from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False,
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout'],
            duration_encoder_adapters=None,
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex,
            decoder_adapters=None,
            generator_adapters=None,
            **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @dataclass
    class TrainingOutputs:
        audio: torch.FloatTensor
        duration_logits: torch.FloatTensor
        rounded_durations: torch.LongTensor
        f0_pred: torch.FloatTensor
        n_pred: torch.FloatTensor

    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_training_outputs: bool = False,
    ) -> Union[tuple[torch.FloatTensor, torch.LongTensor], "KModel.TrainingOutputs"]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be `(batch, tokens)`, got {tuple(input_ids.shape)}")
        if ref_s.dim() != 2 or ref_s.size(0) != input_ids.size(0):
            raise ValueError("ref_s must be `(batch, 256)` matching input_ids batch size")

        outputs: list[KModel.TrainingOutputs] = []
        for i in range(input_ids.size(0)):
            outputs.append(self._forward_single_with_tokens(input_ids[i : i + 1], ref_s[i : i + 1], speed=speed))

        if return_training_outputs:
            if len(outputs) == 1:
                return outputs[0]
            max_audio = max(int(o.audio.size(-1)) for o in outputs)
            max_frames = max(int(o.f0_pred.size(-1)) for o in outputs)
            max_tokens = max(int(o.duration_logits.size(-1)) for o in outputs)
            audio = ref_s.new_zeros((len(outputs), max_audio))
            duration_logits = ref_s.new_zeros((len(outputs), max_tokens))
            rounded = torch.ones((len(outputs), max_tokens), device=ref_s.device, dtype=torch.long)
            f0 = ref_s.new_zeros((len(outputs), max_frames))
            n_pred = ref_s.new_zeros((len(outputs), max_frames))
            for i, out in enumerate(outputs):
                audio[i, : out.audio.size(-1)] = out.audio.squeeze(0)
                duration_logits[i, : out.duration_logits.size(-1)] = out.duration_logits.squeeze(0)
                rounded[i, : out.rounded_durations.size(-1)] = out.rounded_durations.squeeze(0)
                f0[i, : out.f0_pred.size(-1)] = out.f0_pred.squeeze(0)
                n_pred[i, : out.n_pred.size(-1)] = out.n_pred.squeeze(0)
            return self.TrainingOutputs(
                audio=audio,
                duration_logits=duration_logits,
                rounded_durations=rounded,
                f0_pred=f0,
                n_pred=n_pred,
            )

        if len(outputs) == 1:
            return outputs[0].audio, outputs[0].rounded_durations.squeeze(0)
        raise ValueError("forward_with_tokens without return_training_outputs only supports a single example")

    def _forward_single_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        *,
        speed: float,
    ) -> "KModel.TrainingOutputs":
        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration_logits_full = self.predictor.duration_proj(x)
        duration_logits = torch.sigmoid(duration_logits_full).sum(axis=-1) / speed
        pred_dur = torch.round(duration_logits).clamp(min=1).long()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur.squeeze(0))
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).reshape(1, -1)
        return self.TrainingOutputs(
            audio=audio,
            duration_logits=duration_logits,
            rounded_durations=pred_dur,
            f0_pred=F0_pred,
            n_pred=N_pred,
        )

    @torch.no_grad()
    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(
            input_ids,
            ref_s,
            speed,
        )
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(
            input_ids,
            ref_s,
            speed,
        )
        return waveform, duration
