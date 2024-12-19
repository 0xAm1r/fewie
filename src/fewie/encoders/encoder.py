from dataclasses import dataclass
from typing import Optional
import torch
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import ModelOutput

@dataclass
class EncoderOutput(ModelOutput):
    """
    Base class for encoder outputs
    Args:
        embeddings: Hidden-state embeddings at the output of the encoder
    """
    embeddings: Optional[torch.FloatTensor] = None

class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> EncoderOutput:
        pass
