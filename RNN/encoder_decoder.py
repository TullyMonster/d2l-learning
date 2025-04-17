from abc import ABC as _ABC, abstractmethod as _abstractmethod
from typing import Tuple as _Tuple, TypeVar as _TypeVar, Generic as _Generic

from torch import Tensor as _Tensor
from torch import nn as _nn

__all__ = ['AbstractEncoder', 'AbstractDecoder', 'EncoderDecoder']

_EncoderOutput = _TypeVar('_EncoderOutput')
_DecoderState = _TypeVar('_DecoderState')


class AbstractEncoder(_ABC, _nn.Module, _Generic[_EncoderOutput]):
    """编码器基类，将输入序列编码为中间表示"""

    def __init__(self):
        super().__init__()

    @_abstractmethod
    def forward(self, input_seq: _Tensor, **kwargs) -> _EncoderOutput:
        """将输入序列编码为中间表示"""
        raise NotImplementedError


class AbstractDecoder(_ABC, _nn.Module, _Generic[_DecoderState]):
    """解码器基类，将编码器的输出解码为目标序列"""

    def __init__(self):
        super().__init__()

    @_abstractmethod
    def init_state(self, enc_output: _EncoderOutput, **kwargs) -> _DecoderState:
        """初始化解码器状态"""
        raise NotImplementedError

    @_abstractmethod
    def forward(self, input_seq: _Tensor, state: _DecoderState) -> _Tuple[_Tuple[_Tensor, ...], _DecoderState]:
        """执行序列解码"""
        raise NotImplementedError


class EncoderDecoder(_nn.Module, _Generic[_EncoderOutput, _DecoderState]):
    """以组合类的形式提供编码器-解码器架构的完整端到端序列转换实现"""

    def __init__(self, encoder: AbstractEncoder[_EncoderOutput], decoder: AbstractDecoder[_DecoderState]):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input_seq: _Tensor, dec_input_seq: _Tensor, **kwargs) -> _Tuple[_Tensor, ...]:
        """执行完整的编码-解码过程，输出序列"""
        enc_outputs = self.encoder(enc_input_seq, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, **kwargs)
        dec_outputs, _ = self.decoder(dec_input_seq, dec_state)
        return dec_outputs
