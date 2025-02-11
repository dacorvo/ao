from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SubclassTensorArgs:
    original_shape: torch.Size
    original_strides: Tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


def get_block_absmax(inpt_tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    """Iterate through a flattened tensor getting the absmax scalers for each block

    Args:
        inpt_tensor: Input tensor to get scalers for
        block_size: Block size for the scanning window
    Returns:
        torch.Tensor: Tensor of scalers for each block
    """
    assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
    assert (
        inpt_tensor.numel() % block_size
    ) == 0, f"Input tensor must be divisible by block size, got {inpt_tensor.numel()} and {block_size}"

    n_blocks = inpt_tensor.numel() // block_size
    blocks = inpt_tensor.view(n_blocks, block_size)
    block_scalers = blocks.abs().max(dim=1).values
    return block_scalers


class NF4Tensor(torch.Tensor):
    """NF4Tensor class for converting a weight to the QLoRA NF4 format"""

    def __new__(
        cls,
        # Args related for base tensor construction
        tensor_meta: SubclassTensorArgs,
        # Args stored on the instance
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):
        """Create a new NF4Tensor object
        Args:
            tensor_meta: Metadata for the tensor
            block_size: Size of the quantization block
            n_blocks: Number of blocks to cover the full tensor
            scaler_block_size: Block size for the scalar quantization
            quantized_scalers: Quantized scalers data' represented a uint8 tensor
            quantization_factor: Quantization factor, single scalar represented as torch.Tensor
            scaler_mean: Mean of the scalers
            quantized_data: Quantized data represented as uint8 tensor
            nf4: NF4 tensor LUT for the quantization and dequantization

        """

        nf4tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor_meta.original_shape,
            tensor_meta.original_strides,
            tensor_meta.storage_offset,
            dtype=tensor_meta.dtype,
            device=tensor_meta.device,
            requires_grad=tensor_meta.requires_grad,
        )
        return nf4tensor

    def __init__(
        self,
        tensor_meta: SubclassTensorArgs,
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):
        """Initialize the NF4Tensor class"""
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.scaler_block_size = scaler_block_size
        self.quantized_scalers = quantized_scalers
        self.quantization_factor = quantization_factor
        self.scaler_mean = scaler_mean
        self.quantized_data = quantized_data
        self.nf4 = nf4

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        inpt_tensor: torch.Tensor,
        block_size: int = 64,
        scaler_block_size: int = 256,
    ):
        assert inpt_tensor.dtype == torch.bfloat16
        assert (
            inpt_tensor.numel() % block_size == 0
        ), "Input tensor must be divisible by block size"
        assert inpt_tensor.dtype == torch.bfloat16, "Input tensor must be bfloat16"
        assert inpt_tensor.is_contiguous, "Input tensor must be contiguous!"
        # I think I want do this
        # assert not inpt_tensor.requires_grad, "Input tensor must not require grad"
        device = inpt_tensor.device
        # Cache the tensor on the class def
        nf4 = torch.tensor(
            [
                -1.0000,
                -0.6962,
                -0.5251,
                -0.3949,
                -0.2844,
                -0.1848,
                -0.0911,
                0.0000,
                0.0796,
                0.1609,
                0.2461,
                0.3379,
                0.4407,
                0.5626,
                0.7230,
                1.0000,
            ],
            device=device,
            dtype=torch.bfloat16,
        )
        n_blocks = inpt_tensor.numel() // block_size
        # Double quantization
        (
            quantized_scalers,
            quantization_factor,
            scaler_mean,
        ) = cls.double_quantize_scalers(
            inpt_tensor.flatten(), block_size, scaler_block_size
        )
        quantized_data = cls.convert_to_norm_float_weight(
            inpt_tensor, n_blocks, block_size, nf4
        )
        tensor_meta = SubclassTensorArgs(
            inpt_tensor.size(),
            inpt_tensor.stride(),
            inpt_tensor.storage_offset(),
            inpt_tensor.dtype,
            inpt_tensor.device,
            inpt_tensor.requires_grad,
        )
        return cls(
            tensor_meta,
            block_size,
            n_blocks,
            scaler_block_size,
            quantized_scalers,
            quantization_factor,
            scaler_mean,
            quantized_data,
            nf4=nf4,
        )

    @staticmethod
    def double_quantize_scalers(
        inpt_tensor: torch.Tensor,
        block_size: int,
        scaler_block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used to achieve the double quantization of the scalers
        We take the input tensor first calculate the absmax quantization factors for each block.
        We then find the mean of our positive absmax scalers. We subtract this mean from the scalers
        And then we calculate the absmax quantization factors for each block again. We then quantize the scalers to int8.

        Args:
            inpt_tensor: Input tensor to convert to QLoRA format, typically a weight tensor

        Returns:
            torch.Tensor: Tensor of per_block quantization factors stored in int8 format
                size: (n_blocks)
            torch.Tensor: Tensor of per_scaler_block quantization factors stored in int16 format
                size: (n_scaler_blocks)
        """
        assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
        assert (
            inpt_tensor.numel() % scaler_block_size
        ) == 0, f"Input tensor must be divisible by block size, got {inpt_tensor.numel()} and {scaler_block_size}"

        # First round of quantization
        # Produces: A tensor of size (n_blocks) of inpt_tensor.dtype
        scalers_1 = get_block_absmax(inpt_tensor, block_size)
        scalers_1_mean = scalers_1.mean()
        scalers_1 = scalers_1 - scalers_1_mean
        # Second round of quantization
        assert (
            scalers_1.numel() % scaler_block_size == 0
        ), "Number of scalers must be divisible by scaler block size"
        n_scaler_blocks = scalers_1.numel() // scaler_block_size
        scaler_blocks = scalers_1.view(n_scaler_blocks, scaler_block_size)

        scaler_absmax = get_block_absmax(scalers_1, scaler_block_size)
        scaler_absmax = scaler_absmax.unsqueeze(-1).expand(
            n_scaler_blocks, scaler_block_size
        )

        quantization_factor = 256 / (2 * scaler_absmax)
        # Length equal to weight numel // block_size
        quantized_scaler_blocks = scaler_blocks * quantization_factor
        quantized_scaler_blocks = quantized_scaler_blocks.round()
        quantized_scaler_blocks = quantized_scaler_blocks.clamp(-128, 127)

        # This is needed to make sure that quantization_factor remains a repeated view of n_scaler_blocks
        # For some reason the 127/scaler_absmax realizes n_scaler entries when only n_scaler_blocks are needed
        # The following will grab the first entry for the n_scaler_blocks which is the same across the scaler_block_size
        quantization_factor = quantization_factor[:, 0]

        return (
            quantized_scaler_blocks.flatten().to(torch.int8),
            quantization_factor.view(n_scaler_blocks),
            scalers_1_mean,
        )

    def dequantize_scalers(
        self,
        inpt_tensor: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_block_size: int,
    ) -> torch.Tensor:
        """Used to unpack the double quantized scalers

        Args;
            inpt_tensor: Input tensor to convert to QLoRA format this is the quantized scalers in int8 format
            quantization_factor: Tensor of per_scaler_block quantization factors stored in inpt_weight.dtype
                size: (n_scaler_blocks)
            scaler_block_size: Scaler block size to use for double quantization.

        """
        assert inpt_tensor.dim() == 1, "Input tensor must be flattened"
        assert (
            inpt_tensor.numel() % scaler_block_size
        ) == 0, f"Input tensor must be divisible by block size, got {inpt_tensor.numel()} and {scaler_block_size}"
        n_scaler_blocks = inpt_tensor.numel() // scaler_block_size
        inpt_tensor = inpt_tensor.view(n_scaler_blocks, scaler_block_size)
        dequantized = (inpt_tensor / quantization_factor.unsqueeze(-1)).flatten().to(
            torch.bfloat16
        ) + self.scaler_mean
        return dequantized

    @staticmethod
    def convert_to_norm_float_weight(
        inpt_tensor: torch.Tensor, n_blocks: int, block_size: int, nf4: torch.tensor
    ) -> torch.Tensor:
        """Convert a tensor to the normalized float weight format"""
        flattened_tensor = inpt_tensor.flatten()
        #  Since we are using uint8 we will encode 2 entries per byte
        numel = inpt_tensor.numel()
        assert (
            numel % 2 == 0
        ), "Number of elements must be even just to not have to think about the end"
        # Reshape the flattened tensor into blocks of size self.block_size
        blocks = flattened_tensor.view(n_blocks, block_size)

        # Scale the blocks
        scalers = get_block_absmax(inpt_tensor.flatten(), block_size)
        scales = scalers.unsqueeze(-1).expand(n_blocks, block_size)
        scaled_blocks = blocks / scales

        # Returns a flattened tensor with each element quantized to nf4 index
        quantized_blocks = NF4Tensor.quantize_tensor_nearest(
            scaled_blocks.flatten(), nf4
        )

        # Combine the quantized elements into uint8 values
        # This lays out two consecutive elements in the same byte
        # [a, b, c, d] -> [ab, cd]
        # The size of combined blocks will be half the size of the original tensor
        combined_blocks = quantized_blocks[::2] << 4 | quantized_blocks[1::2]

        return combined_blocks.to(torch.uint8)

    def get_original_weight(self) -> torch.Tensor:
        """Get the original weight from the normalized float weight format"""
        # Since we are using uint8 we will decode 2 entries per byte
        # Shift elements down 4 and select out the bottom 4 bits
        first_elements = (self.quantized_data >> 4).to(torch.long)
        second_elements = (self.quantized_data & 0b1111).to(torch.long)

        # Dequantize every element
        dequantized_first = self.dequantize(first_elements, self.nf4)
        dequantized_second = self.dequantize(second_elements, self.nf4)

        # Build up matrix of scalers repeated for each element in the block
        # Since first and second elements make up a full block
        # we expand out to half the size of the full block
        scalers = self.dequantize_scalers(
            self.quantized_scalers, self.quantization_factor, self.scaler_block_size
        )
        repeated = scalers.unsqueeze(-1).expand(scalers.size(0), self.block_size // 2)

        scaled_first = dequantized_first * repeated.flatten()
        scaled_second = dequantized_second * repeated.flatten()

        # Flip them to be vertical and them stack them together horizontally
        # Upon flattening this will interleave the elements
        scaled_first = scaled_first.unsqueeze(-1).transpose(0, 1)
        scaled_second = scaled_second.unsqueeze(-1).transpose(0, 1)
        return torch.stack([scaled_first, scaled_second], dim=-1).reshape(self.shape)

    @staticmethod
    def quantize_tensor_nearest(
        value: torch.float16, nf4: torch.Tensor
    ) -> torch.Tensor:
        """Quantize a float16 tensor to nf4 format to nearest and not rounded up"""
        value = value.unsqueeze(-1)  # (numel, 1)
        # Compare the value tensor with the nf4 tensor element-wise
        diff = (value - nf4).abs()
        closest_nf4 = diff.min(dim=-1).indices
        return closest_nf4

    @staticmethod
    def dequantize(value: torch.Tensor, nf4: torch.Tensor) -> torch.Tensor:
        """Dequantize a nf4 value to float16 format"""
        # return nf4.index_select(0, value)
        return nf4[value]

    def unpack(
        self,
    ) -> Tuple[
        int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Size
    ]:
        return (
            self.block_size,
            self.n_blocks,
            self.scaler_block_size,
            self.quantized_scalers,
            self.quantization_factor,
            self.scaler_mean,
            self.quantized_data,
        )

    def __repr__(self):
        return f"Quantized Data: {self.quantized_data}\nScalers: {self.quantized_scalers}\n"

    def __str__(self):
        return f"NF4Tensor({self.shape}, {self.block_size})"

    def __tensor_flatten__(self):
        tensor_meta = SubclassTensorArgs(
            self.shape,
            self.stride(),
            self.storage_offset(),
            self.dtype,
            self.device,
            self.requires_grad,
        )
        ctx = {
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
            "scaler_block_size": self.scaler_block_size,
            "tensor_meta": tensor_meta,
        }
        return [
            "quantized_data",
            "scaler_mean",
            "quantization_factor",
            "quantized_scalers",
            "nf4",
        ], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 5, "Expected 5 inner tensors"
        return NF4Tensor(
            metadata["tensor_meta"],
            metadata["block_size"],
            metadata["n_blocks"],
            metadata["scaler_block_size"],
            inner_tensors["quantized_scalers"],
            inner_tensors["quantization_factor"],
            inner_tensors["scaler_mean"],
            inner_tensors["quantized_data"],
            inner_tensors["nf4"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """TODO we are not supporting torch dispatch at the moment
        instead we have created a Autograd.Function to handle the linear
        """
        raise NotImplementedError("NF4Tensor does not support torch dispatch")

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl


class LinearNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: NF4Tensor):
        """Save the quantized nf4 weight for backward pass"""
        ctx.nf4_weight = weight
        return F.linear(input, weight.get_original_weight())

    @staticmethod
    def backward(ctx, grad_output):
        """The nf4 weight will never require grad so we can just return the grad_output @ weight.get_original_weight()"""
        weight: NF4Tensor = ctx.nf4_weight
        return grad_output @ weight.get_original_weight(), None


def linear_nf4(input: torch.Tensor, weight: NF4Tensor) -> torch.Tensor:
    """Apply a linear operation with the NF4Tensor weight

    Args:
        input: Input tensor
        weight: NF4Tensor weight
    """
    return LinearNF4.apply(input, weight)
