import torch
from dataclasses import dataclass
from typing import List


@dataclass
class CircularTensorList:
    """
    CircularTensorList represents a circular buffer for storing PyTorch tensors.
    """
    initLen: int = 0  # Default length of the circular buffer
    pointer: int = -1  # Pointer to current position in the circular buffer
    buffer: List[torch.Tensor] = None  # Initialize the buffer

    def __post_init__(self) -> None:
        """
        Post-initialization method to initialize the circular buffer with empty tensors.
        """
        self.buffer = [torch.empty(0) for _ in range(self.initLen)]  # Initialize with empty tensors

    def reset(self) -> None:
        """
        Method to reset the circular buffer to a new length.
        """
        self.pointer = -1  # Reset the pointer to initial position
        self.buffer = [torch.empty(0) for _ in range(self.initLen)]  # Reinitialize buffer

    def move_pointer(self) -> None:
        """
        Method to move the circular buffer pointer to the next position.
        """
        self.pointer = (self.pointer + 1) % self.initLen

    def cover(self, iptTensor: torch.Tensor) -> None:
        """
        Method to cover the current position of the circular buffer with an input tensor.

        Parameters:
        - iptTensor: Input tensor to cover the current position.
        """
        self.buffer[self.pointer] = iptTensor

    def record_next(self, iptTensor: torch.Tensor) -> None:
        """
        Method to record an input tensor in the circular buffer, after moving the pointer to the next position.

        Parameters:
        - iptTensor: Input tensor to be recorded.
        """
        self.move_pointer()
        self.cover(iptTensor)

    def get_buffer(self) -> List[torch.Tensor]:
        """
        Method to get the current state of the circular buffer.
        """
        return self.buffer


def compute_temporal_conv(circular_tensor_list: CircularTensorList, kernel: torch.Tensor,
                          pointer: int = None) -> torch.Tensor:
    """
        Computes temporal convolution using CircularTensorList.

        Parameters:
        - circular_tensor_list: An instance of CircularTensorList containing the input tensors.
        - kernel: A 2D tensor representing the convolution kernel.
        - pointer: Optional pointer to start from, defaults to the last position in the circular buffer.

        Returns:
        - optMatrix: The result of the temporal convolution.
    """

    # default value for pointer
    if pointer is None:
        pointer = circular_tensor_list.pointer

    k1 = circular_tensor_list.initLen
    k2 = kernel.size(0)
    length = min(k1, k2)

    # init output matrix
    if circular_tensor_list.buffer[pointer] is None or circular_tensor_list.buffer[
        pointer].numel() == 0:
        return None

    optMatrix = torch.zeros_like(circular_tensor_list.buffer[pointer])

    for t in range(length):
        j = (pointer - t) % k1
        if circular_tensor_list.buffer[j] is not None and circular_tensor_list.buffer[
            j].numel() > 0:
            optMatrix += circular_tensor_list.buffer[j] * kernel[t].view(-1, 1)

    return optMatrix


def compute_circularlist_conv(circular_tensor_cell: CircularTensorList,
                              temporalkernel: torch.Tensor) -> torch.Tensor:
    optMatrix = compute_temporal_conv(circular_tensor_cell, temporalkernel, circular_tensor_cell.pointer)
    return optMatrix
