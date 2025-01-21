from gflownet.envs.cube import ContinuousCube
import torch.nn.functional as F
from typing import List, Union, Optional
from torchtyping import TensorType
from gflownet.utils.common import tfloat
import torch


Metal = ['Al', 'Ca', 'Fe', 'Mg', 'Mn', 'Ni']
M_list = torch.tensor([
    [0.00149, 0.02, 0.0803, 2.759, 0.0779],
    [0.0055, 0.1374, 0.7697, 1.417, 1.142],
    [3.303, 1.379, 2.4, 0.97, 2.238],
    [0.00155, 5.167, 1.306, 0.744, 1.438],
    [0, 0.0288, 0.0521, 0.0132, 0.0541],
    [0.328, 0, 0, 0, 0]
], dtype=torch.float32).T

abcdelist = torch.tensor(
    [0.27105, 0.567, 0.5632, 0.935, 0.6885], dtype=torch.float32)


class MVP(ContinuousCube):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: clips
        the states into [0, 1] and maps them to [-1.0, 1.0]

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, device=self.device, float_type=self.float)

        return states

    def states2proxy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: clips the
        states into [0, 1] and maps them to [CELL_MIN, CELL_MAX]

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        # Compute the sum along the state dimension for each batch element
        states = tfloat(states, device=self.device, float_type=self.float)
        states = torch.clip(states, min=0.0, max=1.0)
        states = 0.5 * states + 0.1
        state_sum = torch.sum(
            states, dim=1, keepdim=True)  # Shape: [batch, 1]
        state_e = 1-state_sum

        

        # Concatenate the states with their sums along the last dimension
        # Shape: [batch, state_dim + 1]
        states = torch.cat([states, state_e], dim=1)
        states, outline=self.convert_batch(states)

        

        output = torch.cat(
            [states, outline], dim=1)

        return output

    def state2readable(self, state: List) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """

        return str(state).replace("(", "[").replace(")", "]").replace(",", ""), str(self.states2proxy([state]))

    def process_data_set(self, samples):
        """Convert each sample from the CSV (e.g. a string of list) to the internal representation."""
        processed_samples = []
        for s in samples:
            # assuming the state is stored as a string representation of a list:
            processed_sample = eval(s)
            processed_samples.append(processed_sample)
        return processed_samples

    def convert_batch(self,input_batch):


        #acording to the papaer
        """
        Convert a batch of ABCDE values to metal compositions.

        Each input vector should be of the form [A, B, C, D, E].
        If the last component (E) is less than 0.1, it is set to 0.1 and the row is renormalized.
        The function also computes an outline which is 0.1 - original_E when E was low.

        Args:
            input_batch (torch.Tensor): Tensor of shape (batch_size, 5) containing the [A, B, C, D, E] values.
            abcdelist (torch.Tensor): 1D tensor of shape (5,) used for scaling.
            M_list (torch.Tensor): Conversion matrix of shape (5, 6).

        Returns:
            converted (torch.Tensor): Tensor of shape (batch_size, 6) with normalized metal compositions.
            outline (torch.Tensor): Tensor of shape (batch_size,) with the outline adjustments.
        """
        # Ensure a copy of the input to avoid modifying the original tensor.
        x = input_batch  # shape: (batch_size, 5)
        x=x#coverting normalized abcde to adjusted

        # Compute outline for entries where E < 0.1.
        # Save the original E values.
        original_E = x[:, 4]

        # Create outline tensor (for each sample: outline = max(0, 0.1 - original_E))
        outline = torch.clamp(0.1 - original_E, min=0).unsqueeze(-1)

        # For rows where E < 0.1, set E to 0.1.
        mask = x[:, 4] < 0.1
        if mask.any():
            x[mask, 4] = 0.1
            # Renormalize each row so that the sum becomes 1.
            row_sums = x[mask].sum(dim=1, keepdim=True)
            x[mask] = x[mask] / row_sums

        # Now, perform the conversion:
        # 1. Divide each vector elementwise by abcdelist.
        #    Make sure abcdelist has shape (1,5) to allow broadcasting.
        # skipping this step because we aer working with normalised ABCDE
        #x_scaled = x / abcdelist.view(1, -1)

        # 2. Multiply by the conversion matrix.
        #    The multiplication: (batch_size, 5) @ (5, 6) results in (batch_size, 6).
        converted = torch.matmul(x, M_list)

        # 3. Normalize so that each row sums to 1.
        converted = converted / converted.sum(dim=1, keepdim=True)

        return converted, -outline
    

    



