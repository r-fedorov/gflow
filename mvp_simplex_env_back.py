from gflownet.envs.cube import ContinuousCube
from torch.distributions import Bernoulli, Beta, Categorical, MixtureSameFamily, OneHotCategorical
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from torchtyping import TensorType
from gflownet.utils.common import tfloat,tbool
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


class SimplexWalker(ContinuousCube):
    def __init__(self,
                 random_distr_params = {
                "beta_weights": 1.0,
                "beta_alpha": 10.0,
                "beta_beta": 10.0,
                "bernoulli_bts_prob": 0.1,
                "bernoulli_eos_prob": 0.1,
                "one_hot_logits": torch.zeros(4) + 1.0,  
                },
                fixed_distr_params = {
                "beta_weights": 1.0,
                "beta_alpha": 10.0,
                "beta_beta": 10.0,
                "bernoulli_bts_prob": 0.1,
                "bernoulli_eos_prob": 0.1,
                "one_hot_logits": torch.zeros(4) + 1.0,  
                }, **kwargs):
        self.random_distr_params=random_distr_params
        self.fixed_distr_params=fixed_distr_params


        super().__init__(fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,**kwargs)



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
        return 2.0 * torch.clip(states, min=0.0, max=1.0)

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
        outline = 1-state_sum
        state_e = 1-state_sum
        outline = torch.clip(outline, min=-0.6*self.n_dim, max=0.0)
        state_e = torch.clip(state_e, min=0.1, max=0.6)

        # Concatenate the states with their sums along the last dimension
        # Shape: [batch, state_dim + 1]
        states = torch.cat([states, state_e], dim=1)
        state_norm = states / torch.sum(
            states, dim=1, keepdim=True)

        metal_prop = self._compute_metal_proportions(state_norm)

        states = torch.cat(
            [metal_prop, outline], dim=1)

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
        states, outline = self.convert_batch(states)

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

    def convert_batch(self, input_batch):

        # acording to the papaer
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
        x = x  # coverting normalized abcde to adjusted

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
        # x_scaled = x / abcdelist.view(1, -1)

        # 2. Multiply by the conversion matrix.
        #    The multiplication: (batch_size, 5) @ (5, 6) results in (batch_size, 6).
        converted = torch.matmul(x, M_list)

        # 3. Normalize so that each row sums to 1.
        converted = converted / converted.sum(dim=1, keepdim=True)

        return converted, -outline
    
    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        ...
        The environment consists of both continuous and discrete actions.
        ...
        D x C x 3 + 2
        ...
        """
        # Continuous part: mixture of Beta distributions
        if params is None:
            raise ValueError("The 'params' argument is None.")
            print("params:", params)
        self._len_policy_output_cont = self.n_dim * self.n_comp * 3
        policy_output_cont = torch.empty(
            self._len_policy_output_cont,
            dtype=self.float,
            device=self.device,
        )
        policy_output_cont[0::3] = params["beta_weights"]
        policy_output_cont[1::3] = self._beta_params_to_policy_outputs("alpha", params)
        policy_output_cont[2::3] = self._beta_params_to_policy_outputs("beta", params)

        # Logit for Bernoulli distribution to model back-to-source (BTS) action
        policy_output_bts_logit = torch.logit(
            tfloat(
                [params["bernoulli_bts_prob"]],
                float_type=self.float,
                device=self.device,
            )
        )
        # Logit for Bernoulli distribution to model EOS action

        if ["bernoulli_eos_prob"] is None:

            raise ValueError("bernoulli_eos_prob argument is None.")
            
        policy_output_eos_logit = torch.logit(
            tfloat(
                [params["bernoulli_eos_prob"]],
                float_type=self.float,
                device=self.device,
            )
        )


        # ---- NEW PART: one‐hot dimension‐selection logits ----

        
        policy_output_one_hot_logit = tfloat(
            params["one_hot_logits"],  # shape n_dim
            float_type=self.float,
            device=self.device,
        )


        # Ensure shape [n_dim]
        #assert policy_output_one_hot_logit.shape[-1] == self.n_dim

        # Concatenate all outputs.  Final shape becomes:
        #   n_dim * n_comp * 3  (beta mixture)
        # + 1                   (bts logit)
        # + 1                   (eos logit)
        # + n_dim               (one‐hot logits)
        # = n_dim * n_comp * 3 + 2 + n_dim
        policy_output = torch.cat(
            (
                policy_output_cont,
                policy_output_bts_logit,  # shape [1]
                policy_output_eos_logit,  # shape [1]
                policy_output_one_hot_logit,  # shape [n_dim]
            )
        )
        return policy_output

    def relative_to_absolute_increments(
        self,
        states: TensorType["n_states", "n_dim"],
        increments_rel: TensorType["n_states", "n_dim"],
        increments_one_hot: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Returns a batch of absolute increments (actions) given a batch of states,
        relative increments and minimum_increments.

        Given a dimension value x, a relative increment r, and a minimum increment m,
        then the absolute increment a is given by:

        Forward:

        a = m + r * (1 - x - m)

        Backward:

        a = m + r * (x - m)
        """

        # making only one incriment per dimesnion (the same one that was adjsuteb by the beta distribution)
        # Find the indices of the maximum values along the last dimension

        min_increments = torch.full_like(
            increments_rel, self.min_incr, dtype=self.float, device=self.device
        )
        #min_increments = min_increments*increments_one_hot

        # also one_hot for incriments

        if is_backward:
            
            #C = torch.sum(states, dim=1, keepdim=True)
            #room=torch.clamp(C-min_increments,min=0.0,max=1.0)
            room = torch.clamp(states - min_increments, 0.0, 1.0)
            a = min_increments + increments_rel * room


            #return min_increments + increments_rel * room
            return a
        else:
            B = (1.0 - torch.sum(states, dim=1, keepdim=True))
            room=torch.clamp(B-min_increments,min=0.0,max=1.0)

            return min_increments + increments_rel * room

    def absolute_to_relative_increments(
        self,
        states: TensorType["n_states", "n_dim"],
        increments_abs: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Returns a batch of relative increments (as sampled by the Beta distributions)
        given a batch of states, absolute increments (actions) and minimum_increments.

        Given a dimension value x, an absolute increment a, and a minimum increment m,
        then the relative increment r is given by:

        Forward:

        r = (a - m) / (1 - x - m)
        r= (a-incr)/C

        Backward:

        r = (a - m) / (x - m)
        r=(a-m) /B
        """
        # making only one incriment per dimesnion (the same one that was adjsuteb by the beta distribution)
        # Find the indices of the maximum values along the last dimension

        min_increments = torch.full_like(
            increments_abs, self.min_incr, dtype=self.float, device=self.device
        )
        #min_increments = min_increments*increments_one_hot
  
        if is_backward:
            increments_rel = (increments_abs - min_increments) / (
                states - min_increments
            )
            # Add epsilon to numerator and denominator if values are unbounded
            if not torch.all(torch.isfinite(increments_rel)):
                increments_rel = (increments_abs - min_increments + 1e-9) / (
                    states - min_increments + 1e-9
                )
            return increments_rel
        else:

            B = (1.0 - torch.sum(states, dim=1, keepdim=True))
            return (increments_abs - min_increments) / B - min_increments

    def _get_jacobian_diag(
        self,
        states_from: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Computes the diagonal entries of the Jacobian for the transformation between
        relative increments and the change in state.

        In the modified version the update rules are:

        Forward:
            s'_d = s_d + m + r_d * (B - m),
            where B = (1 - sum(s)) / n_dim.
            Thus, ∂s'_d/∂r_d = B - m   and  ∂r_d/∂s'_d = 1 / (B - m).

        Backward:
            s_d = s'_d - m - r_d * (C - m),
            where C = (sum(s') / n_dim).
            Thus, ∂s_d/∂r_d = C - m   and  ∂r_d/∂s_d = 1 / (C - m).

        Here, states_from is interpreted as the fixed state (s in the forward move and s' in
        the backward move) used for the transformation.
        """
        if is_backward:
            min_increments = torch.full_like(
                states_from, self.min_incr, dtype=self.float, device=self.device
            )
    
            return 1.0 / ((states_from - min_increments))

        else:
            # For a forward update we assume:

            # Compute B for each sample:
            B = (1.0 - torch.sum(states_from, dim=1, keepdim=True))
            # The Jacobian for each coordinate is then 1 / (B - m)
            return 1.0 / (B - self.min_incr)

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        The action space is continuous, thus the mask is not only of invalid actions as
        in discrete environments, but also an indicator of "special cases", for example
        states from which only certain actions are possible.

        The values of True/False intend to approximately stick to the semantics in
        discrete environments, where the mask is of "invalid" actions, but it is
        important to note that a direct interpretation in this sense does not always
        apply.

        For example, the mask values of special cases are True if the special cases they
        refer to are "invalid". In other words, the values are False if the state has
        the special case.

        The forward mask has the following structure:

        - 0 : whether a continuous action is invalid. True if the value at any
          dimension is larger than 1 - min_incr, or if done is True. False otherwise.
        - 1 : special case when the state is the source state. False when the state is
          the source state, True otherwise.
        - 2 : whether EOS action is invalid. EOS is valid from any state, except the
          source state or if done is True.
        - -n_dim: : dimensions that should be ignored when sampling actions or
          computing logprobs. This can be used for trajectories that may have
          multiple dimensions coupled or fixed. For each dimension, True if ignored,
          False, otherwise.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        # If done, the entire mask is True (all actions are "invalid" and no special
        # cases)
        if done:
            return [True] * self.mask_dim

        mask = [False] * self.mask_dim_base + self.ignored_dims
        # If the state is the source state, EOS is invalid
        if self._get_effective_dims(state) == self._get_effective_dims(self.source):
            mask[2] = True
        # If the state is not the source, indicate not special case (True)
        else:
            mask[1] = True

        # If the sum of all dimensions is greater than 1 - min_incr, then continuous
        # actions are invalid (True) and EOS is valid

        if sum(self._get_effective_dims(state)) >= 1 - self.min_incr:

            mask[0] = True
            mask[2] = False
        # If any dimension of the state is greater then 1-icr,then continuous
        # actions are invalid

        if any([s > 1 - self.min_incr and sum(self._get_effective_dims(state)) >= 0 for s in self._get_effective_dims(state)]):
            mask[0] = True
  

        #print("forward_mask",mask)
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        The action space is continuous, thus the mask is not only of invalid actions as
        in discrete environments, but also an indicator of "special cases", for example
        states from which only certain actions are possible.

        In order to approximately stick to the semantics in discrete environments,
        where the mask is of "invalid" actions, that is the value is True if an action
        is invalid, the mask values of special cases are True if the special cases they
        refer to are "invalid". In other words, the values are False if the state has
        the special case.

        The backward mask has the following structure:

        - 0 : whether a continuous action is invalid. True if the value at any
          dimension is smaller than min_incr, or if done is True. False otherwise.
        - 1 : special case when back-to-source action is the only possible action.
          False if any dimension is smaller than min_incr, True otherwise.
        - 2 : whether EOS action is invalid. False only if done is True, True
          (invalid) otherwise.
        - -n_dim: : dimensions that should be ignored when sampling actions or
          computing logprobs. this can be used for trajectories that may have
          multiple dimensions coupled or fixed. for each dimension, true if ignored,
          false, otherwise. By default, no dimension is ignored.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True] * self.mask_dim_base + self.ignored_dims
        # If the state is the source state, entire mask is True
        if self._get_effective_dims(state) == self._get_effective_dims(self.source):
            return mask
        # If done, only valid action is EOS.
        if done:
            mask[2] = False
            return mask

        # is the sum of states is smaller then the ,min_incriment -> go to the sourse
        """if sum(self._get_effective_dims(state)) < self.min_incr:
            mask[1] = False
            return mask"""
        if any([s < self.min_incr for s in self._get_effective_dims(state)]):
            mask[1] = False
            return mask
        # Otherwise, continuous actions are valid
        mask[0] = False
        #print("backward_mask",mask)
        return mask

    def _sample_actions_batch_forward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a a batch of forward actions from a batch of policy outputs.

        An action indicates, for each dimension, the absolute increment of the
        dimension value. However, in order to ensure that trajectories have finite
        length, increments must have a minumum increment (self.min_incr) except if the
        originating state is the source state (special case, see
        get_mask_invalid_actions_forward()). Furthermore, absolute increments must also
        be smaller than the distance from the dimension value to the edge of the cube
        (1.0). In order to accomodate these constraints, first relative
        increments (in [0, 1]) are sampled from a (mixture of) Beta distribution(s),
        where 0.0 indicates an absolute increment of min_incr and 1.0 indicates an
        absolute increment of 1 - x + min_incr (going to the edge).

        Therefore, given a dimension value x, a relative increment r, a minimum
        increment m and a maximum value 1, the absolute increment a is given by:

        a = m + r * (1 - x - m)

        The continuous distribution to sample the continuous action described above
        must be mixed with the discrete distribution to model the sampling of the EOS
        action. The EOS action can be sampled from any state except from the source
        state or whether the trajectory is done. That the EOS action is invalid is
        indicated by mask[-1] being False.

        Finally, regarding the constraints on the increments, the following special
        cases are taken into account:

        - The originating state is the source state: in this case, the minimum
          increment is 0.0 instead of self.min_incr. This is to ensure that the entire
          state space can be reached. This is indicated by mask[-2] being False.
        - The value at any dimension is at a distance from the cube edge smaller than the
          minimum increment (x > 1 - m). In this case, only EOS is valid.
          This is indicated by mask[0] being True (continuous actions are invalid).
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        is_eos = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        # Determine source states
        is_source = ~mask[:, 1]
        # EOS is the only possible action if continuous actions are invalid (mask[0] is
        # True)
        is_eos_forced = mask[:, 0]
        is_eos[is_eos_forced] = True
        # Ensure that is_eos_forced does not include any source state
        assert not torch.any(torch.logical_and(is_source, is_eos_forced))
        # Sample EOS from Bernoulli distribution
        do_eos = torch.logical_and(~is_source, ~is_eos_forced)
        if torch.any(do_eos):
            is_eos_sampled = torch.zeros_like(do_eos)
            logits_eos = self._get_policy_eos_logit(policy_outputs)[do_eos]
            distr_eos = Bernoulli(logits=logits_eos)
            is_eos_sampled[do_eos] = tbool(
                distr_eos.sample(), device=self.device)
            is_eos[is_eos_sampled] = True
        # Sample (relative) increments if EOS is not the (sampled or forced) action

        do_increments = ~is_eos
        if torch.any(do_increments):
            if sampling_method == "uniform":
                raise NotImplementedError()
            elif sampling_method == "policy":
                distr_increments, one_hot_dist = self._make_increments_distribution(
                    policy_outputs[do_increments]
                )

            ####################################

            # we will sample a distribution here

            beta_increments = distr_increments.sample()
            one_hot = one_hot_dist.sample()
            #print(beta_increments.shape)
            #print(one_hot.shape)
            # increments = one_hot * beta_increments
            increments = beta_increments
            mask[do_increments, -self.n_dim:] = one_hot.logical_not()
            # we will use it later to mask out the 0-jacobian

            ##################################

            is_relative = ~is_source[do_increments]

            states_from_rel = tfloat(
                states_from_tensor[do_increments],
                float_type=self.float,
                device=self.device,
            )[is_relative]
            increments[is_relative] = self.relative_to_absolute_increments(
                states_from_rel,
                increments[is_relative],
                one_hot,
                is_backward=False,
            )
        # Build actions
        actions_tensor = torch.full(
            (n_states, self.n_dim + 1), torch.inf, dtype=self.float, device=self.device
        )
        if torch.any(do_increments):
            # Make increments of ignored dimensions zero
            increments = self._mask_ignored_dimensions(
                mask[do_increments], increments)
            # Add dimension is_source and add to actions tensor
            actions_tensor[do_increments] = torch.cat(
                (increments, torch.zeros((increments.shape[0], 1))), dim=1
            )
        actions_tensor[is_source, -1] = 1
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions,  None

    def _sample_actions_batch_backward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a a batch of backward actions from a batch of policy outputs.

        An action indicates, for each dimension, the absolute increment of the
        dimension value. However, in order to ensure that trajectories have finite
        length, increments must have a minumum increment (self.min_incr). Furthermore,
        absolute increments must also be smaller than the distance from the dimension
        value to the edge of the cube. In order to accomodate these constraints, first
        relative increments (in [0, 1]) are sampled from a (mixture of) Beta
        distribution(s), where 0.0 indicates an absolute increment of min_incr and 1.0
        indicates an absolute increment of x (going back to the source).

        Therefore, given a dimension value x, a relative increment r, a minimum
        increment m and a maximum value 1, the absolute increment a is given by:

        a = m + r * (x - m)

        The continuous distribution to sample the continuous action described above
        must be mixed with the discrete distribution to model the sampling of the back
        to source (BTS) action. While the BTS action is also a continuous action, it
        needs to be modelled with a (discrete) Bernoulli distribution in order to
        ensure that this action has positive likelihood.

        Finally, regarding the constraints on the increments, the special case where
        the trajectory is done and the only possible action is EOS, is also taken into
        account.
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        is_bts = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        # EOS is the only possible action only if done is True (mask[2] is False)
        is_eos = ~mask[:, 2]
        # Back-to-source (BTS) is the only possible action if mask[1] is False
        is_bts_forced = ~mask[:, 1]
        is_bts[is_bts_forced] = True
        # Sample BTS from Bernoulli distribution
        do_bts = torch.logical_and(~is_bts_forced, ~is_eos)
        if torch.any(do_bts):
            is_bts_sampled = torch.zeros_like(do_bts)
            logits_bts = self._get_policy_source_logit(policy_outputs)[do_bts]
            distr_bts = Bernoulli(logits=logits_bts)
            is_bts_sampled[do_bts] = tbool(
                distr_bts.sample(), device=self.device)
            is_bts[is_bts_sampled] = True
        # Sample relative increments if actions are neither BTS nor EOS
        do_increments = torch.logical_and(~is_bts, ~is_eos)
        if torch.any(do_increments):
            if sampling_method == "uniform":
                raise NotImplementedError()
            elif sampling_method == "policy":
                distr_increments, one_hot_dist = self._make_increments_distribution(
                    policy_outputs[do_increments]
                )

            beta_increments = distr_increments.sample()
            one_hot = one_hot_dist.sample()
            # increments = one_hot * beta_increments
            increments = beta_increments
            mask[do_increments, -self.n_dim:] = one_hot.logical_not()
            # we will use it later to mask out the 0-jacobian
            # Shape of increments_rel: [n_do_increments, n_dim]

            # Compute absolute increments from all sampled relative increments
            states_from_rel = tfloat(
                states_from, float_type=self.float, device=self.device
            )[do_increments]
            increments = self.relative_to_absolute_increments(
                states_from_rel,
                increments,
                one_hot,
                is_backward=True,
            )
        # Build actions
        actions_tensor = torch.zeros(
            (n_states, self.n_dim + 1), dtype=self.float, device=self.device
        )
        actions_tensor[is_eos] = tfloat(
            self.eos, float_type=self.float, device=self.device
        )
        if torch.any(do_increments):
            # Make increments of ignored dimensions zero
            increments = self._mask_ignored_dimensions(
                mask[do_increments], increments)
            # Add dimension is_source and add to actions tensor
            actions_tensor[do_increments] = torch.cat(
                (increments, torch.zeros((increments.shape[0], 1))), dim=1
            )
        if torch.any(is_bts):
            # BTS actions are equal to the originating states
            actions_bts = tfloat(
                states_from, float_type=self.float, device=self.device
            )[is_bts]
            actions_bts = torch.cat(
                (actions_bts, torch.ones((actions_bts.shape[0], 1))), dim=1
            )
            actions_tensor[is_bts] = actions_bts
            # Make ignored dimensions zero
            actions_tensor[is_bts, :-1] = self._mask_ignored_dimensions(
                mask[is_bts], actions_tensor[is_bts, :-1]
            )
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions, None

    def _get_logprobs_forward(
        self,
        policy_outputs: torch.Tensor,       # shape [n_states, policy_output_dim]
        actions: torch.Tensor,              # shape [n_states, actions_dim]
        mask: torch.Tensor,                 # shape [n_states, 3]
        states_from: list,                  # length = n_states
    ) -> torch.Tensor:
        """
        Computes log probabilities of forward actions for a batch of states.

        `actions` has shape [n_states, n_dim+1], where the +1 is the is_source dimension or EOS dimension.
        """

        n_states = policy_outputs.shape[0]
        states_from_tensor = torch.as_tensor(
            states_from, dtype=self.float, device=self.device
        )

        # Initialize per-state log-prob terms
        logprobs_eos = torch.zeros(n_states, dtype=self.float, device=self.device)
        logprobs_increments_rel = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=self.device
        )
        log_jacobian_diag = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=self.device
        )
        one_hot_logp_full = torch.zeros(n_states, dtype=self.float, device=self.device)

        # Identify which states are source, forced EOS, etc.
        is_source = ~mask[:, 1]        # mask[:,1] = True => "special source" is invalid
        is_eos_forced = mask[:, 0]     # mask[:,0] = True => only EOS is valid
        is_eos = torch.zeros(n_states, dtype=torch.bool, device=self.device)

        # Force EOS if continuous actions are invalid
        is_eos[is_eos_forced] = True

        # Bernoulli sample for “do EOS?” except for source or forced-EOS states
        eos_tensor = torch.as_tensor(
            self.eos, dtype=self.float, device=self.device
        )  # shape [n_dim+1]
        do_eos = torch.logical_and(~is_source, ~is_eos_forced)  # can potentially do EOS
        if torch.any(do_eos):
            # Among the do_eos states, see if they actually took the EOS action
            is_eos_sampled = torch.zeros_like(do_eos)
            is_eos_sampled[do_eos] = torch.all(actions[do_eos] == eos_tensor, dim=1)

            # Mark those states as EOS
            is_eos[is_eos_sampled] = True

            # Probability of taking EOS from policy
            logits_eos = self._get_policy_eos_logit(policy_outputs)[do_eos]  # shape [#do_eos]
            distr_eos = Bernoulli(logits=logits_eos)

            # log P(EOS or not)
            logprobs_eos[do_eos] = distr_eos.log_prob(is_eos_sampled[do_eos].float())

        # Now handle increments for states that did not do EOS
        do_increments = ~is_eos
        if torch.any(do_increments):
            # Sub-batch
            increments_sub = actions[do_increments, :-1]       # shape [M, n_dim]
            policy_sub = policy_outputs[do_increments]         # shape [M, policy_output_dim]
            states_from_sub = states_from_tensor[do_increments]# shape [M, n_dim]

            # Convert absolute -> relative increments for non-source states
            is_relative_sub = ~is_source[do_increments]  # shape [M]
            if torch.any(is_relative_sub):
                increments_sub[is_relative_sub] = self.absolute_to_relative_increments(
                    states_from_sub[is_relative_sub],
                    increments_sub[is_relative_sub],
                    is_backward=False,
                )

            # Compute log-Jacobian for non-source states
            # (We fill into the big [n_states, n_dim] tensor in the correct positions.)
            log_jacobian_diag_sub = torch.zeros_like(increments_sub)
            if torch.any(is_relative_sub):
                jac_diag_vals = self._get_jacobian_diag(
                    states_from_sub[is_relative_sub],
                    is_backward=False,
                )  # shape [#is_relative_sub, 1 or n_dim]
                log_jacobian_diag_sub[is_relative_sub] = torch.log(jac_diag_vals)

            # Place sub-batch back into full array
            log_jacobian_diag[do_increments] = log_jacobian_diag_sub

            # Mask out ignored dimensions in the full log_jacobian_diag
            log_jacobian_diag = self._mask_ignored_dimensions(mask, log_jacobian_diag)

            # Build distribution(s) on the sub-batch
            distr_increments, one_hot_dist_sub = self._make_increments_distribution(policy_sub)

            # Beta mixture log-probs
            # clamp to avoid Beta logprob(0.0 or 1.0) => -inf
            increments_sub_clamped = torch.clamp(
                increments_sub, min=self.epsilon, max=1.0 - self.epsilon
            )
            logprobs_increments_rel_sub = distr_increments.log_prob(increments_sub_clamped)
            # shape [M, n_dim]

            # Write back into full array
            logprobs_increments_rel[do_increments] = logprobs_increments_rel_sub

            # Now compute the one-hot dimension choice log-prob
            chosen_indices_sub = torch.argmax(increments_sub, dim=1)  # shape [M]
            effective_one_hot_sub = F.one_hot(chosen_indices_sub, num_classes=self.n_dim).float()
            one_hot_logp_sub = one_hot_dist_sub.log_prob(effective_one_hot_sub)
            one_hot_logp_full[do_increments] = one_hot_logp_sub

        # Combine all log-prob terms into final
        log_det_jacobian = torch.sum(log_jacobian_diag, dim=1)           # shape [n_states]
        sumlog_increments = torch.sum(logprobs_increments_rel, dim=1)    # shape [n_states]

        logprobs = logprobs_eos + sumlog_increments + log_det_jacobian + one_hot_logp_full
        return logprobs


    def _get_logprobs_backward(
        self,
        policy_outputs: torch.Tensor, 
        actions: torch.Tensor,
        mask: torch.Tensor,
        states_from: list,
    ) -> torch.Tensor:
        """
        Computes log probabilities of backward actions for a batch of states.
        """
        n_states = policy_outputs.shape[0]
        states_from_tensor = torch.as_tensor(states_from, dtype=self.float, device=self.device)

        # Initialize outputs
        logprobs_bts = torch.zeros(n_states, dtype=self.float, device=self.device)
        logprobs_increments_rel = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=self.device
        )
        log_jacobian_diag = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=self.device
        )
        one_hot_logp_full = torch.zeros(n_states, dtype=self.float, device=self.device)

        # Identify forced BTS vs. forced EOS
        is_eos = ~mask[:, 2]             # done => only EOS
        is_bts = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        is_bts_forced = ~mask[:, 1]      # mask[:,1] = False => must do BTS
        is_bts[is_bts_forced] = True

        # Bernoulli sample to see if we do BTS (except forced EOS states)
        do_bts = torch.logical_and(~is_bts_forced, ~is_eos)
        if torch.any(do_bts):
            # Check if the user took a BTS action => actions == states_from
            is_bts_sampled = torch.zeros_like(do_bts)
            is_bts_sampled[do_bts] = torch.all(
                actions[do_bts, :-1] == states_from_tensor[do_bts], dim=1
            )
            is_bts[is_bts_sampled] = True

            # Probability of BTS from the policy
            logits_bts = self._get_policy_source_logit(policy_outputs)[do_bts]
            distr_bts = Bernoulli(logits=logits_bts)

            logprobs_bts[do_bts] = distr_bts.log_prob(is_bts_sampled[do_bts].float())

        # Non-BTS, non-EOS => increments
        do_increments = torch.logical_and(~is_bts, ~is_eos)
        if torch.any(do_increments):
            # Sub-batch
            increments_sub = actions[do_increments, :-1]          # shape [M, n_dim]
            policy_sub = policy_outputs[do_increments]            # shape [M, policy_output_dim]
            states_from_sub = states_from_tensor[do_increments]   # shape [M, n_dim]

            # Convert absolute -> relative increments for backward
            # (since we store them as absolute in `actions`)
            increments_sub = self.absolute_to_relative_increments(
                states_from_sub,
                increments_sub,
                is_backward=True,
            )
            # clamp if needed (but typically we just clamp before Beta log-prob)
            # e.g. increments_sub = torch.clamp(increments_sub, self.epsilon, 1.0 - self.epsilon)

            # Compute Jacobian
            log_jacobian_diag_sub = torch.log(
                self._get_jacobian_diag(
                    states_from_sub,
                    is_backward=True,
                )
            )  # shape [M, n_dim]
            # Place into full
            log_jacobian_diag[do_increments] = log_jacobian_diag_sub

            # Mask ignored dimensions
            log_jacobian_diag = self._mask_ignored_dimensions(mask, log_jacobian_diag)

            # Build the Beta mixture + one-hot distribution
            distr_increments, one_hot_dist_sub = self._make_increments_distribution(policy_sub)

            increments_sub_clamped = torch.clamp(
                increments_sub, min=self.epsilon, max=1.0 - self.epsilon
            )
            logprobs_increments_sub = distr_increments.log_prob(increments_sub_clamped)

            # Place into full
            logprobs_increments_rel[do_increments] = logprobs_increments_sub
            logprobs_increments_rel = self._mask_ignored_dimensions(mask, logprobs_increments_rel)

            # Dimension choice
            chosen_indices_sub = torch.argmax(increments_sub, dim=1)  # shape [M]
            effective_one_hot_sub = F.one_hot(chosen_indices_sub, num_classes=self.n_dim).float()
            one_hot_logp_sub = one_hot_dist_sub.log_prob(effective_one_hot_sub)
            one_hot_logp_full[do_increments] = one_hot_logp_sub

        # Sum up everything
        log_det_jacobian = torch.sum(log_jacobian_diag, dim=1)
        sumlog_increments = torch.sum(logprobs_increments_rel, dim=1)

        # Combine
        logprobs = logprobs_bts + sumlog_increments + log_det_jacobian + one_hot_logp_full

        # If forced EOS, we set those logprobs = 0
        logprobs[is_eos] = 0.0

        return logprobs

    def _make_increments_distribution(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
    ) -> Tuple[MixtureSameFamily, OneHotCategorical]:
        """
        Constructs the mixture distribution for sampling relative increments (one per dimension)
        as well as the one-hot distribution for selecting which dimension is updated.

        Assumes that the policy_output contains the Beta mixture parameters in the first part and,
        additionally, a set of logits for the one-hot distribution (for instance, the last self.n_dim values).
        """

        # Extract parameters for the Beta mixture (for each dimension and component)
        mix_logits = self._get_policy_betas_weights(policy_outputs).reshape(
            -1, self.n_dim, self.n_comp
        )
        mix = Categorical(logits=mix_logits)

        alphas = self._get_policy_betas_alpha(policy_outputs).reshape(
            -1, self.n_dim, self.n_comp
        )
        alphas = self.beta_params_max * \
            torch.sigmoid(alphas) + self.beta_params_min

        betas = self._get_policy_betas_beta(policy_outputs).reshape(
            -1, self.n_dim, self.n_comp
        )
        betas = self.beta_params_max * \
            torch.sigmoid(betas) + self.beta_params_min
        beta_distr = Beta(alphas, betas)

        mixture = MixtureSameFamily(mix, beta_distr)

        # --- One-hot Distribution ---
        # not very elegant, but it works
        one_hot_logits = policy_outputs[:, -self.n_dim:]
        one_hot_distr = OneHotCategorical(logits=one_hot_logits)

        return mixture, one_hot_distr