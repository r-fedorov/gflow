import torch
from gflownet.utils.common import tfloat
from gflownet.proxy.base import Proxy
import numpy as np
import joblib
import os
from robo_models import PretrainNet, RetrainNet


class Mars6DProxy(Proxy):
    """
    A custom proxy that computes an 'energy' from a 6D state.
    """

    def __init__(
        self,
        device,
        float_precision,
        reward_min=0.0,
        do_clip_rewards=False,
        **kwargs
    ):

        super().__init__(
            device=device,
            float_precision=float_precision,
            reward_min=reward_min,
            do_clip_rewards=do_clip_rewards,
            **kwargs
        )
        self._optimum = torch.tensor(
            5., device=self.device, dtype=self.float)
        # i know there is no such thing as an know optimum, but if dont initalise it,
        #  the evaluator will crash. This value dose not affet the performance of the model
        #  and only used for some evaluation stuff in the evalautor. The naximum in in Nature was
        #
        #
        input_dim = 6
        input_dim_retrain = input_dim + 3
        pre_model_dir = 'retraining_models/pre-model'
        pre_model_path = os.path.join(pre_model_dir, 'pretrain_model.pth')
        retrain_model_dir = 'retraining_models/model'
        retrain_model_path = os.path.join(
            retrain_model_dir, 'retrain_model.pth')
        scaler_path = os.path.join(retrain_model_dir, 'norm.pkl')
        target_scaler = joblib.load(scaler_path)

        self.pre_model = PretrainNet(input_dim=input_dim).to(device)
        self.pre_model.load_state_dict(
            torch.load(pre_model_path, map_location=device))
        self.pre_model.eval()
        self.retrain_model = RetrainNet(input_dim=input_dim_retrain).to(device)
        self.retrain_model.load_state_dict(
            torch.load(retrain_model_path, map_location=device))
        self.retrain_model.eval()
        self.target_scaler = joblib.load(scaler_path)

    def __call__(self, states):
        """
        states: can be a list of lists, a NumPy array, or a torch Tensor
                of shape [batch_size, 5] in 'proxy format'.

        Returns a torch Tensor of shape [batch_size], containing energies.
        """
        # Convert states to float tensor on correct device
        states = tfloat(states, float_type=self.float, device=self.device)
        # Assure shape is [batch_size, 7] 6 metals+outline value
        assert states.shape[1] == 7, "Expected states of shape [N, 7]."

        x = states[:, :-1]
        outline = states[:, 6]

        energy =  self.model_pred(x)

        return energy+outline

    def model_pred(self, data):
        """
        Predict the target for a single sample or batch of samples.

        If `data` is a 1D array-like of 6 elements, it is reshaped to (1, 6).
        If `data` is a 2D array-like of shape (n, 6), then n samples are processed.

        The function augments each sample with additional features computed via
        the pre-trained model, applies the final regression model, inverts the scaling,
        and returns the negative of the prediction.

        Args:
            data: array-like, shape (6,) or (n, 6)

        Returns:
            predictions: NumPy array of predictions (shape: (n,) for n samples)
        """
        # Ensure data is a NumPy array
        x_input = np.array(data, dtype=np.float32)

        # If input is a 1D array with 5 elements, reshape it as a batch of size 1.
        if x_input.ndim == 1:
            x_input = x_input.reshape(1, -1)

        # Convert the original features to a torch tensor
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(self.device)

        # Get additional features from the pre-trained model
        with torch.no_grad():
            pre_features_tensor = self.pre_model(x_tensor)

        # Convert to numpy and ensure shape is (n, 3)
        pre_features = pre_features_tensor.cpu()

        # Concatenate the original features (n,5) with the pre_features (n,3) to get (n,9)
        x_augmented = np.concatenate([x_input, pre_features], axis=1)

        # Convert the augmented input to a torch tensor for the final model
        x_final = torch.tensor(
            x_augmented, dtype=torch.float32).to(self.device)

        # Get the final model prediction
        with torch.no_grad():
            pred_tensor = self.retrain_model(x_final)
        pred_scaled = pred_tensor.cpu()  # shape will be (n,1)

        # Inverse transform the predictions. norm expects a 2D array.
        pred_original = self.target_scaler.inverse_transform(pred_scaled)

        # Return the negative of the predictions as a 1D array
        return torch.Tensor(pred_original.flatten())
