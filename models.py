import torch
from flax import nnx
import jax
import jax.numpy as jnp
import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SymbolicExtractor(nnx.Module):
    def __init__(self, symbolic_shape, rng = nnx.Rngs(0)):
        super().__init__()
        self.linear1 = nnx.Linear(symbolic_shape, 16, rngs=rng)
        self.linear2 = nnx.Linear(16, 16, rngs=rng)
        self.linear3 = nnx.Linear(16, 16, rngs=rng)
    
    def __call__(self, symbolic_obs):
        x = jax.nn.relu(self.linear1(symbolic_obs))
        x = jax.nn.relu(self.linear2(x))
        x = self.linear3(x)

        x = jnp.reshape(x, (-1, 16))

        return x

class DomainMapExtractor(nnx.Module):
    def __init__(self, img_shape, rng = nnx.Rngs(0)):
        super().__init__()
        self.linear_len = 16 * img_shape[0] * img_shape[1]

        self.conv1 = nnx.Conv(1, 16, kernel_size=(3,3), rngs=rng)
        self.conv2 = nnx.Conv(16, 16, kernel_size=(3,3), rngs=rng)
        self.linear1 = nnx.Linear(self.linear_len, 16, rngs=rng)

    def __call__(self, domain_map):
        x = jnp.expand_dims(domain_map, axis=-1)
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jnp.reshape(x, (-1, self.linear_len))

        x = self.linear1(x)

        return x

class MultiTaxi(nnx.Module):
    def __init__(self, img_shape, symbolic_shape, n_actions, rng = nnx.Rngs(0)):
        super().__init__()
        self.domain_map_extractor = DomainMapExtractor(img_shape, rng)
        self.symbolic_extractor = SymbolicExtractor(symbolic_shape, rng)

        self.linear1 = nnx.Linear(32, 32, rngs=rng)
        self.linear2 = nnx.Linear(32, n_actions, rngs=rng)

    def __call__(self, symbolic_obs, domain_map):
        x1 = self.domain_map_extractor(domain_map)
        x2 = self.symbolic_extractor(symbolic_obs)

        x = jnp.concatenate([x1, x2], axis=-1)
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        raise NotImplementedError("This example is not implemented yet.")

        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "symbolic":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "domain_map":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)