from flax import nnx
import jax
import jax.numpy as jnp


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
