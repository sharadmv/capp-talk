import jax
import jax.numpy as jnp
import numpy as np

def f(x):
  y = jnp.sin(x)
  z = jnp.exp(y)
  return jnp.cos(z)

print("=======================")
print("Jaxpr:")
print(jax.make_jaxpr(f)(2.))
print("=======================")

inverse_registry = {}

def eval_jaxpr_inverse(jaxpr, consts, *outputs):
  
  env = {}

  def read_env(var):
    if isinstance(var, jax.core.Literal):
      return var.val
    assert var in env, var
    return env[var]

  def write_env(var, val):
    env[var] = val

  jax.util.safe_map(write_env, jaxpr.outvars, outputs)
  jax.util.safe_map(write_env, jaxpr.constvars, consts)
  
  for eqn in jaxpr.eqns[::-1]:
    eqn_outputs = jax.util.safe_map(read_env, eqn.outvars)
    inverse_rule = inverse_registry[eqn.primitive]
    eqn_inputs = inverse_rule(*eqn_outputs)
    jax.util.safe_map(write_env, eqn.invars, eqn_inputs)
  return read_env(jaxpr.invars[0])

inverse_registry[jax.lax.cos_p] = lambda x: [jnp.arccos(x)]
inverse_registry[jax.lax.exp_p] = lambda x: [jnp.log(x)]
inverse_registry[jax.lax.sin_p] = lambda x: [jnp.arcsin(x)]

closed_jaxpr = jax.make_jaxpr(f)(2.)
jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts

out = f(0.5)
print("f^-1(f(0.5)) =", eval_jaxpr_inverse(jaxpr, consts, out))
