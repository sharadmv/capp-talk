import jax
import jax.numpy as jnp

def f(x):
  return jnp.cos(jnp.exp(jnp.sin(x)))

jaxpr = jax.make_jaxpr(f)(2.).jaxpr
print(jaxpr)

inverse_registry = {}

def eval_jaxpr_inverse(jaxpr, output):

  env = {}
  def read_env(atom):
    if type(atom) is jax.core.Literal: return atom.val
    return env[atom]
  def write_env(var, val):
    env[var] = val

  write_env(jaxpr.outvars[0], output)

  for eqn in jaxpr.eqns[::-1]:
    output = read_env(eqn.outvars[0])
    input = inverse_registry[eqn.primitive](output)
    write_env(eqn.invars[0], input)
  return read_env(jaxpr.invars[0])

inverse_registry[jax.lax.sin_p] = lambda x: jnp.arcsin(x)
inverse_registry[jax.lax.exp_p] = lambda x: jnp.log(x)
inverse_registry[jax.lax.cos_p] = lambda x: jnp.arccos(x)

print("Inverting f:", eval_jaxpr_inverse(jaxpr, f(-0.2)))

add_one_p = jax.core.Primitive("add_one")

add_one_p.def_impl(lambda x: x + 1)
add_one_p.def_abstract_eval(lambda x: jax.core.ShapedArray(x.shape, x.dtype))
inverse_registry[add_one_p] = lambda x: x - 1

print("============")
def g(x):
  return add_one_p.bind(f(x))

jaxpr = jax.make_jaxpr(g)(2.).jaxpr

print("Inverting g:", eval_jaxpr_inverse(jaxpr, g(-0.2)))
