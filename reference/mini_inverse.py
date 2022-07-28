import jax
import jax.numpy as jnp
import numpy as np

def f(x):
  y = jnp.sin(x)
  z = jnp.exp(y)
  return jnp.cos(z)

inverse_registry = {}

def eval_jaxpr(jaxpr, consts, input):
  
  env = {}

  def read_env(var):
    if isinstance(var, jax.core.Literal): return var.val
    return env[var]

  def write_env(var, val):
    env[var] = val

  write_env(jaxpr.invars[0], input)
  for const, constvar in zip(jaxpr.constvars, consts):
    write_env(constvar, const)
  
  for eqn in jaxpr.eqns:
    eqn_input = read_env(eqn.invars[0])
    eqn_output = eqn.primitive.bind(eqn_input)
    write_env(eqn.outvars[0], eqn_output)
  return read_env(jaxpr.outvars[0])

def eval_jaxpr_inverse(jaxpr, consts, output):
  
  env = {}

  def read_env(var):
    if isinstance(var, jax.core.Literal): return var.val
    return env[var]

  def write_env(var, val):
    env[var] = val

  write_env(jaxpr.outvars[0], output)
  for const, constvar in zip(jaxpr.constvars, consts):
    write_env(constvar, const)
  
  for eqn in jaxpr.eqns[::-1]:
    eqn_output = read_env(eqn.outvars[0])
    inverse_rule = inverse_registry[eqn.primitive]
    eqn_input = inverse_rule(eqn_output)
    write_env(eqn.invars[0], eqn_input)
  return read_env(jaxpr.invars[0])

inverse_registry[jax.lax.cos_p] = lambda x: jnp.arccos(x)
inverse_registry[jax.lax.exp_p] = lambda x: jnp.log(x)
inverse_registry[jax.lax.sin_p] = lambda x: jnp.arcsin(x)

closed_jaxpr = jax.make_jaxpr(f)(2.)
print("=======================")
print("Jaxpr:")
print(closed_jaxpr)
print("=======================")
jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts

out = f(0.5)
print("f^-1(f(0.5)) =", eval_jaxpr_inverse(jaxpr, consts, out))

print("======== Adding primitives ======")
add_one_p = jax.core.Primitive("add_one")

add_one_p.def_impl(lambda x: x + 1)
add_one_p.def_abstract_eval(lambda x: x)

inverse_registry[add_one_p] = lambda x: x - 1

def g(x):
  return add_one_p.bind(f(x))


closed_jaxpr = jax.make_jaxpr(g)(2.)
print("=======================")
print("Jaxpr:")
print(closed_jaxpr)
print("=======================")
jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts

out = g(0.5)
print("g^-1(g(0.5)) =", eval_jaxpr_inverse(jaxpr, consts, out))

print("=========Extension (ILDJ)=========")

def ildj(f, x):
  closed_jaxpr = jax.make_jaxpr(f)(x)
  jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.consts
  def f_inv(x):
    return eval_jaxpr_inverse(jaxpr, consts, x)
  jac = jax.jacrev(f_inv)(x)
  if jac.shape == ():
    return jnp.log(jnp.abs(jac))
  return jnp.linalg.slogdet(jax.jacrev(f_inv)(x))[1]

print("|J(g)(0.5)| =", ildj(g, g(0.5)))
