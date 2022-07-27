import jax
import jax.numpy as jnp

def f(x):
  y = jnp.sin(x)
  z = jnp.exp(y)
  return z * 2.

print("==========Printing a jaxpr=============")
print("Jaxpr:")
jaxpr = jax.make_jaxpr(f)(2.).jaxpr
print(jaxpr)
print()

print("==========The components of a jaxpr=============")
print("Invars:", jaxpr.invars)
print("Equation: <outvars> = <primitive> <invars>")
print("Eqns:", jaxpr.eqns)
print("Outvars:", jaxpr.outvars)
print()

print("==========Interpreting a jaxpr=============")
print(jax.core.eval_jaxpr(jaxpr, (), 2.))
print()

print("==========Staging out a jaxpr=============")
print(jax.jit(f).lower(2.).compiler_ir())

print("==========Custom interpreters for jaxpr=============")
print("Forward-mode autodiff (`jvp`):", jax.jvp(f, (2.,), (1.,))[1])
print("Reverse-mode autodiff (`vjp/grad`):", jax.grad(f)(2.))
print("Vectorized-map (`vmap`):", jax.vmap(f)(jnp.arange(2.)))
print("Compositions (`vmap(grad(f))`):", jax.vmap(jax.grad(f))(jnp.arange(2.)))
print()
