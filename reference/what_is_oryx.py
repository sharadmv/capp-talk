import jax
import jax.numpy as jnp
from jax import random
from oryx.core.interpreters import log_prob as lp

import tensorflow_probability.substrates.jax as tfp

# TALK ABOUT ORYX INVERSE HERE

random_normal_p = jax.core.Primitive('random_normal')

random_normal_p.def_impl(lambda key: random.normal(key))
random_normal_p.def_abstract_eval(
    lambda _: jax.core.ShapedArray((), jnp.float32))

def f(key):
  return random_normal_p.bind(key)

print("=========Jaxpr=========")
print(jax.make_jaxpr(f)(random.PRNGKey(0)))
print("Evaluates to:", f(random.PRNGKey(0)))
print()

print("======Log prob interpreter====")

def _log_prob_rule(incells, outcells, **_):
  outcell, = outcells
  if not outcell.top():
    return incells, outcells, None
  outval = outcell.val
  return incells, outcells, tfp.distributions.Normal(0., 1.).log_prob(outval)

lp.log_prob_rules[random_normal_p] = _log_prob_rule
print("log p(-0.2) =", lp.log_prob(f)(-0.2))
print()

print("======Handling change of variable=====")

def g(key): # Log-normal distribution
  return jnp.exp(random_normal_p.bind(key))

print(jax.make_jaxpr(g)(random.PRNGKey(0)))
print("Evaluates to:", g(random.PRNGKey(0)))

print("log p(1.) =", lp.log_prob(g)(1.))
print("grad(log p)(1.) =", jax.grad(lp.log_prob(g))(1.))
print()

def h(key): # Logit-normal distribution
  return jax.nn.sigmoid(random_normal_p.bind(key))

print("log p(0.2) =", lp.log_prob(h)(0.2))
print("grad(log p)(0.2) =", jax.grad(lp.log_prob(h))(0.2))
print()

print("======Learnable distributions=====")

def f(key, w):
  return w + random_normal_p.bind(key)

w_grad = jax.grad(lp.log_prob(f), argnums=1)(2., -0.2)
print("grad_w(log p(x=1.2 | w)) =", w_grad)
