from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences import docstrings as doc
from edward.inferences.util import make_intercept
from edward.models.core import Node, trace


@doc.set_doc(
    arg_model=doc.arg_model[:-1],
    arg_align_latent=doc.arg_align_latent_monte_carlo[:-1],
    args=(doc.arg_align_data +
          doc.arg_state +
          doc.arg_independent_chain_ndims +
          doc.arg_target_log_prob +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_mcmc_programs=doc.notes_mcmc_programs,
    notes_conditional_inference=doc.notes_conditional_inference)
def metropolis_hastings(model,
                        proposal,
                        align_latent,
                        align_proposal,
                        align_data,
                        state=None,
                        independent_chain_ndims=0,
                        target_log_prob=None,
                        collections=None,
                        *args, **kwargs):
  """Metropolis-Hastings [@metropolis1953equation; @hastings1970monte].

  MH draws a sample from `proposal` given the last sample. The
  proposed sample is accepted with log-probability given by

  $\\text{ratio} =
        \log p(x, z^{\\text{new}}) - \log p(x, z^{\\text{old}}) -
        \log g(z^{\\text{new}} \mid z^{\\text{old}}) +
        \log g(z^{\\text{old}} \mid z^{\\text{new}})$

  where $p$ is the model's joint density over observed and latent
  variables, and $g$ is the proposal's density.

  Args:
  @{arg_model}
    proposal: function whose inputs are each state. It returns a new
      collection (Python list) of states given the inputs, $z'\sim
      g(z' \mid z)$.
  @{arg_align_latent}
    align_proposal:
  @{args}

  Returns:
  @{returns}

  #### Notes

  @{notes_mcmc_programs}

  @{notes_conditional_inference}

  #### Examples

  Consider the following setup.
  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  def proposal(mu):
    proposal_mu = Normal(loc=mu, scale=0.5, name="proposal/mu")
  ```
  In graph mode, build `tf.Variable`s which are updated via the Markov
  chain. The update op is fetched at runtime over many iterations.
  ```python
  qmu = tf.get_variable("qmu", initializer=1.)
  new_state, _ = ed.metropolis_hastings(
      model, proposal,
      state=qmu,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x_data" if name == "x" else None,
      x_data=x_data)
  qmu_update = qmu.assign(new_state)
  ```
  In eager mode, call the function at runtime, updating its inputs
  such as `state`.
  ```python
  qmu = 1.
  new_log_prob = None
  for _ in range(1000):
    new_state, new_log_prob = ed.metropolis_hastings(
        model, proposal,
        state=qmu,
        align_latent=lambda name: "qmu" if name == "mu" else None,
        align_proposal=lambda name: "proposal/mu" if name == "mu" else None,
        align_data=lambda name: "x_data" if name == "x" else None,
        target_log_prob=new_log_prob,
        x_data=x_data)
    qmu = new_state
  ```
  """
  def _target_log_prob_fn(*fargs):
    """Target's unnormalized log-joint density as a function of states."""
    posterior_trace = {state.name.split(':')[0]: Node(arg)
                       for state, arg in zip(states, fargs)}
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    model_trace = trace(model, intercept=intercept, *args, **kwargs)

    global inverse_align_latent
    inverse_align_latent = {}
    p_log_prob = 0.0
    for name, node in six.iteritems(model_trace):
      if align_latent(name) is not None or align_data(name) is not None:
        if align_latent(name) is not None:
          inverse_align_latent[align_latent(name)] = name
        rv = node.value
        p_log_prob += tf.reduce_sum(rv.log_prob(rv.value))
    return p_log_prob

  def _proposal_fn(*fargs):
    """Takes inputted states and returns (proposed states, log Hastings ratio).

    This implementation doesn't let `proposal take *args, **kwargs as
    input (i.e., it cannot be amortized). We also assume proposal
    returns same size and order as inputted states.
    """
    global inverse_align_latent
    # Build g(new | old): new states are drawn given old states as input.
    new_trace = trace(proposal, *fargs)
    new_states = []
    old_proposal_trace = {}
    for state, farg in zip(states, fargs):
      name = state.name.split(':')[0]
      new_state = new_trace[align_proposal(inverse_align_latent[name])].value
      new_state_name = new_state.name.split(':')[0]
      old_proposal_trace[new_state_name] = Node(farg)
      new_states.append(new_state)
    align_latent = lambda name: name if name in old_proposal_trace else None
    intercept = make_intercept(
        old_proposal_trace, align_data, align_latent, args, kwargs)
    # Build g(old | new): `value`s set to old states; new states are input.
    old_trace = trace(proposal, intercept=intercept, *new_states)
    old_states = [old_trace[align_proposal(
                    inverse_align_latent[state.name.split(':')[0]])].value
                  for state in states]
    old_states = []
    for state, farg in zip(states, fargs):
      name = state.name.split(':')[0]
      old_state = old_trace[align_proposal(inverse_align_latent[name])].value
      old_states.append(old_state)
    # Compute log p(old | new) - log p(new | old).
    log_hastings_ratio = 0.0
    for old_state, new_state in zip(old_states, new_states):
      log_hastings_ratio += tf.reduce_sum(old_state.log_prob(old_state.value))
      log_hastings_ratio -= tf.reduce_sum(new_state.log_prob(new_state.value))
    return new_states, log_hastings_ratio

  is_list_like = lambda x: isinstance(x, (tuple, list))
  maybe_list = lambda x: list(x) if is_list_like(x) else [x]
  states = maybe_list(state)

  out = _metropolis_hastings_kernel(
      target_log_prob_fn=_target_log_prob_fn,
      proposal_fn=_proposal_fn,
      state=state,
      independent_chain_ndims=independent_chain_ndims,
      target_log_prob=target_log_prob)
  return out


def _metropolis_hastings_kernel(target_log_prob_fn,
                                proposal_fn,
                                state,
                                independent_chain_ndims=0,
                                target_log_prob=None,
                                seed=None,
                                name=None):
  """Runs one iteration of Metropolis-Hastings.

  Args:
    state: `Tensor` or list of `Tensor`s each representing part of the
      chain's state.
    target_log_prob_fn: Python callable which takes an argument like
      `*state_tensors` (i.e., Python expanded) and returns the target
      distribution's (possibly unnormalized) log-density. Output has
      same shape as `target_log_prob`.
    proposal_fn: Python callable which takes an argument like
      `*state_tensors` (i.e., Python expanded) and returns a tuple of
      a list of proposed states of same size as input, and a log
      Hastings ratio `Tensor` of same shape as `target_log_prob`. If
      proposal is symmetric, set the second value to `None` to enable
      more efficient computation than explicitly supplying a tensor of
      zeros.
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leading dimensions (in each state) which index independent chains.
      Default value: `0` (i.e., only one chain).
    target_log_prob: `Tensor` of shape (independent_chain_dims,) if
      independent_chain_ndims == 1 else ().
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      @{tf.set_random_seed}
      for behavior.
    name: A name of the operation (optional).
  """
  is_list_like = lambda x: isinstance(x, (tuple, list))
  maybe_list = lambda x: list(x) if is_list_like(x) else [x]
  states = maybe_list(state)
  with tf.name_scope(name, "metropolis_hastings_kernel", states):
    with tf.name_scope("init"):
      if target_log_prob is None:
        target_log_prob = target_log_prob_fn(*states)
    proposed_state, log_hastings_ratio = proposal_fn(*states)
    proposed_states = maybe_list(proposed_state)
    if log_hastings_ratio is None:
      # Assume proposal is symmetric so log Hastings ratio is zero,
      # log p(old | new) - log p(new | old) = 0.
      log_hastings_ratio = 0.

    target_log_prob_proposed_states = target_log_prob_fn(*proposed_states)

    with tf.name_scope(
            "accept_reject",
            states + [target_log_prob, target_log_prob_proposed_states]):
      log_accept_prob = (target_log_prob_proposed_states - target_log_prob +
                         log_hastings_ratio)
      log_draws = tf.log(tf.random_uniform(tf.shape(log_accept_prob),
                                           seed=seed,
                                           dtype=log_accept_prob.dtype))
      is_proposal_accepted = log_draws < log_accept_prob
      next_states = [tf.where(is_proposal_accepted, proposed_state, state)
                     for proposed_state, state in zip(proposed_states, states)]
      next_log_prob = tf.where(is_proposal_accepted,
                               target_log_prob_proposed_states,
                               target_log_prob)
    maybe_flatten = lambda x: x if is_list_like(state) else x[0]
    next_state = maybe_flatten(next_states)
    proposed_state = maybe_flatten(proposed_states)
    return [
        next_state,
        is_proposal_accepted,
        next_log_prob,
        proposed_state,
    ]
