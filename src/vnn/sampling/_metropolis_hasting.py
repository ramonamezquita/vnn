import torch
from tqdm import trange

from ._probmodel import ProbabilisticModel
from ._proposal import Proposal


def accept(
    log_tgt_new: torch.Tensor,
    log_tgt_old: torch.Tensor,
    log_old_given_new: torch.Tensor,
    log_new_given_old: torch.Tensor,
) -> torch.BoolTensor:
    """Returns True if accepted."""

    log_u = torch.rand(()).log()
    log_ratio = (log_tgt_new + log_old_given_new) - (log_tgt_old + log_new_given_old)
    log_alpha = torch.minimum(torch.tensor(0.0), log_ratio)
    return log_u <= log_alpha  # accept if True.


@torch.no_grad()
def metropolis_hasting(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    model: ProbabilisticModel,
    proposal: Proposal,
    initial_guess: dict[str, torch.Tensor],
    n_iterations: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Run a Metropolis-Hastings MCMC sampler for a NN posterior distribution.

    Parameters
    ----------
    X : torch.Tensor
        Observed covariate values.

    y : torch.Tensor
        Observed target values.

    model : Model
        Model defining the forward function, log likelihood and the log prior.

    proposal : Proposal
        Proposal distribution defining q(new | old).

    initial_guess: dict[str, torch.Tensor
        Initial guess.

    n_iterations : int, default=100
        Number of iterations.

    Returns
    -------
    samples : list of dict[str, torch.Tensor]
    """

    samples: list[dict[str, torch.Tensor]] = []
    pbar = trange(n_iterations, desc="MH Sampling")

    W_old = initial_guess
    parameters_names = list(initial_guess)
    accepts_counter: int = 0

    for i in pbar:
        # Gen (new) proposal weights based on current weights.
        W_new = proposal.propose(W_old)

        # --------------------- #
        # Log of target density #
        # --------------------- #
        log_tgt_old = model.log_prob(W_old, X, y)
        log_tgt_new = model.log_prob(W_new, X, y)

        # ----------------------- #
        # Log of proposal density #
        # ----------------------- #
        # The log of the proposal is also evaluated on:
        # - old given new
        # - new given old.
        log_old_given_new = sum(
            proposal.log_prob(W_old[name], condition_on=W_new[name]).sum()
            for name in parameters_names
        )
        log_new_given_old = sum(
            proposal.log_prob(W_new[name], condition_on=W_old[name]).sum()
            for name in parameters_names
        )

        # ---------------- #
        # Accept or Reject #
        # ---------------- #
        if accept(
            log_tgt_new=log_tgt_new,
            log_tgt_old=log_tgt_old,
            log_new_given_old=log_new_given_old,
            log_old_given_new=log_old_given_new,
        ):
            accepts_counter += 1
            W_old = W_new
            samples.append(W_new)

        else:
            samples.append(W_old)

        if (i + 1) % 10 == 0:  # update every 10 steps.
            acc_rate = accepts_counter / (i + 1)
            pbar.set_postfix({"acc_rate": f"{acc_rate:.6f}"})

    return samples
