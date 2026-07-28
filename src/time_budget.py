"""Shared time-budget helper for chunked fw2 jobs.

Long steps (drift, stability) checkpoint per unit of work but have no natural
chunk boundary. This helper lets them exit cleanly at a checkpoint once a wall-
clock budget is hit, so fw2 `--repeat-until-exit 0` restarts them from the
checkpoint instead of a hard timeout kill that loses the in-flight unit.

Usage:
    from src.time_budget import TimeBudget
    budget = TimeBudget(args.time_budget_min, args.time_budget_exit_code)
    ...
    for unit in work:
        do_unit(unit); save_checkpoint()
        budget.check_and_exit(done=n_done, total=n_total)  # exits 75 if over
"""
import sys
import time


def continuation_required(*, expired, done=None, total=None):
    """Return whether an expired slice still has unfinished durable atoms."""
    if not expired:
        return False
    try:
        if done is not None and total is not None and float(done) >= float(total):
            return False
    except (TypeError, ValueError):
        pass
    return True


class TimeBudget:
    def __init__(self, minutes: float = 0, exit_code: int = 75):
        self.deadline = time.monotonic() + minutes * 60 if minutes and minutes > 0 else None
        self.exit_code = exit_code

    def expired(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def check_and_exit(self, done=None, total=None, log=print):
        """Exit at a checkpoint boundary if the budget is spent. Call AFTER a
        checkpoint save so resume continues cleanly."""
        if continuation_required(expired=self.expired(), done=done, total=total):
            where = f" at {done}/{total}" if done is not None else ""
            log(f"Time budget reached{where} — clean exit "
                f"{self.exit_code} (resume continues from checkpoint)")
            sys.stdout.flush()
            sys.exit(self.exit_code)


def add_time_budget_args(parser):
    parser.add_argument("--time-budget-min", type=float, default=0,
                        help="Exit cleanly at a checkpoint after N minutes "
                             "(0 disables); pair with fw2 --repeat-until-exit 0")
    parser.add_argument("--time-budget-exit-code", type=int, default=75,
                        help="Exit code on budget stop (75 for fw2 repeat)")
