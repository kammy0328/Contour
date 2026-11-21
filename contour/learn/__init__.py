"""Learning on this computer from the footage worked on (nothing is sent anywhere), and the rule for when it
may start."""


def allowed(project, cut=None) -> bool:
    """Learning from footage happens only in a project where it was switched on by hand (project.set_learn; off
    unless chosen, instruction 2026-11-20 §4-4), and not for a cut taken out of it: the crops and samples collected
    from a cut, and the automatic learning that starts from them (self-training, the model weights of the fusion,
    identity and status rules). What was learned before is still used, and nothing here touches the caches that
    editing and rendering need."""
    if project is None or not getattr(project, "learn", False) or not getattr(project, "learn_set", False):
        return False
    return cut is None or bool(getattr(cut, "learn", True))


def chosen_off(project, cut=None) -> bool:
    """Switched off by hand (the project, or this cut): what was collected from it is removed, as before. A project
    that never chose keeps what was collected under the old rule; nothing new is added to it."""
    if cut is not None and not getattr(cut, "learn", True):
        return True
    return bool(project is not None and getattr(project, "learn_set", False) and not getattr(project, "learn", False))
