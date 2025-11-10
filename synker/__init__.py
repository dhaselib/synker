# synker/__init__.py

from .scott import Scott
from .silverman import Silverman
from .kde import kde
from .kl_div import KL_div
from .ckl_div import CKL_div
from .synthetic import Synthetic
from .pinkde import Pinkde

__all__ = ["Scott", "Silverman", "kde", "KL_div","CKL_div", "Synthetic","Pinkde"]