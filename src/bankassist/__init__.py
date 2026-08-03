"""BankAssist AI — a governed, multi-agent banking assistant.

Kept deliberately free of imports so that ``import bankassist`` succeeds with no
environment configured (AC-L1-1). Configuration is resolved lazily via
``bankassist.config.get_settings``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
