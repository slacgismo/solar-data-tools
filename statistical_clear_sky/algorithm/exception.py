"""
Defines exceptions used in the context of this module "algorithm"
"""

class ProblemStatusError(Exception):
    """Error thrown when SCSF algorithm experiences something other than
    an 'optimal' problem status during one of
    the solve steps."""
