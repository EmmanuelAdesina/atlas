"""Atlas portfolio optimization layer.

Sequential bet selection and stake sizing for a single live match.
All pricing (EV, Kelly) is delegated to the existing live engine.
"""

from atlas.portfolio.sequential import (
    SequentialConfig,
    SequentialDecision,
    SequentialPortfolio,
)

__all__ = ["SequentialConfig", "SequentialDecision", "SequentialPortfolio"]
