"""DeepReview baseline (limitation-generation only).

A faithful, stripped-down re-implementation of the DeepReview multi-stage
"Review-with-Thinking" framework (Zhu et al., ACL 2025) that emits ONLY the
final list of limitations / weaknesses for a paper.

Cite as:
    Zhu, M., Weng, Y., Yang, L., & Zhang, Y. (2025).
    "DeepReview: Improving LLM-based Paper Review with Human-like Deep
     Thinking Process." Proceedings of ACL 2025.
"""

__version__ = "0.1.0" 