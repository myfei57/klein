"""
Public API for Klein's two-phase change protocol for resource rendering.

Prepare versioned candidate plans with L{TwoPhaseResourceStore.plan}, then
complete the write with L{TwoPhaseResourceStore.commit}; see
L{klein._changes} for the protocol semantics.
"""

from ._changes import (
    DELETE,
    Applied,
    CandidatePlan,
    Change,
    ChangeApplicationError,
    ChangeProtocolError,
    CommitResult,
    DuplicateIdentifier,
    EmptyChanges,
    ResourceSnapshot,
    StaleSnapshot,
    TwoPhaseChangeAPI,
    TwoPhaseResourceStore,
)

__all__ = (
    "DELETE",
    "Applied",
    "CandidatePlan",
    "Change",
    "ChangeApplicationError",
    "ChangeProtocolError",
    "CommitResult",
    "DuplicateIdentifier",
    "EmptyChanges",
    "ResourceSnapshot",
    "StaleSnapshot",
    "TwoPhaseChangeAPI",
    "TwoPhaseResourceStore",
)
