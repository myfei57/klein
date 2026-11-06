# -*- test-case-name: klein.test.test_resource_migration -*-

"""
Compatibility migration for legacy Klein resource-rendering structures.

A resource-rendering structure is the JSON-like C{dict} produced or consumed
when Klein renders a resource (for example the slot-data dictionary produced by
a L{klein.Plating} route).  Such a structure carries an integer schema version
under L{VERSION_KEY}; L{migrate_resource_rendering} upgrades a legacy structure
to a requested target version by running it through a registered chain of
L{MigrationStep} definitions.

The command obeys the following rules:

    - each field touched by a migration step is reported as a L{FieldRecord},

    - fields unknown to the migration chain are passed through untouched,

    - added fields use fill-in defaults: a value already present is never
      overwritten, mirroring the C{defaults.copy(); structure.update()}
      semantics of L{klein.Plating}, and the standard defaults (C{code=200},
      C{headers={}}, C{body=""}) mirror L{klein.Response},

    - the input structure is never mutated; when anything fails (corrupt
      input, a field conflict, or a failing asynchronous transform) the
      original structure is preserved on the raised exception,

    - migration is idempotent: migrating an already-current structure yields
      no field records and an equal structure.

Versions for which no L{MigrationStep} is registered are treated as identity
steps, so structures may skip intermediate versions.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from types import MappingProxyType
from typing import Any, TypeGuard

import attr

from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from twisted.web.iweb import IRequest

#: Key under which a resource-rendering structure records its schema version.
VERSION_KEY = "version"

#: Field was introduced by a migration step and filled with a default.
FIELD_ADDED = "added"

#: Field was removed by a migration step.
FIELD_REMOVED = "removed"

#: Field was renamed by a migration step.
FIELD_RENAMED = "renamed"

#: Field value was converted by a migration step transform.
FIELD_TRANSFORMED = "transformed"


#: A transform receives the field's current value, a read-only view of the
#: whole structure at the start of the step, and the request the migration is
#: running for (if any).  It may return a value directly or a L{Deferred}.
FieldTransform = Callable[..., Any]


def _mappingOfStrToObject(
    value: Mapping[str, object],
) -> dict[str, object]:
    """
    Converter for C{str -> object} mapping attributes.
    """
    return dict(value)


def _mappingOfStrToStr(value: Mapping[str, str]) -> dict[str, str]:
    """
    Converter for C{str -> str} mapping attributes.
    """
    return dict(value)


def _mappingOfTransforms(
    value: Mapping[str, FieldTransform],
) -> dict[str, FieldTransform]:
    """
    Converter for transform mapping attributes.
    """
    return dict(value)


def _mappingOfSteps(
    value: Mapping[int, "MigrationStep"],
) -> dict[int, "MigrationStep"]:
    """
    Converter for step registry attributes.
    """
    return dict(value)


def _setOfStr(value: Iterable[str]) -> frozenset[str]:
    """
    Converter for C{str} set attributes.
    """
    return frozenset(value)


class _Unset:
    """
    Sentinel marking the absence of an old or new value in a L{FieldRecord}.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


UNSET = _Unset()


class ResourceMigrationError(Exception):
    """
    Base class for errors raised while migrating resource-rendering
    structures.
    """


class MigrationRegistryError(ResourceMigrationError):
    """
    A collection of L{MigrationStep}s was defined incorrectly.
    """


class CorruptedStructureError(ResourceMigrationError):
    """
    The structure passed to L{migrate_resource_rendering} is not a valid
    versioned resource-rendering structure.
    """

    def __init__(self, structure: object, reason: str) -> None:
        self.structure = structure
        self.reason = reason
        super().__init__(
            f"corrupt resource-rendering structure: {reason}",
        )


class UnreachableVersionError(ResourceMigrationError):
    """
    The structure is newer than the requested target version, and migrations
    never downgrade.
    """

    def __init__(
        self,
        structure: Mapping[str, object],
        from_version: int,
        to_version: int,
    ) -> None:
        self.structure = structure
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(
            f"cannot migrate resource-rendering structure "
            f"v{from_version} down to v{to_version}",
        )


class FieldConflictError(ResourceMigrationError):
    """
    A migration step wanted to write a field that was already present in the
    structure with a different value.

    The original structure is left untouched.
    """

    def __init__(
        self,
        field: str,
        incoming: object,
        existing: object,
        from_version: int,
        to_version: int,
        structure: Mapping[str, object],
    ) -> None:
        self.field = field
        self.incoming = incoming
        self.existing = existing
        self.from_version = from_version
        self.to_version = to_version
        self.structure = structure
        super().__init__(
            f"conflicting values for {field!r} while migrating "
            f"v{from_version} to v{to_version}: "
            f"{incoming!r} vs {existing!r}",
        )


class FieldMigrationError(ResourceMigrationError):
    """
    A field transform raised (synchronously) or its L{Deferred} failed.

    The original structure is preserved unchanged on C{structure}, and the
    request context (when one was supplied to the migration command) is kept
    on C{request}, so the same command can safely be retried.
    """

    def __init__(
        self,
        field: str,
        from_version: int,
        to_version: int,
        structure: Mapping[str, object],
        failure: Failure,
        request: IRequest | None = None,
        asynchronous: bool = False,
    ) -> None:
        self.field = field
        self.from_version = from_version
        self.to_version = to_version
        self.structure = structure
        self.failure = failure
        self.request = request
        self.asynchronous = asynchronous
        super().__init__(
            f"could not migrate field {field!r} from v{from_version} to "
            f"v{to_version} for request {request!r}: "
            f"{failure.getErrorMessage()}",
        )


def _isPositiveInt(value: object) -> TypeGuard[int]:
    """
    Versions must be real positive integers, not booleans.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


@attr.frozen
class FieldDefault:
    """
    A default value for an added field that C{factory} creates afresh on
    every migration, mirroring L{attr.ib(factory=...)}.
    """

    factory: Callable[[], object]


@attr.frozen
class MigrationStep:
    """
    One schema-version step, upgrading a structure from C{version} to
    C{version + 1}.

    @param version: The source schema version of this step.

    @param add: Fields to fill in when absent.  Values are plain defaults, or
        L{FieldDefault} instances wrapping zero-argument factories.  Existing
        values are never overwritten.

    @param remove: Fields to drop.

    @param rename: Mapping of old field names to new field names.  If the new
        name is already present with a different value, L{FieldConflictError}
        is raised at migration time.

    @param transforms: Mapping of field names to L{FieldTransform} callables
        applied to the value when the field is present.  A transform may
        return a L{Deferred}.
    """

    version: int
    add: Mapping[str, object] = attr.ib(
        factory=dict,
        converter=_mappingOfStrToObject,
    )
    remove: frozenset[str] = attr.ib(
        factory=frozenset,
        converter=_setOfStr,
    )
    rename: Mapping[str, str] = attr.ib(
        factory=dict,
        converter=_mappingOfStrToStr,
    )
    transforms: Mapping[str, FieldTransform] = attr.ib(
        factory=dict,
        converter=_mappingOfTransforms,
    )

    def __attrs_post_init__(self) -> None:
        if not _isPositiveInt(self.version):
            raise MigrationRegistryError(
                f"migration step versions must be positive integers, "
                f"not {self.version!r}",
            )

        touched = (
            set(self.add)
            | self.remove
            | set(self.rename)
            | set(self.rename.values())
            | set(self.transforms)
        )
        if VERSION_KEY in touched:
            raise MigrationRegistryError(
                f"migration step v{self.version} must not touch the "
                f"{VERSION_KEY!r} field",
            )

        sources = set(self.rename)
        targets = list(self.rename.values())
        if len(set(targets)) != len(targets):
            raise MigrationRegistryError(
                f"migration step v{self.version} renames multiple fields "
                f"to the same name",
            )
        if sources.intersection(targets):
            raise MigrationRegistryError(
                f"migration step v{self.version} chains renames within a "
                f"single step",
            )
        for source, target in self.rename.items():
            if source == target:
                raise MigrationRegistryError(
                    f"migration step v{self.version} renames {source!r} to "
                    f"itself",
                )

        if self.remove.intersection(self.add):
            raise MigrationRegistryError(
                f"migration step v{self.version} both adds and removes the "
                f"same field",
            )
        if self.remove.intersection(sources):
            raise MigrationRegistryError(
                f"migration step v{self.version} both renames and removes "
                f"the same field",
            )
        if self.remove.intersection(self.transforms):
            raise MigrationRegistryError(
                f"migration step v{self.version} both transforms and removes "
                f"the same field",
            )


@attr.frozen
class RenderingMigrations:
    """
    A registry of L{MigrationStep}s, keyed by their source version.

    Versions without a registered step are migrated as identity steps, so
    skipped versions (including versions that made no schema change) need no
    declaration.
    """

    steps: Mapping[int, MigrationStep] = attr.ib(
        factory=dict,
        converter=_mappingOfSteps,
    )

    def __attrs_post_init__(self) -> None:
        for source, step in self.steps.items():
            if not _isPositiveInt(source):
                raise MigrationRegistryError(
                    f"migration steps must be keyed by positive integer "
                    f"versions, not {source!r}",
                )
            if source != step.version:
                raise MigrationRegistryError(
                    f"step registered for v{source} claims to be "
                    f"v{step.version}",
                )

    def step(self, source_version: int) -> MigrationStep | None:
        """
        Return the step registered for C{source_version}, or L{None} when
        that version is an identity step.
        """
        return self.steps.get(source_version)


@attr.frozen
class FieldRecord:
    """
    A record of what happened to a single field within one migration step.
    """

    field: str
    from_version: int
    to_version: int
    action: str
    old: object = UNSET
    new: object = UNSET


@attr.frozen
class MigrationResult:
    """
    The outcome of L{migrate_resource_rendering}.

    @ivar structure: The migrated structure.  On a no-op migration this is a
        fresh copy equal to the input.

    @ivar records: One L{FieldRecord} per field action, in execution order.

    @ivar from_version: The schema version the input structure declared.

    @ivar to_version: The schema version of the returned structure.
    """

    structure: dict[str, object]
    records: tuple[FieldRecord, ...]
    from_version: int
    to_version: int

    @property
    def changed(self) -> bool:
        """
        Whether the migration altered anything.
        """
        return self.from_version != self.to_version or bool(self.records)


#: Compatibility chain shipped with Klein.
#:
#: Version 1 resource-rendering structures used ``status``/``content``;
#: version 2 uses the L{klein.Response} names ``code``/``body`` and carries
#: an explicit ``headers`` mapping.  Defaults for missing fields follow
#: L{klein.Response}: ``code=200``, ``headers={}``, ``body=""``, and, as with
#: L{klein.Plating} defaults, they only fill in absent fields.
DEFAULT_RESOURCE_RENDERING_MIGRATIONS = RenderingMigrations(
    {
        1: MigrationStep(
            version=1,
            rename={"status": "code", "content": "body"},
            add={
                "code": 200,
                "headers": FieldDefault(dict),
                "body": "",
            },
        ),
    },
)


def _validateStructure(structure: object) -> int:
    """
    Validate the incoming structure and return its declared version.
    """
    if not isinstance(structure, Mapping):
        raise CorruptedStructureError(
            structure,
            "expected a mapping of field names to values",
        )
    if VERSION_KEY not in structure:
        raise CorruptedStructureError(
            structure,
            f"missing {VERSION_KEY!r} field",
        )
    version = structure[VERSION_KEY]
    if not _isPositiveInt(version):
        raise CorruptedStructureError(
            structure,
            f"{VERSION_KEY!r} must be a positive integer, not {version!r}",
        )
    return version


@inlineCallbacks
def _applyStep(
    step: MigrationStep,
    working: Mapping[str, object],
    records: list[FieldRecord],
    original: Mapping[str, object],
    request: IRequest | None,
) -> Generator[Deferred[Any], Any, dict[str, object]]:
    """
    Apply C{step} to C{working}, appending L{FieldRecord}s to C{records}.

    Returns a L{Deferred} firing with a brand-new structure dictionary; the
    input mapping is never modified.
    """
    from_version = step.version
    to_version = step.version + 1
    updated: dict[str, object] = dict(working)
    view = MappingProxyType(dict(working))

    for oldName, newName in step.rename.items():
        if oldName in updated:
            incoming = updated[oldName]
            if newName in updated and updated[newName] != incoming:
                raise FieldConflictError(
                    newName,
                    incoming,
                    updated[newName],
                    from_version,
                    to_version,
                    original,
                )
            updated.pop(oldName)
            updated[newName] = incoming
            records.append(
                FieldRecord(
                    field=newName,
                    from_version=from_version,
                    to_version=to_version,
                    action=FIELD_RENAMED,
                    old=incoming,
                    new=incoming,
                ),
            )

    for name, transform in step.transforms.items():
        if name not in updated:
            continue
        oldValue = updated[name]
        try:
            outcome = transform(oldValue, view, request)
        except Exception:
            raise FieldMigrationError(
                name,
                from_version,
                to_version,
                original,
                Failure(),
                request=request,
                asynchronous=False,
            )
        if isinstance(outcome, Deferred):
            try:
                newValue = yield outcome
            except Exception:
                raise FieldMigrationError(
                    name,
                    from_version,
                    to_version,
                    original,
                    Failure(),
                    request=request,
                    asynchronous=True,
                )
        else:
            newValue = outcome
        updated[name] = newValue
        records.append(
            FieldRecord(
                field=name,
                from_version=from_version,
                to_version=to_version,
                action=FIELD_TRANSFORMED,
                old=oldValue,
                new=newValue,
            ),
        )

    # Defaults only ever fill in absent fields, matching Plating semantics.
    for name, default in step.add.items():
        if name in updated:
            continue
        if isinstance(default, FieldDefault):
            value = default.factory()
        else:
            value = default
        updated[name] = value
        records.append(
            FieldRecord(
                field=name,
                from_version=from_version,
                to_version=to_version,
                action=FIELD_ADDED,
                old=UNSET,
                new=value,
            ),
        )

    for name in sorted(step.remove):
        if name not in updated:
            continue
        oldValue = updated.pop(name)
        records.append(
            FieldRecord(
                field=name,
                from_version=from_version,
                to_version=to_version,
                action=FIELD_REMOVED,
                old=oldValue,
                new=UNSET,
            ),
        )

    updated[VERSION_KEY] = to_version
    return updated


@inlineCallbacks
def migrate_resource_rendering(
    structure: object,
    target_version: int,
    *,
    request: IRequest | None = None,
    migrations: RenderingMigrations = DEFAULT_RESOURCE_RENDERING_MIGRATIONS,
) -> Generator[Deferred[Any], Any, MigrationResult]:
    """
    Migrate a legacy Klein resource-rendering C{structure} to
    C{target_version}.

    All outcomes, including validation failures, are delivered on the
    returned L{Deferred}.

    @param structure: A mapping carrying a positive integer C{version} field.

    @param target_version: The schema version to migrate to.  It must be at
        least the structure's current version; migrations never downgrade.

    @param request: The L{IRequest} being served, if any.  It is passed to
        asynchronous field transforms and preserved on
        L{FieldMigrationError} so a failed migration can be correlated with
        its request and safely retried.

    @param migrations: The L{RenderingMigrations} registry defining version
        steps.  Versions without a registered step are identity steps.

    @return: A L{Deferred} firing with a L{MigrationResult}, or failing with
        L{CorruptedStructureError}, L{ResourceMigrationError},
        L{UnreachableVersionError}, L{FieldConflictError}, or
        L{FieldMigrationError}.  The input structure is never mutated and is
        kept on the failure, so re-invoking this command with the same
        arguments is a safe retry.
    """
    if not _isPositiveInt(target_version):
        raise ResourceMigrationError(
            f"target version must be a positive integer, "
            f"not {target_version!r}",
        )

    current_version = _validateStructure(structure)
    assert isinstance(structure, Mapping)
    original: Mapping[str, object] = structure

    if current_version > target_version:
        raise UnreachableVersionError(
            original,
            current_version,
            target_version,
        )

    working: dict[str, object] = dict(original)
    records: list[FieldRecord] = []
    for source in range(current_version, target_version):
        step = migrations.step(source)
        if step is None:
            # A skipped version with no schema change.
            working[VERSION_KEY] = source + 1
            continue
        working = yield _applyStep(
            step,
            working,
            records,
            original,
            request,
        )

    return MigrationResult(
        structure=working,
        records=tuple(records),
        from_version=current_version,
        to_version=target_version,
    )
