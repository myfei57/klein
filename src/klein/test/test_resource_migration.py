# -*- test-case-name: klein.test.test_resource_migration -*-

"""
Tests for L{klein._resource_migration}.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import attr

from twisted.internet.defer import Deferred, fail, succeed
from twisted.trial.unittest import SynchronousTestCase
from twisted.web.iweb import IRequest

from .. import (
    FIELD_ADDED,
    FIELD_REMOVED,
    FIELD_RENAMED,
    FIELD_TRANSFORMED,
    VERSION_KEY,
    CorruptedStructureError,
    FieldConflictError,
    FieldMigrationError,
    FieldRecord,
    MigrationRegistryError,
    MigrationResult,
    MigrationStep,
    RenderingMigrations,
    ResourceMigrationError,
    UnreachableVersionError,
    migrate_resource_rendering,
)
from .._resource_migration import UNSET


@attr.frozen
class _DummyRequest:
    """
    A minimal stand-in for an L{IRequest} carrying identifiable context.
    """

    uri: bytes


def _dummyRequest(uri: bytes) -> IRequest:
    """
    Construct a dummy request typed as L{IRequest}.
    """
    return cast(IRequest, _DummyRequest(uri=uri))


def _migrate(
    testcase: SynchronousTestCase,
    structure: object,
    target: int,
    **kwargs: object,
) -> MigrationResult:
    """
    Run the migration command synchronously and return its result.
    """
    deferred: Deferred[MigrationResult] = migrate_resource_rendering(
        structure,
        target,
        **kwargs,  # type: ignore[arg-type]
    )
    testcase.assertIsInstance(deferred, Deferred)
    return testcase.successResultOf(deferred)


def _summary(result: MigrationResult) -> list[tuple[str, str]]:
    """
    Condense records to ``(action, field)`` pairs.
    """
    return [(record.action, record.field) for record in result.records]


class DefaultChainTests(SynchronousTestCase):
    """
    Tests for the shipped L{DEFAULT_RESOURCE_RENDERING_MIGRATIONS} chain.
    """

    def test_legacyStructureMigratedWithRecords(self) -> None:
        """
        Legacy ``status``/``content`` fields are renamed to ``code``/``body``
        and every field action is recorded.
        """
        structure = {
            VERSION_KEY: 1,
            "status": 404,
            "content": "hello",
            "trace": {"id": 7},
        }
        result = _migrate(self, structure, 2)

        self.assertEqual(
            result.structure,
            {
                VERSION_KEY: 2,
                "trace": {"id": 7},
                "code": 404,
                "body": "hello",
                "headers": {},
            },
        )
        self.assertEqual(
            _summary(result),
            [
                (FIELD_RENAMED, "code"),
                (FIELD_RENAMED, "body"),
                (FIELD_ADDED, "headers"),
            ],
        )
        self.assertEqual(
            result.records[0].old,
            404,
        )
        self.assertTrue(result.changed)

    def test_defaultsFollowExistingProjectSemantics(self) -> None:
        """
        Missing fields get the L{klein.Response} defaults (code 200, headers
        mapping, empty body), but supplied values are never overwritten.
        """
        result = _migrate(self, {VERSION_KEY: 1}, 2)
        self.assertEqual(result.structure["code"], 200)
        self.assertEqual(result.structure["body"], "")
        self.assertEqual(result.structure["headers"], {})

        supplied = {
            VERSION_KEY: 1,
            "code": 201,
            "body": "explicit",
            "headers": {"x": "y"},
        }
        result = _migrate(self, supplied, 2)
        self.assertEqual(
            result.structure,
            {
                VERSION_KEY: 2,
                "code": 201,
                "body": "explicit",
                "headers": {"x": "y"},
            },
        )
        self.assertEqual(_summary(result), [])

    def test_factoryDefaultsAreFresh(self) -> None:
        """
        Factory defaults produce an independent value on every migration.
        """
        first = _migrate(self, {VERSION_KEY: 1}, 2)
        second = _migrate(self, {VERSION_KEY: 1}, 2)
        self.assertEqual(first.structure["headers"], {})
        self.assertIsNot(
            first.structure["headers"],
            second.structure["headers"],
        )

    def test_unknownFieldsArePreserved(self) -> None:
        """
        Fields unknown to the migration chain survive unchanged, including
        their values and order.
        """
        unknown = {"nested": [1, 2, 3]}
        structure = {
            VERSION_KEY: 1,
            "status": 200,
            "custom": unknown,
            "other": 42,
        }
        result = _migrate(self, structure, 2)
        self.assertIs(result.structure["custom"], unknown)
        self.assertEqual(result.structure["other"], 42)
        self.assertEqual(result.structure["body"], "")
        self.assertEqual(
            list(result.structure),
            [
                VERSION_KEY,
                "custom",
                "other",
                "code",
                "headers",
                "body",
            ],
        )

    def test_renameToEqualExistingValueSucceeds(self) -> None:
        """
        A rename whose destination already holds the same value is not a
        conflict.
        """
        result = _migrate(
            self,
            {VERSION_KEY: 1, "status": 200, "code": 200},
            2,
        )
        self.assertEqual(result.structure["code"], 200)
        self.assertNotIn("status", result.structure)

    def test_inputStructureIsNeverMutated(self) -> None:
        """
        The returned structure is a new object; the input stays untouched.
        """
        structure = {VERSION_KEY: 1, "status": 200}
        result = _migrate(self, structure, 2)
        self.assertIsNot(result.structure, structure)
        self.assertEqual(structure, {VERSION_KEY: 1, "status": 200})


class IdempotencyTests(SynchronousTestCase):
    """
    A second migration run must be a no-op.
    """

    def test_secondRunProducesNoChanges(self) -> None:
        """
        Re-running the same migration yields no records and an equal
        structure.
        """
        structure = {
            VERSION_KEY: 1,
            "status": 404,
            "content": "hi",
            "custom": 1,
        }
        first = _migrate(self, structure, 2)
        second = _migrate(self, first.structure, 2)
        self.assertEqual(second.records, ())
        self.assertFalse(second.changed)
        self.assertEqual(second.structure, first.structure)
        self.assertEqual(second.from_version, 2)
        self.assertEqual(second.to_version, 2)

    def test_secondRunAfterMultiStepChain(self) -> None:
        """
        Idempotency holds after chains spanning registered steps and skipped
        versions.
        """
        migrations = RenderingMigrations(
            {
                1: MigrationStep(version=1, rename={"old": "new"}),
                2: MigrationStep(version=2, add={"added": 9}),
            },
        )
        first = _migrate(
            self,
            {VERSION_KEY: 1, "old": "v", "unknown": None},
            4,
            migrations=migrations,
        )
        self.assertEqual(first.structure[VERSION_KEY], 4)
        second = _migrate(
            self,
            first.structure,
            4,
            migrations=migrations,
        )
        self.assertEqual(second.records, ())
        self.assertEqual(second.structure, first.structure)

    def test_alreadyCurrentStructure(self) -> None:
        """
        Migrating to the current version is an empty migration.
        """
        structure = {VERSION_KEY: 2, "code": 200, "body": ""}
        result = _migrate(self, structure, 2)
        self.assertFalse(result.changed)
        self.assertEqual(result.records, ())
        self.assertEqual(result.structure, structure)


class SkippedVersionTests(SynchronousTestCase):
    """
    Structures may skip intermediate versions, with or without registered
    steps for those versions.
    """

    def test_chainsThroughRegisteredSteps(self) -> None:
        """
        Migrating v1 to v3 runs both the v1 and v2 steps and records each
        field with its actual step versions.
        """
        migrations = RenderingMigrations(
            {
                1: MigrationStep(version=1, rename={"old": "new"}),
                2: MigrationStep(
                    version=2,
                    add={"added": 7},
                    remove=frozenset({"gone"}),
                ),
            },
        )
        result = _migrate(
            self,
            {VERSION_KEY: 1, "old": "v", "gone": 1},
            3,
            migrations=migrations,
        )
        self.assertEqual(
            result.structure,
            {VERSION_KEY: 3, "new": "v", "added": 7},
        )
        self.assertEqual(
            _summary(result),
            [
                (FIELD_RENAMED, "new"),
                (FIELD_ADDED, "added"),
                (FIELD_REMOVED, "gone"),
            ],
        )
        self.assertEqual(result.records[0].from_version, 1)
        self.assertEqual(result.records[0].to_version, 2)
        self.assertEqual(result.records[1].from_version, 2)
        self.assertEqual(result.records[2].action, FIELD_REMOVED)
        self.assertEqual(result.records[2].old, 1)
        self.assertIs(result.records[2].new, UNSET)

    def test_versionsWithoutStepsAreIdentity(self) -> None:
        """
        Intermediate versions with no registered step just advance the
        version and preserve everything else.
        """
        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, rename={"old": "new"})},
        )
        result = _migrate(
            self,
            {VERSION_KEY: 1, "old": "v", "keep": True},
            4,
            migrations=migrations,
        )
        self.assertEqual(result.structure[VERSION_KEY], 4)
        self.assertEqual(result.structure["new"], "v")
        self.assertTrue(result.structure["keep"])
        self.assertEqual(_summary(result), [(FIELD_RENAMED, "new")])
        self.assertTrue(result.changed)

    def test_startingAfterTheLastRegisteredStep(self) -> None:
        """
        A structure already past every registered step reaches the target
        through identity steps alone.
        """
        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, rename={"old": "new"})},
        )
        result = _migrate(
            self,
            {VERSION_KEY: 2, "new": "v"},
            4,
            migrations=migrations,
        )
        self.assertEqual(result.structure[VERSION_KEY], 4)
        self.assertEqual(result.records, ())


class FieldConflictTests(SynchronousTestCase):
    """
    Conflicting fields abort the migration and preserve the original
    structure.
    """

    def test_renameConflictsWithExistingValue(self) -> None:
        """
        Renaming onto a differently-valued existing field raises
        L{FieldConflictError}.
        """
        structure = {
            VERSION_KEY: 1,
            "status": 404,
            "code": 500,
        }
        failure = self.failureResultOf(
            migrate_resource_rendering(structure, 2),
            FieldConflictError,
        )
        error = cast(FieldConflictError, failure.value)
        self.assertEqual(error.field, "code")
        self.assertEqual(error.incoming, 404)
        self.assertEqual(error.existing, 500)
        self.assertEqual(error.from_version, 1)
        self.assertEqual(error.to_version, 2)
        self.assertIs(error.structure, structure)
        self.assertEqual(
            structure,
            {VERSION_KEY: 1, "status": 404, "code": 500},
        )

    def test_conflictInLaterStepIsAtomic(self) -> None:
        """
        A conflict two steps in leaves the input at its original version
        with no partially migrated fields.
        """
        migrations = RenderingMigrations(
            {
                1: MigrationStep(version=1, rename={"a": "b"}),
                2: MigrationStep(version=2, rename={"b": "c"}),
            },
        )
        structure = {VERSION_KEY: 1, "a": 1, "c": 2}
        failure = self.failureResultOf(
            migrate_resource_rendering(structure, 3, migrations=migrations),
            FieldConflictError,
        )
        error = cast(FieldConflictError, failure.value)
        self.assertEqual(error.from_version, 2)
        self.assertIs(error.structure, structure)
        self.assertEqual(structure, {VERSION_KEY: 1, "a": 1, "c": 2})


class CorruptedInputTests(SynchronousTestCase):
    """
    Corrupt structures and invalid targets are rejected without mutating
    their input.
    """

    def assertCorrupt(self, structure: object, reason: str) -> None:
        """
        Migrating C{structure} raises L{CorruptedStructureError} quoting
        C{reason}.
        """
        failure = self.failureResultOf(
            migrate_resource_rendering(structure, 2),
            CorruptedStructureError,
        )
        error = cast(CorruptedStructureError, failure.value)
        self.assertIs(error.structure, structure)
        self.assertIn(reason, error.reason)

    def test_notAMapping(self) -> None:
        """
        Non-mapping input is corrupt.
        """
        for bad in (None, 42, ["version", 1], "version=1"):
            self.assertCorrupt(bad, "mapping")

    def test_missingVersion(self) -> None:
        """
        A structure without a version field is corrupt.
        """
        self.assertCorrupt({"code": 200}, "missing")

    def test_invalidVersion(self) -> None:
        """
        Non-integer, boolean, zero and negative versions are corrupt.
        """
        for bad in ("1", 1.0, True, False, 0, -1):
            self.assertCorrupt({VERSION_KEY: bad}, "positive integer")

    def test_invalidTargetVersion(self) -> None:
        """
        A non-positive-integer target fails the result with
        L{ResourceMigrationError}.
        """
        for bad in (0, -2, "2", 2.0, True):
            failure = self.failureResultOf(
                migrate_resource_rendering(
                    {VERSION_KEY: 1},
                    bad,  # type: ignore[arg-type]
                ),
                ResourceMigrationError,
            )
            self.assertIn("target version", str(failure.value))

    def test_downgradeIsUnreachable(self) -> None:
        """
        Migrating to an older version fails with
        L{UnreachableVersionError} and keeps the input.
        """
        structure = {VERSION_KEY: 3, "code": 200}
        failure = self.failureResultOf(
            migrate_resource_rendering(structure, 1),
            UnreachableVersionError,
        )
        error = cast(UnreachableVersionError, failure.value)
        self.assertIs(error.structure, structure)
        self.assertEqual(error.from_version, 3)
        self.assertEqual(error.to_version, 1)
        self.assertEqual(structure, {VERSION_KEY: 3, "code": 200})


class RegistryDefinitionTests(SynchronousTestCase):
    """
    Invalid L{MigrationStep} and L{RenderingMigrations} definitions fail
    fast with L{MigrationRegistryError}.
    """

    def test_invalidStepVersion(self) -> None:
        """
        Step versions must be positive integers.
        """
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=0)

    def test_stepsMustNotTouchVersionKey(self) -> None:
        """
        No step may add, remove, rename or transform the version field.
        """
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=1, add={VERSION_KEY: 1})
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=1, remove=frozenset({VERSION_KEY}))
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=1, rename={"x": VERSION_KEY})
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(
                version=1,
                transforms={VERSION_KEY: lambda v, s, r: v},
            )

    def test_renameRules(self) -> None:
        """
        Rename targets must be unique, must not chain and must differ from
        their source.
        """
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=1, rename={"a": "c", "b": "c"})
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=1, rename={"a": "b", "b": "c"})
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=1, rename={"a": "a"})

    def test_conflictingFieldOperations(self) -> None:
        """
        A field may not be both added/removed, renamed/removed or
        transformed/removed within one step.
        """
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(version=1, add={"x": 1}, remove=frozenset({"x"}))
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(
                version=1,
                rename={"x": "y"},
                remove=frozenset({"x"}),
            )
        with self.assertRaises(MigrationRegistryError):
            MigrationStep(
                version=1,
                remove=frozenset({"x"}),
                transforms={"x": lambda v, s, r: v},
            )

    def test_registryKeysMustMatchSteps(self) -> None:
        """
        The registry key for a step must equal the step's source version.
        """
        with self.assertRaises(MigrationRegistryError):
            RenderingMigrations({2: MigrationStep(version=1)})
        with self.assertRaises(MigrationRegistryError):
            RenderingMigrations({0: MigrationStep(version=0)})


def _upper(
    value: object,
    structure: Mapping[str, object],
    request: IRequest | None,
) -> str:
    """
    Synchronous transform used by the tests below.
    """
    return cast(str, value).upper()


class TransformTests(SynchronousTestCase):
    """
    Tests for synchronous and asynchronous field transforms.
    """

    def test_synchronousTransformIsRecorded(self) -> None:
        """
        A transform's old and new values are recorded per field.
        """
        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, transforms={"name": _upper})},
        )
        result = _migrate(
            self,
            {VERSION_KEY: 1, "name": "klein"},
            2,
            migrations=migrations,
        )
        self.assertEqual(result.structure["name"], "KLEIN")
        [record] = result.records
        self.assertEqual(record.action, FIELD_TRANSFORMED)
        self.assertEqual(record.field, "name")
        self.assertEqual(record.old, "klein")
        self.assertEqual(record.new, "KLEIN")

    def test_transformReceivesStructureAndRequest(self) -> None:
        """
        Transforms receive a read-only view of the structure and the
        request context.
        """
        seen: list[tuple[object, Mapping[str, object], IRequest | None]] = []

        def capture(
            value: object,
            structure: Mapping[str, object],
            request: IRequest | None,
        ) -> object:
            seen.append((value, dict(structure), request))
            return value

        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, transforms={"name": capture})},
        )
        request = _dummyRequest(b"/render")
        structure = {VERSION_KEY: 1, "name": "x", "other": 2}
        _migrate(
            self,
            structure,
            2,
            request=request,
            migrations=migrations,
        )
        value, view, passed = seen[0]
        self.assertEqual(value, "x")
        self.assertEqual(view, structure)
        self.assertIs(passed, request)

    def test_absentFieldsAreNotTransformed(self) -> None:
        """
        Transforms only run when the field exists, so they add no records
        for missing fields.
        """
        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, transforms={"name": _upper})},
        )
        result = _migrate(
            self,
            {VERSION_KEY: 1},
            2,
            migrations=migrations,
        )
        self.assertEqual(result.records, ())

    def test_asynchronousTransformSucceeds(self) -> None:
        """
        A transform returning a L{Deferred} is awaited and its result is
        recorded.
        """

        def later(
            value: object,
            structure: Mapping[str, object],
            request: IRequest | None,
        ) -> Deferred[str]:
            return succeed(cast(str, value) + "!")

        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, transforms={"name": later})},
        )
        result = _migrate(
            self,
            {VERSION_KEY: 1, "name": "hi"},
            2,
            migrations=migrations,
        )
        self.assertEqual(result.structure["name"], "hi!")
        self.assertEqual(result.records[0].action, FIELD_TRANSFORMED)

    def test_synchronousFailurePreservesOriginalAndRetries(self) -> None:
        """
        A raising transform aborts synchronously, and the command retries
        successfully with the same arguments.
        """
        attempts = []

        def flaky(
            value: object,
            structure: Mapping[str, object],
            request: IRequest | None,
        ) -> object:
            attempts.append(request)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            return "recovered"

        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, transforms={"name": flaky})},
        )
        request = _dummyRequest(b"/sync")
        structure = {VERSION_KEY: 1, "name": "old"}

        failure = self.failureResultOf(
            migrate_resource_rendering(
                structure,
                2,
                request=request,
                migrations=migrations,
            ),
            FieldMigrationError,
        )
        error = cast(FieldMigrationError, failure.value)
        self.assertFalse(error.asynchronous)
        self.assertIs(error.request, request)
        self.assertEqual(error.field, "name")
        self.assertIs(error.structure, structure)
        self.assertTrue(error.failure.check(RuntimeError))
        self.assertEqual(
            structure,
            {VERSION_KEY: 1, "name": "old"},
        )

        result = _migrate(
            self,
            structure,
            2,
            request=request,
            migrations=migrations,
        )
        self.assertEqual(result.structure["name"], "recovered")
        self.assertEqual(len(attempts), 2)
        self.assertIs(attempts[1], request)

    def test_asynchronousFailurePreservesRequestAndRetries(self) -> None:
        """
        A failed asynchronous transform preserves the request context and
        the untouched original, and the identical command retries cleanly.
        """
        attempts = []

        def flaky(
            value: object,
            structure: Mapping[str, object],
            request: IRequest | None,
        ) -> Deferred[str]:
            attempts.append(request)
            if len(attempts) == 1:
                return fail(RuntimeError("transient"))
            return succeed("migrated")

        migrations = RenderingMigrations(
            {1: MigrationStep(version=1, transforms={"note": flaky})},
        )
        request = _dummyRequest(b"/async")
        structure = {VERSION_KEY: 1, "note": "before"}

        failure = self.failureResultOf(
            migrate_resource_rendering(
                structure,
                2,
                request=request,
                migrations=migrations,
            ),
            FieldMigrationError,
        )
        error = cast(FieldMigrationError, failure.value)
        self.assertTrue(error.asynchronous)
        self.assertIs(error.request, request)
        self.assertIs(error.structure, structure)
        self.assertEqual(error.field, "note")
        self.assertEqual(error.from_version, 1)
        self.assertEqual(error.to_version, 2)
        self.assertTrue(error.failure.check(RuntimeError))
        self.assertIn("/async", str(error))
        self.assertEqual(
            structure,
            {VERSION_KEY: 1, "note": "before"},
        )

        result = _migrate(
            self,
            structure,
            2,
            request=request,
            migrations=migrations,
        )
        self.assertEqual(result.structure[VERSION_KEY], 2)
        self.assertEqual(result.structure["note"], "migrated")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts, [request, request])
        self.assertEqual(
            _summary(result),
            [(FIELD_TRANSFORMED, "note")],
        )

    def test_asynchronousFailureMidChainIsAtomic(self) -> None:
        """
        When a later step's transform fails, nothing from the earlier step
        is committed and the retry performs the whole chain.
        """
        attempts = []

        def flaky(
            value: object,
            structure: Mapping[str, object],
            request: IRequest | None,
        ) -> Deferred[str]:
            attempts.append(request)
            if len(attempts) == 1:
                return fail(RuntimeError("later step failed"))
            return succeed(cast(str, value) + "!")

        migrations = RenderingMigrations(
            {
                1: MigrationStep(version=1, rename={"old": "new"}),
                2: MigrationStep(version=2, transforms={"new": flaky}),
            },
        )
        request = _dummyRequest(b"/chain")
        structure = {VERSION_KEY: 1, "old": "x"}

        failure = self.failureResultOf(
            migrate_resource_rendering(
                structure,
                3,
                request=request,
                migrations=migrations,
            ),
            FieldMigrationError,
        )
        error = cast(FieldMigrationError, failure.value)
        self.assertTrue(error.asynchronous)
        self.assertEqual(error.field, "new")
        self.assertIs(error.structure, structure)
        self.assertEqual(structure, {VERSION_KEY: 1, "old": "x"})

        result = _migrate(
            self,
            structure,
            3,
            request=request,
            migrations=migrations,
        )
        self.assertEqual(
            result.structure,
            {VERSION_KEY: 3, "new": "x!"},
        )
        self.assertEqual(
            _summary(result),
            [(FIELD_RENAMED, "new"), (FIELD_TRANSFORMED, "new")],
        )
        self.assertEqual(len(attempts), 2)

    def test_removedFieldsAreRecordedInStableOrder(self) -> None:
        """
        Removed fields produce records in a deterministic order.
        """
        migrations = RenderingMigrations(
            {
                1: MigrationStep(
                    version=1,
                    remove=frozenset({"zeta", "alpha", "mid"}),
                ),
            },
        )
        result = _migrate(
            self,
            {VERSION_KEY: 1, "zeta": 1, "alpha": 2, "mid": 3},
            2,
            migrations=migrations,
        )
        self.assertEqual(
            _summary(result),
            [
                (FIELD_REMOVED, "alpha"),
                (FIELD_REMOVED, "mid"),
                (FIELD_REMOVED, "zeta"),
            ],
        )
        for record in result.records:
            self.assertIs(record.new, UNSET)

    def test_recordsAreFieldRecords(self) -> None:
        """
        The command reports L{FieldRecord} instances.
        """
        result = _migrate(self, {VERSION_KEY: 1, "status": 200}, 2)
        for record in result.records:
            self.assertIsInstance(record, FieldRecord)
