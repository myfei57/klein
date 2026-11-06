from typing import TYPE_CHECKING

from ._app import (
    Klein,
    KleinErrorHandler,
    KleinRenderable,
    KleinRouteHandler,
    handle_errors,
    route,
    run,
    subroute,
    url_for,
    urlFor,
)
from ._dihttp import RequestComponent, RequestURL, Response
from ._form import Field, FieldValues, Form, RenderableForm
from ._plating import Plating
from ._requirer import Requirer
from ._resource_migration import (
    DEFAULT_RESOURCE_RENDERING_MIGRATIONS,
    FIELD_ADDED,
    FIELD_REMOVED,
    FIELD_RENAMED,
    FIELD_TRANSFORMED,
    VERSION_KEY,
    CorruptedStructureError,
    FieldConflictError,
    FieldDefault,
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
from ._session import Authorization, SessionProcurer
from ._version import __version__ as _incremental_version

if TYPE_CHECKING:
    # Inform mypy of import shenanigans.
    from .resource import _SpecialModuleObject

    resource = _SpecialModuleObject(None)
else:
    from . import resource

__all__ = (
    "DEFAULT_RESOURCE_RENDERING_MIGRATIONS",
    "FIELD_ADDED",
    "FIELD_REMOVED",
    "FIELD_RENAMED",
    "FIELD_TRANSFORMED",
    "Klein",
    "KleinErrorHandler",
    "KleinRenderable",
    "KleinRouteHandler",
    "Plating",
    "Field",
    "FieldConflictError",
    "FieldDefault",
    "FieldMigrationError",
    "FieldRecord",
    "FieldValues",
    "Form",
    "CorruptedStructureError",
    "MigrationRegistryError",
    "MigrationResult",
    "MigrationStep",
    "RenderingMigrations",
    "RequestComponent",
    "RequestURL",
    "ResourceMigrationError",
    "Response",
    "RenderableForm",
    "SessionProcurer",
    "Authorization",
    "Requirer",
    "UnreachableVersionError",
    "VERSION_KEY",
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "handle_errors",
    "migrate_resource_rendering",
    "resource",
    "route",
    "run",
    "subroute",
    "urlFor",
    "url_for",
)


# Make it a str, for backwards compatibility
__version__ = _incremental_version.base()

__author__ = "The Klein contributors (see AUTHORS)"
__license__ = "MIT"
__copyright__ = f"Copyright 2011-2021 {__author__}"
