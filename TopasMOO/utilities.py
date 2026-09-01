"""
Helpful utilities for TopasMOO, called elsewhere in the codebase.
"""
import hashlib
import importlib.util
import logging
import os
import sys
from typing import Any

import numpy as np

from .exceptions import InvalidParameterError

logger = logging.getLogger(__name__)


def _import_from_absolute_path(fullpath):
    """Import a module from an absolute path under a name unique to that path.

    Loads ``GenerateTopasScripts``/``TopasObjectiveFunction`` from an arbitrary
    ``OptimizationDirectory`` using :func:`importlib.util.spec_from_file_location`.
    The module is registered in ``sys.modules`` under a name derived from the
    absolute path, so two optimizations pointing at different directories in the
    same process never collide on the bare ``GenerateTopasScripts`` name (which
    would otherwise silently reuse the first project's module).

    The script's own directory is temporarily prepended to ``sys.path`` while the
    module executes, so a user script can ``import`` helper modules that live
    alongside it in the optimization directory.

    :param fullpath: Absolute path to the ``.py`` file to load.

    :raises ModuleNotFoundError: If no file exists at ``fullpath``.
    """
    fullpath = os.fspath(fullpath)
    if not os.path.isfile(fullpath):
        raise ModuleNotFoundError(f"No module file found at {fullpath}")

    script_dir = os.path.dirname(os.path.abspath(fullpath))
    stem = os.path.splitext(os.path.basename(fullpath))[0]
    digest = hashlib.md5(os.path.abspath(fullpath).encode()).hexdigest()[:8]
    unique_name = f"_topasmoo_userscript_{stem}_{digest}"

    spec = importlib.util.spec_from_file_location(unique_name, fullpath)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not create import spec for {fullpath}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing so the module can import sibling helpers and so
    # the docstring's "registered under a unique name" guarantee actually holds.
    sys.modules[unique_name] = module
    sys.path.insert(0, script_dir)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique_name, None)
        raise
    finally:
        try:
            sys.path.remove(script_dir)
        except ValueError:
            pass
    return module


def _load_user_callable(module, attr_name, file_path, signature_hint):
    """Return ``module.attr_name``, raising a clear error if it isn't callable.

    The user project must define functions with these exact names and
    signatures; a missing or non-callable attribute is reported up front with
    the expected signature rather than failing later with a bare AttributeError.

    :param module: Imported user module.
    :param attr_name: Function name the contract requires (e.g.
        ``"TopasObjectiveFunction"``).
    :param file_path: Path the module was loaded from, for the error message.
    :param signature_hint: Human-readable expected signature.

    :raises InvalidParameterError: If the attribute is missing or not callable.
    """
    func = getattr(module, attr_name, None)
    if not callable(func):
        msg = (
            f"{file_path} must define a callable '{attr_name}'. "
            f"Expected signature: {signature_hint}."
        )
        logger.error(msg)
        raise InvalidParameterError(msg)
    return func

def _tensor_to_float(value: Any) -> float:
    """
    Convert a PyTorch tensor / NumPy / Python scalar to a Python float.

    Any tensor entering an f-string, log line, comparison against a Python
    float, or a NumPy-consuming function must go through this helper.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
if hasattr(value, "item"):
    try:
        return float(value.item())
    except (ValueError, RuntimeError, TypeError):
        pass
arr = np.asarray(value, dtype=float).reshape(-1)
if arr.size != 1 or not np.isfinite(arr[0]):
    raise TypeError(
        f"Expected a single finite scalar value; got shape {np.asarray(value).shape}."
    )
return float(arr[0])

