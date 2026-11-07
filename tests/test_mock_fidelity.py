"""Checks that the mock serialem module behaves like SerialEM's real Python module.

The command table (SPACEtomo/modules/dummy/sem_commands.json) is exported from SerialEM's MacroMasterList.h per
version; these tests keep the mock in line with it (drift) and pin down the module's calling conventions.
"""

import inspect

import pytest
import serialem as sem

# Functions of the real module that are not script commands, and mock-only hooks
MODULE_ONLY = {"ConnectToSEM", "ScriptIsInitialized", "PutImageInBuffer", "ReturnAllValuesAsTuples",
               "ReportNumModuleFuncs", "SetBufferImageTimeout"}
MOCK_ONLY = {"RegisterNavSaveCallback"}


def _commands(version):
    return sem._COMMANDS[version]["commands"]


# --- Drift between mock and SerialEM's command table ---

def test_mock_signatures_accept_serialem_arity(sem_version):
    """Every mocked command accepts as few and as many positional arguments as SerialEM allows."""

    too_narrow = []
    for name, func in sem._RAW.items():
        if name not in _commands(sem_version):
            continue
        min_args, keys = _commands(sem_version)[name]
        n_required = next((i for i, key in enumerate(keys) if key.islower()), len(keys))
        signature = inspect.signature(func)
        for n_args in {max(n_required, min_args), len(keys)}:
            try:
                signature.bind(*range(n_args))
            except TypeError:
                too_narrow.append(f"{name}{signature} cannot take {n_args} args (SerialEM keys {keys!r})")
    assert not too_narrow, "\n".join(too_narrow)


def test_mock_has_no_unknown_commands():
    """Capitalised mock functions must be real SerialEM commands (catches typos like CLearPersistentVars)."""

    unknown = sorted(name for name, obj in vars(sem).items()
                     if name[:1].isupper() and inspect.isfunction(obj)
                     and name not in sem._ALL_COMMANDS | MODULE_ONLY | MOCK_ONLY)
    assert not unknown


def test_commands_follow_emulated_version(sem_version):
    commands = _commands(sem_version)
    for name in sem._RAW:
        assert hasattr(sem, name) == (name in commands), name

    assert hasattr(sem, "GetLowDoseAreaParams") == (sem_version != "4.2")     # added in 4.3


# --- Calling conventions of the real module ---

@pytest.mark.parametrize("call, message", [
    (lambda: sem.ReportApertureSize("C"), "must be int, not str"),
    (lambda: sem.ReportApertureSize(1.0), "must be int, not float"),
    (lambda: sem.CameraProperties(1, 25.0), "must be int, not float"),
    (lambda: sem.GetVariable(3), "must be str, not int"),
    (lambda: sem.MoveStageTo("1", 2), "must be real number, not str"),
    (lambda: sem.ReportMag(1), "takes no arguments"),
    (lambda: sem.SetApertureSize(1), "takes exactly 2 arguments"),
    (lambda: sem.SetImageShift(x=1, y=2), "no keyword arguments"),
])
def test_wrong_arguments_raise_type_error(call, message):
    with pytest.raises(TypeError, match=message):
        call()


def test_number_arguments_accept_ints_and_numpy():
    import numpy as np
    sem.MoveStageTo(np.float32(1.5), 2)
    sem.SetApertureSize(np.int64(1), True)


def test_reported_values_are_floats_or_tuples():
    assert isinstance(sem.ReportMag(), tuple) and all(isinstance(val, float) for val in sem.ReportMag())
    assert isinstance(sem.ReportSpotSize(), float)
    assert sem.Delay(1) is None


def test_return_all_values_as_tuples():
    try:
        sem.ReturnAllValuesAsTuples(1)
        assert isinstance(sem.ReportSpotSize(), tuple)
    finally:
        sem.ReturnAllValuesAsTuples(0)


@pytest.mark.parametrize("version, date, expected", [("40200", "20250228", 1), ("40300", "", None), ("50000", "", 0),
                                                    ("40200", "20990101", 0)])     # version and date must both be reached
def test_is_version_at_least(sem_version, version, date, expected):
    if expected is None:
        expected = int(sem_version != "4.2")
    assert sem.IsVersionAtLeast(version, date) == expected
