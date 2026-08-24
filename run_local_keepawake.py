"""Keep the display safe while manga_translator runs.

This laptop (RTX 5090, nvlddmkm) deadlocks in the driver when Windows turns
the display off while a CUDA job is running: bugcheck 0x19C
DRVSETMONITORPOWERSTATE_HANG_nvlddmkm!rmapiLockAcquire -- see crash-report.md.
Holding ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED suppresses the idle
display-off and standby timers for exactly the lifetime of this process, and
the OS drops the hold automatically if the process dies, so a crash cannot
leave the power settings altered.

This file can also switch the external monitor to a lower refresh rate for
the duration of the run (MT_EXT_HZ=<n>, e.g. 60) and restore it afterwards.
DISABLED BY DEFAULT (MT_EXT_HZ unset/0) after 2026-08-13 evening: the first
live run with the auto-switch enabled wedged the display engine anyway
(green screen at 60 Hz, no DSC, ~46% link utilization, ended in bugcheck
0x9F), and an earlier same-day wedge happened at 4K@120 YCbCr422 10-bit
(also no DSC, ended in 0x133) — so the wedge is not bandwidth/DSC-gated and
each mode-set is just an extra dice roll on this rig. Keep the feature
opt-in for experiments only. MT_EXT_MONITOR selects the target monitor by
PnP id substring (default AUSAA34, the XG27UQDMS). The change is dynamic
(not persisted), so even a hard crash reverts it on reboot.

Usage:  python run_local_keepawake.py <manga_translator args...>
Equivalent to:  python -m manga_translator <args...>
"""

import ctypes
import os
import runpy
import sys
from ctypes import wintypes

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

ENUM_CURRENT_SETTINGS = -1
DISPLAY_DEVICE_ATTACHED_TO_DESKTOP = 0x00000001
DM_PELSWIDTH = 0x00080000
DM_PELSHEIGHT = 0x00100000
DM_DISPLAYFREQUENCY = 0x00400000
DISP_CHANGE_SUCCESSFUL = 0


class DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("DeviceName", ctypes.c_wchar * 32),
        ("DeviceString", ctypes.c_wchar * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", ctypes.c_wchar * 128),
        ("DeviceKey", ctypes.c_wchar * 128),
    ]


class DEVMODEW(ctypes.Structure):
    _fields_ = [
        ("dmDeviceName", ctypes.c_wchar * 32),
        ("dmSpecVersion", wintypes.WORD),
        ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD),
        ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD),
        ("dmPositionX", ctypes.c_long),
        ("dmPositionY", ctypes.c_long),
        ("dmDisplayOrientation", wintypes.DWORD),
        ("dmDisplayFixedOutput", wintypes.DWORD),
        ("dmColor", ctypes.c_short),
        ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short),
        ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short),
        ("dmFormName", ctypes.c_wchar * 32),
        ("dmLogPixels", wintypes.WORD),
        ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD),
        ("dmPelsHeight", wintypes.DWORD),
        ("dmDisplayFlags", wintypes.DWORD),
        ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD),
        ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD),
        ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD),
        ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD),
        ("dmPanningHeight", wintypes.DWORD),
    ]


def _find_external_display(user32, monitor_id_substr):
    r"""Return the adapter DeviceName (e.g. '\\.\DISPLAY1') whose attached
    monitor's PnP id contains monitor_id_substr, or None."""
    i = 0
    while True:
        dd = DISPLAY_DEVICEW()
        dd.cb = ctypes.sizeof(dd)
        if not user32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            return None
        if dd.StateFlags & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP:
            mon = DISPLAY_DEVICEW()
            mon.cb = ctypes.sizeof(mon)
            if user32.EnumDisplayDevicesW(dd.DeviceName, 0, ctypes.byref(mon), 0) \
                    and monitor_id_substr.lower() in mon.DeviceID.lower():
                return dd.DeviceName
        i += 1


def _set_refresh(user32, device_name, hz):
    """Switch device_name to hz keeping its current resolution. Returns the
    previous refresh rate on success, None on failure. Dynamic only (flags=0),
    so a crash/reboot reverts to the registry mode automatically."""
    dm = DEVMODEW()
    dm.dmSize = ctypes.sizeof(dm)
    if not user32.EnumDisplaySettingsW(device_name, ENUM_CURRENT_SETTINGS,
                                       ctypes.byref(dm)):
        return None
    prev = dm.dmDisplayFrequency
    if prev == hz:
        return None
    dm.dmDisplayFrequency = hz
    dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY
    res = user32.ChangeDisplaySettingsExW(device_name, ctypes.byref(dm),
                                          None, 0, None)
    if res != DISP_CHANGE_SUCCESSFUL:
        print(f"[keepawake] WARNING: refresh change to {hz} Hz failed "
              f"(code {res}); continuing at {prev} Hz", file=sys.stderr)
        return None
    print(f"[keepawake] external monitor {prev} Hz -> {hz} Hz for this run",
          file=sys.stderr)
    return prev


def main() -> None:
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    prev = kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    if prev == 0:
        # 0 means the call failed. Translation itself still works, so only
        # warn: the display may idle off mid-run, which risks the 0x19C crash.
        print(
            "[keepawake] WARNING: SetThreadExecutionState failed; "
            "display may sleep mid-run (crash risk, see crash-report.md)",
            file=sys.stderr,
        )

    # Optional (opt-in via MT_EXT_HZ) external refresh drop for the run.
    # Default off — see the module docstring for why.
    run_hz_raw = os.environ.get("MT_EXT_HZ", "0")
    monitor_id = os.environ.get("MT_EXT_MONITOR", "AUSAA34")
    device_name = None
    restore_hz = None
    try:
        run_hz = int(run_hz_raw)
    except ValueError:
        run_hz = 0
    if run_hz > 0:
        try:
            device_name = _find_external_display(user32, monitor_id)
            if device_name is not None:
                restore_hz = _set_refresh(user32, device_name, run_hz)
        except OSError as e:
            print(f"[keepawake] WARNING: refresh switch skipped ({e})",
                  file=sys.stderr)

    try:
        sys.argv = ["manga_translator", *sys.argv[1:]]
        runpy.run_module("manga_translator", run_name="__main__", alter_sys=True)
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        if device_name is not None and restore_hz is not None:
            try:
                _set_refresh(user32, device_name, restore_hz)
            except OSError as e:
                print(f"[keepawake] WARNING: refresh restore failed ({e}); "
                      f"set it back manually in Settings", file=sys.stderr)


if __name__ == "__main__":
    main()
