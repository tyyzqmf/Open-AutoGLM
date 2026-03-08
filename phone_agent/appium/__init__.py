"""Appium device module — routes screenshot and input through a remote Appium session.

The Appium endpoint URL is read from the environment variable
``APPIUM_ENDPOINT_URL``.  Set it before launching Open-AutoGLM:

    export APPIUM_ENDPOINT_URL="https://devicefarm-interactive-global..."
    python main.py --device-type appium ...
"""

from phone_agent.appium.connection import AppiumConnection, list_devices
from phone_agent.appium.device import (
    back,
    double_tap,
    home,
    launch_app,
    long_press,
    swipe,
    tap,
)
from phone_agent.appium.input import clear_text, type_text
from phone_agent.appium.screenshot import get_current_app, get_screenshot

__all__ = [
    "AppiumConnection",
    "list_devices",
    "get_screenshot",
    "get_current_app",
    "tap",
    "double_tap",
    "long_press",
    "swipe",
    "back",
    "home",
    "launch_app",
    "type_text",
    "clear_text",
]


def detect_and_set_adb_keyboard(device_id=None):
    return "native"


def restore_keyboard(ime, device_id=None):
    pass
