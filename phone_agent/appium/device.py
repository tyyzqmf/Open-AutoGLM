"""Device control via Appium WebDriver."""

import time

from phone_agent.appium.connection import AppiumConnection


def _driver():
    return AppiumConnection().driver


def tap(x: int, y: int, device_id: str | None = None, delay: float | None = None) -> None:
    """Tap at (x, y) in Appium logical coordinates."""
    import time
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    from selenium.webdriver.common.actions.pointer_input import PointerInput
    from selenium.webdriver.common.actions import interaction

    print(f"[Appium] tap ({x}, {y})")
    d = _driver()
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    actions = ActionBuilder(d, mouse=finger)
    actions.pointer_action.move_to_location(x, y)
    actions.pointer_action.pointer_down()
    actions.pointer_action.pause(0.1)
    actions.pointer_action.pointer_up()
    actions.perform()
    if delay:
        time.sleep(delay)


def double_tap(x: int, y: int, device_id: str | None = None, delay: float | None = None) -> None:
    """Double-tap at (x, y)."""
    import time
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    from selenium.webdriver.common.actions.pointer_input import PointerInput
    from selenium.webdriver.common.actions import interaction

    print(f"[Appium] double_tap ({x}, {y})")
    d = _driver()
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    for _ in range(2):
        actions = ActionBuilder(d, mouse=finger)
        actions.pointer_action.move_to_location(x, y)
        actions.pointer_action.pointer_down()
        actions.pointer_action.pause(0.05)
        actions.pointer_action.pointer_up()
        actions.perform()
        time.sleep(0.1)
    if delay:
        time.sleep(delay)


def long_press(
    x: int,
    y: int,
    duration_ms: int = 3000,
    device_id: str | None = None,
    delay: float | None = None,
) -> None:
    """Long-press at (x, y) for duration_ms milliseconds."""
    import time
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    from selenium.webdriver.common.actions.pointer_input import PointerInput
    from selenium.webdriver.common.actions import interaction

    print(f"[Appium] long_press ({x}, {y}) {duration_ms}ms")
    d = _driver()
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    actions = ActionBuilder(d, mouse=finger)
    actions.pointer_action.move_to_location(x, y)
    actions.pointer_action.pointer_down()
    actions.pointer_action.pause(duration_ms / 1000.0)
    actions.pointer_action.pointer_up()
    actions.perform()
    if delay:
        time.sleep(delay)


def swipe(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration_ms: int | None = None,
    device_id: str | None = None,
    delay: float | None = None,
) -> None:
    """Swipe from (start_x, start_y) to (end_x, end_y)."""
    import time
    from selenium.webdriver.common.actions.action_builder import ActionBuilder
    from selenium.webdriver.common.actions.pointer_input import PointerInput
    from selenium.webdriver.common.actions import interaction

    print(f"[Appium] swipe ({start_x},{start_y}) → ({end_x},{end_y})")
    d = _driver()
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    actions = ActionBuilder(d, mouse=finger)
    actions.pointer_action.move_to_location(start_x, start_y)
    actions.pointer_action.pointer_down()
    actions.pointer_action.pause((duration_ms or 500) / 1000.0)
    actions.pointer_action.move_to_location(end_x, end_y)
    actions.pointer_action.pointer_up()
    actions.perform()
    if delay:
        time.sleep(delay)


def back(device_id: str | None = None, delay: float | None = None) -> None:
    """Press the Android back button."""
    print("[Appium] back")
    _driver().press_keycode(4)  # KEYCODE_BACK
    if delay:
        time.sleep(delay)


def home(device_id: str | None = None, delay: float | None = None) -> None:
    """Press the Android home button."""
    print("[Appium] home")
    _driver().press_keycode(3)  # KEYCODE_HOME
    if delay:
        time.sleep(delay)


def launch_app(
    app_name: str, device_id: str | None = None, delay: float | None = None
) -> bool:
    """Launch an app by package name or friendly name.

    Args:
        app_name: Android package name (e.g. "com.example.app") or a
                  friendly name that maps to a package in apps.py.

    Returns:
        True if the app was activated, False otherwise.
    """
    from phone_agent.config.apps import APP_PACKAGES

    # Resolve friendly name → package list
    packages = APP_PACKAGES.get(app_name.lower(), [app_name])
    if isinstance(packages, str):
        packages = [packages]

    d = _driver()
    for pkg in packages:
        try:
            print(f"[Appium] launch_app: {pkg}")
            d.execute_script("mobile: activateApp", {"appId": pkg})
            time.sleep(2)
            if delay:
                time.sleep(delay)
            return True
        except Exception as e:
            print(f"[Appium] launch_app {pkg} failed: {e}")

    return False
