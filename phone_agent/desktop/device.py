"""Device control for desktop using pyautogui (cross-platform)."""

import subprocess
import time

import pyautogui

from phone_agent.config.apps_desktop import APP_PACKAGES_DESKTOP

# Disable pyautogui fail-safe (moving mouse to corner won't stop script)
pyautogui.FAILSAFE = False


def tap(
    x: int, y: int, device_id: str | None = None, delay: float | None = None
) -> None:
    """
    Click at coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.
        device_id: Ignored for desktop.
        delay: Optional delay after action in seconds.
    """
    print(f"[Desktop] 执行 Tap: ({x}, {y})")
    pyautogui.click(x, y)
    print(f"[Desktop] Tap 完成")
    if delay:
        time.sleep(delay)


def double_tap(
    x: int, y: int, device_id: str | None = None, delay: float | None = None
) -> None:
    """
    Double click at coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.
        device_id: Ignored for desktop.
        delay: Optional delay after action in seconds.
    """
    pyautogui.doubleClick(x, y)
    if delay:
        time.sleep(delay)


def long_press(
    x: int,
    y: int,
    duration_ms: int = 3000,
    device_id: str | None = None,
    delay: float | None = None,
) -> None:
    """
    Right click at coordinates (simulates long press).

    Args:
        x: X coordinate.
        y: Y coordinate.
        duration_ms: Ignored for desktop.
        device_id: Ignored for desktop.
        delay: Optional delay after action in seconds.
    """
    pyautogui.rightClick(x, y)
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
    """
    Drag from start to end coordinates.

    Args:
        start_x: Start X coordinate.
        start_y: Start Y coordinate.
        end_x: End X coordinate.
        end_y: End Y coordinate.
        duration_ms: Duration in milliseconds.
        device_id: Ignored for desktop.
        delay: Optional delay after action in seconds.
    """
    duration = (duration_ms / 1000.0) if duration_ms else 0.5
    pyautogui.moveTo(start_x, start_y)
    pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
    if delay:
        time.sleep(delay)


def back(device_id: str | None = None, delay: float | None = None) -> None:
    """
    Press Alt+Left (browser back).

    Args:
        device_id: Ignored for desktop.
        delay: Optional delay after action in seconds.
    """
    pyautogui.hotkey("alt", "left")
    if delay:
        time.sleep(delay)


def home(device_id: str | None = None, delay: float | None = None) -> None:
    """
    Press Windows/Command key (show desktop/start menu).

    Args:
        device_id: Ignored for desktop.
        delay: Optional delay after action in seconds.
    """
    pyautogui.press("win")
    if delay:
        time.sleep(delay)


def launch_app(
    app_name: str, device_id: str | None = None, delay: float | None = None
) -> bool:
    """
    Launch a desktop application.

    Args:
        app_name: Application name (must be in APP_PACKAGES_DESKTOP).
        device_id: Ignored for desktop.
        delay: Optional delay after action in seconds.

    Returns:
        True if app was launched, False otherwise.
    """
    if app_name not in APP_PACKAGES_DESKTOP:
        return False

    app_path = APP_PACKAGES_DESKTOP[app_name]

    try:
        subprocess.Popen(app_path, shell=True)
        time.sleep(2)  # Wait for app to start
        if delay:
            time.sleep(delay)
        return True
    except Exception:
        return False
