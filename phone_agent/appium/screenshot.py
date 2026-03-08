"""Screenshot capture via Appium WebDriver."""

from dataclasses import dataclass


@dataclass
class Screenshot:
    """Screenshot data container (mirrors desktop.screenshot.Screenshot)."""

    width: int
    height: int
    base64_data: str


def get_screenshot(device_id: str | None = None, timeout: int = 10) -> Screenshot:
    """Capture the current device screen via Appium.

    Args:
        device_id: Ignored (kept for API compatibility).
        timeout: Ignored (kept for API compatibility).

    Returns:
        Screenshot with base64-encoded PNG data.
    """
    import base64
    import io

    from phone_agent.appium.connection import AppiumConnection

    conn = AppiumConnection()
    driver = conn.driver

    b64 = driver.get_screenshot_as_base64()
    if not b64:
        raise RuntimeError("Appium 截图返回空数据")

    # Decode to get actual pixel dimensions
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        width, height = img.size
    except Exception:
        # Fallback: read from driver window size
        size = driver.get_window_size()
        width, height = size["width"], size["height"]

    return Screenshot(width=width, height=height, base64_data=b64)


def get_current_app(device_id: str | None = None) -> str:
    """Return the current foreground app package name.

    Args:
        device_id: Ignored (kept for API compatibility).

    Returns:
        App package name, or "Unknown" on failure.
    """
    from phone_agent.appium.connection import AppiumConnection

    try:
        conn = AppiumConnection()
        driver = conn.driver
        # Works on Android; may return None on other platforms
        app_id = driver.execute_script("mobile: getCurrentPackage")
        return app_id or "Unknown"
    except Exception:
        return "Unknown"
