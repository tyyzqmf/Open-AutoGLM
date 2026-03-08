"""Text input via Appium WebDriver."""

from phone_agent.appium.connection import AppiumConnection


def _driver():
    return AppiumConnection().driver


def type_text(text: str, device_id: str | None = None) -> None:
    """Type text into the currently focused element.

    Args:
        text: Text to type.
        device_id: Ignored (kept for API compatibility).
    """
    print(f"[Appium] type_text: {text!r}")
    try:
        _driver().execute_script("mobile: type", {"text": text})
    except Exception:
        # Fallback: send keys to active element
        from appium.webdriver.common.appiumby import AppiumBy
        el = _driver().find_element(AppiumBy.XPATH, "//*[@focused='true']")
        el.send_keys(text)


def clear_text(device_id: str | None = None) -> None:
    """Clear text from the currently focused element.

    Args:
        device_id: Ignored (kept for API compatibility).
    """
    print("[Appium] clear_text")
    try:
        from appium.webdriver.common.appiumby import AppiumBy
        el = _driver().find_element(AppiumBy.XPATH, "//*[@focused='true']")
        el.clear()
    except Exception as e:
        print(f"[Appium] clear_text failed: {e}")
