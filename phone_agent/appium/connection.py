"""Appium connection management — reads endpoint from APPIUM_ENDPOINT_URL env var."""

import os
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device_id: str
    name: str = "Appium Device"


class AppiumConnection:
    """Singleton Appium WebDriver connection.

    The endpoint URL is read from the ``APPIUM_ENDPOINT_URL`` environment
    variable.  Set it before launching Open-AutoGLM:

        export APPIUM_ENDPOINT_URL="https://devicefarm-interactive-global..."
        python main.py --device-type appium "打开斗地主"
    """

    _instance: "AppiumConnection | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._driver = None
            cls._instance._initialized = False
        return cls._instance

    def _ensure_connected(self):
        if self._initialized and self._driver is not None:
            return

        endpoint_url = os.environ.get("APPIUM_ENDPOINT_URL", "")
        if not endpoint_url:
            raise RuntimeError(
                "APPIUM_ENDPOINT_URL environment variable is not set. "
                "Run setup_device_farm.py first and export the endpoint URL."
            )

        try:
            from appium import webdriver as appium_webdriver
            from selenium.webdriver.common.options import ArgOptions as AppiumOptions
        except ImportError as exc:
            raise ImportError(
                "Appium-Python-Client is required. "
                "Install with: pip install Appium-Python-Client>=3.0"
            ) from exc

        session_id = os.environ.get("APPIUM_SESSION_ID", "")
        if session_id:
            # Attach to existing session owned by the main process — no new session created
            print(f"[Appium] Attach 到已有 session: {session_id}")
            self._driver = appium_webdriver.Remote(
                command_executor=endpoint_url,
                options=AppiumOptions(),
            )
            # Override the session_id to attach instead of creating a new one
            self._driver.session_id = session_id
            self._initialized = True
            print("[Appium] ✅ Attach 成功（复用主进程 session）")
            return

        device_arn = os.environ.get("APPIUM_DEVICE_ARN", "")
        options = AppiumOptions()
        if device_arn:
            options.set_capability("deviceName", device_arn)
        options.set_capability("newCommandTimeout", 300)
        # Note: APK must be pre-installed when the Device Farm Remote Access Session
        # is created (via appArn in create_remote_access_session call).
        # Do NOT pass 'app' capability here — Device Farm Appium endpoint does not support it.

        print(f"[Appium] 连接 endpoint: {endpoint_url[:80]}...")
        self._driver = appium_webdriver.Remote(
            command_executor=endpoint_url,
            options=options,
        )
        self._initialized = True
        print("[Appium] ✅ 连接成功")

    @property
    def driver(self):
        self._ensure_connected()
        return self._driver

    def quit(self):
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
            self._initialized = False
        AppiumConnection._instance = None


def list_devices() -> list[DeviceInfo]:
    """Return a single virtual device entry representing the Appium session."""
    endpoint_url = os.environ.get("APPIUM_ENDPOINT_URL", "")
    if not endpoint_url:
        return []
    return [DeviceInfo(device_id="appium", name="Device Farm Appium Device")]
