"""Appium connection management — thread-local driver instances.

Each thread gets its own Appium WebDriver session, isolated from other threads.
Configuration is passed explicitly via ``connect()`` or falls back to
environment variables (``APPIUM_ENDPOINT_URL``, ``APPIUM_DEVICE_ARN``,
``APPIUM_SESSION_ID``).
"""

import os
import threading
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    device_id: str
    name: str = "Appium Device"


# Thread-local storage — each thread has its own _driver / _initialized.
_tls = threading.local()


class AppiumConnection:
    """Thread-local Appium WebDriver connection.

    In multi-threaded parallel runs each thread must operate its own device.
    ``AppiumConnection()`` always returns the **same Python object** (kept for
    API compatibility), but the underlying ``driver`` is stored per-thread in
    ``threading.local()``.

    Preferred usage (thread-safe)::

        conn = AppiumConnection()
        conn.connect(endpoint_url="https://...", device_arn="arn:...")
        conn.driver.get_screenshot_as_base64()

    Legacy usage (env-var, single-thread only)::

        os.environ["APPIUM_ENDPOINT_URL"] = "https://..."
        conn = AppiumConnection()
        conn._ensure_connected()
    """

    # Keep a single Python-level instance for backward compat — the actual
    # state lives in _tls so each thread is isolated.
    _instance: "AppiumConnection | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Explicit connect — preferred for multi-thread
    # ------------------------------------------------------------------

    def connect(
        self,
        endpoint_url: str,
        device_arn: str = "",
        session_id: str = "",
        new_command_timeout: int = 300,
    ):
        """Create an Appium driver for the **current thread**.

        Safe to call from multiple threads concurrently — each thread gets
        its own WebDriver instance connected to its own Device Farm session.
        """
        from appium import webdriver as appium_webdriver
        from selenium.webdriver.common.options import ArgOptions as AppiumOptions

        tid = threading.current_thread().name

        if session_id:
            print(f"[Appium][{tid}] Attach 到已有 session: {session_id}")
            driver = appium_webdriver.Remote(
                command_executor=endpoint_url,
                options=AppiumOptions(),
            )
            driver.session_id = session_id
            print(f"[Appium][{tid}] ✅ Attach 成功")
        else:
            options = AppiumOptions()
            if device_arn:
                options.set_capability("deviceName", device_arn)
            options.set_capability("newCommandTimeout", new_command_timeout)
            print(f"[Appium][{tid}] 连接 endpoint: ...{endpoint_url[-40:]}")
            driver = appium_webdriver.Remote(
                command_executor=endpoint_url,
                options=options,
            )
            print(f"[Appium][{tid}] ✅ 连接成功 (session={driver.session_id[:12]}...)")

        _tls.driver = driver
        _tls.initialized = True

    # ------------------------------------------------------------------
    # Legacy env-var path (backward compat, single-thread)
    # ------------------------------------------------------------------

    def _ensure_connected(self):
        if getattr(_tls, "initialized", False) and getattr(_tls, "driver", None) is not None:
            return

        endpoint_url = os.environ.get("APPIUM_ENDPOINT_URL", "")
        if not endpoint_url:
            raise RuntimeError(
                "APPIUM_ENDPOINT_URL environment variable is not set. "
                "Run setup_device_farm.py first and export the endpoint URL."
            )

        session_id = os.environ.get("APPIUM_SESSION_ID", "")
        device_arn = os.environ.get("APPIUM_DEVICE_ARN", "")
        self.connect(
            endpoint_url=endpoint_url,
            device_arn=device_arn,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Driver property
    # ------------------------------------------------------------------

    @property
    def driver(self):
        self._ensure_connected()
        return _tls.driver

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def quit(self):
        """Close the Appium session for the **current thread**."""
        drv = getattr(_tls, "driver", None)
        if drv is not None:
            try:
                drv.quit()
            except Exception:
                pass
            _tls.driver = None
            _tls.initialized = False


def list_devices() -> list[DeviceInfo]:
    """Return a single virtual device entry representing the Appium session."""
    endpoint_url = os.environ.get("APPIUM_ENDPOINT_URL", "")
    if not endpoint_url:
        return []
    return [DeviceInfo(device_id="appium", name="Device Farm Appium Device")]
