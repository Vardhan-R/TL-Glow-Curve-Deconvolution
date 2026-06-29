from streamlit_cookies_manager_ext import EncryptedCookieManager
import gc
import streamlit as st
import time


def get_cookie_manager(wait_iters: int = 50, sleep_time: float = 0.1) -> EncryptedCookieManager:
    print("Getting cookie manager...")
    try:
        # Create a global cookie manager with a consistent prefix and password
        cookie_manager = EncryptedCookieManager(password=st.secrets["cookie_manager"]["password"], prefix=st.secrets["cookie_manager"]["prefix"])
        print("Cookie manager created successfully.")
    except Exception as e:
        print(f"Failed to create cookie manager: {e}")
        for obj in gc.get_objects():
            if isinstance(obj, EncryptedCookieManager):
                if "_cookie_manager" in dir(obj):
                    if obj._cookie_manager._prefix == st.secrets["cookie_manager"]["prefix"]:
                        cookie_manager = obj
                        break
        print("Cookie manager found.")

    print("Checking if cookie manager is ready...")
    for _ in range(wait_iters):
        if cookie_manager.ready():
            print("Cookie manager is ready.")
            return cookie_manager
        print("Cookie manager is not ready. Waiting...")
        time.sleep(sleep_time)
    print("Cookie manager is not ready. Returning...")
    return cookie_manager
