from streamlit_cookies_manager import EncryptedCookieManager
import time

def initCookies(wait_iters: int = 50, sleep_time: float = 0.1) -> EncryptedCookieManager:
	# Create a global cookie manager with a consistent prefix and password
	cookies = EncryptedCookieManager(prefix="EiOP_test", password="some_insecure_password")
	for _ in range(wait_iters):
		if cookies.ready():
			break
		time.sleep(sleep_time)
	return cookies
