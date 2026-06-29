from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import psycopg
import streamlit as st

from modules.cookie_manager import get_cookie_manager


def hash_password(password: str) -> str:
    encoded_password = password.encode()
    kdf = PBKDF2HMAC(
        algorithm=SHA256(),
        length=32,
        salt=st.secrets["auth_manager"]["salt"].encode(),
        iterations=480_000,
    )
    key = urlsafe_b64encode(kdf.derive(encoded_password))
    return key.decode()


def setup_database() -> bool:
    """Initializes the users table."""
    print("Setting up database...")
    try:
        conn = psycopg.connect(**st.secrets["db_config"])
    except Exception as e:
        print(f"Failed to get database connection: {e}")
        return False

    # Users table
    print("Creating users table...")
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK (role IN ('super_admin', 'admin', 'standard'))
                )
            """)

        print("Committing changes...")
        conn.commit()
        print("Changes committed successfully.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"Failed to create users table: {e}")
        return False
    finally:
        if conn:
            conn.close()


def register(username: str, password: str) -> bool:
    try:
        conn = psycopg.connect(**st.secrets["db_config"])
    except Exception as e:
        print(f"Failed to get database connection: {e}")
        return False

    try:
        with conn.cursor() as cursor:
            # Check if username already exists
            cursor.execute("SELECT 1 FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return False

            # Insert new user with hashed password and 'standard' role, and create an associated wallet
            password_hash = hash_password(password)
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, 'standard')", (username, password_hash))

        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Failed to register user: {e}")
        return False
    finally:
        if conn:
            conn.close()


def load_user(username: str, password: str) -> bool:
    try:
        conn = psycopg.connect(**st.secrets["db_config"])
    except Exception as e:
        print(f"Failed to get database connection: {e}")
        return False

    try:
        with conn.cursor() as cursor:
            # Fetch the stored password hash for the given username
            cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
            result = cursor.fetchone()
            if not result:
                print("Username not found.")
                return False

            stored_password_hash = result[0]
            if not stored_password_hash or hash_password(password) != stored_password_hash:
                print("Incorrect password.")
                return False

            cursor.execute(
                "SELECT 1 FROM users WHERE username = %s AND password_hash = %s",
                (username, hash_password(password)),
            )
            if not cursor.fetchone():
                print("Failed to load user data.")
                return False
    except Exception as e:
        print(f"Failed to load user data: {e}")
        return False
    finally:
        if conn:
            conn.close()

    cm = get_cookie_manager()
    if not cm.ready():
        print("Cookie manager is not ready. Please refresh the page.")
        return False

    cm["username"] = username
    cm.save()

    return True


def logout() -> None:
    cm = get_cookie_manager()
    if not cm.ready():
        print("Cookie manager is not ready. Please refresh the page.")
    cm["username"] = ""
    cm.save()
    st.session_state.uploaded = False
