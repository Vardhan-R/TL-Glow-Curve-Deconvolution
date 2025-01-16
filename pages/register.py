from hashlib import sha256
from pages.common.cookies_manager import initCookies
from pages.common.databases_manager import executeSQL
from sqlalchemy import create_engine
import streamlit as st

def checkUsername(username: str) -> bool:
    curr_user_id = sha256(username.encode()).hexdigest()
    sql = "SELECT user_id FROM users_credentials"
    res = executeSQL(sql, st.session_state.engine)
    return all(row.user_id != curr_user_id for row in res)

cookies = initCookies()

# Ensure that the cookies are ready
if not cookies.ready():
    st.error("Cookies not initialised yet.")
    st.stop()

if cookies.get("user_id", "") != "":
    # Already logged in
    st.switch_page("./pages/main.py")

if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite+pysqlite:///pages/common/databases/server_side.db", echo=True)

col_1, col_2 = st.columns([4, 1])
with col_1:
    st.title("Register")
with col_2:
    st.page_link("./home.py", label="Back to Home", icon='🏠')

with st.form("register"):
    usrn = st.text_input("Username", max_chars=256, placeholder="Username", label_visibility="collapsed")
    pswd = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
    c_pswd = st.text_input("Confirm password", type="password", placeholder="Confirm password", label_visibility="collapsed")
    submitted = st.form_submit_button("Register")
    if submitted:
        if usrn != "":
            if checkUsername(usrn):
                if pswd == c_pswd:
                    # Store user credentials
                    sql = "INSERT INTO users_credentials (username, password_hash, user_id) VALUES (:usrn, :pswd_hash, :usr_id)"
                    params = [{"usrn": usrn, "pswd_hash": sha256(pswd.encode()).hexdigest(), "usr_id": sha256(usrn.encode()).hexdigest()}]
                    executeSQL(sql, st.session_state.engine, True, params)

                    # Create user's database
                    st.session_state.engine = create_engine(f"sqlite+pysqlite:///pages/common/databases/users/db_{cookies["user_id"]}.db", echo=True)

                    # Create table `workspaces`
                    sql = """
                    CREATE TABLE IF NOT EXISTS workspaces (
                        workspace_id INT AUTO_INCREMENT PRIMARY KEY,    -- Primary Key, Auto Increment
                        workspace_name VARCHAR(255) NOT NULL,           -- String, Not Null
                        file_path VARCHAR(255),                         -- String, Not Null
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- Datetime, Default: Current Timestamp
                    )
                    """
                    executeSQL(sql, st.session_state.engine, True, params)

                    # Create table `parameter_sets`
                    sql = """
                    CREATE TABLE IF NOT EXISTS parameter_sets (
                        parameter_set_id INT AUTO_INCREMENT PRIMARY KEY,                -- Primary Key, Auto Increment
                        workspace_id INT NOT NULL,                                      -- Foreign Key referencing workspaces(workspace_id)
                        parameter_set_name VARCHAR(255),                                -- Name of the parameter set
                        n INT NOT NULL,                                                 -- Integer parameter "n"
                        scale_factor FLOAT NOT NULL,                                    -- Scale factor
                        prop_coeff_tol FLOAT NOT NULL,                                  -- Proportionality coefficient tolerance
                        fitting_method VARCHAR(255) NOT NULL,                           -- Method for fitting (["COBYLA", "COBYQA", "SLSQP", "trust-constr"])
                        base_line BOOLEAN NOT NULL,                                     -- Base line
                        exp_base BOOLEAN,                                               -- Base type (constant or exponential)
                        offset FLOAT,                                                   -- Offset
                        exp_coeff FLOAT,                                                -- Exp. coeff.
                        exp_power FLOAT,                                                -- Exp. power
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,                  -- Datetime, Default: Current Timestamp
                        FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)  -- Foreign Key Constraint
                    )
                    """
                    executeSQL(sql, st.session_state.engine, True, params)

                    # Create table `parameter_arrays`
                    sql = """
                    CREATE TABLE IF NOT EXISTS parameter_arrays (
                        array_id INT AUTO_INCREMENT PRIMARY KEY,                                    -- Primary Key, Auto Increment
                        parameter_set_id INT NOT NULL,                                              -- Foreign Key referencing parameter_sets(parameter_set_id)
                        array_type VARCHAR(63) NOT NULL,                                            -- Type of array (e.g., b, I_m, T_m, E)
                        array_index INT NOT NULL,                                                   -- Index of the element in the array
                        array_value FLOAT NOT NULL,                                                 -- Value of the array element
                        FOREIGN KEY (parameter_set_id) REFERENCES parameter_sets(parameter_set_id)  -- Foreign Key Constraint
                    )
                    """
                    executeSQL(sql, st.session_state.engine, True, params)

                    st.success("Registered successfully!")
                    st.switch_page("./pages/login.py")
                else:
                    st.error("Passwords don't match.")
            else:
                st.error("Username is taken.")
        else:
            st.error("Enter a username.")
