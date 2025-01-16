from pages.common.databases_manager import executeSQL
from sqlalchemy import create_engine
import streamlit as st

if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite+pysqlite:///pages/common/databases/server_side.db", echo=True)

if st.button("Create Tables"):
    sql = """
    CREATE TABLE IF NOT EXISTS users_credentials (
        username VARCHAR(256),
        password_hash VARCHAR(256),
        user_id VARCHAR(256) PRIMARY KEY
    )
    """
    executeSQL(sql, st.session_state.engine, True)
    st.success("Succesfully created the table(s)!")
