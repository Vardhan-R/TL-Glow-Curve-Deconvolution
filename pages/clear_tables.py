from pages.common.databases_manager import executeSQL
from sqlalchemy import create_engine
import streamlit as st

if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite+pysqlite:///pages/common/databases/server_side.db", echo=True)

if st.button("Clear Tables"):
    sql = "DELETE FROM users_credentials WHERE TRUE"
    executeSQL(sql, st.session_state.engine, True)
    st.success("Succesfully cleared the table(s)!")
