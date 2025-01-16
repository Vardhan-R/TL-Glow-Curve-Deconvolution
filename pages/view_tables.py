from pages.common.databases_manager import executeSQL
from sqlalchemy import create_engine
import streamlit as st

if "engine" not in st.session_state:
    st.session_state.engine = create_engine("sqlite+pysqlite:///pages/common/databases/server_side.db", echo=True)

if st.button("View Tables"):
    sql = "SELECT * FROM users_credentials"
    res = executeSQL(sql, st.session_state.engine, True)
    for row in res:
        st.write(row)
