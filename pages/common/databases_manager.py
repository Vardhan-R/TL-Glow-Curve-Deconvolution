from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

def executeSQL(sql: str, engine: Engine, commit: bool = False, params: list[dict] | None = None):
    with Session(engine) as session:
        res = session.execute(text(sql), params)

        if commit:
            session.commit()

    return res
