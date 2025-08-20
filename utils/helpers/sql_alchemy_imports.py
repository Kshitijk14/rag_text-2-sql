# put data into sqlite db
from sqlalchemy import (
    create_engine,
    text,
    inspect,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)