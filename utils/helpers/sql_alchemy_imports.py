# put data into sqlite db
from sqlalchemy import (
    create_engine,
    text,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)