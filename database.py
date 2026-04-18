from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# Replace with your PostgreSQL credentials
DB_USER = "postgres"
DB_PASSWORD = "codedcoder"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "plantfastapi_db"





DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables in PostgreSQL
Base.metadata.create_all(bind=engine)