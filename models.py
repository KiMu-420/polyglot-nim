import os
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    clerk_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    manifestos = relationship("Manifesto", back_populates="owner")

class Manifesto(Base):
    __tablename__ = "manifestos"
    id = Column(String, primary_key=True, default=lambda: os.urandom(8).hex())
    owner_id = Column(String, ForeignKey("users.clerk_id"))
    title = Column(String, default="My Manifesto")
    content = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    owner = relationship("User", back_populates="manifestos")