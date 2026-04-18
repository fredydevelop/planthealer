# from fastapi import FastAPI, Depends, HTTPException, status
from fastapi import FastAPI, UploadFile,Depends, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User
from schemas import UserCreate, UserLogin, Token
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io


# FastAPI app
app = FastAPI()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# JWT settings
SECRET_KEY = "your-secret-key"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30



models = {
    "tomato": load_model("models/tomato/tomato_model.keras"),
    "bell_pepper": load_model("models/bell_pepper/retry_bellpepper_model.keras"),
    "potato": load_model("models/potato/new_potato_model.keras"),
}

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Hash password
def get_password_hash(password):
    return pwd_context.hash(password)

# Verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Create JWT token
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Register endpoint
@app.post("/register", status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if len(user.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password must not exceed 72 bytes"
        )

    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    # print("Plain:", user.password)
    return {"message": "User registered successfully"}

# Login endpoint
@app.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    # print("Hashed:", db_user.hashed_password)

    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict(
    plant: str = Form(...),
    file: UploadFile = File(...)
):
    plant = plant.strip().lower()
    print("PLANT RECEIVED:", plant)
    #Check plant type
    if plant not in models:
        raise HTTPException(status_code=400, detail="Invalid plant type")

    model = models[plant]

    #  Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "plant": plant,
        "prediction": predicted_class,
        "confidence": confidence
    }
    
# if __name__ == "__main__":

#     uvicorn.run(app, host="0.0.0.0", port=8000)