from fastapi import (
    FastAPI,
    UploadFile,
    Depends,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
    status,
)
from dotenv import load_dotenv
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User
from schemas import (
    UserCreate,
    UserLogin,
    Token,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    ProfileResponse,
    ProfileUpdate,
)
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from tensorflow.keras.models import load_model
from PIL import Image
from security_tokens import generate_email_token, verify_email_token
from email_utils import send_verification_email, send_reset_email
import numpy as np
import io
import os



from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
import os

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROFILE_DIR = UPLOAD_DIR / "profile"
SCAN_DIR = UPLOAD_DIR / "scans"

PROFILE_DIR.mkdir(parents=True, exist_ok=True)
SCAN_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


models = {
    "tomato": load_model("models/tomato/tomato_model.keras"),
    "bell_pepper": load_model("models/bell_pepper/retry_bellpepper_model.keras"),
    "potato": load_model("models/potato/new_potato_model.keras"),
}


tomato_classes = [
    "Bacterial_spot",
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "YellowLeaf_Curl_Virus",
    "mosaic_virus",
]

bell_pepper_classes = [
    "Bacterial_spot",
    "Healthy",
]

potato_classes = [
    "Early_blight",
    "Late_blight",
    "Healthy",
]

class_labels = {
    "tomato": tomato_classes,
    "bell_pepper": bell_pepper_classes,
    "potato": potato_classes,
}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    return user


@app.get("/")
def root():
    return {"message": "PlantHealer API is running"}


@app.post("/register", status_code=201)
def register(
    user: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    if len(user.password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password must not exceed 72 bytes",
        )

    hashed_password = get_password_hash(user.password)

    new_user = User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_verified=False,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = generate_email_token(user.email, "verify_email")
    background_tasks.add_task(send_verification_email, user.email, token)

    return {
        "message": "User registered successfully. Please check your email to verify your account."
    }


@app.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):
    data = verify_email_token(token, max_age=3600)

    if not data or data.get("purpose") != "verify_email":
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")

    user = db.query(User).filter(User.email == data["email"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_verified = True
    db.commit()

    return {"message": "Email verified successfully"}


@app.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not db_user.is_verified:
        raise HTTPException(
            status_code=403,
            detail="Please verify your email before logging in",
        )

    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == payload.email).first()

    if user:
        token = generate_email_token(payload.email, "reset_password")
        background_tasks.add_task(send_reset_email, payload.email, token)

    return {
        "message": "If an account with that email exists, a reset link has been sent."
    }


@app.post("/reset-password")
def reset_password(
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    data = verify_email_token(payload.token, max_age=3600)

    if not data or data.get("purpose") != "reset_password":
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user = db.query(User).filter(User.email == data["email"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if len(payload.new_password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password must not exceed 72 bytes",
        )

    user.hashed_password = get_password_hash(payload.new_password)
    db.commit()

    return {"message": "Password reset successfully"}


@app.get("/profile", response_model=ProfileResponse)
def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


@app.put("/profile", response_model=ProfileResponse)
def update_profile(
    payload: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.full_name is not None:
        current_user.full_name = payload.full_name

    if payload.phone is not None:
        current_user.phone = payload.phone

    if payload.profile_image is not None:
        current_user.profile_image = payload.profile_image

    db.commit()
    db.refresh(current_user)

    return current_user


@app.post("/predict")
async def predict(
    plant: str = Form(...),
    file: UploadFile = File(...),
):
    plant = plant.strip().lower()

    if plant not in models:
        raise HTTPException(status_code=400, detail="Invalid plant type")

    model = models[plant]

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((256, 256))  # adjust if your training size was different
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    labels = class_labels[plant]
    predicted_label = labels[predicted_class_index]

    return {
        "plant": plant,
        "prediction": predicted_label,
        "prediction_index": predicted_class_index,
        "confidence": confidence,
    }

@app.post("/upload/profile-photo")
async def upload_profile_photo(
    request: Request,
    uid: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    filename = f"{uid}_{uuid.uuid4().hex}{ext}"
    file_path = PROFILE_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = str(request.base_url) + f"uploads/profile/{filename}"

    return {
        "message": "Profile photo uploaded successfully",
        "image_url": image_url
    }


@app.post("/upload/scan-image")
async def upload_scan_image(
    request: Request,
    uid: str = Form(...),
    plant: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    ext = os.path.splitext(file.filename or "")[1] or ".jpg"
    filename = f"{uid}_{plant}_{uuid.uuid4().hex}{ext}"
    file_path = SCAN_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = str(request.base_url) + f"uploads/scans/{filename}"

    return {
        "message": "Scan image uploaded successfully",
        "image_url": image_url
    }
