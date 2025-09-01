from fastapi import FastAPI, Depends, UploadFile, File, Form, Request, HTTPException, Cookie
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import Response
import bcrypt
from fastapi import Query
from starlette.middleware.sessions import SessionMiddleware
from typing import Optional
from sqlalchemy.orm import sessionmaker
import torch.nn as nn
from models import Base
from sqlalchemy.orm import joinedload
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from typing import List
from PIL import Image
import joblib
import os
import io
import sqlite3
import torch
from torchvision import models, transforms
from passlib.context import CryptContext
import uuid
import sys
import numpy as np
from pydantic import BaseModel, EmailStr, validator
import json

from fastapi import Path

from database import SessionLocal, init_db
from models import Prediction
from sqlalchemy import create_engine

from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

app = FastAPI()
FEEDBACK_FILE = "feedbacks.json"
app.add_middleware(SessionMiddleware, secret_key="your_secret_key_here")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing only!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@app.on_event("startup")
def on_startup():
    # Create tables (if they don't exist)
    Base.metadata.create_all(bind=engine)
    print("Database connected and tables created (if not exist)")

from torchvision import models
import torch.nn as nn

class_to_idx = {'high': 0, 'low': 1}
idx_to_class = {v: k for k, v in class_to_idx.items()}

urgency_model = models.resnet18(weights=None)  # pretrained weights not needed now
urgency_model.fc = nn.Linear(urgency_model.fc.in_features, 2)  # 2 classes
urgency_model.load_state_dict(torch.load("urgency_resnet18_finetuned.pth", map_location=torch.device("cpu")))
urgency_model.eval()

# -------------------- PATHS --------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # .../backend
ML_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "ml"))  # .../ml

# -------------------- LOAD ISSUE-TYPE (Keras) MODEL --------------------
h5_path = os.path.join(ML_DIR, "garbage_vs_pothole_cnn.h5")
savedmodel_path = os.path.join(ML_DIR, "garbage_vs_pothole_cnn")

if os.path.exists(h5_path):
    keras_model = load_model(h5_path)
elif os.path.exists(savedmodel_path):
    keras_model = load_model(savedmodel_path)
else:
    raise FileNotFoundError("Could not find issue type model")

# -------------------- LOAD URGENCY MODEL --------------------
# -------------------- INIT APP --------------------

# -------------------- STATIC PATHS --------------------
image_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "images"))
os.makedirs(image_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=image_dir), name="images")

static_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "static"))
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# -------------------- TEMPLATES --------------------
templates_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "templates"))
templates = Jinja2Templates(directory=templates_dir)

# -------------------- INIT DATABASE --------------------
init_db()

# -------------------- PASSWORD HASHING --------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# -------------------- DEPENDENCIES --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- USER DB INIT --------------------
def init_user_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            city TEXT NOT NULL,
            area TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

init_user_db()

# -------------------- EMPLOYEE DB INIT --------------------

def init_employee_db():
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            department TEXT NOT NULL,
            city TEXT NOT NULL,
            area TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_employee_db()


# -------------------- SCHEMAS --------------------
class ComplaintRequest(BaseModel):
    complaint_text: str

class UserIn(BaseModel):
    username: str
    password: str
    city: str
    area: str

class UserLogin(BaseModel):
    username: str
    password: str

class EmployeeSignup(BaseModel):
    username: str
    password: str
    govt_id: str
    department: str
    city: str
    area: str

class EmployeeLogin(BaseModel):
    username: str
    password: str
# -------------------- HELPERS --------------------

def classify_issue_keras(image_data: Image.Image):
    """Return (issue_type, confidence).
    issue_type in {"Garbage", "Potholes"}.
    Works with both sigmoid(1) and softmax(2) outputs.
    """
    img_resized = image_data.resize((224, 224))
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = keras_model.predict(img_array)
    print("[DEBUG] TF raw prediction:", prediction)
    sys.stdout.flush()

    # Sigmoid case: shape (1, 1)
    if len(prediction.shape) == 2 and prediction.shape[1] == 1:
        prob = float(prediction[0][0])
        # IMPORTANT: assumes label 1 == Potholes, 0 == Garbage
        issue_type = "Potholes" if prob >= 0.5 else "Garbage"
        confidence = prob if issue_type == "Potholes" else (1.0 - prob)
    else:
        # Softmax case: shape (1, 2)
        # Folder order was mentioned as Garbage first, then Potholes
        classes = ["Garbage", "Potholes"]
        pred_class = int(np.argmax(prediction[0]))
        issue_type = classes[pred_class]
        confidence = float(np.max(prediction[0]))

    return issue_type, confidence

def classify_urgency_torch(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor_img = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = urgency_model(tensor_img)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return idx_to_class[pred]

# -------------------- ROUTES --------------------
@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/login")
def login(user: UserLogin):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password, city, area FROM users WHERE username=?", (user.username,))
    row = cursor.fetchone()
    conn.close()

    if row and pwd_context.verify(user.password, row[0]):
        response = JSONResponse(content={
            "username": user.username,
            "city": row[1],
            "area": row[2]
        })
        # cookies valid for 1 day
        response.set_cookie(key="username", value=user.username, max_age=86400)
        response.set_cookie(key="city", value=row[1], max_age=86400)
        response.set_cookie(key="area", value=row[2], max_age=86400)
        return response
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/set_location")
def set_location(response: Response, city: str = Form(...), area: str = Form(...)):
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="city", value=city, max_age=86400)
    response.set_cookie(key="area", value=area, max_age=86400)
    return response


# -------------------- DASHBOARD --------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, city: str = Cookie(None), area: str = Cookie(None), db: Session = Depends(get_db)):
    if not city or not area:
        return RedirectResponse(url="/auth?next=/dashboard")
    complaints = db.query(Prediction).filter(Prediction.city == city, Prediction.area == area).all()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "predictions": complaints,
        "city": city,
        "area": area
    })
def get_complaints(db: Session):
    complaints = db.query(Prediction).all()
    return complaints

@app.get("/admin_dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    department: str = Cookie(None),
    city: str = Cookie(None),
    area: str = Cookie(None),
    db: Session = Depends(get_db)
):
    if not (department and city and area):
        return RedirectResponse("/admin", status_code=302)

    complaints = (
        db.query(Prediction)
        .filter_by(
            issue_type=department,
            area=area,
            city=city
        )
        .all()
    )

    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "issues": complaints
    })

# -------------------- APIS FOR DATA --------------------
@app.get("/get_predictions")
def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).all()
    return [
        {
            "id": pred.id,
            "complaint_text": pred.complaint_text,
            "issue_type": pred.issue_type,
            "urgency": pred.urgency,
            "timestamp": str(pred.timestamp),
            "image_path": pred.image_path
        }
        for pred in predictions
    ]

@app.get("/complaints", response_class=JSONResponse)
def get_complaints(db: Session = Depends(get_db)):
    complaints = db.query(Prediction).all()
    return [
        {
            "id": c.id,
            "complaint_text": c.complaint_text,
            "issue_type": c.issue_type,
            "urgency": c.urgency,
            "timestamp": str(c.timestamp),
            "image_path": c.image_path
        }
        for c in complaints
    ]

# -------------------- AUTH ROUTES --------------------
@app.post("/signup")
def signup(user: UserIn):
    hashed_password = pwd_context.hash(user.password)
    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, hashed_password, city, area) VALUES (?, ?, ?, ?)",
                       (user.username, hashed_password, user.city, user.area))
        conn.commit()
        conn.close()
        return {"message": "User created successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already taken")

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("username")
    response.delete_cookie("city")
    response.delete_cookie("area")
    return response

class EmployeeSignup(BaseModel):
    employee_id: str
    password: str
    department: str
    city: str
    area: str

@app.post("/employee_signup")
async def employee_signup(data: EmployeeSignup):
    hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO employees (employee_id, hashed_password, department, city, area)
            VALUES (?, ?, ?, ?, ?)
        """, (data.employee_id, hashed_password, data.department, data.city, data.area))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Employee ID already exists")
    finally:
        conn.close()

    return {"message": "Signup successful"}

class LoginData(BaseModel):
    employee_id: str
    password: str

# Function to fetch employee from DB
def get_employee(employee_id: str):
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, hashed_password, department, city, area FROM employees WHERE employee_id = ?", (employee_id,))
    row = cursor.fetchone()
    conn.close()
    return row  # None if not found

@app.post("/employee_login")
async def employee_login(login_data: LoginData, response: Response):
    employee = get_employee(login_data.employee_id)
    if not employee:
        raise HTTPException(status_code=401, detail="Invalid employee ID or password")

    db_employee_id, hashed_password, department, city, area = employee

    if not bcrypt.checkpw(login_data.password.encode('utf-8'), hashed_password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid employee ID or password")

    # Optionally set cookies here
    response.set_cookie(key="employee_id", value=db_employee_id, httponly=True)
    response.set_cookie(key="city", value=city)
    response.set_cookie(key="area", value=area)
    response.set_cookie(key="department", value=department)

    return {
        "employee_id": db_employee_id,
        "department": department,
        "city": city,
        "area": area
    }
# -------------------- GRIEVANCE TRACKING --------------------
@app.get("/track_grievance")
def track_grievance(grievance_number: str, db: Session = Depends(get_db)):
    grievance = db.query(Prediction).filter(Prediction.grievance_number == grievance_number).first()
    if not grievance:
        raise HTTPException(status_code=404, detail="Grievance not found")
    return {
        "grievance_number": grievance.grievance_number,
        "status": grievance.status,
        "complaint_text": grievance.complaint_text,
        "timestamp": str(grievance.timestamp),
        "image_path": grievance.image_path
    }

@app.get("/add_grievance", response_class=HTMLResponse)
async def add_grievance_form(request: Request):
    username = request.cookies.get("username")
    if not username:
        # Redirect to your auth page with next param
        return RedirectResponse(url="/auth?next=/add_grievance", status_code=302)

    # Render the add grievance form if logged in
    return templates.TemplateResponse("grievance.html", {"request": request})

@app.post("/add_grievance")
async def add_grievance(
    request: Request,
    complaint_text: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    username = request.cookies.get("username")
    city = request.cookies.get("city")
    area = request.cookies.get("area")

    if not all([username, city, area]):
        return RedirectResponse(url="/login?next=/add_grievance", status_code=302)

    try:
        contents = await image.read()
        image_data = Image.open(io.BytesIO(contents)).convert("RGB")

        issue_type, confidence = classify_issue_keras(image_data)
        urgency = classify_urgency_torch(image_data)

        grievance_number = uuid.uuid4().hex[:8]
        safe_issue_type = issue_type.lower().replace(" ", "_")
        save_dir = os.path.join(image_dir, safe_issue_type)
        os.makedirs(save_dir, exist_ok=True)

        image_filename = f"{grievance_number}_{image.filename}"
        save_path = os.path.join(save_dir, image_filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        image_path = f"{safe_issue_type}/{image_filename}"
        print(f"Received complaint at location: latitude={latitude}, longitude={longitude}")
 

        new_grievance = Prediction(
            grievance_number=grievance_number,
            complaint_text=complaint_text,
            issue_type=issue_type,
            urgency=urgency,
            timestamp=datetime.now(),
            image_path=image_path,
            city=city,
            area=area,
            status="Reported",
            latitude=latitude,
            longitude=longitude,
            # reported_by=user.id  ← optional
        )

        db.add(new_grievance)
        db.commit()
        db.refresh(new_grievance)

        return {
            "message": f"Thank you for reporting. Your grievance number is: {grievance_number}",
            "grievance_number": grievance_number
        }

    except Exception as e:
        return {"error": str(e)}


class StatusUpdate(BaseModel):
    issue_id: int
    status: str

@app.post("/update_status")
def update_status(payload: StatusUpdate, db: Session = Depends(get_db)):
    issue = db.query(Prediction).filter(Prediction.id == payload.issue_id).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")

    valid_statuses = ["reported", "acknowledged", "in progress", "completed"]
    if payload.status.lower() not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status value")

    issue.status = payload.status.title()  # Title-case if desired
    db.commit()
    return JSONResponse({"message": f"Status updated to {issue.status}"})

@app.delete("/delete_issue/{issue_id}")
def delete_issue(issue_id: int = Path(...), db: Session = Depends(get_db)):
    issue = db.query(Prediction).filter(Prediction.id == issue_id).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")

    db.delete(issue)
    db.commit()
    return JSONResponse(content={"message": "Issue deleted successfully"})


from pydantic import BaseModel, EmailStr, validator


def save_feedback(data: dict):
    # Read existing feedbacks or create new list
    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                feedbacks = json.load(f)
        except json.JSONDecodeError:
            # File is empty or invalid JSON, start fresh
            feedbacks = []

    feedbacks.append(data)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedbacks, f, indent=2)

class Feedback(BaseModel):
    timeliness: int
    satisfaction: int
    ease: int
    feedbackText: str
    email: Optional[EmailStr] = None
    contact_permission: Optional[bool] = False

    @validator('timeliness', 'satisfaction', 'ease')
    def rating_must_be_1_to_5(cls, v):
        if v < 1 or v > 5:
            raise ValueError("Rating must be between 1 and 5")
        return v


@app.post("/submit-feedback")
async def submit_feedback(
    timeliness: int = Form(...),
    satisfaction: int = Form(...),
    ease: int = Form(...),
    feedbackText: str = Form(...),
    email: Optional[str] = Form(None),
    contact_permission: Optional[bool] = Form(False),
):
    try:
        feedback = Feedback(
            timeliness=timeliness,
            satisfaction=satisfaction,
            ease=ease,
            feedbackText=feedbackText,
            email=email,
            contact_permission=contact_permission,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    feedback_data = feedback.dict()
    feedback_data["submitted_at"] = datetime.utcnow().isoformat()

    save_feedback(feedback_data)
    return {"message": "Feedback submitted successfully!"}

# -------------------- MISC --------------------
@app.get("/grievance", response_class=HTMLResponse)
def show_grievance_form(request: Request):
    return templates.TemplateResponse("grievance.html", {"request": request})

@app.get("/auth", response_class=HTMLResponse)
def auth_page(request: Request):
    return templates.TemplateResponse("auth.html", {"request": request})

@app.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})



@app.get("/admin", response_class=HTMLResponse)
def admin_login_page(request: Request, next: str = Query("/admin_dashboard")):
    return templates.TemplateResponse("admin_auth.html", {"request": request, "next": next})


from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from fastapi.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return PlainTextResponse(f"404 Not Found: {request.url}", status_code=404)
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return PlainTextResponse(str(exc), status_code=400)

for route in app.routes:
    print(route.path)



@app.get("/ping")
def ping():
    return {"msg": "pong"}
