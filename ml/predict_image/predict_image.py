from fastapi import FastAPI, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import uuid
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import sys

app = FastAPI()

# Load your trained model once
model = tf.keras.models.load_model("../garbage_vs_pothole_cnn.h5")

# Dependency to get DB session (implement this according to your setup)
def get_db():
    # your DB session creation here
    pass

@app.post("/add_grievance")
async def add_grievance(
    complaint_text: str = Form(...),
    city: str = Form(...),
    area: str = Form(...),
    urgency: str = Form("Medium"),  # default urgency
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save the uploaded image
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join(upload_dir, image_filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Load and preprocess image for prediction
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    print("Raw model prediction:", prediction)
    sys.stdout.flush()  # Make sure the print gets flushed immediately

    issue_type = "Garbage" if prediction[0][0] < 0.5 else "Pothole"

    grievance_number = str(uuid.uuid4())[:8]

    new_grievance = Prediction(
        grievance_number=grievance_number,
        complaint_text=complaint_text,
        issue_type=issue_type,
        urgency=urgency,
        timestamp=datetime.now(),
        image_path=image_path,
        city=city,
        area=area,
        status="Reported"
    )

    db.add(new_grievance)
    db.commit()
    db.refresh(new_grievance)

    return {
        "message": f"Thank you for reporting. Your grievance number is: {grievance_number}",
        "grievance_number": grievance_number
    }
