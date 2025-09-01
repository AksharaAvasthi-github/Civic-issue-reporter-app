from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
# models.py
from database import Base  # 🔁 remove the dot


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    grievance_number = Column(String, unique=True, index=True)
    complaint_text = Column(String)
    issue_type = Column(String)
    urgency = Column(String)
    timestamp = Column(DateTime)
    image_path = Column(String, nullable=True)
    city = Column(String)
    area = Column(String)
    status = Column(String, default="Reported")
    latitude = Column(Float, nullable=True)   # ✅ Add this
    longitude = Column(Float, nullable=True)  # ✅ And 

   
# models.py

import torch
import torch.nn as nn
import torchvision.models as models

class UrgencyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(UrgencyModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)

    

