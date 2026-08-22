from pydantic import BaseModel, Field
from typing import Literal

class StudentFeatures(BaseModel):
    gender: Literal["Female", "Male"]
    study_time_hours: float = Field(ge=0)
    attendance_percent: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0)
    parental_education: Literal["High School", "Bachelors", "Masters", "PhD"]
    internet_access: Literal["Yes", "No"]
    extracurricular_activities: Literal["Yes", "No"]
    part_time_job: Literal["Yes", "No"]
    previous_grade: float = Field(ge=0)

class PredictionResponse(BaseModel):
    final_grade: str
    model_version: str

class ModelInfoResponse(BaseModel):
    name: str
    version: str
    alias: str
    run_id: str

class BatchPredictionRequest(BaseModel):
    students: list[StudentFeatures] = Field(..., min_length=1, max_length=500)

class BatchPredictionResponse(BaseModel):
    predictions: list[str]
    model_version: str