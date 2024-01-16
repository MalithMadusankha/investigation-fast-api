from pydantic import BaseModel
from bson import ObjectId

class DataModel(BaseModel):
    provience: str
    district: str
    city: str
    month: str   
    year: int
    