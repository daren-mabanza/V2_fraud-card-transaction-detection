from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    amt: float = Field(gt=0, strict=True)
    
    heure: int = Field(gt=0, le=23, strict=True)
    
    jour: Literal[
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    
    category: Literal[
        "entertainment", "food_dining", "gas_transport", "grocery_net",
        "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
        "misc_pos", "personal_care", "shopping_net", "shopping_pos",
        "travel",
    ]
    
    state: Literal[
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
        "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
        "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
        "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV",
        "WY",
    ]
    
    
    
class ModelInfoOutput(BaseModel):
    
    model: str
    version: str
    seuil: float = Field(gt=0, lt=1)
    variables: list[str]
    
    