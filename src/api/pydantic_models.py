from pydantic import BaseModel, Field


class CreditRiskInput(BaseModel):
    Total_Amount: float = Field(..., example=15000.0)
    Average_Amount: float = Field(..., example=500.0)
    Transaction_Count: int = Field(..., example=30)
    Std_Amount: float = Field(..., example=200.5)

    class Config:
        schema_extra = {
            "example": {
                "Total_Amount": 15000.0,
                "Average_Amount": 500.0,
                "Transaction_Count": 30,
                "Std_Amount": 200.5
            }
        }
