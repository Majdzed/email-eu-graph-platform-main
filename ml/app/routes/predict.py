from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from utils.model_manager import model_manager

router = APIRouter()

class PredictRequest(BaseModel):
    node_id: int = Field(..., ge=0, description="Node ID must be a non-negative integer")

class SimulationRequest(BaseModel):
    neighbor_ids: List[int] = Field(..., min_length=1, description="Must provide at least one neighbor node ID")

@router.post("/department")
async def predict_department(request: PredictRequest):
    """Predict department for an existing node"""
    result = model_manager.predict_department(request.node_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
        
    return {
        **result,
        "message": "Node classification powered by Random Forest"
    }

@router.post("/department/simulate")
async def simulate_department(request: SimulationRequest):
    """Simulate department for a new hypothetical node"""
    result = model_manager.simulate_department(request.neighbor_ids)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.get("/department/unseen")
async def predict_unseen(index: Optional[int] = Query(None)):
    """Evaluate model on a sample from the unseen test set"""

    result = model_manager.predict_unseen(index)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
