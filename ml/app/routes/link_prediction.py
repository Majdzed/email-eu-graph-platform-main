from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
from utils.model_manager import model_manager

router = APIRouter()

class LinkPredictRequest(BaseModel):
    source: int = Field(..., ge=0, description="Source node ID must be a non-negative integer")
    target: int = Field(..., ge=0, description="Target node ID must be a non-negative integer")

class LinkSimulationRequest(BaseModel):
    neighbor_ids: List[int] = Field(..., min_length=1, description="Must provide at least one neighbor node ID")
    target: int = Field(..., ge=0, description="Target node ID must be a non-negative integer")

@router.post("/predict")
async def predict_links(request: LinkPredictRequest):
    """Predict link probability between two existing nodes"""
    result = model_manager.predict_link(request.source, request.target)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.post("/simulate")
async def simulate_link(request: LinkSimulationRequest):
    """Simulate link probability for a hypothetical new node"""
    result = model_manager.simulate_link(request.neighbor_ids, request.target)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
