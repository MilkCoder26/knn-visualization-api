from typing import List, Union
from pydantic import BaseModel, Field

Label = Union[str, int]


class TrainingSet(BaseModel):
    """Le dataset d'entraînement envoyé par le frontend à chaque requête."""
    x_train: List[List[float]] = Field(..., description="Points d'entraînement, ex: [[1.0, 2.0], ...]")
    y_train: List[Label] = Field(..., description="Labels correspondants")
    k: int = Field(3, ge=1, description="Nombre de voisins")
    distance_fn: str = Field("euclidian", pattern="^(euclidian|manhattan)$")


class PredictRequest(TrainingSet):
    x_test: List[List[float]] = Field(..., description="Points à prédire")


class PredictResponse(BaseModel):
    predictions: List[Label]


class NeighborsRequest(TrainingSet):
    point: List[float] = Field(..., description="Le point unique dont on veut les voisins")


class NeighborItem(BaseModel):
    distance: float
    label: Label
    point: List[float]
    is_selected: bool  # fait partie des k plus proches ou non


class NeighborsResponse(BaseModel):
    neighbors: List[NeighborItem]
    prediction: Label


class DecisionBoundaryRequest(TrainingSet):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: int = Field(50, ge=5, le=200, description="Nombre de points par axe (grille resolution x resolution)")


class DecisionBoundaryResponse(BaseModel):
    grid_x: List[float]
    grid_y: List[float]
    predictions: List[List[Label]]  # matrice [resolution][resolution]
