from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import csv
import io

from .model import KNN
from .schemas import (
    PredictRequest,
    PredictResponse,
    NeighborsRequest,
    NeighborsResponse,
    NeighborItem,
    DecisionBoundaryRequest,
    DecisionBoundaryResponse,
)

app = FastAPI(title="KNN Playground API", version="0.1.0")

# En dev, on ouvre le CORS large. À restreindre en prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _validate_training_set(req.x_train, req.y_train)
    model = KNN(req.x_train, req.y_train, k=req.k, distance_fn=req.distance_fn)
    predictions = model.test(req.x_test)
    return PredictResponse(predictions=predictions)


@app.post("/neighbors", response_model=NeighborsResponse)
def neighbors(req: NeighborsRequest):
    """
    Renvoie TOUS les voisins triés par distance, en marquant lesquels font
    partie des k plus proches. Pensé pour l'animation frontend : on peut
    dessiner toutes les lignes et surligner celles qui comptent pour le vote.
    """
    _validate_training_set(req.x_train, req.y_train)
    model = KNN(req.x_train, req.y_train, k=req.k, distance_fn=req.distance_fn)

    # On calcule nous-mêmes les distances ici (plutôt que via model.find_neighbors)
    # pour garder le point d'origine associé à chaque distance/label.
    triples = []
    for training_point, label in zip(req.x_train, req.y_train):
        d = model.distance(training_point, req.point)
        triples.append((d, label, training_point))
    triples.sort(key=lambda t: t[0])

    items = [
        NeighborItem(distance=d, label=label, point=point, is_selected=i < req.k)
        for i, (d, label, point) in enumerate(triples)
    ]

    selected = [(d, label) for d, label, _ in triples[: req.k]]
    prediction = model.vote(selected)
    return NeighborsResponse(neighbors=items, prediction=prediction)


@app.post("/decision-boundary", response_model=DecisionBoundaryResponse)
def decision_boundary(req: DecisionBoundaryRequest):
    """
    Calcule les prédictions sur une grille régulière pour dessiner la
    frontière de décision (heatmap/contour côté frontend).
    Attention : ne fonctionne que pour des données 2D (x_train de dimension 2).
    """
    _validate_training_set(req.x_train, req.y_train)
    if any(len(p) != 2 for p in req.x_train):
        raise HTTPException(400, "decision-boundary ne supporte que des points 2D")

    model = KNN(req.x_train, req.y_train, k=req.k, distance_fn=req.distance_fn)

    step_x = (req.x_max - req.x_min) / (req.resolution - 1)
    step_y = (req.y_max - req.y_min) / (req.resolution - 1)
    grid_x = [req.x_min + i * step_x for i in range(req.resolution)]
    grid_y = [req.y_min + i * step_y for i in range(req.resolution)]

    predictions = []
    for gy in grid_y:
        row = []
        for gx in grid_x:
            neighbors = model.find_neighbors([gx, gy])[: req.k]
            row.append(model.vote(neighbors))
        predictions.append(row)

    return DecisionBoundaryResponse(grid_x=grid_x, grid_y=grid_y, predictions=predictions)


@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Parse un CSV et le renvoie sous forme utilisable directement par le
    frontend (x_train = toutes les colonnes sauf la dernière, y_train = dernière colonne).
    """
    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if not rows:
        raise HTTPException(400, "Fichier vide")

    header = rows[0]
    data_rows = rows[1:]

    x_train, y_train = [], []
    for row in data_rows:
        if not row:
            continue
        try:
            features = [float(v) for v in row[:-1]]
        except ValueError:
            raise HTTPException(400, f"Ligne non numérique détectée: {row}")
        x_train.append(features)
        y_train.append(row[-1])

    return {"header": header, "x_train": x_train, "y_train": y_train}


def _validate_training_set(x_train, y_train):
    if len(x_train) != len(y_train):
        raise HTTPException(400, "x_train et y_train doivent avoir la même longueur")
    if len(x_train) == 0:
        raise HTTPException(400, "x_train est vide")
