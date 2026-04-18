from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import numpy as np
from PIL import Image
import io

app = FastAPI()


@app.post("/predict")
async def predict(
    plant: str = Form(...),
    file: UploadFile = File(...)
):
    # ✅ Check plant type
    if plant not in models:
        raise HTTPException(status_code=400, detail="Invalid plant type")

    model = models[plant]

    # ✅ Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # ✅ Preprocess (adjust based on your training)
    image = image.resize((224, 224))  # adjust if needed
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # ✅ Predict
    prediction = model.predict(image)

    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "plant": plant,
        "prediction": predicted_class,
        "confidence": confidence
    }