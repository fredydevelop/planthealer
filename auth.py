# from fastapi import FastAPI, File, UploadFile

# import uvicorn

# import numpy as np

# from io import BytesIO

# from PIL import Image

# import tensorflow as tf

# import os

# from pathlib import Path




# BASE_DIR = Path(__file__).resolve().parent

# tomatmodel_path = BASE_DIR / "models" / "tomato"/ "tomato_model.keras"

# potatmodel_path = BASE_DIR / "models" / "potato"/ "new_potato_model.keras"

# bellmodel_path = BASE_DIR / "models" / "bell_pepper"/ "retry_bellpepper_model.keras"



# try:

#     tomatmodel = tf.keras.models.load_model(tomatmodel_path)


#     potatmodel = tf.keras.models.load_model(potatmodel_path)


#     bellmodel = tf.keras.models.load_model(bellmodel_path)


# except Exception as e:

#     print("Error loading models:", e)

#     raise e



# bellclassnames = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

# tomatclassnames = ['Bacterial_spot', 'Early_blight', 'Healthy', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 'YellowLeaf__Curl_Virus', 'mosaic_virus']

# potatclassnames = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']



# def read_file_as_image(data)-> np.ndarray:

#     image = np.array(Image.open(BytesIO(data)))

#     return image


# app = FastAPI()


# @app.post("/tomato/predict")

# async def tomatprediction(file: UploadFile = File(...)):


#     image =  read_file_as_image(await file.read())

#     batch_img = np.expand_dims(image,0)

#     prediction = tomatmodel.predict(batch_img)

#     predicted_class= tomatclassnames[np.argmax(prediction[0])]

#     predicted_confidence = np.max(prediction[0]) * 100

#     return {

#         "predicted": predicted_class,

#         "prediction confidence": predicted_confidence,

#     }

   



# @app.post("/bell_pepper/predict")

# async def bellprediction(file: UploadFile = File(...)):


#     image =  read_file_as_image(await file.read())

#     batch_img = np.expand_dims(image,0)

#     prediction = bellmodel.predict(batch_img)

#     predicted_class= bellclassnames[np.argmax(prediction[0])]

#     predicted_confidence = np.max(prediction[0]) * 100

#     return {

#         "predicted": predicted_class,

#         "Prediction confidence": predicted_confidence,

#     }



# @app.post("/potato/predict")

# async def potatprediction(file: UploadFile = File(...)):


#     image =  read_file_as_image(await file.read())

#     batch_img = np.expand_dims(image,0)

#     prediction = potatmodel.predict(batch_img)

#     predicted_class= potatclassnames[np.argmax(prediction[0])]

#     predicted_confidence = np.max(prediction[0]) * 100

#     return {

#         "predicted": predicted_class,

#         "Prediction confidence": predicted_confidence,

#     }


     

# # if __name__ == "__main__":

# #     uvicorn.run(app,host="localhost", port=8000)