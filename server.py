from typing import Annotated, BinaryIO
from fastapi import FastAPI, File, Form, UploadFile
from transformers import pipeline
from PIL import Image
from io import BytesIO


app = FastAPI()


classifier = pipeline(
    "image-classification",
    model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
)


@app.post("/uploadfile")
async def create_upload_file(file: Annotated[bytes, File()]):
    img = Image.open(BytesIO(file))

    res = classifier(img)

    print("predicted:", res, " ", "type:", type(res))
    return {"predicted": res}
