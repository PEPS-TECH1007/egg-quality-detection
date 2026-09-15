from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response

app = FastAPI()

latest_image = None


@app.get("/")
def home():
    return {"status": "Egg camera bridge is running"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    global latest_image

    latest_image = await file.read()

    return {
        "status": "success",
        "message": "Image received"
    }


@app.get("/latest")
def get_latest_image():
    if latest_image is None:
        return {
            "status": "no_image"
        }

    return Response(
        content=latest_image,
        media_type="image/jpeg"
    )
