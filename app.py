import os
from utils import cvutils
import uvicorn
from fastapi import FastAPI, Header, HTTPException
import logging
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import cv2
import aiohttp
import asyncio
import Webfunc
import requests


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


app = FastAPI()

class InputDoc(BaseModel):
    tokenc : str
    text  : str

class Item(BaseModel):
    tokene : str
    url  : str
    effect : str

class Item_style(BaseModel):
    tokenb : str
    url  : str
    url2 : str
    ratio : float

@app.on_event("startup")
def startup():
    logger.info("Starting...")


@app.on_event("shutdown")
def shutdown():
    logger.info("Shutting down...")


@app.get("/")
def root():
    return {"message": "API is running"}



@app.post("/cv")
def effect(input : Item):
    if input.tokene == os.getenv('tokene'):
        url = input.url
        effect = input.effect
        if requests.head(url).headers['Content-Type'] == 'image/png':
            img = cvutils.GnP(url)
            imgout = cvutils.effectsdoer(img, effect)
            imgout = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
            if imgout is None:
                raise HTTPException(status_code=400, detail="Invalid effect")
            bytes = cvutils.to_bytes(imgout)
            return StreamingResponse(bytes, media_type="image/png")

        elif requests.head(url).headers['Content-Type'] == 'image/gif':
            gifout = []
            frames = cvutils.get_gif(url)
            for frame in frames:
                frameout = cv2.CvtColor(frame, cv2.COLOR_RGBA2RGB)
                framesout = cvutils.effectsdoer(frameout, effect)
                framesout = cv2.cvtColor(framesout, cv2.COLOR_BGR2RGB)
                if framesout is None:
                    raise HTTPException(status_code=400, detail="Invalid effect")
                gifout.append(framesout)
            imgs = [Image.fromarray(out) for out in gifout]
            output = io.BytesIO()
            bytes = imgs[0].save(bytes, save_all=True,append_images=imgs[1:],duration=50, loop=0, format='PNG')
            return StreamingResponse(bytes, media_type="image/gif")
    else: return {"error": "get the fu*king password man"}

@app.post("/style")
def style(input : Item):
    if input.tokene == os.getenv('tokene'):
        url = input.url
        effect = input.effect
        if requests.head(url).headers['Content-Type'] == 'image/png':
            img = cvutils.GnP(url)
            model = cvutils.getmodel(effect)
            if model is None: raise HTTPException(status_code=400, detail="Invalid style")
            imgout = cvutils.style_transfer(img, model)
            imgout = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)
            bytes = cvutils.to_bytes(imgout)
            return StreamingResponse(bytes, media_type="image/png")

        elif requests.head(url).headers['Content-Type'] == 'image/gif':
            gifout = []
            frames = cvutils.get_gif(url)
            for frame in frames:
                model = cvutils.getmodel(effect)
                if frame is None:
                    raise HTTPException(status_code=400, detail="Invalid effect")
                frameout = cv2.CvtColor(frame, cv2.COLOR_RGBA2RGB)
                framesout = cvutils.style_transfer(frame, model)
                framesout = cv2.cvtColor(framesout, cv2.COLOR_BGR2RGB)
                gifout.append(framesout)
            imgs = [Image.fromarray(out) for out in gifout]
            output = io.BytesIO()
            bytes = imgs[0].save(bytes, save_all=True,append_images=imgs[1:],duration=50, loop=0, format='PNG')
            return StreamingResponse(bytes, media_type="image/gif")

    else: return {"error": "get the fu*king password man"}
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
