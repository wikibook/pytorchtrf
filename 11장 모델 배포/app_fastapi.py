# app_fastapi.py
import io
import torch
import base64
from PIL import Image
from torch.nn import functional as F
from torchvision import models, transforms


class VGG16Model:
    def __init__(self, weight_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48235, 0.45882, 0.40784],
                    std=[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
                )
            ]
        )
        self.model = models.vgg16(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

    def preprocessing(self, data):
        decode = base64.b64decode(data)
        bytes = io.BytesIO(decode)
        image = Image.open(bytes)
        input_data = self.transform(image).to(self.device)
        return input_data
    
    @torch.no_grad()
    def predict(self, input):
        input_data = self.preprocessing(input)
        outputs = self.model(input_data.unsqueeze(0))
        probs = F.softmax(outputs, dim=-1)
        
        index = int(probs[0].argmax(axis=-1))
        label = "개" if index == 1 else "고양이"
        score = float(probs[0][index])

        return {
            "label": label,
            "score": score
        }
        
# app_fastapi.py
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException


app = FastAPI()
vgg = VGG16Model(weight_path="./VGG16.pt")


class Item(BaseModel):
    base64: str


def get_model():
    return vgg


@app.post("/predict")
async def inference(item: Item, model: VGG16Model = Depends(get_model)):
    try:
        return model.predict(item.base64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app="app_fastapi:app", host="0.0.0.0", port=8000, workers=2)