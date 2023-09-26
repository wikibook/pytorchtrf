# app_flask.py
import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForSequenceClassification


class BertModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def load_model(cls, weight_path):
        cls.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path="bert-base-multilingual-cased",
            do_lower_case=False,
        )
        cls.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="bert-base-multilingual-cased",
            num_labels=2
        ).to(cls.device)
        cls.model.load_state_dict(torch.load(weight_path, map_location=cls.device))
        cls.model.eval()
        
    @classmethod
    def preprocessing(cls, data):
        input_data = cls.tokenizer(
            text=data,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ).to(cls.device)
        return input_data

    @classmethod
    @torch.no_grad()
    def predict(cls, input):
        input_data = cls.preprocessing(input)
        outputs = cls.model(**input_data).logits
        probs = F.softmax(outputs, dim=-1)
        
        index = int(probs[0].argmax(axis=-1))
        label = "긍정" if index == 1 else "부정"
        score = float(probs[0][index])

        return {
            "label": label,
            "score": score
        }
        
        
# app_flask.py
import json
from flask import Flask, request, Response


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def inference():
    data = request.get_json()
    text = data["text"]

    try:
        return Response(
            response=json.dumps(BertModel.predict(text), ensure_ascii=False),
            status=200,
            mimetype="application/json",
        )

    except Exception as e:
        return Response(
            response=json.dumps({"error": str(e)}, ensure_ascii=False),
            status=500,
            mimetype="application/json",
        )


if __name__ == "__main__":
    BertModel.load_model(weight_path="./BertForSequenceClassification.pt")
    app.run(host="0.0.0.0", port=8000)