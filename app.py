import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
import torch.nn as nn

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/processed"
MODEL_PATH = "model/ゴルディアス.pt"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.pool1(self.act1(self.bn1(self.conv1(x))))
        out2 = self.pool2(self.act2(self.bn2(self.conv2(out1))))
        out3 = self.act3(self.bn3(self.conv3(out2)))
        out = self.conv4(out3)
        out = self.conv5(out)
        out = self.sigmoid(self.conv6(out))
        return out


def load_model(model_path):
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def process_image(image_path, model, output_path, threshold=0.8):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        mask = model(input_tensor).squeeze().numpy()

    mask = (mask > threshold).astype(np.uint8) * 255
    mask = Image.fromarray(mask).resize(image.size, Image.LANCZOS)

    transparent_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    cutout = Image.composite(image.convert("RGBA"), transparent_image, mask.convert("L"))
    cutout.save(output_path, format="PNG")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or file.filename == "":
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(input_path)

    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
    process_image(input_path, model, output_path)

    output_url = url_for("static", filename=f"processed/{output_filename}")
    return render_template("DownLoad.html", output_url=output_url)


@app.route("/sns")
def sns():
    return render_template("SNS.html")


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    app.run(debug=True)
