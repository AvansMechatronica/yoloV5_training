
from roboflow import Roboflow
rf = Roboflow(api_key="K3sks4IiHf1jC7nMw6YN")
project = rf.workspace("gerard-harkema-9zmag").project("simplefruits")
version = project.version(1)
dataset = version.download("yolov5")