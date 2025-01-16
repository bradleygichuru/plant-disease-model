from transformers import pipeline

classifier = pipeline(
    "image-classification",
    model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
)
res = classifier("./002213fb-b620-4593-b9ac-6a6cc119b100___Com.G_TgS_FL 8360.jpeg")
print("predicted:", res, " ", "type:", type(res))
