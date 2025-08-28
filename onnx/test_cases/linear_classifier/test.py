import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
input_data = np.array([0.2909665, 0.35581952, 0.16257513, 0.5522083, 0.22992867], dtype=np.float32)
input_data = input_data.reshape(1, -1)

outputs = session.run(None, {input_name: input_data})
outputs = [output.astype(np.float32) for output in outputs]
np.set_printoptions(suppress=True, precision=20)
print(outputs)
