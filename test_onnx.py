import onnxruntime
import numpy as np


class OnnxInfer:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=["CPUExecutionProvider"]
        )

    def infer(self, inputs):
        outputs = self.ort_session.run(None, {"obs": inputs})
        return outputs[0]


if __name__ == "__main__":
    O = OnnxInfer("ONNX.onnx")
    inputs = np.random.uniform(size=51).astype(np.float32)
    print(inputs)
    print(O.infer(inputs))
