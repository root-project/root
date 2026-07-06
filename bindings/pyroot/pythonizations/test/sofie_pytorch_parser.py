import os
import shutil
import unittest

import numpy as np
import ROOT

# Test the SOFIE PyTorch parser (TMVA::Experimental::SOFIE::PyTorch::Parse) by
# parsing TorchScript models, generating and compiling the inference code and
# comparing the SOFIE results with the outputs of the original PyTorch models.
#
# This is the Python translation of the former C++ googletest
# tmva/sofie/test/TestRModelParserPyTorch.C.

WORK_DIR = "pytorch_parser_models"


def generate_pytorch_models(dst_dir):
    import torch
    import torch.nn as nn

    def train_and_save(model, x, y, name, n_iterations):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for _ in range(n_iterations):
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        m = torch.jit.script(model)
        torch.jit.save(m, f"{dst_dir}/{name}.pt")

    # Sequential model
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
        nn.SELU(),
    )
    train_and_save(model, torch.randn(2, 4), torch.randn(2, 6), "PyTorchModelSequential", 2000)

    # Module model
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(6, 36)
            self.fc2 = nn.Linear(36, 12)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            x = torch.transpose(x, 1, 0)
            return x

    train_and_save(Model(), torch.randn(2, 6), torch.randn(12, 2), "PyTorchModelModule", 2000)

    # Convolution model
    model = nn.Sequential(
        nn.Conv2d(6, 5, 3, stride=2),
        nn.ReLU(),
    )
    train_and_save(model, torch.randn(5, 6, 5, 5), torch.randn(5, 5, 2, 2), "PyTorchModelConvolution", 100)


class SOFIE_PyTorch_Parser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isdir(WORK_DIR):
            shutil.rmtree(WORK_DIR)
        os.makedirs(WORK_DIR)
        print("Generating PyTorch models for testing")
        generate_pytorch_models(WORK_DIR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(WORK_DIR)

    def generate_and_test_inference(self, model_name, input_tensor, atol):
        import torch

        model_path = f"{WORK_DIR}/{model_name}.pt"

        # Parse the TorchScript model. The PyTorch parser needs the input
        # tensor shapes, which also fix the batch size (the first dimension).
        input_shapes = ROOT.std.vector["std::vector<std::size_t>"]()
        input_shapes.push_back(input_tensor.shape)
        rmodel = ROOT.TMVA.Experimental.SOFIE.PyTorch.Parse(model_path, input_shapes)

        # Generate and compile the SOFIE inference code
        batch_size = input_tensor.shape[0]
        rmodel.Generate(ROOT.TMVA.Experimental.SOFIE.Options.kDefault, batch_size)
        header_path = f"{WORK_DIR}/{model_name}.hxx"
        rmodel.OutputGenerated(header_path)
        print(f"Compiling SOFIE model {model_name}")
        if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
            raise AssertionError(f"Error compiling header file {header_path}")
        sofie_model_namespace = getattr(ROOT, "TMVA_SOFIE_" + model_name)
        inference_session = sofie_model_namespace.Session(header_path.removesuffix(".hxx") + ".dat")

        sofie_inference_result = np.asarray(inference_session.infer(input_tensor)).flatten()

        # Evaluate the original model with PyTorch on the same input
        torch_model = torch.jit.load(model_path)
        torch_model.eval()
        torch_inference_result = torch_model(torch.from_numpy(input_tensor)).detach().numpy()

        # Compare the output tensor sizes and values
        self.assertEqual(len(sofie_inference_result), torch_inference_result.size)
        np.testing.assert_allclose(
            sofie_inference_result,
            torch_inference_result.flatten(),
            atol=atol,
            rtol=0.0,  # explicitly disable relative tolerance (NumPy uses |a - b| <= atol + rtol * |b|)
        )

    def test_sequential_model(self):
        input_tensor = np.array(
            [[-1.6207, 0.6133, 0.5058, -1.2560], [-0.7750, -1.6701, 0.8171, -0.2858]], dtype=np.float32
        )
        self.generate_and_test_inference("PyTorchModelSequential", input_tensor, atol=1e-6)

    def test_module_model(self):
        input_tensor = np.array(
            [
                [0.5516, 0.3585, -0.4854, -1.3884, 0.8057, -0.9449],
                [0.5626, -0.6466, -1.8818, 0.4736, 1.1102, 1.8694],
            ],
            dtype=np.float32,
        )
        self.generate_and_test_inference("PyTorchModelModule", input_tensor, atol=1e-6)

    def test_convolution_model(self):
        input_tensor = np.arange(1, 751, dtype=np.float32).reshape(5, 6, 5, 5)
        self.generate_and_test_inference("PyTorchModelConvolution", input_tensor, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
