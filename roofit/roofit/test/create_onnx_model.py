import numpy as np
import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """Single-input MLP regression model."""

    def __init__(self, in_features=10, hidden=32, out_features=1, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


class TwoInputMLP(nn.Module):
    """Dual-input MLP: concatenates two input tensors then runs an MLP."""

    def __init__(self, in_features_a=10, in_features_b=5, hidden=32, out_features=1, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features_a + in_features_b, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x_a, x_b):
        return self.net(torch.cat([x_a, x_b], dim=-1))


def export_model(
    model,
    input_shapes,
    onnx_path,
    seed=0,
):
    """
    Train `model` on synthetic regression data and export it to ONNX.

    `input_shapes` is a list of per-sample feature shapes, one per input tensor.
    For a single-input model pass e.g. [(10,)]; for two inputs [(10,), (5,)].
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = model.to(device)

    # ---- Export to ONNX ----
    model.eval()
    example_inputs = tuple(torch.randn(1, *shape, device=device) for shape in input_shapes)
    exported = torch.export.export(model, example_inputs)
    torch.onnx.export(
        exported,
        args=(),
        f=onnx_path,
        external_data=False,
        dynamo=True,
    )
    return model


def run_inference_and_save(model, example_inputs, name, with_hessian=False):
    """Run forward+backward with fixed inputs and save prediction + gradients."""
    model = model.cpu()
    inputs = [t.clone().detach().requires_grad_(True) for t in example_inputs]
    y = model(*inputs)
    y.backward()

    print(f"[{name}] prediction:", y.item())
    for i, x in enumerate(inputs):
        print(f"[{name}] input[{i}] gradient:", x.grad)

    np.savetxt(f"{name}_pred.txt", y.detach().numpy())
    for i, x in enumerate(inputs):
        np.savetxt(f"{name}_grad_{i}.txt", x.grad.detach().numpy())

    if with_hessian:
        # Full Hessian over the concatenation of all (flattened) input
        # tensors, matching the parameter ordering used on the RooFit side.
        blocks = torch.autograd.functional.hessian(
            lambda *args: model(*args).squeeze(), tuple(t.clone().detach() for t in example_inputs)
        )
        n_per_input = [t.numel() for t in example_inputs]
        hess = np.block([
            [blocks[i][j].reshape(n_per_input[i], n_per_input[j]).numpy() for j in range(len(example_inputs))]
            for i in range(len(example_inputs))
        ])
        np.savetxt(f"{name}_hessian.txt", hess)


def main():
    # ---- Model 1: single input ----
    model1 = SmallMLP(in_features=10, hidden=32, out_features=1)
    model1 = export_model(
        model1,
        input_shapes=[(10,)],
        onnx_path="regression_mlp.onnx",
    )
    run_inference_and_save(
        model1,
        example_inputs=[torch.tensor([[0.1] * 10])],
        name="regression_mlp",
    )

    # ---- Model 2: two inputs ----
    model2 = TwoInputMLP(in_features_a=10, in_features_b=5, hidden=32, out_features=1)
    model2 = export_model(
        model2,
        input_shapes=[(10,), (5,)],
        onnx_path="regression_mlp_two_input.onnx",
    )
    run_inference_and_save(
        model2,
        example_inputs=[
            torch.tensor([[0.1] * 10]),
            torch.tensor([[0.2] * 5]),
        ],
        name="regression_mlp_two_input",
    )

    # ---- Models 3 and 4: like models 1 and 2, but with a smooth activation
    # function so that the Hessian with respect to the inputs is non-trivial
    # (a ReLU MLP is piecewise-linear in its inputs, so its input Hessian
    # vanishes almost everywhere). Used for the Hessian tests.
    model3 = SmallMLP(in_features=10, hidden=32, out_features=1, activation=nn.Tanh)
    model3 = export_model(
        model3,
        input_shapes=[(10,)],
        onnx_path="regression_mlp_tanh.onnx",
    )
    run_inference_and_save(
        model3,
        example_inputs=[torch.tensor([[0.1] * 10])],
        name="regression_mlp_tanh",
        with_hessian=True,
    )

    model4 = TwoInputMLP(in_features_a=10, in_features_b=5, hidden=32, out_features=1, activation=nn.Tanh)
    model4 = export_model(
        model4,
        input_shapes=[(10,), (5,)],
        onnx_path="regression_mlp_two_input_tanh.onnx",
    )
    run_inference_and_save(
        model4,
        example_inputs=[
            torch.tensor([[0.1] * 10]),
            torch.tensor([[0.2] * 5]),
        ],
        name="regression_mlp_two_input_tanh",
        with_hessian=True,
    )


if __name__ == "__main__":
    main()
