import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---- 1) Define a small fully-connected regression model ----
class SmallMLP(nn.Module):
    def __init__(self, in_features=10, hidden=32, out_features=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)


def write_onnx_model(onnx_path):

    # ---- 2) Create synthetic regression data ----
    torch.manual_seed(0)

    num_samples = 1000
    in_features = 10

    X = torch.randn(num_samples, in_features)

    # True function: linear combination + noise
    true_w = torch.randn(in_features, 1)
    true_b = torch.randn(1)

    y = X @ true_w + true_b + 0.1 * torch.randn(num_samples, 1)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # ---- 3) Train the model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallMLP(in_features=10, hidden=32, out_features=1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")

    # ---- 4) Export the trained model to ONNX ----
    model.eval()

    # Create a torch.export program in the parent process.
    # This does not require importing ONNX.
    example_input = (torch.randn(1, 10, device=device),)
    exported = torch.export.export(model, example_input)

    torch.onnx.export(
        exported,
        args=(),
        f=onnx_path,
        external_data=False,
        dynamo=True,
    )

    return model


def main():

    onnx_path = "regression_mlp.onnx"

    model = write_onnx_model(onnx_path)

    x = torch.tensor([[0.1] * 10], requires_grad=True)

    y = model(x)

    y.backward()

    print("prediction:", y.item())
    print("input gradient:", x.grad)

    np.savetxt("regression_mlp_pred.txt", y.detach().numpy())
    np.savetxt("regression_mlp_grad.txt", x.grad.detach().numpy())


if __name__ == "__main__":
    main()
