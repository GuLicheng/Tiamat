import torch

# Computer every step manually

# Linear Regression
# f = w * x

# example: f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model output
def forward(x):
    return w * x

# loss
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

print(f"Prediction before training : f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
n_iters = 100  # requires more iterations compared with manually gradientdescent
for epoch in range(n_iters):

    y_pred = forward(X)

    # loss
    l = loss(y=Y, y_pred=y_pred)

    # calculate gradients = backward
    l.backward()

    # update weight
    # w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # zero the gradients after updating
    w.grad.zero_()

    if epoch % 2 == 0:
        print(f"epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}")

print(f"Prediction after training : f(5) = {forward(5):.3f}")



