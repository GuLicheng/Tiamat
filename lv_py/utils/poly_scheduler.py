from torch.optim.lr_scheduler import LambdaLR 

def poly(optimizer, num_epochs, power = 0.9):
    return LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** power)