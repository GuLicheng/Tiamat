# from revover import Net
from configuration import cfg
from loader.data_loader import DATA_LOADER
from torch.utils.data import DataLoader



def main():
    dataloader_ = DataLoader(dataset=DATA_LOADER, batch_size=32, shuffle=True)
    model_ = Net(model=cfg[], data_loader=dataloader_, in_channels=3, out_channels=1)
    model_.to(device=DEVICE)
    model_.train()
    # torch.save(model.state_dict(), 'params.pkl')
    print("OK")

if __name__ == '__main__':
    main()
