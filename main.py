import torchvision
from torchvision import transforms

from src.model import *
from src.nn_utils import *

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

def load_fake_data(batch_size, resize=None):
    """Download the FakeData dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    fakeData_train = torchvision.datasets.FakeData(50000, image_size=(3, 224, 224), num_classes=10, transform=trans)
    fakeData_val = torchvision.datasets.FakeData(10000, image_size=(3, 224, 224), num_classes=10, transform=trans)
    fakeData_test = torchvision.datasets.FakeData(10000, image_size=(3, 224, 224), num_classes=10, transform=trans)

    return (torch.utils.data.DataLoader(fakeData_train, batch_size, shuffle=True,
                                        num_workers=2),
            torch.utils.data.DataLoader(fakeData_val, batch_size, shuffle=False,
                                        num_workers=2),
            torch.utils.data.DataLoader(fakeData_test, batch_size, shuffle=False,
                                        num_workers=2))


def load_data_cifar10(batch_size, resize=None):
    """Download the CIFAR10 dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    cifar10_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True)
    cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10_train, [3000, 47000],
                                                           generator=torch.Generator().manual_seed(42))
    return (torch.utils.data.DataLoader(cifar10_train, batch_size, shuffle=True,
                            num_workers=2),
            torch.utils.data.DataLoader(cifar10_val, batch_size, shuffle=False,
                            num_workers=2),
            torch.utils.data.DataLoader(cifar10_test, batch_size, shuffle=False,
                            num_workers=2))


if __name__ == '__main__':
    print("Hi pycharm")

    net = shufflenet_g8_w1()

    X = torch.rand(1, 3, 224, 224)
    X = net(X)
    print(net.__class__.__name__, 'output shape:\t', X.shape)
    assert (tuple(X.size()) == (1, 10))

    batch_size, lr, num_epochs = 256, 0.1, 5
    train_iter, val_iter, test_iter = load_fake_data(batch_size)

    # train_iter, val_iter, test_iter = load_data_cifar10(batch_size, resize=224)

    train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(net, train_iter, val_iter, test_iter, num_epochs,
                                                                     lr, try_gpu())

    plot_loss(train_loss_all, val_loss_all)
    plot_accuracy(train_acc_all, val_acc_all)
