import torchvision
from torchvision import transforms
from torchsummary import summary

from src.model import *
from src.nn_utils import *

def load_fake_data(batch_size, resize=None):
    """Get the FakeData dataset generators."""
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


def load_data_cifar10(batch_size, resize=224):
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


def test_output_shape(input_shape, num_classes=0):
    y = net(input_shape)
    print(net.__class__.__name__, 'output shape:\t', y.shape)
    if num_classes > 0:
        output_shape = torch.rand(1, num_classes)
        assert (y.size() == output_shape.size())


def print_net(net, input_shape):
    print(net)
    summary(net, input_shape)


if __name__ == '__main__':
    print("Hi pycharm")

    net = shufflenet_g8_w1()

    X = torch.rand(1, 3, 224, 224)

    test_output_shape(X, num_classes=10)
    export_onnx(net, X, filename="shuffleNet")

    batch_size, lr, num_epochs = 256, 0.1, 5
    train_iter, val_iter, test_iter = load_fake_data(batch_size)

    if 0:
        train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(net, train_iter, val_iter, test_iter, num_epochs,
                                                                         lr, try_gpu())

        plot_loss(train_loss_all, val_loss_all)
        plot_accuracy(train_acc_all, val_acc_all)
