from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class AWGN(object):
    def __init__(self, sigma, vmin=0, vmax=1):
        assert isinstance(sigma, (int, float))
        self.sigma = sigma
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, image):
        noise = torch.from_numpy(np.random.normal(0.0, self.sigma, image.shape).astype(np.float32))
        # Above supports std of 0

        # noise = torch.normal(0.0, self.sigma, image.shape)
        output = image + noise
        torch.clip(output, self.vmin, self.vmax, out=output)

        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = np.zeros((10,10))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            indices = np.array([target.numpy().squeeze(), pred.numpy().squeeze()]).transpose()
            for i in indices:
                confusion_matrix[i[0],i[1]] += 1

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    # suppress: suppress scientific notation
    with np.printoptions(precision=3, suppress=True):
        print(np.array(confusion_matrix))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--test-AWGN', action='store_true', default=False,
                        help='Test AWGN transformation')
    parser.add_argument('--sigma', type=float, default=0.0, metavar='N',
                        help='standard deviation of AWGN')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    if args.test_AWGN:
        transform = transforms.Compose([transforms.ToTensor()])

        test_dataset  = datasets.MNIST('../data', train=False, transform=transform)

        ##DEBUG STEP
        print(type(test_dataset[0][0]))
        print(test_dataset[0][0].shape)
        print(type(test_dataset[0][0].numpy()))
        print(test_dataset[0][0].numpy().shape)

        # Display effects of AWGN on first image
        fig = plt.figure()
        image = test_dataset[0][0]
        print(image.type())

        for i, sigma in enumerate([0.001, 0.3, 0.6, 1.0]):
            add_noise = AWGN(sigma)
            transformed_sample = add_noise(image)
            # print(transformed_sample)

            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            plt.axis('off')

            ax.set_title("$\sigma = ${}".format(sigma))
            plt.imshow(transformed_sample.numpy().squeeze(), cmap='gray')

        fig.suptitle("Image effects of different noise levels")
        plt.show()


        #

        # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    else:
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])


        dataset1 = datasets.MNIST('../data', train=True, download=True,
                           transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                           transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
