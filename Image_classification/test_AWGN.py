from __future__ import print_function
import argparse
import torch
import main
from torchvision import datasets, transforms



def run():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test-AWGN', action='store_true', default=False,
                        help='Test AWGN transformation')
    parser.add_argument('--sigma', type=float, default=0.0, metavar='N',
                        help='standard deviation of AWGN')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    test_kwargs = {'batch_size': args.test_batch_size}

    ## import model from main and import weights
    model = main.Net().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    for i, sigma in enumerate([0, 0.3, 0.6, 1.0]):
        transform = transforms.Compose([transforms.ToTensor(), main.AWGN(sigma), transforms.Normalize((0.1307,), (0.3081,))])

        test_dataset = datasets.MNIST('../data', train=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        print(f"\nAccuracy for AWGN with a standard deviation of {sigma}")
        main.test(model, device, test_loader)


if __name__ == '__main__':
    run()