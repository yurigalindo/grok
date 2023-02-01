import os, argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import random
import matplotlib.pyplot as plt
import numpy as np

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def data_preparation(args, seed_id=42, test_samples=100, test_batchsize=100):
    # Data generation - different seed each time
    data = parity(args.n, args.k, args.N, seed=seed_id*17)
    train_dataset = TensorDataset(data[0], data[1])
    train_dataloader = DataLoader(train_dataset, batch_size=args.B, shuffle=True)

    data = parity(args.n, args.k, test_samples, seed=2001) # fixed test samples
    test_dataset = TensorDataset(data[0], data[1])
    test_dataloader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=True)

    return train_dataloader, test_dataloader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=40, type=int, help='string dimension')
    parser.add_argument('--k', default=3, type=int, help='parity dimension')
    parser.add_argument('--N', default=1000, type=int, help='number of training samples')
    parser.add_argument('--B', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=0.01, type=float, help='value of weight decay')
    parser.add_argument('--width', default=1000, type=int, help='width of network')
    parser.add_argument('--n_seeds', default=5, type=int, help='number of random seeds')
    parser.add_argument('--train', action='store_true', help='train models')
    parser.add_argument('--sparsity-sampling', default=10, help='every how many epochs we compute the sparsity of the network')
    
    return parser.parse_args()

def main():
    args = get_args()
    loss_fn = MyHingeLoss()

    base_dir = f'./results/n_{args.n}/k_{args.k}/N_{args.N}/lr_{args.lr}/wd_{args.weight_decay}/width_{args.width}'
    os.makedirs(base_dir, exist_ok=True)

    fig_path = os.path.join(base_dir, 'figures')
    os.makedirs(fig_path, exist_ok=True)
    
    losses, accs, normss = {'train': [], 'test': []}, {'train': [], 'test': []}, []
    mem_epochs, gen_epochs = [], []
    if args.train:
        for seed_id in range(args.n_seeds):
            torch.manual_seed(seed_id)

            # Data & save_dir preparation
            train_dataloader, test_dataloader = data_preparation(args, seed_id)
            path = os.path.join(base_dir, f'seed{seed_id}_checkpoints')
            os.makedirs(path, exist_ok=True)

            # Model & Optim initialization
            model = FF1(input_dim=args.n, width=args.width)
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            mem_epochs.append(-1)
            gen_epochs.append(-1)
            train_loss, test_loss = [], []
            train_acc, test_acc = [], []
            norms = {'feats':  [],
                    'conx':  []}
            for epoch in range(args.epochs):
                if (epoch % 100 == 0):
                    print(f"Epoch {epoch + 1}\n-------------------------------")

                # Norm statistics
                norms['feats'].append(torch.linalg.norm(list(model.parameters())[0], dim=1).detach().cpu().numpy())
                norms['conx'].append(torch.squeeze(list(model.parameters())[2]).detach().cpu().numpy())

                # Loss & Accuracy statistics
                train_loss.append(loss_calc(train_dataloader, model, loss_fn))
                test_loss.append(loss_calc(test_dataloader, model, loss_fn))

                train_acc.append(acc_calc(train_dataloader, model))
                test_acc.append(acc_calc(test_dataloader, model))

                # Save memorizing / generalizing network
                if (train_acc[-1] > 0.98 and mem_epochs[-1] < 0):
                    print(f'Saving memorizing model - epoch {epoch}')
                    torch.save(model.state_dict(), os.path.join(path, 'memorization.pt'))
                    mem_epochs[-1] = epoch
                if (test_acc[-1] > 0.98 and gen_epochs[-1] < 0):
                    print(f'Saving initially generalizing model - epoch {epoch}')
                    torch.save(model.state_dict(), os.path.join(path, 'initial_generalization.pt'))
                    gen_epochs[-1] = epoch
                if (epoch == args.epochs - 1):
                    print(f'Saving (final) generalizing model - epoch {epoch}')
                    torch.save(model.state_dict(), os.path.join(path, 'generalization.pt'))
                    gen_epochs[-1] = epoch

                # Save model
                torch.save(model.state_dict(), os.path.join(path, f'model_{epoch}.pt'))
                # Train model
                for id_batch, (x_batch, y_batch) in enumerate(train_dataloader):

                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred = model(x_batch)

                    optimizer.zero_grad()
                    loss = loss_fn(pred, y_batch).mean()
                    loss.backward()
                    optimizer.step()
            
            losses['train'].append(train_loss)
            losses['test'].append(test_loss)  

            accs['train'].append(train_acc)
            accs['test'].append(test_acc)

            normss.append(norms)


        # Save files
        with open(os.path.join(base_dir, 'normss'), "wb") as fp:
            pickle.dump(normss, fp)

        with open(os.path.join(base_dir, 'mem_epochs'), "wb") as fp:
            pickle.dump(mem_epochs, fp)

        with open(os.path.join(base_dir, 'gen_epochs'), "wb") as fp:
            pickle.dump(gen_epochs, fp)

        # Save and Plot train (test) curves (acc & loss)
        m1, std1 = mean_and_std_across_seeds(losses['train'])
        np.save(os.path.join(base_dir, 'mean_train_loss'), m1)
        np.save(os.path.join(base_dir, 'std_train_loss'), std1)

        m2, std2 = mean_and_std_across_seeds(losses['test'])
        np.save(os.path.join(base_dir, 'mean_test_loss'), m2)
        np.save(os.path.join(base_dir, 'std_test_loss'), std2)

        m3, std3 = mean_and_std_across_seeds(accs['train'])
        np.save(os.path.join(base_dir, 'mean_train_acc'), m3)
        np.save(os.path.join(base_dir, 'std_train_acc'), std3)

        m4, std4 = mean_and_std_across_seeds(accs['test'])
        np.save(os.path.join(base_dir, 'mean_test_acc'), m4)
        np.save(os.path.join(base_dir, 'std_test_acc'), std4)

        plt.plot(m1, linestyle='-', label='train')
        plt.plot(m2, linestyle='-', label='test')
        plt.fill_between([i for i in range(args.epochs)], m1 - std1, m1 + std1, alpha = 0.3)
        plt.fill_between([i for i in range(args.epochs)], m2 - std2, m2 + std2, alpha = 0.3)
        plt.title('Loss')
        plt.xlabel('epochs')
        plt.xscale('log')
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'loss.pdf'))
        plt.close()

        plt.plot(m3, linestyle='-', label='train')
        plt.plot(m4, linestyle='-', label='test')
        plt.fill_between([i for i in range(args.epochs)], m3 - std3, m3 + std3, alpha = 0.3)
        plt.fill_between([i for i in range(args.epochs)], m4 - std4, m4 + std4, alpha = 0.3)
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.xscale('log')
        plt.legend()
        plt.savefig(os.path.join(fig_path, 'acc.pdf'))
        plt.close()

    else:
        with open(os.path.join(base_dir, 'normss'), "rb") as fp:
            normss = pickle.load(fp)

        with open(os.path.join(base_dir, 'mem_epochs'), "rb") as fp:
            mem_epochs = pickle.load(fp)

        with open(os.path.join(base_dir, 'gen_epochs'), "rb") as fp:
            gen_epochs = pickle.load(fp)


    # Show norm evolutions
    for seed_id in range(args.n_seeds):
        best = normss[seed_id]['feats'][gen_epochs[seed_id]].argmax()
        prev_best = normss[seed_id]['feats'][mem_epochs[seed_id]].argmax()

        traj, prev_traj = [], []
        for k in range(args.epochs):
            traj.append(normss[seed_id]['feats'][k][best])
            prev_traj.append(normss[seed_id]['feats'][k][prev_best])

        plt.plot(traj, label='generalizing neuron', lw=2, color='crimson')
        plt.plot(prev_traj, label='memorizing neuron', lw=2, color='navy')
        plt.title(f'Norm evolution of neurons belonging to different circuits - seed {seed_id}')
        plt.xscale('log')
        plt.xlabel('epochs')
        plt.ylabel(r'$\| w \|$')
        plt.legend()
        plt.savefig(os.path.join(fig_path, f'norm_contrast_seed{seed_id}.pdf'))
        plt.show()
        plt.close()

        trajs = []
        for neuron in range(args.width):
            trajs.append([])
            for k in range(args.epochs):
                trajs[-1].append(normss[seed_id]['feats'][k][neuron])

        for neuron in range(args.width):
            plt.plot(trajs[neuron])
        plt.title(f'Norm evolution of all neurons - seed {seed_id}')
        plt.xlabel('epochs')
        plt.ylabel(r'$\| w \|$')
        plt.xscale('log')
        plt.savefig(os.path.join(fig_path, f'all_neurons_norm_seed{seed_id}.pdf'))
        plt.show()
        plt.close()


    # Global sparsity over time
    sparsities = []
    for seed_id in range(args.n_seeds):
        sparsity = []
        path = os.path.join(base_dir, f'seed{seed_id}_checkpoints')
        train_dataloader, _ = data_preparation(args, seed_id) # load the training dataset of that seed_id
        for epoch in range(0, args.epochs, args.sparsity_sampling):
            _model = FF1().to(device)
            _model.load_state_dict(torch.load(os.path.join(path, f'model_{epoch}.pt')))
            if (epoch < 20):
                # warm up for irregular behavior in the beginning - non monotonic accuracy (pruning helps apparently?)
                size, _ = circuit_discovery_linear(epoch, _model, normss[seed_id], train_dataloader, device)
            else:
                size, _ = circuit_discovery_binary(epoch, _model, normss[seed_id], train_dataloader, device)
                if (size == float('inf')): 
                    # binary search failed?!
                    size, _ = circuit_discovery_linear(epoch, _model, normss[seed_id], train_dataloader, device)

            sparsity.append(size)
        sparsities.append(sparsity)

    mean_sparsity, std_sparsity = mean_and_std_across_seeds(sparsities)
    np.save(os.path.join(base_dir, 'mean_sparsity_over_time'), mean_sparsity)
    np.save(os.path.join(base_dir, 'std_sparsity_over_time'), std_sparsity)

    plt.plot([i for i in range(0, args.epochs, args.sparsity_sampling)], mean_sparsity, linestyle='-')
    plt.fill_between([i for i in range(0, args.epochs, args.sparsity_sampling)], mean_sparsity - std_sparsity, mean_sparsity + std_sparsity, alpha = 0.3)
    plt.title('Sparsity of network')
    plt.xlabel('epochs')
    plt.ylabel('#neurons')
    plt.xscale('log')
    plt.savefig(os.path.join(fig_path, 'sparsity.pdf'))
    plt.close()

    # Subnetworks calculations & reconstruction accuracy
    # for seed_id in range(args.n_seeds):
    #     sparsity = []
    #     path = os.path.join(base_dir, f'seed{seed_id}_checkpoints')
    #     train_dataloader, _ = data_preparation(args, seed_id) # load the training dataset of that seed_id
    # mem_model = FF1().to(device)
    # mem_model.load_state_dict(torch.load(os.path.join(path, f'memorization.pt')))
    # mem_size, mem_idx = circuit_discovery_linear(epoch, mem_model, normss[seed_id], train_dataloader, device)
    # print(f'Memorizing circuit has size equal to {mem_size}')

    # init_gen_model = FF1().to(device)
    # init_gen_model.load_state_dict(torch.load(os.path.join(path, f'initial_generalization.pt')))
    # init_gen_size, init_gen_idx = circuit_discovery_linear(epoch, init_gen_model, normss[seed_id], train_dataloader, device)
    # print(f'Initial generalizing circuit has size equal to {mem_size}')

    # gen_model = FF1().to(device)
    # gen_model.load_state_dict(torch.load(os.path.join(path, f'generalization.pt')))
    # mem_size, mem_idx = circuit_discovery_linear(epoch, _model, normss[seed_id], train_dataloader, device)
    # print(f'Memorizing circuit has size equal to {mem_size}')


if __name__ == '__main__':
    main()
