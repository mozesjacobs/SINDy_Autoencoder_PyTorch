import torch
from tqdm import tqdm

def train(net, train_loader, train_board, optim, epoch, clip, lambdas):
    net.train()

    # losses
    epoch_l_recon = 0
    epoch_l_dx = 0
    epoch_l_dz = 0
    epoch_l_reg = 0

    # for each batch
    for x, dx in tqdm(train_loader, desc="Training", total=len(train_loader), dynamic_ncols=True):
        l_recon, l_dx, l_dz, l_reg = net(x, dx, lambdas)
        epoch_l_recon += l_recon.item()
        epoch_l_dx += l_dx.item()
        epoch_l_dz += l_dz.item()
        epoch_l_reg += l_reg.item()

        # backprop
        batch_loss = (l_recon + l_dx + l_dz + l_reg) / len(x)
        optim.zero_grad()
        batch_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()    
    
    # average
    num_batches = len(train_loader)
    epoch_l_recon /= num_batches
    epoch_l_dx /= num_batches
    epoch_l_dz /= num_batches
    epoch_l_reg /= num_batches

    # tensorboard
    train_board.add_scalar('L recon', epoch_l_recon, epoch)
    train_board.add_scalar('L dx', epoch_l_dx, epoch)
    train_board.add_scalar('L dz', epoch_l_dz, epoch)
    train_board.add_scalar('L regularization', epoch_l_reg, epoch)


def test(net, test_loader, test_board, epoch, timesteps, lambdas):
    net.eval()
    total_recon, total_dx, total_dz, total_reg = 0, 0, 0, 0
    for x, dx in tqdm(test_loader, desc="Testing", total=len(test_loader), dynamic_ncols=True):
        l_recon, l_dx, l_dz, l_reg = net(x, dx, lambdas)
        total_recon += l_recon.item()
        total_dx += l_dx.item()
        total_dz += l_dz.item()
        total_reg += l_reg.item()
    num_batches = len(test_loader)
    test_board.add_scalar('L recon', total_recon / num_batches, epoch)
    test_board.add_scalar('L dx', total_dx / num_batches, epoch)
    test_board.add_scalar('L dz', total_dz / num_batches, epoch)
    test_board.add_scalar('L regularization', total_reg / num_batches, epoch)
