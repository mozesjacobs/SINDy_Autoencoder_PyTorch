import torch
from tqdm import tqdm

def train(net, train_loader, train_board, optim, epoch, clip, lambdas):
    net.train()

    # losses
    epoch_l_recon = 0
    epoch_l_dx = 0
    epoch_l_dz = 0
    epoch_l_reg = 0
    epoch_kld = 0

    # for each batch
    for x, dx, dz in tqdm(train_loader, desc="Training", total=len(train_loader), dynamic_ncols=True):
        l_recon, l_dx, l_dz, l_reg, kl = net(x, dx, lambdas)
        epoch_l_recon += l_recon.item()
        epoch_l_dx += l_dx.item()
        epoch_l_dz += l_dz.item()
        epoch_l_reg += l_reg.item()
        epoch_kld += kl.item()

        # backprop
        batch_loss = (l_recon + l_dx + l_dz + l_reg + kl) / len(x)
        optim.zero_grad()
        batch_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()

        # update the mask
        net.threshold_mask[torch.abs(net.sindy_coefficients) < net.sequential_threshold] = 0
    
    # average
    num_batches = len(train_loader)
    epoch_l_recon /= num_batches
    epoch_l_dx /= num_batches
    epoch_l_dz /= num_batches
    epoch_l_reg /= num_batches
    epoch_kld /= num_batches

    # tensorboard
    train_board.add_scalar('L recon', epoch_l_recon, epoch)
    train_board.add_scalar('L dx', epoch_l_dx, epoch)
    train_board.add_scalar('L dz', epoch_l_dz, epoch)
    train_board.add_scalar('L regularization', epoch_l_reg, epoch)
    train_board.add_scalar('KLD', epoch_kld, epoch)


def test(net, test_loader, test_board, epoch, timesteps, lambdas):
    net.eval()
    total_recon, total_dx, total_dz, total_reg, total_kld = 0, 0, 0, 0, 0
    for x, dx, dz in tqdm(test_loader, desc="Testing", total=len(test_loader), dynamic_ncols=True):
        l_recon, l_dx, l_dz, l_reg, kl = net(x, dx, lambdas)
        total_recon += l_recon.item()
        total_dx += l_dx.item()
        total_dz += l_dz.item()
        total_reg += l_reg.item()
        total_kld += kl.item()
    num_batches = len(test_loader)
    test_board.add_scalar('L recon', total_recon / num_batches, epoch)
    test_board.add_scalar('L dx', total_dx / num_batches, epoch)
    test_board.add_scalar('L dz', total_dz / num_batches, epoch)
    test_board.add_scalar('L regularization', total_reg / num_batches, epoch)
    test_board.add_scalar('KLD', total_kld / num_batches, epoch)
