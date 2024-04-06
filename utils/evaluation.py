import torch
from torch.nn import functional as F

def loss_f_funciton(labels, parameters, data):
    output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
    loss = F.cross_entropy(output, labels)
    return loss

def out_f(data, parameters):
    output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
    return output

def reg_f(params, hparams, loss):
    loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(hparams))) + 0.01*torch.pow(torch.norm(params,2),2)
    return loss_regu

def test_avg(data_loader, parameters, device):
    loss_avg, num = 0.0, 0
    correct_predictions = 0
    total_samples = 0
    for _, (images, labels) in enumerate(data_loader):
        images = torch.reshape(images, (images.size()[0], -1)).to(device)
        labels = labels.to(device)

        logits = torch.matmul(images, torch.t(parameters[:, :784])) + parameters[:, 784]
        predictions = torch.argmax(logits, dim=1)

        correct_predictions += torch.sum(predictions == labels).item()
        total_samples += labels.size(0)

        loss = F.cross_entropy(logits, labels)
        loss_avg += loss 
        num += 1

    loss_avg = loss_avg/num
    accuracy = correct_predictions / total_samples
    return accuracy, loss_avg.detach()


def loss_train_avg(data_loader, parameters, device, batch_num):
    loss_avg, num = 0.0, 0
    for index, (images, labels) in enumerate(data_loader):
        if index>= batch_num:
            break
        else:
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels = labels.to(device)
            loss = loss_f_funciton(labels, parameters, images)
            loss_avg += loss 
            num += 1
    loss_avg = loss_avg/num
    return loss_avg.detach()


def gradient_fy(args, labels, params, data, output):
    loss = F.cross_entropy(output, labels)
    grad = torch.autograd.grad(loss, params)[0]
    return grad

def gradient_gy(args, labels_cp, params, data, hparams, output, reg_f, create_graph=True):
    # For MNIST data-hyper cleaning experiments
    loss = F.cross_entropy(output, labels_cp, reduction='none')
    # For NewsGroup l2reg expriments
    # loss = F.cross_entropy(output, labels_cp)
    loss_regu = reg_f(params, hparams, loss)
    grad = torch.autograd.grad(loss_regu, params, create_graph=create_graph)[0]
    return grad

def gradient_gx(args, labels_cp, params, data, hparams, output, reg_f, create_graph=True):
    # For MNIST data-hyper cleaning experiments
    loss = F.cross_entropy(output, labels_cp, reduction='none')
    # For NewsGroup l2reg expriments
    # loss = F.cross_entropy(output, labels_cp)
    loss_regu = reg_f(params, hparams, loss)
    grad = torch.autograd.grad(loss_regu, hparams, create_graph=create_graph)[0]
    return grad
