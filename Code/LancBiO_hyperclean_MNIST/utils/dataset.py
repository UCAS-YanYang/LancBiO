import torch
import random
from torchvision import datasets
import torchvision.transforms as transforms


def get_data_loaders(args):
    val = args.validation_size
    tr = args.training_size


    dataset = datasets.MNIST(root=args.data_path, train=True, download=True,
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    
    kwargs = {'num_workers': 0, 'pin_memory': True}

    tr_subset_indices = range(tr)
    tr_subset_dataset = torch.utils.data.Subset(dataset, tr_subset_indices)
    train_sampler = torch.utils.data.sampler.SequentialSampler(tr_subset_dataset)
    train_loader = torch.utils.data.DataLoader(tr_subset_dataset, sampler=train_sampler,
        batch_size=args.batch_size, **kwargs)

    
    val_subset_indices = random.sample(range(len(dataset)), val)
    val_subset_dataset = torch.utils.data.Subset(dataset, val_subset_indices)
    val_loader = torch.utils.data.DataLoader(val_subset_dataset, shuffle=True, 
        batch_size=args.batch_size, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=args.data_path, train=False,
                        download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=args.test_size)
    return train_loader, test_loader, val_loader

def nositify(labels, noise_rate, n_class):
    num = noise_rate*(labels.size()[0])
    num = int(num)
    randint = torch.randint(1, 10, (num,))
    index = torch.randperm(labels.size()[0])[:num]
    labels[index] = (labels[index]+randint) % n_class
    return labels

def build_val_data(args, val_index, images_val_list, labels_val_list, images_list, labels_list, device):
    val_index = -(val_index)
    val_data_list, val_labels_list = [], []
    
    images, labels = images_val_list[val_index[0]], labels_val_list[val_index[0]]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels = labels.to(device)
    val_data_list.append(images)
    val_labels_list.append(labels)

    ind_yy = random.randint(0, len(images_list)-1)
    images, labels = images_list[ind_yy], labels_list[ind_yy]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels_cp = labels.to(device)
    val_data_list.append(images)
    val_labels_list.append(labels_cp)

    ind_xy = random.randint(0, len(images_list)-1)
    images, labels = images_list[ind_xy], labels_list[ind_xy]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels_cp = labels.to(device)
    val_data_list.append(images)
    val_labels_list.append(labels_cp)

    return [(val_data_list, val_labels_list), ind_yy, ind_xy]