import hparams

from torch.utils.data import DataLoader
from data_utils import TextMelLoader, TextMelCollate


def prepare_dataloaders():
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_sampler = None
    shuffle = False
    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader


if __name__ == "__main__":
    train_loader = prepare_dataloaders()
