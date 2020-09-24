import torch


def create_loader(
    dataset, *, dryrun=False, batch_size=1, num_workers=0, collate_fn=None
):
    """Create data loader from dataset

    Note: The data loader here is a pytorch data loader object which does not
    assume tensor_type to be pytorch tensor. We only require input dataset to
    support `__getitem__` and `__len__` mothod to iterate over items in
    the dataset.

    Since `collate_fn` method in `torch.utils.data.DataLoader` behave
    differently when automatic batching is on, we might need to override
    this method. If `create_loader` method became too complicated in order to
    support different estimators, we might expect different estimator to
    have their own create_loader method.

    https://pytorch.org/docs/stable/data.html#working-with-collate-fn

    Args:
        dataset (Dataset): dataset object derived from
            `datasetinsights.data.datasets.Dataset` class.
        dryrun (bool): indicator whether to use a very small subset of the
            dataset. This subset is useful to make sure we can quickly run
            estimator without loading the whole dataset. (default: False)
        batch_size (int): how many samples per batch to load (default: 1)
        num_workers (int): number of parallel workers used for data loader.
            Set to `0` to run on a single thread (instead of `1` which might
            introduce overhead). (default: 0)
    Returns:
        `torch.utils.data.DataLoader` object as data loader
    """
    if dataset.split == "train":
        sampler = torch.utils.data.RandomSampler(dataset)
        drop_last = True
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
        drop_last = False

    if dryrun:
        sampler = torch.utils.data.SubsetRandomSampler(range(batch_size))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return data_loader
