from itertools import cycle
from typing import List, Optional, Union

from scvi.data import AnnDataManager
from scvi.dataloaders._concat_dataloader import ConcatDataLoader


class _scCausalIterator:
    """
    Iterator for background and target dataloader pairs as found in the contrastive
    analysis setting.

    Each iteration of this iterator returns a dictionary with two elements:
    "background", containing one batch of data from the background dataloader, and
    "target", containing one batch of data from the target dataloader.
    """

    def __init__(self, background, target):
        self.background = iter(background)
        self.target = iter(target)

    def __iter__(self):
        return self

    def __next__(self):
        bg_samples = next(self.background)
        tg_samples = next(self.target)
        return {"background": bg_samples, "target": tg_samples}


class scCausalDataLoader(ConcatDataLoader):
    """
    Data loader to load data of each condition for scCausalVAE model.

    Each iteration of the data loader returns a dictionary containing background and
    target data points, indexed by "background" and "target", respectively.
    Args:
    ----
        adata: AnnData object that has been registered via `setup_anndata`.
        indices_list: List where each element is a list of indices in the adata to load
        shuffle: Whether the data should be shuffled.
        batch_size: Mini-batch size to load for background and target data.
        data_and_attributes: Dictionary with keys representing keys in data
            registry (`adata.uns["_scvi"]`) and value equal to desired numpy
            loading type (later made into torch tensor). If `None`, defaults to all
            registered data.
        drop_last: If int, drops the last batch if its length is less than
            `drop_last`. If `drop_last == True`, drops last non-full batch.
            If `drop_last == False`, iterate over all batches.
        **data_loader_kwargs: Keyword arguments for `torch.utils.data.DataLoader`.
    """

    def __init__(
            self,
            adata_manager: AnnDataManager,
            indices_list: List[List[int]],
            shuffle: bool = False,
            batch_size: int = 128,
            data_and_attributes: Optional[dict] = None,
            drop_last: Union[bool, int] = False,
            **data_loader_kwargs,
    ) -> None:
        super().__init__(
            adata_manager=adata_manager,
            indices_list=indices_list,
            shuffle=shuffle,
            batch_size=batch_size,
            data_and_attributes=data_and_attributes,
            drop_last=drop_last,
            **data_loader_kwargs,
        )

    def __iter__(self):
        """

        Iter method for scCausal data loader.

        Will iter over the dataloader with the most data while cycling through
        the data in the other dataloaders.
        """

        iter_list = [
            cycle(dl) if dl != self.largest_dl else dl for dl in self.dataloaders
        ]

        # return _scCausalIterator(background=iter_list[0], target=iter_list[1])
        return zip(*iter_list)