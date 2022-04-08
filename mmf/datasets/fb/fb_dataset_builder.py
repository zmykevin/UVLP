import logging

from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.utils.configuration import get_global_config
from mmf.utils.distributed import get_world_size


logger = logging.getLogger(__name__)


# TODO: Handle checkpointing for onbox datamodule later
# TODO: Handle shutdown of onbox datamodule gracefully later
class FBDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, dataset_name, *args, **kwargs):
        super().__init__(dataset_name, *args, **kwargs)
        use_onbox = get_global_config("training.use_onbox_dataloader") or False
        self._use_onbox = use_onbox and get_world_size() > 1

    def _unwrap_dataset(self, dataset):
        from pytorch.data.fb.iterable_dataset_wrapper import IterableDatasetWrapper

        # Handle backwards compatibility for builders returning instances
        # of IterableDatasetWrapper. Fallback to OSS in case of world size == 1
        if isinstance(dataset, IterableDatasetWrapper):
            current_dataset = dataset.dataset
            current_dataset.dataset_name = dataset.dataset_name
            current_dataset.dataset_type = dataset.dataset_type
            if hasattr(current_dataset, "format_for_prediction"):
                current_dataset.format_for_prediction = dataset.format_for_prediction

            dataset = current_dataset

        return dataset

    def setup(self, *args, **kwargs):
        from stl.lightning.data.on_box_data_module import OnBoxDataModule

        super().setup(self, *args, **kwargs)
        # TODO: Later handle config options through dataloader args
        # Note that onbox dataloader only works when distributed module is
        # initialized which in case of world_size == 1 in MMF is not
        if self._use_onbox:
            self.onbox_datamodule = OnBoxDataModule(
                identity_prefix=self.dataset_name,
                # Full Sync should be used in all cases except one GPU
                # but we don't support distributed in one GPU case and hence
                # we don't need it.
                use_full_sync_dataloader=True,
            )

            if get_world_size() == 1:
                logger.warning(
                    "Falling back to OSS dataloader as MMF doesn't "
                    + "support DDP for world size == 1"
                )
            self.onbox_datamodule.train_dataset = self._unwrap_dataset(
                self.train_dataset
            )
            self.onbox_datamodule.val_dataset = self._unwrap_dataset(self.val_dataset)
            self.onbox_datamodule.test_dataset = self._unwrap_dataset(self.test_dataset)

    def build_dataloader(self, dataset_instance, dataset_type):
        if self._use_onbox and hasattr(self, "onbox_datamodule"):
            return getattr(self.onbox_datamodule, f"{dataset_type}_dataloader")()
        else:
            return super().build_dataloader(dataset_instance, dataset_type)

    def teardown(self, *args, **kwargs):
        if self._use_onbox:
            self.onbox_datamodule.teardown()
