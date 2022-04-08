# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from dataclasses import dataclass
from typing import Type

from mmf.common.registry import registry
from mmf.datasets.fb.builders.hive.builder import HiveConfigType, HiveDatasetBuilder
from mmf.utils.general import get_batch_size
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

EVERSTORE_GROUP_ID = 3402181706576512
EVERSTORE_FEATURE_ID = 41556585


@dataclass
class EverstoreConfigType(HiveConfigType):
    # column for everstore handle
    everstore_col: str
    # column for post_id which will be used to fetch everstore handle
    feature_store_post_id_col: str
    # if read soft-delete images, important to integrity domain
    read_deleted: bool


@registry.register_builder("everstore")
class EverstoreDatasetBuilder(HiveDatasetBuilder):
    def __init__(self, dataset_name="everstore", *args, **kwargs):
        super().__init__(dataset_name, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/fb/datasets/everstore.yaml"

    def _build_fs_everstore_handle_fetcher(self, post_id_col: str):
        from pytorch.data.fb.enrichment import FeatureStoreEnrichment

        return FeatureStoreEnrichment(
            lookup_key_2_value={"post_id": "cast({} as varchar)".format(post_id_col)},
            output_columns=["everstore_handle"],
            features=FeatureStoreEnrichment.Features(
                group_id=EVERSTORE_GROUP_ID, feature_ids=[EVERSTORE_FEATURE_ID]
            ),
        )

    def load(
        self, config: EverstoreConfigType, dataset_type: str = "train", *args, **kwargs
    ) -> Type[Dataset]:
        from pytorch.data.fb.dataset_base import InMemoryShuffleSpec
        from pytorch.data.fb.enrichment import (
            EverstoreEnrichment,
        )
        from pytorch.data.fb.hive_dataset import (
            HiveDataset,
            HiveShuffleSpec,
            HiveSplitShuffleOrder,
        )
        from pytorch.data.fb.iterable_dataset_wrapper import IterableDatasetWrapper

        warning_msg = "Only set everstore_col or feature_store_post_id_col"
        assert (
            not config.everstore_col or not config.feature_store_post_id_col
        ), warning_msg

        filters = self.build_filters(config, dataset_type)
        schema = list(config.schema)
        # Image data collected from everstore_col will be in "image"
        # key of the data dict returned from the the dataset
        schema.append("image")

        # Setting for fetching everstore handle

        enrichments = [
            EverstoreEnrichment(
                lookup_value=config.everstore_col
                if config.everstore_col
                else "everstore_handle",
                output_column="image",
                # read soft-deleted contents for integrity cases
                options=EverstoreEnrichment.Options(
                    read_deleted=config.get("read_deleted", True),
                    repeat_when_throttled=True,
                ),
            )
        ]
        if config.feature_store_post_id_col:
            enrichments.insert(
                0,
                self._build_fs_everstore_handle_fetcher(
                    config.feature_store_post_id_col
                ),
            )

        hive_dataset = (
            HiveDataset(**config.constructor)
            .enrichments(enrichments)
            .schema(schema)
            .filter(filters)
            .transform(self.build_transform(config))
            .batch(get_batch_size(), drop_incomplete=True)
        )

        if dataset_type == "train":
            # TODO: to also expose the rest of shuffle orders
            logger.info("Using HiveSplitShuffleOrder.SHUFFLE_ALL")
            hive_dataset.shuffle(
                HiveShuffleSpec(
                    shuffle_order=HiveSplitShuffleOrder.SHUFFLE_ALL,
                )
            )
            if config.get("memory_limit_in_bytes", -1) > 0:
                logger.info(
                    "Using InMemoryShuffleSpec with memory_limit_in_bytes"
                    + f" = {config.memory_limit_in_bytes}"
                )
                hive_dataset.shuffle(
                    InMemoryShuffleSpec(
                        memory_limit_in_bytes=config.memory_limit_in_bytes,
                    )
                )

        dataset = IterableDatasetWrapper(hive_dataset)
        dataset.dataset_name = self.dataset_name
        dataset.dataset_type = dataset_type

        dataset.format_for_prediction = self.build_transform(
            config, "prediction_processor"
        )
        return dataset
