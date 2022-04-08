# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union

from mmf.common.registry import registry
from mmf.datasets.fb.fb_dataset_builder import FBDatasetBuilder
from mmf.utils.build import build_processors
from mmf.utils.general import get_batch_size
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SetNameMappingType:
    train: str
    val: str
    test: str


@dataclass
class BatchProcessorType:
    type: str
    params: Dict[str, Any]


@dataclass
class HiveProcessorsType:
    batch_processor: BatchProcessorType


@dataclass
class HiveDatasetConstructorType:
    namespace: str
    table: str
    partitions: Union[None, str]


@dataclass
class HiveConfigType:
    # Config type encapsulates parameters
    # to functions that pytorch.data.fb.HiveDataset
    # provides for making Hive queries

    # Constructor arguments are directly passed to
    # pytorch.data.fb.HiveDataset's constructor
    # Please follow its documentation to understand
    # which all options are support
    constructor: HiveDatasetConstructorType
    # The table column which defined which set a sample
    # belongs to. For e.g. train, val, test
    set_name_key: str
    # Set name mapping defines which values map to train, val and test
    # for the column `set_name_key`. For example, your train can be
    # called training, val -> validation, test -> testing
    # Specify a dict which maps train, val and test to a corresponding
    # values
    set_name_mapping: SetNameMappingType
    # What all columns need to be pulled from the Table and if they
    # map to something specific like Everstore key handle etc
    schema: List[str]
    # Filters are same as filters that you use to limit your hive
    # SQL query. A default filter is applied by MMF to limit only
    # the rows particular to current set (train|val|test)
    filters: List[str]
    # processors you are going to use in this dataset. By default,
    # batch_processor defined here will be used and the data from
    # HiveDataset will be passed to your `batch_processor` which
    # you can use to adapt the data according to your model and
    # then return a SampleList
    processors: HiveProcessorsType
    # memory_limit_in_bytes for HiveDataset InMemoryShuffleSpec
    memory_limit_in_bytes: int


@registry.register_builder("hive")
class HiveDatasetBuilder(FBDatasetBuilder):
    def __init__(self, dataset_name="hive", *args, **kwargs):
        super().__init__(dataset_name, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/fb/datasets/hive.yaml"

    def build(self, config: HiveConfigType, *args, **kwargs):
        # No need to build in fb datasets
        pass

    def build_transform(self, config: HiveConfigType, key: str = "batch_processor"):
        processor_config = config.processors
        extra_args = {"data_dir": config.data_dir}
        processor_dict = build_processors(processor_config, **extra_args)
        return processor_dict[key]

    def build_filters(self, config: HiveConfigType, dataset_type: str):
        filters = list(config.get("filters", []))
        if "set_name_mapping" in config and "set_name_key" in config:
            set_name = config.set_name_mapping[dataset_type]
            set_filter = f"{config.set_name_key}='{set_name}'"
            filters.append(set_filter)
        return filters

    def load(
        self, config: HiveConfigType, dataset_type: str = "train", *args, **kwargs
    ) -> Type[Dataset]:
        from pytorch.data.fb.dataset_base import InMemoryShuffleSpec
        from pytorch.data.fb.hive_dataset import (
            HiveDataset,
            HiveShuffleSpec,
            HiveSplitShuffleOrder,
        )
        from pytorch.data.fb.iterable_dataset_wrapper import IterableDatasetWrapper

        filters = self.build_filters(config, dataset_type)
        hive_dataset = (
            HiveDataset(**config.constructor)
            .schema(config.schema)
            .filter(filters)
            .transform(self.build_transform(config))
            .batch(get_batch_size())
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
