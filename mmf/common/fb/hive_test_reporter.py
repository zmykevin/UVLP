# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import pytorch_lightning as pl
from mmf.common.registry import registry
from mmf.common.test_reporter import TestReporter
from mmf.utils.distributed import is_main
from omegaconf import MISSING
import random
from datetime import datetime

logger = logging.getLogger(__name__)


def get_tmp_tablename(tableprefix="tmp_mmf"):
    return (tableprefix + "_" + str(uuid.uuid4())).replace("-", "_")


@registry.register_test_reporter("hive")
class HiveTestReporter(TestReporter):
    @dataclass
    class Config(TestReporter.Config):
        # Table namespace
        namespace: str = MISSING
        # Table name
        table_name: str = MISSING
        # the field that will be used as primary key
        id_field: str = MISSING
        # oncall
        oncall: str = MISSING
        # number of days to retain the table
        retention_days: int = 1
        # How often to write to hive
        flush_every: int = 2
        # num threads to be used by hive
        hive_num_threads: int = 8
        # batch size used by hive for writing
        hive_batch_size: int = 100
        # hive writer qps_report_interval
        qps_report_interval: int = 1024

    def __init__(
        self,
        datamodules: List[pl.LightningDataModule],
        config: Config,
        dataset_type: str = "train",
    ):
        super().__init__(datamodules, config, dataset_type)

        self.run_id = os.environ.get("WORKFLOW_RUN_ID")
        if self.run_id:
            self.run_id = int(self.run_id)
        else:
            # assing a random number as run_id
            self.run_id = random.Random(datetime.now()).randint(0, 1000)
        self.batch_number = 0
        self.dump_thread = None

        assert (
            self.test_reporter_config.id_field
        ), "id_field cannot be missing for hive test reporter"

        if "table_name" not in self.test_reporter_config:
            message = "oncall must be specified to create temp tables"
            assert self.test_reporter_config.oncall, message
            if is_main():
                self.test_reporter_config.table_name = get_tmp_tablename()
                logger.warning(
                    "No table name was provided. Creating temp table "
                    + f"{self.test_reporter_config.namespace}."
                    + f"{self.test_reporter_config.table_name}"
                )
                logger.warning(
                    "Note that this will create different table \
                    for each dataset type (i.e val, test)"
                )
        if is_main():
            self.setup_table(
                self.test_reporter_config.namespace,
                self.test_reporter_config.table_name,
                self.test_reporter_config.retention_days,
                self.test_reporter_config.oncall,
            )

            logger.info(
                f"Flushing writer every {self.test_reporter_config.flush_every}"
            )

    def flush_report(self):
        if not is_main():
            self.report = []
            return

        if self.dump_thread:
            logger.info("Waiting for dump thread to complete...")
            self.dump_thread.join()

        self.dump_thread = threading.Thread(target=self.hive_dump, args=(self.report,))
        self.dump_thread.start()
        self.report = []

    def hive_dump(self, report):
        import koski.dataframes as kd
        import ujson  # @manual=third-party//ultrajson:ultrajson
        from pytorch.data.fb.hive_writer.hive_writer import (
            HiveWriter,
            Params as HiveWriterParams,
        )

        writer = HiveWriter(
            HiveWriterParams(
                namespace=self.test_reporter_config.namespace,
                table=self.test_reporter_config.table_name,
                ctx=kd.create_test_ctx(),
                partition={
                    "run_id": self.run_id,
                    "shard": self.batch_number,
                    "dataset_name": self.dataset_type,
                },
                num_threads=self.test_reporter_config.hive_num_threads,
                batch_size=self.test_reporter_config.hive_batch_size,
                qps_report_interval=self.test_reporter_config.qps_report_interval,
            )
        )

        for i in range(len(report)):
            writer.add_row(
                [
                    report[i][self.test_reporter_config.id_field],
                    f"{ujson.dumps(report[i])}",
                ]
            )

        writer.flush()
        logger.info(
            "Wrote predictions to {}.{}".format(
                self.test_reporter_config.namespace,
                self.test_reporter_config.table_name,
            )
        )

    def setup_table(self, namespace, table_name, retention_days, oncall):
        from torch.fb.vision.data.writers import hive_table_util

        columns_with_type = OrderedDict([("id", "BIGINT"), ("payload", "VARCHAR")])
        partition_columns_with_type = OrderedDict(
            [("run_id", "BIGINT"), ("shard", "INTEGER"), ("dataset_name", "VARCHAR")]
        )
        create_table_query = hive_table_util.create_hive_table_query(
            table_name,
            columns_with_type,
            partition_columns_with_type,
            retention_days,
            oncall,
        )
        hive_table_util.run_query(namespace, create_table_query)

    def prepare_batch(self, batch):
        self.batch_number += 1
        if self.batch_number % self.test_reporter_config.flush_every == 0:
            self.flush_report()
        return super().prepare_batch(batch)

    def add_to_report(self, *args, **kwargs):
        super().add_to_report(*args, **kwargs)
        # Always empty report in case of HiveTestReporter on ranks
        # other than master to save memory as HiveTestReporter would
        # mostly be used for inference
        if not is_main():
            self.report = []
