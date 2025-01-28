#!/usr/bin/env python3

from __future__ import annotations
import asyncio
import pathlib
from yandex_cloud_ml_sdk import AsyncYCloudML
from yandex_cloud_ml_sdk.auth import YandexCloudCLIAuth


def local_path(path: str) -> pathlib.Path:
    return pathlib.Path(__file__).parent / path


async def main():

    sdk = AsyncYCloudML(
        folder_id="b1g9no86lklqacng5kr3",
        auth="",
    )

    # Создаем датасет для дообучения базовой модели YandexGPT Lite
    dataset_draft = sdk.datasets.from_path_deferred(
        task_type="TextToTextGeneration",
        path="dataset.jsonl",
        upload_format="jsonlines",
        name="YandexGPT tuning",
    )

    # Дождемся окончания загрузки данных и создания датасета
    operation = await dataset_draft.upload()
    dataset = await operation
    print(f"new {dataset=}")


if __name__ == "__main__":
    asyncio.run(main())
