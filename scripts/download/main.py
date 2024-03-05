import os
import pathlib
import urllib.parse
import zipfile

import hydra
import requests
from loguru import logger
from omegaconf import DictConfig


class YandexDiskDownloader:
    # Source: https://github.com/SecFathy/YandexDown/blob/main/YandexCLI.py
    def __init__(self, link, download_location, batch):
        self.link = link
        self.download_location = download_location
        self.batch = batch

    def _download_file(self, path, save_dir, unpack=True):
        url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={self.link}&path={path}"
        response = requests.get(url)
        download_url = response.json()["href"]
        file_name = urllib.parse.unquote(download_url.split("filename=")[1].split("&")[0])

        save_dir.mkdir(exist_ok=True, parents=True)
        save_path_zip = save_dir / file_name

        with open(save_path_zip, "wb") as file:
            download_response = requests.get(download_url, stream=True)
            for chunk in download_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    file.flush()
            logger.info(f"Saved to {save_path_zip}")

        if unpack:
            save_path = save_dir / file_name.split(".")[0]
            with zipfile.ZipFile(save_path_zip, mode="r") as archieve:
                archieve.extractall(save_path)
            logger.info(f"Zip is unpacked to {save_path}")

            os.remove(save_path_zip)


    def download(self):
        images_url = f"https://cloud-api.yandex.net/v1/disk/public/resources?public_key={self.link}&path=/images/&limit=1000"
        images_response = requests.get(images_url)
        all_images = images_response.json()["_embedded"]["items"]

        logger.info(f"The total number of images is {len(all_images)}")
        logger.info(f"The chosen batch is {self.batch}")
        for i, file in enumerate(all_images):
            if  self.batch[0] <= i and i < self.batch[1]:
                logger.info(f"File {i + 1}/{len(all_images)}: {file['name']}")
                self._download_file(path=file['path'], save_dir=self.download_location / "images")

        logger.info(f"Loading metadata")
        self._download_file(path="/biospecimen.cart.2024-01-18.json", save_dir=self.download_location, unpack=False)
        self._download_file(path="/metadata.cart.2024-01-18.json", save_dir=self.download_location, unpack=False)

        logger.info("Download complete.")


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    save_dir = pathlib.Path(cfg.data.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading data")
    downloader = YandexDiskDownloader(cfg.data.link, save_dir, cfg.data.batch)
    downloader.download()


if __name__ == "__main__":
    main()
