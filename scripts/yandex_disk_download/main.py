import os
import pathlib
import shutil
import urllib.parse
import zipfile

import hydra
import requests
from loguru import logger
from omegaconf import DictConfig


class YandexDiskDownloader:
    # Source: https://github.com/SecFathy/YandexDown/blob/main/YandexCLI.py
    def __init__(self, link, download_location):
        self.link = link
        self.download_location = download_location

    def unzip(self, save_path):
        # Source:
        # https://stackoverflow.com/questions/61928119/python-how-to-extract-files-without-including-parent-directory
        with zipfile.ZipFile(save_path, mode="r") as archieve:
            for file_info in archieve.infolist():
                # Only extract regular files
                if file_info.is_dir():
                    continue

                file_path = file_info.filename

                # Split at slashes, at most one time, and take the second part
                # so that we skip the root directory part
                extracted_path = file_path.split("/", 1)[1]

                # Combine with the destination directory
                extracted_path = os.path.join(self.download_location, extracted_path)

                # Make sure the directory for the file exists
                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)

                # Extract the file. Don't use `z.extract` as it will concatenate
                # the full path from inside the zip.
                # WARNING: This code does not check for path traversal vulnerabilities
                # Refer to the big warning inside the ZipFile module for more details
                with open(extracted_path, "wb") as dst:
                    with archieve.open(file_info, "r") as src:
                        shutil.copyfileobj(src, dst)

    def download(self):
        url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={self.link}"
        response = requests.get(url)
        download_url = response.json()["href"]
        file_name = urllib.parse.unquote(download_url.split("filename=")[1].split("&")[0])
        save_path = os.path.join(self.download_location, file_name)

        logger.info(save_path)
        with open(save_path, "wb") as file:
            download_response = requests.get(download_url, stream=True)
            for chunk in download_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    file.flush()

        # self.unzip(save_path)
        # os.remove(save_path)
        logger.info("Download complete.")


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    save_dir = pathlib.Path(cfg.data.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading data")

    downloader = YandexDiskDownloader(cfg.data.link, save_dir)
    downloader.download()


if __name__ == "__main__":
    main()
