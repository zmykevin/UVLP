#import mmf.utils.download as download
import collections
import datetime
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import requests
import tqdm
from mmf.utils.file_io import PathManager

GOOGLE_DRIVE_SUBSTR = "drive.google"
MMF_PREFIX = "mmf://"
MMF_PREFIX_REPLACEMENT = "https://dl.fbaipublicfiles.com/mmf/data/"

def make_dir(path):
    """
    Make the directory and any nonexistent parent directories (`mkdir -p`).
    """
    # the current working directory is a fine path
    if path != "":
        PathManager.mkdirs(path)


def move(path1, path2):
    """
    Rename the given file.
    """
    shutil.move(path1, path2)


def copy(path1, path2):
    """
    Copy the given file from path1 to path2.
    """
    shutil.copy(path1, path2)


def remove_dir(path):
    """
    Remove the given directory, if it exists.
    """
    shutil.rmtree(path, ignore_errors=True)


def decompress(path, fname, delete_original=True):
    """
    Unpack the given archive file to the same directory.

    Args:
        path(str): The folder containing the archive. Will contain the contents.
        fname (str): The filename of the archive file.
        delete_original (bool, optional): If true, the archive will be deleted
                                          after extraction. Default to True.
    """
    print("Unpacking " + fname)
    fullpath = os.path.join(path, fname)
    shutil.unpack_archive(fullpath, path)
    if delete_original:
        os.remove(fullpath)

def check_header(url, from_google=False):
    """
    Performs a HEAD request to check if the URL / Google Drive ID is live.
    """
    session = requests.Session()
    if from_google:
        URL = "https://docs.google.com/uc?export=download"
        response = session.head(URL, params={"id": url}, stream=True)
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) "
            + "AppleWebKit/537.36 (KHTML, like Gecko) "
            + "Chrome/77.0.3865.90 Safari/537.36"
        }
        response = session.head(url, allow_redirects=True, headers=headers)
    status = response.status_code
    session.close()

    assert status == 200, (
        "The url {} is broken. If this is not your own url,"
        + " please open up an issue on GitHub"
    ).format(url)

def download(url, path, fname, redownload=True, disable_tqdm=False):
    """
    Download file using `requests`.

    If ``redownload`` is set to false, then will not download tar file again if it is
    present (default ``True``).

    Returns whether download actually happened or not
    """
    outfile = os.path.join(path, fname)
    download = not PathManager.isfile(outfile) or redownload
    retry = 10
    exp_backoff = [2 ** r for r in reversed(range(retry))]

    pbar = None
    if download:
        # First test if the link is actually downloadable
        check_header(url)
        if not disable_tqdm:
            print("[ Downloading: " + url + " to " + outfile + " ]")
        pbar = tqdm.tqdm(
            unit="B", unit_scale=True, desc=f"Downloading {fname}", disable=disable_tqdm
        )

    while download and retry >= 0:
        resume_file = outfile + ".part"
        resume = PathManager.isfile(resume_file)
        if resume:
            resume_pos = os.path.getsize(resume_file)
            mode = "ab"
        else:
            resume_pos = 0
            mode = "wb"
        response = None

        with requests.Session() as session:
            try:
                header = (
                    {"Range": "bytes=%d-" % resume_pos, "Accept-Encoding": "identity"}
                    if resume
                    else {}
                )
                response = session.get(url, stream=True, timeout=5, headers=header)

                # negative reply could be 'none' or just missing
                if resume and response.headers.get("Accept-Ranges", "none") == "none":
                    resume_pos = 0
                    mode = "wb"

                CHUNK_SIZE = 32768
                total_size = int(response.headers.get("Content-Length", -1))
                # server returns remaining size if resuming, so adjust total
                total_size += resume_pos
                pbar.total = total_size
                done = resume_pos

                with PathManager.open(resume_file, mode) as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                retry -= 1
                pbar.clear()
                if retry >= 0:
                    print("Connection error, retrying. (%d retries left)" % retry)
                    time.sleep(exp_backoff[retry])
                else:
                    print("Retried too many times, stopped retrying.")
            finally:
                if response:
                    response.close()
    if retry < 0:
        raise RuntimeWarning("Connection broken too many times. Stopped retrying.")

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeWarning(
                "Received less data than specified in "
                + "Content-Length header for "
                + url
                + ". There may be a download problem."
            )
        move(resume_file, outfile)

    if pbar:
        pbar.close()

    return download


def parse_url(url):
    if url.find(MMF_PREFIX) == -1:
        return url
    else:
        return MMF_PREFIX_REPLACEMENT + url[len(MMF_PREFIX) :]

# resources = "- url: mmf://datasets/coco/defaults/features/coco_train2017.tar.gz \
#                file_name: coco_train2017.tar.gz \
#                hashcode: 7815fa155f3ab438bbb753bd0ae746add35bafedfbd3db582141c2b99b817ddc"

raw_url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
#raw_url = "mmf://datasets/flickr30k/defaults/features/features.tar.gz"
url = parse_url(raw_url)
file_name = "annotations.zip"
#hashcode = "679af7902f342480c1e039bb0be3ddabb8b88a4c45a2c17f9b97e4f10689475a"
target_path = "/home/zmykevin/fb_intern/data/mmf_data/datasets/flickr30k/annotations"




download(url, target_path, file_name)
print("Download accomplished")