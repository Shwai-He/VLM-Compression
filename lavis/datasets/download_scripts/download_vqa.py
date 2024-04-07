"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import time
import logging
import requests
import numpy as np
from pathlib import Path

from omegaconf import OmegaConf
import sys

sys.path.append('/mnt/petrelfs/dongdaize.d/workspace/sh/ECoFLaP/LAVIS')
from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
    get_abs_path,
    get_cache_path,
)

header_mzl = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Cookie": "__Secure-ENID=12.SE=eSJGY4uSfDwvY0qlxBdF4mkIPqSCAZwFFB1QgzleBYGJischjd6gJEwPx6uoBOxVl7ayk-aFuZxVuAbHFTzgMn4wqUPBtj_auvzpyKg-C16UYIuFYslsENXNyS6c7Yu42GnVqqgxt50gxevbm9A1umadRzMXdoVX-mBr0K4fgMSSbG--Pr3H6fxvYeZBFep0fKnRnAkW3U_Ra5_EjxE6hWzIR2ZAkdn1GUU; HSID=Ao5U5-pY-8RjtwbWd; SSID=ALN2hDL5yxCsqtXgq; APISID=BlWW67_CXVy98SCk/A_Az9-vrMUUcMBFOD; SAPISID=CgIwOTBD_4V4WvWi/Ae47ppKU3jIdu46bs; __Secure-1PAPISID=CgIwOTBD_4V4WvWi/Ae47ppKU3jIdu46bs; __Secure-3PAPISID=CgIwOTBD_4V4WvWi/Ae47ppKU3jIdu46bs; SEARCH_SAMESITE=CgQIypkB; SID=dQiiv34cFWOcoKCrR4NTwfNe_rCfaPUqxkizIKmPzRGIGWguyTP80Ah5cZUy3hiU01PapQ.; __Secure-1PSID=dQiiv34cFWOcoKCrR4NTwfNe_rCfaPUqxkizIKmPzRGIGWguxySLAzAxGW2MDV9qpdQ3Qg.; __Secure-3PSID=dQiiv34cFWOcoKCrR4NTwfNe_rCfaPUqxkizIKmPzRGIGWguRFqkTHemgjJbhhdyamLErA.; 1P_JAR=2023-12-22-11; AEC=Ackid1R6QuaPyLbY3-dARRwHryAZZKkwf4NYr6kPbrJkbuHAR8krNfNdaOs; __Secure-1PSIDTS=sidts-CjEBPVxjSm_ksuw2e4bt-F8kuwhw1M0JFhhV9mew4v0M3bn8gHg2iHMiWbli7w80zVduEAA; __Secure-3PSIDTS=sidts-CjEBPVxjSm_ksuw2e4bt-F8kuwhw1M0JFhhV9mew4v0M3bn8gHg2iHMiWbli7w80zVduEAA; NID=511=hbedNQOcy9Wd6wrV5wM31r5S4EXeGETiIclMe-Q33PvLrM0r8wrUGDsGGgXtNZZ8e-9NocVx71ln6JEoVl4HB5kAjsfGIMmk-hCS_E8xtpc2Cx7HsILEnDJi6A_m8tKfiif7HmVd0d01J75B-RvI7XceK9GsZEQORhrLLf48srbUUgl5nDyC43UGPWCfhg0RWZwpfCwOp6yRvuzJeAw7DoThc5PyToiGki7mQBVmhhWixLQYpcfMRAftX7djJTNJ5gqx1m-6XGZKtIjtZFTuDTv2tbMmW-Lq4wwNiNtkfPNFgHilDig8I3ga4L1gVuZ4FuCfI5z0H8XZHxzlD6LGvJqGHEHhhLBlVyELT3OU35uvKvST018wYuByOexlsoDXkMabP5FcVQDLwyX8G0zXvhshdE1wdX2whvt4Mh7XK46YW14Z_W4McPDEans2gC2gJL9WCyd1M7h1zJDHZAuh70uq-WHZMNAXUKV38wZIMcX-9GZn4AJoYmr9c-d7xtgoD1tL7ScST9g8B1XVHU1IZwwmmLeWnSWmXgprq1Fn1AlG-C2ot6wnhCcgEMwti-c1T5yvLkeTNz1Ov54g_VKVuS5VUuk3SMjkPsWBdq-eSRmGmkSx1RjUCwRl9kJwUN8xjQ; SIDCC=ABTWhQF-mewH7ysCt4DlLzqYNzo4hmGuXxz3kjQnCtpEpzTc_XDeeUsKfwFoyFiCMWxOmOSHB-JK; __Secure-1PSIDCC=ABTWhQG64BsSr35O8euVAZOnuyT2QmJrdIUgGzt0_TYDOWYpAi9DXpHBRy1AdY90ln-92MM0te4; __Secure-3PSIDCC=ABTWhQGlhRh-tT26Axgh0MkGxmn6MqY-X7VZkZR9KFf0Y1_vHB2x2kfciju0YhYfqz490OrJeuaa", 
    # "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    # "X-Forwarded-For": "64.18.15.200",
}

header_gbot = {
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
}

# headers = [header_mzl, header_gbot]
headers = [header_mzl]

def download_file(url, filename):
    max_retries = 20
    cur_retries = 0

    header = headers[0]

    while cur_retries < max_retries:
        try:
            r = requests.get(url, headers=header, timeout=10)
            with open(filename, "wb") as f:
                f.write(r.content)

            break
        except Exception as e:
            logging.info(" ".join(repr(e).splitlines()))
            logging.error(url)
            cur_retries += 1

            # random sample a header from headers
            header = headers[np.random.randint(0, len(headers))]

    time.sleep(3 + cur_retries * 2)


DATA_URL = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",  # md5: 0da8c0bd3d6becc4dcb32757491aca88
    "val": "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
    "test": "http://images.cocodataset.org/zips/test2014.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
    "test2015": "http://images.cocodataset.org/zips/test2015.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
}


# def download_datasets(root, url):
#     download_and_extract_archive(url=url, download_root=root, extract_root=storage_dir)


if __name__ == "__main__":

    config_path = get_abs_path("configs/datasets/coco/defaults_vqa.yaml")

    config = OmegaConf.load(
        config_path
    )
    # storage_dir = OmegaConf.load(
    #     config_path
    # ).datasets.coco_vqa.build_info.annotations.storage

    print(config)
    print(type(config))

    for k, v in config.datasets.coco_vqa.build_info.annotations.items():

        # print(k, v)
        for url, storage_path in zip(v.url, v.storage):
            storage_path = Path(get_cache_path(storage_path))

            # if storage_path.exists():
            #     print(f"File already exists at {storage_path}. Aborting.")
            #     exit(0)

            try:
                print("Downloading {} to {}".format(url, storage_path))
                download_file(url, storage_path)
            except Exception as e:
                # remove download dir if failed
                print("Failed to download file. Aborting.")
