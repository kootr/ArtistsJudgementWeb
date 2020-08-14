import os
import argparse
import sys

from google_images_download import google_images_download

# -s: Google Imagesにかける検索キーワード (デフォルト "tokyo")
# -n: ダウンロードする画像の数


def main(search, num_images):
    save_directory = "./images/" + search
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    config = {
        "Records": [
            {
                "keywords": search,
                "no_numbering": False,
                "limit": 100,
                "output_directory": "./images",
                "image_directory": search,
                "chromedriver": "/usr/local/bin/chromedriver",
            }
        ]
    }

    response = google_images_download.googleimagesdownload()
    for rc in config["Records"]:
        response.download(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options for scraping images")
    parser.add_argument("-s", "--search", default="tokyo", type=str, help="search term")
    parser.add_argument(
        "-n", "--num_images", default=10, type=int, help="num of images to scrap"
    )
    args = parser.parse_args()
    try:
        main(search=args.search, num_images=args.num_images)
    except KeyboardInterrupt:
        pass
    sys.exit()
