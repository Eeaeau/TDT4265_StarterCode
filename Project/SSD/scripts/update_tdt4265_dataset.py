import pathlib
import requests
import zipfile
import tqdm


zip_url = "https://folk.ntnu.no/haakohu/tdt4265_2022_dataset.zip"
dataset_path = pathlib.Path("data", "tdt4265_2022")


def download_image_zip(zip_path):
    response = requests.get(zip_url, stream=True)
    total_length = int(response.headers.get("content-length"))
    assert response.status_code == 200, \
        f"Did not download the images. Contact the TA. Status code: {response.status_code}"
    zip_path.parent.mkdir(exist_ok=True, parents=True)
    with open(zip_path, "wb") as fp:
        for data in tqdm.tqdm(
                response.iter_content(chunk_size=4096), total=total_length/4096,
                desc="Downloading images."):
            fp.write(data)


def download_dataset():
    print("Extracting images.")
    work_dir = pathlib.Path("/work", "datasets", "tdt4265_2022")
    if work_dir.is_dir():
        print("You are working on a computer with the dataset under work_dataset:", work_dir)
        print("Doing nothing.")
        return
    zip_path = pathlib.Path("datasets", "tdt4265", "dataset.zip")
    if not zip_path.is_file():
        print(f"Download the zip file and place it in the path: {zip_path.absolute()}")
        download_image_zip(zip_path)
    with zipfile.ZipFile(zip_path, "r") as fp:
        fp.extractall(dataset_path)


if __name__ == "__main__":
    # Download labels
    dataset_path.mkdir(exist_ok=True, parents=True)
    download_dataset()
    print("Dataset extracted to:", dataset_path)
