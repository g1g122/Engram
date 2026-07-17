import os
import json
import shutil
import tarfile
import argparse
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict

import gdown

DATASETS = {
    "PACS": "download_pacs",
    "VLCS": "download_vlcs",
    "OfficeHome": "download_office_home",
    "TerraIncognita": "download_terra_incognita",
    "DomainNet": "download_domain_net",
}


def has_files(path):
    path = Path(path)
    return path.exists() and any(path.iterdir())


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[skip] archive exists: {dst}")
        return dst
    
    print(f"[download] {url}")
    print(f"[to] {dst}")

    if "drive.google.com" in url:
        gdown.download(url, str(dst), quiet=False)
    else:
        urllib.request.urlretrieve(url, str(dst))

    return dst


def extract_archive(path, dst_dir=None, remove=True):
    path = Path(path)
    dst_dir = Path(dst_dir) if dst_dir is not None else path.parent

    print(f"[extract] {path} -> {dst_dir}")

    if path.suffix == ".zip":
        with ZipFile(path, "r") as zf:
            zf.extractall(dst_dir)
    elif path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(dst_dir)
    elif path.suffix == ".tar":
        with tarfile.open(path, "r:") as tar:
            tar.extractall(dst_dir)
    else:
        raise ValueError(f"Unsupported archive format: {path}")

    if remove:
        path.unlink()

    return dst_dir


def download_and_extract(url, dst, remove=True):
    archive = download_file(url, dst)
    extract_archive(archive, archive.parent, remove=remove)


def safe_rename(src, dst):
    src = Path(src)
    dst = Path(dst)

    if dst.exists():
        if any(dst.iterdir()):
            print(f"[skip] target exists: {dst}")
            return
        dst.rmdir()

    if not src.exists():
        raise FileNotFoundError(f"Expected extracted path does not exist: {src}")
    
    print(f"[rename] {src} -> {dst}")
    src.rename(dst)


def download_pacs(data_dir):
    data_dir = Path(data_dir)
    full_path = data_dir / "PACS"

    if has_files(full_path):
        print(f"[skip] PACS appears to exist at {full_path}")
        return
    
    official_url = (
        "https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?"
        "resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ&usp=sharing"
    )

    print(
        "[note] The DomainBed PACS Google Drive link seems to be unavailable. "
        f"If the download fails, please open {official_url}, going to the 'Raw images' "
        f"folder, download PACS.zip, place it under {full_path} and rerun this script. "
        "The script will automatically skip the download and directly extract the archive."
    )

    url = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"
    download_and_extract(url, data_dir / "PACS.zip")

    safe_rename(data_dir / "kfold", full_path)


def download_vlcs(data_dir):
    data_dir = Path(data_dir)
    full_path = data_dir / "VLCS"

    if has_files(full_path):
        print(f"[skip] VLCS appears to exist at {full_path}")
        return
    
    print(
        "[note] The original DomainBed Google Drive link for the ~4GB "
        "full-resolution VLCS archive appears to be unavailable. This script "
        "provides a fallback path that can convert the official compressed VLCS "
        "release to the required ImageFolder layout. The image count should match, "
        "but the archive size and image resolution differ. If you need the "
        "full-resolution VLCS dataset, please find a working preprocessed archive "
        "or rebuild it from the original source datasets."
    )

    url = ("https://download1514.mediafire.com/a1nmdvb7o8tgb1WF7ZUqAHtQyiYZVFiVifuYP2dBbNU2r1-dGhy7J45_"
           "jLsoihTtSLQd_pId_p7pGV6VSpAwqgGPp6_q5YNk4tnCC24nqPdUhFekQuk5pWWAupHpY4Lr39jxmUHk0VVCOTW805a_"
           "vwZQY1bXYv5yORw5iLXtSJNvoA/7yv132lgn1v267r/vlcs.tar.gz")
    download_and_extract(url, data_dir / "VLCS.tar.gz")

    build_vlcs_imagefolder(full_path)


def build_vlcs_imagefolder(vlcs_dir):
    vlcs_dir = Path(vlcs_dir)

    domain_map = {
        "CALTECH": "Caltech101",
        "LABELME": "LabelMe",
        "PASCAL": "VOC2007",
        "SUN": "SUN09",
    }

    class_map = {
        "0": "bird",
        "1": "car",
        "2": "chair",
        "3": "dog",
        "4": "person",
    }

    converted_dir = vlcs_dir.with_name(vlcs_dir.name + "_imagefolder")

    if converted_dir.exists():
        shutil.rmtree(converted_dir)

    copied = 0
    for src_domain, dst_domain in domain_map.items():
        for split in ["train", "test", "crossval"]:
            split_dir = vlcs_dir / src_domain / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Expected VLCS split does not exist: {split_dir}")
            
            for src_class, dst_class in class_map.items():
                class_dir = split_dir / src_class
                if not class_dir.exists():
                    raise FileNotFoundError(f"Expected VLCS class does not exist: {class_dir}")
                
                dst_dir = converted_dir / dst_domain / dst_class
                dst_dir.mkdir(parents=True, exist_ok=True)

                for src in class_dir.iterdir():
                    if not src.is_file():
                        continue

                    dst = dst_dir / f"{src_domain}_{src_class}_{src.name}"
                    shutil.copyfile(src, dst)
                    copied += 1

    original_dir = vlcs_dir.with_name(vlcs_dir.name + "_official")
    if original_dir.exists():
        shutil.rmtree(original_dir)
    
    vlcs_dir.rename(original_dir)
    converted_dir.rename(vlcs_dir)

    print(f"[done] converted VLCS to ImageFolder layout with {copied} images")
    print(f"[keep] original VLCS layout saved at {original_dir}")


def download_office_home(data_dir):
    data_dir = Path(data_dir)
    full_path = data_dir / "office_home"

    if has_files(full_path):
        print(f"[skip] OfficeHome appears to exist at {full_path}")
        return
    
    official_url = (
        "https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?"
        "usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw"
    )
    
    print(
        "[note] The DomainBed Office-Home Google Drive link seems to be unavailable. "
        f"If the download fails, please download Office-Home manually from {official_url}, "
        f"place OfficeHomeDataset_10072016.zip under {data_dir}, and rerun this script. "
        "The script will automatically skip the download and extract the archive when it already exists."
    )

    url = "https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC"

    archive = data_dir / "office_home.zip"
    official_archive = data_dir / "OfficeHomeDataset_10072016.zip"

    if official_archive.exists():
        archive = official_archive

    download_and_extract(url, archive)

    safe_rename(data_dir / "OfficeHomeDataset_10072016", full_path)


def download_domain_net(data_dir):
    data_dir = Path(data_dir)
    full_path = ensure_dir(data_dir / "domain_net")

    urls = [
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
    ]

    for url in urls:
        archive_name = url.split("/")[-1]
        domain_name = archive_name.replace(".zip", "")
        expected_dir = full_path / domain_name

        if has_files(expected_dir):
            print(f"[skip] DomainNet domain exists: {expected_dir}")
            continue

        download_and_extract(url, full_path / archive_name)

    remove_domain_net_duplicates(full_path)


def remove_domain_net_duplicates(domain_net_dir):
    duplicate_file = duplicate_file = Path(__file__).with_name("domain_net_duplicates.txt")

    if not duplicate_file.exists():
        print("[warn] {duplicate_file} not found; skip duplicate removal")
        return

    removed = 0
    with duplicate_file.open("r", encoding="utf-8") as f:
        for line in f:
            rel_path = line.strip()
            if not rel_path:
                continue

            path = Path(domain_net_dir) / rel_path
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass

    print(f"[done] removed {removed} DomainNet duplicate files")


def download_terra_incognita(data_dir):
    data_dir = Path(data_dir)
    full_path = ensure_dir(data_dir / "terra_incognita")

    expected_locations = [
        full_path / "location_38",
        full_path / "location_46",
        full_path / "location_100",
        full_path / "location_43",
    ]

    if all(has_files(path) for path in expected_locations):
        print(f"[skip] TerraIncognita appears to exist at {full_path}")
        return

    images_url = (
        "https://storage.googleapis.com/public-datasets-lila/"
        "caltechcameratraps/eccv_18_all_images_sm.tar.gz"
    )
    annotations_url = (
        "https://storage.googleapis.com/public-datasets-lila/"
        "caltechcameratraps/eccv_18_annotations.tar.gz"
    )

    download_and_extract(images_url, full_path / "terra_incognita_images.tar.gz")
    download_and_extract(annotations_url, full_path / "eccv_18_annotations.tar.gz")

    build_terra_incognita_imagefolder(full_path)


def build_terra_incognita_imagefolder(full_path):
    full_path = Path(full_path)

    include_locations = {"38", "46", "100", "43"}
    include_categories = {
        "bird",
        "bobcat",
        "cat",
        "coyote",
        "dog",
        "empty",
        "opossum",
        "rabbit",
        "raccoon",
        "squirrel",
    }

    images_folder = full_path / "eccv_18_all_images_sm"
    annotations_folder = full_path / "eccv_18_annotation_files"

    annotation_files = [
        annotations_folder / "cis_test_annotations.json",
        annotations_folder / "cis_val_annotations.json",
        annotations_folder / "train_annotations.json",
        annotations_folder / "trans_test_annotations.json",
        annotations_folder / "trans_val_annotations.json",
    ]

    data = defaultdict(list)
    for annotation_file in annotation_files:
        print(f"[read] {annotation_file}")
        with annotation_file.open("r", encoding="utf-8") as f:
            annots = json.load(f)
        for key, value in annots.items():
            data[key].extend(value)

    category_by_id = {item["id"]: item["name"] for item in data["categories"]}

    annotations_by_image_id = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image_id[annotation["image_id"]].append(annotation)

    copied = 0
    for image in data["images"]:
        location = str(image["location"])
        if location not in include_locations:
            continue

        image_id = image["id"]
        image_name = image["file_name"]
        image_annotations = annotations_by_image_id.get(image_id, [])

        for annotation in image_annotations:
            category = category_by_id[annotation["category_id"]]
            if category not in include_categories:
                continue

            dst_dir = full_path / f"location_{location}" / category
            dst_dir.mkdir(parents=True, exist_ok=True)

            src = images_folder / image_name
            dst = dst_dir / image_name

            if not dst.exists():
                shutil.copyfile(src, dst)
                copied += 1

    print(f"[done] copied {copied} TerraIncognita images into ImageFolder layout")

    if images_folder.exists():
        shutil.rmtree(images_folder)
    if annotations_folder.exists():
        shutil.rmtree(annotations_folder)


def download_dataset(name, data_dir):
    if name not in DATASETS:
        choices = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset: {name}. Choices: {choices}")

    fn = globals()[DATASETS[name]]
    fn(data_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Download DomainBed-style domain generalization datasets")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="Directory to store datasets. Default: ./datasets",)
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(DATASETS),
                        help="Datasets to download",)  # PACS, VLCS, OfficeHome, TerraIncognita, DomainNet
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = ensure_dir(args.data_dir)

    for name in args.datasets:
        print(f"\n=== {name} ===")
        download_dataset(name, data_dir)

    print("\n[done] all requested datasets processed")


if __name__ == "__main__":
    main()
