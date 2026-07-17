import os
from pathlib import Path

from PIL import ImageFile
from torchvision.datasets import ImageFolder

from .transforms import build_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    "PACS",
    "VLCS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
]


def get_dataset_class(dataset_name):
    if dataset_name not in DATASETS:
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


def num_domains(dataset_name):
    return len(get_dataset_class(dataset_name).DOMAINS)


def resolve_domain_indices(domains, selected_domains=None):
    selected_domains = selected_domains or []
    indices = set()

    for item in selected_domains:
        if isinstance(item, int):
            if item < 0 or item >= len(domains):
                raise ValueError(f"Domain index out of range: {item}")
            indices.add(item)
        elif isinstance(item, str):
            if item not in domains:
                choices = ",".join(domains)
                raise ValueError(f"Unknown domain: {item}. Choices: {choices}")
            indices.add(domains.index(item))
        else:
            raise TypeError(f"Domain must be an int index or str name, got {type(item).__name__}")

    return indices


class MultipleDomainDataset:
    N_STEPS = 5001
    CHECKPOINT_FREQ = 100
    N_WORKERS = 4
    DOMAINS = None
    INPUT_SHAPE = (3, 224, 224)

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
    

class MultipleDomainImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_domains=None, augment=False, image_size=224, normalization="clip"):
        super().__init__()

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {root}")

        self.domains = list(self.DOMAINS)
        test_domain_indices = resolve_domain_indices(self.domains, test_domains)

        self.datasets = []

        for domain_index, domain in enumerate(self.domains):
            domain_path = root / domain
            if not domain_path.is_dir():
                raise FileNotFoundError(f"Domain directory not found: {domain_path}")

            is_test_domain = domain_index in test_domain_indices

            transform = build_transform(
                train=not is_test_domain,
                augment=augment and not is_test_domain,
                image_size=image_size,
                normalization=normalization,
            )

            self.datasets.append(ImageFolder(domain_path, transform=transform))

        self.input_shape = (3, image_size, image_size)
        self.num_classes = len(self.datasets[-1].classes)


class PACS(MultipleDomainImageFolder):
    CHECKPOINT_FREQ = 300
    DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]

    def __init__(self, root, **kwargs):
        super().__init__(Path(root) / "PACS", **kwargs)


class VLCS(MultipleDomainImageFolder):
    CHECKPOINT_FREQ = 300
    DOMAINS = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

    def __init__(self, root, **kwargs):
        super().__init__(Path(root) / "VLCS", **kwargs)


class OfficeHome(MultipleDomainImageFolder):
    CHECKPOINT_FREQ = 300
    DOMAINS = ["Art", "Clipart", "Product", "Real World"]

    def __init__(self, root, **kwargs):
        super().__init__(Path(root) / "office_home", **kwargs)


class TerraIncognita(MultipleDomainImageFolder):
    CHECKPOINT_FREQ = 300
    DOMAINS = ["location_100", "location_38", "location_43", "location_46"]

    def __init__(self, root, **kwargs):
        super().__init__(Path(root) / "terra_incognita", **kwargs)


class DomainNet(MultipleDomainImageFolder):
    N_STEPS = 15001
    CHECKPOINT_FREQ = 1000
    DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, root, **kwargs):
        super().__init__(Path(root) / "domain_net", **kwargs)
