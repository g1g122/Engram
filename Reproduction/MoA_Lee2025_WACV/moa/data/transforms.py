from torchvision import transforms

NORMALIZATION_STATS = {
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}


def build_transform(train=False, augment=False, image_size=224, normalization="clip"):
    if normalization not in NORMALIZATION_STATS:
        choices = ", ".join(sorted(NORMALIZATION_STATS))
        raise ValueError(f"Unknown normalization: {normalization}. Choices: {choices}")

    stats = NORMALIZATION_STATS[normalization]

    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(stats["mean"], stats["std"]),
    ])
