import os
import ast
import h5py
import numpy as np
import pandas as pd
import tifffile as tif
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


def transform_to_list(val):
    return ast.literal_eval(val)


class Shapes3D(torch.utils.data.Dataset):
    def __init__(self, directory, test=False):
        super().__init__()
        if not test:
            dataset_path = directory + "/3dshapes_train_ood.h5"
        else:
            dataset_path = directory + "/3dshapes_test_ood.h5"
        print(dataset_path)
        dataset = h5py.File(dataset_path, "r")
        self.images = dataset["images"]
        self.labels = dataset["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.images[idx].transpose((2, 0, 1)) / 255.0
        label = self.labels[idx]

        # if we remove shape conditioning
        #         label = np.delete(label, 4)

        sample = torch.tensor(sample)
        label = torch.tensor(label)
        return sample, label


class SingleCellDataset(torch.utils.data.Dataset):
    img_channels = 5

    mean = (14.61494155, 28.87414807, 32.61800756, 36.41028929, 22.27769592)
    std = (28.80363038, 31.39763568, 32.06969697, 32.35857192, 25.8217434)
    max_pixel_value = 255.0

    num_experiments = 5
    num_plates = 136
    num_wells = 80

    normalize_per_plate = False
    split_number = 1
    ratios = (0.2, 0.3)

    def __init__(self, dataset_params, df_path, transform, mode="train"):
        self.root_dir = dataset_params.root_dir
        self.metadata_dir = dataset_params.metadata_dir
        self.img_size = 224
        self.mode = mode
        self.get_stats()
        self.data = self.get_data_as_list(os.path.join(self.metadata_dir, df_path), mode=mode)
        self.profile_embeddings_dir = os.path.join(self.metadata_dir, "profile_embeddings")
        self.style_embeddings_dir = os.path.join(self.metadata_dir, "style_embeddings")
        self.compound_embeddings_dir = os.path.join(self.metadata_dir, "compound_embeddings")
        self.transform = transform

    def attr_from_dict(self, param_dict):
        self.name = self.__class__.__name__
        for key in param_dict:
            setattr(self, key, param_dict[key])

    def get_data_as_list(self, path, dmso_flag=False, mode="train"):
        data_list = []
        datainfo = pd.read_csv(path, index_col=0, engine="python")

        datainfo["crops"] = datainfo["new_crops"].apply(transform_to_list)

        experimentlist = datainfo.replicate_ID.values
        platelist = datainfo.Metadata_Plate_ID.values
        welllist = datainfo.Metadata_Well_ID.values
        platenames = datainfo.Metadata_Plate.values
        wellnames = datainfo.Metadata_Well.values
        sitenames = datainfo.Metadata_Site.values
        img_names = datainfo.combined_paths.tolist()
        crops = datainfo.crops.tolist()
        compounds = datainfo.Metadata_broad_sample.values

        dataframe = pd.DataFrame(
            list(zip(img_names, experimentlist, platelist, welllist, crops, platenames, wellnames, sitenames, compounds)),
            columns=["img_path", "experiment", "plate", "well", "crops", "platenames", "wellnames", "sitenames", "compounds"],
        )
        data = dataframe

        img_paths = data["img_path"].values.tolist()
        experiments = data["experiment"].values.tolist()
        plates = data["plate"].values.tolist()
        wells = data["well"].values.tolist()
        crops = data["crops"].values.tolist()
        platenames = data["platenames"].values.tolist()
        wellnames = data["wellnames"].values.tolist()
        sitenames = data["sitenames"].values.tolist()
        compounds = data["compounds"].values.tolist()

        data_list = [
            {
                "img_path": img_path,
                "experiment": experiment,
                "plate": plate,
                "well": well,
                "crops": crop,
                "platename": platename,
                "wellname": wellname,
                "sitename": sitename,
                "compound": compound
            }
            for img_path, experiment, plate, well, crop, platename, wellname, sitename, compound in zip(
                img_paths, experiments, plates, wells, crops, platenames, wellnames, sitenames, compounds
                )
        ]

        self.df = pd.DataFrame(data_list)
        return data_list

    def get_stats(self):
        path = "/raid/cian/user/sergei.pnev/data/JUMP/cpg0004-lincs/metadata/top20_moa_plate_norms_dmso.csv"
        datainfo = pd.read_csv(path, index_col=0, engine="python")

        mean = [1.0, 1.0, 1.0, 1.0, 1.0]
        std = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean = np.mean(mean, axis=0) * 0.0
        self.std = np.mean(std, axis=0)

        self.norm_df = datainfo
        self.norm_df["plate"] = [i for i in range(len(self.norm_df))]
        self.mean_columns = ["DNA_mean", "ER_mean", "RNA_mean", "AGP_mean", "Mito_mean"]
        self.std_columns = ["DNA_std", "ER_std", "RNA_std", "AGP_std", "Mito_std"]

    def plate_normalize_image(self, x, p):
        x = x.float()
        if self.normalize_per_plate:
            plate_stats = self.norm_df[self.norm_df.Metadata_Plate == p]

            mean_vals = (plate_stats[self.mean_columns].values) * (1 / 255.0)
            std_vals = (plate_stats[self.std_columns].values) * (1 / 255.0)

            norm_values_mean = (
                torch.tensor(np.array([mean_vals]))
                .view(5, 1, 1)
                .type(torch.FloatTensor)
            )
            norm_values_std = (
                torch.tensor(np.array([std_vals])).view(5, 1, 1).type(torch.FloatTensor)
            )

            x = (x - norm_values_mean) / (norm_values_std)
        else:

            mean_vals = np.array(
                [14.61494155, 28.87414807, 32.61800756, 36.41028929, 22.27769592]
            ) * (1 / 255.0)
            std_vals = np.array(
                [28.80363038, 31.39763568, 32.06969697, 32.35857192, 25.8217434]
            ) * (1 / 255.0)

            norm_values_mean = (
                torch.tensor(np.array([mean_vals]))
                .view(5, 1, 1)
                .type(torch.FloatTensor)
            )
            norm_values_std = (
                torch.tensor(np.array([std_vals])).view(5, 1, 1).type(torch.FloatTensor)
            )

            x = (x - norm_values_mean) / (norm_values_std)

        return x

    def __getitem__(self, idx):
        crop = self.data[idx]["crops"]
        experiment = torch.as_tensor(self.data[idx]["experiment"])
        plate = torch.as_tensor(self.data[idx]["plate"])
        well = torch.as_tensor(self.data[idx]["well"])
        platename = self.data[idx]["platename"]
        wellname = self.data[idx]["wellname"]
        sitename = self.data[idx]["sitename"]
        compound = self.data[idx]["compound"]

        crop_idx = np.random.choice(len(crop), size=1)[0]
        x, y = crop[crop_idx]
        image_name = f"{platename}/{wellname}-{sitename}"
        image_path = f"{self.root_dir}/{image_name}-{x}-{y}.tif"
        image = tif.imread(image_path)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.normalize_per_plate:
            image = self.plate_normalize_image(image, platename)

        emb_name = f"{platename}/{wellname}-{sitename}-{x}-{y}.pt"
        profile_emb = torch.load(os.path.join(self.profile_embeddings_dir, emb_name), weights_only=True)
        style_emb = torch.load(os.path.join(self.style_embeddings_dir, emb_name), weights_only=True)
        compound_emb = torch.load(os.path.join(self.compound_embeddings_dir, f"{compound}.pt"), weights_only=True)

        profile_emb = torch.tensor(profile_emb)

        profile_emb = (profile_emb - profile_emb.mean()) / profile_emb.std()
        style_emb = (style_emb - style_emb.mean()) / style_emb.std()
        compound_emb = (compound_emb - compound_emb.mean()) / compound_emb.std()
#         profile_emb = F.normalize(torch.tensor(profile_emb), dim=0)
#         style_emb = F.normalize(style_emb, dim=0)
#         compound_emb = F.normalize(compound_emb, dim=0)

        label = (experiment, plate, well)

        return image, profile_emb, style_emb, compound_emb, label, crop_idx, image_name

    def __len__(self):
        return self.df.shape[0]


def get_dataset(args, only_test=False, all=False):
    train_set = None
    val_set = None
    test_set = None

    if args.dataset == "jump":
        transform = A.Compose([
            A.Resize(96, 96),
            A.Normalize([0] * 5, [1] * 5),
            ToTensorV2(),
        ])

        train_set = SingleCellDataset(args, args.df_path_train, transform)
        val_set = SingleCellDataset(args, args.df_path_test, transform)
        test_set = SingleCellDataset(args, args.df_path_test, transform)

        print(f"Training set containing {len(train_set)} single cell images.")
        print(f"Test set containing {len(test_set)} single cell images.")

        args.data_type = "img"
        args.in_size, args.out_size = args.in_size, 5
        print("args.in_size: ", args.in_size)
        args.data_size = (args.out_size, args.img_size, args.img_size)

    elif args.dataset == "3dshapes":
        train_set = Shapes3D("/raid/cian/user/sergei.pnev/data", test=False)
        val_set = Shapes3D(
            "/raid/cian/user/sergei.pnev/data", test=True)
        test_set = Shapes3D("/raid/cian/user/sergei.pnev/data", test=True)

        print(f"Training set containing {len(train_set)} 3d shapes.")
        print(f"Test set containing {len(test_set)} 3d shapes.")

        args.data_type = "img"
        args.in_size, args.out_size = args.in_size, 3
        print("args.in_size: ", args.in_size)
        args.data_size = (args.out_size, args.img_size, args.img_size)

    else:
        raise NotImplementedError()

    if only_test:
        return test_set

    elif all:
        return train_set, val_set, test_set

    else:
        return train_set, test_set
