from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from pandas import read_csv
from os import listdir
from os.path import join


class DermaDataset(Dataset):
    IMAGE_EXTENSION = ".jpg"
    IMAGE_COLUMN = "maibi_id"
    LABEL_COLUMN = "dx"
    classes = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec', 'other']
    
    def __init__(self, image_directory: str, meta_data_path: str, transform=None):
        # Get the image files
        self.image_directory = image_directory
        all_files = listdir(self.image_directory)
        self.image_files = [f for f in all_files if f[-len(self.IMAGE_EXTENSION):] == self.IMAGE_EXTENSION]

        # Read in the metadata csv
        unindexed_meta_data = read_csv(meta_data_path)

        # Set the "IMAGE_COLUMN" as the index for the metadata
        self.meta_data = unindexed_meta_data.set_index([self.IMAGE_COLUMN])

        # Set the transforms
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image_file_name = self.image_files[idx]
        label = self.get_integer_label_for_image_name(image_file_name)
        image_path = join(self.image_directory, image_file_name)
        image = default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def get_integer_label_for_image_name(self, image_name:str) -> int:
        text_label = self.get_text_label_for_image_name(image_name)
        if text_label in self.classes:
            return self.classes.index(text_label)
        else:
            return self.classes.index("other")
        
    def get_text_label_for_image_name(self, image_name: str) -> str:
        no_extension = image_name[:-len(self.IMAGE_EXTENSION)]
        return self.meta_data.loc[no_extension][self.LABEL_COLUMN]
            