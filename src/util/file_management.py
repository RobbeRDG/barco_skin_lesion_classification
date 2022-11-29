import pandas as pd
import os


def filter_metadata(img_folder, metadata_path, metadata_export_path):
    metadata = pd.read_csv(metadata_path)

    image_names = os.listdir(img_folder)
    image_names = map(lambda x: x[:-len('.jpg')], image_names)

    filtered_metadata = metadata[metadata['maibi_id'].isin(image_names)]

    filtered_metadata.to_csv(metadata_export_path, ',')


