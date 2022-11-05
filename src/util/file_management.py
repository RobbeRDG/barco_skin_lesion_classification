from os.path import join

def setup_file_system(in_colab):
    if in_colab:
        from google.colab import drive

        # Set the base and mount path
        MOUNT_PATH_DRIVE = '/content/drive'
        BASE_PATH = join(
            MOUNT_PATH_DRIVE, 
            "MyDrive/barco_skin_lesion_classification"
        )

        # Mount the google drive
        drive.mount(MOUNT_PATH_DRIVE)

        return BASE_PATH

    else:
        return "/workspaces/barco_skin_lesion_classification"