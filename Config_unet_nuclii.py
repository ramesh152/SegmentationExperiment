import os

from trixi.util import Config


def get_config():
    # Set your own path, if needed.
    #data_root_dir = os.path.abspath('data')  # The path where the downloaded dataset is stored.
    #data_root_dir = "/home/ramesh/Desktop/WS/Implementation/experiment/Data/Filtereddataset"
    #data_root_dir ="/home/ramesh/Desktop/IIITB/experiment/data/FilteredDataSet"
    data_root_dir = "/home/ramesh/Desktop/IIITB/experiment/data/NucliiData"
    taskName = "Nuclii Segmentation"
    #taskName = "Task09_Spleen"
    c = Config(
        update_from_argv=True,
        # Train parameters
        num_classes=1,
        in_channels=1,
        batch_size=6,
        patch_size=64,
        n_epochs=50,
        learning_rate=0.00001,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Segmentation_Experiment_Unet_Nuclii',
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        start_visdom=True,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=False,
        checkpoint_dir=' ',

        # Adapt to your own path, if needed.
        #google_drive_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
        
        dataset_name = taskName,
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, taskName,'imagesTr'),  # This is where your training and validation data is stored
       
        #data_test_dir=os.path.join(data_root_dir, 'Task04_Hippocampus/preprocessed'),  # This is where your test data is stored
        split_dir= os.path.join(data_root_dir, taskName,'preprocessed'),  # This is where the 'splits.pkl' file is located, that holds your splits.
    )

    print(c)
    return c

