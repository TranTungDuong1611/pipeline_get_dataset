# Pipeline Get Dataset
## 1. Set up
```bash
    git clone https://github.com/TranTungDuong1611/pipeline_get_dataset.git
    cd pipeline_get_dataset
```
## 2. Get fake data
- **Step 1:** Clone `gan-control` repository
``` bash
    git clone https://github.com/amazon-science/gan-control.git
```

- **Step 2:** Download weights
    - You can follow the instruction in `gan-control` repository to get weights and extract all the weight in folder `resources/gan_models`
    - If you feel lazy to download and set up weights from the internet, you should run this `script` for short :))
    ```bash
        python download_weights.py
    ```

- **Step 3:** Run the command to get fake data
```bash
    usage: get_gan_data.py [-h] [--save_folder SAVE_FOLDER] [--change_pose CHANGE_POSE] [--smile SMILE] [--change_hair_color CHANGE_HAIR_COLOR]
                       [--color COLOR] [--num_images NUM_IMAGES] [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]

    options:
    -h, --help            show this help message and exit
    --save_folder SAVE_FOLDER
                            path to save images
    --change_pose CHANGE_POSE
                            True|False
    --smile SMILE         True|False
    --change_hair_color CHANGE_HAIR_COLOR
                            True|False
    --color COLOR         bloand|black
    --num_images NUM_IMAGES
                            total generated images
    --batch_size BATCH_SIZE
                            batch_size
    --image_size IMAGE_SIZE
                            shape of generated images
```
**Example:**

```bash
    python get_gan_data.py --save_folder=pipeline_get_dataset --change_pose=True --smile=True --change_hair_color=True --color=bloand --num_images=100 --batch_size=16 --image_size=512
```
