import os

os.makedirs(os.path.join("ADE20K", "datalist"), exist_ok=True)
for method in ["training", "validation"]:
    root_path = os.path.join("ADE20K", "images", "ADE", method)
    with open(os.path.join("ADE20K", "datalist", f"{method}.txt"), "w") as f:
        for folder in os.listdir(root_path):
            for sub_folder in os.listdir(os.path.join(root_path, folder)):
                files = os.listdir(os.path.join(root_path, folder, sub_folder))
                images = [os.path.join(root_path, folder, sub_folder, file) for file in files if file.endswith(".jpg")]
                annotations = [os.path.join(root_path, folder, sub_folder, file) for file in files if file.endswith("_seg.png")]
                for image, annotation in zip(images, annotations):
                    f.write(f"{image} {annotation}\n")
    