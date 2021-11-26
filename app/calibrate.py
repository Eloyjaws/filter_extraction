# Use pyyaml to save and load camera parameters
import yaml

with open(r'data/camera_parameters.yaml', 'w') as file:
    documents = yaml.dump(dict_file, file)

with open(r'data/camera_parameters.yaml') as file:
    doc = yaml.safe_load(file, Loader=yaml.FullLoader)


if __name__ == "__main__":
    camera_id = "One_Plus"
    print(f"Running Calibration Routine - Camera ID: {camera_id}")
