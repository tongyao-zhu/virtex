import argparse
import pandas as pd
import torch
import os


def get_tensor(csv_dir, image_list_path, mode, tensor_dir):
    df = pd.read_csv(os.path.join(csv_dir, f"PHOENIX-2014-T.{mode}.corpus.csv"), delimiter = "|")

    video_index_list = list(df.name)

    with open(image_list_path, 'r') as f:
        image_list = f.readlines()

    print(f"image list has {len((image_list))} lines")
    tensor_list = []

    mode_tensor = torch.load(os.path.join(tensor_dir, f"{mode}_tensor.pt"))
    print(f'finished loading mode tensor, has length of {len(mode_tensor)}')

    seen_video = set()
    for line in image_list:
        if mode in line:
            video_name = (line.strip().split("/")[-2])
            video_index = video_index_list.index(video_name)
            number = int((line.strip().split("/")[-1])[-8:-4])
            frame_index = number-1
            seen_video.add(video_name)
            tensor_list.append(mode_tensor[video_index][frame_index])
            assert( (mode_tensor[video_index][frame_index]==torch.zeros(1048)).all() == False)
    print(f"finished_processing all lines, tensor_list has shape {len(tensor_list)}")
    torch.save(torch.stack(tensor_list), f"./features_phoenix/finished_tensor_{mode}.pt")
    print(f"finished saving")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--tensor_dir", type=str,default="./features_phoenix", help="directory that stores feature")
    ap.add_argument("--mode", default=None, type=str, help="feature tensor file after combination")
    ap.add_argument("--image_list_path", default=None, help="the list containing all images", required=True)
    ap.add_argument("--csv_dir",type=str, default=100, help="the directory containing csv information")
    args = ap.parse_args()
    print(f"you've entered the following arguments: {args}")
    get_tensor(args.csv_dir, args.image_list_path, args.mode, args.tensor_dir)
