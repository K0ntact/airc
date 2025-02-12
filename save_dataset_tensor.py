import os
import torch
import argparse
from tqdm import tqdm
from dataset import TimesformerGRUData


def main(vid_folder_path, label_path, output_dir):
    dataset = TimesformerGRUData(vid_folder_path=vid_folder_path, label_path=label_path)
    os.makedirs(output_dir, exist_ok=True)

    video_windows = {}
    for vid_path, tensor_idx in dataset.video_tensor_idx_mapping:
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        if vid_name not in video_windows:
            video_windows[vid_name] = []
        video_windows[vid_name].append(tensor_idx)

    for vid_name, window_ids in tqdm(video_windows.items(), desc="Saving tensors and labels"):
        for window_id in window_ids:
            vid_path = os.path.join(vid_folder_path, vid_name + '.mp4')
            tensor, label = dataset._load_frame_tensor(str(vid_path), window_id)
            output_path = os.path.join(output_dir, f'{vid_name}_window_{window_id}.pt')
            torch.save({'tensor': tensor, 'label': label}, output_path)

    print(f"Saved tensors and labels to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save tensors and labels of each video window from ATMADataset")
    parser.add_argument('--vid_folder_path', type=str, default="./datasets/ATMA-V/videos/train/aug", help="Path to the video folder")
    parser.add_argument('--label_path', type=str, default="./datasets/ATMA-V/labels/labels.txt", help="Path to the label file")
    parser.add_argument('--output_dir', type=str, default="./datasets/ATMA-V/tensors/train", help="Directory to save the output tensors and labels")

    args = parser.parse_args()
    main(args.vid_folder_path, args.label_path, args.output_dir)
