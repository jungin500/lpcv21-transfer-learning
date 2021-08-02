import glob
import torchvision.transforms as T
import cv2
import numpy as np
import torch
import os


class IdentityTransform:
    def __call__(self, image_resized):
        return image_resized

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CvToTensor:
    def __call__(self, image_resized):
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_resized = np.transpose(image_resized, (2, 0, 1))
        image_resized = np.divide(image_resized, 255.0, dtype=np.float32)
        return image_resized

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SubfolderImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, input_folder, transforms_list = [IdentityTransform()], request_sizes = [(224, 224)]) -> None:
        super().__init__()

        if len(transforms_list) != len(request_sizes):
            raise RuntimeError("transforms_list Requires per-size transforms")

        self.file_list = []
        self.request_sizes = request_sizes

        self.transforms_list = transforms_list
        self.posttransforms = T.Compose([
            CvToTensor()
        ])

        for subfolder_name in glob.glob(os.path.join(input_folder, '**')):

            image_file_list = list(glob.glob(os.path.join(subfolder_name, '**')))
            for image_location in image_file_list:
                # maybe we can use subfolder_name if required?
                self.file_list.append(image_location)

    def describe(self):
        print("Loaded data folders:", len(self.file_list))

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, x):
        image_original = cv2.imread(self.file_list[x])
        image = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
        return [
            image_original,
            *[
                self.posttransforms(
                    self.transforms_list[i](
                        cv2.resize(image, size)
                    )
                )
                for i, size in enumerate(self.request_sizes)
            ]
        ]

    @staticmethod
    def collate_fn(batch):
        assert type(batch) == list and len(batch) > 0 and len(batch[0]) > 0, "Batch is empty or not a list"

        original_images = []
        sizes_per_image = [[] for _ in range(len(batch[0]) - 1)]

        image_sizes_count = len(batch[0]) - 1

        for single_getitem_output in batch:
            # Adds original images (type preserved)
            original_images.append(single_getitem_output[0])

            # Adds image (will be converted to torch.Tensor type)
            for id in range(image_sizes_count):  # 1 -> 0, 2 -> 1, etc ...
                sizes_per_image[id].append(single_getitem_output[id + 1])

        original_images = np.array(original_images, dtype=original_images[0].dtype)
        sizes_per_image = [torch.Tensor(batch_list) for batch_list in sizes_per_image]

        return original_images, *sizes_per_image


'''
class RoundRobinVideoLoader(object):
    def __init__(self, batch_size = 16, video_file_list = None, shuffle=False):
        super().__init__()

        if video_file_list is None:
            video_file_list = glob.glob(os.path.join(os.getcwd(), "videos", "*.m4v"))
        print("Current Directory:", os.getcwd())
        print("Search Directory:", os.path.join(os.getcwd(), "videos", "*.m4v"))

        self.video_file_list = video_file_list[:]
        if shuffle:
            random.shuffle(self.video_file_list)


        self.caps = []
        for video_file in self.video_file_list:
            self.caps.append(
                cv2.VideoCapture(video_file)
            )

        self.batch_size = batch_size
        self.next_cap_id = 0

        self.total_videos = len(self.video_file_list)
        self.total_frames = int(np.sum([
            int(self.caps[i].get(cv2.CAP_PROP_FRAME_COUNT)) for i in range(len(self.caps))
        ]))

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        frames = []

        for _ in range(self.batch_size):
            cap_id = self.next_cap_id
            cap = self.caps[cap_id]

            ret, frame = cap.read()
            if not ret:
                self.caps[cap_id].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("cap: Video returned error after rewinding to first frame")
            
            frames.append(frame)

            self.next_cap_id = ((self.next_cap_id + 1) % self.total_videos)
        
        return frames

    # Not more required as vid2img is implemented
'''