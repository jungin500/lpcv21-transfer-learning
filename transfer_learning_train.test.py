import unittest

class TestTransferLearningTrain(unittest.TestCase):
    def test_dataset_load_5item(self):
        # from transfer_learning_train import RoundRobinVideoLoader
        from transfer_learning_train import SubfolderImageOnlyDataset, IdentityTransform
        import torch
        from torch.utils.data import DataLoader
        import numpy as np
        import glob, os

        # loader = RoundRobinVideoLoader(batch_size=64, video_file_list=glob.glob(os.path.join("videos", "*.m4v")), shuffle=True)
        # print("Total frames:", len(loader))

        # for frame_batches in enumerate(loader):
        #     print(len(frame_batches))

        dataset = SubfolderImageOnlyDataset(
            input_folder='./data/',
            request_sizes = [(640, 384), (300, 300)],
            transforms_list=[
                IdentityTransform(),
                IdentityTransform()
            ]
            )
        dataset.describe()

        batch_size = 2
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=True, num_workers=0,
            collate_fn=SubfolderImageOnlyDataset.collate_fn, pin_memory=True)

        for i, image_per_shape in enumerate(dataloader):
            if i >= 5:
                break

            # Check data type
            self.assertEqual(len(image_per_shape), 3)  # 1 original image + 2 requests

            # # We don't know original image shape but know channel!
            # self.assertEqual(type(image_per_shape[0]), np.ndarray)
            # self.assertEqual(image_per_shape[0].shape[0], batch_size)
            # # self.assertEqual(image_per_shape[0].shape[1], ???)
            # # self.assertEqual(image_per_shape[0].shape[2], ???)
            # self.assertEqual(image_per_shape[0].shape[3], 3)
            
            # WE SHOULD FORCE ITS DIMENSION ORDER!
            self.assertEqual(type(image_per_shape[0]), np.ndarray)
            self.assertEqual(image_per_shape[0].shape[0], batch_size)
            # self.assertEqual(image_per_shape[0].shape[1], ???)
            # self.assertEqual(image_per_shape[0].shape[2], ???)
            self.assertEqual(image_per_shape[0].shape[3], 3)
            
            self.assertEqual(type(image_per_shape[1]), torch.Tensor)
            self.assertEqual(image_per_shape[1].shape[0], batch_size)
            self.assertEqual(image_per_shape[1].shape[1], 3)
            self.assertEqual(image_per_shape[1].shape[2], dataset.request_sizes[0][1])  # channel swap
            self.assertEqual(image_per_shape[1].shape[3], dataset.request_sizes[0][0])  # channel swap

            self.assertEqual(type(image_per_shape[2]), torch.Tensor)
            self.assertEqual(image_per_shape[2].shape[0], batch_size)
            self.assertEqual(image_per_shape[2].shape[1], 3)
            self.assertEqual(image_per_shape[2].shape[2], dataset.request_sizes[1][1])  # channel swap
            self.assertEqual(image_per_shape[2].shape[3], dataset.request_sizes[1][0])  # channel swap
            
            # Verbose print
            print("%dth Data: " % i, end='')
            for image in image_per_shape:
                if type(image) == torch.Tensor:
                    print(image.shape, end=' ')
                elif type(image) == np.ndarray:
                    print(image.shape, end=' ')
                elif type(image) == tuple:
                    print(len(image), end='=>')
                    print(image[0].shape, end=' ')
            print()



if __name__ == '__main__':
    unittest.main(verbosity=2)