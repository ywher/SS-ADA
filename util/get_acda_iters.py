# import math
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from dataset.semi import SemiDataset

class AC_iters():
    def __init__(self, cfg):
            """
            Initializes the GetACDAIters class.

            Args:
                cfg (dict): Configuration dictionary containing the following keys:
                    - dataset: Dictionary containing dataset related configurations.
                    - trainer: Dictionary containing trainer related configurations.
                        - active: Dictionary containing active learning related configurations.
                            - init_ratio: Initial sample ratio for active learning.
                            - total_ratio: Final sample ratio for active learning.
                            - sample_epoch: Number of sample rounds for active learning.
                        - epochs: Total number of epochs for training.
                    - batch_size: Batch size for training.
                    - num_gpus: Number of GPUs for training.
            
            Returns:
                None
            """
            self.dataset_size = cfg["n_sample"]
            self.batch_size = cfg["batch_size"]
            self.initial_sample_ratio = cfg["active"]["init_ratio"]
            self.final_sample_ratio = cfg["active"]["total_ratio"]
            self.sample_rounds = cfg["active"]["sample_epoch"]
            self.total_epochs = cfg["epochs"]
            self.num_gpus = cfg["n_gpus"]
            # total_iters: total number of iterations for the ACDA algorithm
            # dataloader_iters: a list of iteration counts for each sample round 
            # sample_begin_iters: a list of iteration counts at the beginning of each sample round
            self.total_iters, self.dataloader_iters, self.sample_begin_iters, self.sample_increment = self.calculate_total_iters()
            self.total_samples = self.total_iters * self.batch_size * self.num_gpus

    def calculate_total_iters(self):
        """
            Calculates the total number of iterations for the ACDA algorithm.

            Returns:
                total_iters (int): The total number of iterations.
                dataloader_iters (list): A list of iteration counts for each sample round.
                sample_begin_iters (list): A list of iteration counts at the beginning of each sample round.
                sample_increment (list): A list of iteration increments for each sample round.
        """
        # calculate initial sample size
        initial_sample_size = int(self.dataset_size * self.initial_sample_ratio + 0.5)
        
        # calculate final sample size
        final_sample_size = int(self.dataset_size * self.final_sample_ratio + 0.5)
        
        # calculate the sample size for each epoch
        epoch_sample_size = final_sample_size - initial_sample_size
        
        # calculate the sample increment for each sample round
        if len(self.sample_rounds) == 1:
            sample_increment = [epoch_sample_size]
        else:
            # calculate the size of each segment
            segment_size = epoch_sample_size // len(self.sample_rounds)
            # calculate the size of the last segment
            last_segment_size = epoch_sample_size - segment_size * (len(self.sample_rounds) - 1)
            # calculate the sample increment for each sample round
            sample_increment = [segment_size] * (len(self.sample_rounds) - 1) + [last_segment_size]

        
        # calculate the sample size for each sample round
        sup_samples = [initial_sample_size] + [initial_sample_size + sum(sample_increment[:i+1]) for i in range(len(self.sample_rounds))]
        unsup_samples = [self.dataset_size - sup for sup in sup_samples]
        # dataloader_iters = [int(unsup_sample // (self.num_gpus * self.batch_size)) for unsup_sample in unsup_samples]
        # Calculate data loader iteration counts list
        dataloader_iters = [int(max(unsup_sample, sup_sample) // (self.num_gpus * self.batch_size)) for unsup_sample, sup_sample in zip(unsup_samples, sup_samples)]
        
        # Calculate total iteration counts, initialized with the iteration counts of the first sampling round multiplied by the corresponding number of rounds
        total_iters = dataloader_iters[0] * self.sample_rounds[0]

        # Initialize a list to store the iteration counts at the beginning of each sampling round, adding the iteration counts of the first sampling round multiplied by the corresponding number of rounds to the list
        sample_begin_iters = [dataloader_iters[0] * self.sample_rounds[0]]

        # If the number of sampling rounds is greater than 1, continue iterating to calculate the total iteration counts and the list of iteration counts at the beginning of each sampling round
        if len(self.sample_rounds) > 1:
            for i in range(1, len(self.sample_rounds)):
                # Calculate the iteration counts of the current sampling round multiplied by the number of rounds between the current round and the previous round
                tmp_iters = (dataloader_iters[i]) * (self.sample_rounds[i] - self.sample_rounds[i-1])
                # Add to the total iteration counts
                total_iters += tmp_iters
                # Add the iteration counts at the beginning of the current sampling round to the list, which is the iteration counts of the previous sampling round plus the iteration counts of the current round
                sample_begin_iters.append(sample_begin_iters[-1] + tmp_iters)

        # Finally, add the iteration counts of the last sampling round multiplied by the remaining training rounds
        total_iters += dataloader_iters[-1] * (self.total_epochs - self.sample_rounds[-1])

        
        return total_iters, dataloader_iters, sample_begin_iters, sample_increment
    
    def get_cur_epoch_init_iters(self, epoch):
        """
        Calculates the initial iteration number for a given epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            int: The initial iteration number for the given epoch.
        """
        assert epoch <= self.total_epochs, "epoch should be less than or equal to total_epochs"
        if epoch < self.sample_rounds[0]:
            return self.dataloader_iters[0] * epoch
        elif epoch >= self.sample_rounds[-1]:
            return self.total_iters - self.dataloader_iters[-1] * (self.total_epochs - epoch)
        else:
            for i in range(1, len(self.sample_rounds)):
                if epoch < self.sample_rounds[i]:
                    return self.sample_begin_iters[i - 1] + self.dataloader_iters[i] * (epoch - self.sample_rounds[i-1])
        
if __name__ == "__main__":
    # 示例输入参数
    dataset_size = 140
    initial_sample_ratio = 0.01
    final_sample_ratio = 0.5
    sample_rounds = [10, 20, 30]
    total_epochs = 100
    batch_size = 4
    num_gpus = 1
    
    yaml_path = "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs/parking_bev2024_acda_bisenetv1_single.yaml"
    cfg = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)
    
    ac_iter = AC_iters(cfg)

    print('self.total_iters, self.dataloader_iters, self.sample_begin_iters')
    print(ac_iter.total_iters, ac_iter.dataloader_iters, ac_iter.sample_begin_iters)
    
    for sample_round in [1, 5, 10, 20, 50]:
        print('sample_round:', sample_round)
        print('ac_iter.get_cur_epoch_init_iters(sample_round):', ac_iter.get_cur_epoch_init_iters(sample_round))
        
    trainset_s = SemiDataset(cfg['source']['type'], cfg['source']['data_root'], 'source', cfg['crop_size'], cfg['source']['data_list'], ac_iters=ac_iter)
    trainloader_s = DataLoader(trainset_s, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True)
    print("len(trainset_s):", len(trainset_s))
    print("len(trainloader_s):", len(trainloader_s))