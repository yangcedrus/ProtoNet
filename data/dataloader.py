import paddle
from paddle.io import DataLoader

from .sampler import CategoriesSampler
from .dataset import GeneralDataset

from paddle.vision.transforms import Resize, CenterCrop,RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, Compose

MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

def get_dataloader(config, mode='train'):

	if mode == 'train':
		trfms = Compose([
			RandomResizedCrop((84, 84)),
			ColorJitter(0.4, 0.4, 0.4),
			RandomHorizontalFlip(),
			ToTensor(),
			Normalize(MEAN, STD),
		])
	else:
		trfms = Compose([
			Resize((92,92)),
			CenterCrop((84,84)),
			ToTensor(),
			Normalize(MEAN, STD)
		])
	
	dataset = dataset = GeneralDataset(data_root=config.data_root, mode=mode, use_memory=config.use_memory, trfms=trfms)

	sampler = CategoriesSampler(label_list=dataset.label_list,
									label_num=dataset.label_num,
									episode_size=config.episode_size,
									episode_num=config.train_episode
									if mode == 'train' else config.test_episode,
									way_num=config.way_num
									if mode == 'train' else config.test_way,
									image_num=config.shot_num + config.query_num)
	dataloader = DataLoader(dataset, batch_sampler=sampler,
									num_workers=config.n_gpu * 4, collate_fn=None)
									
	return dataloader
