import paddle
from paddle.io import Dataset, DataLoader

from . import CategoriesSampler, GeneralDataset

from paddle.vision.transforms import RandomResizedCrop, RandomHorizonFlip, ColorJitter, ToTensor(), Normalize, Compose

MEAN = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
STD = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

def get_dataloader(config, mode='train'):
	trfms = Compose([
		RandomResizedCrop(84, 84),
		ColorJitter(0.4, 0.4, 0.4),
		RandomHorizonFlip(),
		ToTensor(),
		Normalize(MEAN, STD),
	])
	
	dataset = dataset = GeneralDataset(data_root=config['data_root'], mode=mode, use_memory=config['use_memory'], transformer=trfms)

	sampler = CategoriesSampler(label_list=dataset.label_list,
									label_num=dataset.label_num,
									episode_size=config['episode_size'],
									episode_num=config['train_episode']
									if mode == 'train' else config['test_episode'],
									way_num=config['way_num'],
									image_num=config['shot_num'] + config['query_num'])
	dataloader = DataLoader(dataset, batch_sampler=sampler,
									num_workers=config['n_gpu'] * 4, pin_memory=True,
									collate_fn=collate_fn)
									
	return dataloader