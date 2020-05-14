import os
import torch
import pickle
import numpy as np
from PIL import Image
from functools import partial
from collections import defaultdict
from collections import Counter
from torch.utils import data
import torchvision.transforms as transforms

class Vocab:
    def __init__(self, formula_file, threshold=0):
        self.formula_file = formula_file
        self.unk_threshold = threshold   # 在is_eligible函数中，判断是否保存倒token表中
        self.token2idx = {}
        self.idx2token = {}
        self.__load()

    def __build(self):
        self.start_token = 0
        self.end_token = 1
        self.pad_token = 2
        self.unk_token = 3
        self.frequency = defaultdict(int)
        self.total = 0

        formulas = open(self.formula_file, 'r', encoding="latin_1")
        lines = formulas.readlines()
        # 统计
        for line in lines:
            tokens = line.rstrip('\n').strip(' ').split()
            for token in tokens:
                self.frequency[token] += 1
                self.total += 1

        self.token2idx = {'<f>' : 0, '</f>' : 1, '<pad>' : 2, '<unk>' : 3}
        self.idx2token = {0 : '<f>', 1 : '</f>', 2 : '<pad>', 3 : '<unk>'}
        idx = 4

        formulas = open(self.formula_file, 'r', encoding="latin_1")
        lines = formulas.readlines()
        # 取数据
        for line in lines:
            tokens = line.rstrip('\n').strip(' ').split()
            for token in tokens:
                if self.__is_eligible(token) and token not in self.token2idx:
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token
                    idx += 1
        
        # 保存 Python 对象
        if not os.path.isdir('vocab'):
            os.mkdir('vocab')
        
        with open(os.path.join('vocab', 'vocab.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def __is_eligible(self, token):   #出现次数与阈值相比较，决定是否保存
        if self.frequency[token] >= self.unk_threshold:
            return True
        return False

    def __load(self):
        try:
            with open(os.path.join('vocab', 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
                self.token2idx = vocab.token2idx
                self.idx2token = vocab.idx2token
                self.start_token = vocab.start_token
                self.unk_token = vocab.unk_token
                self.pad_token = vocab.pad_token
                self.end_token = vocab.end_token
                self.frequency = vocab.frequency
                self.total = vocab.total
        except:
            self.__build()

    def formulas2tensor(self, formulas):
        '''Convert formula to numpy.

        Args:
            formulas (list): A batch of formulas.

        Returns:
	        A numpy object contains index of formulas's token
	    '''
        tensor = []

        # 获取 batch size 中最长字符串的长度
        sorted_formulas = sorted(formulas, key=lambda x: len(x.split()), reverse=True)
        max_len = len(sorted_formulas[0].split())
        
        batch_size = len(formulas)

        # 遍历 batch
        for i in range(batch_size):
            length = len(formulas[i].split())
            temp_formula = [self.token2idx['<pad>']] * max_len # batch 要求长度一致，向最长串对齐
            temp_formula[0] = self.token2idx['<f>']
            temp_formula[1:length] = [self.token2idx[token] if token in self.token2idx else self.token2idx['<unk>'] for token in formulas[i].split()]
            temp_formula[-1] = self.token2idx['</f>']
            tensor.append(temp_formula)

        # tensor = torch.from_numpy(np.array(tensor))
        tensor = np.array(tensor)
         	
        return tensor

    def tensor2formula(self, tensor, pretty=False, tags=True):
        if not pretty:
            if tags:
                return ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0]))
            else:
                return ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0])
                                if self.idx2token[tensor[i]] not in ['<f>', '</f>', '<pad>'])
        else:
            s = ' '.join(self.idx2token[tensor[i]] for i in range(tensor.shape[0]))
            end = s.find('</f>')
            if end != -1 : end = end - 1
            s = s[4:end]
            s = s.replace('<pad>', '')
            s = s.replace('<unk>', '')
            return s

def custom_collate_fn(vocab, batch):
    imgs, labels = zip(*batch)

    labels = vocab.formulas2tensor(labels)

    return torch.cat([img.unsqueeze(0) for img in imgs]), labels

def get_train_data(formula_file, label_file, img_path):
    images = []
    labels = []
    
    with open(formula_file, 'r', encoding='latin_1') as f:
        formulas = [line.rstrip('\n').strip(' ') for line in f.readlines()]

    with open(label_file, 'r', encoding='latin_1') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').strip(' ').split()
            if (len(line) < 2):
                continue
            
            idx = int(line[0])
            labels.append(formulas[idx])
            images.append(os.path.join(img_path, line[1] + ".png"))

    return images, labels

class CustomDataset(data.Dataset):
    r"""
    Custom dataset.

    Arguments:
        root_path (string): root dir stroing dataset.
        method (string, optional): train, validate or test (default: ``train``).
    """

    def __init__(self, root_path, method='train', transform=None):
        '''args:
        root_path: root dir stroing dataset
        method: train, validate or test
        '''
        super(CustomDataset, self).__init__()
        assert(method in ["train", "validate", "test"])

        self.root_path = root_path
        self.method = method
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        else:
            self.transform = transform

        img_path = os.path.join(root_path, "formula_images")
        formula_file = os.path.join(root_path, "im2latex_formulas.lst")
        label_file = os.path.join(root_path, 'im2latex_{}.lst'.format(method))
        
        self.images, self.labels = get_train_data(formula_file, label_file, img_path)

        assert(len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img)
        img = self.transform(img)

        return img, label

def main():
    batch_size = 16
    num_workers = 0
    root_path = './image2latex100k'

    vocab = Vocab(os.path.join(root_path, "im2latex_formulas.lst"))

    train_dataset = CustomDataset(root_path, "train")
    '''
    partial function: https://www.liaoxuefeng.com/wiki/1016959663602400/1017454145929440
    '''
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=partial(custom_collate_fn, vocab))
    
    # show test
    img, label = next(iter(train_loader))
    print('image shape: {}'.format(img.shape))
    print('label shape: {}'.format(label.shape))
    print('\n'.join(vocab.tensor2formula(label[i]) for i in range(label.shape[0])))

if __name__ == '__main__':
    main()