from torch.utils.data.sampler import Sampler
import random


class CTSampler(Sampler):
    """Each batch of data consist of patches from one single CT, like in RCNN
       i.e. we sample patches from the same CT for a certain batch. Set num_per_ct to
       be the same as batch_size to meet this demand.

       Or we JUST SAMPLE SAME NUMBER OF PATCHES from each CT.
    """

    def __init__(self, train_patch_path, num_per_ct=16, pos_fraction=0.5, shuffle_ct=False, numct_perbatch = 6000):

        with open(train_patch_path) as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]

        # ct_dict = {'ct_sub_dir': [(line, idx), (line, idx), ...]}
        self.ct_dict = {}
        for idx, line in enumerate(lines):
            path_line = line.split(' ')[0]
            path_part = path_line.split('/')[0:4]
            sub_dir = '/'.join(path_part)
            if sub_dir not in self.ct_dict:
                self.ct_dict[sub_dir] = []
            self.ct_dict[sub_dir].append((line, idx))

        self.ct_num = numct_perbatch
        self.num_per_ct = num_per_ct
        self.indices = []
        self.pos_fraction = pos_fraction
        self.shuffle_ct = shuffle_ct

        self.pos_num = 0
        self.neg_num = 0
        self.lower = 0
        self.zero = 0

    def __iter__(self):
        self.pos_num = 0
        self.neg_num = 0
        self.lower = 0
        self.zero = 0

        self.indices = []
        ct_sub_dirs = list(self.ct_dict.keys())
        #random.shuffle(ct_sub_dirs)
        ct_sub_dirs = [random.choice(ct_sub_dirs) for _ in xrange(self.ct_num)]

        ct_sub_dirs = ct_sub_dirs[:self.ct_num]
        for sub_dir in ct_sub_dirs:
            pos_list, neg_list = self._split_lines(self.ct_dict[sub_dir])
            sample_list = self._sample_patch(pos_list, neg_list, self.num_per_ct, self.pos_fraction)
            self.indices.extend(self._get_index(sample_list))
        if self.shuffle_ct:
            random.shuffle(self.indices)
        print("pos sample num is: %d, neg sample num is: %d (approximately)" % (self.pos_num, self.neg_num))
        print("ct with lower sampled pos num: %d, ct with zero pos num: %d" % (self.lower, self.zero))
        return iter(self.indices)

    def _split_lines(self, line_idx_list):
        pos_list = []
        neg_list = []
        for line_idx in line_idx_list:
            label = int(line_idx[0].split(' ')[1])
            if label == 0:
                neg_list.append(line_idx)
            else:
                pos_list.append(line_idx)

        return pos_list, neg_list

    def _sample_patch(self, pos_list, neg_list, sample_num, pos_fraction=0.5):
        sample_list = []

        pos_per_image = int(round((sample_num * pos_fraction)))
        pos_per_this_image = min(pos_per_image, len(pos_list))
        sampled_pos = random.sample(pos_list, pos_per_this_image)

        if pos_per_this_image < sample_num * pos_fraction:
            self.lower += 1
        if pos_per_this_image == 0:
            self.zero += 1
        self.pos_num += pos_per_this_image
        neg_per_this_image = sample_num - pos_per_this_image
        neg_per_this_image = min(neg_per_this_image, len(neg_list))
        sampled_neg = random.sample(neg_list, neg_per_this_image)
        self.neg_num += neg_per_this_image

        sample_list = sampled_pos + sampled_neg
        random.shuffle(sample_list)

        while len(sample_list) < sample_num:
            end_idx = min((sample_num - len(sample_list)), len(sample_list))
            sample_list = sample_list + sample_list[:end_idx]
        assert len(sample_list) == sample_num

        return sample_list

    def _get_index(self, sample_list):
        return [line[1] for line in sample_list]

    def __len__(self):
        return (self.ct_num * self.num_per_ct)

