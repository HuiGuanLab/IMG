import os
import codecs
import numpy as np
from tqdm import tqdm
from collections import Counter
import collections
from nltk.tokenize import word_tokenize
from util.data_util import load_json, load_lines, load_pickle, save_pickle, time_to_index
import pickle
import json
PAD, UNK = "<PAD>", "<UNK>"
class CharadesProcessor:
    def __init__(self):
        super(CharadesProcessor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data(self, data, scope):
        results = []
        dic=collections.defaultdict(int)
        got=[]
        for line in tqdm(data, total=len(data), desc='process charades-sta {}'.format(scope)):
            line=json.loads(line.strip())
            vid = line['vid']
            start_time=line['relevant_windows'][0][0]
            end_time=line['relevant_windows'][0][1]
            sentence = line['query']
            qid = "qid"+"_"+str(line['qid'])
            duration = float(line['duration'])
            start_time = max(0.0, float(start_time))
            end_time = min(float(end_time), duration)
            words = word_tokenize(sentence.strip().lower(), language="english")
            new_qid=vid+"_"+str(dic[vid])
            if new_qid in got:
                print("error!!!!!")
                exit(0)
            else:
                got.append(new_qid)
            record = {'sample_id': self.idx_counter,'qid':str(qid),'new_qid':str(new_qid), 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                      'duration': duration, 'words': words}
            dic[vid]+=1
            results.append(record)
            self.idx_counter += 1
        return results

    def convert(self, data_dir):
        self.reset_idx_counter()
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))
        # load raw data
        
        train_data = load_lines(os.path.join(data_dir, 'charades_sta_train_tvr_format.jsonl'))

        test_data = load_lines(os.path.join(data_dir, 'charades_sta_test_tvr_format.jsonl'))
        
        train_set = self.process_data(train_data, scope='train')
        test_set = self.process_data(test_data, scope='test')
        
        return train_set, None, test_set

class CharadesAMProcessor:
    def __init__(self):
        super(CharadesAMProcessor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data(self, data, scope):
        results = []
        dic=collections.defaultdict(int)
        got=[]
        for line in tqdm(data, total=len(data), desc='process charades-sta {}'.format(scope)):
            line=json.loads(line.strip())
            vid = line['vid']
            start_time=line['relevant_windows'][0][0]
            end_time=line['relevant_windows'][0][1]
            sentence = line['query']
            qid = "qid"+"_"+str(line['qid'])
            duration = float(line['duration'])
            start_time = max(0.0, float(start_time))
            end_time = min(float(end_time), duration)
            words = word_tokenize(sentence.strip().lower(), language="english")
            new_qid=vid+"_"+str(dic[vid])
            if new_qid in got:
                print("error!!!!!")
                exit(0)
            else:
                got.append(new_qid)
            record = {'sample_id': self.idx_counter,'qid':str(qid),'new_qid':str(new_qid), 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                      'duration': duration, 'words': words}
            dic[vid]+=1
            results.append(record)
            self.idx_counter += 1
        return results

    def convert(self, data_dir):
        self.reset_idx_counter()
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))
        # load raw data
        
        train_data = load_lines(os.path.join(data_dir, 'charades_sta_train_tvr_format.jsonl'))

        test_data = load_lines(os.path.join(data_dir, 'charades_audiomatter_tvr_format.jsonl'))
        
        train_set = self.process_data(train_data, scope='train')
        test_set = self.process_data(test_data, scope='test')
        
        return train_set, None, test_set

class ActivityNetProcessor:
    def __init__(self):
        super(ActivityNetProcessor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data(self, data, scope):
        results = []
        for vid, data_item in tqdm(data.items(), total=len(data), desc='process activitynet {}'.format(scope)):
            duration = float(data_item['duration'])
            for timestamp, sentence,qid in zip(data_item["timestamps"], data_item["sentences"],data_item["qid"]):
                start_time = max(0.0, float(timestamp[0]))
                end_time = min(float(timestamp[1]), duration)
                words = word_tokenize(sentence.strip().lower(), language="english")
                record = {'sample_id': self.idx_counter,'qid':str(qid), 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                          'duration': duration, 'words': words}
                results.append(record)
                self.idx_counter += 1
        return results

    def convert(self, data_dir):
        self.reset_idx_counter()
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))
        # load raw data
        train_data = load_json(os.path.join(data_dir, 'new_train.json'))
        val_data = load_json(os.path.join(data_dir, 'new_val_2.json'))
        test_data = load_json(os.path.join(data_dir, 'new_val_1.json'))
        # process data
        train_set = self.process_data(train_data, scope='train')
        val_set = self.process_data(val_data, scope='val')
        test_set = self.process_data(test_data, scope='test')
        return train_set, val_set, test_set


def load_glove(glove_path):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict, glove_path):
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)
    return np.asarray(vectors)


def vocab_emb_gen(datasets, emb_path):
    # generate word dict and vectors
    emb_vocab = load_glove(emb_path)
    word_counter, char_counter = Counter(), Counter()
    for data in datasets:
        for record in data:
            for word in record['words']:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict, emb_path)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 5]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors


def dataset_gen(data, vfeat_lens, word_dict, char_dict, max_pos_len, scope):
    dataset = list()
    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            continue
        s_ind, e_ind, _ = time_to_index(record['s_time'], record['e_time'], vfeat_lens[vid], record['duration'])
        word_ids, char_ids = [], []
        for word in record['words'][0:max_pos_len]:
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)
        result = {'sample_id': record['sample_id'],'qid':record['qid'],'new_qid':record['new_qid'], 'vid': record['vid'], 's_time': record['s_time'],
                  'e_time': record['e_time'], 'duration': record['duration'], 'words': record['words'],
                  's_ind': int(s_ind), 'e_ind': int(e_ind),'v_len': vfeat_lens[vid], 'w_ids': word_ids,
                  'c_ids': char_ids}
        dataset.append(result)
    return dataset


def gen_or_load_dataset_iv2(configs,new_json,old_json):
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    data_dir = os.path.join('data', 'dataset', configs.task)
    if configs.task=="charadesAM":
        data_dir = os.path.join('data', 'dataset', 'charades')
    
    if configs.task == 'charades' or configs.task =='charadesAM':
        feature_dir = os.path.join('data', 'features', 'charades', 'i3d_video')
    else:
        feature_dir = os.path.join('data', 'features', 'charades', 'c3d_video')
    
    if configs.suffix is None:
        save_path = os.path.join(configs.save_dir, '_'.join([configs.task,configs.visual_features,str(configs.max_pos_len)]) +'.pkl')
    else:
        save_path = os.path.join(configs.save_dir,'_'.join([configs.task,configs.visual_features,str(configs.max_pos_len),configs.suffix]) +'.pkl')
    if os.path.exists(save_path):
        dataset = load_pickle(save_path)
        return dataset
    
    emb_path = os.path.join('data', 'features', 'glove.840B.300d.txt')
    # load video feature length
    vfeat_lens=new_json
    # old_vfeat_lens=old_json
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(configs.max_pos_len, vfeat_len)
    # for vid, vfeat_len in old_vfeat_lens.items():
    #     old_vfeat_lens[vid] = min(configs.max_pos_len, vfeat_len)
    # load data
    if configs.task == 'charades':
        processor = CharadesProcessor()
    elif configs.task == 'activitynet':
        processor = ActivityNetProcessor()
    elif configs.task == 'charadesAM':
        processor = CharadesAMProcessor()
    else:
        raise ValueError('Unknown task {}!!!'.format(configs.task))
    
    train_data, val_data, test_data = processor.convert(data_dir)
    # generate dataset
    data_list = [train_data, test_data] if val_data is None else [train_data, val_data, test_data]
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, emb_path)
    
    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'train')
    val_set = None if val_data is None else dataset_gen(val_data, vfeat_lens, word_dict, char_dict,
                                                        configs.max_pos_len, 'val')
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'test')
    # save dataset
    n_val = 0 if val_set is None else len(val_set)
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': n_val,
               'n_test': len(test_set), 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, save_path)
    return dataset
