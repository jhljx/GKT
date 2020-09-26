import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class KTDataset(Dataset):

    def __init__(self, features, questions, answers):
        self.features = features
        self.questions = questions
        self.answers = answers
        # self.seq_len = seq_len

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    (features, questions, answers) = zip(*batch)
    features = [torch.tensor(feat) for feat in features]
    questions = [torch.tensor(qt) for qt in questions]
    answers = [torch.tensor(ans) for ans in answers]
    # features_len = [len(feat) for feat in features]
    # questions_len = [len(qt) for qt in questions]
    # answers_len = [len(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad


def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, use_cuda=True):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    df = pd.read_csv(file_path)
    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {file_path}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {file_path}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {file_path}")

    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises

    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []

    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['correct'].tolist())
        seq_len_list.append(series['correct'].shape[0])

    df.groupby('user_id').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(df['skill_with_answer'].max() + 1)
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = question_dim
    assert feature_dim == 2 * question_dim

    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    if graph_type == 'Dense':
        graph = build_dense_graph(concept_num)
    elif graph_type == 'Transition':
        graph = build_transition_graph(question_list, seq_len_list, student_num, concept_num)
    elif graph_type == 'DKT':
        graph = build_dkt_graph(dkt_graph_path, concept_num)
    else:
        graph = None
    if use_cuda:
        graph = graph.cuda()
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader


def build_transition_graph(question_list, seq_len_list, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    for i in range(student_num):
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph