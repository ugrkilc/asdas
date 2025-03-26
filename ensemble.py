import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse


# ARGUMENT PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, choices=['ntu60_xsub', 'ntu60_xview', 'ntu120_xsub', 'ntu120_xset'])
args = parser.parse_args()


if args.dataset == 'ntu60_xsub':   
    label_path = 'dataset/ntu60_xsub_val_label.pkl'
    joint_path = 'work_dir/ntu60/xsub/joint/test_score.pkl'
    bone_path = 'work_dir/ntu60/xsub/bone/test_score.pkl'
    joint_motion_path = 'work_dir/ntu60/xsub/joint_motion/test_score.pkl'
    bone_motion_path = 'work_dir/ntu60/xsub/bone_motion/test_score.pkl'
    
elif args.dataset == 'ntu60_xview':
    label_path = 'dataset/ntu60_xview_val_label.pkl'
    joint_path = 'work_dir/ntu60/xview/joint/test_score.pkl'
    bone_path = 'work_dir/ntu60/xview/bone/test_score.pkl'
    joint_motion_path = 'work_dir/ntu60/xview/joint_motion/test_score.pkl'
    bone_motion_path = 'work_dir/ntu60/xview/bone_motion/test_score.pkl'

elif args.dataset == 'ntu120_xsub':   
    label_path = 'dataset/ntu120_xsub_val_label.pkl'
    joint_path = 'work_dir/ntu120/xsub/joint/test_score.pkl'
    bone_path = 'work_dir/ntu120/xsub/bone/test_score.pkl'
    joint_motion_path = 'work_dir/ntu120/xsub/joint_motion/test_score.pkl'
    bone_motion_path = 'work_dir/ntu120/xsub/bone_motion/test_score.pkl'
    
elif args.dataset == 'ntu120_xset':
    label_path = 'dataset/ntu120_xset_val_label.pkl'
    joint_path = 'work_dir/ntu120/xset/joint/test_score.pkl'
    bone_path = 'work_dir/ntu120/xset/bone/test_score.pkl'
    joint_motion_path = 'work_dir/ntu120/xset/joint_motion/test_score.pkl'
    bone_motion_path = 'work_dir/ntu120/xset/bone_motion/test_score.pkl'
else:
    raise ValueError("Invalid dataset specified.")


with open(label_path, 'rb') as f:
    label = np.array(pickle.load(f))

with open(joint_path, 'rb') as f:
    joint_scores = list(pickle.load(f).items())

with open(bone_path, 'rb') as f:
    bone_scores = list(pickle.load(f).items())

with open(joint_motion_path, 'rb') as f:
    joint_motion_scores = list(pickle.load(f).items())

with open(bone_motion_path, 'rb') as f:
    bone_motion_scores = list(pickle.load(f).items())


correct_predictions = 0
total_predictions = 0
top5_correct_predictions = 0


true_labels = []
predicted_labels = []

for i in tqdm(range(len(label[0])), desc="Evaluating"):
    _, true_label = label[:, i]
    _, joint_score = joint_scores[i]
    _, bone_score = bone_scores[i]
    _, joint_motion_score = joint_motion_scores[i]
    _, bone_motion_score = bone_motion_scores[i]
    

    total_score = joint_score + bone_score + joint_motion_score + bone_motion_score

    top5_predictions = total_score.argsort()[-5:]
    top5_correct_predictions += int(int(true_label) in top5_predictions)
    

    predicted_label = np.argmax(total_score)
    correct_predictions += int(predicted_label == int(true_label))
    
    total_predictions += 1

    true_labels.append(int(true_label))
    predicted_labels.append(predicted_label)


top1_accuracy = correct_predictions / total_predictions
top5_accuracy = top5_correct_predictions / total_predictions


print('Top-1 Accuracy: {:.3f}%'.format(100. * top1_accuracy))
print('Top-5 Accuracy: {:.3f}%'.format(100. * top5_accuracy))