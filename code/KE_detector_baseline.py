import torch
import pickle
import numpy as np

# Read the triplets file
with open("data/VOA/ke_labels.pkl", "rb") as f:
    data_metadata = pickle.load(f)

# Compute the portion of KEs labeled fake
count_fake, count_total = 0, 0
per_data_point_count = []
per_data_point_count_atleast1linked = []
for artID, art_data in data_metadata.items():
    count_fake += art_data[-1].count(1)
    count_total += len(art_data[-1])
    per_data_point_count.append(art_data[-1].count(1)/(len(art_data[-1])+0.00001))
    # count the ones that are linked to atleast one proper noun entity
    # cuz these ones are more salient and may have higher change of being manipualted
    uhh_count = 0
    for trip_idx in range(len(art_data[-1])):
        try:
            if art_data[-1][trip_idx] == 1 and ("m." in art_data[0][trip_idx][0][1] or "m." in art_data[2][trip_idx][2][1]):
                uhh_count += 1
        except:
            pass
    per_data_point_count_atleast1linked.append(uhh_count/(len(art_data[-1])+0.00001))

pct1 = count_fake / count_total
pct2 = np.mean(np.array(per_data_point_count))
pct3 = np.mean(np.array(per_data_point_count_atleast1linked))
print("The percentage of KEs labeled as manipulated in the dataset, depending on how counting is performed: ")
print(pct1, pct2, pct3)

# Now compute the prediction score
tp, fp, fn = 0.0, 0.0, 0.0
for artID, art_data in data_metadata.items():
    gt_labels = art_data[-1]
    pred_labels =  pct2 * torch.ones_like(torch.tensor(gt_labels))
    pred_labels = torch.bernoulli(pred_labels).tolist()
    pred_labels = [int(x) for x in pred_labels]
    gt_labels = torch.tensor(gt_labels)
    pred_labels = torch.tensor(pred_labels)
    tp += ((gt_labels == pred_labels) & (gt_labels == 1)).sum()
    fp += ((gt_labels != pred_labels) & (gt_labels == 0)).sum()
    fn += ((gt_labels != pred_labels) & (gt_labels == 1)).sum()
print("Fscore: " + str(tp/(tp+0.5*(fp+fn))))
