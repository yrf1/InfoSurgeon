import os, json, spacy

nlp = spacy.load("en_core_web_sm")

dataset = "NYTimes"
output = {}
for fID in os.listdir("data/"+dataset+"/txt"):
    fID = fID.replace(".txt","")
    print(fID)
    output[fID] = {}
    art_txt = open("data/"+dataset+"/txt/"+fID+".txt").read()
    art_entities = [x.text for x in nlp(art_txt).ents]
    for i in range(3):
        if os.path.exists("data/"+dataset+"/caption/txt/"+fID+"_cap_"+str(i)+".txt"):
            ind_facs = []
            cap_txt = ftxt = open("data/"+dataset+"/caption/txt/"+fID+"_cap_"+str(i)+".txt").read()
            cap_entities = [x.text for x in nlp(cap_txt).ents]
            overlap = set(art_entities).intersection(set(cap_entities))
            num_overlap = len(overlap)
            binary_overlap = int(num_overlap>0)
            ind_facs.append(binary_overlap)
            ind_facs.append(num_overlap)
            ind_facs.append(len(cap_entities))
            output[fID][i] = ind_facs
with open("data/"+dataset+"/ind_facs.json", "w") as f:
    json.dump(output, f)
