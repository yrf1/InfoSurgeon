import os
import json
import pickle
import pandas as pd


def parse_entity(entity_file, entity_fb_link):
    entity_dict, swap_dict = {}, ({},{})
    entity_ID_ptr = ""
    for idx, line in enumerate(entity_file):
        line = line.split("\t")
        entity_ID_cur = line[0]
        if entity_ID_cur != entity_ID_ptr:
            entity_subtype = line[2].split("#")[1]
            entity_type = entity_subtype.split(".")[0]
            entity_dict[entity_ID_cur] = {"type": entity_type, "subtype": entity_subtype}
            entity_ID_ptr = entity_ID_cur
            swap_dict[0][entity_ID_cur] = entity_type
            if entity_type not in swap_dict[1]:
                swap_dict[1][entity_type] = []
        elif line[1] != "type":
            if line[1] != "link":
                entity_ID, mention_type, mention, offset, _ = line
                this_fb_link = entity_fb_link[offset] if offset in entity_fb_link else ""
                article, offset = offset.split(":")
                if mention_type == "canonical_mention":
                    if article not in entity_dict:
                        entity_dict[article] = {}
                    entity_dict[article][entity_ID] = entity_dict[entity_ID_cur]
                    entity_dict[article][entity_ID]["mention"] = mention[1:-1]
                    entity_dict[article][entity_ID]["offsets"] = []
                    del entity_dict[entity_ID_cur]
                    swap_dict[1][entity_type].append((article,entity_ID))
                elif mention_type == "mention" or mention_type == "nominal_mention":
                    entity_dict[article][entity_ID]["offsets"].append(offset)
                if "freebase_link" not in entity_dict[article][entity_ID]:
                    entity_dict[article][entity_ID]["freebase_link"] = this_fb_link
                elif this_fb_link != "" and entity_dict[article][entity_ID]["freebase_link"] == "":
                    entity_dict[article][entity_ID]["freebase_link"] = this_fb_link
            elif "link" not in entity_dict[article][entity_ID]:
                entity_ID, _, link_ID, _, _ = line
                entity_dict[article][entity_ID]["geobase_link"] = link_ID
    return entity_dict, swap_dict

def parse_relation(relation_file):
    relation_dict = {}
    for idx, line in enumerate(relation_file):
        line = line.split("\t")
        if line[1] == "type":
            relation_ID, _, relation_type = line
            relation_dict[relation_ID] = {"type": relation_type.split("#")[1]}
        elif "https:" in line[1]: 
            relation_ID, relation_type, entity_ID, offset, _ = line
            article, offset = offset.split(":")
            if article not in relation_dict:
                relation_dict[article] = {}
            if relation_ID not in relation_dict[article]:
                relation_dict[article][relation_ID] = relation_dict[relation_ID]
                relation_dict[article][relation_ID]["offset"] = offset
                del relation_dict[relation_ID]
            if "entity1" not in relation_dict[article][relation_ID]:
                relation_dict[article][relation_ID]["entity1"] = entity_ID
            elif "entity2" not in relation_dict[article][relation_ID]:
                relation_dict[article][relation_ID]["entity2"] = entity_ID
    return relation_dict

def parse_event(event_file, event_time_file):
    event_dict = {}
    for idx, line in enumerate(event_file):
        line = line.split("\t")
        if line[1] == "canonical_mention.actual":
            event_ID, _, mention, offset, _ = line
            article, offset = offset.split(":")
            if article not in event_dict:
                event_dict[article] = {}
            event_dict[article][event_ID] = {"mention": mention, "offset": offset, "args": []}
        elif "https:" in line[1]:
            event_ID, arg_type, entity_ID, offset, _ = line
            article, offset = offset.split(":")
            arg_type = arg_type.split("#")[1]
            event_dict[article][event_ID]["args"].append((arg_type, entity_ID, offset))
    for idx, line in enumerate(event_time_file):
        line = line.split("\t")
        if line[1] == "canonical_mention.actual":
            article_ID = line[3].split(":")[0]
        if len(line[1]) == 2 and line[2] != '"inf"':
            event_ID = line[0]
            event_dict[article_ID][event_ID][line[1]] = line[2][1:-1]
    return event_dict

def parse_KG(dir):
    f = open(dir+"/en_full_link.cs").read().split("\n")
    entity_file = list(filter(lambda x: x[:8] == ":Entity_", f))
    relation_file = list(filter(lambda x: x[:10] == ":Relation_", f))
    event_file = list(filter(lambda x: x[:7] == ":Event_", f))
    try:
        event_time_file = open(dir+"/event/events_4tuple.cs").read().strip().split("\n")
    except:
        event_time_file = ""
    entity_fb_link = parse_FreeBase_link(dir)
    return parse_entity(entity_file, entity_fb_link), parse_relation(relation_file), \
        parse_event(event_file, event_time_file)

def parse_FreeBase_link(data_dir):
    ### the file can be out of order :( ###
    entity_link_file = data_dir + "/edl/en.linking.freebase.cs"
    entity_link_file = open(entity_link_file).read().strip().split("\n")[2:]
    entity_link_dict, entity_link_dict_rev, i = {}, {}, 0
    while i < len(entity_link_file):
        entity_link_line = entity_link_file[i].split("\t")
        if entity_link_line[1] == "link":
            entityID, _, link = entity_link_line
            if entityID not in entity_link_dict_rev:
                entity_link_dict_rev[entityID] = {"offsets":[], "link":None}
            entity_link_dict_rev[entityID]["link"] = link
        elif entity_link_line[1] != "type":
            entityID, _, mention, offset, _ = entity_link_line
            if entityID not in entity_link_dict_rev:
                entity_link_dict_rev[entityID] = {"offsets":[], "link":None}
            entity_link_dict_rev[entityID]["offsets"].append(offset)
        """if entity_link_line[1] == "type":
            j = i + 1
            entity_offsets = []
            canonical_mention = ""
            while entity_link_file[j].split("\t")[1] != "link":
                if entity_link_file[j].split("\t")[1] == "canonical_mention":
                    canonical_mention = entity_link_file[j].split("\t")[2][1:-1]
                entity_offsets.append(entity_link_file[j].split("\t")[3])
                j += 1
            for entity_offset in entity_offsets:
                entity_offset = entity_offset.replace(".","_")
                entity_link_dict[entity_offset] = (entity_link_file[j].split("\t")[2], canonical_mention)
            i = j + 1"""
        i += 1
    for entityID, entity_link_data in entity_link_dict_rev.items():
        for offset in entity_link_data["offsets"]:
            offset = offset.replace(".", "_")
            entity_link_dict[offset] = entity_link_data["link"]
    return entity_link_dict

def parse_KE(entity_dict, relation_dict, event_dict, articleID, data_dir="", mode="detect"):
    triplet_mention_list, triplet_KB_id_list = [], []
    if articleID in relation_dict:
        for relationID, relation in relation_dict[articleID].items():
            try:
                en1_ID, r, en2_ID = relation["entity1"], relation["type"], relation["entity2"]
                en1_mention, en2_mention = entity_dict[articleID][en1_ID]["mention"], entity_dict[articleID][en2_ID]["mention"]
                triplet_mention_list.append((en1_mention, r, en2_mention))
                FB_link1 = entity_dict[articleID][en1_ID]["freebase_link"] if "freebase_link" in \
                        entity_dict[articleID][en1_ID] else ""
                GB_link1 = entity_dict[articleID][en1_ID]["geobase_link"] if "geobase_link" in \
                        entity_dict[articleID][en1_ID] else ""
                FB_link2 = entity_dict[articleID][en2_ID]["freebase_link"] if "freebase_link" in \
                        entity_dict[articleID][en1_ID] else ""
                GB_link2 = entity_dict[articleID][en2_ID]["geobase_link"] if "geobase_link" in \
                        entity_dict[articleID][en2_ID] else ""
                en1_type, en2_type = entity_dict[articleID][en1_ID]["subtype"], entity_dict[articleID][en2_ID]["subtype"]
                triplet_KB_id_list.append(((en1_ID, FB_link1, GB_link1, en1_type, en1_mention), (r, relation["offset"]), \
                        (en2_ID, FB_link2, GB_link2, en2_type, en2_mention)))
            except:
                pass
    for event, event_data in event_dict[articleID].items():
        for i, event_arg_data1 in enumerate(event_data["args"]):
            for j, event_arg_data2 in enumerate(event_data["args"]):
                if j > i:
                    try:
                        en1 = entity_dict[articleID][event_arg_data1[1]]
                        en1_mention = en1["mention"]
                        en1_ID, en1_type = event_arg_data1[1], en1["subtype"]
                        en1_link = en1["freebase_link"] if "freebase_link" in en1 else "NIL"
                        en1_GB_link = en1["geobase_link"] if "geobase_link" in en1 else ""
                        en2 = entity_dict[articleID][event_arg_data2[1]]
                        en2_mention = en2["mention"]
                        en2_ID, en2_type = event_arg_data2[1], en2["subtype"]
                        en2_link = en2["freebase_link"] if "freebase_link" in en2 else "NIL"
                        en2_GB_link =  en2["geobase_link"] if "geobase_link" in en2 else ""
                        event_edge_label = ""
                        event_arg1_tokenized = event_arg_data1[0].split(".")
                        event_arg2_tokenized = event_arg_data2[0].split(".")
                        for i in range(len(event_arg1_tokenized)):
                            if i <= len(event_arg1_tokenized) and \
                                    event_arg1_tokenized[i] == event_arg2_tokenized[i]:
                                event_edge_label += event_arg1_tokenized[i] + "."
                        for tok in event_arg1_tokenized:
                            if tok not in event_edge_label:
                                event_edge_label += tok
                        event_edge_label += "-"
                        for tok in event_arg2_tokenized:
                            if tok not in event_edge_label:
                                event_edge_label += tok
                        if event_edge_label[-1] == "-" or event_edge_label[-1] == ".":
                            event_edge_label = event_edge_label.strip(".")
                            event_edge_label = event_edge_label.strip("-")
                        triplet_mention_list.append((en1_mention, event_edge_label, en2_mention))
                        triplet_KB_id_list.append(((en1_ID, en1_link, en1_GB_link, en1_type, en1_mention), (event_edge_label,event_data["offset"]), \
                                (en2_ID, en2_link, en2_GB_link, en2_type, en2_mention)))
                    except:
                        pass
    if mode == "generate":
        triplet_mentions = ""
        for triplet in triplet_mention_list:
            h, r, t = triplet
            triplet_mentions += "<"+h+", "+r+", "+t+"> "
        for event, event_data in event_dict[articleID].items():
            if len(event_data["arg_roles"]) <= 1 and event_data["trigger"] not in triplet_mentions:
                triplet_mentions += "<"+event_data["trigger"]+ "> "
        for entity, entity_data in entity_dict[articleID].items():
            entity_mention = entity_data["mention"]
            if entity_mention not in triplet_mentions and \
                    "XXXX" not in entity_mention and "P1" not in entity_mention:
                triplet_mentions += "<"+entity_mention+"> "
        article_headline = open(data_dir+"rsd/"+articleID+".rsd.txt").read().split("\n")[0]
        triplet_mention_list = article_headline + "\n"+triplet_mentions.replace(".actual", "")
    return triplet_mention_list, triplet_KB_id_list

def parse_fake_KE(parsed_KE, intended_annotation):
    intended_annotation = intended_annotation.values[0]
    #print(parsed_KE)
    #print( intended_annotation)
    return ""

def process_data(data_dir, source_type="body_txt", mode="detect"):
    data = {}
    if os.path.exists(data_dir+"annotation_direct_entity_swapping.csv"):
        intended_annotations = pd.read_csv(data_dir+"annotation_direct_entity_swapping.csv")
    (KG_entity_dict, swap_dict), KG_relation_dict, KG_event_dict = parse_KG(data_dir)
    for articleID in os.listdir(data_dir+"/rsd"):
        articleID = articleID.replace(".rsd.txt", "")
        if articleID not in KG_event_dict:
            continue
        parsed_KE, parsed_KE_metadata = parse_KE(KG_entity_dict, KG_relation_dict, KG_event_dict, articleID, data_dir, mode)
        try:
            fake_KE_label = parse_fake_KE(parsed_KE, intended_annotations[intended_annotations["id"]==articleID]["annotation"]) \
                if "fake" in data_dir else ""
        except:
            fake_KE_label = ""
        data[articleID] = {}
        data[articleID]["text"] = {"source": source_type, "triplet_mentions": parsed_KE, "triplet_metadata": \
            parsed_KE_metadata, "triplet_annotations": fake_KE_label, "doc_label": "fake" in data_dir}
    if mode == "detect":
        with open(data_dir+"detection_stage_data.pkl", "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif mode == "generate":
        df = []
        for articleID, article_data in data.items():
            article_txt = open(data_dir+"rsd/"+articleID+".rsd.txt").read()
            df.append([articleID, article_data["triplet_mentions"], article_txt])
        pd.DataFrame(df, columns=["articleID", "source", "target"]).to_csv(data_dir+"KG2txt_train_data.csv", index=False)
