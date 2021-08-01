import os, csv, json, random, pandas as pd
from datetime import date

train_val_test_split_mode = "online"

orig_real_art_fpath = "data/NYTimes_orig/real_arts/"
orig_fake_art_fpath = "data/NYTimes_orig/large_out.jsonl"
orig_fake_art_test_fpath = "data/NYTimes_orig/fake_arts_mega/"
orig_cap_fpath = "data/NYTimes_orig/captioning_dataset.json"
orig_img_fpath = "data/NYTimes_orig/img_urls_all.json"

dest_dir = "data/NYTimes"
mapping = []
old_mapping = pd.read_csv("data/NYTimes_data_mappings.csv", header=None)

with open(orig_cap_fpath, "r") as f:
    orig_cap = json.load(f)
with open(orig_img_fpath, "r") as f:
    orig_img = json.load(f)

if not os.path.exists(dest_dir):
    os.system("mkdir " + dest_dir)
if not os.path.exists(dest_dir + "/txt/"):
    os.system("mkdir " + dest_dir + "/txt/")
if not os.path.exists(dest_dir + "/caption/"):
    os.system("mkdir " + dest_dir + "/caption/")
    os.system("mkdir " + dest_dir + "/caption/txt")
if not os.path.exists(dest_dir + "/title/"):
    os.system("mkdir " + dest_dir + "/title/")
    os.system("mkdir " + dest_dir + "/title/txt")
if not os.path.exists(dest_dir + "/vision/"):
    os.system("mkdir " + dest_dir + "/vision/")
    os.system("mkdir " + dest_dir + "/vision/data")
    os.system("mkdir " + dest_dir + "/vision/data/jpg")
    os.system("mkdir " + dest_dir + "/vision/data/jpg/jpg")
if not os.path.exists(dest_dir + "/metadata/"):
    os.system("mkdir " + dest_dir + "/metadata/")
    os.system("mkdir " + dest_dir + "/metadata/txt")

def grab_img_cap(fname, new_fname, i, orig_cap_idx):
    if i in orig_img[fname] and \
        i in orig_cap[fname]["images"]:
        img_fname = dest_dir+"/vision/data/jpg/jpg/"+new_fname+"_img_"+i+".jpg"
        os.system("curl -o " + img_fname + " " + orig_img[fname][orig_cap_idx])
        with open(dest_dir+"/caption/txt/"+new_fname+"_cap_"+i+".txt", "w") as f:
            f.write(orig_cap[fname]["images"][orig_cap_idx].lstrip().rstrip())

def grab_title(fname, new_fname):
    title = orig_cap[fname]["headline"]["main"]
    if len(title) < 1 or title == None:
        title = orig_cap[orig]["headline"]["seo"]
    date_ = orig_cap[fname]["article_url"].split("www.nytimes.com/")[1][:10]
    d, m, y = int(date_.split("/")[2]), int(date_.split("/")[1]), int(date_.split("/")[0])
    date_ = date(day=d, month=m, year=y).strftime('%B %d %Y')
    title += "\n" + date_ + "\t NYTimes"
    with open(dest_dir+"/title/txt/"+new_fname+"_title.txt", "w") as f:
        f.write(title)

def get_fID_from_metadata(title, text):
    fID = ""
    for k, v in orig_cap.items():
        if text in v["article"].replace('"','').replace("'",""):
            fID = k
    if fID == "": 
        for k, v in orig_cap.items():
            if fID == "" and "main" in v["headline"]:
                if title == v["headline"]["main"]: 
                    fID = k
            if fID == "" and "print_headline" in v["headline"]:
                if title == v["headline"]["print_headline"]:
                    fID = k
    return fID

new_fname_list = list(range(2*32000))

for fname in os.listdir(orig_real_art_fpath):
    real_art = open(orig_real_art_fpath+fname).read()
    fname = fname.replace(".txt","")
    new_fname = str(random.choice(new_fname_list)) 
    new_fname_list.remove(int(new_fname))
    new_real_art = "\n".join(item for item in real_art.split('\n') if item)
    with open(dest_dir+"/txt/"+new_fname+".txt", "w") as f:
        f.write(new_real_art)
    caption_indices = list(orig_cap[fname]["images"].keys())
    for i, orig_cap_idx in enumerate(caption_indices):
        if i > 2:
            continue
        grab_img_cap(fname, new_fname, str(i), orig_cap_idx)
    grab_title(fname, new_fname)
    train_val_test_split = random.choices(["train","val","test"],weights=[0.6,0.2,0.2])[0] \
                                          if train_val_test_split_mode == "online" else \
                                          old_mapping[old_mapping[1] == fname][3].values[0]
    mapping.append([new_fname, fname, 0, train_val_test_split]) 

with open(orig_fake_art_fpath, 'r') as json_file:
    head_out_test_fnames = list(filter(lambda x: x.replace(".rsd.txt",""), os.listdir(orig_fake_art_test_fpath)))
    for idx, fake_art in enumerate(list(json_file)):
        train_val_test_split = random.choices(["train","val","test"],weights=[0.6,0.2,0.2])[0] \
                                              if train_val_test_split_mode == "online" else \
                                              old_mapping[old_mapping[1] == fname][3].values[0]
        fake_art = json.loads(fake_art)
        fname = get_fID_from_metadata(fake_art["title"],fake_art["text"])
        if train_val_test_split == "test": 
            fname = random.choice(head_out_test_fnames)
            head_out_test_fnames.remove(fname)
            fake_art = open(orig_fake_art_test_fpath+fname+".txt").read()
        new_fname = str(random.choice(new_fname_list)) 
        new_fname_list.remove(int(new_fname))
        with open(dest_dir+"/txt/"+new_fname+".txt", "w") as f:
            f.write(fake_art["gens_article"][0])
        caption_indices = list(orig_cap[fname]["images"].keys())
        for i, orig_cap_idx in enumerate(caption_indices):
            if i > 2:
                continue
            grab_img_cap(fname, new_fname, str(i), orig_cap_idx)
        grab_title(fname, new_fname)
        mapping.append([new_fname, fname, 1, train_val_test_split]) 

with open(dest_dir+"/mapping.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(mapping)
