# origin_path="/media/disk2/lja/IMG_code/IMG/data/dataset/charades/charades_sta_test_qid.txt"
# origin_am_path="/media/disk2/lja/IMG_code/IMG/data/dataset/charades/charades_audiomatter.txt"
# dict_find={}
# with open(origin_path,"r") as reader:
#     l=reader.readlines()
#     for i in l:
#         qid=i.split("##")[1]
#         video_text=i.split("##")[0]+" "+i.split("##")[2]
#         dict_find[video_text]=qid

# new_l=[]
# with open(origin_am_path,"r") as reader:
#     l=reader.readlines()
#     for i in l:
#         video_text=i.split("##")[0]+" "+i.split("##")[1]
#         new_l.append("##".join([i.split("##")[0],dict_find[video_text],i.split("##")[1]]))
# with open("charades_audiomatte_qid.txt","w") as writer:
#     writer.writelines(new_l)
# import json
# data=[]
# new_data=[]
origin_am_path="/media/disk2/lja/IMG/data/dataset/charades/charades_audiomatter.txt"
vt={}
with open(origin_am_path,"r") as reader:
    l=reader.readlines()
    for i in l:
        video_text=i.split(" ")[0]+" "+i.split("##")[1].strip()
        if video_text in vt:
            print(video_text)
        else:
            vt[video_text]=1
# with open("/media/disk2/lja/IMG_code/IMG/data/dataset/charades/charades_sta_test_tvr_format.jsonl","r") as f:
#     for line in f:
#         data.append(json.loads(line))
#     for i in data:
#         video_text=i['vid']+" "+i['query']
#         if video_text not in vt:
#             continue
#         else:
#             new_data.append(i)
# with open("charades_audiomatterhao_tvr_format.json", 'w', encoding='utf-8') as f:
#     for item in new_data:
#         f.write(json.dumps(item, ensure_ascii=False) + '\n')


