import pickle

path = '/home/ubuntu/code/LowLight/mmdetexp/MaskRCNN600x400/Formal/r50_fpn_nothing_baseline_8cls40.8_12E/results.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
    # print(len(data))
    # exit()
    for i in data:
        print(len(i))
        
        # for j in i:
            # print(j)
        exit()
    # pass