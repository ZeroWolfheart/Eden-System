import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "/home/matrix-code/Escritorio/EdenSystem/frutas/annots"
CLUSTERS = 12

def load_dataset(path, max_dim):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        print(xml_file)
        tree = ET.parse(xml_file)
        
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        
        max_img = max(height,width)
        escala = max_dim/max_img
        height = round(height*escala)
        width  = round(width*escala)

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) #/ width
            ymin = int(obj.findtext("bndbox/ymin")) #/ height
            xmax = int(obj.findtext("bndbox/xmax")) #/ width
            ymax = int(obj.findtext("bndbox/ymax")) #/ height
            
            xmax = round(xmax*escala) /width
            xmin = round(xmin*escala) /width
            ymax = round(ymax*escala) /height
            ymin = round(ymin*escala) /height
            
            # print([xmax, xmin, ymax, ymin, height, width])
            dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)

max_dim = 200
data = load_dataset(ANNOTATIONS_PATH, max_dim)
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
