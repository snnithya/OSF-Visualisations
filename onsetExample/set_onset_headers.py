import pandas as pd
import os

'''
This script adds headers (originally missing) to the annotation files
'''
annotationFolder = 'Data/OSF/Annotations/'

for root, _, fileNames in os.walk(annotationFolder):
    for fileName in fileNames:
        df = pd.read_csv(os.path.join(root, fileName), header=None, names=
        ['Inst', 'Label'])
        df.to_csv(os.path.join(root, fileName), index=False)