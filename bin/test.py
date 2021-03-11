import sys
import os

sys.path.append(os.path.abspath(os.path.join('../src')))

from ..src.model import DodonaphyModel

def testFunc(v1, v2):
    print(v1)
    print(v2)
    # return DodonaphyModel(v1['D'],v1['L'],v1['S'])
    
