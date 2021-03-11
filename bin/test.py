import sys
import os
import rdata

# sys.path.append(os.path.abspath(os.path.join('../src')))

from model import DodonaphyModel

def testFunc():
    dat1 = rdata.conversion.convert(
        rdata.parser.parse_file("/home/azad/UTS-RF-Projects/pyDodonaphy/dphy_data.rda")
    )
    dat2 = rdata.conversion.convert(
        rdata.parser.parse_file("/home/azad/UTS-RF-Projects/pyDodonaphy/param_init.rda")
    )
    
    mymod = DodonaphyModel(int(dat1['dphy_dat']['D']),  int(dat1['dphy_dat']['L']), int(dat1['dphy_dat']['S']))
    mymod.learn(dat1['dphy_dat'], dat2['param_init'])

    
testFunc()