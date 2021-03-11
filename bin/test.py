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
    # print(v1['dphy_dat']['D'])
    # print(v1['dphy_dat']['L'])
    # print(v1['dphy_dat']['S'])
    # print(dat2['param_init']["leaf_r"])
    
    mymod = DodonaphyModel(int(dat1['dphy_dat']['D']),  int(dat1['dphy_dat']['L']), int(dat1['dphy_dat']['S']))
    # print(mymod.generatedQuantities["blens"])
    mymod.learn(dat1['dphy_dat'], dat2['param_init'])


    # result = pyreadr.read_r("~/UTS-RF-Projects/pyDodonaphy/dphy_data.Rds")
    # print(result.keys())
    # df1 = result[None]
    # v1 = rdata.conversion.convert(v1)
    # v2 = rdata.parser.parse_file("~/UTS-RF-Projects/pyDodonaphy/initint.rda")
    # v2 = rdata.conversion.convert(v1)
    # print(v1['D'])
    # print(v2)
    # return DodonaphyModel(v1['D'],v1['L'],v1['S'])
    # s = open('/home/azad/UTS-RF-Projects/pyDodonaphy/dphy_dat.dic','r').read()
    # dat = ast.literal_eval(s)
    # print(dat)
    
testFunc()