#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 09:21:55 2021

@author: 151569
"""

import numpy as np
from dodonaphy.utils import utilFunc
from pytest import approx

def test_2d():
    
    D = np.ones((3,3),float)
    np.fill_diagonal(D, 0.0)
    dim=2
    emm = utilFunc.hydra(D, dim)
    
    # Compare to output of hydra in r 
    assert emm['curvature'] == approx(1)
    assert emm['dim'] == approx(dim)
    assert emm['directional'] == approx(np.array([
        [0.8660254038, -0.5000000000],
        [-0.8660254038, -0.5000000000],
        [0.0000000000, 1.0000000000]
        ]))
    assert emm['r'] == approx(np.array([0.2182178902, 0.2182178902, 0.2182178902]))
    assert emm['theta'] == approx(np.array([ -0.5235987756, -2.6179938780, 1.5707963268]))

def test_3d():
    
    D = np.array([[0,1,2.5,3,1], [1,0,2.2,1,2], [2.5,2.2,0,3,1], [3,1,3,0,2], [1,2,1,2,0]])
    dim=3
    emm = utilFunc.hydra(D, dim, equi_adj = 0)
    
    # Compare to output of hydra in r 
    # NB: some directions reversed from r due to opposite eigenvectors
    assert emm['curvature'] == approx(1)
    assert emm['dim'] == approx(dim)
    assert emm['r'] == approx(np.array([0.6274604254, 0.2705702432, 0.6461609880, 0.6779027826, 0.2182178902]))
    assert emm['directional'] == approx(np.array([
    [0.0821320214, -0.7981301820, 0.5968605730],
    [-0.5711268681, -0.6535591501, -0.4966634050],
    [-0.2070516064, 0.6620783581, 0.7202651457],
    [0.0811501819, 0.1418186867, -0.9865607473],
    [0.8008637619, 0.2731045145, 0.5329457375]
    ]))