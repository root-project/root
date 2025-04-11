# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # for common

from common import check_cppyy_backend
check_cppyy_backend()

import ROOT
ROOT.gROOT.ProcessLine('.L root_6023.h+')

c = ROOT.instance()

a = c.data
b = c.value()
