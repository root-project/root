# -*- coding: utf-8 -*-
import cppyy
cppyy.gbl.gROOT.ProcessLine('.L root_6023.h+')

c = cppyy.gbl.instance()

a = c.data
b = c.value()
