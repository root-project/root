# -*- coding: utf-8 -*-
import cppyy
cppyy.gbl.gROOT.ProcessLine('.L root_6023.h+')

c = cppyy.gbl.instance()

print c.data
print c.value()
