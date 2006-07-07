# @(#)root/gdml:$Name:  $:$Id: ROOTGDML.py,v 1.2 2006/06/13 20:46:53 rdm Exp $
# Author: Witold Pokorski   05/06/2006

from math import *
from units import *

import ROOT
import writer
import ROOTwriter
import libPyROOT

# get TGeoManager and top volume
geomgr = ROOT.gGeoManager
topV = geomgr.GetTopVolume()

# instanciate writer
gdmlwriter = writer.writer('geo.gdml')
binding = ROOTwriter.ROOTwriter(gdmlwriter)

# dump materials
matlist = geomgr.GetListOfMaterials()
binding.dumpMaterials(matlist)

# dump solids
shapelist = geomgr.GetListOfShapes()
binding.dumpSolids(shapelist)

# dump geo tree
print 'Traversing geometry tree'
gdmlwriter.addSetup('default', '1.0', topV.GetName()+'_at_'+str(libPyROOT.AddressOf(topV)[0]))
binding.examineVol(topV)

# write file
gdmlwriter.writeFile()


