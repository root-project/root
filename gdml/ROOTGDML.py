from math import *
from units import *

import ROOT
import writer
import ROOTwriter

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
gdmlwriter.addSetup('default', '1.0', topV.GetName())
binding.examineVol(topV)

# write file
gdmlwriter.writeFile()


