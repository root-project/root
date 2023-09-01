\defgroup Geometry_gdml GDML tools
\ingroup Geometry
\brief GDML tools for geometry classes

The `$ROOTSYS/gdml` directory contains a set of Python modules designed
for writing out Geometry Description Markup Language (GDML) files.
There is also a C++ implementation for the import of GDML into ROOT.
They act as a converter between the GDML geometry files and the TGeo
geometry structures (and vice versa).

### GDML->ROOT

As this binding is integrated into the ROOT installation, you need to
enable the use of the binding at the configure point of the ROOT
installation.  This can be done like so:

~~~ {.cpp}
./configure  --enable-gdml
~~~

On doing this the libraries will be built by issuing the standard ROOT
make command. The GDML to TGeo converter uses the TXMLEngine to parse
the GDML files. This XML parser is a DOM parser and returns the DOM
tree to the class TGDMLParse.  This class then interprets the GDML file
and adds the bindings in their TGeo equivalent.

The GDML schema is fully supported with a few exceptions:

  - Replica Volumes are not supported
  - Loops           are not supported
  - Matrices        are not supported

These will hopefully be added in the near future.

Once you have enabled GDML in the configure process for ROOT, to import
a GDML file, this can be done using TGeoManager::Import. This automatically
calls the right method to parse the GDML by detecting the .gdml file
extension. Here is how to do it:

~~~ {.cpp}
TGeoManager::Import("test.gdml");
~~~

Replace test.gdml with the gdml filename you want to import. Once the
GDML file has been successfully imported, you can view the geometry by
calling:

~~~ {.cpp}
gGeoManager->GetTopVolume()->Draw("ogl");
~~~

For any questions or comments about the GDML->ROOT binding please contact ben.lloyd@cern.ch


### ROOT->GDML

The TGeo to GDML converter allows to export ROOT geometries (TGeo
geometry trees) as GDML files. The writer module writes a GDML file
out of the 'in-memory' representation of the geometry. The actual
application-specific (ROOT) binding is implemented in ROOTwriter
module. It contains 'binding methods' for TGeo geometry classes which
can be exported in GDML format. Please refere to the comment part of
the ROOTwriter.py file for the list of presently supported TGeo
classes. The ROOTwriter class contains also three methods,
dumpMaterials, dumpSolids and examineVol which need to be called in
order to export materials, solids and geometry tree respectively.

The TGeo to GDML converter is now interfaced to the
TGeoManager::Export method which automatically calls the appropriate
Python scripts whenever the geometry output file has the .gdml
extension.

Alternatively, one can also use the ROOT->GDML converter directly from
the Python prompt (assuming the TGeo geometry has already been loaded
into memory in one or another way), for example:


~~~ {.cpp}
from math import *

import ROOT
import writer
import ROOTwriter

# get TGeoManager and
# get the top volume of the existing (in-memory) geometry tree
geomgr = ROOT.gGeoManager
topV = geomgr.GetTopVolume()

# instanciate writer
gdmlwriter = writer.writer('mygeo.gdml')
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
~~~

For all other functionality questions or comments, or even GDML in general,
please email Witold.Pokorski@cern.ch
