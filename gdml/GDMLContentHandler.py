#!/usr/bin/env python2.3
# -*- Mode: Python -*-
#
import processes
import xml.sax

# This class is an implementation of SAX ContentHandler for parsing GDML files.
# xml.sax.parse method should be called with an instance of this class as
# the second argument (the first argument of xml.sax.parse method should be
# the name of the file to be parsed).  

# The constructor of this class requires a 'binding' as the argument.
# The 'binding' is an application-specific mapping of GDML elements (materials,
# solids, etc) to specific objects which should be instanciated by the converted.
# In the present case (ROOT) the binding is implemented in the ROOTBinding module.
# This class requires 'processes' module where appropriate methods for all the
# allowed GDML elements are implemented.

# Apart from the standard method required by the ContentHandler (startElement and
# endElement), this class implements also two additional (GDML specific) methods.
# The WordVolume method allows to access the pointer to the world volume once the
# geometry file has been parsed.
# The AuxiliaryData method allows to access the map (volume, data) of
# any potential (optional) auxiliary data (like colour attributes, sensitive detectors, etc)
# associated to specific volumes.

# For any question or remarks concerning this code, please send an email to
# Witold.Pokorski@cern.ch.

class GDMLContentHandler(xml.sax.ContentHandler):
    def __init__(self, binding):
        self.stack = []
        self.proc = processes.processes(binding)
    
    def startElement(self,name,attrs):
        self.stack.append([name,attrs,[]])
    
    def endElement(self,name):
        elem = self.stack.pop()

        if len(self.stack):
            self.stack[-1][2].append(elem)

        if self.proc.gdmlel_dict.has_key(elem[0]):
            self.proc.gdmlel_dict[name](self.proc,elem)

    def WorldVolume(self):
        return self.proc.world
            
    def AuxiliaryData(self):
        return self.proc.auxmap
