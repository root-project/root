#!/usr/bin/env python2.3
# -*- Mode: Python -*-
#
import sys
import xml.sax
import ROOT
import ROOTBinding
import GDMLContentHandler

ROOT.gSystem.Load("libGeom")

gdmlhandler = GDMLContentHandler.GDMLContentHandler(ROOTBinding.ROOTBinding())

filename = 'test.gdml'
if sys.argv.__len__() > 1:
    filename = sys.argv[1]
    
xml.sax.parse(filename, gdmlhandler)
geomgr = ROOT.gGeoManager

geomgr.SetTopVolume(gdmlhandler.WorldVolume())
geomgr.CloseGeometry()
geomgr.DefaultColors()

gdmlhandler.WorldVolume().Draw("ogl")



