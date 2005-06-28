#
# author  Valeri Fine (fine@bnl.gov)
#-------------------------------------------------------------------------
# Qmake include file to add the rules to create RootCint Dictionary
#-------------------------------------------------------------------------
#
# $Id: rootlibs.pri,v 1.3 2005/06/17 00:56:19 fine Exp $
#
# Copyright (C) 2002 by Valeri Fine.  All rights reserved.
#
# This file may be distributed under the terms of the Q Public License
# as defined by Trolltech AS of Norway and appearing in the file
# LICENSE.QPL included in the packaging of this file.
#-------------------------------------------------------------------------
#
# Usage: (aux file for rootcint.pri qmake include file)
# -----
#        Defines the qmake LIBS, INCLUDEPATH, and  DEPENDPATH variables to link 
#        some the custom shared library against of the manadatory ROOT libraries
#
#        This is the "private" qmake include file. 
#        It should NOT be used alone. 
#
#        USE "rootcint.pri" instead
#-------------------------------------------------------------------------------------
# 1. Include this file into your project with QMAKE include statement:
#
#    !exists ($(ROOTSYS)/include/rootlibs.pri){
#        message "The rootlibs.pri was not found"
#    }
#    exists ($(ROOTSYS)/include/rootlibs.pri){
#       include ($(ROOTSYS)/include/rootlibs.pri)
#    }
#

#-- permanent components to be included into any ".pro" file to build the RootCint dictionary

win32 {
   LIBS	+=                                                                                                \
      -include:_G__cpp_setupG__Hist       -include:_G__cpp_setupG__Graf1  -include:_G__cpp_setupG__G3D     \
      -include:_G__cpp_setupG__GPad       -include:_G__cpp_setupG__Tree   -include:_G__cpp_setupG__Rint    \
      -include:_G__cpp_setupG__PostScript -include:_G__cpp_setupG__Matrix -include:_G__cpp_setupG__Physics \
      -include:_G__cpp_setupG__Gui1       -include:_G__cpp_setupG__Geom1   
    
   exists( $(ROOTSYS)/lib/libTable.lib ) {
      LIBS	+= -include:_G__cpp_setupG__Table
   }   

   exists( $(ROOTSYS)/lib/libQtGui.lib ) {
      LIBS	+=  -include:_G__cpp_setupG__QtGUI     
   }   
   
   LIBS	+=                                                                                              \
    "$(ROOTSYS)/lib/libCore.lib"   "$(ROOTSYS)/lib/libCint.lib"     "$(ROOTSYS)/lib/libHist.lib"         \
    "$(ROOTSYS)/lib/libGraf.lib"   "$(ROOTSYS)/lib/libGraf3d.lib"   "$(ROOTSYS)/lib/libGpad.lib"         \
    "$(ROOTSYS)/lib/libTree.lib"   "$(ROOTSYS)/lib/libRint.lib"     "$(ROOTSYS)/lib/libPostscript.lib"   \
    "$(ROOTSYS)/lib/libMatrix.lib" "$(ROOTSYS)/lib/libPhysics.lib"  "$(ROOTSYS)/lib/libGui.lib"          \
    "$(ROOTSYS)/lib/libGeom.lib"   "$(ROOTSYS)/lib/libTable.lib"                                         \
    "$(ROOTSYS)/lib/libGQt.lib"   
    
   exists( $(ROOTSYS)/lib/libTable.lib ) {
      LIBS	+=  "$(ROOTSYS)/lib/libTable.lib"
   }   
    
   exists( $(ROOTSYS)/lib/libQtGui.so ) {
      LIBS	+=  "$(ROOTSYS)/lib/libQtGui.lib"
   }   
}

mac {
   CONFIG +=  no_smart_library_merge
   LIBS	   += $$system(root-config --glibs) 
    
    exists( $(ROOTSYS)/lib/libTable.so ) {
        LIBS	+= -lTable    
    }   

    exists( $(ROOTSYS)/lib/libQtGui.so ) {
        LIBS	+=  -u _G__cpp_setupG__QtGUI     
    }   
   
    LIBS	   +=  -lGQt 
    exists( $(ROOTSYS)/lib/libQtGui.so ) {
        LIBS	+=  -lQtGui   
    }   
}

unix {
    LIBS	+= $$system(root-config --glibs) -lGQt 
    
    exists( $(ROOTSYS)/lib/libTable.so ) {
        LIBS	+= -lTable    
    }   
    
    LIBS += -lGQt 
    
    exists( $(ROOTSYS)/lib/libQtGui.so ) {
          LIBS	+=  -lQtGui  
          message ( "Found Qt extensions library !!!") 
      }   
}
