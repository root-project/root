#
# author  Valeri Fine (fine@bnl.gov)
#-------------------------------------------------------------------------
# Qmake include file to add the rules to create RootCint Dictionary
#-------------------------------------------------------------------------
#
# $Id: rootlibs.pri,v 1.28 2008/09/25 22:19:24 fine Exp $
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

# QMake must be defined by Qmake alone but ... It is not :(  Sept 6, 2005 V.Fine)


# define whether the current QMake is from Qt4 distribution

MYVERSION = $$[QMAKE_VERSION] 
ISQT4 = $$find(MYVERSION, ^[2-9])


unix {
  QMAKE_EXTENSION_SHLIB = so
}

mac {
#  I dropped using .dylib as extension with 10.5 as MacOS X now allows .so . Fons. 
  QMAKE_EXTENSION_SHLIB = dylib
  CONFIG += no_smart_library_merge
}

win32 {
  QMAKE_EXTENSION_SHLIB = dll
}

#-- permanent components to be included into any ".pro" file to build the RootCint dictionary

win32 {
   LIBS	+=                                                                                                 \
      -include:_G__cpp_setupG__Hist       -include:_G__cpp_setupG__G3D                                     \
      -include:_G__cpp_setupG__GPad       -include:_G__cpp_setupG__Tree   -include:_G__cpp_setupG__Rint    \
      -include:_G__cpp_setupG__PostScript -include:_G__cpp_setupG__Matrix -include:_G__cpp_setupG__Physics \
      -include:_G__cpp_setupG__Gui1       -include:_G__cpp_setupG__Geom1 
    
   exists( $$(ROOTSYS)/lib/libRIO.lib ) {
      LIBS	+= -include:_G__cpp_setupG__IO
   }   

   exists( $$(ROOTSYS)/lib/libTable.lib ) {
      LIBS	+= -include:_G__cpp_setupG__Table
   }   

   exists( $%(ROOTSYS)/lib/libQtRootGui.lib ) {
      LIBS	+=  -include:_G__cpp_setupG__QtGUI     
   }   
   
   LIBS	+=                                                                                               \
    "$(ROOTSYS)/lib/libCore.lib"   "$(ROOTSYS)/lib/libCint.lib"     "$(ROOTSYS)/lib/libHist.lib"         \
    "$(ROOTSYS)/lib/libGraf.lib"   "$(ROOTSYS)/lib/libGraf3d.lib"   "$(ROOTSYS)/lib/libGpad.lib"         \
    "$(ROOTSYS)/lib/libTree.lib"   "$(ROOTSYS)/lib/libRint.lib"     "$(ROOTSYS)/lib/libPostscript.lib"   \
    "$(ROOTSYS)/lib/libMatrix.lib" "$(ROOTSYS)/lib/libPhysics.lib"  "$(ROOTSYS)/lib/libGui.lib"          \
    "$(ROOTSYS)/lib/libGeom.lib"   "$(ROOTSYS)/lib/libTable.lib"                                         \
    "$(ROOTSYS)/lib/libGQt.lib"   
    
   exists( $$(ROOTSYS)/lib/libRIO.lib ) {
      LIBS	+= "$(ROOTSYS)/lib/libRIO.lib" 
   }   

   exists( $$(ROOTSYS)/lib/libTable.lib ) {
      LIBS	+=  "$(ROOTSYS)/lib/libTable.lib"
   }   
       
   exists( $$(ROOTSYS)/lib/libQtRootGui.lib ) {
      LIBS	+=  "$(ROOTSYS)/lib/libQtRootGui.lib"
   }   
}

unix {
    libFile = $$(QTROOTSYSDIR)/lib/libGQt.$$QMAKE_EXTENSION_SHLIB
    exists ($$libFile ) {
        LIBS += -L$(QTROOTSYSDIR)/lib
        LIBS += -lGQt  
    }
    libFile = $$(QTROOTSYSDIR)/lib/libQtRootGui.$$QMAKE_EXTENSION_SHLIB
    exists ($$libFile ) {
        LIBS += -lQtRootGui
    }
    LIBS	+= $$system(${ROOTSYS}/bin/root-config --glibs)
    libFile = $$(ROOTSYS)/lib/libTable.$$QMAKE_EXTENSION_SHLIB 
    exists( $$libFile ) {
        LIBS += -lTable
    }   
    
    LIBS *= -lGQt 
    
    exists( $$(ROOTSYS)/lib/libQtRootGui.$$QMAKE_EXTENSION_SHLIB ) {
          LIBS	*=  -lQtRootGui  
          message ( "Found Qt extensions library !!!") 
    }
}
FORCELINKLIST	+=                                                                  \
        _G__cpp_setupG__Hist        _G__cpp_setupG__Graf1   _G__cpp_setupG__G3D     \
        _G__cpp_setupG__GPad        _G__cpp_setupG__Tree    _G__cpp_setupG__Rint    \
        _G__cpp_setupG__PostScript  _G__cpp_setupG__Matrix  _G__cpp_setupG__Physics \
        _G__cpp_setupG__Gui1        _G__cpp_setupG__Geom1   _G__cpp_setup_initG__IO 

mac {
  equals(TEMPLATE, app_fake) {
  # this trick does not work yet (To be fixed. V.Fine)
      LIBS	+=  $$join( FORCELINKLIST, " -u ")                                                                

      exists( $$(ROOTSYS)/lib/libTable.lib ) {
         LIBS	+= -u _G__cpp_setupG__Table
      }   

      exists( $$(ROOTSYS)/lib/libQtRootGui.lib ) {
         LIBS	+=  -u _G__cpp_setupG__QtGUI     
      }
  }
# -- trick to force the trivial symbolic link under UNIX

  equals(TEMPLATE, lib) {
     sharedso.target       = lib$${TARGET}.so 
     sharedso.commands     =  ( rm  -f  $(DESTDIR)$$sharedso.target; ln -s  lib$${TARGET}.$$QMAKE_EXTENSION_SHLIB $$sharedso.target; mv -f  $$sharedso.target $(DESTDIR) )

     QMAKE_EXTRA_UNIX_TARGETS += sharedso
     POST_TARGETDEPS          += $$sharedso.target
     QMAKE_CLEAN              += $$sharedso.target
  }
}
