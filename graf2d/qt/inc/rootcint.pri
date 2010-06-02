#
# author  Valeri Fine (fine@bnl.gov)
#-------------------------------------------------------------------------
# Qmake include file to add the rules to create RootCint Dictionary
#-------------------------------------------------------------------------
#
# $Id: rootcint.pri,v 1.8 2009/03/22 04:58:00 fine Exp $
#
# Copyright (C) 2002 by Valeri Fine.  All rights reserved.
#
# This file may be distributed under the terms of the Q Public License
# as defined by Trolltech AS of Norway and appearing in the file
# LICENSE.QPL included in the packaging of this file.
#-------------------------------------------------------------------------
#
# Usage:
# -----
# To link against of the ROOT Qt layer and generate the RootCint dictionary with qmake 
#-------------------------------------------------------------------------------------
# 1. Include this file into your project with QMAKE inlcude statement:
#
#    !exists ($$(ROOTSYS)/include/rootcint.pri){
#        message "The Rootcint.pri was not found"
#    }
#    exists ($$(ROOTSYS)/include/rootcint.pri){
#       include ($$(ROOTSYS)/include/rootcint.pri)
#    }
#
# 2. Provide the list of the class header files followed by the appropriated LinkDef.f file
#    within your prpoject with CREATE_ROOT_DICT_FOR_CLASSES  QMAKE variable
# -----------------------------------------------
#
#   For example
#
#    . . . 
#    !exists ($$(ROOTSYS)/include/rootcint.pri){
#        message "The Rootcint.pri was not found"
#    }
#    exists ($$(ROOTSYS)/include/rootcint.pri){
#       include ($$(ROOTSYS)/include/rootcint.pri)
#       CREATE_ROOT_DICT_FOR_CLASSES  = ${HEADERS} MyParticle.h MyDetector.h MyEvent.h ShowerMain.h 
#       CREATE_ROOT_DICT_FOR_CLASSES *= ${HEADERS} RSLinkDef.h
#    }
#    . . . 
# -----------------------------------------------
#
# 3. Run "qmake"
# 4. Run "make"
#
# -----------------------------------------------

#-- permanent components to be included into any ".pro" file to build the RootCint dictionary
#

# -- define the LIBS and INCLUDEPATH variable
#
# -------  define rootlibs.pri -----------------
ROOTLIBSPRI = 
ROOTLIBSPRIFILE = rootlibs.pri
win32: DEFINES +=  _CRT_SECURE_NO_WARNINGS
# Wow !!! Qt exists function for the nested include file applies the relative path
exists ($$ROOTLIBSPRIFILE){
#  ROOTLIBSPRI = inc/$$ROOTLIBSPRIFILE
  ROOTLIBSPRI = $$ROOTLIBSPRIFILE
}

isEmpty(ROOTLIBSPRI) { 
  exists ($$(QTROOTSYSDIR)/include/$$ROOTLIBSPRIFILE){
     ROOTLIBSPRI = $$(QTROOTSYSDIR)/include/$$ROOTLIBSPRIFILE
  }
}

isEmpty(ROOTLIBSPRI) { 
  exists ($$(ROOTSYS)/include/$$ROOTLIBSPRIFILE){
     ROOTLIBSPRI = $$(ROOTSYS)/include/$$ROOTLIBSPRIFILE
  }
}

isEmpty(ROOTLIBSPRI){
    message ("The $$ROOTLIBSPRIFILE was not found")
}

!isEmpty(ROOTLIBSPRI){
   include ($$ROOTLIBSPRI)
}

#
# ----------  define rootcintrule.pri  -----------------
# -- define the RootCint ROOT dictionary building rule
#
ROOTCINTRULEPRI = 
ROOTCINTRULEPRIFILE = rootcintrule.pri

exists ($$ROOTCINTRULEPRIFILE){
     ROOTCINTRULEPRI = $$ROOTCINTRULEPRIFILE 
  }

isEmpty(ROOTCINTRULEPRI) { 
  exists ($$(QTROOTSYSDIR)/include/$$ROOTCINTRULEPRIFILE ){
     ROOTCINTRULEPRI = $$(QTROOTSYSDIR)/include/$$ROOTCINTRULEPRIFILE 
  }
}
isEmpty(ROOTCINTRULEPRI) { 
  exists ($$(ROOTSYS)/include/$$ROOTCINTRULEPRIFILE ){
     ROOTCINTRULEPRI = $$(ROOTSYS)/include/$$ROOTCINTRULEPRIFILE 
  }
}

isEmpty(ROOTCINTRULEPRI){
    message ("The $$ROOTCINTRULEPRIFILE was not found")
}
!isEmpty(ROOTCINTRULEPRI){
   include ($$ROOTCINTRULEPRI)
}
# ------------------ Mac OS settings -----------------------
macx|darwin-g++ {
  QMAKE_MACOSX_DEPLOYMENT_TARGET = $$system(/usr/bin/sw_vers -productVersion | cut -d. -f1-2)
  message( Configuring Qt for Mac OS $$QMAKE_MACOSX_DEPLOYMENT_TARGET build ! )
}
