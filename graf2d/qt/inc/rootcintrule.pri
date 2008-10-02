#
# author  Valeri Fine (fine@bnl.gov)
#-------------------------------------------------------------------------
# Qmake include file to add the rules to create RootCint Dictionary
#-------------------------------------------------------------------------
#
# $Id: rootcintrule.pri,v 1.20 2008/09/25 22:19:24 fine Exp $
#
# Copyright (C) 2002 by Valeri Fine.  All rights reserved.
#
# This file may be distributed under the terms of the Q Public License
# as defined by Trolltech AS of Norway and appearing in the file
# LICENSE.QPL included in the packaging of this file.
#-------------------------------------------------------------------------
#
##
# Usage: (aux file for rootcint.pri qmake include file)
# -----
#        Defines the qmake rule generate the RootCint dictionary with qmake
#
#        This is the "private" qmake include file. 
#        It should NOT be used alone. 
#
#        USE "rootcint.pri" instead
#-------------------------------------------------------------------------------------
# 1. Include this file into your project with QMAKE include statement:
#
#    !exists ($$(ROOTSYS)/include/rootcintrule.pri){
#        message "The rootcintrules.pri was not found"
#    }
#    exists ($$(ROOTSYS)/include/rootcintrule.pri){
#       include ($$(ROOTSYS)/include/rootcintrule.pri)
#    }
#
# 2. Provide the list of the class header files followed by the appropriated LinkDef.h file
#    within your prpoject with CREATE_ROOT_DICT_FOR_CLASSES  QMAKE variable
# -----------------------------------------------
#
#   For example
#
#    . . . 
#    !exists ($$(ROOTSYS)/include/rootcintrule.pri){
#        message "The rootcintrule.pri was not found"
#    }
#    exists ($$(ROOTSYS)/include/rootcintrule.pri){
#       include ($(ROOTSYS)/include/rootcintrule.pri)
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

# define whether the current QMake is from Qt4 distribution

MYVERSION = $$[QMAKE_VERSION] 
ISQT4 = $$find(MYVERSION, ^[2-9])

win32 {
  # Required to use dynamic_cast
   CONFIG *= rtti
}
include__File = $$(QTROOTSYSDIR)/include
exists ( $$include__File ){

  DEPENDPATH *= $(QTROOTSYSDIR)/include

  win32 {
     INCLUDEPATH	*= "%QTROOTSYSDIR%/include"
  }

  unix {
     INCLUDEPATH	*= "$(QTROOTSYSDIR)/include"
  }
}

include__File = $$(ROOTSYS)/include
exists ($$include__File){

  DEPENDPATH *= $(ROOTSYS)/include

  win32 {
     INCLUDEPATH	*= "%ROOTSYS%/include"
  }

  unix {
     INCLUDEPATH	*= "$(ROOTSYS)/include"
  }
}
!isEmpty( CREATE_ROOT_DICT_FOR_CLASSES ) {
   DICTDEFINES += -DQT_VERSION=0x30000
   QT_VERSION=$$[QT_VERSION]
   contains( QT_VERSION, "^4.*" ) {
     DICTDEFINES -= -DQT_VERSION=0x30000
     DICTDEFINES *= -DQT_VERSION=0x40000
   }

  ROOT_CINT_TARGET = $${TARGET}
  SOURCES	  *= $${ROOT_CINT_TARGET}Dict.cxx 

  rootcint.target       = $${ROOT_CINT_TARGET}Dict.cxx 

  win32:  rootcint.commands    +="$$(ROOTSYS)\bin\rootcint.exe"
  unix:   rootcint.commands    +=$(ROOTSYS)/bin/rootcint

  rootcint.commands    +=  -f $$rootcint.target  -c -p $$DICTDEFINES $(INCPATH) $$CREATE_ROOT_DICT_FOR_CLASSES
  rootcint.depends      = $$CREATE_ROOT_DICT_FOR_CLASSES
  
  rootcintecho.commands = @echo "Generating dictionary $$rootcint.target for $$CREATE_ROOT_DICT_FOR_CLASSES classes"

unix:   QMAKE_EXTRA_UNIX_TARGETS += rootcintecho rootcint 
win32:  QMAKE_EXTRA_WIN_TARGETS  += rootcintecho rootcint 

  QMAKE_CLEAN       +=  $${ROOT_CINT_TARGET}Dict.cxx $${ROOT_CINT_TARGET}Dict.h
}
