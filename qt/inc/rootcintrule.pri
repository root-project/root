#
# author  Valeri Fine (fine@bnl.gov)
#-------------------------------------------------------------------------
# Qmake include file to add the rules to create RootCint Dictionary
#-------------------------------------------------------------------------
#
# $Id: rootcintrule.pri,v 1.6 2005/06/25 23:32:02 fine Exp $
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
#    !exists ($(ROOTSYS)/include/rootcintrule.pri){
#        message "The rootcintrules.pri was not found"
#    }
#    exists ($(ROOTSYS)/include/rootcintrule.pri){
#       include ($(ROOTSYS)/include/rootcintrule.pri)
#    }
#
# 2. Provide the list of the class header files followed by the appropriated LinkDef.h file
#    within your prpoject with CREATE_ROOT_DICT_FOR_CLASSES  QMAKE variable
# -----------------------------------------------
#
#   For example
#
#    . . . 
#    !exists ($(ROOTSYS)/include/rootcintrule.pri){
#        message "The rootcintrule.pri was not found"
#    }
#    exists ($(ROOTSYS)/include/rootcintrule.pri){
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

exists ($(ROOTSYS)/include){

  DEPENDPATH *= $(ROOTSYS)/include

  win32 {
     INCLUDEPATH	*= "%ROOTSYS%/include"
  }

  mac {
     INCLUDEPATH	*= "$(ROOTSYS)/include" 
  }

  unix {
     INCLUDEPATH	*= "$(ROOTSYS)/include"
  }
}
!isEmpty( CREATE_ROOT_DICT_FOR_CLASSES ) {
  SOURCES	  *= ${QMAKE_TARGET}Dict.cxx 

  rootcint.target       = ${QMAKE_TARGET}Dict.cxx 
  rootcint.commands    += $(ROOTSYS)/bin/rootcint -f $$rootcint.target  -c $(CXXFLAGS) $$CREATE_ROOT_DICT_FOR_CLASSES
  rootcint.depends      = $$CREATE_ROOT_DICT_FOR_CLASSES
  
  rootcintecho.commands = @echo "Generating dictionary $$rootcint.target for $$CREATE_ROOT_DICT_FOR_CLASSES classes"

  QMAKE_EXTRA_UNIX_TARGETS += rootcintecho rootcint 

  QMAKE_CLEAN       +=  ${QMAKE_TARGET}Dict.cxx ${QMAKE_TARGET}Dict.h
}
