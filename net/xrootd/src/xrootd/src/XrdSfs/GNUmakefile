#              $Id: GNUmakefile,v 1.12 2010/07/30 03:19:14 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdSfsNative.cc

  
OBJECTS = \
        $(OBJDIR)/XrdSfsNative.o

LIBRARY =

TARGETS = OBJECTFILE

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdOuc% ../XrdOuc
vpath XrdSec% ../XrdSec
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdSfs done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS)

$(OBJDIR)/XrdSfsNative.o:  XrdSfsInterface.hh XrdSfsNative.hh  XrdSecInterface.hh \
                           XrdSysError.hh     XrdOucErrInfo.hh XrdSysLogger.hh \
                           XrdSysHeaders.hh \
                           XrdSfsNative.cc    XrdSfsAio.hh  ../XrdVersion.hh
	@echo Compiling XrdSfsNative.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSfsNative.o XrdSfsNative.cc
