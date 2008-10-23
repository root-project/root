#              $Id: GNUmakefile,v 1.12 2008/09/07 04:39:10 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

MORELIBS = -L$(LIBDIR) -lXrdOuc $(LIBKRB4)

LIBDEPS = $(LIBDIR)/libXrdOuc.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdSecProtocolkrb4.cc

OBJECTS = \
        $(OBJDIR)/XrdSecProtocolkrb4.o

TARGETS = $(LIBDIR)/libXrdSeckrb4.$(TYPESHLIB)

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
	@echo Make XrdSeckrb4 done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(LIBDEPS)
	@echo Creating shared library $(TARGETS) 
	$(ECHO)$(CC) $(OBJECTS) $(LDSO) $(MORELIBS) $(LIBS) -o $(TARGETS);

$(OBJDIR)/XrdSecProtocolkrb4.o: XrdSecProtocolkrb4.cc XrdSecInterface.hh \
                                XrdSysPthread.hh      XrdOucTokenizer.hh \
                                XrdOucErrInfo.hh      XrdSysHeaders.hh
	@echo Compiling XrdSecProtocolkrb4.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) $(INCKRB4) -o $(OBJDIR)/XrdSecProtocolkrb4.o XrdSecProtocolkrb4.cc
