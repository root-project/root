#              $Id: GNUmakefile,v 1.3 2009/09/01 09:53:25 furano Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

MORELIBS = -L$(LIBDIR) -lXrdOuc -lXrdSys

LIBDEPS = $(LIBDIR)/libXrdOuc.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdSecProtocolunix.cc

OBJECTS = \
        $(OBJDIR)/XrdSecProtocolunix.o

TARGETS = $(LIBDIR)/libXrdSecunix.$(TYPESHLIB)

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
	@echo Make XrdSecunix done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(LIBDEPS)
	@echo Creating shared library $(TARGETS) 
	$(ECHO)$(CC) $(OBJECTS) $(LDSO) $(MORELIBS) $(LIBS) -o $(TARGETS);

$(OBJDIR)/XrdSecProtocolunix.o: XrdSecProtocolunix.cc XrdSecInterface.hh \
                                XrdSysHeaders.hh \
                                XrdSecEntity.hh       XrdOucErrInfo.hh
	@echo Compiling XrdSecProtocolunix.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdSecProtocolunix.o XrdSecProtocolunix.cc
