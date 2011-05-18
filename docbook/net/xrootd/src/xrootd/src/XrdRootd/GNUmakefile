#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

LFLAGS  = -L$(LIBDIR) -lXrdOuc $(LDSO)

LIBDEPS = $(LIBDIR)/libXrdOuc.a

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdRootdProtocol.cc

  
OBJECTS = \
        $(OBJDIR)/XrdRootdProtocol.o

LIBRARY = $(LIBDIR)/libXrdRootd.$(TYPESHLIB)

TARGETS = $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                           -
#------------------------------------------------------------------------------#

vpath XrdSys% ../XrdSys
vpath XrdOuc% ../XrdOuc
vpath  Xrd%   ../Xrd
  
#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdRootd done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(LIBDEPS)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(OBJECTS) $(LFLAGS) -o $(LIBRARY)

$(OBJDIR)/XrdRootdProtocol.o: XrdRootdProtocol.hh XrdLink.hh \
                              XrdScheduler.hh     XrdTrace.hh \
                              XrdSysError.hh      XrdOucTrace.hh \
                              XrdRootdProtocol.cc
	@echo Compiling XrdRootdProtocol.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdRootdProtocol.o XrdRootdProtocol.cc
