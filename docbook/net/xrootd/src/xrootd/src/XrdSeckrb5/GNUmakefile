#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

MORELIBS = -L$(LIBDIR) -lXrdSut -lXrdNet -lXrdOuc -lXrdNetUtil -lXrdSys \
             $(LIBKRB5)

LIBDEPS = $(LIBDIR)/libXrdSut.a     $(LIBDIR)/libXrdNet.a \
          $(LIBDIR)/libXrdNetUtil.a $(LIBDIR)/libXrdOuc.a \
          $(LIBDIR)/libXrdSys.a 

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
  
SOURCES = \
        XrdSecProtocolkrb5.cc

OBJECTS = \
        $(OBJDIR)/XrdSecProtocolkrb5.o

TARGETS = $(LIBDIR)/libXrdSeckrb5.$(TYPESHLIB)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdOuc% ../XrdOuc
vpath XrdSec% ../XrdSec
vpath XrdSys% ../XrdSys
vpath XrdSut% ../XrdSut

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdSeckrb5 done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(LIBDEPS)
	@echo Creating shared library $(TARGETS) 
	$(ECHO)$(CC) $(OBJECTS) $(LDSO) $(MORELIBS) $(LIBS) -o $(TARGETS);

$(OBJDIR)/XrdSecProtocolkrb5.o: XrdSecProtocolkrb5.cc XrdSecInterface.hh \
                                XrdSysPthread.hh      XrdOucTokenizer.hh \
                                XrdOucErrInfo.hh      XrdSysHeaders.hh
	@echo Compiling XrdSecProtocolkrb5.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) $(INCKRB5) -o $(OBJDIR)/XrdSecProtocolkrb5.o XrdSecProtocolkrb5.cc
