#              $Id: GNUmakefile,v 1.3 2010/07/30 03:18:34 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdBwm.cc  \
        XrdBwmConfig.cc \
        XrdBwmHandle.cc \
        XrdBwmLogger.cc \
        XrdBwmPolicy1.cc

OBJECTS  = \
        $(OBJDIR)/XrdBwm.o  \
        $(OBJDIR)/XrdBwmConfig.o \
        $(OBJDIR)/XrdBwmHandle.o \
        $(OBJDIR)/XrdBwmLogger.o \
        $(OBJDIR)/XrdBwmPolicy1.o

MORELIBS = -L$(LIBDIR) -lXrdAcc -lXrdNet -lXrdOuc -lXrdSys

LIBDEP   = $(LIBDIR)/libXrdAcc.a \
           $(LIBDIR)/libXrdOuc.a \
           $(LIBDIR)/libXrdSys.a

LIBARCH  = $(LIBDIR)/libXrdBwm.a
LIBRARY  = $(LIBDIR)/libXrdBwm.$(TYPESHLIB)

TARGETS  = $(LIBARCH) $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet% ../XrdNet
vpath XrdBwm% ../XrdBwm
vpath XrdOuc% ../XrdOuc
vpath XrdAcc% ../XrdAcc
vpath XrdSec% ../XrdSec
vpath XrdSfs% ../XrdSfs
vpath XrdSys% ../XrdSys

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdBwm done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS) $(OBJFS) $(LIBDEP)
	@echo Creating archive $(LIBARCH)
	$(ECHO)rm -f $(LIBARCH)
	$(ECHO)ar -rc $(LIBARCH) $(OBJECTS)
	$(ECHO)ranlib $(LIBARCH)

	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTS) $(OBJFS) $(LDSO) $(MORELIBS) $(LIBS)  -o $(LIBRARY)

$(OBJDIR)/XrdBwm.o:  XrdBwm.cc           XrdBwm.hh \
                     XrdAccAuthorize.hh \
                     XrdBwmHandle.hh     XrdBwmTrace.hh \
                     XrdNetDNS.hh \
                     XrdOucEnv.hh        XrdOucTrace.hh \
                     XrdSecEntity.hh \
                     XrdSfsAio.hh        XrdSfsInterface.hh \
                     XrdSysError.hh      XrdSysLogger.hh    XrdSysPlatform.hh \
                     XrdSysPthread.hh    XrdSysHeaders.hh \
                  ../XrdVersion.hh
	@echo Compiling XrdBwm.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdBwm.o XrdBwm.cc

$(OBJDIR)/XrdBwmConfig.o:  XrdBwmConfig.cc \
                           XrdAccAuthorize.hh \
                           XrdBwm.hh          XrdBwmLogger.hh  XrdBwmPolicy.hh\
                           XrdBwmPolicy1.hh   XrdBwmTrace.hh \
                           XrdOuca2x.hh       XrdOucEnv.hh     XrdOucStream.hh\
                           XrdOucTrace.hh \
                           XrdSysError.hh     XrdSysHeaders.hh XrdSysPlugin.hh
	@echo Compiling XrdBwmConfig.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdBwmConfig.o XrdBwmConfig.cc

$(OBJDIR)/XrdBwmHandle.o: XrdBwmHandle.cc   XrdBwmHandle.hh \
                          XrdBwmLogger.hh   XrdBwmPolicy.hh   XrdBwmTrace.hh \
                          XrdOucErrInfo.hh  \
                          XrdSfsInterface.hh \
                          XrdSysError.hh    XrdSysPlatform.hh  XrdSysPthread.hh\
                          ../XProtocol/XProtocol.hh
	@echo Compiling XrdBwmHandle.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdBwmHandle.o XrdBwmHandle.cc

$(OBJDIR)/XrdBwmLogger.o: XrdBwmLogger.cc   XrdBwmLogger.hh \
                          XrdNetOpts.hh     XrdNetSocket.hh \
                          XrdOucProg.hh     XrdOucStream.hh \
                          XrdSysError.hh    XrdSysPlatform.hh \
                          XrdSysPthread.hh
	@echo Compiling XrdBwmLogger.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdBwmLogger.o XrdBwmLogger.cc

$(OBJDIR)/XrdBwmPolicy1.o: XrdBwmPolicy1.cc   XrdBwmPolicy1.hh \
                           XrdSysPthread.hh
	@echo Compiling XrdBwmPolicy1.cc 
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdBwmPolicy1.o XrdBwmPolicy1.cc
