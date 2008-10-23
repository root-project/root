#              $Id: GNUmakefile,v 1.7 2008/09/07 04:30:39 abh Exp $

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

INCLUDE = -I. -I.. -I/usr/local/src/CASTOR2 -I/usr/local/src/CASTOR2/h

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#

SOURCES  = \
        XrdCS2DCM.cc \
        XrdCS2DCM2cs.cc \
        XrdCS2DCMConfig.cc \
        XrdCS2DCMFile.cc \
        XrdCS2DCMMain.cc \
        XrdCS2Req.cc \
        XrdCS2Xmi.cc

OBJBINS  = \
        $(OBJDIR)/XrdCS2DCM.o \
        $(OBJDIR)/XrdCS2DCM2cs.o \
        $(OBJDIR)/XrdCS2DCMConfig.o \
        $(OBJDIR)/XrdCS2DCMFile.o \
        $(OBJDIR)/XrdCS2DCMMain.o

OBJECTS  = \
        $(OBJDIR)/XrdCS2Req.o \
        $(OBJDIR)/XrdCS2Xmi.o

SUNLIBS  = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdSys -lpthread -lnsl -lsocket -lrt
LNXLIBS  = -L$(LIBDIR) -lXrdNet -lXrdOuc -lXrdSys -lpthread -lnsl -lrt
CS2LIBS  = $(LNXLIBS) -LCASTOR2LIB -lshift

LIBRARY  = $(LIBDIR)/libXrdCS2.$(TYPESHLIB)

TARGETS   = libXrdCS2.so XrdCS2d

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

vpath XrdNet%  ../XrdNet
vpath XrdOlb%  ../XrdOlb
vpath XrdOuc%  ../XrdOuc
vpath XrdSys%  ../XrdSys
vpath Xrd%     ../Xrd

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdCS2 done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

libXrdCS2.so: $(OBJECTS) $(LIBDEP)
	@echo Creating shared library $(LIBRARY) 
	$(ECHO)$(CC) $(CFLAGS) $(OBJECTS) $(LDSO) $(CS2LIBS)  -o $(LIBRARY)

XrdCS2d: $(OBJBINS)
	@echo Creating executable $(BINDIR)/XrdCS2d
	$(ECHO)$(CC) $(CFLAGS) $(OBJBINS) $(CS2LIBS) -lXrd -o $(BINDIR)/XrdCS2d

$(OBJDIR)/XrdCS2DCM.o:     XrdCS2DCM.hh  XrdCS2DCM.cc  XrdCS2DCMFile.hh \
                           XrdSysError.hh   XrdSysHeaders.hh XrdSysPthread.hh \
                           XrdOucStream.hh  XrdOucUtils.hh \
                           XrdScheduler.hh  XrdTrace.hh
	@echo Compiling XrdCS2DCM.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCS2DCM.o XrdCS2DCM.cc

$(OBJDIR)/XrdCS2DCM2cs.o: XrdCS2DCM.hh  XrdCS2DCM2cs.cc \
                          XrdSysError.hh   XrdSysLogger.hh  XrdTrace.hh \
                          XrdSysHeaders.hh \
                          XrdSysTimer.hh   XrdSysPthread.hh XrdNetLink.hh
	@echo Compiling XrdCS2DCM2cs.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCS2DCM2cs.o XrdCS2DCM2cs.cc

$(OBJDIR)/XrdCS2DCMConfig.o: XrdCS2DCM.hh  XrdCS2DCMConfig.cc  XrdCS2DCMFile.hh \
                             XrdNetOpts.hh    XrdNetSocket.hh  XrdOucUtils.hh \
                             XrdSysError.hh   XrdSysLogger.hh  XrdSysTimer.hh \
                             XrdSysHeaders.hh XrdScheduler.hh  XrdTrace.hh    \
                             XrdNetLink.hh    XrdNetWork.hh
	@echo Compiling XrdCS2DCMConfig.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCS2DCMConfig.o XrdCS2DCMConfig.cc

$(OBJDIR)/XrdCS2DCMFile.o: XrdCS2DCM.hh   XrdCS2DCMFile.cc \
                           XrdSysError.hh XrdTrace.hh
	@echo Compiling XrdCS2DCMFile.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCS2DCMFile.o XrdCS2DCMFile.cc

$(OBJDIR)/XrdCS2DCMMain.o: XrdCS2DCM.hh XrdCS2DCMMain.cc \
                           XrdSysError.hh    XrdSysLogger.hh   XrdSysPthread.hh \
                           XrdSysHeaders.hh  XrdScheduler.hh   XrdTrace.hh
	@echo Compiling XrdCS2DCMMain.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCS2DCMMain.o XrdCS2DCMMain.cc

$(OBJDIR)/XrdCS2Req.o:     XrdCS2Req.cc      XrdCS2Req.hh    \
                           XrdOlbReq.hh      XrdOlbTrace.hh     XrdOlbXmi.hh \
                           XrdSysError.hh    XrdOucName2Name.hh \
                           XrdSysPlatform.hh
	@echo Compiling XrdCS2Req.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCS2Req.o XrdCS2Req.cc

$(OBJDIR)/XrdCS2Xmi.o:     XrdCS2Xmi.cc      XrdCS2Xmi.hh    \
                           XrdCS2Req.hh \
                           XrdScheduler.hh     XrdSysPlatform.hh \
                           XrdSysError.hh      XrdOucName2Name.hh \
                           XrdSysPthread.hh    XrdSysTimer.hh \
                           XrdOlbTrace.hh      XrdOlbTypes.hh \
                           XrdOlbReq.hh        XrdOlbXmi.hh
	@echo Compiling XrdCS2Xmi.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) -o $(OBJDIR)/XrdCS2Xmi.o XrdCS2Xmi.cc
