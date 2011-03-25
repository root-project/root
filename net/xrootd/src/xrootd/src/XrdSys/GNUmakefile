#              $Id$

#------------------------------------------------------------------------------#
#                       C o m m o n   V a r i a b l e s                        #
#------------------------------------------------------------------------------#
  
include ../GNUmake.env

#------------------------------------------------------------------------------#
#                             C o m p o n e n t s                              #
#------------------------------------------------------------------------------#
SOURCES = \
        XrdSysDir.cc \
        XrdSysError.cc \
        XrdSysFAttr.cc \
        XrdSysLogger.cc \
        XrdSysPlatform.cc \
        XrdSysPlugin.cc   \
        XrdSysPriv.cc     \
        XrdSysPthread.cc     \
        XrdSysTimer.cc       \
        XrdSysXSLock.cc

OBJECTS = \
        $(OBJDIR)/XrdSysDir.o \
        $(OBJDIR)/XrdSysError.o \
        $(OBJDIR)/XrdSysFAttr.o \
        $(OBJDIR)/XrdSysLogger.o \
        $(OBJDIR)/XrdSysPlatform.o \
        $(OBJDIR)/XrdSysPlugin.o       \
        $(OBJDIR)/XrdSysPriv.o    \
        $(OBJDIR)/XrdSysPthread.o     \
        $(OBJDIR)/XrdSysTimer.o        \
        $(OBJDIR)/XrdSysXSLock.o

LIBRARY = $(LIBDIR)/libXrdSys.a

TARGETS = $(LIBRARY)

#------------------------------------------------------------------------------#
#                           S e a r c h   P a t h s                            #
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#                          I n i t i a l   R u l e s                           #
#------------------------------------------------------------------------------#
 
include ../GNUmake.options

anything: $(TARGETS)
	@echo Make XrdSys done.

#------------------------------------------------------------------------------#
#                           D e p e n d e n c i e s                            #
#------------------------------------------------------------------------------#

$(TARGETS): $(OBJECTS)
	@echo Creating archive $(LIBRARY) 
	$(ECHO)rm -f $(LIBRARY)
	$(ECHO)ar -rc $(LIBRARY) $(OBJECTS)
	$(ECHO)ranlib $(LIBRARY)

$(OBJDIR)/XrdSysDir.o: XrdSysDir.cc  XrdSysDir.hh
	@echo Compiling XrdSysDir.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdSysDir.cc -o $(OBJDIR)/XrdSysDir.o

$(OBJDIR)/XrdSysError.o: XrdSysError.cc XrdSysError.hh XrdSysLogger.hh \
                         XrdSysHeaders.hh XrdSysPlatform.hh
	@echo Compiling XrdSysError.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdSysError.cc -o $(OBJDIR)/XrdSysError.o

$(OBJDIR)/XrdSysFAttr.o: XrdSysFAttr.cc XrdSysFAttr.hh \
                         XrdSysFAttrBsd.icc XrdSysFAttrLnx.icc \
                         XrdSysFAttrMac.icc XrdSysFAttrSun.icc \
                         XrdSysError.hh     XrdSysPlatform.hh
	@echo Compiling XrdSysFAttr.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdSysFAttr.cc -o $(OBJDIR)/XrdSysFAttr.o

$(OBJDIR)/XrdSysLogger.o: XrdSysLogger.cc XrdSysLogger.hh \
                          XrdSysTimer.hh  XrdSysPlatform.hh XrdSysPthread.hh
	@echo Compiling XrdSysLogger.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdSysLogger.cc -o $(OBJDIR)/XrdSysLogger.o

$(OBJDIR)/XrdSysPlatform.o: XrdSysPlatform.cc  XrdSysPlatform.hh
	@echo Compiling XrdSysPlatform.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSETRESUID) $(INCLUDE) XrdSysPlatform.cc -o $(OBJDIR)/XrdSysPlatform.o

$(OBJDIR)/XrdSysPlugin.o: XrdSysPlugin.cc XrdSysPlugin.hh  \
                          XrdSysError.hh  XrdSysHeaders.hh
	@echo Compiling XrdSysPlugin.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSTRLCPY) $(INCLUDE) XrdSysPlugin.cc -o $(OBJDIR)/XrdSysPlugin.o

$(OBJDIR)/XrdSysPriv.o: XrdSysPriv.cc  XrdSysPriv.hh XrdSysHeaders.hh
	@echo Compiling XrdSysPriv.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFSETRESUID) $(INCLUDE) XrdSysPriv.cc -o $(OBJDIR)/XrdSysPriv.o

$(OBJDIR)/XrdSysPthread.o: XrdSysPthread.cc XrdSysPthread.hh XrdSysError.hh
	@echo Compiling XrdSysPthread.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(CFNOSTRICTALIAS) $(INCLUDE) XrdSysPthread.cc -o $(OBJDIR)/XrdSysPthread.o

$(OBJDIR)/XrdSysTimer.o: XrdSysTimer.cc XrdSysTimer.hh  
	@echo Compiling XrdSysTimer.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdSysTimer.cc -o $(OBJDIR)/XrdSysTimer.o

$(OBJDIR)/XrdSysXSLock.o: XrdSysError.hh XrdSysHeaders.hh \
                          XrdSysXSLock.hh XrdSysXSLock.cc
	@echo Compiling XrdSysXSLock.cc
	$(ECHO)$(CC) -c $(CFLAGS) $(INCLUDE) XrdSysXSLock.cc -o $(OBJDIR)/XrdSysXSLock.o
