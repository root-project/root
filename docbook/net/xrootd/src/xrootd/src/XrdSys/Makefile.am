#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

AM_CPPFLAGS = -I$(top_srcdir)/src

#############################################################################
## extra installation directory definitions

xrootdsysincdir = $(includedir)/xrootd/XrdSys

#############################################################################
## deliverables (to be built / installed)

lib_LTLIBRARIES = libXrdSys.la

# convenience lib since these files are also used for builds in other dirs
noinst_LTLIBRARIES = libXrdSysErrLog.la

xrootdsysinc_HEADERS = \
     XrdSysDir.hh    XrdSysLogger.hh    XrdSysPlugin.hh \
     XrdSysPthread.hh  XrdSysTimer.hh   XrdWin32.hh \
     XrdSysError.hh  XrdSysPlatform.hh  XrdSysPriv.hh \
     XrdSysSemWait.hh  XrdSysXSLock.hh  XrdSysHeaders.hh


#############################################################################
# Build rules

libXrdSysErrLog_la_SOURCES = \
        XrdSysError.cc XrdSysError.hh      \
        XrdSysLogger.cc XrdSysLogger.hh

libXrdSys_la_SOURCES = \
        XrdSysDir.cc XrdSysDir.hh \
        XrdSysPlugin.cc XrdSysPlugin.hh \
        XrdSysPriv.cc XrdSysPriv.hh \
        XrdSysPlatform.cc \
        XrdSysPthread.cc XrdSysPthread.hh \
        XrdSysSemWait.hh \
        XrdSysTimer.cc XrdSysTimer.hh \
        XrdSysXSLock.cc XrdSysXSLock.hh

libXrdSys_la_LIBADD = $(LIBPTHREAD) $(SEM_LIBRT) $(LIBNSL) libXrdSysErrLog.la
libXrdSys_la_CXXFLAGS = -prefer-pic

# sources not named in rules that still need to get rolled into the distribution tarball
EXTRA_DIST = XrdWin32.cc  XrdWin32.hh
