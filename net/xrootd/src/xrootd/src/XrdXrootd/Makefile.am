#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

#       Note: since libXrdOfs needs to resolve symbols from xrootd's binary
#       itself, the xrootd binary must be compiled/linked with
#       -fPIC/-rdynamic (for gcc)

bin_PROGRAMS = xrootd
man8_MANS = ../../docs/man/xrootd.8

INCLUDES = -I$(top_srcdir)/src -I$(top_builddir)/src

EXTRA_DIST = XrdXrootdStat.icc

xrootd_SOURCES = \
        XrdXrootdAdmin.cc XrdXrootdAdmin.hh \
        XrdXrootdAio.cc XrdXrootdAio.hh       \
        XrdXrootdCallBack.cc XrdXrootdCallBack.hh \
        XrdXrootdConfig.cc XrdXrootdXPath.hh   \
        XrdXrootdFile.cc XrdXrootdFile.hh XrdXrootdFileLock.hh      \
        XrdXrootdFileLock1.cc XrdXrootdFileLock1.hh \
        XrdXrootdJob.cc XrdXrootdJob.hh \
        XrdXrootdLoadLib.cc \
	XrdXrootdPio.cc XrdXrootdPio.hh \
        XrdXrootdMonitor.cc XrdXrootdMonitor.hh XrdXrootdMonData.hh \
        XrdXrootdPrepare.cc XrdXrootdPrepare.hh  \
        XrdXrootdProtocol.cc XrdXrootdProtocol.hh  \
        XrdXrootdResponse.cc XrdXrootdResponse.hh  \
        XrdXrootdStats.cc XrdXrootdStats.hh     \
        XrdXrootdXeq.cc        \
        XrdXrootdXeqAio.cc XrdXrootdTrace.hh XrdXrootdReqID.hh

xrootd_LDADD = -ldl \
                  ../Xrd/libXrd.la \
                  ../XrdOfs/libXrdOfs.la \
                  ../XrdAcc/libXrdAcc.la \
                  ../XrdOss/libXrdOss.la \
                  ../XrdFrm/libXrdFrm.la \
                  ../XrdCms/libXrdCms.la \
                  ../XrdNet/libXrdNet.la \
                  ../XrdNet/libXrdNetUtil.la \
                  ../XrdOuc/libXrdOuc.la \
                  ../XrdSys/libXrdSys.la 

xrootd_LDFLAGS = -static -export-dynamic

#if IS_GCC
#xrootd_CXXFLAGS = -fPIC
# the program should export its symbol table so that a module's
# unresolved functions can be satisfied by it:
# -export-dynamic
#xrootd_LDFLAGS += -static -export-dynamic -fPIC
#endif
