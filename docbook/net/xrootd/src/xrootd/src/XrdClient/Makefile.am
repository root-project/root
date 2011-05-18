#######################################################################
## Makefile.am for xrootd
##
##
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################


AM_CPPFLAGS = -I$(top_srcdir)/src $(READLINE_INCDIR) $(JAVA_INCDIR)  -I$(top_builddir)/src

#############################################################################
## extra directory definitions for install area
javaclassdir = $(JAVA_CLASSPATH)
xrootdclientincdir = $(includedir)/xrootd/XrdClient


#############################################################################
## deliverables (to be built / installed)
bin_PROGRAMS = xrdcp TestXrdClient TestXrdClient_read xrd xprep xrdstagetool

## old automake versions do not know += operator for adding to a variable, so we have to do this
## very awkwardly using the EXTRA_ definition
if ENABLE_JAVAINT
  BUILDJAVA = libXrdClientAdminJNI.la
endif

if ENABLE_PERLINT
  BUILDPERL = libXrdClientAdmin.la
endif
lib_LTLIBRARIES = libXrdClient.la $(BUILDPERL) $(BUILDJAVA)

xrootdclientinc_HEADERS = XrdClient.hh XrdClientAdmin.hh XrdClientCallback.hh \
               XrdClientVector.hh XrdClientConnMgr.hh \
               XrdClientAbs.hh XrdClientUnsolMsg.hh XrdClientMessage.hh \
               XrdClientProtocol.hh XrdClientSock.hh XrdClientUrlInfo.hh \
               XrdClientConn.hh XrdClientConst.hh XrdClientReadCache.hh \
               XrdClientInputBuffer.hh XrdClientEnv.hh XrdClientThread.hh \
               XrdClientPhyConnection.hh XrdClientLogConnection.hh \
               XrdClientDebug.hh XrdClientReadV.hh XrdClientPSock.hh \
               XrdCpMthrQueue.hh XrdCpWorkLst.hh XrdClientAbsMonIntf.hh \
               XrdcpXtremeRead.hh

#############################################################################
# Build rules

# the following source may be built during the build itself
BUILT_SOURCES=XrdClientAdmin_c_wrap.cc

libXrdClient_la_SOURCES = \
        XrdClientAbs.cc XrdClientAbs.hh                \
        XrdClient.cc XrdClient.hh                   \
        XrdClientSock.cc XrdClientSock.hh               \
        XrdClientConn.cc XrdClientConn.hh               \
        XrdClientConnMgr.cc XrdClientConnMgr.hh XrdClientUnsolMsg.hh           \
        XrdClientDebug.cc XrdClientDebug.hh              \
        XrdClientInputBuffer.cc XrdClientInputBuffer.hh        \
        XrdClientLogConnection.cc XrdClientLogConnection.hh      \
        XrdClientMessage.cc XrdClientMessage.hh XrdClientProtocol.hh          \
        XrdClientPhyConnection.cc XrdClientPhyConnection.hh      \
        XrdClientReadCache.cc XrdClientReadCache.hh          \
        XrdClientUrlInfo.cc XrdClientUrlInfo.hh            \
        XrdClientUrlSet.cc XrdClientUrlSet.hh             \
        XrdClientThread.cc XrdClientThread.hh             \
        XrdClientProtocol.cc            \
        XrdClientAdmin.cc XrdClientAdmin.hh XrdClientVector.hh             \
        XrdClientEnv.cc XrdClientEnv.hh XrdClientConst.hh                \
        XrdCpMthrQueue.cc  XrdCpMthrQueue.hh             \
        XrdCpWorkLst.cc XrdCpWorkLst.hh                \
        XrdClientSid.cc XrdClientSid.hh \
        XrdClientPSock.cc XrdClientPSock.hh \
	XrdClientMStream.cc XrdClientMStream.hh \
        XrdClientReadV.cc XrdClientReadV.hh \
        XrdClientReadAhead.cc XrdClientReadAhead.hh \
        XrdcpXtremeRead.cc XrdcpXtremeRead.hh

libXrdClient_la_LIBADD = $(LIBSUNMATH) \
              ../XrdSys/libXrdSys.la \
	      ../XrdCrypto/libXrdCrypto.la \
              ../XrdSut/libXrdSut.la \
              ../XrdNet/libXrdNet.la \
              ../XrdOuc/libXrdOuc.la \
              ../XrdSys/libXrdSys.la \
              ../XrdNet/libXrdNetUtil.la

clientlibs =  libXrdClient.la \
              -ldl $(LIBZ)

xrdcp_SOURCES = Xrdcp.cc
# Note: static is required for Darwin. Linux and Solaris nicely manage to integrate
#       all needed symbols with shared lib linking
xrdcp_LDFLAGS = -static -export-dynamic
xrdcp_LDADD = $(clientlibs)



xrd_SOURCES = XrdCommandLine.cc
xrd_LDADD = $(clientlibs) $(READLINE_LIBDIR) $(READLINE_LIBS)



TestXrdClient_SOURCES = XrdClientAdmin_c.cc XrdClientAdmin_c.hh \
                        TestXrdClient.cc
TestXrdClient_LDADD =  $(clientlibs)



TestXrdClient_read_SOURCES = TestXrdClient_read.cc
TestXrdClient_read_LDADD =  $(clientlibs)


xprep_SOURCES = XrdClientPrep.cc
xprep_LDADD = $(clientlibs)

$(OBJECT_XPREP) $(BINLIBS) $(LIBM) $(LIBS)


xrdstagetool_SOURCES = XrdStageTool.cc
xrdstagetool_LDADD = $(clientlibs)

$(OBJECT_STAGETOOL) $(BINLIBS) $(LIBM) $(LIBS)

if ENABLE_JAVAINT
libXrdClientAdminJNI_la_SOURCES = XrdClientAdminJNI.cc
libXrdClientAdminJNI_la_LIBADD = $(clientlibs)                                                        
all-local: XrdClientAdminJNI.class
XrdClientAdminJNI.class: XrdClientAdminJNI.java
	$(JAVAC) -d ./ $(srcdir)/XrdClientAdminJNI.java
install-exec-local:  XrdClientAdminJNI.class
	$(INSTALL) -d $(javaclassdir)/xrootdadmin
	$(INSTALL) -m 644 xrootdadmin/XrdClientAdminJNI.class $(javaclassdir)/xrootdadmin/XrdClientAdminJNI.class
uninstall-local:
	rm -f $(javaclassdir)/xrootdadmin/XrdClientAdminJNI.class
endif



clean-local:
	rm -f xrootdadmin/XrdClientAdminJNI.class XrdClientAdmin_c_wrap.cc


# To have the swigfile compiled with the c++ compiler (and because it
# always has been in the repository as .c) we copy it to .cc. If a
# suitable swig is available, we instead create it anew
# This needs to be done outside the ENABLE_PERLINT clause, to get a working
# "make dist" target
SWIG_TARGETS = XrdClientAdmin_c_wrap.cc XrdClientAdmin.pm
.PHONY: FORCE
if REBUILD_SWIGFILE
# This creates both the .cc file and the .pm perl module
$(SWIG_TARGETS): XrdClientAdmin_c.hh
	echo Creating swig perl interface
	$(SWIG) -perl -o $@ $(srcdir)/XrdClientAdmin_c.hh
CLEANFILES = $(SWIG_TARGETS)
else
XrdClientAdmin_c_wrap.cc: XrdClientAdmin.pm
	cp $(srcdir)/XrdClientAdmin_c_wrap.c XrdClientAdmin_c_wrap.cc
XrdClientAdmin.pm: FORCE
	if diff $(srcdir)/XrdClientAdmin.pm XrdClientAdmin.pm >/dev/null 2>&1; then \
	  echo "Identical - nothing to do";                                         \
	else                                                                        \
	  cp $(srcdir)/XrdClientAdmin.pm XrdClientAdmin.pm;                         \
	fi
endif

FORCE:

if ENABLE_PERLINT

libXrdClientAdmin_la_SOURCES = XrdClientAdmin_c.cc \
                               XrdClientAdmin_c_wrap.cc
libXrdClientAdmin_la_LIBADD = $(clientlibs)                                                        

libXrdClientAdmin_la_CXXFLAGS = $(PERL_CCOPTS)
install-exec-local: libXrdClientAdmin.la XrdClientAdmin.pm
	$(INSTALL) -m 644 XrdClientAdmin.pm  $(DESTDIR)/$(libdir)/XrdClientAdmin.pm
endif

# we need to include the pregenerated swig file in the distribution
EXTRA_DIST = XrdClientAdmin_c_wrap.c XrdClientAdmin.pm
