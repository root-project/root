#######################################################################
## Makefile.am for token authorization module for xrootd
##
##
## Author: Andreas Joachim Peters <andreas.joachim.peters@cern.ch>
##         Derek Feichtinger <derek.feichtinger@cern.ch>
## Initial version: 1.8.2005
##
## Version info: $Id$
## Checked in by $Author$
#######################################################################

lib_LTLIBRARIES = libXrdTokenAuthzOfs.la
bin_PROGRAMS = xrm

INCLUDES = -I$(top_srcdir)/src $(AXMLR_INC) $(SENV_INC) $(TKAUTHZ_INCDIR)

###############################################
# sources of the original XrdOfs implementation
OFSSOURCES = \
        XrdOfsConfig.cc \
        XrdOfsEvs.cc \
        XrdOfsHandle.cc \
        XrdOfs.cc

# Note: Because of the layout of the current code, two
#       replacements are necessary in XrdOfs.cc
#       1) read in the new icc file to define the global XrdOfsFS
#       2) the XrdSfsGetFileSystem via which the main executable
#          calls the dynamic module must exist only once
$(OFSSOURCES) :
	cp $(top_srcdir)/src/XrdOfs/$@ .
	sed -e 's/XrdOfs\/XrdOfs.icc/XrdTokenAuthzOfs.icc/' \
            -e 's/XrdOfs\/XrdOfs.hh/XrdTokenAuthzOfs.hh/' \
            -e 's/XrdSfsGetFileSystem/XrdSfsGetFileSystemINACTIVE/' \
            -e 's/extern *XrdOfs *XrdOfsFS/extern XrdTokenAuthzOfs XrdOfsFS/' $@ > ./tmp_build
	mv ./tmp_build $@

###############################################

libXrdTokenAuthzOfs_la_SOURCES = \
        XrdTokenAuthzOfs.cc XrdTokenAuthzOfs.hh \
        XrdTokenAuthzOfs.icc
nodist_libXrdTokenAuthzOfs_la_SOURCES = $(OFSSOURCES)

xrm_SOURCES = xrm.cc

libXrdTokenAuthzOfs_la_LDFLAGS = -module
libXrdTokenAuthzOfs_la_LIBTOOLFLAGS = --tag=disable-static

libXrdTokenAuthzOfs_la_LIBADD = $(top_builddir)/src/XrdAcc/libXrdAcc.la \
                         $(top_builddir)/src/XrdOss/libXrdOss.la \
                         $(top_builddir)/src//XrdNet/libXrdNet.la \
                         $(top_builddir)/src//XrdOuc/libXrdOuc.la \
                         $(top_builddir)/src/XrdXr/libXrdXr.la \
                         $(TKAUTHZ_LIBDIR) -lTokenAuthz 

xrm_LDADD = $(top_builddir)/src/XrdClient/libXrdClient.la \
            $(top_builddir)/src/XrdSec/libXrdSec.la \
            $(top_builddir)/src/XrdOuc/libXrdOuc.la \
            $(top_builddir)/src/XrdNet/libXrdNet.la \
            $(top_builddir)/src/XrdSys/libXrdSys.la \
            -ldl


CLEANFILES = $(OFSSOURCES)

## for now we still distribute the separate configure.ac to be used with the old build
EXTRA_DIST = configure.ac acinclude.m4

