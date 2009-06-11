################################################################################
##                        Makefile.am for xrootd
##
## (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University
##       All Rights Reserved. See XrdInfo.cc for complete License Terms
##   Produced by Andrew Hanushevsky for Stanford University under contract
##              DE-AC03-76-SFO0515 with the Department of Energy
##
##
##
## Makefile/configure Author: Derek Feichtinger <derek.feichtinger@cern.ch>
## Initial version: 1.8.2005
##
## Version info: $Id: Makefile.am,v 1.13 2009/03/06 18:02:42 dfeich Exp $
## Checked in by $Author: dfeich $
################################################################################

# Tell GNU make >=3.79 not to run the top level in parallel since
# inter sub-makefile dependencies are not correct.
# To be fixed using a flat Makefile.
# (F. Rademakers, 31 Oct 2005)
.NOTPARALLEL:

AUTOMAKE_OPTIONS = foreign 1.6

SUBDIRS = src

## This is needed on MacOS in order to correctly build shared libraries with weak
## symbols
if NEED_MACOS_DEPLOYMENT_TARGET
export MACOSX_DEPLOYMENT_TARGET = 10.3
endif

.PHONY: doc silent

silent:
	@cd src; $(MAKE) | sed -n -e '/libtool/d' \
            -e 's/.*-o  *\([^ ]*\).*/++++ Creating \1 ++++/p' \
            -e '/make/p'

if ENABLE_DOC
all-local: doc

clean-local:
	rm -rf doxydoc


doc: Doxyfile
	@if test ! -e doxydoc/html/index.html; then \
           echo "BUILDING DOXYGEN DOCUMENTATION"; \
           $(DOXYGEN) Doxyfile; \
        else \
           echo "Doxygen doc already built. Remove the directory"; \
           echo "'doxydoc' to trigger rebuilding"; \
        fi
endif



################################################################
## Here we define general includes (not specific to any module)
xrootdincdir = $(includedir)/xrootd

xrootdinc_HEADERS = src/XrdVersion.hh

xrootdprotocolincdir = $(includedir)/xrootd/XProtocol

xrootdprotocolinc_HEADERS = src/XProtocol/XProtocol.hh src/XProtocol/XPtypes.hh


################################################################
## Here we define where things in etc and in utils should end up

xrootdstartupdir = $(sysconfdir)/xrootd

dist_xrootdstartup_SCRIPTS = etc/StartOLB \
etc/StartXRD \
etc/StopOLB \
etc/StopXRD \
etc/XrdOlbMonPerf 

dist_xrootdstartup_DATA = etc/StartXRD.cf.example \
etc/StartOLB.cf.example \
etc/xrootd.cf.example



xrootdutilsdir = $(XROOTDUTILSDIR)

dist_xrootdutils_SCRIPTS = utils/XrdOlbNotify.pm \
utils/fs_stat \
utils/mps_MigrPurg \
utils/mps_PreStage \
utils/mps_Stage \
utils/mps_Xeq \
utils/mps_prep \
utils/xrdcksum \
utils/xrdmd5sum \
utils/XrdCmsNotify.pm

dist_xrootdutils_DATA = utils/ooss_CAlloc.pm \
utils/ooss_Lock.pm \
utils/ooss_MonP.pm \
utils/ooss_name2name.pm \
utils/mps_monStage.pm

# Extra files/directories to go into the distribution tarball
EXTRA_DIST = Doxyfile README.Maintainer bootstrap.sh config pthreads-win32 \
             test/test1.sh test/test2.sh test/testconfig.sh.in \
             getAutotools.sh


TESTS = test/test1.sh test/test2.sh





#### WARNING: this rpm target is put in for convenience. It makes
####        the assumption that builds are done in the standard location
####         Needs to be improved
rpm:    dist
	cp $(DIST_ARCHIVES) /usr/src/redhat/SOURCES/
	rpmbuild -bb xrootd.spec
