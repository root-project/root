# Module.mk for afdsmgrd module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Author: Dario Berzano <dario.berzano@cern.ch>, 01/04/2011

MODNAME      := afdsmgrd
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)

AFDSMGRDDIR  := $(MODDIR)
AFDSMGRDBIN  := bin/afdsmgrd
AFDSMGRDTAR  := $(shell cd $(AFDSMGRDDIR) && ls -1 afdsmgrd-v*.tar.gz | tail -n1)

# This one triggers the build from ROOT
ALLEXECS     += $(AFDSMGRDBIN)

# Phony targets
.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

# The tar file is unpacked and the project is configured and built through
# afdsmgrd's custom built system. Proper variables are passed to ./configure,
# including the option to enable/disable ApMon, and files are installed under
# $ROOTSYS properly directly by invoking "make install"
$(AFDSMGRDBIN): $(ALLLIBS)
	@( cd $(AFDSMGRDDIR) && \
	   if [ ! -d afdsmgrd ]; then \
	     echo "*** Unpacking afdsmgrd tarball ***" ; \
	     tar xzf "$(AFDSMGRDTAR)" ; \
	   fi && \
	   cd afdsmgrd && \
	   if [ ! -d build ]; then \
	     ./configure --with-rootsys="$(ROOT_SRCDIR)" "$(AFDSMGRDAPMON)" \
	       --prefix="$(ROOT_SRCDIR)" --root-mode ; \
	   fi && \
	   make install ; \
	 )

# Default target invoked when building this module
all-$(MODNAME): $(AFDSMGRDBIN)

# Simple cleanup of afdsmgrd. The "clean" target of afdsmgrd is called without
# removing the directory where the source was unpacked. If the project hasn't
# been built yet, it just exits without printing a single message
clean-$(MODNAME):
	@( rm -f $(ROOT_SRCDIR)/$(AFDSMGRDBIN) ; \
	   cd "$(AFDSMGRDDIR)" ; \
	   if [ -d afdsmgrd ]; then \
	     echo "*** Cleaning up afdsmgrd ***" ; \
	     cd afdsmgrd && \
	     make clean ; \
	   fi ; \
	)

# Append this clean-afdsmgrd action to the global "clean" target in main
# Makefile
clean:: clean-$(MODNAME)

# Deep cleanup of afdsmgrd. If the unpack directory exists, it is removed and a
# message is printed out. If the directory does not exist (afdsmgrd was never
# built before) it does nothing without printing a single message
#
# TODO: add make uninstall to CMake
distclean-$(MODNAME):
	@( rm -f $(ROOT_SRCDIR)/$(AFDSMGRDBIN) ; \
	   rm -f $(ROOT_SRCDIR)/etc/proof/afdsmgrd.conf.example ; \
	   rm -f $(ROOT_SRCDIR)/etc/proof/sysconfig/afdsmgrd.example ; \
	   rm -f $(ROOT_SRCDIR)/etc/proof/init.d/afdsmgrd ; \
	   rm -f $(ROOT_SRCDIR)/etc/proof/afdsmgrd-macros/StageXrd.C ; \
	   rm -f $(ROOT_SRCDIR)/etc/proof/afdsmgrd-root.sh ; \
	   rm -f $(ROOT_SRCDIR)/etc/proof/afdsmgrd-exec-wrapper ; \
	   rm -fr $(ROOT_SRCDIR)/etc/proof/sysconfig ; \
	   rm -fr $(ROOT_SRCDIR)/etc/proof/init.d ; \
	   rm -fr $(ROOT_SRCDIR)/etc/proof/afdsmgrd-macros ; \
	   cd "$(AFDSMGRDDIR)" ; \
	   if [ -d afdsmgrd ]; then \
	     echo "*** Completely cleaning up afdsmgrd ***" ; \
	     rm -r afdsmgrd ; \
	   fi ; \
	)

# Append this distclean-afdsmgrd action to the global "distclean" target in
# main Makefile
distclean:: distclean-$(MODNAME)
