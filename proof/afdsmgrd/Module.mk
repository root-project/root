# Module.mk for afdsmgrd module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Author: Dario Berzano <dario.berzano@cern.ch>, 01/04/2011

MODNAME      := afdsmgrd
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)

AFDSMGRDDIR  := $(MODDIR)
AFDSMGRDBIN  := bin/afdsmgrd
AFDSMGRDTAR  := $(shell cd $(AFDSMGRDDIR) && ls -1 afdsmgrd-v*.tar.gz | tail -1)

AFDSMGRDPREP   := $(AFDSMGRDDIR)/afdsmgrdPrepare
AFDSMGRDINST   := $(AFDSMGRDDIR)/afdsmgrdInstall
AFDSMGRDUNPACK := $(AFDSMGRDDIR)/afdsmgrdUnpack

AFDSMGRDBUILDDIR := $(AFDSMGRDDIR)/afdsmgrd/build

# This one triggers the build from ROOT
ALLEXECS     += $(AFDSMGRDBIN)

# ApMon: from Makefile.config
ifeq ($(AFDSMGRDAPMON),)
	AFDSMGRDAPMONDISABLED := 1
else
	AFDSMGRDAPMONDISABLED := 0
endif

# Phony targets
.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

# This target is directly invoked by ROOT's make
$(AFDSMGRDBIN): $(AFDSMGRDINST)

# Build afdsmgrd and install it under ROOT's source (ROOT_MODE is on)
$(AFDSMGRDINST): $(AFDSMGRDPREP)
	@echo 'Building afdsmgrd'
	@$(MAKE) install -C "$(AFDSMGRDBUILDDIR)" --no-print-directory
	@touch "$(AFDSMGRDINST)"

# Prepare afdsmgrd: run cmake
$(AFDSMGRDPREP): $(AFDSMGRDUNPACK)
	@echo 'Preparing afdsmgrd'
	@( mkdir -p "$(AFDSMGRDBUILDDIR)" && \
	   cd "$(AFDSMGRDBUILDDIR)" && \
	   cmake .. \
	     -DROOT_MODE=TRUE \
	     -DROOTSYS="$(ROOT_SRCDIR)" \
         -DApMon_PREFIX="$(AFDSMGRDAPMON)" \
         -DApMon_DISABLED="$(AFDSMGRDAPMONDISABLED)" \
	     -DCMAKE_BUILD_TYPE=Release )
	@touch "$(AFDSMGRDPREP)"

# Unpack latest afdsmgrd tarball
$(AFDSMGRDUNPACK): $(ALLLIBS)
	@echo 'Unpacking afdsmgrd tarball'
	@( cd "$(AFDSMGRDDIR)" && \
	   tar xzf "$(AFDSMGRDTAR)" && \
	   touch "$(AFDSMGRDUNPACK)" )

# Default target invoked when building this module
all-$(MODNAME): $(AFDSMGRDBIN)

# Shallow cleanup of afdsmgrd: a rebuild will be forced at next "make". This is
# a silent target if afdsmgrd hasn't been built (yet)
clean-$(MODNAME):
	@( if [ -f "$(AFDSMGRDBUILDDIR)/Makefile" ] ; then \
	     echo 'Shallow cleanup of afdsmgrd' ; \
	     $(MAKE) clean -C "$(AFDSMGRDBUILDDIR)" --no-print-directory ; \
	     rm -f "$(AFDSMGRDINST)" ; \
	   fi )

# Append this clean-afdsmgrd action to the global "clean" target
clean:: clean-$(MODNAME)

# Deep cleanup of afdsmgrd: generated files are unstaged from ROOT's source
# directory and all compilation-time temporary files are removed
distclean-$(MODNAME):
	@( if [ -f "$(AFDSMGRDBUILDDIR)/Makefile" ] ; then \
	     echo 'Unstaging afdsmgrd files from ROOT' ; \
	     $(MAKE) uninstall -C "$(AFDSMGRDBUILDDIR)" --no-print-directory ; \
	   fi )
	@( if [ -d "$(AFDSMGRDDIR)/afdsmgrd" ] ; then \
	     echo 'Deep cleanup of afdsmgrd' ; \
	     rm -rf "$(AFDSMGRDPREP)" "$(AFDSMGRDINST)" "$(AFDSMGRDUNPACK)" \
	       "$(AFDSMGRDDIR)/afdsmgrd" ; \
	   fi )

# Append this distclean-afdsmgrd action to the global "distclean" target
distclean:: distclean-$(MODNAME)
