# Module.mk for afdsmgrd module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Author: Dario Berzano, 12/01/2011

MODNAME      := afdsmgrd
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)

AFDSMGRDDIR  := $(MODDIR)
AFDSMGRDBIN  := bin/afdsmgrd
AFDSMGRDTAR  := $(shell cd $(AFDSMGRDDIR) && echo afdsmgrd-v*.tar.gz)

# List of pairs source file - destination file. Source files are relative to
# either build/src or src dir (automatically choosen), destination files are
# relative to $(ROOT_SRCDIR)
AFDSMGRDCP := src/afdsmgrd                              $(AFDSMGRDBIN) \
              etc/init.d/afdsmgrd                       etc/proof/afdsmgrdctl \
              etc/sysconfig/afdsmgrd                    etc/proof/afdsmgrd_sysconfig.conf \
              etc/xrootd/afdsmgrd.cf.example            etc/proof/afdsmgrd.cf.example \
              etc/xrootd/single_dataset_src.cf.example  etc/proof/afdsmgrd_single_dataset_src.cf.example

# This one triggers the build from ROOT
ALLEXECS     += $(AFDSMGRDBIN)

.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

# The tar file is unpacked and the project is configured and built through
# afdsmgrd's custom built system. Proper variables are passed to ./configure,
# including the option to enable/disable ApMon, and files are installed into
# proper ROOT places
$(AFDSMGRDBIN):
	@( cd $(AFDSMGRDDIR) && \
	   if [ ! -d afdsmgrd ]; then \
	     echo "*** Unpacking afdsmgrd tarball ***" ; \
	     tar xzf "$(AFDSMGRDTAR)" ; \
	   fi && \
	   cd afdsmgrd && \
	   if [ ! -d build ]; then \
	     ./configure --with-rootsys="$(ROOT_SRCDIR)" "$(AFDSMGRDAPMON)" \
	       --prefix="$(ROOT_SRCDIR)" ; \
	   fi && \
	   make ; \
	 )
	@( P=0 ; \
	   for F in $(AFDSMGRDCP) ; do \
	     if test x$$P = x0 ; then \
	       P=1 ; S="$(AFDSMGRDDIR)/afdsmgrd/build/$$F" ; \
	       if [ ! -e "$$S" ]; then \
	         S="$(AFDSMGRDDIR)/afdsmgrd/$$F" ; \
	       fi ; \
	     else \
	       P=0 ; D="$(ROOT_SRCDIR)/$$F" ; \
	       rsync -a "$$S" "$$D" ; \
	     fi ; \
	   done ; \
	 )

all-$(MODNAME): $(AFDSMGRDBIN)

# Simple cleanup of afdsmgrd. The "clean" target of afdsmgrd is called without
# removing the directory where the source was unpacked. If the project hasn't
# been built yet, it just exits without printing a single message
clean-$(MODNAME):
	@( rm -f $(ROOT_SRCDIR)/$(AFDSMGRDBIN) ; \
	   cd "$(AFDSMGRDDIR)" ; \
	   if [ -d afdsmgrd ]; then \
	     echo "*** Cleaning up afdsmgrd ***" ; \
	     cd afdsmgrd ; \
	     make clean ; \
	   fi ; \
	)

clean:: clean-$(MODNAME)

# Deep cleanup of afdsmgrd. If the unpack directory exists, it is removed and a
# message is printed out. If the directory does not exist (afdsmgrd was never
# built before) it does nothing without printing a single message
distclean-$(MODNAME):
	@( cd "$(AFDSMGRDDIR)" ; \
	   if [ -d afdsmgrd ]; then \
	     echo "*** Completely cleaning up afdsmgrd ***" ; \
	     rm -r afdsmgrd ; \
	   fi ; \
	)
	@( P=0 ; \
	   for F in $(AFDSMGRDCP) ; do \
	     if test ! x$$P = x0 ; then \
	       P=1 ; \
	     else \
	       P=0 ; \
	       rm -f "$(ROOT_SRCDIR)/$$F" ; \
	     fi ; \
	   done ; \
	 )


distclean:: distclean-$(MODNAME)
