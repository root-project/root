# Module.mk for asimage module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 8/8/2002

MODDIR       := asimage
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ASIMAGEDIR   := $(MODDIR)
ASIMAGEDIRS  := $(ASIMAGEDIR)/src
ASIMAGEDIRI  := $(ASIMAGEDIR)/inc

ASTEPVERS    := libAfterImage
ASTEPDIRI    := $(MODDIRS)/$(ASTEPVERS)

##### libAfterImage #####
ASTEPLIBA    := $(MODDIRS)/$(ASTEPVERS)/libAfterImage.a
ASTEPLIB     := $(LPATH)/libAfterImage.a

##### libASImage #####
ASIMAGEL     := $(MODDIRI)/LinkDef.h
ASIMAGEDS    := $(MODDIRS)/G__ASImage.cxx
ASIMAGEDO    := $(ASIMAGEDS:.cxx=.o)
ASIMAGEDH    := $(ASIMAGEDS:.cxx=.h)

ASIMAGEH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
ASIMAGES     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ASIMAGEO     := $(ASIMAGES:.cxx=.o)

ASIMAGEDEP   := $(ASIMAGEO:.o=.d) $(ASIMAGEDO:.o=.d)

ASIMAGELIB   := $(LPATH)/libASImage.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEH))
ALLLIBS     += $(ASIMAGELIB)

# include all dependency files
INCLUDEFILES += $(ASIMAGEDEP)

##### local rules #####
include/%.h:    $(ASIMAGEDIRI)/%.h
		cp $< $@

$(ASTEPLIB):    $(ASTEPLIBA)
		cp $< $@

$(ASTEPLIBA):
		@(if [ ! -r $@ ]; then \
			echo "*** Building $@..."; \
			cd $(ASIMAGEDIRS); \
			if [ ! -d $(ASTEPVERS) ]; then \
				if [ "x`which gtar 2>/dev/null | awk '{if ($$1~/gtar/) print $$1;}'`" != "x" ]; then \
					gtar zxf $(ASTEPVERS).tar.gz; \
				else \
					gunzip -c $(ASTEPVERS).tar.gz | tar xf -; \
				fi; \
			fi; \
			cd $(ASTEPVERS); \
			ACC=; \
			ACFLAGS="-O"; \
			if [ $(CC) = "icc" ]; then \
				ACC="icc"; \
			fi; \
			GNUMAKE=$(MAKE) CC=$$ACC CFLAGS=$$ACFLAGS \
			./configure \
			--with-ttf=no --with-gif=no --with-afterbase=no; \
			$(MAKE); \
		fi)

$(ASIMAGELIB):  $(ASIMAGEO) $(ASIMAGEDO) $(ASTEPLIB) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImage.$(SOEXT) $@ \
		   "$(ASIMAGEO) $(ASIMAGEDO)" \
		   "$(ASIMAGELIBEXTRA) $(ASTEPLIB) $(ASEXTRALIBDIR) $(ASEXTRALIB)"

$(ASIMAGEDS):   $(ASIMAGEH) $(ASIMAGEL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEH) $(ASIMAGEL)

$(ASIMAGEDO):   $(ASIMAGEDS) $(ASTEPLIB)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I$(ASTEPDIRI) -I. -o $@ -c $<

all-asimage:    $(ASTEPLIB) $(ASIMAGELIB)

clean-asimage:
		@rm -f $(ASIMAGEO) $(ASIMAGEDO)
		-@(if [ -d $(ASIMAGEDIRS)/$(ASTEPVERS) ]; then \
			cd $(ASIMAGEDIRS)/$(ASTEPVERS); \
			$(MAKE) clean; \
		fi)

clean::         clean-asimage

distclean-asimage: clean-asimage
		@rm -f $(ASIMAGEDEP) $(ASIMAGEDS) $(ASIMAGEDH) $(ASIMAGELIB)
		@rm -rf $(ASTEPLIB) $(ASIMAGEDIRS)/$(ASTEPVERS)

distclean::     distclean-asimage

##### extra rules ######
$(ASIMAGEO): %.o: %.cxx $(ASTEPLIB)
	$(CXX) $(OPT) $(CXXFLAGS) -I$(ASTEPDIRI) -o $@ -c $<
