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

ifeq ($(BUILTINASIMAGE), yes)
ASTEPVERS    := libAfterImage
ASTEPDIRS    := $(MODDIRS)/$(ASTEPVERS)
ASTEPDIRI    := -I$(MODDIRS)/$(ASTEPVERS)
else
ASTEPDIRI    := $(ASINCDIR:%=-I%)
ASTEPDIRS    :=
ASTEPVERS    :=
endif

##### libAfterImage #####
ifeq ($(BUILTINASIMAGE), yes)
ifeq ($(PLATFORM),win32)
ASTEPLIBA    := $(ASTEPDIRS)/libAfterImage.lib
ASTEPLIB     := $(LPATH)/libAfterImage.lib
else
ASTEPLIBA    := $(ASTEPDIRS)/libAfterImage.a
ASTEPLIB     := $(LPATH)/libAfterImage.a
endif
ASTEPDEP     := $(ASTEPLIB)
else
ASTEPLIBA    := $(ASLIBDIR) $(ASLIB)
ASTEPLIB     := $(ASLIBDIR) $(ASLIB)
ASTEPDEP     :=
endif

##### libASImage #####
ASIMAGEL     := $(MODDIRI)/LinkDef.h
ASIMAGEDS    := $(MODDIRS)/G__ASImage.cxx
ASIMAGEDO    := $(ASIMAGEDS:.cxx=.o)
ASIMAGEDH    := $(ASIMAGEDS:.cxx=.h)

ASIMAGEH     := $(MODDIRI)/TASImage.h
ASIMAGES     := $(MODDIRS)/TASImage.cxx
ASIMAGEO     := $(ASIMAGES:.cxx=.o)

ASIMAGEDEP   := $(ASIMAGEO:.o=.d) $(ASIMAGEDO:.o=.d)

ASIMAGELIB   := $(LPATH)/libASImage.$(SOEXT)

##### libASImageGui #####
ASIMAGEGUIL  := $(MODDIRI)/LinkDefGui.h
ASIMAGEGUIDS := $(MODDIRS)/G__ASImageGui.cxx
ASIMAGEGUIDO := $(ASIMAGEGUIDS:.cxx=.o)
ASIMAGEGUIDH := $(ASIMAGEGUIDS:.cxx=.h)

ASIMAGEGUIH  := $(MODDIRI)/TASPaletteEditor.h
ASIMAGEGUIS  := $(MODDIRS)/TASPaletteEditor.cxx
ASIMAGEGUIO  := $(ASIMAGEGUIS:.cxx=.o)

ASIMAGEGUIDEP := $(ASIMAGEGUIO:.o=.d) $(ASIMAGEGUIDO:.o=.d)

ASIMAGEGUILIB := $(LPATH)/libASImageGui.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEGUIH))
ALLLIBS     += $(ASIMAGELIB) $(ASIMAGEGUILIB)

# include all dependency files
INCLUDEFILES += $(ASIMAGEDEP) $(ASIMAGEGUIDEP)

##### local rules #####
include/%.h:    $(ASIMAGEDIRI)/%.h
		cp $< $@

ifeq ($(BUILTINASIMAGE), yes)
$(ASTEPLIB):    $(ASTEPLIBA)
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)

$(ASTEPLIBA):   $(ASTEPDIRS).tar.gz
ifeq ($(PLATFORM),win32)
		@(if [ -d $(ASTEPDIRS) ]; then \
			rm -rf $(ASTEPDIRS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(ASIMAGEDIRS); \
		if [ ! -d $(ASTEPVERS) ]; then \
			gunzip -c $(ASTEPVERS).tar.gz | tar xf -; \
		fi; \
		cd $(ASTEPVERS); \
		unset MAKEFLAGS; \
		nmake FREETYPEDIRI=-I../../../$(FREETYPEDIRI) -nologo -f libAfterImage.mak \
		CFG="libAfterImage - Win32 Release")
else
		@(if [ -d $(ASTEPDIRS) ]; then \
			rm -rf $(ASTEPDIRS); \
		fi; \
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
		ACC=$(CC); \
		ACFLAGS="-O"; \
		if [ "$(CC)" = "icc" ]; then \
			ACC="icc"; \
		fi; \
		if [ "$(ARCH)" = "sgicc64" ]; then \
			ACC="gcc -mabi=64"; \
		fi; \
		if [ "$(ARCH)" = "hpuxia64acc" ]; then \
			ACC="cc +DD64 -Ae"; \
			ACCALT="gcc -mlp64"; \
		fi; \
		if [ "$(ARCH)" = "linuxppc64gcc" ]; then \
			ACC="gcc -m64"; \
		fi; \
		if [ "$(ARCH)" = "linuxx8664gcc" ]; then \
			ACC="gcc -m64"; \
			MMX="--enable-mmx-optimization=no"; \
		fi; \
		if [ "$(ASJPEGINCDIR)" != "" ]; then \
			JPEGINCDIR="--with-jpeg-includes=$(ASJPEGINCDIR)"; \
		fi; \
		if [ "$(ASPNGINCDIR)" != "" ]; then \
			PNGINCDIR="--with-png-includes=$(ASPNGINCDIR)"; \
		fi; \
		if [ "$(ASTIFFINCDIR)" != "" ]; then \
			TIFFINCDIR="--with-tiff-includes=$(ASTIFFINCDIR)"; \
		fi; \
		if [ "$(ASGIFINCDIR)" != "" ]; then \
			GIFINCDIR="--with-gif-includes=$(ASGIFINCDIR)"; \
			NOUNGIF="--with-ungif --with-builtin-ungif=no"; \
		else \
			NOUNGIF="--with-builtin-ungif"; \
		fi; \
		if [ "$(FREETYPEDIRI)" != "" ]; then \
			TTFINCDIR="--with-ttf-includes=-I../../../$(FREETYPEDIRI)"; \
		fi; \
		GNUMAKE=$(MAKE) CC=$$ACC CFLAGS=$$ACFLAGS \
		./configure \
		--with-ttf $$TTFINCDIR \
		--with-afterbase=no \
		--disable-glx \
		$$MMX \
		--with-builtin-ungif \
		$$JPEGINCDIR \
		$$PNGINCDIR \
		$$TIFFINCDIR \
		$$GIFINCDIR; \
		$(MAKE))
endif
endif

##### libASImage #####
$(ASIMAGELIB):  $(ASIMAGEO) $(ASIMAGEDO) $(ASTEPDEP) $(FREETYPEDEP) \
                $(MAINLIBS) $(ASIMAGELIBDEP)
ifeq ($(PLATFORM),win32)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImage.$(SOEXT) $@ \
		   "$(ASIMAGEO) $(ASIMAGEDO)" \
		   "$(ASIMAGELIBEXTRA) $(ASTEPLIB) $(ASEXTRALIBDIR) \
                    $(ASEXTRALIB) $(FREETYPELDFLAGS) $(FREETYPELIB)"
else
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImage.$(SOEXT) $@ \
		   "$(ASIMAGEO) $(ASIMAGEDO)" \
		   "$(ASIMAGELIBEXTRA) $(ASTEPLIB) $(ASEXTRALIBDIR) \
                    $(ASEXTRALIB) $(XLIBS) $(FREETYPELDFLAGS) $(FREETYPELIB)"
endif

$(ASIMAGEDS):   $(ASIMAGEH) $(ASIMAGEL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEH) $(ASIMAGEL)

$(ASIMAGEDO):   $(ASIMAGEDS) $(ASTEPLIB)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(ASTEPDIRI) -I. -o $@ -c $<

##### libASImageGui #####
$(ASIMAGEGUILIB):  $(ASIMAGEGUIO) $(ASIMAGEGUIDO) $(ASTEPDEP) $(FREETYPEDEP) \
                   $(MAINLIBS) $(ASIMAGEGUILIBDEP)
ifeq ($(PLATFORM),win32)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImageGui.$(SOEXT) $@ \
		   "$(ASIMAGEGUIO) $(ASIMAGEGUIDO)" \
		   "$(ASIMAGEGUILIBEXTRA) $(ASTEPLIB) $(ASEXTRALIBDIR) \
                    $(ASEXTRALIB) $(FREETYPELDFLAGS) $(FREETYPELIB)"
else
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImageGui.$(SOEXT) $@ \
		   "$(ASIMAGEGUIO) $(ASIMAGEGUIDO)" \
		   "$(ASIMAGEGUILIBEXTRA) $(ASTEPLIB) $(ASEXTRALIBDIR) \
                    $(ASEXTRALIB) $(XLIBS) $(FREETYPELDFLAGS) $(FREETYPELIB)"
endif

$(ASIMAGEGUIDS): $(ASIMAGEGUIH) $(ASIMAGEGUIL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEGUIH) $(ASIMAGEGUIL)

$(ASIMAGEGUIDO): $(ASIMAGEGUIDS) $(ASTEPLIB)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(ASTEPDIRI) -I. -o $@ -c $<

all-asimage:    $(ASIMAGELIB) $(ASIMAGEGUILIB)

map-asimage:    $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(ASIMAGELIB) \
		   -d $(ASIMAGELIBDEP) -c $(ASIMAGEL)

map-asimagegui: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(ASIMAGEGUILIB) \
		   -d $(ASIMAGEGUILIBDEP) -c $(ASIMAGEGUIL)

map::           map-asimage map-asimagegui

clean-asimage:
		@rm -f $(ASIMAGEO) $(ASIMAGEDO) $(ASIMAGEGUIO) $(ASIMAGEGUIDO)
ifeq ($(BUILTINASIMAGE), yes)
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(ASTEPDIRS) ]; then \
			cd $(ASTEPDIRS); \
			unset MAKEFLAGS; \
			nmake -nologo -f libAfterImage.mak clean \
			CFG="libAfterImage - Win32 Release"; \
		fi)
else
		-@(if [ -d $(ASTEPDIRS) ]; then \
			cd $(ASTEPDIRS); \
			$(MAKE) clean; \
		fi)
endif
endif

clean::         clean-asimage

distclean-asimage: clean-asimage
		@rm -f $(ASIMAGEDEP) $(ASIMAGEDS) $(ASIMAGEDH) $(ASIMAGELIB) \
		   $(ASIMAGEGUIDEP) $(ASIMAGEGUIDS) $(ASIMAGEGUIDH) \
		   $(ASIMAGEGUILIB)

ifeq ($(BUILTINASIMAGE), yes)
		@rm -rf $(ASTEPLIB)
endif
		@rm -rf $(ASTEPDIRS)

distclean::     distclean-asimage

##### extra rules ######
$(ASIMAGEO): %.o: %.cxx $(ASTEPLIB) $(FREETYPEDEP)
	$(CXX) $(OPT) $(FREETYPEINC) $(CXXFLAGS) $(ASTEPDIRI) -o $@ -c $<

$(ASIMAGEGUIO): %.o: %.cxx $(ASTEPLIB)
	$(CXX) $(OPT) $(CXXFLAGS) $(ASTEPDIRI) -o $@ -c $<
