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
ASTEPSRCS    := $(MODDIRS)/$(ASTEPVERS).tar.gz
ifeq ($(BUILTINASIMAGE),yes)
ASTEPDIRS    := $(MODDIRS)/$(ASTEPVERS)
ASTEPDIRI    := -I$(MODDIRS)/$(ASTEPVERS)
ASTEPETAG    := $(MODDIRS)/headers.d
else
ASTEPDIRS    :=
ASTEPDIRI    := $(ASINCDIR:%=-I%)
endif

##### libAfterImage #####
ifeq ($(BUILTINASIMAGE),yes)
ifeq ($(PLATFORM),win32)
ASTEPLIBA    := $(ASTEPDIRS)/libAfterImage.lib
ASTEPLIB     := $(LPATH)/libAfterImage.lib
ifeq (yes,$(WINRTDEBUG))
ASTEPBLD      = "libAfterImage - Win32 Debug"
else
ASTEPBLD      = "libAfterImage - Win32 Release"
endif
else
ASTEPLIBA    := $(ASTEPDIRS)/libAfterImage.a
ASTEPLIB     := $(LPATH)/libAfterImage.a
endif
ASTEPDEP     := $(ASTEPLIB)
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
ASTEPDBG      = "--enable-gdb"
else
ASTEPDBG      =
endif
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

ASIMAGEH     := $(MODDIRI)/TASImage.h $(MODDIRI)/TASImagePlugin.h
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

##### libASPluginGS #####
ASIMAGEGSL  := $(MODDIRI)/LinkDefGS.h
ASIMAGEGSDS := $(MODDIRS)/G__ASImageGS.cxx
ASIMAGEGSDO := $(ASIMAGEGSDS:.cxx=.o)
ASIMAGEGSDH := $(ASIMAGEGSDS:.cxx=.h)

ASIMAGEGSH  := $(MODDIRI)/TASPluginGS.h
ASIMAGEGSS  := $(MODDIRS)/TASPluginGS.cxx
ASIMAGEGSO  := $(ASIMAGEGSS:.cxx=.o)

ASIMAGEGSDEP := $(ASIMAGEGSO:.o=.d) $(ASIMAGEGSDO:.o=.d)

ASIMAGEGSLIB := $(LPATH)/libASPluginGS.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEGUIH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEGSH))
ALLLIBS     += $(ASIMAGELIB) $(ASIMAGEGUILIB) $(ASIMAGEGSLIB)

# include all dependency files
INCLUDEFILES += $(ASIMAGEDEP) $(ASIMAGEGUIDEP) $(ASIMAGEGSDEP)

##### local rules #####
include/%.h:    $(ASIMAGEDIRI)/%.h $(ASTEPETAG)
		cp $< $@

ifeq ($(BUILTINASIMAGE),yes)
$(ASTEPLIB):    $(ASTEPLIBA)
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)

$(ASTEPETAG):   $(ASTEPSRCS)
		@(if [ -d $(ASTEPDIRS) ]; then \
		   rm -rf $(ASTEPDIRS); \
		fi; \
		echo "*** Extracting libAfterimage source ..."; \
		cd $(ASIMAGEDIRS); \
		if [ ! -d $(ASTEPVERS) ]; then \
		   gunzip -c $(ASTEPVERS).tar.gz | tar xf -; \
		   etag=`basename $(ASTEPETAG)` ; \
		   touch $$etag ; \
		fi)

$(ASTEPLIBA):   $(ASTEPETAG)
ifeq ($(PLATFORM),win32)
		@(cd $(ASTEPDIRS); \
		unset MAKEFLAGS; \
		nmake FREETYPEDIRI=-I../../../$(FREETYPEDIRI) \
                -nologo -f libAfterImage.mak \
		CFG=$(ASTEPBLD) NMAKECXXFLAGS="$(BLDCXXFLAGS) -I../../../build/win -FIw32pragma.h")
else
		@(cd $(ASTEPDIRS); \
		ACC=$(CC); \
		ACFLAGS="-O"; \
		if [ "$(CC)" = "icc" ]; then \
			ACC="icc"; \
		fi; \
		if [ "$(ARCH)" = "solarisCC5" ]; then \
			ACFLAGS += " -erroff=E_WHITE_SPACE_IN_DIRECTIVE"; \
		fi; \
		if [ "$(ARCH)" = "sgicc64" ]; then \
			ACC="gcc -mabi=64"; \
		fi; \
		if [ "$(ARCH)" = "hpuxia64acc" ]; then \
			ACC="cc +DD64 -Ae +W863"; \
			ACCALT="gcc -mlp64"; \
		fi; \
		if [ "$(ARCH)" = "macosx64" ]; then \
			ACC="gcc -m64"; \
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
		if [ "$(ASTIFFINCDIR)" = "--with-tiff=no" ]; then \
			TIFFINCDIR="$(ASTIFFINCDIR)"; \
		elif [ "$(ASTIFFINCDIR)" != "" ]; then \
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
		$(ASTEPDBG) \
		$(NOUNGIF) \
		--with-jpeg \
		$$JPEGINCDIR \
		--with-png \
		$$PNGINCDIR \
		$$TIFFINCDIR \
		$$GIFINCDIR; \
		$(MAKE))
endif
endif

##### libASImage #####
$(ASIMAGELIB):  $(ASIMAGEO) $(ASIMAGEDO) $(ASTEPDEP) $(FREETYPEDEP) \
                $(ORDER_) $(MAINLIBS) $(ASIMAGELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImage.$(SOEXT) $@ \
		   "$(ASIMAGEO) $(ASIMAGEDO)" \
		   "$(ASIMAGELIBEXTRA) $(ASTEPLIB) \
                    $(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(ASEXTRALIBDIR) $(ASEXTRALIB) $(XLIBS)"

$(ASIMAGEDS):   $(ASIMAGEH) $(ASIMAGEL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEH) $(ASIMAGEL)

##### libASImageGui #####
$(ASIMAGEGUILIB):  $(ASIMAGEGUIO) $(ASIMAGEGUIDO) $(ASTEPDEP) $(FREETYPEDEP) \
                   $(ORDER_) $(MAINLIBS) $(ASIMAGEGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImageGui.$(SOEXT) $@ \
		   "$(ASIMAGEGUIO) $(ASIMAGEGUIDO)" \
		   "$(ASIMAGEGUILIBEXTRA) $(ASTEPLIB) \
                    $(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(ASEXTRALIBDIR) $(ASEXTRALIB) $(XLIBS)"

$(ASIMAGEGUIDS): $(ASIMAGEGUIH) $(ASIMAGEGUIL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEGUIH) $(ASIMAGEGUIL)

##### libASPluginGS #####
$(ASIMAGEGSLIB):  $(ASIMAGEGSO) $(ASIMAGEGSDO) $(ASTEPDEP) $(FREETYPEDEP) \
                  $(ORDER_) $(MAINLIBS) $(ASIMAGEGSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASPluginGS.$(SOEXT) $@ \
		   "$(ASIMAGEGSO) $(ASIMAGEGSDO)" \
		   "$(ASIMAGEGSLIBEXTRA) $(ASTEPLIB) \
                    $(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(ASEXTRALIBDIR) $(ASEXTRALIB) $(XLIBS)"

$(ASIMAGEGSDS): $(ASIMAGEGSH) $(ASIMAGEGSL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEGSH) $(ASIMAGEGSL)

all-asimage:    $(ASIMAGELIB) $(ASIMAGEGUILIB) $(ASIMAGEGSLIB)

map-asimage:    $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(ASIMAGELIB) \
		   -d $(ASIMAGELIBDEP) -c $(ASIMAGEL)

map-asimagegui: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(ASIMAGEGUILIB) \
		   -d $(ASIMAGEGUILIBDEP) -c $(ASIMAGEGUIL)

map-asimagegs: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(ASIMAGEGSLIB) \
		   -d $(ASIMAGEGSLIBDEP) -c $(ASIMAGEGSL)

map::           map-asimage map-asimagegui map-asimagegs

clean-asimage:
		@rm -f $(ASIMAGEO) $(ASIMAGEDO) $(ASIMAGEGUIO) $(ASIMAGEGUIDO) \
		   $(ASIMAGEGSO) $(ASIMAGEGSDO)
ifeq ($(BUILTINASIMAGE),yes)
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(ASTEPDIRS) ]; then \
			cd $(ASTEPDIRS); \
			unset MAKEFLAGS; \
			nmake -nologo -f libAfterImage.mak clean \
			CFG=$(ASTEPBLD); \
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
		   $(ASIMAGEGUILIB) \
		   $(ASIMAGEGSDEP) $(ASIMAGEGSDS) $(ASIMAGEGSDH) \
		   $(ASIMAGEGSLIB)
ifeq ($(BUILTINASIMAGE),yes)
		@rm -f $(ASTEPLIB) $(ASTEPETAG)
		@rm -rf $(ASIMAGEDIRS)/$(ASTEPVERS)
endif

distclean::     distclean-asimage

##### extra rules ######
$(ASIMAGEO): $(ASTEPLIB) $(FREETYPEDEP)
$(ASIMAGEO): CXXFLAGS += $(FREETYPEINC) $(ASTEPDIRI)

$(ASIMAGEGUIO) $(ASIMAGEGSO) $(ASIMAGEGSDO) $(ASIMAGEGUIDO) $(ASIMAGEDO): \
  $(ASTEPLIB)
$(ASIMAGEGUIO) $(ASIMAGEGSO) $(ASIMAGEGSDO) $(ASIMAGEGUIDO) $(ASIMAGEDO): \
  CXXFLAGS += $(ASTEPDIRI)
