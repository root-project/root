# Module.mk for asimage module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 8/8/2002

MODNAME      := asimage
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ASIMAGEDIR   := $(MODDIR)
ASIMAGEDIRS  := $(ASIMAGEDIR)/src
ASIMAGEDIRI  := $(ASIMAGEDIR)/inc

ASTEPVERS    := libAfterImage
ifeq ($(BUILTINASIMAGE),yes)
ASTEPDIRS    := $(call stripsrc,$(MODDIRS)/$(ASTEPVERS))
ASTEPDIRI    := -I$(ASTEPDIRS)
ASTEPMAKE    := $(ASTEPDIRS)/Makefile
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
ifeq ($(ASPNGINCDIR),)
# needed for the afterimage built-in libpng
ASEXTRALIB   += -lz
endif
ASTEPDEP     := $(ASTEPLIB)
##### To trigger the debug printouts for libafterimage when ROOTBUILD=debug
##### ifeq (debug,$(findstring debug,$(ROOTBUILD)))
##### ASTEPDBG      = "--enable-gdb"
##### else
ASTEPDBG      =
##### endif
else
ASTEPLIBA    := $(ASLIBDIR) $(ASLIB)
ASTEPLIB     := $(ASLIBDIR) $(ASLIB)
ASTEPDEP     :=
endif

##### libASImage #####
ASIMAGEL     := $(MODDIRI)/LinkDef.h
ASIMAGEDS    := $(call stripsrc,$(MODDIRS)/G__ASImage.cxx)
ASIMAGEDO    := $(ASIMAGEDS:.cxx=.o)
ASIMAGEDH    := $(ASIMAGEDS:.cxx=.h)

ASIMAGEH     := $(MODDIRI)/TASImage.h $(MODDIRI)/TASImagePlugin.h \
                $(MODDIRI)/TASPluginGS.h
ASIMAGES     := $(MODDIRS)/TASImage.cxx $(MODDIRS)/TASPluginGS.cxx
ASIMAGEO     := $(call stripsrc,$(ASIMAGES:.cxx=.o))

ASIMAGEDEP   := $(ASIMAGEO:.o=.d) $(ASIMAGEDO:.o=.d)

ASIMAGELIB   := $(LPATH)/libASImage.$(SOEXT)
ASIMAGEMAP   := $(ASIMAGELIB:.$(SOEXT)=.rootmap)

##### libASImageGui #####
ASIMAGEGUIL  := $(MODDIRI)/LinkDefGui.h
ASIMAGEGUIDS := $(call stripsrc,$(MODDIRS)/G__ASImageGui.cxx)
ASIMAGEGUIDO := $(ASIMAGEGUIDS:.cxx=.o)
ASIMAGEGUIDH := $(ASIMAGEGUIDS:.cxx=.h)

ASIMAGEGUIH  := $(MODDIRI)/TASPaletteEditor.h
ASIMAGEGUIS  := $(MODDIRS)/TASPaletteEditor.cxx
ASIMAGEGUIO  := $(call stripsrc,$(ASIMAGEGUIS:.cxx=.o))

ASIMAGEGUIDEP := $(ASIMAGEGUIO:.o=.d) $(ASIMAGEGUIDO:.o=.d)

ASIMAGEGUILIB := $(LPATH)/libASImageGui.$(SOEXT)
ASIMAGEGUIMAP := $(ASIMAGEGUILIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ASIMAGEGUIH))
ALLLIBS     += $(ASIMAGELIB) $(ASIMAGEGUILIB)
ALLMAPS     += $(ASIMAGEMAP) $(ASIMAGEGUIMAP)

# include all dependency files
INCLUDEFILES += $(ASIMAGEDEP) $(ASIMAGEGUIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ASIMAGEDIRI)/%.h
		cp $< $@

ifeq ($(BUILTINASIMAGE),yes)
$(ASTEPLIB):    $(ASTEPLIBA)
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)

$(ASTEPMAKE):
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		$(MAKEDIR)
		@$(RSYNC) --exclude '.svn' --exclude '*.o' --exclude '*.a' $(ASIMAGEDIRS)/$(ASTEPVERS) $(call stripsrc,$(ASIMAGEDIRS))
endif
ifeq ($(PLATFORM),win32)
		@touch $(ASTEPMAKE)
else
		@(cd $(ASTEPDIRS); \
		ACC="$(CC)"; \
		ACFLAGS="-O"; \
		if [ "$(CC)" = "icc" ]; then \
			ACC="icc"; \
			ACFLAGS="$$ACFLAGS -wd188 -wd869 -wd2259 -wd1418 -wd1419 -wd593 -wd981 -wd1599 -wd181 -wd177 -wd1572"; \
		fi; \
		if [ "$(ARCH)" = "solarisCC5" ]; then \
			ACFLAGS="$$ACFLAGS -erroff=E_WHITE_SPACE_IN_DIRECTIVE"; \
		fi; \
		if [ "$(ARCH)" = "solaris64CC5" ]; then \
			ACC="$$ACC -m64"; \
			ACFLAGS="$$ACFLAGS -KPIC -erroff=E_WHITE_SPACE_IN_DIRECTIVE"; \
		fi; \
		if [ "$(ARCH)" = "sgicc64" ]; then \
			ACC="$$ACC -mabi=64"; \
		fi; \
		if [ "$(ARCH)" = "hpuxia64acc" ]; then \
			ACC="$$ACC +DD64 -Ae +W863"; \
		fi; \
		if [ "$(ARCH)" = "macosx" ]; then \
			ACC="$$ACC -m32"; \
		fi; \
		if [ "$(ARCH)" = "macosx64" ]; then \
			ACC="$$ACC -m64"; \
		fi; \
		if [ "$(ARCH)" = "linuxppc64gcc" ]; then \
			ACC="$$ACC -m64"; \
		fi; \
		if [ "$(ARCH)" = "linux" ]; then \
			ACC="$$ACC -m32"; \
		fi; \
		if [ "$(ARCH)" = "linuxx8664gcc" ]; then \
			ACC="$$ACC -m64"; \
			MMX="--enable-mmx-optimization=no"; \
		fi; \
		if [ "$(ARCH)" = "linuxicc" ]; then \
			ACC="$$ACC -m32"; \
		fi; \
		if [ "$(ARCH)" = "linuxx8664icc" ]; then \
			ACC="$$ACC -m64"; \
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
		fi; \
		if [ "$(FREETYPEDIRI)" != "" ]; then \
			TTFINCDIR="--with-ttf-includes=-I../../../../$(FREETYPEDIRI)"; \
		fi; \
		GNUMAKE=$(MAKE) CC=$$ACC CFLAGS=$$ACFLAGS \
		./configure \
		--with-ttf $$TTFINCDIR \
		--with-afterbase=no \
		--without-svg \
		--disable-glx \
		$$MMX \
		$(ASTEPDBG) \
		--with-builtin-ungif \
		$$GIFINCDIR \
		--with-jpeg \
		$$JPEGINCDIR \
		--with-png \
		$$PNGINCDIR \
		$$TIFFINCDIR)
endif

$(ASTEPLIBA):   $(ASTEPMAKE) $(wildcard $(ASTEPDIRS)/*.h) $(wildcard $(ASTEPDIRS)/*.c)
ifeq ($(PLATFORM),win32)
		@(cd $(ASTEPDIRS); \
		echo "*** Building libAfterImage ..." ; \
		unset MAKEFLAGS; \
		nmake FREETYPEDIRI=-I../../../../$(FREETYPEDIRI) \
                -nologo -f libAfterImage.mak \
		CFG=$(ASTEPBLD) NMAKECXXFLAGS="$(BLDCXXFLAGS) -I$(shell cygpath -w $(ROOT_SRCDIR))/build/win -FIw32pragma.h /wd4244")
else
		@(cd $(ASTEPDIRS); \
		echo "*** Building libAfterImage ..." ; \
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

$(ASIMAGEDS):   $(ASIMAGEH) $(ASIMAGEL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEH) $(ASIMAGEL)

$(ASIMAGEMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(ASIMAGEL)
		$(RLIBMAP) -o $@ -l $(ASIMAGELIB) \
		   -d $(ASIMAGELIBDEPM) -c $(ASIMAGEL)

##### libASImageGui #####
$(ASIMAGEGUILIB):  $(ASIMAGEGUIO) $(ASIMAGEGUIDO) $(ASTEPDEP) $(FREETYPEDEP) \
                   $(ORDER_) $(MAINLIBS) $(ASIMAGEGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libASImageGui.$(SOEXT) $@ \
		   "$(ASIMAGEGUIO) $(ASIMAGEGUIDO)" \
		   "$(ASIMAGEGUILIBEXTRA) $(ASTEPLIB) \
                    $(FREETYPELDFLAGS) $(FREETYPELIB) \
		    $(ASEXTRALIBDIR) $(ASEXTRALIB) $(XLIBS)"

$(ASIMAGEGUIDS): $(ASIMAGEGUIH) $(ASIMAGEGUIL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ASIMAGEGUIH) $(ASIMAGEGUIL)

$(ASIMAGEGUIMAP): $(RLIBMAP) $(MAKEFILEDEP) $(ASIMAGEGUIL)
		$(RLIBMAP) -o $(ASIMAGEGUIMAP) -l $(ASIMAGEGUILIB) \
		   -d $(ASIMAGEGUILIBDEPM) -c $(ASIMAGEGUIL)

all-$(MODNAME): $(ASIMAGELIB) $(ASIMAGEGUILIB) $(ASIMAGEMAP) $(ASIMAGEGUIMAP)

clean-$(MODNAME):
		@rm -f $(ASIMAGEO) $(ASIMAGEDO) $(ASIMAGEGUIO) $(ASIMAGEGUIDO)
ifeq ($(BUILTINASIMAGE),yes)
ifeq ($(PLATFORM),win32)
		@(if [ -f $(ASTEPMAKE) ]; then \
			cd $(ASTEPDIRS); \
			unset MAKEFLAGS; \
			nmake -nologo -f libAfterImage.mak clean \
			CFG=$(ASTEPBLD); \
		fi)
else
		@(if [ -f $(ASTEPMAKE) ]; then \
			cd $(ASTEPDIRS); \
			$(MAKE) clean; \
		fi)
endif
endif

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ASIMAGEDEP) $(ASIMAGEDS) $(ASIMAGEDH) \
		   $(ASIMAGELIB) $(ASIMAGEMAP) \
		   $(ASIMAGEGUIDEP) $(ASIMAGEGUIDS) $(ASIMAGEGUIDH) \
		   $(ASIMAGEGUILIB) $(ASIMAGEGUIMAP)
ifeq ($(BUILTINASIMAGE),yes)
		@rm -f $(ASTEPLIB)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(ASTEPDIRS)
else
ifeq ($(PLATFORM),win32)
		@(if [ -f $(ASTEPMAKE) ]; then \
			cd $(ASTEPDIRS); \
			unset MAKEFLAGS; \
			nmake -nologo -f libAfterImage.mak distclean \
			CFG=$(ASTEPBLD); \
		fi)
else
		@(if [ -f $(ASTEPMAKE) ]; then \
			cd $(ASTEPDIRS); \
			$(MAKE) distclean; \
		fi)
endif
endif
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(ASIMAGEO): $(ASTEPDEP) $(FREETYPEDEP)
$(ASIMAGEO): CXXFLAGS += $(FREETYPEINC) $(ASTEPDIRI)

$(ASIMAGEGUIO) $(ASIMAGEGUIDO) $(ASIMAGEDO): $(ASTEPDEP)
$(ASIMAGEGUIO) $(ASIMAGEGUIDO) $(ASIMAGEDO): CXXFLAGS += $(ASTEPDIRI)
