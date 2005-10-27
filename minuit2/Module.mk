# Module.mk for minuit2 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 07/05/2003

#MINUIT2INCDIR := $(ROOTSYS)/include
#MINUIT2LIBDIR := $(ROOTSYS)/lib
# MINUITINCDIR := /Users/moneta/mathlibs/Minuit-1_7_1/include
# MINUITLIBDIR := /Users/moneta/mathlibs/Minuit-1_7_1/lib

MODDIR       := minuit2
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MINUIT2DIR    := $(MODDIR)
MINUIT2DIRS   := $(MINUIT2DIR)/src
MINUIT2DIRI   := $(MINUIT2DIR)/inc

##### libMinuit2 #####
MINUIT2L      := $(MODDIRI)/LinkDef.h
MINUIT2DS     := $(MODDIRS)/G__Minuit2.cxx
MINUIT2DO     := $(MINUIT2DS:.cxx=.o)
MINUIT2DH     := $(MINUIT2DS:.cxx=.h)

MINUIT2H      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MINUIT2S      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MINUIT2O      := $(MINUIT2S:.cxx=.o)

MINUIT2DEP    := $(MINUIT2O:.o=.d) $(MINUIT2DO:.o=.d)

MINUIT2LIB    := $(LPATH)/libMinuit2.$(SOEXT)


MINUITBASEVERS     := Minuit-1_7_6
MINUITBASESRCS     := $(MODDIRS)/$(MINUITBASEVERS).tar.gz
MINUITBASEDIRS     := $(MODDIRS)/$(MINUITBASEVERS)
MINUITBASEDIRI     := -I$(MODDIRS)/$(MINUITBASEVERS)
MINUITBASEETAG     := $(MODDIRS)/headers.d


##### liblcg_Minuit #####
ifeq ($(PLATFORM),win32)
MINUITBASELIBA      := $(MINUITBASEDIRS)/lcg_Minuit.lib
MINUITBASELIB       := $(LPATH)/libminuitbase.lib
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
MINUITBASEBLD        = "DEBUG=1"
else
MINUITBASEBLD        = ""
endif
else
MINUITBASELIBA      := $(MINUITBASEDIRS)/src/.libs/liblcg_Minuit.a
MINUITBASELIB       := $(LPATH)/libminuitbase.a
endif
MINUITBASEDEP       := $(MINUITBASELIB)
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
MINUITBASEDBG      = "--enable-gdb"
else
MINUITBASEDBG      =
endif


# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MINUIT2H))
ALLLIBS     += $(MINUIT2LIB)

# include all dependency files
INCLUDEFILES += $(MINUIT2LIBDEP)

##### local rules #####
include/%.h:    $(MINUIT2DIRI)/%.h
		cp $< $@


$(MINUITBASELIB):   $(MINUITBASELIBA)
		cp $< $@
ifeq ($(PLATFORM),macosx)
		ranlib $@
endif
ifeq ($(PLATFORM), win32)
		cp $(MINUITBASEDIRS)/lcg_Minuit.dll $(LPATH)/../bin/libbaseminuit.dll
endif

$(MINUITBASELIBA):  $(MINUITBASESRCS)
		echo "building Minuit library first"
ifeq ($(PLATFORM),win32)
		@(if [ -d $(MINUITBASEDIRS) ]; then \
			rm -rf $(MINUITBASEDIRS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(MINUIT2DIRS); \
		if [ ! -d $(MINUITBASEVERS) ]; then \
			gunzip -c $(MINUITBASEVERS).tar.gz | tar xf -; \
		fi; \
		cd $(MINUITBASEVERS); \
		unset MAKEFLAGS; \
		nmake -f makefile.msc $(MINUITBASEBLD))
#		GNUMAKE=$(MAKE) ./configure $(MINUITBASEDBG) CC=cl LD=cl CFLAGS="$(CFLAGS)" ;  \
#		cd minuit; sed -e 's/ln -s/cp -p/' Makefile > MakefileNew; mv MakefileNew Makefile; cd ../; \
#		$(MAKE)) \
# 		unset MAKEFLAGS; \
# 		nmake -nologo -f minuit.mak \
# 		CFG=$(MINUITBASEBLD))
else
		@(if [ -d $(MINUITBASEDIRS) ]; then \
			rm -rf $(MINUITBASEDIRS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(MINUIT2DIRS); \
		if [ ! -d $(MINUITBASEVERS) ]; then \
			gunzip -c $(MINUITBASEVERS).tar.gz | tar xf -; \
		fi; \
		cd $(MINUITBASEVERS); \
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
		fi; \
		if [ "$(ARCH)" = "linuxppc64gcc" ]; then \
			ACC="gcc -m64"; \
		fi; \
		if [ "$(ARCH)" = "linuxx8664gcc" ]; then \
			ACC="gcc -m64"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure $(MINUITBASEDBG) CXXFLAGS="$(OPTFLAGS) $(CXXFLAGS)";  \
		$(MAKE))
endif

$(MINUIT2LIB):  $(MINUITBASEDEP) $(MINUIT2O) $(MINUIT2DO) $(MAINLIBS) $(MINUITBASELIBDEP)
		@echo "Doing Minuit lib for platform "$(PLATFORM)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMinuit2.$(SOEXT) $@ "$(MINUIT2O) $(MINUIT2DO)" \
		   "$(MINUITLIBEXTRA) $(MINUITBASELIB)" 

$(MINUIT2DS):   $(MINUIT2H) $(MINUIT2L) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MINUITBASEDIRI) $(MINUIT2H) $(MINUIT2L)

$(MINUIT2DO):   $(MINUIT2DS)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(MINUITBASEDIRI) -I. -o $@ -c $< 


all-minuit2:    $(MINUIT2LIB) 

# all-minuit2:    untar-minuit
# 		echo "make all" $(MINUIT2LIB)

test-minuit2: 	$(MINUIT2LIB)
		cd $(MINUIT2DIR)/test; make

clean-minuit2:
		@rm -f $(MINUIT2O) $(MINUIT2DO)
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(MINUITBASEDIRS) ]; then \
			cd $(MINUITBASEDIRS); \
			unset MAKEFLAGS; \
			nmake -nologo -f Makefile.msc clean \
			CFG=$(MINUITBASEBLD); \
		fi)
else
		-@(if [ -d $(MINUITBASEDIRS) ]; then \
			cd $(MINUITBASEDIRS); \
			$(MAKE) clean; \
		fi)
endif

clean::         clean-minuit2

distclean-minuit2: clean-minuit2
		@rm -f $(MINUIT2DEP) $(MINUIT2DS) $(MINUIT2DH) $(MINUIT2LIB)

distclean::     distclean-minuit2
##### extra rules ######

$(MINUIT2O): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(MINUITBASEDIRI) -o $@ -c $< 
