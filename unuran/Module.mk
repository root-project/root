# Module.mk for mathmore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := unuran
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UNURANDIR  := $(MODDIR)
UNURANDIRS := $(UNURANDIR)/src
UNURANDIRI := $(UNURANDIR)/inc

UNRVERS      := unuran-0.8.1-root
UNRSRCS      := $(MODDIRS)/$(UNRVERS).tar.gz
UNRDIRS      := $(MODDIRS)/$(UNRVERS)


###pre-compiled GSL DLL require Unuran to be compiled with -DGSL_DLL 
#ifeq ($(PLATFORM),win32)
#GSLFLAGS += "-DGSL_DLL"
#endif


#UNRLIBDIR := /Users/moneta/mathlibs/Unuran/lib

ifeq ($(PLATFORM),win32)
UNRLIBS := $(MODDIRS)/$(UNRVERS)/src/.libs/libunuran.lib
else
UNRLIBS := $(MODDIRS)/$(UNRVERS)/src/.libs/libunuran.a
endif
#UNRFLAGS := -I$(MODDIRS)/$(UNRVERS)/src -I$(MODDIRS)/$(UNRVERS) -I$(MODDIRS)/$(UNRVERS)/src/utils -DHAVE_CONFIG_H
UNRFLAGS :=  -I$(MODDIRS)/$(UNRVERS)/src 

##### libUnuran #####
UNURANL    := $(MODDIRI)/LinkDef.h
UNURANDS   := $(MODDIRS)/G__Unuran.cxx
UNURANDO   := $(UNURANDS:.cxx=.o)
UNURANDH   := $(UNURANDS:.cxx=.h)
UNURANDH1  := $(MODDIRI)/TUnuran.h

UNURANH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
UNURANS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

UNURANO    := $(UNURANS:.cxx=.o)
#UNURANO    += $(UNURANS:.c=.o)


UNURANDEP  := $(UNURANO:.o=.d) $(UNURANDO:.o=.d)

UNURANLIB  := $(LPATH)/libUnuran.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(UNURANH))
ALLLIBS      += $(UNURANLIB)

# include all dependency files
INCLUDEFILES += $(UNURANDEP) 

##### local rules #####
include/%.h: 	$(UNURANDIRI)/%.h
		cp $< $@

$(UNRLIBS):	$(UNRSRCS)
#ifeq ($(PLATFORM),win32)
#		echo "build unuran"
#		echo $(ARCH)
#		echo $(CC)
#		echo $(CFLAGS)
#else
		echo "build unuran"
		@(if [ -d $(UNRDIRS) ]; then \
			rm -rf $(UNRDIRS); \
		fi; \
		echo "*** Building unuran..."; \
		cd $(UNURANDIRS); \
		if [ ! -d $(UNRVERS) ]; then \
			gunzip -c $(UNRVERS).tar.gz | tar xf -; \
		fi; \
		cd $(UNRVERS); \
		ACC=$(CC); \
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
			ACC="gcc -m64 -fPIC"; \
		fi; \
		if [ "$(ARCH)" = "linuxx8664gcc" ]; then \
			ACC="gcc"; \
			ACFLAGS="-m64 -fPIC"; \
		fi; \
		if [ "$(ARCH)" = "win32" ]; then \
			export LD="cl"; \
			ACC="cl.exe"; \
			ACFLAGS="-MD -G5 -GX"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure --prefix=`pwd`/$(MODDIRS)/$(UNRVERS) CC="$$ACC"  \
		CFLAGS="$$ACFLAGS $(GSLOPT)" $(GSLDBG); \
		$(MAKE))



$(UNURANLIB): $(UNRLIBS) $(UNURANO) $(UNURANDO) $(ORDER_) $(MAINLIBS)
		echo $(UNROBJS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libUnuran.$(SOEXT) $@     \
		   "$(UNURANO) $(UNURANDO)"             \
		   "$(UNURANLIBEXTRA) $(UNRLIBS)"

$(UNURANDS):  $(UNURANDH1) $(UNURANL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(UNRFLAGS) $(UNURANDH1) $(UNURANL)

all-unuran:   $(UNURANLIB)

map-unuran:   $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(UNURANLIB) \
		   -d $(UNURANLIBDEP) -c $(UNURANL) $(UNURANLINC)

map::           map-unuran

clean-unuran:
		@rm -f $(UNURANO) $(UNURANDO)
		-@(if [ -d $(UNRDIRS) ]; then \
			cd $(UNRDIRS); \
			$(MAKE) clean; \
		fi)


clean::         clean-unuran

distclean-unuran: clean-unuran
		@rm -f $(UNURANDEP) $(UNURANDS) $(UNURANDH) $(UNURANLIB)
		@mv $(UNRSRCS) $(UNURANDIRS)/-$(UNRVERS).tar.gz
		@rm -rf $(UNURANDIRS)/$(UNRVERS)
		@mv $(UNURANDIRS)/-$(UNRVERS).tar.gz $(UNRSRCS)

distclean::     distclean-unuran

##### extra rules ######
##$(UNURANO): CFLAGS += $(UNRFLAGS)
$(UNURANO): CXXFLAGS += $(UNRFLAGS) 
