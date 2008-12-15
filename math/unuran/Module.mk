# Module.mk for mathmore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := unuran
MODDIR       := math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UNURANDIR    := $(MODDIR)
UNURANDIRS   := $(UNURANDIR)/src
UNURANDIRI   := $(UNURANDIR)/inc

UNRVERS      := unuran-1.3.0-root

UNRSRCS      := $(MODDIRS)/$(UNRVERS).tar.gz
UNRDIRS      := $(MODDIRS)/$(UNRVERS)
UNRALLDIRS   := $(MODDIRS)/unuran-*-root
UNURANETAG   := $(UNURANDIRS)/headers.d
UNRCFG       := $(UNURANDIRS)/$(UNRVERS)/config.h

UNRS 	     := $(wildcard $(UNRDIRS)/src/utils/*.c)\
                $(wildcard $(UNRDIRS)/src/methods/*.c) \
                $(wildcard $(UNRDIRS)/src/specfunct/*.c) \
                $(wildcard $(UNRDIRS)/src/distr/*.c) \
                $(wildcard $(UNRDIRS)/src/distributions/*.c) \
                $(wildcard $(UNRDIRS)/src/parser/*.c) \
                $(wildcard $(UNRDIRS)/src/tests/*.c) \
                $(wildcard $(UNRDIRS)/src/uniform/*.c) \
                $(wildcard $(UNRDIRS)/src/urng/*.c)
UNRO         := $(UNRS:.c=.o)

ifeq ($(PLATFORM),win32)
UNRLIBS      := $(MODDIRS)/$(UNRVERS)/src/.libs/libunuran.lib
else
UNRLIBS      := $(MODDIRS)/$(UNRVERS)/src/.libs/libunuran.a
endif

UNRFLAGS     :=  -I$(MODDIRS)/$(UNRVERS)/src

##### libUnuran #####
UNURANL      := $(MODDIRI)/LinkDef.h
UNURANDS     := $(MODDIRS)/G__Unuran.cxx
UNURANDO     := $(UNURANDS:.cxx=.o)
UNURANDH     := $(UNURANDS:.cxx=.h)
UNURANDH1    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))


UNURANH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
UNURANS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

UNURANO      := $(UNURANS:.cxx=.o)

UNURANDEP    := $(UNURANO:.o=.d) $(UNURANDO:.o=.d)

UNURANLIB    := $(LPATH)/libUnuran.$(SOEXT)
UNURANMAP    := $(UNURANLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(UNURANH))
ALLLIBS      += $(UNURANLIB)
ALLMAPS      += $(UNURANMAP)

# include all dependency files
INCLUDEFILES += $(UNURANDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h: 	$(UNURANDIRI)/%.h $(UNURANETAG)
		cp $< $@

# force untar in first pass, so that in second pass UNRS is correctly set
ifeq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean),)
-include $(UNURANETAG)
endif
$(UNURANDEP):   $(UNRCFG)

$(UNURANETAG):	$(UNRSRCS)
		echo "** untar unuran"
		@(if  [ -d $(UNRDIRS) ]; then \
			rm -rf $(UNRDIRS); \
		fi; \
		cd $(UNURANDIRS); \
		rm -rf unuran*root; \
		if [ ! -d $(UNRVERS) ]; then \
			gunzip -c $(UNRVERS).tar.gz | tar xf -; \
		   etag=`basename $(UNURANETAG)` ; \
		   touch $$etag ; \
		fi); 
#configure unuran (required for creating the config.h used by unuran source files)
$(UNRCFG):	$(UNRANETAG)
		@echo "** configure unuran"
		@(cd $(UNURANDIRS)/$(UNRVERS) ; \
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
		GNUMAKE=$(MAKE) ./configure  CC="$$ACC"  \
		CFLAGS="$$ACFLAGS");

$(UNURANLIB):   $(UNRCFG) $(UNRO) $(UNURANO) $(UNURANDO) $(ORDER_) $(MAINLIBS) $(UNURANLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libUnuran.$(SOEXT) $@     \
		   "$(UNURANO) $(UNURANDO)"             \
		   "$(UNURANLIBEXTRA) $(UNRO)"

$(UNURANDS):    $(UNRINIT) $(UNURANDH1) $(UNURANL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(UNRFLAGS) $(UNURANDH1) $(UNURANL)

$(UNURANMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(UNURANL) $(UNURANLINC)
		$(RLIBMAP) -o $(UNURANMAP) -l $(UNURANLIB) \
		   -d $(UNURANLIBDEPM) -c $(UNURANL) $(UNURANLINC)

all-$(MODNAME): $(UNURANLIB) $(UNURANMAP)

clean-$(MODNAME):
		@rm -f $(UNURANO) $(UNURANDO)
		-@(if [ -d $(UNRDIRS) ]; then \
			cd $(UNRDIRS); \
			$(MAKE) clean; \
		fi)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): 
		@rm -f $(UNURANO) $(UNURANDO) $(UNURANETAG) $(UNURANDEP) $(UNURANDS) $(UNURANDH) $(UNURANLIB) $(UNURANMAP)
ifneq ($(wildcard $(UNRSRCS)), )
		@mv $(UNRSRCS) $(UNURANDIRS)/-$(UNRVERS).tar.gz
endif
ifeq ($(UNURKEEP),yes)
		@mv $(UNURANDIRS)/$(UNRVERS) $(UNURANDIRS)/$(UNRVERS).keep
endif
		@rm -rf $(UNURANDIRS)/unuran-*-root
ifeq ($(UNURKEEP),yes)
		@mv $(UNURANDIRS)/$(UNRVERS).keep $(UNURANDIRS)/$(UNRVERS)
endif
ifneq ($(wildcard $(UNRSRCS)), )
		@mv $(UNURANDIRS)/-$(UNRVERS).tar.gz $(UNRSRCS)
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######

$(UNURANO): CXXFLAGS += $(UNRFLAGS)

ifeq ($(PLATFORM),win32)
$(UNRO): CFLAGS := $(filter-out -FIsehmap.h,$(filter-out -Iinclude,$(CFLAGS) -I$(UNRDIRS) -I$(UNRDIRS)/src/ -I$(UNRDIRS)/src/utils -DHAVE_CONFIG_H))
else
$(UNRO): CFLAGS := $(filter-out -Iinclude,$(CFLAGS) -I$(UNRDIRS) -I$(UNRDIRS)/src/ -I$(UNRDIRS)/src/utils -DHAVE_CONFIG_H)
endif
ifeq ($(CC),icc)
$(UNRO): CFLAGS += -mp
endif
