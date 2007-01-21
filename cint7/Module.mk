# Module.mk for cint module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR        := cint7
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

CINT7DIR      := $(MODDIR)
CINT7DIRS     := $(CINT7DIR)/src
CINT7DIRI     := $(CINT7DIR)/inc
CINT7DIRM     := $(CINT7DIR)/main
CINT7DIRT     := $(CINT7DIR)/tool
CINT7DIRL     := $(CINT7DIR)/lib
CINT7DIRDLLS  := $(CINT7DIR)/include
CINT7DIRSTL   := $(CINT7DIR)/stl
CINT7DIRDLLSTL:= $(CINT7DIRL)/dll_stl

##### libCint #####
CINT7H1       := $(wildcard $(CINT7DIRS)/*.h)
CINT7H2       := $(wildcard $(CINT7DIR)/G__ci.h $(CINT7DIR)/G__ci_fproto.h $(CINT7DIR)/G__security.h)
CINT7H1T      := $(patsubst $(CINT7DIRS)/%.h,include/%.h,$(CINT7H1))
CINT7H2T      := $(patsubst $(CINT7DIR)/%.h,include/%.h,$(CINT7H2))
CINT7S1       := $(wildcard $(MODDIRS)/*.c)
CINT7S2       := $(filter-out $(MODDIRS)/v6_dmy%,$(wildcard $(MODDIRS)/*.cxx))

CINT7S1       += $(CINT7DIRM)/G__setup.c

CINT7ALLO     := $(CINT7S1:.c=.o) $(CINT7S2:.cxx=.o)
CINT7ALLDEP   := $(CINT7ALLO:.o=.d)

CINT7CONF     := $(CINT7DIRI)/configcint.h
CINT7CONFMK   := $(MODDIR)/configcint.mk

CINT7S1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINT7S1))

CINT7S2       := $(filter-out $(MODDIRS)/v6_sunos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/v6_macos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/v6_winnt.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/v6_newsos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/v6_loadfile_tmp.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/v6_pragma_tmp.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/allstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/kccstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/sunstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/sun5strm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/gcc3strm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/longif3.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/accstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/iccstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/fakestrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/vcstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/vc7strm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/bcstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/vcstrmold.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/alphastrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/Apiifold.%,$(CINT7S2))

# strip off possible leading path from compiler command name
CXXCMD       := $(shell echo $(CXX) | sed s/".*\/"//)

ifeq ($(CXXCMD),KCC)
CINT7S2       += $(MODDIRS)/kccstrm.cxx
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
CINT7S2       += $(MODDIRS)/longif3.cxx
else
ifeq ($(PLATFORM),linux)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),hurd)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),fbsd)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),obsd)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),hpux)
ifeq ($(ARCH),hpuxia64acc)
CINT7S2       += $(MODDIRS)/accstrm.cxx
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
CINT7S2       += $(MODDIRS)/longif3.cxx
else
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),solaris)
ifeq ($(SUNCC5),true)
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
CINT7S2       += $(MODDIRS)/longif3.cxx
ifeq ($(findstring $(CXXFLAGS),-library=iostream,no%Cstd),)
CINT7S2       += $(MODDIRS)/sunstrm.cxx
#CINT7S2       += $(MODDIRS)/sun5strm.cxx
else
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
else
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),aix3)
CINT7S1       += $(MODDIRS)/dlfcn.c
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix5)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),sgi)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),alpha)
CINT7S2       += $(MODDIRS)/alphastrm.cxx
endif
ifeq ($(PLATFORM),alphagcc)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),sunos)
CINT7S1       += $(MODDIRS)/sunos.c
endif
ifeq ($(PLATFORM),macos)
CINT7S2       += $(MODDIRS)/v6_macos.cxx
CINT7S2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),macosx)
CINT7S2       += $(MODDIRS)/libstrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINT7S2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINT7S2       += $(MODDIRS)/v6_winnt.cxx
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
CINT7S2       += $(MODDIRS)/longif3.cxx
ifeq ($(VC_MAJOR),13)
 ifeq ($(VC_MINOR),10)
  CINT7S2       += $(MODDIRS)/vc7strm.cxx
 else
  CINT7S2       += $(MODDIRS)/iccstrm.cxx
 endif
else
 ifeq ($(VC_MAJOR),14)
  CINT7S2       += $(MODDIRS)/vc7strm.cxx
 else
  CINT7S2       += $(MODDIRS)/iccstrm.cxx
 endif
endif
endif
ifeq ($(PLATFORM),vms)
CINT7S2       += $(MODDIRS)/fakestrm.cxx
endif
ifeq ($(CXXCMD),icc)
CINT7S2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
ifneq ($(ICC_GE_9),)
CINT7S2       += $(MODDIRS)/gcc3strm.cxx
else
CINT7S2       += $(MODDIRS)/iccstrm.cxx
endif
CINT7S2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),3)
CINT7S2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
CINT7S2       += $(MODDIRS)/gcc3strm.cxx
CINT7S2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),4)
CINT7S2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
CINT7S2       += $(MODDIRS)/gcc3strm.cxx
CINT7S2       += $(MODDIRS)/longif3.cxx
endif
ifeq ($(CXXCMD),xlC)
ifeq ($(PLATFORM),macosx)
CINT7S2       := $(filter-out $(MODDIRS)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/longif.%,$(CINT7S2))
CINT7S2       += $(MODDIRS)/gcc3strm.cxx
CINT7S2       += $(MODDIRS)/longif3.cxx
endif
endif

CINT7S        := $(CINT7S1) $(CINT7S2)
CINT7O        := $(CINT7S1:.c=.o) $(CINT7S2:.cxx=.o)
CINT7TMPO     := $(subst v6_loadfile.o,v6_loadfile_tmp.o,$(CINT7O))
CINT7TMPO     := $(subst v6_pragma.o,v6_pragma_tmp.o,$(CINT7TMPO))
CINT7TMPINC   := -I$(MODDIR)/inc -I$(MODDIR)/include -I$(MODDIR)/stl -I$(MODDIR)/lib -Iinclude
CINT7DEP      := $(CINT7O:.o=.d)
CINT7DEP      += $(MODDIRS)/v6_loadfile_tmp.d
CINT7DEP      += $(MODDIRS)/v6_loadfile_tmp.d
CINT7ALLDEP   += $(MODDIRS)/v6_pragma_tmp.d
CINT7ALLDEP   += $(MODDIRS)/v6_pragma_tmp.d

CINT7LIB      := $(LPATH)/libCint7.$(SOEXT)

##### cint #####
CINT7EXES     := $(CINT7DIRM)/cppmain.cxx
CINT7EXEO     := $(CINT7EXES:.cxx=.o)
CINT7EXEDEP   := $(CINT7EXEO:.o=.d)
CINT7TMP      := $(CINT7DIRM)/cint7_tmp$(EXEEXT)
CINT7         := bin/cint7$(EXEEXT)

##### makecint #####
MAKECINT7S    := $(CINT7DIRT)/makecint.cxx
MAKECINT7O    := $(MAKECINT7S:.cxx=.o)
MAKECINT7     := bin/makecint7$(EXEEXT)

##### iosenum.h #####
IOSENUM7      := $(MODDIR)/include/iosenum.h
IOSENUM7C     := $(MODDIR)/include/iosenum.cxx
ifeq ($(GCC_MAJOR),4)
IOSENUM7A     := $(MODDIR)/include/iosenum.$(ARCH)3
else
ifeq ($(GCC_MAJOR),3)
IOSENUM7A     := $(MODDIR)/include/iosenum.$(ARCH)3
else
IOSENUM7A     := $(MODDIR)/include/iosenum.$(ARCH)
endif
endif

CINT7_STDIOH   := $(MODDIR)/include/stdio.h
CINT7_MKINCLD  := $(MODDIR)/include/mkincld
CINT7_MKINCLDS := $(MODDIR)/include/mkincld.c
CINT7_MKINCLDO := $(MODDIR)/include/mkincld.o

# used in the main Makefile
#ALLHDRS     += $(patsubst $(CINT7DIRS)/%.h,include/%.h,$(CINT7H1))
#ALLHDRS     += $(patsubst $(CINT7DIR)/%.h,include/%.h,$(CINT7H2))
#ALLHDRS     += $(CINT7CONF)

ALLLIBS      += $(CINT7LIB)
ALLEXECS     += $(CINT7) $(MAKECINT7) $(CINT7TMP)

# include all dependency files
INCLUDEFILES += $(CINT7DEP) $(CINT7EXEDEP)

CINT7CXXFLAGS := $(filter-out -Iinclude -DG__REGEXP,$(CINTCXXFLAGS))
CINT7CFLAGS := $(filter-out -Iinclude -DG__REGEXP,$(CINTCFLAGS))

CINT7CXXFLAGS += -DG__CINTBODY -DG__HAVE_CONFIG -DG__NOMAKEINFO
CINT7CFLAGS += -DG__CINTBODY -DG__HAVE_CONFIG -DG__NOMAKEINFO

CINT7CXXFLAGS += -I$(CINT7DIRI) -I$(CINT7DIRS) -I$(CINT7DIR)/reflex/inc -Iinclude
CINT7CFLAGS += -I$(CINT7DIRI) -I$(CINT7DIRS) -I$(CINT7DIR)/reflex/inc -Iinclude

ifeq ($(PLATFORM),win32)
REFLEXLL := lib/libReflex.lib
else
REFLEXLL := -Llib -lReflex
endif

ifneq ($(PLATFORM),fbsd) 
ifneq ($(PLATFORM),obsd)
REFLEXLL   += -ldl 
endif 
endif

##### local rules #####
$(CINT7LIB): $(CINT7O) $(CINT7LIBDEP) $(REFLEXLIB)
	$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" \
	   libCint7.$(SOEXT) $@ "$(CINT7O)" "$(CINT7LIBEXTRA) $(REFLEXLL)"

$(CINT7): $(CINT7EXEO) $(CINT7LIB) $(REFLEXLIB)
	$(LD) $(LDFLAGS) -o $@ $(CINT7EXEO) $(RPATH) $(CINT7LIBS) $(CILIBS)

#From cint7:
#$(CINTTMP): $(SETUPO) $(MAINO) $(G__CFG_READLINELIB) $(CINTTMPOBJ) $(REFLEXLIBDEP)
#        @echo "Linking $@"
#        $(CMDECHO)$(G__CFG_LD) $(G__CFG_LDFLAGS) $(G__CFG_LDOUT)$@ \
#          $(SETUPO) $(MAINO) $(CINTTMPOBJ) $(REFLEXLINK) \
#          $(G__CFG_READLINELIB) $(G__CFG_CURSESLIB) $(G__CFG_DEFAULTLIBS)

$(CINT7TMP) : $(CINT7EXEO) $(CINT7TMPO) $(REFLEXLIB)
	$(LD) $(LDFLAGS) -o $@ $(CINT7EXEO) $(CINT7TMPO) $(RPATH) \
	   $(REFLEXLL) $(CILIBS)

$(MAKECINT7) : $(MAKECINT7O)
	$(LD) $(LDFLAGS) -o $@ $(MAKECINT7O)

$(IOSENUM7) : $(IOSENUM7A)
	cp $< $@

$(IOSENUM7A) : $(CINT7TMP) $(CINT7_STDIOH)
	@(if test ! -r $@ ; \
	  then \
	    $(CINT7TMP) $(CINT7TMPINC) $(IOSENUM7C) > /dev/null ; \
	    mv iosenum.h $@ ; \
	  else \
	    touch $@ ; \
	  fi)

$(CINT7_STDIOH) : $(CINT7_MKINCLDS)
	$(CC) $(OPT) $(CINT7CFLAGS) $(CXXOUT)$(CINT7_MKINCLDO) -c $<
	$(LD) $(LDFLAGS) -o $(CINT7_MKINCLD) $(CINT7_MKINCLDO)
	cd $(dir $(CINT7_MKINCLD)) ; ./mkincld

all-cint7 : $(CINT7LIB) $(CINT7) $(CINT7TMP) $(MAKECINT7) $(IOSENUM7)

clean-cint7 :
	@rm -f $(CINT7TMPO) $(CINT7ALLO) $(CINT7EXEO) $(MAKECINT7O)

clean :: clean-cint7

distclean-cint7 : clean-cint7
	@rm -rf $(CINT7ALLDEP) $(CINT7LIB) $(IOSENUM7) $(IOSENUM7A) \
	  $(CINT7EXEDEP) \
          $(CINT7) $(CINT7TMP) $(MAKECINT7) $(CINT7DIRM)/*.exp \
          $(CINT7DIRM)/*.lib $(CINT7DIRS)/v6_loadfile_tmp.cxx \
	  $(CINT7DIRS)/v6_pragma_tmp.cxx

distclean :: distclean-cint7

##### extra rules ######
$(CINT7DIRS)/libstrm.o :  CINT7CXXFLAGS += -I$(CINT7DIRL)/stream
$(CINT7DIRS)/sunstrm.o :  CINT7CXXFLAGS += -I$(CINT7DIRL)/snstream
$(CINT7DIRS)/sun5strm.o : CINT7CXXFLAGS += -I$(CINT7DIRL)/snstream
$(CINT7DIRS)/vcstrm.o :   CINT7CXXFLAGS += -I$(CINT7DIRL)/vcstream
$(CINT7DIRS)/%strm.o :    CINT7CXXFLAGS += -I$(CINT7DIRL)/$(notdir $(basename $@))

$(MAKECINT7O) $(CINT7ALLO) : $(CINT7CONF)

$(CINT7DIRS)/v6_stdstrct.o :     CINT7CXXFLAGS += -I$(CINT7DIRL)/stdstrct
$(CINT7DIRS)/v6_loadfile_tmp.o : CINT7CXXFLAGS += -UHAVE_CONFIG -DROOTBUILD -DG__BUILDING_CINTTMP
$(CINT7DIRS)/v6_pragma_tmp.o : CINT7CXXFLAGS += -UHAVE_CONFIG -DROOTBUILD -DG__BUILDING_CINTTMP

$(CINT7DIRS)/v6_loadfile_tmp.cxx : $(CINT7DIRS)/v6_loadfile.cxx
	cp -f $< $@

$(CINT7DIRS)/v6_pragma_tmp.cxx : $(CINT7DIRS)/v6_pragma.cxx
	cp -f $< $@

#$(CINT7H1T) : include/% : $(CINT7DIRS)/%
#	cp $< $@
#	@if test ! -d $(CINT7DIR)/inc; then mkdir $(CINT7DIR)/inc; fi
#	cp $< $(CINT7DIR)/inc/$(notdir $<)

#$(CINT7H2T) : include/% : $(CINT7DIR)/%
#	cp $< $@
#	@if test ! -d $(CINT7DIR)/inc; then mkdir $(CINT7DIR)/inc; fi
#	cp $< $(CINT7DIR)/inc/$(notdir $<)

##### configcint.h
ifeq ($(CPPPREP),)
# cannot use "CPPPREP?=", as someone might set "CPPPREP="
  CPPPREP = $(CXX) -E -C
endif
include $(CINT7CONFMK)
##### configcint.h - END
