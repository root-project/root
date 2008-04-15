# Module.mk for cint module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIRBASE   := cint
MODDIR       := $(MODDIRBASE)/cint7
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

CINT7DIR      := $(MODDIR)
CINT7DIRS     := $(CINT7DIR)/src
CINT7DIRI     := $(CINT7DIR)/inc
CINT7DIRM     := $(CINT7DIR)/main
CINT7DIRT     := $(MODDIRBASE)/tool
CINT7DIRL     := $(CINT7DIR)/lib
CINT7DIRDLLS  := $(CINT7DIR)/include
CINT7DIRSTL   := $(CINT7DIR)/stl
CINT7DIRDLLSTL:= $(CINT7DIRL)/dll_stl
CINT7DIRSD    := $(CINT7DIRS)/dict

# for configcint:
CINTDIR       := $(CINT7DIR)
# for cintdlls:
CINTDIRSTL    := $(CINT7DIRSTL)
CINTDIRDLLS   := $(CINT7DIRDLLS)
CINTDIRDLLSTL := $(CINT7DIRDLLSTL)
CINTDIRL      := $(CINT7DIRL)


##### libCint #####
CINT7CONF     := $(CINT7DIRI)/configcint.h
CINTCONF      := $(CINT7CONF)
CINT7H        := $(filter-out $(CINT7CONF),$(wildcard $(CINT7DIRI)/*.h))
CINT7HT       := $(patsubst $(CINT7DIRI)/%.h,include/%.h,$(CINT7H))
CINT7S1       := $(wildcard $(MODDIRS)/*.c)
CINT7S2       := $(wildcard $(CINT7DIRS)/*.cxx) $(CINT7DIRSD)/longif.cxx $(CINT7DIRSD)/Apiif.cxx $(CINT7DIRSD)/stdstrct.o

CINT7S1       += $(CINT7DIRM)/G__setup.c

CINT7ALLO     := $(CINT7S1:.c=.o) $(CINT7S2:.cxx=.o)
CINT7ALLDEP   := $(CINT7ALLO:.o=.d)

CINT7CONFMK   := $(MODDIRBASE)/ROOT/configcint.mk
CINTCONFMK    := $(CINT7CONFMK)

CINT7S1       := $(filter-out $(MODDIRS)/dlfcn.%,$(CINT7S1))

CINT7S2       := $(filter-out $(MODDIRS)/sunos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/macos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/winnt.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/newsos.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/loadfile_tmp.%,$(CINT7S2))
CINT7S2       := $(filter-out $(MODDIRS)/pragma_tmp.%,$(CINT7S2))

# strip off possible leading path from compiler command name
CXXCMD       := $(shell echo $(CXX) | sed s/".*\/"//)

ifeq ($(CXXCMD),KCC)
CINT7S2       += $(CINT7DIRSD)/kccstrm.cxx
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
else
ifeq ($(PLATFORM),linux)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),hurd)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),fbsd)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),obsd)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),hpux)
ifeq ($(ARCH),hpuxia64acc)
CINT7S2       += $(CINT7DIRSD)/accstrm.cxx
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
else
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),solaris)
ifeq ($(SUNCC5),true)
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
ifeq ($(findstring $(CXXFLAGS),-library=iostream,no%Cstd),)
CINT7S2       += $(CINT7DIRSD)/sunstrm.cxx
#CINT7S2       += $(CINT7DIRSD)/sun5strm.cxx
else
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
else
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),aix3)
CINT7S1       += $(CINT7DIRS)/dlfcn.c
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),aix5)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),sgi)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),alpha)
CINT7S2       += $(CINT7DIRSD)/alphastrm.cxx
endif
ifeq ($(PLATFORM),alphagcc)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
endif
ifeq ($(PLATFORM),sunos)
CINT7S1       += $(CINT7DIRS)/sunos.c
endif
ifeq ($(PLATFORM),macos)
CINT7S2       += $(CINT7DIRS)/macos.cxx
CINT7S2       += $(CINT7DIRSD)/fakestrm.cxx
endif
ifeq ($(PLATFORM),macosx)
CINT7S2       += $(CINT7DIRSD)/libstrm.cxx
endif
ifeq ($(PLATFORM),lynxos)
CINT7S2       += $(CINT7DIRSD)/fakestrm.cxx
endif
ifeq ($(PLATFORM),win32)
CINT7S2       += $(CINT7DIRS)/winnt.cxx
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
ifeq ($(VC_MAJOR),13)
 ifeq ($(VC_MINOR),10)
  CINT7S2       += $(CINT7DIRSD)/vc7strm.cxx
 else
  CINT7S2       += $(CINT7DIRSD)/iccstrm.cxx
 endif
else
 ifeq ($(VC_MAJOR),14)
  CINT7S2       += $(CINT7DIRSD)/vc7strm.cxx
 else
  CINT7S2       += $(CINT7DIRSD)/iccstrm.cxx
 endif
endif
endif
ifeq ($(PLATFORM),vms)
CINT7S2       += $(CINT7DIRSD)/fakestrm.cxx
endif
ifeq ($(CXXCMD),icc)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
ifneq ($(ICC_GE_9),)
CINT7S2       += $(CINT7DIRSD)/gcc3strm.cxx
else
CINT7S2       += $(CINT7DIRSD)/iccstrm.cxx
endif
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),3)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/gcc3strm.cxx
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
ifeq ($(GCC_MAJOR),4)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/gcc4strm.cxx
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
ifeq ($(CXXCMD),xlC)
ifeq ($(PLATFORM),macosx)
CINT7S2       := $(filter-out $(CINT7DIRSD)/libstrm.%,$(CINT7S2))
CINT7S2       := $(filter-out $(CINT7DIRSD)/longif.%,$(CINT7S2))
CINT7S2       += $(CINT7DIRSD)/gcc3strm.cxx
CINT7S2       += $(CINT7DIRSD)/longif3.cxx
endif
endif

CINT7S        := $(CINT7S1) $(CINT7S2)
CINT7O        := $(CINT7S1:.c=.o) $(CINT7S2:.cxx=.o)
CINT7TMPO     := $(subst loadfile.o,loadfile_tmp.o,$(CINT7O))
CINT7TMPO     := $(subst pragma.o,pragma_tmp.o,$(CINT7TMPO))
CINTTMPO       = $(CINT7TMPO) $(REFLEXO)
CINT7TMPINC   := -I$(MODDIR)/inc -I$(MODDIR)/include -I$(MODDIR)/stl -I$(MODDIR)/lib -Iinclude
CINT7DEP      := $(CINT7O:.o=.d)
CINT7DEP      += $(CINT7DIRS)/loadfile_tmp.d
CINT7DEP      += $(CINT7DIRS)/loadfile_tmp.d
CINT7ALLDEP   += $(CINT7DIRS)/pragma_tmp.d
CINT7ALLDEP   += $(CINT7DIRS)/pragma_tmp.d

CINT7LIB      := $(LPATH)/libCint.$(SOEXT)
CINTLIB       := $(CINT7LIB)

##### cint #####
CINT7EXES     := $(CINT7DIRM)/cppmain.cxx
CINT7EXEO     := $(CINT7EXES:.cxx=.o)
CINT7EXEDEP   := $(CINT7EXEO:.o=.d)
CINT7TMP      := $(CINT7DIRM)/cint_tmp$(EXEEXT)
CINTTMP       :=  $(CINT7TMP)
CINT7         := bin/cint$(EXEEXT)
CINT          := $(CINT7)

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
ALLHDRS     += $(CINT7HT)
ALLHDRS     += $(CINT7CONF)

ALLLIBS      += $(CINT7LIB)
ALLEXECS     += $(CINT7) $(MAKECINT7) $(CINT7TMP)

# include all dependency files
INCLUDEFILES += $(CINT7DEP) $(CINT7EXEDEP)

CINT7CXXFLAGS := $(filter-out -Iinclude -DG__REGEXP,$(CINTCXXFLAGS))
CINT7CFLAGS   := $(filter-out -Iinclude -DG__REGEXP,$(CINTCFLAGS))

CINT7CXXFLAGS += -DG__CINTBODY -DG__HAVE_CONFIG -DG__NOMAKEINFO
CINT7CFLAGS   += -DG__CINTBODY -DG__HAVE_CONFIG -DG__NOMAKEINFO

CINT7CXXFLAGS += -I$(CINT7DIRI) -I$(CINT7DIRS) -I$(CINT7DIRSD) -I$(CINT7DIR)/reflex/inc -Iinclude
CINT7CFLAGS   += -I$(CINT7DIRI) -I$(CINT7DIRS) -I$(CINT7DIRSD) -I$(CINT7DIR)/reflex/inc -Iinclude

CINTCXXFLAGS   = $(CINT7CXXFLAGS)
CINTCFLAGS     = $(CINT7CFLAGS)

CINTLIBS      := $(CINT7LIBS)

ifeq ($(PLATFORM),win32)
REFLEXLL := lib/libReflex.lib
else
REFLEXLL := -Llib -lReflex
ifneq ($(PLATFORM),fbsd) 
ifneq ($(PLATFORM),obsd)
REFLEXLL   += -ldl 
endif 
endif
endif

##### local rules #####
include/%.h:    $(CINT7DIRI)/%.h
	cp $< $@

$(CINT7LIB): $(CINT7O) $(CINT7LIBDEP) $(REFLEXLIB)
	$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" \
	   libCint.$(SOEXT) $@ "$(CINT7O)" "$(CINT7LIBEXTRA) $(REFLEXLL)"

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
	   $(REFLEXO) $(CILIBS)

$(MAKECINT7) : $(MAKECINT7O)
	$(LD) $(LDFLAGS) -o $@ $(MAKECINT7O)

$(IOSENUM7) : $(IOSENUM7A)
	cp $< $@

$(IOSENUM7A) : $(CINT7TMP) $(CINT7_STDIOH)
	@(if test ! -r $@ ; \
	  then \
	    PATH=$PWD/bin:$$PATH \
	    LD_LIBRARY_PATH=$$PWD/lib:$$LD_LIBRARY_PATH \
	    DYLD_LIBRARY_PATH=$$PWD/lib:$$DYLD_LIBRARY_PATH \
	    $(CINT7TMP) $(CINT7TMPINC) $(IOSENUM7C) > /dev/null ; \
	    mv iosenum.h $@ ; \
	  else \
	    touch $@ ; \
	  fi)

$(CINT7_STDIOH) : $(CINT7_MKINCLDS)
	$(CC) $(OPT) $(CINT7CFLAGS) $(CXXOUT)$(CINT7_MKINCLDO) -c $<
	$(LD) $(LDFLAGS) -o $(CINT7_MKINCLD) $(CINT7_MKINCLDO)
	cd $(dir $(CINT7_MKINCLD)) ; ./mkincld

all-cint  : all-cint7
all-cint7 : $(CINT7LIB) $(CINT7) $(CINT7TMP) $(MAKECINT7) $(IOSENUM7)

clean-cint7 :
	@rm -f $(CINT7TMPO) $(CINT7ALLO) $(CINT7EXEO) $(MAKECINT7O)

clean :: clean-cint7

distclean-cint7 : clean-cint7
	@rm -rf $(CINT7ALLDEP) $(CINT7LIB) $(IOSENUM7) $(CINT7EXEDEP) \
          $(CINT7) $(CINT7TMP) $(MAKECINT7) $(CINT7DIRM)/*.exp \
          $(CINT7DIRM)/*.lib $(CINT7DIRS)/loadfile_tmp.cxx \
	  $(CINT7DIRS)/pragma_tmp.cxx \
	  $(CINT7HT) $(CINTCONF)

distclean :: distclean-cint7

##### extra rules ######
$(CINT7DIRS)/libstrm.o :  CINT7CXXFLAGS += -I$(CINT7DIRL)/stream
$(CINT7DIRS)/sunstrm.o :  CINT7CXXFLAGS += -I$(CINT7DIRL)/snstream
$(CINT7DIRS)/sun5strm.o : CINT7CXXFLAGS += -I$(CINT7DIRL)/snstream
$(CINT7DIRS)/vcstrm.o :   CINT7CXXFLAGS += -I$(CINT7DIRL)/vcstream
$(CINT7DIRS)/%strm.o :    CINT7CXXFLAGS += -I$(CINT7DIRL)/$(notdir $(basename $@))

$(MAKECINT7O) $(CINT7ALLO) : $(CINT7CONF)

$(MAKECINT7O): CXXFLAGS := $(CINT7CXXFLAGS)
$(CINT7DIRSD)/stdstrct.o :    CINT7CXXFLAGS += -I$(CINT7DIRL)/stdstrct
$(CINT7DIRS)/loadfile_tmp.o : CINT7CXXFLAGS += -UHAVE_CONFIG -DROOTBUILD -DG__BUILDING_CINTTMP
$(CINT7DIRS)/pragma_tmp.o :   CINT7CXXFLAGS += -UHAVE_CONFIG -DROOTBUILD -DG__BUILDING_CINTTMP

$(CINT7DIRS)/loadfile_tmp.cxx : $(CINT7DIRS)/loadfile.cxx
	cp -f $< $@

$(CINT7DIRS)/pragma_tmp.cxx : $(CINT7DIRS)/pragma.cxx
	cp -f $< $@

##### configcint.h
ifeq ($(CPPPREP),)
# cannot use "CPPPREP?=", as someone might set "CPPPREP="
  CPPPREP = $(CXX) -E -C
endif
include $(CINT7CONFMK)
##### configcint.h - END
