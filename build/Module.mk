# Module.mk for build module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

ifneq ($(HOST),)

RMKDEP       := $(BUILDTOOLSDIR)/bin/rmkdepend$(EXEEXT)

else

MODNAME      := build
MODDIR       := $(ROOT_SRCDIR)/$(MODNAME)

RMKDEPDIR    := $(MODDIR)/rmkdepend
BINDEXPDIR   := $(MODDIR)/win/bindexplib
DROPDIR      := $(MODDIR)/unix/drop_from_path

##### rmkdepend #####
RMKDEPH      := $(wildcard $(RMKDEPDIR)/*.h)
RMKDEPS1     := $(wildcard $(RMKDEPDIR)/*.c)
RMKDEPS2     := $(wildcard $(RMKDEPDIR)/*.cxx)
RMKDEPO1     := $(call stripsrc,$(RMKDEPS1:.c=.o))
RMKDEPO2     := $(call stripsrc,$(RMKDEPS2:.cxx=.o))
RMKDEPO      := $(RMKDEPO1) $(RMKDEPO2)
RMKDEP       := bin/rmkdepend$(EXEEXT)
RMKDEPCFLAGS := -DINCLUDEDIR=\"/usr/include\" -DOBJSUFFIX=\".o\"

##### drop_from_path #####
ifneq ($(PLATFORM),win32)
DROPH      := $(wildcard $(DROPDIR)/*.h)
DROPS      := $(wildcard $(DROPDIR)/*.c)
DROPO      := $(call stripsrc,$(DROPS:.c=.o))
DROP       := bin/drop_from_path$(EXEEXT)
endif

##### bindexplib #####
ifeq ($(PLATFORM),win32)
BINDEXPS     := $(wildcard $(BINDEXPDIR)/*.cxx)
BINDEXPO     := $(call stripsrc,$(BINDEXPS:.cxx=.o))
BINDEXP      := bin/bindexplib$(EXEEXT)

W32PRAGMA    := $(ROOT_SRCDIR)/build/win/w32pragma.h
ALLHDRS      += include/w32pragma.h
endif

POSTBIN      += $(DROP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(RMKDEP):      $(RMKDEPO)
		$(LD) $(LDFLAGS) -o $@ $(RMKDEPO)

$(DROP):        $(DROPO)
		$(LD) $(LDFLAGS) -o $@ $(DROPO)

ifeq ($(PLATFORM),win32)
include/%.h:    $(ROOT_SRCDIR)/build/win/%.h
		cp $< $@

$(BINDEXP):     $(BINDEXPO)
		$(LD) $(LDFLAGS) -o $@ $(BINDEXPO)

$(BINDEXPO):    $(ORDER_) $(RMKDEP)

all-$(MODNAME): $(RMKDEP) $(BINDEXP)
else
all-$(MODNAME): $(RMKDEP) $(DROP)
endif

clean-$(MODNAME):
		@rm -f $(RMKDEPO) $(BINDEXPO) $(DROPO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RMKDEP) $(BINDEXP) $(DROP)

distclean::     distclean-$(MODNAME)


##### dependencies #####
$(call stripsrc,$(RMKDEPDIR)/cppsetup.o): $(RMKDEPDIR)/def.h $(RMKDEPDIR)/ifparser.h
$(call stripsrc,$(RMKDEPDIR)/ifparser.o): $(RMKDEPDIR)/ifparser.h
$(call stripsrc,$(RMKDEPDIR)/include.o):  $(RMKDEPDIR)/def.h
$(call stripsrc,$(RMKDEPDIR)/main.o):     $(RMKDEPDIR)/def.h $(RMKDEPDIR)/imakemdep.h
$(call stripsrc,$(RMKDEPDIR)/parse.o):    $(RMKDEPDIR)/def.h
$(call stripsrc,$(RMKDEPDIR)/pr.o):       $(RMKDEPDIR)/def.h
$(call stripsrc,$(RMKDEPDIR)/mainroot.o): $(RMKDEPDIR)/def.h

##### local rules #####
$(RMKDEPO1): CFLAGS += $(RMKDEPCFLAGS)
$(RMKDEPO2): CXXFLAGS += $(RMKDEPCFLAGS)

endif
