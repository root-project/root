# Module.mk for reflex module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := reflex
MODDIR       := $(ROOT_SRCDIR)/cint/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

REFLEXDIR    := $(MODDIR)
REFLEXDIRS   := $(REFLEXDIR)/src
REFLEXDIRI   := $(REFLEXDIR)/inc

REFLEXLIB    := $(LPATH)/libReflex.$(SOEXT)

REFLEXS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
REFLEXO      := $(call stripsrc,$(REFLEXS:.cxx=.o))

# genreflex
ifeq ($(PLATFORM),win32)
RFLX_REFLEXLL   = lib/libReflex.lib
else
# test suite
RFLX_REFLEXLL   = -Llib -lReflex
ifeq ($(ARCH),linuxicc)
RFLX_REFLEXLL   += -limf
endif
ifeq ($(ARCH),linuxx8664icc)
RFLX_REFLEXLL   += -limf
endif
ifneq ($(PLATFORM),fbsd)
ifneq ($(PLATFORM),obsd)
RFLX_REFLEXLL   += -ldl
endif
endif
endif

ifeq ($(PLATFORM),solaris)
RFLX_REFLEXLL   += -ldemangle
endif

RFLX_GENMAPX   = bin/genmap$(EXEEXT)
