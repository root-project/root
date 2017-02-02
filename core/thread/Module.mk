# Module.mk for thread module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := thread
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

THREADDIR    := $(MODDIR)
THREADDIRS   := $(THREADDIR)/src
THREADDIRI   := $(THREADDIR)/inc

##### libThread #####
THREADL      := $(MODDIRI)/LinkDef.h
THREADDS     := $(call stripsrc,$(MODDIRS)/G__Thread.cxx)
THREADDO     := $(THREADDS:.cxx=.o)
THREADDH     := $(THREADDS:.cxx=.h)

THREADH      := $(MODDIRI)/TCondition.h $(MODDIRI)/TConditionImp.h \
                $(MODDIRI)/TMutex.h $(MODDIRI)/TMutexImp.h \
                $(MODDIRI)/TRWLock.h $(MODDIRI)/TSemaphore.h \
                $(MODDIRI)/TThread.h $(MODDIRI)/TThreadFactory.h \
                $(MODDIRI)/TThreadImp.h $(MODDIRI)/TAtomicCount.h \
                $(MODDIRI)/TThreadPool.h $(MODDIRI)/ThreadLocalStorage.h \
                $(MODDIRI)/ROOT/TThreadedObject.hxx \
                $(MODDIRI)/ROOT/TSpinMutex.hxx

ifeq ($(IMT),yes)
THREADH      += $(MODDIRI)/ROOT/TThreadExecutor.hxx
endif

ifneq ($(ARCH),win32)
THREADH      += $(MODDIRI)/TPosixCondition.h $(MODDIRI)/TPosixMutex.h \
                $(MODDIRI)/TPosixThread.h $(MODDIRI)/TPosixThreadFactory.h \
                $(MODDIRI)/PosixThreadInc.h
# Headers that should be copied to $ROOTSYS/include but should not be
# passed directly to rootcint
THREADH_EXT  += $(MODDIRI)/TAtomicCountGcc.h $(MODDIRI)/TAtomicCountPthread.h
else
THREADH      += $(MODDIRI)/TWin32Condition.h $(MODDIRI)/TWin32Mutex.h \
                $(MODDIRI)/TWin32Thread.h $(MODDIRI)/TWin32ThreadFactory.h
THREADH_EXT  += $(MODDIRI)/TWin32AtomicCount.h
endif

THREADS      := $(MODDIRS)/TCondition.cxx $(MODDIRS)/TConditionImp.cxx \
                $(MODDIRS)/TMutex.cxx $(MODDIRS)/TMutexImp.cxx \
                $(MODDIRS)/TRWLock.cxx $(MODDIRS)/TSemaphore.cxx \
                $(MODDIRS)/TThread.cxx $(MODDIRS)/TThreadFactory.cxx \
                $(MODDIRS)/TThreadImp.cxx
ifneq ($(ARCH),win32)
THREADS      += $(MODDIRS)/TPosixCondition.cxx $(MODDIRS)/TPosixMutex.cxx \
                $(MODDIRS)/TPosixThread.cxx $(MODDIRS)/TPosixThreadFactory.cxx
else
THREADS      += $(MODDIRS)/TWin32Condition.cxx $(MODDIRS)/TWin32Mutex.cxx \
                $(MODDIRS)/TWin32Thread.cxx $(MODDIRS)/TWin32ThreadFactory.cxx
endif

THREADO      := $(call stripsrc,$(THREADS:.cxx=.o))

THREADDEP    := $(THREADO:.o=.d) $(THREADDO:.o=.d)

ifeq ($(BUILDTBB),yes)
THREADIMTS   := $(MODDIRS)/TImplicitMT.cxx
THREADIMTO   := $(call stripsrc,$(THREADIMTS:.cxx=.o))
THREADIMTDEP := $(THREADIMTO:.o=.d)
else
THREADIMTS   :=
THREADIMTO   :=
THREADIMTDEP :=
endif

THREADLIB    := $(LPATH)/libThread.$(SOEXT)
THREADMAP    := $(THREADLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
THREADH_REL  := $(patsubst $(MODDIRI)/%,include/%,$(THREADH))
ALLHDRS      += $(THREADH_REL) $(patsubst $(MODDIRI)/%,include/%, $(THREADH_EXT))
ALLLIBS      += $(THREADLIB)
ALLMAPS      += $(THREADMAP)
ifeq ($(CXXMODULES),yes)
  # We need to prefilter ThreadLocalStorage.h because this is a non-modular header,
  # on which depends libCore. Otherwise we end up having a libThread->libCore->libThread
  # header file dependency.
  THREADH_FILTERED_REL := $(filter-out include/ThreadLocalStorage.h, $(THREADH_REL))
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(THREADH_FILTERED_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Core_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(THREADLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

CXXFLAGS     += $(OSTHREADFLAG)
CFLAGS       += $(OSTHREADFLAG)

# include all dependency files
INCLUDEFILES += $(THREADDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(THREADDIRI)/%.h
		cp $< $@

include/%.hxx:  $(THREADDIRI)/%.hxx
		mkdir -p include/ROOT
		cp $< $@

$(THREADLIB):   $(THREADO) $(THREADDO) $(THREADIMTO) \
		   $(ORDER_) $(MAINLIBS) $(THREADLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libThread.$(SOEXT) $@ \
		   "$(THREADO) $(THREADDO) $(THREADIMTO)" \
		   "$(THREADLIBEXTRA) $(OSTHREADLIBDIR) $(OSTHREADLIB) $(TBBLIBDIR) $(TBBLIB)"

$(call pcmrule,THREAD)
	$(noop)

$(THREADDS):    $(THREADH) $(THREADL) $(ROOTCLINGSTAGE1DEP) $(call pcmdep,THREAD)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE1) -f $@ $(call dictModule,THREAD) -c $(THREADH) $(THREADL) && touch lib/libThread_rdict.pcm

$(THREADMAP):   $(THREADH) $(THREADL) $(ROOTCLINGSTAGE1DEP) $(call pcmdep,THREAD)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE1) -r $(THREADDS) $(call dictModule,THREAD) -c $(THREADH) $(THREADL)

all-$(MODNAME): $(THREADLIB)

clean-$(MODNAME):
		@rm -f $(THREADO) $(THREADDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(THREADDEP) $(THREADDS) $(THREADDH) $(THREADLIB) $(THREADMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(BUILDTBB),yes)
$(THREADO) $(THREADIMTO): CXXFLAGS += $(TBBINCDIR:%=-I%)
endif
