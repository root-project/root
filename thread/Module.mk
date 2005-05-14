# Module.mk for thread module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := thread
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

THREADDIR    := $(MODDIR)
THREADDIRS   := $(THREADDIR)/src
THREADDIRI   := $(THREADDIR)/inc

##### libThread #####
THREADL      := $(MODDIRI)/LinkDef.h
THREADDS     := $(MODDIRS)/G__Thread.cxx
THREADDO     := $(THREADDS:.cxx=.o)
THREADDH     := $(THREADDS:.cxx=.h)

THREADH      := $(MODDIRI)/TCondition.h $(MODDIRI)/TConditionImp.h \
                $(MODDIRI)/TMutex.h $(MODDIRI)/TMutexImp.h \
                $(MODDIRI)/TRWLock.h $(MODDIRI)/TSemaphore.h \
                $(MODDIRI)/TThread.h $(MODDIRI)/TThreadFactory.h \
                $(MODDIRI)/TThreadImp.h
ifneq ($(ARCH),win32)
THREADH      += $(MODDIRI)/TPosixCondition.h $(MODDIRI)/TPosixMutex.h \
                $(MODDIRI)/TPosixThread.h $(MODDIRI)/TPosixThreadFactory.h \
                $(MODDIRI)/PosixThreadInc.h
else
THREADH      += $(MODDIRI)/TWin32Condition.h $(MODDIRI)/TWin32Mutex.h \
                $(MODDIRI)/TWin32Thread.h $(MODDIRI)/TWin32ThreadFactory.h
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

THREADO      := $(THREADS:.cxx=.o)

THREADDEP    := $(THREADO:.o=.d) $(THREADDO:.o=.d)

THREADLIB    := $(LPATH)/libThread.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(THREADH))
ALLLIBS      += $(THREADLIB)

CXXFLAGS     += $(OSTHREADFLAG)
CFLAGS       += $(OSTHREADFLAG)
CINTCXXFLAGS += $(OSTHREADFLAG)
CINTCFLAGS   += $(OSTHREADFLAG)

ifeq ($(PLATFORM),win32)
CXXFLAGS     += -D_WIN32_WINNT=0x0400
endif

# include all dependency files
INCLUDEFILES += $(THREADDEP)

##### local rules #####
include/%.h:    $(THREADDIRI)/%.h
		cp $< $@

$(THREADLIB):   $(THREADO) $(THREADDO) $(MAINLIBS) $(THREADLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libThread.$(SOEXT) $@ "$(THREADO) $(THREADDO)" \
		   "$(THREADLIBEXTRA) $(OSTHREADLIBDIR) $(OSTHREADLIB)"

$(THREADDS):    $(THREADH) $(THREADL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(THREADH) $(THREADL)

$(THREADDO):    $(THREADDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-thread:     $(THREADLIB)

map-thread:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(THREADLIB) \
		   -d $(THREADLIBDEP) -c $(THREADL)

map::           map-thread

clean-thread:
		@rm -f $(THREADO) $(THREADDO)

clean::         clean-thread

distclean-thread: clean-thread
		@rm -f $(THREADDEP) $(THREADDS) $(THREADDH) $(THREADLIB)

distclean::     distclean-thread
