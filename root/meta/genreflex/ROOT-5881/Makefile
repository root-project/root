DICTNAME=stringIO_allClasses_rflx.cpp
OFILENAME=stringTree.root
LIBBASENAME=StringDict
LIBNAME=lib$(LIBBASENAME).$(DllSuf)
EXENAME=writeTree
CXXCOMPILER=$(shell root-config --cxx)

CLEAN_TARGETS += *.log *.so *.d *pcm *rflx* dummy*.C *.root *success $(OFILENAME) $(EXENAME)
TEST_TARGETS += execstringI

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

$(LIBNAME): $(DICTNAME)
	$(CMDECHO) $(CXXCOMPILER) -shared -fPIC $(shell root-config --cflags) $(DICTNAME) -o $@ >$@.log 2>&1

$(EXENAME): $(LIBNAME)
	$(CMDECHO) $(CXXCOMPILER) $@.cpp -fPIC -o $@ $(shell root-config --cflags --libs) -l $(LIBBASENAME) -L ./ > $@_compilation.log 2>&1

$(OFILENAME): $(EXENAME)
	$(CMDECHO) ./$(EXENAME) > $(EXENAME)_execution.log 2>&1

execstringI.log: $(OFILENAME)
