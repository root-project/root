# Disable until a portable and minimal way to express the test is found
# indeed this formulation would fail on windows and actually fail on OSx10.8
test:
	@echo nothing > /dev/null

clean:
	@echo nothing > /dev/null

# DICTNAME=stringIO_allClasses_rflx.cpp
# OFILENAME=stringTree.root
# LIBBASENAME=StringDict
# LIBNAME=lib$(LIBBASENAME).$(DllSuf)
# EXENAME=writeTree
# CXXCOMPILER=$(shell root-config --cxx)
# 
# CLEAN_TARGETS += *.log *.so *.d *pcm *rflx* dummy*.C *.root *success $(OFILENAME) $(EXENAME)
# TEST_TARGETS += execstringI
# 
# ifeq ($(strip $(ROOTTEST_HOME)),)
#   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
#   ifeq ($(strip $(ROOTTEST_HOME)),)
#      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
#   endif
#   ifeq ($(strip $(ROOTTEST_HOME)),)
#      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
#   endif
# endif
# 
# include $(ROOTTEST_HOME)/scripts/Rules.mk
# include $(ROOTTEST_HOME)/scripts/Reflex.mk
# 
# #$(LIBNAME): $(DICTNAME)
# #	$(CMDECHO) $(CXXCOMPILER) -shared -fPIC $(shell root-config --cflags) $(DICTNAME) -o $@ >$@.log 2>&1
# # Strategy from root/io/alloc/Makefile
# $(DICTNAME) : stringIO_allClasses.h stringIO_allClasses_selection.xml $(ROOTCINT)
#	$(CMDECHO) genreflex stringIO_allClasses -s stringIO_allClasses_selection.xml
#
# dictObject.$(ObjSuf): $(DICTNAME)
#
# $(LIBNAME): dictObject.$(ObjSuf)
#	$(BuildFromObjs)
#
# # We need a way to build the exe in a portable way
# $(EXENAME): $(LIBNAME)
# 	$(CMDECHO) $(CXXCOMPILER) $@.cpp -fPIC -o $@ $(shell root-config --cflags --libs) -l $(LIBBASENAME) -L ./ > $@_compilation.log 2>&1
# 
# $(OFILENAME): $(EXENAME)
# 	$(CMDECHO) ./$(EXENAME) > $(EXENAME)_execution.log 2>&1
# 
# execstringI.log: $(OFILENAME)