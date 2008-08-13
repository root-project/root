RFLXDICT_WRAPPER:=t_rflx_wrap.cxx
RFLXDICT_LIB:=t_rflx_wrap_cxx.$(DllSuf)

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log $(RFLXDICT_LIB) t.exe t_rflx.* *.rootmap
TEST_TARGETS += cintrun

# Which tests to run depends on availability of reflex / python:
HAVEGCCXML := $(shell which gccxml 2>/dev/null)
ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
 HAVEREFLEX := $(shell root-config --has-reflex)
endif

ifeq ($(HAVEREFLEX),yes)
 TEST_TARGETS += reflexrun
endif

# also in Rules.mk, but that needs TEST_TARGETS to be defined first.
HAVEPYTHON := $(shell root-config --has-python)

ifeq ($(HAVEPYTHON),yes)
 TEST_TARGETS += pythoncintrun
 ifeq ($(HAVEREFLEX),yes)
  TEST_TARGETS += pythonreflexrun
 endif
endif

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


cintrun reflexrun pythoncintrun pythonreflexrun: % : %.log
	$(TestDiff)


cintrun.log: cintrun.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) | grep -v 'creating shared library' > $@ 2>&1

reflexrun.log: $(RFLXDICT_LIB) reflexrun.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) > $@ 2>&1

pythoncintrun.log: testcint.py
	$(CMDECHO)$(PYTHON) testcint.py | grep -v 'creating shared library' > $@ 2>&1

pythonreflexrun.log: $(RFLXDICT_LIB) testreflex.py
	$(CMDECHO)$(PYTHON) testreflex.py > $@ 2>&1

BUILDER:=build_t_rflx_wrap_cxx.C
$(RFLXDICT_LIB): t_rflx.cpp
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(BUILDER) >> $@_build.log 2>&1

t_rflx.cpp: t.h selection.xml
	$(CMDECHO)genreflex t.h -s selection.xml --gccxmlopt="--gccxml-cxxflags -U__GXX_WEAK__" --gccxmlpath=$(dir $(HAVEGCCXML)) > $@_build.log 2>&1
