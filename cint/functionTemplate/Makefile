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
ifneq ($(HAS_PYTHON),)
   HAVEPYTHON := $(HAS_PYTHON)
else
   HAVEPYTHON := $(shell root-config --has-python)
endif

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


RFLXDICT_LIB:=t_rflx_wrap_cxx.$(DllSuf)

cintrun reflexrun pythoncintrun pythonreflexrun: % : %.log
	$(TestDiff)


cintrun.log: cintrun.C t_h.$(DllSuf)
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) > $@ 2>&1 || ( cat $@ && false )

reflexrun.log: $(RFLXDICT_LIB) reflexrun.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) > $@ 2>&1 || ( cat $@ && false )

pythoncintrun.log: testcint.py t_h.$(DllSuf)
	$(CMDECHO)$(PYTHON) testcint.py - -b 2>&1 | grep -v "just a comment" > $@ || ( cat $@ && false )

pythonreflexrun.log: $(RFLXDICT_LIB) testreflex.py
	$(CMDECHO)$(PYTHON) testreflex.py - -b 2>&1 | grep -v "just a comment" > $@ || ( cat $@ && false )

t_rflx.cpp: t.h selection.xml
	$(CMDECHO)genreflex t.h -s selection.xml --gccxmlopt="--gccxml-cxxflags -U__GXX_WEAK__" > $@_build.log 2>&1

$(RFLXDICT_LIB): t_rflx.cpp
