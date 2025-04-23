CLEAN_TARGETS += $(ALL_LIBRARIES) *.log $(RFLXDICT_LIB) t.exe t_rflx.* *.rootmap
TEST_TARGETS += cintrun referenceUse

# Which tests to run depends on availability of cintex / python:
HAVEGCCXML := $(shell which gccxml 2>/dev/null)
ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
 HAVECINTEX := $(findstring cintex,$(shell root-config --features))
endif

ifeq ($(HAVECINTEX),cintex)
 TEST_TARGETS += reflexrun
endif

# also in Rules.mk, but that needs TEST_TARGETS to be defined first.
ifeq ($(FIXCLING),)
ifneq ($(HAS_PYTHON),)
   HAVEPYTHON := $(HAS_PYTHON)
else
   HAVEPYTHON := $(shell root-config --has-python)
endif
endif

ifeq ($(HAVEPYTHON),yes)
 TEST_TARGETS += pythoncintrun
 ifeq ($(HAVECINTEX),cintex)
  TEST_TARGETS += pythonreflexrun
 endif
endif

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk


RFLXDICT_LIB:=t_rflx_wrap_cxx.$(DllSuf)

cintrun reflexrun pythoncintrun pythonreflexrun: % : %.log
	$(TestDiffCintSpecific)


cintrun.log: cintrun.C t_h.$(DllSuf)
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) > $@ 2>&1 || ( cat $@ && false )

reflexrun.log: $(RFLXDICT_LIB) reflexrun.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) > $@ 2>&1 || ( cat $@ && false )

pythoncintrun.log: testcint.py t_h.$(DllSuf)
	$(CMDECHO)unset DISPLAY; $(PYTHON) testcint.py - -b 2>&1 | grep -v "just a comment" > $@ || ( cat $@ && false )

pythonreflexrun.log: $(RFLXDICT_LIB) testreflex.py
	$(CMDECHO)unset DISPLAY; $(PYTHON) testreflex.py - -b 2>&1 | grep -v "just a comment" > $@ || ( cat $@ && false )

t_rflx.cpp: t.h selection.xml
	$(CMDECHO)genreflex t.h -s selection.xml --gccxmlopt="--gccxml-cxxflags -U__GXX_WEAK__" > $@_build.log 2>&1

$(RFLXDICT_LIB): t_rflx.cpp

referenceUse.log: MyClassReferenceUse_C.$(DllSuf)

referenceUse: referenceUse.log
	$(TestDiff)
