CLEAN_TARGETS += $(ALL_LIBRARIES) *.log libt_rflx.$(DllSuf) t.exe t_rflx.* *_wrap.* *.rootmap

# which tests we run depends on availability of reflex / python:
TEST_TARGETS += compiled cintrun

HAVEGCCXML := $(shell which gccxml 2>/dev/null)
ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
 HAVEREFLEX := $(shell root-config --has-reflex)
# GCC4.3 needs GCCXML>= 0.9:
 ifneq ($(subst g++,,$(CXX)),$(CXX))
  GCC_VER:=$(shell g++ --version | head -n1)
  GCC_VER_MAJOR:=$(shell echo '$(GCC_VER)' | sed 's,^.*[^.]\([[:digit:]]\+\)\..*$$,\1,')
  GCC_VER_MINOR:=$(shell echo '$(GCC_VER)' | sed 's,^.*\b$(GCC_VER_MAJOR)\.\([[:digit:]]\+\)\..*$$,\1,')
  ifeq ($(GCC_VER_MAJOR),4)
   GCC_VER_GE_43:=$(shell if [ $(GCC_VER_MINOR) -ge 3 ]; then echo yes; fi)
  else
   ifeq ($(subst $(GCC_VER_MAJOR),,123), 123)
    GCC_VER_GE_43:=yes
   endif
  endif
  ifeq ($(GCC_VER_GE_43),yes)
   GCCXML_VER:=$(shell $(HAVEGCCXML) --version | head -n1 | sed 's,^.*\s\([[:digit:]]\+\.[[:digit:]]\+\).*,\1,')
   ifneq ($(findstring $(GCCXML_VER),0.7 0.6),)
    HAVEREFLEX :=
   endif
  endif
 endif
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


ifeq ($(PLATFORM),win32)
 RFLX_REFLEXLL   = $(ROOTSYS)/lib/libReflex.lib
else
 RFLX_REFLEXLL   = -lReflex
 ifneq ($(PLATFORM),fbsd)
  ifneq ($(PLATFORM),obsd)
   RFLX_REFLEXLL   += -ldl
  endif
 endif
endif

compiled cintrun reflexrun pythoncintrun pythonreflexrun: % : %.log
	$(TestDiff)


compiled.log: t.exe
	$(CMDECHO)./t.exe > $@ 2>&1

CINTNORMALIZER:=sed -e 's,\bt_h\.[^[:space:]]*,,' -e 's,tagnum=[[:digit:]]\+,tagnum=***,' -e 's/0x[0-9a-f]\{7,\}/0xSOMEHEX/g'
cintrun.log: cintrun.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) | $(CINTNORMALIZER) > $@ 2>&1

reflexrun.log: libt_rflx.$(DllSuf) reflexrun.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(@:.log=.C) | $(CINTNORMALIZER) > $@ 2>&1

pythoncintrun.log: testcint.py
	$(CMDECHO)$(PYTHON) testcint.py > $@ 2>&1

pythonreflexrun.log: libt_rflx.$(DllSuf) testreflex.py
	$(CMDECHO)$(PYTHON) testreflex.py > $@ 2>&1

t.exe: t.h t.cxx
ifeq ($(ARCH),win32)
	$(CMDECHO)$(CXX) t.cxx $(CXXFLAGS) /Fe$@ > $@_build.log 2>&1
else
	$(CMDECHO)$(CXX) -c t.cxx $(CXXFLAGS) $(OutPutOpt)t.$(ObjSuf) > $@_build.log 2>&1
	$(CMDECHO)$(CXX) $(@:.exe=.$(ObjSuf)) $(OutPutOpt)$@ >> $@_build.log 2>&1
endif

BUILDER:=libt_rflx_tmp.C
WRAPPER:=t_rflx_wrap.cxx
libt_rflx.$(DllSuf): t_rflx.cpp
	$(CMDECHO)echo '#ifndef __CINT__' > $(WRAPPER) 2> $@_build.log
	$(CMDECHO)echo '#include "$<"' >> $(WRAPPER) 2> $@_build.log
	$(CMDECHO)echo '#endif' >> $(WRAPPER) 2> $@_build.log
	$(CMDECHO)echo 'int $(<:.cpp=_wrap)() {return 0;}' >> $(WRAPPER) 2> $@_build.log
	$(CMDECHO)echo '{gSystem->Load("libReflex"); gROOT->ProcessLine(".L $(WRAPPER)+"); exit();}' > $(BUILDER)
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $(BUILDER) >> $@_build.log 2>&1
	$(CMDECHO)mv $(WRAPPER:.cxx=_cxx.$(DllSuf)) $@ >> $@_build.log 2>&1
	$(CMDECHO)rm $(BUILDER) $(WRAPPER) >> $@_build.log 2>&1

t_rflx.cpp: t.h selection.xml
	$(CMDECHO)genreflex t.h -s selection.xml --gccxmlpath=$(dir $(HAVEGCCXML)) > $@_build.log 2>&1
