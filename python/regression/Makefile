# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS +=  $(ALL_LIBRARIES) *.log *.clog *.pyc

#Set the list of target to make while testing
TEST_TARGETS += PyROOT_regressiontests 
#PyROOT_advancedtests

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

pylib = $(strip $(shell python -c 'import distutils.sysconfig, sys; print (distutils.sysconfig.get_config_var("LDLIBRARY") or ("Python%d%d" % (sys.version_info[0],sys.version_info[1]))),' ) )
ifeq ($(PLATFORM),win32)
pylibpath = $(strip $(shell python -c 'import os, sys; path = os.path.join(sys.prefix, "libs"); upath = path.replace("\\","/"); print upath; ' ) )
export LIB:=$(pylibpath);$(LIB)
endif

# special case for Scott.C: link with python library
Scott_C.$(DllSuf): Scott.C $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\",\"$(pylib)\"\) > Scott_C.build.log 2>&1


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

%_C.$(DllSuf): %.C $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\",\"libPyROOT\"\) > $*_C.build.log 2>&1

LoKiNamespace_C.clean.log: LoKiNamespace_C.$(DllSuf)
	$(CMDECHO) rm -f LoKiNamespace_C.rootmap > LoKiNamespace_C.clean.log

PyROOT_regressiontests.log: LoKiNamespace_C.clean.log Marco_C.$(DllSuf) MuonTileID_C.$(DllSuf) Scott3_C.$(DllSuf) Scott2_C.$(DllSuf) Scott_C.$(DllSuf) ULongLong_C.$(DllSuf) Till_C.$(DllSuf)

ifeq ($(shell root-config --arch),win32)
# don't run the python file; it doesn't work on Windows:
# WEXITSTATUS not known on windows
PyROOT_regressiontests.log:
	$(CMDECHO) touch $@

PyROOT_regressiontests: PyROOT_regressiontests.log
	$(WarnFailTest)

else

PyROOT_regressiontests: PyROOT_regressiontests.log
	$(TestDiff)

endif
