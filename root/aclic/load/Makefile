TEST_TARGETS +=

CLEAN_TARGETS += $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


execautoloadNamespace.log: MyClass_cxx.$(DllSuf)

execautoload.log : autoloadtest_C.$(DllSuf)

linktest_C.$(DllSuf): linktest.C $(ROOT_LOC)/lib/libPhysics.so
	$(BuildWithLib)

assertReload.elog: ReloadScript.C

