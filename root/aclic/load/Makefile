TEST_TARGETS += autoloadNamespace_test autoload_test

CLEAN_TARGETS += $(ALL_LIBRARIES) *.log autoloadNamespace autoload

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


autoloadNamespace_test : autoloadNamespace

autoloadNamespace : autoloadNamespace.log
	$(TestDiff) && touch $@

autoloadNamespace.log : MyClass_cxx.$(DllSuf)


autoload_test : autoload

autoload : autoload.log
	$(TestDiff) && touch $@

autoload.log : autoloadtest_C.$(DllSuf)

autoloadNamespace.log: MyClass_cxx.$(DllSuf)

linktest_C.$(DllSuf): linktest.C $(ROOT_LOC)/lib/libPhysics.so
	$(BuildWithLib)
