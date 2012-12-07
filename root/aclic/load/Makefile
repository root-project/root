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

assertReload.elog: ReloadScript.C assertReload.C
ifneq ($(ClingWorkAroundMissingUnloading),)
# No point in testing reloading ..
	$(CMDECHO) cp assertReload.ref assertReload.elog
else
	$(CMDECHO) $(CALLROOTEXE) -q -b -l assertReload.C 2>assertReload.full.elog > assertReload.log; result=$$? ; \
	   cat assertReload.full.elog | grep -v _cint.cxx > assertReload.elog ; test $$result -eq 0 || handleError.sh --log=assertReload.elog --result=$$result --test=assertReload.C
endif
