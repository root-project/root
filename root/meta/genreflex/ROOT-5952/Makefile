CLEAN_TARGETS += *.log dummy* *pcm *o *rflx.* $(ALL_LIBRARIES)
TEST_TARGETS += selLog

ifeq ($(strip $(ROOTTEST_HOME)),)
        export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk


selLog.log: classes.h
	$(CMDECHO) genreflex classes.h -s classes_selection.xml >$@ 2>&1

selLog: selLog.log
	$(TestDiff)
