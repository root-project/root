CLEAN_TARGETS += *.log xr.* 
TEST_TARGETS += root_5626

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

root_5626_genreflex.log:
	$(CMDECHO) genreflex classes.h -s classes_def.xml -o xr.cc >$@ 2>&1

root_5626.log: root_5626_genreflex.log
	$(CMDECHO) $(CXX) -c xr.cc `root-config --cflags`  >$@ 2>&1

# The ref is empty: we do not expect compilation errors :)
root_5626: root_5626.log
	$(TestDiff) && touch $@
