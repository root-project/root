CLEAN_TARGETS += *.log root_5657 xr.cc xr.h libROOT-5657.so loadLib
TEST_TARGETS += root_5657

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

# Create the dict source
root_5657_genreflex.log:
        # Makefile kung-fu to avoid printing warning messages
        # We know there is a deprecation warning: it is the point of the whole exercise :)
	$(CMDECHO)genreflex classes.h -s classes_def.xml -o xr.cc --quiet 2>/dev/null ||true

# Compile it into a lib, compile a program to load it and run it
root_5657.log: root_5657_genreflex.log
        # in this case, i.e. a testsuite, we don't even want the warning at 
        # compiletime.
	$(CMDECHO) $(CXX) xr.cc `root-config --cflags` -fPIC  -shared -o libROOT-5657.so -Wno-deprecated
	$(CMDECHO) $(CXX) loadLib.cpp -o loadLib `root-config --cflags --libs` -fPIC
	$(CMDECHO) ./loadLib >$@ 2>&1
	

# now load it into the interpreter
root_5657: root_5657.log
	$(TestDiff) && touch $@
