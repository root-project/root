CLEAN_TARGETS += *.log root_5657 xr.* libROOT-5657.so
TEST_TARGETS += root_5657

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

# Create the dict source
xr.cc:  classes.h
        # Makefile kung-fu to avoid printing warning messages
        # We know there is a deprecation warning: it is the point of the whole exercise :)
	$(CMDECHO)genreflex classes.h -s classes_def.xml -o xr.cc --quiet 2>/dev/null ||true

# Compile it into a lib: command line less than optimal, but needed if genreflex is used (no linkdef)
libROOT-5657.so:xr.cc
	$(CMDECHO) $(CXX) xr.cc `root-config --cflags --libs` -fPIC  -shared -o libROOT-5657.so -Wno-deprecated

# Load the lib and check for warnings
root_5657.log: libROOT-5657.so
        # in this case, i.e. a testsuite, we don't even want the warning at 
        # compiletime.
	$(CMDECHO) root -b -l -q loadLib.C >$@ 2>&1


# Compare warnings
root_5657: root_5657.log
	$(TestDiff) && touch $@
