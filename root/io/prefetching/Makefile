# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog atlasFlushed/atlasFlushed.*  atlasFlushed/*.$(ObjSuf) atlasFlushed/*.pcm atlasFlushed/*.log  atlasFlushed/atlasFlushedProjectDict*

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += PrefetchReading 

# Search for Rules.mk in roottest/scripts
# Algorithm:  Find the current working directory and remove everything after
#  '*roottest/'.  Append the path for Rules.mk from within roottest, which 
#  should be 'scripts/Rules.mk'.  The roottest path is saved in the
#  ROOTTEST_HOME variable for use by the SUBDIRECTORIES variable and is 
#  exported to eliminate the overhead of findding the path again during 
#  recursive calls of gmake.
# Since all makefiles should be under roottest or one of its
#  subdirectories and all recursions of gmake are called by 
#    'cd [DIR]; gmake ...'
#  this algorithm should not fail in finding /roottest/ in the
#  current working directory.  
# Issues:  This algorithm will fail if a makefile is called from outside the 
#  roottest folder, as in executing 'gmake -f ~/roottest/Makefile' from 
#  the home directory.

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


# The name of this target should be changed according to the test being run.  Any name changes
#  should also be reflected in the TEST_TARGETS variable above.

mytest:
	$(CMDECHO) echo mytest > /dev/null


# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


# Remember that sometest.log can be made automatically from
#  from runsometest.C
# This target attempts to create a log file (which should be the output of running the test), and
#  compares its output with a prebuilt reference file.  The reference file should be copied from
#  the log files produced during what is considered a successful run of the test.  The testWithDiff
#  files should be renamed according to the test being run.
# By default the TestDiff variable is set to compare the .log and .ref files
# To run a diff and ignore whitespacing use $(TestDiffW)

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

RUN_PREFETCH=yes
ifeq ($(FAIL)$(PLATFORM),win32)
RUN_PREFETCH=no
endif
ifneq ($(PLATFORM),win32)
ifneq ($(PLATFORM),macos)
HAS_LDD:=$(shell which ldd 2>/dev/null)
endif
endif

ifneq ($(HAS_LDD),)
# Disable for pthread version older than 2.3.5 (we know ROOT thread do not work well with pthread 2.3.4)
pthreadCheck:=$(shell ldd $(ROOTSYS)/lib/libThread.so | grep libpthread.so | cut -d'>' -f2 | cut -d' ' -f2 | cut -d'.' -f1 2>&1 )
ifneq ($(pthreadCheck),)
pthreadCheck:=$(shell ls $(pthreadCheck)-* | sed -e 's/.*\([0-9]\).\([0-9]\).\([0-9]\).*/\1 \\\* 10000 + 100 \\\* \2 + \3 \\\< 20305/' | xargs expr )
ifeq ($(pthreadCheck),1)
RUN_PREFETCH=no
endif
endif #pthreadCheck not empty
endif #has ldd

PrefetchReading.clog: atlasFlushed/atlasFlushed.$(DllSuf)

ifeq ($(RUN_PREFETCH),no)
PrefetchReading: 
	$(WarnFailTest)
else
PrefetchReading: PrefetchReading.clog
	$(TestDiff)
endif

# We should really generate a .d file
%.$(ObjSuf): atlasFlushed/%.cxx $(ROOTCORELIBS)
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_o_cxx.build.log 2>&1

atlasFlushed/atlasFlushedProjectDict1.$(ObjSuf) atlasFlushed/atlasFlushedProjectDict2.$(ObjSuf) atlasFlushed/atlasFlushedProjectSource.$(ObjSuf): $(ROOTCORELIBS)

atlasFlushed/atlasFlushedProjectDict1.cxx: atlasFlushed/atlasFlushedLinkDef1.h
	$(CMDECHO) cd atlasFlushed; rootcint -f atlasFlushedProjectDict1.cxx -c atlasFlushedProjectHeaders.h atlasFlushedLinkDef1.h

atlasFlushed/atlasFlushedProjectDict2.cxx: atlasFlushed/atlasFlushedLinkDef2.h
	$(CMDECHO) cd atlasFlushed ; rootcint -f atlasFlushedProjectDict2.cxx -c atlasFlushedProjectHeaders.h atlasFlushedLinkDef2.h

atlasFlushed/atlasFlushed.$(DllSuf): atlasFlushed/atlasFlushedProjectDict1.$(ObjSuf) atlasFlushed/atlasFlushedProjectDict2.$(ObjSuf) atlasFlushed/atlasFlushedProjectSource.$(ObjSuf)
	$(BuildFromObjs)
