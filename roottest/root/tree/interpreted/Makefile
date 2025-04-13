# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += ClassInfoNew 

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


mytest:
	$(CMDECHO) echo mytest > /dev/null

ClassInfoNew:
ifeq ($(FAIL),)
	$(WarnFailTest)
	@echo Warning Run the following in valgrind:
	@echo Warning "root[] .L macroFriends2.C"
	@echo Warning "root[] macroFriends()"
	@echo Warning "G__ClassInfo::New would induce an error"
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

testWithDiff: testWithDiff.log
	$(TestDiff)

