# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += ClassInfoNew 

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

ClassInfoNew:
ifeq ($(FAIL),)
	$(WarnFailTest)
	@echo Run the following in valgrind:
	@echo "root[] .L macroFriends2.C"
	@echo "root[] macroFriends()"
	@echo "G__ClassInfo::New would induce an error"
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

testWithDiff: testWithDiff.log
	$(TestDiff)

