# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += stlIoTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

-include TestHelpers_C.d

-include dequeHolder_C.d
-include listHolder_C.d
-include mapHolder_C.d
-include multimapHolder_C.d
-include multisetHolder_C.d
-include setHolder_C.d
-include vectorHolder_C.d

-include dtest_C.d
-include ltest_C.d
-include mtest_C.d
-include ntest_C.d
-include stest_C.d
-include ttest_C.d
-include vtest_C.d

FTEST = $(wildcard ?test.C)
FTEST_DLL = $(subst .C,_C.$(DllSuf),$(FTEST))

HOLDER = $(wildcard *Holder.C)
HOLDER_DLL = $(subst .C,_C.$(DllSuf),$(HOLDER))

deque: dtest_C.$(DllSuf) dequeHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) 
	$(CMDECHO) echo deque up to date

list: ltest_C.$(DllSuf) listHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) 
	$(CMDECHO) echo list up to date

set: stest_C.$(DllSuf) setHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) 
	$(CMDECHO) echo set up to date

map: mtest_C.$(DllSuf) mapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) 
	$(CMDECHO) echo map up to date

multimap: ttest_C.$(DllSuf) multimapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) 
	$(CMDECHO) echo multimap up to date

multiset: ttest_C.$(DllSuf) multisetHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) 
	$(CMDECHO) echo multiset up to date

vector: vtest_C.$(DllSuf) vectorHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) 
	$(CMDECHO) echo vector up to date

stllibs: $(FTEST_DLL) $(HOLDER_DLL) TestHelpers_C.$(DllSuf) 

# debug := $(shell echo $(FTEST_DLL) 1>&2 ) 

%.o: %.C
	@touch $@

#vectorHolder.o:
#	@touch vectorHolder.o

#dequeHolder.o:
#	@touch dequeHolder.o

#vtest.o:
#	@touch vtest.o

#TestHelpers.o:
#	@touch TestHelpers.o

%_C.$(DllSuf) : %.C %.o
	$(CMDECHO) echo '.x ../../build.C("$<")' | root.exe -l -b -n
#	$(CMDECHO) root.exe -q -l -b ../../build.C\(\"$<\"\)

stlIoTest: stllibs 
	$(CMDECHO) echo stlIoTest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


