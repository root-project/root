# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.d *.log

#Set the list of target to make while testing
TEST_TARGETS += stlIoTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

LPWD := $(shell echo `pwd -P`)
#debug := $(shell echo $(LPWD) 1>&2 )
LIBDIR := $(subst .,-,$(subst /,-,$(shell root-config --version))).libs/$(LPWD)
#debug := $(shell echo $(LIBDIR) 1>&2 )

#-include TestHelpers_C.d

#-include dequeHolder_C.d
#-include listHolder_C.d
#-include mapHolder_C.d
#-include multimapHolder_C.d
#-include multisetHolder_C.d
#-include setHolder_C.d
#-include vectorHolder_C.d

#-include dtest_C.d
#-include ltest_C.d
#-include mtest_C.d
#-include ntest_C.d
#-include stest_C.d
#-include ttest_C.d
#-include vtest_C.d

FTEST = $(wildcard ?test.C)
FTEST_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(FTEST)) )

HOLDER = $(wildcard *Holder.C)
HOLDER_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(HOLDER)))

TEST_DLL = $(LIBDIR)TestHelpers_C.$(DllSuf) $(FTEST_DLL) $(HOLDER_DLL)

DEPENDENCY_FILES = $(TEST_DLL:.so=.d)

-include $(DEPENDENCY_FILES)

deque: $(addprefix $(LIBDIR),dtest_C.$(DllSuf) dequeHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo deque up to date

list: $(addprefix $(LIBDIR),ltest_C.$(DllSuf) listHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo list up to date

set: $(addprefix $(LIBDIR),stest_C.$(DllSuf) setHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo set up to date

map: $(addprefix $(LIBDIR),mtest_C.$(DllSuf) mapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo map up to date

multimap: $(addprefix $(LIBDIR),ttest_C.$(DllSuf) multimapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo multimap up to date

multiset: $(addprefix $(LIBDIR),ttest_C.$(DllSuf) multisetHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo multiset up to date

vector: $(addprefix $(LIBDIR),vtest_C.$(DllSuf) vectorHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo vector up to date

stllibs: $(FTEST_DLL) $(HOLDER_DLL) TestHelpers_C.$(DllSuf) 

# debug := $(shell echo $(FTEST_DLL) 1>&2 ) 

%.o: %.C
	@touch $@

$(LIBDIR)%_C.$(DllSuf) : %.C %.o
	$(CMDECHO) root.exe -q -l -b ../../build.C\(\"$<\"\)
	@touch $@
#	$(CMDECHO) echo '.x ../../build.C("$<")' | root.exe -l -b -n

stlIoTest.log: $(TEST_DLL)
	$(CMDECHO) root.exe -q -l -b brun.C > stlIoTest.log 2>&1 

stlIoTest: stlIoTest.log
	$(CMDECHO) diff -b stlIoTest.log stlIoTest.ref 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


