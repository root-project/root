# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.d *.log

#Set the list of target to make while testing
TEST_TARGETS += stlIoTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

LPWD := $(shell echo `pwd -P`)
#debug := $(shell echo $(LPWD) 1>&2 )
LIBDIR := $(subst .,-,$(subst /,-,$(shell root-config --version))).libs$(LPWD)/
#debug := $(shell echo $(LIBDIR) 1>&2 )

FTEST = $(wildcard ?test.C)
FTEST_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(FTEST)) )

HOLDER = $(wildcard *Holder.C)
HOLDER_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(HOLDER)))

TEST_DLL = $(LIBDIR)TestHelpers_C.$(DllSuf) $(FTEST_DLL) $(HOLDER_DLL)
#debug := $(shell echo $(TEST_DLL) 1>&2 )

DEPENDENCY_FILES = $(TEST_DLL:.so=.df)
#debug := $(shell echo $(DEPENDENCY_FILES) 1>&2 )

-include $(DEPENDENCY_FILES)

#.PRECIOUS :  $(addprefix $(LIBDIR),$(subst .C,o.$(DllSuf),$(HOLDER) $(FTEST) $(LIBDIR)TestHelpers_C))

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

#$(LIBDIR)%.o: %.C
#	@touch $@

LIBDIR_SED=$(subst /,\\/,$(LIBDIR))
LIBDIR_SED2=$(subst /,\\/,$(LPWD)/./)
#debug := $(shell echo $(LIBDIR) 1>&2 )
$(LIBDIR)%_C.df: $(LIBDIR)%_C.d
	@cat $< | sed -e s/$(LIBDIR_SED2)// | sed -e s/$*.o/$(LIBDIR_SED)$*_C.so/ > $@

$(LIBDIR)%_C.$(DllSuf) : %.C 
	$(CMDECHO) root.exe -q -l -b ../../build.C\(\"$<\"\)
	@touch $@
#	$(CMDECHO) echo '.x ../../build.C("$<")' | root.exe -l -b -n

stlIoTest.log: $(TEST_DLL)
#	$(CMDECHO) root.exe -q -l -b brun.C > stlIoTest.log 2>&1 
	$(CMDECHO) root.exe -q -l -b brun.C 2>&1 | tee full.log | grep -v 'does not support' > stlIoTest.log && ( cat full.log | grep 'does not support'; echo )

stlIoTest: stlIoTest.log
	$(CMDECHO) diff -b stlIoTest.log stlIoTest.ref 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


