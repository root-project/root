
#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.d *.log $(LIBDIR)/*

#Set the list of target to make while testing
TEST_TARGETS += stlIoTest

# adjust the location of Rules.mk as needed.
ifeq "x$(wildcard CVS)" "xCVS"
PACKAGEABLE=yes
RULES_DIR=../..
else
RULES_DIR=.
endif

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


VERSION := $(subst .,-,$(subst /,-,$(shell root-config --version)))
LPWD := $(shell echo `pwd -P`)
#debug := $(shell echo $(LPWD) 1>&2 )
LIBDIRTOP := $(VERSION).libs
LIBDIR := $(LIBDIRTOP)$(LPWD)/
FILEDIR := $(VERSION)
#debug := $(shell echo $(LIBDIR) 1>&2 )

$(FILEDIR):
	$(CMDECHO)mkdir -p $(FILEDIR)

$(LIBDIR):
	$(CMDECHO)mkdir -p $(LIBDIR)

directories: $(FILEDIR) $(LIBDIR)

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

deque: directories $(addprefix $(LIBDIR),dtest_C.$(DllSuf) dequeHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo deque up to date

list: directories $(addprefix $(LIBDIR),ltest_C.$(DllSuf) listHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo list up to date

set: directories $(addprefix $(LIBDIR),stest_C.$(DllSuf) setHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo set up to date

map: directories $(addprefix $(LIBDIR),mtest_C.$(DllSuf) mapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo map up to date

multimap: directories $(addprefix $(LIBDIR),ttest_C.$(DllSuf) multimapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo multimap up to date

multiset: directories $(addprefix $(LIBDIR),ttest_C.$(DllSuf) multisetHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo multiset up to date

vector: directories $(addprefix $(LIBDIR),vtest_C.$(DllSuf) vectorHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo vector up to date

stllibs: directories $(FTEST_DLL) $(HOLDER_DLL) TestHelpers_C.$(DllSuf) 

# debug := $(shell echo $(FTEST_DLL) 1>&2 ) 

#$(LIBDIR)%.o: %.C
#	@touch $@

LIBDIR_SED=$(subst /,\\/,$(LIBDIR))
LIBDIR_SED2=$(subst /,\\/,$(LPWD)/./)
#debug := $(shell echo $(LIBDIR) 1>&2 )
$(LIBDIR)%_C.df: $(LIBDIR)%_C.d
	@cat $< | sed -e s/$(LIBDIR_SED2)// | sed -e s/$*.o/$(LIBDIR_SED)$*_C.so/ > $@

$(LIBDIR)%_C.$(DllSuf) : %.C 
	$(CMDECHO) root.exe -q -l -b $(RULES_DIR)/build.C\(\"$<\"\) >& $*_C.build.log
	$(CMDECHO)touch $@
#	$(CMDECHO) echo '.x $(RULES_DIR)/build.C("$<")' | root.exe -l -b -n

stlIoTest.log: directories $(TEST_DLL)
#	$(CMDECHO) root.exe -q -l -b brun.C > stlIoTest.log 2>&1 
	$(CMDECHO) root.exe -q -l -b brun.C 2>&1 | tee full.log | grep -v 'does not support' | grep -v 'did not support' > stlIoTest.log && ( cat full.log | grep 'not support'; echo )

stlIoTest: stlIoTest.log
	$(CMDECHO) diff -b stlIoTest.log stlIoTest.ref 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

ifneq "x$(PACKAGEABLE)" "x"
package:
	rm -rf stltest
	mkdir -p stltest
	cp *.C *.h Makefile *.ref ../../Rules.mk ../../build.C stltest
	( \
	   tar_name=stlioTest; \
	   inc=0; \
	   tar_file=~/Transfer/$$tar_name.$$inc.tar.gz; \
	   while test -e $$tar_file; do \
		   inc=$$(($$inc+1)); \
		   tar_file=~/Transfer/$$tar_name.$$inc.tar.gz; \
	   done; \
	   tar cfz $$tar_file stltest; \
      echo generated $$tar_file; \
   )
endif


