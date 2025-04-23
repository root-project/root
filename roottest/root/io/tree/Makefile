# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += test.root copy.root *.so *.out *.d test2.root \
		 *.root.log tref_test_pid*.root *.log $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += missingClass reftest tref_pid_test

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


missingClass: missingClass.log
	$(TestDiffW)


missingClass.log: test.root copy.root read.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b read.C > $@ 2>&1 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

test.root: classes.C write.C
		  $(CMDECHO) $(CALLROOTEXE) -q -l -b write.C > write.out 2>&1

copy.root: classes.C copy.C test.root
		  $(CMDECHO) $(CALLROOTEXE) -q -l -b copy.C > copy.out 2>&1

ifneq ($(ClingWorkAroundMissingAutoLoading),)
tref_test_rd.log test2.root: CALLROOTEXE += -e 'gSystem->Load("libTree");'
endif

test2.root: tref_test.C
		  $(CMDECHO) $(CALLROOTEXE) -q -l -b tref_test.C > tref_test.log

tref_test_rd.log: tref_test_rd.C test2.root
		  $(CMDECHO) $(CALLROOTEXE) -q -l -b tref_test_rd.C > tref_test_rd.log

reftest: tref_test_rd.log
		  $(TestDiff)


#################################

TTP   := tref_test_pid
TTP1  := echo '{gSystem->CompileMacro("$(TTP).C","k");'
TTP2  := | root -l -b 

$(TTP).root: $(TTP).C
	$(CMDECHO)echo 'create("$@")'			 		> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'create("$@");}' $(TTP2) 			>> $@.log 2>&1

$(TTP)_r.root: $(TTP).root
	$(CMDECHO)echo 'addref("$<", "$@", "r1", "N");'			> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'addref("$<", "$@", "r1", "N");}' $(TTP2) 	>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r1");' 				>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r1");}' $(TTP2) 		>> $@.log 2>&1

$(TTP)_rc.root: $(TTP)_r.root
	$(CMDECHO)echo 'clone("$<", "$@");' 				> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'clone("$<", "$@");}' $(TTP2) 		>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r1"); readall("$@");'	 	>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r1"); readall("$@");}' $(TTP2)>> $@.log 2>&1

$(TTP)_rcr.root: $(TTP)_rc.root
	$(CMDECHO)echo 'addref("$<", "$@", "r2", "r1");'		> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'addref("$<", "$@", "r2", "r1");}' $(TTP2) 	>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r1"); readall("$@");' 		>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r1"); readall("$@");}' $(TTP2)>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r2"); readall("$@");' 		>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r2"); readall("$@");}' $(TTP2)>> $@.log 2>&1

$(TTP)_rcrr.root: $(TTP)_rcr.root
	$(CMDECHO)echo 'addref("$<", "$@", "r3", "N");'			> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'addref("$<", "$@", "r3", "N");}' $(TTP2) 	>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r1"); readall("$@");' 		>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r1"); readall("$@");}' $(TTP2)>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r2"); readall("$@");' 		>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r2"); readall("$@");}' $(TTP2)>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r3"); readall("$@");' 		>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r3"); readall("$@");}' $(TTP2)>> $@.log 2>&1
	$(CMDECHO)echo 'readauto("$@", "r1:r2:r3");'	>> $@.log 2>&1
	$(CMDECHO)$(TTP1) 'readauto("$@", "r1:r2:r3");}' $(TTP2)>> $@.log 2>&1

$(TTP).log: $(TTP)_rcrr.root
	$(CMDECHO)rm -f $@ ; cat `ls $(TTP)*.log | sort ` | grep -v ACLiC> $@

tref_test_pid: $(TTP).log
		  $(TestDiff)
