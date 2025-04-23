CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *rflx*
# TEST_TARGETS += cleanloadAllLibs
#execloadAllLibsAZ execloadAllLibsZA

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
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# runloadAllLibs: libtest_dictrflx.$(DllSuf)
#
# cleanloadAllLibs.log: runloadAllLibs
# 	$(CMDECHO) cat loadAllLibs.log|grep -v Wouter|grep -v NIKHEF|grep -v sourceforge
#
# cleanloadAllLibs: cleanloadAllLibs.log
# 	$(TestDiff)
#
# #execloadAllLibsAZ.log: libdummy_dictrflx.$(DllSuf)
#
# #cleanloadAllLibsAZ.log: execloadAllLibsAZ.log
# #	$(CMDECHO) cat execloadAllLibsAZ.log|grep -v Wouter|grep -v NIKHEF|grep -v sourceforge
#
# #cleanloadAllLibsAZ: cleanloadAllLibsAZ.log
# #	$(TestDiff)
#
# #cleanloadAllLibsZA.log: execloadAllLibsZA.log
# #	$(CMDECHO) cat execloadAllLibsZA.log|grep -v Wouter|grep -v NIKHEF|grep -v sourceforge
#
# #cleanloadAllLibsZA: cleanloadAllLibsZA.log
# #	$(TestDiff)
