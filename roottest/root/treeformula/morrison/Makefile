all: tests
test: tests


TEST_TARGETS += ArraySyntax
CLEAN_TARGETS += *Dict* Event.root main$(ExeSuf) $(ALL_LIBRARIES) *.log


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


#ArraySyntax: Event.root
#	@echo built

ArraySyntax.log: main_C.$(DllSuf)  $(ROOTCORELIBS) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q run.C > ArraySyntax.log 2>&1

ArraySyntax: ArraySyntax.log
	$(TestDiff)

#main$(ExeSuf) $(ROOT_LOC)/lib/libTreePlayer.$(DllSuf)
#	$(CMDECHO) ./main

main$(ExeSuf): main.C fooDict.C barDict.C foo.C bar.C $(ROOTCORELIBS) $(ROOTV)
	$(CMDECHO) $(CXX) -o$@ $(CXXFLAGS) $(filter %.C,$^) -I. $(LDSTARTOPT) $(ROOTLIBS)

main_C.$(DllSuf) : main.$(ObjSuf) fooDict.$(ObjSuf) barDict.$(ObjSuf) foo.$(ObjSuf) bar.$(ObjSuf) $(ROOTCORELIBS) $(ROOTV)
	$(BuildFromObjs)

%Dict.C: %.h $(ROOTCINT)
	$(CMDECHO) rootcint -f $@ $<+


