all: tests
test: tests


TEST_TARGETS += ArraySyntax
CLEAN_TARGETS += *Dict* *.root main$(ExeSuf) $(ALL_LIBRARIES) *.log


ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

include $(ROOTTEST_HOME)/scripts/Rules.mk


#ArraySyntax: Event.root
#	@echo built

ArraySyntax: main_C.$(DllSuf)  $(ROOTCORELIBS) $(ROOTV)
	$(CMDECHO) root.exe -b -l -q run.C > ArraySyntax.log

#main$(ExeSuf) $(ROOTSYS)/lib/libTreePlayer.$(DllSuf)
#	$(CMDECHO) ./main

main$(ExeSuf): main.C fooDict.C barDict.C foo.C bar.C $(ROOTCORELIBS) $(ROOTV)
	$(CMDECHO) $(CXX) -o$@ $(CXXFLAGS) $(filter %.C,$^) -I. $(LDSTARTOPT) $(shell eval root-config --libs)

main_C.$(DllSuf) : main.$(ObjSuf) fooDict.$(ObjSuf) barDict.$(ObjSuf) foo.$(ObjSuf) bar.$(ObjSuf) $(ROOTCORELIBS) $(ROOTV)
	$(BuildFromObjs)

%Dict.C: %.h $(ROOTCINT)
	$(CMDECHO) rootcint -f $@ -c $<


