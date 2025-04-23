CLEAN_TARGETS += file.root memory.root *~ *.$(ObjSuf) *Dict.cxx *Dict.h *.$(DllSuf) reader writer Foo.so \
	 *.log $(ALL_LIBRARIES) *.exe trial.root
TEST_TARGETS +=  check Memory ucharshow

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

LIBS = $(ROOTLIBS)

ifeq ($(PLATFORM),macosx)
ifeq ($(MACOSX_MINOR),)
  export MACOSX_MINOR := $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2)
endif
endif

#all:	Foo.$(DllSuf) reader$(ExeSuf) writer$(ExeSuf)
#test: check

#Foo.$(DllSuf): Foo.$(ObjSuf)
#	$(BuildFromObjs)

FooDict.cxx: Foo.h
	$(CMDECHO) rootcint -f FooDict.cxx Foo.h+

Foo.cxx : Foo.h

Foo.$(DllSuf): FooDict.$(ObjSuf) Foo.$(ObjSuf) Foo.h Foo.cxx
ifeq ($(ARCH),aix)
                $(CMDECHO) /usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
                $(CMDECHO) /usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(PLATFORM),win32)
	$(CMDECHO) bindexplib Foo Foo.$(ObjSuf) FooDict.$(ObjSuf) > Foo.def
	$(CMDECHO) lib -nologo -MACHINE:IX86  Foo.$(ObjSuf) FooDict.$(ObjSuf) -def:Foo.def \
                   $(OutPutOpt)$(subst dll,lib,$@)
	$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) Foo.$(ObjSuf) FooDict.$(ObjSuf) Foo.exp $(LIBS) \
                   $(OutPutOpt)$@
else
	$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $(LIBS) Foo.$(ObjSuf) FooDict.$(ObjSuf) $(OutPutOpt) $@
endif
endif
endif

#	g++ -shared -Wl,-soname,Foo.$(DllSuf) -o Foo.$(DllSuf) Foo.$(ObjSuf) FooDict.$(ObjSuf)

reader$(ExeSuf) :reader.$(ObjSuf)
	$(CMDECHO) $(LD) reader.$(ObjSuf) $(LDFLAGS) $(LIBS) $(OutPutOpt)$@



#g++ -o $@ reader.$(ObjSuf) `root-config --libs` \
#	-Wl,-rpath,`root-config --libdir`

writer$(ExeSuf):writer.$(ObjSuf) Foo.$(DllSuf) 
	$(CMDECHO) $(LD) writer.$(ObjSuf) $(LDFLAGS) $(LIBS) Foo.$(LibSuf) $(OutPutOpt)$@

ifneq ($(ClingWorkAroundMissingDynamicScope),)
write.log: CALLROOTEXE += -e 'gSystem->Load("Foo.so");'

reader.log: CALLROOTEXE += -e 'gSystem->Load("Foo.so");'
endif

ifneq ($(ClingWorkAroundMissingAutoLoading),)
write.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'

reader.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

write.log:  writer$(ExeSuf) Foo.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q writer.C > write.log

reader.log: Foo.$(DllSuf) reader$(ExeSuf) write.log
	$(CMDECHO) $(CALLROOTEXE) -b -l -q reader.C > reader.log

check: reader.log
	$(CMDECHO)diff -b -w reader.ref reader.log

Memory: Memory.log
	$(TestDiff)

ucharshow: ucharshow.log
	$(TestDiff)
