CLEAN_TARGETS += *.root *~ *.$(ObjSuf) *Dict.cxx *Dict.h *.$(DllSuf) reader writer Foo.so \
	 write.out
TEST_TARGETS +=  check

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


LIBS = $(ROOTLIBS)

#all:	Foo.$(DllSuf) reader$(ExeSuf) writer$(ExeSuf)
#test: check

Foo.$(DllSuf): Foo.h Foo.cxx
	$(CMDECHO) rootcint -f FooDict.cxx -c Foo.h+
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c Foo.cxx 
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c FooDict.cxx 
ifeq ($(ARCH),aix)
                $(CMDECHO) /usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
                $(CMDECHO) /usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
                $(CMDECHO) $(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
                $(CMDECHO) $(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
                   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(PLATFORM),win32)
	$(CMDECHO) bindexplib Foo Foo.$(ObjSuf) FooDict.$(ObjSuf) > Foo.def
	$(CMDECHO) lib -nologo -MACHINE:IX86  Foo.$(ObjSuf) FooDict.$(ObjSuf) -def:Foo.def \
                   $(OutPutOpt)$(subst dll,lib,$@)
	$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) Foo.$(ObjSuf) FooDict.$(ObjSuf) Foo.exp $(LIBS) \
                   $(OutPutOpt)$@
else
	$(CMDECHO) $(LD) $(SOFLAGS) $(LDFLAGS) $(LIBSFORLINK) Foo.$(ObjSuf) FooDict.$(ObjSuf) $(OutPutOpt) $@
endif
endif
endif
endif

#	g++ -shared -Wl,-soname,Foo.$(DllSuf) -o Foo.$(DllSuf) Foo.$(ObjSuf) FooDict.$(ObjSuf)

reader$(ExeSuf) :reader.C
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $<  
	$(CMDECHO) $(LD) reader.$(ObjSuf) $(LDFLAGS) $(LIBS) $(OutPutOpt)$@



#g++ -o $@ reader.$(ObjSuf) `root-config --libs` \
#	-Wl,-rpath,`root-config --libdir`

writer$(ExeSuf):writer.C Foo.$(DllSuf) 
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< 
ifeq ($(PLATFORM),win32)
	$(CMDECHO) $(LD) writer.$(ObjSuf) $(LDFLAGS) $(LIBS) Foo.lib $(OutPutOpt)$@
else
	$(CMDECHO) $(LD) writer.$(ObjSuf) $(LDFLAGS) $(LIBS) Foo.so $(OutPutOpt)$@
endif

#	g++ -o $@ writer.$(ObjSuf) Foo.$(DllSuf) `root-config --libs` \
#		-Wl,-rpath,`root-config --libdir` -Wl,-rpath,.

dist:
	mkdir -p treeproblem 
	cp Foo.{cxx,h} tree.{C,h} reader.C writer.C Makefile README \
		treeproblem
	tar -czvf treeproblem.tar.gz treeproblem 
	rm -rf treeproblem 

check:Foo.$(DllSuf) reader$(ExeSuf) writer$(ExeSuf) 
	$(CMDECHO) root -b -l -q writer.C > write.out
ifeq ($(FAIL),) 
	@echo Test known to fail in `pwd`
else
	root -b -l -q reader.C
endif

#clean:
#	rm -f *.root *~ *.$(ObjSuf) *Dict.cxx *Dict.h *.$(DllSuf) reader writer

