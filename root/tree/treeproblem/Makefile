include ../../Rules.mk

LIBS = $(ROOTLIBS)

all:	Foo.$(DllSuf) reader writer 

Foo.$(DllSuf):Foo.h Foo.cxx
	rootcint -f FooDict.cxx -c Foo.h+
	$(CXX) $(CXXFLAGS) -c Foo.cxx 
	$(CXX) $(CXXFLAGS) -c FooDict.cxx 
ifeq ($(ARCH),aix)
                /usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
                /usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
                $(LD) $(SOFLAGS) $(EVENTO) $(OutPutOpt) $(EVENTSO)
                $(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
                   $(OutPutOpt) $(subst .$(DllSuf),.so,$@)
else
ifeq ($(PLATFORM),win32)
	bindexplib Foo Foo.$(ObjSuf) FooDict.$(ObjSuf) > Foo.def
	lib -nologo -MACHINE:IX86  Foo.$(ObjSuf) FooDict.$(ObjSuf) -def:Foo.def \
                   $(OutPutOpt)$(subst dll,lib,$@)
	$(LD) $(SOFLAGS) $(LDFLAGS) Foo.$(ObjSuf) FooDict.$(ObjSuf) Foo.exp $(LIBS) \
                   $(OutPutOpt)$@
else
                $(LD) $(SOFLAGS) $(LDFLAGS) $(LIBSFORLINK) $^ $(OutPutOpt) $@
endif
endif
endif
endif

#	g++ -shared -Wl,-soname,Foo.$(DllSuf) -o Foo.$(DllSuf) Foo.$(ObjSuf) FooDict.$(ObjSuf)

reader:reader.C
	$(CXX) $(CXXFLAGS) -c $<  
	$(LD) reader.$(ObjSuf) $(LDFLAGS) $(LIBS) $(OutPutOpt)$@

	#g++ -o $@ reader.$(ObjSuf) `root-config --libs` \
	#	-Wl,-rpath,`root-config --libdir`

writer:writer.C Foo.$(DllSuf) 
	$(CXX) $(CXXFLAGS) -c $< 
	$(LD) writer.$(ObjSuf) $(LDFLAGS) $(LIBS) Foo.lib $(OutPutOpt)$@

#	g++ -o $@ writer.$(ObjSuf) Foo.$(DllSuf) `root-config --libs` \
#		-Wl,-rpath,`root-config --libdir` -Wl,-rpath,.

dist:
	mkdir -p treeproblem 
	cp Foo.{cxx,h} tree.{C,h} reader.C writer.C Makefile README \
		treeproblem
	tar -czvf treeproblem.tar.gz treeproblem 
	rm -rf treeproblem 

check:Foo.$(DllSuf) reader writer 
	root -b -l -q writer.C 
	root -b -l -q reader.C

#clean:
#	rm -f *.root *~ *.$(ObjSuf) *Dict.cxx *Dict.h *.$(DllSuf) reader writer

CLEAN_TARGET += *.root *~ *.$(ObjSuf) *Dict.cxx *Dict.h *.$(DllSuf) reader writer
