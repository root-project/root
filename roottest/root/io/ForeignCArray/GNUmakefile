TEST_TARGETS = ForeignCArrayTest
CLEAN_TARGETS = MyClassesDict*.* mytest$(EXESUF)

include ../../Rules.mk


MyClassesDict.cxx: MyClasses.h LinkDef.h $(ROOTSYS)/bin/rootcint
	rootcint -f MyClassesDict.cxx MyClasses.h LinkDef.h

mytest$(ExeSuf): MyClassesDict.$(ObjSuf) mytest.$(ObjSuf)
	$(LD)  $^ $(LDFLAGS) $(ROOTLIBS) $(OutPutOpt)$@

#	c++ $(CXXP) -o mytest$(ExeSuf) mytest.$(ObjSuf) MyClassesDict.$(ObjSuf) $(LDFLAGS)

ForeignCArrayTest: mytest$(ExeSuf)
	@[ ! -e mytest$(ExeSuf) ] || ./mytest$(ExeSuf)


mytest.$(ObjSuf): MyClasses.h 




