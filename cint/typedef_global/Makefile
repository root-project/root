TEST_TARGETS += scopeTest scopeTest2

include ../../root/Rules.mk


baddict.C: mycl.C LinkDef.h
	$(CMDECHO) rootcint -f baddict.C -c -DBAD mycl.C LinkDef.h

gooddict.C: mycl.C LinkDef.h
	$(CMDECHO) rootcint -f gooddict.C -c mycl.C LinkDef.h

myclbad.so: baddict.C
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),alphacxx)
# due to a bug in cxx/ld under osf3.xx, one cannot use cxx to generate
# a shared library. One must use ld instead.
		$(CMDECHO) ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
		   -expect_unresolved "*" -g0 -O1 -shared \
		   /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
		   -o $@ $^ -lcxxstd -lcxx -lexc -lots -lc
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(CMDECHO) $(LD) $(SOFLAGS) $^ -lstdc++ $(OutPutOpt) $@
		$(CMDECHO) $(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $@
else
ifeq ($(PLATFORM),win32)
		$(CMDECHO) $(LD) $(CXXFLAGS) $(SOFLAGS) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
else
		$(CMDECHO) $(LD) $(CXXFLAGS) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@
endif
endif
endif
endif
endif


myclgood.so: gooddict.C
ifeq ($(ARCH),aix)
		/usr/ibmcxx/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),aix5)
		/usr/vacpp/bin/makeC++SharedLib $(OutPutOpt) $@ $(LIBS) -p 0 $^
else
ifeq ($(ARCH),alphacxx)
# due to a bug in cxx/ld under osf3.xx, one cannot use cxx to generate
# a shared library. One must use ld instead.
		$(CMDECHO) ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
		   -expect_unresolved "*" -g0 -O1 -shared \
		   /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
		   -o $@ $^ -lcxxstd -lcxx -lexc -lots -lc
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(CMDECHO) $(LD) $(SOFLAGS) $^ -lstdc++ $(OutPutOpt) $@
		$(CMDECHO) $(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $@
else
ifeq ($(PLATFORM),win32)
		$(CMDECHO) $(LD) $(CXXFLAGS) $(SOFLAGS) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
else
		$(CMDECHO) $(LD) $(CXXFLAGS) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@
endif
endif
endif
endif
endif

scopeTest: run.C myclbad.so
	$(CMDECHO) root.exe -l -q -b run.C  2>&1 | grep -v Processing | sed -e s/myclbad.so/mycl.so/ > output 
	$(CMDECHO) diff good.output output

scopeTest2: run.C myclgood.so
	$(CMDECHO) root.exe -l -q -b rungood.C   2>&1 | grep -v Processing | sed -e s/myclgood.so/mycl.so\ / > output  
	$(CMDECHO) diff good.output output

CLEAN_TARGET += *.so output baddict* gooddict*

