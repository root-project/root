TEST_TARGETS += scopeTest

include ../../root/Rules.mk


baddict.C: mycl.C LinkDef.h
	rootcint -f baddict.C -c -DBAD mycl.C LinkDef.h

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
		ld -L/usr/lib/cmplrs/cxx -rpath /usr/lib/cmplrs/cxx \
		   -expect_unresolved "*" -g0 -O1 -shared \
		   /usr/lib/cmplrs/cc/crt0.o /usr/lib/cmplrs/cxx/_main.o \
		   -o $@ $^ -lcxxstd -lcxx -lexc -lots -lc
else
ifeq ($(ARCH),macosx)
# We need to make both the .dylib and the .so
		$(LD) $(SOFLAGS) $^ -lstdc++ $(OutPutOpt) $@
		$(LD) -bundle -undefined suppress -Wl,-x $(LDFLAGS) $^ \
		   $(OutPutOpt) $@
else
ifeq ($(PLATFORM),win32)
		$(LD) $(CXXFLAGS) $(SOFLAGS) $(LDFLAGS) $^ $(LIBS) $(OutPutOpt)$@
else
		$(LD) $(CXXFLAGS) $(SOFLAGS) $(LDFLAGS) $^ $(OutPutOpt) $@
endif
endif
endif
endif
endif

scopeTest: run.C myclbad.so
	root.exe -q -b run.C > output 2>&1 
	diff good.output output

CLEAN_TARGET += *.so output baddict* 

