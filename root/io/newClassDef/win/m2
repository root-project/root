local_all: main
.PHONY: main

include $(ROOTSYS)/test/Makefile

ifeq ($(ARCH),win32)
CXXFLAGS += /TP /GX
LIBSFORLINK = $(LIBS)
else
LIBSFORLINK = 
endif
PACKAGES = namespace template InheritMulti
SHARED_LIBS = $(PACKAGES:%=%.$(DllSuf)) 
DICTS = $(PACKAGES:%=%dict.C)
DICTS_H = $(PACKAGES:%=%dict.h)

main: $(SHARED_LIBS) array_cxx.$(DllSuf)
#	echo $(CXX)
#	echo $(CXXFLAGS)
#	echo $(SHARED_LIBS)

test: main Run.C
	root -q -l -b "Run.C(\"$(DllSuf)\")"
	@echo "All test succeeded in `pwd`"

$(DICTS) : %dict.C : %.h %linkdef.h
	rootcint -f $@ -c $^


$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf)
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
		bindexplib $* $^ > $*.def
		lib -nologo -MACHINE:IX86 $^ -def:$*.def \
		   $(OutPutOpt)$(subst dll,lib,$@)
		$(LD) $(SOFLAGS) $(LDFLAGS) $^ $*.exp $(LIBS) \
		   $(OutPutOpt)$@
else
		$(LD) $(SOFLAGS) $(LDFLAGS) $(LIBSFORLINK) $^ $(OutPutOpt) $@
endif
endif
endif
endif

#	$(LD) $(SOFLAGS) $(LDFLAGS) $(LIBSFORLINK) $^ $(OutPutOpt)$@

local_clean:
	rm -rf ti_files $(DICTS) $(DICTS_H) ii_files ti_files
	rm -rf *.$(ObjSuf) *.$(DllSuf) *.lib *.def *.root so_locations *~

clean: local_clean


array_cxx.$(DllSuf) : array.cxx
	root.exe -l -b -q '.L $<++' 

.C.$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
