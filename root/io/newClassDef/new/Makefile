local_all: main
.PHONY: main

include $(ROOTSYS)/test/Makefile

ifeq ($(PLATFORM),win32)
CXXFLAGS += /TP
LIBSFORLINK = $(LIBS)
else
ifeq ($(ARCH),solarisCC)
CXXFLAGS += -ptr.
else
LIBSFORLINK = 
endif
endif

ifeq ($(ARCH),aix5)
PACKAGES = namespace template InheritMulti
else
PACKAGES = namespace template nstemplate InheritMulti
endif

SHARED_LIBS = $(PACKAGES:%=%.$(DllSuf)) 
DICTS = $(PACKAGES:%=%dict.C)
DICTS_H = $(PACKAGES:%=%dict.h)

main: $(SHARED_LIBS)
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

#		$(LD) $(SOFLAGS) $(LDFLAGS) $(LIBSFORLINK) $^ $(OutPutOpt)$@

local_clean:
	rm -rf ti_files $(DICTS) $(DICTS_H) ii_files ti_files
	rm -rf *.$(ObjSuf) *.$(DllSuf) *.lib *.def *.root so_locations *~
	rm -rf *.exp .def

clean: local_clean
	

.C.$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<

