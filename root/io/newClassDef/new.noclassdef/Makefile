local_all: main
.PHONY: main

include $(ROOTSYS)/test/Makefile

ifeq ($(PLATFORM),win32)
CXXFLAGS += /TP
LIBSFORLINK = $(LIBS)
else
LIBSFORLINK = 
endif
PACKAGES = namespace template nstemplate
SHARED_LIBS = $(PACKAGES:%=%.$(DllSuf)) 
DICTS = $(PACKAGES:%=%dict.C)
DICTS_H = $(PACKAGES:%=%dict.h)

main: $(SHARED_LIBS)
#	echo $(CXX)
#	echo $(CXXFLAGS)
#	echo $(SHARED_LIBS)

test: main Run.C
	root -q -l -b Run.C

$(DICTS) : %dict.C : %.h %linkdef.h
	rootcint -f $@ -c $^


$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf)
	$(LD) $(SOFLAGS) $(LDFLAGS) $(LIBSFORLINK) $^ $(OutPutOpt) $@

local_clean:
	rm -rf ti_files $(DICTS) $(DICTS_H) ii_files ti_files
	rm -rf *.$(ObjSuf) *.$(DllSuf) *.lib *.def *.root so_locations

clean: local_clean

.C.$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
