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



$(DICTS) : %dict.C : %.h %linkdef.h
	rootcint -f $@ -c $^


$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf)
	$(LD) $(SOFLAGS) $(LDFLAGS) $(LIBSFORLINK) $^ $(OutPutOpt) $@

local_clean:
	rm -rf ti_files $(DICTS) $(DICTS_H) 
	rm -rf *.$(ObjSuf) *.$(DllSuf) *.lib *.def *.root

clean: local_clean

.C.$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<
