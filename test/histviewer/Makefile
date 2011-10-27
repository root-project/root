############# MACROS ##############

RC        := root-config
ifeq ($(shell which $(RC) 2>&1 | sed -ne "s@.*/$(RC)@$(RC)@p"),$(RC))
include $(shell $(RC) --etcdir)/Makefile.arch
else
ROOTSYS = ../..
include $(ROOTSYS)/etc/Makefile.arch
endif

DICT      := HistViewerDict.$(SrcSuf)
DICTH     := $(DICT:.$(SrcSuf)=.h)
DICTO     := $(DICT:.$(SrcSuf)=.$(ObjSuf))

SRCS      := $(wildcard *.$(SrcSuf))

HDRS      := $(SRCS:.$(SrcSuf)=.h) LinkDef.h
HDRS      := $(filter-out $(DICTH),$(HDRS))

TARGETLIB := libHistViewer.$(DllSuf)

OBJS      := $(SRCS:.$(SrcSuf)=.$(ObjSuf))

############# RULES ###############

.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<

############# TARGETS #############

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf) .$(DllSuf)

all:    $(TARGETLIB)

$(DICT): $(HDRS)
	@echo "Generating dictionary $@..."
	$(ROOTCINT) -f $@ -c $^

$(TARGETLIB): $(OBJS) $(DICTO)
ifeq ($(PLATFORM),macosx)
	$(LD) $(SOFLAGS)$@ $(LDFLAGS) $^ $(OutPutOpt) $@ $(GLIBS)
else
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(GLIBS) $(OutPutOpt)$@
endif

distclean: clean
	@rm -f $(TARGETLIB) $(DICT) $(DICTH)

clean:
	@rm -f $(OBJS) $(DICTO)
