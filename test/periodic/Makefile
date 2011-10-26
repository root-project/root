############# MACROS ##############

RC   := root-config
ifeq ($(shell which $(RC) 2>&1 | sed -ne "s@.*/$(RC)@$(RC)@p"),$(RC))
include $(shell $(RC) --etcdir)/Makefile.arch
else
$(error Please make sure $(RC) can be found in path)
endif

DICT  := NdbDict.$(SrcSuf)
DICTH := $(DICT:.$(SrcSuf)=.h)
DICTO := $(DICT:.$(SrcSuf)=.$(ObjSuf))

DBSRCS := $(wildcard Ndb*.$(SrcSuf))

XSSRCS := $(wildcard XS*.$(SrcSuf))

SRCS := $(XSSRCS) $(DBSRCS)

HDRS := $(SRCS:.$(SrcSuf)=.h) NdbLinkDef.h
HDRS := $(filter-out $(DICTH),$(HDRS))

TARGETLIB = libNdb.$(DllSuf)
TARGETPRG = XSGui$(ExeSuf)

DBOBJS = $(DBSRCS:.$(SrcSuf)=.$(ObjSuf))
XSOBJS = $(XSSRCS:.$(SrcSuf)=.$(ObjSuf))

############# RULES ###############

.$(SrcSuf).$(ObjSuf):
	$(CXX) $(CXXFLAGS) -c $<

############# TARGETS #############

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) $(ExeSuf) .$(DllSuf)

all:	$(TARGETPRG) $(TARGETLIB)

$(DICT): $(HDRS)
	@echo "Generating dictionary $@..."
	$(ROOTCINT) -f $@ -c $^

$(TARGETPRG): $(XSOBJS) $(DBOBJS) $(DICTO)
	$(LD) $(LDFLAGS) $^ $(GLIBS) $(OutPutOpt)$@
	@echo "$@ done"

$(TARGETLIB): $(DBOBJS) $(DICTO)
ifeq ($(PLATFORM),macosx)
	$(LD) $(SOFLAGS)$@ $(LDFLAGS) $^ $(OutPutOpt) $@ $(LIBS)
else
	$(LD) $(SOFLAGS) $(LDFLAGS) $^ $(GLIBS) $(OutPutOpt)$@
endif

distclean: clean
	@rm -f $(TARGETPRG) $(TARGETLIB) $(DICT) $(DICTH)

clean:
	@rm -f $(DBOBJS) $(XSOBJS) $(DICTO)
