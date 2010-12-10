#

include $(ROOTSYS)/test/Makefile.arch

#------------------------------------------------------------------------------

STRESSTMVAO   = stressTMVA.$(ObjSuf)
STRESSTMVAS   = stressTMVA.$(SrcSuf)
STRESSTMVA    = stressTMVA$(ExeSuf)

OBJS          = $(STRESSTMVAO)

PROGRAMS      = $(STRESSTMVA)

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)

all:            $(PROGRAMS)

$(STRESSTMVA):  $(STRESSTMVAO) 
	$(LD) $(LDFLAGS) $< $(LIBS) -lTMVA -lMinuit -lXMLIO -lMLP -lTreePlayer $(OutPutOpt)$@
	$(MT_EXE)
	@echo "$@ done"

.$(SrcSuf).$(ObjSuf):
	$(CXX) -g $(CXXFLAGS) -c $<


