SRC=$(wildcard *.cxx)
HDR=$(wildcard *.h)
LIB=libTreeReader_C.so
ROOTFILE=tr.root
all: test

$(LIB): $(SRC) $(HDR)
	@echo '#include "MyParticle.h"' > libTreeReader.C
	@echo '#include "TTreeReader.cxx"' >> libTreeReader.C
	@echo '#include "TTreeReaderValue.cxx"' >> libTreeReader.C
	@echo '#ifdef __CINT__' >> libTreeReader.C 
	@echo '#pragma link C++ class MyPos+;' >> libTreeReader.C 
	@echo '#pragma link C++ class MyParticle+;' >> libTreeReader.C 
	@echo '#pragma link C++ class ParticleHolder+;' >> libTreeReader.C 
	@echo '#pragma link C++ class TTreeReader+;' >> libTreeReader.C 
	@echo '#pragma link C++ class TTreeReaderValuePtrBase+;' >> libTreeReader.C 
	@echo '#pragma link C++ class TTreeReaderArrayBase+;' >> libTreeReader.C 
	@echo '#endif // __CINT__' >> libTreeReader.C 
	@root -l -b -q build.C\(\"libTreeReader.C\",\"L\"\)

$(ROOTFILE): twrite.C $(HDR)
	root -l -b -q build.C\(\"twrite.C\",\"x\",\"$(LIB)\"\)

test: $(LIB) $(ROOTFILE)
	root -l -b -q build.C\(\"tread.C\",\"x\",\"$(LIB)\"\)
	#root.exe -l -b -q tread.C

clean:
	rm -f *.so libTreeReader.C *.d AutoDict_* *~ $(ROOTFILE)
