LIB:=libReent.so
LLVM_PREFIX="$(shell llvm-config --prefix)"
INC:=-I$(LLVM_PREFIX)/include
CXXFLAGS+=-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -DHOST_TARGET=\"$(shell llvm-config --host-target)\" -DCLANG_SYS_HEADERS=\"$(LLVM_PREFIX)/Headers\"

all: run

$(LIB): reent.cxx
	g++ -shared -o$@ $< $(INC) $(CXXFLAGS) -fPIC

run: $(LIB)
	cat runreent.cxx | cling
