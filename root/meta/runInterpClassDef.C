// Check that the interpreter can create instances of interpreted classes
// that use ClassDef() - but don't have a dictionary and thus do not implement
// many of the functions declared in Rtypes.h's regular ClassDef() macro.

struct InterpClass {
  ClassDef(InterpClass,1);
};

void runInterpClassDef() {
  InterpClass testCase;
}
