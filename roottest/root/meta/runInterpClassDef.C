// Check that the interpreter can create instances of interpreted classes
// that use ClassDef() - but don't have a dictionary and thus do not implement
// many of the functions declared in Rtypes.h's regular ClassDef() macro.

struct InterpClass {
  ClassDef(InterpClass,1);
};


struct Base {
   ClassDefNV(Base, 1)
};
struct Derived: public Base {
   // Make sure that this Base::IsA() is *not* virtual!
   TClass *IsA() const { return (TClass*) (intptr_t) -1; }
};
// Test for the call of Error function
// from within the Inner struct (ROOT-7441)
struct Outer: public TObject {
  struct Inner {
    ClassDef(Inner, 1);
  };
};


int runInterpClassDef() {
  InterpClass testCase;

  Derived d;
  Base* b = &d;
  if (((intptr_t)b->IsA()) == -1)
    return 1; // FAILURE
  return 0;
}
