#ifndef userClass_C
#ifdef ClingWorkAroundMultipleInclude
#define userClass_C
#endif

class TopLevel { 
public: 
   virtual ~TopLevel() {}
   int t;
};
class BottomOne : public TopLevel {
public:
   int b;
};
#endif

// Expose this to the interpreter, even if the library already pulled
// this header (with __ROOTCLING__ and thus without BottomMissing) into
// the interpreter.
#if !defined(__MAKECINT__) && !defined(__ROOTCLING__)
class BottomMissing : public TopLevel {
public:
   int c;
};
#endif
