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
   int b;
};
#if !defined(__MAKECINT__) && !defined(__ROOTCLING__)
class BottomMissing : public TopLevel {
   int c;
};
#endif
#endif
