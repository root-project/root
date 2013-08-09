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
#if !defined(__MAKECINT__) && !defined(__ROOTCLING__)
class BottomMissing : public TopLevel {
public:
   int c;
};
#endif
#endif
