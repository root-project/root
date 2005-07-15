class TopLevel { 
public: 
   virtual ~TopLevel() {}
   int t;
};
class BottomOne : public TopLevel {
   int b;
};
#if !defined(__MAKECINT__)
class BottomMissing : public TopLevel {
   int c;
};
#endif
