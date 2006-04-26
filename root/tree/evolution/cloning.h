class cloning {
 public:
#if defined(ONE)
   float fVar;
   cloning() : fVar(3.33) {}
      //ClassDef(cloning,1);
#elif defined(TWO)
   double fVar;
   cloning() : fVar(7.77) {}
      //ClassDef(cloning,2);
#else
#error missing case
#endif
};

#ifdef __MAKECINT__
#pragma link C++ class cloning+;
#endif
