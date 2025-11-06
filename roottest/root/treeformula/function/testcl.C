class testcl {
public:
   static double calc(double x, double y) { return x*y; }
   static TObject *crap(double x) {
      return gROOT;
   }

   static Longptr_t stuff(double x) {
      return (Longptr_t)ROOT::GetROOT();
   }
};

double mycalc(double t) {
   return 2*t;
};

double mycalc(double t, double y) {
   return y*t;
};

TObject *crap(double x) {
   return gROOT;
};

TObject crap2(double x) {
   return *gROOT;
};
