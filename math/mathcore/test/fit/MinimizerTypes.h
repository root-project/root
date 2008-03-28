#ifndef ROOT_Fit_MinimizerTypes
#define ROOT_Fit_MinimizerTypes

// define a different type so easy to see in Shark
struct TMINUIT {
   static std::string name() { return "Minuit"; }
   static std::string name2() { return ""; }
};
struct TFUMILI {
   static std::string name() { return "Fumili"; }
   static std::string name2() { return ""; }
};
struct MINUIT2 {
   static std::string name() { return "Minuit2"; }
   static std::string name2() { return ""; }
};
struct FUMILI2 {
   static std::string name() { return "Fumili2"; }
   static std::string name2() { return ""; }
};
struct DUMMY {
   static std::string name() { return "Dummy"; }
   static std::string name2() { return ""; }
};
struct GSL_FR {
   static std::string name() { return "GSLMultiMin"; }
   static std::string name2() { return "ConjugateFR"; }
};
struct GSL_PR {
   static std::string name() { return "GSLMultiMin"; }
   static std::string name2() { return "ConjugatePR"; }
};
struct GSL_BFGS {
   static std::string name() { return "GSLMultiMin"; }
   static std::string name2() { return "BFGS"; }
};
struct GSL_BFGS2 {
   static std::string name() { return "GSLMultiMin"; }
   static std::string name2() { return "BFGS2"; }
};
struct GSL_NLS {
   static std::string name() { return "GSLMultiFit"; }
   static std::string name2() { return ""; }
};

struct LINEAR { 
   static std::string name() { return "Linear"; }
   static std::string name2() { return ""; }
};

#endif
