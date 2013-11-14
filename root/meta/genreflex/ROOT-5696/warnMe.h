
// to be probably replaced by #if __cplusplus >= 201103L
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  #define CONSTEXPR constexpr 
#else
  #define CONSTEXPR const 
#endif


class A{

public:

   CONSTEXPR float _growth_factor() { return 1.5; } //from boost
   int returnInt (int i) { register int a=12; return a*i;}
};

