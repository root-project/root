int core();

int assertSpaces() {
   gInterpreter->ProcessLine(".autodict 0");
   return core();
}

template <typename T>
class X {
public:
   X(): c() {}
   T s[12 * sizeof(T)];
   const T c;
   T* p;
   const T* cp;

   unsigned int f() {
      return 12;
   }
   int g(unsigned long p = sizeof(double)) {
      return (int)p;
   }

   int h(long p = sizeof(unsigned short int)) {
      return (int)p;
   }

   int i(long p = sizeof(double)) {
      return (unsigned short)p;
   }
};

int core() {
   X<long long> xll;
   X<long double> xld;
   X<unsigned long int> xuli;
   X<const X<unsigned long long>*> xcx;

   xll.s[0] = 87465876ll;
   if (xll.c) return 1;
   xll.p = &xll.s[12];
   xll.cp = &xll.c;

   xld.s[9] = xll.s[0];
   if (xld.s[9] != 87465876ll) return 1;

   xcx.s[2] = 0;

   xll.f();
   xll.g();
   xll.h();
   xll.i();

   return 0;
};

