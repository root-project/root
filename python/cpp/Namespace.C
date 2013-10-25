namespace PR_NS_A {
   int sa = 1;

   typedef int TD_NS_A_t;
   TD_NS_A_t tsa = -1;
   const TD_NS_A_t ctsa = -1;

   struct PR_ST_B {
      PR_ST_B() { fb = -2; }
      int fb;
      static int sb;

      struct PR_ST_C {
         PR_ST_C() { fc = -3; }
         int fc;
         static int sc;
      };
   };

   namespace PR_NS_D {
      int sd = 4;

      struct PR_ST_E {
         PR_ST_E() { fe = -5; }
         int fe;
         static int se;

         struct PR_ST_F {
            PR_ST_F() { ff = -6; }
            int ff;
            static int sf;
         };
      };

   } // namespace PR_NS_D

} // namespace PR_NS_A

namespace PR_NS_A {      // second namespace to check for updates

   TD_NS_A_t tsa2 = -1;
   const TD_NS_A_t ctsa2 = -1;

} // namespace PR_NS_A

int PR_NS_A::PR_ST_B::sb                   = 2;
int PR_NS_A::PR_ST_B::PR_ST_C::sc          = 3;
int PR_NS_A::PR_NS_D::PR_ST_E::se          = 5;
int PR_NS_A::PR_NS_D::PR_ST_E::PR_ST_F::sf = 6;


class CtorWithDefaultInGBL {
public:
   int data;
   CtorWithDefaultInGBL( int i = -1 ) : data( i ) {}
};

namespace PR_NS_A {
   class CtorWithDefaultInNS {
   public:
      int data;
      CtorWithDefaultInNS( int i = -1 ) : data( i ) {}
   };
}

