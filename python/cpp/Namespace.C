namespace A {
   int sa = 1;

   struct B {
      B() { fb = -2; }
      int fb;
      static int sb;

      struct C {
         C() { fc = -3; }
         int fc;
         static int sc;
      };
   };

   namespace D {
      int sd = 4;

      struct E {
         E() { fe = -5; }
         int fe;
         static int se;

         struct F {
            F() { ff = -6; }
            int ff;
            static int sf;
         };
      };

   } // namespace D

} // namespace A

int A::B::sb       = 2;
int A::B::C::sc    = 3;
int A::D::E::se    = 5;
int A::D::E::F::sf = 6;
