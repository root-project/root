// Tests for Get

template <typename U>
struct St {
   struct A {
      int a;
      static int s;
   };

   typedef A T;
};


#ifndef __GCCXML__
template <typename U>
int St<U>::A::s = 43;
#endif

template class St<int>;
