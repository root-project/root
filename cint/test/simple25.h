template <typename T> class C {};
template <typename T> class iterator 
{
public:
   typedef T value;
   typedef T& reference;      
};
template <typename T> class cont
{
public:
   typedef iterator<T> iter;
   typename iter::reference at() 
   { 
#ifndef __CINT__
      static typename iter::value t;
#endif
      return t; 
   }
   //typename iter::reference& at2();
};

cont<C<float> > c;
