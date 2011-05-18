class forarg {};
template <class T, class const_reference> class simple {
public:
   typedef T value;
   typedef forarg &refvalue;
   typedef const value &const_refvalue;
   simple() {};
   forarg Init0() {};
   const forarg& Init1(forarg &one) {};
   refvalue Init2(forarg *one) {};
   const_refvalue Init3(forarg *one) {};
   const_refvalue Init4(forarg *one) const {};
   const_reference operator*() const {};
};

simple<forarg*,const forarg*&> x;

int main() 
{
   return 0;
}
