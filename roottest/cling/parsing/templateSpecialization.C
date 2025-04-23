class null_type {};

template <class T0 = null_type> class tuple;

template <class T0 > class tuple
{
   typedef null_type inherited;  
};

// The empty tuple
template<> class tuple<null_type> 
{
public:
  typedef int missing;
  typedef null_type inherited;
};

tuple<null_type>::inherited good1;
tuple<null_type>::missing good2;

tuple<>::inherited bad1; // comes from incorrect template
tuple<>::missing bad2;

