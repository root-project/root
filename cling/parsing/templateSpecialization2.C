namespace A {
class null_type;

template <
  class T0 = null_type, class T1 = null_type, class T2 = null_type,
  class T3 = null_type, class T4 = null_type, class T5 = null_type,
  class T6 = null_type, class T7 = null_type, class T8 = null_type,
  class T9 = null_type>
class tuple;

template <class T0, class T1, class T2, class T3, class T4,
          class T5, class T6, class T7, class T8, class T9>

class tuple //:
   //  public detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
{
   typedef null_type inherited;  
};

// The empty tuple
template <>
class tuple<null_type, null_type, null_type, null_type, null_type, 
            null_type, null_type, null_type, null_type, null_type> :
  public null_type
{
public:
  typedef int missing;
  typedef null_type inherited;
};

tuple<null_type, null_type, null_type, null_type, null_type, 
      null_type, null_type, null_type, null_type, null_type>::missing xyz1;
tuple<>::missing xyz;

}
