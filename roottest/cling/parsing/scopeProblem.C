namespace boost {
   template < typename T> struct is_function {  enum value_holder { value = false }; };


   template <bool If> struct IF { typedef int RET; };
   
#ifdef WORKS
   template <class T> struct wrap_non_storeable_type {
      typedef typename IF<boost::is_function<T>::value>::RET type;
   };
#else
   template <class T> struct wrap_non_storeable_type {
      typedef typename IF< ::boost::is_function<T>::value>::RET type;
   };
#endif
   
} // end of namespace boost

typedef boost::is_function<long>::value_holder vv_t;
typedef boost::wrap_non_storeable_type<long> tt_t;
