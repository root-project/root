namespace boost {
   namespace tuples {
      class null_type {};
      template <typename T0=null_type> class tuple {};
   }
}
namespace boost {
   using tuples::tuple;
}

boost::tuples::null_type zz;
boost::tuples::tuple<int> aa;
boost::tuple<int> bb;
