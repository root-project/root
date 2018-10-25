#ifndef SERIALIZE_BOOST_H
#define SERIALIZE_BOOST_H 1

namespace Detail {

// Specialization for std::tuple
template <std::size_t I, class Archive, class... Args>
struct tuple_serialize_helper {
   static int impl(Archive& ar, std::tuple<Args...>& t, const unsigned int version) {
      ar & std::get<I - 1>(t);
      tuple_serialize_helper<I - 1, Archive, Args...>::impl(ar, t, version);
      return 0;
   }
};

template <class Archive, class... Args>
struct tuple_serialize_helper<0, Archive, Args...> {
   static int impl(Archive& ar, std::tuple<Args...>& t, const unsigned int) {
      ar & std::get<0>(t);
      return 0;
  }
};

}

namespace boost {
namespace serialization {

// Serialize RunInfo
template <typename Archive, class... Args>
auto serialize(Archive& archive, std::tuple<Args...>& t,
               const unsigned int v) -> void {
   Detail::tuple_serialize_helper<sizeof...(Args), Archive, Args...>::impl(archive, t, v);
}

}
}
#endif // SERIALIZE_BOOST_H
