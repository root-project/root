/// \file
/// \warning This file contains implementation details that will change without notice. User code should never include
/// this header directly.

#ifndef ROOT_RHistUtils
#define ROOT_RHistUtils

namespace ROOT {
namespace Experimental {
namespace Internal {

template <typename T, typename... Ts>
struct LastType : LastType<Ts...> {};
template <typename T>
struct LastType<T> {
   using type = T;
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
