// @(#)root/foundation:
// Author: Axel Naumann, 2017-06-02

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTypeTraits
#define ROOT_TTypeTraits

#include <type_traits>

namespace ROOT{

///\class ROOT::TypeTraits::
template <class T>
class IsSmartOrDumbPtr: public std::integral_constant<bool, std::is_pointer<T>::value> {};

template <class P>
class IsSmartOrDumbPtr<std::shared_ptr<P>>: public std::true_type {};

template <class P>
class IsSmartOrDumbPtr<std::unique_ptr<P>>: public std::true_type {};

}
#endif //ROOT_TTypeTraits
