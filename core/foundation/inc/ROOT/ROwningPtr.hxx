/**
 \file ROOT/ROwningPtr.hxx
 \ingroup core
 \author Jonas Rembser
 \author Vincenzo Eduardo Padulano
 \date 2022-08
*/

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ROWNINGPTR
#define ROOT_ROWNINGPTR

#include <memory>
namespace ROOT {

template <typename T, typename = std::enable_if_t<std::is_pointer<T>::value>>
class ROwningPtr {
public:
   using pointer = T;
   using element_type = typename std::remove_pointer<pointer>::type;
   using reference = typename std::add_lvalue_reference<element_type>::type;

   ROwningPtr(T ptr) : _ptr{ptr} {}

   operator pointer() { return _ptr; }
   operator std::unique_ptr<element_type>() { return std::unique_ptr<element_type>{_ptr}; }

   reference operator*() const { return *_ptr; }
   pointer operator->() const { return _ptr; }

private:
   T _ptr;
};

} // namespace ROOT
#endif