// -*- c++ -*-
/*  
  Copyright 2000, Karl Einar Nelson

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
*/
#ifndef    SIGC_TRAIT
#define    SIGC_TRAIT

#include <sigc++/sigcconfig.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

#ifdef SIGC_CXX_SPECIALIZE_REFERENCES
template <class T>
struct Trait
  {
    typedef T  type;
    typedef const T& ref;
  };

template <class T>
struct Trait<T&>
  {
    typedef T& type;
    typedef T& ref;
  };
#else
// for really dumb compilers, we have to copy rather than reference
template <class T>
struct Trait
  {
    typedef T  type;
    typedef T  ref;
  };
#endif

#ifdef SIGC_CXX_VOID_RETURN
template <>
struct Trait<void>
  {
    typedef void type;
  };
#else
template <>
struct Trait<void>
  {
    typedef int type;
  };
#endif

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif  // SIGC_TRAIT
