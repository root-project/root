// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef __GNU_CXX_HASH_H
#define __GNU_CXX_HASH_H

#if defined(__GNUC__)
# if defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 800)
#  define __gnu_cxx std
# endif
# if (__GNUC__ < 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ < 3))
// For gcc, the hash_map and hash_set classes are in the extensions area
#  include <ext/hash_set>
#  include <ext/hash_map>
# else
// GCC >= 4.3:
// silence warning
#  define _BACKWARD_BACKWARD_WARNING_H
#  include <backward/hash_set>
#  include <backward/hash_map>
# endif
#elif defined(_WIN32)
// MSDEV.NET has hash_map and hash_set, but different hash functions!
# include <hash_map>
# include <hash_set>
// We normailze everything to __gnu_cxx for windows
# define __gnu_cxx stdext
#elif defined(__ICC) || defined(__ECC)
# include <hash_map>
# include <hash_set>
// We normailze everything to __gnu_cxx for ICC end ECC
# define __gnu_cxx std
#elif defined(__SUNPRO_CC) || defined(_AIX) || (defined(__alpha) && !defined(__linux))
# include <map>
# define __gnu_cxx std
// This is not What we want to do in the end !! FIXME !!
# define hash_map map
# define hash_multimap multimap
namespace std {
inline size_t
distance(std::string* _l,
         std::string* _r) {
   return _r - _l;
}


}
#endif

#include <cstring>


#if defined(_WIN32)

namespace __gnu_cxx {
inline size_t
__gnu_cxx_hash_string(const char* s) {
   unsigned long __h = 0;

   for ( ; *s; ++s) {
      __h = 5 * __h + *s;
   }
   return size_t(__h);
}


template <> class hash_compare<const char*> {
   typedef const char* Key;

public:
   static const size_t bucket_size = 4;
   static const size_t min_buckets = 8;
   size_t
   operator ()(const Key& k) const { return __gnu_cxx_hash_string(k); }

   bool
   operator ()(const Key& k1,
               const Key& k2) const { return strcmp(k1, k2) < 0; }

};

template <> class hash_compare<const char**> {
   typedef const char** Key;

public:
   static const size_t bucket_size = 4;
   static const size_t min_buckets = 8;
   size_t
   operator ()(const Key& k) const { return __gnu_cxx_hash_string(*k); }

   bool
   operator ()(const Key& k1,
               const Key& k2) const { return strcmp(*k1, *k2) < 0; }

};

template <> class hash_compare<const std::string*> {
   typedef const std::string* Key;

public:
   static const size_t bucket_size = 4;
   static const size_t min_buckets = 8;
   size_t
   operator ()(const Key& k) const { return __gnu_cxx_hash_string(k->c_str()); }

   bool
   operator ()(const Key& k1,
               const Key& k2) const { return *k1 < *k2; }

};

/*
   inline size_t hash_value( const std::string * s ) {
   return hash_value(s->c_str());
   }
 */
} // namespace __gnu_cxx

#endif // __ICC, __ECC, _WIN32


#if defined(__GNUC__)
namespace __gnu_cxx {
template <> struct hash<const char**> {
   size_t
   operator ()(const char** __s) const {
# if defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 800)
      return hash_value(* __s);
# else
      return __stl_hash_string(* __s);
# endif
   }
};


template <> struct hash<const std::string*> {
   size_t
   operator ()(const std::string* __s) const {
# if defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 800)
      return hash_value(* __s);
# else
      return __stl_hash_string(__s->c_str());
# endif
   }


};
}

namespace std {
template <> struct equal_to<const char*> :
   public binary_function<const char*, const char*, bool> {
   bool
   operator ()(const char* const& _Left,
               const char* const& _Right) const {
      return strcmp(_Left, _Right) == 0;
   }


};

template <> struct equal_to<const char**> :
   public binary_function<const char**, const char**, bool> {
   bool
   operator ()(const char** const& _Left,
               const char** const& _Right) const {
      return strcmp(*_Left, *_Right) == 0;
   }


};

template <> struct equal_to<const std::string*> :
   public binary_function<const std::string*, const std::string*, bool> {
   bool
   operator ()(const std::string* const& _Left,
               const std::string* const& _Right) const {
      return * _Left == * _Right;
   }


};

} // namespace std

#endif // __GNUC__

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER <= 800)

namespace __gnu_cxx {
template <> class hash_compare<const char*> {
   typedef const char* Key;

public:
   static const size_t bucket_size = 4;
   static const size_t min_buckets = 8;
   size_t
   operator ()(const Key& k) const { return hash_value(k); }

   bool
   operator ()(const Key& k1,
               const Key& k2) const { return strcmp(k1, k2) < 0; }

};

template <> class hash_compare<const char**> {
   typedef const char** Key;

public:
   static const size_t bucket_size = 4;
   static const size_t min_buckets = 8;
   size_t
   operator ()(const Key& k) const { return hash_value(*k); }

   bool
   operator ()(const Key& k1,
               const Key& k2) const { return strcmp(*k1, *k2) < 0; }

};

template <> class hash_compare<const std::string*> {
   typedef const std::string* Key;

public:
   static const size_t bucket_size = 4;
   static const size_t min_buckets = 8;
   size_t
   operator ()(const Key& k) const { return hash_value(*k); }

   bool
   operator ()(const Key& k1,
               const Key& k2) const { return *k1 < *k2; }

};

template <> struct less<const std::string*> {
   typedef const std::string* const Key;
   bool
   operator ()(Key& k1,
               Key& k2) const { return *k1 < *k2; }

};

} // namespace __gnu_cxx

#endif // __INTEL_COMPILER


#if defined(__SUNPRO_CC)

namespace __gnu_cxx {
template <> struct less<const char*> {
   typedef const char* Key;
   bool
   operator ()(Key& k1,
               Key& k2) const { return strcmp(k1, k2) < 0; }

};

template <> struct less<const char**> {
   typedef const char** Key;
   bool
   operator ()(Key& k1,
               Key& k2) const { return strcmp(*k1, *k2) < 0; }

};

template <> struct less<const std::string*> {
   typedef const std::string* Key;
   bool
   operator ()(Key& k1,
               Key& k2) const { return *k1 < *k2; }

};

} // namespace __gnu_cxx

#endif // __SUNPRO_CC


#if defined(_AIX)

namespace __gnu_cxx {
template <> struct less<const char*> {
   typedef const char* Key;
   bool
   operator ()(Key k1,
               Key k2) const { return strcmp(k1, k2) < 0; }

};

template <> struct less<const char**> {
   typedef const char** Key;
   bool
   operator ()(Key k1,
               Key k2) const { return strcmp(*k1, *k2) < 0; }

};

template <> struct less<const std::string*> {
   typedef const std::string* Key;
   bool
   operator ()(Key k1,
               Key k2) const { return *k1 < *k2; }

};

} // namespace __gnu_cxx

#endif // _AIX

#endif // __GNU_CXX_HASH_H
