// This is a Mock extracted from LHCb's FastAllocVector code

#ifndef MOCK_FastAllocVector_H
#define MOCK_FastAllocVector_H 1

#define XSTR(x) STR(x)
#define STR(x) #x


// Include files
#include <vector>

// need to use macro as template typedefs don't work yet :(

// Check if memory pools are isabled completely
#ifndef GOD_NOALLOC

#if defined(__GNUC__) && !defined( _LIBCPP_VERSION )

   #if __GNUC__ > 3
      // This is GCC 4 and above. Use the allocators
      #include <ext/mt_allocator.h>
      #define LHCb_FastAllocVector_allocator(TYPE) __gnu_cxx::__mt_alloc< TYPE >
   #else

   #if __GNUC_MINOR__ > 3
      // This is gcc 3.4.X so has the custom allocators
      #include <ext/mt_allocator.h>
      #define LHCb_FastAllocVector_allocator(TYPE) __gnu_cxx::__mt_alloc< TYPE >

   #else
      // This is older than gcc 3.4.X so use standard allocator
      #define LHCb_FastAllocVector_allocator(TYPE) std::allocator< TYPE >
      #endif
#endif

// GOD_NOALLOC DEFINED
#else
   // Not GNUC, so disable allocators
   #define LHCb_FastAllocVector_allocator(TYPE) std::allocator< TYPE >
#endif

#else

// GOD NOALLOC - Disable memory pools completely
#define LHCb_FastAllocVector_allocator(TYPE) std::allocator< TYPE >

#endif

namespace {
std::vector<int,LHCb_FastAllocVector_allocator(int)> inst;
}

#endif
