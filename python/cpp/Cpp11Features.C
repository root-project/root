/*
  File: roottest/python/cpp/Cpp11Features.C
  Author: WLavrijsen@lbl.gov
  Created: 11/25/13
  Last: 11/26/13
*/

#if defined(__GXX_EXPERIMENTAL_CXX0X) || __cplusplus >= 201103L

#include <memory>


// for shared_ptr testing
class MyCounterClass {
public:
   static int counter;

public:
   MyCounterClass() { ++counter; }
   MyCounterClass( const MyCounterClass& ) { ++counter; }
   ~MyCounterClass() { --counter; }
};

int MyCounterClass::counter = 0;

std::shared_ptr< MyCounterClass > CreateMyCounterClass() {
  return std::shared_ptr< MyCounterClass >( new MyCounterClass );
}


// from gcc's ext/concurrence.h, typed enum static defined in header
namespace PyTest {

  enum _Lock_policy { _S_single, _S_mutex, _S_atomic };
  static const _Lock_policy __default_lock_policy = _S_mutex;

} // PyTest namespace


#endif
