/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// 030107exception.txt

#ifndef R_EXCEPT_H
#define R_EXCEPT_H

#include <stddef.h>
#include <exception>
//#include "Rtypes.h"

// Work-around for SunOS anomaly caused by math.h, which defines a
// structure called exception, resulting in ambiguous usage in the
// generated Cint file.
#if defined(__INTEL_COMPILER)  
using std::exception;
#else
#define exception std::exception
#endif

// Exception class r_exception, derived from std::exception, and
// defined at global scope.
class r_exception: public exception {
  public:
  r_exception() : exception() { }
  virtual ~r_exception() throw() { }
};

// Exception class r_space_exception, derived from std::exception, and
// defined in r_space namespace.
namespace r_space {

  class r_space_exception: public exception {
    public:
    r_space_exception()  : exception(){ }
    virtual ~r_space_exception() throw() { }
  };

  namespace eh1 {
    class eh_exception1 : public exception {
    public:
      eh_exception1() : exception() { }
      virtual ~eh_exception1() throw() { }
    };

    namespace errorhandling {
      class eh_exception: public exception {
      public:
	eh_exception() : exception() { }
	virtual ~eh_exception() throw() { }
      };
    }
  }
}

// Function which throws r_exception.
void throw_r_exception(void) throw(r_exception) {
  throw r_exception(); 
}

// Function which throws r_space_exception.
void throw_r_space_exception(void) throw(r_space::r_space_exception) {
  throw r_space::r_space_exception();
}

// Function which throws std::exception.
void throw_std_exception(void) throw(exception) {
  throw exception(); 
}

// Function which throws r_space_exception.
void throw_r_space_eh_exception(void) throw(r_space::eh1::eh_exception1) {
  throw r_space::eh1::eh_exception1();
}


// Function which throws r_space_exception.
void throw_r_space_eh_errorhandling_exception(void) throw(r_space::eh1::errorhandling::eh_exception) {
  throw r_space::eh1::errorhandling::eh_exception();
}


void throw_int(void) throw(int) {throw 1234;}
void throw_long(void) throw(long) {throw 5678L;}
static int xxx=999;
void throw_void(void) throw(void*) {throw((void*)(&xxx));}
void throw_float(void) throw(float) {throw (float)1.23;}
void throw_double(void) throw(double) {throw 3.14;}

#ifdef __MAKECINT__
#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;
#endif

#endif

