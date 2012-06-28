/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// _climits.h

#ifndef G__CLIMITS_DLL
//  Interpreted version of climits header.  At install time, the setup script
//  tries to compile stl/climits.dll.  If it is successful, compiled version
//  is used.  numeric_limits<> definition in this file is used only if
//  stl/climits.dll is not found.  You can find compiled numeric_limits<>
//  definition in lib/prec_stl/climits and lib/dll_stl/clim.h.
//
//  2004 Jan 3,  Masaharu Goto

#include <limits.h>

enum float_round_style {
  round_indeterminate       = -1,
  round_toward_zero         =  0,
  round_to_nearest          =  1,
  round_toward_infinity     =  2,
  round_toward_neg_infinity =  3
};
enum float_denorm_style {
  denorm_indeterminate = -1,
  denorm_absent = 0,
  denorm present = 1
};

template<class T> class numeric_limits {
 private:
  bool isunsigned() const {return( ((T)(-1)>0) ? 1 : 0 ) ;}
  bool isfloat() const {return( ((T)(0.5)>0) ? 1 : 0 ) ;}
 public:
  static const bool is_specialized = false;
  static T min() throw() { 
    if(isunsigned()) {
      return 0;
    }
    else if(isfloat()) {
      switch(sizeof(T)) {
      case sizeof(float):  return(1.17549435082228751e-38);
      case sizeof(double): return(2.22507385850720138e-308);
      }
    }
    else {
      switch(sizeof(T)) {
      case sizeof(char): return(CHAR_MIN);
      case sizeof(short): return(SHRT_MIN);
      case sizeof(int): return(INT_MIN);
      case sizeof(long): return(LONG_MIN);
      case sizeof(long long): return(LONG_MIN);
      }
    }
    return(0);
  }
  static T max() throw() {
    if(isunsigned()) {
      switch(sizeof(T)) {
      case sizeof(char): return(UCHAR_MAX);
      case sizeof(short): return(USHRT_MAX);
      case sizeof(int): return(UINT_MAX);
      case sizeof(long): return(ULONG_MAX);
      case sizeof(long long): return(ULONG_MAX);
      }
    }
    else if(isfloat()) {
      switch(sizeof(T)) {
      case sizeof(float):  return(3.40282346638528860e38);
      case sizeof(double): return(1.79769313486231571e308);
      }
    }
    else {
      switch(sizeof(T)) {
      case sizeof(char): return(CHAR_MAX);
      case sizeof(short): return(SHRT_MAX);
      case sizeof(int): return(INT_MAX);
      case sizeof(long): return(LONG_MAX);
      case sizeof(long long): return(LONG_MAX);
      }
    }
    return(0);
  }
  static const int  digits = 0;
  static const int  digits10 = 0;
  static const bool is_signed = false;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const int  radix = 0;
  static T epsilon() throw() {
    if(isfloat()) {
      switch(sizeof(T)) {
      case sizeof(float):  return(1.19209289550781250e-7);
      case sizeof(double): return(2.22044604925031308e-16);
      }
    }
    else return(0);
  }
  static T round_error() throw() {
    if(isfloat()) return(1.0);
    else return(0);
  }
  
  static const int  min_exponent = 0;
  static const int  min_exponent10 = 0;
  static const int  max_exponent = 0;
  static const int  max_exponent10 = 0;
  
  static const bool has_infinity = false;
  static const bool has_quiet_NaN = false;
  static const bool has_signaling_NaN = false;
  static const float_denorm_style has_denorm = denorm_absent;
  static const bool has_denorm_loss = false;
  static T infinity() throw() { return(0); }
  static T quiet_NaN() throw() { return(0); }
  static T signaling_NaN() throw() { return(0); }
  static T denorm_min() throw() { return(0); }
  
  static const bool is_iec559 = false;
  static const bool is_bounded = false;
  static const bool is_modulo = false;
  
  static const bool traps = false;
  static const bool tinyness_before = false;
  static const float_round_style round_style = round_toward_zero;
};

#endif // G__CLIMITS_DLL
