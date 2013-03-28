// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Tools
#define Reflex_Tools

// Include files
#include "Reflex/Kernel.h"
#include <vector>
#include <typeinfo>
#include <string>

namespace Reflex {
// forward declarations
class Function;
class Type;

enum EFUNDAMENTALTYPE {
   kCHAR,
   kSIGNED_CHAR,
   kSHORT_INT,
   kINT,
   kLONG_INT,
   kUNSIGNED_CHAR,
   kUNSIGNED_SHORT_INT,
   kUNSIGNED_INT,
   kUNSIGNED_LONG_INT,
   kBOOL,
   kFLOAT,
   kDOUBLE,
   kLONG_DOUBLE,
   kVOID,
   kLONGLONG,
   kULONGLONG,
   kNOTFUNDAMENTAL
};


namespace Tools {
/**
 * GetFundamentalType will return an enum representing the
 * fundamental type which was passed in, or NOFUNDAMENTALTYPE
 * @param typ the type passed into the system
 * @return enum representing kind of fundamental type
 */
RFLX_API EFUNDAMENTALTYPE FundamentalType(const Type& typ);


/**
 * Demangle will call the internal demangling routines and
 * return the demangled string of a At
 * @param ti the mangled At string
 * @return the demangled string
 */
RFLX_API std::string Demangle(const std::type_info& ti);


/**
 * StringSplit will return a vector of split strings
 * @param  splitValues returns the vector with split strings
 * @param  str the input string
 * @param  delim the delimiter on which to split
 */
RFLX_API void StringSplit(std::vector<std::string>& splitValues,
                          const std::string& str,
                          const std::string& delim = ",");


/**
 * StringSplitPair will return two values which are split
 * @param  val1 returns the first value
 * @param  val2 returns the second value
 * @param  str the string to be split
 * @param  delim the delimiter on which to split
 */
RFLX_API void StringSplitPair(std::string& val1,
                              std::string& val2,
                              const std::string& str,
                              const std::string& delim = ",");


/**
 * StringStrip will strip off Empty spaces of a string
 * @param str a reference to a string to be stripped
 */
RFLX_API void StringStrip(std::string& str);


/**
 * StringVec2String will take a vector of strings and return the
 * vector containees concatenated by commas
 * @param vec the vector to be converted
 * @return string of comma concatenated containees
 */
RFLX_API std::string StringVec2String(const std::vector<std::string>& vec);


RFLX_API std::string BuildTypeName(Type& t,
                                   unsigned int modifiers);


RFLX_API std::vector<std::string> GenTemplateArgVec(const std::string& name);

/**
 * GetTemplateComponents extract from 'Name' a template name and a vector containing its argument.
 *
 */
RFLX_API void GetTemplateComponents(const std::string& Name,
                                    std::string& templatename,
                                    std::vector<std::string>& args);

/**
 * GetBasePosition will return the position in a
 * string where the unscoped At begins
 */
RFLX_API size_t GetBasePosition(const std::string& name);


RFLX_API size_t GetFirstScopePosition(const std::string& name, size_t& start);


/**
 * Get the At part of a given At/member Name
 */
RFLX_API std::string GetScopeName(const std::string& name,
                                  bool startFromLeft = false);


/**
 * Get the BaseAt (unscoped) Name of a At/member Name
 */
RFLX_API std::string GetBaseName(const std::string& name,
                                 bool startFromLeft = false);


/**
 * IsTemplated returns true if the At (class) is templated
 * @param Name the At Name
 * @return true if At is templated
 */
RFLX_API bool IsTemplated(const char* name);


/**
 * templateArguments returns a string containing the template arguments
 * of a templated At (including outer angular brackets)
 * @param Name the Name of the templated At
 * @return template arguments of the templated At
 */
RFLX_API std::string GetTemplateArguments(const char* name);


/**
 * GetTemplateName returns the Name of the template At (without arguments)
 * @param Name the Name of the template At
 * @return template Name
 */
RFLX_API std::string GetTemplateName(const char* name);


RFLX_API std::string NormalizeName(const std::string& name);


RFLX_API std::string NormalizeName(const char* name);


/**
 * MakeVector is a utility function to create and initialize a std::vector of
 * number of items
 * @param t1 vector element
 * @return output vector
 */
template <typename T>
inline std::vector<T>
MakeVector(T t0) {
   std::vector<T> v;
   v.reserve(1);
   v.push_back(t0);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1) {
   std::vector<T> v;
   v.reserve(2);
   v.push_back(t0); v.push_back(t1);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2) {
   std::vector<T> v;
   v.reserve(3);
   v.push_back(t0); v.push_back(t1); v.push_back(t2);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3) {
   std::vector<T> v;
   v.reserve(4);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4) {
   std::vector<T> v;
   v.reserve(5);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5) {
   std::vector<T> v;
   v.reserve(6);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6) {
   std::vector<T> v;
   v.reserve(7);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5); v.push_back(t6);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7) {
   std::vector<T> v;
   v.reserve(8);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5); v.push_back(t6); v.push_back(t7);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8) {
   std::vector<T> v;
   v.reserve(9);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9) {
   std::vector<T> v;
   v.reserve(10);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10) {
   std::vector<T> v;
   v.reserve(11);
   v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
   v.push_back(t10);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11) {
   std::vector<T> v;
   v.reserve(12);
   v.push_back(t0);  v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5);  v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
   v.push_back(t10); v.push_back(t11);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12) {
   std::vector<T> v;
   v.reserve(13);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2); v.push_back(t3); v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8); v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13) {
   std::vector<T> v;
   v.reserve(14);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3); v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8); v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14) {
   std::vector<T> v;
   v.reserve(15);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15) {
   std::vector<T> v;
   v.reserve(16);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16) {
   std::vector<T> v;
   v.reserve(17);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17) {
   std::vector<T> v;
   v.reserve(18);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18) {
   std::vector<T> v;
   v.reserve(19);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19) {
   std::vector<T> v;
   v.reserve(20);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20) {
   std::vector<T> v;
   v.reserve(21);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21) {
   std::vector<T> v;
   v.reserve(22);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22) {
   std::vector<T> v;
   v.reserve(23);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23) {
   std::vector<T> v;
   v.reserve(24);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24) {
   std::vector<T> v;
   v.reserve(25);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24,
           T t25) {
   std::vector<T> v;
   v.reserve(26);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   v.push_back(t25);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24,
           T t25,
           T t26) {
   std::vector<T> v;
   v.reserve(27);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   v.push_back(t25); v.push_back(t26);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24,
           T t25,
           T t26,
           T t27) {
   std::vector<T> v;
   v.reserve(28);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   v.push_back(t25); v.push_back(t26); v.push_back(t27);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24,
           T t25,
           T t26,
           T t27,
           T t28) {
   std::vector<T> v;
   v.reserve(29);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   v.push_back(t25); v.push_back(t26); v.push_back(t27); v.push_back(t28);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24,
           T t25,
           T t26,
           T t27,
           T t28,
           T t29) {
   std::vector<T> v;
   v.reserve(30);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   v.push_back(t25); v.push_back(t26); v.push_back(t27); v.push_back(t28); v.push_back(t29);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24,
           T t25,
           T t26,
           T t27,
           T t28,
           T t29,
           T t30) {
   std::vector<T> v;
   v.reserve(31);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   v.push_back(t25); v.push_back(t26); v.push_back(t27); v.push_back(t28); v.push_back(t29);
   v.push_back(t30);
   return v;
}


template <typename T>
inline std::vector<T>
MakeVector(T t0,
           T t1,
           T t2,
           T t3,
           T t4,
           T t5,
           T t6,
           T t7,
           T t8,
           T t9,
           T t10,
           T t11,
           T t12,
           T t13,
           T t14,
           T t15,
           T t16,
           T t17,
           T t18,
           T t19,
           T t20,
           T t21,
           T t22,
           T t23,
           T t24,
           T t25,
           T t26,
           T t27,
           T t28,
           T t29,
           T t30,
           T t31) {
   std::vector<T> v;
   v.reserve(32);
   v.push_back(t0);  v.push_back(t1);  v.push_back(t2);  v.push_back(t3);  v.push_back(t4);
   v.push_back(t5);  v.push_back(t6);  v.push_back(t7), v.push_back(t8);  v.push_back(t9);
   v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
   v.push_back(t15); v.push_back(t16); v.push_back(t17); v.push_back(t18); v.push_back(t19);
   v.push_back(t20); v.push_back(t21); v.push_back(t22); v.push_back(t23); v.push_back(t24);
   v.push_back(t25); v.push_back(t26); v.push_back(t27); v.push_back(t28); v.push_back(t29);
   v.push_back(t30); v.push_back(t31);
   return v;
}


template <typename T> class CheckPointer {
public:
   static void*
   Get(const T& value) { return (void*) &value; }

};

template <typename T> class CheckPointer<T*> {
public:
   static void*
   Get(const T& value) { return (void*) value; }

};

}    // namespace Tools
} // namespace Reflex

#endif // Reflex_Tools
