// @(#)root/reflex:$Name:  $:$Id: Tools.h,v 1.2 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Tools
#define ROOT_Reflex_Tools

// Include files
#include "Reflex/Kernel.h"
#include <vector>
#include <typeinfo>
#include <string>

namespace ROOT {
  namespace Reflex {

    // forward declarations
    class Function;
    class Type;

   namespace Tools {

      /**
       * Demangle will call the internal demangling routines and
       * return the demangled string of a At 
       * @param ti the mangled At string
       * @return the demangled string
       */
      std::string Demangle( const std::type_info & ti );


      /**
       * StringSplit will return a vector of splitted strings
       * @param  splitValues returns the vector with splitted strings
       * @param  str the input string
       * @param  delim the delimiter on which to split 
       */
      void StringSplit(std::vector < std:: string > & splitValues,
                       const std::string & str, 
                       const std::string & delim = ",");


      /** 
       * StringSplitPair will return two values which are split
       * @param  val1 returns the first value
       * @param  val2 returns the second value
       * @param  str the string to be split
       * @param  delim the delimiter on which to split
       */
      void StringSplitPair(std::string & val1,
                           std::string & val2,
                           const std::string & str,
                           const std::string & delim = ",");


      /**
       * StringStrip will strip off Empty spaces of a string
       * @param str a reference to a string to be stripped
       */
      void StringStrip(std::string & str);

      std::string BuildTypeName( Type & t, 
                                 unsigned int modifiers );


      std::vector<std::string> GenTemplateArgVec( const std::string & name );


      /**
       * getUnscopedPosition will return the position in a
       * string where the unscoped At begins
       */
      size_t GetBasePosition( const std::string & name );


      /**
       * Get the At part of a given At/member Name
       */
      std::string GetScopeName( const std::string & name );


      /** 
       * Get the BaseAt (unscoped) Name of a At/member Name
       */
      std::string GetBaseName( const std::string & name );


      /**
       * IsTemplated returns true if the At (class) is templated
       * @param Name the At Name
       * @return true if At is templated
       */
      bool IsTemplated( const char * name );


      /**
       * templateArguments returns a string containing the template arguments
       * of a templated At (including outer angular brackets)
       * @param Name the Name of the templated At
       * @return template arguments of the templated At
       */
      std::string GetTemplateArguments( const char * name );


      /** 
       * GetTemplateName returns the Name of the template At (without arguments)
       * @param Name the Name of the template At
       * @return template Name
       */
      std::string GetTemplateName( const char * name );
   

      /**
       * MakeVector is a utility function to create and initialize a std::vector of
       * number of items
       * @param t1 vector element
       * @return output vector
       */
      template <typename T >
        inline std::vector<T> MakeVector(T t0) { 
        std::vector<T> v; 
        v.push_back(t0);
        return v; 
      }
  
      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1);
        return v;
      }
  
      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8, T t9 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8, T t9, T t10 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
        v.push_back(t10);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8, T t9, T t10, T t11 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
        v.push_back(t10); v.push_back(t11);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8, T t9, T t10, T t11, T t12 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
        v.push_back(t10); v.push_back(t11); v.push_back(t12);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8, T t9, T t10, T t11, T t12, T t13 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
        v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8, T t9, T t10, T t11, T t12, T t13, T t14 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
        v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
        return v;
      }

      template <typename T >
        inline std::vector<T> MakeVector(T t0, T t1, T t2, T t3, T t4, T t5, T t6, T t7,
                                         T t8, T t9, T t10, T t11, T t12, T t13, T t14, T t15 ) {
        std::vector<T> v;
        v.push_back(t0); v.push_back(t1); v.push_back(t2); v.push_back(t3); v.push_back(t4);
        v.push_back(t5); v.push_back(t6); v.push_back(t7), v.push_back(t8); v.push_back(t9);
        v.push_back(t10); v.push_back(t11); v.push_back(t12); v.push_back(t13); v.push_back(t14);
        v.push_back(t15);
        return v;
      }

      template < typename T > struct CheckPointer { 
        static void * Get(const T & value) { return (void*) & value; } 
      };

      template < typename T > struct CheckPointer < T * > { 
        static void * Get(const T & value) { return (void*) value; } 
      };

    } // namespace Tools
  } // namespace Reflex
} // namespace ROOT

#endif // ROOT_Reflex_Tools
