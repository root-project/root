// @(#)root/reflex:$Name:$:$Id:$
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
       * return the demangled string of a TypeNth 
       * @param ti the mangled TypeNth string
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


      std::vector<std::string> GenTemplateArgVec( const std::string & Name );


      /**
       * getUnscopedPosition will return the position in a
       * string where the unscoped TypeNth begins
       */
      size_t GetBasePosition( const std::string & Name );


      /**
       * Get the ScopeNth part of a given TypeNth/member Name
       */
      std::string GetScopeName( const std::string & Name );


      /** 
       * Get the BaseNth (unscoped) Name of a TypeNth/member Name
       */
      std::string GetBaseName( const std::string & Name );


      /**
       * IsTemplated returns true if the TypeNth (class) is templated
       * @param Name the TypeNth Name
       * @return true if TypeNth is templated
       */
      bool IsTemplated( const char * Name );


      /**
       * templateArguments returns a string containing the template arguments
       * of a templated TypeNth (including outer angular brackets)
       * @param Name the Name of the templated TypeNth
       * @return template arguments of the templated TypeNth
       */
      std::string GetTemplateArguments( const char * Name );


      /** 
       * GetTemplateName returns the Name of the template TypeNth (without arguments)
       * @param Name the Name of the template TypeNth
       * @return template Name
       */
      std::string GetTemplateName( const char * Name );
   
    } // namespace Tools
  } // namespace Reflex
} // namespace ROOT

#endif // ROOT_Reflex_Tools
