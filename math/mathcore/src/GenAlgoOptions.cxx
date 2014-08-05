// @(#)root/mathcore:$Id$
// Author: L. Moneta Nov 2010
/**********************************************************************
 *                                                                    *
 * Copyright (c) 2010  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// implementation file for static methods of GenAlgoOptions
// this file contains also the pointer to the static std::map<algorithm name, options>

#include "Math/GenAlgoOptions.h"
#include <cassert>

// for toupper
#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here

namespace ROOT {
namespace Math {

typedef std::map<std::string, ROOT::Math::GenAlgoOptions > OptionsMap;

namespace GenAlgoOptUtil {

   // map with the generic options for all ROOT::Math numerical algorithm
   static OptionsMap gAlgoOptions;


   IOptions * DoFindDefault(std::string & algoname, OptionsMap & gOpts) {
      // internal function to retrieve the
      // default extra options for the given algorithm type
      // return zero if not found
      // store always name in upper case
      std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) toupper );

      OptionsMap::iterator pos = gOpts.find(algoname);
      if (pos !=  gOpts.end() ) {
         return &(pos->second);
      }
      return 0;
   }
}

   IOptions * GenAlgoOptions::FindDefault(const char * algo) {
      // find default options - return 0 if not found
      std::string algoname(algo);
      OptionsMap & gOpts = GenAlgoOptUtil::gAlgoOptions;
      return GenAlgoOptUtil::DoFindDefault(algoname, gOpts);
   }

   IOptions & GenAlgoOptions::Default(const char * algo) {
      // create default extra options for the given algorithm type
      std::string algoname(algo);
      OptionsMap & gOpts = GenAlgoOptUtil::gAlgoOptions;
      IOptions * opt = GenAlgoOptUtil::DoFindDefault(algoname, gOpts);
      if (opt == 0) {
         // create new extra options for the given type
         std::pair<OptionsMap::iterator,bool> ret = gOpts.insert( OptionsMap::value_type(algoname, ROOT::Math::GenAlgoOptions()) );
         assert(ret.second);
         opt = &((ret.first)->second);
      }
      return *opt;
   }

   void GenAlgoOptions::PrintAllDefault(std::ostream & os) {
      const OptionsMap & gOpts = GenAlgoOptUtil::gAlgoOptions;
      for (  OptionsMap::const_iterator pos = gOpts.begin();
          pos != gOpts.end(); ++pos) {
            os << "Default specific options for algorithm "  << pos->first << " : " << std::endl;
            (pos->second).Print(os);
      }
   }

   } // end namespace Math

} // end namespace ROOT

