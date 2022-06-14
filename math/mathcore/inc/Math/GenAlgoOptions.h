// @(#)root/mathcore:$Id$
// Author: L. Moneta Nov 2010

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2010  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Math_GenAlgoOptions
#define ROOT_Math_GenAlgoOptions


#include "Math/IOptions.h"

#include <map>
#include <iomanip>
#include <string>
#include <vector>

namespace ROOT {
      namespace Math {

//_______________________________________________________________________________
/**
    class implementing generic options for a numerical algorithm
    Just store the options in a map of string-value pairs

    @ingroup NumAlgo
*/
class GenAlgoOptions : public IOptions {

public:

   GenAlgoOptions() /* : fExtraOptions(0) */  {}

   ~GenAlgoOptions() override {}// { if (fExtraOptions) delete fExtraOptions; }

   // use default copy constructor and assignment operator

   /** generic  methods for  retrieving options */


   // methods implementing the  IOptions interface

   IOptions * Clone() const override {
      return new GenAlgoOptions(*this);
   }

   // t.b.d need probably to implement in a .cxx file for CINT


   bool GetRealValue(const char * name, double & val) const override {
      const double * pval = FindValue(name, fRealOpts);
      if (!pval) return false;
      val = *pval;
      return true;
   }

   bool GetIntValue(const char * name, int & val) const override {
      const int * pval = FindValue(name, fIntOpts);
      if (!pval) return false;
      val = *pval;
      return true;
   }

   bool GetNamedValue(const char * name, std::string & val) const override {
      const std::string * pval = FindValue(name, fNamOpts);
      if (!pval) return false;
      val = *pval;
      return true;
   }

   /// method wich need to be re-implemented by the derived classes
   void SetRealValue(const char * name, double val) override  {
      InsertValue(name, fRealOpts, val);
   }

   void SetIntValue(const char * name , int val) override {
      InsertValue(name, fIntOpts, val);
   }

   void SetNamedValue(const char * name, const char * val) override {
      InsertValue(name, fNamOpts, std::string(val));
   }

   std::vector<std::string>  GetAllNamedKeys() {
      std::vector<std::string> names;
      names.reserve(fNamOpts.size());
      // start by named options
      for (auto const & e : fNamOpts)
         names.push_back(e.first);
      return names;
   }
   std::vector<std::string>  GetAllRealKeys() {
      std::vector<std::string> names;
      names.reserve(fRealOpts.size());
      // start by named options
      for (auto const & e : fRealOpts)
         names.push_back(e.first);
      return names;
   }
   std::vector<std::string>  GetAllIntKeys() {
      std::vector<std::string> names;
      names.reserve(fIntOpts.size());
      // start by named options
      for (auto const & e : fIntOpts)
         names.push_back(e.first);
      return names;
   }

   /// print options
   void Print(std::ostream & os = std::cout ) const override {
      Print(fNamOpts,os);
      Print(fIntOpts,os);
      Print(fRealOpts,os);
   }


   // static methods to retrieve the default options

   // find the option given a name
   // return 0 if the option is not found
   static IOptions * FindDefault(const char * algoname);

   // retrieve options given the name
   // if option is not found create a new GenAlgoOption for the given name
   static IOptions & Default(const char * algoname);

   /// print all the default options
   static void PrintAllDefault(std::ostream & os = std::cout);


protected:



private:

   template<class M>
   static const typename M::mapped_type * FindValue(const std::string &  name, const M & opts) {
      typename M::const_iterator pos;
      pos = opts.find(name);
      if (pos == opts.end()) {
         return 0;
      }
      return  &((*pos).second);
   }

   template<class M>
   static void InsertValue(const std::string &name, M & opts, const typename M::mapped_type & value) {
      typename M::iterator pos;
      pos = opts.find(name);
      if (pos != opts.end()) {
         pos->second = value;
      }
      else {
         opts.insert(typename M::value_type(name, value) );
      }
   }

   template<class M>
   static void Print( const M & opts, std::ostream & os) {
      //const std::ios_base::fmtflags prevFmt = os.flags();
      for (typename M::const_iterator pos = opts.begin(); pos != opts.end(); ++pos)
         os << std::setw(25) << pos->first << " : " << std::setw(15) << pos->second << std::endl;
   }

   std::map<std::string, int>         fIntOpts;    ///< map of the integer options
   std::map<std::string, double>      fRealOpts;   ///< map of the real options
   std::map<std::string, std::string> fNamOpts;    ///< map of the named options

};



   } // end namespace Math

} // end namespace ROOT

#endif
