// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Aug 15 2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Math_IOptions
#define ROOT_Math_IOptions

#include <iostream>
#include <string>

namespace ROOT {


   namespace Math {

//_______________________________________________________________________________
/**
    Generic interface for defining configuration options of a numerical algorithm

    @ingroup NumAlgo
*/
class IOptions {

public:

   IOptions() /* : fExtraOptions(0) */  {}

   virtual ~IOptions() {}// { if (fExtraOptions) delete fExtraOptions; }

   // copy the options
   virtual IOptions * Clone() const = 0;

   /** generic  methods for  retrieving options */

   /// set option value
   void SetValue(const char * name, double val) { SetRealValue(name,val);}
   void SetValue(const char * name, int val) { SetIntValue(name,val);}
   void SetValue(const char * name, const char * val) { SetNamedValue(name,val);}


   double  RValue(const char * name) const;
   int   IValue(const char * name) const;
   std::string  NamedValue(const char * name) const;


   // generic method to retrieve  a type
   template <typename T>
   bool GetValue(const char * name, T & t) const {
      bool ret = DoGetValue(name, t);
      //if (!ret )  MATH_ERROR_MSG("IOptions::GetValue","option is not existing - returns 0");
      return ret;
   }


   // methods to be re-implemented in the derived classes

   virtual bool GetRealValue(const char *, double &) const { return false; }
   virtual bool GetIntValue(const char *, int &) const { return false; }
   virtual bool GetNamedValue(const char *, std::string &) const { return false; }

   virtual void SetRealValue(const char * , double );
   virtual void SetIntValue(const char * , int );
   virtual void SetNamedValue(const char * , const char * );

   virtual void Print(std::ostream & = std::cout ) const;

private:

   bool DoGetValue(const char *name, double &val) const { return GetRealValue(name,val); }

   bool DoGetValue(const char *name, int &val) const { return GetIntValue(name,val); }

   bool DoGetValue(const char *name, std::string &val) const { return GetNamedValue(name,val); }


};


   } // end namespace Math

} // end namespace ROOT

#endif
