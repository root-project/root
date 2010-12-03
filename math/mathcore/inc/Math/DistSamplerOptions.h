// @(#)root/mathcore:$Id$ 
// Author: L. Moneta Fri Aug 15 2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Math_DistSamplerOptions
#define ROOT_Math_DistSamplerOptions

#include <string>

#include <iostream>

namespace ROOT { 
   

namespace Math { 


class IOptions;      

//_______________________________________________________________________________
/** 
    DistSampler options 

    @ingroup NumAlgo
*/
class DistSamplerOptions {

public:

   // static methods for setting and retrieving the default options 

   static void SetDefaultSampler(const char * type, const char * algo = 0);
   static void SetDefaultAlgorithm(const char * algo );
   static void SetDefaultPrintLevel(int level);

   static const std::string & DefaultSampler();
   static const std::string & DefaultAlgorithm(); 
   static int DefaultPrintLevel(); 

   /// retrieve extra options - if not existing create a IOptions 
   static ROOT::Math::IOptions & Default(const char * name);

   // find extra options - return 0 if not existing 
   static ROOT::Math::IOptions * FindDefault(const char * name);

   /// print all the default options for the name given
   static void PrintDefault(const char * name = 0, std::ostream & os = std::cout); 

public:

   // constructor using the default options 
   // pass optionally a pointer to the additional options
   // otehrwise look if they exist for this default minimizer
   // and in that case they are copied in the constructed instance
   DistSamplerOptions(IOptions * extraOpts = 0);

   // destructor  
   ~DistSamplerOptions();

   // copy constructor 
   DistSamplerOptions(const DistSamplerOptions & opt);

   /// assignment operators 
   DistSamplerOptions & operator=(const DistSamplerOptions & opt);

   /** non-static methods for  retrivieng options */

   /// set print level
   int PrintLevel() const { return fLevel; }

   /// return extra options (NULL pointer if they are not present)
   IOptions * ExtraOptions() const { return fExtraOptions; }

   /// type of minimizer
   const std::string & Sampler() const { return fSamplerType; }

   /// type of algorithm
   const std::string & Algorithm() const { return fAlgoType; }

   /// print all the options 
   void Print(std::ostream & os = std::cout) const; 

   /** non-static methods for setting options */

   /// set print level
   void SetPrintLevel(int level) { fLevel = level; }

   /// set minimizer type
   void SetSampler(const char * type) { fSamplerType = type; }

   /// set minimizer algorithm
   void SetAlgorithm(const char *type) { fAlgoType = type; }

   /// set extra options (in this case pointer is cloned)
   void  SetExtraOptions(const IOptions & opt); 


private:

   int fLevel;               // debug print level 
   std::string fSamplerType;   // DistSampler type (Minuit, Minuit2, etc..
   std::string fAlgoType;    // DistSampler algorithmic specification (Migrad, Minimize, ...)

   // extra options
   ROOT::Math::IOptions *   fExtraOptions;  // extra options 
     
};

   } // end namespace Math

} // end namespace ROOT

#endif
