// @(#)root/mathcore:$Id$ 
// Author: L. Moneta Fri Aug 15 2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/DistSamplerOptions.h"

#include "Math/GenAlgoOptions.h"

// case of using ROOT plug-in manager
#ifndef MATH_NO_PLUGIN_MANAGER
#include "TEnv.h"
#endif 

#include <iomanip>

namespace ROOT { 
   

namespace Math { 

   namespace Sampler { 
      static std::string gDefaultSampler = "Unuran"; // take from /etc/system.rootrc in ROOT Fitter
      static std::string gDefaultAlgorithm = "";
      static int  gDefaultPrintLevel  = 0; 
   }


void DistSamplerOptions::SetDefaultSampler(const char * type, const char * algo ) {   
   // set the default minimizer type and algorithm
   if (type) Sampler::gDefaultSampler = std::string(type); 
   if (algo) Sampler::gDefaultAlgorithm = std::string(algo);
}
void DistSamplerOptions::SetDefaultAlgorithm(const char * algo ) {   
   // set the default minimizer type and algorithm
   if (algo) Sampler::gDefaultAlgorithm = std::string(algo);
}
void DistSamplerOptions::SetDefaultPrintLevel(int level) {
   // set the default printing level 
   Sampler::gDefaultPrintLevel = level; 
}

const std::string & DistSamplerOptions::DefaultAlgorithm() { return Sampler::gDefaultAlgorithm; }
int    DistSamplerOptions::DefaultPrintLevel()       { return Sampler::gDefaultPrintLevel; }

const std::string & DistSamplerOptions::DefaultSampler() 
{ 
   // return default minimizer
   // if is "" (no default is set) read from etc/system.rootrc
   // use form /etc/ ??

   return Sampler::gDefaultSampler; 
}


DistSamplerOptions::DistSamplerOptions(IOptions * extraOpts): 
   fLevel( Sampler::gDefaultPrintLevel),
   fExtraOptions(extraOpts)
{
   // constructor using  the default options

   fSamplerType = DistSamplerOptions::DefaultSampler();

   fAlgoType =  DistSamplerOptions::DefaultAlgorithm();


   // check if extra options exists (copy them if needed)
   if (!fExtraOptions) { 
      IOptions * gopts = FindDefault( fSamplerType.c_str() );
      if (gopts) fExtraOptions = gopts->Clone();
   }
}


DistSamplerOptions::DistSamplerOptions(const DistSamplerOptions & opt) : fExtraOptions(0) {  
   // copy constructor 
   (*this) = opt; 
}

DistSamplerOptions & DistSamplerOptions::operator=(const DistSamplerOptions & opt) {  
   // assignment operator 
   if (this == &opt) return *this; // self assignment
   fLevel = opt.fLevel;
   fSamplerType = opt.fSamplerType; 
   fAlgoType = opt.fAlgoType; 

   if (fExtraOptions) delete fExtraOptions; 
   fExtraOptions = 0; 
   if (opt.fExtraOptions)  fExtraOptions =  (opt.fExtraOptions)->Clone();
   return *this;
}

DistSamplerOptions::~DistSamplerOptions() { 
   if (fExtraOptions) delete fExtraOptions; 
}

void DistSamplerOptions::SetExtraOptions(const IOptions & opt) {  
   // set extra options (clone the passed one)
   if (fExtraOptions) delete fExtraOptions; 
   fExtraOptions = opt.Clone(); 
}

void DistSamplerOptions::Print(std::ostream & os) const {
   //print all the options
   os << std::setw(25) << "DistSampler Type"        << " : " << std::setw(15) << fSamplerType << std::endl;
   os << std::setw(25) << "DistSampler Algorithm"   << " : " << std::setw(15) << fAlgoType << std::endl;
   os << std::setw(25) << "Print Level"            << " : " << std::setw(15) << fLevel << std::endl;
   
   if (ExtraOptions()) { 
      os << fSamplerType << " specific options :"  << std::endl;
      ExtraOptions()->Print(os);
   }
}

IOptions & DistSamplerOptions::Default(const char * name) { 
   // create default extra options for the given algorithm type 
   return GenAlgoOptions::Default(name);
}

IOptions * DistSamplerOptions::FindDefault(const char * name) { 
   // find extra options for the given algorithm type 
   return GenAlgoOptions::FindDefault(name);
}

void DistSamplerOptions::PrintDefault(const char * name, std::ostream & os) {
   //print default options
   DistSamplerOptions tmp;
   tmp.Print(os);
   if (!tmp.ExtraOptions() ) {
      IOptions * opt = FindDefault(name);
      os << "Specific options for "  << name << std::endl;
      if (opt) opt->Print(os);
   }
}




} // end namespace Math

} // end namespace ROOT

