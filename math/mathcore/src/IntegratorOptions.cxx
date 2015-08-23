// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Aug 15 2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#include "Math/IntegratorOptions.h"
#include "Math/Integrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/GenAlgoOptions.h"

#include "RConfigure.h"

#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here

#include <map>

namespace ROOT {

namespace Math {

   // eventually could take values from /etc/system.rootrc

namespace IntegOneDim {

#ifdef R__HAS_MATHMORE
   static int gDefaultIntegrator = IntegrationOneDim::kADAPTIVESINGULAR;
#else
   static int gDefaultIntegrator = IntegrationOneDim::kGAUSS;
#endif
   static double gDefaultAbsTolerance = 1.E-09;
   static double gDefaultRelTolerance = 1.E-09;
   static unsigned int gDefaultWKSize = 1000;
   static unsigned int gDefaultNPointsLegendre = 10;
   static unsigned int gDefaultNPointsGSLAdaptive = 3;  // corresponds to 31 points
   static unsigned int gDefaultNPoints = gDefaultNPointsGSLAdaptive;


}

namespace IntegMultiDim {

   static int gDefaultIntegrator = IntegrationMultiDim::kADAPTIVE;
   // by default do not use absolute tolerance in AdaptiveIntegration multidim.
   // If an absolute tolerance is given integration of shar peaks often failed
   static double gDefaultAbsTolerance = 0.0;  
   static double gDefaultRelTolerance = 1.E-09;
   static unsigned int gDefaultWKSize = 100000;
   static unsigned int gDefaultNCalls = 100000;


}


// some utility functions

namespace IntegOptionsUtil {


   // traits for the specific methods 1D - ND
   template<class OptionType>
   struct OptionTrait {
      static int N() { return 0; }
      static int N(const OptionType & ) { return 0; }
      static const char * DescriptionOfN() {return 0; }
   };
   template<>
   struct OptionTrait<IntegratorOneDimOptions> {
      typedef IntegratorOneDimOptions OptType;
      static int N() { return OptType::DefaultNPoints(); }
      static int N(const OptType & opt) { return opt.NPoints(); }
      static const char * DescriptionOfN() {return  "Rule (Npoints)";}
   };
   template<>
   struct OptionTrait<IntegratorMultiDimOptions> {
      typedef IntegratorMultiDimOptions OptType;
      static int N() { return OptType::DefaultNCalls(); }
      static int N(const OptType & opt) { return opt.NCalls(); }
      static const char * DescriptionOfN() {return "(max) function calls";}
   };


   //print  option values (not the default ones)
   template <class OptionType>
   void Print(std::ostream & os,const OptionType & opt) {
      //print all the options
      os << std::setw(25) << "Integrator Type"        << " : " << std::setw(15) << opt.Integrator() << std::endl;
      os << std::setw(25) << "Absolute tolerance"     << " : " << std::setw(15) << opt.AbsTolerance() << std::endl;
      os << std::setw(25) << "Relative tolerance"     << " : " << std::setw(15) << opt.RelTolerance() << std::endl;
      os << std::setw(25) << "Workspace size"         << " : " << std::setw(15) << opt.WKSize() << std::endl;
      typedef  OptionTrait<OptionType> OPT;
      os << std::setw(25) << OPT::DescriptionOfN()    << " : " << std::setw(15) << OPT::N(opt) << std::endl;
      if (opt.ExtraOptions()) {
         os << opt.Integrator() << " specific options :"  << std::endl;
         opt.ExtraOptions()->Print(os);
      }
   }


   /// print default  options
   template <class OptionType>
   void PrintDefault(const char * name, std::ostream & os) {
      //print default options
      std::string integName = (name != 0) ? name : OptionType::DefaultIntegrator();
      os << "Default options for numerical integrator "  << integName << " : " << std::endl;
      os << std::setw(25) << "Absolute tolerance"     << " : " << std::setw(15) << OptionType::DefaultAbsTolerance() << std::endl;
      os << std::setw(25) << "Relative tolerance"     << " : " <<std::setw(15) << OptionType::DefaultRelTolerance() << std::endl;
      os << std::setw(25) << "Workspace size"         << " : " << std::setw(15) << OptionType::DefaultWKSize() << std::endl;
      typedef  OptionTrait<OptionType> OPT;
      os << std::setw(25) <<  OPT::DescriptionOfN()   << " : " << std::setw(15) << OPT::N() << std::endl;
      IOptions * opts = GenAlgoOptions::FindDefault(integName.c_str());
      if (opts) opts->Print(os);
   }

}


/// constructor (protected) to avoid user creating this class
BaseIntegratorOptions::BaseIntegratorOptions() :
   fIntegType(-1),
   fWKSize(0), fNCalls(0),
   fAbsTolerance(0), fRelTolerance(0),
   fExtraOptions(0)
{}

BaseIntegratorOptions::BaseIntegratorOptions(const BaseIntegratorOptions & opt) : fExtraOptions(0) {
   // copy constructor
   (*this) = opt;
}

BaseIntegratorOptions & BaseIntegratorOptions::operator=(const BaseIntegratorOptions & opt) {
   // assignment operator
   if (this == &opt) return *this; // self assignment
   fWKSize = opt.fWKSize;
   fNCalls = opt.fNCalls;
   fAbsTolerance = opt.fAbsTolerance;
   fRelTolerance = opt.fRelTolerance;
   fIntegType = opt.fIntegType;

//    std::cout << " copy options for " << fIntegName << std::endl;
//    std::cout << fExtraOptions << std::endl;
//    if (fExtraOptions) fExtraOptions->Print(std::cout);

//    std::cout << opt.fExtraOptions << std::endl;
//    if (opt.fExtraOptions) (opt.fExtraOptions)->Print(std::cout);



   ClearExtra();
   if (opt.fExtraOptions)  fExtraOptions =  (opt.fExtraOptions)->Clone();
   return *this;
}


void BaseIntegratorOptions::ClearExtra() {
   // delete extra options
   if (fExtraOptions) delete fExtraOptions;
   fExtraOptions = 0;
}

void BaseIntegratorOptions::SetExtraOptions(const IOptions & opt) {
   // delete extra options
   ClearExtra();
   fExtraOptions = opt.Clone();
}



// one dim specific  methods

// implementation of non-static methods

IntegratorOneDimOptions::IntegratorOneDimOptions(IOptions * opts):
   BaseIntegratorOptions()
{
   fWKSize       = IntegOneDim::gDefaultWKSize;
   fNCalls       = IntegOneDim::gDefaultNPoints;
   fAbsTolerance = IntegOneDim::gDefaultAbsTolerance;
   fRelTolerance = IntegOneDim::gDefaultRelTolerance;
   fIntegType    = IntegOneDim::gDefaultIntegrator;

   fExtraOptions = opts; // N.B. ownership of pointer is given to the class !

   // check  the default options if opts = 0
   if (!fExtraOptions) {
      std::string igname = DefaultIntegrator();
      IOptions * gopts = FindDefault( igname.c_str() );
      if (gopts) fExtraOptions = gopts->Clone();
   }
}

void IntegratorOneDimOptions::SetIntegrator(const char * algo ) {
   // set the default 1D integrator
   if (!algo) return;
   fIntegType = (int) IntegratorOneDim::GetType(algo);
}

std::string  IntegratorOneDimOptions::Integrator() const {
   return  IntegratorOneDim::GetName((IntegratorOneDim::Type) fIntegType);
}

void IntegratorOneDimOptions::Print(std::ostream & os) const {
   //print all the options
   IntegOptionsUtil::Print(os, *this);
}

// one dim integrator options:  implementation for static methods

/// print default  options
void IntegratorOneDimOptions::PrintDefault(const char * name, std::ostream & os) {
   //print default options
   IntegOptionsUtil::PrintDefault<IntegratorOneDimOptions>(name,os);
}



void IntegratorOneDimOptions::SetDefaultIntegrator(const char * algo ) {
   // set the default 1D integrator
   if (!algo) return;
   IntegrationOneDim::Type type =  IntegratorOneDim::GetType(algo);
   if (type == IntegrationOneDim::kDEFAULT) return;  // this is possible only when invalid name was specified
   IntegOneDim::gDefaultIntegrator = (int) type;
   if (IntegOneDim::gDefaultIntegrator == IntegrationOneDim::kLEGENDRE)
      IntegOneDim::gDefaultNPoints = IntegOneDim::gDefaultNPointsLegendre;
   if (IntegOneDim::gDefaultIntegrator == IntegrationOneDim::kADAPTIVE)
      IntegOneDim::gDefaultNPoints = IntegOneDim::gDefaultNPointsGSLAdaptive;
}


std::string  IntegratorOneDimOptions::DefaultIntegrator() {
   // return default integrator name
   return  IntegratorOneDim::GetName((IntegratorOneDim::Type) IntegOneDim::gDefaultIntegrator);
}

IntegratorOneDim::Type  IntegratorOneDimOptions::DefaultIntegratorType() {
   // return default integrator type (enum)
   return  (IntegratorOneDim::Type) IntegOneDim::gDefaultIntegrator;
}


void IntegratorOneDimOptions::SetDefaultAbsTolerance(double tol) {
   // set the default tolerance
   IntegOneDim::gDefaultAbsTolerance = tol;
}
void IntegratorOneDimOptions::SetDefaultRelTolerance(double tol) {
   // set the default tolerance
   IntegOneDim::gDefaultRelTolerance = tol;
}

void IntegratorOneDimOptions::SetDefaultWKSize(unsigned int size) {
   // set the default workspace size
   IntegOneDim::gDefaultWKSize = size;
}
void IntegratorOneDimOptions::SetDefaultNPoints(unsigned int n) {
   // set the default number of points for the integration rule
   IntegOneDim::gDefaultNPoints = n;
}


double IntegratorOneDimOptions::DefaultAbsTolerance()        { return IntegOneDim::gDefaultAbsTolerance; }
double IntegratorOneDimOptions::DefaultRelTolerance()        { return IntegOneDim::gDefaultRelTolerance; }
unsigned int IntegratorOneDimOptions::DefaultWKSize()        { return IntegOneDim::gDefaultWKSize; }
unsigned int IntegratorOneDimOptions::DefaultNPoints()        { return IntegOneDim::gDefaultNPoints; }


IOptions & IntegratorOneDimOptions::Default(const char * algo) {
   // create default extra options for the given algorithm type
   return GenAlgoOptions::Default(algo);
}

IOptions * IntegratorOneDimOptions::FindDefault(const char * algo) {
   // find extra options for the given algorithm type
   return GenAlgoOptions::FindDefault(algo);
}

//////////////////////////////////////////////////////
//Multi-dim integration options implementation
/////////////////////////////////////////////////////////

IntegratorMultiDimOptions::IntegratorMultiDimOptions(IOptions * opts):
   BaseIntegratorOptions()
{
   fWKSize       = IntegMultiDim::gDefaultWKSize;
   fNCalls       = IntegMultiDim::gDefaultNCalls;
   fAbsTolerance = IntegMultiDim::gDefaultAbsTolerance;
   fRelTolerance = IntegMultiDim::gDefaultRelTolerance;
   fIntegType    = IntegMultiDim::gDefaultIntegrator;

   fExtraOptions = opts; // N.B. ownership of pointer is given to the class !

   // check  the default options if opts = 0
   if (!fExtraOptions) {
      IOptions * gopts = FindDefault( DefaultIntegrator().c_str() );
      if (gopts) fExtraOptions = gopts->Clone();
   }
}

void IntegratorMultiDimOptions::SetIntegrator(const char * algo ) {
   // set the default integrator
   if (!algo) return;
   fIntegType = (int) IntegratorMultiDim::GetType(algo);
}

std::string IntegratorMultiDimOptions::Integrator() const {
   return  IntegratorMultiDim::GetName((IntegratorMultiDim::Type) fIntegType);
}

void IntegratorMultiDimOptions::Print(std::ostream & os) const {
   //print all the options
   IntegOptionsUtil::Print(os, *this);
}

// multi dim integrator options:  implementation for static methods

/// print default  options
void IntegratorMultiDimOptions::PrintDefault(const char * name, std::ostream & os) {
   //print default options
   IntegOptionsUtil::PrintDefault<IntegratorMultiDimOptions>(name,os);
}


void IntegratorMultiDimOptions::SetDefaultIntegrator(const char * algo ) {
   // set the default integrator
   if (!algo) return;
   // check if type is correct
   IntegrationMultiDim::Type type =  IntegratorMultiDim::GetType(algo);
   if (type == IntegrationMultiDim::kDEFAULT) return;  // this is possible only when invalid name was specified
   IntegMultiDim::gDefaultIntegrator = (int) type;
}


std::string  IntegratorMultiDimOptions::DefaultIntegrator() {
   // return default integrator name
   return  IntegratorMultiDim::GetName((IntegratorMultiDim::Type) IntegMultiDim::gDefaultIntegrator);
}

IntegratorMultiDim::Type  IntegratorMultiDimOptions::DefaultIntegratorType() {
   // return default integrator type (enum)
   return  (IntegratorMultiDim::Type) IntegMultiDim::gDefaultIntegrator;
}


void IntegratorMultiDimOptions::SetDefaultAbsTolerance(double tol) {
   // set the default tolerance
   IntegMultiDim::gDefaultAbsTolerance = tol;
}

void IntegratorMultiDimOptions::SetDefaultRelTolerance(double tol) {
   // set the default tolerance
   IntegMultiDim::gDefaultRelTolerance = tol;
}

void IntegratorMultiDimOptions::SetDefaultWKSize(unsigned int size) {
   // set the default workspace size
   IntegMultiDim::gDefaultWKSize = size;
}
void IntegratorMultiDimOptions::SetDefaultNCalls(unsigned int ncall) {
   // set the default (max) function calls
   IntegMultiDim::gDefaultNCalls = ncall;
}


double IntegratorMultiDimOptions::DefaultAbsTolerance()        { return IntegMultiDim::gDefaultAbsTolerance; }
double IntegratorMultiDimOptions::DefaultRelTolerance()        { return IntegMultiDim::gDefaultRelTolerance; }
unsigned int IntegratorMultiDimOptions::DefaultWKSize()        { return IntegMultiDim::gDefaultWKSize; }
unsigned int IntegratorMultiDimOptions::DefaultNCalls()        { return IntegMultiDim::gDefaultNCalls; }


IOptions & IntegratorMultiDimOptions::Default(const char * algo) {
   // create default extra options for the given algorithm type
   return GenAlgoOptions::Default(algo);
}

IOptions * IntegratorMultiDimOptions::FindDefault(const char * algo) {
   // create default extra options for the given algorithm type
   return GenAlgoOptions::FindDefault(algo);
}


} // end namespace Math

} // end namespace ROOT

