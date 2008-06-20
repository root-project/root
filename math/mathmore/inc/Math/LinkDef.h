// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005 



#ifdef __CINT__

#pragma extra_include "Math/IFunctionfwd.h";
#pragma extra_include "Math/IFunction.h";


#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Math;


#include "LinkDef_Func.h" 
#include "LinkDef_RootFinding.h"


#pragma link C++ class ROOT::Math::ParamFunction<ROOT::Math::IParametricGradFunctionOneDim>+;

#ifndef _WIN32  
// virtual inheritance gives problem when making dictionary on Windows 
#pragma link C++ class ROOT::Math::Polynomial+;
#else 
#pragma link C++ class ROOT::Math::Polynomial-;
#endif

#pragma link C++ class ROOT::Math::Chebyshev+;
#pragma link C++ class ROOT::Math::Derivator+;


//#pragma extra_include "TF1.h";


#pragma link C++ namespace ROOT::Math::Integration;
#pragma link C++ class ROOT::Math::GSLIntegrator+;

#pragma link C++ namespace ROOT::Math::Minim1D;
#pragma link C++ class ROOT::Math::GSLMinimizer1D+;

#pragma link C++ class ROOT::Math::Interpolator+;

// random  numbers
#pragma link C++ class ROOT::Math::GSLRandomEngine+;

#pragma link C++ class ROOT::Math::GSLRngMT+;
#pragma link C++ class ROOT::Math::GSLRngTaus+;
#pragma link C++ class ROOT::Math::GSLRngRanLux+;
#pragma link C++ class ROOT::Math::GSLRngGFSR4+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngMT>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngTaus>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngRanLux>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngGFSR4>+;

#pragma link C++ class ROOT::Math::KelvinFunctions+;

#pragma link C++ class ROOT::Math::GSLMinimizer+;
#pragma link C++ class ROOT::Math::GSLSimAnMinimizer+;
#pragma link C++ class ROOT::Math::GSLSimAnFunc+;
#pragma link C++ class ROOT::Math::GSLSimAnParams+;
#pragma link C++ class ROOT::Math::GSLSimAnnealing+;

//#pragma link C++ class std::vector<ROOT::Math::IGradientFunctionMultiDim *>+;
#pragma link C++ class ROOT::Math::GSLNLSMinimizer-;
#pragma link C++ class ROOT::Math::LSResidualFunc-;

// #ifndef _WIN32  // exclude for same problem of virtual inheritance
// #pragma link C++ class ROOT::Math::LSResidualFunc+;
// #else
// #pragma link C++ class ROOT::Math::LSResidualFunc-;
// #endif

#pragma link C++ class ROOT::Math::GSLMCIntegrator+;



#endif //__CINT__
