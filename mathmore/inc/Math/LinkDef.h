// @(#)root/mathmore:$Name:  $:$Id: LinkDef.h,v 1.3 2005/09/19 18:14:01 brun Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 



#include "LinkDef_SpecFunc.h" 
#include "LinkDef_StatFunc.h" 
#include "LinkDef_Func.h" 
#include "TF1.h"

#include "LinkDef_RootFinding.h"

#ifdef __CINT__

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Math;

#ifndef _WIN32
#pragma link C++ class ROOT::Math::ParamFunction+;
#pragma link C++ class ROOT::Math::Polynomial+;
#pragma link C++ class ROOT::Math::WrappedFunction<ROOT::Math::Polynomial>+;
#endif

#pragma link C++ class ROOT::Math::Chebyshev+;


#pragma link C++ class ROOT::Math::Derivator+;
//#pragma link C++ class ROOT::Math::WrappedFunction<double (&)(double)>+;

#pragma extra_include "TF1.h";
//#pragma link C++ class ROOT::Math::WrappedFunction< ::TF1 &>+;
// maybe idem with IGenFunction... to be seen after checking with Philippe


#pragma link C++ namespace ROOT::Math::Integration;
#pragma link C++ class ROOT::Math::Integrator+;


#pragma link C++ class ROOT::Math::Interpolator+;


#pragma link C++ class ROOT::Math::GSLRandomEngine+;

#pragma link C++ class ROOT::Math::GSLRngMT+;
#pragma link C++ class ROOT::Math::GSLRngTaus+;
#pragma link C++ class ROOT::Math::GSLRngRanLux+;
#pragma link C++ class ROOT::Math::GSLRngGFSR4+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngMT>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngTaus>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngRanLux>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngGFSR4>+;



#endif //__CINT__
