// @(#)root/mathmore:$Name:  $:$Id: LinkDef.h,v 1.1 2005/09/18 17:33:47 brun Exp $
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


//#pragma link C++ class ROOT::Math::ParamFunction+;
#pragma link C++ class ROOT::Math::Polynomial+;
#pragma link C++ class ROOT::Math::Chebyshev+;


#pragma link C++ class ROOT::Math::Derivator+;
#pragma link C++ class ROOT::Math::WrappedFunction<ROOT::Math::Polynomial>+;
//#pragma link C++ class ROOT::Math::WrappedFunction<double (&)(double)>+;

#pragma extra_include "TF1.h";
//#pragma link C++ class ROOT::Math::WrappedFunction< ::TF1 &>+;
// maybe idem with IGenFunction... to be seen after checking with Philippe


#pragma link C++ namespace ROOT::Math::Integration;
#pragma link C++ class ROOT::Math::Integrator+;


#pragma link C++ class ROOT::Math::Interpolator+;



#endif //__CINT__
