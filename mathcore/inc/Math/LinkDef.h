// @(#)root/mathcore:$Id$


#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Math;


#pragma link C++ typedef ROOT::Math::IGenFunction;
#pragma link C++ typedef ROOT::Math::IMultiGenFunction;
#pragma link C++ typedef ROOT::Math::IGradFunction;
#pragma link C++ typedef ROOT::Math::IMultiGradFunction;



#pragma link C++ class ROOT::Math::IBaseFunctionOneDim+;
#pragma link C++ class ROOT::Math::IGradientOneDim+;
#pragma link C++ class ROOT::Math::IGradientFunctionOneDim+;
#pragma link C++ class ROOT::Math::IBaseParam+;

#pragma link C++ class ROOT::Math::IParametricFunctionOneDim+;
#pragma link C++ class ROOT::Math::IParametricGradFunctionOneDim+;

#pragma link C++ class ROOT::Math::IBaseFunctionMultiDim+;
#pragma link C++ class ROOT::Math::IGradientMultiDim+;
#pragma link C++ class ROOT::Math::IGradientFunctionMultiDim+;
#pragma link C++ class ROOT::Math::IParametricFunctionMultiDim+;
#pragma link C++ class ROOT::Math::IParametricGradFunctionMultiDim+;

#pragma link C++ class ROOT::Math::Functor-;
#pragma link C++ class ROOT::Math::GradFunctor-;
#pragma link C++ class ROOT::Math::Functor1D-;
#pragma link C++ class ROOT::Math::GradFunctor1D-;

#pragma link C++ class ROOT::Math::Minimizer+;
#pragma link C++ class ROOT::Math::IntegratorOneDim+;
#pragma link C++ class ROOT::Math::IntegratorMultiDim+;
#pragma link C++ class ROOT::Math::VirtualIntegrator+;
#pragma link C++ class ROOT::Math::VirtualIntegratorOneDim+;
#pragma link C++ class ROOT::Math::VirtualIntegratorMultiDim+;
#pragma link C++ class ROOT::Math::AdaptiveIntegratorMultiDim+;
#pragma link C++ typedef ROOT::Math::Integrator;

#pragma link C++ class ROOT::Math::BasicFitMethodFunction<ROOT::Math::IBaseFunctionMultiDim>+;
#pragma link C++ class ROOT::Math::BasicFitMethodFunction<ROOT::Math::IGradientFunctionMultiDim>+;

#pragma link C++ class ROOT::Math::Factory+;


#include "LinkDef_Func.h" 
#include "LinkDef_GenVector.h" 

#endif
