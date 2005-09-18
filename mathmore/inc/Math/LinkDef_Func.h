// @(#)root/mathmore:$Name:  $:$Id: LinkDef_Func.h,v 1.1 2005/09/08 07:14:56 brun Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 

#ifdef __CINT__

/* #pragma link C++ nestedclasses; */
/* #pragma link C++ nestedtypedef; */

/* #pragma link C++ namespace ROOT; */
/* #pragma link C++ namespace ROOT::Math; */
//#pragma link C++ namespace ROOT::Math::Roots;
/* #pragma link C++ namespace ROOT::Math::Integration; */

/* #pragma link C++ function ROOT::Math::chisquared_prob( double , double); */
/* #pragma link C++ function ROOT::Math::chisquared_quant( double , double); */
/* #pragma link C++ function ROOT::Math::fdistribution_prob( double , double, double); */
/* #pragma link C++ function ROOT::Math::fdistribution_quant( double , double, double); */
/* #pragma link C++ function ROOT::Math::gamma_prob( double , double, double); */
/* #pragma link C++ function ROOT::Math::gamma_quant( double , double, double); */
/* #pragma link C++ function ROOT::Math::tdistribution_prob( double , double); */
/* #pragma link C++ function ROOT::Math::tdistribution_quant( double , double); */


/* #pragma link C++ class ROOT::Math::IGenFunction+; */
/* #pragma link C++ class ROOT::Math::IParamFunction+; */
/* #pragma link C++ class ROOT::Math::ParamFunction+; */
/* #pragma link C++ class ROOT::Math::Polynomial+; */
/* #pragma link C++ class ROOT::Math::Derivator+; */
/* #pragma link C++ class ROOT::Math::GSLDerivator+; */
/* #pragma link C++ class ROOT::Math::IIntegrator+; */
/* #pragma link C++ class ROOT::Math::Integrator+; */
/* #pragma link C++ class ROOT::Math::Interpolator+; */
/* #pragma link C++ class ROOT::Math::GSLIntegrator+; */
//#pragma link C++ class ROOT::Math::GSLIntegrationWorkspace+;
/* #pragma link C++ class ROOT::Math::GSLRootFinder+; */
/* #pragma link C++ class ROOT::Math::GSLRootFinderDeriv+; */
/* #pragma link C++ class ROOT::Math::Roots::Bisection+; */
/* #pragma link C++ class ROOT::Math::Roots::Newton+; */
/* #pragma link C++ class ROOT::Math::Roots::Brent+; */
/* #pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::Bisection>+; */
/* #pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::FalsePos>+; */


//#pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::>+;
//#pragma link C++ class ROOT::Math::RootFinder+;

//#pragma link C++ enum ROOT::Math::Integration::Type;
//#pragma link C++ enum ROOT::Math::Integration::GKRule;

//#pragma link C++ function  ROOT::Math::GSLDerivator::Eval(const ROOT::Math::Polynomial&,double,double);
/* #pragma link C++ function ROOT::Math::RootFinder<ROOT::Math::Roots::Bisection>::SetFunction(const ROOT::Math::Polynomial&,double,double); */
/* #pragma link C++ function ROOT::Math::RootFinder<ROOT::Math::Roots::FalsePos>::SetFunction(const ROOT::Math::Polynomial&,double,double); */
/* #pragma link C++ function ROOT::Math::Roots::Brent::SetFunction(double (*)(double, void*), void*, double, double); */

#endif
