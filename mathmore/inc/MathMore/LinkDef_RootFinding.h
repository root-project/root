// @(#)root/mathmore:$Name:  $:$Id: LinkDef_RootFinding.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 



#pragma link C++ namespace ROOT::Math::Roots;

#pragma link C++ class ROOT::Math::GSLRootFinder+;
#pragma link C++ class ROOT::Math::GSLRootFinderDeriv+;

#pragma link C++ class ROOT::Math::Roots::Bisection+;
#pragma link C++ class ROOT::Math::Roots::Brent+;
#pragma link C++ class ROOT::Math::Roots::FalsePos+;
#pragma link C++ class ROOT::Math::Roots::Newton+;
#pragma link C++ class ROOT::Math::Roots::Secant+;
#pragma link C++ class ROOT::Math::Roots::Steffenson+;

#pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::Bisection>+;
#pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::Brent>+;
#pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::FalsePos>+;
#pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::Newton>+;
#pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::Secant>+;
#pragma link C++ class ROOT::Math::RootFinder<ROOT::Math::Roots::Steffenson>+;






/* #pragma link C++ function ROOT::Math::RootFinder<ROOT::Math::Roots::Bisection>::SetFunction(const ROOT::Math::Polynomial&,double,double); */
/* #pragma link C++ function ROOT::Math::RootFinder<ROOT::Math::Roots::FalsePos>::SetFunction(const ROOT::Math::Polynomial&,double,double); */
/* #pragma link C++ function ROOT::Math::RootFinder<ROOT::Math::Roots::Secant>::SetFunction(const ROOT::Math::Polynomial&,double,double); */

//#pragma link C++ function ROOT::Math::Roots::Brent::SetFunction(double (*)(double, void*), void*, double, double);
