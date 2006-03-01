// @(#)root/mathmore:$Name:  $:$Id: LinkDef_SpecFunc.h,v 1.2 2006/01/23 15:52:59 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 

#ifdef __CINT__


#pragma link C++ function ROOT::Math::assoc_laguerre(unsigned,unsigned,double);
#pragma link C++ function ROOT::Math::assoc_legendre(unsigned,unsigned,double);
#pragma link C++ function ROOT::Math::comp_ellint_1(double);
#pragma link C++ function ROOT::Math::comp_ellint_2(double);
#pragma link C++ function ROOT::Math::comp_ellint_3(double, double);
#pragma link C++ function ROOT::Math::conf_hyperg(double,double,double);
#pragma link C++ function ROOT::Math::conf_hypergU(double,double,double);
#pragma link C++ function ROOT::Math::cyl_bessel_i(double,double);
#pragma link C++ function ROOT::Math::cyl_bessel_j(double,double);
#pragma link C++ function ROOT::Math::cyl_bessel_k(double,double);
#pragma link C++ function ROOT::Math::cyl_neumann(double,double);
#pragma link C++ function ROOT::Math::ellint_1(double,double);
#pragma link C++ function ROOT::Math::ellint_2(double,double);
#pragma link C++ function ROOT::Math::ellint_3(double,double,double);
#pragma link C++ function ROOT::Math::expint(double);
#pragma link C++ function ROOT::Math::hyperg(double,double,double,double);
#pragma link C++ function ROOT::Math::legendre(unsigned,double);
#pragma link C++ function ROOT::Math::riemann_zeta(double);
#pragma link C++ function ROOT::Math::sph_bessel(unsigned,double);
#pragma link C++ function ROOT::Math::sph_neumann(unsigned,double);

#endif
