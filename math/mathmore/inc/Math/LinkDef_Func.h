// @(#)root/mathmore:$Id: LinkDef_SpecFunc.h 19826 2007-09-19 19:56:11Z rdm $
// Authors: L. Moneta, A. Zsenei   08/2005 

#ifdef __CINT__

// define header gurad symbols to avoid CINT re-including the file 
#pragma link C++ global ROOT_Math_SpecFuncMathMore;
#pragma link C++ global ROOT_Math_PdfFuncMathMore;
#pragma link C++ global ROOT_Math_QuantFuncMathMore;

// special functions

#pragma link C++ function ROOT::Math::assoc_laguerre(unsigned,double,double);
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
#pragma link C++ function ROOT::Math::laguerre(unsigned,double);
#pragma link C++ function ROOT::Math::legendre(unsigned,double);
#pragma link C++ function ROOT::Math::riemann_zeta(double);
#pragma link C++ function ROOT::Math::sph_bessel(unsigned,double);
#pragma link C++ function ROOT::Math::sph_legendre(unsigned,unsigned,double);
#pragma link C++ function ROOT::Math::sph_neumann(unsigned,double);
#pragma link C++ function ROOT::Math::airy_Ai(double);
#pragma link C++ function ROOT::Math::airy_Bi(double);
#pragma link C++ function ROOT::Math::airy_Ai_deriv(double)
#pragma link C++ function ROOT::Math::airy_Bi_deriv(double);
#pragma link C++ function ROOT::Math::airy_zero_Ai(unsigned int);
#pragma link C++ function ROOT::Math::airy_zero_Bi(unsigned int);
#pragma link C++ function ROOT::Math::airy_zero_Ai_deriv(unsigned int)
#pragma link C++ function ROOT::Math::airy_zero_Bi_deriv(unsigned int);
#pragma link C++ function ROOT::Math::wigner_3j(int,int,int,int,int,int);
#pragma link C++ function ROOT::Math::wigner_6j(int,int,int,int,int,int);
#pragma link C++ function ROOT::Math::wigner_9j(int,int,int,int,int,int,int,int,int);

// statistical functions: 

//pdf 
#pragma link C++ function ROOT::Math::noncentral_chisquared_pdf(double,double,double);


// quantiles: inverses of cdf

#pragma link C++ function ROOT::Math::tdistribution_quantile_c(double,double);
#pragma link C++ function ROOT::Math::tdistribution_quantile(double,double);

#pragma link C++ namespace ROOT::MathMore;
#pragma link C++ function ROOT::MathMore::chisquared_quantile(double,double);
#pragma link C++ function ROOT::MathMore::gamma_quantile(double,double,double);

#endif
