// @(#)root/mathmore:$Name:  $:$Id: LinkDef_StatFunc.h,v 1.2 2006/01/23 15:52:59 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 


// PDF-s
#ifdef __CINT__


#pragma link C++ function ROOT::Math::chisquared_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::chisquared_cdf( double , double, double);
#pragma link C++ function ROOT::Math::fdistribution_cdf_c( double , double, double, double);
#pragma link C++ function ROOT::Math::fdistribution_cdf( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_cdf_c( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_cdf( double , double, double, double);
#pragma link C++ function ROOT::Math::tdistribution_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::tdistribution_cdf( double , double, double);


// and their inverses (also those contained in mathcore)

#pragma link C++ function ROOT::Math::breitwigner_quantile_c(double,double);
#pragma link C++ function ROOT::Math::breitwigner_quantile(double,double);
#pragma link C++ function ROOT::Math::cauchy_quantile_c(double,double);
#pragma link C++ function ROOT::Math::cauchy_quantile(double,double);
#pragma link C++ function ROOT::Math::chisquared_quantile_c(double,double);
#pragma link C++ function ROOT::Math::chisquared_quantile(double,double);
#pragma link C++ function ROOT::Math::exponential_quantile_c(double,double);
#pragma link C++ function ROOT::Math::exponential_quantile(double,double);
#pragma link C++ function ROOT::Math::gamma_quantile_c(double,double,double);
#pragma link C++ function ROOT::Math::gamma_quantile(double,double,double);
#pragma link C++ function ROOT::Math::gaussian_quantile_c(double,double);
#pragma link C++ function ROOT::Math::gaussian_quantile(double,double);
#pragma link C++ function ROOT::Math::lognormal_quantile_c(double,double,double);
#pragma link C++ function ROOT::Math::lognormal_quantile(double,double,double);
#pragma link C++ function ROOT::Math::normal_quantile_c(double,double);
#pragma link C++ function ROOT::Math::normal_quantile(double,double);
#pragma link C++ function ROOT::Math::tdistribution_quantile_c(double,double);
#pragma link C++ function ROOT::Math::tdistribution_quantile(double,double);
#pragma link C++ function ROOT::Math::uniform_quantile_c(double,double,double);
#pragma link C++ function ROOT::Math::uniform_quantile(double,double,double);

#endif
