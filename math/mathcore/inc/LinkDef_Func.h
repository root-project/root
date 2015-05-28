// @(#)root/mathcore:$Id$
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
#ifdef __CINT__

// special functionses

#pragma link C++ function ROOT::Math::erf( double );
#pragma link C++ function ROOT::Math::erfc( double );
#pragma link C++ function ROOT::Math::tgamma( double );
#pragma link C++ function ROOT::Math::lgamma( double );
#pragma link C++ function ROOT::Math::beta( double , double);

// statistical functions : pdf
#pragma link C++ function ROOT::Math::beta_pdf( double , double, double);
#pragma link C++ function ROOT::Math::binomial_pdf( unsigned int , double, unsigned int);
#pragma link C++ function ROOT::Math::negative_binomial_pdf( unsigned int , double, double);
#pragma link C++ function ROOT::Math::breitwigner_pdf( double , double, double);
#pragma link C++ function ROOT::Math::cauchy_pdf( double , double, double);
#pragma link C++ function ROOT::Math::chisquared_pdf( double , double, double);
#pragma link C++ function ROOT::Math::crystalball_pdf( double , double, double, double, double);
#pragma link C++ function ROOT::Math::exponential_pdf( double , double, double);
#pragma link C++ function ROOT::Math::fdistribution_pdf( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_pdf( double , double, double, double);
#pragma link C++ function ROOT::Math::gaussian_pdf( double , double, double);
#pragma link C++ function ROOT::Math::bigaussian_pdf( double , double, double,double,double,double,double);
#pragma link C++ function ROOT::Math::landau_pdf( double , double, double);
#pragma link C++ function ROOT::Math::lognormal_pdf( double , double, double, double);
#pragma link C++ function ROOT::Math::normal_pdf( double , double, double);
#pragma link C++ function ROOT::Math::poisson_pdf( unsigned int , double);
#pragma link C++ function ROOT::Math::tdistribution_pdf( double , double, double);
#pragma link C++ function ROOT::Math::uniform_pdf( double , double, double, double);

// statistical functions : cdf
#pragma link C++ function ROOT::Math::breitwigner_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::breitwigner_cdf( double , double, double);
#pragma link C++ function ROOT::Math::cauchy_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::cauchy_cdf( double , double, double);
#pragma link C++ function ROOT::Math::chisquared_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::chisquared_cdf( double , double, double);
#pragma link C++ function ROOT::Math::exponential_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::exponential_cdf( double , double, double);
#pragma link C++ function ROOT::Math::gaussian_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::gaussian_cdf( double , double, double);
#pragma link C++ function ROOT::Math::lognormal_cdf_c( double , double, double, double);
#pragma link C++ function ROOT::Math::lognormal_cdf( double , double, double, double);
#pragma link C++ function ROOT::Math::normal_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::normal_cdf( double , double, double);
#pragma link C++ function ROOT::Math::uniform_cdf_c( double , double, double, double);
#pragma link C++ function ROOT::Math::uniform_cdf( double , double, double, double);

#pragma link C++ function ROOT::Math::beta_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::beta_cdf( double , double, double);
#pragma link C++ function ROOT::Math::fdistribution_cdf_c( double , double, double, double);
#pragma link C++ function ROOT::Math::fdistribution_cdf( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_cdf_c( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_cdf( double , double, double, double);
#pragma link C++ function ROOT::Math::tdistribution_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::tdistribution_cdf( double , double, double);
#pragma link C++ function ROOT::Math::landau_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::landau_cdf( double , double, double);

#pragma link C++ function ROOT::Math::binomial_cdf( unsigned int , double, unsigned int);
#pragma link C++ function ROOT::Math::binomial_cdf_c( unsigned int , double, unsigned int);
#pragma link C++ function ROOT::Math::negative_binomial_cdf( unsigned int , double, double);
#pragma link C++ function ROOT::Math::negative_binomial_cdf_c( unsigned int , double, double);
#pragma link C++ function ROOT::Math::poisson_cdf( unsigned int , double);
#pragma link C++ function ROOT::Math::poisson_cdf_c( unsigned int , double);


// statitsical functions: quantiles
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
#pragma link C++ function ROOT::Math::uniform_quantile_c(double,double,double);
#pragma link C++ function ROOT::Math::uniform_quantile(double,double,double);

//Chebyshev functions and polynomials
#pragma link C++ function ROOT::Math::Chebyshev0(double, double);
#pragma link C++ function ROOT::Math::Chebyshev1(double, double, double);
#pragma link C++ function ROOT::Math::Chebyshev2(double, double, double, double);
#pragma link C++ function ROOT::Math::Chebyshev3(double, double, double, double, double);
#pragma link C++ function ROOT::Math::Chebyshev4(double, double, double, double, double, double);
#pragma link C++ function ROOT::Math::Chebyshev5(double, double, double, double, double, double, double);
#pragma link C++ function ROOT::Math::ChebyshevN(unsigned int,double, const double*);
#pragma link C++ class ROOT::Math::ChebyshevPol+;


#endif
