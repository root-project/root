// @(#)root/mathcore:$Name:  $:$Id: LinkDef_Func.h,v 1.3 2006/12/06 15:08:52 moneta Exp $
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/
#ifdef __CINT__

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Math;


#pragma link C++ class ROOT::Math::IBaseFunction<ROOT::Math::OneDim>+;
#pragma link C++ class ROOT::Math::IGradientFunction<ROOT::Math::OneDimt>+;
#pragma link C++ class ROOT::Math::IBaseFunction<ROOT::Math::MultiDim>+;
#pragma link C++ class ROOT::Math::IGradientFunction<ROOT::Math::MultiDim>+;

#pragma link C++ class ROOT::Math::IParametricFunction<ROOT::Math::OneDim>+;
#pragma link C++ class ROOT::Math::IParametricGradFunction<ROOT::Math::OneDim>+;
#pragma link C++ class ROOT::Math::IParametricFunction<ROOT::Math::MultiDim>+;
#pragma link C++ class ROOT::Math::IParametricGradFunction<ROOT::Math::MultiDim>+;


#pragma link C++ function ROOT::Math::erf( double );
#pragma link C++ function ROOT::Math::erfc( double );
#pragma link C++ function ROOT::Math::tgamma( double );
#pragma link C++ function ROOT::Math::lgamma( double );
#pragma link C++ function ROOT::Math::beta( double , double);

#pragma link C++ function ROOT::Math::binomial_pdf( unsigned int , double, unsigned int);
#pragma link C++ function ROOT::Math::breitwigner_pdf( double , double, double);
#pragma link C++ function ROOT::Math::cauchy_pdf( double , double, double);
#pragma link C++ function ROOT::Math::chisquared_pdf( double , double, double);
#pragma link C++ function ROOT::Math::exponential_pdf( double , double, double);
#pragma link C++ function ROOT::Math::fdistribution_pdf( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_pdf( double , double, double, double);
#pragma link C++ function ROOT::Math::gaussian_pdf( double , double, double);
#pragma link C++ function ROOT::Math::lognormal_pdf( double , double, double, double);
#pragma link C++ function ROOT::Math::normal_pdf( double , double, double);
#pragma link C++ function ROOT::Math::poisson_pdf( unsigned int , double);
#pragma link C++ function ROOT::Math::tdistribution_pdf( double , double, double);
#pragma link C++ function ROOT::Math::uniform_pdf( double , double, double, double);

#pragma link C++ function ROOT::Math::breitwigner_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::breitwigner_cdf( double , double, double);
#pragma link C++ function ROOT::Math::cauchy_cdf_c( double , double, double);
#pragma link C++ function ROOT::Math::cauchy_cdf( double , double, double);
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


#endif
