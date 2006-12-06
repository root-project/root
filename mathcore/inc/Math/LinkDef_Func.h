// @(#)root/mathcore:$Name:  $:$Id: LinkDef_Func.h,v 1.2 2006/11/17 18:18:47 moneta Exp $
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
#pragma link C++ function ROOT::Math::breitwigner_pdf( double , double);
#pragma link C++ function ROOT::Math::cauchy_pdf( double , double);
#pragma link C++ function ROOT::Math::chisquared_pdf( double , double);
#pragma link C++ function ROOT::Math::exponential_pdf( double , double);
#pragma link C++ function ROOT::Math::fdistribution_pdf( double , double, double);
#pragma link C++ function ROOT::Math::gamma_pdf( double , double, double);
#pragma link C++ function ROOT::Math::gaussian_pdf( double , double);
#pragma link C++ function ROOT::Math::lognormal_pdf( double , double, double);
#pragma link C++ function ROOT::Math::normal_pdf( double , double);
#pragma link C++ function ROOT::Math::poisson_pdf( unsigned int , double);
#pragma link C++ function ROOT::Math::tdistribution_pdf( double , double);
#pragma link C++ function ROOT::Math::uniform_pdf( double , double, double);

#pragma link C++ function ROOT::Math::breitwigner_prob( double , double);
#pragma link C++ function ROOT::Math::breitwigner_quant( double , double);
#pragma link C++ function ROOT::Math::cauchy_prob( double , double);
#pragma link C++ function ROOT::Math::cauchy_quant( double , double);
#pragma link C++ function ROOT::Math::exponential_prob( double , double);
#pragma link C++ function ROOT::Math::exponential_quant( double , double);
#pragma link C++ function ROOT::Math::gaussian_prob( double , double);
#pragma link C++ function ROOT::Math::gaussian_quant( double , double);
#pragma link C++ function ROOT::Math::lognormal_prob( double , double, double);
#pragma link C++ function ROOT::Math::lognormal_quant( double , double, double);
#pragma link C++ function ROOT::Math::normal_prob( double , double);
#pragma link C++ function ROOT::Math::normal_quant( double , double);
#pragma link C++ function ROOT::Math::uniform_prob( double , double, double);
#pragma link C++ function ROOT::Math::uniform_quant( double , double, double);


#endif
