// @(#)root/mathmore:$Name:  $:$Id: LinkDef_StatFunc.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 


// PDF-s

#pragma link C++ function ROOT::Math::chisquared_prob( double , double, double);
#pragma link C++ function ROOT::Math::chisquared_quant( double , double, double);
#pragma link C++ function ROOT::Math::fdistribution_prob( double , double, double, double);
#pragma link C++ function ROOT::Math::fdistribution_quant( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_prob( double , double, double, double);
#pragma link C++ function ROOT::Math::gamma_quant( double , double, double, double);
#pragma link C++ function ROOT::Math::tdistribution_prob( double , double, double);
#pragma link C++ function ROOT::Math::tdistribution_quant( double , double, double);


// and their inverses (also those contained in mathcore)

#pragma link C++ function ROOT::Math::breitwigner_prob_inv(double,double);
#pragma link C++ function ROOT::Math::breitwigner_quant_inv(double,double);
#pragma link C++ function ROOT::Math::cauchy_prob_inv(double,double);
#pragma link C++ function ROOT::Math::cauchy_quant_inv(double,double);
#pragma link C++ function ROOT::Math::chisquared_prob_inv(double,double);
#pragma link C++ function ROOT::Math::chisquared_quant_inv(double,double);
#pragma link C++ function ROOT::Math::exponential_prob_inv(double,double);
#pragma link C++ function ROOT::Math::exponential_quant_inv(double,double);
#pragma link C++ function ROOT::Math::gamma_prob_inv(double,double,double);
#pragma link C++ function ROOT::Math::gamma_quant_inv(double,double,double);
#pragma link C++ function ROOT::Math::gaussian_prob_inv(double,double);
#pragma link C++ function ROOT::Math::gaussian_quant_inv(double,double);
#pragma link C++ function ROOT::Math::lognormal_prob_inv(double,double,double);
#pragma link C++ function ROOT::Math::lognormal_quant_inv(double,double,double);
#pragma link C++ function ROOT::Math::normal_prob_inv(double,double);
#pragma link C++ function ROOT::Math::normal_quant_inv(double,double);
#pragma link C++ function ROOT::Math::tdistribution_prob_inv(double,double);
#pragma link C++ function ROOT::Math::tdistribution_quant_inv(double,double);
#pragma link C++ function ROOT::Math::uniform_prob_inv(double,double,double);
#pragma link C++ function ROOT::Math::uniform_quant_inv(double,double,double);
