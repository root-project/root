// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005



#ifdef __CINT__

#pragma extra_include "Math/IFunctionfwd.h";
#pragma extra_include "Math/IFunction.h";

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Math;
#pragma link C++ namespace ROOT::MathMore;


#include "LinkDef_Func.h"
#include "LinkDef_RootFinding.h"

#pragma link C++ class ROOT::Math::MathMoreLib+;

#pragma link C++ class ROOT::Math::ParamFunction<ROOT::Math::IParametricGradFunctionOneDim>+;

#pragma link C++ class ROOT::Math::Polynomial+;
#pragma link C++ class ROOT::Math::ChebyshevApprox+;
#pragma link C++ class ROOT::Math::Derivator+;

#pragma link C++ class ROOT::Math::Vavilov+;
#pragma link C++ class ROOT::Math::VavilovAccurate+;
#pragma link C++ class ROOT::Math::VavilovFast+;

#ifndef _WIN32
// virtual inheritance gives problem when making dictionary on Windows
#pragma link C++ class ROOT::Math::Polynomial+;
#pragma link C++ class ROOT::Math::VavilovAccuratePdf+;
#pragma link C++ class ROOT::Math::VavilovAccurateCdf+;
#pragma link C++ class ROOT::Math::VavilovAccurateQuantile+;
#else
#pragma link C++ class ROOT::Math::Polynomial-;
#pragma link C++ class ROOT::Math::VavilovAccuratePdf-;
#pragma link C++ class ROOT::Math::VavilovAccurateCdf-;
#pragma link C++ class ROOT::Math::VavilovAccurateQuantile-;
#endif


//#pragma extra_include "TF1.h";


#pragma link C++ namespace ROOT::Math::Integration;
#pragma link C++ class ROOT::Math::GSLIntegrator+;

#pragma link C++ namespace ROOT::Math::Minim1D;
#pragma link C++ class ROOT::Math::GSLMinimizer1D+;

#pragma link C++ class ROOT::Math::Interpolator+;

// random  numbers
#pragma link C++ class ROOT::Math::GSLRandomEngine+;

#pragma link C++ class ROOT::Math::GSLRngMT+;
#pragma link C++ class ROOT::Math::GSLRngTaus+;
#pragma link C++ class ROOT::Math::GSLRngRanLux+;
#pragma link C++ class ROOT::Math::GSLRngRanLuxS1+;
#pragma link C++ class ROOT::Math::GSLRngRanLuxS2+;
#pragma link C++ class ROOT::Math::GSLRngRanLuxD1+;
#pragma link C++ class ROOT::Math::GSLRngRanLuxD2+;
#pragma link C++ class ROOT::Math::GSLRngGFSR4+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngMT>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngTaus>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngRanLux>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngRanLuxS1>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngRanLuxS2>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngRanLuxD1>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngRanLuxD2>+;
#pragma link C++ class ROOT::Math::Random<ROOT::Math::GSLRngGFSR4>+;

#pragma link C++ typedef ROOT::Math::RandomMT;
#pragma link C++ typedef ROOT::Math::RandomTaus;
#pragma link C++ typedef ROOT::Math::RandomRanLux;
#pragma link C++ typedef ROOT::Math::RandomGFSR4;


#pragma link C++ class ROOT::Math::GSLQRngSobol+;
#pragma link C++ class ROOT::Math::GSLQRngNiederreiter2+;
#pragma link C++ class ROOT::Math::QuasiRandom<ROOT::Math::GSLQRngSobol>+;
#pragma link C++ class ROOT::Math::QuasiRandom<ROOT::Math::GSLQRngNiederreiter2>+;
#pragma link C++ typedef ROOT::Math::QuasiRandomSobol;
#pragma link C++ typedef ROOT::Math::QuasiRandomNiederreiter;
#pragma link C++ class ROOT::Math::GSLQuasiRandomEngine+;



#pragma link C++ class ROOT::Math::KelvinFunctions+;

#pragma link C++ class ROOT::Math::GSLMinimizer+;
#pragma link C++ class ROOT::Math::GSLSimAnMinimizer+;
#pragma link C++ class ROOT::Math::GSLSimAnFunc+;
#pragma link C++ class ROOT::Math::GSLSimAnParams+;
#pragma link C++ class ROOT::Math::GSLSimAnnealing+;

#pragma link C++ class ROOT::Math::GSLNLSMinimizer-;

#pragma link C++ class ROOT::Math::GSLMCIntegrator+;
#pragma link C++ class ROOT::Math::VegasParameters+;
#pragma link C++ class ROOT::Math::MiserParameters+;

#pragma link C++ class ROOT::Math::GSLMultiRootFinder+;
#pragma link C++ typedef ROOT::Math::MultiRootFinder;

#pragma link C++ class ROOT::Math::VoigtRelativistic + ;

#endif //__CINT__
