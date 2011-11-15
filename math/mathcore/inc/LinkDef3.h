// @(#)root/mathcore:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace ROOT::Fit;

#pragma link C++ class ROOT::Fit::DataRange;
#pragma link C++ class ROOT::Fit::DataOptions;

#pragma link C++ class ROOT::Fit::Fitter;
#pragma link C++ class ROOT::Fit::FitConfig+;
#pragma link C++ class ROOT::Fit::FitData+;
#pragma link C++ class ROOT::Fit::BinData+;
#pragma link C++ class ROOT::Fit::UnBinData+;
#pragma link C++ class ROOT::Fit::SparseData+;
#pragma link C++ class ROOT::Fit::FitResult+;
#pragma link C++ class ROOT::Fit::ParameterSettings+;

///skip  the dictionary for the fit method functions 
#ifndef _WIN32

#pragma link C++ class ROOT::Fit::Chi2FCN<ROOT::Math::IBaseFunctionMultiDim>-;
#pragma link C++ class ROOT::Fit::Chi2FCN<ROOT::Math::IGradientFunctionMultiDim>-;
#pragma link C++ class ROOT::Fit::LogLikelihoodFCN<ROOT::Math::IBaseFunctionMultiDim>-;
#pragma link C++ class ROOT::Fit::LogLikelihoodFCN<ROOT::Math::IGradientFunctionMultiDim>-;
#pragma link C++ class ROOT::Fit::PoissonLikelihoodFCN<ROOT::Math::IBaseFunctionMultiDim>-;
#pragma link C++ class ROOT::Fit::PoissonLikelihoodFCN<ROOT::Math::IGradientFunctionMultiDim>-;


#pragma link C++ typedef ROOT::Fit::Chi2Function;
#pragma link C++ typedef ROOT::Fit::Chi2GradFunction;
#pragma link C++ typedef ROOT::Fit::PoissonLLFunction;
#pragma link C++ typedef ROOT::Fit::PoissonLLGradFunction;
#pragma link C++ typedef ROOT::Fit::LogLikelihoodFunction;
#pragma link C++ typedef ROOT::Fit::LogLikelihoodGradFunction;

#endif

//fitter template functions
#pragma link C++ function ROOT::Fit::Fitter::Fit(const ROOT::Fit::BinData &, const ROOT::Math::IParametricFunctionMultiDim&);
#pragma link C++ function ROOT::Fit::Fitter::Fit(const ROOT::Fit::UnBinData &,const ROOT::Math::IParametricFunctionMultiDim &);
#pragma link C++ function ROOT::Fit::Fitter::Fit(const ROOT::Fit::BinData &, const ROOT::Math::IParametricGradFunctionMultiDim&);
#pragma link C++ function ROOT::Fit::Fitter::Fit(const ROOT::Fit::UnBinData &,const ROOT::Math::IParametricGradFunctionMultiDim &);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::BinData &,const ROOT::Math::IParametricFunctionMultiDim&,bool);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::UnBinData &,const ROOT::Math::IParametricFunctionMultiDim&,bool);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::BinData &,const ROOT::Math::IParametricGradFunctionMultiDim&,bool);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::UnBinData &,const ROOT::Math::IParametricGradFunctionMultiDim&,bool);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::BinData &);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::UnBinData &);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::BinData &, bool);
#pragma link C++ function ROOT::Fit::Fitter::LikelihoodFit(const ROOT::Fit::UnBinData &, bool);

#pragma link C++ class vector<ROOT::Fit::ParameterSettings>;

#endif
