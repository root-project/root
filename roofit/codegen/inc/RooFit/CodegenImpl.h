/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2024
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_CodegenImpl_h
#define RooFit_Detail_CodegenImpl_h

#include <RooFit/CodegenContext.h>

#include <type_traits>

class ParamHistFunc;
class PiecewiseInterpolation;
class RooAbsArg;
class RooAbsReal;
class RooAddPdf;
class RooAddition;
class RooBernstein;
class RooBifurGauss;
class RooCBShape;
class RooChebychev;
class RooConstVar;
class RooConstraintSum;
class RooEffProd;
class RooEfficiency;
class RooExponential;
class RooExtendPdf;
class RooFormulaVar;
class RooGamma;
class RooGaussian;
class RooGenericPdf;
class RooHistFunc;
class RooHistPdf;
class RooLandau;
class RooLognormal;
class RooMultiVarGaussian;
class RooParamHistFunc;
class RooPoisson;
class RooPolyVar;
class RooPolynomial;
class RooProduct;
class RooRatio;
class RooRealIntegral;
class RooRealSumFunc;
class RooRealSumPdf;
class RooRealVar;
class RooRecursiveFraction;
class RooUniform;
class RooWrapperPdf;

namespace RooStats {
namespace HistFactory {
class FlexibleInterpVar;
}
} // namespace RooStats

namespace RooFit {

namespace Detail {
class RooFixedProdPdf;
class RooNLLVarNew;
class RooNormalizedPdf;
} // namespace Detail

namespace Experimental {

class CodegenContext;

void codegenImpl(RooFit::Detail::RooFixedProdPdf &arg, CodegenContext &ctx);
void codegenImpl(RooFit::Detail::RooNLLVarNew &arg, CodegenContext &ctx);
void codegenImpl(RooFit::Detail::RooNormalizedPdf &arg, CodegenContext &ctx);
void codegenImpl(ParamHistFunc &arg, CodegenContext &ctx);
void codegenImpl(PiecewiseInterpolation &arg, CodegenContext &ctx);
void codegenImpl(RooAbsArg &arg, CodegenContext &ctx);
void codegenImpl(RooAddPdf &arg, CodegenContext &ctx);
void codegenImpl(RooAddition &arg, CodegenContext &ctx);
void codegenImpl(RooBernstein &arg, CodegenContext &ctx);
void codegenImpl(RooBifurGauss &arg, CodegenContext &ctx);
void codegenImpl(RooCBShape &arg, CodegenContext &ctx);
void codegenImpl(RooChebychev &arg, CodegenContext &ctx);
void codegenImpl(RooConstVar &arg, CodegenContext &ctx);
void codegenImpl(RooConstraintSum &arg, CodegenContext &ctx);
void codegenImpl(RooEffProd &arg, CodegenContext &ctx);
void codegenImpl(RooEfficiency &arg, CodegenContext &ctx);
void codegenImpl(RooExponential &arg, CodegenContext &ctx);
void codegenImpl(RooExtendPdf &arg, CodegenContext &ctx);
void codegenImpl(RooFormulaVar &arg, CodegenContext &ctx);
void codegenImpl(RooGamma &arg, CodegenContext &ctx);
void codegenImpl(RooGaussian &arg, CodegenContext &ctx);
void codegenImpl(RooGenericPdf &arg, CodegenContext &ctx);
void codegenImpl(RooHistFunc &arg, CodegenContext &ctx);
void codegenImpl(RooHistPdf &arg, CodegenContext &ctx);
void codegenImpl(RooLandau &arg, CodegenContext &ctx);
void codegenImpl(RooLognormal &arg, CodegenContext &ctx);
void codegenImpl(RooMultiVarGaussian &arg, CodegenContext &ctx);
void codegenImpl(RooParamHistFunc &arg, CodegenContext &ctx);
void codegenImpl(RooPoisson &arg, CodegenContext &ctx);
void codegenImpl(RooPolyVar &arg, CodegenContext &ctx);
void codegenImpl(RooPolynomial &arg, CodegenContext &ctx);
void codegenImpl(RooProduct &arg, CodegenContext &ctx);
void codegenImpl(RooRatio &arg, CodegenContext &ctx);
void codegenImpl(RooRealIntegral &arg, CodegenContext &ctx);
void codegenImpl(RooRealSumFunc &arg, CodegenContext &ctx);
void codegenImpl(RooRealSumPdf &arg, CodegenContext &ctx);
void codegenImpl(RooRealVar &arg, CodegenContext &ctx);
void codegenImpl(RooRecursiveFraction &arg, CodegenContext &ctx);
void codegenImpl(RooStats::HistFactory::FlexibleInterpVar &arg, CodegenContext &ctx);
void codegenImpl(RooUniform &arg, CodegenContext &ctx);
void codegenImpl(RooWrapperPdf &arg, CodegenContext &ctx);

std::string codegenIntegralImpl(RooAbsReal &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooBernstein &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooBifurGauss &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooCBShape &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooChebychev &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooEfficiency &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooExponential &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooGamma &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooGaussian &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooHistFunc &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooHistPdf &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooLandau &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooLognormal &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooMultiVarGaussian &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooPoisson &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooPolyVar &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooPolynomial &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooRealSumPdf &arg, int code, const char *rangeName, CodegenContext &ctx);
std::string codegenIntegralImpl(RooUniform &arg, int code, const char *rangeName, CodegenContext &ctx);

template <class Arg_t, int P>
std::string codegenIntegralImpl(Arg_t &arg, int code, const char *rangeName, CodegenContext &ctx, Prio<P> p)
{
   if constexpr (std::is_same<Prio<P>, PrioLowest>::value) {
      return codegenIntegralImpl(arg, code, rangeName, ctx);
   } else {
      return codegenIntegralImpl(arg, code, rangeName, ctx, p.next());
   }
}

template <class Arg_t>
struct CodegenIntegralImplCaller {

   static auto call(RooAbsReal &arg, int code, const char *rangeName, CodegenContext &ctx)
   {
      return codegenIntegralImpl(static_cast<Arg_t &>(arg), code, rangeName, ctx, PrioHighest{});
   }
};

} // namespace Experimental
} // namespace RooFit

#endif
