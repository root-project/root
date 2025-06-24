/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2023
 *   Jonas Rembser, CERN 2024
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/CodegenImpl.h>

#include <RooFit/CodegenContext.h>

#include <RooAddPdf.h>
#include <RooAddition.h>
#include <RooBernstein.h>
#include <RooBifurGauss.h>
#include <RooCBShape.h>
#include <RooCategory.h>
#include <RooChebychev.h>
#include <RooConstVar.h>
#include <RooConstraintSum.h>
#include <RooEffProd.h>
#include <RooEfficiency.h>
#include <RooExponential.h>
#include <RooExtendPdf.h>
#include <RooFit/Detail/RooNLLVarNew.h>
#include <RooFit/Detail/RooNormalizedPdf.h>
#include <RooFormulaVar.h>
#include <RooGamma.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLandau.h>
#include <RooLognormal.h>
#include <RooMultiPdf.h>
#include <RooMultiVarGaussian.h>
#include <RooParamHistFunc.h>
#include <RooPoisson.h>
#include <RooPolyVar.h>
#include <RooPolynomial.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooRatio.h>
#include <RooRealIntegral.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooRecursiveFraction.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooUniform.h>
#include <RooWrapperPdf.h>

#include "RooFitImplHelpers.h"

#include <TInterpreter.h>

namespace RooFit {
namespace Experimental {

namespace {

std::string mathFunc(std::string const &name)
{
   return "RooFit::Detail::MathFuncs::" + name;
}

void rooHistTranslateImpl(RooAbsArg const &arg, CodegenContext &ctx, int intOrder, RooDataHist const &dataHist,
                          const RooArgSet &obs, bool correctForBinSize, bool cdfBoundaries)
{
   if (intOrder != 0 && !(!cdfBoundaries && !correctForBinSize && intOrder == 1 && obs.size() == 1)) {
      ooccoutE(&arg, InputArguments) << "RooHistPdf::weight(" << arg.GetName()
                                     << ") ERROR: codegen currently only supports non-interpolation cases."
                                     << std::endl;
      return;
   }

   if (intOrder == 1) {
      RooAbsBinning const &binning = *dataHist.getBinnings()[0];
      std::string weightArr = dataHist.declWeightArrayForCodeSquash(ctx, correctForBinSize);
      ctx.addResult(&arg, ctx.buildCall(mathFunc("interpolate1d"), binning.lowBound(), binning.highBound(), *obs[0],
                                        binning.numBins(), weightArr));
      return;
   }
   std::string const &offset = dataHist.calculateTreeIndexForCodeSquash(ctx, obs);
   std::string weightArr = dataHist.declWeightArrayForCodeSquash(ctx, correctForBinSize);
   ctx.addResult(&arg, "*(" + weightArr + " + " + offset + ")");
}

std::string realSumPdfTranslateImpl(CodegenContext &ctx, RooAbsArg const &arg, RooArgList const &funcList,
                                    RooArgList const &coefList, bool normalize)
{
   bool noLastCoeff = funcList.size() != coefList.size();

   std::string const &funcName = ctx.buildArg(funcList);
   std::string const &coeffName = ctx.buildArg(coefList);
   std::string const &coeffSize = std::to_string(coefList.size());

   std::string sum = ctx.getTmpVarName();
   std::string coeffSum = ctx.getTmpVarName();
   ctx.addToCodeBody(&arg, "double " + sum + " = 0;\ndouble " + coeffSum + "= 0;\n");

   std::string iterator = "i_" + ctx.getTmpVarName();
   std::string subscriptExpr = "[" + iterator + "]";

   std::string code = "for(int " + iterator + " = 0; " + iterator + " < " + coeffSize + "; " + iterator + "++) {\n" +
                      sum + " += " + funcName + subscriptExpr + " * " + coeffName + subscriptExpr + ";\n";
   code += coeffSum + " += " + coeffName + subscriptExpr + ";\n";
   code += "}\n";

   if (noLastCoeff) {
      code += sum + " += " + funcName + "[" + coeffSize + "]" + " * (1 - " + coeffSum + ");\n";
   } else if (normalize) {
      code += sum + " /= " + coeffSum + ";\n";
   }
   ctx.addToCodeBody(&arg, code);

   return sum;
}

} // namespace

void codegenImpl(RooFit::Detail::RooFixedProdPdf &arg, CodegenContext &ctx)
{
   if (arg.cache()._isRearranged) {
      ctx.addResult(&arg, ctx.buildCall(mathFunc("ratio"), *arg.cache()._rearrangedNum, *arg.cache()._rearrangedDen));
   } else {
      ctx.addResult(&arg, ctx.buildCall(mathFunc("product"), arg.cache()._partList, arg.cache()._partList.size()));
   }
}

void codegenImpl(ParamHistFunc &arg, CodegenContext &ctx)
{
   std::string const &idx = arg.dataHist().calculateTreeIndexForCodeSquash(ctx, arg.dataVars(), true);
   std::string const &paramNames = ctx.buildArg(arg.paramList());

   ctx.addResult(&arg, paramNames + "[" + idx + "]");
}

void codegenImpl(PiecewiseInterpolation &arg, CodegenContext &ctx)
{
   auto const &interpCodes = arg.interpolationCodes();

   std::size_t n = interpCodes.size();

   std::string resName = "total_" + ctx.getTmpVarName();
   for (std::size_t i = 0; i < n; ++i) {
      if (interpCodes[i] != interpCodes[0]) {
         oocoutE(&arg, InputArguments)
            << "FlexibleInterpVar::evaluate ERROR:  Code Squashing AD does not yet support having "
               "different interpolation codes for the same class object "
            << std::endl;
      }
   }

   // The PiecewiseInterpolation class is used in the context of HistFactory
   // models, where is is always used the same way: all RooAbsReals in _lowSet,
   // _histSet, and also nominal are 1D RooHistFuncs with with same structure.
   //
   // Therefore, we can make a big optimization: we get the bin index only once
   // here in the generated code for PiecewiseInterpolation. Then, we also
   // rearrange the histogram data in such a way that we can always pass the
   // same arrays to the free function that implements the interpolation, just
   // with a dynamic offset calculated from the bin index.
   RooDataHist const &nomHist = dynamic_cast<RooHistFunc const &>(*arg.nominalHist()).dataHist();
   int nBins = nomHist.numEntries();
   std::vector<double> valsNominal;
   std::vector<double> valsLow;
   std::vector<double> valsHigh;
   for (int i = 0; i < nBins; ++i) {
      valsNominal.push_back(nomHist.weight(i));
   }
   for (int i = 0; i < nBins; ++i) {
      for (std::size_t iParam = 0; iParam < n; ++iParam) {
         valsLow.push_back(dynamic_cast<RooHistFunc const &>(arg.lowList()[iParam]).dataHist().weight(i));
         valsHigh.push_back(dynamic_cast<RooHistFunc const &>(arg.highList()[iParam]).dataHist().weight(i));
      }
   }
   std::string idxName = ctx.getTmpVarName();
   std::string valsNominalStr = ctx.buildArg(valsNominal);
   std::string valsLowStr = ctx.buildArg(valsLow);
   std::string valsHighStr = ctx.buildArg(valsHigh);
   std::string nStr = std::to_string(n);
   std::string code;

   std::string lowName = ctx.getTmpVarName();
   std::string highName = ctx.getTmpVarName();
   std::string nominalName = ctx.getTmpVarName();
   code +=
      "unsigned int " + idxName + " = " +
      nomHist.calculateTreeIndexForCodeSquash(ctx, dynamic_cast<RooHistFunc const &>(*arg.nominalHist()).variables()) +
      ";\n";
   code += "double const* " + lowName + " = " + valsLowStr + " + " + nStr + " * " + idxName + ";\n";
   code += "double const* " + highName + " = " + valsHighStr + " + " + nStr + " * " + idxName + ";\n";
   code += "double " + nominalName + " = *(" + valsNominalStr + " + " + idxName + ");\n";

   std::string funcCall = ctx.buildCall(mathFunc("flexibleInterp"), interpCodes[0], arg.paramList(), n, lowName,
                                        highName, 1.0, nominalName, 0.0);
   code += "double " + resName + " = " + funcCall + ";\n";

   if (arg.positiveDefinite()) {
      code += resName + " = " + resName + " < 0 ? 0 : " + resName + ";\n";
   }

   ctx.addToCodeBody(&arg, code);
   ctx.addResult(&arg, resName);
}

////////////////////////////////////////////////////////////////////////////////
/// This function defines a translation for each RooAbsReal based object that can be used
/// to express the class as simple C++ code. The function adds the code represented by
/// each class as an std::string (that is later concatenated with code strings from translate calls)
/// to form the C++ code that AD tools can understand. Any class that wants to support AD, has to
/// implement this function.
///
/// \param[in] ctx An object to manage auxiliary information for code-squashing. Also takes the
/// code string that this class outputs into the squashed code through the 'addToCodeBody' function.
void codegenImpl(RooAbsArg &arg, CodegenContext &ctx)
{
   std::stringstream errorMsg;
   errorMsg << "Translate function for class \"" << arg.ClassName() << "\" has not yet been implemented.";
   oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
   return ctx.addResult(&arg, "1.0");
}

void codegenImpl(RooAddPdf &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, realSumPdfTranslateImpl(ctx, arg, arg.pdfList(), arg.coefList(), true));
}

void codegenImpl(RooMultiVarGaussian &arg, CodegenContext &ctx)
{
   auto const &covI = arg.covarianceMatrixInverse();
   std::span<const double> covISpan{covI.GetMatrixArray(), static_cast<size_t>(covI.GetNoElements())};
   ctx.addResult(&arg,
                 ctx.buildCall(mathFunc("multiVarGaussian"), arg.xVec().size(), arg.xVec(), arg.muVec(), covISpan));
}

void codegenImpl(RooMultiPdf &arg, CodegenContext &ctx)
{
   int numPdfs = arg.getNumPdfs();

   // MathFunc call

   if (numPdfs > 2) { // the value of this number should be discussed.Beyound a certain number of indices MathFunc call
                      // becomes more efficient.
      ctx.addResult(&arg, ctx.buildCall(mathFunc("multipdf"), arg.indexCategory(), arg.getPdfList()));

      std::cout << "MathFunc call used\n";

   } else {

      // Ternary nested expression
      std::string indexExpr = ctx.getResult(arg.indexCategory());

      // int numPdfs = arg.getNumPdfs();
      std::string expr;

      for (int i = 0; i < numPdfs; ++i) {
         RooAbsPdf *pdf = arg.getPdf(i);
         std::string pdfExpr = ctx.getResult(*pdf);

         expr += "(" + indexExpr + " == " + std::to_string(i) + " ? (" + pdfExpr + ") : ";
      }

      expr += "0.0";
      expr += std::string(numPdfs, ')'); // Close all ternary operators

      ctx.addResult(&arg, expr);
      std::cout << "Ternary expression call used \n";
   }
}

// RooCategory index added.
void codegenImpl(RooCategory &arg, CodegenContext &ctx)
{
   int idx = ctx.observableIndexOf(arg);
   if (idx < 0) {

      idx = 1;
      ctx.addVecObs(arg.GetName(), idx);
   }

   std::string result = std::to_string(arg.getCurrentIndex());
   ctx.addResult(&arg, result);
}

void codegenImpl(RooAddition &arg, CodegenContext &ctx)
{
   if (arg.list().empty()) {
      ctx.addResult(&arg, "0.0");
   }
   std::string result;
   if (arg.list().size() > 1)
      result += "(";

   std::size_t i = 0;
   for (auto *component : static_range_cast<RooAbsReal *>(arg.list())) {

      if (!dynamic_cast<RooFit::Detail::RooNLLVarNew *>(component) || arg.list().size() == 1) {
         result += ctx.getResult(*component);
         ++i;
         if (i < arg.list().size())
            result += '+';
         continue;
      }
      result += ctx.buildFunction(*component, ctx.outputSizes()) + "(params, obs, xlArr)";
      ++i;
      if (i < arg.list().size())
         result += '+';
   }
   if (arg.list().size() > 1)
      result += ')';
   ctx.addResult(&arg, result);
}

void codegenImpl(RooBernstein &arg, CodegenContext &ctx)
{
   arg.fillBuffer();
   ctx.addResult(&arg, ctx.buildCall(mathFunc("bernstein"), arg.x(), arg.xmin(), arg.xmax(), arg.coefList(),
                                     arg.coefList().size()));
}

void codegenImpl(RooBifurGauss &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg,
                 ctx.buildCall(mathFunc("bifurGauss"), arg.getX(), arg.getMean(), arg.getSigmaL(), arg.getSigmaR()));
}

void codegenImpl(RooCBShape &arg, CodegenContext &ctx)
{
   ctx.addResult(
      &arg, ctx.buildCall(mathFunc("cbShape"), arg.getM(), arg.getM0(), arg.getSigma(), arg.getAlpha(), arg.getN()));
}

void codegenImpl(RooChebychev &arg, CodegenContext &ctx)
{
   // first bring the range of the variable _x to the normalised range [-1, 1]
   // calculate sum_k c_k T_k(x) where x is given in the normalised range,
   // c_0 = 1, and the higher coefficients are given in _coefList
   double xmax = static_cast<RooAbsRealLValue const &>(arg.x()).getMax(arg.refRangeName());
   double xmin = static_cast<RooAbsRealLValue const &>(arg.x()).getMin(arg.refRangeName());

   ctx.addResult(&arg,
                 ctx.buildCall(mathFunc("chebychev"), arg.coefList(), arg.coefList().size(), arg.x(), xmin, xmax));
}

void codegenImpl(RooConstVar &arg, CodegenContext &ctx)
{
   // Just return a stringy-field version of the const value.
   // Formats to the maximum precision.
   constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
   std::stringstream ss;
   ss.precision(max_precision);
   // Just use toString to make sure we do not output 'inf'.
   // This is really ugly for large numbers...
   ss << std::fixed << RooNumber::toString(arg.getVal());
   ctx.addResult(&arg, ss.str());
}

void codegenImpl(RooConstraintSum &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("constraintSum"), arg.list(), arg.list().size()));
}

void codegenImpl(RooGamma &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall("TMath::GammaDist", arg.getX(), arg.getGamma(), arg.getMu(), arg.getBeta()));
}

void codegenImpl(RooFormulaVar &arg, CodegenContext &ctx)
{
   arg.getVal(); // to trigger the creation of the TFormula
   std::string funcName = arg.getUniqueFuncName();
   ctx.collectFunction(funcName);
   ctx.addResult(&arg, ctx.buildCall(funcName, arg.dependents()));
}

void codegenImpl(RooEffProd &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("effProd"), arg.eff(), arg.pdf()));
}

void codegenImpl(RooEfficiency &arg, CodegenContext &ctx)
{
   RooAbsCategory const &cat = arg.cat();
   int sigCatIndex = cat.lookupIndex(arg.sigCatName());
   ctx.addResult(&arg, ctx.buildCall(mathFunc("efficiency"), arg.effFunc(), cat, sigCatIndex));
}

void codegenImpl(RooExponential &arg, CodegenContext &ctx)
{
   // Build a call to the stateless exponential defined later.
   std::string coef;
   if (arg.negateCoefficient()) {
      coef += "-";
   }
   coef += ctx.getResult(arg.coefficient());
   ctx.addResult(&arg, "std::exp(" + coef + " * " + ctx.getResult(arg.variable()) + ")");
}

void codegenImpl(RooExtendPdf &arg, CodegenContext &ctx)
{
   // Use the result of the underlying pdf.
   ctx.addResult(&arg, ctx.getResult(arg.pdf()));
}

void codegenImpl(RooGaussian &arg, CodegenContext &ctx)
{
   // Build a call to the stateless gaussian defined later.
   ctx.addResult(&arg, ctx.buildCall(mathFunc("gaussian"), arg.getX(), arg.getMean(), arg.getSigma()));
}

void codegenImpl(RooGenericPdf &arg, CodegenContext &ctx)
{
   arg.getVal(); // to trigger the creation of the TFormula
   std::string funcName = arg.getUniqueFuncName();
   ctx.collectFunction(funcName);
   ctx.addResult(&arg, ctx.buildCall(funcName, arg.dependents()));
}

void codegenImpl(RooHistFunc &arg, CodegenContext &ctx)
{
   rooHistTranslateImpl(arg, ctx, arg.getInterpolationOrder(), arg.dataHist(), arg.variables(), false,
                        arg.getCdfBoundaries());
}

void codegenImpl(RooHistPdf &arg, CodegenContext &ctx)
{
   rooHistTranslateImpl(arg, ctx, arg.getInterpolationOrder(), arg.dataHist(), arg.variables(), !arg.haveUnitNorm(),
                        arg.getCdfBoundaries());
}

void codegenImpl(RooLandau &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("landau"), arg.getX(), arg.getMean(), arg.getSigma()));
}

void codegenImpl(RooLognormal &arg, CodegenContext &ctx)
{
   std::string funcName = arg.useStandardParametrization() ? "logNormalEvaluateStandard" : "logNormal";
   ctx.addResult(&arg, ctx.buildCall(mathFunc(funcName), arg.getX(), arg.getShapeK(), arg.getMedian()));
}

void codegenImpl(RooFit::Detail::RooNLLVarNew &arg, CodegenContext &ctx)
{
   if (arg.binnedL() && !arg.pdf().getAttribute("BinnedLikelihoodActiveYields")) {
      std::stringstream errorMsg;
      errorMsg << "codegen: binned likelihood optimization is only supported when raw pdf "
                  "values can be interpreted as yields."
               << " This is not the case for HistFactory models written with ROOT versions before 6.26.00";
      oocoutE(&arg, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
   }

   std::string weightSumName = RooFit::Detail::makeValidVarName(arg.GetName()) + "WeightSum";
   std::string resName = RooFit::Detail::makeValidVarName(arg.GetName()) + "Result";
   ctx.addResult(&arg, resName);
   ctx.addToGlobalScope("double " + weightSumName + " = 0.0;\n");
   ctx.addToGlobalScope("double " + resName + " = 0.0;\n");

   const bool needWeightSum = arg.expectedEvents() || arg.simCount() > 1;

   if (needWeightSum) {
      auto scope = ctx.beginLoop(&arg);
      ctx.addToCodeBody(weightSumName + " += " + ctx.getResult(arg.weightVar()) + ";\n");
   }
   if (arg.simCount() > 1) {
      std::string simCountStr = std::to_string(static_cast<double>(arg.simCount()));
      ctx.addToCodeBody(resName + " += " + weightSumName + " * std::log(" + simCountStr + ");\n");
   }

   // Begin loop scope for the observables and weight variable. If the weight
   // is a scalar, the context will ignore it for the loop scope. The closing
   // brackets of the loop is written at the end of the scopes lifetime.
   {
      auto scope = ctx.beginLoop(&arg);
      std::string term = ctx.buildCall(mathFunc("nll"), arg.pdf(), arg.weightVar(), arg.binnedL(), 0);
      ctx.addToCodeBody(&arg, resName + " += " + term + ";");
   }
   if (arg.expectedEvents()) {
      std::string expected = ctx.getResult(*arg.expectedEvents());
      ctx.addToCodeBody(resName + " += " + expected + " - " + weightSumName + " * std::log(" + expected + ");\n");
   }
}

void codegenImpl(RooFit::Detail::RooNormalizedPdf &arg, CodegenContext &ctx)
{
   // For now just return function/normalization integral.
   ctx.addResult(&arg, ctx.getResult(arg.pdf()) + "/" + ctx.getResult(arg.normIntegral()));
}

void codegenImpl(RooParamHistFunc &arg, CodegenContext &ctx)
{
   std::string const &idx = arg.dataHist().calculateTreeIndexForCodeSquash(ctx, arg.xList());
   std::string arrName = ctx.buildArg(arg.paramList());
   std::string result = arrName + "[" + idx + "]";
   if (arg.relParam()) {
      // get weight[idx] * binv[idx]. Here we get the bin volume for the first element as we assume the distribution to
      // be binned uniformly.
      double binV = arg.dataHist().binVolume(0);
      std::string weightArr = arg.dataHist().declWeightArrayForCodeSquash(ctx, false);
      result += " * *(" + weightArr + " + " + idx + ") * " + std::to_string(binV);
   }
   ctx.addResult(&arg, result);
}

void codegenImpl(RooPoisson &arg, CodegenContext &ctx)
{
   std::string xName = ctx.getResult(arg.getX());
   if (!arg.getNoRounding())
      xName = "std::floor(" + xName + ")";

   ctx.addResult(&arg, ctx.buildCall(mathFunc("poisson"), xName, arg.getMean()));
}

void codegenImpl(RooPolyVar &arg, CodegenContext &ctx)
{
   const unsigned sz = arg.coefList().size();
   if (!sz) {
      ctx.addResult(&arg, std::to_string(arg.lowestOrder() ? 1. : 0.));
      return;
   }

   ctx.addResult(&arg, ctx.buildCall(mathFunc("polynomial"), arg.coefList(), sz, arg.lowestOrder(), arg.x()));
}

void codegenImpl(RooPolynomial &arg, CodegenContext &ctx)
{
   const unsigned sz = arg.coefList().size();
   if (!sz) {
      ctx.addResult(&arg, std::to_string(arg.lowestOrder() ? 1. : 0.));
      return;
   }

   ctx.addResult(&arg, ctx.buildCall(mathFunc("polynomial<true>"), arg.coefList(), sz, arg.lowestOrder(), arg.x()));
}

void codegenImpl(RooProduct &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("product"), arg.realComponents(), arg.realComponents().size()));
}

void codegenImpl(RooRatio &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("ratio"), arg.numerator(), arg.denominator()));
}

namespace {

std::string codegenIntegral(RooAbsReal &arg, int code, const char *rangeName, CodegenContext &ctx)
{
   using Func = std::string (*)(RooAbsReal &, int, const char *, CodegenContext &);

   Func func;

   TClass *tclass = arg.IsA();

   // Cache the overload resolutions
   static std::unordered_map<TClass *, Func> dispatchMap;

   auto found = dispatchMap.find(tclass);

   if (found != dispatchMap.end()) {
      func = found->second;
   } else {
      // Can probably done with CppInterop in the future to avoid string manipulation.
      std::stringstream cmd;
      cmd << "&RooFit::Experimental::CodegenIntegralImplCaller<" << tclass->GetName() << ">::call;";
      func = reinterpret_cast<Func>(gInterpreter->ProcessLine(cmd.str().c_str()));
      dispatchMap[tclass] = func;
   }

   return func(arg, code, rangeName, ctx);
}

} // namespace

void codegenImpl(RooRealIntegral &arg, CodegenContext &ctx)
{
   if (arg.numIntCatVars().empty() && arg.numIntRealVars().empty()) {
      ctx.addResult(&arg, codegenIntegral(const_cast<RooAbsReal &>(arg.integrand()), arg.mode(), arg.intRange(), ctx));
      return;
   }

   if (arg.intVars().size() != 1 || arg.numIntRealVars().size() != 1) {
      std::stringstream errorMsg;
      errorMsg << "Only analytical integrals and 1D numeric integrals are supported for AD for class"
               << arg.integrand().GetName();
      oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   auto &intVar = static_cast<RooAbsRealLValue &>(*arg.numIntRealVars()[0]);

   std::string obsName = ctx.getTmpVarName();
   std::string oldIntVarResult = ctx.getResult(intVar);
   ctx.addResult(&intVar, "obs[0]");

   std::string funcName = ctx.buildFunction(arg.integrand(), {});

   std::stringstream ss;

   ss << "double " << obsName << "[1];\n";

   std::string resName = RooFit::Detail::makeValidVarName(arg.GetName()) + "Result";
   ctx.addResult(&arg, resName);
   ctx.addToGlobalScope("double " + resName + " = 0.0;\n");

   // TODO: once Clad has support for higher-order functions (follow also the
   // Clad issue #637), we could refactor this code into an actual function
   // instead of hardcoding it here as a string.
   ss << "{\n"
      << "   const int n = 1000; // number of sampling points\n"
      << "   double d = " << intVar.getMax(arg.intRange()) << " - " << intVar.getMin(arg.intRange()) << ";\n"
      << "   double eps = d / n;\n"
      << "   for (int i = 0; i < n; ++i) {\n"
      << "      " << obsName << "[0] = " << intVar.getMin(arg.intRange()) << " + eps * i;\n"
      << "      double tmpA = " << funcName << "(params, " << obsName << ", xlArr);\n"
      << "      " << obsName << "[0] = " << intVar.getMin(arg.intRange()) << " + eps * (i + 1);\n"
      << "      double tmpB = " << funcName << "(params, " << obsName << ", xlArr);\n"
      << "      " << resName << " += (tmpA + tmpB) * 0.5 * eps;\n"
      << "   }\n"
      << "}\n";

   ctx.addToGlobalScope(ss.str());

   ctx.addResult(&intVar, oldIntVarResult);
}

void codegenImpl(RooRealSumFunc &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, realSumPdfTranslateImpl(ctx, arg, arg.funcList(), arg.coefList(), false));
}

void codegenImpl(RooRealSumPdf &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, realSumPdfTranslateImpl(ctx, arg, arg.funcList(), arg.coefList(), false));
}

void codegenImpl(RooRealVar &arg, CodegenContext &ctx)
{
   if (!arg.isConstant()) {
      ctx.addResult(&arg, arg.GetName());
   }
   // Just return a formatted version of the const value.
   // Formats to the maximum precision.
   constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
   std::stringstream ss;
   ss.precision(max_precision);
   // Just use toString to make sure we do not output 'inf'.
   // This is really ugly for large numbers...
   ss << std::fixed << RooNumber::toString(arg.getVal());
   ctx.addResult(&arg, ss.str());
}

void codegenImpl(RooRecursiveFraction &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("recursiveFraction"), arg.variables(), arg.variables().size()));
}

void codegenImpl(RooStats::HistFactory::FlexibleInterpVar &arg, CodegenContext &ctx)
{
   auto const &interpCodes = arg.interpolationCodes();

   unsigned int n = interpCodes.size();

   int interpCode = interpCodes[0];
   // To get consistent codes with the PiecewiseInterpolation
   if (interpCode == 4) {
      interpCode = 5;
   }

   for (unsigned int i = 1; i < n; i++) {
      if (interpCodes[i] != interpCodes[0]) {
         oocoutE(&arg, InputArguments)
            << "FlexibleInterpVar::evaluate ERROR:  Code Squashing AD does not yet support having "
               "different interpolation codes for the same class object "
            << std::endl;
      }
   }

   std::string const &resName = ctx.buildCall(mathFunc("flexibleInterp"), interpCode, arg.variables(), n, arg.low(),
                                              arg.high(), arg.globalBoundary(), arg.nominal(), 1.0);
   ctx.addResult(&arg, resName);
}

void codegenImpl(RooUniform &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, "1.0");
}

void codegenImpl(RooWrapperPdf &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.getResult(arg.function()));
}

////////////////////////////////////////////////////////////////////////////////
/// This function defines the analytical integral translation for the class.
///
/// \param[in] code The code that decides the integrands.
/// \param[in] rangeName Name of the normalization range.
/// \param[in] ctx An object to manage auxiliary information for code-squashing.
///
/// \returns The representative code string of the integral for the given object.
std::string codegenIntegralImpl(RooAbsReal &arg, int, const char *, CodegenContext &)
{
   std::stringstream errorMsg;
   errorMsg << "An analytical integral function for class \"" << arg.ClassName() << "\" has not yet been implemented.";
   oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
   throw std::runtime_error(errorMsg.str().c_str());
}

std::string codegenIntegralImpl(RooBernstein &arg, int, const char *rangeName, CodegenContext &ctx)
{
   arg.fillBuffer(); // to get the right xmin() and xmax()
   auto &x = dynamic_cast<RooAbsRealLValue const &>(arg.x());
   return ctx.buildCall(mathFunc("bernsteinIntegral"), x.getMin(rangeName), x.getMax(rangeName), arg.xmin(), arg.xmax(),
                        arg.coefList(), arg.coefList().size());
}

std::string codegenIntegralImpl(RooBifurGauss &arg, int code, const char *rangeName, CodegenContext &ctx)
{
   auto &constant = code == 1 ? arg.getMean() : arg.getX();
   auto &integrand = dynamic_cast<RooAbsRealLValue const &>(code == 1 ? arg.getX() : arg.getMean());

   return ctx.buildCall(mathFunc("bifurGaussIntegral"), integrand.getMin(rangeName), integrand.getMax(rangeName),
                        constant, arg.getSigmaL(), arg.getSigmaR());
}

std::string codegenIntegralImpl(RooCBShape &arg, int /*code*/, const char *rangeName, CodegenContext &ctx)
{
   auto &m = dynamic_cast<RooAbsRealLValue const &>(arg.getM());
   return ctx.buildCall(mathFunc("cbShapeIntegral"), m.getMin(rangeName), m.getMax(rangeName), arg.getM0(),
                        arg.getSigma(), arg.getAlpha(), arg.getN());
}

std::string codegenIntegralImpl(RooChebychev &arg, int, const char *rangeName, CodegenContext &ctx)
{
   auto &x = dynamic_cast<RooAbsRealLValue const &>(arg.x());
   double xmax = x.getMax(arg.refRangeName());
   double xmin = x.getMin(arg.refRangeName());
   unsigned int sz = arg.coefList().size();

   return ctx.buildCall(mathFunc("chebychevIntegral"), arg.coefList(), sz, xmin, xmax, x.getMin(rangeName),
                        x.getMax(rangeName));
}

std::string codegenIntegralImpl(RooEfficiency &, int, const char *, CodegenContext &)
{
   return "1.0";
}

std::string codegenIntegralImpl(RooExponential &arg, int code, const char *rangeName, CodegenContext &ctx)
{
   bool isOverX = code == 1;

   std::string constant;
   if (arg.negateCoefficient() && isOverX) {
      constant += "-";
   }
   constant += ctx.getResult(isOverX ? arg.coefficient() : arg.variable());

   auto &integrand = dynamic_cast<RooAbsRealLValue const &>(isOverX ? arg.variable() : arg.coefficient());

   double min = integrand.getMin(rangeName);
   double max = integrand.getMax(rangeName);

   if (!isOverX && arg.negateCoefficient()) {
      std::swap(min, max);
      min = -min;
      max = -max;
   }

   return ctx.buildCall(mathFunc("exponentialIntegral"), min, max, constant);
}

std::string codegenIntegralImpl(RooGamma &arg, int, const char *rangeName, CodegenContext &ctx)
{
   auto &x = dynamic_cast<RooAbsRealLValue const &>(arg.getX());
   const std::string a =
      ctx.buildCall("ROOT::Math::gamma_cdf", x.getMax(rangeName), arg.getGamma(), arg.getBeta(), arg.getMu());
   const std::string b =
      ctx.buildCall("ROOT::Math::gamma_cdf", x.getMin(rangeName), arg.getGamma(), arg.getBeta(), arg.getMu());
   return a + " - " + b;
}

std::string codegenIntegralImpl(RooGaussian &arg, int code, const char *rangeName, CodegenContext &ctx)
{
   auto &constant = code == 1 ? arg.getMean() : arg.getX();
   auto &integrand = dynamic_cast<RooAbsRealLValue const &>(code == 1 ? arg.getX() : arg.getMean());

   return ctx.buildCall(mathFunc("gaussianIntegral"), integrand.getMin(rangeName), integrand.getMax(rangeName),
                        constant, arg.getSigma());
}

namespace {

std::string rooHistIntegralTranslateImpl(int code, RooAbsArg const &arg, RooDataHist const &dataHist,
                                         const RooArgSet &obs, bool histFuncMode)
{
   if (((2 << obs.size()) - 1) != code) {
      oocoutE(&arg, InputArguments) << "RooHistPdf::integral(" << arg.GetName()
                                    << ") ERROR: AD currently only supports integrating over all histogram observables."
                                    << std::endl;
      return "";
   }
   return std::to_string(dataHist.sum(histFuncMode));
}

} // namespace

std::string codegenIntegralImpl(RooHistFunc &arg, int code, const char *, CodegenContext &)
{
   return rooHistIntegralTranslateImpl(code, arg, arg.dataHist(), arg.variables(), true);
}

std::string codegenIntegralImpl(RooHistPdf &arg, int code, const char *, CodegenContext &)
{
   return rooHistIntegralTranslateImpl(code, arg, arg.dataHist(), arg.variables(), false);
}

std::string codegenIntegralImpl(RooLandau &arg, int, const char *rangeName, CodegenContext &ctx)
{
   // Don't do anything with "code". It can only be "1" anyway (see
   // implementation of getAnalyticalIntegral).
   auto &x = dynamic_cast<RooAbsRealLValue const &>(arg.getX());
   const std::string a = ctx.buildCall("ROOT::Math::landau_cdf", x.getMax(rangeName), arg.getSigma(), arg.getMean());
   const std::string b = ctx.buildCall("ROOT::Math::landau_cdf", x.getMin(rangeName), arg.getSigma(), arg.getMean());
   return ctx.getResult(arg.getSigma()) + " * " + "(" + a + " - " + b + ")";
}

std::string codegenIntegralImpl(RooLognormal &arg, int, const char *rangeName, CodegenContext &ctx)
{
   std::string funcName = arg.useStandardParametrization() ? "logNormalIntegralStandard" : "logNormalIntegral";
   auto &x = dynamic_cast<RooAbsRealLValue const &>(arg.getX());
   return ctx.buildCall(mathFunc(funcName), x.getMin(rangeName), x.getMax(rangeName), arg.getMedian(), arg.getShapeK());
}

std::string codegenIntegralImpl(RooMultiVarGaussian &arg, int code, const char *rangeName, CodegenContext &)
{
   if (code != -1) {
      std::stringstream errorMsg;
      errorMsg << "Partial integrals over RooMultiVarGaussian are not supported.";
      oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   return std::to_string(arg.analyticalIntegral(code, rangeName));
}

std::string codegenIntegralImpl(RooPoisson &arg, int code, const char *rangeName, CodegenContext &ctx)
{
   assert(code == 1 || code == 2);
   std::string xName = ctx.getResult(arg.getX());
   if (!arg.getNoRounding())
      xName = "std::floor(" + xName + ")";

   auto &integrand = dynamic_cast<RooAbsRealLValue const &>(code == 1 ? arg.getX() : arg.getMean());
   // Since the integral function is the same for both codes, we need to make sure the indexed observables do not appear
   // in the function if they are not required.
   xName = code == 1 ? "0" : xName;
   return ctx.buildCall(mathFunc("poissonIntegral"), code, arg.getMean(), xName, integrand.getMin(rangeName),
                        integrand.getMax(rangeName), arg.getProtectNegativeMean());
}

std::string codegenIntegralImpl(RooPolyVar &arg, int, const char *rangeName, CodegenContext &ctx)
{
   auto &x = dynamic_cast<RooAbsRealLValue const &>(arg.x());
   const double xmin = x.getMin(rangeName);
   const double xmax = x.getMax(rangeName);
   const unsigned sz = arg.coefList().size();
   if (!sz)
      return std::to_string(arg.lowestOrder() ? xmax - xmin : 0.0);

   return ctx.buildCall(mathFunc("polynomialIntegral"), arg.coefList(), sz, arg.lowestOrder(), xmin, xmax);
}

std::string codegenIntegralImpl(RooPolynomial &arg, int, const char *rangeName, CodegenContext &ctx)
{
   auto &x = dynamic_cast<RooAbsRealLValue const &>(arg.x());
   const double xmin = x.getMin(rangeName);
   const double xmax = x.getMax(rangeName);
   const unsigned sz = arg.coefList().size();
   if (!sz)
      return std::to_string(arg.lowestOrder() ? xmax - xmin : 0.0);

   return ctx.buildCall(mathFunc("polynomialIntegral<true>"), arg.coefList(), sz, arg.lowestOrder(), xmin, xmax);
}

std::string codegenIntegralImpl(RooRealSumPdf &arg, int code, const char *rangeName, CodegenContext &ctx)
{
   // Re-use translate, since integration is linear.
   return realSumPdfTranslateImpl(ctx, arg, arg.funcIntListFromCache(code, rangeName), arg.coefList(), false);
}

std::string codegenIntegralImpl(RooUniform &arg, int code, const char *rangeName, CodegenContext &)
{
   // The integral of a uniform distribution is static, so we can just hardcode
   // the result in a string.
   return std::to_string(arg.analyticalIntegral(code, rangeName));
}

} // namespace Experimental
} // namespace RooFit
