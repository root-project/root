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
#include <RooFFTConvPdf.h>
#include <RooFit/Detail/RooNLLVarNew.h>
#include <RooFit/Detail/RooNormalizedPdf.h>
#include <RooFormulaVar.h>
#include <RooFunctor1DBinding.h>
#include <RooFunctorBinding.h>
#include <RooGamma.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLandau.h>
#include <RooLognormal.h>
#include <RooMultiPdf.h>
#include <RooMultiVarGaussian.h>
#include <RooONNXFunc.h>
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
#include <RooSimultaneous.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooUniform.h>
#include <RooWrapperPdf.h>

#include "RooFitImplHelpers.h"

#include <TInterpreter.h>

#include <cmath>
#include <locale>
#include <memory>
#include <unordered_set>

namespace RooFit::Experimental {

namespace {

// Return a stringy-field version of the value, formatted to maximum precision.
std::string doubleToString(double val)
{
   std::stringstream ss;
   // The formatting must not depend on the global locale: a comma decimal
   // separator (e.g. from a German locale) would corrupt the generated C++.
   ss.imbue(std::locale::classic());
   ss << std::setprecision(std::numeric_limits<double>::max_digits10) << val;
   return ss.str();
}

std::string mathFunc(std::string const &name)
{
   return "RooFit::Detail::MathFuncs::" + name;
}

/// Name of the variable in the generated code that holds the sum of the cache
/// bins of a RooFFTConvPdf, shared between the code for the pdf value and for
/// its normalization integral. Like other variables named after objects, it
/// assumes that makeValidVarName() maps different names to different strings.
std::string fftConvSumName(RooFFTConvPdf const &arg)
{
   return RooFit::Detail::makeValidVarName(arg.GetName()) + "_fftConvSum";
}

/// Whether all bins of the binning have the same width. RooAbsBinning::isUniform()
/// is only true for RooUniformBinning, but a RooBinning can be uniform too.
bool hasUniformBins(RooAbsBinning const &binning)
{
   if (binning.isUniform()) {
      return true;
   }
   const int nBins = binning.numBins();
   const double width = (binning.highBound() - binning.lowBound()) / nBins;
   for (int i = 0; i < nBins; ++i) {
      if (std::abs(binning.binWidth(i) - width) > 1e-10 * std::abs(width)) {
         return false;
      }
   }
   return true;
}

void rooHistTranslateImpl(RooAbsArg const &arg, CodegenContext &ctx, int intOrder, RooDataHist const &dataHist,
                          const RooArgSet &obs, bool correctForBinSize, bool cdfBoundaries)
{
   if (intOrder != 0) {
      RooAbsBinning const *binning = obs.size() == 1 ? dataHist.getBinnings()[0].get() : nullptr;
      if (cdfBoundaries || !binning || !hasUniformBins(*binning)) {
         std::stringstream errorMsg;
         errorMsg << "RooHistPdf::weight(" << arg.GetName()
                  << ") ERROR: codegen only supports interpolation for one-dimensional histograms with uniform "
                     "binning and without cdf boundaries.";
         ooccoutE(&arg, InputArguments) << errorMsg.str() << std::endl;
         throw std::runtime_error(errorMsg.str());
      }
      std::string weightArr = dataHist.declWeightArrayForCodeSquash(ctx, correctForBinSize);
      ctx.addResult(&arg, ctx.buildCall(mathFunc("interpolate1dPolynomial"), binning->lowBound(), binning->highBound(),
                                        *obs[0], binning->numBins(), weightArr, intOrder));
      return;
   }
   std::string const &offset = dataHist.calculateTreeIndexForCodeSquash(ctx, obs);
   std::string weightArr = dataHist.declWeightArrayForCodeSquash(ctx, correctForBinSize);
   ctx.addResult(&arg, "(" + weightArr + ")[" + offset + "]");
}

std::string
realSumPdfTranslateImpl(CodegenContext &ctx, RooArgList const &funcList, RooArgList const &coefList, bool normalize)
{
   bool noLastCoeff = funcList.size() != coefList.size();

   // Decide the scope from the inputs rather than from the RooRealSumPdf
   // itself: this function is also used for the integral of the sum, whose
   // inputs are the integrals of the components. Those usually don't depend
   // on the data even though the sum does, and then the code must not end up
   // inside the event loop, because the integral is also the expected number
   // of events that the extended likelihood uses after the loop.
   bool scopeIndependent = true;
   for (RooAbsArg const *func : funcList) {
      scopeIndependent = scopeIndependent && ctx.isScopeIndependent(func);
   }
   for (RooAbsArg const *coef : coefList) {
      scopeIndependent = scopeIndependent && ctx.isScopeIndependent(coef);
   }
   auto addCode = [&](std::string const &code) { ctx.addToCodeBody(code, scopeIndependent); };

   std::string const &funcName = ctx.buildArg(funcList);
   std::string const &coeffName = ctx.buildArg(coefList);
   std::string const &coeffSize = std::to_string(coefList.size());

   std::string sum = ctx.getTmpVarName();
   std::string coeffSum = ctx.getTmpVarName();
   addCode("double " + sum + " = 0;\ndouble " + coeffSum + "= 0;\n");

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
   addCode(code);

   return sum;
}

} // namespace

void codegenImpl(RooFit::Detail::RooFixedProdPdf &arg, CodegenContext &ctx)
{
   if (arg.isRearranged()) {
      ctx.addResult(&arg, ctx.buildCall(mathFunc("ratio"), *arg.rearrangedNum(), *arg.rearrangedDen()));
   } else {
      ctx.addResult(&arg, ctx.buildCall(mathFunc("product"), *arg.partList(), arg.partList()->size()));
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
   ctx.addResult(&arg, realSumPdfTranslateImpl(ctx, arg.pdfList(), arg.coefList(), true));
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

   // The value of this number should be discussed. Beyound a certain number of
   // indices MathFunc call becomes more efficient.
   if (numPdfs > 2) {
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

void codegenImpl(RooSimultaneous &arg, CodegenContext &ctx)
{
   // A RooSimultaneous appears as a node in the compute graph only when its
   // index category is not an observable (see the "switch mode" in
   // RooSimultaneous::compileForNormSet()): its value is then the one of the
   // component selected by the current category state, like for RooMultiPdf.
   // With an observable index category, the likelihood is decomposed into
   // per-channel terms and the RooSimultaneous itself is never translated.
   if (!arg.indexCat().isFundamental()) {
      std::stringstream errorMsg;
      errorMsg << "RooSimultaneous \"" << arg.GetName()
               << "\" with a derived, non-observable index category can't be translated.";
      oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
   }

   std::string const &indexExpr = ctx.getResult(arg.indexCat());

   // Nested ternary expression that selects the pdf matching the category
   // state, keyed on the state index numbers.
   std::string expr;
   std::size_t numStates = 0;
   for (auto const &nameIdx : arg.indexCat()) {
      RooAbsPdf *pdf = arg.getPdf(nameIdx.first.c_str());
      if (!pdf) {
         continue;
      }
      expr += "(" + indexExpr + " == " + std::to_string(nameIdx.second) + " ? (" + ctx.getResult(*pdf) + ") : ";
      ++numStates;
   }
   expr += "0.0" + std::string(numStates, ')');

   ctx.addResult(&arg, expr);
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
      result += ctx.buildFunction(*component, ctx.dependsOnData()) + "(params, obs, xlArr)";
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
   ctx.addResult(&arg, doubleToString(arg.getVal()));
}

void codegenImpl(RooConstraintSum &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("constraintSum"), arg.list(), arg.list().size()));
}

// Generate RooFit codegen wrappers for RooFunctorBinding and similar objects,
// emitting both the primal function call and its gradient pullback for
// Clad-based AD.
template <class RooArg_t>
void functorCodegenImpl(RooArg_t &arg, RooArgList const &variables, CodegenContext &ctx)
{
   if (!arg.function()->HasGradient()) {
      std::stringstream errorMsg;
      errorMsg << "Functor wrapped by \"" << arg.GetName() << "\" doesn't provide a gradient function."
               << " RooFit codegen is therefore not supported.";
      oocoutE(&arg, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
   }

   std::string funcAddrStr = TString::Format("0x%zx", reinterpret_cast<std::size_t>(arg.function())).Data();
   std::string wrapperName = "roo_functor_" + funcAddrStr;

   static std::unordered_set<std::string> wrapperNames;

   if (wrapperNames.find(wrapperName) == wrapperNames.end()) {

      wrapperNames.insert(wrapperName);

      std::string pullbackName = wrapperName + "_pullback";
      std::string nStr = std::to_string(std::size(variables));

      std::string type;
      if constexpr (std::is_same_v<RooArg_t, RooFunctor1DBinding> || std::is_same_v<RooArg_t, RooFunctor1DPdfBinding>)
         type = "::ROOT::Math::IGradientFunctionOneDim";
      else
         type = "::ROOT::Math::IGradientFunctionMultiDim";

      std::string funcAddrCasted = "reinterpret_cast<" + type + " const *>(" + funcAddrStr + ")";

      std::string code;

      code += "double " + wrapperName +
              "(double const *x) {\n"
              "   return " +
              funcAddrCasted +
              "->operator()(x);\n"
              "}\n\n"
              "namespace clad::custom_derivatives {\n\n"
              "void " +
              pullbackName +
              "(double const* x, double d_y, double *d_x) {\n"
              "   double output[" +
              nStr +
              "]{};\n"
              "   " +
              funcAddrCasted +
              "->Gradient(x, output);\n"
              "   for (int i = 0; i < " +
              nStr +
              "; ++i) {\n"
              "      d_x[i] += output[i] * d_y;\n"
              "   }\n"
              "}\n"
              "} // namespace clad::custom_derivatives\n";

      gInterpreter->Declare(code.c_str());
   }

   ctx.addResult(&arg, ctx.buildCall(wrapperName, variables));
}

void codegenImpl(RooFunctor1DBinding &arg, CodegenContext &ctx)
{
   functorCodegenImpl(arg, arg.variable(), ctx);
}

void codegenImpl(RooFunctor1DPdfBinding &arg, CodegenContext &ctx)
{
   functorCodegenImpl(arg, arg.variable(), ctx);
}

void codegenImpl(RooFunctorBinding &arg, CodegenContext &ctx)
{
   functorCodegenImpl(arg, arg.variables(), ctx);
}

void codegenImpl(RooFunctorPdfBinding &arg, CodegenContext &ctx)
{
   functorCodegenImpl(arg, arg.variables(), ctx);
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
   // We have to force the array type to be "double" because that's what the
   // declared function wrapped by the TFormula expects.
   auto inputVar = ctx.buildArg(arg.dependents(), /*arrayType=*/"double");
   ctx.addResult(&arg, funcName + "(" + inputVar + ")");
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

void codegenImpl(RooFFTConvPdf &arg, CodegenContext &ctx)
{
   // Like RooFFTConvPdf::fillCacheSlice(), the generated code samples both
   // input pdfs on the extended grid, convolves the samples and reads out the
   // cache bins, which it then interpolates at the observable value. All but
   // the interpolation only depends on the parameters and is placed outside
   // of the loop over the data.

   if (arg.getPdfConvVar()) {
      std::stringstream errorMsg;
      errorMsg << "RooFFTConvPdf \"" << arg.GetName()
               << "\": a convolution observable that is a function of the actual observable (pdfConvVar) is not "
                  "supported by codegen.";
      oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
   }

   RooAbsReal const &convVar = arg.getConvVar();

   // The convolution would have to be redone for each value of any other
   // observable of the input pdfs (RooFFTConvPdf caches slices for them).
   for (RooAbsReal const *pdf : {&arg.getPdf1(), &arg.getPdf2()}) {
      std::unique_ptr<RooArgSet> vars{pdf->getVariables()};
      for (RooAbsArg const *var : *vars) {
         if (var->namePtr() != convVar.namePtr() && ctx.observableIndexOf(*var) >= 0) {
            std::stringstream errorMsg;
            errorMsg << "RooFFTConvPdf \"" << arg.GetName() << "\": the input pdf \"" << pdf->GetName()
                     << "\" depends on the observable \"" << var->GetName()
                     << "\" besides the convolution observable, which is not supported by codegen.";
            oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
            throw std::runtime_error(errorMsg.str());
         }
      }
   }

   RooFFTConvPdf::SamplingGrid const grid = arg.samplingGrid();
   const int nBins = grid.numBins;
   const int nBuf = grid.numBufferBins;
   const int nSamples = grid.numSamples();

   // The sampled arrays live on the stack of the generated function, and so do
   // their adjoints in the derivatives that clad generates from it.
   constexpr int maxSamples = 32768;
   if (nSamples > maxSamples) {
      std::stringstream errorMsg;
      errorMsg << "RooFFTConvPdf \"" << arg.GetName() << "\": the " << nBins << " bins of the cache binning of \""
               << convVar.GetName() << "\" give " << nSamples << " samples including the buffers, more than the "
               << maxSamples << " supported by codegen. Reduce the number of bins or use another evaluation backend.";
      oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
   }

   // Generate the functions that evaluate the input pdfs, with the convolution
   // observable as their observable argument. The inputs were compiled for the
   // observables of the cache (see RooFFTConvPdf::compileServersNormSet()),
   // but their overall normalization is a constant factor that cancels, so a
   // RooNormalizedPdf wrapper is bypassed instead of evaluating its integral
   // at every grid point.
   auto sampledInput = [](RooAbsReal const &pdf) -> RooAbsReal const & {
      if (auto *normalized = dynamic_cast<RooFit::Detail::RooNormalizedPdf const *>(&pdf)) {
         return normalized->pdf();
      }
      return pdf;
   };

   std::string const convVarResult = ctx.getResult(convVar);
   ctx.addResult(convVar.GetName(), "obs[0]");
   std::string const func1 = ctx.buildFunction(sampledInput(arg.getPdf1()), {});
   std::string const func2 = ctx.buildFunction(sampledInput(arg.getPdf2()), {});
   ctx.addResult(convVar.GetName(), convVarResult);

   std::string const in1 = ctx.getTmpVarName();
   std::string const in2 = ctx.getTmpVarName();
   std::string const obsArr = ctx.getTmpVarName();
   std::string const conv = ctx.getTmpVarName();
   std::string const cache = ctx.getTmpVarName();
   std::string const sum = fftConvSumName(arg);
   std::string const idx = "i_" + ctx.getTmpVarName();

   // The samples are taken at the bin centers, shifted for each pdf (see
   // RooFFTConvPdf::setShift()). Flat and Mirror only sample the observable
   // range and fill the buffer bins from it, like RooFFTConvPdf::scanPdf().
   auto sampleAt = [&](std::string const &lowStr, std::string const &index) {
      std::stringstream ss;
      ss.imbue(std::locale::classic()); // the generated code is C++, not locale-dependent text
      ss << "   " << obsArr << "[0] = " << lowStr << " + (" << idx << " + 0.5) * " << doubleToString(grid.binWidth)
         << " - " << doubleToString(grid.shift1) << ";\n"
         << "   " << in1 << "[" << index << "] = " << func1 << "(params, " << obsArr << ", xlArr);\n"
         << "   " << obsArr << "[0] = " << lowStr << " + (" << idx << " + 0.5) * " << doubleToString(grid.binWidth)
         << " - " << doubleToString(grid.shift2) << ";\n"
         << "   " << in2 << "[" << index << "] = " << func2 << "(params, " << obsArr << ", xlArr);\n";
      return ss.str();
   };

   std::stringstream code;
   code.imbue(std::locale::classic()); // the generated code is C++, not locale-dependent text
   code << "double " << in1 << "[" << nSamples << "];\n"
        << "double " << in2 << "[" << nSamples << "];\n"
        << "double " << obsArr << "[1];\n";

   if (arg.bufferStrategy() == RooFFTConvPdf::Extend) {
      code << "#pragma clad checkpoint loop\n"
           << "for (int " << idx << " = 0; " << idx << " < " << nSamples << "; ++" << idx << ") {\n"
           << sampleAt(doubleToString(grid.lowExt()), idx) << "}\n";
   } else {
      std::string const inner = std::to_string(nBuf) + " + " + idx;
      code << "#pragma clad checkpoint loop\n"
           << "for (int " << idx << " = 0; " << idx << " < " << nBins << "; ++" << idx << ") {\n"
           << sampleAt(doubleToString(grid.low), inner) << "}\n";
      if (arg.bufferStrategy() == RooFFTConvPdf::Flat) {
         // Buffers hold the values at the boundaries of the observable range
         for (std::string const &in : {in1, in2}) {
            code << "for (int " << idx << " = 0; " << idx << " < " << nBuf << "; ++" << idx << ") {\n"
                 << "   " << in << "[" << idx << "] = " << in << "[" << nBuf << "];\n"
                 << "   " << in << "[" << nBuf + nBins << " + " << idx << "] = " << in << "[" << nBuf + nBins - 1
                 << "];\n"
                 << "}\n";
         }
      } else {
         // Buffers hold the mirror image of the sampled range
         for (std::string const &in : {in1, in2}) {
            code << "for (int " << idx << " = 1; " << idx << " <= " << nBuf << "; ++" << idx << ") {\n"
                 << "   " << in << "[" << nBuf << " - " << idx << "] = " << in << "[" << nBuf << " + " << idx << "];\n"
                 << "   " << in << "[" << nBuf + nBins - 1 << " + " << idx << "] = " << in << "[" << nBuf + nBins
                 << " - " << idx << "];\n"
                 << "}\n";
         }
      }
   }

   // Convolve and read out the cache bins, clamping round-off negatives like
   // RooFFTConvPdf does. The sum of the cache bins is the normalization
   // integral, see codegenIntegralImpl() for RooFFTConvPdf.
   code << "double " << conv << "[" << nSamples << "] = {0.};\n"
        << mathFunc("fftConvolveAdd") << "(" << nSamples << ", " << in1 << ", " << in2 << ", " << conv << ");\n"
        << "double " << cache << "[" << nBins << "];\n"
        << "double " << sum << " = 0.;\n"
        << "for (int " << idx << " = 0; " << idx << " < " << nBins << "; ++" << idx << ") {\n"
        << "   const double val = " << conv << "[(" << idx << " + " << grid.outputOffset << ") % " << nSamples << "];\n"
        << "   " << cache << "[" << idx << "] = val < 0. ? 0. : val;\n"
        << "   " << sum << " += " << cache << "[" << idx << "];\n"
        << "}\n";

   ctx.addToCodeBody(code.str(), /*isScopeIndep=*/true);

   // Interpolate the cache at the observable value, with the bin size
   // correction and the clamping at zero of RooHistPdf::evaluate().
   std::string const interp = ctx.buildCall(mathFunc("interpolate1dPolynomial"), grid.low, grid.high, convVar, nBins,
                                            cache, arg.getInterpolationOrder());
   std::string const result = ctx.getTmpVarName();
   ctx.addToCodeBody(&arg, "double " + result + " = " + interp + " / " + doubleToString(grid.binWidth) + ";\n" +
                              result + " = " + result + " < 0. ? 0. : " + result + ";\n");
   ctx.addResult(&arg, result);
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
   // We have to force the array type to be "double" because that's what the
   // declared function wrapped by the TFormula expects.
   auto inputVar = ctx.buildArg(arg.dependents(), /*arrayType=*/"double");
   ctx.addResult(&arg, funcName + "(" + inputVar + ")");
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
   if (arg.getInterpolationOrder() > 1) {
      // The interpolating polynomial can undershoot zero between the bins,
      // which RooHistPdf::evaluate() clamps.
      std::string const &res = ctx.getResult(arg);
      ctx.addResult(&arg, res + " < 0. ? 0. : " + res);
   }
}

void codegenImpl(RooLandau &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, ctx.buildCall(mathFunc("landau"), arg.getX(), arg.getMean(), arg.getSigma()));
}

void codegenImpl(RooLognormal &arg, CodegenContext &ctx)
{
   std::string funcName = arg.useStandardParametrization() ? "logNormalStandard" : "logNormal";
   ctx.addResult(&arg, ctx.buildCall(mathFunc(funcName), arg.getX(), arg.getShapeK(), arg.getMedian()));
}

namespace {

void codegenChi2(RooFit::Detail::RooNLLVarNew &arg, CodegenContext &ctx)
{
   using FuncMode = RooFit::Detail::RooNLLVarNew::FuncMode;

   std::string resName = RooFit::Detail::makeValidVarName(arg.GetName()) + "Result";
   ctx.addResult(&arg, resName);
   ctx.addToGlobalScope("double " + resName + " = 0.0;\n");

   // DataError::None means "no errors": every bin contributes zero.
   if (arg.chi2ErrorType() == RooDataHist::None) {
      return;
   }

   // Compute the per-bin normalization factor (constant with respect to the
   // loop).
   std::string normFactor;
   if (arg.funcMode() == FuncMode::Function) {
      normFactor = "1.0";
   } else if (arg.funcMode() == FuncMode::ExtendedPdf) {
      normFactor = ctx.getResult(*arg.expectedEvents());
   } else { // Pdf
      std::string weightSumName = RooFit::Detail::makeValidVarName(arg.GetName()) + "WeightSum";
      ctx.addToGlobalScope("double " + weightSumName + " = 0.0;\n");
      {
         auto scope = ctx.beginLoop(&arg);
         ctx.addToCodeBody(weightSumName + " += " + ctx.getResult(arg.weightVar()) + ";\n");
      }
      normFactor = weightSumName;
   }

   auto scope = ctx.beginLoop(&arg);
   const std::string mu =
      "(" + ctx.getResult(arg.func()) + " * " + ctx.getResult(*arg.binVolumes()) + " * " + normFactor + ")";

   std::string term;
   switch (arg.chi2ErrorType()) {
   case RooDataHist::Expected: term = ctx.buildCall(mathFunc("chi2Expected"), mu, arg.weightVar()); break;
   case RooDataHist::SumW2:
      term = ctx.buildCall(mathFunc("chi2Symmetric"), mu, arg.weightVar(), arg.weightSquaredVar());
      break;
   case RooDataHist::Poisson:
      term = ctx.buildCall(mathFunc("chi2Asymmetric"), mu, arg.weightVar(), *arg.weightErrLo(), *arg.weightErrHi());
      break;
   default: break;
   }
   ctx.addToCodeBody(&arg, resName + " += " + term + ";");
}

} // namespace

void codegenImpl(RooFit::Detail::RooNLLVarNew &arg, CodegenContext &ctx)
{
   if (arg.statistic() == RooFit::Detail::RooNLLVarNew::Statistic::Chi2) {
      return codegenChi2(arg, ctx);
   }

   if (arg.binnedL() && !arg.func().getAttribute("BinnedLikelihoodActiveYields")) {
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
      std::string term = ctx.buildCall(mathFunc("nll"), arg.func(), arg.weightVar(), arg.binnedL(), 0);
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

// Support for the deprecated RooParamHistFunc, to be removed in ROOT 6.46
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
void codegenImpl(RooParamHistFunc &arg, CodegenContext &ctx)
{
   std::string const &idx = arg.dataHist().calculateTreeIndexForCodeSquash(ctx, arg.xList());
   std::string arrName = ctx.buildArg(arg.paramList());
   std::stringstream result;
   result << arrName << "[" << idx << "]";
   if (arg.relParam()) {
      // get weight[idx] * binv[idx]. Here we get the bin volume for the first element as we assume the distribution to
      // be binned uniformly.
      double binV = arg.dataHist().binVolume(0);
      std::string weightArr = arg.dataHist().declWeightArrayForCodeSquash(ctx, false);
      result << " * *(" << weightArr << " + " << idx + ") * " << doubleToString(binV);
   }
   ctx.addResult(&arg, result.str());
}
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif

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

   if (dynamic_cast<RooFFTConvPdf const *>(&arg.integrand())) {
      // Sub-range integrals of an interpolated cache end up here, where the
      // integrand would redo the FFT at every integration point.
      std::stringstream errorMsg;
      errorMsg << "RooFFTConvPdf \"" << arg.integrand().GetName()
               << "\": only the integral over the full range of the convolution observable is supported by codegen.";
      oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
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
      << "   #pragma clad checkpoint loop\n"
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
   ctx.addResult(&arg, realSumPdfTranslateImpl(ctx, arg.funcList(), arg.coefList(), false));
}

void codegenImpl(RooRealSumPdf &arg, CodegenContext &ctx)
{
   ctx.addResult(&arg, realSumPdfTranslateImpl(ctx, arg.funcList(), arg.coefList(), false));
}

void codegenImpl(RooRealVar &arg, CodegenContext &ctx)
{
   if (!arg.isConstant()) {
      ctx.addResult(&arg, arg.GetName());
   }
   ctx.addResult(&arg, doubleToString(arg.getVal()));
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

std::string codegenIntegralImpl(RooFFTConvPdf &arg, int /*code*/, const char *rangeName, CodegenContext &ctx)
{
   // Like for RooHistPdf, the integral over the full range is the sum of the
   // cache bins, which the code of the convolution computes. Sub-range
   // integrals would be partial sums, which are not supported.
   auto const &convVar = static_cast<RooAbsRealLValue const &>(arg.getConvVar());
   RooFFTConvPdf::SamplingGrid const grid = arg.samplingGrid();
   if (convVar.getMin(rangeName) != grid.low || convVar.getMax(rangeName) != grid.high) {
      std::stringstream errorMsg;
      errorMsg << "RooFFTConvPdf \"" << arg.GetName()
               << "\": only the integral over the full range of the convolution observable is supported by codegen.";
      oocoutE(&arg, Minimization) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str());
   }
   // Make sure the convolution code, which includes the sum, is generated
   ctx.getResult(arg);
   return fftConvSumName(arg);
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
   return doubleToString(dataHist.sum(histFuncMode));
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

   return doubleToString(arg.analyticalIntegral(code, rangeName));
}

void codegenImpl(RooONNXFunc &arg, CodegenContext &ctx)
{
   std::stringstream ss;
   ss << arg.outerWrapperName() << "(";
   for (std::size_t i = 0; i < arg.nInputTensors(); ++i) {
      ss << ctx.buildArg(arg.inputTensorList(i)) << std::endl;
      if (i != arg.nInputTensors() - 1) {
         ss << ", ";
      }
   }
   ss << ")";

   ctx.addResult(&arg, ss.str());
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
   return realSumPdfTranslateImpl(ctx, arg.funcIntListFromCache(code, rangeName), arg.coefList(), false);
}

std::string codegenIntegralImpl(RooUniform &arg, int code, const char *rangeName, CodegenContext &)
{
   // The integral of a uniform distribution is static, so we can just hardcode
   // the result in a string.
   return doubleToString(arg.analyticalIntegral(code, rangeName));
}

} // namespace RooFit::Experimental
