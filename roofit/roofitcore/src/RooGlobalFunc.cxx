/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// Global helper functions

#include <RooGlobalFunc.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooBatchCompute.h>
#include <RooCategory.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooNumIntConfig.h>
#include <RooRealConstant.h>
#include <RooRealVar.h>

#include <TH1.h>
#include <TInterpreter.h>

#include <algorithm>

namespace RooFit {

/// Get the global choice for the RooBatchCompute library that RooFit will load.
/// \see RooFit::setBatchCompute().
std::string getBatchCompute()
{
   return RooBatchCompute::getBatchComputeChoice();
}

/// Globally select the RooBatchCompute CPU implementation that will be loaded
/// in RooFit.
/// Supported options are "auto" (default), "avx512", "avx2", "avx", "sse", "generic".
/// \note It is not possible to change the selection after RooFit has already
/// loaded a library (which is usually triggered by likelihood creation or fitting).
void setBatchCompute(std::string const &value)
{
   return RooBatchCompute::setBatchComputeChoice(value);
}

// anonymous namespace for helper functions for the implementation of the global functions
namespace {

template <class T>
RooCmdArg processImportItem(std::string const &key, T *val)
{
   return Import(key.c_str(), *val);
}

template <class T>
RooCmdArg processLinkItem(std::string const &key, T *val)
{
   return Link(key.c_str(), *val);
}

RooCmdArg processSliceItem(RooCategory *key, std::string const &val)
{
   return Slice(*key, val.c_str());
}

template <class Key_t, class Val_t, class Func_t>
RooCmdArg processMap(const char *name, Func_t func, std::map<Key_t, Val_t> const &map)
{
   RooCmdArg container(name, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
   for (auto const &item : map) {
      container.addArg(func(item.first, item.second));
   }
   container.setProcessRecArgs(true, false);
   return container;
}

template <class Key_t, class Val_t, class Func_t>
RooCmdArg processFlatMap(const char *name, Func_t func, Detail::FlatMap<Key_t, Val_t> const &map)
{
   RooCmdArg container(name, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
   for (std::size_t i = 0; i < map.keys.size(); ++i) {
      container.addArg(func(map.keys[i], map.vals[i]));
   }
   container.setProcessRecArgs(true, false);
   return container;
}

int interpretString(std::string const &s)
{
   return gInterpreter->ProcessLine(s.c_str());
}

Style_t interpretLineStyleString(std::string const &style)
{
   using Map = std::unordered_map<std::string, Style_t>;
   // Style dictionary to define matplotlib conventions
   static Map styleMap{{"-", kSolid}, {"--", kDashed}, {":", kDotted}, {"-.", kDashDotted}};
   auto found = styleMap.find(style);
   if (found != styleMap.end())
      return found->second;
   // Use interpreter if style was not matched in the style map
   return gInterpreter->ProcessLine(style.c_str());
}

} // namespace

namespace Experimental {

RooCmdArg ParallelGradientOptions(bool enable, int orderStrategy, int chainFactor)
{
   return RooCmdArg("ParallelGradientOptions", enable, chainFactor, orderStrategy, 0, nullptr, nullptr, nullptr,
                    nullptr);
}
RooCmdArg ParallelDescentOptions(bool enable, int splitStrategy, int numSplits)
{
   return RooCmdArg("ParallelDescentOptions", enable, numSplits, splitStrategy, 0, nullptr, nullptr, nullptr, nullptr);
}

} // namespace Experimental

// RooAbsReal::plotOn arguments
RooCmdArg DrawOption(const char *opt)
{
   return RooCmdArg("DrawOption", 0, 0, 0, 0, opt, nullptr, nullptr, nullptr);
}
RooCmdArg Slice(const RooArgSet &sliceSet)
{
   RooCmdArg out{"SliceVars", 0};
   out.setSet(0, sliceSet);
   return out;
}
RooCmdArg Slice(RooCategory &cat, const char *label)
{
   // We don't support adding multiple slices for a single category by
   // concatenating labels with a comma. Users were trying to do that, and were
   // surprised it did not work. So we explicitly check if there is a comma,
   // and if there is, we will give some helpful advice on how to get to the
   // desired plot.
   std::string lbl{label};
   if (lbl.find(',') != std::string::npos) {
      std::stringstream errorMsg;
      errorMsg << "RooFit::Slice(): you tried to pass a comma-separated list of state labels \"" << label
               << "\" for a given category, but selecting multiple slices like this is not supported!"
               << " If you want to make a plot of multiple slices, use the ProjWData() command where you pass a "
                  "dataset that includes "
                  "the desired slices. If the slices are a subset of all slices, then you can create such a dataset "
                  "with RooAbsData::reduce(RooFit::Cut(\"cat==cat::label_1 || cat==cat::label_2 || ...\")). You can "
                  "find some examples in the rf501_simultaneouspdf tutorial.";
      oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
      throw std::invalid_argument(errorMsg.str().c_str());
   }
   return RooCmdArg("SliceCat", 0, 0, 0, 0, label, nullptr, &cat, nullptr);
}
RooCmdArg Slice(std::map<RooCategory *, std::string> const &arg)
{
   return processMap("SliceCatMany", processSliceItem, arg);
}

RooCmdArg Project(const RooArgSet &projSet)
{
   RooCmdArg out{"Project", 0};
   out.setSet(0, projSet);
   return out;
}
RooCmdArg ProjWData(const RooArgSet &projSet, const RooAbsData &projData, bool binData)
{
   RooCmdArg out{"ProjData", binData, 0, 0, 0, nullptr, nullptr, nullptr, &projData};
   out.setSet(0, projSet);
   return out;
}
RooCmdArg ProjWData(const RooAbsData &projData, bool binData)
{
   return RooCmdArg("ProjData", binData, 0, 0, 0, nullptr, nullptr, nullptr, &projData);
}
RooCmdArg Asymmetry(const RooCategory &cat)
{
   return RooCmdArg("Asymmetry", 0, 0, 0, 0, nullptr, nullptr, &cat, nullptr);
}
RooCmdArg Precision(double prec)
{
   return RooCmdArg("Precision", 0, 0, prec, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ShiftToZero()
{
   return RooCmdArg("ShiftToZero", 1);
}
RooCmdArg Normalization(double scaleFactor)
{
   return RooCmdArg("Normalization", RooAbsReal::Relative, 0, scaleFactor, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Range(const char *rangeName, bool adjustNorm)
{
   return RooCmdArg("RangeWithName", adjustNorm, 0, 0, 0, rangeName, nullptr, nullptr, nullptr);
}
RooCmdArg Range(double lo, double hi, bool adjustNorm)
{
   return RooCmdArg("Range", adjustNorm, 0, lo, hi, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg NormRange(const char *rangeNameList)
{
   return RooCmdArg("NormRange", 0, 0, 0, 0, rangeNameList, nullptr, nullptr, nullptr);
}
RooCmdArg VLines()
{
   return RooCmdArg("VLines", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg LineColor(TColorNumber color)
{
   return RooCmdArg("LineColor", color.number());
}
RooCmdArg LineStyle(Style_t style)
{
   return RooCmdArg("LineStyle", style);
}
RooCmdArg LineStyle(std::string const &style)
{
   return LineStyle(interpretLineStyleString(style));
}
RooCmdArg LineWidth(Width_t width)
{
   return RooCmdArg("LineWidth", width);
}
RooCmdArg FillColor(TColorNumber color)
{
   return RooCmdArg("FillColor", color.number());
}
RooCmdArg FillStyle(Style_t style)
{
   return RooCmdArg("FillStyle", style);
}
RooCmdArg FillStyle(std::string const &style)
{
   return FillStyle(interpretString(style));
}
RooCmdArg ProjectionRange(const char *rangeName)
{
   return RooCmdArg("ProjectionRange", 0, 0, 0, 0, rangeName, nullptr, nullptr, nullptr);
}
RooCmdArg Name(const char *name)
{
   return RooCmdArg("Name", 0, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg Invisible(bool inv)
{
   return RooCmdArg("Invisible", inv, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg AddTo(const char *name, double wgtSel, double wgtOther)
{
   return RooCmdArg("AddTo", 0, 0, wgtSel, wgtOther, name, nullptr, nullptr, nullptr);
}
RooCmdArg EvalErrorValue(double val)
{
   return RooCmdArg("EvalErrorValue", 1, 0, val, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg MoveToBack()
{
   return RooCmdArg("MoveToBack", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg VisualizeError(const RooFitResult &fitres, double Z, bool EVmethod)
{
   return RooCmdArg("VisualizeError", EVmethod, 0, Z, 0, nullptr, nullptr, &fitres, nullptr);
}
RooCmdArg VisualizeError(const RooFitResult &fitres, const RooArgSet &param, double Z, bool EVmethod)
{
   return RooCmdArg("VisualizeError", EVmethod, 0, Z, 0, nullptr, nullptr, &fitres, nullptr, nullptr, nullptr, &param);
}
RooCmdArg VisualizeError(const RooDataSet &paramData, double Z)
{
   return RooCmdArg("VisualizeErrorData", 0, 0, Z, 0, nullptr, nullptr, &paramData, nullptr);
}
RooCmdArg ShowProgress()
{
   return RooCmdArg("ShowProgress", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooAbsPdf::plotOn arguments
RooCmdArg Components(const char *compSpec)
{
   return RooCmdArg("SelectCompSpec", 0, 0, 0, 0, compSpec, nullptr, nullptr, nullptr);
}
RooCmdArg Normalization(double scaleFactor, Int_t scaleType)
{
   return RooCmdArg("Normalization", scaleType, 0, scaleFactor, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooAbsData::plotOn arguments
RooCmdArg Cut(const char *cutSpec)
{
   return RooCmdArg("CutSpec", 0, 0, 0, 0, cutSpec, nullptr, nullptr, nullptr);
}
RooCmdArg Cut(const RooFormulaVar &cutVar)
{
   return RooCmdArg("CutVar", 0, 0, 0, 0, nullptr, nullptr, &cutVar, nullptr);
}
RooCmdArg Binning(const RooAbsBinning &binning)
{
   return RooCmdArg("Binning", 0, 0, 0, 0, nullptr, nullptr, &binning, nullptr);
}
RooCmdArg Binning(const char *binningName)
{
   return RooCmdArg("BinningName", 0, 0, 0, 0, binningName, nullptr, nullptr, nullptr);
}
RooCmdArg Binning(int nBins, double xlo, double xhi)
{
   return RooCmdArg("BinningSpec", nBins, 0, xlo, xhi, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg MarkerStyle(Style_t style)
{
   return RooCmdArg("MarkerStyle", style, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg MarkerStyle(std::string const &color)
{
   return MarkerStyle(interpretString(color));
}
RooCmdArg MarkerSize(Size_t size)
{
   return RooCmdArg("MarkerSize", 0, 0, size, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg MarkerColor(TColorNumber color)
{
   return RooCmdArg("MarkerColor", color.number());
}
RooCmdArg CutRange(const char *rangeName)
{
   return RooCmdArg("CutRange", 0, 0, 0, 0, rangeName, nullptr, nullptr, nullptr);
}
RooCmdArg XErrorSize(double width)
{
   return RooCmdArg("XErrorSize", 0, 0, width, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg RefreshNorm()
{
   return RooCmdArg("RefreshNorm", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Efficiency(const RooCategory &cat)
{
   return RooCmdArg("Efficiency", 0, 0, 0, 0, nullptr, nullptr, &cat, nullptr);
}
RooCmdArg Rescale(double factor)
{
   return RooCmdArg("Rescale", 0, 0, factor, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooDataHist::ctor arguments
RooCmdArg Weight(double wgt)
{
   return RooCmdArg("Weight", 0, 0, wgt, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Index(RooCategory &icat)
{
   return RooCmdArg("IndexCat", 0, 0, 0, 0, nullptr, nullptr, &icat, nullptr);
}
RooCmdArg Import(const char *state, TH1 &histo)
{
   return RooCmdArg("ImportDataSlice", 0, 0, 0, 0, state, nullptr, &histo, nullptr);
}
RooCmdArg Import(const char *state, RooDataHist &dhist)
{
   return RooCmdArg("ImportDataSlice", 0, 0, 0, 0, state, nullptr, &dhist, nullptr);
}
RooCmdArg Import(TH1 &histo, bool importDensity)
{
   return RooCmdArg("ImportHisto", importDensity, 0, 0, 0, nullptr, nullptr, &histo, nullptr);
}

RooCmdArg Import(const std::map<std::string, RooDataHist *> &arg)
{
   return processMap("ImportDataSliceMany", processImportItem<RooDataHist>, arg);
}
RooCmdArg Import(const std::map<std::string, TH1 *> &arg)
{
   return processMap("ImportDataSliceMany", processImportItem<TH1>, arg);
}

// RooDataSet::ctor arguments
RooCmdArg WeightVar(const char *name, bool reinterpretAsWeight)
{
   if (name == nullptr)
      return RooCmdArg::none(); // Passing a nullptr name means no weight variable
   return RooCmdArg("WeightVarName", reinterpretAsWeight, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg WeightVar(const RooRealVar &arg, bool reinterpretAsWeight)
{
   return RooCmdArg("WeightVar", reinterpretAsWeight, 0, 0, 0, nullptr, nullptr, &arg, nullptr);
}
RooCmdArg Link(const char *state, RooAbsData &data)
{
   return RooCmdArg("LinkDataSlice", 0, 0, 0, 0, state, nullptr, &data, nullptr);
}
RooCmdArg Import(const char *state, RooAbsData &data)
{
   return RooCmdArg("ImportDataSlice", 0, 0, 0, 0, state, nullptr, &data, nullptr);
}
RooCmdArg Import(RooAbsData &data)
{
   return RooCmdArg("ImportData", 0, 0, 0, 0, nullptr, nullptr, &data, nullptr);
}
RooCmdArg Import(TTree &tree)
{
   return RooCmdArg("ImportTree", 0, 0, 0, 0, nullptr, nullptr, reinterpret_cast<TObject *>(&tree), nullptr);
}
RooCmdArg ImportFromFile(const char *fname, const char *tname)
{
   return RooCmdArg("ImportFromFile", 0, 0, 0, 0, fname, tname, nullptr, nullptr);
}
RooCmdArg StoreError(const RooArgSet &aset)
{
   return RooCmdArg("StoreError", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &aset);
}
RooCmdArg StoreAsymError(const RooArgSet &aset)
{
   return RooCmdArg("StoreAsymError", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &aset);
}
RooCmdArg OwnLinked()
{
   return RooCmdArg("OwnLinked", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

RooCmdArg Import(const std::map<std::string, RooDataSet *> &arg)
{
   return processMap("ImportDataSliceMany", processImportItem<RooDataSet>, arg);
}
RooCmdArg Link(const std::map<std::string, RooAbsData *> &arg)
{
   return processMap("LinkDataSliceMany", processLinkItem<RooAbsData>, arg);
}

// RooChi2Var::ctor / RooNLLVar arguments
RooCmdArg Extended(bool flag)
{
   return RooCmdArg("Extended", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg DataError(Int_t etype)
{
   return RooCmdArg("DataError", etype);
}
RooCmdArg DataError(std::string const &etype)
{
   return DataError(RooAbsData::errorTypeFromString(etype));
}
RooCmdArg NumCPU(Int_t nCPU, Int_t interleave)
{
   return RooCmdArg("NumCPU", nCPU, interleave, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ModularL(bool flag)
{
   return RooCmdArg("ModularL", flag, 0, 0, 0, nullptr, nullptr, nullptr);
}
RooCmdArg Parallelize(Int_t nWorkers)
{
   return RooCmdArg("Parallelize", nWorkers, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg TimingAnalysis(bool flag)
{
   return RooCmdArg("TimingAnalysis", flag, 0, 0, 0, nullptr, nullptr, nullptr);
}
RooCmdArg BatchMode(std::string const &batchMode)
{
   oocoutW(nullptr, InputArguments)
      << "The BatchMode() command argument is deprecated. Please use EvalBackend() instead." << std::endl;
   std::string lower = batchMode;
   std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
   if (lower == "off") {
      return EvalBackend::Legacy();
   } else if (lower == "cpu") {
      return EvalBackend::Cpu();
   } else if (lower == "cuda") {
      return EvalBackend::Cuda();
   }
   throw std::runtime_error("Only supported string values for BatchMode() are \"off\", \"cpu\", or \"cuda\".");
}
/// Integrate the PDF over bins. Improves accuracy for binned fits. Switch off using `0.` as argument. \see
/// RooAbsPdf::fitTo().
RooCmdArg IntegrateBins(double precision)
{
   return RooCmdArg("IntegrateBins", 0, 0, precision);
}

// RooAbsCollection::printLatex arguments
RooCmdArg Columns(Int_t ncol)
{
   return RooCmdArg("Columns", ncol, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg OutputFile(const char *fileName)
{
   return RooCmdArg("OutputFile", 0, 0, 0, 0, fileName, nullptr, nullptr, nullptr);
}
RooCmdArg Sibling(const RooAbsCollection &sibling)
{
   return RooCmdArg("Sibling", 0, 0, 0, 0, nullptr, nullptr, &sibling, nullptr);
}
RooCmdArg Format(const char *what, const RooCmdArg &arg1, const RooCmdArg &arg2, const RooCmdArg &arg3,
                 const RooCmdArg &arg4, const RooCmdArg &arg5, const RooCmdArg &arg6, const RooCmdArg &arg7,
                 const RooCmdArg &arg8)
{
   RooCmdArg ret("FormatArgs", 0, 0, 0, 0, what, nullptr, nullptr, nullptr);
   ret.addArg(arg1);
   ret.addArg(arg2);
   ret.addArg(arg3);
   ret.addArg(arg4);
   ret.addArg(arg5);
   ret.addArg(arg6);
   ret.addArg(arg7);
   ret.addArg(arg8);
   ret.setProcessRecArgs(false);
   return ret;
}

// RooAbsRealLValue::frame arguments
RooCmdArg Title(const char *name)
{
   return RooCmdArg("Title", 0, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg Bins(Int_t nbin)
{
   return RooCmdArg("Bins", nbin, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg AutoSymRange(const RooAbsData &data, double marginFactor)
{
   return RooCmdArg("AutoRange", 1, 0, marginFactor, 0, nullptr, nullptr, &data, nullptr);
}
RooCmdArg AutoRange(const RooAbsData &data, double marginFactor)
{
   return RooCmdArg("AutoRange", 0, 0, marginFactor, 0, nullptr, nullptr, &data, nullptr);
}

// RooAbsData::reduce arguments
RooCmdArg SelectVars(const RooArgSet &vars)
{
   RooCmdArg out{"SelectVars", 0};
   out.setSet(0, vars);
   return out;
}
RooCmdArg EventRange(Int_t nStart, Int_t nStop)
{
   return RooCmdArg("EventRange", nStart, nStop, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooAbsPdf::fitTo arguments

EvalBackend::EvalBackend(EvalBackend::Value value) : RooCmdArg{"EvalBackend", static_cast<int>(value)} {}
EvalBackend::EvalBackend(std::string const &name) : EvalBackend{toValue(name)} {}
EvalBackend::Value EvalBackend::toValue(std::string const &name)
{
   std::string lower = name;
   std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
   if (lower == toName(Value::Legacy))
      return Value::Legacy;
   if (lower == toName(Value::Cpu))
      return Value::Cpu;
   if (lower == toName(Value::Cuda))
      return Value::Cuda;
   if (lower == toName(Value::Codegen))
      return Value::Codegen;
   if (lower == toName(Value::CodegenNoGrad))
      return Value::CodegenNoGrad;
   throw std::runtime_error("Only supported string values for EvalBackend() are \"legacy\", \"cpu\", \"cuda\", "
                            "\"codegen\", or \"codegen_no_grad\".");
}
EvalBackend EvalBackend::Legacy()
{
   return EvalBackend(Value::Legacy);
}
EvalBackend EvalBackend::Cpu()
{
   return EvalBackend(Value::Cpu);
}
EvalBackend EvalBackend::Cuda()
{
   return EvalBackend(Value::Cuda);
}
EvalBackend EvalBackend::Codegen()
{
   return EvalBackend(Value::Codegen);
}
EvalBackend EvalBackend::CodegenNoGrad()
{
   return EvalBackend(Value::CodegenNoGrad);
}
std::string EvalBackend::name() const
{
   return toName(value());
}
std::string EvalBackend::toName(EvalBackend::Value value)
{
   if (value == Value::Legacy)
      return "legacy";
   if (value == Value::Cpu)
      return "cpu";
   if (value == Value::Cuda)
      return "cuda";
   if (value == Value::Codegen)
      return "codegen";
   if (value == Value::CodegenNoGrad)
      return "codegen_no_grad";
   return "";
}
EvalBackend::Value &EvalBackend::defaultValue()
{
   static Value value = Value::Cpu;
   return value;
}

RooCmdArg PrefitDataFraction(double data_ratio)
{
   return RooCmdArg("Prefit", 0, 0, data_ratio, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Optimize(Int_t flag)
{
   return RooCmdArg("Optimize", flag);
}
RooCmdArg Verbose(bool flag)
{
   return RooCmdArg("Verbose", flag);
}
RooCmdArg Save(bool flag)
{
   return RooCmdArg("Save", flag);
}
RooCmdArg Timer(bool flag)
{
   return RooCmdArg("Timer", flag);
}
RooCmdArg PrintLevel(Int_t level)
{
   return RooCmdArg("PrintLevel", level);
}
RooCmdArg MaxCalls(int n)
{
   return RooCmdArg("MaxCalls", n);
}
RooCmdArg Warnings(bool flag)
{
   return RooCmdArg("Warnings", flag);
}
RooCmdArg Strategy(Int_t code)
{
   return RooCmdArg("Strategy", code);
}
RooCmdArg InitialHesse(bool flag)
{
   return RooCmdArg("InitialHesse", flag);
}
RooCmdArg Hesse(bool flag)
{
   return RooCmdArg("Hesse", flag);
}
RooCmdArg Minos(bool flag)
{
   return RooCmdArg("Minos", flag);
}
RooCmdArg Minos(const RooArgSet &minosArgs)
{
   RooCmdArg out{"Minos", 1};
   out.setSet(0, minosArgs);
   return out;
}
RooCmdArg SplitRange(bool flag)
{
   return RooCmdArg("SplitRange", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg SumCoefRange(const char *rangeName)
{
   return RooCmdArg("SumCoefRange", 0, 0, 0, 0, rangeName, nullptr, nullptr, nullptr);
}
RooCmdArg Constrain(const RooArgSet &params)
{
   for (RooAbsArg *param : params) {
      if (!dynamic_cast<RooRealVar *>(param)) {
         std::stringstream errorMsg;
         errorMsg << "RooFit::Constrain(): you passed the argument \"" << param->GetName()
                  << "\", but it's not a RooRealVar!"
                  << " You can only constrain parameters, which must be RooRealVars.";
         oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
         throw std::invalid_argument(errorMsg.str().c_str());
      }
   }
   return RooCmdArg("Constrain", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &params);
}
RooCmdArg GlobalObservablesSource(const char *sourceName)
{
   return {"GlobalObservablesSource", 0, 0, 0, 0, sourceName, nullptr, nullptr, nullptr};
}
RooCmdArg GlobalObservablesTag(const char *tagName)
{
   return RooCmdArg("GlobalObservablesTag", 0, 0, 0, 0, tagName, nullptr, nullptr, nullptr);
}
RooCmdArg ExternalConstraints(const RooArgSet &cpdfs)
{
   return RooCmdArg("ExternalConstraints", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &cpdfs);
}
RooCmdArg PrintEvalErrors(Int_t numErrors)
{
   return RooCmdArg("PrintEvalErrors", numErrors, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg EvalErrorWall(bool flag)
{
   return RooCmdArg("EvalErrorWall", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg SumW2Error(bool flag)
{
   return RooCmdArg("SumW2Error", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg AsymptoticError(bool flag)
{
   return RooCmdArg("AsymptoticError", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg CloneData(bool flag)
{
   oocoutI(nullptr, InputArguments) << "The deprecated RooFit::CloneData(" << flag
                                    << ") option passed to createNLL() is ignored." << std::endl;
   return RooCmdArg("CloneData", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Integrate(bool flag)
{
   return RooCmdArg("Integrate", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Minimizer(const char *type, const char *alg)
{
   return RooCmdArg("Minimizer", 0, 0, 0, 0, type, alg, nullptr, nullptr);
}

RooCmdArg Offset(std::string const &mode)
{
   std::string lower = mode;
   std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
   OffsetMode modeVal = OffsetMode::None;
   if (lower == "none") {
      modeVal = OffsetMode::None;
   } else if (lower == "initial") {
      modeVal = OffsetMode::Initial;
   } else if (lower == "bin") {
      modeVal = OffsetMode::Bin;
   }
   return RooCmdArg("OffsetLikelihood", static_cast<int>(modeVal));
}

/// When parameters are chosen such that a PDF is undefined, try to indicate to the minimiser how to leave this region.
/// \param strength Strength of hints for minimiser. Set to zero to switch off.
RooCmdArg RecoverFromUndefinedRegions(double strength)
{
   return RooCmdArg("RecoverFromUndefinedRegions", 0, 0, strength, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooAbsPdf::paramOn arguments
RooCmdArg Label(const char *str)
{
   return RooCmdArg("Label", 0, 0, 0, 0, str, nullptr, nullptr, nullptr);
}
RooCmdArg Layout(double xmin, double xmax, double ymin)
{
   return RooCmdArg("Layout", Int_t(ymin * 10000), 0, xmin, xmax, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Parameters(const RooArgSet &params)
{
   RooCmdArg out{"Parameters", 0};
   out.setSet(0, params);
   return out;
}
RooCmdArg ShowConstants(bool flag)
{
   return RooCmdArg("ShowConstants", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooTreeData::statOn arguments
RooCmdArg What(const char *str)
{
   return RooCmdArg("What", 0, 0, 0, 0, str, nullptr, nullptr, nullptr);
}

// RooProdPdf::ctor arguments
RooCmdArg Conditional(const RooArgSet &pdfSet, const RooArgSet &depSet, bool depsAreCond)
{
   return RooCmdArg("Conditional", depsAreCond, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &pdfSet,
                    &depSet);
};

// RooAbsPdf::generate arguments
RooCmdArg ProtoData(const RooDataSet &protoData, bool randomizeOrder, bool resample)
{
   return RooCmdArg("PrototypeData", randomizeOrder, resample, 0, 0, nullptr, nullptr, &protoData, nullptr);
}
RooCmdArg NumEvents(Int_t numEvents)
{
   return RooCmdArg("NumEvents", numEvents, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg NumEvents(double numEvents)
{
   return RooCmdArg("NumEventsD", 0, 0, numEvents, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ExpectedData(bool flag)
{
   return RooCmdArg("ExpectedData", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Asimov(bool flag)
{
   return ExpectedData(flag);
}
RooCmdArg AutoBinned(bool flag)
{
   return RooCmdArg("AutoBinned", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg GenBinned(const char *tag)
{
   return RooCmdArg("GenBinned", 0, 0, 0, 0, tag, nullptr, nullptr, nullptr);
}
RooCmdArg AllBinned()
{
   return RooCmdArg("GenBinned", 0, 0, 0, 0, "*", nullptr, nullptr, nullptr);
}

// RooAbsRealLValue::createHistogram arguments
RooCmdArg YVar(const RooAbsRealLValue &var, const RooCmdArg &arg)
{
   return RooCmdArg("YVar", 0, 0, 0, 0, nullptr, nullptr, &var, nullptr, &arg);
}
RooCmdArg ZVar(const RooAbsRealLValue &var, const RooCmdArg &arg)
{
   return RooCmdArg("ZVar", 0, 0, 0, 0, nullptr, nullptr, &var, nullptr, &arg);
}
RooCmdArg AxisLabel(const char *name)
{
   return RooCmdArg("AxisLabel", 0, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg Scaling(bool flag)
{
   return RooCmdArg("Scaling", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooAbsReal::createHistogram arguments
RooCmdArg IntrinsicBinning(bool flag)
{
   return RooCmdArg("IntrinsicBinning", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooAbsData::createHistogram arguments
RooCmdArg AutoSymBinning(Int_t nbins, double marginFactor)
{
   return RooCmdArg("AutoRangeData", 1, nbins, marginFactor, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg AutoBinning(Int_t nbins, double marginFactor)
{
   return RooCmdArg("AutoRangeData", 0, nbins, marginFactor, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooAbsReal::fillHistogram arguments
RooCmdArg IntegratedObservables(const RooArgSet &intObs)
{
   return RooCmdArg("IntObs", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &intObs, nullptr);
};

// RooAbsReal::createIntegral arguments
RooCmdArg NumIntConfig(const RooNumIntConfig &cfg)
{
   return RooCmdArg("NumIntConfig", 0, 0, 0, 0, nullptr, nullptr, &cfg, nullptr);
}

// RooMCStudy::ctor arguments
RooCmdArg Silence(bool flag)
{
   return RooCmdArg("Silence", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg FitModel(RooAbsPdf &pdf)
{
   return RooCmdArg("FitModel", 0, 0, 0, 0, nullptr, nullptr, &pdf, nullptr);
}
RooCmdArg FitOptions(const RooCmdArg &arg1, const RooCmdArg &arg2, const RooCmdArg &arg3, const RooCmdArg &arg4,
                     const RooCmdArg &arg5, const RooCmdArg &arg6)
{
   RooCmdArg ret("FitOptArgs", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
   ret.addArg(arg1);
   ret.addArg(arg2);
   ret.addArg(arg3);
   ret.addArg(arg4);
   ret.addArg(arg5);
   ret.addArg(arg6);
   ret.setProcessRecArgs(false);
   return ret;
}
RooCmdArg Binned(bool flag)
{
   return RooCmdArg("Binned", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg BootStrapData(const RooDataSet &dset)
{
   return RooCmdArg("BootStrapData", 0, 0, 0, 0, nullptr, nullptr, &dset, nullptr);
}

// RooMCStudy::plot* arguments
RooCmdArg Frame(const RooCmdArg &arg1, const RooCmdArg &arg2, const RooCmdArg &arg3, const RooCmdArg &arg4,
                const RooCmdArg &arg5, const RooCmdArg &arg6)
{
   RooCmdArg ret("FrameArgs", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
   ret.addArg(arg1);
   ret.addArg(arg2);
   ret.addArg(arg3);
   ret.addArg(arg4);
   ret.addArg(arg5);
   ret.addArg(arg6);
   ret.setProcessRecArgs(false);
   return ret;
}
RooCmdArg FrameBins(Int_t nbins)
{
   return RooCmdArg("Bins", nbins, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg FrameRange(double xlo, double xhi)
{
   return RooCmdArg("Range", 0, 0, xlo, xhi, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg FitGauss(bool flag)
{
   return RooCmdArg("FitGauss", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooRealVar::format arguments
RooCmdArg ShowName(bool flag)
{
   return RooCmdArg("ShowName", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ShowValue(bool flag)
{
   return RooCmdArg("ShowValue", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ShowError(bool flag)
{
   return RooCmdArg("ShowError", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ShowAsymError(bool flag)
{
   return RooCmdArg("ShowAsymError", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ShowUnit(bool flag)
{
   return RooCmdArg("ShowUnit", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg AutoPrecision(Int_t ndigit)
{
   return RooCmdArg("AutoPrecision", ndigit, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg FixedPrecision(Int_t ndigit)
{
   return RooCmdArg("FixedPrecision", ndigit, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg TLatexStyle(bool flag)
{
   return RooCmdArg("TLatexStyle", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg LatexStyle(bool flag)
{
   return RooCmdArg("LatexStyle", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg LatexTableStyle(bool flag)
{
   return RooCmdArg("LatexTableStyle", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg VerbatimName(bool flag)
{
   return RooCmdArg("VerbatimName", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooMsgService::addReportingStream arguments
RooCmdArg Topic(Int_t topic)
{
   return RooCmdArg("Topic", topic, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ObjectName(const char *name)
{
   return RooCmdArg("ObjectName", 0, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg ClassName(const char *name)
{
   return RooCmdArg("ClassName", 0, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg BaseClassName(const char *name)
{
   return RooCmdArg("BaseClassName", 0, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg TagName(const char *name)
{
   return RooCmdArg("LabelName", 0, 0, 0, 0, name, nullptr, nullptr, nullptr);
}
RooCmdArg OutputStream(std::ostream &os)
{
   return RooCmdArg("OutputStream", 0, 0, 0, 0, nullptr, nullptr, new RooHelpers::WrapIntoTObject<std::ostream>(os),
                    nullptr);
}
RooCmdArg Prefix(bool flag)
{
   return RooCmdArg("Prefix", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg Color(TColorNumber color)
{
   return RooCmdArg("Color", color.number());
}

// RooWorkspace::import() arguments
RooCmdArg RenameConflictNodes(const char *suffix, bool ro)
{
   return RooCmdArg("RenameConflictNodes", ro, 0, 0, 0, suffix, nullptr, nullptr, nullptr);
}
RooCmdArg RecycleConflictNodes(bool flag)
{
   return RooCmdArg("RecycleConflictNodes", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg RenameAllNodes(const char *suffix)
{
   return RooCmdArg("RenameAllNodes", 0, 0, 0, 0, suffix, nullptr, nullptr, nullptr);
}
RooCmdArg RenameAllVariables(const char *suffix)
{
   return RooCmdArg("RenameAllVariables", 0, 0, 0, 0, suffix, nullptr, nullptr, nullptr);
}
RooCmdArg RenameAllVariablesExcept(const char *suffix, const char *except)
{
   return RooCmdArg("RenameAllVariables", 0, 0, 0, 0, suffix, except, nullptr, nullptr);
}
RooCmdArg RenameVariable(const char *in, const char *out)
{
   return RooCmdArg("RenameVar", 0, 0, 0, 0, in, out, nullptr, nullptr);
}
RooCmdArg Rename(const char *suffix)
{
   return RooCmdArg("Rename", 0, 0, 0, 0, suffix, nullptr, nullptr, nullptr);
}
RooCmdArg Embedded(bool flag)
{
   return RooCmdArg("Embedded", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg NoRecursion(bool flag)
{
   return RooCmdArg("NoRecursion", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

// RooSimCloneTool::build() arguments
RooCmdArg SplitParam(const char *varname, const char *catname)
{
   return RooCmdArg("SplitParam", 0, 0, 0, 0, varname, catname, nullptr, nullptr);
}
RooCmdArg SplitParam(const RooRealVar &var, const RooAbsCategory &cat)
{
   return RooCmdArg("SplitParam", 0, 0, 0, 0, var.GetName(), cat.GetName(), nullptr, nullptr);
}
RooCmdArg SplitParamConstrained(const char *varname, const char *catname, const char *rsname)
{
   return RooCmdArg("SplitParamConstrained", 0, 0, 0, 0, varname, catname, nullptr, nullptr, nullptr, rsname);
}
RooCmdArg SplitParamConstrained(const RooRealVar &var, const RooAbsCategory &cat, const char *rsname)
{
   return RooCmdArg("SplitParamConstrained", 0, 0, 0, 0, var.GetName(), cat.GetName(), nullptr, nullptr, nullptr,
                    rsname);
}
RooCmdArg Restrict(const char *catName, const char *stateNameList)
{
   return RooCmdArg("Restrict", 0, 0, 0, 0, catName, stateNameList, nullptr, nullptr);
}

// RooAbsPdf::createCdf() arguments
RooCmdArg SupNormSet(const RooArgSet &nset)
{
   RooCmdArg out{"SupNormSet", 0};
   out.setSet(0, nset);
   return out;
}
RooCmdArg ScanParameters(Int_t nbins, Int_t intOrder)
{
   return RooCmdArg("ScanParameters", nbins, intOrder, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ScanNumCdf()
{
   return RooCmdArg("ScanNumCdf", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ScanAllCdf()
{
   return RooCmdArg("ScanAllCdf", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}
RooCmdArg ScanNoCdf()
{
   return RooCmdArg("ScanNoCdf", 1, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

RooCmdArg MultiArg(const RooCmdArg &arg1, const RooCmdArg &arg2, const RooCmdArg &arg3, const RooCmdArg &arg4,
                   const RooCmdArg &arg5, const RooCmdArg &arg6, const RooCmdArg &arg7, const RooCmdArg &arg8)
{
   RooCmdArg ret("MultiArg", 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
   ret.addArg(arg1);
   ret.addArg(arg2);
   ret.addArg(arg3);
   ret.addArg(arg4);
   ret.addArg(arg5);
   ret.addArg(arg6);
   ret.addArg(arg7);
   ret.addArg(arg8);
   ret.setProcessRecArgs(true, false);
   return ret;
}

RooConstVar &RooConst(double val)
{
   return RooRealConstant::value(val);
}

namespace Detail {

RooCmdArg SliceFlatMap(FlatMap<RooCategory *, std::string> const &args)
{
   return processFlatMap("SliceCatMany", processSliceItem, args);
}
RooCmdArg ImportFlatMap(FlatMap<std::string, RooDataHist *> const &args)
{
   return processFlatMap("ImportDataSliceMany", processImportItem<RooDataHist>, args);
}
RooCmdArg ImportFlatMap(FlatMap<std::string, TH1 *> const &args)
{
   return processFlatMap("ImportDataSliceMany", processImportItem<TH1>, args);
}
RooCmdArg ImportFlatMap(FlatMap<std::string, RooDataSet *> const &args)
{
   return processFlatMap("ImportDataSliceMany", processImportItem<RooDataSet>, args);
}
RooCmdArg LinkFlatMap(FlatMap<std::string, RooAbsData *> const &args)
{
   return processFlatMap("LinkDataSliceMany", processLinkItem<RooAbsData>, args);
}

} // namespace Detail

} // namespace RooFit

namespace RooFitShortHand {

RooConstVar &C(double value)
{
   return RooFit::RooConst(value);
}

} // namespace RooFitShortHand
