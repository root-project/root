/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooGlobalFunc.h,v 1.14 2007/07/16 21:04:28 wouter Exp $
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
#ifndef ROO_GLOBAL_FUNC
#define ROO_GLOBAL_FUNC

#include "RooCmdArg.h"
#include "RooLinkedList.h"
#include "RooArgSet.h"

#include "ROOT/RConfig.hxx"
#include <TColor.h>

#include <map>
#include <memory>
#include <string>

class RooDataHist ;
class RooDataSet ;
class RooFitResult ;
class RooAbsPdf ;
class RooAbsRealLValue ;
class RooRealConstant ;
class RooMsgService ;
class RooFormulaVar ;
class RooAbsData ;
class RooCategory ;
class RooAbsReal ;
class RooAbsBinning ;
class RooAbsCollection ;
class RooAbsPdf ;
class RooConstVar ;
class RooRealVar ;
class RooAbsCategory ;
class RooNumIntConfig ;
class RooHist;

class TH1 ;
class TTree ;

/*! \namespace RooFit
The namespace RooFit contains mostly switches that change the behaviour of functions of PDFs
(or other types of arguments).

These switches are documented with the relevant functions, e.g. RooAbsPdf::fitTo().
For an introduction to RooFit (not the namespace), check the [user's guides](https://root.cern/root-user-guides-and-manuals),
[courses](https://root.cern/learn/courses) or [the RooFit chapter of the Manual](https://root.cern/manual/roofit/).
*/
namespace RooFit {

/// Verbosity level for RooMsgService::StreamConfig in RooMsgService
enum MsgLevel { DEBUG=0, INFO=1, PROGRESS=2, WARNING=3, ERROR=4, FATAL=5 } ;
/// Topics for a RooMsgService::StreamConfig in RooMsgService
enum MsgTopic { Generation=1, Minimization=2, Plotting=4, Fitting=8, Integration=16, LinkStateMgmt=32,
    Eval=64, Caching=128, Optimization=256, ObjectHandling=512, InputArguments=1024, Tracing=2048,
    Contents=4096, DataHandling=8192, NumericIntegration=16384, FastEvaluations=1<<15, HistFactory=1<<16, IO=1<<17 };
enum MPSplit { BulkPartition=0, Interleave=1, SimComponents=2, Hybrid=3 } ;

/// Alias of MsgLevel::NumericIntegration for backwards compatibility.
/// \see https://github.com/root-project/root/issues/19422
constexpr static auto NumIntegration = NumericIntegration;

/// For setting the offset mode with the Offset() command argument to
/// RooAbsPdf::fitTo()
enum class OffsetMode { None, Initial, Bin };

namespace Experimental {

/// Configuration options for parallel minimization with multiprocessing library
RooCmdArg ParallelGradientOptions(bool enable=true, int orderStrategy=0, int chainFactor=1) ;
RooCmdArg ParallelDescentOptions(bool enable=false, int splitStrategy=0, int numSplits=0) ;

void writeCodegenDebugMacro(RooAbsReal const &absReal, std::string const &name);
inline void writeCodegenDebugMacro(std::unique_ptr<RooAbsReal> const &absReal, std::string const &name)
{
   return writeCodegenDebugMacro(*absReal, name);
}

} // Experimental

std::string getBatchCompute();
void setBatchCompute(std::string const &value);

/**
 * \defgroup CmdArgs RooFit command arguments
 * These arguments can be passed to functions of RooFit objects.
 * \ingroup Roofitmain
 * @{
 */

/**
 * \defgroup Plotting Arguments for plotOn functions
 * @{
 */
RooCmdArg DrawOption(const char* opt) ;
RooCmdArg Normalization(double scaleFactor) ;
RooCmdArg Slice(const RooArgSet& sliceSet) ;
RooCmdArg Slice(RooCategory& cat, const char* label) ;
RooCmdArg Slice(std::map<RooCategory*, std::string> const&) ;
RooCmdArg Project(const RooArgSet& projSet) ;
RooCmdArg ProjWData(const RooAbsData& projData, bool binData=false) ;
RooCmdArg ProjWData(const RooArgSet& projSet, const RooAbsData& projData, bool binData=false) ;
RooCmdArg Asymmetry(const RooCategory& cat) ;
RooCmdArg Precision(double prec) ;
RooCmdArg ShiftToZero() ;
RooCmdArg Range(const char* rangeName, bool adjustNorm=true) ;
RooCmdArg Range(double lo, double hi, bool adjustNorm=true) ;
RooCmdArg NormRange(const char* rangeNameList) ;
RooCmdArg VLines() ;
RooCmdArg LineColor(TColorNumber color) ;
RooCmdArg LineStyle(Style_t style) ;
RooCmdArg LineStyle(std::string const &style) ;
RooCmdArg LineWidth(Width_t width) ;
RooCmdArg FillColor(TColorNumber color) ;
RooCmdArg FillStyle(Style_t style) ;
RooCmdArg FillStyle(std::string const &style) ;
RooCmdArg ProjectionRange(const char* rangeName) ;
RooCmdArg Name(const char* name) ;
RooCmdArg Invisible(bool inv=true) ;
RooCmdArg AddTo(const char* name, double wgtSel=1.0, double wgtOther=1.0) ;
RooCmdArg EvalErrorValue(double value) ;
RooCmdArg MoveToBack()  ;
RooCmdArg VisualizeError(const RooDataSet& paramData, double Z=1) ;
RooCmdArg VisualizeError(const RooFitResult& fitres, double Z=1, bool linearMethod=true) ;
RooCmdArg VisualizeError(const RooFitResult& fitres, const RooArgSet& param, double Z=1, bool linearMethod=true) ;
RooCmdArg ShowProgress() ;

// RooAbsPdf::plotOn arguments
RooCmdArg Normalization(double scaleFactor, Int_t scaleType) ;
template<class... Args_t>
RooCmdArg Components(Args_t &&... argsOrArgSet) {
  RooCmdArg out{"SelectCompSet",0};
  out.setSet(0, RooArgSet{std::forward<Args_t>(argsOrArgSet)...});
  return out;
}
RooCmdArg Components(const char* compSpec) ;

// RooAbsData::plotOn arguments
RooCmdArg Cut(const char* cutSpec) ;
RooCmdArg Cut(const RooFormulaVar& cutVar) ;
RooCmdArg Binning(const RooAbsBinning& binning) ;
RooCmdArg Binning(const char* binningName) ;
RooCmdArg Binning(int nBins, double xlo=0.0, double xhi=0.0) ;
RooCmdArg MarkerStyle(Style_t style) ;
RooCmdArg MarkerStyle(std::string const &style) ;
RooCmdArg MarkerSize(Size_t size) ;
RooCmdArg MarkerColor(TColorNumber color) ;
RooCmdArg CutRange(const char* rangeName) ;
RooCmdArg XErrorSize(double width) ;
RooCmdArg RefreshNorm() ;
RooCmdArg Efficiency(const RooCategory& cat) ;
RooCmdArg Rescale(double factor) ;

/** @} */

/**
 * \defgroup ConstructorArgs Arguments for various constructors
 * @{
 */
// RooDataHist::ctor arguments
RooCmdArg Weight(double wgt) ;
RooCmdArg Index(RooCategory& icat) ;
RooCmdArg Import(const char* state, TH1& histo) ;
RooCmdArg Import(const std::map<std::string,TH1*>&) ;
RooCmdArg Import(const char* state, RooDataHist& dhist) ;
RooCmdArg Import(const std::map<std::string,RooDataHist*>&) ;
RooCmdArg Import(TH1& histo, bool importDensity=false) ;

// RooDataSet::ctor arguments
RooCmdArg WeightVar(const char* name="weight", bool reinterpretAsWeight=false) ;
RooCmdArg WeightVar(const RooRealVar& arg, bool reinterpretAsWeight=false) ;
RooCmdArg Import(const char* state, RooAbsData& data) ;
RooCmdArg Import(const std::map<std::string,RooDataSet*>& ) ;
template<class DataPtr_t>
RooCmdArg Import(std::map<std::string,DataPtr_t> const& map) {
    RooCmdArg container("ImportDataSliceMany",0,0,0,0,nullptr,nullptr,nullptr,nullptr) ;
    for (auto const& item : map) {
      container.addArg(Import(item.first.c_str(), *item.second)) ;
    }
    container.setProcessRecArgs(true,false) ;
    return container ;
}

RooCmdArg Link(const char* state, RooAbsData& data) ;
RooCmdArg Link(const std::map<std::string,RooAbsData*>&) ;
RooCmdArg Import(RooAbsData& data) ;
RooCmdArg Import(TTree& tree) ;
RooCmdArg ImportFromFile(const char* fname, const char* tname) ;
RooCmdArg StoreError(const RooArgSet& aset) ;
RooCmdArg StoreAsymError(const RooArgSet& aset) ;
RooCmdArg OwnLinked() ;

/** @} */

// RooAbsPdf::printLatex arguments
RooCmdArg Columns(Int_t ncol) ;
RooCmdArg OutputFile(const char* fileName) ;
RooCmdArg Format(const char* what, const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
                 const RooCmdArg& arg3={},const RooCmdArg& arg4={},
                 const RooCmdArg& arg5={},const RooCmdArg& arg6={},
                 const RooCmdArg& arg7={},const RooCmdArg& arg8={}) ;
RooCmdArg Sibling(const RooAbsCollection& sibling) ;

// RooAbsRealLValue::frame arguments
RooCmdArg Title(const char* name) ;
RooCmdArg Bins(Int_t nbin) ;
RooCmdArg AutoSymRange(const RooAbsData& data, double marginFactor=0.1) ;
RooCmdArg AutoRange(const RooAbsData& data, double marginFactor=0.1) ;

// RooAbsData::createHistogram arguments
RooCmdArg AutoSymBinning(Int_t nbins=100, double marginFactor=0.1) ;
RooCmdArg AutoBinning(Int_t nbins=100, double marginFactor=0.1) ;

// RooAbsReal::fillHistogram arguments
RooCmdArg IntegratedObservables(const RooArgSet& intObs) ;

// RooAbsData::reduce arguments
RooCmdArg SelectVars(const RooArgSet& vars) ;
RooCmdArg EventRange(Int_t nStart, Int_t nStop) ;


/**
 * \defgroup Fitting Arguments for fitting
 * @{
 */
// RooChi2Var::ctor / RooNLLVar arguments
RooCmdArg Extended(bool flag=true) ;
RooCmdArg DataError(Int_t) ;
RooCmdArg DataError(std::string const&) ;
RooCmdArg NumCPU(Int_t nCPU, Int_t interleave=0) ;
RooCmdArg Parallelize(int nWorkers) ;
RooCmdArg ModularL(bool flag=false) ;
RooCmdArg TimingAnalysis(bool timingAnalysis) ;

RooCmdArg BatchMode(std::string const &batchMode = "cpu")
   R__DEPRECATED(6, 44, "Use EvalBackend() instead of BatchMode().");
// The const char * overload is necessary, otherwise the compiler will cast a
// C-Style string to a bool and choose the BatchMode(bool) overload if one
// calls for example BatchMode("off").
RooCmdArg BatchMode(const char *batchMode)
   R__DEPRECATED(6, 44, "Use EvalBackend() instead of BatchMode().");
RooCmdArg BatchMode(bool batchModeOn)
   R__DEPRECATED(6, 44, "Use EvalBackend() instead of BatchMode().");

RooCmdArg IntegrateBins(double precision);

// RooAbsPdf::fitTo arguments
RooCmdArg PrefitDataFraction(double data_ratio = 0.0) ;
RooCmdArg Optimize(Int_t flag = 2);

class EvalBackend : public RooCmdArg {
public:
   enum class Value { Legacy, Cpu, Cuda, Codegen, CodegenNoGrad };

   EvalBackend(Value value);

   EvalBackend(std::string const &name);

   static EvalBackend Legacy()
// Deprecation macro skipped when building RooFit itself, so we don't get
// warnings when unit testing deprecated features.
#ifndef ROOFIT_BUILDS_ITSELF
      R__DEPRECATED(6, 44,
                    "The legacy evaluation backend will be removed in ROOT 6.44. "
                    "Use the default \"cpu\" evaluation backend, i.e. simply don't pass any EvalBackend() "
                    "command argument.")
#endif
         ;
   static EvalBackend Cpu();
   static EvalBackend Cuda();
   static EvalBackend Codegen();
   static EvalBackend CodegenNoGrad();

   Value value() const { return static_cast<Value>(getInt(0)); }

   bool operator==(EvalBackend const &other) const { return value() == other.value(); }

   bool operator!=(EvalBackend const &other) const { return value() != other.value(); }

   std::string name() const;

   static Value &defaultValue();
private:
   static Value toValue(std::string const& name);
   static std::string toName(Value value);
};

////////////////////////////////////////////////////////////////////////////////
/// Create a RooCmdArg to declare conditional observables.
/// \param[in] argsOrArgSet Can either be one or more RooRealVar with the
//                          observables or a single RooArgSet containing them.
template<class... Args_t>
RooCmdArg ConditionalObservables(Args_t &&... argsOrArgSet) {
  RooCmdArg out{"ProjectedObservables",0};
  out.setSet(0, RooArgSet{std::forward<Args_t>(argsOrArgSet)...});
  return out;
}

// obsolete, for backward compatibility
template<class... Args_t>
RooCmdArg ProjectedObservables(Args_t &&... argsOrArgSet) {
  return ConditionalObservables(std::forward<Args_t>(argsOrArgSet)...);
}

RooCmdArg Verbose(bool flag=true) ;
RooCmdArg Save(bool flag=true) ;
RooCmdArg Timer(bool flag=true) ;
RooCmdArg PrintLevel(Int_t code) ;
RooCmdArg Warnings(bool flag=true) ;
RooCmdArg Strategy(Int_t code) ;
RooCmdArg InitialHesse(bool flag=true) ;
RooCmdArg Hesse(bool flag=true) ;
RooCmdArg Minos(bool flag=true) ;
RooCmdArg Minos(const RooArgSet& minosArgs) ;
RooCmdArg SplitRange(bool flag=true) ;
RooCmdArg SumCoefRange(const char* rangeName) ;
RooCmdArg Constrain(const RooArgSet& params) ;
RooCmdArg MaxCalls(int n) ;

template<class... Args_t>
RooCmdArg GlobalObservables(Args_t &&... argsOrArgSet) {
  RooCmdArg out{"GlobalObservables",0};
  out.setSet(0, RooArgSet{std::forward<Args_t>(argsOrArgSet)...});
  return out;
}
RooCmdArg GlobalObservablesSource(const char* sourceName);
RooCmdArg GlobalObservablesTag(const char* tagName) ;
RooCmdArg ExternalConstraints(const RooArgSet& constraintPdfs) ;
RooCmdArg PrintEvalErrors(Int_t numErrors) ;
RooCmdArg EvalErrorWall(bool flag) ;
RooCmdArg SumW2Error(bool flag) ;
RooCmdArg AsymptoticError(bool flag) ;
RooCmdArg CloneData(bool flag) ;
RooCmdArg Integrate(bool flag) ;
RooCmdArg Minimizer(const char* type, const char* alg=nullptr) ;
RooCmdArg Offset(std::string const& mode);
// The const char * overload is necessary, otherwise the compiler will cast a
// C-Style string to a bool and choose the Offset(bool) overload if one
// calls for example Offset("off").
inline RooCmdArg Offset(const char * mode) { return Offset(std::string(mode)); }
// For backwards compatibility
inline RooCmdArg Offset(bool flag=true) { return flag ? Offset("initial") : Offset("off"); }
RooCmdArg RecoverFromUndefinedRegions(double strength);
/** @} */

// RooAbsPdf::paramOn arguments
RooCmdArg Label(const char* str) ;
RooCmdArg Layout(double xmin, double xmax=0.99, double ymin=0.95) ;
RooCmdArg Parameters(const RooArgSet& params) ;
RooCmdArg ShowConstants(bool flag=true) ;

// RooTreeData::statOn arguments
RooCmdArg What(const char* str) ;

// RooProdPdf::ctor arguments
RooCmdArg Conditional(const RooArgSet& pdfSet, const RooArgSet& depSet, bool depsAreCond=false) ;

/**
 * \defgroup Generating Arguments for generating data
 * @{
 */
// RooAbsPdf::generate arguments
RooCmdArg ProtoData(const RooAbsData& protoData, bool randomizeOrder=false, bool resample=false) ;
RooCmdArg NumEvents(Int_t numEvents) ;
RooCmdArg NumEvents(double numEvents) ;
RooCmdArg AutoBinned(bool flag=true) ;
RooCmdArg GenBinned(const char* tag) ;
RooCmdArg AllBinned() ;
RooCmdArg ExpectedData(bool flag=true) ;
RooCmdArg Asimov(bool flag=true) ;

/** @} */

// RooAbsRealLValue::createHistogram arguments
RooCmdArg YVar(const RooAbsRealLValue& var, const RooCmdArg& arg={}) ;
RooCmdArg ZVar(const RooAbsRealLValue& var, const RooCmdArg& arg={}) ;
RooCmdArg AxisLabel(const char* name) ;
RooCmdArg Scaling(bool flag) ;


// RooAbsReal::createHistogram arguments
RooCmdArg IntrinsicBinning(bool flag=true) ;

// RooAbsReal::createIntegral arguments
template<class... Args_t>
RooCmdArg NormSet(Args_t &&... argsOrArgSet) {
  RooCmdArg out{"NormSet",0};
  out.setSet(0, RooArgSet{std::forward<Args_t>(argsOrArgSet)...});
  return out;
}
RooCmdArg NumIntConfig(const RooNumIntConfig& cfg) ;

// RooMCStudy::ctor arguments
RooCmdArg Silence(bool flag=true) ;
RooCmdArg FitModel(RooAbsPdf& pdf) ;
RooCmdArg FitOptions(const RooCmdArg& arg1                ,const RooCmdArg& arg2={},
                     const RooCmdArg& arg3={},const RooCmdArg& arg4={},
                     const RooCmdArg& arg5={},const RooCmdArg& arg6={}) ;
RooCmdArg Binned(bool flag=true) ;

// RooMCStudy::plot* arguments
RooCmdArg Frame(const RooCmdArg& arg1                ,const RooCmdArg& arg2={},
                const RooCmdArg& arg3={},const RooCmdArg& arg4={},
                const RooCmdArg& arg5={},const RooCmdArg& arg6={}) ;
RooCmdArg FrameBins(Int_t nbins) ;
RooCmdArg FrameRange(double xlo, double xhi) ;
RooCmdArg FitGauss(bool flag=true) ;

// RooRealVar::format arguments
RooCmdArg AutoPrecision(Int_t ndigit=2) ;
RooCmdArg FixedPrecision(Int_t ndigit=2) ;
RooCmdArg TLatexStyle(bool flag=true) ;
RooCmdArg LatexStyle(bool flag=true) ;
RooCmdArg LatexTableStyle(bool flag=true) ;
RooCmdArg VerbatimName(bool flag=true) ;

// RooMsgService::addReportingStream arguments
RooCmdArg Topic(Int_t topic) ;
RooCmdArg ObjectName(const char* name) ;
RooCmdArg ClassName(const char* name) ;
RooCmdArg BaseClassName(const char* name) ;
RooCmdArg TagName(const char* name) ;
RooCmdArg OutputStream(std::ostream& os) ;
RooCmdArg Prefix(bool flag) ;
RooCmdArg Color(TColorNumber color) ;

// RooWorkspace::import() arguments
RooCmdArg RenameConflictNodes(const char* suffix, bool renameOrigNodes=false) ;
RooCmdArg RenameAllNodes(const char* suffix) ;
RooCmdArg RenameAllVariables(const char* suffix) ;
RooCmdArg RenameAllVariablesExcept(const char* suffix,const char* exceptionList) ;
RooCmdArg RenameVariable(const char* inputName, const char* outputName) ;
RooCmdArg Rename(const char* suffix) ;
RooCmdArg RecycleConflictNodes(bool flag=true) ;
RooCmdArg Embedded(bool flag=true) ;
RooCmdArg NoRecursion(bool flag=true) ;

// RooSimCloneTool::build() arguments
RooCmdArg SplitParam(const char* varname, const char* catname) ;
RooCmdArg SplitParam(const RooRealVar& var, const RooAbsCategory& cat) ;
RooCmdArg SplitParamConstrained(const char* varname, const char* catname, const char* rsname) ;
RooCmdArg SplitParamConstrained(const RooRealVar& var, const RooAbsCategory& cat, const char* rsname) ;
RooCmdArg Restrict(const char* catName, const char* stateNameList) ;

// RooAbsPdf::createCdf() arguments
RooCmdArg SupNormSet(const RooArgSet& nset) ;
RooCmdArg ScanParameters(Int_t nbins,Int_t intOrder) ;
RooCmdArg ScanNumCdf() ;
RooCmdArg ScanAllCdf() ;
RooCmdArg ScanNoCdf() ;

// Generic container arguments (to be able to supply more command line arguments)
RooCmdArg MultiArg(const RooCmdArg& arg1, const RooCmdArg& arg2,
         const RooCmdArg& arg3={},const RooCmdArg& arg4={},
         const RooCmdArg& arg5={},const RooCmdArg& arg6={},
         const RooCmdArg& arg7={},const RooCmdArg& arg8={}) ;

RooConstVar& RooConst(double val) ;

// End group CmdArgs:
/**
 * @}
 */

/**
 * \defgroup Residuals Residuals and pulls of binned data vs. a fitted model
 *
 * RooFit::makeResidHist() and RooFit::makePullHist() construct per-bin
 * residual or pull distributions of binned data with respect to a fitted
 * model. In contrast to RooPlot::residHist() and RooPlot::pullHist(),
 * which interpolate or average the curve of a plotted pdf, these functions
 * integrate the model itself exactly over each bin. This avoids the
 * biased residuals (the "residual wiggle" and inflated \f$\chi^2\f$ values)
 * that appear when a sharply peaked pdf is compared with binned data.
 * @{
 */

/// Creates a RooHist with the per-bin residuals `data` - `fitModel`.
///
/// For each bin, the model expectation is the model integrated exactly over
/// the bin (with RooAbsReal::createIntegral()), normalized to the weight of
/// the data inside the normalization range. Unbinned input data is binned
/// internally. The supported RooCmdArgs are:
///   - `Binning(binning)` / `Binning(nbins)` / `Binning(nbins, xlo, xhi)` :
///     the binning used to bin unbinned input data (default: the current
///     binning of the observable)
///   - `Range(lo, hi)` / `Range("name")` : the range that the model
///     expectation is normalized in. Only bins inside this range get points
///     in the result. Range("name") can be passed multiple times for a union
///     of named ranges (sidebands). Use it for sideband fits or blinded data,
///     so that the model is normalized in the same range as the fitted data.
///   - `Normalization(double)` : additional scale factor for the expectation
///     (default 1.0). The scale types of Normalization(scale, type) are not
///     supported.
///   - `DataError(RooAbsData::ErrorType)` : error model used for the point errors,
///     one of Poisson, SumW2 or Auto
///     (default: Auto, meaning SumW2 for weighted and Poisson for unweighted data)
///   - `Name(const char*)`, `Title(const char*)` : name and title of the created RooHist
///
/// \note The residuals are computed with respect to `fitModel` itself, not
/// with respect to the curve of a plotted pdf. Make sure that the arguments
/// are consistent with what is plotted (same normalization and range, same
/// error model via DataError()), or the residual/pull histogram will not
/// match the displayed data and curve.
///
/// \note If the model depends on observables beyond the data's observable
/// (e.g. a conditional pdf), they are evaluated at their current values.
/// There is no exact equivalent for a projection with ProjWData().
///
/// \note For internal use of RooAbsReal::createIntegral(), a scratch named
/// range is left on the model's observable after the call.
///
/// Only 1-dimensional data is supported, otherwise std::invalid_argument is
/// thrown; the same exception is thrown for empty input data, unknown named
/// ranges or binnings, or when the model has no support in the normalization
/// range. For higher-dimensional fits, project both the data and the
/// model onto the observable of interest:
/// ~~~{.cpp}
/// std::unique_ptr<RooAbsData> projData{data.reduce(RooArgSet{x})};
/// std::unique_ptr<RooAbsPdf> projModel{model.createProjection(RooArgSet{y})};
/// auto pulls = RooFit::makePullHist(*projModel, *projData);
/// ~~~
/// For a residual/pull histogram against one component of a composite pdf,
/// pass the component pdf itself, scaled by its coefficient:
/// ~~~{.cpp}
/// auto& background = static_cast<RooAbsPdf&>(*addPdf.pdfList().find("bkg"));
/// auto pulls = RooFit::makePullHist(background, data,
///                                   RooFit::Normalization(bkgFraction.getVal()));
/// ~~~
std::unique_ptr<RooHist>
makeResidHist(RooAbsReal &fitModel, RooAbsData const &data, RooCmdArg const &arg1 = {}, RooCmdArg const &arg2 = {},
              RooCmdArg const &arg3 = {}, RooCmdArg const &arg4 = {}, RooCmdArg const &arg5 = {},
              RooCmdArg const &arg6 = {}, RooCmdArg const &arg7 = {}, RooCmdArg const &arg8 = {});

/// Like makeResidHist(), but the residuals are divided by the corresponding
/// data uncertainty, creating a pull distribution.
std::unique_ptr<RooHist>
makePullHist(RooAbsReal &fitModel, RooAbsData const &data, RooCmdArg const &arg1 = {}, RooCmdArg const &arg2 = {},
             RooCmdArg const &arg3 = {}, RooCmdArg const &arg4 = {}, RooCmdArg const &arg5 = {},
             RooCmdArg const &arg6 = {}, RooCmdArg const &arg7 = {}, RooCmdArg const &arg8 = {});

/**
 * @}
 */

namespace Detail {

// Function to pack an arbitrary number of RooCmdArgs into a RooLinkedList. Implementation detail of many high-level RooFit functions.
template <typename... Args>
inline std::unique_ptr<RooLinkedList> createCmdList(RooCmdArg const* arg1, Args &&...args)
{
   auto cmdList = std::make_unique<RooLinkedList>();
   for (auto &arg : {arg1, static_cast<RooCmdArg const *>(args)...}) {
      cmdList->Add(const_cast<RooCmdArg *>(arg));
   }
   return cmdList;
}

inline std::unique_ptr<RooLinkedList> createCmdList()
{
   return std::make_unique<RooLinkedList>();
}

inline std::unique_ptr<RooLinkedList> createCmdList(RooLinkedList const *cmdList)
{
   auto cmdListCopy = std::make_unique<RooLinkedList>();
   for (auto *arg : *cmdList) {
      cmdListCopy->Add(arg);
   }
   return cmdListCopy;
}

// RooFit-internal helper struct to build a map object that only uses
// std::vector, which can be implicitly converted to std::map in C++. Used to
// avoid std::map in pythonizations.
template <class Key_t, class Val_t>
struct FlatMap {
   std::vector<Key_t> keys;
   std::vector<Val_t> vals;
};

template <class Key_t, class Val_t>
auto flatMapToStdMap(FlatMap<Key_t, Val_t> const& flatMap) {
   std::map<Key_t, Val_t> out;
   for (std::size_t i = 0; i < flatMap.keys.size(); ++i) {
      out[flatMap.keys[i]] = flatMap.vals[i];
   }
   return out;
}

// Internal variant of Slice(), Import(), and Link(), that take flat maps instead of std::map.
RooCmdArg SliceFlatMap(FlatMap<RooCategory *, std::string> const &args);
RooCmdArg ImportFlatMap(FlatMap<std::string, RooDataHist *> const &args);
RooCmdArg ImportFlatMap(FlatMap<std::string, TH1 *> const &args);
RooCmdArg ImportFlatMap(FlatMap<std::string, RooDataSet *> const &args);
RooCmdArg LinkFlatMap(FlatMap<std::string, RooAbsData *> const &args);

} // namespace Detail

} // namespace RooFit

namespace RooFitShortHand {

RooConstVar& C(double value);

} // namespace RooFitShortHand

#endif
