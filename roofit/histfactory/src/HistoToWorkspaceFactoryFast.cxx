// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////

/** \class RooStats::HistFactory::HistoToWorkspaceFactoryFast
 * \ingroup HistFactory
 * This class provides helper functions for creating likelihood models from histograms.
 * It is used by RooStats::HistFactory::MakeModelAndMeasurementFast.
 *
 * A tutorial showing how to create a HistFactory model is hf001_example.C
 */


#include <RooAddition.h>
#include <RooBinWidthFunction.h>
#include <RooBinning.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFit/ModelConfig.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooGamma.h>
#include <RooGaussian.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooHistFunc.h>
#include <RooMultiVarGaussian.h>
#include <RooNumIntConfig.h>
#include <RooPoisson.h>
#include <RooPolyVar.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooProfileLL.h>
#include <RooRandom.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>

#include "RooStats/RooStatsUtils.h"
#include "RooStats/HistFactory/PiecewiseInterpolation.h"
#include "RooStats/HistFactory/ParamHistFunc.h"
#include "RooStats/AsymptoticCalculator.h"

#include "HFMsgService.h"

#include "TH1.h"
#include "TStopwatch.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"

// specific to this package
#include <RooStats/HistFactory/Detail/HistFactoryImpl.h>
#include "RooStats/HistFactory/LinInterpVar.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistFactoryException.h"

#include <algorithm>
#include <memory>
#include <utility>

constexpr double alphaLow = -5.0;
constexpr double alphaHigh = 5.0;

std::vector<double> histToVector(TH1 const &hist)
{
   // Must get the full size of the TH1 (No direct method to do this...)
   int numBins = hist.GetNbinsX() * hist.GetNbinsY() * hist.GetNbinsZ();
   std::vector<double> out(numBins);
   int histIndex = 0;
   for (int i = 0; i < numBins; ++i) {
      while (hist.IsBinUnderflow(histIndex) || hist.IsBinOverflow(histIndex)) {
         ++histIndex;
      }
      out[i] = hist.GetBinContent(histIndex);
      ++histIndex;
   }
   return out;
}

// use this order for safety on library loading
using namespace RooStats;
using std::string, std::vector, std::make_unique, std::pair, std::unique_ptr, std::map;

using namespace RooStats::HistFactory::Detail;
using namespace RooStats::HistFactory::Detail::MagicConstants;

ClassImp(RooStats::HistFactory::HistoToWorkspaceFactoryFast);

namespace RooStats{
namespace HistFactory{

  HistoToWorkspaceFactoryFast::HistoToWorkspaceFactoryFast(RooStats::HistFactory::Measurement& measurement) :
    HistoToWorkspaceFactoryFast{measurement, Configuration{}} {}

  HistoToWorkspaceFactoryFast::HistoToWorkspaceFactoryFast(RooStats::HistFactory::Measurement& measurement,
                                                           Configuration const& cfg) :
    fSystToFix( measurement.GetConstantParams() ),
    fParamValues( measurement.GetParamValues() ),
    fNomLumi( measurement.GetLumi() ),
    fLumiError( measurement.GetLumi()*measurement.GetLumiRelErr() ),
    fLowBin( measurement.GetBinLow() ),
    fHighBin( measurement.GetBinHigh() ),
    fCfg{cfg} {

    // Set Preprocess functions
    SetFunctionsToPreprocess( measurement.GetPreprocessFunctions() );

  }

  void HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement( const std::string& ModelName, RooWorkspace* ws_single, Measurement& measurement ) {

    // Configure a workspace by doing any
    // necessary post-processing and by
    // creating a ModelConfig

    // Make a ModelConfig and configure it
    ModelConfig * proto_config = static_cast<ModelConfig *>(ws_single->obj("ModelConfig"));
    if( proto_config == nullptr ) {
      std::cout << "Error: Did not find 'ModelConfig' object in file: " << ws_single->GetName()
      << std::endl;
      throw hf_exc();
    }

    if( measurement.GetPOIList().empty() ) {
      cxcoutWHF << "No Parametetrs of interest are set" << std::endl;
    }


    std::stringstream sstream;
    sstream << "Setting Parameter(s) of Interest as: ";
    for(auto const& item : measurement.GetPOIList()) {
      sstream << item << " ";
    }
    cxcoutIHF << sstream.str() << std::endl;

    RooArgSet params;
    for(auto const& poi_name : measurement.GetPOIList()) {
      if(RooRealVar* poi = (RooRealVar*) ws_single->var(poi_name)){
        params.add(*poi);
      }
      else {
   std::cout << "WARNING: Can't find parameter of interest: " << poi_name
        << " in Workspace. Not setting in ModelConfig." << std::endl;
   //throw hf_exc();
      }
    }
    proto_config->SetParametersOfInterest(params);

    // Name of an 'edited' model, if necessary
    std::string NewModelName = "newSimPdf"; // <- This name is hard-coded in HistoToWorkspaceFactoryFast::EditSyt.  Probably should be changed to : std::string("new") + ModelName;

    // Set the ModelConfig's Params of Interest
    RooAbsData* expData = ws_single->data("asimovData");
    if( !expData ) {
      std::cout << "Error: Failed to find dataset: " << expData
      << " in workspace" << std::endl;
      throw hf_exc();
    }
    if(!measurement.GetPOIList().empty()){
      proto_config->GuessObsAndNuisance(*expData, RooMsgService::instance().isActive(nullptr, RooFit::HistFactory, RooFit::INFO));
    }

    // Now, let's loop over any additional asimov datasets
    // that we need to make

    // Get the pdf
    // Notice that we get the "new" pdf, this is the one that is
    // used in the creation of these asimov datasets since they
    // are fitted (or may be, at least).
    RooAbsPdf* pdf = ws_single->pdf(NewModelName);
    if( !pdf ) pdf = ws_single->pdf( ModelName );
    const RooArgSet* observables = ws_single->set("observables");

    // Create a SnapShot of the nominal values
    std::string SnapShotName = "NominalParamValues";
    ws_single->saveSnapshot(SnapShotName, ws_single->allVars());

    for( unsigned int i=0; i<measurement.GetAsimovDatasets().size(); ++i) {

      // Set the variable values and "const" ness with the workspace
      RooStats::HistFactory::Asimov& asimov = measurement.GetAsimovDatasets().at(i);
      std::string AsimovName = asimov.GetName();

      cxcoutPHF << "Generating additional Asimov Dataset: " << AsimovName << std::endl;
      asimov.ConfigureWorkspace(ws_single);
      std::unique_ptr<RooAbsData> asimov_dataset{AsymptoticCalculator::GenerateAsimovData(*pdf, *observables)};

      cxcoutPHF << "Importing Asimov dataset" << std::endl;
      bool failure = ws_single->import(*asimov_dataset, RooFit::Rename(AsimovName.c_str()));
      if( failure ) {
        std::cout << "Error: Failed to import Asimov dataset: " << AsimovName
        << std::endl;
   throw hf_exc();
      }

      // Load the snapshot at the end of every loop iteration
      // so we start each loop with a "clean" snapshot
      ws_single->loadSnapshot(SnapShotName.c_str());
    }

    // Cool, we're done
    return; // ws_single;
  }


  // We want to eliminate this interface and use the measurement directly
  RooFit::OwningPtr<RooWorkspace> HistoToWorkspaceFactoryFast::MakeSingleChannelModel( Measurement& measurement, Channel& channel ) {

    // This is a pretty light-weight wrapper function
    //
    // Take a fully configured measurement as well as
    // one of its channels
    //
    // Return a workspace representing that channel
    // Do this by first creating a vector of EstimateSummary's
    // and this by configuring the workspace with any post-processing

    // Get the channel's name
    string ch_name = channel.GetName();

    // Create a workspace for a SingleChannel from the Measurement Object
    std::unique_ptr<RooWorkspace> ws_single{this->MakeSingleChannelWorkspace(measurement, channel)};
    if( ws_single == nullptr ) {
      cxcoutF(HistFactory) << "Error: Failed to make Single-Channel workspace for channel: " << ch_name
      << " and measurement: " << measurement.GetName() << std::endl;
      throw hf_exc();
    }

    // Finally, configure that workspace based on
    // properties of the measurement
    HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement( "model_"+ch_name, ws_single.get(), measurement );

    return RooFit::makeOwningPtr(std::move(ws_single));

  }

  RooFit::OwningPtr<RooWorkspace> HistoToWorkspaceFactoryFast::MakeCombinedModel( Measurement& measurement ) {

    // This function takes a fully configured measurement
    // which may contain several channels and returns
    // a workspace holding the combined model
    //
    // This can be used, for example, within a script to produce
    // a combined workspace on-the-fly
    //
    // This is a static function (for now) to make
    // it a one-liner


    Configuration config;
    return MakeCombinedModel(measurement,config);
  }

  RooFit::OwningPtr<RooWorkspace> HistoToWorkspaceFactoryFast::MakeCombinedModel( Measurement& measurement, const Configuration& config) {

    // This function takes a fully configured measurement
    // which may contain several channels and returns
    // a workspace holding the combined model
    //
    // This can be used, for example, within a script to produce
    // a combined workspace on-the-fly
    //
    // This is a static function (for now) to make
    // it a one-liner

    RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::INFO, 0u, RooFit::ObjectHandling, false);

    // First, we create an instance of a HistFactory
    HistoToWorkspaceFactoryFast histFactory(measurement, config);

    // Loop over the channels and create the individual workspaces
    vector<std::unique_ptr<RooWorkspace>> channel_workspaces;
    vector<string>        channel_names;

    for(HistFactory::Channel& channel : measurement.GetChannels()) {

      if( ! channel.CheckHistograms() ) {
        cxcoutFHF << "MakeModelAndMeasurementsFast: Channel: " << channel.GetName()
            << " has uninitialized histogram pointers" << std::endl;
        throw hf_exc();
      }

      string ch_name = channel.GetName();
      channel_names.push_back(ch_name);

      // GHL: Renaming to 'MakeSingleChannelWorkspace'
      channel_workspaces.emplace_back(histFactory.MakeSingleChannelModel(measurement, channel));
    }


    // Now, combine the individual channel workspaces to
    // form the combined workspace
    std::unique_ptr<RooWorkspace> ws{histFactory.MakeCombinedModel( channel_names, channel_workspaces )};


    // Configure the workspace
    HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement("simPdf", ws.get(), measurement);

    // Done.  Return the pointer
    return RooFit::makeOwningPtr(std::move(ws));

  }

namespace {

template <class Arg_t, typename... Args_t>
Arg_t &emplace(RooWorkspace &ws, std::string const &name, Args_t &&...args)
{
   Arg_t arg{name.c_str(), name.c_str(), std::forward<Args_t>(args)...};
   ws.import(arg, RooFit::RecycleConflictNodes());
   return *dynamic_cast<Arg_t *>(ws.arg(name));
}

} // namespace

/// Create observables of type RooRealVar. Creates 1 to 3 observables, depending on the type of the histogram.
RooArgList HistoToWorkspaceFactoryFast::createObservables(const TH1 *hist, RooWorkspace &proto) const {
  RooArgList observables;

  for (unsigned int idx=0; idx < fObsNameVec.size(); ++idx) {
    if (!proto.var(fObsNameVec[idx])) {
      const TAxis *axis = (idx == 0) ? hist->GetXaxis() : (idx == 1 ? hist->GetYaxis() : hist->GetZaxis());
      int nbins = axis->GetNbins();
      // create observable
      RooRealVar &obs = emplace<RooRealVar>(proto, fObsNameVec[idx], axis->GetXmin(), axis->GetXmax());
      if(strlen(axis->GetTitle())>0) obs.SetTitle(axis->GetTitle());
      obs.setBins(nbins);
      if (axis->IsVariableBinSize()) {
        RooBinning binning(nbins, axis->GetXbins()->GetArray());
        obs.setBinning(binning);
      }
    }

    observables.add(*proto.var(fObsNameVec[idx]));
  }

  return observables;
}

  /// Create the nominal hist function from `hist`, and register it in the workspace.
  RooHistFunc* HistoToWorkspaceFactoryFast::MakeExpectedHistFunc(const TH1* hist,RooWorkspace &proto, string prefix,
      const RooArgList& observables) const {
    if(hist) {
      cxcoutI(HistFactory) << "processing hist " << hist->GetName() << std::endl;
    } else {
      cxcoutF(HistFactory) << "hist is empty" << std::endl;
      R__ASSERT(hist != nullptr);
      return nullptr;
    }

    // determine histogram dimensionality
    unsigned int histndim(1);
    std::string classname = hist->ClassName();
    if      (classname.find("TH1")==0) { histndim=1; }
    else if (classname.find("TH2")==0) { histndim=2; }
    else if (classname.find("TH3")==0) { histndim=3; }
    R__ASSERT( histndim==fObsNameVec.size() );

    prefix += "_Hist_alphanominal";

    RooDataHist histDHist(prefix + "DHist","",observables,hist);

    return &emplace<RooHistFunc>(proto, prefix, observables,histDHist,0);
  }

  namespace {

  void makeGaussianConstraint(RooAbsArg& param, RooWorkspace& proto, bool isUniform,
                              std::vector<std::string> & constraintTermNames) {
      std::string paramName = param.GetName();
      std::string nomName = "nom_" + paramName;
      std::string constraintName = paramName + "Constraint";

      // do nothing if the constraint term already exists
      if(proto.pdf(constraintName)) return;

      // case systematic is uniform (assume they are like a Gaussian but with
      // a large width (100 instead of 1)
      const double gaussSigma = isUniform ? 100. : 1.0;
      if (isUniform) {
         cxcoutIHF << "Added a uniform constraint for " << paramName << " as a Gaussian constraint with a very large sigma " << std::endl;
      }

      constraintTermNames.emplace_back(constraintName);
      RooRealVar &paramVar = *proto.var(paramName);
      RooRealVar &nomParam = emplace<RooRealVar>(proto, nomName, 0., -10., 10.);
      nomParam.setConstant();
      emplace<RooGaussian>(proto, constraintName, paramVar, nomParam, gaussSigma);
      paramVar.setError(gaussSigma); // give param initial error to match gaussSigma
      const_cast<RooArgSet*>(proto.set("globalObservables"))->add(nomParam);
  }

  /// Make list of abstract parameters that interpolate in space of variations.
  RooArgList makeInterpolationParameters(std::vector<HistoSys> const& histoSysList, RooWorkspace& proto) {
    RooArgList params( ("alpha_Hist") );

    for(auto const& histoSys : histoSysList) {
      params.add(getOrCreate<RooRealVar>(proto, "alpha_" + histoSys.GetName(), alphaLow, alphaHigh));
    }

    return params;
  }

  /// Create a linear interpolation object that holds nominal and systematics, import it into the workspace,
  /// and return a pointer to it.
  RooAbsArg* makeLinInterp(RooArgList const& interpolationParams,
                           RooHistFunc* nominalFunc,
                           RooWorkspace& proto, const std::vector<HistoSys>& histoSysList,
                           const string& prefix,
                           const RooArgList& obsList) {

    // now make function that linearly interpolates expectation between variations
    // get low/high variations to interpolate between
    std::vector<double> low;
    std::vector<double> high;
    RooArgSet lowSet;
    RooArgSet highSet;
    for(unsigned int j=0; j<histoSysList.size(); ++j){
      std::string str = prefix + "_" + std::to_string(j);

      const HistoSys& histoSys = histoSysList.at(j);
      auto lowDHist = std::make_unique<RooDataHist>(str+"lowDHist","",obsList, histoSys.GetHistoLow());
      auto highDHist = std::make_unique<RooDataHist>(str+"highDHist","",obsList, histoSys.GetHistoHigh());
      lowSet.addOwned(std::make_unique<RooHistFunc>((str+"low").c_str(),"",obsList,std::move(lowDHist),0));
      highSet.addOwned(std::make_unique<RooHistFunc>((str+"high").c_str(),"",obsList,std::move(highDHist),0));
    }

    // this is sigma(params), a piece-wise linear interpolation
    PiecewiseInterpolation interp(prefix.c_str(),"",*nominalFunc,lowSet,highSet,interpolationParams);
    interp.setPositiveDefinite();
    interp.setAllInterpCodes(4); // LM: change to 4 (piece-wise linear to 6th order polynomial interpolation + linear extrapolation )
    // KC: interpo codes 1 etc. don't have proper analytic integral.
    RooArgSet obsSet(obsList);
    interp.setBinIntegrator(obsSet);
    interp.forceNumInt();

    proto.import(interp, RooFit::RecycleConflictNodes()); // individual params have already been imported in first loop of this function

    return proto.arg(prefix);
  }

  }

  // GHL: Consider passing the NormFactor list instead of the entire sample
  std::unique_ptr<RooProduct> HistoToWorkspaceFactoryFast::CreateNormFactor(RooWorkspace& proto, string& channel, string& sigmaEpsilon, Sample& sample, bool doRatio){

    std::vector<string> prodNames;

    vector<NormFactor> normList = sample.GetNormFactorList();
    vector<string> normFactorNames;
    vector<string> rangeNames;

    string overallNorm_times_sigmaEpsilon = sample.GetName() + "_" + channel + "_scaleFactors";
    auto sigEps = proto.arg(sigmaEpsilon);
    assert(sigEps);
    auto normFactor = std::make_unique<RooProduct>(overallNorm_times_sigmaEpsilon.c_str(), overallNorm_times_sigmaEpsilon.c_str(), RooArgList(*sigEps));

    if(!normList.empty()){

      for(NormFactor &norm : normList) {
        string varname = norm.GetName();
        if(doRatio) {
          varname += "_" + channel;
        }

        // GHL: Check that the NormFactor doesn't already exist
        //      (it may have been created as a function expression
        //       during preprocessing)
        std::stringstream range;
        range << "[" << norm.GetVal() << "," << norm.GetLow() << "," << norm.GetHigh() << "]";

        if( proto.obj(varname) == nullptr) {
          cxcoutI(HistFactory) << "making normFactor: " << norm.GetName() << std::endl;
          // remove "doRatio" and name can be changed when ws gets imported to the combined model.
          emplace<RooRealVar>(proto, varname, norm.GetVal(), norm.GetLow(), norm.GetHigh());
          proto.var(varname)->setError(0); // ensure factor is assigned an initial error, even if its zero
        }

        prodNames.push_back(varname);
        rangeNames.push_back(range.str());
        normFactorNames.push_back(varname);
      }


      for (const auto& name : prodNames) {
        auto arg = proto.arg(name);
        assert(arg);
        normFactor->addTerm(arg);
      }

    }

    unsigned int rangeIndex=0;
    for( vector<string>::iterator nit = normFactorNames.begin(); nit!=normFactorNames.end(); ++nit){
      if( count (normFactorNames.begin(), normFactorNames.end(), *nit) > 1 ){
        cxcoutI(HistFactory) <<"<NormFactor Name =\""<<*nit<<"\"> is duplicated for <Sample Name=\""
            << sample.GetName() << "\">, but only one factor will be included.  \n Instead, define something like"
            << "\n\t<Function Name=\""<<*nit<<"Squared\" Expression=\""<<*nit<<"*"<<*nit<<"\" Var=\""<<*nit<<rangeNames.at(rangeIndex)
            << "\"> \nin your top-level XML's <Measurement> entry and use <NormFactor Name=\""<<*nit<<"Squared\" in your channel XML file."<< std::endl;
      }
      ++rangeIndex;
    }

    return normFactor;
  }

   void HistoToWorkspaceFactoryFast::AddConstraintTerms(RooWorkspace& proto, Measurement & meas, string prefix,
                         string interpName,
                         std::vector<OverallSys>& systList,
                         vector<string>& constraintTermNames,
                         vector<string>& totSystTermNames) {

    // add variables for all the relative overall uncertainties we expect
    totSystTermNames.push_back(prefix);

    RooArgSet params(prefix.c_str());
    vector<double> lowVec;
    vector<double> highVec;

    std::map<std::string, double>::iterator itconstr;
    for(unsigned int i = 0; i < systList.size(); ++i) {

      OverallSys& sys = systList.at(i);
      std::string strname = sys.GetName();
      const char * name = strname.c_str();

      // case of no systematic (is it possible)
      if (meas.GetNoSyst().count(sys.GetName()) > 0 ) {
        cxcoutI(HistFactory) << "HistoToWorkspaceFast::AddConstraintTerm - skip systematic " << sys.GetName() << std::endl;
        continue;
      }
      // case systematic is a  gamma constraint
      if (meas.GetGammaSyst().count(sys.GetName()) > 0 ) {
         double relerr = meas.GetGammaSyst().find(sys.GetName() )->second;
         if (relerr <= 0) {
           cxcoutI(HistFactory) << "HistoToWorkspaceFast::AddConstraintTerm - zero uncertainty assigned - skip systematic  " << sys.GetName() << std::endl;
           continue;
         }
         const double tauVal = 1./(relerr*relerr);
         const double sqtau = 1./relerr;
         RooRealVar &beta = emplace<RooRealVar>(proto, "beta_" + strname, 1., 0., 10.);
         // the global observable (y_s)
         RooRealVar &yvar = emplace<RooRealVar>(proto, "nom_" + std::string{beta.GetName()}, tauVal, 0., 10.);
         // the rate of the gamma distribution (theta)
         RooRealVar &theta = emplace<RooRealVar>(proto, "theta_" + strname, 1./tauVal);
         // find alpha as function of beta
         RooPolyVar &alphaOfBeta = emplace<RooPolyVar>(proto, "alphaOfBeta_" + strname, beta, RooArgList{-sqtau, sqtau});

         // add now the constraint itself  Gamma_beta_constraint(beta, y+1, tau, 0 )
         // build the gamma parameter k = as y_s + 1
         RooAddition &kappa = emplace<RooAddition>(proto, "k_" + std::string{yvar.GetName()}, RooArgList{yvar, 1.0});
         RooGamma &gamma = emplace<RooGamma>(proto, std::string{beta.GetName()} + "Constraint", beta, kappa, theta, RooFit::RooConst(0.0));
         if (RooMsgService::instance().isActive(nullptr, RooFit::HistFactory, RooFit::DEBUG)) {
           alphaOfBeta.Print("t");
           gamma.Print("t");
         }
         constraintTermNames.push_back(gamma.GetName());
         // set global observables
         yvar.setConstant(true);
         const_cast<RooArgSet*>(proto.set("globalObservables"))->add(yvar);

         // add alphaOfBeta in the list of params to interpolate
         params.add(alphaOfBeta);
         cxcoutI(HistFactory) << "Added a gamma constraint for " << name << std::endl;

      }
      else {
         RooRealVar& alpha = getOrCreate<RooRealVar>(proto, prefix + sys.GetName(), 0, alphaLow, alphaHigh);
         // add the Gaussian constraint part
         const bool isUniform = meas.GetUniformSyst().count(sys.GetName()) > 0;
         makeGaussianConstraint(alpha, proto, isUniform, constraintTermNames);

         // check if exists a log-normal constraint
         if (meas.GetLogNormSyst().count(sys.GetName()) == 0 &&  meas.GetGammaSyst().count(sys.GetName()) == 0 ) {
            // just add the alpha for the parameters of the FlexibleInterpVar function
            params.add(alpha);
         }
                  // case systematic is a  log-normal constraint
         if (meas.GetLogNormSyst().count(sys.GetName()) > 0 ) {
            // log normal constraint for parameter
            const double relerr = meas.GetLogNormSyst().find(sys.GetName() )->second;

            RooFormulaVar &alphaOfBeta = emplace<RooFormulaVar>(
               proto, "alphaOfBeta_" + sys.GetName(), "x[0]*(pow(x[1],x[2])-1.)",
               RooArgList{emplace<RooRealVar>(proto, "tau_" + sys.GetName(), 1. / relerr),
                          emplace<RooRealVar>(proto, "kappa_" + sys.GetName(), 1. + relerr), alpha});

            cxcoutI(HistFactory) << "Added a log-normal constraint for " << name << std::endl;
            if (RooMsgService::instance().isActive(nullptr, RooFit::HistFactory, RooFit::DEBUG)) {
              alphaOfBeta.Print("t");
            }
            params.add(alphaOfBeta);
         }

      }
      // add low/high vectors
      lowVec.push_back(sys.GetLow());
      highVec.push_back(sys.GetHigh());

    }  // end sys loop

    if(!systList.empty()){
       // this is epsilon(alpha_j), a piece-wise linear interpolation
       //      LinInterpVar interp( (interpName).c_str(), "", params, 1., lowVec, highVec);

       assert(!params.empty());
       assert(lowVec.size() == params.size());

       FlexibleInterpVar interp( (interpName).c_str(), "", params, 1., lowVec, highVec);
       interp.setAllInterpCodes(4); // LM: change to 4 (piece-wise exponential to 6th order polynomial interpolation + exponential extrapolation )
       //interp.setAllInterpCodes(0); // simple linear interpolation
       proto.import(interp); // params have already been imported in first loop of this function
    } else{
       // some strange behavior if params,lowVec,highVec are empty.
       //cout << "WARNING: No OverallSyst terms" << std::endl;
       emplace<RooConstVar>(proto, interpName, 1.); // params have already been imported in first loop of this function
    }
  }


  void  HistoToWorkspaceFactoryFast::MakeTotalExpected(RooWorkspace& proto, const string& totName,
                         const vector<RooProduct*>& sampleScaleFactors, std::vector<vector<RooAbsArg*>>& sampleHistFuncs) const {
    assert(sampleScaleFactors.size() == sampleHistFuncs.size());

    // for ith bin calculate totN_i =  lumi * sum_j expected_j * syst_j

    if (fObsNameVec.empty() && !fObsName.empty())
      throw std::logic_error("HistFactory didn't process the observables correctly. Please file a bug report.");

    auto firstHistFunc = dynamic_cast<const RooHistFunc*>(sampleHistFuncs.front().front());
    if (!firstHistFunc) {
      auto piecewiseInt = dynamic_cast<const PiecewiseInterpolation*>(sampleHistFuncs.front().front());
      firstHistFunc = dynamic_cast<const RooHistFunc*>(piecewiseInt->nominalHist());
    }
    assert(firstHistFunc);

    // Prepare a function to divide all bin contents by bin width to get a density:
    auto &binWidth = emplace<RooBinWidthFunction>(proto, totName + "_binWidth", *firstHistFunc, true);

    // Loop through samples and create products of their functions:
    RooArgSet coefList;
    RooArgSet shapeList;
    for (unsigned int i=0; i < sampleHistFuncs.size(); ++i) {
      assert(!sampleHistFuncs[i].empty());
      coefList.add(*sampleScaleFactors[i]);

      std::vector<RooAbsArg*>& thisSampleHistFuncs = sampleHistFuncs[i];
      thisSampleHistFuncs.push_back(&binWidth);

      if (thisSampleHistFuncs.size() == 1) {
        // Just one function. Book it.
        shapeList.add(*thisSampleHistFuncs.front());
      } else {
        // Have multiple functions. We need to multiply them.
        std::string name = thisSampleHistFuncs.front()->GetName();
        auto pos = name.find("Hist_alpha");
        if (pos != std::string::npos) {
          name = name.substr(0, pos) + "shapes";
        } else if ( (pos = name.find("nominal")) != std::string::npos) {
          name = name.substr(0, pos) + "shapes";
        }

        RooProduct shapeProduct(name.c_str(), thisSampleHistFuncs.front()->GetTitle(), RooArgSet(thisSampleHistFuncs.begin(), thisSampleHistFuncs.end()));
        proto.import(shapeProduct, RooFit::RecycleConflictNodes());
        shapeList.add(*proto.function(name));
      }
    }

    // Sum all samples
    RooRealSumPdf tot(totName.c_str(), totName.c_str(), shapeList, coefList, true);
    tot.specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator")  ;
    tot.specialIntegratorConfig(true)->method2D().setLabel("RooBinIntegrator")  ;
    tot.specialIntegratorConfig(true)->methodND().setLabel("RooBinIntegrator")  ;
    tot.forceNumInt();

    // for mixed generation in RooSimultaneous
    tot.setAttribute("GenerateBinned"); // for use with RooSimultaneous::generate in mixed mode

    // Enable the binned likelihood optimization
    if(fCfg.binnedFitOptimization) {
      tot.setAttribute("BinnedLikelihood");
    }

    proto.import(tot, RooFit::RecycleConflictNodes());
  }

  //////////////////////////////////////////////////////////////////////////////

  void HistoToWorkspaceFactoryFast::PrintCovarianceMatrix(RooFitResult* result, RooArgSet* params, string filename){

    FILE* covFile = fopen ((filename).c_str(),"w");
    fprintf(covFile," ") ;
    for (auto const *myargi : static_range_cast<RooRealVar *>(*params)) {
      if(myargi->isConstant()) continue;
      fprintf(covFile," & %s",  myargi->GetName());
    }
    fprintf(covFile,"\\\\ \\hline \n" );
    for (auto const *myargi : static_range_cast<RooRealVar *>(*params)) {
      if(myargi->isConstant()) continue;
      fprintf(covFile,"%s", myargi->GetName());
      for (auto const *myargj : static_range_cast<RooRealVar *>(*params)) {
        if(myargj->isConstant()) continue;
        std::cout << myargi->GetName() << "," << myargj->GetName();
        fprintf(covFile, " & %.2f", result->correlation(*myargi, *myargj));
      }
      std::cout << std::endl;
      fprintf(covFile, " \\\\\n");
    }
    fclose(covFile);

  }


  ///////////////////////////////////////////////
  std::unique_ptr<RooWorkspace> HistoToWorkspaceFactoryFast::MakeSingleChannelWorkspace(Measurement& measurement, Channel& channel) {

    // check inputs (see JIRA-6890 )

    if (channel.GetSamples().empty()) {
      Error("MakeSingleChannelWorkspace",
          "The input Channel does not contain any sample - return a nullptr");
      return nullptr;
    }

    const TH1* channel_hist_template = channel.GetSamples().front().GetHisto();
    if (channel_hist_template == nullptr) {
      channel.CollectHistograms();
      channel_hist_template = channel.GetSamples().front().GetHisto();
    }
    if (channel_hist_template == nullptr) {
      std::ostringstream stream;
      stream << "The sample " << channel.GetSamples().front().GetName()
                   << " in channel " << channel.GetName() << " does not contain a histogram. This is the channel:\n";
      channel.Print(stream);
      Error("MakeSingleChannelWorkspace", "%s", stream.str().c_str());
      return nullptr;
    }

    if( ! channel.CheckHistograms() ) {
      std::cout << "MakeSingleChannelWorkspace: Channel: " << channel.GetName()
                      << " has uninitialized histogram pointers" << std::endl;
      throw hf_exc();
    }



    // Set these by hand inside the function
    vector<string> systToFix = measurement.GetConstantParams();
    bool doRatio=false;

    // to time the macro
    TStopwatch t;
    t.Start();
    //ES// string channel_name=summary[0].channel;
    string channel_name = channel.GetName();

    /// MB: reset observable names for each new channel.
    fObsNameVec.clear();

    /// MB: label observables x,y,z, depending on histogram dimensionality
    /// GHL: Give it the first sample's nominal histogram as a template
    ///      since the data histogram may not be present
    if (fObsNameVec.empty()) { GuessObsNameVec(channel_hist_template); }

    for ( unsigned int idx=0; idx<fObsNameVec.size(); ++idx ) {
      fObsNameVec[idx] = "obs_" + fObsNameVec[idx] + "_" + channel_name ;
    }

    if (fObsNameVec.empty()) {
      fObsName= "obs_" + channel_name; // set name ov observable
      fObsNameVec.push_back( fObsName );
    }

    if (fObsNameVec.empty() || fObsNameVec.size() > 3) {
      throw hf_exc("HistFactory is limited to 1- to 3-dimensional histograms.");
    }

    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\tStarting to process '"
        << channel_name << "' channel with " << fObsNameVec.size() << " observables"
        << "\n-----------------------------------------\n" << std::endl;

    //
    // our main workspace that we are using to construct the model
    //
    auto protoOwner = std::make_unique<RooWorkspace>(channel_name.c_str(), (channel_name+" workspace").c_str());
    RooWorkspace &proto = *protoOwner;
    auto proto_config = make_unique<ModelConfig>("ModelConfig", &proto);
    proto_config->SetWorkspace(proto);

    // preprocess functions
    for(auto const& func : fPreprocessFunctions){
      cxcoutI(HistFactory) << "will preprocess this line: " << func << std::endl;
      proto.factory(func);
      proto.Print();
    }

    RooArgSet likelihoodTerms("likelihoodTerms");
    RooArgSet constraintTerms("constraintTerms");
    vector<string> likelihoodTermNames;
    vector<string> constraintTermNames;
    vector<string> totSystTermNames;
    // All histogram functions to be multiplied in each sample
    std::vector<std::vector<RooAbsArg*>> allSampleHistFuncs;
    std::vector<RooProduct*> sampleScaleFactors;

    std::vector< pair<string,string> >   statNamePairs;
    std::vector< pair<const TH1*, std::unique_ptr<TH1>> > statHistPairs; // <nominal, error>
    const std::string statFuncName = "mc_stat_" + channel_name;

    string prefix;
    string range;

    /////////////////////////////
    // shared parameters
    // this is ratio of lumi to nominal lumi.  We will include relative uncertainty in model
    auto &lumiVar = getOrCreate<RooRealVar>(proto, "Lumi", fNomLumi, 0.0, 10 * fNomLumi);

    // only include a lumiConstraint if there's a lumi uncert, otherwise just set the lumi constant
    if(fLumiError != 0) {
      auto &nominalLumiVar = emplace<RooRealVar>(proto, "nominalLumi", fNomLumi, 0., fNomLumi + 10. * fLumiError);
      emplace<RooGaussian>(proto, "lumiConstraint", lumiVar, nominalLumiVar, fLumiError);
      proto.var("Lumi")->setError(fLumiError/fNomLumi); // give initial error value
      proto.var("nominalLumi")->setConstant();
      proto.defineSet("globalObservables","nominalLumi");
      //likelihoodTermNames.push_back("lumiConstraint");
      constraintTermNames.push_back("lumiConstraint");
    } else {
      proto.var("Lumi")->setConstant();
      proto.defineSet("globalObservables",RooArgSet()); // create empty set as is assumed it exists later
    }
    ///////////////////////////////////
    // loop through estimates, add expectation, floating bin predictions,
    // and terms that constrain floating to expectation via uncertainties
    // GHL: Loop over samples instead, which doesn't contain the data
    for (Sample& sample : channel.GetSamples()) {
      string overallSystName = sample.GetName() + "_" + channel_name + "_epsilon";

      string systSourcePrefix = "alpha_";

      // constraintTermNames and totSystTermNames are vectors that are passed
      // by reference and filled by this method
      AddConstraintTerms(proto,measurement, systSourcePrefix, overallSystName,
          sample.GetOverallSysList(), constraintTermNames , totSystTermNames);

      allSampleHistFuncs.emplace_back();
      std::vector<RooAbsArg*>& sampleHistFuncs = allSampleHistFuncs.back();

      // GHL: Consider passing the NormFactor list instead of the entire sample
      auto normFactors = CreateNormFactor(proto, channel_name, overallSystName, sample, doRatio);
      assert(normFactors);

      // Create the string for the object
      // that is added to the RooRealSumPdf
      // for this channel
//      string syst_x_expectedPrefix = "";

      // get histogram
      //ES// TH1* nominal = it->nominal;
      const TH1* nominal = sample.GetHisto();

      // MB : HACK no option to have both non-hist variations and hist variations ?
      // get histogram
      // GHL: Okay, this is going to be non-trivial.
      //      We will loop over histosys's, which contain both
      //      the low hist and the high hist together.

      // Logic:
      //        - If we have no HistoSys's, do part A
      //        - else, if the histo syst's don't match, return (we ignore this case)
      //        - finally, we take the syst's and apply the linear interpolation w/ constraint
      string expPrefix = sample.GetName() + "_" + channel_name;
      // create roorealvar observables
      RooArgList observables = createObservables(sample.GetHisto(), proto);
      RooHistFunc* nominalHistFunc = MakeExpectedHistFunc(sample.GetHisto(), proto, expPrefix, observables);
      assert(nominalHistFunc);

      if(sample.GetHistoSysList().empty()) {
        // If no HistoSys
        cxcoutI(HistFactory) << sample.GetName() + "_" + channel_name + " has no variation histograms " << std::endl;

        sampleHistFuncs.push_back(nominalHistFunc);
      } else {
        // If there ARE HistoSys(s)
        // name of source for variation
        string constraintPrefix = sample.GetName() + "_" + channel_name + "_Hist_alpha";

        // make list of abstract parameters that interpolate in space of variations
        RooArgList interpParams = makeInterpolationParameters(sample.GetHistoSysList(), proto);

        // next, create the constraint terms
        for(std::size_t i = 0; i < interpParams.size(); ++i) {
          bool isUniform = measurement.GetUniformSyst().count(sample.GetHistoSysList()[i].GetName()) > 0;
          makeGaussianConstraint(interpParams[i], proto, isUniform, constraintTermNames);
        }

        // finally, create the interpolated function
        sampleHistFuncs.push_back( makeLinInterp(interpParams, nominalHistFunc, proto,
            sample.GetHistoSysList(), constraintPrefix, observables) );
      }

      sampleHistFuncs.front()->SetTitle( (nominal && strlen(nominal->GetTitle())>0) ? nominal->GetTitle() : sample.GetName().c_str() );

      ////////////////////////////////////
      // Add StatErrors to this Channel //
      ////////////////////////////////////

      if( sample.GetStatError().GetActivate() ) {

        if( fObsNameVec.size() > 3 ) {
          cxcoutF(HistFactory) << "Cannot include Stat Error for histograms of more than 3 dimensions."
              << std::endl;
          throw hf_exc();
        } else {

          // If we are using StatUncertainties, we multiply this object
          // by the ParamHistFunc and then pass that to the
          // RooRealSumPdf by appending it's name to the list

          cxcoutI(HistFactory) << "Sample: "     << sample.GetName()  << " to be included in Stat Error "
              << "for channel " << channel_name
              << std::endl;

          string UncertName  = sample.GetName() + "_" + channel_name + "_StatAbsolUncert";
          std::unique_ptr<TH1> statErrorHist;

          if( sample.GetStatError().GetErrorHist() == nullptr ) {
            // Make the absolute stat error
            cxcoutI(HistFactory) << "Making Statistical Uncertainty Hist for "
                << " Channel: " << channel_name
                << " Sample: "  << sample.GetName()
                << std::endl;
            statErrorHist.reset(MakeAbsolUncertaintyHist( UncertName, nominal));
          } else {
            // clone the error histograms because in case the sample has not error hist
            // it is created in MakeAbsolUncertainty
            // we need later to clean statErrorHist
            statErrorHist.reset(static_cast<TH1*>(sample.GetStatError().GetErrorHist()->Clone()));
            // We assume the (relative) error is provided.
            // We must turn it into an absolute error
            // using the nominal histogram
            cxcoutI(HistFactory) << "Using external histogram for Stat Errors for "
                << "\tChannel: " << channel_name
                << "\tSample: "  << sample.GetName()
                << "\tError Histogram: " << statErrorHist->GetName() << std::endl;
            // Multiply the relative stat uncertainty by the
            // nominal to get the overall stat uncertainty
            statErrorHist->Multiply( nominal );
            statErrorHist->SetName( UncertName.c_str() );
          }

          // Save the nominal and error hists
          // for the building of constraint terms
          statHistPairs.emplace_back(nominal, std::move(statErrorHist));

          // To do the 'conservative' version, we would need to do some
          // intervention here.  We would probably need to create a different
          // ParamHistFunc for each sample in the channel.  The would nominally
          // use the same gamma's, so we haven't increased the number of parameters
          // However, if a bin in the 'nominal' histogram is 0, we simply need to
          // change the parameter in that bin in the ParamHistFunc for this sample.
          // We also need to add a constraint term.
          //  Actually, we'd probably not use the ParamHistFunc...?
          //  we could remove the dependence in this ParamHistFunc on the ith gamma
          //  and then create the poisson term: Pois(tau | n_exp)Pois(data | n_exp)


          // Next, try to get the common ParamHistFunc (it may have been
          // created by another sample in this channel)
          // or create it if it doesn't yet exist:
          RooAbsReal* paramHist = dynamic_cast<ParamHistFunc*>(proto.function(statFuncName) );
          if( paramHist == nullptr ) {

            // Get a RooArgSet of the observables:
            // Names in the list fObsNameVec:
            RooArgList theObservables;
            std::vector<std::string>::iterator itr = fObsNameVec.begin();
            for (int idx=0; itr!=fObsNameVec.end(); ++itr, ++idx ) {
              theObservables.add( *proto.var(*itr) );
            }

            // Create the list of terms to
            // control the bin heights:
            std::string ParamSetPrefix  = "gamma_stat_" + channel_name;
            RooArgList statFactorParams = ParamHistFunc::createParamSet(proto,
                ParamSetPrefix,
                theObservables,
                defaultGammaMin, defaultStatErrorGammaMax);

            ParamHistFunc statUncertFunc(statFuncName.c_str(), statFuncName.c_str(),
                theObservables, statFactorParams );

            proto.import( statUncertFunc, RooFit::RecycleConflictNodes() );

            paramHist = proto.function( statFuncName);
          }

          // apply stat function to sample
          sampleHistFuncs.push_back(paramHist);
        }
      } // END: if DoMcStat


      ///////////////////////////////////////////
      // Create a ShapeFactor for this channel //
      ///////////////////////////////////////////

      if( !sample.GetShapeFactorList().empty() ) {

        if( fObsNameVec.size() > 3 ) {
          cxcoutF(HistFactory) << "Cannot include Stat Error for histograms of more than 3 dimensions."
              << std::endl;
          throw hf_exc();
        } else {

          cxcoutI(HistFactory) << "Sample: "     << sample.GetName() << " in channel: " << channel_name
              << " to be include a ShapeFactor."
              << std::endl;

          for(ShapeFactor& shapeFactor : sample.GetShapeFactorList()) {

            std::string funcName = channel_name + "_" + shapeFactor.GetName() + "_shapeFactor";
            RooAbsArg *paramHist = proto.function(funcName);
            if( paramHist == nullptr ) {

              RooArgList theObservables;
              for(std::string const& varName : fObsNameVec) {
                theObservables.add( *proto.var(varName) );
              }

              // Create the Parameters
              std::string funcParams = "gamma_" + shapeFactor.GetName();

              // GHL: Again, we are putting hard ranges on the gamma's
              //      We should change this to range from 0 to /inf
              RooArgList shapeFactorParams = ParamHistFunc::createParamSet(proto,
                  funcParams,
                  theObservables, defaultGammaMin, defaultShapeFactorGammaMax);

              // Create the Function
              ParamHistFunc shapeFactorFunc( funcName.c_str(), funcName.c_str(),
                  theObservables, shapeFactorParams );

              // Set an initial shape, if requested
              if( shapeFactor.GetInitialShape() != nullptr ) {
                TH1* initialShape = static_cast<TH1*>(shapeFactor.GetInitialShape()->Clone());
                cxcoutI(HistFactory) << "Setting Shape Factor: " << shapeFactor.GetName()
                   << " to have initial shape from hist: "
                   << initialShape->GetName()
                   << std::endl;
                shapeFactorFunc.setShape( initialShape );
              }

              // Set the variables constant, if requested
              if( shapeFactor.IsConstant() ) {
                cxcoutI(HistFactory) << "Setting Shape Factor: " << shapeFactor.GetName()
                   << " to be constant" << std::endl;
                shapeFactorFunc.setConstant(true);
              }

              proto.import( shapeFactorFunc, RooFit::RecycleConflictNodes() );
              paramHist = proto.function(funcName);

            } // End: Create ShapeFactor ParamHistFunc

            sampleHistFuncs.push_back(paramHist);
          } // End loop over ShapeFactor Systematics
        }
      } // End: if ShapeFactorName!=""


      ////////////////////////////////////////
      // Create a ShapeSys for this channel //
      ////////////////////////////////////////

      if( !sample.GetShapeSysList().empty() ) {

        if( fObsNameVec.size() > 3 ) {
          cxcoutF(HistFactory) << "Cannot include Stat Error for histograms of more than 3 dimensions."
              << std::endl;
          throw hf_exc();
        }

          // List of ShapeSys ParamHistFuncs
          std::vector<string> ShapeSysNames;

          for(RooStats::HistFactory::ShapeSys& shapeSys : sample.GetShapeSysList()) {

            // Create the ParamHistFunc's
            // Create their constraint terms and add them
            // to the list of constraint terms

            // Create a single RooProduct over all of these
            // paramHistFunc's

            // Send the name of that product to the RooRealSumPdf

            cxcoutI(HistFactory) << "Sample: " << sample.GetName() << " in channel: " << channel_name
                << " to include a ShapeSys." << std::endl;

            std::string funcName = channel_name + "_" + shapeSys.GetName() + "_ShapeSys";
            ShapeSysNames.push_back( funcName );
            auto paramHist = static_cast<ParamHistFunc*>(proto.function(funcName));
            if( paramHist == nullptr ) {

              //std::string funcParams = "gamma_" + it->shapeFactorName;
              //paramHist = CreateParamHistFunc( proto, fObsNameVec, funcParams, funcName );

              RooArgList theObservables;
              for(std::string const& varName : fObsNameVec) {
                theObservables.add( *proto.var(varName) );
              }

              // Create the Parameters
              std::string funcParams = "gamma_" + shapeSys.GetName();
              RooArgList shapeFactorParams = ParamHistFunc::createParamSet(proto,
                  funcParams,
                  theObservables, defaultGammaMin, defaultShapeSysGammaMax);

              // Create the Function
              ParamHistFunc shapeFactorFunc( funcName.c_str(), funcName.c_str(),
                  theObservables, shapeFactorParams );

              proto.import( shapeFactorFunc, RooFit::RecycleConflictNodes() );
              paramHist = static_cast<ParamHistFunc*>(proto.function(funcName));

            } // End: Create ShapeFactor ParamHistFunc

            // Create the constraint terms and add
            // them to the workspace (proto)
            // as well as the list of constraint terms (constraintTermNames)

            // The syst should be a fractional error
            const TH1* shapeErrorHist = shapeSys.GetErrorHist();

            // Constraint::Type shapeConstraintType = Constraint::Gaussian;
            Constraint::Type systype = shapeSys.GetConstraintType();
            if( systype == Constraint::Gaussian) {
              systype = Constraint::Gaussian;
            }
            if( systype == Constraint::Poisson ) {
              systype = Constraint::Poisson;
            }

            auto shapeConstraintsInfo = createGammaConstraints(
                paramHist->paramList(), histToVector(*shapeErrorHist),
                minShapeUncertainty,
                systype);
            for (auto const& term : shapeConstraintsInfo.constraints) {
               proto.import(*term, RooFit::RecycleConflictNodes());
               constraintTermNames.emplace_back(term->GetName());
            }
            // Add the "observed" value to the list of global observables:
            RooArgSet *globalSet = const_cast<RooArgSet *>(proto.set("globalObservables"));
            for (RooAbsArg * glob : shapeConstraintsInfo.globalObservables) {
               globalSet->add(*proto.var(glob->GetName()));
            }


          } // End: Loop over ShapeSys vector in this EstimateSummary

          // Now that we have the list of ShapeSys ParamHistFunc names,
          // we create the total RooProduct
          // we multiply the expected function

          for(std::string const& name : ShapeSysNames) {
            sampleHistFuncs.push_back(proto.function(name));
          }

      } // End: !GetShapeSysList.empty()


      // GHL: This was pretty confusing before,
      //      hopefully using the measurement directly
      //      will improve it
      RooAbsArg *lumi = proto.arg("Lumi");
      if( !sample.GetNormalizeByTheory() ) {
        if (!lumi) {
          lumi = &emplace<RooRealVar>(proto, "Lumi", measurement.GetLumi());
        } else {
          static_cast<RooAbsRealLValue*>(lumi)->setVal(measurement.GetLumi());
        }
      }
      assert(lumi);
      normFactors->addTerm(lumi);

      // Append the name of the "node"
      // that is to be summed with the
      // RooRealSumPdf
      proto.import(*normFactors, RooFit::RecycleConflictNodes());
      auto normFactorsInWS = dynamic_cast<RooProduct*>(proto.arg(normFactors->GetName()));
      assert(normFactorsInWS);

      sampleScaleFactors.push_back(normFactorsInWS);
    } // END: Loop over EstimateSummaries

    // If a non-zero number of samples call for
    // Stat Uncertainties, create the statFactor functions
    if(!statHistPairs.empty()) {

      // Create the histogram of (binwise)
      // stat uncertainties:
      unique_ptr<TH1> fracStatError( MakeScaledUncertaintyHist( channel_name + "_StatUncert" + "_RelErr", statHistPairs) );
      if( fracStatError == nullptr ) {
        cxcoutE(HistFactory) << "Error: Failed to make ScaledUncertaintyHist for: "
            << channel_name + "_StatUncert" + "_RelErr" << std::endl;
        throw hf_exc();
      }

      // Using this TH1* of fractinal stat errors,
      // create a set of constraint terms:
      auto chanStatUncertFunc = static_cast<ParamHistFunc*>(proto.function( statFuncName ));
      cxcoutI(HistFactory) << "About to create Constraint Terms from: "
          << chanStatUncertFunc->GetName()
          << " params: " << chanStatUncertFunc->paramList()
          << std::endl;

      // Get the constraint type and the
      // rel error threshold from the (last)
      // EstimateSummary looped over (but all
      // should be the same)

      // Get the type of StatError constraint from the channel
      Constraint::Type statConstraintType = channel.GetStatErrorConfig().GetConstraintType();
      if( statConstraintType == Constraint::Gaussian) {
        cxcoutI(HistFactory) << "Using Gaussian StatErrors in channel: " << channel.GetName() << std::endl;
      }
      if( statConstraintType == Constraint::Poisson ) {
        cxcoutI(HistFactory) << "Using Poisson StatErrors in channel: " << channel.GetName()  << std::endl;
      }

      double statRelErrorThreshold = channel.GetStatErrorConfig().GetRelErrorThreshold();
      auto statConstraintsInfo = createGammaConstraints(
          chanStatUncertFunc->paramList(), histToVector(*fracStatError),
          statRelErrorThreshold,
          statConstraintType);
      for (auto const& term : statConstraintsInfo.constraints) {
         proto.import(*term, RooFit::RecycleConflictNodes());
         constraintTermNames.emplace_back(term->GetName());
      }
      // Add the "observed" value to the list of global observables:
      RooArgSet *globalSet = const_cast<RooArgSet *>(proto.set("globalObservables"));
      for (RooAbsArg * glob : statConstraintsInfo.globalObservables) {
         globalSet->add(*proto.var(glob->GetName()));
      }

    } // END: Loop over stat Hist Pairs


    ///////////////////////////////////
    // for ith bin calculate totN_i =  lumi * sum_j expected_j * syst_j
    MakeTotalExpected(proto, channel_name+"_model",
        sampleScaleFactors, allSampleHistFuncs);
    likelihoodTermNames.push_back(channel_name+"_model");

    //////////////////////////////////////
    // fix specified parameters
    for(unsigned int i=0; i<systToFix.size(); ++i){
      RooRealVar* temp = proto.var(systToFix.at(i));
      if(!temp) {
        cxcoutW(HistFactory) << "could not find variable " << systToFix.at(i)
            << " could not set it to constant" << std::endl;
      } else {
        // set the parameter constant
        temp->setConstant();
      }
    }

    //////////////////////////////////////
    // final proto model
    for(unsigned int i=0; i<constraintTermNames.size(); ++i){
      RooAbsArg* proto_arg = proto.arg(constraintTermNames[i]);
      if( proto_arg==nullptr ) {
        cxcoutF(HistFactory) << "Error: Cannot find arg set: " << constraintTermNames.at(i)
            << " in workspace: " << proto.GetName() << std::endl;
        throw hf_exc();
      }
      constraintTerms.add( *proto_arg );
      //  constraintTerms.add(* proto_arg(proto.arg(constraintTermNames[i].c_str())) );
    }
    for(unsigned int i=0; i<likelihoodTermNames.size(); ++i){
      RooAbsArg* proto_arg = (proto.arg(likelihoodTermNames[i]));
      if( proto_arg==nullptr ) {
        cxcoutF(HistFactory) << "Error: Cannot find arg set: " << likelihoodTermNames.at(i)
            << " in workspace: " << proto.GetName() << std::endl;
        throw hf_exc();
      }
      likelihoodTerms.add( *proto_arg );
    }
    proto.defineSet("constraintTerms",constraintTerms);
    proto.defineSet("likelihoodTerms",likelihoodTerms);

    // list of observables
    RooArgList observables;
    std::string observablesStr;

    for(std::string const& name : fObsNameVec) {
      observables.add( *proto.var(name) );
      if (!observablesStr.empty()) { observablesStr += ","; }
      observablesStr += name;
    }

    // We create two sets, one for backwards compatibility
    // The other to make a consistent naming convention
    // between individual channels and the combined workspace
    proto.defineSet("observables", observablesStr.c_str());
    proto.defineSet("observablesSet", observablesStr.c_str());

    // Create the ParamHistFunc
    // after observables have been made
    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\timport model into workspace"
        << "\n-----------------------------------------\n" << std::endl;

    auto model = make_unique<RooProdPdf>(
        ("model_"+channel_name).c_str(),    // MB : have changed this into conditional pdf. Much faster for toys!
        "product of Poissons across bins for a single channel",
        constraintTerms, RooFit::Conditional(likelihoodTerms,observables));
    // can give channel a title by setting title of corresponding data histogram
    if (channel.GetData().GetHisto() && strlen(channel.GetData().GetHisto()->GetTitle())>0) {
       model->SetTitle(channel.GetData().GetHisto()->GetTitle());
    }
    proto.import(*model,RooFit::RecycleConflictNodes());

    proto_config->SetPdf(*model);
    proto_config->SetObservables(observables);
    proto_config->SetGlobalObservables(*proto.set("globalObservables"));
    //    proto.writeToFile(("results/model_"+channel+".root").c_str());
    // fill out nuisance parameters in model config
    //    proto_config->GuessObsAndNuisance(*proto.data("asimovData"));
    proto.import(*proto_config,proto_config->GetName());
    proto.importClassCode();

    ///////////////////////////
    // make data sets
    // THis works and is natural, but the memory size of the simultaneous dataset grows exponentially with channels
    // New Asimov Generation: Use the code in the Asymptotic calculator
    // Need to get the ModelConfig...
    int asymcalcPrintLevel = 0;
    if (RooMsgService::instance().isActive(nullptr, RooFit::HistFactory, RooFit::INFO)) asymcalcPrintLevel = 1;
    if (RooMsgService::instance().isActive(nullptr, RooFit::HistFactory, RooFit::DEBUG)) asymcalcPrintLevel = 2;
    AsymptoticCalculator::SetPrintLevel(asymcalcPrintLevel);
    unique_ptr<RooAbsData> asimov_dataset(AsymptoticCalculator::GenerateAsimovData(*model, observables));
    proto.import(*asimov_dataset, RooFit::Rename("asimovData"));

    // GHL: Determine to use data if the hist isn't 'nullptr'
    if(TH1 const* mnominal = channel.GetData().GetHisto()) {
      // This works and is natural, but the memory size of the simultaneous
      // dataset grows exponentially with channels.
      std::unique_ptr<RooDataSet> dataset;
      if(!fCfg.storeDataError){
        dataset = std::make_unique<RooDataSet>("obsData","",*proto.set("observables"), RooFit::WeightVar("weightVar"));
      } else {
        const char* weightErrName="weightErr";
        proto.factory(TString::Format("%s[0,-1e10,1e10]",weightErrName));
        dataset = std::make_unique<RooDataSet>("obsData","",*proto.set("observables"), RooFit::WeightVar("weightVar"), RooFit::StoreError(*proto.var(weightErrName)));
      }
      ConfigureHistFactoryDataset( *dataset, *mnominal, proto, fObsNameVec );
      proto.import(*dataset);
    } // End: Has non-null 'data' entry


    for(auto const& data : channel.GetAdditionalData()) {
      if(data.GetName().empty()) {
        cxcoutF(HistFactory) << "Error: Additional Data histogram for channel: " << channel.GetName()
                << " has no name! The name always needs to be set for additional datasets, "
                << "either via the \"Name\" tag in the XML or via RooStats::HistFactory::Data::SetName()." << std::endl;
        throw hf_exc();
      }
      std::string const& dataName = data.GetName();
      TH1 const* mnominal = data.GetHisto();
      if( !mnominal ) {
        cxcoutF(HistFactory) << "Error: Additional Data histogram for channel: " << channel.GetName()
                << " with name: " << dataName << " is nullptr" << std::endl;
        throw hf_exc();
      }

      // THis works and is natural, but the memory size of the simultaneous dataset grows exponentially with channels
      RooDataSet dataset{dataName, "", *proto.set("observables"), RooFit::WeightVar("weightVar")};
      ConfigureHistFactoryDataset( dataset, *mnominal, proto, fObsNameVec );
      proto.import(dataset);

    }

    if (RooMsgService::instance().isActive(nullptr, RooFit::HistFactory, RooFit::INFO)) {
      proto.Print();
    }

    return protoOwner;
  }


  void HistoToWorkspaceFactoryFast::ConfigureHistFactoryDataset( RooDataSet& obsDataUnbinned,
                         TH1 const& mnominal,
                         RooWorkspace& proto,
                         std::vector<std::string> const& obsNameVec) {

    // Take a RooDataSet and fill it with the entries
    // from a TH1*, using the observable names to
    // determine the columns

     if (obsNameVec.empty() ) {
        Error("ConfigureHistFactoryDataset","Invalid input - return");
        return;
     }

    TAxis const* ax = mnominal.GetXaxis();
    TAxis const* ay = mnominal.GetYaxis();
    TAxis const* az = mnominal.GetZaxis();

    // check whether the dataset needs the errors stored explicitly
    const bool storeWeightErr = obsDataUnbinned.weightVar()->getAttribute("StoreError");

    for (int i=1; i<=ax->GetNbins(); ++i) { // 1 or more dimension

      double xval = ax->GetBinCenter(i);
      proto.var( obsNameVec[0] )->setVal( xval );

      if(obsNameVec.size()==1) {
   double fval = mnominal.GetBinContent(i);
   double ferr = storeWeightErr ? mnominal.GetBinError(i) : 0.;
   obsDataUnbinned.add( *proto.set("observables"), fval, ferr );
      } else { // 2 or more dimensions

   for(int j=1; j<=ay->GetNbins(); ++j) {
     double yval = ay->GetBinCenter(j);
     proto.var( obsNameVec[1] )->setVal( yval );

     if(obsNameVec.size()==2) {
       double fval = mnominal.GetBinContent(i,j);
       double ferr = storeWeightErr ? mnominal.GetBinError(i, j) : 0.;
       obsDataUnbinned.add( *proto.set("observables"), fval, ferr );
     } else { // 3 dimensions

       for(int k=1; k<=az->GetNbins(); ++k) {
         double zval = az->GetBinCenter(k);
         proto.var( obsNameVec[2] )->setVal( zval );
         double fval = mnominal.GetBinContent(i,j,k);
         double ferr = storeWeightErr ? mnominal.GetBinError(i, j, k) : 0.;
         obsDataUnbinned.add( *proto.set("observables"), fval, ferr );
       }
     }
   }
      }
    }
  }

  void HistoToWorkspaceFactoryFast::GuessObsNameVec(const TH1* hist)
  {
    fObsNameVec = std::vector<string>{"x", "y", "z"};
    fObsNameVec.resize(hist->GetDimension());
  }


   RooFit::OwningPtr<RooWorkspace>
   HistoToWorkspaceFactoryFast::MakeCombinedModel(std::vector<std::string> ch_names,
                                                  std::vector<std::unique_ptr<RooWorkspace>> &chs)
  {
    RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::INFO, 0, RooFit::ObjectHandling, false);

     // check first the inputs (see JIRA-6890)
     if (ch_names.empty() || chs.empty() ) {
        Error("MakeCombinedModel","Input vectors are empty - return a nullptr");
        return nullptr;
     }
     if (chs.size()  <  ch_names.size() ) {
        Error("MakeCombinedModel","Input vector of workspace has an invalid size - return a nullptr");
        return nullptr;
     }

    //
    /// These things were used for debugging. Maybe useful in the future
    //

    map<string, RooAbsPdf*> pdfMap;
    vector<RooAbsPdf*> models;

    RooArgList obsList;
    for(unsigned int i = 0; i< ch_names.size(); ++i){
      obsList.add(*static_cast<ModelConfig *>(chs[i]->obj("ModelConfig"))->GetObservables());
    }
    cxcoutI(HistFactory) <<"full list of observables:\n" << obsList << std::endl;

    RooArgSet globalObs;
    std::map<std::string, int> channelMap;
    for(unsigned int i = 0; i< ch_names.size(); ++i){
      string channel_name=ch_names[i];
      if (i == 0 && isdigit(channel_name[0])) {
        throw std::invalid_argument("The first channel name for HistFactory cannot start with a digit. Got " + channel_name);
      }
      if (channel_name.find(',') != std::string::npos) {
        throw std::invalid_argument("Channel names for HistFactory cannot contain ','. Got " + channel_name);
      }

      channelMap[channel_name] = i;
      RooWorkspace * ch=chs[i].get();

      RooAbsPdf* model = ch->pdf("model_"+channel_name);
      if(!model) std::cout <<"failed to find model for channel"<< std::endl;
      //      std::cout << "int = " << model->createIntegral(*obsN)->getVal() << std::endl;
      models.push_back(model);
      globalObs.add(*ch->set("globalObservables"), /*silent=*/true); // silent because observables might exist in other channel.

      //      constrainedParams->add( * ch->set("constrainedParams") );
      pdfMap[channel_name]=model;
    }

    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\tEntering combination"
        << "\n-----------------------------------------\n" << std::endl;
    auto combined = std::make_unique<RooWorkspace>("combined");


    RooCategory& channelCat = emplace<RooCategory>(*combined, "channelCat", channelMap);

    auto simPdf= std::make_unique<RooSimultaneous>("simPdf","",pdfMap, channelCat);
    auto combined_config = std::make_unique<ModelConfig>("ModelConfig", combined.get());
    combined_config->SetWorkspace(*combined);
    //    combined_config->SetNuisanceParameters(*constrainedParams);

    combined->import(globalObs);
    combined->defineSet("globalObservables",globalObs);
    combined_config->SetGlobalObservables(*combined->set("globalObservables"));

    combined->defineSet("observables",{obsList, channelCat}, /*importMissing=*/true);
    combined_config->SetObservables(*combined->set("observables"));


    // Now merge the observable datasets across the channels
    for(RooAbsData * data : chs[0]->allData()) {
      // We are excluding the Asimov data, because it needs to be regenerated
      // later after the parameter values are set.
      if(std::string("asimovData") == data->GetName()) {
        continue;
      }
      // Loop through channels, get their individual datasets,
      // and add them to the combined dataset
      std::map<std::string, RooAbsData*> dataMap;
      for(unsigned int i = 0; i < ch_names.size(); ++i){
        dataMap[ch_names[i]] = chs[i]->data(data->GetName());
      }
      combined->import(RooDataSet{data->GetName(), "", obsList, RooFit::Index(channelCat),
                                  RooFit::WeightVar("weightVar"), RooFit::Import(dataMap)});
    }


    if (RooMsgService::instance().isActive(nullptr, RooFit::HistFactory, RooFit::INFO))
      combined->Print();

    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
            << "\tImporting combined model"
            << "\n-----------------------------------------\n" << std::endl;
    combined->import(*simPdf,RooFit::RecycleConflictNodes());

    for(auto const& param_itr : fParamValues) {
      // make sure they are fixed
      std::string paramName = param_itr.first;
      double paramVal = param_itr.second;

      if(RooRealVar* temp = combined->var( paramName )) {
        temp->setVal( paramVal );
        cxcoutI(HistFactory) <<"setting " << paramName << " to the value: " << paramVal <<  std::endl;
      } else
        cxcoutE(HistFactory) << "could not find variable " << paramName << " could not set its value" << std::endl;
    }


    for(unsigned int i=0; i<fSystToFix.size(); ++i){
      // make sure they are fixed
      if(RooRealVar* temp = combined->var(fSystToFix[i])) {
        temp->setConstant();
        cxcoutI(HistFactory) <<"setting " << fSystToFix.at(i) << " constant" << std::endl;
      } else
        cxcoutE(HistFactory) << "could not find variable " << fSystToFix.at(i) << " could not set it to constant" << std::endl;
    }

    ///
    /// writing out the model in graphViz
    ///
    //    RooAbsPdf* customized=combined->pdf("simPdf");
    //combined_config->SetPdf(*customized);
    combined_config->SetPdf(*simPdf);
    //    combined_config->GuessObsAndNuisance(*simData);
    //    customized->graphVizTree(("results/"+fResultsPrefixStr.str()+"_simul.dot").c_str());
    combined->import(*combined_config,combined_config->GetName());
    combined->importClassCode();
    //    combined->writeToFile("results/model_combined.root");


    ////////////////////////////////////////////
    // Make toy simultaneous dataset
    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\tcreate toy data"
        << "\n-----------------------------------------\n" << std::endl;


    // now with weighted datasets
    // First Asimov

    // Create Asimov data for the combined dataset
    std::unique_ptr<RooAbsData> asimov_combined{AsymptoticCalculator::GenerateAsimovData(
                                  *combined->pdf("simPdf"),
                                  obsList)};
    if( asimov_combined ) {
      combined->import( *asimov_combined, RooFit::Rename("asimovData"));
    }
    else {
      std::cout << "Error: Failed to create combined asimov dataset" << std::endl;
      throw hf_exc();
    }

    return RooFit::makeOwningPtr(std::move(combined));
  }


  TH1* HistoToWorkspaceFactoryFast::MakeAbsolUncertaintyHist( const std::string& Name, const TH1* Nominal ) {

    // Take a nominal TH1* and create
    // a TH1 representing the binwise
    // errors (taken from the nominal TH1)

    auto ErrorHist = static_cast<TH1*>(Nominal->Clone( Name.c_str() ));
    ErrorHist->Reset();

    int numBins   = Nominal->GetNbinsX()*Nominal->GetNbinsY()*Nominal->GetNbinsZ();
    int binNumber = 0;

    // Loop over bins
    for( int i_bin = 0; i_bin < numBins; ++i_bin) {

      binNumber++;
      // Ignore underflow / overflow
      while( Nominal->IsBinUnderflow(binNumber) || Nominal->IsBinOverflow(binNumber) ){
   binNumber++;
      }

      double histError = Nominal->GetBinError( binNumber );

      // Check that histError != NAN
      if( histError != histError ) {
   std::cout << "Warning: In histogram " << Nominal->GetName()
        << " bin error for bin " << i_bin
        << " is NAN.  Not using Error!!!"
        << std::endl;
   throw hf_exc();
   //histError = sqrt( histContent );
   //histError = 0;
      }

      // Check that histError ! < 0
      if( histError < 0  ) {
   std::cout << "Warning: In histogram " << Nominal->GetName()
        << " bin error for bin " << binNumber
        << " is < 0.  Setting Error to 0"
        << std::endl;
   //histError = sqrt( histContent );
   histError = 0;
      }

      ErrorHist->SetBinContent( binNumber, histError );

    }

    return ErrorHist;

  }

  // Take a list of < nominal, absolError > TH1* pairs
  // and construct a single histogram representing the
  // total fractional error as:

  // UncertInQuad(bin i) = Sum: absolUncert*absolUncert
  // Total(bin i)        = Sum: Value
  //
  // TotalFracError(bin i) = Sqrt( UncertInQuad(i) ) / TotalBin(i)
  std::unique_ptr<TH1> HistoToWorkspaceFactoryFast::MakeScaledUncertaintyHist( const std::string& Name, std::vector< std::pair<const TH1*, std::unique_ptr<TH1>> > const& HistVec ) const {


    unsigned int numHists = HistVec.size();

    if( numHists == 0 ) {
      cxcoutE(HistFactory) << "Warning: Empty Hist Vector, cannot create total uncertainty" << std::endl;
      return nullptr;
    }

    const TH1* HistTemplate = HistVec.at(0).first;
    int numBins = HistTemplate->GetNbinsX()*HistTemplate->GetNbinsY()*HistTemplate->GetNbinsZ();

  // Check that all histograms
  // have the same bins
  for( unsigned int i = 0; i < HistVec.size(); ++i ) {

    const TH1* nominal = HistVec.at(i).first;
    const TH1* error   = HistVec.at(i).second.get();

    if( nominal->GetNbinsX()*nominal->GetNbinsY()*nominal->GetNbinsZ() != numBins ) {
      cxcoutE(HistFactory) << "Error: Provided hists have unequal bins" << std::endl;
      return nullptr;
    }
    if( error->GetNbinsX()*error->GetNbinsY()*error->GetNbinsZ() != numBins ) {
      cxcoutE(HistFactory) << "Error: Provided hists have unequal bins" << std::endl;
      return nullptr;
    }
  }

  std::vector<double> TotalBinContent( numBins, 0.0);
  std::vector<double> HistErrorsSqr( numBins, 0.0);

  int binNumber = 0;

  // Loop over bins
  for( int i_bins = 0; i_bins < numBins; ++i_bins) {

    binNumber++;
    while( HistTemplate->IsBinUnderflow(binNumber) || HistTemplate->IsBinOverflow(binNumber) ){
      binNumber++;
    }

    for( unsigned int i_hist = 0; i_hist < numHists; ++i_hist ) {

      const TH1* nominal = HistVec.at(i_hist).first;
      const TH1* error   = HistVec.at(i_hist).second.get();

      //int binNumber = i_bins + 1;

      double histValue  = nominal->GetBinContent( binNumber );
      double histError  = error->GetBinContent( binNumber );

      if( histError != histError ) {
        cxcoutE(HistFactory) << "In histogram " << error->GetName()
        << " bin error for bin " << binNumber
        << " is NAN.  Not using error!!";
        throw hf_exc();
      }

      TotalBinContent.at(i_bins) += histValue;
      HistErrorsSqr.at(i_bins)   += histError*histError; // Add in quadrature

    }
  }

  binNumber = 0;

  // Creat the output histogram
  TH1* ErrorHist = static_cast<TH1*>(HistTemplate->Clone( Name.c_str() ));
  ErrorHist->Reset();

  // Fill the output histogram
  for( int i = 0; i < numBins; ++i) {

    //    int binNumber = i + 1;
    binNumber++;
    while( ErrorHist->IsBinUnderflow(binNumber) || ErrorHist->IsBinOverflow(binNumber) ){
      binNumber++;
    }

    double ErrorsSqr = HistErrorsSqr.at(i);
    double TotalVal  = TotalBinContent.at(i);

    if( TotalVal <= 0 ) {
      cxcoutW(HistFactory) << "Warning: Sum of histograms for bin: " << binNumber
      << " is <= 0.  Setting error to 0"
      << std::endl;

      ErrorHist->SetBinContent( binNumber, 0.0 );
      continue;
    }

    double RelativeError = sqrt(ErrorsSqr) / TotalVal;

    // If we otherwise get a NAN
    // it's an error
    if( RelativeError != RelativeError ) {
      cxcoutE(HistFactory) << "Error: bin " << i << " error is NAN\n"
          << " HistErrorsSqr: " << ErrorsSqr
      << " TotalVal: " << TotalVal;
      throw hf_exc();
    }

    // 0th entry in vector is
    // the 1st bin in TH1
    // (we ignore underflow)

    // Error and bin content are interchanged because for some reason, the other functions
    // use the bin content to convey the error ...
    ErrorHist->SetBinError(binNumber, TotalVal);
    ErrorHist->SetBinContent(binNumber, RelativeError);

    cxcoutI(HistFactory) << "Making Total Uncertainty for bin " << binNumber
         << " Error = " << sqrt(ErrorsSqr)
         << " CentralVal = " << TotalVal
         << " RelativeError = " << RelativeError << "\n";

  }

  return std::unique_ptr<TH1>(ErrorHist);
}


} // namespace RooStats
} // namespace HistFactory

