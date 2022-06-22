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


#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif

#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooAddition.h"
#include "RooProduct.h"
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooGaussian.h"
#include "RooPoisson.h"
#include "RooExponential.h"
#include "RooRandom.h"
#include "RooCategory.h"
#include "RooSimultaneous.h"
#include "RooMultiVarGaussian.h"
#include "RooNumIntConfig.h"
#include "RooNLLVar.h"
#include "RooProfileLL.h"
#include "RooFitResult.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"
#include "RooHistPdf.h"
#include "RooRealSumPdf.h"
#include "RooWorkspace.h"
#include "RooCustomizer.h"
#include "RooPlot.h"
#include "RooHelpers.h"
#include "RooBinning.h"
#include "RooBinWidthFunction.h"
#include "RooStats/RooStatsUtils.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/HistFactory/PiecewiseInterpolation.h"
#include "RooStats/HistFactory/ParamHistFunc.h"
#include "RooStats/AsymptoticCalculator.h"
#include "HFMsgService.h"

#include "TH1.h"
#include "TTree.h"
#include "TStopwatch.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"

// specific to this package
#include "RooStats/HistFactory/LinInterpVar.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"
#include "RooStats/HistFactory/Measurement.h"
#include "Helper.h"

#include <algorithm>
#include <memory>
#include <utility>

#define VERBOSE

#define alpha_Low "-5"
#define alpha_High "5"
#define NoHistConst_Low "0"
#define NoHistConst_High "2000"

// use this order for safety on library loading
using namespace RooFit ;
using namespace RooStats ;
using namespace std ;

ClassImp(RooStats::HistFactory::HistoToWorkspaceFactoryFast);

namespace RooStats{
namespace HistFactory{

  HistoToWorkspaceFactoryFast::HistoToWorkspaceFactoryFast() : 
       fNomLumi(1.0), fLumiError(0),   
       fLowBin(0), fHighBin(0)  
  {}

  HistoToWorkspaceFactoryFast::~HistoToWorkspaceFactoryFast(){
  }

  HistoToWorkspaceFactoryFast::HistoToWorkspaceFactoryFast(RooStats::HistFactory::Measurement& measurement ) :
    fSystToFix( measurement.GetConstantParams() ),
    fParamValues( measurement.GetParamValues() ),
    fNomLumi( measurement.GetLumi() ),
    fLumiError( measurement.GetLumi()*measurement.GetLumiRelErr() ),
    fLowBin( measurement.GetBinLow() ),
    fHighBin( measurement.GetBinHigh() ) {

    // Set Preprocess functions
    SetFunctionsToPreprocess( measurement.GetPreprocessFunctions() );

  }

  void HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement( const std::string& ModelName, RooWorkspace* ws_single, Measurement& measurement ) {

    // Configure a workspace by doing any
    // necessary post-processing and by
    // creating a ModelConfig

    // Make a ModelConfig and configure it
    ModelConfig * proto_config = (ModelConfig *) ws_single->obj("ModelConfig");
    if( proto_config == NULL ) {
      std::cout << "Error: Did not find 'ModelConfig' object in file: " << ws_single->GetName() 
		<< std::endl;
      throw hf_exc();
    }

    std::vector<std::string> poi_list = measurement.GetPOIList();
    if( poi_list.size()==0 ) {
      cxcoutWHF << "No Parametetrs of interest are set" << std::endl;
    }


    std::stringstream sstream;
    sstream << "Setting Parameter(s) of Interest as: ";
    for(unsigned int i = 0; i < poi_list.size(); ++i) {
      sstream << poi_list.at(i) << " ";
    }
    cxcoutIHF << sstream.str() << endl;

    RooArgSet params;
    for( unsigned int i = 0; i < poi_list.size(); ++i ) {
      std::string poi_name = poi_list.at(i);
      RooRealVar* poi = (RooRealVar*) ws_single->var( poi_name.c_str() );
      if(poi){
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
    if(poi_list.size()!=0){
      proto_config->GuessObsAndNuisance(*expData, RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::INFO));
    }

    // Now, let's loop over any additional asimov datasets
    // that we need to make

    // Get the pdf
    // Notice that we get the "new" pdf, this is the one that is
    // used in the creation of these asimov datasets since they
    // are fitted (or may be, at least).
    RooAbsPdf* pdf = ws_single->pdf(NewModelName.c_str());
    if( !pdf ) pdf = ws_single->pdf( ModelName.c_str() );
    const RooArgSet* observables = ws_single->set("observables");

    // Create a SnapShot of the nominal values 
    std::string SnapShotName = "NominalParamValues";
    ws_single->saveSnapshot(SnapShotName.c_str(), ws_single->allVars());

    for( unsigned int i=0; i<measurement.GetAsimovDatasets().size(); ++i) {

      // Set the variable values and "const" ness with the workspace
      RooStats::HistFactory::Asimov& asimov = measurement.GetAsimovDatasets().at(i);
      std::string AsimovName = asimov.GetName();

      cxcoutPHF << "Generating additional Asimov Dataset: " << AsimovName << std::endl;
      asimov.ConfigureWorkspace(ws_single);
      std::unique_ptr<RooDataSet> asimov_dataset{static_cast<RooDataSet*>(AsymptoticCalculator::GenerateAsimovData(*pdf, *observables))};

      cxcoutPHF << "Importing Asimov dataset" << std::endl;
      bool failure = ws_single->import(*asimov_dataset, Rename(AsimovName.c_str()));
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


  // We want to eliminate this interface and use the measurment directly
  RooWorkspace* HistoToWorkspaceFactoryFast::MakeSingleChannelModel( Measurement& measurement, Channel& channel ) {

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
    RooWorkspace* ws_single = this->MakeSingleChannelWorkspace(measurement, channel);
    if( ws_single == NULL ) {
      cxcoutF(HistFactory) << "Error: Failed to make Single-Channel workspace for channel: " << ch_name
		<< " and measurement: " << measurement.GetName() << std::endl;
      throw hf_exc();
    }

    // Finally, configure that workspace based on
    // properties of the measurement
    HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement( "model_"+ch_name, ws_single, measurement );

    return ws_single;

  }

  RooWorkspace* HistoToWorkspaceFactoryFast::MakeCombinedModel( Measurement& measurement ) {

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
    HistoToWorkspaceFactoryFast factory( measurement );

    // Loop over the channels and create the individual workspaces
    vector<std::unique_ptr<RooWorkspace>> channel_workspaces;
    vector<string>        channel_names;
    
    for( unsigned int chanItr = 0; chanItr < measurement.GetChannels().size(); ++chanItr ) {
    
      HistFactory::Channel& channel = measurement.GetChannels().at( chanItr );

      if( ! channel.CheckHistograms() ) {
        cxcoutFHF << "MakeModelAndMeasurementsFast: Channel: " << channel.GetName()
		      << " has uninitialized histogram pointers" << std::endl;
        throw hf_exc();
      }

      string ch_name = channel.GetName();
      channel_names.push_back(ch_name);

      // GHL: Renaming to 'MakeSingleChannelWorkspace'
      RooWorkspace* ws_single = factory.MakeSingleChannelModel( measurement, channel );
      
      channel_workspaces.emplace_back(ws_single);

    }

    
    // Now, combine the individual channel workspaces to
    // form the combined workspace
    RooWorkspace* ws = factory.MakeCombinedModel( channel_names, channel_workspaces );


    // Configure the workspace
    HistoToWorkspaceFactoryFast::ConfigureWorkspaceForMeasurement( "simPdf", ws, measurement );

    // Done.  Return the pointer
    return ws;

  }

/// Create observables of type RooRealVar. Creates 1 to 3 observables, depending on the type of the histogram.
RooArgList HistoToWorkspaceFactoryFast::createObservables(const TH1 *hist, RooWorkspace *proto) const {
  RooArgList observables;

  for (unsigned int idx=0; idx < fObsNameVec.size(); ++idx) {
    if (!proto->var(fObsNameVec[idx].c_str())) {
      const TAxis *axis = (idx == 0) ? hist->GetXaxis() : (idx == 1 ? hist->GetYaxis() : hist->GetZaxis());
      Int_t nbins = axis->GetNbins();
      Double_t xmin = axis->GetXmin();
      Double_t xmax = axis->GetXmax();
      // create observable
      auto obs = static_cast<RooRealVar*>(proto->factory(
          Form("%s[%f,%f]", fObsNameVec[idx].c_str(), xmin, xmax)));
      obs->setBins(nbins);
      if (axis->IsVariableBinSize()) {
        RooBinning binning(nbins, axis->GetXbins()->GetArray());
        obs->setBinning(binning);
      }
    }

    observables.add(*proto->var(fObsNameVec[idx].c_str()));
  }

  return observables;
}

  /// Create the nominal hist function from `hist`, and register it in the workspace.
  RooHistFunc* HistoToWorkspaceFactoryFast::MakeExpectedHistFunc(const TH1* hist,RooWorkspace* proto, string prefix,
      const RooArgList& observables) const {
    if(hist) {
      cxcoutI(HistFactory) << "processing hist " << hist->GetName() << endl;
    } else {
      cxcoutF(HistFactory) << "hist is empty" << endl;
      R__ASSERT(hist != 0); 
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

    RooDataHist histDHist((prefix + "DHist").c_str(),"",observables,hist);
    RooHistFunc histFunc(prefix.c_str(),"",observables,histDHist,0);

    proto->import(histFunc, RecycleConflictNodes());
    auto histFuncInWS = static_cast<RooHistFunc*>(proto->arg(prefix.c_str()));

    return histFuncInWS;
  }

  namespace {

  void makeGaussianConstraint(RooAbsArg& param, RooWorkspace& proto, bool isUniform,
                              std::vector<std::string> & constraintTermNames) {
      std::string paramName = param.GetName();
      std::string constraintName = paramName + "Constraint";

      // do nothing if the constraint term already exists
      if(proto.pdf(constraintName.c_str())) return;

      // case systematic is uniform (asssume they are like a Gaussian but with
      // a large width (100 instead of 1)
      const double gaussSigma = isUniform ? 100. : 1.0;
      if (isUniform) {
         cxcoutIHF << "Added a uniform constraint for " << paramName << " as a Gaussian constraint with a very large sigma " << std::endl;
      }

      std::stringstream command;
      command << "Gaussian::" << constraintName << "(" << paramName << ",nom_" << paramName << "[0.,-10,10],"
              << gaussSigma << ")";
      constraintTermNames.emplace_back(proto.factory( command.str().c_str() )->GetName());
      auto * normParam = proto.var((std::string("nom_") + paramName).c_str());
      normParam->setConstant();
      const_cast<RooArgSet*>(proto.set("globalObservables"))->add(*normParam);
  }

  /// Make list of abstract parameters that interpolate in space of variations.
  RooArgList makeInterpolationParameters(std::vector<HistoSys> const& histoSysList, RooWorkspace& proto) {
    RooArgList params( ("alpha_Hist") );

    // range is set using defined macro (see top of the page)
    string range=string("[")+alpha_Low+","+alpha_High+"]";

    for(auto const& histoSys : histoSysList) {
      const std::string histoSysName = histoSys.GetName();
      RooRealVar* temp = proto.var(("alpha_" + histoSysName).c_str());
      if(!temp){
        temp = (RooRealVar*) proto.factory(("alpha_" + histoSysName + range).c_str());
      }
      params.add(* temp );
    }

    return params;
  }

  /// Create a linear interpolation object that holds nominal and systematics, import it into the workspace,
  /// and return a pointer to it.
  RooAbsArg* makeLinInterp(RooArgList const& interpolationParams,
                           RooHistFunc* nominalFunc,
                           RooWorkspace* proto, const std::vector<HistoSys>& histoSysList,
                           const string& prefix,
                           const RooArgList& observables) {

    // now make function that linearly interpolates expectation between variations
    // get low/high variations to interpolate between
    vector<double> low, high;
    RooArgSet lowSet, highSet;
    //ES// for(unsigned int j=0; j<lowHist.size(); ++j){
    for(unsigned int j=0; j<histoSysList.size(); ++j){
      std::stringstream str;
      str<<"_"<<j;

      const HistoSys& histoSys = histoSysList.at(j);
      RooDataHist* lowDHist = new RooDataHist((prefix+str.str()+"lowDHist").c_str(),"",observables, histoSys.GetHistoLow());
      RooDataHist* highDHist = new RooDataHist((prefix+str.str()+"highDHist").c_str(),"",observables, histoSys.GetHistoHigh());
      RooHistFunc* lowFunc = new RooHistFunc((prefix+str.str()+"low").c_str(),"",observables,*lowDHist,0) ;
      RooHistFunc* highFunc = new RooHistFunc((prefix+str.str()+"high").c_str(),"",observables,*highDHist,0) ;
      lowSet.add(*lowFunc);
      highSet.add(*highFunc);
    }

    // this is sigma(params), a piece-wise linear interpolation
    PiecewiseInterpolation interp(prefix.c_str(),"",*nominalFunc,lowSet,highSet,interpolationParams);
    interp.setPositiveDefinite();
    interp.setAllInterpCodes(4); // LM: change to 4 (piece-wise linear to 6th order polynomial interpolation + linear extrapolation )
    // KC: interpo codes 1 etc. don't have proper analytic integral.
    RooArgSet observableSet(observables);
    interp.setBinIntegrator(observableSet);
    interp.forceNumInt();

    proto->import(interp, RecycleConflictNodes()); // individual params have already been imported in first loop of this function

    auto interpInWS = proto->arg(prefix.c_str());
    assert(interpInWS);

    return interpInWS;
  }

  }

  // GHL: Consider passing the NormFactor list instead of the entire sample
  std::unique_ptr<RooProduct> HistoToWorkspaceFactoryFast::CreateNormFactor(RooWorkspace* proto, string& channel, string& sigmaEpsilon, Sample& sample, bool doRatio){

    std::vector<string> prodNames;

    vector<NormFactor> normList = sample.GetNormFactorList();
    vector<string> normFactorNames, rangeNames;


    string overallNorm_times_sigmaEpsilon = sample.GetName() + "_" + channel + "_scaleFactors";
    auto sigEps = proto->arg(sigmaEpsilon.c_str());
    assert(sigEps);
    auto normFactor = std::make_unique<RooProduct>(overallNorm_times_sigmaEpsilon.c_str(), overallNorm_times_sigmaEpsilon.c_str(), RooArgList(*sigEps));

    if(normList.size() > 0){

      for(vector<NormFactor>::iterator itr = normList.begin(); itr != normList.end(); ++itr){

        NormFactor& norm = *itr;

        string varname;
        if(doRatio) {
          varname = norm.GetName() + "_" + channel;
        }
        else {
          varname=norm.GetName();
        }

        // GHL: Check that the NormFactor doesn't already exist
        //      (it may have been created as a function expression
        //       during preprocessing)
        std::stringstream range;
        range << "[" << norm.GetVal() << "," << norm.GetLow() << "," << norm.GetHigh() << "]";

        if( proto->obj(varname.c_str()) == NULL) {
          cxcoutI(HistFactory) << "making normFactor: " << norm.GetName() << endl;
          // remove "doRatio" and name can be changed when ws gets imported to the combined model.
          proto->factory((varname + range.str()).c_str());
        }

        if(norm.GetConst()) {
          //	  proto->var(varname.c_str())->setConstant();
          //	  cout <<"setting " << varname << " constant"<<endl;
          cxcoutW(HistFactory) << "Const attribute to <NormFactor> tag is deprecated, will ignore." <<
              " Instead, add \n\t<ParamSetting Const=\"True\"> " << varname << " </ParamSetting>\n" <<
              " to your top-level XML's <Measurement> entry" << endl;
        }
        prodNames.push_back(varname);
        rangeNames.push_back(range.str());
        normFactorNames.push_back(varname);
      }


      for (const auto& name : prodNames) {
        auto arg = proto->arg(name.c_str());
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
            << "\"> \nin your top-level XML's <Measurment> entry and use <NormFactor Name=\""<<*nit<<"Squared\" in your channel XML file."<< endl;
      }
      ++rangeIndex;
    }

    return normFactor;
  }        

   void HistoToWorkspaceFactoryFast::AddConstraintTerms(RooWorkspace* proto, Measurement & meas, string prefix, 
						       string interpName,
						       std::vector<OverallSys>& systList, 
						       vector<string>& constraintTermNames, 
						       vector<string>& totSystTermNames) {

    // add variables for all the relative overall uncertainties we expect
    // range is set using defined macro (see top of the page)

    string range=string("[0,")+alpha_Low+","+alpha_High+"]";
    totSystTermNames.push_back(prefix);

    RooArgSet params(prefix.c_str());
    vector<double> lowVec, highVec;

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
         double tauVal = 1./(relerr*relerr);
         double sqtau = 1./relerr;
         RooAbsArg * beta = proto->factory(TString::Format("beta_%s[1,0,10]",name) );
         // the global observable (y_s)
         RooAbsArg * yvar = proto->factory(TString::Format("nom_%s[%f,0,10]",beta->GetName(),tauVal)) ;
         // the rate of the gamma distribution (theta)
         RooAbsArg * theta = proto->factory(TString::Format("theta_%s[%f]",name,1./tauVal));
         // find alpha as function of beta
         RooAbsArg* alphaOfBeta = proto->factory(TString::Format("PolyVar::alphaOfBeta_%s(beta_%s,{%f,%f})",name,name,-sqtau,sqtau));

         // add now the constraint itself  Gamma_beta_constraint(beta, y+1, tau, 0 )
         // build the gamma parameter k = as y_s + 1
         RooAbsArg * kappa = proto->factory(TString::Format("sum::k_%s(%s,1.)",name,yvar->GetName()) ); 
         RooAbsArg * gamma = proto->factory(TString::Format("Gamma::%sConstraint(%s, %s, %s, 0.0)",beta->GetName(),beta->GetName(), kappa->GetName(), theta->GetName() ) );  
         if (RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::DEBUG)) {
           alphaOfBeta->Print("t");
           gamma->Print("t");
         }
         constraintTermNames.push_back(gamma->GetName());
         // set global observables
         RooRealVar * gobs = dynamic_cast<RooRealVar*>(yvar); assert(gobs);
         gobs->setConstant(true);
         const_cast<RooArgSet*>(proto->set("globalObservables"))->add(*yvar);

         // add alphaOfBeta in the list of params to interpolate
         params.add(*alphaOfBeta);
         cxcoutI(HistFactory) << "Added a gamma constraint for " << name << std::endl;
                  
      }
      else {
         RooRealVar* alpha = (RooRealVar*) proto->var((prefix + sys.GetName()).c_str());
         if(!alpha) {
            alpha = (RooRealVar*) proto->factory((prefix + sys.GetName() + range).c_str());
         }
         // add the Gaussian constraint part
         const bool isUniform = meas.GetUniformSyst().count(sys.GetName()) > 0;
         makeGaussianConstraint(*alpha, *proto, isUniform, constraintTermNames);

         // check if exists a log-normal constraint
         if (meas.GetLogNormSyst().count(sys.GetName()) == 0 &&  meas.GetGammaSyst().count(sys.GetName()) == 0 ) {             
            // just add the alpha for the parameters of the FlexibleInterpVar function 
            params.add(*alpha);                    
         }
                  // case systematic is a  log-normal constraint 
         if (meas.GetLogNormSyst().count(sys.GetName()) > 0 ) {
            // log normal constraint for parameter
            double relerr = meas.GetLogNormSyst().find(sys.GetName() )->second;
            double tauVal = 1./relerr; 
            std::string tauName = "tau_" + sys.GetName();
            proto->factory(TString::Format("%s[%f]",tauName.c_str(),tauVal ) );
            double kappaVal = 1. + relerr; 
            std::string kappaName = "kappa_" + sys.GetName();
            proto->factory(TString::Format("%s[%f]",kappaName.c_str(),kappaVal ) );
            const char * alphaName = alpha->GetName(); 
            
            std::string alphaOfBetaName = "alphaOfBeta_" + sys.GetName(); 
            RooAbsArg * alphaOfBeta = proto->factory(TString::Format("expr::%s('%s*(pow(%s,%s)-1.)',%s,%s,%s)",alphaOfBetaName.c_str(),
                                                                     tauName.c_str(),kappaName.c_str(),alphaName,
                                                                     tauName.c_str(),kappaName.c_str(),alphaName ) );

            cxcoutI(HistFactory) << "Added a log-normal constraint for " << name << std::endl;
            if (RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::DEBUG))
              alphaOfBeta->Print("t");
            params.add(*alphaOfBeta);
         }

      }
      // add low/high vectors
      double low = sys.GetLow();
      double high = sys.GetHigh();
      lowVec.push_back(low);
      highVec.push_back(high);

    }  // end sys loop 

    if(systList.size() > 0){
       // this is epsilon(alpha_j), a piece-wise linear interpolation
       //      LinInterpVar interp( (interpName).c_str(), "", params, 1., lowVec, highVec);
       
       assert( params.getSize() > 0);
       assert(int(lowVec.size()) == params.getSize() );
       
       FlexibleInterpVar interp( (interpName).c_str(), "", params, 1., lowVec, highVec);      
       interp.setAllInterpCodes(4); // LM: change to 4 (piece-wise linear to 6th order polynomial interpolation + linear extrapolation )
       //interp.setAllInterpCodes(0); // simple linear interpolation
       proto->import(interp); // params have already been imported in first loop of this function
    } else{
       // some strange behavior if params,lowVec,highVec are empty.  
       //cout << "WARNING: No OverallSyst terms" << endl;
       RooConstVar interp( (interpName).c_str(), "", 1.);
       proto->import(interp); // params have already been imported in first loop of this function
    }

    // std::cout << "after creating FlexibleInterpVar " << std::endl;
    // proto->Print();
    
  }


  void  HistoToWorkspaceFactoryFast::MakeTotalExpected(RooWorkspace* proto, const string& totName,
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
    const std::string binWidthFunctionName = totName + "_binWidth";
    RooBinWidthFunction binWidth(binWidthFunctionName.c_str(), "Divide by bin width to obtain probability density", *firstHistFunc, true);
    proto->import(binWidth);
    auto binWidthWS = proto->function(binWidthFunctionName.c_str());
    assert(binWidthWS);

    // Loop through samples and create products of their functions:
    RooArgSet coefList;
    RooArgSet shapeList;
    for (unsigned int i=0; i < sampleHistFuncs.size(); ++i) {
      assert(!sampleHistFuncs[i].empty());
      coefList.add(*sampleScaleFactors[i]);

      std::vector<RooAbsArg*>& thisSampleHistFuncs = sampleHistFuncs[i];
      thisSampleHistFuncs.push_back(binWidthWS);

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

        RooProduct shapeProduct(name.c_str(), name.c_str(), RooArgSet(thisSampleHistFuncs.begin(), thisSampleHistFuncs.end()));
        proto->import(shapeProduct, RecycleConflictNodes());
        shapeList.add(*proto->function(name.c_str()));
      }
    }

    // Sum all samples
    RooRealSumPdf tot(totName.c_str(), totName.c_str(), shapeList, coefList, true);
    tot.specialIntegratorConfig(kTRUE)->method1D().setLabel("RooBinIntegrator")  ;
    tot.specialIntegratorConfig(kTRUE)->method2D().setLabel("RooBinIntegrator")  ;
    tot.specialIntegratorConfig(kTRUE)->methodND().setLabel("RooBinIntegrator")  ;
    tot.forceNumInt();

    // for mixed generation in RooSimultaneous
    tot.setAttribute("GenerateBinned"); // for use with RooSimultaneous::generate in mixed mode

    proto->import(tot, RecycleConflictNodes());
  }

  void HistoToWorkspaceFactoryFast::AddPoissonTerms(RooWorkspace* proto, string prefix, string obsPrefix, string expPrefix, int lowBin, int highBin, 
           vector<string>& likelihoodTermNames){
    /////////////////////////////////
    // Relate observables to expected for each bin
    // later modify variable named expPrefix_i to be product of terms
    RooArgSet Pois(prefix.c_str());
    for(Int_t i=lowBin; i<highBin; ++i){
      std::stringstream str;
      str<<"_"<<i;
      //string command("Poisson::"+prefix+str.str()+"("+obsPrefix+str.str()+","+expPrefix+str.str()+")");
      string command("Poisson::"+prefix+str.str()+"("+obsPrefix+str.str()+","+expPrefix+str.str()+",1)");//for no rounding
      RooAbsArg* temp = (proto->factory( command.c_str() ) );

      // output
      cout << "Poisson Term " << command << endl;
      ((RooAbsPdf*) temp)->setEvalErrorLoggingMode(RooAbsReal::PrintErrors);
      //cout << temp << endl;

      likelihoodTermNames.push_back( temp->GetName() );
      Pois.add(* temp );
    }  
    proto->defineSet(prefix.c_str(),Pois); // add argset to workspace
  }

   void HistoToWorkspaceFactoryFast::SetObsToExpected(RooWorkspace* proto, string obsPrefix, string expPrefix, int lowBin, int highBin){ 
    /////////////////////////////////
    // set observed to expected
     TTree* tree = new TTree();
     Double_t* obsForTree = new Double_t[highBin-lowBin];
     RooArgList obsList("obsList");

     for(Int_t i=lowBin; i<highBin; ++i){
       std::stringstream str;
       str<<"_"<<i;
       RooRealVar* obs = (RooRealVar*) proto->var((obsPrefix+str.str()).c_str());
       cout << "expected number of events called: " << expPrefix << endl;
       RooAbsReal* exp = proto->function((expPrefix+str.str()).c_str());
       if(obs && exp){
         
         //proto->Print();
         obs->setVal(  exp->getVal() );
         cout << "setting obs"+str.str()+" to expected = " << exp->getVal() << " check: " << obs->getVal() << endl;
         
         // add entry to array and attach to tree
         obsForTree[i] = exp->getVal();
         tree->Branch((obsPrefix+str.str()).c_str(), obsForTree+i ,(obsPrefix+str.str()+"/D").c_str());
         obsList.add(*obs);
       }else{
         cout << "problem retrieving obs or exp " << obsPrefix+str.str() << obs << " " << expPrefix+str.str() << exp << endl;
       }
     }  
     tree->Fill();
     RooDataSet* data = new RooDataSet("expData","", tree, obsList); // one experiment

     delete tree;
     delete [] obsForTree;

     proto->import(*data);

     delete data; 

  }

  //////////////////////////////////////////////////////////////////////////////

  void HistoToWorkspaceFactoryFast::EditSyst(RooWorkspace* proto, const char* pdfNameChar, 
					     map<string,double> gammaSyst, 
					     map<string,double> uniformSyst, 
					     map<string,double> logNormSyst, 
					     map<string,double> noSyst) {
    string pdfName(pdfNameChar);

    ModelConfig * combined_config = (ModelConfig *) proto->obj("ModelConfig");
    if( combined_config==NULL ) {
      std::cout << "Error: Failed to find object 'ModelConfig' in workspace: " 
		<< proto->GetName() << std::endl;
      throw hf_exc();
    }
    //    const RooArgSet * constrainedParams=combined_config->GetNuisanceParameters();
    //    RooArgSet temp(*constrainedParams);
    string edit="EDIT::newSimPdf("+pdfName+",";
    string editList;
    string lastPdf=pdfName;
    string precede="";
    unsigned int numReplacements = 0;
    unsigned int nskipped = 0;
    map<string,double>::iterator it;

    
    // add gamma terms and their constraints
    for(it=gammaSyst.begin(); it!=gammaSyst.end(); ++it) {
      //cout << "edit for " << it->first << "with rel uncert = " << it->second << endl;
      if(! proto->var(("alpha_"+it->first).c_str())){
	//cout << "systematic not there" << endl;
	nskipped++; 
	continue;
      }
      numReplacements++;      

      double relativeUncertainty = it->second;
      double scale = 1/sqrt((1+1/pow(relativeUncertainty,2)));
      
      // this is the Gamma PDF and in a form that doesn't have roundoff problems like the Poisson does
      proto->factory(Form("beta_%s[1,0,10]",it->first.c_str()));
      proto->factory(Form("y_%s[%f]",it->first.c_str(),1./pow(relativeUncertainty,2))) ;
      proto->factory(Form("theta_%s[%f]",it->first.c_str(),pow(relativeUncertainty,2))) ;
      proto->factory(Form("Gamma::beta_%sConstraint(beta_%s,sum::k_%s(y_%s,one[1]),theta_%s,zero[0])",
			  it->first.c_str(),
			  it->first.c_str(),
			  it->first.c_str(),
			  it->first.c_str(),
			  it->first.c_str())) ;

      /*
      // this has some problems because N in poisson is rounded to nearest integer     
      proto->factory(Form("Poisson::beta_%sConstraint(y_%s[%f],prod::taub_%s(taus_%s[%f],beta_%s[1,0,5]))",
			  it->first.c_str(),
			  it->first.c_str(),
			  1./pow(relativeUncertainty,2),
			  it->first.c_str(),
			    it->first.c_str(),
			  1./pow(relativeUncertainty,2),
			  it->first.c_str()
			  ) ) ;
      */
      //	combined->factory(Form("expr::alphaOfBeta('(beta-1)/%f',beta)",scale));
      //	combined->factory(Form("expr::alphaOfBeta_%s('(beta_%s-1)/%f',beta_%s)",it->first.c_str(),it->first.c_str(),scale,it->first.c_str()));
      proto->factory(Form("PolyVar::alphaOfBeta_%s(beta_%s,{%f,%f})",it->first.c_str(),it->first.c_str(),-1./scale,1./scale));
	
      // set beta const status to be same as alpha
      if(proto->var(Form("alpha_%s",it->first.c_str()))->isConstant()) {
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(true);
      }
      else {
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(false);
      }
      // set alpha const status to true
      //      proto->var(Form("alpha_%s",it->first.c_str()))->setConstant(true);

      // replace alphas with alphaOfBeta and replace constraints
      editList+=precede + "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint";
      precede=",";
      editList+=precede + "alpha_"+it->first+"=alphaOfBeta_"+ it->first;

      /*
      if( proto->pdf(("alpha_"+it->first+"Constraint").c_str()) && proto->var(("alpha_"+it->first).c_str()) )
      cout << " checked they are there" << proto->pdf(("alpha_"+it->first+"Constraint").c_str()) << " " << proto->var(("alpha_"+it->first).c_str()) << endl;
      else
	cout << "NOT THERE" << endl;
      */

      // EDIT seems to die if the list of edits is too long.  So chunck them up.
      if(numReplacements%10 == 0 && numReplacements+nskipped!=gammaSyst.size()){
	edit="EDIT::"+lastPdf+"_("+lastPdf+","+editList+")";
	lastPdf+="_"; // append an underscore for the edit
	editList=""; // reset edit list
	precede="";
	cout << "Going to issue this edit command\n" << edit<< endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne)
	  cxcoutWHF << "---------------------\n WARNING: failed to make EDIT\n\n" << endl;
	
      }
    }

    // add uniform terms and their constraints
    for(it=uniformSyst.begin(); it!=uniformSyst.end(); ++it) {
      cout << "edit for " << it->first << "with rel uncert = " << it->second << endl;
      if(! proto->var(("alpha_"+it->first).c_str())){
	cout << "systematic not there" << endl;
	nskipped++; 
	continue;
      }
      numReplacements++;      

      // this is the Uniform PDF
      proto->factory(Form("beta_%s[1,0,10]",it->first.c_str()));
      proto->factory(Form("Uniform::beta_%sConstraint(beta_%s)",it->first.c_str(),it->first.c_str()));
      proto->factory(Form("PolyVar::alphaOfBeta_%s(beta_%s,{-1,1})",it->first.c_str(),it->first.c_str()));
      
      // set beta const status to be same as alpha
      if(proto->var(Form("alpha_%s",it->first.c_str()))->isConstant())
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(true);
      else
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(false);
      // set alpha const status to true
      //      proto->var(Form("alpha_%s",it->first.c_str()))->setConstant(true);

      // replace alphas with alphaOfBeta and replace constraints
      cout <<         "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint" << endl;
      editList+=precede + "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint";
      precede=",";
      cout <<         "alpha_"+it->first+"=alphaOfBeta_"+ it->first << endl;
      editList+=precede + "alpha_"+it->first+"=alphaOfBeta_"+ it->first;

      if( proto->pdf(("alpha_"+it->first+"Constraint").c_str()) && proto->var(("alpha_"+it->first).c_str()) )
	cout << " checked they are there" << proto->pdf(("alpha_"+it->first+"Constraint").c_str()) << " " << proto->var(("alpha_"+it->first).c_str()) << endl;
      else
	cout << "NOT THERE" << endl;

      // EDIT seems to die if the list of edits is too long.  So chunck them up.
      if(numReplacements%10 == 0 && numReplacements+nskipped!=gammaSyst.size()){
	edit="EDIT::"+lastPdf+"_("+lastPdf+","+editList+")";
	lastPdf+="_"; // append an underscore for the edit
	editList=""; // reset edit list
	precede="";
	cout << edit<< endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne)
	  cxcoutWHF <<  "---------------------\n WARNING: failed to make EDIT\n\n" << endl;
	
      }
    }

    /////////////////////////////////////////
    ////////////////////////////////////


    // add lognormal terms and their constraints
    for(it=logNormSyst.begin(); it!=logNormSyst.end(); ++it) {
      cout << "edit for " << it->first << "with rel uncert = " << it->second << endl;
      if(! proto->var(("alpha_"+it->first).c_str())){
	cout << "systematic not there" << endl;
	nskipped++; 
	continue;
      }
      numReplacements++;      

      double relativeUncertainty = it->second;
      double kappa = 1+relativeUncertainty;
      // when transforming beta -> alpha, need alpha=1 to be +1sigma value.
      // the P(beta>kappa*\hat(beta)) = 16%
      // and \hat(beta) is 1, thus
      double scale = relativeUncertainty;
      //double scale = kappa; 

      const char * cname  = it->first.c_str(); 

      // this is the LogNormal
      proto->factory(TString::Format("beta_%s[1,0,10]",cname));
      proto->factory(TString::Format("nom_beta_%s[1]",cname));
      proto->factory(TString::Format("kappa_%s[%f]",cname,kappa));
      proto->factory(TString::Format("Lognormal::beta_%sConstraint(beta_%s,nom_beta_%s,kappa_%s)",
                                     cname, cname, cname, cname)) ;
      proto->factory(TString::Format("PolyVar::alphaOfBeta_%s(beta_%s,{%f,%f})",cname,cname,-1./scale,1./scale));
      
      
      // set beta const status to be same as alpha
      if(proto->var(TString::Format("alpha_%s",cname))->isConstant())
	proto->var(TString::Format("beta_%s",cname))->setConstant(true);
      else
	proto->var(TString::Format("beta_%s",cname))->setConstant(false);
      // set alpha const status to true
      //      proto->var(TString::Format("alpha_%s",cname))->setConstant(true);

      // replace alphas with alphaOfBeta and replace constraints
      cout <<         "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint" << endl;
      editList+=precede + "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint";
      precede=",";
      cout <<         "alpha_"+it->first+"=alphaOfBeta_"+ it->first << endl;
      editList+=precede + "alpha_"+it->first+"=alphaOfBeta_"+ it->first;

      if( proto->pdf(("alpha_"+it->first+"Constraint").c_str()) && proto->var(("alpha_"+it->first).c_str()) )
	cout << " checked they are there" << proto->pdf(("alpha_"+it->first+"Constraint").c_str()) << " " << proto->var(("alpha_"+it->first).c_str()) << endl;
      else
	cout << "NOT THERE" << endl;

      // EDIT seems to die if the list of edits is too long.  So chunck them up.
      if(numReplacements%10 == 0 && numReplacements+nskipped!=gammaSyst.size()){
	edit="EDIT::"+lastPdf+"_("+lastPdf+","+editList+")";
	lastPdf+="_"; // append an underscore for the edit
	editList=""; // reset edit list
	precede="";
	cout << edit<< endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne)
	  cxcoutWHF << "\n\n ---------------------\n WARNING: failed to make EDIT\n\n" << endl;
	
      }
      // add global observables
      const RooArgSet * gobs = proto->set("globalObservables");
      RooArgSet gobsNew(*gobs); 
      gobsNew.add(*proto->var(TString::Format("nom_beta_%s",cname)) );
      proto->removeSet("globalObservables");
      proto->defineSet("globalObservables",gobsNew);
      gobsNew.Print();
      
    }

    /////////////////////////////////////////

    // MB: remove a systematic constraint
    for(it=noSyst.begin(); it!=noSyst.end(); ++it) {

      cout << "remove constraint for parameter" << it->first << endl;
      if(! proto->var(("alpha_"+it->first).c_str()) || ! proto->pdf(("alpha_"+it->first+"Constraint").c_str()) ) {
	cout << "systematic not there" << endl;
	nskipped++; 
	continue;
      }
      numReplacements++;      

      // dummy replacement pdf
      if ( !proto->var("one") ) { proto->factory("one[1.0]"); }
      proto->var("one")->setConstant();

      // replace constraints
      cout << "alpha_"+it->first+"Constraint=one" << endl;
      editList+=precede + "alpha_"+it->first+"Constraint=one";
      precede=",";

      // EDIT seems to die if the list of edits is too long.  So chunck them up.
      if(numReplacements%10 == 0 && numReplacements+nskipped!=gammaSyst.size()){
	edit="EDIT::"+lastPdf+"_("+lastPdf+","+editList+")";
	lastPdf+="_"; // append an underscore for the edit
	editList=""; // reset edit list
	precede="";
	cout << edit << endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne) {
	  cxcoutWHF << "---------------------\n WARNING: failed to make EDIT\n\n" << endl;
	}
      }
    }

    /////////////////////////////////////////

    // commit last bunch of edits
    edit="EDIT::newSimPdf("+lastPdf+","+editList+")";
    cout << edit<< endl;
    proto->factory( edit.c_str() );
    //    proto->writeToFile(("results/model_"+fRowTitle+"_edited.root").c_str());
    RooAbsPdf* newOne = proto->pdf("newSimPdf");
    if(newOne){
      // newOne->graphVizTree(("results/"+pdfName+"_"+fRowTitle+"newSimPdf.dot").c_str());
      combined_config->SetPdf(*newOne);
    }
    else{
      cxcoutWHF << "\n\n ---------------------\n WARNING: failed to make EDIT\n\n" << endl;
    }
  }

  void HistoToWorkspaceFactoryFast::PrintCovarianceMatrix(RooFitResult* result, RooArgSet* params, string filename){
    // Change-> Now a static utility

    FILE* covFile = fopen ((filename).c_str(),"w"); 

    TIter iti = params->createIterator();
    TIter itj = params->createIterator();
    RooRealVar *myargi, *myargj; 
    fprintf(covFile," ") ;
    while ((myargi = (RooRealVar *)iti.Next())) { 
      if(myargi->isConstant()) continue;
      fprintf(covFile," & %s",  myargi->GetName());
    }
    fprintf(covFile,"\\\\ \\hline \n" );
    iti.Reset();
    while ((myargi = (RooRealVar *)iti.Next())) { 
      if(myargi->isConstant()) continue;
      fprintf(covFile,"%s", myargi->GetName());
      itj.Reset();
      while ((myargj = (RooRealVar *)itj.Next())) { 
        if(myargj->isConstant()) continue;
        cout << myargi->GetName() << "," << myargj->GetName();
        fprintf(covFile, " & %.2f", result->correlation(*myargi, *myargj));
      }
      cout << endl;
      fprintf(covFile, " \\\\\n");
    }
    fclose(covFile);
    
  }


  ///////////////////////////////////////////////
  RooWorkspace* HistoToWorkspaceFactoryFast::MakeSingleChannelWorkspace(Measurement& measurement, Channel& channel) {

    // check inputs (see JIRA-6890 )

    if (channel.GetSamples().empty()) {
      Error("MakeSingleChannelWorkspace",
          "The input Channel does not contain any sample - return a nullptr");
      return 0;
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
      return 0;
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

    if (fObsNameVec.size() == 0 || fObsNameVec.size() >= 3) {
      throw hf_exc("HistFactory is limited to 1- to 3-dimensional histograms.");
    }

    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\tStarting to process '"
        << channel_name << "' channel with " << fObsNameVec.size() << " observables"
        << "\n-----------------------------------------\n" << endl;

    //
    // our main workspace that we are using to construct the model
    //
    RooWorkspace* proto = new RooWorkspace(channel_name.c_str(), (channel_name+" workspace").c_str());
    auto proto_config = make_unique<ModelConfig>("ModelConfig", proto);
    proto_config->SetWorkspace(*proto);

    // preprocess functions
    vector<string>::iterator funcIter = fPreprocessFunctions.begin();
    for(;funcIter!= fPreprocessFunctions.end(); ++funcIter){
      cxcoutI(HistFactory) << "will preprocess this line: " << *funcIter <<endl;
      proto->factory(funcIter->c_str());
      proto->Print();
    }

    RooArgSet likelihoodTerms("likelihoodTerms"), constraintTerms("constraintTerms");
    vector<string> likelihoodTermNames, constraintTermNames, totSystTermNames;
    // All histogram functions to be multiplied in each sample
    std::vector<std::vector<RooAbsArg*>> allSampleHistFuncs;
    std::vector<RooProduct*> sampleScaleFactors;

    std::vector< pair<string,string> >   statNamePairs;
    std::vector< pair<const TH1*, std::unique_ptr<TH1>> > statHistPairs; // <nominal, error>
    const std::string statFuncName = "mc_stat_" + channel_name;

    string prefix, range;

    /////////////////////////////
    // shared parameters
    // this is ratio of lumi to nominal lumi.  We will include relative uncertainty in model
    std::stringstream lumiStr;
    // lumi range
    lumiStr << "Lumi[" << fNomLumi << ",0," << 10.*fNomLumi << "]";
    proto->factory(lumiStr.str().c_str());
    cxcoutI(HistFactory) << "lumi str = " << lumiStr.str() << endl;

    std::stringstream lumiErrorStr;
    lumiErrorStr << "nominalLumi["<<fNomLumi << ",0,"<<fNomLumi+10*fLumiError<<"]," << fLumiError ;
    proto->factory(("Gaussian::lumiConstraint(Lumi,"+lumiErrorStr.str()+")").c_str());
    proto->var("nominalLumi")->setConstant();
    proto->defineSet("globalObservables","nominalLumi");
    //likelihoodTermNames.push_back("lumiConstraint");
    constraintTermNames.push_back("lumiConstraint");
    cxcoutI(HistFactory) << "lumi Error str = " << lumiErrorStr.str() << endl;

    //proto->factory((string("SigXsecOverSM[1.,0.5,1..8]").c_str()));
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

      if(sample.GetHistoSysList().size() == 0) {
        // If no HistoSys
        cxcoutI(HistFactory) << sample.GetName() + "_" + channel_name + " has no variation histograms " << endl;

        sampleHistFuncs.push_back(nominalHistFunc);
      } else {
        // If there ARE HistoSys(s)
        // name of source for variation
        string constraintPrefix = sample.GetName() + "_" + channel_name + "_Hist_alpha"; 

        // make list of abstract parameters that interpolate in space of variations
        RooArgList interpParams = makeInterpolationParameters(sample.GetHistoSysList(), *proto);

        // next, cerate the constraint terms
        for(std::size_t i = 0; i < interpParams.size(); ++i) {
          bool isUniform = measurement.GetUniformSyst().count(sample.GetHistoSysList()[i].GetName()) > 0;
          makeGaussianConstraint(interpParams[i], *proto, isUniform, constraintTermNames);
        }

        // finally, create the interpolated function
        sampleHistFuncs.push_back( makeLinInterp(interpParams, nominalHistFunc, proto,
            sample.GetHistoSysList(), constraintPrefix, observables) );
      }

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

          if( sample.GetStatError().GetErrorHist() == NULL ) {
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
          ParamHistFunc* paramHist = dynamic_cast<ParamHistFunc*>( proto->function(statFuncName.c_str()) );
          if( paramHist == nullptr ) {

            // Get a RooArgSet of the observables:
            // Names in the list fObsNameVec:
            RooArgList theObservables;
            std::vector<std::string>::iterator itr = fObsNameVec.begin();
            for (int idx=0; itr!=fObsNameVec.end(); ++itr, ++idx ) {
              theObservables.add( *proto->var(itr->c_str()) );
            }

            // Create the list of terms to
            // control the bin heights:
            std::string ParamSetPrefix  = "gamma_stat_" + channel_name;
            Double_t gammaMin = 0.0;
            Double_t gammaMax = 10.0;
            RooArgList statFactorParams = ParamHistFunc::createParamSet(*proto,
                ParamSetPrefix.c_str(),
                theObservables,
                gammaMin, gammaMax);

            ParamHistFunc statUncertFunc(statFuncName.c_str(), statFuncName.c_str(),
                theObservables, statFactorParams );

            proto->import( statUncertFunc, RecycleConflictNodes() );

            paramHist = (ParamHistFunc*) proto->function( statFuncName.c_str() );
          }

          // apply stat function to sample
          sampleHistFuncs.push_back(paramHist);
        }
      } // END: if DoMcStat


      ///////////////////////////////////////////
      // Create a ShapeFactor for this channel //
      ///////////////////////////////////////////

      if( sample.GetShapeFactorList().size() > 0 ) {

        if( fObsNameVec.size() > 3 ) {
          cxcoutF(HistFactory) << "Cannot include Stat Error for histograms of more than 3 dimensions."
              << std::endl;
          throw hf_exc();
        } else {

          cxcoutI(HistFactory) << "Sample: "     << sample.GetName() << " in channel: " << channel_name
              << " to be include a ShapeFactor."
              << std::endl;

          for(unsigned int i=0; i < sample.GetShapeFactorList().size(); ++i) {

            ShapeFactor& shapeFactor = sample.GetShapeFactorList().at(i);

            std::string funcName = channel_name + "_" + shapeFactor.GetName() + "_shapeFactor";
            ParamHistFunc* paramHist = (ParamHistFunc*) proto->function( funcName.c_str() );
            if( paramHist == nullptr ) {

              RooArgList theObservables;
              std::vector<std::string>::iterator itr = fObsNameVec.begin();
              for (int idx=0; itr!=fObsNameVec.end(); ++itr, ++idx ) {
                theObservables.add( *proto->var(itr->c_str()) );
              }

              // Create the Parameters
              std::string funcParams = "gamma_" + shapeFactor.GetName();

              // GHL: Again, we are putting hard ranges on the gamma's
              //      We should change this to range from 0 to /inf
              RooArgList shapeFactorParams = ParamHistFunc::createParamSet(*proto,
                  funcParams.c_str(),
                  theObservables, 0, 1000);

              // Create the Function
              ParamHistFunc shapeFactorFunc( funcName.c_str(), funcName.c_str(),
                  theObservables, shapeFactorParams );

              // Set an initial shape, if requested
              if( shapeFactor.GetInitialShape() != NULL ) {
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

              proto->import( shapeFactorFunc, RecycleConflictNodes() );
              paramHist = (ParamHistFunc*) proto->function( funcName.c_str() );

            } // End: Create ShapeFactor ParamHistFunc

            sampleHistFuncs.push_back(paramHist);
          } // End loop over ShapeFactor Systematics
        }
      } // End: if ShapeFactorName!=""


      ////////////////////////////////////////
      // Create a ShapeSys for this channel //
      ////////////////////////////////////////

      if( sample.GetShapeSysList().size() != 0 ) {

        if( fObsNameVec.size() > 3 ) {
          cxcoutF(HistFactory) << "Cannot include Stat Error for histograms of more than 3 dimensions."
              << std::endl;
          throw hf_exc();
        } else {

          // List of ShapeSys ParamHistFuncs
          std::vector<string> ShapeSysNames;

          for( unsigned int i = 0; i < sample.GetShapeSysList().size(); ++i) {

            // Create the ParamHistFunc's
            // Create their constraint terms and add them
            // to the list of constraint terms

            // Create a single RooProduct over all of these
            // paramHistFunc's

            // Send the name of that product to the RooRealSumPdf

            RooStats::HistFactory::ShapeSys& shapeSys = sample.GetShapeSysList().at(i);

            cxcoutI(HistFactory) << "Sample: " << sample.GetName() << " in channel: " << channel_name
                << " to include a ShapeSys." << std::endl;

            std::string funcName = channel_name + "_" + shapeSys.GetName() + "_ShapeSys";
            ShapeSysNames.push_back( funcName );
            ParamHistFunc* paramHist = (ParamHistFunc*) proto->function( funcName.c_str() );
            if( paramHist == NULL ) {

              //std::string funcParams = "gamma_" + it->shapeFactorName;
              //paramHist = CreateParamHistFunc( proto, fObsNameVec, funcParams, funcName );

              RooArgList theObservables;
              std::vector<std::string>::iterator itr = fObsNameVec.begin();
              for(; itr!=fObsNameVec.end(); ++itr ) {
                theObservables.add( *proto->var(itr->c_str()) );
              }

              // Create the Parameters
              std::string funcParams = "gamma_" + shapeSys.GetName();
              RooArgList shapeFactorParams = ParamHistFunc::createParamSet(*proto,
                  funcParams.c_str(),
                  theObservables, 0, 10);

              // Create the Function
              ParamHistFunc shapeFactorFunc( funcName.c_str(), funcName.c_str(),
                  theObservables, shapeFactorParams );

              proto->import( shapeFactorFunc, RecycleConflictNodes() );
              paramHist = (ParamHistFunc*) proto->function( funcName.c_str() );

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

            Double_t minShapeUncertainty = 0.0;
            RooArgList shapeConstraints = createStatConstraintTerms(proto, constraintTermNames,
                *paramHist, shapeErrorHist,
                systype,
                minShapeUncertainty);

          } // End: Loop over ShapeSys vector in this EstimateSummary

          // Now that we have the list of ShapeSys ParamHistFunc names,
          // we create the total RooProduct
          // we multiply the expected functio


          for( unsigned int i = 0; i < ShapeSysNames.size(); ++i ) {
            auto func = proto->function(ShapeSysNames.at(i).c_str());
            assert(func);
            sampleHistFuncs.push_back(func);
          }

        } // End: NumObsVar == 1

      } // End: GetShapeSysList.size() != 0


      // GHL: This was pretty confusing before,
      //      hopefully using the measurement directly
      //      will improve it
      auto lumi = proto->arg("Lumi");
      if( !sample.GetNormalizeByTheory() ) {
        if (!lumi) {
          TString lumiParamString;
          lumiParamString += measurement.GetLumi();
          lumiParamString.ReplaceAll(' ', TString());
          lumi = proto->factory(("Lumi[" + lumiParamString + "]").Data());
        } else {
          static_cast<RooAbsRealLValue*>(lumi)->setVal(measurement.GetLumi());
        }
      }
      assert(lumi);
      normFactors->addTerm(lumi);

      // Append the name of the "node"
      // that is to be summed with the
      // RooRealSumPdf
      proto->import(*normFactors, RecycleConflictNodes());
      auto normFactorsInWS = dynamic_cast<RooProduct*>(proto->arg(normFactors->GetName()));
      assert(normFactorsInWS);

      sampleScaleFactors.push_back(normFactorsInWS);
    } // END: Loop over EstimateSummaries

    // If a non-zero number of samples call for
    // Stat Uncertainties, create the statFactor functions
    if(!statHistPairs.empty()) {

      // Create the histogram of (binwise)
      // stat uncertainties:
      unique_ptr<TH1> fracStatError( MakeScaledUncertaintyHist( channel_name + "_StatUncert" + "_RelErr", statHistPairs) );
      if( fracStatError == NULL ) {
        cxcoutE(HistFactory) << "Error: Failed to make ScaledUncertaintyHist for: "
            << channel_name + "_StatUncert" + "_RelErr" << std::endl;
        throw hf_exc();
      }

      // Using this TH1* of fractinal stat errors, 
      // create a set of constraint terms:
      ParamHistFunc* chanStatUncertFunc = (ParamHistFunc*) proto->function( statFuncName.c_str() );
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
      RooArgList statConstraints = createStatConstraintTerms(proto, constraintTermNames, 
          *chanStatUncertFunc, fracStatError.get(),
          statConstraintType,
          statRelErrorThreshold);

    } // END: Loop over stat Hist Pairs


    ///////////////////////////////////
    // for ith bin calculate totN_i =  lumi * sum_j expected_j * syst_j 
    MakeTotalExpected(proto, channel_name+"_model",
        sampleScaleFactors, allSampleHistFuncs);
    likelihoodTermNames.push_back(channel_name+"_model");

    //////////////////////////////////////
    // fix specified parameters
    for(unsigned int i=0; i<systToFix.size(); ++i){
      RooRealVar* temp = proto->var((systToFix.at(i)).c_str());
      if(temp) {
        // set the parameter constant
        temp->setConstant();

        // remove the corresponding auxiliary observable from the global observables
        RooRealVar* auxMeas = NULL;
        if(systToFix.at(i)=="Lumi"){
          auxMeas = proto->var("nominalLumi");
        } else {
          auxMeas = proto->var(TString::Format("nom_%s",temp->GetName()));
        }

        if(auxMeas){
          const_cast<RooArgSet*>(proto->set("globalObservables"))->remove(*auxMeas);
        } else{
          cxcoutE(HistFactory) << "could not corresponding auxiliary measurement  "
              << TString::Format("nom_%s",temp->GetName()) << endl;
        }
      } else {
        cxcoutE(HistFactory) << "could not find variable " << systToFix.at(i)
	         << " could not set it to constant" << endl;
      }
    }

    //////////////////////////////////////
    // final proto model
    for(unsigned int i=0; i<constraintTermNames.size(); ++i){
      RooAbsArg* proto_arg = (proto->arg(constraintTermNames[i].c_str()));
      if( proto_arg==NULL ) {
        cxcoutF(HistFactory) << "Error: Cannot find arg set: " << constraintTermNames.at(i)
		      << " in workspace: " << proto->GetName() << std::endl;
        throw hf_exc();
      }
      constraintTerms.add( *proto_arg );
      //  constraintTerms.add(* proto_arg(proto->arg(constraintTermNames[i].c_str())) );
    }
    for(unsigned int i=0; i<likelihoodTermNames.size(); ++i){
      RooAbsArg* proto_arg = (proto->arg(likelihoodTermNames[i].c_str())); 
      if( proto_arg==NULL ) {
        cxcoutF(HistFactory) << "Error: Cannot find arg set: " << likelihoodTermNames.at(i)
		      << " in workspace: " << proto->GetName() << std::endl;
        throw hf_exc();
      }
      likelihoodTerms.add( *proto_arg );
    }
    proto->defineSet("constraintTerms",constraintTerms);
    proto->defineSet("likelihoodTerms",likelihoodTerms);

    // list of observables
    RooArgList observables;
    std::string observablesStr;

    std::vector<std::string>::iterator itr = fObsNameVec.begin();
    for(; itr!=fObsNameVec.end(); ++itr ) {
      observables.add( *proto->var(itr->c_str()) );
      if (!observablesStr.empty()) { observablesStr += ","; }
      observablesStr += *itr;
    }

    // We create two sets, one for backwards compatability
    // The other to make a consistent naming convention
    // between individual channels and the combined workspace
    proto->defineSet("observables", TString::Format("%s",observablesStr.c_str()));
    proto->defineSet("observablesSet", TString::Format("%s",observablesStr.c_str()));

    // Create the ParamHistFunc
    // after observables have been made
    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\timport model into workspace"
        << "\n-----------------------------------------\n" << endl;

    auto model = make_unique<RooProdPdf>(
        ("model_"+channel_name).c_str(),    // MB : have changed this into conditional pdf. Much faster for toys!
        "product of Poissons accross bins for a single channel",
        constraintTerms, Conditional(likelihoodTerms,observables));
    proto->import(*model,RecycleConflictNodes());

    proto_config->SetPdf(*model);
    proto_config->SetObservables(observables);
    proto_config->SetGlobalObservables(*proto->set("globalObservables"));
    //    proto->writeToFile(("results/model_"+channel+".root").c_str());
    // fill out nuisance parameters in model config
    //    proto_config->GuessObsAndNuisance(*proto->data("asimovData"));
    proto->import(*proto_config,proto_config->GetName());
    proto->importClassCode();

    ///////////////////////////
    // make data sets
    // THis works and is natural, but the memory size of the simultaneous dataset grows exponentially with channels
    const char* weightName="weightVar";
    proto->factory(TString::Format("%s[0,-1e10,1e10]",weightName));
    proto->defineSet("obsAndWeight",TString::Format("%s,%s",weightName,observablesStr.c_str()));

    // New Asimov Generation: Use the code in the Asymptotic calculator 
    // Need to get the ModelConfig...
    int asymcalcPrintLevel = 0;
    if (RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::INFO)) asymcalcPrintLevel = 1;
    if (RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::DEBUG)) asymcalcPrintLevel = 2;
    AsymptoticCalculator::SetPrintLevel(asymcalcPrintLevel);
    unique_ptr<RooAbsData> asimov_dataset(AsymptoticCalculator::GenerateAsimovData(*model, observables));
    proto->import(dynamic_cast<RooDataSet&>(*asimov_dataset), Rename("asimovData"));

    // GHL: Determine to use data if the hist isn't 'NULL'
    if(TH1 const* mnominal = channel.GetData().GetHisto()) {
      // This works and is natural, but the memory size of the simultaneous
      // dataset grows exponentially with channels.
      RooDataSet dataset{"obsData","",*proto->set("obsAndWeight"),weightName};
      ConfigureHistFactoryDataset( dataset, *mnominal, *proto, fObsNameVec );
      proto->import(dataset);
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
		          << " with name: " << dataName << " is NULL" << std::endl;
        throw hf_exc();
      }

      // THis works and is natural, but the memory size of the simultaneous dataset grows exponentially with channels
      RooDataSet dataset{dataName.c_str(), "", *proto->set("obsAndWeight"), weightName};
      ConfigureHistFactoryDataset( dataset, *mnominal, *proto, fObsNameVec );
      proto->import(dataset);

    }

    if (RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::INFO))
      proto->Print();

    return proto;
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

    for (int i=1; i<=ax->GetNbins(); ++i) { // 1 or more dimension

      double xval = ax->GetBinCenter(i);
      proto.var( obsNameVec[0].c_str() )->setVal( xval );

      if(obsNameVec.size()==1) {
   double fval = mnominal.GetBinContent(i);
   obsDataUnbinned.add( *proto.set("obsAndWeight"), fval );
      } else { // 2 or more dimensions

   for(int j=1; j<=ay->GetNbins(); ++j) {
     double yval = ay->GetBinCenter(j);
     proto.var( obsNameVec[1].c_str() )->setVal( yval );

     if(obsNameVec.size()==2) {
       double fval = mnominal.GetBinContent(i,j);
       obsDataUnbinned.add( *proto.set("obsAndWeight"), fval );
     } else { // 3 dimensions

       for(int k=1; k<=az->GetNbins(); ++k) {
         double zval = az->GetBinCenter(k);
         proto.var( obsNameVec[2].c_str() )->setVal( zval );
         double fval = mnominal.GetBinContent(i,j,k);
         obsDataUnbinned.add( *proto.set("obsAndWeight"), fval );
       }
     }
   }
      }
    }
  }

  void HistoToWorkspaceFactoryFast::GuessObsNameVec(const TH1* hist)
  {
    fObsNameVec.clear();

    // determine histogram dimensionality 
    unsigned int histndim(1);
    std::string classname = hist->ClassName();
    if      (classname.find("TH1")==0) { histndim=1; }
    else if (classname.find("TH2")==0) { histndim=2; }
    else if (classname.find("TH3")==0) { histndim=3; }

    for ( unsigned int idx=0; idx<histndim; ++idx ) {
      if (idx==0) { fObsNameVec.push_back("x"); }
      if (idx==1) { fObsNameVec.push_back("y"); }
      if (idx==2) { fObsNameVec.push_back("z"); }
    }
  }


  RooWorkspace* HistoToWorkspaceFactoryFast::MakeCombinedModel(vector<string> ch_names, vector<std::unique_ptr<RooWorkspace>>& chs)
  {
    RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::INFO, 0, RooFit::ObjectHandling, false);

     // check first the inputs (see JIRA-6890)
     if (ch_names.empty() || chs.empty() ) {
        Error("MakeCombinedModel","Input vectors are empty - return a nullptr");
        return 0;
     }
     if (chs.size()  <  ch_names.size() ) {
        Error("MakeCombinedModel","Input vector of workspace has an invalid size - return a nullptr");
        return 0;
     }

    //
    /// These things were used for debugging. Maybe useful in the future
    //

    map<string, RooAbsPdf*> pdfMap;
    vector<RooAbsPdf*> models;

    RooArgList obsList;
    for(unsigned int i = 0; i< ch_names.size(); ++i){
      ModelConfig * config = (ModelConfig *) chs[i]->obj("ModelConfig");
      obsList.add(*config->GetObservables());
    }
    cxcoutI(HistFactory) <<"full list of observables:\n" << obsList << std::endl;

    RooArgSet globalObs;
    stringstream channelString;
    channelString << "channelCat[";
    for(unsigned int i = 0; i< ch_names.size(); ++i){
      string channel_name=ch_names[i];
      if (i == 0 && isdigit(channel_name[0])) {
        throw std::invalid_argument("The first channel name for HistFactory cannot start with a digit. Got " + channel_name);
      }
      if (channel_name.find(',') != std::string::npos) {
        throw std::invalid_argument("Channel names for HistFactory cannot contain ','. Got " + channel_name);
      }

      if (i == 0) channelString << channel_name ;
      else channelString << ',' << channel_name ;
      RooWorkspace * ch=chs[i].get();
      
      RooAbsPdf* model = ch->pdf(("model_"+channel_name).c_str());
      if(!model) cout <<"failed to find model for channel"<<endl;
      //      cout << "int = " << model->createIntegral(*obsN)->getVal() << endl;;
      models.push_back(model);
      globalObs.add(*ch->set("globalObservables"), /*silent=*/true); // silent because observables might exist in other channel.

      //      constrainedParams->add( * ch->set("constrainedParams") );
      pdfMap[channel_name]=model;
    }
    channelString << "]";

    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\tEntering combination"
        << "\n-----------------------------------------\n" << endl;
    RooWorkspace* combined = new RooWorkspace("combined");
    //    RooWorkspace* combined = chs[0];
    

    RooCategory* channelCat = dynamic_cast<RooCategory*>( combined->factory(channelString.str().c_str()) );
    if (!channelCat) throw std::runtime_error("Unable to construct a category from string " + channelString.str());

    auto simPdf= std::make_unique<RooSimultaneous>("simPdf","",pdfMap, *channelCat);
    auto combined_config = std::make_unique<ModelConfig>("ModelConfig", combined);
    combined_config->SetWorkspace(*combined);
    //    combined_config->SetNuisanceParameters(*constrainedParams);

    combined->import(globalObs);
    combined->defineSet("globalObservables",globalObs);
    combined_config->SetGlobalObservables(*combined->set("globalObservables"));
    

    ////////////////////////////////////////////
    // Make toy simultaneous dataset
    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
        << "\tcreate toy data for " << channelString.str()
        << "\n-----------------------------------------\n" << endl;
    

    // now with weighted datasets
    // First Asimov
    //RooDataSet * simData=NULL;
    combined->factory("weightVar[0,-1e10,1e10]");
    obsList.add(*combined->var("weightVar"));

    // Create Asimov data for the combined dataset
    std::unique_ptr<RooDataSet> asimov_combined{static_cast<RooDataSet*>(AsymptoticCalculator::GenerateAsimovData(*simPdf,
                                  obsList))};
    if( asimov_combined ) {
      combined->import( *asimov_combined, Rename("asimovData"));
    }
    else {
      std::cout << "Error: Failed to create combined asimov dataset" << std::endl;
      throw hf_exc();
    }

    // Now merge the observable datasets across the channels
    for(RooAbsData * data : chs[0]->allData()) {
      // Only include RooDataSets where a data with the same name doesn't only
      // exist in the merged workspace (like the "asimovData").
      if(dynamic_cast<RooDataSet*>(data) && !combined->data(data->GetName())) {
        MergeDataSets(combined, chs, ch_names, data->GetName(), obsList, channelCat);
      }
    }


    obsList.add(*channelCat);
    combined->defineSet("observables",obsList);
    combined_config->SetObservables(*combined->set("observables"));

    if (RooMsgService::instance().isActive(static_cast<TObject*>(nullptr), RooFit::HistFactory, RooFit::INFO))
      combined->Print();

    cxcoutP(HistFactory) << "\n-----------------------------------------\n"
            << "\tImporting combined model"
            << "\n-----------------------------------------\n" << endl;
    combined->import(*simPdf,RecycleConflictNodes());
    //combined->import(*simPdf, RenameVariable("SigXsecOverSM","SigXsecOverSM_comb"));
    // cout << "check pointer " << simPdf << endl;
    //    cout << "check val " << simPdf->getVal() << endl;

    std::map< std::string, double>::iterator param_itr = fParamValues.begin();
    for( ; param_itr != fParamValues.end(); ++param_itr ){
      // make sure they are fixed
      std::string paramName = param_itr->first;
      double paramVal = param_itr->second;

      if(RooRealVar* temp = combined->var( paramName.c_str() )) {
        temp->setVal( paramVal );
        cxcoutI(HistFactory) <<"setting " << paramName << " to the value: " << paramVal <<  endl;
      } else 
        cxcoutE(HistFactory) << "could not find variable " << paramName << " could not set its value" << endl;
    }


    for(unsigned int i=0; i<fSystToFix.size(); ++i){
      // make sure they are fixed
      if(RooRealVar* temp = combined->var(fSystToFix[i].c_str())) {
        temp->setConstant();
        cxcoutI(HistFactory) <<"setting " << fSystToFix.at(i) << " constant" << endl;
      } else 
        cxcoutE(HistFactory) << "could not find variable " << fSystToFix.at(i) << " could not set it to constant" << endl;
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

    return combined;
  }


  RooDataSet* HistoToWorkspaceFactoryFast::MergeDataSets(RooWorkspace* combined,
                      std::vector<std::unique_ptr<RooWorkspace>>& wspace_vec,
                      std::vector<std::string> const& channel_names,
                      std::string const& dataSetName,
                      RooArgList obsList,
                      RooCategory* channelCat) {

    // Create the total dataset
    std::unique_ptr<RooDataSet> simData;

    // Loop through channels, get their individual datasets,
    // and add them to the combined dataset
    for(unsigned int i = 0; i< channel_names.size(); ++i){

      // Grab the dataset for the existing channel
      cxcoutPHF << "Merging data for channel " << channel_names[i].c_str() << std::endl;
      RooDataSet* obsDataInChannel = (RooDataSet*) wspace_vec[i]->data(dataSetName.c_str());
      if( !obsDataInChannel ) {
	std::cout << "Error: Can't find DataSet: " << dataSetName
		  << " in channel: " << channel_names.at(i)
		  << std::endl;
	throw hf_exc();
      }

      // Create the new Dataset
      auto tempData = std::make_unique<RooDataSet>(channel_names[i].c_str(),"",
                    obsList, Index(*channelCat),
                    WeightVar("weightVar"),
                    Import(channel_names[i].c_str(),*obsDataInChannel));
      if(simData) {
   simData->append(*tempData);
      }
      else {
   simData = std::move(tempData);
      }
    } // End Loop Over Channels
      
    // Check that we successfully created the dataset
    // and import it into the workspace
    if(simData) {
      combined->import(*simData, Rename(dataSetName.c_str()));
      return static_cast<RooDataSet*>(combined->data(dataSetName.c_str()));
    }
    else {
      std::cout << "Error: Unable to merge observable datasets" << std::endl;
      throw hf_exc();
      return nullptr;
    }
  }
    

  TH1* HistoToWorkspaceFactoryFast::MakeAbsolUncertaintyHist( const std::string& Name, const TH1* Nominal ) {

    // Take a nominal TH1* and create
    // a TH1 representing the binwise
    // errors (taken from the nominal TH1)

    TH1* ErrorHist = (TH1*) Nominal->Clone( Name.c_str() );
    ErrorHist->Reset();
    
    Int_t numBins   = Nominal->GetNbinsX()*Nominal->GetNbinsY()*Nominal->GetNbinsZ();
    Int_t binNumber = 0;

    // Loop over bins
    for( Int_t i_bin = 0; i_bin < numBins; ++i_bin) {

      binNumber++;
      // Ignore underflow / overflow
      while( Nominal->IsBinUnderflow(binNumber) || Nominal->IsBinOverflow(binNumber) ){
	binNumber++;
      }

      Double_t histError = Nominal->GetBinError( binNumber );
    
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
      return NULL;
    }
    
    const TH1* HistTemplate = HistVec.at(0).first;
    Int_t numBins = HistTemplate->GetNbinsX()*HistTemplate->GetNbinsY()*HistTemplate->GetNbinsZ();

  // Check that all histograms
  // have the same bins
  for( unsigned int i = 0; i < HistVec.size(); ++i ) {
    
    const TH1* nominal = HistVec.at(i).first;
    const TH1* error   = HistVec.at(i).second.get();

    if( nominal->GetNbinsX()*nominal->GetNbinsY()*nominal->GetNbinsZ() != numBins ) {
      cxcoutE(HistFactory) << "Error: Provided hists have unequal bins" << std::endl;
      return NULL;
    }
    if( error->GetNbinsX()*error->GetNbinsY()*error->GetNbinsZ() != numBins ) {
      cxcoutE(HistFactory) << "Error: Provided hists have unequal bins" << std::endl;
      return NULL;
    }
  }

  std::vector<double> TotalBinContent( numBins, 0.0);
  std::vector<double> HistErrorsSqr( numBins, 0.0);

  Int_t binNumber = 0;

  // Loop over bins
  for( Int_t i_bins = 0; i_bins < numBins; ++i_bins) {
    
    binNumber++;
    while( HistTemplate->IsBinUnderflow(binNumber) || HistTemplate->IsBinOverflow(binNumber) ){
      binNumber++;
    }
    
    for( unsigned int i_hist = 0; i_hist < numHists; ++i_hist ) {
      
      const TH1* nominal = HistVec.at(i_hist).first;
      const TH1* error   = HistVec.at(i_hist).second.get();

      //Int_t binNumber = i_bins + 1;

      Double_t histValue  = nominal->GetBinContent( binNumber );
      Double_t histError  = error->GetBinContent( binNumber );
      /*
      std::cout << " Getting Bin content for Stat Uncertainty"
		<< " Nom name: " << nominal->GetName()
		<< " Err name: " << error->GetName()
		<< " HistNumber: " << i_hist << " bin: " << binNumber
		<< " Value: " << histValue << " Error: " << histError
		<< std::endl;
      */

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
  TH1* ErrorHist = (TH1*) HistTemplate->Clone( Name.c_str() );
  ErrorHist->Reset();

  // Fill the output histogram
  for( Int_t i = 0; i < numBins; ++i) {

    //    Int_t binNumber = i + 1;
    binNumber++;
    while( ErrorHist->IsBinUnderflow(binNumber) || ErrorHist->IsBinOverflow(binNumber) ){
      binNumber++;
    }

    Double_t ErrorsSqr = HistErrorsSqr.at(i);
    Double_t TotalVal  = TotalBinContent.at(i);

    if( TotalVal <= 0 ) {
      cxcoutW(HistFactory) << "Warning: Sum of histograms for bin: " << binNumber
		<< " is <= 0.  Setting error to 0"
		<< std::endl;

      ErrorHist->SetBinContent( binNumber, 0.0 );
      continue;
    }

    Double_t RelativeError = sqrt(ErrorsSqr) / TotalVal;

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



  RooArgList HistoToWorkspaceFactoryFast::
  createStatConstraintTerms( RooWorkspace* proto, vector<string>& constraintTermNames,
			     ParamHistFunc& paramHist, const TH1* uncertHist,
			     Constraint::Type type, Double_t minSigma ) {


  // Take a RooArgList of RooAbsReal's and
  // create N constraint terms (one for
  // each gamma) whose relative uncertainty
  // is the value of the ith RooAbsReal
  //
  // The integer "type" controls the type
  // of constraint term:
  //
  // type == 0 : NONE
  // type == 1 : Gaussian
  // type == 2 : Poisson
  // type == 3 : LogNormal

  RooArgList ConstraintTerms;

  RooArgList paramSet = paramHist.paramList();

  // Must get the full size of the TH1
  // (No direct method to do this...)
  Int_t numBins   = uncertHist->GetNbinsX()*uncertHist->GetNbinsY()*uncertHist->GetNbinsZ();
  Int_t numParams = paramSet.getSize();
  //  Int_t numBins   = uncertHist->GetNbinsX()*uncertHist->GetNbinsY()*uncertHist->GetNbinsZ();

  // Check that there are N elements
  // in the RooArgList
  if( numBins != numParams ) {
    std::cout << "Error: In createStatConstraintTerms, encountered bad number of bins" << std::endl;
    std::cout << "Given histogram with " << numBins << " bins,"
	      << " but require exactly " << numParams << std::endl;
    throw hf_exc();
  }

  Int_t TH1BinNumber = 0;
  for( Int_t i = 0; i < paramSet.getSize(); ++i) {

    TH1BinNumber++;

    while( uncertHist->IsBinUnderflow(TH1BinNumber) || uncertHist->IsBinOverflow(TH1BinNumber) ){
      TH1BinNumber++;
    }

    RooRealVar& gamma = (RooRealVar&) (paramSet[i]);

    cxcoutI(HistFactory) << "Creating constraint for: " << gamma.GetName()
	      << ". Type of constraint: " << type <<  std::endl;

    // Get the sigma from the hist
    // (the relative uncertainty)
    const double sigmaRel = uncertHist->GetBinContent(TH1BinNumber); 

    // If the sigma is <= 0, 
    // do cont create the term
    if( sigmaRel <= 0 ){
      cxcoutI(HistFactory) << "Not creating constraint term for "
		<< gamma.GetName() 
		<< " because sigma = " << sigmaRel
		<< " (sigma<=0)" 
		<< " (TH1 bin number = " << TH1BinNumber << ")"
		<< std::endl;
      gamma.setConstant(kTRUE);
      continue;
    }
  
    // set reasonable ranges for gamma parameters
    gamma.setMax( 1 + 5*sigmaRel );
    gamma.setMin( 0. );         

    // Make Constraint Term
    std::string constrName = string(gamma.GetName()) + "_constraint";
    std::string nomName = string("nom_") + gamma.GetName();
    std::string sigmaName = string(gamma.GetName()) + "_sigma";
    std::string poisMeanName = string(gamma.GetName()) + "_poisMean";

    if( type == Constraint::Gaussian ) {

      // Type 1 : RooGaussian
    
      // Make sigma

      RooConstVar constrSigma( sigmaName.c_str(), sigmaName.c_str(), sigmaRel );
    
      // Make "observed" value
      RooRealVar constrNom(nomName.c_str(), nomName.c_str(), 1.0,0,10);
      constrNom.setConstant( true );

      // Make the constraint: 
      RooGaussian gauss( constrName.c_str(), constrName.c_str(),
			 constrNom, gamma, constrSigma );
      
      proto->import( gauss, RecycleConflictNodes() );
      
      // Give reasonable starting point for pre-fit errors by setting it to the absolute sigma
      // Mostly useful for pre-fit plotting.
      gamma.setError(sigmaRel);
    } else if( type == Constraint::Poisson ) {
    
      Double_t tau = 1/sigmaRel/sigmaRel; // this is correct Poisson equivalent to a Gaussian with mean 1 and stdev sigma

      // Make nominal "observed" value
      RooRealVar constrNom(nomName.c_str(), nomName.c_str(), tau);
      constrNom.setMin(0);
      constrNom.setConstant( true );
    
      // Make the scaling term
      std::string scalingName = string(gamma.GetName()) + "_tau";
      RooConstVar poissonScaling( scalingName.c_str(), scalingName.c_str(), tau);
    
      // Make mean for scaled Poisson
      RooProduct constrMean( poisMeanName.c_str(), poisMeanName.c_str(), RooArgSet(gamma, poissonScaling) );
      //proto->import( constrSigma, RecycleConflictNodes() );
      //proto->import( constrSigma );

      // Type 2 : RooPoisson
      RooPoisson pois(constrName.c_str(), constrName.c_str(), constrNom, constrMean);
      pois.setNoRounding(true);
      proto->import( pois, RecycleConflictNodes() );
      
      if (std::string(gamma.GetName()).find("gamma_stat") != std::string::npos) {
        // Give reasonable starting point for pre-fit errors.
        // Mostly useful for pre-fit plotting.
        gamma.setError(sigmaRel);
      }

    } else {

      std::cout << "Error: Did not recognize Stat Error constraint term type: "
		<< type << " for : " << paramHist.GetName() << std::endl;
      throw hf_exc();
    }
  
    // If the sigma value is less
    // than a supplied threshold,
    // set the variable to constant
    if( sigmaRel < minSigma ) {
      cxcoutW(HistFactory) << "Warning:  Bin " << i << " = " << sigmaRel
		<< " and is < " << minSigma
		<< ". Setting: " << gamma.GetName() << " to constant"
		<< std::endl;
      gamma.setConstant(kTRUE);
    }
  
    constraintTermNames.push_back( constrName );    
    ConstraintTerms.add( *proto->pdf(constrName.c_str()) );

    // Add the "observed" value to the 
    // list of global observables:
    RooArgSet* globalSet = const_cast<RooArgSet*>(proto->set("globalObservables"));
  
    RooRealVar* nomVarInWorkspace = proto->var(nomName.c_str());
    if( ! globalSet->contains(*nomVarInWorkspace) ) {
      globalSet->add( *nomVarInWorkspace );	
    }
  
  } // end loop over parameters
  
  return ConstraintTerms;
  
}

} // namespace RooStats
} // namespace HistFactory

