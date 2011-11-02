// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
</p>
END_HTML
*/
//


#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif

// Roofit/Roostat include
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooAddition.h"
#include "RooProduct.h"
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooRandom.h"
#include "RooCategory.h"
#include "RooSimultaneous.h"
#include "RooMultiVarGaussian.h"
#include "RooNumIntConfig.h"
#include "RooMinuit.h"
#include "RooNLLVar.h"
#include "RooProfileLL.h"
#include "RooFitResult.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"
#include "RooHistPdf.h"
#include "RooRealSumPdf.h"
#include "RooProduct.h"
#include "RooWorkspace.h"
#include "RooCustomizer.h"
#include "RooPlot.h"
#include "RooMsgService.h"
#include "RooStats/RooStatsUtils.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/HistFactory/PiecewiseInterpolation.h"

#include "TH2F.h"
#include "TH3F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TLine.h"
#include "TTree.h"
#include "TMarker.h"
#include "TStopwatch.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TVectorD.h"
#include "TMatrixDSym.h"

// specific to this package
//#include "RooStats/HistFactory/Helper.h"
#include "Helper.h"
#include "RooStats/HistFactory/LinInterpVar.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"

#define VERBOSE

#define alpha_Low "-5"
#define alpha_High "5"
#define NoHistConst_Low "0"
#define NoHistConst_High "2000"

// use this order for safety on library loading
using namespace RooFit ;
using namespace RooStats ;
using namespace std ;
//using namespace RooMsgService ;

ClassImp(RooStats::HistFactory::HistoToWorkspaceFactoryFast)

namespace RooStats{
namespace HistFactory{

  HistoToWorkspaceFactoryFast::HistoToWorkspaceFactoryFast() : 
       fNomLumi(0), fLumiError(0),   
       fLowBin(0), fHighBin(0),  
       fOut_f(0), pFile(0)           
  {}

  HistoToWorkspaceFactoryFast::~HistoToWorkspaceFactoryFast(){
    if (pFile!=0) { fclose(pFile); }
  }

  HistoToWorkspaceFactoryFast::HistoToWorkspaceFactoryFast(string filePrefix, string row, vector<string> syst, double nomL, double lumiE, int low, int high, TFile* f):
      fFileNamePrefix(filePrefix),
      fRowTitle(row),
      fSystToFix(syst),
      fNomLumi(nomL),
      fLumiError(lumiE),
      fLowBin(low),
      fHighBin(high),
      fOut_f(f) {

    fResultsPrefixStr<< "_" << fRowTitle;
    while(fRowTitle.find("\\ ")!=string::npos){
      int pos=fRowTitle.find("\\ ");
      fRowTitle.replace(pos, 1, "");
    }
    pFile = fopen ((filePrefix+"_results.table").c_str(),"a"); 
    //RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;

  }

  string HistoToWorkspaceFactoryFast::FilePrefixStr(string prefix){

    stringstream ss;
    ss << prefix << "_" << fNomLumi<< "_" << fLumiError<< "_" << fLowBin<< "_" << fHighBin<< "_"<<fRowTitle;

    return ss.str();
  }

  void HistoToWorkspaceFactoryFast::ProcessExpectedHisto(TH1* hist,RooWorkspace* proto, string prefix, string
						       productPrefix, string systTerm, double /*low*/ , double
						       /*high*/, int /*lowBin*/, int /*highBin*/ ){
    if(hist) {
      cout << "processing hist " << hist->GetName() << endl;
    } else {
      cout << "hist is empty" << endl;
      R__ASSERT(hist != 0); 
      return;                  
    }

    /// require dimension >=1 or <=3
    if (fObsNameVec.empty() && !fObsName.empty()) { fObsNameVec.push_back(fObsName); }    
    R__ASSERT( fObsNameVec.size()>=1 && fObsNameVec.size()<=3 );

    /// determine histogram dimensionality 
    unsigned int histndim(1);
    std::string classname = hist->ClassName();
    if      (classname.find("TH1")==0) { histndim=1; }
    else if (classname.find("TH2")==0) { histndim=2; }
    else if (classname.find("TH3")==0) { histndim=3; }
    R__ASSERT( histndim==fObsNameVec.size() );

    /// create roorealvar observables
    RooArgList observables;
    std::vector<std::string>::iterator itr = fObsNameVec.begin();
    for (int idx=0; itr!=fObsNameVec.end(); ++itr, ++idx ) {
      if ( !proto->var(itr->c_str()) ) {
	TAxis* axis(0);
	if (idx==0) { axis = hist->GetXaxis(); }
	if (idx==1) { axis = hist->GetYaxis(); }
	if (idx==2) { axis = hist->GetZaxis(); }
	Int_t nbins = axis->GetNbins();	
	Double_t xmin = axis->GetXmin();
	Double_t xmax = axis->GetXmax(); 	
	// create observable
	proto->factory(Form("%s[%f,%f]",itr->c_str(),xmin,xmax));
	proto->var(itr->c_str())->setBins(nbins);
      }
      observables.add( *proto->var(itr->c_str()) );
    }

    RooDataHist* histDHist = new RooDataHist((prefix+"nominalDHist").c_str(),"",observables,hist);
    RooHistFunc* histFunc = new RooHistFunc((prefix+"_nominal").c_str(),"",observables,*histDHist,0) ;
    //RooHistPdf* histPdf = new RooHistPdf((prefix+"_nominalpdf").c_str(),"",observables,*histDHist,0);

    proto->import(*histFunc);
    //proto->import(*histPdf);

    /// now create the product of the overall efficiency times the sigma(params) for this estimate
    proto->factory(("prod:"+productPrefix+"("+prefix+"_nominal,"+systTerm+")").c_str() );    
    //    proto->Print();
  }

  void HistoToWorkspaceFactoryFast::AddMultiVarGaussConstraint(RooWorkspace* proto, string prefix,int lowBin, int highBin, vector<string>& constraintTermNames){
    // these are the nominal predictions: eg. the mean of some space of variations
    // later fill these in a loop over histogram bins

    TVectorD mean(highBin); //-lowBin); // MB: fix range
    cout << "a" << endl;
    for(Int_t i=lowBin; i<highBin; ++i){
      std::stringstream str;
      str<<"_"<<i;
      RooRealVar* temp = proto->var((prefix+str.str()).c_str());
      mean(i) = temp->getVal();
    }

    TMatrixDSym Cov(highBin-lowBin);
    for(int i=lowBin; i<highBin; ++i){
      for(int j=0; j<highBin-lowBin; ++j){
        if(i==j) { Cov(i,j) = sqrt(mean(i)); } // MB : this doesn't make sense to me if lowBin!=0 (?)
	else { Cov(i,j) = 0; } 
      }
    }
    // can't make MultiVarGaussian with factory yet, do it by hand
    RooArgList floating( *(proto->set(prefix.c_str() ) ) );
    RooMultiVarGaussian constraint((prefix+"Constraint").c_str(),"",
             floating, mean, Cov);
             
    proto->import(constraint);

    constraintTermNames.push_back(constraint.GetName());
  }


  void HistoToWorkspaceFactoryFast::LinInterpWithConstraint(RooWorkspace* proto, TH1* nominal, vector<TH1*> lowHist, vector<TH1*> highHist, 
             vector<string> sourceName, string prefix, string productPrefix, string systTerm, 
                                                            int /*lowBin*/, int /*highBin */, vector<string>& constraintTermNames){
    // these are the nominal predictions: eg. the mean of some space of variations
    // later fill these in a loop over histogram bins

    // require dimension >=1 or <=3
    if (fObsNameVec.empty() && !fObsName.empty()) { fObsNameVec.push_back(fObsName); }    
    R__ASSERT( fObsNameVec.size()>=1 && fObsNameVec.size()<=3 );

    // determine histogram dimensionality 
    unsigned int histndim(1);
    std::string classname = nominal->ClassName();
    if      (classname.find("TH1")==0) { histndim=1; }
    else if (classname.find("TH2")==0) { histndim=2; }
    else if (classname.find("TH3")==0) { histndim=3; }
    R__ASSERT( histndim==fObsNameVec.size() );
    //    cout <<"In LinInterpWithConstriants and histndim = " << histndim <<endl;

    // create roorealvar observables
    RooArgList observables;
    std::vector<std::string>::iterator itr = fObsNameVec.begin();
    for (int idx=0; itr!=fObsNameVec.end(); ++itr, ++idx ) {
      if ( !proto->var(itr->c_str()) ) {
	TAxis* axis(0);
	if (idx==0) { axis = nominal->GetXaxis(); }
	if (idx==1) { axis = nominal->GetYaxis(); }
	if (idx==2) { axis = nominal->GetZaxis(); }
	Int_t nbins = axis->GetNbins();	
	Double_t xmin = axis->GetXmin();
	Double_t xmax = axis->GetXmax(); 	
	// create observable
	proto->factory(Form("%s[%f,%f]",itr->c_str(),xmin,xmax));
	proto->var(itr->c_str())->setBins(nbins);
      }
      observables.add( *proto->var(itr->c_str()) );
    }

    RooDataHist* nominalDHist = new RooDataHist((prefix+"nominalDHist").c_str(),"",observables,nominal);
    RooHistFunc* nominalFunc = new RooHistFunc((prefix+"nominal").c_str(),"",observables,*nominalDHist,0) ;

    // make list of abstract parameters that interpolate in space of variations
    RooArgList params( ("alpha_Hist") );
    // range is set using defined macro (see top of the page)
    string range=string("[")+alpha_Low+","+alpha_High+"]";
    for(unsigned int j=0; j<lowHist.size(); ++j){
      std::stringstream str;
      str<<"_"<<j;

      RooRealVar* temp = (RooRealVar*) proto->var(("alpha_"+sourceName.at(j)).c_str());
      if(!temp){
        temp = (RooRealVar*) proto->factory(("alpha_"+sourceName.at(j)+range).c_str());

        // now add a constraint term for these parameters
        string command=("Gaussian::alpha_"+sourceName.at(j)+"Constraint(alpha_"+sourceName.at(j)+",nom_alpha_"+sourceName.at(j)+"[0.,-10,10],1.)");
        cout << command << endl;
        constraintTermNames.push_back(  proto->factory( command.c_str() )->GetName() );
	proto->var(("nom_alpha_"+sourceName.at(j)).c_str())->setConstant();
	const_cast<RooArgSet*>(proto->set("globalObservables"))->add(*proto->var(("nom_alpha_"+sourceName.at(j)).c_str()));
      } 
      params.add(* temp );
    }

    // now make function that linearly interpolates expectation between variations
    // get low/high variations to interpolate between
    vector<double> low, high;
    RooArgSet lowSet, highSet;
    for(unsigned int j=0; j<lowHist.size(); ++j){
      std::stringstream str;
      str<<"_"<<j;
      lowHist.at(j);
      highHist.at(j);
      RooDataHist* lowDHist = new RooDataHist((prefix+str.str()+"lowDHist").c_str(),"",observables,lowHist.at(j));
      RooDataHist* highDHist = new RooDataHist((prefix+str.str()+"highDHist").c_str(),"",observables,highHist.at(j));
      RooHistFunc* lowFunc = new RooHistFunc((prefix+str.str()+"low").c_str(),"",observables,*lowDHist,0) ;
      RooHistFunc* highFunc = new RooHistFunc((prefix+str.str()+"high").c_str(),"",observables,*highDHist,0) ;
      lowSet.add(*lowFunc);
      highSet.add(*highFunc);
    }
    
    // this is sigma(params), a piece-wise linear interpolation
    PiecewiseInterpolation interp(prefix.c_str(),"",*nominalFunc,lowSet,highSet,params);
    interp.setPositiveDefinite();
    interp.setAllInterpCodes(0); // MB : change default to 1? = piece-wise log interpolation from pice-wise linear (=0)
    // KC: interpo codes 1 etc. don't have proper analytic integral.
    
    //    cout << "check: " << interp.getVal() << endl;
    proto->import(interp); // individual params have already been imported in first loop of this function
    
    // now create the product of the overall efficiency times the sigma(params) for this estimate
    proto->factory(("prod:"+productPrefix+"("+prefix+","+systTerm+")").c_str() );    
    //    proto->Print();

  }

  string HistoToWorkspaceFactoryFast::AddNormFactor(RooWorkspace * proto, string & channel, string & sigmaEpsilon, EstimateSummary & es, bool doRatio){
    string overallNorm_times_sigmaEpsilon ;
    string prodNames;
    vector<EstimateSummary::NormFactor> norm=es.normFactor;
    if(norm.size()){
      for(vector<EstimateSummary::NormFactor>::iterator itr=norm.begin(); itr!=norm.end(); ++itr){
        cout << "making normFactor: " << itr->name << endl;
        // remove "doRatio" and name can be changed when ws gets imported to the combined model.
        std::stringstream range;
        range<<"["<<itr->val<<","<<itr->low<<","<<itr->high<<"]";

        string varname;
        if(!prodNames.empty()) prodNames+=",";
        if(doRatio) {
          varname=itr->name+"_"+channel;
        }
        else {
          varname=itr->name;
        }
        proto->factory((varname+range.str()).c_str());
        prodNames+=varname;
      }
      overallNorm_times_sigmaEpsilon = es.name+"_"+channel+"_overallNorm_x_sigma_epsilon";
      proto->factory(("prod::"+overallNorm_times_sigmaEpsilon+"("+prodNames+","+sigmaEpsilon+")").c_str());
    }

    if(!overallNorm_times_sigmaEpsilon.empty())
      return overallNorm_times_sigmaEpsilon;
    else
      return sigmaEpsilon;
  }        


  void HistoToWorkspaceFactoryFast::AddEfficiencyTerms(RooWorkspace* proto, string prefix, string interpName,
        map<string,pair<double,double> > systMap, 
        vector<string>& constraintTermNames, vector<string>& totSystTermNames){
    // add variables for all the relative overall uncertainties we expect
    
    // range is set using defined macro (see top of the page)
    string range=string("[0,")+alpha_Low+","+alpha_High+"]";
    //string range="[0,-1,1]";
    totSystTermNames.push_back(prefix);
    //bool first=true;
    RooArgSet params(prefix.c_str());
    vector<double> lowVec, highVec;
    for(map<string,pair<double,double> >::iterator it=systMap.begin(); it!=systMap.end(); ++it){
      // add efficiency term
      RooRealVar* temp = (RooRealVar*) proto->var((prefix+ it->first).c_str());
      if(!temp){
        temp = (RooRealVar*) proto->factory((prefix+ it->first +range).c_str());

        string command=("Gaussian::"+prefix+it->first+"Constraint("+prefix+it->first+",nom_"+prefix+it->first+"[0.,-10,10],1.)");
        cout << command << endl;
        constraintTermNames.push_back(  proto->factory( command.c_str() )->GetName() );
	proto->var(("nom_"+prefix+it->first).c_str())->setConstant();
	const_cast<RooArgSet*>(proto->set("globalObservables"))->add(*proto->var(("nom_"+prefix+it->first).c_str()));	
      } 
      params.add(*temp);

      // add constraint in terms of bifrucated gauss with low/high as sigmas
      std::stringstream lowhigh;
      double low = it->second.first; 
      double high = it->second.second;
      lowVec.push_back(low);
      highVec.push_back(high);
      
    }
    if(systMap.size()>0){
      // this is epsilon(alpha_j), a piece-wise linear interpolation
      //      LinInterpVar interp( (interpName).c_str(), "", params, 1., lowVec, highVec);
      FlexibleInterpVar interp( (interpName).c_str(), "", params, 1., lowVec, highVec);
      interp.setAllInterpCodes(1); // changing default to piece-wise log interpolation from pice-wise linear
      proto->import(interp); // params have already been imported in first loop of this function
    } else{
      // some strange behavior if params,lowVec,highVec are empty.  
      //cout << "WARNING: No OverallSyst terms" << endl;
      RooConstVar interp( (interpName).c_str(), "", 1.);
      proto->import(interp); // params have already been imported in first loop of this function
    }
    
  }


  void  HistoToWorkspaceFactoryFast::MakeTotalExpected(RooWorkspace* proto, string totName, string /**/, string /**/, 
                                                       int /*lowBin*/, int /*highBin */, vector<string>& syst_x_expectedPrefixNames, 
                                                       vector<string>& normByNames){

    // for ith bin calculate totN_i =  lumi * sum_j expected_j * syst_j 
    string command;
    string coeffList="";
    string shapeList="";
    string prepend="";

    if (fObsNameVec.empty() && !fObsName.empty()) { fObsNameVec.push_back(fObsName); } 

    double binWidth(1.0);
    std::string obsNameVecStr;
    std::vector<std::string>::iterator itr = fObsNameVec.begin();
    for (; itr!=fObsNameVec.end(); ++itr) {
      std::string obsName = *itr;
      binWidth *= proto->var(obsName.c_str())->numBins()/(proto->var(obsName.c_str())->getMax() - proto->var(obsName.c_str())->getMin()) ; // MB: Note: requires fixed bin sizes
      if (obsNameVecStr.size()>0) { obsNameVecStr += "_"; }
      obsNameVecStr += obsName;
    }

    vector<string>::iterator it=syst_x_expectedPrefixNames.begin();
    for(unsigned int j=0; j<syst_x_expectedPrefixNames.size();++j){
      std::stringstream str;
      str<<"_"<<j;
      // repatative, but we need one coeff for each term in the sum
      // maybe can be avoided if we don't use bin width as coefficient
      command=string(Form("binWidth_%s_%d[%f]",obsNameVecStr.c_str(),j,binWidth));     
      proto->factory(command.c_str());
      proto->var(Form("binWidth_%s_%d",obsNameVecStr.c_str(),j))->setConstant();
      coeffList+=prepend+"binWidth_"+obsNameVecStr+str.str();

      command="prod::L_x_"+syst_x_expectedPrefixNames.at(j)+"("+normByNames.at(j)+","+syst_x_expectedPrefixNames.at(j)+")";
      proto->factory(command.c_str());
      shapeList+=prepend+"L_x_"+syst_x_expectedPrefixNames.at(j);
      prepend=",";
    }    

    proto->defineSet("coefList",coeffList.c_str());
    proto->defineSet("shapeList",shapeList.c_str());
    //    proto->factory(command.c_str());
    RooRealSumPdf tot(totName.c_str(),totName.c_str(),*proto->set("shapeList"),*proto->set("coefList"),kTRUE);
    proto->import(tot);
    //    proto->Print();
    
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

     proto->import(*data);

  }

  void HistoToWorkspaceFactoryFast::Customize(RooWorkspace* proto, const char* pdfNameChar, map<string,string> renameMap) {
    cout << "in customizations" << endl;
    string pdfName(pdfNameChar);
    map<string,string>::iterator it;
    string edit="EDIT::customized("+pdfName+",";
    string preceed="";
    for(it=renameMap.begin(); it!=renameMap.end(); ++it) {
      cout << it->first + "=" + it->second << endl;
      edit+=preceed + it->first + "=" + it->second;
      preceed=",";
    }
    edit+=")";
    cout << edit<< endl;
    proto->factory( edit.c_str() );
  }

  //_____________________________________________________________
  void HistoToWorkspaceFactoryFast::EditSyst(RooWorkspace* proto, const char* pdfNameChar, 
					     map<string,double> gammaSyst, map<string,double> uniformSyst, map<string,double> logNormSyst, map<string,double> noSyst) {
    string pdfName(pdfNameChar);

    //cout << "HistoToWorkspaceFactoryFast::EditSyst() : gamma = " << gammaSyst.size() << ", uniform = " << uniformSyst.size() << ", noconst = " << noSyst.size() << endl;

    ModelConfig * combined_config = (ModelConfig *) proto->obj("ModelConfig");
    //    const RooArgSet * constrainedParams=combined_config->GetNuisanceParameters();
    //    RooArgSet temp(*constrainedParams);
    string edit="EDIT::newSimPdf("+pdfName+",";
    string editList;
    string lastPdf=pdfName;
    string preceed="";
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
      if(proto->var(Form("alpha_%s",it->first.c_str()))->isConstant())
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(true);
      else
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(false);
      // set alpha const status to true
      //      proto->var(Form("alpha_%s",it->first.c_str()))->setConstant(true);

      // replace alphas with alphaOfBeta and replace constraints
      //cout <<         "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint" << endl;
      editList+=preceed + "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint";
      preceed=",";
      //      cout <<         "alpha_"+it->first+"=alphaOfBeta_"+ it->first << endl;
      editList+=preceed + "alpha_"+it->first+"=alphaOfBeta_"+ it->first;

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
	preceed="";
	cout << "Going to issue this edit command\n" << edit<< endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne)
	  cout << "\n\n ---------------------\n WARNING: failed to make EDIT\n\n" << endl;
	
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
      editList+=preceed + "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint";
      preceed=",";
      cout <<         "alpha_"+it->first+"=alphaOfBeta_"+ it->first << endl;
      editList+=preceed + "alpha_"+it->first+"=alphaOfBeta_"+ it->first;

      if( proto->pdf(("alpha_"+it->first+"Constraint").c_str()) && proto->var(("alpha_"+it->first).c_str()) )
	cout << " checked they are there" << proto->pdf(("alpha_"+it->first+"Constraint").c_str()) << " " << proto->var(("alpha_"+it->first).c_str()) << endl;
      else
	cout << "NOT THERE" << endl;

      // EDIT seems to die if the list of edits is too long.  So chunck them up.
      if(numReplacements%10 == 0 && numReplacements+nskipped!=gammaSyst.size()){
	edit="EDIT::"+lastPdf+"_("+lastPdf+","+editList+")";
	lastPdf+="_"; // append an underscore for the edit
	editList=""; // reset edit list
	preceed="";
	cout << edit<< endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne)
	  cout << "\n\n ---------------------\n WARNING: failed to make EDIT\n\n" << endl;
	
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

      // this is the LogNormal
      proto->factory(Form("beta_%s[1,0,10]",it->first.c_str()));
      proto->factory(Form("kappa_%s[%f]",it->first.c_str(),kappa));
      proto->factory(Form("Lognormal::beta_%sConstraint(beta_%s,one[1],kappa_%s)",
			  it->first.c_str(),
			  it->first.c_str(),
			  it->first.c_str())) ;
      proto->factory(Form("PolyVar::alphaOfBeta_%s(beta_%s,{%f,%f})",it->first.c_str(),it->first.c_str(),-1./scale,1./scale));
      //      proto->factory(Form("PolyVar::alphaOfBeta_%s(beta_%s,{%f,%f})",it->first.c_str(),it->first.c_str(),-1.,1./scale));
      
      // set beta const status to be same as alpha
      if(proto->var(Form("alpha_%s",it->first.c_str()))->isConstant())
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(true);
      else
	proto->var(Form("beta_%s",it->first.c_str()))->setConstant(false);
      // set alpha const status to true
      //      proto->var(Form("alpha_%s",it->first.c_str()))->setConstant(true);

      // replace alphas with alphaOfBeta and replace constraints
      cout <<         "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint" << endl;
      editList+=preceed + "alpha_"+it->first+"Constraint=beta_" + it->first+ "Constraint";
      preceed=",";
      cout <<         "alpha_"+it->first+"=alphaOfBeta_"+ it->first << endl;
      editList+=preceed + "alpha_"+it->first+"=alphaOfBeta_"+ it->first;

      if( proto->pdf(("alpha_"+it->first+"Constraint").c_str()) && proto->var(("alpha_"+it->first).c_str()) )
	cout << " checked they are there" << proto->pdf(("alpha_"+it->first+"Constraint").c_str()) << " " << proto->var(("alpha_"+it->first).c_str()) << endl;
      else
	cout << "NOT THERE" << endl;

      // EDIT seems to die if the list of edits is too long.  So chunck them up.
      if(numReplacements%10 == 0 && numReplacements+nskipped!=gammaSyst.size()){
	edit="EDIT::"+lastPdf+"_("+lastPdf+","+editList+")";
	lastPdf+="_"; // append an underscore for the edit
	editList=""; // reset edit list
	preceed="";
	cout << edit<< endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne)
	  cout << "\n\n ---------------------\n WARNING: failed to make EDIT\n\n" << endl;
	
      }
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
      editList+=preceed + "alpha_"+it->first+"Constraint=one";
      preceed=",";

      // EDIT seems to die if the list of edits is too long.  So chunck them up.
      if(numReplacements%10 == 0 && numReplacements+nskipped!=gammaSyst.size()){
	edit="EDIT::"+lastPdf+"_("+lastPdf+","+editList+")";
	lastPdf+="_"; // append an underscore for the edit
	editList=""; // reset edit list
	preceed="";
	cout << edit << endl;
	proto->factory( edit.c_str() );
	RooAbsPdf* newOne = proto->pdf(lastPdf.c_str());
	if(!newOne) { cout << "\n\n ---------------------\n WARNING: failed to make EDIT\n\n" << endl; }
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
      cout << "\n\n ---------------------\n WARNING: failed to make EDIT\n\n" << endl;
    }
  }

  void HistoToWorkspaceFactoryFast::PrintCovarianceMatrix(RooFitResult* result, RooArgSet* params, string filename){
    //    FILE * pFile;
    pFile = fopen ((filename).c_str(),"w"); 


    TIter iti = params->createIterator();
    TIter itj = params->createIterator();
    RooRealVar *myargi, *myargj; 
    fprintf(pFile," ") ;
    while ((myargi = (RooRealVar *)iti.Next())) { 
      if(myargi->isConstant()) continue;
      fprintf(pFile," & %s",  myargi->GetName());
    }
    fprintf(pFile,"\\\\ \\hline \n" );
    iti.Reset();
    while ((myargi = (RooRealVar *)iti.Next())) { 
      if(myargi->isConstant()) continue;
      fprintf(pFile,"%s", myargi->GetName());
      itj.Reset();
      while ((myargj = (RooRealVar *)itj.Next())) { 
        if(myargj->isConstant()) continue;
        cout << myargi->GetName() << "," << myargj->GetName();
        fprintf(pFile, " & %.2f", result->correlation(*myargi, *myargj));
      }
      cout << endl;
      fprintf(pFile, " \\\\\n");
    }
    fclose(pFile);
    
  }


  ///////////////////////////////////////////////
  RooWorkspace* HistoToWorkspaceFactoryFast::MakeSingleChannelModel(vector<EstimateSummary> summary, vector<string> systToFix, bool doRatio)
  {
    // to time the macro
    TStopwatch t;
    t.Start();
    string channel=summary[0].channel;




    /// MB: reset observable names for each new channel.
    fObsNameVec.clear();

    /// MB: label observables x,y,z, depending on histogram dimensionality
    if (fObsNameVec.empty()) { GuessObsNameVec( summary.at(0).nominal ); }

    for ( unsigned int idx=0; idx<fObsNameVec.size(); ++idx ) {
      fObsNameVec[idx] = "obs_" + fObsNameVec[idx] + "_" + summary[0].channel ;
    }

    if (fObsNameVec.empty()) {
      //    fObsName.c_str()=Form("%s_%s",summary.at(0).nominal->GetXaxis()->GetName()],summary[0].channel.c_str()); // set name ov observable
      fObsName= "obs_"+summary[0].channel; // set name ov observable
      fObsNameVec.push_back( fObsName );
    }

    R__ASSERT( fObsNameVec.size()>=1 && fObsNameVec.size()<=3 );

    cout << "\n\n-------------------\nStarting to process " << channel << " channel with " << fObsNameVec.size() << " observables" << endl;

    //
    // our main workspace that we are using to construct the model
    //
    RooWorkspace* proto = new RooWorkspace(summary[0].channel.c_str(),(summary[0].channel+" workspace").c_str());
    ModelConfig * proto_config = new ModelConfig("ModelConfig", proto);
    proto_config->SetWorkspace(*proto);

    // preprocess functions
    vector<string>::iterator funcIter = fPreprocessFunctions.begin();
    for(;funcIter!= fPreprocessFunctions.end(); ++funcIter){
      cout <<"will preprocess this line: " << *funcIter <<endl;
      proto->factory(funcIter->c_str());
      proto->Print();
    }


    RooArgSet likelihoodTerms("likelihoodTerms"), constraintTerms("constraintTerms");
    vector<string> likelihoodTermNames, constraintTermNames, totSystTermNames,syst_x_expectedPrefixNames, normalizationNames;

    string prefix, range;


    /////////////////////////////
    // shared parameters
    // this is ratio of lumi to nominal lumi.  We will include relative uncertainty in model
    std::stringstream lumiStr;
    // lumi range
    lumiStr<<"["<<fNomLumi<<",0,"<<10.*fNomLumi<<"]";
    proto->factory(("Lumi"+lumiStr.str()).c_str());
    cout << "lumi str = " << lumiStr.str() << endl;
    
    std::stringstream lumiErrorStr;
    lumiErrorStr << "nominalLumi["<<fNomLumi << ",0,"<<fNomLumi+10*fLumiError<<"]," << fLumiError ;
    proto->factory(("Gaussian::lumiConstraint(Lumi,"+lumiErrorStr.str()+")").c_str());
    proto->var("nominalLumi")->setConstant();
    proto->defineSet("globalObservables","nominalLumi");
    //likelihoodTermNames.push_back("lumiConstraint");
    constraintTermNames.push_back("lumiConstraint");
    cout << "lumi Error str = " << lumiErrorStr.str() << endl;
    
    //proto->factory((string("SigXsecOverSM[1.,0.5,1..8]").c_str()));
    ///////////////////////////////////
    // loop through estimates, add expectation, floating bin predictions, 
    // and terms that constrain floating to expectation via uncertainties
    vector<EstimateSummary>::iterator it = summary.begin();
    for(; it!=summary.end(); ++it){
      if(it->name=="Data") continue;

      string overallSystName = it->name+"_"+it->channel+"_epsilon"; 
      string systSourcePrefix = "alpha_";
      AddEfficiencyTerms(proto,systSourcePrefix, overallSystName,
			 it->overallSyst, constraintTermNames /*likelihoodTermNames*/, totSystTermNames);    

      overallSystName = AddNormFactor(proto, channel, overallSystName, *it, doRatio); 

      // MB : HACK no option to have both non-hist variations and hist variations ?
      // get histogram
      TH1* nominal = it->nominal;
      if(it->lowHists.size() == 0){
        cout << it->name+"_"+it->channel+" has no variation histograms " <<endl;
        string expPrefix=it->name+"_"+it->channel;//+"_expN";
        string syst_x_expectedPrefix=it->name+"_"+it->channel+"_overallSyst_x_Exp";
        ProcessExpectedHisto(nominal,proto,expPrefix,syst_x_expectedPrefix,overallSystName,atoi(NoHistConst_Low),atoi(NoHistConst_High),fLowBin,fHighBin);
        syst_x_expectedPrefixNames.push_back(syst_x_expectedPrefix);
      } else if(it->lowHists.size() != it->highHists.size()){
        cout << "problem in "+it->name+"_"+it->channel 
	     << " number of low & high variation histograms don't match" << endl;
        return 0;
      } else {
        string constraintPrefix = it->name+"_"+it->channel+"_Hist_alpha"; // name of source for variation
        string syst_x_expectedPrefix = it->name+"_"+it->channel+"_overallSyst_x_HistSyst";
        LinInterpWithConstraint(proto, nominal, it->lowHists, it->highHists, it->systSourceForHist,
              constraintPrefix, syst_x_expectedPrefix, overallSystName, 
				fLowBin, fHighBin, constraintTermNames /*likelihoodTermNames*/);
        syst_x_expectedPrefixNames.push_back(syst_x_expectedPrefix);
      }

      if(it->normName=="")
        normalizationNames.push_back( "Lumi" );
      else
        normalizationNames.push_back( it->normName);
    }
    //    proto->Print();

    ///////////////////////////////////
    // for ith bin calculate totN_i =  lumi * sum_j expected_j * syst_j 
    MakeTotalExpected(proto,channel+"_model",channel,"Lumi",fLowBin,fHighBin, 
          syst_x_expectedPrefixNames, normalizationNames);
    likelihoodTermNames.push_back(channel+"_model");

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
	  auxMeas = proto->var(Form("nom_%s",temp->GetName()));
	}

	if(auxMeas){
	  const_cast<RooArgSet*>(proto->set("globalObservables"))->remove(*auxMeas);
	} else{
	  cout << "could not corresponding auxiliary measurement  " << Form("nom_%s",temp->GetName()) << endl;
	}
      } else {
	cout << "could not find variable " << systToFix.at(i) << " could not set it to constant" << endl;
      }
    }

    //////////////////////////////////////
    // final proto model
    for(unsigned int i=0; i<constraintTermNames.size(); ++i){
      constraintTerms.add(* (proto->arg(constraintTermNames[i].c_str())) );
    }
    for(unsigned int i=0; i<likelihoodTermNames.size(); ++i){
      likelihoodTerms.add(* (proto->arg(likelihoodTermNames[i].c_str())) );
    }
    proto->defineSet("constraintTerms",constraintTerms);
    proto->defineSet("likelihoodTerms",likelihoodTerms);
    //  proto->Print();

    // list of observables
    RooArgList observables;
    std::string observablesStr;
    std::vector<std::string>::iterator itr = fObsNameVec.begin();
    for (int idx=0; itr!=fObsNameVec.end(); ++itr, ++idx ) {
      observables.add( *proto->var(itr->c_str()) );
      if (!observablesStr.empty()) { observablesStr += ","; }
      observablesStr += *itr;
    }
    proto->defineSet("observablesSet",Form("%s",observablesStr.c_str()));

    cout <<"-----------------------------------------"<<endl;
    cout <<"import model into workspace" << endl;

    RooProdPdf* model = new RooProdPdf(("model_"+channel).c_str(),    // MB : have changed this into conditional pdf. Much faster for toys!
               "product of Poissons accross bins for a single channel",
	       constraintTerms, Conditional(likelihoodTerms,observables));  //likelihoodTerms);
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
    proto->factory(Form("%s[0,-1e10,1e10]",weightName));
    proto->defineSet("obsAndWeight",Form("%s,%s",weightName,observablesStr.c_str()));
    RooAbsData* data = model->generateBinned(observables,ExpectedData());

    /// Asimov dataset
    RooDataSet* asimovDataUnbinned = new RooDataSet("asimovData","",*proto->set("obsAndWeight"),weightName);
    /*
    double binWidthW(1.0);
    itr = fObsNameVec.begin();
    for (; itr!=fObsNameVec.end(); ++itr) {
      std::string obsName = *itr;
      binWidthW *= proto->var(obsName.c_str())->numBins()/(proto->var(obsName.c_str())->getMax() - proto->var(obsName.c_str())->getMin()) ; 
    }
    */
    for(int i=0; i<data->numEntries(); ++i){
      data->get(i)->Print("v");
      //cout << "GREPME : " << i << " " << data->weight() <<endl;
      asimovDataUnbinned->add( *data->get(i), data->weight() );
    }
    proto->import(*asimovDataUnbinned);


    if (summary.at(0).name=="Data") { 

      // THis works and is natural, but the memory size of the simultaneous dataset grows exponentially with channels
      RooDataSet* obsDataUnbinned = new RooDataSet("obsData","",*proto->set("obsAndWeight"),weightName);

      TH1* mnominal = summary.at(0).nominal;
      TAxis* ax = mnominal->GetXaxis(); 
      TAxis* ay = mnominal->GetYaxis(); 
      TAxis* az = mnominal->GetZaxis(); 	

      for (int i=1; i<=ax->GetNbins(); ++i) { // 1 or more dimension
	Double_t xval = ax->GetBinCenter(i);
	proto->var( fObsNameVec[0].c_str() )->setVal( xval );
	if        (fObsNameVec.size()==1) {
	  Double_t fval = mnominal->GetBinContent(i);
	  obsDataUnbinned->add( *proto->set("obsAndWeight"), fval );
	} else { // 2 or more dimensions
	  for (int j=1; j<=ay->GetNbins(); ++j) {
	    Double_t yval = ay->GetBinCenter(j);
	    proto->var( fObsNameVec[1].c_str() )->setVal( yval );
	    if (fObsNameVec.size()==2) { 
	      Double_t fval = mnominal->GetBinContent(i,j);
	      obsDataUnbinned->add( *proto->set("obsAndWeight"), fval );
	    } else { // 3 dimensions 
	      for (int k=1; k<=az->GetNbins(); ++k) {
		Double_t zval = az->GetBinCenter(k);
		proto->var( fObsNameVec[2].c_str() )->setVal( zval );
		Double_t fval = mnominal->GetBinContent(i,j,k);
		obsDataUnbinned->add( *proto->set("obsAndWeight"), fval );
	      }
	    }
	  }
	}
      }
      
      proto->import(*obsDataUnbinned);
    }

    proto->Print();
    return proto;
  }


  void HistoToWorkspaceFactoryFast::GuessObsNameVec(TH1* hist)
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


  RooWorkspace* HistoToWorkspaceFactoryFast::MakeCombinedModel(vector<string> ch_names, vector<RooWorkspace*> chs)
  {

    //
    /// These things were used for debugging. Maybe useful in the future
    //

    map<string, RooAbsPdf*> pdfMap;
    vector<RooAbsPdf*> models;
    stringstream ss;

    RooArgList obsList;
    for(unsigned int i = 0; i< ch_names.size(); ++i){
      ModelConfig * config = (ModelConfig *) chs[i]->obj("ModelConfig");
      obsList.add(*config->GetObservables());
    }
    cout <<"full list of observables:"<<endl;
    obsList.Print();

    RooArgSet globalObs;
    for(unsigned int i = 0; i< ch_names.size(); ++i){
      string channel_name=ch_names[i];

      if (ss.str().empty()) ss << channel_name ;
      else ss << ',' << channel_name ;
      RooWorkspace * ch=chs[i];
      
      RooAbsPdf* model = ch->pdf(("model_"+channel_name).c_str());
      if(!model) cout <<"failed to find model for channel"<<endl;
      //      cout << "int = " << model->createIntegral(*obsN)->getVal() << endl;;
      models.push_back(model);
      globalObs.add(*ch->set("globalObservables"));

      //      constrainedParams->add( * ch->set("constrainedParams") );
      pdfMap[channel_name]=model;
    }
    //constrainedParams->Print();

    cout << "\n\n------------------\n Entering combination" << endl;
    RooWorkspace* combined = new RooWorkspace("combined");
    //    RooWorkspace* combined = chs[0];
    

    RooCategory* channelCat = (RooCategory*) combined->factory(("channelCat["+ss.str()+"]").c_str());
    RooSimultaneous * simPdf= new RooSimultaneous("simPdf","",pdfMap, *channelCat);
    ModelConfig * combined_config = new ModelConfig("ModelConfig", combined);
    combined_config->SetWorkspace(*combined);
    //    combined_config->SetNuisanceParameters(*constrainedParams);

    combined->import(globalObs);
    combined->defineSet("globalObservables",globalObs);
    combined_config->SetGlobalObservables(*combined->set("globalObservables"));
    

    ////////////////////////////////////////////
    // Make toy simultaneous dataset
    cout <<"-----------------------------------------"<<endl;
    cout << "create toy data for " << ss.str() << endl;
    

    // now with weighted datasets
    // First Asimov
    RooDataSet * simData=NULL;
    combined->factory("weightVar[0,-1e10,1e10]");
    obsList.add(*combined->var("weightVar"));
    for(unsigned int i = 0; i< ch_names.size(); ++i){
      cout << "merging data for channel " << ch_names[i].c_str() << endl;
      RooDataSet * tempData=new RooDataSet(ch_names[i].c_str(),"", obsList, Index(*channelCat),
					   WeightVar("weightVar"),
					  Import(ch_names[i].c_str(),*(RooDataSet*)chs[i]->data("asimovData")));
      if(simData){
	simData->append(*tempData);
      delete tempData;
      }else{
	simData = tempData;
      }
    }
    
    if (simData) combined->import(*simData,Rename("asimovData"));

    // now obs
    if(chs[0]->data("obsData")){
      simData=NULL;
      for(unsigned int i = 0; i< ch_names.size(); ++i){
	cout << "merging data for channel " << ch_names[i].c_str() << endl;
	RooDataSet * tempData=new RooDataSet(ch_names[i].c_str(),"", obsList, Index(*channelCat),
					     WeightVar("weightVar"),
					     Import(ch_names[i].c_str(),*(RooDataSet*)chs[i]->data("obsData")));
	if(simData){
	  simData->append(*tempData);
	  delete tempData;
	}else{
	  simData = tempData;
	}
      }
      
      if (simData) combined->import(*simData,Rename("obsData"));
    }
    


    //    obsList.Print();
    //    combined->import(obsList);
    //    combined->Print();
    obsList.add(*channelCat);
    combined->defineSet("observables",obsList);
    combined_config->SetObservables(*combined->set("observables"));


    combined->Print();

    cout << "\n\n----------------\n Importing combined model" << endl;
    combined->import(*simPdf,RecycleConflictNodes());
    //combined->import(*simPdf, RenameVariable("SigXsecOverSM","SigXsecOverSM_comb"));
    cout << "check pointer " << simPdf << endl;
    //    cout << "check val " << simPdf->getVal() << endl;

    for(unsigned int i=0; i<fSystToFix.size(); ++i){
      // make sure they are fixed
      RooRealVar* temp = combined->var((fSystToFix.at(i)).c_str());
      if(temp) {
        temp->setConstant();
        cout <<"setting " << fSystToFix.at(i) << " constant" << endl;
      } else 
	cout << "could not find variable " << fSystToFix.at(i) << " could not set it to constant" << endl;
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

  ///////////////////////////////////////////////
  void HistoToWorkspaceFactoryFast::FitModel(RooWorkspace * combined, string channel, string /*model_name*/, string data_name, bool /*doParamInspect*/)
  {
    cout << "In Fit Model"<<endl;
    ModelConfig * combined_config = (ModelConfig *) combined->obj("ModelConfig");
    if(!combined_config){
      cout << "no model config " << "ModelConfig" << " exiting" << endl;
      return;
    }
    //    RooDataSet * simData = (RooDataSet *) combined->obj(data_name.c_str());
    RooAbsData* simData = combined->data(data_name.c_str());
    if(!simData){
      cout << "no data " << data_name << " exiting" << endl;
      return;
    }
    //    const RooArgSet * constrainedParams=combined_config->GetNuisanceParameters();
    const RooArgSet * POIs=combined_config->GetParametersOfInterest();
    if(!POIs){
      cout << "no poi " << data_name << " exiting" << endl;
      return;
    }

    //RooAbsPdf* model=combined->pdf(model_name.c_str()); 
    RooAbsPdf* model=combined_config->GetPdf();
    //    RooArgSet* allParams = model->getParameters(*simData);

    ///////////////////////////////////////
    //Do combined fit
    //RooMsgService::instance().setGlobalKillBelow(RooMsgService::INFO) ;
    cout << "\n\n---------------" << endl;
    cout << "---------------- Doing "<< channel << " Fit" << endl;
    cout << "---------------\n\n" << endl;
    //    RooFitResult* result = model->fitTo(*simData, Minos(kTRUE), Save(kTRUE), PrintLevel(1));
    model->fitTo(*simData, Minos(kTRUE), PrintLevel(1));
    //    PrintCovarianceMatrix(result, allParams, "results/"+FilePrefixStr(channel)+"_corrMatrix.table" );

    //
    // assuming there is only on poi
    // 
    RooRealVar* poi = 0; // (RooRealVar*) POIs->first();
    // for results tables
    TIterator* params_itr=POIs->createIterator();
    TObject* params_obj=0;
    while((params_obj=params_itr->Next())){
      poi = (RooRealVar*) params_obj;
      cout << "printing results for " << poi->GetName() << " at " << poi->getVal()<< " high " << poi->getErrorLo() << " low " << poi->getErrorHi()<<endl;
    }
    fprintf(pFile, " %.4f / %.4f  ", poi->getErrorLo(), poi->getErrorHi());

    RooAbsReal* nll = model->createNLL(*simData);
    RooAbsReal* profile = nll->createProfile(*poi);
    RooPlot* frame = poi->frame();
    FormatFrameForLikelihood(frame);
    TCanvas* c1 = new TCanvas( channel.c_str(), "",800,600);
    nll->plotOn(frame, ShiftToZero(), LineColor(kRed), LineStyle(kDashed));
    profile->plotOn(frame);
    frame->SetMinimum(0);
    frame->SetMaximum(2.);
    frame->Draw();
    //    c1->SaveAs( ("results/"+FilePrefixStr(channel)+"_profileLR.eps").c_str() );
    c1->SaveAs( (fFileNamePrefix+"_"+channel+"_"+fRowTitle+"_profileLR.eps").c_str() );

    fOut_f->mkdir(channel.c_str())->mkdir("Summary")->cd();

    // an example of calculating profile for a nuisance parameter not poi
    /*
    RooRealVar* alpha_isrfsr = (RooRealVar*) combined->var("alpha_isrfsr");
    RooAbsReal* profile_isrfsr = nll->createProfile(*alpha_isrfsr);
    poi->setVal(0.55);
    poi->setConstant();

    RooPlot* frame_isrfsr = alpha_isrfsr->frame();
    profile_isrfsr->plotOn(frame_isrfsr, Precision(0.1));
    TCanvas c_isrfsr = new TCanvas( "combined", "",800,600);
    FormatFrameForLikelihood(frame_isrfsr, "alpha_{isrfsr}");
    frame_isrfsr->Draw();
    fOut_f->cd("Summary");
    c1->Write((FilePrefixStr(channel).str()+"_profileLR_alpha_isrfsr").c_str() );
    delete frame; delete c1;
    poi->setConstant(kFALSE);
    */

    RooCurve* curve=frame->getCurve();
    Int_t curve_N=curve->GetN();
    Double_t* curve_x=curve->GetX();
    delete frame; delete c1;

    //
    // Verbose output from MINUIT
    //
    /*
    RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
    profile->getVal();
    RooMinuit* minuit = ((RooProfileLL*) profile)->minuit();
    minuit->setPrintLevel(5) ; // Print MINUIT messages
    minuit->setVerbose(5) ; // Print RooMinuit messages with parameter 
                                // changes (corresponds to the Verbose() option of fitTo()
    */
  
    Double_t * x_arr = new Double_t[curve_N];
    Double_t * y_arr_nll = new Double_t[curve_N];
//     Double_t y_arr_prof_nll[curve_N];
//     Double_t y_arr_prof[curve_N];

    for(int i=0; i<curve_N; i++){
      double f=curve_x[i];
      poi->setVal(f);
      x_arr[i]=f;
      y_arr_nll[i]=nll->getVal();
    }
    TGraph * g = new TGraph(curve_N, x_arr, y_arr_nll);
    g->SetName((FilePrefixStr(channel)+"_nll").c_str());
    g->Write(); 
    delete g;
    delete [] x_arr;
    delete [] y_arr_nll;

    /** find out what's inside the workspace **/
    //combined->Print();

  }


void HistoToWorkspaceFactoryFast::FormatFrameForLikelihood(RooPlot* frame, string /*XTitle*/, string YTitle){

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadColor(0);
    gStyle->SetCanvasColor(255);
    gStyle->SetTitleFillColor(255);
    gStyle->SetFrameFillColor(0);  
    gStyle->SetStatColor(255);
    
    RooAbsRealLValue* var = frame->getPlotVar();
    double xmin = var->getMin();
    double xmax = var->getMax();
    
    frame->SetTitle("");
    //      frame->GetXaxis()->SetTitle(XTitle.c_str());
    frame->GetXaxis()->SetTitle(var->GetTitle());
    frame->GetYaxis()->SetTitle(YTitle.c_str());
    frame->SetMaximum(2.);
    frame->SetMinimum(0.);
    TLine * line = new TLine(xmin,.5,xmax,.5);
    line->SetLineColor(kGreen);
    TLine * line90 = new TLine(xmin,2.71/2.,xmax,2.71/2.);
    line90->SetLineColor(kGreen);
    TLine * line95 = new TLine(xmin,3.84/2.,xmax,3.84/2.);
    line95->SetLineColor(kGreen);
    frame->addObject(line);
    frame->addObject(line90);
    frame->addObject(line95);
}
  
  TDirectory * HistoToWorkspaceFactoryFast::Makedirs( TDirectory * file, vector<string> names ){
    if(! file) return file;
    string path="";
    TDirectory* ptr=0;
    for(vector<string>::iterator itr=names.begin(); itr != names.end(); ++itr){
      if( ! path.empty() ) path+="/";
      path+=(*itr);
      ptr=file->GetDirectory(path.c_str());
      if( ! ptr ) ptr=file->mkdir((*itr).c_str());
      file=file->GetDirectory(path.c_str());
    }
    return ptr;
  }
  TDirectory * HistoToWorkspaceFactoryFast::Mkdir( TDirectory * file, string name ){
    if(! file) return file;
    TDirectory* ptr=0;
    ptr=file->GetDirectory(name.c_str());
    if( ! ptr )  ptr=file->mkdir(name.c_str());
    return ptr;
  }

}

}

