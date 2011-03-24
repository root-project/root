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
This is a package that creates a RooFit probability density function from ROOT histograms 
of expected distributions and histograms that represent the +/- 1 sigma variations 
from systematic effects. The resulting probability density function can then be used
with any of the statistical tools provided within RooStats, such as the profile 
likelihood ratio, Feldman-Cousins, etc.  In this version, the model is directly
fed to a likelihodo ratio test, but it needs to be further factorized.</p>

<p>
The user needs to provide histograms (in picobarns per bin) and configure the job
with XML.  The configuration XML is defined in the file config/Config.dtd, but essentially
it is organized as follows (see config/Combination.xml and config/ee.xml for examples)</p>

<ul>
<li> - a top level 'Combination' that is composed of:</li>
<ul>
 <li>- several 'Channels' (eg. ee, emu, mumu), which are composed of:</li>
 <ul>
  <li>- several 'Samples' (eg. signal, bkg1, bkg2, ...), each of which has:</li>
  <ul>
   <li> - a name</li>
   <li> - if the sample is normalized by theory (eg N = L*sigma) or not (eg. data driven)</li>
   <li> - a nominal expectation histogram</li>
   <li> - a named 'Normalization Factor' (which can be fixed or allowed to float in a fit)</li>
   <li> - several 'Overall Systematics' in normalization with:</li>
   <ul>
    <li> - a name</li>
    <li> - +/- 1 sigma variations (eg. 1.05 and 0.95 for a 5% uncertainty)</li>
   </ul>
   <li>- several 'Histogram Systematics' in shape with:</li>
   <ul>
    <li>- a name (which can be shared with the OverallSyst if correlated)</li>
    <li>- +/- 1 sigma variational histograms</li>
   </ul>
  </ul>
 </ul>
 <li>- several 'Measurements' (corresponding to a full fit of the model) each of which specifies</li>
 <ul>
  <li>- a name for this fit to be used in tables and files</li>
  <ul>
   <li>      - what is the luminosity associated to the measurement in picobarns</li>
   <li>      - which bins of the histogram should be used</li>
   <li>      - what is the relative uncertainty on the luminosity </li>
   <li>      - what is (are) the parameter(s) of interest that will be measured</li>
   <li>      - which parameters should be fixed/floating (eg. nuisance parameters)</li>
  </ul>
 </ul>
</ul>
END_HTML
*/
//


// from std
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>

// from root
#include "TFile.h"
#include "TH1F.h"
#include "TDOMParser.h"
#include "TXMLAttr.h"
#include "TString.h"

// from roofit
#include "RooStats/ModelConfig.h"

// from this package
#include "Helper.h"
#include "ConfigParser.h"
#include "RooStats/HistFactory/EstimateSummary.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"


using namespace RooFit;
using namespace RooStats;
using namespace HistFactory;


// main is int MakeModelAndMeasurements
void fastDriver(string input ){
  // TO DO:
  // would like to fully factorize the XML parsing.  
  // No clear need to have some here and some in ConfigParser

  /*** read in the input xml ***/
  TDOMParser xmlparser;
  Int_t parseError = xmlparser.ParseFile( input.c_str() );
  if( parseError ) { 
    std::cerr << "Loading of xml document \"" << input
          << "\" failed" << std::endl;
  } 

  cout << "reading input : " << input << endl;
  TXMLDocument* xmldoc = xmlparser.GetXMLDocument();
  TXMLNode* rootNode = xmldoc->GetRootNode();

  if( rootNode->GetNodeName() == TString( "Combination" ) ){
    string outputFileName, outputFileNamePrefix;
    vector<string> xml_input;

    TListIter attribIt = rootNode->GetAttributes();
    TXMLAttr* curAttr = 0;
    while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
      if( curAttr->GetName() == TString( "OutputFilePrefix" ) ) {
        outputFileNamePrefix=string(curAttr->GetValue());
        cout << "output file is : " << outputFileName << endl;
      }
    } 
    TXMLNode* node = rootNode->GetChildren();
    while( node != 0 ) {
      if( node->GetNodeName() == TString( "Input" ) ) {
        xml_input.push_back(node->GetText());
      }
      node = node->GetNextNode();
    }
    node = rootNode->GetChildren();
    while( node != 0 ) {
      if( node->GetNodeName() == TString( "Measurement" ) ) {

        Double_t nominalLumi=0, lumiRelError=0, lumiError=0;
        Int_t lowBin=0, highBin=0;
        string rowTitle, POI, mode;
        vector<string> systToFix;
        map<string,double> gammaSyst;
        map<string,double> uniformSyst;
        map<string,double> logNormSyst;
	bool exportOnly = false;

	//        TListIter attribIt = node->GetAttributes();
	//        TXMLAttr* curAttr = 0;
        attribIt = node->GetAttributes();
        curAttr = 0;
        while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
          if( curAttr->GetName() == TString( "Lumi" ) ) {
            nominalLumi=atof(curAttr->GetValue());
          }
          if( curAttr->GetName() == TString( "LumiRelErr" ) ) {
            lumiRelError=atof(curAttr->GetValue());
          }
          if( curAttr->GetName() == TString( "BinLow" ) ) {
            lowBin=atoi(curAttr->GetValue());
          }
          if( curAttr->GetName() == TString( "BinHigh" ) ) {
            highBin=atoi(curAttr->GetValue());
          }
          if( curAttr->GetName() == TString( "Name" ) ) {
            rowTitle=curAttr->GetValue();
            outputFileName=outputFileNamePrefix+"_"+rowTitle+".root";
          }
          if( curAttr->GetName() == TString( "Mode" ) ) {
            mode=curAttr->GetValue();
          }
          if( curAttr->GetName() == TString( "ExportOnly" ) ) {
            if(curAttr->GetValue() == TString( "True" ) )
	      exportOnly = true;
	    else
	      exportOnly = false;
          }
        }
        lumiError=nominalLumi*lumiRelError;

        TXMLNode* mnode = node->GetChildren();
        while( mnode != 0 ) {
          if( mnode->GetNodeName() == TString( "POI" ) ) {
            POI=mnode->GetText();
          }
          if( mnode->GetNodeName() == TString( "ParamSetting" ) ) {
	    //            TListIter attribIt = mnode->GetAttributes();
            //TXMLAttr* curAttr = 0;
	    attribIt = mnode->GetAttributes();
	    curAttr = 0;
            while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
              if( curAttr->GetName() == TString( "Const" ) ) {
                if(curAttr->GetValue()==TString("True")){
                  AddSubStrings(systToFix, mnode->GetText());
                }
              }
            }
          }
          if( mnode->GetNodeName() == TString( "ConstraintTerm" ) ) {
            vector<string> syst; string type = ""; double rel = 0;
            AddSubStrings(syst,mnode->GetText());
	    //            TListIter attribIt = mnode->GetAttributes();
	    //            TXMLAttr* curAttr = 0;
	    attribIt = mnode->GetAttributes();
            curAttr = 0;
            while( ( curAttr = dynamic_cast< TXMLAttr* >( attribIt() ) ) != 0 ) {
              if( curAttr->GetName() == TString( "Type" ) ) {
                type = curAttr->GetValue();
              }
              if( curAttr->GetName() == TString( "RelativeUncertainty" ) ) {
                rel = atof(curAttr->GetValue());
              }
            }
            if (type=="Gamma" && rel!=0) {
              for (vector<string>::const_iterator it=syst.begin(); it!=syst.end(); it++) gammaSyst[(*it).c_str()] = rel;
            }
            if (type=="Uniform" && rel!=0) {
              for (vector<string>::const_iterator it=syst.begin(); it!=syst.end(); it++) uniformSyst[(*it).c_str()] = rel;
            }
            if (type=="LogNormal" && rel!=0) {
              for (vector<string>::const_iterator it=syst.begin(); it!=syst.end(); it++) logNormSyst[(*it).c_str()] = rel;
            }
          }
          mnode = mnode->GetNextNode();
        }

        /* Do measurement */
        cout << "using lumi = " << nominalLumi << " and lumiError = " << lumiError
             << " including bins between " << lowBin << " and " << highBin << endl;
        cout << "fixing the following parameters:"  << endl;
        for(vector<string>::iterator itr=systToFix.begin(); itr!=systToFix.end(); ++itr){
          cout << "   " << *itr << endl;
        }

        /***
            Construction of Model. Only requirement is that they return vector<vector<EstimateSummary> >
	    This is where we use the factory.
        ***/

        vector<vector<EstimateSummary> > summaries;
        if(xml_input.empty()){
          cerr << "no input channels found" << endl;
          exit(1);
        }


        vector<RooWorkspace*> chs;
        vector<string> ch_names;
        TFile* outFile = new TFile(outputFileName.c_str(), "recreate");
        HistoToWorkspaceFactoryFast factory(outputFileNamePrefix, rowTitle, systToFix, nominalLumi, lumiError, lowBin, highBin , outFile);


        // for results tables
        fprintf(factory.pFile, " %s &", rowTitle.c_str() );

        // read the xml for each channel and combine
        for(vector<string>::iterator itr=xml_input.begin(); itr!=xml_input.end(); ++itr){
          vector<EstimateSummary> oneChannel;
          // read xml
          ReadXmlConfig(*itr, oneChannel, nominalLumi);
          // not really needed anymore
          summaries.push_back(oneChannel);
          // use factory to create the workspace
          string ch_name=oneChannel[0].channel;
          ch_names.push_back(ch_name);
          RooWorkspace * ws = factory.MakeSingleChannelModel(oneChannel, systToFix);
          chs.push_back(ws);
          // set poi in ModelConfig
          ModelConfig * proto_config = (ModelConfig *) ws->obj("ModelConfig");
          cout << "Setting Parameter of Interest as :" << POI << endl;
          RooRealVar* poi = (RooRealVar*) ws->var(POI.c_str());
          RooArgSet * params= new RooArgSet;
          params->add(*poi);
          proto_config->SetParameters(*params);


          // Gamma/Uniform Constraints:
          // turn some Gaussian constraints into Gamma/Uniform/LogNorm constraints, rename model newSimPdf
	  if(gammaSyst.size()>0 || uniformSyst.size()>0 || logNormSyst.size()>0) {
	    factory.EditSyst(ws,("model_"+oneChannel[0].channel).c_str(),gammaSyst,uniformSyst,logNormSyst);
	    proto_config->SetPdf(*ws->pdf("newSimPdf"));
          }

	  // fill out ModelConfig and export
          RooAbsData* expData = ws->data("asimovData");
	  proto_config->GuessObsAndNuisance(*expData);
	  ws->writeToFile((outputFileNamePrefix+"_"+ch_name+"_"+rowTitle+"_model.root").c_str());

	  // do fit unless exportOnly requested
	  if(!exportOnly){
	    if(ws->data("obsData")){
	      factory.FitModel(ws, ch_name, "newSimPdf", "obsData", false);
	    } else {
	      factory.FitModel(ws, ch_name, "newSimPdf", "asimovData", false);
	    }

	  }
          fprintf(factory.pFile, " & " );
        }

        /***
	    Make the combined model:
            If you want output histograms in root format, create and pass it to the combine routine.
            "combine" : will do the individual cross-section measurements plus combination

        ***/


          
        if(mode.find("comb")!=string::npos){ 
          RooWorkspace* ws=factory.MakeCombinedModel(ch_names,chs);
          // Gamma/Uniform Constraints:
          // turn some Gaussian constraints into Gamma/Uniform/logNormal constraints, rename model newSimPdf
	  if(gammaSyst.size()>0 || uniformSyst.size()>0 || logNormSyst.size()>0) 
	    factory.EditSyst(ws,"simPdf",gammaSyst,uniformSyst,logNormSyst);
          //
          // set parameter of interest according to the configuration
          //
          ModelConfig * combined_config = (ModelConfig *) ws->obj("ModelConfig");
          cout << "Setting Parameter of Interest as :" << POI << endl;
          RooRealVar* poi = (RooRealVar*) ws->var((POI).c_str());
          //RooRealVar* poi = (RooRealVar*) ws->var((POI+"_comb").c_str());
          RooArgSet * params= new RooArgSet;
          cout << poi << endl;
          params->add(*poi);
          combined_config->SetParameters(*params);
          ws->Print();

          // Set new PDF if there are gamma/uniform constraint terms
          if(gammaSyst.size()>0 || uniformSyst.size()>0 || logNormSyst.size()>0) 
	    combined_config->SetPdf(*ws->pdf("newSimPdf"));

          RooAbsData* simData = ws->data("asimovData");
	  combined_config->GuessObsAndNuisance(*simData);
	  //	  ws->writeToFile(("results/model_combined_edited.root").c_str());
	  ws->writeToFile((outputFileNamePrefix+"_combined_"+rowTitle+"_model.root").c_str());

	  // TO DO:
          // Totally factorize the statistical test in "fit Model" to a different area
	  if(!exportOnly){
	    if(ws->data("obsData")){
	      factory.FitModel(ws, "combined", "simPdf", "obsData", false);
	    } else {
	      factory.FitModel(ws, "combined", "simPdf", "asimovData", false);
	    }
	  }
        }


        fprintf(factory.pFile, " \\\\ \n");

        outFile->Close();
        delete outFile;

      }
      node = node->GetNextNode(); // next measurement
    }
  }
}



