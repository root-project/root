/// \file
/// \ingroup tutorial_unfold
/// \notebook
/// Test program for the classes TUnfoldDensity and TUnfoldBinning.
///
/// A toy test of the TUnfold package
///
/// This is an example of unfolding a two-dimensional distribution
/// also using an auxiliary measurement to constrain some background
///
/// The example comprises several macros
///  - testUnfold5a.C   create root files with TTree objects for
///                      signal, background and data
///            - write files  testUnfold5_signal.root
///                           testUnfold5_background.root
///                           testUnfold5_data.root
///
///  - testUnfold5b.C   create a root file with the TUnfoldBinning objects
///            - write file  testUnfold5_binning.root
///
///  - testUnfold5c.C   loop over trees and fill histograms based on the
///                      TUnfoldBinning objects
///            - read  testUnfold5_binning.root
///                    testUnfold5_signal.root
///                    testUnfold5_background.root
///                    testUnfold5_data.root
///
///            - write testUnfold5_histograms.root
///
///  - testUnfold5d.C   run the unfolding
///            - read  testUnfold5_histograms.root
///            - write testUnfold5_result.root
///                    testUnfold5_result.ps
///
/// \macro_output
/// \macro_code
///
///  **Version 17.6, in parallel to changes in TUnfold**
///
/// #### History:
///  - Version 17.5, updated for reading binning from XML file
///  - Version 17.4, updated for reading binning from XML file
///  - Version 17.3, updated for reading binning from XML file
///  - Version 17.2, updated for reading binning from XML file
///  - Version 17.1, in parallel to changes in TUnfold
///  - Version 17.0 example for multi-dimensional unfolding
///
///  This file is part of TUnfold.
///
///  TUnfold is free software: you can redistribute it and/or modify
///  it under the terms of the GNU General Public License as published by
///  the Free Software Foundation, either version 3 of the License, or
///  (at your option) any later version.
///
///  TUnfold is distributed in the hope that it will be useful,
///  but WITHOUT ANY WARRANTY; without even the implied warranty of
///  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
///  GNU General Public License for more details.
///
///  You should have received a copy of the GNU General Public License
///  along with TUnfold.  If not, see <http://www.gnu.org/licenses/>.
///
/// \author Stefan Schmitt DESY, 14.10.2008

// uncomment this to read the binning schemes from the root file
// by default the binning is read from the XML file
// #define READ_BINNING_CINT


#include <iostream>
#include <map>
#include <cmath>
#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#ifndef READ_BINNING_CINT
#include <TDOMParser.h>
#include <TXMLDocument.h>
#include "TUnfoldBinningXML.h"
#else
#include "TUnfoldBinning.h"
#endif

using namespace std;


void testUnfold5c()
{
  // switch on histogram errors
  TH1::SetDefaultSumw2();

  //=======================================================
  // Step 1: open file to save histograms and binning schemes

  TFile *outputFile=new TFile("testUnfold5_histograms.root","recreate");

  //=======================================================
  // Step 2: read binning from XML
  //         and save them to output file

#ifdef READ_BINNING_CINT
  TFile *binningSchemes=new TFile("testUnfold5_binning.root");
#endif

  TUnfoldBinning *detectorBinning,*generatorBinning;

  outputFile->cd();

  // read binning schemes in XML format
#ifndef READ_BINNING_CINT
  TDOMParser parser;
  Int_t error=parser.ParseFile("testUnfold5binning.xml");
  if(error) cout<<"error="<<error<<" from TDOMParser\n";
  TXMLDocument const *XMLdocument=parser.GetXMLDocument();
  detectorBinning=
     TUnfoldBinningXML::ImportXML(XMLdocument,"detector");
  generatorBinning=
     TUnfoldBinningXML::ImportXML(XMLdocument,"generator");
#else
  binningSchemes->GetObject("detector",detectorBinning);
  binningSchemes->GetObject("generator",generatorBinning);

  delete binningSchemes;
#endif
  detectorBinning->Write();
  generatorBinning->Write();

  if(detectorBinning) {
     detectorBinning->PrintStream(cout);
  } else {
     cout<<"could not read 'detector' binning\n";
  }
  if(generatorBinning) {
     generatorBinning->PrintStream(cout);
  } else {
     cout<<"could not read 'generator' binning\n";
  }

  // pointers to various nodes in the binning scheme
  const TUnfoldBinning *detectordistribution=
     detectorBinning->FindNode("detectordistribution");

  const TUnfoldBinning *signalBinning=
     generatorBinning->FindNode("signal");

  const TUnfoldBinning *bgrBinning=
     generatorBinning->FindNode("background");

  // write binning schemes to output file

  //=======================================================
  // Step 3: book and fill data histograms

  Float_t etaRec,ptRec,discr,etaGen,ptGen;
  Int_t istriggered,issignal;

  outputFile->cd();

  TH1 *histDataReco=detectorBinning->CreateHistogram("histDataReco");
  TH1 *histDataTruth=generatorBinning->CreateHistogram("histDataTruth");

  TFile *dataFile=new TFile("testUnfold5_data.root");
  TTree *dataTree=(TTree *) dataFile->Get("data");

  if(!dataTree) {
     cout<<"could not read 'data' tree\n";
  }

  dataTree->ResetBranchAddresses();
  dataTree->SetBranchAddress("etarec",&etaRec);
  dataTree->SetBranchAddress("ptrec",&ptRec);
  dataTree->SetBranchAddress("discr",&discr);
  // for real data, only the triggered events are available
  dataTree->SetBranchAddress("istriggered",&istriggered);
  // data truth parameters
  dataTree->SetBranchAddress("etagen",&etaGen);
  dataTree->SetBranchAddress("ptgen",&ptGen);
  dataTree->SetBranchAddress("issignal",&issignal);
  dataTree->SetBranchStatus("*",1);


  cout<<"loop over data events\n";

  for(Int_t ievent=0;ievent<dataTree->GetEntriesFast();ievent++) {
     if(dataTree->GetEntry(ievent)<=0) break;
     // fill histogram with reconstructed quantities
     if(istriggered) {
        Int_t binNumber=
           detectordistribution->GetGlobalBinNumber(ptRec,etaRec,discr);
        histDataReco->Fill(binNumber);
     }
     // fill histogram with data truth parameters
     if(issignal) {
        // signal has true eta and pt
        Int_t binNumber=signalBinning->GetGlobalBinNumber(ptGen,etaGen);
        histDataTruth->Fill(binNumber);
     } else {
        // background only has reconstructed pt and eta
        Int_t binNumber=bgrBinning->GetGlobalBinNumber(ptRec,etaRec);
        histDataTruth->Fill(binNumber);
     }
  }

  delete dataTree;
  delete dataFile;

  //=======================================================
  // Step 4: book and fill histogram of migrations
  //         it receives events from both signal MC and background MC

  outputFile->cd();

  TH2 *histMCGenRec=TUnfoldBinning::CreateHistogramOfMigrations
     (generatorBinning,detectorBinning,"histMCGenRec");

  TFile *signalFile=new TFile("testUnfold5_signal.root");
  TTree *signalTree=(TTree *) signalFile->Get("signal");

  if(!signalTree) {
     cout<<"could not read 'signal' tree\n";
  }

  signalTree->ResetBranchAddresses();
  signalTree->SetBranchAddress("etarec",&etaRec);
  signalTree->SetBranchAddress("ptrec",&ptRec);
  signalTree->SetBranchAddress("discr",&discr);
  signalTree->SetBranchAddress("istriggered",&istriggered);
  signalTree->SetBranchAddress("etagen",&etaGen);
  signalTree->SetBranchAddress("ptgen",&ptGen);
  signalTree->SetBranchStatus("*",1);

  cout<<"loop over MC signal events\n";

  for(Int_t ievent=0;ievent<signalTree->GetEntriesFast();ievent++) {
     if(signalTree->GetEntry(ievent)<=0) break;

     // bin number on generator level for signal
     Int_t genBin=signalBinning->GetGlobalBinNumber(ptGen,etaGen);

     // bin number on reconstructed level
     // bin number 0 corresponds to non-reconstructed events
     Int_t recBin=0;
     if(istriggered) {
        recBin=detectordistribution->GetGlobalBinNumber(ptRec,etaRec,discr);
     }
     histMCGenRec->Fill(genBin,recBin);
  }

  delete signalTree;
  delete signalFile;

  TFile *bgrFile=new TFile("testUnfold5_background.root");
  TTree *bgrTree=(TTree *) bgrFile->Get("background");

  if(!bgrTree) {
     cout<<"could not read 'background' tree\n";
  }

  bgrTree->ResetBranchAddresses();
  bgrTree->SetBranchAddress("etarec",&etaRec);
  bgrTree->SetBranchAddress("ptrec",&ptRec);
  bgrTree->SetBranchAddress("discr",&discr);
  bgrTree->SetBranchAddress("istriggered",&istriggered);
  bgrTree->SetBranchStatus("*",1);

  cout<<"loop over MC background events\n";

  for(Int_t ievent=0;ievent<bgrTree->GetEntriesFast();ievent++) {
     if(bgrTree->GetEntry(ievent)<=0) break;

     // here, for background only reconstructed quantities are known
     // and only the reconstructed events are relevant
     if(istriggered) {
        // bin number on generator level for background
        Int_t genBin=bgrBinning->GetGlobalBinNumber(ptRec,etaRec);
        // bin number on reconstructed level
        Int_t recBin=detectordistribution->GetGlobalBinNumber
           (ptRec,etaRec,discr);
        histMCGenRec->Fill(genBin,recBin);
     }
  }

  delete bgrTree;
  delete bgrFile;

  outputFile->Write();
  delete outputFile;

}
