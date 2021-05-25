/// \file
/// \ingroup tutorial_unfold
/// \notebook
/// Test program for the classes TUnfoldDensity and TUnfoldBinning
///
/// A toy test of the TUnfold package
///
/// This example is documented in conference proceedings:
///
///   arXiv:1611.01927
/// 12th Conference on Quark Confinement and the Hadron Spectrum (Confinement XII)
///
/// This is an example of unfolding a two-dimensional distribution
/// also using an auxiliary measurement to constrain some background
///
/// The example comprises several macros
///  - testUnfold7a.C   create root files with TTree objects for
///                      signal, background and data
///            - write files  testUnfold7_signal.root
///                           testUnfold7_background.root
///                           testUnfold7_data.root
///
///  - testUnfold7b.C   loop over trees and fill histograms based on the
///                      TUnfoldBinning objects
///            - read  testUnfold7binning.xml
///                    testUnfold7_signal.root
///                    testUnfold7_background.root
///                    testUnfold7_data.root
///
///            - write testUnfold7_histograms.root
///
///  - testUnfold7c.C   run the unfolding
///            - read  testUnfold7_histograms.root
///            - write testUnfold7_result.root
///                    testUnfold7_result.ps
///
/// \macro_output
/// \macro_code
///
///  **Version 17.6, in parallel to changes in TUnfold**
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


/* below is the content of the file testUnfold7binning.xml,
   which is required as input to run this example.

<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE TUnfoldBinning SYSTEM "tunfoldbinning.dtd">
<TUnfoldBinning>
<BinningNode name="fine" firstbin="1" factor="1">
  <Axis name="pt" lowEdge="0.">
   <Bin repeat="20" width="1."/>
   <Bin repeat="12" width="2.5"/>
   <Bin location="overflow" width="10"/>
  </Axis>
</BinningNode>
<BinningNode name="coarse" firstbin="1" factor="1">
  <Axis name="pt" lowEdge="0.">
   <Bin repeat="10" width="2"/>
   <Bin repeat="6" width="5"/>
   <Bin location="overflow" width="10"/>
  </Axis>
</BinningNode>
</TUnfoldBinning>

 */

#include <iostream>
#include <map>
#include <cmath>
#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TDOMParser.h>
#include <TXMLDocument.h>
#include "TUnfoldBinningXML.h"

using namespace std;


void testUnfold7b()
{
  // switch on histogram errors
  TH1::SetDefaultSumw2();

  //=======================================================
  // Step 1: open file to save histograms and binning schemes

  TFile *outputFile=new TFile("testUnfold7_histograms.root","recreate");

  //=======================================================
  // Step 2: read binning from XML
  //         and save them to output file

  TUnfoldBinning *fineBinningRoot,*coarseBinningRoot;

  outputFile->cd();

  // read binning schemes in XML format

  TDOMParser parser;
  TString dir = gSystem->UnixPathName(gSystem->GetDirName(__FILE__));
  Int_t error = parser.ParseFile(dir+"/testUnfold7binning.xml");
  if(error) {
     cout<<"error="<<error<<" from TDOMParser\n";
     cout<<"==============================================================\n";
     cout<<"Maybe the file testUnfold7binning.xml is missing?\n";
     cout<<"The content of the file is included in the comments section\n";
     cout<<"of this macro \"testUnfold7b.C\"\n";
     cout<<"==============================================================\n";
  }
  TXMLDocument const *XMLdocument=parser.GetXMLDocument();
  fineBinningRoot=
     TUnfoldBinningXML::ImportXML(XMLdocument,"fine");
  coarseBinningRoot=
     TUnfoldBinningXML::ImportXML(XMLdocument,"coarse");

  // write binning schemes to output file
  fineBinningRoot->Write();
  coarseBinningRoot->Write();

  if(fineBinningRoot) {
     fineBinningRoot->PrintStream(cout);
  } else {
     cout<<"could not read 'detector' binning\n";
  }
  if(coarseBinningRoot) {
     coarseBinningRoot->PrintStream(cout);
  } else {
     cout<<"could not read 'generator' binning\n";
  }

  TUnfoldBinning const *fineBinning=fineBinningRoot;//->FindNode("ptfine");
  TUnfoldBinning const *coarseBinning=coarseBinningRoot;//->FindNode("ptcoarse");


  //=======================================================
  // Step 3: book and fill data histograms

  Float_t ptRec[3],ptGen[3],weight;
  Int_t isTriggered,isSignal;

  outputFile->cd();

  TH1 *histDataRecF=fineBinning->CreateHistogram("histDataRecF");
  TH1 *histDataRecC=coarseBinning->CreateHistogram("histDataRecC");
  TH1 *histDataBgrF=fineBinning->CreateHistogram("histDataBgrF");
  TH1 *histDataBgrC=coarseBinning->CreateHistogram("histDataBgrC");
  TH1 *histDataGen=coarseBinning->CreateHistogram("histDataGen");

  TFile *dataFile=new TFile("testUnfold7_data.root");
  TTree *dataTree=(TTree *) dataFile->Get("data");

  if(!dataTree) {
     cout<<"could not read 'data' tree\n";
  }

  dataTree->ResetBranchAddresses();
  dataTree->SetBranchAddress("ptrec",ptRec);
  //dataTree->SetBranchAddress("discr",&discr);
  // for real data, only the triggered events are available
  dataTree->SetBranchAddress("istriggered",&isTriggered);
  // data truth parameters
  dataTree->SetBranchAddress("ptgen",ptGen);
  dataTree->SetBranchAddress("issignal",&isSignal);
  dataTree->SetBranchStatus("*",1);

  cout<<"loop over data events\n";

#define VAR_REC (ptRec[2])
#define VAR_GEN (ptGen[2])

  for(Int_t ievent=0;ievent<dataTree->GetEntriesFast();ievent++) {
     if(dataTree->GetEntry(ievent)<=0) break;
     // fill histogram with reconstructed quantities
     if(isTriggered) {
        int binF=fineBinning->GetGlobalBinNumber(VAR_REC);
        int binC=coarseBinning->GetGlobalBinNumber(VAR_REC);
        histDataRecF->Fill(binF);
        histDataRecC->Fill(binC);
        if(!isSignal) {
           histDataBgrF->Fill(binF);
           histDataBgrC->Fill(binC);
        }
     }
     // fill histogram with data truth parameters
     if(isSignal) {
        int binGen=coarseBinning->GetGlobalBinNumber(VAR_GEN);
        histDataGen->Fill(binGen);
     }
  }

  delete dataTree;
  delete dataFile;

  //=======================================================
  // Step 4: book and fill histogram of migrations
  //         it receives events from both signal MC and background MC

  outputFile->cd();

  TH2 *histMcsigGenRecF=TUnfoldBinning::CreateHistogramOfMigrations
     (coarseBinning,fineBinning,"histMcsigGenRecF");
  TH2 *histMcsigGenRecC=TUnfoldBinning::CreateHistogramOfMigrations
     (coarseBinning,coarseBinning,"histMcsigGenRecC");
  TH1 *histMcsigRecF=fineBinning->CreateHistogram("histMcsigRecF");
  TH1 *histMcsigRecC=coarseBinning->CreateHistogram("histMcsigRecC");
  TH1 *histMcsigGen=coarseBinning->CreateHistogram("histMcsigGen");

  TFile *signalFile=new TFile("testUnfold7_signal.root");
  TTree *signalTree=(TTree *) signalFile->Get("signal");

  if(!signalTree) {
     cout<<"could not read 'signal' tree\n";
  }

  signalTree->ResetBranchAddresses();
  signalTree->SetBranchAddress("ptrec",&ptRec);
  //signalTree->SetBranchAddress("discr",&discr);
  signalTree->SetBranchAddress("istriggered",&isTriggered);
  signalTree->SetBranchAddress("ptgen",&ptGen);
  signalTree->SetBranchAddress("weight",&weight);
  signalTree->SetBranchStatus("*",1);

  cout<<"loop over MC signal events\n";

  for(Int_t ievent=0;ievent<signalTree->GetEntriesFast();ievent++) {
     if(signalTree->GetEntry(ievent)<=0) break;

     int binC=0,binF=0;
     if(isTriggered) {
        binF=fineBinning->GetGlobalBinNumber(VAR_REC);
        binC=coarseBinning->GetGlobalBinNumber(VAR_REC);
     }
     int binGen=coarseBinning->GetGlobalBinNumber(VAR_GEN);
     histMcsigGenRecF->Fill(binGen,binF,weight);
     histMcsigGenRecC->Fill(binGen,binC,weight);
     histMcsigRecF->Fill(binF,weight);
     histMcsigRecC->Fill(binC,weight);
     histMcsigGen->Fill(binGen,weight);
  }

  delete signalTree;
  delete signalFile;

  outputFile->cd();

  TH1 *histMcbgrRecF=fineBinning->CreateHistogram("histMcbgrRecF");
  TH1 *histMcbgrRecC=coarseBinning->CreateHistogram("histMcbgrRecC");

  TFile *bgrFile=new TFile("testUnfold7_background.root");
  TTree *bgrTree=(TTree *) bgrFile->Get("background");

  if(!bgrTree) {
     cout<<"could not read 'background' tree\n";
  }

  bgrTree->ResetBranchAddresses();
  bgrTree->SetBranchAddress("ptrec",&ptRec);
  //bgrTree->SetBranchAddress("discr",&discr);
  bgrTree->SetBranchAddress("istriggered",&isTriggered);
  bgrTree->SetBranchAddress("weight",&weight);
  bgrTree->SetBranchStatus("*",1);

  cout<<"loop over MC background events\n";

  for(Int_t ievent=0;ievent<bgrTree->GetEntriesFast();ievent++) {
     if(bgrTree->GetEntry(ievent)<=0) break;

     // here, for background only reconstructed quantities are known
     // and only the reconstructed events are relevant
     if(isTriggered) {
        int binF=fineBinning->GetGlobalBinNumber(VAR_REC);
        int binC=coarseBinning->GetGlobalBinNumber(VAR_REC);
        histMcbgrRecF->Fill(binF,weight);
        histMcbgrRecC->Fill(binC,weight);
     }
  }

  delete bgrTree;
  delete bgrFile;

  outputFile->Write();
  delete outputFile;

}
