/// \file
/// \ingroup tutorial_unfold5
/// \notebook -nodraw
/// Version 17.0 example for multi-dimensional unfolding
///
/// \macro_output
/// \macro_code
///
/// \author Stefan Schmitt, DESY

#include <iostream>
#include <map>
#include <cmath>
#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include "TUnfoldBinning.h"

using namespace std;

void testUnfold5c()
{
  // switch on histogram errors
  TH1::SetDefaultSumw2();

  //=======================================================
  // Step 1: open file to save histograms and binning schemes

  TFile *outputFile=new TFile("testUnfold5_histograms.root","recreate");

  //=======================================================
  // Step 2: read binning from file
  //         and save them to output file

  TFile *binningSchemes=new TFile("testUnfold5_binning.root");

  TUnfoldBinning *detectorBinning,*generatorBinning;

  outputFile->cd();

  binningSchemes->GetObject("detector",detectorBinning);
  binningSchemes->GetObject("generator",generatorBinning);

  delete binningSchemes;

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

  // pointers to various nodes in the bining scheme
  const TUnfoldBinning *detectordistribution=
     detectorBinning->FindNode("detectordistribution");

  const TUnfoldBinning *signalBinning=
     generatorBinning->FindNode("signal");

  const TUnfoldBinning *bgrBinning=
     generatorBinning->FindNode("background");

  // write binnig schemes to output file

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
