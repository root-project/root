// @(#)root/tmva $Id: TMVA_Factory.cxx,v 1.3 2006/05/08 15:39:03 brun Exp $   
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Factory                                                          *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// This is the main MVA steering class: it creates all MVA methods,     
// and guides them through the training, testing and evaluation         
// phases. It also manages multiple MVA handling in case of distinct    
// phase space requirements (cuts).                                     
//_______________________________________________________________________

#include <iostream>
#include "Riostream.h"
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TMVA_MethodBase.h"
#include "TString.h"
#include "TFile.h"

#include "TLeaf.h"
#include "TEventList.h"
#include "TH2F.h"
#include "TText.h"
#include "TStyle.h"
#include "TMatrixF.h"
#include "TMatrixDSym.h"
#include "TPaletteAxis.h"
#include "TMVA_Factory.h"
#include "TMVA_Tools.h"
#include "TMVA_AsciiConverter.h"
#include "TMVA_MethodCuts.h"
#include "TMVA_MethodFisher.h"
#include "TMVA_MethodTMlpANN.h"
#include "TMVA_MethodCFMlpANN.h"
#include "TMVA_MethodLikelihood.h"
#include "TMVA_MethodVariable.h"
#include "TMVA_MethodHMatrix.h"
#include "TMVA_MethodPDERS.h"
#include "TMVA_MethodBDT.h"

#define DEBUG_TMVA_Factory   kFALSE
#define MinNoTrainingEvents 10
#define MinNoTestEvents     1
#define basketsize 1280000

const TString BCwhite__f  ( "\033[1;37m" );
const TString BCred__f    ( "\033[31m"   );
const TString BCblue__f   ( "\033[34m"   );
const TString BCblue__b   ( "\033[44m"   );
const TString BCred__b    ( "\033[1;41m"   );
const TString EC__        ( "\033[0m"    );
const TString BClblue__b  ( "\033[1;44m" );

using namespace std;

ClassImp(TMVA_Factory)

//_______________________________________________________________________
TMVA_Factory::TMVA_Factory( TString jobName, TFile* theTargetFile, TString theOption )
  : fSignalFile      ( 0 ), 
    fBackgFile       ( 0 ), 
    fTrainingTree    ( 0 ),
    fTestTree        ( 0 ),
    fMultiCutTestTree( 0 ),
    fSignalTree      ( 0 ),
    fBackgTree       ( 0 ),
    fTargetFile      ( theTargetFile ),
    fOptions         ( theOption ),
    fMultipleMVAs    (kFALSE),
    fMultipleStoredOptions(kFALSE)

{  
  // default  constructor
  fJobName = jobName;
  this->Greeting("Color");
  // interpret option string 
  // at present, only verbose option defined
  TString s = fOptions;
  s.ToUpper();
  if (s.Contains("V")) fVerbose = kTRUE;
  else                 fVerbose = kFALSE;

  fLocalTDir = gDirectory;
}

//_______________________________________________________________________
TMVA_Factory::TMVA_Factory( TFile* theTargetFile)
  : fSignalFile      ( 0 ), 
    fBackgFile       ( 0 ), 
    fTrainingTree    ( 0 ),
    fTestTree        ( 0 ),
    fMultiCutTestTree( 0 ),
    fSignalTree      ( 0 ),
    fBackgTree       ( 0 ),
    fTargetFile      ( theTargetFile ),
    fOptions         ( ""),
    fMultipleMVAs    (kFALSE),
    fMultipleStoredOptions(kFALSE)

{  
  fJobName = "";
  this->Greeting("Color");
  // interpret option string 
  // at present, only verbose option defined
  TString s = fOptions;
  s.ToUpper();
  if (s.Contains("V")) fVerbose = kTRUE;
  else                 fVerbose = kFALSE;
  
  fLocalTDir = gDirectory;
}

//_______________________________________________________________________
void TMVA_Factory::Greeting(TString op){
  op.ToUpper();
  if (op.Contains("COLOR") || op.Contains("COLOUR") ) {
    cout << "--- " << GetName() << ": " << BCred__f 
	 << "_______________________________ _ _ _ _ _ _ _" << EC__ << endl;
    cout << "--- " << GetName() << ": " << BCblue__f
	 << BCred__b << BCwhite__f << " // " << EC__
	 << BCwhite__f << BClblue__b 
	 << "|\\  /|| \\  //  /\\\\\\\\\\\\\\\\\\\\\\\\ \\ \\ \\ \\ \\ \\ \\ " << EC__ << endl;
    cout << "--- " << GetName() << ": "<< BCblue__f
	 << BCred__b << BCwhite__f << "//  " << EC__
	 << BCwhite__f << BClblue__b 
	 << "| \\/ ||  \\//  /--\\\\\\\\\\\\\\\\\\\\\\\\f\\a\\c\\t\\o\\r\\y\\" << EC__ << endl;
  }
  else {
    cout << "--- " << GetName() << ": " 
 	 << "_______________________________ _ _ _ _ _ _ _" << endl;
    cout << "--- " << GetName() << ":  // "
	 << "|\\  /|| \\  //  /\\\\\\\\\\\\\\\\\\\\\\\\ \\ \\ \\ \\ \\ \\ \\ " << endl;
    cout << "--- " << GetName() << ": //  " 
	 << "| \\/ ||  \\//  /--\\\\\\\\\\\\\\\\\\\\\\\\f\\a\\c\\t\\o\\r\\y\\" << endl;
  }
  
}




//_______________________________________________________________________
TMVA_Factory::~TMVA_Factory( void )
{
  // default destructor
  //
  // *** segmentation fault occurs when deleting this object :-( ***
  //   fTrainingTree->Delete();
  //
  // *** cannot delete: need to clarify ownership :-( ***
  //   fSignalTree->Delete();
  //   fBackgTree->Delete();
  this->DeleteAllMethods();
}

//_______________________________________________________________________
void TMVA_Factory::DeleteAllMethods( void )
{
  // delete methods
  vector<TMVA_MethodBase*>::iterator itrMethod    = fMethods.begin();
  vector<TMVA_MethodBase*>::iterator itrMethodEnd = fMethods.end();
  for(; itrMethod != itrMethodEnd; itrMethod++) {
    if (Verbose())
      cout << "--- " << GetName() << " <verbose>: delete method: " 
	   << (*itrMethod)->GetName() 
	   << endl;    
    delete (*itrMethod);
  }
  // erase method vector
  itrMethod    = fMethods.begin();
  itrMethodEnd = fMethods.end();
  fMethods.erase(itrMethod ,itrMethodEnd );

}

//_______________________________________________________________________
Bool_t TMVA_Factory::SetInputTrees(TTree* signal, TTree* background)
{
  this->SetSignalTree(signal);
  this->SetBackgroundTree(background);
  return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA_Factory::SetInputTrees(TTree* inputTree, TCut SigCut, TCut BgCut)
{
  fSignalTree = inputTree->CloneTree(0);
  fBackgTree  = inputTree->CloneTree(0);
   
  TIter next_branch1( fSignalTree->GetListOfBranches() );
  while (TBranch *branch = (TBranch*)next_branch1())
    branch->SetBasketSize(basketsize);

  TIter next_branch2( fBackgTree->GetListOfBranches() );
  while (TBranch *branch = (TBranch*)next_branch2())
    branch->SetBasketSize(basketsize);

  inputTree->Draw(">>signalList",SigCut,"goff");
  TEventList *signalList = (TEventList*)gDirectory->Get("signalList");

  inputTree->Draw(">>backgList",BgCut,"goff");    
  TEventList *backgList = (TEventList*)gDirectory->Get("backgList");

  if (backgList->GetN() == inputTree->GetEntries()){
    TCut bgcut= !SigCut;
    inputTree->Draw(">>backgList",bgcut,"goff");    
    backgList = (TEventList*)gDirectory->Get("backgList");
  }
  signalList->Print();
  backgList->Print();
  
  for (Int_t i=0;i<inputTree->GetEntries(); i++) {
    inputTree->GetEntry(i);
     if ((backgList->Contains(i)) && (signalList->Contains(i))){
       cout << "--- " << GetName() << ": WARNING  Event "<<i
 	   <<" is selected for signal and background sample! Skip it!"<<endl;
       continue;
     }
    if (signalList->Contains(i)) fSignalTree->Fill();
    if (backgList->Contains(i)) fBackgTree->Fill();
  }

   delete signalList;
   delete  backgList;
   return (0 != fSignalTree && 0 != fBackgTree) ? kTRUE : kFALSE;
    
}

//_______________________________________________________________________
Bool_t TMVA_Factory::SetInputTrees( TString datFileS, TString datFileB )
{
  // create trees from these files
  fSignalTree = new TTree( "TreeS", "Tree (S)" );
  fBackgTree  = new TTree( "TreeB", "Tree (B)" );

  TMVA_AsciiConverter* sConv = new TMVA_AsciiConverter( datFileS, fSignalTree );
  TMVA_AsciiConverter* bConv = new TMVA_AsciiConverter( datFileB, fBackgTree  );

  if (!sConv->GetFileStatus()) {
    cout << "--- " << GetName() << ": Error: could not open file: " << datFileS << endl;
    return kFALSE;
  }
  if (!bConv->GetFileStatus()) {
    cout << "--- " << GetName() << ": Error: could not open file: " << datFileB << endl;
    return kFALSE;
  }

  fSignalTree->Write();
  fBackgTree ->Write();

  delete sConv;
  delete bConv;

  return (0 != fSignalTree && 0 != fBackgTree) ? kTRUE : kFALSE;
}    

//_______________________________________________________________________
void TMVA_Factory::BookMultipleMVAs(TString theVariable, Int_t nbins, Double_t *array)
{
  // here the mess starts!
  fMultipleMVAs=kTRUE;
  fMultiTrain=kFALSE , fMultiTest=kFALSE , fMultiEvalVar=kFALSE , fMultiEval=kFALSE ; 
  // at least one sanity check:
  // this method must be called *before* any Method has been booked!
  if ( fMethods.size() > 0){
    cout << "--- " << GetName() << ": ERROR! BookMultipleMVAs must be called befor booking any Method!"<<endl;
    exit(1);
  }
   fMultipleStoredOptions=kFALSE;
   fMultiVar1=theVariable;

   // print out
   cout << "--- " << GetName() << " : MulitCut Analysis Booked:  "
	<<theVariable<<" is splitted in "<<nbins<<" bins:"<< endl;

   // check if already some bins are booked
   // if yes add the new bins to *every* existing one
   if (fMultipleMVAnames.size() >0){
     
     // loop over existing bins and add the new ones
     // store their values in temporary opjects
     Int_t nOldBins =  fMultipleMVAnames.size();
     //TString SimpleName[nOldBins], Description[nOldBins];
     TString SimpleName[1000], Description[1000]; //please check
     //TCut OldCut[nOldBins];
     TCut OldCut[1000];
     Int_t binc=0;
     for (map<TString, std::pair<TString,TCut> >::iterator oldBin = fMultipleMVAnames.begin();
	  oldBin != fMultipleMVAnames.end(); oldBin++) {
       SimpleName[binc]=oldBin->first;
       Description[binc]=(oldBin->second).first;
       OldCut[binc]=(oldBin->second).second;
       binc++;
     } // end of loop over existing bins
     // erase the old map
     map<TString, std::pair<TString,TCut> >::iterator startBins = fMultipleMVAnames.begin();
     map<TString, std::pair<TString,TCut> >::iterator EndBins = fMultipleMVAnames.end();
     fMultipleMVAnames.erase(startBins, EndBins);

     // create new map
     for(Int_t oldbin=0; oldbin<nOldBins; oldbin++){
       for(Int_t bin=0; bin<(nbins); bin++){
 	 // prepare string for this bin
 	 // FIXME!!! assume at the moment that array is sorted!
 	 //    fMultipleMVAnames = new TMap(nbins);
	 
 	 // simple bin name
 	 TString *binMVAname = new TString( SimpleName[oldbin] + "__" + theVariable + 
 					    Form("_bin_%d",(bin+1)));
 	 // this is the cut in human readable version
 	 TString *binMVAdescription = new TString( Description[oldbin] + " && " 
						   + Form(" %g < ",array[bin])  
 						   + theVariable  
 						   + Form(" < %g",array[bin+1]));
         // create ROOT TCut
 	 TString *binMVAtmp = new TString("("+ theVariable +
 					  Form(" > %g ",array[bin]) +
 					  ") && (" +
 					  theVariable + Form(" < %g",array[bin+1]) +")");
	 
 	 TCut *binMVACut = new TCut(binMVAtmp->Data());
	 
	 
 	 // fill all three into the map
 	 fMultipleMVAnames[*binMVAname] =  std::pair<TString,TCut>(*binMVAdescription, *binMVACut + OldCut[oldbin]);

	 if (Verbose()) cout << "--- " <<  GetName() <<": "
			     <<binMVAname->Data()
			     <<"  ["<< binMVAdescription->Data() << "]  "<<endl;
	 delete binMVAname;
	 delete binMVAdescription;
	 delete binMVAtmp;
	 delete binMVACut;
       }// end of loop over bins
     } // end of loop over oldbins

   }else{ // this is the first time BookMultipleMVAs is being called
     for(Int_t bin=0; bin<(nbins); bin++){
       // prepare string for this bin
       // FIXME!!! assume at the moment that array is sorted!
       //    fMultipleMVAnames = new TMap(nbins);
       
       // simple bin name
       TString *binMVAname = new TString( theVariable + 
					  Form("_bin_%d",(bin+1)));
       // this is the cut in human readable version
       TString *binMVAdescription = new TString( Form("%g < ",array[bin]) + 
						 theVariable + 
						 Form(" < %g",array[bin+1]));
       // create ROOT TCut
       TString *binMVAtmp = new TString("("+ theVariable +
					Form(" > %g ",array[bin]) +
					") && (" +
					theVariable + Form(" < %g",array[bin+1]) +")");
       
       TCut *binMVACut = new TCut(binMVAtmp->Data());
       
       // fill all three into the map
       fMultipleMVAnames[*binMVAname] =  std::pair<TString,TCut>(*binMVAdescription, *binMVACut);
       if (Verbose()) cout << "--- " <<  GetName() <<": "
			   <<binMVAname->Data()
			   <<"  ["<< binMVAdescription->Data() << "]  "<<endl;
       delete binMVAname;
       delete binMVAdescription;
       delete binMVAtmp;
       delete binMVACut;
    }// end of loop over bins
  }// end of if (fMultipleMVAnames.size() >0)

}

//_______________________________________________________________________
void TMVA_Factory::PrepareTrainingAndTestTree( TCut cut, Int_t Ntrain, Int_t Ntest, TString TreeName )
{ 
//!   ------------------------------------------------------
//!   |              |              |        |             |
//!   ------------------------------------------------------
//!                                                        # input signal events
//!                                          # input signal events after cuts
//!   ------------------------------------------------------
//!   |              |              |             |       |
//!   ------------------------------------------------------
//!    \/  \/                      # input bg events
//!                                               # input bg events after cuts
//!      Ntrain/2       Ntest/2                         
//!
//! definitions:
//!
//!         nsigTot = all signal events
//!         nbkgTot = all bkg events
//!         nTot    = nsigTot + nbkgTot
//!         i.g.: nsigTot != nbkgTot
//!         N:M     = use M events after event N (distinct event sample)
//!                   (read as: "from event N to event M")
//!
//! assumptions:
//!
//! 	a) equal number of signal and background events is used for training
//! 	b) any numbers of signal and background events are used for testing
//! 	c) an explicit syntax can violate a)
//!
//! cases (in order of importance)
//!
//! 1)
//!      user gives         : N1
//!      PrepareTree does   : nsig_train=nbkg_train=min(N1,nsigTot,nbkgTot)
//!                           nsig_test =nsig_train:nsigTot, nbkg_test =nsig_train:nbkgTot
//!      -> give warning if nsig_test<=0 || nbkg_test<=0
//!
//! 2)
//!      user gives         : N1, N2
//!      PrepareTree does   : nsig_train=nbkg_train=min(N1,nsigTot,nbkgTot)
//!                           nsig_test =nsig_train:min(N2,nsigTot-nsig_train),
//!                           nbkg_test =nsig_train:min(N2,nbkgTot-nbkg_train)
//!      -> give warning if nsig(bkg)_train != N1, or
//!                      if nsig_test<N2 || nbkg_test<N2
//!
//! 3)
//!      user gives         : -1
//!      PrepareTree does   : nsig_train=nbkg_train=min(nsigTot,nbkgTot)
//!                           nsig_test =nsigTot, nbkg_test=nbkgTot
//!      -> give warning that same samples are used for testing and training
//!
//! 4)
//!      user gives         : -1, -1
//!      PrepareTree does   : nsig_train=nsigTot, nbkg_train=nbkgTot
//! 			  nsig_test =nsigTot, nbkg_test =nbkgTot
//!      -> give warning that same samples are used for testing and training,
//!         and, if nsig_train != nbkg_train, that an unequal number of 
//!         signal and background events are used in training
//!			  
//! ------------------------------------------------------------------------
//! Give in any case the number of signal and background events that are
//! used for testing and training, and tell whether there are overlaps between
//! the samples.
//! ------------------------------------------------------------------------
//! 
//! Addon (Jan 12, 2006) kai
//! if 
  fCut = cut;

  if (fMultipleMVAs && !fMultipleStoredOptions ){
    cout << "--- " << GetName() << ":  Store cut and numbers for multiple MVAs " << endl;
    fMultiCut    = cut;
    fMultiNtrain = Ntrain;
    fMultiNtest  = Ntest;
    return;
  }

  cout << "--- " << endl;
  cout << "--- " << GetName() << ": prepare training and Test samples" << endl;
  cout << "--- " << GetName() << ": num of events in input signal tree     : " 
       << fSignalTree->GetEntries() << endl;
  cout << "--- " << GetName() << ": num of events in input background tree : " 
       << fBackgTree->GetEntries() << endl;
  if (TString(fCut) != "") 
    cout << "--- " << GetName() << ": apply cut on input trees               : " 
	 << fCut << endl;
  else
    cout << "--- " << GetName() << ": no cuts applied" << endl;

  // apply cuts to the input trees and create TEventLists of only the events
  // we would like to use !
  fSignalTree->Draw(">>signalList",fCut,"goff");
  TEventList *signalList = (TEventList*)gDirectory->Get("signalList");

  fBackgTree->Draw(">>backgList",fCut,"goff");
  TEventList *backgList = (TEventList*)gDirectory->Get("backgList");

  if (TString(fCut) != "") {
    cout << "--- " << GetName() << ": num of signal events passing cut       : " 
	 << signalList->GetN() << endl;
    cout << "--- " << GetName() << ": num of background eventspassing cut    : " 
	 << backgList->GetN() << endl;
  }

  Int_t nsig_train(0), nbkg_train(0), nsig_test(0), nbkg_test(0);
  Int_t nsig_test_min(0), nbkg_test_min(0);
  Int_t nsigTot = signalList->GetN();  
  Int_t nbkgTot = backgList->GetN();
  //  Int_t nTot    = nsigTot + nbkgTot;
  Int_t array[3];
  array[1] = nsigTot;
  array[2] = nbkgTot;

  if (Ntrain >0 && Ntest == 0) {
    array[0]      = Ntrain;
    nsig_train    = TMath::MinElement(3,array);
    nbkg_train    = nsig_train;
    nsig_test_min = nsig_train;
    nbkg_test_min = nsig_train;
    
    if ((nsigTot-nsig_train)<=0 ) 
      cout << "--- " << GetName() 
	   << ": WARNING  # signal events for testing <= 0! " << endl;
    if ((nbkgTot-nbkg_test)<=0) 
      cout << "--- " << GetName() 
	   << ": WARNING # background events for testing <= 0! " << endl;
    nsig_test = nsigTot;
    nbkg_test = nbkgTot;
    
  } 
  else if (Ntrain >0 && Ntest > 0) {
    array[0]      = Ntrain;
    nsig_train    = TMath::MinElement(3,array);
    nbkg_train    = nsig_train;
    nsig_test_min = nsig_train;
    nbkg_test_min = nsig_train;

    nsig_test     = TMath::Min(Ntest,nsigTot-nsig_train);
    nbkg_test     = TMath::Min(Ntest,nbkgTot-nsig_train);
    if (nsig_train != Ntrain) 
      cout << "--- " << GetName() 
	   << ": WARNING  less events for training than requested!" << endl;
      if (nsig_test<Ntest || nbkg_test<Ntest)
	cout << "--- " << GetName() 
	     << ": WARNING  less events for testing than requested!" << endl;
      nsig_test += nsig_train;
      nbkg_test += nsig_train;

      
  } 

  else if (Ntrain == -1 && Ntest == 0) {
    nsig_train = TMath::Min(nsigTot,nbkgTot);
    nbkg_train = nsig_train;
    nsig_test  = nsigTot;
    nbkg_test  = nbkgTot;
    nsig_test_min=1;
    nbkg_test_min=1;
    cout << "--- " << GetName() 
	 << ": WARNING! Same samples are used for training and testing" << endl;
    
  }
  else if (Ntrain == -1 && Ntest == -1) {
    nsig_train = nsigTot;
    nbkg_train = nbkgTot;
    nbkg_test  = nbkgTot;
    nsig_test  = nsigTot;
    nsig_test_min=1;
    nbkg_test_min=1;
    cout << "--- " << GetName() 
	 << ": WARNING! Same samples are used for training and testing" << endl;
    cout << "--- " << GetName() 
	 << ": WARNING! An unequal number of signal and background events are used in training" 
	 <<endl;
    
  }


  //! Sanity check (introduced Jan 12, 2006) by Andreas and Kai
  //! idea: You always want more events for testing than training
  if ( (nsig_train + nbkg_train ) > ((nsig_test-nsig_test_min) +  (nbkg_test-nbkg_test_min)) )
    {
      cout << "--- " << GetName()  << ": WARNING selected less events for training than fo testing"<<endl;
      cout << "--- " << GetName()  << ":         will split samples in halfs"<<endl;
      if (nsigTot < nbkgTot){
      nsig_train=nsigTot/2;
      nsig_test_min=nsigTot/2;
      nbkg_train=nsigTot/2;
      nbkg_test_min=nsigTot/2;
      }else{
      nbkg_train=nbkgTot/2;
      nbkg_test_min=nbkgTot/2;
      nsig_train=nbkgTot/2;
      nsig_test_min=nbkgTot/2;
      }
    }
  

  // provide detailed output
  cout << "--- " << GetName() << ": num of training signal events          : 1..." 
       << nsig_train << endl;  
  cout << "--- " << GetName() << ": num of training background events      : 1..." 
       << nbkg_train << endl;  
  cout << "--- " << GetName() << ": num of testing  signal events          : " 
       << nsig_test_min << "..." << nsig_test << endl;
  cout << "--- " << GetName() << ": num of testing  background events      : " 
       << nbkg_test_min << "..." << nbkg_test << endl;
  
  // create new trees
  // variable "type" is used to destinguish "0" = background;  "1" = Signal
  Int_t type; 
  fTrainingTree = new TTree("TrainingTree", "Variables used for MVA training");
  fTrainingTree->Branch( "type", &type, "type/I" , basketsize );

  if (TreeName.Sizeof() >1) TreeName.Prepend("_");
  fTestTree = new TTree("TestTree"+TreeName, "Variables used for MVA testing, and MVA outputs" );
  fTestTree->Branch( "type", &type, "type/I", basketsize );

  Int_t nvars = fInputVariables->size();
  //Float_t v[nvars];
  Float_t v[1000];  //please check
  for (Int_t ivar=0; ivar<nvars; ivar++) {
    
    // Add Branch to training/test Tree
    TString myVar = (*fInputVariables)[ivar];
    fTrainingTree->Branch( myVar, &v[ivar], myVar + "/F", basketsize );
    fTestTree->Branch( myVar, &v[ivar], myVar + "/F", basketsize );
  } // end of loop over input variables

  // loop over signal events first
  type = 1;
  Int_t ac=0;
  for (Int_t i = 0; i < fSignalTree->GetEntries(); i++) {
    if (signalList->Contains(i)) { // survives the cut
      for (Int_t ivar=0; ivar<nvars; ivar++) 
	v[ivar] = (Float_t)TMVA_Tools::GetValue( fSignalTree, i, (*fInputVariables)[ivar] );

      ac++;
      if ( ac <= nsig_train)                         fTrainingTree->Fill();
      if ((ac > nsig_test_min) && (ac <= nsig_test)) fTestTree->Fill();      
    }
  }

  ac=0;
  // now loop over backgound events 
  type = 0;
  for (Int_t i = 0; i < fBackgTree->GetEntries(); i++) {
    if(backgList->Contains(i)){
      for (UInt_t ivar=0; ivar<fInputVariables->size(); ivar++) 
	v[ivar] = (Float_t)TMVA_Tools::GetValue( fBackgTree, i, (*fInputVariables)[ivar] );

      ac++;
      if ( ac <= nbkg_train)                         fTrainingTree->Fill();
      if ((ac > nbkg_test_min) && (ac <= nbkg_test)) fTestTree    ->Fill();
    }
  }

  if (DEBUG_TMVA_Factory) {
    fTestTree->Print();
    fTrainingTree->Print();
    
    fTrainingTree->Show(12);
    fTrainingTree->Show(41);
  }
  if (Verbose()) cout << "--- " << GetName() << " <verbose>: tree preparation finished" << endl;

  // designed plotting output
  if (fTrainingTree->GetEntries() > 0)  PlotVariables( fTrainingTree );

  // first thing: get overall correlation matrix between all variables in tree  
  if (fTrainingTree->GetEntries() > 0)  GetCorrelationMatrix( fTrainingTree );


}

//_______________________________________________________________________
void TMVA_Factory::PlotVariables( TTree* theTree )
{
  Int_t nbins       = 100;
  Float_t timesRMS  = 3.0;

  // create plots of the input variables and check them
  if (Verbose()) cout << "--- " << GetName() << " <verbose>: plot input variables from '"
		      <<theTree->GetName();

  // create directory in output file
  TDirectory *localDir= fLocalTDir->mkdir("input_variables" );
  localDir->cd();
  if (Verbose()) cout<<"' into dir: "<<localDir->GetPath()<<endl;
  TH1F *myhist = new TH1F();
  vector<TString>::iterator   itrVar    = fInputVariables->begin();
  vector<TString>::iterator   itrVarEnd = fInputVariables->end();
  cout << "--- " << endl;
  for(; itrVar != itrVarEnd; itrVar++) {
     TString myVar = *itrVar;  

     // Find out mean and rms of signal and background distributions.
     TString drawOpt = myVar + ">>h";
     theTree->Draw( drawOpt, "type  == 1", "goff" );
     myhist= (TH1F*)gDirectory->Get("h");
     Float_t rmsS  = myhist->GetRMS();
     Float_t meanS = myhist->GetMean();
     theTree->Draw( drawOpt, "type == 0","goff" );
     myhist= (TH1F*)gDirectory->Get("h");
     Float_t rmsB  = myhist->GetRMS();
     Float_t meanB = myhist->GetMean();

     // choose reasonable histogram ranges, by removing outliers
     Float_t xmin = TMath::Max( Float_t( myhist->GetXaxis()->GetXmin() ), 
				TMath::Min( meanS - timesRMS*rmsS, meanB - timesRMS*rmsB ) );
     Float_t xmax = TMath::Min( Float_t( myhist->GetXaxis()->GetXmax() ), 
				TMath::Max( meanS + timesRMS*rmsS, meanB + timesRMS*rmsB ) );

     //
     // ----- signal distribution
     //
     TString histTitle =  myVar + " signal";
     TString histName  =  myVar + "__S";
     drawOpt= myVar+">>h("+TString(nbins);
     TString draw = Form( ">>h(%d,%f,%f)", nbins, xmin, xmax );
     printf("--- %s: create histogram '%s' within range [%0.3g, %0.3g]\n",
	    (const char*)GetName(), (const char*)histName, xmin, xmax );
     drawOpt= myVar + draw;
     theTree->Draw(drawOpt,"type  == 1", "goff");
     myhist= (TH1F*)gDirectory->Get("h");
     myhist->SetName(histName);
     myhist->SetTitle(histTitle);
     myhist->SetXTitle(myVar);
     myhist->SetLineColor(4);

     // check signal histo for outliers!
     Float_t origEntries=myhist->GetEntries();
     Int_t emptyBins=0;
     // count number of empty bins
     for(Int_t bin=1; bin<=myhist->GetNbinsX(); bin++){
       if (myhist->GetBinContent(bin) == 1) emptyBins++;
     }
     if (((Float_t)emptyBins/(Float_t)myhist->GetNbinsX()) > 0.75) {
       cout << " | More than 75% of the bins in hist '"
	    << myhist->GetName() << "' are empty!" << endl;
       cout << " | check plot " << myhist->GetName() << " in output file " << endl;

       // create additional cut to remove outliers
       TString newCutStr = "( "+ myVar 
	 + Form(" > %0.3g ) && (", (myhist->GetMean()-timesRMS*(myhist->GetRMS())))
	 + myVar  
	 + Form(" < %0.3g )", (myhist->GetMean()+timesRMS*(myhist->GetRMS())));

       cout << " | suggested cut to remove outliers: " << newCutStr << endl;

       TCut newCut(newCutStr); 
       newCut += "type  == 1";
       drawOpt.ReplaceAll(">>h",">>g");
       theTree->Draw(drawOpt,newCut, "goff");
       TH1F *myNewhist= (TH1F*)gDirectory->Get("g");
       Float_t removed=origEntries-(myNewhist->GetEntries());
       cout << " | this cut would remove " << removed << " out of "
	    << origEntries << " signal events" << endl;
       Float_t left=0;
       Float_t right=0;
       for(Int_t b=1; b<=myhist->GetNbinsX(); b++){
	 if (b <  myhist->FindBin((myhist->GetRMS()))){
	   left += ((myhist->GetBinContent(b)) - (myNewhist->GetBinContent(b)));
	 }else{
	   right += myhist->GetBinContent(b)-myNewhist->GetBinContent(b);
	 }
       }
       cout << " | "<<left <<" on the low side and "<<
	 right<<" on the high side of the mean"<<endl;
       delete myNewhist;
     }

     TMVA_Tools::NormHist( myhist ); // normalize
     myhist->Write();


     //
     // ----- background distribution
     //
     histTitle = myVar+" background";
     histName  = myVar +"__B";
     draw      = Form( ">>h(%d,%f,%f)",nbins,xmin,xmax );
     drawOpt   = myVar + draw;
     theTree->Draw(drawOpt,"type == 0", "goff");
     myhist = (TH1F*)gDirectory->Get("h");
     TMVA_Tools::NormHist( myhist ); // normalize
     myhist->SetName(histName);
     myhist->SetTitle(histTitle);
     myhist->SetXTitle(myVar);
     myhist->SetLineColor(2);

     myhist->Write();
  }
  delete myhist;

  this->SetLocalDir();
}

//_______________________________________________________________________
void TMVA_Factory::GetCorrelationMatrix( TTree* theTree )
{ 
      
  // first remove type from variable set
  if (Verbose())
    cout << "--- " << GetName() << " <verbose>: retrieve correlation matrix using tree: " 
	 << theTree->GetName() << endl;

  TBranch*         branch = 0;
  vector<TString>* theVars = new vector<TString>;
  TObjArrayIter branchIter( theTree->GetListOfBranches(), kIterForward );
  while ((branch = (TBranch*)branchIter.Next()) != 0) 
    if ((TString)branch->GetName() != "type") theVars->push_back( branch->GetName() );

  Int_t nvar = (int)theVars->size();
  TMatrixD *corrMatS = new TMatrixD( nvar, nvar );
  TMatrixD *corrMatB = new TMatrixD( nvar, nvar );

  // now compute the matrix
  TMVA_Tools::GetCorrelationMatrix( theTree, corrMatS, theVars, 1 );
  TMVA_Tools::GetCorrelationMatrix( theTree, corrMatB, theVars, 0 );

  // print the matrix
  const Int_t prec = 9;
  cout << "--- " << endl;
  cout << "-------------------------------------------------------------------" << endl;
  cout << "--- " << GetName() << ": correlation matrix (signal):" << endl;
  cout << "-------------------------------------------------------------------" << endl;
  cout << "--- " << "           ";
  for (Int_t ivar=0; ivar<nvar; ivar++) cout << setw(prec) << (*theVars)[ivar];
  cout << endl;
  for (Int_t ivar=0; ivar<nvar; ivar++) {
    cout << "--- " << setw(10) << (*theVars)[ivar] << ":";
    for (Int_t jvar=0; jvar<nvar; jvar++) printf("   %+1.3f",(*corrMatS)(ivar, jvar));    
    cout << endl;
  }
  cout << "--- " << endl;
  cout << "-------------------------------------------------------------------" << endl;
  cout << "--- " << GetName() << ": correlation matrix (background):" << endl;
  cout << "-------------------------------------------------------------------" << endl;
  cout << "--- " << "           ";
  for (Int_t ivar=0; ivar<nvar; ivar++) cout << setw(prec) << (*theVars)[ivar];
  cout << endl;
  for (Int_t ivar=0; ivar<nvar; ivar++) {
    cout << "--- " << setw(10) << (*theVars)[ivar] << ":";
    for (Int_t jvar=0; jvar<nvar; jvar++) printf("   %+1.3f",(*corrMatB)(ivar, jvar));    
    cout << endl;
  }
  cout << "-------------------------------------------------------------------" << endl;
  cout << "--- " << endl;

  // ---- histogramming
  this->SetLocalDir();

  // loop over signal and background
  TString      hName[2]  = { "CorrelationMatrixS", "CorrelationMatrixB" };
  TString      hTitle[2] = { "Correlation Matrix (signal)", "Correlation Matrix (background)" };

  // workaround till the TMatrix templates are comonly used
  // this keeps backward compatibility
  TMatrixF*    tmS = new TMatrixF( nvar, nvar );
  TMatrixF*    tmB = new TMatrixF( nvar, nvar );
  for (Int_t ivar=0; ivar<nvar; ivar++) {
    for (Int_t jvar=0; jvar<nvar; jvar++) {
      (*tmS)(ivar, jvar) = (*corrMatS)(ivar,jvar);
      (*tmB)(ivar, jvar) = (*corrMatB)(ivar,jvar);
    }
  }  

  TMatrixF *mObj[2]  = { tmS, tmB };

  // settings
  const Float_t labelSize = 0.055;

  for (Int_t ic=0; ic<2; ic++) { 

    TH2F* h2 = new TH2F( *(mObj[ic]) );
    h2->SetNameTitle( hName[ic], hTitle[ic] );

    for (Int_t ivar=0; ivar<nvar; ivar++) {
      h2->GetXaxis()->SetBinLabel( ivar+1, (*theVars)[ivar] );
      h2->GetYaxis()->SetBinLabel( ivar+1, (*theVars)[ivar] );
    }

    // present in percent, and round off digits
    // also, use absolute value of correlation coefficient (ignore sign)
    h2->Scale( 100.0  ); 
    for (Int_t ibin=1; ibin<=nvar; ibin++)
      for (Int_t jbin=1; jbin<=nvar; jbin++)
	h2->SetBinContent( ibin, jbin, Int_t(TMath::Abs(h2->GetBinContent( ibin, jbin ))) );

    // style settings
    h2->SetStats( 0 );
    h2->GetXaxis()->SetLabelSize( labelSize );
    h2->GetYaxis()->SetLabelSize( labelSize );
    h2->SetMarkerSize( 1.5 );
    h2->SetMarkerColor( 0 );
    h2->LabelsOption( "d" ); // diagonal labels on x axis
    h2->SetLabelOffset( 0.011 );// label offset on x axis
    h2->SetMinimum(    0.0 );
    h2->SetMaximum( +100.0 );

    // -------------------------------------------------------------------------------------
    // just in case one wants to change the position of the color palette axis
    // -------------------------------------------------------------------------------------
    //     gROOT->SetStyle("Plain");
    //     TStyle* gStyle = gROOT->GetStyle( "Plain" );
    //     gStyle->SetPalette( 1, 0 );
    //     TPaletteAxis* paletteAxis 
    //                   = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject( "palette" );
    // -------------------------------------------------------------------------------------

    // write to file
    h2->Write();
    if (Verbose())
      cout << "--- " << GetName() << " <verbose>: created correlation matrix as 2D histogram: " 
	   << h2->GetName() << endl;

    delete h2;
  }
  // ----  

  delete theVars;
  delete corrMatS;
  delete corrMatB;
}


//_______________________________________________________________________
void TMVA_Factory::SetSignalTree(TTree* signal)
{
  fSignalTree=signal;
}

//_______________________________________________________________________
void TMVA_Factory::SetBackgroundTree(TTree* background)
{
  fBackgTree=background;
}

//_______________________________________________________________________
void TMVA_Factory::SetTestTree(TTree* testTree)
{
  fTestTree = testTree;
}


//_______________________________________________________________________
Bool_t TMVA_Factory::BookMethod( TString theMethodName, TString theOption, 
				 TString theNameAppendix ) 
{
  if (fMultipleMVAs && !fMultipleStoredOptions ){
    cout << "--- " << GetName() << ":  Store "<<theMethodName+theNameAppendix
	 <<"  and its options for multiple MVAs " << endl;
    
    fMultipleMVAMethodOptions[theMethodName+theNameAppendix] = 
      std::pair<TString,TString>(theOption, theNameAppendix);
    return kTRUE;
  }

  if (theMethodName != "Variable") 
    cout << "--- " << GetName() << ": create method: " << theMethodName << endl;

  if (theMethodName.Contains("Cuts")) 
    return BookMethod( TMVA_Types::Cuts, theOption, theNameAppendix );
  else if (theMethodName.Contains("Fisher")) 
    return BookMethod( TMVA_Types::Fisher, theOption, theNameAppendix );
  else if (theMethodName.Contains("TMlpANN")) 
    return BookMethod( TMVA_Types::TMlpANN, theOption, theNameAppendix );
  else if (theMethodName.Contains("CFMlpANN")) 
    return BookMethod( TMVA_Types::CFMlpANN, theOption, theNameAppendix );
  else if (theMethodName.Contains("Likelihood")) 
    return BookMethod( TMVA_Types::Likelihood, theOption, theNameAppendix );
  else if (theMethodName.Contains("Variable")) 
    return BookMethod( TMVA_Types::Variable, theOption, theNameAppendix );
  else if (theMethodName.Contains("HMatrix")) 
    return BookMethod( TMVA_Types::HMatrix, theOption, theNameAppendix );
  else if (theMethodName.Contains("PDERS")) 
    return BookMethod( TMVA_Types::PDERS, theOption, theNameAppendix );
  else if (theMethodName.Contains("BDT")) 
    return BookMethod( TMVA_Types::BDT, theOption, theNameAppendix );
  else {
    cout << "--- " << GetName() << ": Error: method: " 
	 << theMethodName << " does not exist ==> abort" << endl;
    exit(1);
  }

  return kFALSE;
}

//_______________________________________________________________________
Bool_t TMVA_Factory::BookMethod( TMVA_Types::MVA theMethod, TString theOption, 
				 TString theNameAppendix ) 
{
  TMVA_MethodBase *method = 0;

   // initialize methods
  if (theMethod == TMVA_Types::Cuts)
    method = new TMVA_MethodCuts      ( fJobName, 
					fInputVariables,
					fTrainingTree,
					theOption, 
					fLocalTDir );

  else if (theMethod == TMVA_Types::Fisher)
    method = new TMVA_MethodFisher    ( fJobName, 
					fInputVariables,
					fTrainingTree, 
					theOption,
					fLocalTDir );

  else if (theMethod == TMVA_Types::TMlpANN) {
    method = new TMVA_MethodTMlpANN   ( fJobName,
					fInputVariables, 
					fTrainingTree, 
					theOption,
					fLocalTDir );    
  
    // special "feature" of TMlpANN: needs also test tree :-(
    TMVA_MethodTMlpANN *tmp = (TMVA_MethodTMlpANN*)method;
    tmp->SetTestTree(fTestTree);    
  }
  
  else if (theMethod == TMVA_Types::CFMlpANN)
    method = new TMVA_MethodCFMlpANN  ( fJobName,  
					fInputVariables, 
					fTrainingTree, 
					theOption,
					fLocalTDir );    
  
  else if (theMethod == TMVA_Types::Likelihood) 
    method = new TMVA_MethodLikelihood( fJobName,  
					fInputVariables, 
					fTrainingTree, 
					theOption, 
					fLocalTDir );    

  else if (theMethod == TMVA_Types::Variable)
    method = new TMVA_MethodVariable  ( fJobName,  
					fInputVariables, 
					fTrainingTree, 
					theOption, 
					fLocalTDir );    

  else if (theMethod == TMVA_Types::HMatrix)
    method = new TMVA_MethodHMatrix   ( fJobName, 
					fInputVariables,
					fTrainingTree, 
					theOption,
					fLocalTDir );

  else if (theMethod == TMVA_Types::PDERS)
    method = new TMVA_MethodPDERS     ( fJobName, 
					fInputVariables,
					fTrainingTree, 
					theOption,
					fLocalTDir );
  
  else if (theMethod == TMVA_Types::BDT)
    method = new TMVA_MethodBDT       ( fJobName, 
					fInputVariables,
					fTrainingTree, 
					theOption,
					fLocalTDir );

  else {
    cout << "--- " << GetName() << ": Error: method: " 
	 << theMethod << " does not exist ==> abort" << endl;
    exit(1);
  }

  if (0 != method) {
    if (theNameAppendix.Sizeof() > 1 )method->AppendToMethodName( theNameAppendix );
    fMethods.push_back( method );
  }

  return (0 != method) ? kTRUE : kFALSE;
}

//_______________________________________________________________________
Bool_t TMVA_Factory::BookMethod( TMVA_MethodBase *theMethod,
				 TString theNameAppendix )
{

  if (NULL != theMethod) {
    if (theNameAppendix.Sizeof() > 1 ){
      theMethod->AppendToMethodName( theNameAppendix );
    }
    fMethods.push_back( theMethod );
  }
  return (0 != theMethod) ? kTRUE : kFALSE;
}

//_______________________________________________________________________
TMVA_MethodBase* TMVA_Factory::GetMVA( TString method )
{
  vector<TMVA_MethodBase*>::iterator itrMethod    = fMethods.begin();
  vector<TMVA_MethodBase*>::iterator itrMethodEnd = fMethods.end();
  for(; itrMethod != itrMethodEnd; itrMethod++) {
    TMVA_MethodBase* MVA = (*itrMethod);    
    if ( (MVA->GetMethodName()).Contains(method)) return MVA;
  }
  return 0;
}

//_______________________________________________________________________
void TMVA_Factory::TrainAllMethods( void ) 
{  

  // if multiple  MVAs 
  if (fMultipleMVAs && !fMultipleStoredOptions ){
    cout << "--- " << GetName() << ": TrainAllMethods will be called for multiple MVAs " << endl;
    fMultiTrain=kTRUE;
    return;
  }
  
  // iterate over methods and train
  vector<TMVA_MethodBase*>::iterator itrMethod    = fMethods.begin();
  vector<TMVA_MethodBase*>::iterator itrMethodEnd = fMethods.end();
  for(; itrMethod != itrMethodEnd; itrMethod++) {
    if (fTrainingTree->GetEntries() > MinNoTrainingEvents){
      cout << "--- " << GetName() << ": train method: " 
	   << ((TMVA_MethodBase*)*itrMethod)->GetMethodName() << endl;
      (*itrMethod)->Train();
    }
    else{
      cout << "--- " << GetName() 
	   << ": WARNING method "<< ((TMVA_MethodBase*)*itrMethod)->GetMethodName() 
	   << " not trained (training tree has no entries)"<<endl; 
    }
  }
}

//_______________________________________________________________________
void TMVA_Factory::TestAllMethods( void )
{

  // if multiple  MVAs 
  if (fMultipleMVAs && !fMultipleStoredOptions ) {
    cout << "--- " << GetName() << ": TestAllMethods will be called for multiple MVAs " << endl;
    fMultiTest=kTRUE;
    return;
  } else if (fTrainingTree == NULL) {
    cout << "--- "<< GetName() 
	 << " you perform testing without training before, hope you  \n"
	 << "--- did give a reasonable test tree and weight files " <<endl;
  } else if ((fTrainingTree->GetEntries() < MinNoTrainingEvents) && fMultipleMVAs && fMultiTrain) {
    cout << "--- "<< GetName() 
	 <<" : WARNING Skip testing since training wasn't performed for this bin"<<endl;
    return;
  }

  // iterate over methods and test
  vector<TMVA_MethodBase*>::iterator itrMethod    = fMethods.begin();
  vector<TMVA_MethodBase*>::iterator itrMethodEnd = fMethods.end();
  for(; itrMethod != itrMethodEnd; itrMethod++) {
    cout << "--- " << GetName() << ": test method: " 
	 << ((TMVA_MethodBase*)*itrMethod)->GetMethodName() << endl;
    (*itrMethod)->PrepareEvaluationTree( fTestTree );
    if (DEBUG_TMVA_Factory) fTestTree->Print();
  }
}

//_______________________________________________________________________
void TMVA_Factory::EvaluateAllVariables( TString options )
{

  // if multiple  MVAs 
  if (fMultipleMVAs && !fMultipleStoredOptions ){
    cout << "--- " << GetName() << ": EvaluateAllVariables will be called for multiple MVAs " << endl;
    fMultiEvalVar=kTRUE;
    return;
  } else if (fTrainingTree == NULL) {
    cout << "--- "<< GetName() 
	 << " you perform testing without training before, hope you  \n"
	 << "--- did give a reasonable test tree and weight files " <<endl;
  }else if ((fTrainingTree->GetEntries() < MinNoTrainingEvents) && fMultipleMVAs && fMultiTrain){
    cout << "--- "<< GetName() 
	 <<" : WARNING Skip evaluation since training wasn't performed for this bin"<<endl;
    return;
  }


  if (Verbose())
    cout << "--- " << GetName() 
	 << " <verbose>: for this each variable needs to be booked as a Method" << endl;
  // iterate over variables and evaluate
  vector<TString>::iterator itrVars    = fInputVariables->begin();
  vector<TString>::iterator itrVarsEnd = fInputVariables->end();
  for (; itrVars != itrVarsEnd; itrVars++) {
    TString s = *itrVars;
    if (options.Contains("V")) s += ":V";
    this->BookMethod( "Variable", s );
  }
}

//_______________________________________________________________________
void TMVA_Factory::EvaluateAllMethods( void )
{
  // if multiple  MVAs 
  if (fMultipleMVAs && !fMultipleStoredOptions ){
    cout << "--- " << GetName() << ": EvaluateAllMethods will be called for multiple MVAs " << endl;
    fMultiEval=kTRUE;
    return;
  } else if (fTrainingTree == NULL) {
    cout << "--- "<< GetName() 
	 << " you perform testing without training before, hope you  \n"
	 << "--- did give a reasonable test tree and weight files " <<endl;
  }else if ((fTrainingTree->GetEntries() < MinNoTrainingEvents) && fMultipleMVAs && fMultiTrain){
    cout << "--- "<< GetName() <<" : WARNING Skip evaluation since training wasn't performed"<<endl;
    return;
  }

  // although equal, we now want to seperate the outpuf for the variables
  // and the real methods
  Int_t    isel; //will be 0 for a Method; 1 for a Variable
  Int_t nmeth_used[2] = {0,0}; //0 Method; 1 Variable

  vector< vector<TString> > mname(2);
  vector< vector<Double_t> > sig(2),sep(2),eff01(2),eff10(2),eff30(2),mutr(2);

  // iterate over methods and evaluate
  vector<TMVA_MethodBase*>::iterator itrMethod    = fMethods.begin();
  vector<TMVA_MethodBase*>::iterator itrMethodEnd = fMethods.end();
  for (; itrMethod != itrMethodEnd; itrMethod++) {
    cout << "--- " << GetName() << ": evaluate method: " 
	 << (*itrMethod)->GetMethodName() << endl;
    isel=0; if ((*itrMethod)->GetMethodName().Contains("Variable")) isel=1;

    // perform the evaluation
    (*itrMethod)->TestInit(fTestTree);
    // do the job
    if ((*itrMethod)->isOK()) (*itrMethod)->Test(fTestTree);
    if ((*itrMethod)->isOK()) {
      mname[isel].push_back( (*itrMethod)->GetMethodName() );  
      sig[isel].push_back  ( (*itrMethod)->GetSignificance() );
      sep[isel].push_back  ( (*itrMethod)->GetSeparation() );
      eff01[isel].push_back( (*itrMethod)->GetEfficiency("Efficiency:0.01", fTestTree)  );
      eff10[isel].push_back( (*itrMethod)->GetEfficiency("Efficiency:0.10", fTestTree)  );
      eff30[isel].push_back( (*itrMethod)->GetEfficiency("Efficiency:0.30", fTestTree)  );
      mutr[isel].push_back ( (*itrMethod)->GetmuTransform(fTestTree) );
      nmeth_used[isel]++;
      (*itrMethod)->WriteHistosToFile( fLocalTDir );
    }
    else {
      cout << "--- " << GetName() << ": Warning: " << (*itrMethod)->GetName() 
	   << " returned isOK flag: " 
	   << (*itrMethod)->isOK() << endl;
    }
  }

  // now sort the variables according to the best 'eff at Beff=0.10'
  for (Int_t k=0; k<2; k++) {
    vector< vector<Double_t> > vtemp;
    vtemp.push_back( eff10[k] ); // this is the vector that is ranked
    vtemp.push_back( eff01[k] );
    vtemp.push_back( eff30[k] );
    vtemp.push_back( sig[k]   );
    vtemp.push_back( sep[k]   );
    vtemp.push_back( mutr[k]  ); 
    vector<TString> vtemps = mname[k];
    TMVA_Tools::UsefulSortDescending( vtemp, &vtemps );
    eff10[k] = vtemp[0];
    eff01[k] = vtemp[1];
    eff30[k] = vtemp[2];
    sig[k]   = vtemp[3];
    sep[k]   = vtemp[4];
    mutr[k]  = vtemp[5];
    mname[k] = vtemps;
  }

  cout << "--- " << endl;
  cout << "--- " << GetName() << ": Evaluation results ranked by best 'signal eff @B=0.10'" << endl;
  cout << "---------------------------------------------------------------------------"   << endl;
  cout << "--- MVA              Signal efficiency:         Signifi- Sepa-    mu-Trans-"   << endl;
  cout << "--- Methods:         @B=0.01  @B=0.10  @B=0.30  cance:   ration:  form:"       << endl;
  cout << "---------------------------------------------------------------------------"   << endl;
  for(Int_t k=0; k<2; k++){
    if (k == 1 && nmeth_used[k] > 0 && !fMultipleMVAs) {
      cout << "---------------------------------------------------------------------------" << endl;
      cout << "--- Input Variables: " << endl
	   << "---------------------------------------------------------------------------" << endl;
    }
    for(Int_t i=0; i<nmeth_used[k]; i++) {
      if (k == 1) mname[k][i].ReplaceAll( "Variable_", "" );
      printf("--- %-15s: %1.3f    %1.3f    %1.3f    %1.3f    %1.3f    %1.3f \n",
 	     (const char*)mname[k][i], 
	     eff01[k][i], eff10[k][i], eff30[k][i], sig[k][i], sep[k][i], mutr[k][i] );
    }
  }
  cout << "---------------------------------------------------------------------------" << endl;
  cout << "--- " << endl;
  
  cout << "--- " << GetName() << ": Write Test Tree '"<< fTestTree->GetName()<<"' to file" << endl;
  this->SetLocalDir();
  fTestTree->Write();
}

//_______________________________________________________________________
void TMVA_Factory::ProcessMultipleMVA( void )
{
  Double_t vd[100];
  Int_t vi[100];
  Float_t vf[100];
  Int_t count_vd=0;
  Int_t count_vi=0;
  Int_t count_vf=0;

  if( fMultipleMVAs ){
    // assume that we have booked all method:
    // all other methods know that they are called from this method!
    fMultipleStoredOptions=kTRUE;

    // loop over bins:
    for (map<TString, std::pair<TString,TCut> >::iterator bin = fMultipleMVAnames.begin();
	 bin != fMultipleMVAnames.end(); bin++) {


      cout << "---------------------------------------------------------------------------"   << endl;  
      cout << "--- " << GetName() << ": Process Bin "<< bin->first<< endl;
      cout << "---                 with cut ["<< (bin->second).first <<"]"<< endl;
      cout << "---------------------------------------------------------------------------"   << endl;      

      TString binName( "multicutTMVA_" +  bin->first );
      fLocalTDir = fTargetFile->mkdir( binName, (bin->second).first);    
      fLocalTDir->cd();


      // prepare trees for this bin
      this->PrepareTrainingAndTestTree( ((bin->second).second)+fMultiCut, fMultiNtrain, fMultiNtest );

      // reset list of methods: 
      if (Verbose()) cout << "--- " << GetName() << " <verbose>: delete previous methods" << endl;
      this->DeleteAllMethods();

      // loop over stored methods
      for (map<TString, std::pair<TString,TString> >::iterator method = fMultipleMVAMethodOptions.begin();
 	   method != fMultipleMVAMethodOptions.end(); method++) {
	
	// book methods
	this->BookMethod(method->first, (method->second).first, (method->second).second ) ;
      } // end of loop over methods

      // set weigt file dir: SetWeightFileDir
      // iterate over methods and test      
      vector<TMVA_MethodBase*>::iterator itrMethod2    = fMethods.begin();
      vector<TMVA_MethodBase*>::iterator itrMethod2End = fMethods.end();
      for(; itrMethod2 != itrMethod2End; itrMethod2++) {
 	TString binDir( "weights/" + bin->first );
	(*itrMethod2)->SetWeightFileDir(binDir);
      }
      
      if (Verbose()) 
	cout << "--- " << GetName() << " <verbose>: booked " << fMethods.size() << " methods" << endl;
      
      if (fMultiTrain)  this->TrainAllMethods();
      if (fMultiTest)   this->TestAllMethods();
      if (fMultiEval)   {
	this->EvaluateAllMethods();
	
	//check if fTestTree contains MVA variables
	Bool_t hasMVA=kFALSE;
	TIter next_branch1( fTestTree->GetListOfBranches() );
	while (TBranch *branch = (TBranch*)next_branch1()){
	  if (((TString)branch->GetName()).Contains("TMVA_")) hasMVA=kTRUE;
	} // end of loop over fTestTree branches
	
	if (hasMVA){
	  if (fMultiCutTestTree == NULL){
	    fMultiCutTestTree = new TTree("MultiCutTree","Combined Test Tree for all bins");
	    TIter next_branch1( fTestTree->GetListOfBranches() );
	    
	    while (TBranch *branch = (TBranch*)next_branch1()){
	      TLeaf *leaf = branch->GetLeaf(branch->GetName());
	      if (((TString)leaf->GetTypeName()).Contains("Double_t"))      	
		fMultiCutTestTree->Branch( (TString)leaf->GetName(), 
					  &vd[++count_vd], 
					    (TString)leaf->GetName() + "/D", basketsize );
	      
	      if (((TString)leaf->GetTypeName()).Contains("Int_t"))      	
		fMultiCutTestTree->Branch( (TString)leaf->GetName(),
					    &vi[++count_vi], 
					    (TString)leaf->GetName() + "/I", basketsize );
	      
	      if (((TString)leaf->GetTypeName()).Contains("Float_t"))      	
		fMultiCutTestTree->Branch( (TString)leaf->GetName(), 
					  &vf[++count_vf], 
					    (TString)leaf->GetName() + "/F", basketsize );
	      
	    } // loop over branches in fTestTree
	  }
	  // loop over fTestTree and fill into MultiCutTestTree
	  for (Int_t ievt=0;ievt<fTestTree->GetEntries(); ievt++) {
	    count_vd=0;	  count_vi=0;	  count_vf=0;
	    TIter next_branch1( fTestTree->GetListOfBranches() );
	    while (TBranch *branch = (TBranch*)next_branch1()){
	      TLeaf *leaf = branch->GetLeaf(branch->GetName());
	    if (((TString)leaf->GetTypeName()).Contains("Double_t"))      	
	      vd[++count_vd]=TMVA_Tools::GetValue( fTestTree, ievt, leaf->GetName());
	    
	    if (((TString)leaf->GetTypeName()).Contains("Int_t"))      	
	      vi[++count_vi]=(Int_t)TMVA_Tools::GetValue( fTestTree, ievt, leaf->GetName());
	    
	    if (((TString)leaf->GetTypeName()).Contains("Float_t"))      	
	      vf[++count_vf]=(Float_t)TMVA_Tools::GetValue( fTestTree, ievt, leaf->GetName());
	    } // loop over branches
	    fMultiCutTestTree->Fill();
	  } // end of loop over fTestTree

	} //end of if(fTesttree has MVA branches
      }// end of if (fMultiEval) 
    } // end loop over bins
    // write global tree to top directory of the file
    fTargetFile->cd();
    if (fMultiCutTestTree != NULL) fMultiCutTestTree->Write();
    if (DEBUG_TMVA_Factory) fMultiCutTestTree->Print();
    // Evaluate MVA methods globally for all multiCut Bins

    // reset list of methods: 
    if (Verbose()) cout << "--- " << GetName() << " <verbose>: delete previous methods" << endl;
    this->DeleteAllMethods();

    // evaluate the combined TestTree
    cout << "--- " << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout << "--- " << GetName() << ": Combined Overall Evaluation:" << endl;
    cout << "-------------------------------------------------------------------" << endl;
    cout << "--- " << "           "<<endl;

    TMVA_MethodBase *method = 0;
    TIter next_branch1( fTestTree->GetListOfBranches() );
    while (TBranch *branch = (TBranch*)next_branch1()){
      TLeaf *leaf = branch->GetLeaf(branch->GetName());
      if (((TString)branch->GetName()).Contains("TMVA_")){
	method = new TMVA_MethodVariable  ( fJobName,  
					   fInputVariables, 
					   fMultiCutTestTree, 
					   (TString)leaf->GetName(), 
					   fTargetFile );   
	fMethods.push_back( method );
      }// is MVA variable
    }

    fLocalTDir = fTargetFile;
    this->EvaluateAllMethods();

    // this is save:
    fMultipleStoredOptions = kFALSE;
  }
  else {
    cout << "--- " << GetName() << ":ERROR!!! ProcessMultipleMVA without bin definitions!"<<endl;
    cout << " Call BookMultipleMVAs in prior!" << endl;
    exit(1);
  }   
  // plot input variables in global tree
  
  if (fMultiCutTestTree != NULL) {
    PlotVariables( fMultiCutTestTree );
    GetCorrelationMatrix( fMultiCutTestTree );
  }
}


//_______________________________________________________________________
void TMVA_Factory::SetLocalDir( void )
{
  fLocalTDir->cd();
}


