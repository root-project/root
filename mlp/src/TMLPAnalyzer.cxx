// @(#)root/mlp:$Name:  $:$Id: TMLPAnalyzer.cxx,v 1.5 2004/09/30 10:13:30 rdm Exp $
// Author: Christophe.Delaere@cern.ch   25/04/04

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
//
// TMLPAnalyzer
//
// This utility class contains a set of tests usefull when developing
// a neural network.
// It allows you to check for unneeded variables, and to control
// the network structure.
//
///////////////////////////////////////////////////////////////////////////

#include "TSynapse.h"
#include "TNeuron.h"
#include "TMultiLayerPerceptron.h"
#include "TMLPAnalyzer.h"
#include "TTree.h"
#include "TTreeFormula.h"
#include "TEventList.h"
#include "TH1D.h"
#include "TProfile.h"
#include "THStack.h"
#include "TLegend.h"
#include "TPad.h"
#include "Riostream.h"

ClassImp(TMLPAnalyzer)

//______________________________________________________________________________
TMLPAnalyzer::~TMLPAnalyzer() { if(fAnalysisTree) delete fAnalysisTree; }

//______________________________________________________________________________
Int_t TMLPAnalyzer::GetLayers()
{
   // Returns the number of layers.

   TString fStructure = fNetwork->GetStructure();
   return fStructure.CountChar(':')+1;
}

//______________________________________________________________________________
Int_t TMLPAnalyzer::GetNeurons(Int_t layer)
{
   // Returns the number of neurons in given layer.

   if(layer==1) {
      TString fStructure = fNetwork->GetStructure();
      TString input      = TString(fStructure(0, fStructure.First(':')));
      return input.CountChar(',')+1;
   }
   else if(layer==GetLayers()) {
      TString fStructure = fNetwork->GetStructure();
      TString output = TString(fStructure(fStructure.Last(':') + 1,
                               fStructure.Length() - fStructure.Last(':')));
      return output.CountChar(',')+1;
   }
   else {
     Int_t cnt=1;
     TString fStructure = fNetwork->GetStructure();
     TString hidden = TString(fStructure(fStructure.First(':') + 1,
                              fStructure.Last(':') - fStructure.First(':') - 1));
     Int_t beg = 0;
     Int_t end = hidden.Index(":", beg + 1);
     Int_t num = 0;
     while (end != -1) {
       num = atoi(TString(hidden(beg, end - beg)).Data());
       cnt++;
       beg = end + 1;
       end = hidden.Index(":", beg + 1);
       if(layer==cnt) return num;
     }
     num = atoi(TString(hidden(beg, hidden.Length() - beg)).Data());
     cnt++;
     if(layer==cnt) return num;
   }
   return -1;
}

//______________________________________________________________________________
TString TMLPAnalyzer::GetNeuronFormula(Int_t idx)
{
   // Returns the formula used as input for neuron (idx) in
   // the first layer.

   TString fStructure = fNetwork->GetStructure();
   TString input      = TString(fStructure(0, fStructure.First(':')));
   Int_t beg = 0;
   Int_t end = input.Index(",", beg + 1);
   TString brName;
   Int_t cnt = 0;
   while (end != -1) {
      brName = TString(input(beg, end - beg));
      beg = end + 1;
      end = input.Index(",", beg + 1);
      if(cnt==idx) return brName;
      cnt++;
   }
   brName = TString(input(beg, input.Length() - beg));
   return brName;
}

//______________________________________________________________________________
void TMLPAnalyzer::CheckNetwork()
{
   // Gives some information about the network in the terminal.

   TString fStructure = fNetwork->GetStructure();
   cout << "Network with structure: " << fStructure.Data() << endl;
   cout << "an input with lower values may not be needed" << endl;
   // Checks if some input variable is not needed
   char var[64], sel[64];
   for (Int_t i = 0; i < GetNeurons(1); i++) {
      sprintf(var,"Diff>>tmp%d",i);
      sprintf(sel,"InNeuron==%d",i);
      fAnalysisTree->Draw(var, sel, "goff");
      TH1F* tmp = (TH1F*)gDirectory->Get(Form("tmp%d",i));
      cout << ((TNeuron*)fNetwork->fFirstLayer[i])->GetName() 
           << " -> " << tmp->GetMean()
           << " +/- " << tmp->GetRMS() << endl;
   }
}

//______________________________________________________________________________
void TMLPAnalyzer::GatherInformations()
{
   // Collect informations about what is usefull in the network.
   // This method has to be called first when analyzing a network.
   // Fills the two analysis trees.

   Double_t shift = 0.1;

   TTree* data = fNetwork->fData;
   TEventList* test = fNetwork->fTest;
   Int_t nEvents = test->GetN();
   Int_t NN = GetNeurons(1);
   Double_t* params = new Double_t[NN];
   Double_t* rms    = new Double_t[NN];
   TTreeFormula** formulas = new TTreeFormula*[NN];
   TString formula;
   Int_t i(0), j(0), k(0), l(0);
   for(i=0; i<NN; i++){
      formula = GetNeuronFormula(i);
      formulas[i] = new TTreeFormula(Form("NF%d",this),formula,data);
      TH1D tmp("tmpb", "tmpb", 1, -FLT_MAX, FLT_MAX);
      data->Draw(Form("%s>>tmpb",formula.Data()),"","goff");
      rms[i]  = tmp.GetRMS();
   }

   Int_t InNeuron = 0;
   Double_t Diff = 0.;
   if(fAnalysisTree) delete fAnalysisTree;
   fAnalysisTree = new TTree("result","analysis");
   fAnalysisTree->SetDirectory(0);
   fAnalysisTree->Branch("InNeuron",&InNeuron,"InNeuron/I");
   fAnalysisTree->Branch("Diff",&Diff,"Diff/D");

   Int_t numOutNodes=GetNeurons(GetLayers());
   Double_t *outVal=new Double_t[numOutNodes];
   Double_t *trueVal=new Double_t[numOutNodes];

   delete fIOTree;
   fIOTree=new TTree("MLP_iotree","MLP_iotree");
   fIOTree->SetDirectory(0);
   TString leaflist;
   for (i=0; i<NN; i++)
      leaflist+=Form("In%d/D:",i);
   leaflist.Remove(leaflist.Length()-1);
   fIOTree->Branch("In", params, leaflist);

   leaflist="";
   for (i=0; i<numOutNodes; i++)
      leaflist+=Form("Out%d/D:",i);
   leaflist.Remove(leaflist.Length()-1);
   fIOTree->Branch("Out", outVal, leaflist);

   leaflist="";
   for (i=0; i<numOutNodes; i++)
      leaflist+=Form("True%d/D:",i);
   leaflist.Remove(leaflist.Length()-1);
   fIOTree->Branch("True", trueVal, leaflist);

   Double_t v1 = 0.;
   Double_t v2 = 0.;
   // Loop on the events in the test sample
   for(j=0; j< nEvents; j++) {
      fNetwork->GetEntry(test->GetEntry(j));

      // Loop on the neurons to evaluate
      for(k=0; k<GetNeurons(1); k++) 
         params[k] = formulas[k]->EvalInstance();

      for(k=0; k<GetNeurons(GetLayers()); k++) {
         outVal[k] = fNetwork->Evaluate(k,params);
         trueVal[k] = ((TNeuron*)fNetwork->fLastLayer[k])->GetBranch();
      }

      fIOTree->Fill();

      // Loop on the input neurons
      for (i = 0; i < GetNeurons(1); i++) {
         InNeuron = i;
         Diff = 0;
         // Loop on the neurons in the output layer
         for(l=0; l<GetNeurons(GetLayers()); l++){
            params[i] += shift*rms[i];
            v1 = fNetwork->Evaluate(l,params);
            params[i] -= 2*shift*rms[i];
            v2 = fNetwork->Evaluate(l,params);
	    Diff += (v1-v2)*(v1-v2);
            // reset to original vealue
            params[i] += shift*rms[i];
	 }
         Diff = TMath::Sqrt(Diff);
         fAnalysisTree->Fill();
      }
   }
   delete[] params;
   delete[] rms;
   delete[] outVal;
   for(i=0; i<GetNeurons(1); i++) delete formulas[i]; delete [] formulas;
   fAnalysisTree->ResetBranchAddresses();
   fIOTree->ResetBranchAddresses();
}

//______________________________________________________________________________
void TMLPAnalyzer::DrawDInput(Int_t i)
{
   // Draws the distribution (on the test sample) of the
   // impact on the network output of a small variation of
   // the ith input.

   char sel[64];
   sprintf(sel, "InNeuron==%d", i);
   fAnalysisTree->Draw("Diff", sel);
}

//______________________________________________________________________________
void TMLPAnalyzer::DrawDInputs()
{
   // Draws the distribution (on the test sample) of the
   // impact on the network output of a small variation of
   // each input.

   THStack* stack  = new THStack("differences","differences");
   TLegend* legend = new TLegend(0.75,0.75,0.95,0.95);
   TH1F* tmp = NULL;
   char var[64], sel[64];
   for(Int_t i = 0; i < GetNeurons(1); i++) {
      sprintf(var, "Diff>>tmp%d", i);
      sprintf(sel, "InNeuron==%d", i);
      fAnalysisTree->Draw(var, sel, "goff");
      tmp = (TH1F*)gDirectory->Get(Form("tmp%d",i));
      tmp->SetDirectory(0);
      tmp->SetLineColor(i+1);
      stack->Add(tmp);
      legend->AddEntry(tmp,((TNeuron*)fNetwork->fFirstLayer[i])->GetName(),"l");
   }
   stack->Draw("nostack");
   legend->Draw();
   gPad->SetLogy();
}

//______________________________________________________________________________
void TMLPAnalyzer::DrawNetwork(Int_t neuron, const char* signal, const char* bg)
{
   // Draws the distribution of the neural network (using ith neuron).
   // Two distributions are drawn, for events passing respectively the "signal"
   // and "background" cuts. Only the test sample is used.

   TTree* data = fNetwork->fData;
   TEventList* test = fNetwork->fTest;
   TEventList* current = data->GetEventList();
   data->SetEventList(test);
   THStack* stack = new THStack("__NNout_TMLPA",Form("Neural net output (neuron %d)",neuron));
   TH1F *bgh  = new TH1F("__bgh_TMLPA", "NN output", 50, 0, -1);
   TH1F *sigh = new TH1F("__sigh_TMLPA", "NN output", 50, 0, -1);
   bgh->SetDirectory(0);
   sigh->SetDirectory(0);
   Int_t nEvents = 0;
   Int_t j=0;
   // build event lists for signal and background
   TEventList* signal_list = new TEventList("__tmpSig_MLPA");
   TEventList* bg_list     = new TEventList("__tmpBkg_MLPA");
   data->Draw(">>__tmpSig_MLPA",signal,"goff");
   data->Draw(">>__tmpBkg_MLPA",bg,"goff");
   // fill the background
   nEvents = bg_list->GetN();
   for(j=0; j< nEvents; j++) {
      bgh->Fill(fNetwork->Result(bg_list->GetEntry(j),neuron));
   }
   // fill the signal
   nEvents = signal_list->GetN();
   for(j=0; j< nEvents; j++) {
      sigh->Fill(fNetwork->Result(signal_list->GetEntry(j),neuron));
   }
   // draws the result
   bgh->SetLineColor(kBlue);
   bgh->SetFillStyle(3008);
   bgh->SetFillColor(kBlue);
   sigh->SetLineColor(kRed);
   sigh->SetFillStyle(3003);
   sigh->SetFillColor(kRed);
   bgh->SetStats(0);
   sigh->SetStats(0);
   stack->Add(bgh);
   stack->Add(sigh);
   TLegend *legend = new TLegend(.75, .80, .95, .95);
   legend->AddEntry(bgh, "Background");
   legend->AddEntry(sigh,"Signal");
   stack->Draw("nostack");
   legend->Draw();
   // restore the default event list
   data->SetEventList(current);
   delete signal_list;
   delete bg_list;
}

//______________________________________________________________________________
TProfile* TMLPAnalyzer::DrawTruthDeviation(Int_t i, Option_t *option /*=""*/) 
{
   // Draws a profile of the difference of the MLP output minus the
   // true value for a given output i, vs the true i, for all test data events.
   // Options are passed to TProfile::Draw
   TString pipehist=Form("MLP_truthdev_%d",i);
   TString drawline;
   drawline.Form("Out.Out%d/True.True%d-1.:True.True%d>>", i,i,i);
   fIOTree->Draw(drawline+pipehist, "", "goff prof");
   TProfile* h=(TProfile*)gROOT->FindObject(pipehist);
   const char* title=((TNeuron*)fNetwork->fLastLayer[i])->GetName();
   if (title && h) {
      h->GetXaxis()->SetTitle(title);
      h->GetYaxis()->SetTitle(Form("Deviation from %s", title));
   }
   h->Draw(option);
   return h;
}

//______________________________________________________________________________
THStack* TMLPAnalyzer::DrawTruthDeviations(Option_t *option /*=""*/)
{
   // Draws a profile of the difference of the MLP output minus the
   // true value vs the true value, stacked for all outputs, for all 
   // test data events.
   // Options are passed to TProfile::Draw
   THStack *hs=new THStack("MLP_TruthDeviation",
                           "Deviation of MLP output from truth");
   TLegend *leg=new TLegend(.7,.7,.95,.95,"MLP output");
   for (Int_t o=0; o<GetNeurons(GetLayers()); o++) {
      TProfile* h=DrawTruthDeviation(o, "goff");
      h->SetLineColor(1+o);
      hs->Add(h, Form("node %d",o));
   }
   if (!option || !strstr(option,"goff")) {
      hs->Draw();
      leg->Draw();
   }
   return hs;
}

//______________________________________________________________________________
TProfile* TMLPAnalyzer::DrawTruthDeviationInOut(Int_t i, Int_t o, 
                                           Option_t *option /*=""*/)
{
   // Draws a profile of the difference of the MLP output o minus the
   // true value of o vs the input value i, for all test data events.
   // Options are passed to TProfile::Draw
   TString pipehist=Form("MLP_truthdev_i%d_o%d",i,o);
   TString drawline;
   drawline.Form("Out.Out%d/True.True%d-1.:In.In%d>>", o,o,i);
   fIOTree->Draw(drawline+pipehist, "", "goff prof");
   TProfile* h=(TProfile*)gROOT->FindObject(pipehist);
   const char* title=((TNeuron*)fNetwork->fFirstLayer[i])->GetName();
   if (title && h)
      h->GetXaxis()->SetTitle(title);
   title=((TNeuron*)fNetwork->fLastLayer[o])->GetName();
   if (title && h)
      h->GetYaxis()->SetTitle(Form("Deviation from output %s", title));
   h->Draw(option);
   return h;
}

//______________________________________________________________________________
THStack* TMLPAnalyzer::DrawTruthDeviationInsOut(Int_t o, Option_t *option /*=""*/)
{
   // Draws a profile of the difference of the MLP output o minus the
   // true value of o vs the input value, stacked for all inputs, for
   // all test data events.
   // Options are passed to TProfile::Draw
   TString sName;
   sName.Form("MLP_TruthDeviationIO_%d", o);
   THStack *hs=new THStack(sName,
                           Form("Deviation of MLP output %o from truth"));
   TLegend *leg=new TLegend(.7,.7,.95,.95,"MLP output");
   for (Int_t o=0; o<GetNeurons(GetLayers()); o++) {
      TProfile* h=DrawTruthDeviation(o, "goff");
      h->SetLineColor(1+o);
      hs->Add(h, Form("node %d",o));
   }
   if (!option || !strstr(option,"goff")) {
      hs->Draw();
      leg->Draw();
   }
   return hs;
}
