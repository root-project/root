// Author: Christophe.Delaere@cern.ch   25/04/04

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
#include "THStack.h"
#include "TLegend.h"
#include "TPad.h"
#include "Riostream.h"

ClassImp(TMLPAnalyzer)

//______________________________________________________________________________
TMLPAnalyzer::~TMLPAnalyzer() { if(analysisTree) delete analysisTree; }

//______________________________________________________________________________
Int_t TMLPAnalyzer::GetLayers()
{
   // Returns the number of layers
   TString fStructure = network->GetStructure();
   return fStructure.CountChar(':')+1;
}

//______________________________________________________________________________
Int_t TMLPAnalyzer::GetNeurons(Int_t layer)
{
   // Returns the number of neurons in given layer 
   if(layer==1) {
      TString fStructure = network->GetStructure();
      TString input      = TString(fStructure(0, fStructure.First(':')));
      return input.CountChar(',')+1;
   }
   else if(layer==GetLayers()) {
      TString fStructure = network->GetStructure();
      TString output = TString(fStructure(fStructure.Last(':') + 1,
                                  fStructure.Length() - fStructure.Last(':')));
      return output.CountChar(',')+1;
   }
   else {
     Int_t cnt=1;
     TString fStructure = network->GetStructure();
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
   // returns the formula used as input for neuron (idx) in 
   // the first layer.
   TString fStructure = network->GetStructure();
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
   // gives some information about the network in the terminal.
   TString fStructure = network->GetStructure();
   cout << "Network with structure: " << fStructure.Data() << endl;
   cout << "an input with lower values may not be needed" << endl;
   // Checks if some input variable is not needed
   for(Int_t i=0;i<GetNeurons(1);i++) {
      analysisTree->Draw(Form("Diff>>tmp%i",i),Form("InNeuron==%i",i),"goff");
      TH1F* tmp = (TH1F*)gDirectory->Get(Form("tmp%i",i));
      cout << GetNeuronFormula(i) << " -> " << tmp->GetMean() 
           << " +/- " << tmp->GetRMS() << endl;
   }
}

//______________________________________________________________________________
void TMLPAnalyzer::GatherInformations()
{
   // Collect informations about what is usefull in the network.
   // This method as to be called first when analyzing a network.
   
#define shift 0.1  
       	
   TTree* data = network->fData;
   TEventList* test = network->fTest;
   Int_t nEvents = test->GetN();
   Double_t* params = new Double_t[GetNeurons(1)];
   Double_t* rms    = new Double_t[GetNeurons(1)];
   TTreeFormula* formulas[GetNeurons(1)];
   TString formula;
   Int_t i(0), j(0), k(0), l(0);
   for(Int_t i=0; i<GetNeurons(1); i++){
      formula = GetNeuronFormula(i);
      formulas[i] = new TTreeFormula(Form("NF%d",this),formula,data);
      TH1D tmp("tmpb", "tmpb", 1, -FLT_MAX, FLT_MAX);
      data->Draw(Form("%s>>tmpb",formula.Data()),"","goff");
      rms[i]  = tmp.GetRMS();
   }
   Int_t InNeuron = 0;
   Double_t Diff = 0.;
   if(analysisTree) delete analysisTree;
   analysisTree = new TTree("result","analysis");
   analysisTree->SetDirectory(0);
   analysisTree->Branch("InNeuron",&InNeuron,"InNeuron/I");
   analysisTree->Branch("Diff",&Diff,"Diff/D");
   // Loop on the input neurons
   Double_t v1 = 0.;
   Double_t v2 = 0.;
   for(i=0; i<GetNeurons(1); i++){
      InNeuron = i;
      // Loop on the events in the test sample
      for(j=0; j< nEvents; j++) {
         network->GetEntry(test->GetEntry(j));
	 // Loop on the neurons to evaluate
	 for(k=0; k<GetNeurons(1); k++) {
            params[k] = formulas[k]->EvalInstance();
         }
	 // Loop on the neurons in the output layer
	 Diff = 0;
         for(l=0; l<GetNeurons(GetLayers()); l++){
            params[i] += shift*rms[i];
            v1 = network->Evaluate(l,params);
            params[i] -= 2*shift*rms[i];
            v2 = network->Evaluate(l,params);
	    Diff += (v1-v2)*(v1-v2);
	 }
         Diff = TMath::Sqrt(Diff);
         analysisTree->Fill();
      }
   }
   delete[] params;
   delete[] rms;
   for(i=0; i<GetNeurons(1); i++) delete formulas[i];
}

//______________________________________________________________________________
void TMLPAnalyzer::DrawDInput(Int_t i)
{
   // Draws the distribution (on the test sample) of the 
   // impact on the network output of a small variation of 
   // the ith input.
   analysisTree->Draw("Diff",Form("InNeuron==%i",i));
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
   for(Int_t i=0;i<GetNeurons(1);i++) {
      analysisTree->Draw(Form("Diff>>tmp%i",i),Form("InNeuron==%i",i),"goff");
      tmp = (TH1F*)gDirectory->Get(Form("tmp%i",i));
      tmp->SetDirectory(0);
      tmp->SetLineColor(i+1);
      stack->Add(tmp);
      legend->AddEntry(tmp,GetNeuronFormula(i),"l");
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
   // and "background" cuts.
   // Only the test sample is used.
   TTree* data = network->fData;
   TEventList* test = network->fTest;
   TEventList* current = data->GetEventList();
   data->SetEventList(test);
   THStack* stack = new THStack("__NNout_TMLPA",Form("Neural net output (neuron %i)",neuron));
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
      bgh->Fill(network->Result(bg_list->GetEntry(j),neuron));
   }
   // fill the signal
   nEvents = signal_list->GetN();
   for(j=0; j< nEvents; j++) {
      sigh->Fill(network->Result(signal_list->GetEntry(j),neuron));
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

