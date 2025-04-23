#include "iostream"
#include <TH1F.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>
#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
using namespace std;

void reader() {

   TFile *hfile = new TFile("unweighted.root","read");
   TTree *LHEF=(TTree*)hfile->Get("LHEF");
   auto nen = LHEF->GetEntries();
   //LHEF->Show();
   TH1F *invm = new TH1F("H_invmass","GeV",100,0,150);
   TTreeReader reader("LHEF",hfile);
   TTreeReaderArray<double> ParticleM(reader,"Particle.M");
   long long e = 0;

   UInt_t nullValues = 0;
   UInt_t lowValues = 0;
   UInt_t highValues = 0;
   UInt_t allValues = 0;
   while (reader.Next())
   {
      for (size_t j=0; j< ParticleM.GetSize() ;j++)
      {
         Double_t k = ParticleM.At(j);
         if ( k < 0.001 ) nullValues++;
         else if (k < 100) lowValues++;
         else highValues++;
         allValues++;
      }
   }
   fprintf(stdout,"all=%u zero=%u low=%u high=%u\n",allValues,nullValues,lowValues,highValues);
   delete hfile;
}

void direct()
{
   TFile *hfile = new TFile("unweighted.root","read");
   TTree *LHEF=(TTree*)hfile->Get("LHEF");

   LHEF->SetBranchStatus("*",0);
   LHEF->SetBranchStatus("Particle.M",1);
   LHEF->SetBranchStatus("Event_size",1);
   Long64_t nentries = LHEF->GetEntries();

   Int_t nParticle;
   Double_t Particle_M[9];

   LHEF->SetMakeClass(true);
   LHEF->SetBranchAddress("Particle",&nParticle);
   LHEF->SetBranchAddress("Particle.M",&(Particle_M[0]));

   UInt_t nullValues = 0;
   UInt_t lowValues = 0;
   UInt_t highValues = 0;
   UInt_t allValues = 0;
   for (Int_t i=0; i< LHEF->GetEntries();i++)
   {
      LHEF->GetEntry(i);
      for (Int_t j=0; j< nParticle ;j++)
      {
         double k = Particle_M[j];
         if ( k < 0.001 ) nullValues++;
         else if (k < 100) lowValues++;
         else highValues++;
         allValues++;
      }
   }
   fprintf(stdout,"all=%u zero=%u low=%u high=%u\n",allValues,nullValues,lowValues,highValues);
   delete hfile;
}

void legacySelector() {
   TFile *hfile = new TFile("unweighted.root","read");
   TTree *LHEF = (TTree*)hfile->Get("LHEF");
   LHEF->MakeSelector("lhef_leg_sel_gen","=legacy");
   LHEF->Process("lhef_leg_sel.C");
   delete hfile;
}

void selector() {
   TFile *hfile = new TFile("unweighted.root","read");
   TTree *LHEF = (TTree*)hfile->Get("LHEF");
   LHEF->MakeSelector("lhef_sel_gen","");
   LHEF->Process("lhef_sel.C");
   delete hfile;
}

void makeClass() {
   TFile *hfile = new TFile("unweighted.root","read");
   TTree *LHEF = (TTree*)hfile->Get("LHEF");
   LHEF->MakeClass("lhef_mc_gen","");
   delete hfile;

   gROOT->LoadMacro("lhef_mc.C");
   gROOT->ProcessLine("lhef_mc m; m.Loop();");
}

int execLHEF()
{
   fprintf(stdout,"direct\n");
   direct();

   fprintf(stdout,"reader\n");
   reader();

   fprintf(stdout,"legacySelector\n");
   legacySelector();

   fprintf(stdout,"selector\n");
   selector();

   fprintf(stdout,"makeClass\n");
   makeClass();

   return 0;
}
