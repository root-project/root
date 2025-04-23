#include <vector>
#include <iostream>

class Flow {
public:
   float sumCharged;
   float sumNeutral;
   void Fill(int seed) {
      sumCharged = 100*seed + 1;
      sumNeutral = 100*seed + 2;
   }
   void Print() {
      std::cout << "charged: " << sumCharged << '\n';
      std::cout << "neutral: " << sumNeutral << '\n';
   }
   ClassDefNV(Flow,4);
};

#ifdef __MAKECINT__
#pragma read sourceClass="Flow" targetClass="Flow" version="[2]" source="float charged" target="sumCharged" code="{ sumCharged = onfile.charged; }"
#pragma read sourceClass="Flow" targetClass="Flow" version="[2]" source="float neutral" target="sumNeutral" code="{ sumNeutral = onfile.neutral; }"
#endif

class Electron {
public:
   int  fValue;
   Flow fFlow;
   
   void Fill(int seed) {
      fValue = seed*10;
      fFlow.Fill(seed);
   }
   void Print() {
      std::cout << "fValue: " << fValue << '\n';
      std::cout << "fFlow\n";
      fFlow.Print();
   }
};

class Holder {
public:
   std::vector<Electron> obj;
   void Fill(int seed = 2) {
      Electron e;
      for(int i = 0; i< seed; ++i) {
         e.Fill(i);
         obj.push_back(e);
      }
   }
   void Print() {
      std::cout << "Holder\n";
      for(unsigned int i = 0; i<obj.size(); ++i) {
         std::cout << "Electron #" << i << '\n';
         obj[i].Print();
      }
   }
};

#include "TFile.h"
#include "TTree.h"

void write() {
   TFile *file = new TFile("electron.root","RECREATE");
   TTree *tree = new TTree("tree","electron tree");
   Holder h;
   h.Fill();
   h.Print();
   tree->Branch("holder.",&h);
   tree->Fill();
   file->Write();
   delete file;
}

int execReadRename1() {
   TFile *file = new TFile("electron.root","READ");
   if (!file) return 1;
   TTree *tree; file->GetObject("tree",tree);
   if (!tree) return 2;
   Holder *h = 0;
   tree->SetBranchAddress("holder.",&h);
   tree->GetEntry(0);
   if (!h) return 3;
   h->Print();
   // tree->Print("debugInfo");
   return 0;
}


