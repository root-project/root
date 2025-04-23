#include "TError.h"
#include "TFile.h"
#include "TFileMerger.h"
#include "TGraph.h"
#include "THnSparse.h"
#include "THStack.h"
#include "TTree.h"

void createInputs(int n = 2) 
{
   for(UInt_t i = 0; i < (UInt_t)n; ++i ) {
      TFile *file = TFile::Open(TString::Format("input%d.root",i),"RECREATE");
      TH1F * h = new TH1F("h1","",10,0,100);
      h->Fill(10.5); h->Fill(20.5);
 
      Int_t nbins[5];
      Double_t xmin[5];
      Double_t xmax[5];
      for(UInt_t j = 0; j < 5; ++j) {
         nbins[j] = 10; xmin[j] = 0; xmax[j] = 10;
      }
      THnSparseF *sparse = new THnSparseF("sparse", "sparse", 5, nbins, xmin, xmax);
      Double_t coord[5] = {0.5, 1.5, 2.5, 3.5, 4.5};
      sparse->Fill(coord);
      sparse->Write();
      
      THStack *stack = new THStack("stack","");
      h = new TH1F("hs_1","",10,0,100);
      h->Fill(10.5); h->Fill(20.5);
      h->SetDirectory(0);
      stack->Add(h);
      h = new TH1F("hs_2","",10,0,100);
      h->Fill(30.5); h->Fill(40.5);
      h->SetDirectory(0);
      stack->Add(h);
      stack->Write();

      TGraph *gr = new TGraph(3);
      gr->SetName("exgraph");
      gr->SetPoint(0,1,1);
      gr->SetPoint(1,2,2);
      gr->SetPoint(2,3,3);
      
      gr->Write();
      
      TTree *tree = new TTree("tree","simplistic tree");
      Int_t data = 0;
      tree->Branch("data",&data);
      for(Int_t l = 0; l < 2; ++l) {
         data = l;
         tree->Fill();
      }
      
      file->Write();
      delete file;
   }
}

bool merge(int n = 2, int limit = 0) {
   TFileMerger merger(kFALSE,kFALSE); // hadd style
   merger.OutputFile(TString::Format("merged%d.root",n));
   if (limit > 0) {
      merger.SetMaxOpenedFiles(limit);
   }
   for(UInt_t i = 0; i < (UInt_t)n; ++i ) {
      if (! merger.AddFile(TString::Format("input%d.root",i))) {
         return false;
      }
   }
   return merger.Merge();
}

bool check(int n = 2) {
   TFile *file = TFile::Open(TString::Format("merged%d.root",n));

   bool result = true;
   TH1F *h; file->GetObject("h1",h);
   if (!h) {
      Error("execFileMerger","h1 is missing\n");
      result = false;
   }
   if (h->GetBinContent(2) != n || h->GetBinContent(3) != n) {
      Error("execFileMerger","h1 not added properly");
      result = false;
   }
   
   THnSparseF *sparse; file->GetObject("sparse",sparse);
   if (!sparse) {
      Error("execFileMerger","sparse is missing\n");
      result = false;
   } else {
      Int_t coordIdx[5] = {1, 2, 3, 4, 5};
      Double_t cont = sparse->GetBinContent(coordIdx);
      if (cont > n + 0.4 || cont < n - 0.4) {
         Error("execFileMerger","sparse merge failed: expected bin content %g, read %g\n",
               (Double_t)n, cont);
         result = false;
      }
      Double_t entries = sparse->GetEntries();
      if (entries > n + 0.4 || entries < n - 0.4) {
         Error("execFileMerger","sparse merge failed: expected %g entries, read %g\n",
               (Double_t)n, entries);
         result = false;
      }
   }
   
   THStack *stack; file->GetObject("stack",stack);
   if (!stack) {
      Error("execFileMerger","stack is missing\n");
      result = false;
   }
   h = (TH1F*)stack->GetHists()->FindObject("hs_1");
   if (!h) {
      Error("execFileMerger","hs_1 is missing\n");
      result = false;
   }
   if (h->GetBinContent(2) != n || h->GetBinContent(3) != n) {
      Error("execFileMerger","hs_1 not added properly");
      result = false;
   }
   h = (TH1F*)stack->GetHists()->FindObject("hs_2");
   if (!h) {
      Error("execFileMerger","hs_2 is missing\n");
      result = false;
   }
   if (h->GetBinContent(4) != n || h->GetBinContent(5) != n) {
      Error("execFileMerger","hs_2 not added properly");
      result = false;
   }

   TGraph *gr; file->GetObject("exgraph",gr);
   if (!gr) {
      Error("execFileMerger","exgraph is missing\n");
      result = false;
   }
   if (gr->GetN() != ( n * 3)) {
      Error("execFileMerger","exgraph not added properly n=%d rather than %d",gr->GetN(),n*3);
      result = false;            
   } else {
      for(Int_t k = 0; k < gr->GetN(); ++k) {
         double x,y;
         gr->GetPoint(k,x,y);
         if ( x != ( (k%3)+1 ) ||  y != ( (k%3)+1 ) ) {
            Error("execFileMerger","exgraph not added properly");
            result = false;            
         }
      }
   }
   
   TTree *tree; file->GetObject("tree",tree);
   if (!tree) {
      Error("execFileMerger","tree is missing\n");
      result = false;
   }
   if (tree->GetEntries() != n*2) {
      Error("execFileMerger","tree does not have the expected number of entries: %lld rather than %d",tree->GetEntries(),n*2);
      result = false;            
   } else {
      if ( tree->GetEntries("data==1") != n ) {
         Error("execFileMerger","tree does not have the expected data. We got %lld entries with 'data==1' rather than %d",tree->GetEntries("data==1"),n);
         tree->Scan();
         result = false;
      }
   }   
   return result;
}

int execFileMerger(int n = 2) {
   createInputs(2*n);
   bool result = merge(n) && check(n);
   if (!result) {
      return 1;
   }
   // Now try again but limit the number of files to test the case where we run out of file descriptor
   result = merge(2 * n, 2) && check( 2 * n);
   return result ? 0 : 1;
}
