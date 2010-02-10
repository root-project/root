#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include "TSystem.h"
#include "TTreePerfStats.h"
#include "TVirtualStreamerInfo.h"
#include "TStreamerElement.h"

TFile *openFileAndLib(const char *i_filename)
{
   // Load library if any
   TString libdir(i_filename);
   Ssiz_t pos = libdir.Index(".root");
   if (pos != kNPOS) {
      libdir.Remove(pos);
   }
   bool haslibrary = !gSystem->AccessPathName(libdir,kReadPermission);
   if (haslibrary) {
      gSystem->Load(TString::Format("%s/%s",libdir.Data(),libdir.Data()));
   }
                    
   // open the local if any
   TFile *file;
   TString filename(i_filename);
   if (gSystem->AccessPathName(filename,kReadPermission) ) {
      // otherwise open the http file
      filename.Prepend("http://root.cern.ch/files/");
   }
   file = TFile::Open( filename );
   
   if (!file) return 0;
   
   // if library not load yet, generate the code, compile it and load it.
   if (!haslibrary) {
      // Fix HepMC if needed.
      TVirtualStreamerInfo *info = (TVirtualStreamerInfo*)file->GetStreamerInfoCache()->FindObject("HepMC::GenVertex");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_event");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      }
      
      file->MakeProject(libdir.Data(),"*","NEW+");
      gSystem->Load(TString::Format("%s/%s",libdir.Data(),libdir.Data()));
   }      
   
   return file;
}

TTree *getTree(TFile *file, const char *treename) {
   TTree *tree;
   
   if (treename) {
      file->GetObject(treename,tree);
      return tree;
   }
   // Try the known names :)
   const char *names [] = { "E","Events","CollectionTree" };
   
   for (unsigned int i = 0; i < sizeof(names); ++i) {
      file->GetObject(names[i],tree);
      if (tree) return tree;
   }
   return 0;
}


void readfile(const char *filename, const char *treename /* = 0 */, Int_t cachesize=-1);

void readfile(const char *filename = "lhcb2.root", Int_t cachesize=-1) {
   readfile(filename,0,cachesize);
}

void readfile(const char *filename, const char *treename /* = 0 */, Int_t cachesize /* =-1 */) {
   
   //gSystem->Load("lhcbdir/lhcbdir");  //shared lib generated with TFile::MakeProject
   TFile *file = openFileAndLib(filename);

   TTree *T = getTree(file,treename);

   TFile::SetReadaheadSize(0);
   Long64_t nentries = T->GetEntries();
   nentries=200;
   int efirst= 0;
   int elast = efirst+nentries;
   T->SetCacheSize(cachesize);
   if (cachesize != 0) {
      T->SetCacheEntryRange(efirst,elast);
      T->AddBranchToCache("*");
      T->StopCacheLearningPhase();
   } 
   
   TTreePerfStats *ps= new TTreePerfStats("ioperf",T);
   
   TRandom r;
   for (Long64_t i=efirst;i<elast;i++) {
      if (i%10 == 0) printf("i = %lld\n",i);
      //if (r.Rndm() > 0.01) continue;
      T->GetEntry(i);
   }
   ps->SaveAs("lhcb2_ioperf.root");
   //ps->Draw();
   ps->Print();
   T->PrintCacheStats();
   //printf("Real Time = %7.3f s, CPUtime = %7.3f s\n",sw.RealTime(),sw.CpuTime());
}
