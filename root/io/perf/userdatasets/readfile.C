#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include "TSystem.h"
#include "TTreePerfStats.h"
#include "TVirtualStreamerInfo.h"
#include "TStreamerElement.h"
#include "TROOT.h"

void fixLHCb(TList *infolist = 0)
{
   TClass::AddRule("KeyedContainer<LHCb::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >     m_sequential   attributes=Owner");
   TClass::AddRule("KeyedContainer<LHCb::GenCollision,Containers::KeyedObjectManager<Containers::hashmap> >    m_sequential   attributes=Owner"); 
   TClass::AddRule("ObjectVector<LHCb::MCRichDigitSummary>    m_vector   attributes=Owner");
}

void fixCMS(TList *infolist = 0)
{
   TClass::AddRule("edm::OwnVector<reco::BaseTagInfo,edm::ClonePolicy<reco::BaseTagInfo> >    data_   attributes=Owner");
   TClass::AddRule("edm::OwnVector<pat::UserData,edm::ClonePolicy<pat::UserData> >            data_   attributes=Owner");
}

void fixATLAS(TList *infolist = 0)
{
   TClass::AddRule("MuonSpShowerContainer_p1 m_showers attributes=Owner");
}

TFile *openFileAndLib(const char *i_filename, bool loadlibrary, bool genreflex)
{
   // Load library if any
   TString libdir(i_filename);
   Ssiz_t pos = libdir.Index(".root");
   if (pos != kNPOS) {
      libdir.Remove(pos);
   }
   
   if (genreflex) {
      gSystem->Load("libCintex");file://localhost/Users/pcanal/root_working/roottest/root/io/perf/userdatasets/readfile.C
      gROOT->ProcessLine("ROOT::Cintex::Cintex::Enable()");
      libdir.Prepend("gen");
   }
   libdir.Prepend("lib");

   bool haslibrary = !gSystem->AccessPathName(libdir,kReadPermission);
   if (loadlibrary && haslibrary) {
      if ( gSystem->Load(TString::Format("%s/%s",libdir.Data(),libdir.Data())) < 0) {
         return 0;
      }
   }
                    
   // open the local if any
   TFile *file;
   TString filename(i_filename);
   if (gSystem->AccessPathName(filename,kReadPermission) && filename.Index(":") == kNPOS) {
      // otherwise open the http file
      filename.Prepend("http://root.cern.ch/files/");
   }
   file = TFile::Open( filename );
   
   if (!file) return 0;
   
   fixLHCb();
   fixCMS();
   fixATLAS();

   // if library not load yet, generate the code, compile it and load it.
   if (loadlibrary && !haslibrary) {
      // Fix HepMC
      if (genreflex) {
         file->MakeProject(libdir.Data(),"*","NEW+genreflex");
      } else {
         file->MakeProject(libdir.Data(),"*","NEW+");
      }
      if ( gSystem->Load(TString::Format("%s/%s",libdir.Data(),libdir.Data())) < 0) {
         return 0;
      }
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
   
   for (unsigned int i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
      file->GetObject(names[i],tree);
      if (tree) return tree;
   }
   return 0;
}


void readfile(const char *filename, const char *options /* = 0 */, Int_t cachesize=-1);

void readfile(const char *filename = "lhcb2.root", Int_t cachesize=-1) {
   readfile(filename,0,cachesize);
}

void readfile(const char *filename, const char *options /* = 0 */, Int_t cachesize /* =-1 */) 
{
   // The support options are:
   //   nolib : do not load any library.
   //   genreflex : use a reflex dictionary.
   //   tree=somename : use a non standard name for the dictionary, this _must_ be the last options.
   
   TString opt(options);
   bool genreflex = opt.Contains("genreflex");
   bool loadlibrary = !opt.Contains("nolib");
   Ssiz_t pos = opt.Index("tree=");
   const char *treename = 0;
   if ( pos != kNPOS) {
      treename = &(opt[pos]);
   }
   
   //gSystem->Load("lhcbdir/lhcbdir");  //shared lib generated with TFile::MakeProject
   TFile *file = openFileAndLib(filename,loadlibrary,genreflex);

   if (file==0) return;
   
   TTree *T = getTree(file,treename);

   TFile::SetReadaheadSize(0);
   Long64_t nentries = T->GetEntries();
   nentries   = 200;
   int efirst = 0;
   int elast  = efirst+nentries;
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
      //if (r.Rndm() > 0.01) continue; to check: 12,14,15
      TBranch * br = ((TBranch*)T->GetListOfBranches()->At(16));
      br = (TBranch*)br->GetListOfBranches()->At(2);
//      for(int b= 44;b< 47;++b) { 
//         TBranch *readbr = ((TBranch*)br->GetListOfBranches()->At(b));
//         fprintf(stdout,"%d %s %d : ",b, readbr->GetName(),readbr->TestBit(kDoNotProcess)); 
//         fprintf(stdout,"%d\n",readbr->GetEntry(i)); 
//      }
      //for(int b= 17;b< 18;++b) ((TBranch*)T->GetListOfBranches()->At(b))->GetEntry(i);
      T->GetEntry(i);
   }
   TString psfilename(filename);
   TString pssuffix("_ioperf.root");
   if (options && options[0]) { pssuffix.Prepend(options); pssuffix.Prepend("_"); } 
   psfilename.ReplaceAll(".root", pssuffix );
   ps->SaveAs(psfilename);
   //ps->Draw();
   ps->Print();
   T->PrintCacheStats();
   //printf("Real Time = %7.3f s, CPUtime = %7.3f s\n",sw.RealTime(),sw.CpuTime());
}
