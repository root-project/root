#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include "TSystem.h"
#include "TTreePerfStats.h"
#include "TVirtualStreamerInfo.h"
#include "TStreamerElement.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TEnv.h"

// Instructions to use this script
//
// You can run the script in different configurations depending
//    on the values given to the 5 arguments (all optional)
//    -options: a string containing of:
//        nolib : do not load any library.
//        genreflex : use a reflex dictionary.
//        tree=somename : use a non standard name for the dictionary, this _must_ be the last options.
//    -cachesize: by default ROOT will take the best value computed when writing
//        the file. You can specify a larger cache, eg 80000000 (80 MBytes)
//        You can disable the treeCache by setting cachesize=0.
//    -i_nentries: number of entries to be read, if set to -1 (the default), the script read the
//        whole tree.
//    -percententries: fraction of the entries to be read (default to all)
//    -percentbranches: fraction of the branches to be read (default to all)
//
//   for example the following session
//       root > .x readfile.C("atlasFlushed.root","",60000000,-1,1,0.33)
//   will use a 60 MBytes cache, reading all branches and only 1/3 of entries
//
// If the file does not exist locally, we look for the file in http://root.cern.ch/files/
//
// Note: The very first time, this script is used on a file, it will call MakeProject
// to create a library named after the file.  For myfile.root, the library is named
// libmyfile/libmyfile.so in the rootcint case and libgenMyfile/libgenMyfile.root
// when using genreflex.

void fixLHCb()
{
   TClass::AddRule("KeyedContainer<LHCb::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >     m_sequential   attributes=Owner");
   TClass::AddRule("KeyedContainer<LHCb::GenCollision,Containers::KeyedObjectManager<Containers::hashmap> >    m_sequential   attributes=Owner");
   TClass::AddRule("ObjectVector<LHCb::MCRichDigitSummary>    m_vector   attributes=Owner");
}

void fixCMS()
{
   TClass::AddRule("edm::OwnVector<reco::BaseTagInfo,edm::ClonePolicy<reco::BaseTagInfo> >    data_   attributes=Owner");
   TClass::AddRule("edm::OwnVector<pat::UserData,edm::ClonePolicy<pat::UserData> >            data_   attributes=Owner");
}

void fixATLAS()
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
      // gSystem->Load("libCintex");
      // gROOT->ProcessLine("ROOT::Cintex::Cintex::Enable()");
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
   TString filename(i_filename);
   if (gSystem->AccessPathName(filename,kReadPermission) && filename.Index(":") == kNPOS) {
      // otherwise open the http file
      filename.Prepend("https://root.cern/files/");

      // configure cache directory
      TFile::SetCacheFileDir(".");
   }
   TFile *file = TFile::Open( filename );

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
   const char *names [] = { "E","Events","CollectionTree","ntuple","T" };

   for (unsigned int i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
      file->GetObject(names[i],tree);
      if (tree) return tree;
   }
   return 0;
}

void readfile(const char *filename, const char *options /* = 0 */, Int_t cachesize=-1, Long64_t i_nentries = -1, Int_t freq = 1000, Float_t percententries = 1.00, Float_t percentbranches = 1.00);

void readfile(const char *filename = "lhcb2.root", Int_t cachesize=-1) {
   readfile(filename,0,cachesize);
}

void readfile(const char *filename, const char *options /* = 0 */, Int_t cachesize /* =-1 */,
              Long64_t i_nentries /* = -1 */, Int_t freq /* = 1000 */, Float_t percententries /* = 1.00 */, Float_t percentbranches /* = 1.00 */)
{
   // The support options are:
   //   nolib : do not load any library.
   //   genreflex : use a reflex dictionary.
   //   tree=somename : use a non standard name for the dictionary, this _must_ be the last options.

   TStopwatch sw;

   TString opt(options);
   bool genreflex = opt.Contains("genreflex");
   bool loadlibrary = !opt.Contains("nolib");
   Ssiz_t pos = opt.Index("tree=");
   const char *treename = 0;
   if ( pos != kNPOS) {
      treename = &(opt[pos+strlen("tree=")]);
   }
   TFile *file = openFileAndLib(filename,loadlibrary,genreflex);

   if (file==0) return;

   TTree *T = getTree(file,treename);

   TFile::SetReadaheadSize(0);  // (256*1024);
   Long64_t nentries = T->GetEntries();
   if (i_nentries >= 0) {
     nentries = i_nentries;
   }
   int efirst = 0;
   int elast  = efirst+nentries;
   if (cachesize == -2) {
      gEnv->SetValue("TFile.AsyncReading", 0);
      cachesize = -1;
   }
   T->SetCacheSize(cachesize);
   if (cachesize != 0) {
      T->SetCacheEntryRange(efirst,elast);
      if (percentbranches < 1.00) {
         int nb = T->GetListOfBranches()->GetEntries();
         int incr = nb * percentbranches;
         for(int b=0;b < nb; b += incr) T->AddBranchToCache(((TBranch*)T->GetListOfBranches()->At(b)),kTRUE);
      } else {
         T->AddBranchToCache("*");
      }
      T->StopCacheLearningPhase();
   }

   TTreePerfStats *ps= new TTreePerfStats("ioperf",T);

   TRandom r;
   for (Long64_t i=efirst;i<elast;i++) {
      if (i%freq == 0) printf("i = %lld\n",i);
      if (r.Rndm() > percententries) continue;
      T->LoadTree(i);
      if (percentbranches < 1.00) {
         int nb = T->GetListOfBranches()->GetEntries();
         int incr = nb * percentbranches;
         for(int b=0;b<nb; b += incr) ((TBranch*)T->GetListOfBranches()->At(b))->GetEntry(i);
         int count = 0;
         int maxcount = 100 + 100 ;
         for(int x = 0; x < maxcount; ++x ) { /* waste cpu */ count = sin(cos(count)); }
      } else {
         T->GetEntry(i);
      }
   }
   TString psfilename(filename);
   TString pssuffix("_ioperf.root");
   if (options && options[0]) { pssuffix.Prepend(options); pssuffix.Prepend("_"); }
   psfilename.ReplaceAll(".root", pssuffix );
   ps->SaveAs(psfilename);
   //ps->Draw();
   if (ps) ps->Print("unzip");
   T->PrintCacheStats();
   printf("Real Time = %7.3f s, CPUtime = %7.3f s\n",sw.RealTime(),sw.CpuTime());
}
