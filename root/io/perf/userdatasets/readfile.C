#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include "TSystem.h"
#include "TTreePerfStats.h"
#include "TVirtualStreamerInfo.h"
#include "TStreamerElement.h"
#include "TROOT.h"

void SetOwner(TVirtualStreamerInfo *info, const char *dataname, Bool_t isowner)
{
   if (info) {
      TObject *el = info->GetElements()->FindObject(dataname);
      if (el) {
         if (isowner) el->ResetBit(TStreamerElement::kDoNotDelete);
         else el->SetBit(TStreamerElement::kDoNotDelete);
      }
   }
}

void SetOwner(TList *infolist, const char *classname, const char *dataname, Bool_t isowner)
{
   TVirtualStreamerInfo *info = 0;
   if (infolist) {
      info = (TVirtualStreamerInfo*)infolist->FindObject(classname);
   } else {
      TClass *cl = TClass::GetClass(classname);
      if (cl) info = cl->GetStreamerInfo();
   }
   SetOwner(info, dataname, isowner);
}

void fixHepMC(TList *infolist = 0) 
{
   if (infolist) {
      TVirtualStreamerInfo *info = (TVirtualStreamerInfo*)infolist->FindObject("HepMC::GenVertex");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_event");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      }
      info = (TVirtualStreamerInfo*)infolist->FindObject("HepMC::GenParticle");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_production_vertex");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
         el = info->GetElements()->FindObject("m_end_vertex");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      }      
      info = (TVirtualStreamerInfo*)infolist->FindObject("HepMC::GenEvent");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_vertex_barcodes");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
         el = info->GetElements()->FindObject("m_particle_barcodes");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
         
         el = info->GetElements()->FindObject("m_signal_process_vertex");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
         el = info->GetElements()->FindObject("m_beam_particle_1");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
         el = info->GetElements()->FindObject("m_beam_particle_2");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      }
      info = (TVirtualStreamerInfo*)infolist->FindObject("HepMC::Flow");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_particle_owner");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      } 
   } else {
      TClass *clGenVertex = TClass::GetClass("HepMC::GenVertex");
      if (clGenVertex && clGenVertex->GetStreamerInfo()) { 
         TObject *el = clGenVertex->GetStreamerInfo()->GetElements()->FindObject("m_event");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      }
      TClass *clGenParticle = TClass::GetClass("HepMC::GenParticle");
      if (clGenParticle && clGenParticle->GetStreamerInfo()) {
         TObject *el = clGenParticle->GetStreamerInfo()->GetElements()->FindObject("m_production_vertex");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
         el = clGenParticle->GetStreamerInfo()->GetElements()->FindObject("m_end_vertex");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      }
      TClass *cl = TClass::GetClass("HepMC::GenEvent");
      if (cl && cl->GetStreamerInfo()) { 
         TObject *el = cl->GetStreamerInfo()->GetElements()->FindObject("m_signal_process_vertex");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
         el = cl->GetStreamerInfo()->GetElements()->FindObject("m_beam_particle_1");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
         el = cl->GetStreamerInfo()->GetElements()->FindObject("m_beam_particle_2");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
         
         el = cl->GetStreamerInfo()->GetElements()->FindObject("m_vertex_barcodes");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
         el = cl->GetStreamerInfo()->GetElements()->FindObject("m_particle_barcodes");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
      }
      cl = TClass::GetClass("HepMC::Flow");
      if (cl && cl->GetStreamerInfo()) { 
         TObject *el = cl->GetStreamerInfo()->GetElements()->FindObject("m_particle_owner");
         if (el) el->SetBit(TStreamerElement::kDoNotDelete);
      }      
   }
}

void fixLHCb(TList *infolist = 0)
{
   SetOwner(infolist,"KeyedContainer<LHCb::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >","m_sequential",kTRUE);
   SetOwner(infolist,"KeyedContainer<LHCb::GenCollision,Containers::KeyedObjectManager<Containers::hashmap> >","m_sequential",kTRUE);
   SetOwner(infolist,"ObjectVector<LHCb::MCRichDigitSummary>","m_vector",kTRUE);
   return;
   
   if (infolist) {
      TVirtualStreamerInfo *info = (TVirtualStreamerInfo*)infolist->FindObject("KeyedContainer<LHCb::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_sequential");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
      }
      info = (TVirtualStreamerInfo*)infolist->FindObject("KeyedContainer<LHCb::GenCollision,Containers::KeyedObjectManager<Containers::hashmap> >");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_sequential");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
      }
      info = (TVirtualStreamerInfo*)infolist->FindObject("ObjectVector<LHCb::MCRichDigitSummary>");
      if (info) {
         TObject *el = info->GetElements()->FindObject("m_vector");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
      }
      
   } else {
      TClass *cl = TClass::GetClass("KeyedContainer<LHCb::HepMCEvent,Containers::KeyedObjectManager<Containers::hashmap> >");
      if (cl && cl->GetStreamerInfo()) {
         TObject *el = cl->GetStreamerInfo()->GetElements()->FindObject("m_sequential");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
      }
      cl = TClass::GetClass("KeyedContainer<LHCb::GenCollision,Containers::KeyedObjectManager<Containers::hashmap> >");
      if (cl && cl->GetStreamerInfo()) {
         TObject *el = cl->GetStreamerInfo()->GetElements()->FindObject("m_sequential");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
      }
      cl = TClass::GetClass("ObjectVector<LHCb::MCRichDigitSummary>");
      if (cl && cl->GetStreamerInfo()) {
         TObject *el = cl->GetStreamerInfo()->GetElements()->FindObject("m_vector");
         if (el) el->ResetBit(TStreamerElement::kDoNotDelete);
      }      
   }
}

void fixCMS(TList *infolist = 0)
{
   SetOwner(infolist,"edm::OwnVector<reco::BaseTagInfo,edm::ClonePolicy<reco::BaseTagInfo> >","data_",kTRUE);
   SetOwner(infolist,"edm::OwnVector<pat::UserData,edm::ClonePolicy<pat::UserData> >","data_",kTRUE);
}

void fixATLAS(TList *infolist = 0)
{
   SetOwner(infolist,"MuonSpShowerContainer_p1","m_showers",kTRUE);
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
      gSystem->Load("libCintex");
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
   
   // if library not load yet, generate the code, compile it and load it.
   if (loadlibrary && !haslibrary) {
      // Fix HepMC
      TList *infolist = (TList*)file->GetStreamerInfoCache();
      fixHepMC( infolist );
      fixLHCb( infolist );
      fixCMS( infolist );
      fixATLAS( infolist );      

      if (genreflex) {
         file->MakeProject(libdir.Data(),"*","NEW+genreflex");
      } else {
         file->MakeProject(libdir.Data(),"*","NEW+");
      }
      if ( gSystem->Load(TString::Format("%s/%s",libdir.Data(),libdir.Data())) < 0) {
         return 0;
      }
   } else {
      // Fix HepMC
      fixHepMC();
      fixLHCb();
      fixCMS();
      fixATLAS();
      
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
   if (!loadlibrary && strcmp(T->GetName(),"Events")==0) {
      // CMS file without library, disable the vector of polymorphic pointers
      // which we don't know yet how to delete properly.
      T->SetBranchStatus("patJets_selectedLayer1Jets__PAT.obj.tagInfos_.data_",kFALSE);
      T->SetBranchStatus("patJets_patcrosscleaner_ccJets_PAT.obj.tagInfos_.data_",kFALSE);
      T->SetBranchStatus("patJets_patcrosscleaner_droppedJets_PAT.obj.tagInfos_.data_",kFALSE);
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
