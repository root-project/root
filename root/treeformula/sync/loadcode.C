//------------------------  jetclass.h ---------------------------

#include <TClonesArray.h>

class Jet: public TObject {
public:
   Float_t  pt;
   
   Int_t    nvert;
   Float_t  v_x[20];       //[nvert]
   Float_t  v_y[20];       //[nvert]
   Float_t  v_z[20];       //[nvert]
   Int_t    nbest;
   
   Int_t    nmu;
   Float_t  mu_pt[100];    //[nmu]
   Int_t    mu_nseg[100];  //[nmu]
   Jet(): TObject() {
      clear();
   }
   Jet(Float_t pt_): TObject()
   {
      clear();
      pt = pt_;
   }
   Jet(const Jet& jet): TObject(jet) {
      for (int i=0; i<jet.nvert; ++i) {
         v_x[i] = jet.v_x[i];
         v_y[i] = jet.v_y[i];
         v_z[i] = jet.v_z[i];
         nvert = i+1;
      }
      for (int i=0; i<jet.nmu; ++i) {
         mu_pt[i]    = jet.mu_pt[i];
         mu_nseg[i]  = jet.mu_nseg[i];
         nmu = i+1;
      }
      pt    = jet.pt;
      nbest = jet.nbest;
   }
   void clear() {
      pt    = 0;
      nvert = 0;
      nmu   = 0;
      nbest = -1;
   }
   ClassDefOverride(Jet,1)
};

class Event: public TObject {
public:
   Int_t run;
   Int_t evt;
   Int_t njet;
   TClonesArray* jet;     //->  jets
   Event(): TObject() {
      //NOTE: run = 0; evt = 0; njet = 0;
      jet = new TClonesArray("Jet");
      clear();
   }
  ~Event() override {
      delete jet;    jet = 0;
   }
   void clear() {
      run   = 0;
      evt   = 0;
      njet  = 0;
      jet   ->Clear();
   }
   void AddJet(const Jet* jet_) {
      njet = jet->GetLast()+1;
      new ((*jet)[njet]) Jet(*jet_);
      njet = jet->GetLast()+1;                    // update nmu
   }
   ClassDefOverride(Event,1)
};

//------------------------ jetclass.C ----------------------------

//#include "jetclass.h"

#include "TROOT.h"
#include <TFile.h>
#include <TTree.h>

#include <iostream>

using std::cout;     using std::endl;

// #4 C/C++ macro ClassImp
ClassImp(Jet)
ClassImp(Event)

void jetwrite(const char* ofname="jetclass.root", Int_t num=5)
{
   TFile* o = new TFile(ofname,"recreate");
   cout<< "jetwrite: output file " << o->GetName() <<endl;
   
   TTree* tree = new TTree("t","jet tree");
   
   Event* ev = new Event();
   
   tree->Bronch("ev", "Event", &ev);
   
   cout<< "jetwrite: start loop" <<endl;
   
   for (int jentry=0; jentry<num; ++jentry)
   {
      cout<< "jentry = " << jentry <<endl;
      
      ev->clear();
      
      Jet* jet = new Jet(100.+jentry);
      
      for (int imu=0; imu<jentry; ++imu) {
         jet->mu_pt[imu]   = 100+jentry;
         jet->mu_nseg[imu] = jentry/3;
         jet->nmu = imu+1;
      }
      for (int iv=0; iv<jentry; ++iv) {
         jet->v_x[iv] = 0.01*jentry;
         jet->v_y[iv] = 0.02*jentry;
         jet->v_z[iv] = jentry;
         jet->nvert = iv+1;
      }
      jet->nbest = jentry-1;
      cout<< "jet->nbest = " << jet->nbest <<endl;
      if (jet->nbest >= 0) cout<< "          jet->v_z[nbest] = " << jet->v_z[jet->nbest] <<endl;
      
      ev->run = 1000 + jentry;
      ev->evt = jentry;
      ev->AddJet(jet);
      
      delete jet;
      
      tree->Fill();
   }
   
   delete ev;
   
   cout<< "jetwrite: Filled events: " << tree->GetEntries() <<endl;
   
   TFile* ofile = tree->GetCurrentFile();
   if (ofile) {
      ofile->Write();
      ofile->Close();
   }
}


//------------------------ jetclass_linkdef.h ------------------------------


#ifdef __MAKECINT__
#pragma link C++ class Jet+;
#pragma link C++ class Event+;
#endif

/*
//------------------------ error message -----------------------------------

 *** Break *** segmentation violation
 Generating stack trace...
 0x0087a133 in TStreamerInfo::GetValueClones(TClonesArray*, int, int, int, int) const + 0x61 from /usr/local/root/lib/libCore.so
 0x0553e0f5 in TBranchElement::GetValue(int, int, bool) const + 0x145 from /usr/local/root/lib/libTree.so
 0x05555fbc in TLeafElement::GetValue(int) const + 0x26 from /usr/local/root/lib/libTree.so
 0x0728b04f in TTreeFormula::EvalInstance(int, char const**) + 0x169 from /usr/local/root/lib/libTreePlayer.so
 0x0727bd32 in TSelectorDraw::ProcessFillMultiple(long long) + 0xe4 from /usr/local/root/lib/libTreePlayer.so
 0x0727ba9f in TSelectorDraw::ProcessFill(long long) + 0x53 from /usr/local/root/lib/libTreePlayer.so
 0x0729981c in TTreePlayer::Process(TSelector*, char const*, long long, long long) + 0x23c from /usr/local/root/lib/libTreePlayer.so
 0x0729393d in TTreePlayer::DrawSelect(char const*, char const*, char const*, long long, long long) + 0x4b7 from /usr/local/root/lib/libTreePlayer.so
 0x05562cf7 in TTree::Draw(char const*, char const*, char const*, long long, long long) + 0x63 from /usr/local/root/lib/libTree.so
 0x0557ef49 in <unknown> from /usr/local/root/lib/libTree.so
 0x00181ccb in G__call_cppfunc + 0x2a5 from /usr/local/root/lib/libCint.so
 0x0017142d in G__interpret_func + 0x721 from /usr/local/root/lib/libCint.so
 0x00155db5 in G__getfunction + 0x134b from /usr/local/root/lib/libCint.so
 0x001e7115 in G__getstructmem + 0x80d from /usr/local/root/lib/libCint.so
 0x001dec7f in G__getvariable + 0x4b7 from /usr/local/root/lib/libCint.so
 0x0014d0a3 in G__getitem + 0x4e8 from /usr/local/root/lib/libCint.so
 0x0014d299 in G__getitem + 0x6de from /usr/local/root/lib/libCint.so
 0x0014bdbf in G__getexpr + 0x7103 from /usr/local/root/lib/libCint.so
 0x0019991c in G__exec_function + 0x1d5 from /usr/local/root/lib/libCint.so
 0x001a05b8 in G__exec_statement + 0x23bd from /usr/local/root/lib/libCint.so
 0x00135265 in G__exec_tempfile_core + 0x2bd from /usr/local/root/lib/libCint.so
 0x00135436 in G__exec_tempfile_fp + 0x22 from /usr/local/root/lib/libCint.so
 0x001a8a88 in G__process_cmd + 0x4859 from /usr/local/root/lib/libCint.so
 0x008554c1 in TCint::ProcessLine(char const*, TInterpreter::EErrorCode*) + 0xa9 from /usr/local/root/lib/libCore.so
 0x007aa7e7 in TApplication::ProcessLine(char const*, bool, int*) + 0x66b from /usr/local/root/lib/libCore.so
 0x005e85a9 in TRint::HandleTermInput() + 0x1dd from /usr/local/root/lib/libRint.so
 0x005e733e in TTermInputHandler::Notify() + 0x24 from /usr/local/root/lib/libRint.so
 0x005e90b0 in TTermInputHandler::ReadNotify() + 0x12 from /usr/local/root/lib/libRint.so
 0x008fded9 in TUnixSystem::CheckDescriptors() + 0x14f from /usr/local/root/lib/libCore.so
 0x008fce41 in TUnixSystem::DispatchOneEvent(bool) + 0x157 from /usr/local/root/lib/libCore.so
 0x00814f28 in TSystem::InnerLoop() + 0x18 from /usr/local/root/lib/libCore.so
 0x00814ecd in TSystem::Run() + 0x6f from /usr/local/root/lib/libCore.so
 0x007ab456 in TApplication::Run(bool) + 0x32 from /usr/local/root/lib/libCore.so
 0x005e8088 in TRint::Run(bool) + 0x3a8 from /usr/local/root/lib/libRint.so
 0x08048e4d in main + 0x71 from /usr/local/root/bin/root.exe
 0x04c76770 in __libc_start_main + 0xf0 from /lib/tls/libc.so.6
 0x08048d4d in TApplicationImp::ShowMembers(TMemberInspector&, char*) + 0x31 from /usr/local/root/bin/root.exe








*/