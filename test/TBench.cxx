// set of classes to compare the performance of STL vector versus
// native Root TClonesArray.
// See main program bench.cxx
   
#include "TRandom.h"
#include "TFile.h"
#include "TTree.h"
#include "TClass.h"
//the next include must be the last one on systems like Windows/NT
#include "TBench.h"

THit hit;
char *demofile = "/tmp/bench.root";

//-------------------------------------------------------------
ClassImp(THit)
//-------------------------------------------------------------
THit::THit() {
  fPulses = 0;
}
THit::THit(const THit &hit) {
  fX = hit.fX;
  fY = hit.fY;
  fZ = hit.fZ;
  for (Int_t i=0;i<10;i++) fTime[i] = hit.fTime[i];
  fPulses = 0;
  fNpulses = hit.fNpulses;
  if (fNpulses == 0) return;
  if (hit.fPulses == 0) return;
  fPulses = new int[fNpulses];
  for (int j=0;j<fNpulses;j++) fPulses[j] = hit.fPulses[j];
}

THit::THit(int t) {
  fPulses = 0;
  Set(t);
}

THit::~THit() {
   if (fPulses) delete [] fPulses;
   fPulses = 0;
}

void THit::Set(int t) {
  fX = gRandom->Gaus(0,1);
  fY = gRandom->Gaus(0,1);
  fZ = gRandom->Gaus(0,10);
  if (fPulses && fNpulses > 0) delete [] fPulses;
  fNpulses = t%20 + 1;
  fPulses = new int[fNpulses];
  for (int j=0;j<fNpulses;j++) fPulses[j] = j+1;
  for (int i=0; i<10; i++) fTime[i] = t+i;
}

TBuffer &operator>>(TBuffer &buf, const THit *&obj)
{
   obj =  new THit();
   ((THit*)obj)->Streamer(buf);
   return buf;
}

TBuffer &operator<<(TBuffer &buf, const THit *obj)
{
   ((THit*)obj)->Streamer(buf);
   return buf;
}
   
//-------------------------------------------------------------
ClassImp(TObjHit)
//-------------------------------------------------------------

TObjHit::TObjHit() :THit() {}
TObjHit::TObjHit(int t) :THit(t) {}

//-------------------------------------------------------------
ClassImp(TSTLhit)
//-------------------------------------------------------------
TSTLhit::TSTLhit()
{
}

TSTLhit::TSTLhit(Int_t nmax)
{
   fNhits = nmax;
   fList1.reserve(nmax);
}

TSTLhit::~TSTLhit() {
}

void TSTLhit::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhit::MakeEvent(int ievent)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.push_back(hit);
   }
}

Int_t TSTLhit::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
  TFile *f=0;
  TTree *T=0;
  TSTLhit *top = this;
  if (mode > 0) {
     f = new TFile(demofile,"recreate","STLhit",compression);  
     T = new TTree("T","Demo tree");
     T->Branch("event","TSTLhit",&top,64000,split);
  }
  for (int ievent=0; ievent<nevents; ievent++) {
     MakeEvent(ievent);
     if (mode > 0) T->Fill();
  }

  if (mode == 0) return 0;
  T->Write();
  delete f;
  f = new TFile(demofile);
  Int_t nbytes = f->GetEND();
  cx = f->GetCompressionFactor();
  delete f;
  return nbytes;
}

Int_t TSTLhit::ReadTree()
{
  TSTLhit *top = this;
  TFile *f = new TFile(demofile);  
  TTree *T = (TTree*)f->Get("T");
  T->SetBranchAddress("event",&top);
  Int_t nevents = (Int_t)T->GetEntries();
  Int_t nbytes = 0;
  for (int ievent=0; ievent<nevents; ievent++) {
     nbytes += T->GetEntry(ievent);
     Clear();
  }
  delete f;
  return nbytes;
}


//-------------------------------------------------------------
ClassImp(TSTLhitStar)
//-------------------------------------------------------------
TSTLhitStar::TSTLhitStar()
{
}

TSTLhitStar::TSTLhitStar(Int_t nmax)
{
   fNhits = nmax;
   fList2.reserve(nmax);
}

TSTLhitStar::~TSTLhitStar() {
}

void TSTLhitStar::Clear(Option_t *)
{
   for (vector<THit*>::iterator it = fList2.begin(); it<fList2.end(); it++) {
      delete (*it);
   }
   fList2.erase(fList2.begin(),fList2.end());
}

void TSTLhitStar::MakeEvent(int ievent)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      fList2.push_back(new THit(j));
   }
}

Int_t TSTLhitStar::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
  TFile *f=0;
  TTree *T=0;
  TSTLhitStar *top = this;
  if (mode > 0) {
     f = new TFile(demofile,"recreate","STLhitStar",compression);  
     T = new TTree("T","Demo tree");
     T->Branch("event","TSTLhitStar",&top,64000,split);
  }
  for (int ievent=0; ievent<nevents; ievent++) {
     MakeEvent(ievent);
     if (mode > 0) T->Fill();
  }

  if (mode == 0) return 0;
  T->Write();
  delete f;
  f = new TFile(demofile);
  Int_t nbytes = f->GetEND();
  cx = f->GetCompressionFactor();
  delete f;
  return nbytes;
}

Int_t TSTLhitStar::ReadTree()
{
  TSTLhitStar *top = this;
  TFile *f = new TFile(demofile);  
  TTree *T = (TTree*)f->Get("T");
  T->SetBranchAddress("event",&top);
  Int_t nevents = (Int_t)T->GetEntries();
  Int_t nbytes = 0;
  for (int ievent=0; ievent<nevents; ievent++) {
     nbytes += T->GetEntry(ievent);
     Clear();
  }
  delete f;
  return nbytes;
}


//-------------------------------------------------------------
ClassImp(TCloneshit)
//-------------------------------------------------------------
TCloneshit::TCloneshit()
{
   fList3 = new TClonesArray("TObjHit");
}

TCloneshit::TCloneshit(Int_t nmax)
{
   fNhits = nmax;
   fList3 = new TClonesArray("TObjHit",nmax);
   TObjHit::Class()->IgnoreTObjectStreamer();
}

TCloneshit::~TCloneshit() {
}

void TCloneshit::Clear(Option_t *)
{
   fList3->Delete();   
   //fList3->Clear();   
}

void TCloneshit::MakeEvent(int ievent)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      new((*fList3)[j]) TObjHit(j);
   }   
}

Int_t TCloneshit::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
  TFile *f=0;
  TTree *T=0;
  TCloneshit *top = this;
  if (mode > 0) {
     f = new TFile(demofile,"recreate","Cloneshit",compression);  
     T = new TTree("T","Demo tree");
     T->Branch("event","TCloneshit",&top,64000,split);
  }
  for (int ievent=0; ievent<nevents; ievent++) {
     MakeEvent(ievent);
     if (mode > 0) T->Fill();
  }

  if (mode == 0) return 0;
  T->Write();
  delete f;
  f = new TFile(demofile);
  Int_t nbytes = f->GetEND();
  cx = f->GetCompressionFactor();
  delete f;
  return nbytes;
}

Int_t TCloneshit::ReadTree()
{
  TCloneshit *top = this;
  TFile *f = new TFile(demofile);  
  TTree *T = (TTree*)f->Get("T");
  T->SetBranchAddress("event",&top);
  Int_t nevents = (Int_t)T->GetEntries();
  Int_t nbytes = 0;
  for (int ievent=0; ievent<nevents; ievent++) {
     nbytes += T->GetEntry(ievent);
  }
  delete f;
  return nbytes;
}

