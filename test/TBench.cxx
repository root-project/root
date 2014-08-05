// set of classes to compare the performance of STL vector versus
// native Root TClonesArray.
// See main program bench.cxx

#include "TRandom.h"
#include "TFile.h"
#include "TTree.h"
#include "TClass.h"
//the next include must be the last one on systems like Windows/NT
#include "TBench.h"
#include "Riostream.h"

THit hit;
#ifdef R__HPUX
namespace std {
   using ::make_pair;
}
#endif
#ifdef R__WIN32
const char *demofile = "bench.root";
#else
const char *demofile = "bench.root";
#endif
const char* demofile_name(const char* tit)  {
   static std::string fn;
#ifdef R__WIN32
   fn = "bench.";
#else
   fn = "bench.";
#endif
   fn += tit;
   fn += ".root";
   return fn.c_str();
}
namespace {
   struct Counter  {
      std::string name;
      int count;
      Counter(const std::string& n) : name(n), count(0) {}
      ~Counter()  {
         print();
      }
      void print(const std::string& msg="")  {
         std::cout << msg << " --- Counter: " << name << " " << count << std::endl;
      }
   };
}

Counter hitCount("THit");

//-------------------------------------------------------------
ClassImp(THit)
//-------------------------------------------------------------
THit::THit() {
   fPulses = 0;
   fNpulses = 0;
   hitCount.count++;
}
THit::THit(const THit &hit) {
   hitCount.count++;
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

THit& THit::operator=(const THit& hit)  {
   fX = hit.fX;
   fY = hit.fY;
   fZ = hit.fZ;
   for (Int_t i=0;i<10;i++) fTime[i] = hit.fTime[i];
   fPulses = 0;
   fNpulses = hit.fNpulses;
   if (fNpulses == 0) return *this;
   if (hit.fPulses == 0) return *this;
   if ( fPulses ) delete [] fPulses;
   fPulses = new int[fNpulses];
   for (int j=0;j<fNpulses;j++) fPulses[j] = hit.fPulses[j];
   return *this;
}

THit::THit(int t) {
   hitCount.count++;
   fPulses = 0;
   Set(t);
}

THit::~THit() {
   hitCount.count--;
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

#if 0
#if defined(R__TEMPLATE_OVERLOAD_BUG)
template <>
#endif
TBuffer &operator>>(TBuffer &buf, THit *&obj)
{
   obj = new THit();
   obj->Streamer(buf);
   return buf;
}
#endif

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
   Clear();
   hitCount.print();
}

void TSTLhit::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhit::MakeEvent(int /*ievent*/)
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
      f = new TFile(demofile_name("TSTLhit"),"recreate","STLhit",compression);
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
   f = new TFile(demofile_name("TSTLhit"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhit::ReadTree()
{
   TSTLhit *top = this;
   TFile *f = new TFile(demofile_name("TSTLhit"));
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
ClassImp(TSTLhitList)
//-------------------------------------------------------------
TSTLhitList::TSTLhitList()
{
}

TSTLhitList::TSTLhitList(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitList::~TSTLhitList() {
   Clear();
   hitCount.print();
}

void TSTLhitList::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhitList::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.push_back(hit);
   }
}

Int_t TSTLhitList::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitList *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitList"),"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitList",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitList"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitList::ReadTree()
{
   TSTLhitList *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitList"));
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
ClassImp(TSTLhitDeque)
//-------------------------------------------------------------
TSTLhitDeque::TSTLhitDeque()
{
}

TSTLhitDeque::TSTLhitDeque(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitDeque::~TSTLhitDeque() {
   Clear();
   hitCount.print();
}

void TSTLhitDeque::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhitDeque::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.push_back(hit);
   }
}

Int_t TSTLhitDeque::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitDeque *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitDeque"),"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitDeque",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitDeque"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitDeque::ReadTree()
{
   TSTLhitDeque *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitDeque"));
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
ClassImp(TSTLhitSet)
//-------------------------------------------------------------
TSTLhitSet::TSTLhitSet()
{
}

TSTLhitSet::TSTLhitSet(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitSet::~TSTLhitSet() {
   Clear();
   hitCount.print();
}

void TSTLhitSet::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhitSet::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.insert(hit);
   }
}

Int_t TSTLhitSet::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitSet *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitSet"),"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitSet",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitSet"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitSet::ReadTree()
{
   TSTLhitSet *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitSet"));
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
ClassImp(TSTLhitMultiset)
//-------------------------------------------------------------
TSTLhitMultiset::TSTLhitMultiset()
{
}

TSTLhitMultiset::TSTLhitMultiset(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitMultiset::~TSTLhitMultiset() {
   Clear();
   hitCount.print();
}

void TSTLhitMultiset::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhitMultiset::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.insert(hit);
   }
}

Int_t TSTLhitMultiset::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitMultiset *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitMultiset"),"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitMultiset",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitMultiset"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitMultiset::ReadTree()
{
   TSTLhitMultiset *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitMultiset"));
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
ClassImp(TSTLhitMap)
//-------------------------------------------------------------
TSTLhitMap::TSTLhitMap()
{
}

TSTLhitMap::TSTLhitMap(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitMap::~TSTLhitMap() {
   Clear();
   hitCount.print();
}

void TSTLhitMap::Clear(Option_t *)
{
   fList1.clear();
}

void TSTLhitMap::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.insert(std::pair<const Int_t ,THit> (j,hit));
   }
}

Int_t TSTLhitMap::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitMap *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitMap"),"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitMap",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitMap"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitMap::ReadTree()
{
   TSTLhitMap *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitMap"));
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
ClassImp(TSTLhitMultiMap)
//-------------------------------------------------------------
TSTLhitMultiMap::TSTLhitMultiMap()
{
}

TSTLhitMultiMap::TSTLhitMultiMap(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitMultiMap::~TSTLhitMultiMap() {
   Clear();
   hitCount.print();
}

void TSTLhitMultiMap::Clear(Option_t *)
{
   fList1.clear();
}

void TSTLhitMultiMap::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      std::pair <const int, THit> temp(j,hit);
      fList1.insert(temp);
   }
}

Int_t TSTLhitMultiMap::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitMultiMap *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitMultiMap"),"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitMultiMap",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitMultiMap"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitMultiMap::ReadTree()
{
   TSTLhitMultiMap *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitMultiMap"));
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

#if 0
//-------------------------------------------------------------
ClassImp(TSTLhitHashSet)
//-------------------------------------------------------------
TSTLhitHashSet::TSTLhitHashSet()
{
}

TSTLhitHashSet::TSTLhitHashSet(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitHashSet::~TSTLhitHashSet() {
   Clear();
   hitCount.print();
}

void TSTLhitHashSet::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhitHashSet::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.insert(hit);
   }
}

Int_t TSTLhitHashSet::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitHashSet *top = this;
   if (mode > 0) {
      f = new TFile(demofile,"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitHashSet",&top,64000,split);
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

Int_t TSTLhitHashSet::ReadTree()
{
   TSTLhitHashSet *top = this;
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
ClassImp(TSTLhitHashMultiSet)
//-------------------------------------------------------------
TSTLhitHashMultiSet::TSTLhitHashMultiSet()
{
}

TSTLhitHashMultiSet::TSTLhitHashMultiSet(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitHashMultiSet::~TSTLhitHashMultiSet() {
   Clear();
   hitCount.print();
}

void TSTLhitHashMultiSet::Clear(Option_t *)
{
   fList1.erase(fList1.begin(),fList1.end());
}

void TSTLhitHashMultiSet::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      hit.Set(j);
      fList1.insert(hit);
   }
}

Int_t TSTLhitHashMultiSet::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitHashMultiSet *top = this;
   if (mode > 0) {
      f = new TFile(demofile,"recreate","STLhit",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitHashMultiSet",&top,64000,split);
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

Int_t TSTLhitHashMultiSet::ReadTree()
{
   TSTLhitHashMultiSet *top = this;
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
#endif

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
   Clear();
   hitCount.print();
}

void TSTLhitStar::Clear(Option_t *)
{
   for (std::vector<THit*>::iterator it = fList2.begin(); it<fList2.end(); it++) {
      delete (*it);
   }
   fList2.erase(fList2.begin(),fList2.end());
   // hitCount.print("Clear");
}

void TSTLhitStar::MakeEvent(int /*ievent*/)
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
      f = new TFile(demofile_name("TSTLhitStar"),"recreate","STLhitStar",compression);
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
   f = new TFile(demofile_name("TSTLhitStar"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitStar::ReadTree()
{
   TSTLhitStar *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitStar"));
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
ClassImp(TSTLhitStarList)
//-------------------------------------------------------------
TSTLhitStarList::TSTLhitStarList()
{
}

TSTLhitStarList::TSTLhitStarList(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitStarList::~TSTLhitStarList() {
   Clear();
   hitCount.print();
}

void TSTLhitStarList::Clear(Option_t *)
{
   for (std::list<THit*>::iterator it = fList2.begin(); it!=fList2.end(); it++) {
      delete (*it);
   }
   fList2.erase(fList2.begin(),fList2.end());
}

void TSTLhitStarList::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      fList2.push_back(new THit(j));
   }
}

Int_t TSTLhitStarList::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitStarList *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitStarList"),"recreate","STLhitStar",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitStarList",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitStarList"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitStarList::ReadTree()
{
   TSTLhitStarList *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitStarList"));
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
ClassImp(TSTLhitStarDeque)
//-------------------------------------------------------------
TSTLhitStarDeque::TSTLhitStarDeque()
{
}

TSTLhitStarDeque::TSTLhitStarDeque(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitStarDeque::~TSTLhitStarDeque() {
   Clear();
   hitCount.print();
}

void TSTLhitStarDeque::Clear(Option_t *)
{
   for (deque<THit*>::iterator it = fList2.begin(); it!=fList2.end(); it++) {
      delete (*it);
   }
   fList2.erase(fList2.begin(),fList2.end());
}

void TSTLhitStarDeque::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      fList2.push_back(new THit(j));
   }
}

Int_t TSTLhitStarDeque::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitStarDeque *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitStarDeque"),"recreate","STLhitStar",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitStarDeque",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitStarDeque"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitStarDeque::ReadTree()
{
   TSTLhitStarDeque *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitStarDeque"));
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
ClassImp(TSTLhitStarSet)
//-------------------------------------------------------------
TSTLhitStarSet::TSTLhitStarSet()
{
}

TSTLhitStarSet::TSTLhitStarSet(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitStarSet::~TSTLhitStarSet() {
   Clear();
   hitCount.print();
}

void TSTLhitStarSet::Clear(Option_t *)
{
   for (set<THit*>::iterator it = fList2.begin(); it!=fList2.end(); it++) {
      delete (*it);
   }
   fList2.erase(fList2.begin(),fList2.end());
   //hitCount.print("End of Clear");
}

void TSTLhitStarSet::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      fList2.insert(new THit(j));
   }
}

Int_t TSTLhitStarSet::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitStarSet *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitStarSet"),"recreate","STLhitStar",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitStarSet",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitStarSet"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitStarSet::ReadTree()
{
   TSTLhitStarSet *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitStarSet"));
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
ClassImp(TSTLhitStarMultiSet)
//-------------------------------------------------------------
TSTLhitStarMultiSet::TSTLhitStarMultiSet()
{
}

TSTLhitStarMultiSet::TSTLhitStarMultiSet(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitStarMultiSet::~TSTLhitStarMultiSet() {
   Clear();
   hitCount.print();
}

void TSTLhitStarMultiSet::Clear(Option_t *)
{
   for (multiset<THit*>::iterator it = fList2.begin(); it!=fList2.end(); it++) {
      delete (*it);
   }
   fList2.erase(fList2.begin(),fList2.end());
}

void TSTLhitStarMultiSet::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      fList2.insert(new THit(j));
   }
}

Int_t TSTLhitStarMultiSet::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitStarMultiSet *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitStarMultiSet"),"recreate","STLhitStar",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitStarMultiSet",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitStarMultiSet"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitStarMultiSet::ReadTree()
{
   TSTLhitStarMultiSet *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitStarMultiSet"));
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
ClassImp(TSTLhitStarMap)
//-------------------------------------------------------------
TSTLhitStarMap::TSTLhitStarMap()
{
}

TSTLhitStarMap::TSTLhitStarMap(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitStarMap::~TSTLhitStarMap() {
   Clear();
   hitCount.print();
}

void TSTLhitStarMap::Clear(Option_t *)
{
   for (std::map<int,THit*>::iterator it = fList2.begin(); it!=fList2.end(); it++) {
      delete (*it).second;
   }
   fList2.clear();
}

void TSTLhitStarMap::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      fList2.insert(std::pair<const Int_t, THit*> (j,new THit(j)));
   }
}

Int_t TSTLhitStarMap::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitStarMap *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitStarMap"),"recreate","STLhitStar",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitStarMap",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitStarMap"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitStarMap::ReadTree()
{
   TSTLhitStarMap *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitStarMap"));
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
ClassImp(TSTLhitStarMultiMap)
//-------------------------------------------------------------
TSTLhitStarMultiMap::TSTLhitStarMultiMap()
{
}

TSTLhitStarMultiMap::TSTLhitStarMultiMap(Int_t nmax)
{
   fNhits = nmax;
}

TSTLhitStarMultiMap::~TSTLhitStarMultiMap() {
   Clear();
   hitCount.print();
}

void TSTLhitStarMultiMap::Clear(Option_t *)
{
   for (multimap<int,THit*>::iterator it = fList2.begin(); it!=fList2.end(); it++) {
      delete (*it).second;
   }
   fList2.clear();
}

void TSTLhitStarMultiMap::MakeEvent(int /*ievent*/)
{
   Clear();
   for (Int_t j=0; j<fNhits; j++) {
      std::pair<const int,THit*> temp(j,new THit(j));
      fList2.insert(temp);
   }
}

Int_t TSTLhitStarMultiMap::MakeTree(int mode, int nevents, int compression, int split, float &cx)
{
   TFile *f=0;
   TTree *T=0;
   TSTLhitStarMultiMap *top = this;
   if (mode > 0) {
      f = new TFile(demofile_name("TSTLhitStarMultiMap"),"recreate","STLhitStar",compression);
      T = new TTree("T","Demo tree");
      T->Branch("event","TSTLhitStarMultiMap",&top,64000,split);
   }
   for (int ievent=0; ievent<nevents; ievent++) {
      MakeEvent(ievent);
      if (mode > 0) T->Fill();
   }

   if (mode == 0) return 0;
   T->Write();
   delete f;
   f = new TFile(demofile_name("TSTLhitStarMultiMap"));
   Int_t nbytes = f->GetEND();
   cx = f->GetCompressionFactor();
   delete f;
   return nbytes;
}

Int_t TSTLhitStarMultiMap::ReadTree()
{
   TSTLhitStarMultiMap *top = this;
   TFile *f = new TFile(demofile_name("TSTLhitStarMultiMap"));
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
   Clear();
   hitCount.print();
}

void TCloneshit::Clear(Option_t *)
{
   fList3->Delete();
   //fList3->Clear();
}

void TCloneshit::MakeEvent(int /*ievent*/)
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

