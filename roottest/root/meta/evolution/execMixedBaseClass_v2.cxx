class ParticleState {
public:
   ParticleState() : fX(15),fY(16),fZ(17) {}
   virtual ~ParticleState() {}
   float fX;
   float fY;
   float fZ;
   ClassDef(ParticleState,10);
};

class Particle : public ParticleState {
public:
   Particle() : fE(0) {}
   float fE;
   ClassDefOverride(Particle,12);
};

class ParticleImpl : public Particle
{
   // Intentionally not incremented to see if we provoke an
   // warning message.
   ClassDefOverride(ParticleImpl,10);
};

class ParticleImplVec : public Particle
{
   // Intentionally not incremented to provoke an
   // warning message.
   ClassDefOverride(ParticleImplVec,10);
};

#include <vector>

class LeafCandidate {
public:
   LeafCandidate() {}
   virtual ~LeafCandidate() {}
   ParticleState m_state;
   ClassDef(LeafCandidate,12);
};

class CompositeRefCandidateT : public LeafCandidate {

   // Intentionally not versioned.
   //   ClassDef(CompositeRefCandidateT,10);
};


class GenParticle : public CompositeRefCandidateT {

   ClassDefOverride(GenParticle,11);
};

class Holder {
public:
   Holder() { fMultiple.resize(3); fParticles.resize(4); fCandidates.resize(5); }
   virtual ~Holder() {}

   ParticleImpl                 fSingle;
   std::vector<Particle>        fEmpty; // must stay empty.
   std::vector<ParticleImplVec> fMultiple;
   std::vector<Particle>        fParticles;
   std::vector<GenParticle>     fCandidates;

   ClassDef(Holder,11);
};

#include "TFile.h"
#include "TTree.h"

void write(const char *filename = "mixedBase_v2.root")
{
   TFile *f = TFile::Open(filename,"RECREATE");
   if (!f) return;
   Particle p;
   f->WriteObject(&p,"part");
   Holder h;
   f->WriteObject(&h,"holder");
   CompositeRefCandidateT cand;
   f->WriteObject(&cand,"cand");

   TTree *t = new TTree("t","t");
   t->Branch("holder.",&h);
   t->Branch("cand.",&cand);
   t->Fill();

   f->Write();
   delete f;
}

void read(const char *filename = "mixedBase_v2.root")
{
   TFile *f = TFile::Open(filename,"READ");
   if (!f) return;
   Particle *p;
   f->GetObject("part",p);
   Holder *h;
   f->GetObject("holder",h);
   CompositeRefCandidateT *cand;
   f->GetObject("cand",cand);

   TTree *t;
   f->GetObject("t",t);
   t->GetEntry(0);

   delete f;
}

void execMixedBaseClass_v2(const char *filename = 0) {
   if (filename) {
      read(filename);
   } else {
      write();
      read("mixedBase_v1.root");
      read("mixedBase_v2.root");
   }
}

