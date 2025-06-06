class Particle {
public:
   Particle() : fX(5),fY(6),fZ(7) {}
   virtual ~Particle() {}
   float fX;
   float fY;
   float fZ;
   ClassDef(Particle,11);
};

class ParticleImpl : public Particle
{
   ClassDefOverride(ParticleImpl,10);
};

class ParticleImplVec : public Particle
{
   ClassDefOverride(ParticleImplVec,10);
};

#include <vector>

class LeafCandidate {
public:
   LeafCandidate() : fX(-5),fY(-6),fZ(-7) {}
   virtual ~LeafCandidate() {}
   float fX;
   float fY;
   float fZ;
   ClassDef(LeafCandidate,11);
};

class CompositeRefCandidateT : public LeafCandidate {

   // Intentionally not versioned.
};

class GenParticle : public CompositeRefCandidateT {

   ClassDefOverride(GenParticle,10);
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

   ClassDef(Holder,10);
};

#include "TFile.h"
#include "TTree.h"

void write(const char *filename = "mixedBase_v1.root")
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

void read(const char *filename = "mixedBase_v1.root")
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

void execMixedBaseClass_v1() {
   write();
   read();
}
