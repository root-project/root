#include <TParticle.h>
#include <TTree.h>
#include <TFile.h>

void execReuseTree(Int_t npart=100)
{
   TParticle *fParticleBuffer=0;
   TFile f("pcbug.root","recreate");
   TTree *fTreeK = new TTree("TreeK","Kinematic Tree");
   
   TBranch *branch = fTreeK->GetBranch("Particles");
   if(!branch) {
      printf ("Creating Branch\n");
      branch = fTreeK->Branch("Particles", &fParticleBuffer, 4000);
   } else {
      printf("Branch should not be there!\n");
      exit(1);
   }
   
   fParticleBuffer = new TParticle();
   for(Int_t i=0; i<npart; ++i) {
      fTreeK->Fill();
   }
   
   for(Int_t i=npart-1; i>=0; --i) {
      fParticleBuffer = 0;
      fTreeK->GetEntry(i);
      if(!fParticleBuffer) printf("Bug!!!!\n");
   }
   
   fTreeK->Write();
   f.Close();
}
