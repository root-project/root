#ifndef TBigDSWriteEvent_Header

#define TBigDSWriteEvent_Header

#include "TObject.h"
#include "TClonesArray.h"

#include "TBigDSWriteParticle.h"


class TBigDSWriteEvent : public TObject
{
  protected :
    Int_t fNRun;                // Run number
    Int_t fNEvent;              // Event number
    Int_t fNParticles;          // Number of particles in event
    Int_t fbi;                  // b_i
    Int_t fbg;                  // b_g 
    Int_t ftriggers;            
    Float_t fvtxx;              // vertex X
    Float_t fvtxy;              // vertex Y
    Float_t fvtxz;              // vertex Z
    Float_t fchi2;              // chi2 of vtx fit
    
    Int_t fNGrey;               // Number of grey particles in event
    Int_t fWFABeam[20];         // beam hits in WFA
    Int_t fWFAInt[5];           // interaction hits in WFA
    Int_t fWFAflag;             // WFA flag
    Int_t fBPDflag;             // BPD flag
//    Int_t fvpc[80];           // VPCs
    Float_t fEveto;             // Veto energy 
    Float_t fWeight;
     
    TClonesArray *fParticles;            // Array with particles
/*     TClonesArray *fgParticles; */
    TIter *fpart_iter;                   // ptr to iterator
        
  public :
    TBigDSWriteEvent();                         // Default constructor
    ~TBigDSWriteEvent();                                // destrucor
    
    Int_t GetNRun() {return fNRun;};            // Get run Number
    Int_t GetNEvent() {return fNEvent;};        // Get event number
    Int_t GetNParticles() {return fNParticles;};        // Get number of particles
    Int_t GetBi() {return fbi;};                // Get number of gated interactions
    Int_t GetBg() {return fbg;};                // Get number of gated beams
    Int_t GetTriggers() {return ftriggers;};    // Get number of gated triggers
    Int_t GetNGrey() {return fNGrey;};          // Get number of grey particles
    Int_t GetWFAFlag() {return fWFAflag;};              // Get flag
    Int_t GetBPDFlag() {return fBPDflag;};              // Get flag
    Float_t GetVtxX() {return fvtxx;};          // Get X position of the main vertex
    Float_t GetVtxY() {return fvtxy;};          // Get Y position of the main vertex
    Float_t GetVtxZ() {return fvtxz;};          // Get Z position of the main vertex
    Float_t GetChi2() {return fchi2;};          // Get chi^2 the main vertex fit
    Float_t GetEveto() {return fEveto;};        // Get Veto energy
    Float_t GetWeight(){return fWeight;};       // Get weight
    
    void SetNRun(Int_t run) {fNRun=run;};       // Set run number
    void SetNEvent(Int_t event) {fNEvent=event;}; 
                                                // Set event number
    void SetNParticles(Int_t particles) {fNParticles=particles;};
                                // Set number of particles.
                                // Warning AddParticle increase value of fNParticles
    void SetNGrey(Int_t grey) {fNGrey=grey;};   // Set number of grey particles
    void SetBi(Int_t bi) {fbi=bi;};             // Set bi
    void SetBg(Int_t bg) {fbg=bg;};             // Set bg
    void SetTriggers(Int_t trigg) {ftriggers=trigg;};           // Set # of triggers
    void SetWFAFlag(Int_t wfaflag) {fWFAflag=wfaflag;};         // Set flag
    void SetBPDFlag(Int_t bpdflag) {fBPDflag=bpdflag;};         // Set flag
    void SetVtxX(Float_t vtxx) {fvtxx=vtxx;};   // Set X position of the main vertex
    void SetVtxY(Float_t vtxy) {fvtxy=vtxy;};   // Set Y position of the main vertex
    void SetVtxZ(Float_t vtxz) {fvtxz=vtxz;};   // Set Z position of the main vertex
    void SetChi2(Float_t chi2) {fchi2=chi2;};   // Set chi2the main vertex
    void SetEveto(Float_t eveto) {fEveto=eveto;}; // Set veto energy
    void SetWeight(Float_t w){fWeight=w;};

    Int_t GetWFABeam(Int_t slot);
    Int_t GetWFAInt(Int_t slot);
    Int_t SetWFABeam(Int_t wfa_b, Int_t slot);
    Int_t SetWFAInt(Int_t wfa_i, Int_t slot);

    TBigDSWriteParticle *AddParticle(Float_t px, Float_t py, Float_t pz);
                // Add particle with 3 momenta in lab
    TBigDSWriteParticle *AddParticle(TBigDSWriteParticle *anopart);
                // Add particle, data members are
                // copied from anopart
    void SetToFirst();                          // sets iterator to first particle
    TBigDSWriteParticle* GetNext();             // returns pointer to next particle
    TClonesArray *GetParticles() {return fParticles;};          // Get array of particles
    //    void Copy(TBigDSWriteEvent *anoevent);                      // Copy anoevent to this event
    void Clear(Option_t*) {fParticles->Clear();};
    void Delete(Option_t*) {fParticles->Delete();};

  ClassDef(TBigDSWriteEvent,2)
};


#endif
