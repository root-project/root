#include "TObject.h"
#include "TClonesArray.h"
#include "TBigDSWriteParticle.h"
#include "TBigDSWriteEvent.h"

//////////////////////////////////////////////////////////////////////////
// This class contain data about one measured event. There are methods 
// for setting all data members and methods which returns values of data
// members.
// WARNING : Method AddParticle also increase value of fNParticles, so 
//           you don't need to set number of particles in event, this is
//           set automatically.
//
// Date: 6.10.2003
// Author: Ondrej Chvala, based on TWriteEvent and TBigDSWriteEvent by Michal Kreps
//

ClassImp(TBigDSWriteEvent)

TBigDSWriteEvent::TBigDSWriteEvent()
{
/*
 * Default constructor. Set all members in event to 0 and init array for
 * particles.
 */
  fNRun=0;
  fNEvent=0;
  fNParticles=0;
  fNGrey=0;
  fEveto=0;
  fNRun=0;
  fNEvent=0;
  fbi=0;
  fbg=0;
  ftriggers=0;
  fvtxx=0;
  fvtxy=0;
  fvtxz=0;
  //if(!fgParticles) 
  fParticles = new TClonesArray("TBigDSWriteParticle",1000);
//  fgParticles = fParticles;
  fpart_iter = new TIter(fParticles);
}

TBigDSWriteEvent::~TBigDSWriteEvent()
{
// Destructor, deletes also all particles.
  delete fpart_iter;
  delete fParticles;
}

Int_t TBigDSWriteEvent::GetWFABeam(Int_t slot) 
{ 
  if(slot < 20) { 
    return fWFABeam[slot];
  } else {
    return -999;
  }
}

Int_t TBigDSWriteEvent::GetWFAInt(Int_t slot) 
{ 
  if(slot < 5) {
    return fWFAInt[slot];
  } else {
    return -999;
  }
}

Int_t TBigDSWriteEvent::SetWFABeam(Int_t wfa_b, Int_t slot) 
{ 
  if(slot < 20) { 
    fWFABeam[slot]=wfa_b;
    return 0;
  } else {
    return -1;
  }
}

Int_t TBigDSWriteEvent::SetWFAInt(Int_t wfa_i, Int_t slot) 
{ 
  if(slot < 5) {
    fWFAInt[slot]=wfa_i;
    return 0;
  } else {
    return -1;
  }
}
    

TBigDSWriteParticle *TBigDSWriteEvent::AddParticle(Float_t px, Float_t py, Float_t pz)
{
//
// Method which add one measured particle to event. Arguments are 3 momenta
// of this particle and method returns pointer to this new particle. This
// pointer is usefull for setting another data members of this particle.
// Method has one side effect, namely automatic increase of number of
// particles in event.
//
  TClonesArray &track=*fParticles;
  new(track[fNParticles++]) TBigDSWriteParticle(px,py,pz);
  return ((TBigDSWriteParticle*)fParticles->At(fNParticles-1));
}
                                                                                                                                               
                                                                                                                                               
TBigDSWriteParticle *TBigDSWriteEvent::AddParticle(TBigDSWriteParticle *anopart)
{
//
// Method which add one measured particle to event. Argument is another
// measured particle which is used for setting data members of new one
// particle. Here is also side effect which increase fNParticles.
// Return value is pointer to added particle.
//
  TClonesArray &track=*fParticles;
  new(track[fNParticles++]) TBigDSWriteParticle(anopart);
  return ((TBigDSWriteParticle*)fParticles->At(fNParticles-1));
}
                                                                                                                                               
// void TBigDSWriteEvent::Copy(TBigDSWriteEvent *anoevent)
// {
// //
// // Method which copy another event to this one. During this operation is
// // previous information in this event destroyed.
// //
//   Int_t npart;
//   TClonesArray *array;
                                                                                                                                               
//   npart=anoevent->GetNParticles();
// //   array=anoevent->GetParticles();
//   fNParticles=0;
//   for (int i=0;i<npart;i++)
//     this->AddParticle((TBigDSWriteParticle*)array->At(i));
    
//   fNRun=anoevent->GetNRun();
//   fNEvent=anoevent->GetNEvent();
//   fNGrey=anoevent->GetNGrey();
//   fEveto=anoevent->GetEveto();
//   fWeight=anoevent->GetWeight();
//   fbi=anoevent->GetBi();
//   fbg=anoevent->GetBg();
//   ftriggers=anoevent->GetTriggers();
//   fvtxx=anoevent->GetVtxX();
//   fvtxy=anoevent->GetVtxY();
//   fvtxz=anoevent->GetVtxZ();
// }

inline void TBigDSWriteEvent::SetToFirst()
{
// Sets the iterator to first particle, so GetNext() method will give first
// one.
  fpart_iter->Reset();
}

inline TBigDSWriteParticle* TBigDSWriteEvent::GetNext()
{
// Returns next particle. If we are at end of the list, it returns 0.
  return( (TBigDSWriteParticle *) fpart_iter->Next() );
}

