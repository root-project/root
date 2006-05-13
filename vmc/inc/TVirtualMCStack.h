// @(#)root/vmc:$Name:  $:$Id: TVirtualMCStack.h,v 1.3 2005/09/04 09:25:01 brun Exp $
// Authors: Ivana Hrivnacova 13/04/2002

#ifndef ROOT_TVirtualMCStack
#define ROOT_TVirtualMCStack

// Class TVirtualMCStack
// ---------------------
// Interface to a user defined particles stack.
//

#include "TObject.h"
#include "TMCProcess.h"

class TParticle;

class TVirtualMCStack : public TObject {
  
public:
   // creators, destructors
   TVirtualMCStack();
   virtual ~TVirtualMCStack();

   // 
   // Methods for stacking 
   //
    
   // Create a new particle and push into stack;
   // Arguments:
   // toBeDone   - 1 if particles should go to tracking, 0 otherwise
   // parent     - number of the parent track, -1 if track is primary
   // pdg        - PDG encoding
   // px, py, pz - particle momentum [GeV/c]
   // e          - total energy [GeV]
   // vx, vy, vx - position [cm]
   // tof        - time of flight [s]    
   // polx, poly, polz - polarization
   // mech       - creator process VMC code 
   // ntr        - track number (is filled by the stack
   // weight     - particle weight 
   // is         - generation status code 
   //
   virtual void  PushTrack(Int_t toBeDone, Int_t parent, Int_t pdg,
  	              Double_t px, Double_t py, Double_t pz, Double_t e,
                      Double_t vx, Double_t vy, Double_t vz, Double_t tof,
                      Double_t polx, Double_t poly, Double_t polz,
                      TMCProcess mech, Int_t& ntr, Double_t weight,
                      Int_t is) = 0;

   // The stack has to provide two pop mechanisms:
   // PopNextTrack() - pops all particles with toBeDone = 1,
   //                  both primaries and seconadies
   // PopPrimaryForTracking() - pops only primary particles with toBeDone = 1,
   //                  stacking of secondaries is done by MC
   //
   virtual TParticle* PopNextTrack(Int_t& itrack) = 0;
   virtual TParticle* PopPrimaryForTracking(Int_t i) = 0;    

   //
   // Set methods
   //
    
   // Set current track number
   virtual void       SetCurrentTrack(Int_t trackNumber) = 0;                           

   //
   // Get methods
   //

   // Total number of tracks 
   virtual Int_t      GetNtrack()    const = 0;

   // Total number of primary tracks 
   virtual Int_t      GetNprimary()  const = 0;

   // Current track particle
   virtual TParticle* GetCurrentTrack() const= 0;    

   // Current track number
   virtual Int_t      GetCurrentTrackNumber() const = 0;

   // Number of the parent of the current track  
   virtual Int_t      GetCurrentParentTrackNumber() const = 0;
    
   ClassDef(TVirtualMCStack,1) //Interface to a particles stack
};

#endif //ROOT_TVirtualMCStack
