// @(#)root/mc:$Name:  $:$Id: TMCParticleType.h,v 1.4 2003/02/26 13:30:14 brun Exp $
// Author: Alice collaboration  

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. */

#ifndef ROOT_TMCParticleType
#define ROOT_TMCParticleType

// 
// List of MonteCarlo particle types
// (used when adding user own particles to MC) 
//

#include "Rtypes.h"
   
enum TMCParticleType
{
  kPTGamma,          // particle tracked as gamma
  kPTElectron,       // particle tracked as electron
  kPTNeutron,        // particle tracked as neutron 
  kPTHadron,         // particle tracked as hadron
  kPTMuon,           // particle tracked as muon
  kPTGeantino,       // particle tracked as geantino
  kPTChargedGeantino,// particle tracked as charged geantino
  kPTOpticalPhoton,  // particle tracked as light photon
  kPTIon,            // particle tracked as heavy ion
  kPTUndefined       // unknown particle type
};  

#endif //ROOT_TMCParticleType
