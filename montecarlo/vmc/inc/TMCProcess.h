// @(#)root/vmc:$Id$
// Author: Alice collaboration

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMCProcess
#define ROOT_TMCProcess
//
// List of MonteCarlo processes
//
#include "Rtypes.h"

const Int_t kMaxMCProcess = 48;

enum TMCProcess
{
  kPPrimary,            // Primary interaction

  kPMultipleScattering, // multiple scattering
  kPCoulombScattering,  // single Coulomb scattering
  kPEnergyLoss,         // continuous energy loss
  kPMagneticFieldL,     // bending in mag. field
  kPDecay,              // particle decay
  kPPair,               // photon pair production or
                        // muon direct pair production
  kPCompton,            // Compton scattering
  kPPhotoelectric,      // photoelectric effect
  kPBrem,               // bremsstrahlung
  kPDeltaRay,           // delta-ray production
  kPAnnihilation,       // positron annihilation
  kPAnnihilationRest,   // positron annihilation at rest
  kPAnnihilationFlight, // positron annihilation in flight

  kPHadronic,           // hadronic interaction
  kPEvaporation,        // nuclear evaporation
  kPNuclearFission,     // nuclear fission
  kPNuclearAbsorption,  // nuclear absorption
  kPPbarAnnihilation,   // antiproton annihilation
  kPNbarAnnihilation,   // antineutron annihilation
  kPNCapture,           // neutron capture
  kPHElastic,           // hadronic elastic scattering
  kPHIElastic,          // hadronic elastic incoherent scattering
  kPHCElastic,          // hadronic elastic coherent scattering
  kPHInhelastic,        // hadronic inelastic scattering
  kPPhotonInhelastic,   // photon inelastic scattering

  kPMuonNuclear,        // muon nuclear interaction
  kPElectronNuclear,    // electron nuclear interaction
  kPPositronNuclear,    // positron nuclear interaction
  kPPhotoNuclear,       // photo nuclear interaction

  kPTOFlimit,           // exceeded time of flight cut
  kPPhotoFission,       // nuclear photofission

  kPRayleigh,           // Rayleigh scattering

  kPNull,               // no mechanism is active, usually at the entrance
                        // of a new volume
  kPStop,               // particle has fallen below energy threshold
                        // and tracking stops

  kPLightAbsorption,    // Cerenkov photon absorption
  kPLightDetection,     // Optical photon detection (absorption + photoelectron production)
  kPLightScattering,    // Cerenkov photon reflection/refraction
  kStepMax,             // step limited by STEMAX

  kPCerenkov,           // Cerenkov photon generation
  kPFeedBackPhoton,     // Feed back photon in RICH -- ALICE specific
  kPLightReflection,    // Cerenkov photon reflection
  kPLightRefraction,    // Cerenkov photon refraction
  kPSynchrotron,        // synchrotron radiation generation
  kPScintillation,      // scintillation

  kPTransportation,     // Transportation
  kPUserDefined,        // User defined process

  kPNoProcess           // unknown process
};

static const char * const TMCProcessName[kMaxMCProcess] = {
  "Primary particle emission",
  "Multiple scattering",
  "Coulomb scattering",
  "Energy loss",
  "Bending in magnetic field",
  "Decay",
  "Lepton pair production",
  "Compton scattering",
  "Photoelectric effect",
  "Bremstrahlung",
  "Delta ray",
  "Positron annihilation",
  "Positron annihilation at rest",
  "Positron annihilation in flight",
  "Hadronic interaction",
  "Nuclear evaporation",
  "Nuclear fission",
  "Nuclear absorbtion",
  "Antiproton annihilation",
  "Antineutron annihilation",
  "Neutron capture",
  "Hadronic elastic",
  "Hadronic incoherent elastic",
  "Hadronic coherent elastic",
  "Hadronic inelastic",
  "Photon inelastic",
  "Muon nuclear interaction",
  "Electron nuclear interaction",
  "Positron nuclear interaction",
  "Photo nuclear interaction",
  "Time of flight limit",
  "Nuclear photofission",
  "Rayleigh effect",
  "No active process",
  "Energy threshold",
  "Light absorption",
  "Light detection",
  "Light scattering",
  "Maximum allowed step",
  "Cerenkov production",
  "Cerenkov feed back photon",
  "Cerenkov photon reflection",
  "Cerenkov photon refraction",
  "Synchrotron radiation",
  "Scintillation",
  "Transportation",
  "User defined process",
  "Unknown process"
};

#endif //ROOT_TMCProcess
