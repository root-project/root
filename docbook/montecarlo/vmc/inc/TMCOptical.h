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

#ifndef ROOT_TMCOptical
#define ROOT_TMCOptical
//
// Enums for setting of optical photon physics
//
#include "Rtypes.h"

enum EMCOpSurfaceModel
{
   kGlisur,                      // original GEANT3 model
   kUnified                      // UNIFIED model
};

enum EMCOpSurfaceType
{
   kDielectric_metal,            // dielectric-metal interface
   kDielectric_dielectric,       // dielectric-dielectric interface
   kFirsov,                      // for Firsov Process
   kXray                         // for x-ray mirror process
};

enum EMCOpSurfaceFinish
{
   kPolished,                    // smooth perfectly polished surface
   kPolishedfrontpainted,        // smooth top-layer (front) paint
   kPolishedbackpainted,         // same is 'polished' but with a back-paint
   kGround,                      // rough surface
   kGroundfrontpainted,          // rough top-layer (front) paint
   kGroundbackpainted            // same as 'ground' but with a back-paint
};

#endif //ROOT_TMCOPtical
