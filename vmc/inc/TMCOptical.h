// @(#)root/vmc:$Name: v4-03-02 $:$Id: TMCProcess.h,v 1.1 2003/07/15 09:56:58 brun Exp $
// Author: Alice collaboration  

#ifndef ROOT_TMCOptical
#define ROOT_TMCOptical
// 
// Enums for setting of optical photon physics
//
#include "Rtypes.h"
   
enum TMCOpSurfaceModel
{
   kGlisur,                      // original GEANT3 model
   kUnified                      // UNIFIED model
};

enum TMCOpSurfaceType
{
   kDielectric_metal,            // dielectric-metal interface
   kDielectric_dielectric,       // dielectric-dielectric interface
   kFirsov,                      // for Firsov Process
   kXray                         // for x-ray mirror process
};

enum TMCOpSurfaceFinish
{
   kPolished,                    // smooth perfectly polished surface
   kPolishedfrontpainted,        // smooth top-layer (front) paint
   kPolishedbackpainted,         // same is 'polished' but with a back-paint
   kGround,                      // rough surface
   kGroundfrontpainted,          // rough top-layer (front) paint
   kGroundbackpainted            // same as 'ground' but with a back-paint
};

#endif //ROOT_TMCOPtical
