// Author: Gerardo Ganis   3/12/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_AuthConst
#define ROOT_AuthConst


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AuthConst                                                            //
//                                                                      //
// Const used in authentication business                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// Number of security levels and masks
// (should be the same as in rpdutils/inc/rpdp.h)
const Int_t       kMAXSEC         = 6;
const Int_t       kMAXSECBUF      = 2048;
const Int_t       kAUTH_REUSE_MSK = 0x1;
const Int_t       kAUTH_CRYPT_MSK = 0x2;
const Int_t       kAUTH_SSALT_MSK = 0x4;
const Int_t       kMAXRSATRIES    = 100;

#endif
