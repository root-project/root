/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

// This linkdef is used to generate the skeleton for the customized
// dictionary to emulated access to the templated methods TTree::Branch
// and TTree::SetBranchAddress

// The procedure is as follow (this should be needed only if and when the
// dictionary format changes).

//     rm tree/src/ManualTree2.cxx
//     gmake tree/src/ManualTree2.cxx
//     replace the implementation of the 2 wrappers by 
//         #include "ManualTree2Body.h"
//     you might have to update the syntax in ManualTree2Body.cxx
//
//     You also we need to remove 
//       #include "base/inc/TROOT.h"
//       #include "base/inc/TDataMember.h"
//     from tree/inc/ManualTree2.h
//     and replace the string "SetBranchAddress<void>" by "SetBranchAddress"
//
//     Remove from tree/inc/ManualTree2.h
//       #define G__PRIVATE_GVALUE

#pragma link C++ function TTree::Branch(const char *, const char *, void **, Int_t, Int_t);
#pragma link C++ function TTree::Branch(const char *, void **, Int_t, Int_t);
#pragma link C++ function TTree::SetBranchAddress(const char*, void**);
#pragma link C++ function TTree::Process(void*, Option_t*, Long64_t, Long64_t);
#pragma link C++ function TChain::Process(void*, Option_t*, Long64_t, Long64_t);
#endif
