/* @(#)root/treeplayer:$Name:  $:$Id: LinkDef.h,v 1.20 2006/04/28 07:09:07 pcanal Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link C++ nestedtypedefs;
#pragma link C++ nestedclasses;

#pragma link C++ class TTreePlayer+;
#pragma link C++ class TTreeFormula-;
#pragma link C++ class TSelectorDraw;
#pragma link C++ class TSelectorEntries;
#pragma link C++ class TFileDrawMap+;
#pragma link C++ class TTreeIndex-;
#pragma link C++ class TChainIndex+;
#pragma link C++ class TChainIndex::TChainIndexEntry+;
#pragma link C++ class TTreeFormulaManager;
#pragma link C++ class TTreeDrawArgsParser+;

#pragma link C++ namespace ROOT;

#pragma link C++ class ROOT::TBranchProxyDirector+;
#pragma link C++ class ROOT::TBranchProxy+;
#pragma link C++ class ROOT::TFriendProxy+;

#pragma link C++ class ROOT::TFriendProxyDescriptor;
#pragma link C++ class ROOT::TBranchProxyDescriptor;
#pragma link C++ class ROOT::TBranchProxyClassDescriptor;

#pragma link C++ class ROOT::TImpProxy<Double_t>+;
#pragma link C++ class ROOT::TImpProxy<Float_t>+;
#pragma link C++ class ROOT::TImpProxy<UInt_t>+;
#pragma link C++ class ROOT::TImpProxy<ULong_t>+;
#pragma link C++ class ROOT::TImpProxy<UShort_t>+;
#pragma link C++ class ROOT::TImpProxy<UChar_t>+;
#pragma link C++ class ROOT::TImpProxy<Int_t>+;
#pragma link C++ class ROOT::TImpProxy<Long_t>+;
#pragma link C++ class ROOT::TImpProxy<Short_t>+;
#pragma link C++ class ROOT::TImpProxy<Char_t>+;
#pragma link C++ class ROOT::TImpProxy<Bool_t>+;

#pragma link C++ class ROOT::TArrayProxy<Double_t>+;
#pragma link C++ class ROOT::TArrayProxy<Float_t>+;
#pragma link C++ class ROOT::TArrayProxy<UInt_t>+;
#pragma link C++ class ROOT::TArrayProxy<ULong_t>+;
#pragma link C++ class ROOT::TArrayProxy<UShort_t>+;
#pragma link C++ class ROOT::TArrayProxy<UChar_t>+;
#pragma link C++ class ROOT::TArrayProxy<Int_t>+;
#pragma link C++ class ROOT::TArrayProxy<Long_t>+;
#pragma link C++ class ROOT::TArrayProxy<Short_t>+;
#pragma link C++ class ROOT::TArrayProxy<Char_t>+;
#pragma link C++ class ROOT::TArrayProxy<Bool_t>+;
   //specialized ! typedef TArrayProxy<Char_t>+;

#pragma link C++ class ROOT::TClaImpProxy<Double_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Float_t>+;
#pragma link C++ class ROOT::TClaImpProxy<UInt_t>+;
#pragma link C++ class ROOT::TClaImpProxy<ULong_t>+;
#pragma link C++ class ROOT::TClaImpProxy<UShort_t>+;
#pragma link C++ class ROOT::TClaImpProxy<UChar_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Int_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Long_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Short_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Char_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Bool_t>+;

#pragma link C++ class ROOT::TClaArrayProxy<Double_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<Float_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<UInt_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<ULong_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<UShort_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<UChar_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<Int_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<Long_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<Short_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<Char_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<Bool_t>+;

#if !defined(_MSC_VER) || (_MSC_VER>1300)
#pragma link C++ class ROOT::TImpProxy<Long64_t>+;
#pragma link C++ class ROOT::TImpProxy<ULong64_t>+;
#pragma link C++ class ROOT::TArrayProxy<Long64_t>+;
#pragma link C++ class ROOT::TArrayProxy<ULong64_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Long64_t>+;
#pragma link C++ class ROOT::TClaImpProxy<ULong64_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<ULong64_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<Long64_t>+;
#endif

#endif
