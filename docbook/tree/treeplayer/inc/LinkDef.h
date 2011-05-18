/* @(#)root/treeplayer:$Id$ */

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
#pragma link C++ class TTreePerfStats+;
#pragma link C++ class TTreeTableInterface;

#pragma link C++ namespace ROOT;

#pragma link C++ class ROOT::TBranchProxyDirector+;
#pragma link C++ class ROOT::TBranchProxy+;
#pragma link C++ class ROOT::TFriendProxy+;

#pragma link C++ class ROOT::TFriendProxyDescriptor;
#pragma link C++ class ROOT::TBranchProxyDescriptor;
#pragma link C++ class ROOT::TBranchProxyClassDescriptor;

#pragma link C++ class ROOT::TImpProxy<double>+;
#pragma link C++ class ROOT::TImpProxy<float>+;
#pragma link C++ class ROOT::TImpProxy<UInt_t>+;
#pragma link C++ class ROOT::TImpProxy<ULong_t>+;
#pragma link C++ class ROOT::TImpProxy<UShort_t>+;
#pragma link C++ class ROOT::TImpProxy<UChar_t>+;
#pragma link C++ class ROOT::TImpProxy<Int_t>+;
#pragma link C++ class ROOT::TImpProxy<Long_t>+;
#pragma link C++ class ROOT::TImpProxy<Short_t>+;
#pragma link C++ class ROOT::TImpProxy<Char_t>+;
#pragma link C++ class ROOT::TImpProxy<Bool_t>+;

#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<double> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<float> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<UInt_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<ULong_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<UShort_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<UChar_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<Int_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<Long_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<Short_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<Char_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<Bool_t> >+;
   //specialized ! typedef TArrayProxy<ROOT::TArrayType<Char_t> >+;

#pragma link C++ class ROOT::TClaImpProxy<double>+;
#pragma link C++ class ROOT::TClaImpProxy<float>+;
#pragma link C++ class ROOT::TClaImpProxy<UInt_t>+;
#pragma link C++ class ROOT::TClaImpProxy<ULong_t>+;
#pragma link C++ class ROOT::TClaImpProxy<UShort_t>+;
#pragma link C++ class ROOT::TClaImpProxy<UChar_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Int_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Long_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Short_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Char_t>+;
#pragma link C++ class ROOT::TClaImpProxy<Bool_t>+;

#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<double> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<float> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<UInt_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<ULong_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<UShort_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<UChar_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<Int_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<Long_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<Short_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<Char_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<Bool_t> >+;

#if !defined(_MSC_VER) || (_MSC_VER>1300)
#pragma link C++ class ROOT::TImpProxy<Long64_t>+;
#pragma link C++ class ROOT::TImpProxy<ULong64_t>+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<Long64_t> >+;
#pragma link C++ class ROOT::TArrayProxy<ROOT::TArrayType<ULong64_t> >+;
#pragma link C++ class ROOT::TClaImpProxy<Long64_t>+;
#pragma link C++ class ROOT::TClaImpProxy<ULong64_t>+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<ULong64_t> >+;
#pragma link C++ class ROOT::TClaArrayProxy<ROOT::TArrayType<Long64_t> >+;
#endif

#endif
