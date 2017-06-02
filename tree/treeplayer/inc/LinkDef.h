/* @(#)root/treeplayer:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CLING__

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
#pragma link C++ class TTreeReader+;
#pragma link C++ class TTreeReaderFast+;
#pragma link C++ class TTreeTableInterface;
#pragma link C++ class TSimpleAnalysis+;
#pragma link C++ class TMPWorkerTree+;
#ifdef R__USE_IMT
#pragma link C++ class ROOT::TTreeProcessorMT-;
#endif

#pragma link C++ class ROOT::Internal::TBranchProxyDirector+;
#pragma link C++ class ROOT::Detail::TBranchProxy+;
#pragma link C++ class ROOT::Internal::TFriendProxy+;

#pragma link C++ class ROOT::Internal::TFriendProxyDescriptor;
#pragma link C++ class ROOT::Internal::TBranchProxyDescriptor;
#pragma link C++ class ROOT::Internal::TBranchProxyClassDescriptor;

#pragma link C++ class ROOT::Internal::TImpProxy<double>+;
#pragma link C++ class ROOT::Internal::TImpProxy<float>+;
#pragma link C++ class ROOT::Internal::TImpProxy<UInt_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<ULong_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<UShort_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<UChar_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<Int_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<Long_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<Short_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<Char_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<Bool_t>+;

#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<double> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<float> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<UInt_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<ULong_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<UShort_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<UChar_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<Int_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<Long_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<Short_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<Char_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<Bool_t> >+;
   //specialized ! typedef TArrayProxy<ROOT::Internal::TArrayType<Char_t> >+;

#pragma link C++ class ROOT::Internal::TClaImpProxy<double>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<float>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<UInt_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<ULong_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<UShort_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<UChar_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<Int_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<Long_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<Short_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<Char_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<Bool_t>+;

#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<double> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<float> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<UInt_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<ULong_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<UShort_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<UChar_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<Int_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<Long_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<Short_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<Char_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<Bool_t> >+;

#if !defined(_MSC_VER) || (_MSC_VER>1300)
#pragma link C++ class ROOT::Internal::TImpProxy<Long64_t>+;
#pragma link C++ class ROOT::Internal::TImpProxy<ULong64_t>+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<Long64_t> >+;
#pragma link C++ class ROOT::Internal::TArrayProxy<ROOT::Internal::TArrayType<ULong64_t> >+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<Long64_t>+;
#pragma link C++ class ROOT::Internal::TClaImpProxy<ULong64_t>+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<ULong64_t> >+;
#pragma link C++ class ROOT::Internal::TClaArrayProxy<ROOT::Internal::TArrayType<Long64_t> >+;
#endif

#pragma link C++ class ROOT::Internal::TTreeReaderValueBase+;
#pragma link C++ class ROOT::Internal::TTreeReaderValueFastBase+;
#pragma link C++ class ROOT::Internal::TTreeReaderArrayBase+;
#pragma link C++ class ROOT::Internal::TNamedBranchProxy+;
#pragma link C++ class TNotifyLink<ROOT::Detail::TBranchProxy>;
#pragma link C++ class TNotifyLink<TTreeReader>;

#endif


