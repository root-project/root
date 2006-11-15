// @(#)root/tmva $Id: Ranking.cxx,v 1.10 2006/10/17 14:02:14 krasznaa Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Ranking                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
// 
// Ranking for variables in method (implementation)
//_______________________________________________________________________

#include "Riostream.h"
#include "TMVA/Ranking.h"
#include "TString.h"

ClassImp(TMVA::Ranking)
   ;

TMVA::Ranking::Ranking()
  : fLogger( "", kINFO )
{
   fRanking.clear();
}

TMVA::Ranking::Ranking( const TString& context, const TString& rankingDiscriminatorName ) 
   : fContext( context ),
     fRankingDiscriminatorName( rankingDiscriminatorName ),
     fLogger( context.Data(), kINFO )
{
   fRanking.clear();
}

TMVA::Ranking::~Ranking() 
{
   fRanking.clear();
}

void TMVA::Ranking::AddRank( Rank& rank )
{
   fRanking.push_back( rank );
   
   // sort according to rank value (descending)
   // Who the hell knows why this does not compile on windos.. write the sorting 
   // reversing myself... (means sorting in "descending" order)
   //   --> std::sort   ( fRanking.begin(), fRanking.end() );
   //   --> std::reverse( fRanking.begin(), fRanking.end() );
   
   UInt_t sizeofarray=fRanking.size();
   Rank  temp(fRanking[0]);
   for (unsigned int i=0; i<sizeofarray; i++) {
      for (unsigned int j=sizeofarray-1; j>i; j--) {
         if (fRanking[j-1] < fRanking[j]) {
            temp = fRanking[j-1];fRanking[j-1] = fRanking[j]; fRanking[j] = temp;
         }
      }
   }
   
   for (UInt_t i=0; i<fRanking.size(); i++) fRanking[i].SetRank( i+1 );
}

void TMVA::Ranking::Print() const
{
   // get maximum length of variable names
   Int_t maxL = 0; 
   for (std::vector<Rank>::const_iterator ir = fRanking.begin(); ir != fRanking.end(); ir++ ) 
      if ((*ir).GetVariable().Length() > maxL) maxL = (*ir).GetVariable().Length();
   
   fLogger << kINFO << "ranking result (top variable is best ranked)" << Endl;
   fLogger << kINFO << "----------------------------------------------------------------" << Endl;
   fLogger << kINFO << setiosflags(ios::left) 
           << setw(5) << "Rank : "
           << setw(maxL+0) << "Variable "
           << resetiosflags(ios::right) 
           << " : " << fRankingDiscriminatorName << Endl;
   fLogger << kINFO << "----------------------------------------------------------------" << Endl;
   for (std::vector<Rank>::const_iterator ir = fRanking.begin(); ir != fRanking.end(); ir++ ) {
      fLogger << kINFO 
              << Form( "%4i : ",(*ir).GetRank() )
              << setw(TMath::Max(maxL+0,9)) << (*ir).GetVariable().Data()
              << Form( " : %3.3e", (*ir).GetRankValue() ) << Endl;
   }
   fLogger << kINFO << "----------------------------------------------------------------" << Endl;
}

// -------------------------------------------------------------------------

TMVA::Rank::Rank( TString variable, Double_t rankValue ) 
   : fVariable( variable ),
     fRankValue( rankValue ),
     fRank( -1 ) 
{}

TMVA::Rank::~Rank() 
{}

Bool_t TMVA::Rank::operator < ( const Rank& other ) const
{ 
   if (fRankValue < other.fRankValue) return true;
   else                               return false;
}
Bool_t TMVA::Rank::operator > ( const Rank& other ) const
{ 
   if (fRankValue > other.fRankValue) return true;
   else                               return false;
}
