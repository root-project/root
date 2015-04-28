// @(#)root/tmva $Id$
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
// 
// Ranking for variables in method (implementation)
//_______________________________________________________________________

#include <iomanip>

#include "TString.h"

#include "TMVA/Ranking.h"
#include "TMVA/MsgLogger.h"

ClassImp(TMVA::Ranking)

//_______________________________________________________________________
TMVA::Ranking::Ranking() 
   : fRanking(),
     fContext(""),
     fRankingDiscriminatorName( "" ),
     fLogger( new MsgLogger("", kINFO) )
{
   // default constructor
}

//_______________________________________________________________________
TMVA::Ranking::Ranking( const TString& context, const TString& rankingDiscriminatorName ) 
   : fRanking(),
     fContext( context ),
     fRankingDiscriminatorName( rankingDiscriminatorName ),
     fLogger( new MsgLogger(fContext.Data(), kINFO) )
{
   // constructor
}

//_______________________________________________________________________
TMVA::Ranking::~Ranking() 
{
   // destructor
   fRanking.clear();
   delete fLogger;
}

//_______________________________________________________________________
void TMVA::Ranking::SetContext( const TString& context) 
{
   fContext = context;
   fLogger->SetSource( fContext.Data() );
}

//_______________________________________________________________________
void TMVA::Ranking::AddRank( const Rank& rank )
{
   // Add a new rank
   // take ownership of it

   // sort according to rank value (descending)
   // Who the hell knows why this does not compile on windos.. write the sorting 
   // reversing myself... (means sorting in "descending" order)
   //   --> std::sort   ( fRanking.begin(), fRanking.end() );
   //   --> std::reverse( fRanking.begin(), fRanking.end() );
   fRanking.push_back( rank );
      
   UInt_t sizeofarray=fRanking.size();
   Rank  temp(fRanking[0]);
   for (UInt_t i=0; i<sizeofarray; i++) {
      for (UInt_t j=sizeofarray-1; j>i; j--) {
         if (fRanking[j-1] < fRanking[j]) {
            temp = fRanking[j-1];fRanking[j-1] = fRanking[j]; fRanking[j] = temp;
         }
      }
   }
   
   for (UInt_t i=0; i<fRanking.size(); i++) fRanking[i].SetRank( i+1 );
}

//_______________________________________________________________________
void TMVA::Ranking::Print() const
{
   // get maximum length of variable names
   Int_t maxL = 0; 
   for (std::vector<Rank>::const_iterator ir = fRanking.begin(); ir != fRanking.end(); ir++ ) 
      if ((*ir).GetVariable().Length() > maxL) maxL = (*ir).GetVariable().Length();
   
   TString hline = "";
   for (Int_t i=0; i<maxL+15+fRankingDiscriminatorName.Length(); i++) hline += "-";
   Log() << kINFO << "Ranking result (top variable is best ranked)" << Endl;
   Log() << kINFO << hline << Endl;
   Log() << kINFO << std::setiosflags(std::ios::left) 
         << std::setw(5) << "Rank : "
         << std::setw(maxL+0) << "Variable "
         << std::resetiosflags(std::ios::right) 
         << " : " << fRankingDiscriminatorName << Endl;
   Log() << kINFO << hline << Endl;
   for (std::vector<Rank>::const_iterator ir = fRanking.begin(); ir != fRanking.end(); ir++ ) {
      Log() << kINFO 
            << Form( "%4i : ",(*ir).GetRank() )
            << std::setw(TMath::Max(maxL+0,9)) << (*ir).GetVariable().Data()
            << Form( " : %3.3e", (*ir).GetRankValue() ) << Endl;
   }
   Log() << kINFO << hline << Endl;
}

// ===============================================================================================

//_______________________________________________________________________
TMVA::Rank::Rank( const TString& variable, Double_t rankValue ) 
   : fVariable( variable ),
     fRankValue( rankValue ),
     fRank( -1 ) 
{
   // constructor
}

//_______________________________________________________________________
TMVA::Rank::~Rank() 
{
   // destructor
}

//_______________________________________________________________________
Bool_t TMVA::Rank::operator< ( const Rank& other ) const
{ 
   // comparison operator <
   if (fRankValue < other.fRankValue) return true;
   else                               return false;
}

//_______________________________________________________________________
Bool_t TMVA::Rank::operator> ( const Rank& other ) const
{ 
   // comparison operator >
   if (fRankValue > other.fRankValue) return true;
   else                               return false;
}
