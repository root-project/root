// @(#)root/tmva $Id: Ranking.cxx,v 1.1 2006/10/09 15:55:02 brun Exp $
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

namespace TMVA {

   Ranking::Ranking() 
   {
      fRanking.clear();
   }

   Ranking::Ranking( const TString& context, const TString& rankingDiscriminatorName ) 
      : fContext( context ),
        fRankingDiscriminatorName( rankingDiscriminatorName )
   {
      fRanking.clear();
   }

   Ranking::~Ranking() 
   {
      fRanking.clear();
   }

   void Ranking::AddRank( Rank& rank )
   {
      fRanking.push_back( rank );

      // sort according to rank value (descending)
      //std::sort   ( fRanking.begin(), fRanking.end() ); //PLEASE FIX ME
      //std::reverse( fRanking.begin(), fRanking.end() );
      for (UInt_t i=0; i<fRanking.size(); i++) fRanking[i].SetRank( i+1 );
   }
    
   void Ranking::Print() const
   {
      // get maximum length of variable names
      Int_t maxL = 0; 
      for (std::vector<Rank>::const_iterator ir = fRanking.begin(); ir != fRanking.end(); ir++ ) 
         if ((*ir).GetVariable().Length() > maxL) maxL = (*ir).GetVariable().Length();

      cout << "--- " << fContext << ": ranked output (top variable is best ranked)" << endl;
      cout << "----------------------------------------------------------------" << endl;
      cout << "--- " << setiosflags(ios::left) 
           << setw(5) << "Rank : "
           << setw(maxL+0) << "Variable "
           << resetiosflags(ios::right) 
           << " : " << fRankingDiscriminatorName << endl;
      cout << "----------------------------------------------------------------" << endl;
      for (std::vector<Rank>::const_iterator ir = fRanking.begin(); ir != fRanking.end(); ir++ ) {
         cout << Form( "--- %4i : ",(*ir).GetRank() );
         cout << setw(TMath::Max(maxL+0,9)) << (*ir).GetVariable().Data();
         cout << Form( " : %3.3e", (*ir).GetRankValue() ) << endl;
      }
      cout << "----------------------------------------------------------------" << endl;
   }

   // -------------------------------------------------------------------------

   Rank::Rank( TString variable, Double_t rankValue ) 
      : fVariable( variable ),
        fRankValue( rankValue ),
        fRank( -1 ) 
   {}

   Rank::~Rank() 
   {}

   Bool_t Rank::operator < ( const Rank& other ) const
   { 
      if (fRankValue < other.fRankValue) return true;
      else                               return false;
   }
   Bool_t Rank::operator > ( const Rank& other ) const
   { 
      if (fRankValue > other.fRankValue) return true;
      else                               return false;
   }
}
