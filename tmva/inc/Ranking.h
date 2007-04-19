// @(#)root/tmva $Id: Ranking.h,v 1.7 2007/01/16 09:37:03 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Ranking                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual ranking class                                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_Ranking
#define ROOT_TMVA_Ranking

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Ranking                                                              //
//                                                                      //
// Defines vector of rank                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TString;

namespace TMVA {

   class Rank;

   class Ranking {

   public:

      Ranking();
      Ranking( const TString& context, const TString& rankingDiscriminatorName );
      virtual ~Ranking();
     
      virtual void AddRank( Rank& rank );
      virtual void Print() const;

      void SetContext  ( const TString context   ) { fContext = context; fLogger.SetSource( context.Data() ); }
      void SetDiscrName( const TString discrName ) { fRankingDiscriminatorName = discrName; }

   private:
                  
      std::vector<Rank>  fRanking;                  // vector of ranks
      TString            fContext;                  // the ranking context
      TString            fRankingDiscriminatorName; // the name of the ranking discriminator

      mutable MsgLogger  fLogger;                   // message logger

      ClassDef(Ranking,0) // method-specific ranking for input variables 
   };

   // --------------------------------------------------------------------------

   class Rank {

   public:

      Rank( TString variable, Double_t rankValue );
      virtual ~Rank();

      // comparison between rank
      Bool_t operator <  ( const Rank& other ) const; 
      Bool_t operator >  ( const Rank& other ) const; 

      const TString& GetVariable()  const { return fVariable; }
      Double_t       GetRankValue() const { return fRankValue; }
      Int_t          GetRank()      const { return fRank; }
      void           SetRank( Int_t rank ) { fRank = rank; }

   private:
      
      TString  fVariable;   // the variable name
      Double_t fRankValue;  // the rank value

      Int_t    fRank;
   };
}

#endif
