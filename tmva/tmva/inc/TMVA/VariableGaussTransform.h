// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Eckhard v. Toerne

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableGaussTransform                                                *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Decorrelation of input variables                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>  - Uni Bonn, Germany              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableGaussTransform
#define ROOT_TMVA_VariableGaussTransform

#include "TMVA/PDF.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableGaussTransform                                               //
//                                                                      //
// Gaussian transformation of input variables.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#include "TH1.h"
#include "TGraph.h"
#include "TSpline.h"
#include "TDirectory.h"
#include "Event.h"

#include "TMVA/VariableTransformBase.h"

namespace TMVA {

   class TMVAGaussPair {

   public:

   TMVAGaussPair( Float_t f, Float_t w ): fF(f), fW(w) {}
      Bool_t  operator >  ( const TMVAGaussPair &p ) const { return fF >  p.fF; }
      Bool_t  operator <  ( const TMVAGaussPair &p ) const { return fF <  p.fF; }
      Bool_t  operator == ( const TMVAGaussPair &p ) const { return fF == p.fF; }
      Float_t GetValue() const { return fF; }
      Float_t GetWeight() const { return fW; }

   private:

      Float_t fF; // the float
      Float_t fW; // the event weight
   };


   class VariableGaussTransform : public VariableTransformBase {

   public:

      VariableGaussTransform( DataSetInfo& dsi, TString strcor=""  );
      virtual ~VariableGaussTransform( void );

      void   Initialize() override;
      Bool_t PrepareTransformation (const std::vector<Event*>&) override;

      const Event* Transform(const Event* const, Int_t cls ) const override;
      const Event* InverseTransform(const Event* const, Int_t cls ) const override;

      void WriteTransformationToStream ( std::ostream& ) const override;
      void ReadTransformationFromStream( std::istream&, const TString& ) override;

      void AttachXMLTo(void* parent) override;
      void ReadFromXML( void* trfnode ) override;

      void PrintTransformation( std::ostream & o ) override;

      // writer of function code
      void MakeFunction( std::ostream& fout, const TString& fncName, Int_t part, UInt_t trCounter, Int_t cls ) override;

   private:

      Bool_t           fFlatNotGauss;
      Int_t            fPdfMinSmooth;
      Int_t            fPdfMaxSmooth;
      //      mutable Event*   fTransformedEvent;

      std::vector< std::vector< TH1F* > >      fCumulativeDist;     ///<! The Cumulative distributions
      //std::vector< std::vector< TGraph* > >    fCumulativeGraph;  ///<! The Cumulative distributions
      //std::vector< std::vector< TSpline3* > >  fCumulativeSpline; ///<! The Cumulative distributions
      std::vector< std::vector< PDF*> >         fCumulativePDF;     ///<  The cumulative PDF

      void GetCumulativeDist( const std::vector<Event*>& );
      void CleanUpCumulativeArrays(TString opt = "ALL");

      // needed for backward compatibility
      UInt_t fElementsperbin;  // av number of events stored per bin in cum dist
      Double_t OldCumulant(Float_t x, TH1* h ) const;

      ClassDefOverride(VariableGaussTransform,0); // Variable transformation: Gauss transformation
   };

} // namespace TMVA

#endif
