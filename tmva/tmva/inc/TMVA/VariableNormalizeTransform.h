// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableNormalizeTransform                                            *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      Decorrelation of input variables                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
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

#ifndef ROOT_TMVA_VariableNormalizeTransform
#define ROOT_TMVA_VariableNormalizeTransform

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableNormalizeTransform                                           //
//                                                                      //
// Linear interpolation class                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMatrixDfwd.h"

#include "TMVA/VariableTransformBase.h"

#include <vector>

namespace TMVA {

   class VariableNormalizeTransform : public VariableTransformBase {

   public:

      typedef std::vector<Float_t>       FloatVector;
      typedef std::vector< FloatVector > VectorOfFloatVectors;
      VariableNormalizeTransform( DataSetInfo& dsi );
      virtual ~VariableNormalizeTransform( void );

      void   Initialize() override;
      Bool_t PrepareTransformation (const std::vector<Event*>&) override;

      const Event* Transform(const Event* const, Int_t cls ) const override;
      const Event* InverseTransform( const Event* const, Int_t cls ) const override;

      void WriteTransformationToStream ( std::ostream& ) const override;
      void ReadTransformationFromStream( std::istream&, const TString& ) override;
      void BuildTransformationFromVarInfo( const std::vector<TMVA::VariableInfo>& var );

      void AttachXMLTo(void* parent) override;
      void ReadFromXML( void* trfnode ) override;

      void PrintTransformation( std::ostream & o ) override;

      // writer of function code
      void MakeFunction( std::ostream& fout, const TString& fncName, Int_t part, UInt_t trCounter, Int_t cls ) override;

      // provides string vector giving explicit transformation
      std::vector<TString>* GetTransformationStrings( Int_t cls ) const override;

   private:

      void CalcNormalizationParams( const std::vector< Event*>& events);

      //      mutable Event*           fTransformedEvent;

      VectorOfFloatVectors                   fMin;       ///<! Min of source range
      VectorOfFloatVectors                   fMax;       ///<! Max of source range

      ClassDefOverride(VariableNormalizeTransform,0); // Variable transformation: normalization
   };

} // namespace TMVA

#endif
