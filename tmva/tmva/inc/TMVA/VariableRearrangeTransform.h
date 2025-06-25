// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableRearrangeTransform                                            *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      rearrangement of input variables                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
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

#ifndef ROOT_TMVA_VariableRearrangeTransform
#define ROOT_TMVA_VariableRearrangeTransform

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableRearrangeTransform                                           //
//                                                                      //
// rearrangement of input variables                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/VariableTransformBase.h"

#include <vector>

namespace TMVA {

   class VariableRearrangeTransform : public VariableTransformBase {

   public:

      typedef std::vector<Float_t>       FloatVector;

      VariableRearrangeTransform( DataSetInfo& dsi );
      virtual ~VariableRearrangeTransform( void );

      void   Initialize() override;
      Bool_t PrepareTransformation (const std::vector<Event*>&) override;

      const Event* Transform(const Event* const, Int_t cls ) const override;
      const Event* InverseTransform( const Event* const, Int_t cls ) const override;

      void WriteTransformationToStream ( std::ostream& ) const override {}
      void ReadTransformationFromStream( std::istream&, const TString& ) override { SetCreated(); }

      void AttachXMLTo(void* parent) override;
      void ReadFromXML( void* trfnode ) override;

      void PrintTransformation( std::ostream & o ) override;

      // writer of function code
      void MakeFunction( std::ostream& fout, const TString& fncName, Int_t part, UInt_t trCounter, Int_t cls ) override;

      // provides string vector giving explicit transformation
      std::vector<TString>* GetTransformationStrings( Int_t cls ) const override;

   private:

      ClassDefOverride(VariableRearrangeTransform,0); // Variable transformation: normalization
   };

} // namespace TMVA

#endif
