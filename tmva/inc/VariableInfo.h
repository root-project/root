// @(#)root/tmva $Id: VariableInfo.h,v 1.7 2007/04/19 06:53:01 brun Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Option                                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Variable type info                                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableInfo
#define ROOT_TMVA_VariableInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableInfo                                                         //
//                                                                      //
// Class for type info of MVA input variable                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

namespace TMVA {

   class VariableInfo {

   public:

      VariableInfo( const TString& expression, int varCounter, char varType = 'F', void * external = 0,
                    Double_t min = 0, Double_t max = 0 );
      VariableInfo();
      VariableInfo( const VariableInfo& other );
      ~VariableInfo() {}
      const TString& GetExpression()       const { return fExpression; }
      const TString& GetInternalVarName()  const { return fInternalVarName; }
      char           GetVarType()          const { return fVarType; }

      Double_t       GetMin () const { return fXminNorm; }
      Double_t       GetMax () const { return fXmaxNorm; }
      Double_t       GetMean() const { return fXmeanNorm; }
      Double_t       GetRMS () const { return fXrmsNorm; }

      void           SetExpression(const TString& s)      { fExpression = s; }
      void           SetInternalVarName(const TString& s) { fInternalVarName = s; }
      void           SetVarType(char c)                    { fVarType = c; }

      void           SetMin (Double_t v) { fXminNorm  = v; }
      void           SetMax (Double_t v) { fXmaxNorm  = v; }
      void           SetMean(Double_t v) { fXmeanNorm = v; }
      void           SetRMS (Double_t v) { fXrmsNorm  = v; }
      void           SetExternalLink(void* p) { fExternalData = p; }
      void           ResetMinMax() {fXminNorm = 1e30; fXmaxNorm = -1e30; };

      void           WriteToStream(std::ostream& o) const;
      void           ReadFromStream(std::istream& istr);
      void*          GetExternalLink() const { return fExternalData; }

      // assignment operator (does not copy external link)
      VariableInfo&   operator=(const TMVA::VariableInfo& rhs);

   private:

      TString  fExpression;      //! the variable expression (can be a formula)
      TString  fInternalVarName; //! the internal variable name
      Char_t   fVarType;         //! the variable type to be used internally ('F'-default or 'I')
      Double_t fXminNorm;        //! minimum value for correlated/decorrelated/PCA variable
      Double_t fXmaxNorm;        //! maximum value for correlated/decorrelated/PCA variable
      Double_t fXmeanNorm;       //! mean value for correlated/decorrelated/PCA variable
      Double_t fXrmsNorm;        //! rms value for correlated/decorrelated/PCA variable
      void*    fExternalData;    //! if the variable content is linked to an external pointer      
      Int_t    fVarCounter;      //! dummy variable
   };

}

#endif
