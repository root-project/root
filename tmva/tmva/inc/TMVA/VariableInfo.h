// @(#)root/tmva $Id$
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

#include "Rtypes.h"
#include "TString.h"
#include "TMVA/Types.h"
#include "TNamed.h"

namespace TMVA {

   class VariableInfo:public TNamed {

   public:

      VariableInfo( const TString& expression, const TString& title, const TString& unit,
                    Int_t varCounter, char varType = 'F', void* external = 0,
                    Double_t min = 0, Double_t max = 0, Bool_t normalized=kTRUE );
      VariableInfo();
      VariableInfo( const VariableInfo& other );
      ~VariableInfo() {}
      const TString& GetExpression()       const { return fExpression; }
      const TString& GetInternalName()     const { return fInternalName; }
      const TString& GetLabel()            const { return fLabel; }
      const TString& GetUnit()             const { return fUnit; }
      char           GetVarType()          const { return fVarType; }

      Double_t       GetMin ()     const { return fXminNorm; }
      Double_t       GetMax ()     const { return fXmaxNorm; }
      Double_t       GetMean()     const { return fXmeanNorm; }
      Double_t       GetRMS ()     const { return fXrmsNorm; }
      Double_t       GetVariance() const { return fXvarianceNorm; }

      void           SetInternalName(const char *name) { fInternalName = name; }

      void           SetMin         ( Double_t v ) { fXminNorm     = v; }
      void           SetMax         ( Double_t v ) { fXmaxNorm     = v; }
      void           SetMean        ( Double_t v ) { fXmeanNorm    = v; }
      void           SetRMS         ( Double_t v ) { fXrmsNorm     = v; }
      void           SetExternalLink( void* p    ) { fExternalData = (char*)p; }
      void           SetVariance    ( Double_t v ) { fXvarianceNorm= v; }
      void           ResetMinMax() { fXminNorm = 1e30; fXmaxNorm = -1e30; }

      void           WriteToStream ( std::ostream& o ) const;
      void           ReadFromStream( std::istream& istr );
      void           ReadFromXML   ( void* varnode );
      void           AddToXML      ( void* varnode );
      void*          GetExternalLink() const { return (void*)fExternalData; }

      // assignment operator (does not copy external link)
      VariableInfo&  operator=(const TMVA::VariableInfo& rhs);

   private:

      // should not be set from outside this class
      void           SetExpression     ( const TString& s ) { fExpression = s; }
      void           SetLabel          ( const TString& s ) { fLabel = s; }
      void           SetUnit           ( const TString& s ) { fUnit  = s; }
      void           SetInternalVarName( const TString& s ) { fInternalName = s; }
      void           SetVarType        ( char c )           { fVarType = c; }

      TString  fExpression;      ///< original variable expression (can be a formula)
      TString  fInternalName;    ///< internal variable name (needs to be regular expression)
      TString  fLabel;           ///< variable label, set by "mylabel := var1 + var2", this is a shortcut
      //TString  fTitle;         ///<! title for axis labels in plots; set by second string in AddVariable
      TString  fUnit;            ///< unit for axis labels in plots; set by third string in AddVariable
      Char_t   fVarType;         ///< the variable type to be used internally ('F'-default or 'I')
      Double_t fXminNorm;        ///< minimum value for correlated/decorrelated/PCA variable
      Double_t fXmaxNorm;        ///< maximum value for correlated/decorrelated/PCA variable
      Double_t fXmeanNorm;       ///< mean value for correlated/decorrelated/PCA variable
      Double_t fXrmsNorm;        ///< rms value for correlated/decorrelated/PCA variable
      Double_t fXvarianceNorm;   ///< variance value for correlated/decorrelated/PCA variable
      Bool_t   fNormalized;      ///< variable gets normalized
      void*    fExternalData;    ///<! if the variable content is linked to an external pointer
      TString  fExternalDataType;///< type of external variable (int, long, double, float) - to be done JS
      Int_t    fVarCounter;      ///< dummy variable
   public:

       ClassDef(VariableInfo,1);
   };

}

#endif
