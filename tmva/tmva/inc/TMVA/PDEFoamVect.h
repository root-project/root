// @(#)root/tmva $Id$
// Author: S. Jadach, Tancredi Carli, Dominik Dannheim, Alexander Voigt

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamVect                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Auxiliary class PDEFoamVect of n-dimensional vector, with dynamic         *
 *      allocation used for the cartesian geometry of the PDEFoam cells           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - TU Dresden, Germany                                    *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDEFoamVect
#define ROOT_TMVA_PDEFoamVect

#include "TObject.h"

namespace TMVA {

   class PDEFoamVect : public TObject {

   private:
      Int_t       fDim;                     ///< Dimension
      Double_t   *fCoords;                  ///< [fDim] Coordinates

   public:
      // constructor
      PDEFoamVect();                                 ///< Constructor
      PDEFoamVect(Int_t);                            ///< USER Constructor
      PDEFoamVect(const PDEFoamVect &);              ///< Copy constructor
      virtual ~PDEFoamVect();                        ///< Destructor

      //////////////////////////////////////////////////////////////////////////////
      //                     Overloading operators                                //
      //////////////////////////////////////////////////////////////////////////////
      PDEFoamVect& operator =( const PDEFoamVect& ); // = operator; Substitution
      Double_t & operator[]( Int_t );                // [] provides POINTER to coordinate
      PDEFoamVect& operator =( Double_t [] );        // LOAD IN entire double vector
      PDEFoamVect& operator =( Double_t );           // LOAD IN double number
      //////////////////////////   OTHER METHODS    //////////////////////////////////
      PDEFoamVect& operator+=( const  PDEFoamVect& );  // +=; add vector u+=v  (FAST)
      PDEFoamVect& operator-=( const  PDEFoamVect& );  // +=; add vector u+=v  (FAST)
      PDEFoamVect& operator*=( const  Double_t&  );    // *=; mult. by scalar v*=x (FAST)
      PDEFoamVect  operator+ ( const  PDEFoamVect& );  // +;  u=v+s, NEVER USE IT, SLOW!!!
      PDEFoamVect  operator- ( const  PDEFoamVect& );  // -;  u=v-s, NEVER USE IT, SLOW!!!
      void       Print(Option_t *option) const;    // Prints vector
      Int_t      GetDim() const { return fDim; }   // Returns dimension
      Double_t   GetCoord(Int_t i) const { return fCoords[i]; }   // Returns coordinate

      ClassDef(PDEFoamVect,2) //n-dimensional vector with dynamical allocation
         }; // end of PDEFoamVect
}  // namespace TMVA

#endif
