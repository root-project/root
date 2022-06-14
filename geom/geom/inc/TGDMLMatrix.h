// @(#)root/gdml:$Id$
// Author: Andrei Gheata 05/12/2018

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGDMLMATRIX
#define ROOT_TGDMLMATRIX

#include <TNamed.h>


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGDMLProperty - A property with a name and a reference name pointing   //
//     to a GDML matrix object                                            //
////////////////////////////////////////////////////////////////////////////

typedef TNamed TGDMLProperty;

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGDMLMatrix - A matrix used for GDML parsing, the objects have to be   //
//     exposed via TGeoManager interfcace to be able to construct optical //
//     surfaces.                                                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGDMLMatrix : public TNamed {
public:
   TGDMLMatrix() {}
   TGDMLMatrix(const char *name, size_t rows,size_t cols);
   TGDMLMatrix(const TGDMLMatrix& rhs);
   TGDMLMatrix& operator=(const TGDMLMatrix& rhs);
  ~TGDMLMatrix() { delete [] fMatrix; }

   void        Set(size_t r, size_t c, Double_t a);
   Double_t    Get(size_t r, size_t c) const;
   size_t      GetRows() const { return fNrows; }
   size_t      GetCols() const { return fNcols; }
   void        SetMatrixAsString(const char *mat) { fTitle = mat; }
   const char *GetMatrixAsString() const { return fTitle.Data(); }

   void        Print(Option_t *option="") const;

 private:

   Int_t  fNelem = 0;                // Number of elements
   size_t fNrows = 0;                // Number of rows
   size_t fNcols = 0;                // Number of columns
   Double_t *fMatrix = nullptr;      // [fNelem] Matrix elements

   ClassDef(TGDMLMatrix, 1)          // Class representing a matrix used temporary for GDML parsing
};

#endif /* ROOT_TGDMLMATRIX */
