// @(#)root/g3d:$Id$
// Author: Nenad Buncic   17/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TShape
#define ROOT_TShape


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TShape                                                               //
//                                                                      //
// Basic shape class                                                    //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TMaterial
#include "TMaterial.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_TAtt3D
#include "TAtt3D.h"
#endif
#ifndef ROOT_X3DBuffer
#include "X3DBuffer.h"
#endif

class TBuffer3D;
class TNode;

class TShape : public TNamed, public TAttLine, public TAttFill, public TAtt3D {

protected:
   Int_t           fNumber;      //Shape number
   Int_t           fVisibility;  //Visibility flag
   TMaterial      *fMaterial;    //Pointer to material

   virtual void    FillBuffer3D(TBuffer3D & buffer, Int_t reqSections) const;
   Int_t           GetBasicColor() const;

   Int_t           ShapeDistancetoPrimitive(Int_t numPoints, Int_t px, Int_t py);


public:
   TShape();
   TShape(const char *name, const char *title, const char *material);
   TShape(const TShape&);
   TShape& operator=(const TShape&);
   virtual         ~TShape();

   virtual const   TBuffer3D &GetBuffer3D(Int_t reqSections) const;
   TMaterial      *GetMaterial()  const {return fMaterial;}
   virtual Int_t   GetNumber()     const {return fNumber;}
   Int_t           GetVisibility() const {return fVisibility;}
   virtual void    Paint(Option_t *option="");
   virtual void    SetName(const char *name);
   virtual void    SetPoints(Double_t *points) const ;
   virtual void    SetVisibility(Int_t vis) {fVisibility = vis;} // *MENU*
   void            TransformPoints(Double_t *points, UInt_t NbPnts) const;

   ClassDef(TShape,2)  //Basic shape
};

R__EXTERN TNode *gNode;

inline void TShape::SetName(const char *) { }

#endif
