// @(#)root/g3d:$Name:  $:$Id: TShape.h,v 1.3 2000/12/13 15:13:47 brun Exp $
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
#ifndef ROOT_TBuffer3D
#include "TBuffer3D.h"
#endif

class TNode;

class TShape : public TNamed, public TAttLine, public TAttFill, public TAtt3D {

protected:
   Int_t           fNumber;      //Shape number
   Int_t           fVisibility;  //Visibility flag
   TMaterial      *fMaterial;    //Pointer to material
   
   Int_t           ShapeDistancetoPrimitive(Int_t numPoints, Int_t px, Int_t py);

public:
                   TShape();
                   TShape(const char *name, const char *title, const char *material);
   virtual         ~TShape();
   TMaterial       *GetMaterial()  const {return fMaterial;}
   virtual Int_t   GetNumber()     const {return fNumber;}
           Int_t   GetVisibility() const {return fVisibility;}
   virtual void    Paint(Option_t *option="");
   virtual void    SetName(const char *name);
   virtual void    SetPoints(Double_t *buffer);
   virtual void    SetVisibility(Int_t vis) {fVisibility = vis;} // *MENU*
           void    TransformPoints(TBuffer3D *buff) const;

   ClassDef(TShape,2)  //Basic shape
};

R__EXTERN TNode *gNode;

inline void TShape::SetName(const char *) { }

#endif

