// @(#)root/g3d:$Id$
// Author: Rene Brun   14/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNodeDiv                                                             //
//                                                                      //
// Description of parameters to divide a 3-D geometry object            //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNodeDiv
#define ROOT_TNodeDiv

#ifndef ROOT_TNode
#include "TNode.h"
#endif


class TNodeDiv  : public TNode {
protected:
   Int_t           fNdiv;        //Number of divisions
   Int_t           fAxis;        //Axis number where object is divided

public:
   TNodeDiv();
   TNodeDiv(const char *name, const char *title, const char *shapename, Int_t ndiv, Int_t axis, Option_t *option="");
   TNodeDiv(const char *name, const char *title, TShape *shape, Int_t ndiv, Int_t axis, Option_t *option="");
   virtual ~TNodeDiv();
   virtual void             Draw(Option_t *option="");
   virtual void             Paint(Option_t *option="");

   ClassDef(TNodeDiv,1)  //Description of parameters to divide a 3-D geometry object
};

#endif
