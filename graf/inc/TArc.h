// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   16/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArc
#define ROOT_TArc


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TArc                                                                 //
//                                                                      //
// Arc of a circle.                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TEllipse
#include "TEllipse.h"
#endif


class TArc : public TEllipse {

public:
        TArc();
        TArc(Float_t x1, Float_t y1,Float_t radius
           , Float_t phimin=0,Float_t phimax=360);
        TArc(const TArc &arc);
        virtual ~TArc();
                void Copy(TObject &arc);
        virtual void DrawArc(Float_t x1, Float_t y1, Float_t radius
                            ,Float_t  phimin=0, Float_t  phimax=360);
        virtual void   SavePrimitive(ofstream &out, Option_t *option);

        ClassDef(TArc,1)  //Arc of a circle
};

#endif
