// @(#)root/g3d:$Name$:$Id$
// Author: Nenad Buncic   18/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCONE
#define ROOT_TCONE


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TCONE                                                                  //
//                                                                        //
// CONE is a conical tube. It has 5 parameters, the half length in z,     //
// the inside and outside radius at the low z limit, and those at the     //
// high z limit.                                                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTUBE
#include "TTUBE.h"
#endif

class TCONE : public TTUBE {

    protected:
        Float_t fRmin2;        // inside radius at the high z limit
        Float_t fRmax2;        // outside radius at the high z limit

    public:
        TCONE();
        TCONE(const char *name, const char *title, const char *material, Float_t dz, Float_t rmin1, Float_t rmax1,
              Float_t rmin2, Float_t rmax2);
        TCONE(const char *name, const char *title, const char *material, Float_t dz, Float_t rmax1, Float_t rmax2 =0);
        virtual ~TCONE();

        Float_t         GetRmin2() {return fRmin2;}
        Float_t         GetRmax2() {return fRmax2;}
        virtual void    SetPoints(Float_t *buff);

        ClassDef(TCONE,1)  //CONE shape
};

#endif
