// @(#)root/x3d:$Name$:$Id$
// Author: Rene Brun   05/09/99
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewerX3D                                                           //
//                                                                      //
// C++ interface to the X3D viewer                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TViewerX3D.h"
#include "TAtt3D.h"
#include "X3DBuffer.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TMath.h"
#include "TROOT.h"
#include "TClass.h"

ClassImp(TViewerX3D)

//______________________________________________________________________________
TViewerX3D::TViewerX3D()
{
}

//______________________________________________________________________________

#ifdef R__UNIX
extern "C" {
  void   x3d_main(Float_t  *longitude,Float_t  *latitude,Float_t  *psi,Option_t *option);
}
#endif


//______________________________________________________________________________
void TViewerX3D::View(TVirtualPad *pad, Option_t *option)
{
#ifdef R__UNIX
    TObject *obj;
    char x3dopt[32];

    TView *view = pad->GetView();
    if(!view) {
        printf("ViewX3D::View, View is not set !");
        return;
    }

    gSize3D.numPoints = 0;
    gSize3D.numSegs   = 0;
    gSize3D.numPolys  = 0;

    TObjLink *lnk = pad->GetListOfPrimitives()->FirstLink();
    while (lnk) {
        obj = lnk->GetObject();
        TAtt3D *att;
#ifdef R__RTTI
        if ((att = dynamic_cast<TAtt3D*>(obj)))
#else
        if ((att = (TAtt3D*)obj->IsA()->DynamicCast(TAtt3D::Class(), obj)))
#endif
           att->Sizeof3D();
        lnk = lnk->Next();
    }

    printf("Total size of x3d primitives:\n");
    printf("     gSize3D.numPoints= %d\n",gSize3D.numPoints);
    printf("     gSize3D.numSegs  = %d\n",gSize3D.numSegs);
    printf("     gSize3D.numPolys = %d\n",gSize3D.numPolys);

    if (!AllocateX3DBuffer()) {
        printf("ViewX3D::View, x3d buffer allocation failure");
        return;
    }

    lnk = pad->GetListOfPrimitives()->FirstLink();
    while (lnk) {
        obj = lnk->GetObject();
        if (obj->InheritsFrom(TAtt3D::Class())) {
           strcpy(x3dopt,"x3d");
           strcat(x3dopt,option);
           obj->Paint(x3dopt);
        }
        lnk = lnk->Next();
    }

    const Float_t kPI = Float_t (TMath::Pi());

    Float_t longitude_rad = ( 90 + view->GetLongitude()) * kPI/180.0;
    Float_t  latitude_rad = (-90 + view->GetLatitude() ) * kPI/180.0;
    Float_t       psi_rad = (      view->GetPsi()      ) * kPI/180.0;


    //*-* Call 'x3d' package *-*
    x3d_main(&longitude_rad, &latitude_rad, &psi_rad, option);

    Int_t irep;

    Float_t longitude_deg = longitude_rad * 180.0/kPI - 90;
    Float_t  latitude_deg = latitude_rad  * 180.0/kPI + 90;
    Float_t       psi_deg = psi_rad       * 180.0/kPI;

    view->SetView(longitude_deg, latitude_deg, psi_deg, irep);

    pad->SetPhi(-90 - longitude_deg);
    pad->SetTheta(90 - latitude_deg);

    pad->Modified(kTRUE);

#endif
}

