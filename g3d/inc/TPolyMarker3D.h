// @(#)root/g3d:$Name:  $:$Id: TPolyMarker3D.h,v 1.2 2000/06/13 12:22:49 brun Exp $
// Author: Nenad Buncic   21/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPolyMarker3D
#define ROOT_TPolyMarker3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPolyMarker3D                                                        //
//                                                                      //
// An array of 3-D points with the same marker.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif
#ifndef ROOT_TAtt3D
#include "TAtt3D.h"
#endif
#ifndef ROOT_X3DBuffer
#include "X3DBuffer.h"
#endif

class TH1;

class TPolyMarker3D : public TObject, public TAttMarker, public TAtt3D {

protected:
        Int_t            fN;            //number of points
        Float_t         *fP;            //[3*fN] Array of X,Y,Z coordinates
        TString          fOption;       //options
        UInt_t           fGLList;       //!The list number for OpenGL view
        Int_t            fLastPoint;    //The index of the last filled point

public:
                          TPolyMarker3D();
                          TPolyMarker3D(Int_t n, Marker_t marker=1, Option_t *option="");
                          TPolyMarker3D(Int_t n, Float_t *p, Marker_t marker=1, Option_t *option="");
                          TPolyMarker3D(Int_t n, Double_t *p, Marker_t marker=1, Option_t *option="");
                          TPolyMarker3D(const TPolyMarker3D &polymarker);
        virtual          ~TPolyMarker3D();

        virtual void      Copy(TObject &polymarker);
        Int_t             DistancetoPrimitive(Int_t px, Int_t py);
        virtual void      Draw(Option_t *option="");
        virtual void      DrawPolyMarker(Int_t n, Float_t *p, Marker_t marker, Option_t *option="");
        virtual void      ExecuteEvent(Int_t event, Int_t px, Int_t py);
        virtual Int_t     GetN() const { return fN;}
        virtual Float_t  *GetP() const { return fP;}
        Option_t         *GetOption() const {return fOption.Data();}
        virtual void      ls(Option_t *option="") const;
        virtual void      Paint(Option_t *option="");
        static  void      PaintH3(TH1 *h, Option_t *option);
        virtual void      PaintPolyMarker(Int_t n, Float_t *p, Marker_t marker, Option_t *option="");
        virtual void      Print(Option_t *option="") const;
        virtual void      SavePrimitive(ofstream &out, Option_t *option);
        void              SetPoint(Int_t n, Double_t x, Double_t y, Double_t z); // *MENU*
        virtual void      SetPolyMarker(Int_t n, Float_t *p, Marker_t marker, Option_t *option="");
        virtual void      SetPolyMarker(Int_t n, Double_t *p, Marker_t marker, Option_t *option="");
        virtual Int_t     SetNextPoint(Double_t x, Double_t y, Double_t z); // *MENU*
        virtual void      Sizeof3D() const;
        static  void      SizeofH3(TH1 *h);
        virtual Int_t     Size() const {return fLastPoint+1;}

        ClassDef(TPolyMarker3D,1)  //An array of 3-D points with the same marker
};

R__EXTERN Size3D gSize3D;

#endif
