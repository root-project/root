// @(#)root/g3d:$Name:  $:$Id: TAxis3D.h,v 1.3 2000/09/05 09:21:22 brun Exp $
// Author: Valery Fine(fine@mail.cern.ch)   07/01/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TAxis3D
#define ROOT_TAxis3D
// $Id: TAxis3D.h,v 1.3 2000/09/05 09:21:22 brun Exp $
// ***********************************************************************
// * C++ class library to paint axis "arround" TView object
// * Copyright(c) 1997~1999  [BNL] Brookhaven National Laboratory, STAR, All rights reserved
// * Author                  Valeri Fine  (fine@bnl.gov)
// * Copyright(c) 1997~1999  Valeri Fine  (fine@bnl.gov)
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// *
// * Permission to use, copy, modify and distribute this software and its
// * documentation for any purpose is hereby granted without fee,
// * provided that the above copyright notice appear in all copies and
// * that both that copyright notice and this permission notice appear
// * in supporting documentation.  Brookhaven National Laboratory makes no
// * representations about the suitability of this software for any
// * purpose.  It is provided "as is" without express or implied warranty.
// ************************************************************************

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAxis3D                                                              //
//                                                                      //
// 3D axice                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TAxis
#include "TAxis.h"
#endif

class TF1;
class TBrowser;
class TGaxis;
class TVirtualPad;
class TView;
class TAxis3D : public TNamed  {

private:
    Int_t   AxisChoice(Option_t *axis);
    void    Build();

protected:
    TAxis           fAxis[3];             //X/Y/Z axis
    TString         fOption;              // Options (is not use yet)
    static  const Char_t *rulerName;      // The default object name
    TAxis          *fSelected;            //!  The selected axis to play with
            Bool_t  fZoomMode;            // Zoom mode for the entire parent TPad

    virtual void    Copy(TObject &hnew);
            void    InitSet();
            Bool_t  SwitchZoom();

public:
    TAxis3D();
    TAxis3D(Option_t *option);
    TAxis3D(const TAxis3D &axis);
    virtual ~ TAxis3D(){;}

    virtual void     Browse(TBrowser *b);

    virtual Int_t    DistancetoPrimitive(Int_t px, Int_t py);
    virtual void     ExecuteEvent(Int_t event, Int_t px, Int_t py);

    virtual Int_t    GetNdivisions(Option_t *axis="X");
    virtual Color_t  GetAxisColor(Option_t *axis="X");
    virtual Color_t  GetLabelColor(Option_t *axis="X");
    virtual Style_t  GetLabelFont(Option_t *axis="X");
    virtual Float_t  GetLabelOffset(Option_t *axis="X");
    virtual Float_t  GetLabelSize(Option_t *axis="X");
    static  TAxis3D *GetPadAxis(TVirtualPad *pad=0);
    virtual Float_t  GetTitleOffset(Option_t *axis="X");
    virtual Float_t  GetTickLength(Option_t *axis="X");

    virtual void     GetCenter(Axis_t *center){fAxis[0].GetCenter(center);}

    virtual void     GetLowEdge(Axis_t *edge){fAxis[0].GetLowEdge(edge);}

    virtual char    *GetObjectInfo(Int_t px, Int_t py);

    Option_t        *GetOption() const {return fOption.Data();}

    virtual TAxis   *GetXaxis() {return &fAxis[0];}
    virtual TAxis   *GetYaxis() {return &fAxis[1];}
    virtual TAxis   *GetZaxis() {return &fAxis[2];}
    virtual Bool_t   IsFolder() const { return kTRUE;}
    virtual void     Paint(Option_t *option="");
            void     PaintAxis(TGaxis *axis, Float_t ang);
    static Double_t *PixeltoXYZ(Double_t px, Double_t py, Double_t *point3D, TView *view =0);
    virtual void     SavePrimitive(ofstream &out, Option_t *option);

    virtual void     SetAxisColor(Color_t color=1, Option_t *axis="*"); // *MENU*
    virtual void     SetAxisRange(Double_t xmin, Double_t xmax, Option_t *axis="*");

    virtual void     SetLabelColor(Color_t color=1, Option_t *axis="*");// *MENU*
    virtual void     SetLabelFont(Style_t font=62, Option_t *axis="*"); // *MENU*
    virtual void     SetLabelOffset(Float_t offset=0.005, Option_t *axis="*"); // *MENU*
    virtual void     SetLabelSize(Float_t size=0.02, Option_t *axis="*"); // *MENU*

    virtual void     SetNdivisions(Int_t n=510, Option_t *axis="*"); // *MENU*
    virtual void     SetOption(Option_t *option=" ") {fOption = option;}
    virtual void     SetTickLength(Float_t length=0.02, Option_t *axis="*"); // *MENU*
    virtual void     SetTitleOffset(Float_t offset=1, Option_t *axis="*"); // *MENU*
    virtual void     SetXTitle(const char *title) {fAxis[0].SetTitle(title);} // *MENU*
    virtual void     SetYTitle(const char *title) {fAxis[1].SetTitle(title);} // *MENU*
    virtual void     SetZTitle(const char *title) {fAxis[2].SetTitle(title);} // *MENU*
    static  TAxis3D *ToggleRulers(TVirtualPad *pad=0);
    static  TAxis3D *ToggleZoom(TVirtualPad *pad=0);
    void             UseCurrentStyle();

    ClassDef(TAxis3D,1)  //3-D ruler painting class
};


inline Bool_t TAxis3D::SwitchZoom(){Bool_t s = fZoomMode; fZoomMode = !fZoomMode; return s;}

#endif
