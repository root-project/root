// @(#)root/histpainter:$Name:  $:$Id: TLego.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Rene Brun, Evgueni Tcherniaev, Olivier Couet   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLego
#define ROOT_TLego


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLego                                                                //
//                                                                      //
// Hidden line removal package.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif


const Int_t kCARTESIAN   = 1;
const Int_t kPOLAR       = 2;
const Int_t kCYLINDRICAL = 3;
const Int_t kSPHERICAL   = 4;
const Int_t kRAPIDITY    = 5;


class TLego : public TObject, public TAttLine, public TAttFill {

private:
   Double_t     fX0;               //
   Double_t     fDX;               //
   Double_t     fRmin[3];          //Lower limits of lego
   Double_t     fRmax[3];          //Upper limits of lego
   Double_t     fU[2000];          //
   Double_t     fD[2000];          //
   Double_t     fT[200];           //
   Double_t     fFunLevel[257];    //Function levels corresponding to colors
   Double_t     fPlines[1200];     //
   Double_t     fAphi[183];        //
   Double_t     fYdl;              //
   Double_t     fYls[4];           //
   Double_t     fVls[12];          //
   Double_t     fQA;               //
   Double_t     fQD;               //
   Double_t     fQS;               //
   Double_t     fXrast;            //
   Double_t     fYrast;            //
   Double_t     fDXrast;           //
   Double_t     fDYrast;           //
   Int_t        fSystem;           //Coordinate system
   Int_t        fNT;               //
   Int_t        fNlevel;           //Number of color levels
   Int_t        fColorLevel[258];  //Color levels corresponding to functions
   Int_t        fColorMain[10];    //
   Int_t        fColorDark[10];    //
   Int_t        fColorTop;         //
   Int_t        fColorBottom;      //
   Int_t        fMesh;             //(=1 if mesh to draw, o otherwise)
   Int_t        fNlines;           //
   Int_t        fLevelLine[200];   //
   Int_t        fLoff;             //
   Int_t        fNqs;              //
   Int_t        fNxrast;           //
   Int_t        fNyrast;           //
   Int_t        fIfrast;           //
   Int_t        *fRaster;          //pointer to raster buffer
   Int_t        fJmask[30];        //
   Int_t        fMask[465];        //

public:
   typedef void (TLego::*DrawFaceFunc_t)(Int_t *, Double_t *, Int_t, Int_t *, Double_t *);
   typedef void (TLego::*LegoFunc_t)(Int_t,Int_t,Int_t&,Double_t*,Double_t*,Double_t*);
   typedef void (TLego::*SurfaceFunc_t)(Int_t,Int_t,Double_t*,Double_t*);

private:
   DrawFaceFunc_t  fDrawFace;        //pointer to face drawing function
   LegoFunc_t      fLegoFunction;    //pointer to lego function
   SurfaceFunc_t   fSurfaceFunction; //pointer to surface function

public:
           TLego();
           TLego(Double_t *rmin, Double_t *rmax, Int_t system=1);
 virtual   ~TLego();
   void    BackBox(Double_t ang);
   void    ClearRaster();
   void    ColorFunction(Int_t nl, Double_t *fl, Int_t *icl, Int_t &irep);
   void    DrawFaceMode1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t);
   void    DrawFaceMode2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t);
   void    DrawFaceMode3(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t);
   void    DrawFaceMove1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt);
   void    DrawFaceMove2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt);
   void    DrawFaceRaster1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt);
   void    DrawFaceRaster2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt);
   void    FillPolygon(Int_t n, Double_t *p, Double_t *f);
   void    FillPolygonBorder(Int_t nn, Double_t *xy);
   void    FindLevelLines(Int_t np, Double_t *f, Double_t *t);
   void    FindPartEdge(Double_t *p1, Double_t *p2, Double_t f1, Double_t f2, Double_t fmin, Double_t fmax, Int_t &kpp, Double_t *pp);
   void    FindVisibleLine(Double_t *p1, Double_t *p2, Int_t ntmax, Int_t &nt, Double_t *t);
   void    FindVisibleDraw(Double_t *r1, Double_t *r2);
   void    FrontBox(Double_t ang);
   void    GouraudFunction(Int_t ia, Int_t ib, Double_t *f, Double_t *t);
   void    InitMoveScreen(Double_t xmin, Double_t xmax);
   void    InitRaster(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, Int_t nx, Int_t ny);
   void    LegoCartesian(Double_t ang, Int_t nx, Int_t ny, const char *chopt);
   void    LegoFunction(Int_t ia, Int_t ib, Int_t &nv, Double_t *ab, Double_t *vv, Double_t *t);
   void    LegoPolar(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LegoCylindrical(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LegoSpherical(Int_t ipsdr, Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LightSource(Int_t nl, Double_t yl, Double_t xscr, Double_t yscr, Double_t zscr, Int_t &irep);
   void    Luminosity(Double_t *anorm, Double_t &flum);
   void    ModifyScreen(Double_t *r1, Double_t *r2);
   void    SetDrawFace(DrawFaceFunc_t pointer);
   void    SetLegoFunction(LegoFunc_t pointer);
   void    SetMesh(Int_t mesh=1) {fMesh=mesh;}
   void    SetSurfaceFunction(SurfaceFunc_t pointer);
   void    SetColorDark(Color_t color, Int_t n=0);
   void    SetColorMain(Color_t color, Int_t n=0);
   void    SideVisibilityDecode(Double_t val, Int_t &iv1, Int_t &iv2, Int_t &iv3, Int_t &iv4, Int_t &iv5, Int_t &iv6, Int_t &ir);
   void    SideVisibilityEncode(Int_t iopt, Double_t phi1, Double_t phi2, Double_t &val);
   void    Spectrum(Int_t nl, Double_t fmin, Double_t fmax, Int_t ic, Int_t idc, Int_t &irep);
   void    SurfaceCartesian(Double_t ang, Int_t nx, Int_t ny, const char *chopt);
   void    SurfacePolar(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    SurfaceCylindrical(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    SurfaceFunction(Int_t ia, Int_t ib, Double_t *f, Double_t *t);
   void    SurfaceSpherical(Int_t ipsdr, Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    SurfaceProperty(Double_t qqa, Double_t qqd, Double_t qqs, Int_t nnqs, Int_t &irep);

   ClassDef(TLego,0)   //Hidden line removal package
};

#endif



