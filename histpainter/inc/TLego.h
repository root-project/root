// @(#)root/histpainter:$Name$:$Id$
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
   Int_t        fSystem;           //Coordinate system
   Int_t        fNT;               //
   Float_t      fX0;               //
   Float_t      fDX;               //
   Float_t      fRmin[3];          //Lower limits of lego
   Float_t      fRmax[3];          //Upper limits of lego
   Float_t      fU[2000];          //
   Float_t      fD[2000];          //
   Float_t      fT[200];           //
   Int_t        fNlevel;           //Number of color levels
   Float_t      fFunLevel[257];    //Function levels corresponding to colors
   Int_t        fColorLevel[258];  //Color levels corresponding to functions
   Int_t        fColorMain[10];    //
   Int_t        fColorDark[10];    //
   Int_t        fColorTop;         //
   Int_t        fColorBottom;      //
   Int_t        fMesh;             //(=1 if mesh to draw, o otherwise)
   Int_t        fNlines;           //
   Int_t        fLevelLine[200];   //
   Float_t      fPlines[1200];     //
   Float_t      fAphi[183];        //
   Int_t        fLoff;             //
   Float_t      fYdl;              //
   Float_t      fYls[4];           //
   Float_t      fVls[12];          //
   Float_t      fQA;               //
   Float_t      fQD;               //
   Float_t      fQS;               //
   Int_t        fNqs;              //
   Int_t        fNxrast;           //
   Int_t        fNyrast;           //
   Float_t      fXrast;            //
   Float_t      fYrast;            //
   Float_t      fDXrast;           //
   Float_t      fDYrast;           //
   Int_t        fIfrast;           //
   Int_t        *fRaster;          //pointer to raster buffer
   Int_t        fJmask[30];        //
   Int_t        fMask[465];        //

public:
   typedef void (TLego::*DrawFaceFunc_t)(Int_t *, Float_t *, Int_t, Int_t *, Float_t *);
   typedef void (TLego::*LegoFunc_t)(Int_t,Int_t,Int_t&,Float_t*,Float_t*,Float_t*);
   typedef void (TLego::*SurfaceFunc_t)(Int_t,Int_t,Float_t*,Float_t*);

private:
   DrawFaceFunc_t  fDrawFace;        //pointer to face drawing function
   LegoFunc_t      fLegoFunction;    //pointer to lego function
   SurfaceFunc_t   fSurfaceFunction; //pointer to surface function

public:
           TLego();
           TLego(Float_t *rmin, Float_t *rmax, Int_t system=1);
 virtual   ~TLego();
   void    BackBox(Float_t ang);
   void    ClearRaster();
   void    ColorFunction(Int_t nl, Float_t *fl, Int_t *icl, Int_t &irep);
   void    DrawFaceMode1(Int_t *icodes, Float_t *xyz, Int_t np, Int_t *iface, Float_t *t);
   void    DrawFaceMode2(Int_t *icodes, Float_t *xyz, Int_t np, Int_t *iface, Float_t *t);
   void    DrawFaceMode3(Int_t *icodes, Float_t *xyz, Int_t np, Int_t *iface, Float_t *t);
   void    DrawFaceMove1(Int_t *icodes, Float_t *xyz, Int_t np, Int_t *iface, Float_t *tt);
   void    DrawFaceMove2(Int_t *icodes, Float_t *xyz, Int_t np, Int_t *iface, Float_t *tt);
   void    DrawFaceRaster1(Int_t *icodes, Float_t *xyz, Int_t np, Int_t *iface, Float_t *tt);
   void    DrawFaceRaster2(Int_t *icodes, Float_t *xyz, Int_t np, Int_t *iface, Float_t *tt);
   void    FillPolygon(Int_t n, Float_t *p, Float_t *f);
   void    FillPolygonBorder(Int_t nn, Float_t *xy);
   void    FindLevelLines(Int_t np, Float_t *f, Float_t *t);
   void    FindPartEdge(Float_t *p1, Float_t *p2, Float_t f1, Float_t f2, Float_t fmin, Float_t fmax, Int_t &kpp, Float_t *pp);
   void    FindVisibleLine(Float_t *p1, Float_t *p2, Int_t ntmax, Int_t &nt, Float_t *t);
   void    FindVisibleDraw(Float_t *r1, Float_t *r2);
   void    FrontBox(Float_t ang);
   void    GouraudFunction(Int_t ia, Int_t ib, Float_t *f, Float_t *t);
   void    InitMoveScreen(Float_t xmin, Float_t xmax);
   void    InitRaster(Float_t xmin, Float_t ymin, Float_t xmax, Float_t ymax, Int_t nx, Int_t ny);
   void    LegoCartesian(Float_t ang, Int_t nx, Int_t ny, const char *chopt);
   void    LegoFunction(Int_t ia, Int_t ib, Int_t &nv, Float_t *ab, Float_t *vv, Float_t *t);
   void    LegoPolar(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LegoCylindrical(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LegoSpherical(Int_t ipsdr, Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LightSource(Int_t nl, Float_t yl, Float_t xscr, Float_t yscr, Float_t zscr, Int_t &irep);
   void    Luminosity(Float_t *anorm, Float_t &flum);
   void    ModifyScreen(Float_t *r1, Float_t *r2);
   void    SetDrawFace(DrawFaceFunc_t pointer);
   void    SetLegoFunction(LegoFunc_t pointer);
   void    SetMesh(Int_t mesh=1) {fMesh=mesh;}
   void    SetSurfaceFunction(SurfaceFunc_t pointer);
   void    SetColorDark(Color_t color, Int_t n=0);
   void    SetColorMain(Color_t color, Int_t n=0);
   void    SideVisibilityDecode(Float_t val, Int_t &iv1, Int_t &iv2, Int_t &iv3, Int_t &iv4, Int_t &iv5, Int_t &iv6, Int_t &ir);
   void    SideVisibilityEncode(Int_t iopt, Float_t phi1, Float_t phi2, Float_t &val);
   void    Spectrum(Int_t nl, Float_t fmin, Float_t fmax, Int_t ic, Int_t idc, Int_t &irep);
   void    SurfaceCartesian(Float_t ang, Int_t nx, Int_t ny, const char *chopt);
   void    SurfacePolar(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    SurfaceCylindrical(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    SurfaceFunction(Int_t ia, Int_t ib, Float_t *f, Float_t *t);
   void    SurfaceSpherical(Int_t ipsdr, Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    SurfaceProperty(Float_t qqa, Float_t qqd, Float_t qqs, Int_t nnqs, Int_t &irep);

   ClassDef(TLego,0)   //Hidden line removal package
};

#endif



