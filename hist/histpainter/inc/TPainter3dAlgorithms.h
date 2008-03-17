// @(#)root/histpainter:$Id$
// Author: Rene Brun, Evgueni Tcherniaev, Olivier Couet   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPainter3dAlgorithms
#define ROOT_TPainter3dAlgorithms


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPainter3dAlgorithms                                                 //
//                                                                      //
// 3D graphics representations package.                                 //
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

class TF3;

class TPainter3dAlgorithms : public TObject, public TAttLine, public TAttFill {

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
   Int_t       *fColorMain;        //
   Int_t       *fColorDark;        //
   Int_t        fColorTop;         //
   Int_t        fColorBottom;      //
   Int_t        fMesh;             //(=1 if mesh to draw, o otherwise)
   Int_t        fNlines;           //
   Int_t        fLevelLine[200];   //
   Int_t        fLoff;             //
   Int_t        fNqs;              //
   Int_t        fNStack;           //Number of histograms in the stack to be painted
   Int_t        fNxrast;           //
   Int_t        fNyrast;           //
   Int_t        fIfrast;           //
   Int_t        *fRaster;          //pointer to raster buffer
   Int_t        fJmask[30];        //
   Int_t        fMask[465];        //
   Double_t     fP8[8][3];         //
   Double_t     fF8[8];            //
   Double_t     fG8[8][3];         //
   Double_t     fFmin;             // IsoSurface minimum function value
   Double_t     fFmax;             // IsoSurface maximum function value
   Int_t        fNcolor;           // Number of colours per Iso surface
   Int_t        fIc1;              // Base colour for the 1st Iso Surface
   Int_t        fIc2;              // Base colour for the 2nd Iso Surface
   Int_t        fIc3;              // Base colour for the 3rd Iso Surface

   static Int_t    fgF3Clipping;   // Clipping box is off (0) or on (1)
   static Double_t fgF3XClip;      // Clipping plne along X
   static Double_t fgF3YClip;      // Clipping plne along Y
   static Double_t fgF3ZClip;      // Clipping plne along Y
   static TF3      *fgCurrentF3;   // Pointer to the 3D function to be paint.


public:
   typedef void (TPainter3dAlgorithms::*DrawFaceFunc_t)(Int_t *, Double_t *, Int_t, Int_t *, Double_t *);
   typedef void (TPainter3dAlgorithms::*LegoFunc_t)(Int_t,Int_t,Int_t&,Double_t*,Double_t*,Double_t*);
   typedef void (TPainter3dAlgorithms::*SurfaceFunc_t)(Int_t,Int_t,Double_t*,Double_t*);

private:
   DrawFaceFunc_t  fDrawFace;        //pointer to face drawing function
   LegoFunc_t      fLegoFunction;    //pointer to lego function
   SurfaceFunc_t   fSurfaceFunction; //pointer to surface function
   
public:
   TPainter3dAlgorithms();
   TPainter3dAlgorithms(Double_t *rmin, Double_t *rmax, Int_t system=1);
   virtual ~TPainter3dAlgorithms();
   void    BackBox(Double_t ang);
   void    ClearRaster();
   void    ColorFunction(Int_t nl, Double_t *fl, Int_t *icl, Int_t &irep);
   void    DefineGridLevels(Int_t ndivz);
   void    DrawFaceGouraudShaded(Int_t *icodes, Double_t xyz[][3], Int_t np, Int_t *iface, Double_t *t);
   void    DrawFaceMode1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t);
   void    DrawFaceMode2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t);
   void    DrawFaceMode3(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *t);
   void    DrawFaceMove1(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt);
   void    DrawFaceMove2(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt);
   void    DrawFaceMove3(Int_t *icodes, Double_t *xyz, Int_t np, Int_t *iface, Double_t *tt);
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
   void    ImplicitFunction(Double_t *rmin, Double_t *rmax, Int_t nx, Int_t ny, Int_t nz, const char *chopt);
   void    IsoSurface (Int_t ns, Double_t *s, Int_t nx, Int_t ny, Int_t nz, Double_t *x, Double_t *y, Double_t *z, const char *chopt);
   void    InitMoveScreen(Double_t xmin, Double_t xmax);
   void    InitRaster(Double_t xmin, Double_t ymin, Double_t xmax, Double_t ymax, Int_t nx, Int_t ny);
   void    LegoCartesian(Double_t ang, Int_t nx, Int_t ny, const char *chopt);
   void    LegoFunction(Int_t ia, Int_t ib, Int_t &nv, Double_t *ab, Double_t *vv, Double_t *t);
   void    LegoPolar(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LegoCylindrical(Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LegoSpherical(Int_t ipsdr, Int_t iordr, Int_t na, Int_t nb, const char *chopt);
   void    LightSource(Int_t nl, Double_t yl, Double_t xscr, Double_t yscr, Double_t zscr, Int_t &irep);
   void    Luminosity(Double_t *anorm, Double_t &flum);
   void    MarchingCube(Double_t fiso, Double_t p[8][3], Double_t f[8], Double_t g[8][3], Int_t &nnod, Int_t &ntria, Double_t xyz[][3], Double_t grad[][3], Int_t itria[][3]);
   void    MarchingCubeCase00(Int_t k1, Int_t k2, Int_t k3, Int_t k4, Int_t k5, Int_t k6, Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeCase03(Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeCase04(Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeCase06(Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeCase07(Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeCase10(Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeCase12(Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeCase13(Int_t &nnod, Int_t &ntria, Double_t xyz[52][3], Double_t grad[52][3], Int_t itria[48][3]);
   void    MarchingCubeSetTriangles(Int_t ntria, Int_t it[][3], Int_t itria[48][3]);
   void    MarchingCubeMiddlePoint(Int_t nnod, Double_t xyz[52][3], Double_t grad[52][3], Int_t it[][3], Double_t *pxyz, Double_t *pgrad);
   void    MarchingCubeSurfacePenetration(Double_t a00, Double_t a10, Double_t a11, Double_t a01, Double_t b00, Double_t b10, Double_t b11, Double_t b01, Int_t &irep);
   void    MarchingCubeFindNodes(Int_t nnod, Int_t *ie, Double_t xyz[52][3], Double_t grad[52][3]);
   void    ModifyScreen(Double_t *r1, Double_t *r2);
   void    SetDrawFace(DrawFaceFunc_t pointer);
   void    SetIsoSurfaceParameters(Double_t fmin, Double_t fmax, Int_t ncolor, Int_t ic1, Int_t ic2, Int_t ic3){fFmin=fmin; fFmax=fmax; fNcolor=ncolor; fIc1=ic1; fIc2=ic2; fIc3=ic3;}
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
   void    TestEdge(Double_t del, Double_t xyz[52][3], Int_t i1, Int_t i2, Int_t iface[3], Double_t abcd[4], Int_t &irep);
   void    ZDepth(Double_t xyz[52][3], Int_t &nface, Int_t iface[48][3], Double_t dface[48][6], Double_t abcd[48][4], Int_t *iorder);

   static void    SetF3(TF3 *f3);
   static void    SetF3ClippingBoxOff();
   static void    SetF3ClippingBoxOn(Double_t xclip, Double_t yclip, Double_t zclip);

   ClassDef(TPainter3dAlgorithms,0)   //Hidden line removal package
};

#endif



