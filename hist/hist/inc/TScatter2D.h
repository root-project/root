// @(#)root/hist:$Id$
// Author: Olivier Couet   03/12/2024

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TScatter2D
#define ROOT_TScatter2D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TScatter2D                                                           //
//                                                                      //
// A scatter plot able to draw five variables on a single plot          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TAttMarker.h"
#include "TGraph2D.h"

class TH3F;

class TScatter2D : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:
   Int_t      fNpoints{-1};        ///< Number of points of arrays fX, fY and fZ
   TH3F      *fHistogram{nullptr}; ///< Pointer to histogram used for drawing axis
   TGraph2D  *fGraph{nullptr};     ///< Pointer to graph holding X, Y and Z positions
   Double_t  *fColor{nullptr};     ///< [fNpoints] array of colors
   Double_t  *fSize{nullptr};      ///< [fNpoints] array of marker sizes
   Double_t   fMaxMarkerSize{5.};  ///< Largest marker size used to paint the markers
   Double_t   fMinMarkerSize{1.};  ///< Smallest marker size used to paint the markers
   Double_t   fMargin{.1};         ///< Margin around the plot in %

public:
   TScatter2D();
   TScatter2D(Int_t n);
   TScatter2D(Int_t n, Double_t *x, Double_t *y, Double_t *z, const Double_t *col = nullptr, const Double_t *size = nullptr);
   ~TScatter2D() override;

   Int_t     DistancetoPrimitive(Int_t px, Int_t py) override;
   void      ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   Double_t *GetColor()  const {return fColor;}                ///< Get the array of colors
   Double_t *GetSize()   const {return fSize;}                 ///< Get the array of marker sizes
   Double_t  GetMargin() const {return fMargin;}               ///< Set the margin around the plot in %
   Double_t  GetMaxMarkerSize() const {return fMaxMarkerSize;} ///< Get the largest marker size used to paint the markers
   Double_t  GetMinMarkerSize() const {return fMinMarkerSize;} ///< Get the smallest marker size used to paint the markers
   TGraph2D *GetGraph()  const {return fGraph;}                ///< Get the graph holding X, Y and Z positions
   TH3F     *GetHistogram() const;                             ///< Get the graph histogram used for drawing axis
   TAxis    *GetXaxis() const ;
   TAxis    *GetYaxis() const ;
   TAxis    *GetZaxis() const ;

   void      SetMaxMarkerSize(Double_t max) {fMaxMarkerSize = max;} ///< Set the largest marker size used to paint the markers
   void      SetMinMarkerSize(Double_t min) {fMinMarkerSize = min;} ///< Set the smallest marker size used to paint the markers
   void      SetMargin(Double_t);
   void      SetHistogram(TH3F *h) {fHistogram = h;}
   void      Print(Option_t *chopt="") const override;
   void      SavePrimitive(std::ostream &out, Option_t *option = "") override;
   void      Paint(Option_t *chopt="") override;


   ClassDefOverride(TScatter2D,1)  //A 2D scatter plot
};
#endif

