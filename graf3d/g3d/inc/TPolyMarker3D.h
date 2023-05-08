// @(#)root/g3d:$Id$
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

#include "TObject.h"
#include "TAttMarker.h"
#include "TAtt3D.h"
#include "TString.h"

class TH1;
class TCollection;

class TPolyMarker3D : public TObject, public TAttMarker, public TAtt3D
{
protected:
   Int_t            fN{0};            //Number of allocated points
   Float_t         *fP{nullptr};      //[3*fN] Array of X,Y,Z coordinates
   TString          fOption;          //Options
   Int_t            fLastPoint{-1};   //The index of the last filled point
   TString          fName;            //Name of polymarker

public:
   TPolyMarker3D();
   TPolyMarker3D(Int_t n, Marker_t marker=1, Option_t *option="");
   TPolyMarker3D(Int_t n, Float_t *p, Marker_t marker=1, Option_t *option="");
   TPolyMarker3D(Int_t n, Double_t *p, Marker_t marker=1, Option_t *option="");
   TPolyMarker3D(const TPolyMarker3D &p);
   TPolyMarker3D& operator=(const TPolyMarker3D&);
   virtual ~TPolyMarker3D();

   void              Copy(TObject &polymarker) const override;
   Int_t             DistancetoPrimitive(Int_t px, Int_t py)  override;
   void              Draw(Option_t *option="") override;
   virtual void      DrawPolyMarker(Int_t n, Float_t *p, Marker_t marker, Option_t *option="");
   void              ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   virtual Int_t     GetLastPoint() const { return fLastPoint;}
   const char       *GetName() const override { return fName.Data(); }
   virtual Int_t     GetN() const { return fN;}
   virtual Float_t  *GetP() const { return fP;}
   virtual void      GetPoint(Int_t n, Float_t &x, Float_t &y, Float_t &z) const;
   virtual void      GetPoint(Int_t n, Double_t &x, Double_t &y, Double_t &z) const;
   Option_t         *GetOption() const  override { return fOption.Data(); }
   void              ls(Option_t *option="") const override;
   virtual Int_t     Merge(TCollection *list);
   void              Paint(Option_t *option="") override;
   void              Print(Option_t *option="") const override;
   void              SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void      SetName(const char *name); // *MENU*
   void              SetPoint(Int_t n, Double_t x, Double_t y, Double_t z); // *MENU*
   virtual void      SetPolyMarker(Int_t n, Float_t *p, Marker_t marker, Option_t *option="");
   virtual void      SetPolyMarker(Int_t n, Double_t *p, Marker_t marker, Option_t *option="");
   virtual Int_t     SetNextPoint(Double_t x, Double_t y, Double_t z); // *MENU*
   virtual Int_t     Size() const {return fLastPoint+1;}

   static  void      PaintH3(TH1 *h, Option_t *option);

   ClassDefOverride(TPolyMarker3D,3);  //An array of 3-D points with the same marker
};

#endif
