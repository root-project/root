// @(#)root/treeviewer:$Name:  $:$Id: TParallelCoordRange.h,v 1.1 2007/07/24 20:00:46 brun Exp $
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TParallelRange
#define ROOT_TParallelRange

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParallelCoordRange                                                  //
//                                                                      //
// A TParallelCoordRange is a range used for parallel                   //
// coordinates plots.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TParallelCoordVar;
class TParallelCoord;
class TParallelCoordSelect;
class TPoint;

class TParallelCoordRange : public TNamed, public TAttLine {
public:
   enum {
      kShowOnPad = BIT(15),
      kLiveUpdate = BIT(16)
   };

private:
   Double_t          fMin;    // Min value for the range.
   Double_t          fMax;    // Max value for the range.
   const Double_t    fSize;   // Size of the painted range.
   TParallelCoordVar *fVar;   // Variable owning the range.
   TParallelCoordSelect* fSelect; // Selection owning the range.
   
   void              PaintSlider(Double_t value,Bool_t fill=kFALSE);
   TPoint*           GetBindingLinePoints(Int_t pos,Int_t mindragged);
   TPoint*           GetSliderPoints(Double_t value);
   TPoint*           GetSliderPoints(Int_t pos);
   
public:
   TParallelCoordRange();
   ~TParallelCoordRange();
   TParallelCoordRange(TParallelCoordVar *var, TParallelCoordSelect* sel, Double_t min=0, Double_t max=0);
   
   virtual void BringOnTop() ;// *MENU*
   virtual void Delete(const Option_t* options=""); // *MENU*
   virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
   virtual void Draw(Option_t *options="");
   virtual void ExecuteEvent(Int_t entry, Int_t px, Int_t py);
   TParallelCoordVar* GetVar() {return fVar;}
   Bool_t IsIn(Double_t evtval);
   virtual void Paint(Option_t *options);
   virtual void SendToBack(); // *MENU*
   virtual void SetLineColor(Color_t col);
   virtual void SetLineWidth(Width_t wid);
   
   ClassDef(TParallelCoordRange,1);
};

#endif

#ifndef ROOT_TParallelCoordSelect
#define ROOT_TParallelCoordSelect

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParallelCoordSelect                                                 //
//                                                                      //
// A TParallelCoordSelect is a specialised TList to hold                //
// TParallelCoordRanges used by TParallelCoord.                         //
//                                                                      //
// Selections of specific entries can be defined over the data set      //
// using parallel coordinates. With that representation, a selection is //
// an ensemble of ranges defined on the axes. Ranges defined on the     //
// same axis are conjugated with OR (an entry must be in one or the     //
// other ranges to be selected). Ranges on different axes are           //
// are conjugated with AND (an entry must be in all the ranges to be    //
// selected).                                                           //
// Several selections can be defined with different colors. It is       //
// possible to generate an entry list from a given selection and apply  //
// it to the tree using the editor ("Apply to tree" button).            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TParallelCoord;
class TParallelCoordRange;

class TParallelCoordSelect : public TList, public TAttLine {
public:
   enum {
      kActivated = BIT(18),
      kShowRanges = BIT(19)
   };

private:
   TString fTitle;
   
public:
   TParallelCoordSelect();    // Default constructor.
   TParallelCoordSelect(const char* title); // Normal constructor.
   ~TParallelCoordSelect();   // Destructor.
   
   const char* GetTitle() const {return fTitle.Data();}
   void        SetActivated(Bool_t on);
   void        SetShowRanges(Bool_t s);
   void        SetTitle(const char* title) {fTitle = title;}
   
   ClassDef(TParallelCoordSelect,0);
};

#endif

