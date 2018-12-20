// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REvePointSet
#define ROOT7_REvePointSet

#include <ROOT/REveElement.hxx>
#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveTreeTools.hxx>

#include "TPointSet3D.h"

class TArrayI;

namespace ROOT {
namespace Experimental {

/******************************************************************************/
// REvePointSet
/******************************************************************************/

class REvePointSet : public REveElement, public TPointSet3D, public REvePointSelectorConsumer, public REveProjectable {
   friend class REvePointSetArray;

private:
   REvePointSet &operator=(const REvePointSet &); // Not implemented

protected:
   TString fTitle;            // Title/tooltip of the REvePointSet.
   TArrayI *fIntIds{nullptr}; // Optional array of integer ideices.
   Int_t fIntIdsPerPoint;     // Number of integer indices assigned to each point.

   void AssertIntIdsSize();

public:
   REvePointSet(Int_t n_points = 0, ETreeVarType_e tv_type = kTVT_XYZ);
   REvePointSet(const char *name, Int_t n_points = 0, ETreeVarType_e tv_type = kTVT_XYZ);
   REvePointSet(const REvePointSet &e);
   virtual ~REvePointSet();

   virtual TObject *GetObject(const REveException &) const
   {
      const TObject *obj = this;
      return const_cast<TObject *>(obj);
   }

   virtual REvePointSet *CloneElement() const { return new REvePointSet(*this); }

   virtual void ClonePoints(const REvePointSet &e);

   void Reset(Int_t n_points = 0, Int_t n_int_ids = 0);
   Int_t GrowFor(Int_t n_points);

   virtual const char *GetTitle() const { return fTitle; }
   virtual const char *GetElementName() const { return TPointSet3D::GetName(); }
   virtual const char *GetElementTitle() const { return fTitle; }
   virtual void SetElementName(const char *n)
   {
      fName = n;
      NameTitleChanged();
   }
   virtual void SetTitle(const char *t)
   {
      fTitle = t;
      NameTitleChanged();
   }
   virtual void SetElementTitle(const char *t)
   {
      fTitle = t;
      NameTitleChanged();
   }
   virtual void SetElementNameTitle(const char *n, const char *t)
   {
      fName = n;
      fTitle = t;
      NameTitleChanged();
   }

   Int_t GetIntIdsPerPoint() const { return fIntIdsPerPoint; }
   Int_t *GetPointIntIds(Int_t p) const;
   Int_t GetPointIntId(Int_t p, Int_t i) const;

   void SetPointIntIds(Int_t *ids);
   void SetPointIntIds(Int_t n, Int_t *ids);

   virtual void SetMarkerColor(Color_t col) { SetMainColor(col); }
   virtual void SetMarkerStyle(Style_t mstyle = 1);
   virtual void SetMarkerSize(Size_t msize = 1);

   virtual void InitFill(Int_t subIdNum);
   virtual void TakeAction(REvePointSelector *);

   virtual void PointSelected(Int_t id); // *SIGNAL*

   virtual void CopyVizParams(const REveElement *el);
   virtual void WriteVizParams(std::ostream &out, const TString &var);

   virtual TClass *ProjectedClass(const REveProjection *p) const;

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset); // override;
   void BuildRenderData();                                   // override;

   ClassDef(REvePointSet, 0); // Set of 3D points with same marker attributes; optionally each point can be assigned an
                              // external TRef or a number of integer indices.
};

/******************************************************************************/
// REvePointSetArray
/******************************************************************************/

class REvePointSetArray : public REveElement, public TNamed, public TAttMarker, public REvePointSelectorConsumer {
   REvePointSetArray(const REvePointSetArray &);            // Not implemented
   REvePointSetArray &operator=(const REvePointSetArray &); // Not implemented

protected:
   REvePointSet **fBins{nullptr}; //  Pointers to subjugated REvePointSet's.
   Int_t fDefPointSetCapacity;    //  Default capacity of subjugated REvePointSet's.
   Int_t fNBins;                  //  Number of subjugated REvePointSet's.
   Int_t fLastBin;                //! Index of the last filled REvePointSet.
   Double_t fMin, fCurMin;        //  Overall and current minimum value of the separating quantity.
   Double_t fMax, fCurMax;        //  Overall and current maximum value of the separating quantity.
   Double_t fBinWidth;            //  Separating quantity bin-width.
   TString fQuantName;            //  Name of the separating quantity.

public:
   REvePointSetArray(const char *name = "REvePointSetArray", const char *title = "");
   virtual ~REvePointSetArray();

   virtual void RemoveElementLocal(REveElement *el);
   virtual void RemoveElementsLocal();

   virtual void SetMarkerColor(Color_t tcolor = 1);
   virtual void SetMarkerStyle(Style_t mstyle = 1);
   virtual void SetMarkerSize(Size_t msize = 1);

   virtual void TakeAction(REvePointSelector *);

   virtual Int_t Size(Bool_t under = kFALSE, Bool_t over = kFALSE) const;

   void InitBins(const char *quant_name, Int_t nbins, Double_t min, Double_t max);
   Bool_t Fill(Double_t x, Double_t y, Double_t z, Double_t quant);
   void SetPointId(TObject *id);
   void CloseBins();

   void SetOwnIds(Bool_t o);

   Int_t GetDefPointSetCapacity() const { return fDefPointSetCapacity; }
   void SetDefPointSetCapacity(Int_t c) { fDefPointSetCapacity = c; }

   Int_t GetNBins() const { return fNBins; }
   REvePointSet *GetBin(Int_t bin) const { return fBins[bin]; }

   Double_t GetMin() const { return fMin; }
   Double_t GetCurMin() const { return fCurMin; }
   Double_t GetMax() const { return fMax; }
   Double_t GetCurMax() const { return fCurMax; }

   void SetRange(Double_t min, Double_t max);

   ClassDef(REvePointSetArray, 0); // Array of REvePointSet's filled via a common point-source; range of displayed REvePointSet's can be
                                   // controlled, based on a separating quantity provided on fill-time by a user.
};

/******************************************************************************/
// REvePointSetProjected
/******************************************************************************/

class REvePointSetProjected : public REvePointSet, public REveProjected {
private:
   REvePointSetProjected(const REvePointSetProjected &);            // Not implemented
   REvePointSetProjected &operator=(const REvePointSetProjected &); // Not implemented

protected:
   virtual void SetDepthLocal(Float_t d);

public:
   REvePointSetProjected();
   virtual ~REvePointSetProjected() {}

   virtual void SetProjection(REveProjectionManager *proj, REveProjectable *model);
   virtual void UpdateProjection();
   virtual REveElement *GetProjectedAsElement() { return this; }

   virtual void PointSelected(Int_t id);

   ClassDef(REvePointSetProjected, 0); // Projected copy of a REvePointSet.
};

} // namespace Experimental
} // namespace ROOT

#endif
