// @(#)root/geom:$Name:  $:$Id: TGeoNode.h,v 1.12 2003/01/20 14:35:48 brun Exp $
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoNode
#define ROOT_TGeoNode

#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif

#ifndef ROOT_TGeoAtt
#include "TGeoAtt.h"
#endif

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TGeoVolume
#include "TGeoVolume.h"
#endif

#ifndef ROOT_TGeoPatternFinder
#include "TGeoPatternFinder.h"
#endif

#ifndef ROOT_TGeoVoxelFinder
#include "TGeoVoxelFinder.h"
#endif

// forward declarations
class TGeoVolume;
class TGeoShape;
class TGeoMedium;
class TGeoMatrix;

/*************************************************************************
 * TGeoNode - base class for logical nodes. They represent volumes
 *   positioned inside a mother volume
 *
 *************************************************************************/

class TGeoNode : public TNamed,
                 public TGeoAtt
{
protected:
   TGeoVolume       *fVolume;         // volume associated with this
   TGeoVolume       *fMother;         // mother volume
   Int_t             fNumber;         // copy number
   Int_t             fNovlp;          // number of overlaps
   Int_t            *fOverlaps;       //[fNovlp] list of indices for overlapping brothers
public:
   enum {
      kGeoNodeMatrix = BIT(10),
      kGeoNodeOffset = BIT(11),
      kGeoNodeVC     = BIT(12),
      kGeoNodeOverlap = BIT(13)
   };

   // constructors
   TGeoNode();
   TGeoNode(const TGeoVolume *vol);
   // destructor
   virtual ~TGeoNode();

   void              Browse(TBrowser *b);
   virtual void      cd() const {;}
   void              CheckShapes();
   void              Draw(Option_t *option="");
   void              DrawOnly(Option_t *option="");
   void              DrawOverlaps(); // *MENU*
   Int_t             FindNode(const TGeoNode *node, Int_t level);
   virtual Int_t     GetByteCount() const {return 44;}
   TGeoNode         *GetDaughter(Int_t ind) const {return fVolume->GetNode(ind);}
   virtual TGeoMatrix *GetMatrix() const = 0;

   Int_t             GetColour() const {return fVolume->GetLineColor();}
   virtual Int_t     GetIndex() const                    {return 0;}
   virtual TGeoPatternFinder *GetFinder() const          {return 0;}
   TGeoMedium       *GetMedium() const                   {return fVolume->GetMedium();}
   TGeoVolume       *GetMotherVolume() const             {return fMother;}
   Int_t             GetNdaughters() const {return fVolume->GetNdaughters();}
   TObjArray        *GetNodes() const {return fVolume->GetNodes();}
   Int_t             GetNumber() const {return fNumber;}
   Int_t            *GetOverlaps(Int_t &novlp) const {novlp=fNovlp; return fOverlaps;}
   TGeoVolume       *GetVolume() const                   {return fVolume;}
   virtual Int_t     GetOptimalVoxels() const {return 0;}
   void              InspectNode() const; // *MENU*
   virtual Bool_t    IsFolder() const {return kTRUE;}
   Bool_t            IsOffset() const {return TObject::TestBit(kGeoNodeOffset);}
   Bool_t            IsOnScreen() const; // *MENU*
   Bool_t            IsOverlapping() const {return TObject::TestBit(kGeoNodeOverlap);}
   Bool_t            IsVirtual() const {return TObject::TestBit(kGeoNodeVC);}
   Bool_t            IsVisible() const {return (TGeoAtt::IsVisible() && fVolume->IsVisible());}
   Bool_t            IsVisDaughters() const {return (TGeoAtt::IsVisDaughters() && fVolume->IsVisDaughters());}

   virtual TGeoNode *MakeCopyNode() const {return 0;}
   void              SaveAttributes(ofstream &out);
   void              SetCurrentPoint(Double_t x, Double_t y, Double_t z) {fVolume->SetCurrentPoint(x,y,z);}// *MENU*
   void              SetVolume(const TGeoVolume *volume) {fVolume = (TGeoVolume*)volume;}
   void              SetNumber(Int_t number)             {fNumber=number;}
   void              SetOverlapping()                    {TObject::SetBit(kGeoNodeOverlap, kTRUE);}
   void              SetVirtual()                        {TObject::SetBit(kGeoNodeVC, kTRUE);}
   void              SetVisibility(Bool_t vis=kTRUE); // *MENU*
   void              SetInvisible()                      {SetVisibility(kFALSE);} // *MENU*
   void              SetAllInvisible()                   {VisibleDaughters(kFALSE);} // *MENU*
   void              SetMotherVolume(const TGeoVolume *mother) {fMother = (TGeoVolume*)mother;}
   void              SetOverlaps(Int_t *ovlp, Int_t novlp);

   virtual void      MasterToLocal(const Double_t *master, Double_t *local) const;
   virtual void      MasterToLocalVect(const Double_t *master, Double_t *local) const;
   virtual void      LocalToMaster(const Double_t *local, Double_t *master) const;
   virtual void      LocalToMasterVect(const Double_t *local, Double_t *master) const;

   virtual void      ls(Option_t *option = "") const;
   virtual void      Paint(Option_t *option = "");
   void              PrintCandidates() const; // *MENU*
   void              PrintOverlaps() const; // *MENU*
   void              VisibleDaughters(Bool_t vis=kTRUE); // *MENU*

  ClassDef(TGeoNode, 2)               // base class for all geometry nodes
};

/*************************************************************************
 * TGeoNodeMatrix - node containing a general transformation
 *
 *************************************************************************/

class TGeoNodeMatrix : public TGeoNode
{
private:
   TGeoMatrix       *fMatrix;         // transf. matrix of fNode in fMother system

public:
   // constructors
   TGeoNodeMatrix();
   TGeoNodeMatrix(const TGeoVolume *vol, const TGeoMatrix *matrix);
   // destructor
   virtual ~TGeoNodeMatrix();

   virtual Int_t     GetByteCount() const;
   virtual Int_t     GetOptimalVoxels() const;
   virtual Bool_t    IsFolder() const {return kTRUE;}
   virtual TGeoMatrix *GetMatrix() const   {return fMatrix;}
   virtual TGeoNode *MakeCopyNode() const;
   void              SetMatrix(const TGeoMatrix *matrix) {fMatrix = (TGeoMatrix*)matrix;}

  ClassDef(TGeoNodeMatrix, 1)               // a geometry node in the general case
};

/*************************************************************************
 * TGeoNodeOffset - node containing only an translation offset
 *
 *************************************************************************/

class TGeoNodeOffset : public TGeoNode
{
private:
   Double_t          fOffset; // X offset for this node with respect to its mother
   Int_t             fIndex;  // index of this node in the division
   TGeoPatternFinder *fFinder; // finder for this node
public:
   // constructors
   TGeoNodeOffset();
   TGeoNodeOffset(const TGeoVolume *vol, Int_t index, Double_t offset);
   // destructor
   virtual ~TGeoNodeOffset();

   virtual void      cd() const           {fFinder->cd(fIndex);}
   Double_t          GetOffset() const {return fOffset;}
   virtual Int_t     GetIndex() const;
   virtual TGeoPatternFinder *GetFinder() const {return fFinder;}
   virtual TGeoMatrix *GetMatrix() const {cd(); return fFinder->GetMatrix();}
   virtual TGeoNode *MakeCopyNode() const;
   void              SetFinder(const TGeoPatternFinder *finder) {fFinder = (TGeoPatternFinder*)finder;}

  ClassDef(TGeoNodeOffset, 1)      // a geometry node with just an offset
};

#endif

