// @(#)root/geom:$Name:  $:$Id: $
// Author: Andrei Gheata   17/02/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoPhysicalNode
#define ROOT_TGeoPhysicalNode

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

#ifndef ROOT_TObject
#include "TGeoNode.h"
#endif

// forward declarations
class TGeoHMatrix;
class TGeoVolume;

/*************************************************************************
 * TGeoPhysicalNode - class representing an unique object associated with a
 *   path.
 *
 *************************************************************************/

class TGeoPhysicalNode : public TObject,
                 public TAttLine
{
protected:
   Int_t             fLevel;          // depth in the geometry tree
   TObjArray        *fMatrices;       // global transformation matrices
   TObjArray        *fNodes;          // branch of nodes
public:
   enum {
      kGeoPNodeFull    = BIT(10),     // full branch is visible (default only last node)
      kGeoPNodeVisible = BIT(11),     // this node is visible (default)
      kGeoPNodeVolAtt  = BIT(12)      // preserve volume attributes (default)
   };

   // constructors
   TGeoPhysicalNode();
   TGeoPhysicalNode(const char *path);
   // destructor
   virtual ~TGeoPhysicalNode();

   void              Align(TGeoMatrix *newmat=0, TGeoShape *newshape=0);
   void              cd() const;
   void              Draw(Option_t *option="");
   Int_t             GetLevel() const {return fLevel;}
   TGeoHMatrix      *GetMatrix(Int_t level=-1) const;
   TGeoNode         *GetMother(Int_t levup=1) const;
   TGeoNode         *GetNode(Int_t level=-1) const;
   TGeoShape        *GetShape(Int_t level=-1) const;
   TGeoVolume       *GetVolume(Int_t level=-1) const;
   
 
   Bool_t            IsVolAttributes() const {return TObject::TestBit(kGeoPNodeVolAtt);}
   Bool_t            IsVisible() const {return TObject::TestBit(kGeoPNodeVisible);}
   Bool_t            IsVisibleFull() const {return TObject::TestBit(kGeoPNodeFull);}

   Bool_t            SetPath(const char *path);
   void              SetBranchAsState();

   void              SetIsVolAtt(Bool_t flag=kTRUE) {TObject::SetBit(kGeoPNodeVolAtt,flag);}
   void              SetVisibility(Bool_t flag=kTRUE)  {TObject::SetBit(kGeoPNodeVisible,flag);}
   void              SetVisibleFull(Bool_t flag=kTRUE) {TObject::SetBit(kGeoPNodeFull,flag);}
   virtual void      Paint(Option_t *option = "");


  ClassDef(TGeoPhysicalNode, 1)               // base class for physical nodes
};

#endif

