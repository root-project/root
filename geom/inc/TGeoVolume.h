// @(#)root/geom:$Name:  $:$Id: TGeoVolume.h,v 1.3 2002/07/10 19:24:16 brun Exp $
// Author: Andrei Gheata   30/05/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :                  date : Wed 24 Oct 2001 01:39:36 PM CEST

#ifndef ROOT_TGeoVolume
#define ROOT_TGeoVolume


#ifndef ROOT_TGeoAtt
#include "TGeoAtt.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

#ifndef ROOT_TAtt3D
#include "TAtt3D.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TGeoShape
#include "TGeoShape.h"
#endif

#ifndef ROOT_TGeoMaterial
#include "TGeoMaterial.h"
#endif

// forward declarations
class TGeoNode;
class TGeoShape;
class TGeoMatrix;
class TGeoVoxelFinder;
class TGeoPatternFinder;

/*************************************************************************
 * TGeoVolume - class description 
 *
 *************************************************************************/

class TGeoVolume : public TNamed,
                   public TGeoAtt,
                   public TAttLine,
                   public TAttFill,
                   public TAtt3D
{
protected :
   enum EGeoVolumeTypes {
      kVolumeDiv     =  BIT(16),
      kVolumeOverlap =  BIT(17),
      kVolumeImportNodes=  BIT(18),
      kVolumeMulti   = BIT(19)
   };
// data members
   Double_t          fUsageCount[2];  // usage of this volume in tracking !!!
   TObjArray        *fNodes;          // array of nodes inside this volume
   TGeoShape        *fShape;          // shape
   TGeoMaterial     *fMaterial;       // material
   TGeoVoxelFinder  *fVoxels;         // finder object for bounding boxes
   TGeoPatternFinder *fFinder;        // finder object for divisions

   TObject          *fField;          // !!! just a hook for now
   TString           fOption;         // option - if any
// methods
public:
   // constructors
   TGeoVolume();
   TGeoVolume(const char *name, TGeoShape *shape=0, TGeoMaterial *mat=0);

   // destructor
   virtual ~TGeoVolume();
   // methods
   virtual void    cd(Int_t inode) const;
   void            Browse(TBrowser *b);
   void            CheckShapes();
   void            ClearNodes() {fNodes = 0;}
   void            ClearShape();
   void            CleanAll();
   Int_t           CountNodes(Int_t nlevels=1000) const; // *MENU*
   Bool_t          Contains(Double_t *point) const {return fShape->Contains(point);}
   Bool_t          IsFolder() const;
   Bool_t          IsRunTime() const {return fShape->IsRunTimeShape();}
   virtual void    AddNode(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option="");       // most general case
   void            AddNodeOffset(TGeoVolume *vol, Int_t copy_no, Double_t offset, Option_t *option="");
   virtual void    AddNodeOverlap(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option="");

   virtual TGeoVolume *Divide(const char *divname, Int_t ndiv, Option_t *option="");
   virtual TGeoVolume *Divide(const char *divname, Int_t ndiv, Double_t start, Double_t step, Option_t *option="");
   virtual TGeoVolume *Divide(const char *divname, Double_t start, Double_t end, Double_t step, Option_t *option="");
   virtual TGeoVolume *Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step);
   virtual TGeoVolume *Divide(const char *divname, Int_t iaxis, Double_t step);
   virtual TGeoVolume *Divide(const char *divname, TObject *userdiv, Double_t *params, Option_t *option="");

   virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
   virtual void    Draw(Option_t *option=""); // *MENU*
   virtual void    DrawOnly(Option_t *option=""); // *MENU*
   virtual void    Paint(Option_t *option="");
   void            PrintNodes() const;
   void            PrintVoxels() const; // *MENU*
   virtual void    ExecuteEvent(Int_t event, Int_t px, Int_t py);
   
   Bool_t          IsValid() const {return fShape->IsValid();} 
   Bool_t          IsVisible() const {return TGeoAtt::IsVisible();}
   TGeoNode       *FindNode(const char *name) const;
   void            FindOverlaps() const;
   TObjArray      *GetNodes() {return fNodes;}
   Int_t           GetNdaughters() const;
   virtual Int_t   GetByteCount() const;
   Double_t        GetUsageCount(Int_t i) const;
   TGeoMaterial   *GetMaterial() const               {return fMaterial;}
   Int_t           GetMedia() const                  {return fMaterial->GetMedia();}
   TObject        *GetField() const                  {return fField;}
   TGeoPatternFinder *GetFinder() const              {return fFinder;}
   TGeoVoxelFinder   *GetVoxels() const              {return fVoxels;}
   Int_t           GetIndex(TGeoNode *node) const;
   TGeoNode       *GetNode(const char *name) const;
   TGeoNode       *GetNode(Int_t i) const {return (TGeoNode*)fNodes->At(i);}
   Int_t           GetNodeIndex(TGeoNode *node, Int_t *check_list, Int_t ncheck) const;
   virtual char   *GetObjectInfo(Int_t px, Int_t py) const;
   Option_t       *GetOption() const { return fOption.Data(); }
   TGeoShape      *GetShape() const                  {return fShape;}
   void            Gsord(Int_t iaxis)                {;}
   Bool_t          IsStyleDefault() const;
   void            InspectMaterial() const; // *MENU*
   void            InspectShape() const {fShape->InspectShape();} // *MENU*
   TGeoVolume     *MakeCopyVolume();
   void            MakeCopyNodes(TGeoVolume *other);
   void            RandomPoints(Int_t npoints=10000, Option_t *option=""); // *MENU*
   void            RandomRays(Int_t nrays=10000); // *MENU*
   void            RenameCopy(Int_t copy_no);
   void            SetAsTopVolume(); // *MENU*
   void            SetCurrentPoint(Double_t x, Double_t y, Double_t z);// *MENU*
   void            SetMaterial(TGeoMaterial *material);
   void            SetNodes(TObjArray *nodes) {fNodes = nodes; TObject::SetBit(kVolumeImportNodes);}
   void            SetShape(TGeoShape *shape);
   void            SetField(TObject *field)          {fField = field;}
   void            SetOption(const char *option);
   virtual void    SetVisibility(Bool_t vis=kTRUE) {TGeoAtt::SetVisibility(vis);} // *MENU*
   virtual void    SetLineColor(Color_t lcolor);
   virtual void    SetLineStyle(Style_t lstyle);
   virtual void    SetLineWidth(Width_t lwidth);
   void            SetInvisible() {SetVisibility(kFALSE);} // *MENU*
   void            SetVoxelFinder(TGeoVoxelFinder *finder) {fVoxels=finder;}
   void            SetFinder(TGeoPatternFinder *finder) {fFinder=finder;}
   virtual void    Sizeof3D() const;
   void            SortNodes();
   Bool_t          Valid() const;
   void            VisibleDaughters(Bool_t vis=kTRUE); // *MENU*
   void            Voxelize(Option_t *option);

  ClassDef(TGeoVolume, 1)              // geometry volume descriptor
};

/*************************************************************************
 * TGeoVolumeMulti - class storing a list of volumes that have to
 *   be handled togeather at build time 
 *
 *************************************************************************/

class TGeoVolumeMulti : public TGeoVolume
{
private:
   TObjArray      *fVolumes;      // list of volumes
   Bool_t          fAttSet;       // flag attributes set
public:
   TGeoVolumeMulti();
   TGeoVolumeMulti(const char* name, TGeoMaterial *mat=0);
   virtual ~TGeoVolumeMulti();
   
   void            AddVolume(TGeoVolume *vol) {fVolumes->Add(vol);}
   TGeoVolume     *GetVolume(Int_t id) const {return (TGeoVolume*)fVolumes->At(id);}
   virtual void    AddNode(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option="");       // most general case
   virtual void    AddNodeOverlap(TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option="");
   virtual TGeoVolume *Divide(const char *divname, Int_t ndiv, Option_t *option="") { return 0;}
   virtual TGeoVolume *Divide(const char *divname, Int_t ndiv, Double_t start, Double_t step, Option_t *option="") {return 0;}
   virtual TGeoVolume *Divide(const char *divname, Double_t start, Double_t end, Double_t step, Option_t *option="") {return 0;}
   virtual TGeoVolume *Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step);
   virtual TGeoVolume *Divide(const char *divname, Int_t iaxis, Double_t step) {return 0;}
   virtual TGeoVolume *Divide(const char *divname, TObject *userdiv, Double_t *params, Option_t *option="") {return 0;}
   TGeoShape      *GetLastShape() const {return GetVolume(fVolumes->GetEntriesFast()-1)->GetShape();} 
   virtual void    SetLineColor(Color_t lcolor);
   virtual void    SetLineStyle(Style_t lstyle);
   virtual void    SetLineWidth(Width_t lwidth);
   virtual void    SetVisibility(Bool_t vis=kTRUE);
   

 ClassDef(TGeoVolumeMulti, 1)     // class to handle multiple volumes in one step
};

inline Int_t TGeoVolume::GetNdaughters() const {
             if (!fNodes) return 0; return (fNodes->GetEntriesFast());}

#endif

