// @(#)root/geom:$Name:  $:$Id: TGeoVolume.h,v 1.29 2003/08/08 09:22:18 brun Exp $
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

#ifndef ROOT_TGeoMedium
#include "TGeoMedium.h"
#endif

#ifndef ROOT_TGeoShape
#include "TGeoShape.h"
#endif

// forward declarations
class TH2F;
class TGeoNode;
class TGeoMaterial;
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
   TObjArray         *fNodes;          // array of nodes inside this volume
   TGeoShape         *fShape;          // shape
   TGeoMedium        *fMedium;         // tracking medium
   TGeoPatternFinder *fFinder;         // finder object for divisions
   TGeoVoxelFinder   *fVoxels;         // finder object for bounding boxes

   TObject           *fField;          //! just a hook for now
   TString            fOption;         //! option - if any
   Int_t              fNumber;         //  volume serial number in the list of volumes
   Int_t              fNtotal;         // total number of physical nodes
public:
   enum EGeoVolumeTypes {
      kVolumeDiv     =     BIT(16),
      kVolumeOverlap =     BIT(17),
      kVolumeImportNodes = BIT(18),
      kVolumeMulti   =     BIT(19),
      kVoxelsXYZ     =     BIT(20),
      kVoxelsCyl     =     BIT(21)
   };
   // constructors
   TGeoVolume();
   TGeoVolume(const char *name, const TGeoShape *shape, const TGeoMedium *med=0);

   // destructor
   virtual ~TGeoVolume();
   // methods
   virtual void    cd(Int_t inode) const;
   void            Browse(TBrowser *b);
   void            CheckShapes();
   void            ClearNodes() {fNodes = 0;}
   void            ClearShape();
   void            CleanAll();
   void            CheckGeometry(Int_t nrays=1, Double_t startx=0, Double_t starty=0, Double_t startz=0) const;
   void            CheckOverlaps(Double_t ovlp=0.1, Option_t *option="") const; // *MENU*
   Int_t           CountNodes(Int_t nlevels=1000);
   Bool_t          Contains(Double_t *point) const {return fShape->Contains(point);}
   virtual Bool_t  IsAssembly() const {return kFALSE;}
   Bool_t          IsFolder() const;
   Bool_t          IsRunTime() const {return fShape->IsRunTimeShape();}
   virtual Bool_t  IsVolumeMulti() const {return kFALSE;}
   virtual void    AddNode(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat=0, Option_t *option="");       // most general case
   void            AddNodeOffset(const TGeoVolume *vol, Int_t copy_no, Double_t offset=0, Option_t *option="");
   virtual void    AddNodeOverlap(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat=0, Option_t *option="");

   virtual TGeoVolume *Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step, Int_t numed=0, Option_t *option="");
   virtual Int_t   DistancetoPrimitive(Int_t px, Int_t py);
   virtual void    Draw(Option_t *option=""); // *MENU*
   virtual void    DrawOnly(Option_t *option=""); // *MENU*
   TH2F           *LegoPlot(Int_t ntheta=20, Double_t themin=0., Double_t themax=180.,
                            Int_t nphi=60, Double_t phimin=0., Double_t phimax=360.,
                            Double_t rmin=0., Double_t rmax=9999999, Option_t *option=""); // *MENU*
   virtual void    Paint(Option_t *option="");
   void            PrintNodes() const;
   void            PrintVoxels() const; // *MENU*
   virtual void    ExecuteEvent(Int_t event, Int_t px, Int_t py);

   Bool_t          IsCylVoxels() const {return TObject::TestBit(kVoxelsCyl);}
   Bool_t          IsXYZVoxels() const {return TObject::TestBit(kVoxelsXYZ);}
   Bool_t          IsValid() const {return fShape->IsValid();}
   Bool_t          IsVisible() const {return TGeoAtt::IsVisible();}
   TGeoNode       *FindNode(const char *name) const;
   void            FindOverlaps() const;
   Bool_t          FindMatrixOfDaughterVolume(TGeoVolume *vol) const;
   TObjArray      *GetNodes() {return fNodes;}
   Int_t           GetNdaughters() const;
   Int_t           GetNtotal() const {return fNtotal;}
   virtual Int_t   GetByteCount() const;
   TGeoMaterial   *GetMaterial() const               {return fMedium->GetMaterial();}
   TGeoMedium     *GetMedium() const                 {return fMedium;}
   TObject        *GetField() const                  {return fField;}
   TGeoPatternFinder *GetFinder() const              {return fFinder;}
   TGeoVoxelFinder   *GetVoxels() const              {return fVoxels;}
   Int_t           GetIndex(const TGeoNode *node) const;
   TGeoNode       *GetNode(const char *name) const;
   TGeoNode       *GetNode(Int_t i) const {return (TGeoNode*)fNodes->At(i);}
   Int_t           GetNodeIndex(const TGeoNode *node, Int_t *check_list, Int_t ncheck) const;
   Int_t           GetNumber() {return fNumber;}
   virtual char   *GetObjectInfo(Int_t px, Int_t py) const;
   Bool_t          GetOptimalVoxels() const;
   Option_t       *GetOption() const { return fOption.Data(); }
   TGeoShape      *GetShape() const                  {return fShape;}
   void            GrabFocus(); // *MENU*
   void            Gsord(Int_t /*iaxis*/)                {;}
   Bool_t          IsStyleDefault() const;
   void            InspectMaterial() const; // *MENU*
   void            InspectShape() const {fShape->InspectShape();} // *MENU*
   void           *Make3DBuffer() const {return fShape->Make3DBuffer(this);}
   virtual TGeoVolume *MakeCopyVolume(TGeoShape *newshape);
   void            MakeCopyNodes(const TGeoVolume *other);
   Bool_t          OptimizeVoxels(); // *MENU*
   void            RandomPoints(Int_t npoints=1000000, Option_t *option=""); // *MENU*
   void            RandomRays(Int_t nrays=10000, Double_t startx=0, Double_t starty=0, Double_t startz=0); // *MENU*
   void            Raytrace(Option_t *option=""); // *MENU*
   void            SetAsTopVolume(); // *MENU*
   void            SetCurrentPoint(Double_t x, Double_t y, Double_t z);// *MENU*
   void            SetCylVoxels(Bool_t flag=kTRUE) {TObject::SetBit(kVoxelsCyl, flag); TObject::SetBit(kVoxelsXYZ, !flag);}
   void            SetNodes(TObjArray *nodes) {fNodes = nodes; TObject::SetBit(kVolumeImportNodes);}
   void            SetShape(const TGeoShape *shape);
   void            SetField(const TObject *field)          {fField = (TObject*)field;}
   void            SetOption(const char *option);
   virtual void    SetVisibility(Bool_t vis=kTRUE); // *MENU*
   virtual void    SetLineColor(Color_t lcolor);
   virtual void    SetLineStyle(Style_t lstyle);
   virtual void    SetLineWidth(Width_t lwidth);
   void            SetInvisible() {SetVisibility(kFALSE);} // *MENU*
   virtual void    SetMedium(const TGeoMedium *medium) {fMedium = (TGeoMedium*)medium;}
   void            SetVoxelFinder(const TGeoVoxelFinder *finder) {fVoxels=(TGeoVoxelFinder*)finder;}
   void            SetFinder(const TGeoPatternFinder *finder) {fFinder=(TGeoPatternFinder*)finder;}
   void            SetNumber(Int_t number) {fNumber = number;}
   virtual void    Sizeof3D() const;
   void            SortNodes();
   Bool_t          Valid() const;
   void            VisibleDaughters(Bool_t vis=kTRUE); // *MENU*
   void            InvisibleAll() {SetInvisible(); VisibleDaughters(kFALSE);} // *MENU*
   void            Voxelize(Option_t *option);
   Double_t        Weight(Double_t precision=0.01, Option_t *option="v"); // *MENU*

  ClassDef(TGeoVolume, 4)              // geometry volume descriptor
};

/*************************************************************************
 * TGeoVolumeMulti - class storing a list of volumes that have to
 *   be handled togeather at build time
 *
 *************************************************************************/

class TGeoVolumeMulti : public TGeoVolume
{
private:
   TObjArray       *fVolumes;      // list of volumes
   TGeoVolumeMulti *fDivision;     // division of this volume
   Int_t            fNumed;        // medium number for divisions
   Int_t            fNdiv;         // number of divisions
   Int_t            fAxis;         // axis of division
   Double_t         fStart;        // division start offset
   Double_t         fStep;         // division step
   Bool_t           fAttSet;       // flag attributes set
public:
   TGeoVolumeMulti();
   TGeoVolumeMulti(const char* name, const TGeoMedium *med=0);
   virtual ~TGeoVolumeMulti();

   void            AddVolume(TGeoVolume *vol);
   TGeoVolume     *GetVolume(Int_t id) const {return (TGeoVolume*)fVolumes->At(id);}
   virtual void    AddNode(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option="");       // most general case
   virtual void    AddNodeOverlap(const TGeoVolume *vol, Int_t copy_no, TGeoMatrix *mat, Option_t *option="");
   virtual TGeoVolume *Divide(const char *divname, Int_t iaxis, Int_t ndiv, Double_t start, Double_t step, Int_t numed=0, Option_t *option="");
   TGeoShape      *GetLastShape() const {return GetVolume(fVolumes->GetEntriesFast()-1)->GetShape();}
   Int_t           GetNvolumes() const {return fVolumes->GetEntriesFast();}
   Int_t           GetAxis() const {return fNdiv;}
   Int_t           GetNdiv() const {return fNdiv;}
   Double_t        GetStart() const {return fStart;}
   Double_t        GetStep() const {return fStep;}
   virtual Bool_t  IsVolumeMulti() const {return kTRUE;}
   virtual TGeoVolume *MakeCopyVolume(TGeoShape *newshape);
   virtual void    SetLineColor(Color_t lcolor);
   virtual void    SetLineStyle(Style_t lstyle);
   virtual void    SetLineWidth(Width_t lwidth);
   virtual void    SetMedium(const TGeoMedium *medium);
   virtual void    SetVisibility(Bool_t vis=kTRUE);


 ClassDef(TGeoVolumeMulti, 3)     // class to handle multiple volumes in one step
};

/*************************************************************************
 * TGeoVolumeAssembly - special assembly of volumes
 *  
 *
 *************************************************************************/

class TGeoVolumeAssembly : public TGeoVolume
{
public:
   TGeoVolumeAssembly();
   TGeoVolumeAssembly(const char *name);
   virtual ~TGeoVolumeAssembly() {;}
   virtual Bool_t  IsAssembly() const {return kTRUE;}

ClassDef(TGeoVolumeAssembly, 1)   // an assembly of volumes
};

inline Int_t TGeoVolume::GetNdaughters() const {
             if (!fNodes) return 0; return (fNodes->GetEntriesFast());}

#endif

