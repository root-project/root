// @(#)root/geom:$Name:  $:$Id: TGeoOverlap.h,v 1.1 2003/02/10 17:23:14 brun Exp $
// Author: Andrei Gheata   09/02/03

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoOverlap
#define ROOT_TGeoOverlap

#ifndef ROOT_TNamed
#include "TNamed.h"
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

class TGeoVolume;
class TPolyMarker3D;
class TBrowser;

/*************************************************************************
 * TGeoOverlap - base class describing geometry overlaps. Overlaps apply
 *   to the nodes contained inside a volume. These should not overlap to
 *   each other nor extrude the shape of their mother volume.
 *
 *************************************************************************/

class TGeoOverlap : public TNamed,
                    public TAttLine,
                    public TAttFill,
                    public TAtt3D
{
public:
   enum EOverlapTypes {
      kGeoExtrusion   = 8,
      kGeoNodeOverlap = 9
   };      
protected:
   Double_t         fOverlap;    // overlap distance
   TGeoVolume      *fVolume;     // volume containing the overlap
   TPolyMarker3D   *fMarker;     // points in the overlapping region

public:
   TGeoOverlap();
   TGeoOverlap(const char *name, TGeoVolume *vol, Double_t ovlp);
   virtual           ~TGeoOverlap();
   
   void              Browse(TBrowser *b);
   virtual Int_t     Compare(const TObject *obj) const;
   virtual Int_t     DistancetoPrimitive(Int_t px, Int_t py);
   virtual void      Draw(Option_t *option="")                   = 0;
   virtual void      ExecuteEvent(Int_t event, Int_t px, Int_t py);
   TPolyMarker3D    *GetPolyMarker() const {return fMarker;}
   virtual TGeoNode *GetNode(Int_t iovlp) const                  = 0;
   Double_t          GetOverlap() const {return fOverlap;}
   TGeoVolume       *GetVolume() const  {return fVolume;}
   Bool_t            IsFolder() const {return kFALSE;}
   virtual Bool_t    IsSortable() const {return kTRUE;}
   virtual void      Paint(Option_t *option="");
   virtual void      PrintInfo() const                           = 0;
   virtual void      Sizeof3D() const                            = 0;
   void              SetNextPoint(Double_t x, Double_t y, Double_t z);
   void              SetVolume(TGeoVolume *vol) {fVolume=vol;}
   void              SetOverlap(Double_t ovlp)  {fOverlap=ovlp;}
   
 ClassDef(TGeoOverlap, 1)         // base class for geometical overlaps
};
 
/*************************************************************************
 *   TGeoExtrusion - class representing the extrusion of a positioned volume
 *      with respect to its mother.
 ************************************************************************/
 
class TGeoExtrusion : public TGeoOverlap
{
private:
   TGeoNode        *fNode;        // extruding daughter

public:
   TGeoExtrusion();
   TGeoExtrusion(const char *name, TGeoVolume *vol, Int_t inode, Double_t ovlp);
   virtual           ~TGeoExtrusion() {;}
   
   virtual TGeoNode *GetNode(Int_t iovlp) const;
   virtual void      Draw(Option_t *option=""); // *MENU*
   virtual void      PrintInfo() const;         // *MENU*
   virtual void      Sizeof3D() const;
   
 ClassDef(TGeoExtrusion, 1)      // class representing an extruding node 
};
 
/*************************************************************************
 *   TGeoNodeOverlap - class representing the overlap of 2 positioned 
 *      nodes inside a mother volume.
 ************************************************************************/
 
class TGeoNodeOverlap : public TGeoOverlap
{
private:
   TGeoNode        *fNode1;       // first node
   TGeoNode        *fNode2;       // second node

public:
   TGeoNodeOverlap();
   TGeoNodeOverlap(const char *name, TGeoVolume *vol, Int_t inode1, Int_t inode2, Double_t ovlp);
   virtual           ~TGeoNodeOverlap() {;}
   
   virtual TGeoNode *GetNode(Int_t iovlp) const;
   virtual void      Draw(Option_t *option=""); // *MENU*
   virtual void      PrintInfo() const;         // *MENU*
   virtual void      Sizeof3D() const;
   
 ClassDef(TGeoNodeOverlap, 1)     // class representing 2 overlapping nodes
};
     
#endif
 
