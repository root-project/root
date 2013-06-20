// @(#):$Id$
// Author: Andrei Gheata   01/03/11

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoBranchArray
#define ROOT_TGeoBranchArray

#ifndef ROOT_TObject
#include "TObject.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoBranchArray - An array of daughter indices making a geometry path. //
//   Can be used to backup/restore a state                                //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoHMatrix;
class TGeoMatrix;
class TGeoNavigator;
class TGeoNode;

class TGeoBranchArray : public TObject
{
protected:
   Int_t             fLevel;          // Branch depth
   Int_t             fMaxLevel;       // Array length
   TGeoNode        **fArray;          //![fMaxLevel] Array of nodes
   TGeoHMatrix      *fMatrix;         // Global matrix (owned)
   TObject          *fClient;         // Client object to notify

public:
   TGeoBranchArray() : TObject(), fLevel(0), fMaxLevel(10), fArray(NULL), fMatrix(NULL), fClient(NULL) {}
   TGeoBranchArray(Int_t level);
   virtual ~TGeoBranchArray();

   TGeoBranchArray(const TGeoBranchArray&);
   TGeoBranchArray& operator=(const TGeoBranchArray&);
   Bool_t operator ==(const TGeoBranchArray& other) const;
   Bool_t operator !=(const TGeoBranchArray& other) const;
   Bool_t operator >(const TGeoBranchArray& other) const;
   Bool_t operator <(const TGeoBranchArray& other) const;
   Bool_t operator >=(const TGeoBranchArray& other) const;
   Bool_t operator <=(const TGeoBranchArray& other) const;
   
   void              AddLevel(Int_t dindex);
   virtual Int_t     Compare(const TObject *obj) const;
   void              CleanMatrix();
   TGeoNode        **GetArray() const   {return fArray;}
   TObject          *GetClient() const  {return fClient;}
   Int_t             GetLevel() const   {return fLevel;}
   TGeoHMatrix      *GetMatrix() const  {return fMatrix;}
   TGeoNode         *GetNode(Int_t level) const {return fArray[level];}
   TGeoNode         *GetCurrentNode() const {return fArray[fLevel];}
   void              GetPath(TString &path) const;
   void              Init(TGeoNode **branch, TGeoMatrix *global, Int_t level);
   void              InitFromNavigator(TGeoNavigator *nav);
   virtual Bool_t    IsSortable() const {return kTRUE;}
   virtual Bool_t    Notify() {return (fClient)?fClient->Notify():kFALSE;}
   virtual void      Print(Option_t *option="") const;
   void              SetClient(TObject *client) {fClient = client;}
   static void       Sort(Int_t n, TGeoBranchArray **array, Int_t *index, Bool_t down=kTRUE);
   static Long64_t   BinarySearch(Long64_t n, const TGeoBranchArray **array, TGeoBranchArray *value);
   void              UpdateNavigator(TGeoNavigator *nav) const;
   
   ClassDef(TGeoBranchArray, 3)
};

struct compareBAasc {
   compareBAasc(TGeoBranchArray **d) : fData(d) {}
   bool operator ()(Int_t i1, Int_t i2) {return **(fData+i1) < **(fData+i2);}
   TGeoBranchArray **fData;
};

struct compareBAdesc {
   compareBAdesc(TGeoBranchArray **d) : fData(d) {}
   bool operator ()(Int_t i1, Int_t i2) {return **(fData+i1) > **(fData+i2);}
   TGeoBranchArray **fData;
};

#endif
