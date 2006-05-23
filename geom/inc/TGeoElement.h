// @(#)root/geom:$Name:  $:$Id: TGeoElement.h,v 1.4 2005/11/18 16:07:58 brun Exp $
// Author: Andrei Gheata   17/06/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoElement
#define ROOT_TGeoElement

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TGeoElementTable;

/*************************************************************************
 * TGeoElement - a chemical element
 *
 *************************************************************************/

class TGeoElement : public TNamed
{
   enum EGeoElement {
      kElemUsed    =   BIT(17),
      kElemDefined =   BIT(18)
   };

protected:
   Int_t                    fZ;          // Z of material
   Double_t                 fA;          // A of material

// methods

public:
   // constructors
   TGeoElement();
   TGeoElement(const char *name, const char *title, Int_t z, Double_t a);
   // destructor
   virtual ~TGeoElement()             {;}
   // methods
   Int_t                    Z() const {return fZ;}
   Double_t                 A() const {return fA;}
   Bool_t                   IsDefined() const {return TObject::TestBit(kElemDefined);}   
   Bool_t                   IsUsed() const {return TObject::TestBit(kElemUsed);}
   void                     SetDefined(Bool_t flag=kTRUE) {TObject::SetBit(kElemDefined,flag);}                    
   void                     SetUsed(Bool_t flag=kTRUE) {TObject::SetBit(kElemUsed,flag);}                    
   TGeoElementTable        *GetElementTable() const;
   

   ClassDef(TGeoElement, 1)              // base element class
};

/*************************************************************************
 * TGeoElementTable - table of elements 
 *
 *************************************************************************/

class TGeoElementTable : public TObject
{
private:
// data members
   Int_t                    fNelements;  // number of elements
   TObjArray               *fList;       // list of elements

   void                     BuildDefaultElements();

protected:
   TGeoElementTable(const TGeoElementTable&); 
   TGeoElementTable& operator=(const TGeoElementTable&); 

public:
   // constructors
   TGeoElementTable();
   TGeoElementTable(Int_t nelements);
   // destructor
   virtual ~TGeoElementTable();
   // methods
   
   void                     AddElement(const char *name, const char *title, Int_t z, Double_t a);
   TGeoElement             *FindElement(const char *name);
   TGeoElement             *GetElement(Int_t z) {return (TGeoElement*)fList->At(z);}
   Int_t                    GetNelements() const {return fNelements;}

   ClassDef(TGeoElementTable, 2)              // table of elements
};

#endif

