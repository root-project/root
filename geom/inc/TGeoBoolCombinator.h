// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   15/01/02
   
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoBoolCombinator
#define ROOT_TGeoBoolCombinator


// forward declarations
class TGeoCompositeShape;
class TGeoNode;

/*************************************************************************
 * TGeoBoolCombinator - class description 
 *
 *************************************************************************/

class TGeoBoolCombinator : public TNamed
{
private :
enum EGeoCombinationType {
   kUnionOnly     = BIT(15),
   kTwoComponents = BIT(16)
};
// data members
   TGeoCompositeShape   *fShape;
// methods

public:
   // constructors
   TGeoBoolCombinator();
   TGeoBoolCombinator(const char *name, const char *formula);
   // destructor
   virtual ~TGeoBoolCombinator();
   // methods
   void                  SetShape(TGeoCompositeShape *shape) {fShape = shape;}
   Bool_t                Compile();
   Bool_t                Contains(Double_t *point) const {return kFALSE;}
   void                  ComputeBBox() {}
   Double_t              DistToSurf(Double_t *point, Double_t *dir) {return 0.0;}

  ClassDef(TGeoBoolCombinator, 1)
};

#endif

