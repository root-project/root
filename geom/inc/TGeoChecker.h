/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata           - date Thu 01 Nov 2001 11:43:42 AM CET

/*************************************************************************
 * TGeoChecker - A simple checker generating random points inside a 
 *   geometry. Generates a tree of points on the surfaces coresponding
 *   to the safety of each generated point
 *
 *************************************************************************/

#ifndef ROOT_TGeoChecker
#define ROOT_TGeoChecker


// forward declarations
class TGeoNode;
class TGeoVolume;
class TTree;

/*************************************************************************
 * TGeoChecker - class description 
 *
 *************************************************************************/

class TGeoChecker : public TObject
{
private :
// data members
   TGeoNode        *fCurrentNode;
   TTree           *fTreePts;
// methods

public:
   // constructors
   TGeoChecker();
   TGeoChecker(const char *treename, const char *filename);
   // destructor
   virtual ~TGeoChecker();
   // methods
   void             SetCurrentNode(TGeoNode *node)   {fCurrentNode = node;}
   void             CreateTree(const char *treename, const char *filename);
   void             Generate(UInt_t npoints=1000000);      // compute safety and fill the tree
   void             Raytrace(Double_t *startpoint, UInt_t npoints=1000000);
   void             ShowPoints(Option_t *option="");

  ClassDef(TGeoChecker, 1)               // a simple geometry checker

//***** Need to add classes and globals to LinkDef.h *****
};

#endif

