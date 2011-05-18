// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/01/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVolumeViewIter
#define ROOT_TVolumeViewIter

#include "TDataSetIter.h"
#include "TVolumePosition.h"

class TObjArray;
class TVolumeView;

class TVolumeViewIter : public TDataSetIter {
private:
protected:
   friend class TVolumeView;
   TObjArray    *fPositions; // the array of the Node position in the absolute system
   virtual const TVolumePosition *GetPosition(Int_t level=0) const;
   virtual TVolumePosition *SetPositionAt(TVolume *node,Double_t x=0, Double_t y=0, Double_t z=0, TRotMatrix *matrix=0);
   virtual TVolumePosition *SetPositionAt(TVolumePosition &curPosition);
public:
   TVolumeViewIter(TVolumeView *view, Int_t depth=1, Bool_t dir=kIterForward);
   ~TVolumeViewIter();
   virtual Bool_t          Notify();
   virtual void            Notify(TDataSet *set);
   virtual void            Reset(TDataSet *l=0,Int_t depth=0);

   virtual TDataSet       *operator[](const Char_t *path);
   TVolumePosition        *operator[](Int_t level);

   TVolumePosition        *UpdateTempMatrix(TVolumePosition *curPosition);
   void                    ResetPosition(Int_t level=0, TVolumePosition *newPosition=0);
   ClassDef(TVolumeViewIter,0)  // Volume view iterator
};

inline Bool_t  TVolumeViewIter::Notify() { return TDataSetIter::Notify();}
inline TDataSet  *TVolumeViewIter::operator[](const Char_t *path)
{return TDataSetIter::operator[](path); }

#endif

