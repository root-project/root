// @(#)root/base:$Id$
// Author: Nenad Buncic   08/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TContextMenuImp
#define ROOT_TContextMenuImp


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TContextMenuImp                                                            //
//                                                                            //
// This class provides an interface to GUI independent                        //
// context sensitive popup menus.                                             //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

class TContextMenu;
class TObject;
class TMethod;
class TFunction;


class TContextMenuImp {

protected:
   TContextMenu *fContextMenu; //TContextMenu associated with this implementation

   TContextMenuImp(const TContextMenuImp& cmi)
     : fContextMenu(cmi.fContextMenu) { }
   TContextMenuImp& operator=(const TContextMenuImp& cmi)
     {if(this!=&cmi) fContextMenu=cmi.fContextMenu;
     return *this;}

public:
   TContextMenuImp(TContextMenu *c=0) : fContextMenu(c) { }
   virtual ~TContextMenuImp();

   virtual TContextMenu *GetContextMenu() const { return fContextMenu; }

   virtual void Dialog(TObject *object, TFunction *function);
   virtual void Dialog(TObject *object, TMethod *method);
   virtual void DisplayPopup(Int_t x, Int_t y);

   ClassDef(TContextMenuImp,0) //Context sensitive popup menu implementation
};

inline void TContextMenuImp::Dialog(TObject *, TFunction *) { }
inline void TContextMenuImp::Dialog(TObject *, TMethod *) { }
inline void TContextMenuImp::DisplayPopup(Int_t, Int_t) { }

#endif
