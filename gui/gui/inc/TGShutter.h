// @(#)root/gui:$Id$
// Author: Fons Rademakers   18/9/2000

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGShutter
#define ROOT_TGShutter


#include "TGFrame.h"
#include "TGCanvas.h"
#include "TGWidget.h"


class TGButton;
class TGCanvas;
class TTimer;
class TList;



class TGShutterItem : public TGVerticalFrame, public TGWidget {

friend class TGShutter;

protected:
   TGButton      *fButton;     ///< shutter item button
   TGCanvas      *fCanvas;     ///< canvas of shutter item
   TGFrame       *fContainer;  ///< container in canvas containing shutter items
   TGLayoutHints *fL1, *fL2;   ///< positioning hints

private:
   TGShutterItem(const TGShutterItem&) = delete;
   TGShutterItem& operator=(const TGShutterItem&) = delete;

public:
   TGShutterItem(const TGWindow *p = nullptr, TGHotString *s = nullptr, Int_t id = -1,
                 UInt_t options = 0);
   virtual ~TGShutterItem();

   TGButton *GetButton() const { return fButton; }
   TGFrame  *GetContainer() const { return fCanvas->GetContainer(); }
   virtual void Selected()  { Emit(" Selected()"); } //*SIGNAL*

   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGShutterItem,0)  // Shutter widget item
};



class TGShutter : public TGCompositeFrame {

protected:
   TTimer         *fTimer;                  ///< Timer for animation
   TGShutterItem  *fSelectedItem;           ///< Item currently open
   TGShutterItem  *fClosingItem;            ///< Item closing down
   TList          *fTrash;                  ///< Items that need to be cleaned up
   Int_t           fHeightIncrement;        ///< Height delta
   Int_t           fClosingHeight;          ///< Closing items current height
   Int_t           fClosingHadScrollbar;    ///< Closing item had a scroll bar
   UInt_t          fDefWidth;               ///< Default width
   UInt_t          fDefHeight;              ///< Default height

private:
   TGShutter(const TGShutter&) = delete;
   TGShutter& operator=(const TGShutter&) = delete;

public:
   TGShutter(const TGWindow *p = nullptr, UInt_t options = kSunkenFrame);
   virtual ~TGShutter();

   virtual void   AddItem(TGShutterItem *item);
   virtual void   RemoveItem(const char *name);
   virtual TGShutterItem *AddPage(const char *item = "Page"); //*MENU*
   virtual void   RemovePage();                    //*MENU*
   virtual void   RenamePage(const char *name);    //*MENU*
   virtual Bool_t HandleTimer(TTimer *t);
   virtual void   Layout();
   virtual void   SetLayoutManager(TGLayoutManager*) { }
   TGShutterItem *GetSelectedItem() const { return fSelectedItem; }
   TGShutterItem *GetItem(const char *name);
   virtual void   SetSelectedItem(TGShutterItem *item);
   virtual void   SetSelectedItem(const char *name);
   virtual void   EnableItem(const char *name, Bool_t on = kTRUE);

   virtual TGDimension GetDefaultSize() const;
   virtual void        SetDefaultSize(UInt_t w, UInt_t h);

   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   virtual Bool_t ProcessMessage(Long_t cmd, Long_t parm1, Long_t parm2);
   virtual void   Selected(TGShutterItem *item) { Emit(" Selected(TGShutterItem*)", item); } //*SIGNAL*

   ClassDef(TGShutter,0)  // Shutter widget
};

#endif
