// @(#)root/gui:$Id$
// Author: Fons Rademakers   13/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGComboBox
#define ROOT_TGComboBox


#include "TGListBox.h"

class TGScrollBarElement;
class TGTextEntry;

class TGComboBoxPopup : public TGCompositeFrame {

protected:
   TGListBox *fListBox;
   TGLBEntry *fSelected;

private:
   TGComboBoxPopup(const TGComboBoxPopup&) = delete;
   TGComboBoxPopup& operator=(const TGComboBoxPopup&) = delete;

public:
   TGComboBoxPopup(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
                   UInt_t options = kVerticalFrame,
                   Pixel_t back = GetWhitePixel());

   virtual Bool_t HandleButton(Event_t *);
   void KeyPressed(TGFrame*, UInt_t, UInt_t);

   void SetListBox(TGListBox *lb) { fListBox = lb; }
   void PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void EndPopup();

   ClassDef(TGComboBoxPopup,0)  // Combobox popup window
};


class TGComboBox : public TGCompositeFrame, public TGWidget {

private:
   TGComboBox(const TGComboBox&) = delete;
   TGComboBox& operator=(const TGComboBox&) = delete;

protected:
   TGLBEntry           *fSelEntry;      ///< selected item frame
   TGTextEntry         *fTextEntry;     ///< text entry
   TGScrollBarElement  *fDDButton;      ///< button controlling drop down of popup
   TGComboBoxPopup     *fComboFrame;    ///< popup containing a listbox
   TGListBox           *fListBox;       ///< the listbox with text items
   const TGPicture     *fBpic;          ///< down arrow picture used in fDDButton
   TGLayoutHints       *fLhs;           ///< layout hints for selected item frame
   TGLayoutHints       *fLhb;           ///< layout hints for fDDButton
   TGLayoutHints       *fLhdd;          ///< layout hints for fListBox

   virtual void Init();

public:
   TGComboBox(const TGWindow *p = nullptr, Int_t id = -1,
              UInt_t options = kHorizontalFrame | kSunkenFrame | kDoubleBorder,
              Pixel_t back = GetWhitePixel());
   TGComboBox(const TGWindow *p, const char *text, Int_t id = -1,
              UInt_t options = kHorizontalFrame | kSunkenFrame | kDoubleBorder,
              Pixel_t back = GetWhitePixel());

   virtual ~TGComboBox();

   virtual void DrawBorder();
   virtual TGDimension GetDefaultSize() const { return TGDimension(fWidth, fHeight); }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleSelection(Event_t *event);
   virtual Bool_t HandleSelectionRequest(Event_t *event);
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);

   virtual void AddEntry(TGString *s, Int_t id)
                        { fListBox->AddEntry(s, id); Resize(); }
   virtual void AddEntry(const char *s, Int_t id)
                        { fListBox->AddEntry(s, id); Resize(); }
   virtual void AddEntry(TGLBEntry *lbe, TGLayoutHints *lhints)
                        { fListBox->AddEntry(lbe, lhints); Resize(); }
   virtual void InsertEntry(TGString *s, Int_t id, Int_t afterID)
                        { fListBox->InsertEntry(s, id, afterID); Resize(); }
   virtual void InsertEntry(const char *s, Int_t id, Int_t afterID)
                        { fListBox->InsertEntry(s, id, afterID); Resize(); }
   virtual void InsertEntry(TGLBEntry *lbe, TGLayoutHints *lhints, Int_t afterID)
                        { fListBox->InsertEntry(lbe, lhints, afterID); Resize(); }
   virtual void NewEntry(const char *s = "Entry")
                        { fListBox->NewEntry(s); Resize(); }       //*MENU*
   virtual void RemoveEntry(Int_t id = -1);                        //*MENU*
   virtual void RemoveAll();                                       //*MENU*
   virtual void Layout();
   virtual Bool_t IsTextInputEnabled() const { return (fTextEntry != 0); }
   virtual void EnableTextInput(Bool_t on);    //*TOGGLE* *GETTER=IsTextInputEnabled
   virtual void RemoveEntries(Int_t from_ID, Int_t to_ID)
                        { fListBox->RemoveEntries(from_ID, to_ID); }
   virtual Int_t GetNumberOfEntries() const
                        { return fListBox->GetNumberOfEntries(); }

   virtual TGListBox    *GetListBox() const { return fListBox; }
   virtual TGTextEntry  *GetTextEntry() const { return fTextEntry; }
   virtual TGLBEntry    *FindEntry(const char *s) const;
   virtual void  Select(Int_t id, Bool_t emit = kTRUE);
   virtual Int_t GetSelected() const { return fListBox->GetSelected(); }
   virtual TGLBEntry *GetSelectedEntry() const
                        { return fListBox->GetSelectedEntry(); }
   virtual void SetTopEntry(TGLBEntry *e, TGLayoutHints *lh);
   virtual void SetEnabled(Bool_t on = kTRUE);   //*TOGGLE* *GETTER=IsEnabled
   virtual Bool_t IsEnabled() const { return  fDDButton->IsEnabled(); }
   virtual void SortByName(Bool_t ascend = kTRUE)
                  { fListBox->SortByName(ascend); }            //*MENU*icon=bld_sortup.png*

   virtual void Selected(Int_t widgetId, Int_t id);                  // *SIGNAL*
   virtual void Selected(Int_t id) { Emit("Selected(Int_t)", id); }  // *SIGNAL*
   virtual void Selected(const char *txt) { Emit("Selected(char*)", txt); } // *SIGNAL*
   virtual void Changed() { Emit("Changed()"); } // *SIGNAL*
   virtual void ReturnPressed();                                     // *SIGNAL*
   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGComboBox,0)  // Combo box widget
};


/** \class TGLineStyleComboBox
The TGLineStyleComboBox user callable and it creates
a combobox for selecting the line style.
*/

class TGLineStyleComboBox : public TGComboBox {

public:
   TGLineStyleComboBox(const TGWindow *p = nullptr, Int_t id = -1,
              UInt_t options = kHorizontalFrame | kSunkenFrame | kDoubleBorder,
              Pixel_t back = GetWhitePixel());

   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGLineStyleComboBox, 0)  // Line style combobox widget
};


/** \class TGLineWidthComboBox
The TGLineWidthComboBox user callable and it creates
a combobox for selecting the line width.
*/

class TGLineWidthComboBox : public TGComboBox {

public:
   TGLineWidthComboBox(const TGWindow *p = 0, Int_t id = -1,
              UInt_t options = kHorizontalFrame | kSunkenFrame | kDoubleBorder,
              Pixel_t back = GetWhitePixel(), Bool_t none=kFALSE);

   virtual void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGLineWidthComboBox, 0)  // Line width combobox widget
};


/** \class TGFontTypeComboBox
The TGFontTypeComboBox is user callable and it creates
a combobox for selecting the font.
*/

const Int_t kMaxFonts = 20;

class TGFontTypeComboBox : public TGComboBox {

protected:
   FontStruct_t fFonts[kMaxFonts];

public:
   TGFontTypeComboBox(const TGWindow *p = 0, Int_t id = -1,
            UInt_t options = kHorizontalFrame | kSunkenFrame | kDoubleBorder,
            Pixel_t bask = GetWhitePixel());
   virtual ~TGFontTypeComboBox();

   ClassDef(TGFontTypeComboBox, 0)  // Font type combobox widget
};

#endif
