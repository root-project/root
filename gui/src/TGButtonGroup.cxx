// @(#)root/gui:$Name:  $:$Id: TGButtonGroup.cxx,v 1.26 2006/07/03 16:10:45 brun Exp $
// Author: Valeriy Onuchin & Fons Rademakers   16/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// The TGButtonGroup widget organizes TGButton widgets in a group.      //
//                                                                      //
// A button group widget makes it easier to deal with groups of buttons.//
// A button in a button group is associated with a unique identifier.   //
// The button group emits a Clicked() signal with this identifier when  //
// the button is clicked. Thus, a button group is an ideal solution     //
// when you have several similar buttons and want to connect all their  //
// Clicked() signals, for example, to one slot.                         //
//                                                                      //
// An exclusive button group switches off all toggle buttons except     //
// the one that was clicked. A button group is by default non-exclusive.//
// All radio buttons that are inserted, will be mutually exclusive even //
// if the button group is non-exclusive.                                //
//                                                                      //
//                                                                      //
// There are two ways of using a button group:                          //
//                                                                      //
//    The button group is a parent widget of a number of buttons,       //
//    i.e. the button group is the parent argument in the button        //
//    constructor. The buttons are assigned identifiers 1, 2, 3 etc.    //
///   in the order they are created or you can specify button id in     //
//    the button constructor. A TGButtonGroup can display a frame and   //
//    a title because it inherits from TGGroupFrame.                    //
//                                                                      //
// Example:                                                             //
//                                                                      //
//    // vertical frame without border and title                        //
//    TGVButtonGroup *bg = new TGVButtonGroup(main_frame);              //
//                                                                      //
//    // create text button with id=1                                   //
//    TGTextButton *button1 = new TGTextButton(bg,"some text");         //
//                                                                      //
//    // create another text button with id=2                           //
//    TGTextButton *button2 = new TGTextButton(bg,"another text");      //
//                                                                      //
//    // map all buttons                                                //
//    bg->Show();                                                       //
//                                                                      //
// NOTE: there is no need to call AddFrame() since the buttons are      //
// automatically added with a default layout hint to their parent,      //
// i.e. the buttongroup. To override the default layout hints use the   //
// SetLayoutHints() method.                                             //
//                                                                      //
//  ButtonGroup Signals:                                                //
//                                                                      //
//    Pressed(Int_t id)  -->  is emitted when a button in the group is  //
//                            pressed down. The id argument is the      //
//                            button's identifier.                      //
//    Released(Int_t id) -->  is emitted when a button in the group is  //
//                            released. The id argument is the button's //
//                            identifier.                               //
//    Clicked(Int_t id)  -->  is emitted when a button in the group is  //
//                            clicked. The id argument is the button's  //
//                            identifier.                               //
//                                                                      //
//                                                                      //
// The TGHButtonGroup widget organizes TGButton widgets in a group      //
// with one horizontal row. TGHButtonGroup is a convenience class that  //
// offers a thin layer on top of TGButtonGroup. It inherits from        //
// TGButtonGroup.                                                       //
//                                                                      //
// The TGVButtonGroup widget organizes TGButton widgets in a group      //
// with one vertical column. TGVButtonGroup is a convenience class that //
// offers a thin layer on top of TGButtonGroup. It inherits from        //
// TGButtonGroup.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGButtonGroup.h"
#include "TGButton.h"
#include "TClass.h"
#include "TGLayout.h"
#include "TList.h"
#include "TGResourcePool.h"
#include "Riostream.h"


ClassImp(TGButtonGroup)
ClassImp(TGHButtonGroup)
ClassImp(TGVButtonGroup)

//______________________________________________________________________________
TGButtonGroup::TGButtonGroup(const TGWindow *parent,
                             const TString &title,
                             UInt_t options,
                             GContext_t norm,
                             FontStruct_t font,
                             ULong_t back) :
   TGGroupFrame(parent, new TGString(title), options, norm, font, back)
{
   // Constructor. Layout 1 row or 1 column.

   Init();
   if (options & kVerticalFrame) {
      SetLayoutManager(new TGVerticalLayout(this));
   } else {
      SetLayoutManager(new TGHorizontalLayout(this));
   }

   fDrawBorder = !title.IsNull();
}

//______________________________________________________________________________
TGButtonGroup::TGButtonGroup(const TGWindow *parent,
                             UInt_t r, UInt_t c,
                             Int_t s, Int_t h,
                             const TString &title,
                             GContext_t norm ,
                             FontStruct_t font ,
                             ULong_t back) :
   TGGroupFrame(parent, new TGString(title), 0, norm, font, back)
{
   // Constructor. Layout defined by TGMatrixLayout:
   //    r = number of rows
   //    c = number of columns
   //    s = interval between frames
   //    h = layout hints

   Init();
   fDrawBorder = !title.IsNull();
   SetLayoutManager(new TGMatrixLayout(this,r,c,s,h));
}

//______________________________________________________________________________
void TGButtonGroup::Init()
{
   // Default init.

   fMapOfButtons = new TMap();  // map of button/id pairs
   fExclGroup    = kFALSE;
   fRadioExcl    = kFALSE;
   fDrawBorder   = kTRUE;
   SetWindowName();
}

//______________________________________________________________________________
TGButtonGroup::~TGButtonGroup()
{
   // Destructor, we do not delete the buttons.

   TIter next(fMapOfButtons);
   register TGButton *item = 0;

   while ((item = (TGButton*)next())) {
      item->SetGroup(0);
   }

   SafeDelete(fMapOfButtons);
}

//______________________________________________________________________________
void TGButtonGroup::SetBorderDrawn(Bool_t enable)
{
   // Makes border to be visible/invisible.

   if (enable != IsBorderDrawn()) {
      fDrawBorder = enable;
      ChangedBy("SetBorderDrawn");        // emit signal
   }
}

//______________________________________________________________________________
void TGButtonGroup::SetExclusive(Bool_t enable)
{
   // Sets the button group to be exclusive if enable is kTRUE,
   // or to be non-exclusive if enable is kFALSE.
   // An exclusive button group switches off all other toggle buttons when
   // one is switched on. This is ideal for groups of radio-buttons
   // A non-exclusive group allow many buttons to be switched on at the same
   // time. The default setting is kFALSE.

   if (enable != IsExclusive()) {
      fExclGroup = enable;
      ChangedBy("SetExclusive");  // emit signal
   }
}

//______________________________________________________________________________
void TGButtonGroup::SetRadioButtonExclusive(Bool_t enable)
{
   // If enable is kTRUE, this button group will treat radio buttons as
   // mutually exclusive, and other buttons according to IsExclusive().
   // This function is called automatically whenever a TGRadioButton
   // is inserted, so you should normally never have to call it.

   if (enable != IsRadioButtonExclusive()) {
      fRadioExcl = enable;
      ChangedBy("SetRadioButtonExclusive"); // emit signal
   }
}

//______________________________________________________________________________
void TGButtonGroup::SetButton(Int_t id, Bool_t down)
{
   // Sets the button with id to be on/down, and if this is an
   // exclusive group, all other button in the group to be off/up.

   TGButton *b = Find(id);

   if (b && (b->IsDown() != down)) {
      b->SetDown(down, kTRUE);
   }
}

//______________________________________________________________________________
Int_t TGButtonGroup::Insert(TGButton *button, Int_t id)
{
   // Inserts a button with the identifier id into the button group.
   // Returns the button identifier.
   //
   // It is not necessary to manually insert buttons that have this button
   // group as their parent widget. An exception is when you want custom
   // identifiers instead of the default 1, 2, 3 etc.
   //
   // The button is assigned the identifier id or an automatically
   // generated identifier.  It works as follows: If id > 0, this
   // identifier is assigned.  If id == -1 (default), the identifier is
   // equal to the number of buttons in the group+1.  If id is any other
   // negative integer, for instance -2, a unique identifier (negative
   // integer <= -2) is generated.
   //
   // Inserting several buttons with id = -1 assigns the identifiers 1,
   // 2, 3, etc.

   if (button->fGroup && button->fGroup != this)
      button->fGroup->Remove(button);

   if (button->fGroup == this) {
      if (id == -1)
         return GetId(button);    // the button is already in group
      else
         button->fGroup->Remove(button);  // want to set a new id
   }

   button->fGroup = this;

   static Int_t seq_no = -2;
   Long_t bid;

   if (id < -1)       bid = seq_no--;
   else if (id == -1) bid = GetCount()+1;
   else               bid = id;

   fMapOfButtons->Add(button, (TObject*)bid);
   AddFrame(button);

   SetRadioButtonExclusive(button->IsA()->InheritsFrom(TGRadioButton::Class()));

   Connect(button, "Pressed()" , "TGButtonGroup", this, "ReleaseButtons()");
   Connect(button, "Pressed()" , "TGButtonGroup", this, "ButtonPressed()");
   Connect(button, "Released()", "TGButtonGroup", this, "ButtonReleased()");
   Connect(button, "Clicked()" , "TGButtonGroup", this, "ButtonClicked()");

   return (Int_t) bid;
}

//______________________________________________________________________________
void TGButtonGroup::Remove(TGButton *button)
{
   // Removes a button from the button group.

   TGButton *item = (TGButton*) fMapOfButtons->Remove(button);
   if (item) {
      button->SetGroup(0);
      button->Disconnect(this);
   }

   RemoveFrame(button);
}

//______________________________________________________________________________
TGButton *TGButtonGroup::Find(Int_t id) const
{
   // Finds and returns a pointer to the button with the specified
   // identifier id. Returns null if the button was not found.

   TIter next(fMapOfButtons);
   register TGButton *item = 0;

   while ((item = (TGButton*)next())) {
      if ((Long_t)fMapOfButtons->GetValue(item) == id) break;   // found
   }

   return item;
}

//______________________________________________________________________________
Int_t TGButtonGroup::GetId(TGButton *button) const
{
   // Finds and returns the id of the button.
   // Returns -1 if the button is not a member of this group.

   TPair *a = (TPair*) fMapOfButtons->FindObject(button);
   if (a)
      return (Int_t)Long_t(a->Value());
   else
      return -1;
}

//______________________________________________________________________________
void TGButtonGroup::ButtonPressed()
{
   // This slot is activated when one of the buttons in the group emits the
   // Pressed() signal.

#if 0
   // Is here for historical purposes and example. Now this is not needed
   // anymore since TGButton has has its own GetSender() method returning
   // the TGButton proper.

   // This is needed since gTQSender points to TQObject part of TGButton
   TGButton *btn = dynamic_cast<TGButton*>((TQObject*)gTQSender);

   if (!btn) {
      Error("ButtonPressed", "gTQSender not a TGButton");
      return;
   }
#else
      TGButton *btn = (TGButton*)gTQSender;
#endif

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Long_t(a->Value());
      Pressed(id);
   }
}

//______________________________________________________________________________
void TGButtonGroup::ButtonReleased()
{
   // This slot is activated when one of the buttons in the group emits the
   // Released() signal.

   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Long_t(a->Value());
      Released(id);
   }
}

//______________________________________________________________________________
void TGButtonGroup::ButtonClicked()
{
   // This slot is activated when one of the buttons in the group emits the
   // Clicked() signal.

   TGButton *btn = (TGButton*)gTQSender;

   TPair *a = (TPair*) fMapOfButtons->FindObject(btn);
   if (a) {
      Int_t id = (Int_t)Long_t(a->Value());
      Clicked(id);
   }
}

//______________________________________________________________________________
void TGButtonGroup::ReleaseButtons()
{
   // This slot is activated when one of the buttons in the
   // exclusive group emits the Pressed() signal.

   if (!fExclGroup && !fRadioExcl) return;

   TGButton *btn = (TGButton*)gTQSender;

   if (!fExclGroup && !btn)
      return;

   TIter next(fMapOfButtons);
   register TGButton *item = 0;

   while ((item = (TGButton*)next())) {    // loop over all buttons
      if (btn != item && item->IsToggleButton() && item->IsOn() &&
          (fExclGroup || item->IsA()->InheritsFrom(TGRadioButton::Class()))) {
         item->SetOn(kFALSE);
      }
   }
}

//______________________________________________________________________________
void TGButtonGroup::Show()
{
   // Show group of buttons.

   MapSubwindows();
   Resize();
   MapRaised();
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGButtonGroup::Hide()
{
   // Hide group of buttons.

   UnmapWindow();
}

//______________________________________________________________________________
void TGButtonGroup::SetTitle(TGString *title)
{
   // Set or change title.

   if (!title) {
      Error("SetTitle", "title cannot be 0, try \"\"");
      return;
   }

   if (strcmp(fText->GetString(), title->GetString())) {
      SetBorderDrawn(title->GetLength() ? kTRUE : kFALSE);
      TGGroupFrame::SetTitle(title);
      ChangedBy("SetTitle");
   }
}

//______________________________________________________________________________
void TGButtonGroup::SetTitle(const char *title)
{
   // Set or change title.

   if (!title) {
      Error("SetTitle", "title cannot be 0, try \"\"");
      return;
   }

   if (strcmp(fText->GetString(), title)) {
      SetBorderDrawn(title && strlen(title));
      TGGroupFrame::SetTitle(title);
      ChangedBy("SetTitle");
   }
}

//______________________________________________________________________________
void TGButtonGroup::SetLayoutHints(TGLayoutHints *l, TGButton *button)
{
   // Set layout hints for the specified button or if button=0 for all
   // buttons.

   TGFrameElement *el;
   TIter next(fList);

   while ((el = (TGFrameElement *)next())) {
      if ((el->fFrame==(TGFrame*)button) || !button) {
         el->fLayout = l ? l : fgDefaultHints;
      }
   }
   Layout();
}

//______________________________________________________________________________
void TGButtonGroup::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a button group widget as a C++ statement(s) on output stream out

   char quote ='"';

   // font + GC
   option = GetName()+5;         // unique digit id of the name
   char parGC[50], parFont[50];
   sprintf(parFont,"%s::GetDefaultFontStruct()",IsA()->GetName());
   sprintf(parGC,"%s::GetDefaultGC()()",IsA()->GetName());
   
   if ((GetDefaultFontStruct() != fFontStruct) || (GetDefaultGC()() != fNormGC)) {
      TGFont *ufont = gClient->GetResourcePool()->GetFontPool()->FindFont(fFontStruct);
      if (ufont) {
         ufont->SavePrimitive(out, option);
         sprintf(parFont,"ufont->GetFontStruct()");
      } 

      TGGC *userGC = gClient->GetResourcePool()->GetGCPool()->FindGC(fNormGC);
      if (userGC) {
         userGC->SavePrimitive(out, option);
         sprintf(parGC,"uGC->GetGC()");
      } 
   }

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // buttongroup frame" << endl;

   out << "   TGButtonGroup *";
   out << GetName() << " = new TGButtonGroup(" << fParent->GetName()
       << ","<< quote << fText->GetString() << quote;

   if (fBackground == GetDefaultFrameBackground()) {
      if (fFontStruct == GetDefaultFontStruct()) {
         if (fNormGC == GetDefaultGC()()) {
            if (!GetOptions()) {
               out <<");" << endl;
            } else {
               out << "," << GetOptionString() <<");" << endl;
            }
         } else {
            out << "," << GetOptionString() << "," << parGC <<");" << endl;
         }
      } else {
         out << "," << GetOptionString() << "," << parGC << "," << parFont <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << "," << parGC << "," << parFont << ",ucolor);" << endl;
   }
   
   // setting layout manager
   out << "   " << GetName() <<"->SetLayoutManager(";
   GetLayoutManager()->SavePrimitive(out, option);
   out << ");"<< endl;

   TGFrameElement *f;
   TIter next(GetList());
   while ((f = (TGFrameElement *)next())) {
      f->fFrame->SavePrimitive(out,option);
      if (f->fFrame->InheritsFrom("TGButton")) continue;
      else {
         out << "   " << GetName() << "->AddFrame(" << f->fFrame->GetName();
         f->fLayout->SavePrimitive(out, option);
         out << ");"<< endl;
      }
   }

   if (IsExclusive())
      out << "   " << GetName() <<"->SetExclusive(kTRUE);" << endl;

   if (IsRadioButtonExclusive())
      out << "   " << GetName() <<"->SetRadioButtonExclusive(kTRUE);" << endl;

   if (!IsBorderDrawn())
      out << "   " << GetName() <<"->SetBorderDrawn(kFALSE);" << endl;


   out << "   " << GetName() << "->Resize(" << GetWidth()
       << "," << GetHeight() << ");" << endl;

   out << "   " << GetName() << "->Show();" << endl;
}

//______________________________________________________________________________
void TGHButtonGroup::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a button group widget as a C++ statement(s) on output stream out

   char quote ='"';

   // font + GC
   option = GetName()+5;         // unique digit id of the name
   char parGC[50], parFont[50];
   sprintf(parFont,"%s::GetDefaultFontStruct()",IsA()->GetName());
   sprintf(parGC,"%s::GetDefaultGC()()",IsA()->GetName());
   
   if ((GetDefaultFontStruct() != fFontStruct) || (GetDefaultGC()() != fNormGC)) {
      TGFont *ufont = gClient->GetResourcePool()->GetFontPool()->FindFont(fFontStruct);
      if (ufont) {
         ufont->SavePrimitive(out, option);
         sprintf(parFont,"ufont->GetFontStruct()");
      } 

      TGGC *userGC = gClient->GetResourcePool()->GetGCPool()->FindGC(fNormGC);
      if (userGC) {
         userGC->SavePrimitive(out, option);
         sprintf(parGC,"uGC->GetGC()");
      } 
   }

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // horizontal buttongroup frame" << endl;

   out << "   TGHButtonGroup *";
   out << GetName() << " = new TGHButtonGroup(" << fParent->GetName()
       << "," << quote << fText->GetString() << quote;
   if (fBackground == GetDefaultFrameBackground()) {

      if (fFontStruct == GetDefaultFontStruct()) {

         if (fNormGC == GetDefaultGC()()) {
            out << ");" << endl;
         } else {
            out << "," << parGC <<");" << endl;
         }
      } else {
         out << "," << parGC << "," << parFont <<");" << endl;
      }
   } else {
      out << "," << parGC << "," << parFont << ",ucolor);" << endl;
   }

   TGFrameElement *f;
   TIter next(GetList());
   while ((f = (TGFrameElement *)next())) {
      f->fFrame->SavePrimitive(out,option);
      if (f->fFrame->InheritsFrom("TGButton")){
         out << "   " << GetName() << "->SetLayoutHints(";
         f->fLayout->SavePrimitive(out, "nocoma");
         out << "," << f->fFrame->GetName();
         out << ");"<< endl;
      }
      else {
         out << "   " << GetName() << "->AddFrame(" << f->fFrame->GetName();
         f->fLayout->SavePrimitive(out, option);
         out << ");"<< endl;
      }
   }

   if (IsExclusive())
      out << "   " << GetName() <<"->SetExclusive(kTRUE);" << endl;

   if (IsRadioButtonExclusive())
      out << "   " << GetName() <<"->SetRadioButtonExclusive(kTRUE);" << endl;

   if (!IsBorderDrawn())
      out << "   " << GetName() <<"->SetBorderDrawn(kFALSE);" << endl;

   out << "   " << GetName() <<"->Resize(" << GetWidth() << ","
       << GetHeight() << ");" << endl;

   out << "   " << GetName() << "->Show();" << endl;
}

//______________________________________________________________________________
void TGVButtonGroup::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a button group widget as a C++ statement(s) on output stream out

   char quote ='"';

   // font + GC
   option = GetName()+5;         // unique digit id of the name
   char parGC[50], parFont[50];
   sprintf(parFont,"%s::GetDefaultFontStruct()",IsA()->GetName());
   sprintf(parGC,"%s::GetDefaultGC()()",IsA()->GetName());
   
   if ((GetDefaultFontStruct() != fFontStruct) || (GetDefaultGC()() != fNormGC)) {
      TGFont *ufont = gClient->GetResourcePool()->GetFontPool()->FindFont(fFontStruct);
      if (ufont) {
         ufont->SavePrimitive(out, option);
         sprintf(parFont,"ufont->GetFontStruct()");
      } 

      TGGC *userGC = gClient->GetResourcePool()->GetGCPool()->FindGC(fNormGC);
      if (userGC) {
         userGC->SavePrimitive(out, option);
         sprintf(parGC,"uGC->GetGC()");
      } 
   }

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << endl << "   // vertical buttongroup frame" << endl;

   out << "   TGVButtonGroup *";
   out << GetName() << " = new TGVButtonGroup(" << fParent->GetName()
       << "," << quote << fText->GetString() << quote;

   if (fBackground == GetDefaultFrameBackground()) {
      if (fFontStruct == GetDefaultFontStruct()) {
         if (fNormGC == GetDefaultGC()()) {
            out <<");" << endl;
         } else {
            out << "," << parGC <<");" << endl;
         }
      } else {
         out << "," << parGC << "," << parFont <<");" << endl;
      }
   } else {
      out << "," << parGC << "," << parFont << ",ucolor);" << endl;
   }

   TGFrameElement *f;
   TIter next(GetList());
   while ((f = (TGFrameElement *)next())) {
      f->fFrame->SavePrimitive(out,option);
      if (f->fFrame->InheritsFrom("TGButton")) continue;
      else {
         out << "   " << GetName() << "->AddFrame(" << f->fFrame->GetName();
         f->fLayout->SavePrimitive(out, option);
         out << ");"<< endl;
      }
   }

   if (IsExclusive())
      out << "   " << GetName() <<"->SetExclusive(kTRUE);" << endl;

   if (IsRadioButtonExclusive())
      out << "   " << GetName() <<"->SetRadioButtonExclusive(kTRUE);" << endl;

   if (!IsBorderDrawn())
      out << "   " << GetName() <<"->SetBorderDrawn(kFALSE);" << endl;

   out << "   " << GetName() << "->Resize(" << GetWidth()
       << "," << GetHeight() << ");"<< endl;
    
   out << "   " << GetName() << "->Show();" << endl;
}
