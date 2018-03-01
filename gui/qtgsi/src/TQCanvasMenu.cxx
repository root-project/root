// @(#)root/qtgsi:$Id$
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQCanvasMenu.h"
#include "TQRootDialog.h"

#include "Riostream.h"

#include "TClass.h"
#include "TROOT.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "TMethodArg.h"
#include "TCanvas.h"
#include "TDataType.h"

ClassImp(TQCanvasMenu);

////////////////////////////////////////////////////////////////////////////////
/// ctor, create the popup menu

TQCanvasMenu::TQCanvasMenu(QWidget* wparent, TCanvas *canvas)
{
   fc       = canvas;
   fCurrObj = 0;
   fParent  = wparent;
   fTabWin  = 0;
   fDialog  = 0;
   fMousePosX = fMousePosY = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// ctor, create the popup menu

TQCanvasMenu::TQCanvasMenu(QWidget* wparent, QWidget *tabWin, TCanvas *canvas)
{
   fc       = canvas;
   fParent  = wparent;
   fTabWin  = tabWin;
   fCurrObj = 0;
   fDialog  = 0;
   fMousePosX = fMousePosY = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TQCanvasMenu::~TQCanvasMenu()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create title for dialog box retrieving argument values.

char *TQCanvasMenu::CreateDialogTitle( TObject *object, TMethod *method )
{
   static char methodTitle[128];

   if (object && method)
      snprintf(methodTitle, 127, "%s::%s", object->ClassName(), method->GetName());
   else
      *methodTitle = 0;
   return methodTitle;

}

////////////////////////////////////////////////////////////////////////////////
/// Create string describing argument (for use in dialog box).

char *TQCanvasMenu::CreateArgumentTitle(TMethodArg *argument)
{
   static Char_t argTitle[128];
   if (argument) {
      snprintf(argTitle, 127, "(%s)  %s", argument->GetTitle(), argument->GetName());
      const char *arg_def = argument->GetDefault();
      if (arg_def && *arg_def) {
         strncat(argTitle, "  [default: ", 127 - strlen(argTitle));
         strncat(argTitle, arg_def, 127 - strlen(argTitle));
         strncat(argTitle, "]", 127 - strlen(argTitle));
      }
   }
   else
      *argTitle = 0;

   return argTitle;
}

////////////////////////////////////////////////////////////////////////////////
/// Perform the corresponding selected TObject  popup
/// in the position defined
/// by x, y coordinates (in user coordinate system).
/// @param obj (TObject*)
/// @param p (QPoint&)

void TQCanvasMenu::Popup(TObject *obj, double x, double y, QMouseEvent *e)
{
   TClass *cls = obj->IsA();

   fCurrObj=obj;
   fMenu.clear();
   fMethods.Clear();

   QString name = cls->GetName();
   name = name + "::" + obj->GetName();

   fMenu.addAction(name , this, SLOT(Execute(int)), 0);
   fMenu.addSeparator();

   cls->GetMenuItems(&fMethods);
   for (auto item : fMethods)
      if (TMethod *method = dynamic_cast<TMethod*>(item))
         fMenu.addAction(method->GetName(), this, SLOT(Execute(int)), 0);

   // hold the position where the mouse was clicked
   fMousePosX = x;
   fMousePosY = y;

   // let Qt decide how to draw the popup Menu otherwise we have a problem that
   // the visible rectangle can get outside the screen (M.T. 03.06.02)
   fMenu.popup(e->globalPos(), /* QAction* */ nullptr);
}

////////////////////////////////////////////////////////////////////////////////
/// Slot defined to execute a method from a selected TObject
/// using TObject::Execute() function.

void TQCanvasMenu::Execute(int id)
{
   if (id < 0) return;
   QString text="";

   TVirtualPad  *psave = gROOT->GetSelectedPad();
   TMethod *method=(TMethod *)fMethods.At(id);
   fc->HandleInput(kButton3Up,gPad->XtoAbsPixel(fMousePosX), gPad->YtoAbsPixel(fMousePosY) );
   if (  method->GetListOfMethodArgs()->First() ) {
      Dialog(fCurrObj,method);
   }
   else {
      gROOT->SetFromPopUp(kTRUE);
      fCurrObj->Execute((char *) method->GetName(), "");
   }
   fc->GetPadSave()->Update();
   fc->GetPadSave()->Modified();
   gROOT->SetSelectedPad(psave);
   gROOT->GetSelectedPad()->Update();
   gROOT->GetSelectedPad()->Modified();
   fc->Modified();
   fc->ForceUpdate();
   gROOT->SetFromPopUp( kFALSE );
}

////////////////////////////////////////////////////////////////////////////////
/// Create dialog object with OK and Cancel buttons. This dialog
/// prompts for the arguments of "method".

void TQCanvasMenu::Dialog(TObject* object, TMethod* method)
{
   if (!(object && method)) return;
   fDialog = new TQRootDialog(fParent, CreateDialogTitle(object, method), object, method);
   fDialog->SetTCanvas(fc);
   // iterate through all arguments and create apropriate input-data objects:
   // inputlines, option menus...
   TMethodArg *argument = 0;
   TIter next(method->GetListOfMethodArgs());
   while ((argument = (TMethodArg *) next())) {
      char       *argname    = CreateArgumentTitle(argument);
      const char *type       = argument->GetTypeName();
      TDataType    *datatype   = gROOT->GetType(type);
      const char *charstar   = "char*";
      char        basictype [32];

      if (datatype) {
         strlcpy(basictype, datatype->GetTypeName(),32);
      }
      else {
         if (strncmp(type, "enum", 4) != 0)
         std::cout << "*** Warning in Dialog(): data type is not basic type, assuming (int)\n";
         strcpy(basictype, "int");
      }

      if (strchr(argname, '*')) {
         strcat(basictype, "*");
         type = charstar;
      }

      TDataMember *m = argument->GetDataMember();
      if (m && m->GetterMethod()) {
         char gettername[256] = "";
         strlcpy(gettername, m->GetterMethod()->GetMethodName(),256);
         m->GetterMethod()->Init(object->IsA(), gettername, "");
         // Get the current value and form it as a text:
         char val[256];
         if (!strncmp(basictype, "char*", 5)) {
            char *tdefval = 0;
            m->GetterMethod()->Execute(object, "", &tdefval);
            if (tdefval && strlen(tdefval))
               strlcpy(val, tdefval, 256);
         }
         else if (!strncmp(basictype, "float", 5) ||
            !strncmp(basictype, "double", 6)) {
            Double_t ddefval = 0.0;
            m->GetterMethod()->Execute(object, "", ddefval);
            snprintf(val, 255, "%g", ddefval);
         }
         else if (!strncmp(basictype, "char", 4) ||
             !strncmp(basictype, "int", 3)  ||
             !strncmp(basictype, "long", 4) ||
             !strncmp(basictype, "short", 5)) {
            Long_t ldefval = 0L;
            m->GetterMethod()->Execute(object, "", ldefval);
            snprintf(val, 255, "%li", ldefval);
         }
         // Find out whether we have options ...
         TList *opt;
         if ((opt = m->GetOptions())) {
            std::cout << "*** Warning in Dialog(): option menu not yet implemented " << opt << std::endl;
            // should stop dialog
            return;
         }
         else {
            // we haven't got options - textfield ...
            fDialog->Add(argname, val, type);
         }
      }
      else {    // if m not found ...
         char val[256] = "";
         const char *tval = argument->GetDefault();
         if (tval) strlcpy(val, tval, 256);
         fDialog->Add(argname, val, type);
      }
   } //end while

   fDialog->Popup();
}
