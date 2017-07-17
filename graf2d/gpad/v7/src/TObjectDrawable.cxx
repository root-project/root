/// \file TObjectDrawable.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!


/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TObjectDrawable.hxx>

#include <ROOT/TDisplayItem.hxx>
#include <ROOT/TLogger.hxx>
#include <ROOT/TVirtualCanvasPainter.hxx>

#include "TClass.h"
#include "TList.h"
#include "TMethod.h"
#include "TMethodCall.h"
#include "TROOT.h"

#include <exception>


void ROOT::Experimental::Internal::TObjectDrawable::Paint(TVirtualCanvasPainter& canv) {
   ROOT::Experimental::TDisplayItem *res = new TOrdinaryDisplayItem<TObject>(fObj.get());
   res->SetOption(fOpts.c_str());

   canv.AddDisplayItem(res);
}

void ROOT::Experimental::Internal::TObjectDrawable::FillMenu(TVirtualCanvasPainter& onCanv) {
   TObject *obj = fObj.get();

   // fill context menu items for the ROOT class

   TClass* cl = obj->IsA();

   TList* lst = new TList;
   cl->GetMenuItems(lst);

   TIter iter(lst);
   TMethod* m = 0;

   while ((m = (TMethod*) iter()) != 0) {

      if (m->IsMenuItem() == kMenuToggle) {
         TString getter;
         if (m->Getter() && strlen(m->Getter()) > 0) {
            getter = m->Getter();
         } else if (strncmp(m->GetName(),"Set",3)==0) {
            getter = TString(m->GetName())(3, strlen(m->GetName())-3);
            if (cl->GetMethodAllAny(TString("Has") + getter)) getter = TString("Has") + getter;
            else if (cl->GetMethodAllAny(TString("Get") + getter)) getter = TString("Get") + getter;
            else if (cl->GetMethodAllAny(TString("Is") + getter)) getter = TString("Is") + getter;
            else getter = "";
         }

         if ((getter.Length()>0) && cl->GetMethodAllAny(getter)) {
            // execute getter method to get current state of toggle item

            TMethodCall* call = new TMethodCall(cl, getter, "");

            if (call->ReturnType() == TMethodCall::kLong) {
               Long_t l(0);
               call->Execute(obj, l);

               onCanv.AddChkMenuItem(m->GetName(), m->GetTitle(), l!=0, Form("%s(%s)", m->GetName(), (l!=0) ? "0" : "1"));

            } else {
               // Error("CheckModifiedFlag", "Cannot get toggle value with getter %s", getter.Data());
            }

            delete call;
         }
      } else {
         onCanv.AddMenuItem(m->GetName(), m->GetTitle(), Form("%s()", m->GetName()));
      }
   }

}


void ROOT::Experimental::Internal::TObjectDrawable::ExecMenu(const std::string &exec)
{
   TObject *obj = fObj.get();

   TString cmd;
   cmd.Form("((%s*) %p)->%s;", obj->ClassName(), obj, exec.c_str());
   printf("TObjectDrawable::ExecMenu Obj %s Execute %s\n", obj->GetName(), cmd.Data());
   gROOT->ProcessLine(cmd);

}
