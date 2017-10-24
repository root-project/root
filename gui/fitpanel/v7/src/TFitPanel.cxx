/// \file ROOT/TFitPanel.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-10-24
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TFitPanel.hxx"

#include <ROOT/TWebWindowsManager.hxx>

#include "TString.h"
#include "TROOT.h"


std::shared_ptr<ROOT::Experimental::TWebWindow> ROOT::Experimental::TFitPanel::GetWindow()
{
   if (!fWindow) {
      fWindow = TWebWindowsManager::Instance()->CreateWindow(false);

      fWindow->SetPanelName("FitPanel");

      fWindow->SetDataCallBack(std::bind(&TFitPanel::ProcessData, this, std::placeholders::_1, std::placeholders::_2));
   }

   return fWindow;
}


void ROOT::Experimental::TFitPanel::Show(const std::string &where)
{
   GetWindow()->Show(where);
}

void ROOT::Experimental::TFitPanel::Hide()
{
   if (!fWindow) return;

   fWindow->CloseConnections();
}

void ROOT::Experimental::TFitPanel::ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "CONN_READY") {
      fConnId = connid;
      printf("Connection established %u\n", fConnId);
      fWindow->Send("SHOW:FitPanel", fConnId);
      return;
   }

   if (arg == "CONN_CLOSED") {
      printf("Connection closed\n");
      fConnId = 0;
      return;
   }

   if (arg.find("DOFIT:")==0) {
      TString exec;
      exec.Form("((ROOT::Experimental::TFitPanel *) %p)->DoFit(%s);", this, arg.c_str()+6);
      printf("Execute %s\n", exec.Data());
      gROOT->ProcessLine(exec.Data());
      return;
   }
}

/// method called from the UI
void ROOT::Experimental::TFitPanel::DoFit(const std::string &dname, const std::string &mname)
{
   printf("DoFit %s %s\n", dname.c_str(), mname.c_str());

}

