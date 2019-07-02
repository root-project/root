/// \file
/// \ingroup tutorial_webgui
///  This macro demonstrates simple openui5 panel, shown with RWebWindow
/// \macro_code
///
/// \author Sergey Linev


#include <ROOT/RWebWindowsManager.hxx>
#include "TBufferJSON.h"
#include <vector>
#include <string>

/** Simple structure for ComboBox item */
struct ComboBoxItem {
   std::string fId;
   std::string fName;
   ComboBoxItem() = default;
   ComboBoxItem(const std::string &id, const std::string &name) : fId(id), fName(name) {}
};

/** Full model used to configure openui5 widget */
struct TestPanelModel {
   std::string fSampleText;
   std::vector<ComboBoxItem> fComboItems;
   std::string fSelectId;
   std::string fButtonText;
};

std::shared_ptr<ROOT::Experimental::RWebWindow> window;
std::unique_ptr<TestPanelModel> model;

void ProcessData(unsigned connid, const std::string &arg)
{
   if (arg == "CONN_READY") {
      printf("connection established %u\n", connid);

      TString json = TBufferJSON::ToJSON(model.get());
      window->Send(connid, std::string("MODEL:") + json.Data());
      return;
   }

   if (arg == "CONN_CLOSED") {
      printf("connection closed\n");
      return;
   }

   if (arg == "REFRESH") {
      // send model to client again
      printf("Resend model\n");
      TString json = TBufferJSON::ToJSON(model.get());
      window->Send(connid, std::string("MODEL:") + json.Data());
      return;
   }

   if (arg.find("MODEL:") == 0) {
      printf("Decode model %s\n", arg.c_str());

      auto m = TBufferJSON::FromJSON<TestPanelModel>(arg.substr(6));
      if (m) {
         printf("Selected %s\n", m->fSelectId.c_str());
         std::swap(model, m);
      }
      return;
   }
}

void server()
{
   // prepare model
   model = std::make_unique<TestPanelModel>();
   model->fSampleText = "This is openui5 widget";
   model->fComboItems = {{"item1", "Text 1"}, {"item2", "Text 2"}, {"item3", "Text 3"}, {"item4", "Text 4"}};
   model->fSelectId = "item2";
   model->fButtonText = "Custom button";

   // create window
   window = ROOT::Experimental::RWebWindowsManager::Instance()->CreateWindow();

   // this is very important, it defines name of openui5 widget
   // "localapp" prefix indicates that all files located in current directory
   window->SetPanelName("localapp.view.TestPanel");

   // this is call-back, invoked when message received via websocket
   window->SetDataCallBack(ProcessData);

   window->SetGeometry(400, 500); // configure window geometry

   window->Show();
}
