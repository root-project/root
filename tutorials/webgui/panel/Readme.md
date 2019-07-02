## OPENUI5 Panel example

This is special kind of openui5-based widget.

It is normal xml::View, but controller should be derived from sap.ui.jsroot.GuiPanelController class.
This class provides number of methods, which should simplify handling of communication between server and client

First of all, when creating RWebWindow, one should configure panel name. Like:

     auto win = ROOT::Experimental::RWebWindowsManager::Instance()->CreateWindow();

     // this is very important, it defines name of openui5 widget, which
     // will run on the client side
     win->SetPanelName("localapp.view.TestPanel");

 
 Namespace "localapp" in this case corresponds to openui5 files, which will be loaded from current directory.
 Therefore `"localapp.view.TestPanel"` means view, which will be loaded from `./view/TestPanel.view.xml` file.
 
 Controller is configured in the XML file and called `"localapp.controller.TestPanel"`. 
 Means it will be loaded from `./controller/TestPanel.controller.js` file.
 
 In the controller one use `onPanelInit` and `onPanelExit` methods to handle initialization and close of widget.
 Also  
  
 