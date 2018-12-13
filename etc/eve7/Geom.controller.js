sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/layout/Splitter',
               'sap/ui/layout/SplitterLayoutData'
],function(Controller, Splitter, SplitterLayoutData) {
   "use strict";

   return Controller.extend("eve.Geom", {
      onInit: function () {
         
         console.log('GEOM CONTROLLER INIT');
         
         console.log('USE CONNECTION', this.getView().getViewData().conn_handle);
      },
      

      onAfterRendering: function(){
      },

      onToolsMenuAction : function (oEvent) {

         var item = oEvent.getParameter("item");

         switch (item.getText()) {
            case "GED Editor": this.getView().byId("Summary").getController().toggleEditor(); break;
         }
      },
      
      showHelp : function(oEvent) {
         alert("User support: root-webgui@cern.ch");
      },
      
      showUserURL : function(oEvent) {
         sap.m.URLHelper.redirect("https://github.com/alja/jsroot/blob/dev/eve7.md", true);
      }
   });
});
