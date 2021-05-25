sap.ui.define([
   'rootui5/panel/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/m/MessageToast'
], function (GuiPanelController, JSONModel, MessageToast) {
   "use strict";

   return GuiPanelController.extend("localapp.controller.TestPanel", {

      // function called from rootui5.panel.Controller
      onPanelInit : function() {
         this.setPanelTitle("TestPanel");
      },

      // function called from rootui5.panel.Controller
      onPanelExit : function() {
      },

      onPanelReceive: function(msg, offset) {
         if (typeof msg != "string") {
            // binary data transfer not used in this example
            var arr = new Float32Array(msg, offset);
            return;
         }

         if (msg.indexOf("MODEL:")==0) {
            var data = JSROOT.parse(msg.substr(6));
            if (data)
               this.getView().setModel(new JSONModel(data));
         } else {
            MessageToast.show("Receive msg: " + msg.substr(0,30));
         }
      },

      handleButtonPress: function() {
         MessageToast.show("Press sample button");
      },

      // just send model as is to the server back
      handleSendPress: function() {
         this.panelSend("MODEL:" + this.getView().getModel().getJSON());
      },

      handleRefreshPress: function() {
         this.panelSend("REFRESH");
      }
   });

});
