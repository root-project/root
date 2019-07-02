sap.ui.define([
   'rootui5/panel/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/m/MessageToast'
], function (GuiPanelController, JSONModel, MessageToast) {
   "use strict";

   return GuiPanelController.extend("localapp.controller.TestPanel", {

      // function called from rootui5.panel.Controller
      onPanelInit : function() {
         if (document) document.title = "TestPanel";
      },

      // function called from rootui5.panel.Controller
      onPanelExit : function() {
      },

      OnWebsocketMsg: function(handle, msg, offset) {
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
            this.getView().byId("SampleText").setText("Get message:\n" + msg);
         }
      },

      handleButtonPress: function() {
         MessageToast.show("Press sample button");
      },

      handleSendPress: function() {
         // just send model as is to the server back
         if (this.websocket)
            this.websocket.Send("MODEL:" + this.getView().getModel().getJSON());
      },

      handleRefreshPress: function() {
         if (this.websocket)
            this.websocket.Send("REFRESH");
      }
   });

});
