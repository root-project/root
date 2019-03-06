sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel'
], function (Controller, JSONModel) {
   "use strict";

   return Controller.extend("rootui5.panel.Controller", {

      onInit: function() {
         var data = this.getView().getViewData();

         if (data && data.handle) {
            this.websocket = data.handle;
            this.websocket.SetReceiver(this); // redirect websocket handling on controller itself
            this.websocket.Send("PANEL_READY"); // confirm panel creation, only then GUI can send commands
         }
         
         // TODO: use more specific API between Canvas and Panel
         if (data && data.masterPanel) {
            this.masterPanel = data.masterPanel;
         }

         if (this.onPanelInit) this.onPanelInit();
      },

      OnWebsocketOpened: function(handle) {
         console.log('Connection established - should never happen');
      },

      OnWebsocketMsg: function(handle, msg) {
          console.log('GuiPanel Get message ' + msg);
      },

      OnWebsocketClosed: function(handle) {
          console.log('GuiPanel closed');
          if (window) window.open('','_self').close(); // window.close();
          delete this.websocket; // remove reference on websocket
      },

      onExit: function() {
         if (this.onPanelExit) 
            this.onPanelExit();
         if (this.websocket) {
            this.websocket.Close();
            delete this.websocket;
         }
      },

      closePanel: function() {
         
         if (this.masterPanel) {
            if (this.masterPanel.showLeftArea) this.masterPanel.showLeftArea("");
         } else {
            if (window) window.open('','_self').close(); // window.close();
         }
      }

   });

});
