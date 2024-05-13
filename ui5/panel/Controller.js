sap.ui.define([
   'sap/ui/core/mvc/Controller'
], function (Controller) {
   "use strict";

   return Controller.extend("rootui5.panel.Controller", {

      // one could define in derived classes followin methods
      //   onPanelInit: function() {},
      //   onPanelReceive: function(msg, offset) {},
      //   onPanelExit: function(),


      onInit: function() {
         let data = this.getView().getViewData();

         this.websocket = data?.handle;
         if (this.websocket) {
            this.websocket.setReceiver(this); // redirect websocket handling on controller itself
            this.websocket.send("PANEL_READY"); // confirm panel creation, only then GUI can send commands
         }

         // assign several core methods which are used like: parse, toJSON, source_dir
         this.jsroot = data?.jsroot;

         // TODO: use more specific API between Canvas and Panel
         this.masterPanel = data?.masterPanel;

         // let correctly reload Panel
         if (!this.masterPanel && this.websocket)
            this.websocket.addReloadKeyHandler();


         if (typeof this.onPanelInit == "function")
            this.onPanelInit();
      },

      onWebsocketOpened: function(handle) {
         console.log('Connection established - should never happen');
      },

      onWebsocketMsg: function(handle, msg, offset) {
         if (typeof this.onPanelReceive == 'function')
            this.onPanelReceive(msg, offset);
         else
            console.log('GuiPanel Get message ' + msg);
      },

      onWebsocketClosed: function(handle) {
          console.log('GuiPanel closed');
          if (window) window.open('','_self').close(); // window.close();
          delete this.websocket; // remove reference on websocket
      },

      setPanelTitle: function(title) {
         if (!this.masterPanel && document)
            document.title = title;
      },

      panelSend: function(msg) {
         if (this.websocket)
            this.websocket.send(msg);
         else
            console.error('No connection available to send message');
      },

      onExit: function() {
         if (typeof this.onPanelExit == 'function')
            this.onPanelExit();
         if (this.websocket) {
            this.websocket.close();
            delete this.websocket;
         }
      },

      /** Method should be used to close panel
       * Depending from used window manager different functionality can be used here */
      closePanel: function() {
         if (this.masterPanel) {
            if (this.masterPanel.showLeftArea) this.masterPanel.showLeftArea("");
         } else {
            if (window) window.open('','_self').close(); // window.close();
         }
      }

   });

});
