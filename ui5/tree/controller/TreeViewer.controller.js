sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/model/json/JSONModel'
],function(Controller, JSONModel) {

   "use strict";

   /** Tree viewer contoller
     * All TTree functionality is loaded after main ui5 rendering is performed */

   return Controller.extend("rootui5.tree.controller.TreeViewer", {
      onInit: function () {

         let viewData = this.getView().getViewData();

         this.websocket = viewData.conn_handle;
         this.jsroot = viewData.jsroot;
         this._embeded = viewData.embeded;

         // this is code for the Components.js
         // this.websocket = Component.getOwnerComponentFor(this.getView()).getComponentData().conn_handle;

         this.websocket.setReceiver(this);
         this.websocket.connect(viewData.conn_href);

         this.queue = []; // received draw messages

         // if true, most operations are performed locally without involving server
         this.standalone = this.websocket.kind == "file";

         this.cfg = {
            standalone: this.websocket.kind == "file",
            not_embeded: !this._embeded
         };
         this.cfg_model = new JSONModel(this.cfg);
         this.getView().setModel(this.cfg_model);

         this.checkSendRequest();
      },

      onWebsocketOpened: function(/*handle*/) {
         this.isConnected = true;

         // when connection established, checked if we can submit request
         this.checkSendRequest();
      },

      onWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');

         if (window && !this._embeded) window.close();

         this.isConnected = false;
      },

      /** Entry point for all data from server */
      onWebsocketMsg: function(handle, msg /*, offset */) {

         // binary data can be send only as addition to draw message
         // here data can be placed in the queue and processed when all other prerequicities are done
         if (typeof msg != "string")
            return console.error(`TreeViewer does not uses binary messages len = ${mgs.byteLength}`);

         let mhdr = msg.substr(0,6);
         msg = msg.substr(6);

         console.log(mhdr, msg.length, msg.substr(0,70), "...");

         switch (mhdr) {
         case "RELOAD":
            this.doReload(true);
            break;
         case "VIEWER:":   // generic viewer configuration
            this.checkViewerMsg(this.jsroot.parse(msg)); // use jsroot.parse while refs are used
            break;
         default:
            console.error(`Non recognized msg ${mhdr} len= ${msg.length}`);
         }
      },

      /** @summary processing viewer configuration */
      checkViewerMsg: function(msg) {
         console.log('checkViewerMsg', msg);
      },

      onBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;

         this.checkSendRequest();
      },

      checkSendRequest: function(force) {
         if (force) this.ask_getdraw = false;

         if (this.isConnected && this.renderingDone) {

            if (this.geo && !this.ask_getdraw) {
               this.websocket.send("GETVIEWER");
               this.ask_getdraw = true;
            }
         }
      },

      performDraw: function() {
      },

      /** @summary Reload geometry description and base drawing, normally not required */
      onRealoadPress: function () {
         this.doReload(true);
      },

      doReload: function(force) {
         if (this.standalone) {
            // offline uscase, maybe iremove later
         } else {
            this.checkSendRequest(force);

            // keep here - if implementing browsing of tree hierarchy
            if (this.model) {
               this.model.clearFullModel();
               this.model.reloadMainModel(force);
            }
         }
      },

      /** Quit ROOT session */
      onQuitRootPress: function() {
         if (!this.standalone)
            this.websocket.send("QUIT_ROOT");
      }

   });

});
