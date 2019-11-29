sap.ui.define(['rootui5/panel/Controller',
               "sap/ui/model/json/JSONModel",
               'sap/ui/core/Fragment',
               "sap/m/Link"
],function(GuiPanelController, JSONModel, Fragment, Link) {

   "use strict";

   /** FileDialog controller */

   return GuiPanelController.extend("rootui5.browser.controller.FileDialog", {

      //function called from GuiPanelController
      onPanelInit : function() {

         // TODO: provide functionality via BrowserModel - once we know that exactly we need

         // create model only for browser - no need for anybody else
         // this.model = new BrowserModel();

         // copy extra attributes from element to node in the browser
         // later can be done automatically
         /* this.model.addNodeAttributes = function(node, elem) {
            node.icon = elem.icon;
            node.fsize = elem.fsize;
            node.mtime = elem.mtime;
            node.ftype = elem.ftype;
            node.fuid = elem.fuid;
            node.fgid = elem.fgid;
            node.className = elem.className
         };
         */

         console.log("CALLING FileDialog.onPanelInit");

         this.kind = "None"; // not yet known
         this.oModel = new JSONModel({ dialogTitle: "Dialog Title", filesList: [{name:"first.txt", counter: 11}, {name:"second.txt", counter: 22}, {name:"third.xml", counter: 33}]});
         this.getView().setModel(this.oModel);

         var pthis = this;

         Fragment.load({name: "rootui5.browser.view.filedialog", controller: this, id: "FileDialogFragment"}).then(function (oFragment) {
            pthis.getView().addDependent(oFragment);

            pthis.getView().byId("dialogPage").addContent(oFragment);
            oFragment.setModel(pthis.oModel);

         });
      },

      /** @brief Use controller with m.Dialog, no separate view
       * @private
       * TODO: make special method to create full dialog here */
      initDialog: function(conn, filename) {

         console.log("CALLING FileDialog.initDialog");
         this.kind = "None"; // not yet known
         this.oModel = new JSONModel({ canEnterFile: true, fileName: filename || "", filesList: [{name:"first.txt", counter: 11}, {name:"second.txt", counter: 22}, {name:"third.xml", counter: 33}]});

         // just initialize, server should confirm creation of channel
         this.websocket = conn;
         conn.SetReceiver(this);
      },

      // returns full file name as array
      getFullFileName: function() {
         var oBreadcrumbs = sap.ui.core.Fragment.byId("FileDialogFragment", "breadcrumbs");
         let oLinks = oBreadcrumbs.getLinks();
         let path = [];
         for (let i = 0; i < oLinks.length; i++) {
            if (i>0) path.push(oLinks[i].getText());
         }
         path.push(this.oModel.getProperty("/fileName"));
         return path;
      },

      onClosePress: async function() {
         if (window) window.open('','_self').close();
         this.isConnected = false;
      },

      updateBReadcrumbs: function(split) {
         var oBreadcrumbs = sap.ui.core.Fragment.byId("FileDialogFragment", "breadcrumbs");
         oBreadcrumbs.removeAllLinks();
         for (let i=-1; i<split.length; i++) {
            let txt = i<0 ? "/": split[i];
            if (i === split.length-1) {
               oBreadcrumbs.setCurrentLocationText(txt);
            } else {
               let link = new Link({text: txt});
               link.attachPress(this, this.onBreadcrumbsPress, this);
               oBreadcrumbs.addLink(link);
            }
         }
      },

     onBreadcrumbsPress: function(oEvent) {
        let sId = oEvent.getSource().sId;
        let oBreadcrumbs = oEvent.getSource().getParent();
        let oLinks = oBreadcrumbs.getLinks();
        let path = [];
        for (let i = 0; i < oLinks.length; i++) {
           if (i>0) path.push(oLinks[i].getText());
           if (oLinks[i].getId() === sId ) break;
        }
        this.websocket.Send('CHPATH:' + JSON.stringify(path));
     },

     processInitMsg: function(msg) {
        console.log('INIT' + msg);

        var cfg = JSON.parse(msg);

        this.kind = cfg.kind; //

        this.updateBReadcrumbs(cfg.path);

        this.oModel.setProperty("/dialogTitle", cfg.title);
        this.oModel.setProperty("/filesList", cfg.brepl.nodes);
     },

     closeFileDialog: function() {
        // add more logic when FileDialog embed into main window
        if (this.did_close) return;
        console.log('TRY TO CLOSE FILE DIALOG');

        if (this.dialog) {
           this.dialog.close();
           this.dialog.destroy();
        } else if (window) {
           window.open('','_self').close();
        }

        this.did_close = true;
     },

      OnWebsocketOpened: function(handle) {
         this.isConnected = true;

         if (this.model)
            this.model.sendFirstRequest(this.websocket);
      },

      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');
         this.closeFileDialog();
         this.isConnected = false;
      },

      /** Entry point for all data from server */
      OnWebsocketMsg: function(handle, msg) {

         if (typeof msg != "string")
            return console.error("Browser do not uses binary messages len = " + mgs.byteLength);

         let mhdr = msg.split(":")[0];
         msg = msg.substr(mhdr.length+1);

         switch (mhdr) {
         case "INMSG":
            this.processInitMsg(msg);
            break;
         case "CLOSE":
            this.closeFileDialog();
            break;
         case "WORKPATH":
            this.updateBReadcrumbs(JSON.parse(msg));
            break;
         case "BREPL":   // browser reply
            var repl = JSON.parse(msg);
            this.oModel.setProperty("/filesList", repl.nodes);
            break;
         default:
            console.error('Non recognized msg ' + mhdr + ' len=' + msg.length);
         }
      },

      onItemPress: function(event) {
         var item = event.getParameters().listItem;
         if (!item) return;

         var ctxt = item.getBindingContext(),
             prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;

         // console.log('Property', prop);

         if (prop && (prop.icon == "sap-icon://folder-blank")) {
            this.oModel.setProperty("/fileName", "");
            return this.websocket.Send('CHDIR:' + item.getTitle()); // dialog send chdir
         }


         this.oModel.setProperty("/fileName", item.getTitle());

         // this is final selection, server should close connection at the end
         // this.websocket.Send('SELECT:' + item.getTitle());
      },

      onBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;
      }
   });

});
