sap.ui.define(['rootui5/panel/Controller',
               'sap/ui/core/Component',
               'sap/ui/core/mvc/Controller',
               'sap/ui/core/Control',
               'sap/ui/core/Icon',
               'sap/ui/core/mvc/XMLView',
               'sap/m/Text',
               'sap/m/CheckBox',
               'sap/m/MessageBox',
               'sap/m/MessageToast',
               'sap/m/TabContainerItem',
               'sap/ui/layout/Splitter',
               "sap/ui/core/ResizeHandler",
               "sap/ui/layout/HorizontalLayout",
               "sap/ui/core/util/File",
               "sap/ui/model/json/JSONModel",
               "rootui5/browser/model/BrowserModel",
               "sap/ui/core/Fragment",
               "sap/m/Link"
],function(GuiPanelController, Component, Controller, CoreControl, CoreIcon, XMLView, mText, mCheckBox, MessageBox, MessageToast, TabContainerItem,
           Splitter, ResizeHandler, HorizontalLayout, File, JSONModel, BrowserModel, Fragment, Link) {

   "use strict";
   // FIXME: cleanup unused modules

   /** FileDialog controller */

   return GuiPanelController.extend("rootui5.browser.controller.FileDialog", {

      //function called from GuiPanelController
      onPanelInit : function() {

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

         this.kind = "None"; // not yet known

         this.oModel = new JSONModel({ dialogTitle: "Dialog Title", filesList: [{name:"first.txt", counter: 11}, {name:"second.txt", counter: 22}, {name:"third.xml", counter: 33}]});
         this.getView().setModel(this.oModel);
      },

      onClosePress: async function() {
         if (window) window.open('','_self').close();
         this.isConnected = false;
      },

      /** @brief Send RBrowserRequest to the browser, not yet used */
      sendBrowserRequest: function(_oper, args) {
         var req = { path: "", first: 0, number: 0, sort: _oper };
         JSROOT.extend(req, args);
         this.websocket.Send("BRREQ:" + JSON.stringify(req));
      },

      updateBReadcrumbs: function(jsonpath) {
         this.currentPath = jsonpath;

        // let json = JSON.parse(jsonString);
        let split = jsonpath.split("/");
        let oBreadcrumbs = this.getView().byId("breadcrumbs");
        oBreadcrumbs.removeAllLinks();
        for (let i=0; i<split.length; i++) {
          if (i === 0) {
             let link = new Link();
             if (split[i].length === 2 && split[i][1] === ':') // Windows drive letter
               link.setText(split[i]);
             else
               link.setText("/");
            link.attachPress(this, this.onBreadcrumbsPress, this);
            oBreadcrumbs.addLink(link);
          } else {
            let link = new Link({text: split[i]});
            link.attachPress(this, this.onBreadcrumbsPress, this);
            oBreadcrumbs.addLink(link);
          }
        }
      },

     onBreadcrumbsPress: function(oEvent) {
        let sId = oEvent.getSource().sId;
        let oBreadcrumbs = oEvent.getSource().getParent();
        let oLinks = oBreadcrumbs.getLinks();
        let path = "/";
        for (let i = 1; i<oLinks.length; i++) {
          if (oLinks[i].sId === sId ) {
            path += oLinks[i].getText();
            break;
          }
          path += oLinks[i].getText() + "/";
        }

        this.websocket.Send('CHDIR:' + path); // dialog send reply itself
     },

     processInitMsg: function(msg) {
        console.log('INIT' + msg);

        var cfg = JSON.parse(msg);

        this.kind = cfg.kind; //

        this.updateBReadcrumbs(cfg.path);

        this.oModel.setProperty("/dialogTitle", cfg.title);
        this.oModel.setProperty("/filesList", cfg.brepl.nodes);
     },

     OnWebsocketOpened: function(handle) {
        this.isConnected = true;

        if (this.model)
           this.model.sendFirstRequest(this.websocket);
      },

      closeFileDialog: function() {
         // add more logic when FileDialog embed into main window
         if (this.did_close) return;
         if (window) window.open('','_self').close();
         this.did_close = true;
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
         case "GETWORKDIR":
            this.updateBReadcrumbs(msg);
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

         if (prop && (prop.icon == "sap-icon://folder-blank"))
            return this.websocket.Send('CHDIR:' + item.getTitle()); // dialog send chdir

         // this is final selection, server should close connection at the end
         this.websocket.Send('SELECT:' + item.getTitle());
      },

      onBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;
      }
   });

});
