sap.ui.define(['rootui5/panel/Controller',
               "sap/ui/model/json/JSONModel",
               "sap/ui/core/Fragment",
               "sap/m/Link",
               "sap/m/Text",
               "sap/m/Button",
               "sap/m/ButtonType",
               "sap/m/Dialog"
],function(GuiPanelController, JSONModel, Fragment, Link, Text, Button, ButtonType, Dialog) {

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

      // returns full file name as array
      getFullFileName: function() {
         var oBreadcrumbs = sap.ui.core.Fragment.byId("FileDialogFragment", "breadcrumbs");
         var oLinks = oBreadcrumbs.getLinks();
         var path = [];
         for (var i = 1; i < oLinks.length; i++) {
            path.push(oLinks[i].getText());
         }

         var lastdir = oBreadcrumbs.getCurrentLocationText();
         if (lastdir) path.push(lastdir);

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
         oBreadcrumbs.setCurrentLocationText("");
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

         // console.log('GET DLG MGS: ', msg.substr(0,30));

         var mhdr = msg.split(":")[0];
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
         case "SELECT_CONFIRMED": // when selected file can be used for SaveAs operation
            this.closeDialog(true);
            break;
         case "NEED_CONFIRM": // need confirmation warning
            this.showWarningDialog();
            break;
         case "NOSELECT_CONFIRMED":
            this.closeDialog(false);
            break;

         default:
            console.error('Non recognized msg ' + mhdr + ' len=' + msg.length);
         }
      },

      showWarningDialog: function() {
         var oDialog = new Dialog({
            title: "Warning",
            type: "Message",
            state: "Warning",
            content: new Text({ text: "File already exists. Overwrite? "}),
            beginButton: new Button({
               text: 'Cancel',
               press: function() {
                  oDialog.close();
               }
            }),
            endButton: new Button({
               text: 'Ok',
               type: ButtonType.Emphasized,
               press: function() {
                  oDialog.close();
                  this.websocket.Send("DLG_CONFIRM_SELECT");
               }.bind(this)
            }),
            afterClose: function() {
               oDialog.destroy();
            }
         });

         oDialog.open();
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


      /** @brief Start SaveAs dialog @private */

      initDialog: async function(conn, filename, handler) {

         console.log("CALLING FileDialog.initDialog");
         this.kind = "None"; // not yet known
         this.oModel = new JSONModel({ canEnterFile: true, fileName: filename || "", filesList: [{name:"first.txt", counter: 11}, {name:"second.txt", counter: 22}, {name:"third.xml", counter: 33}]});

         // just initialize, server should confirm creation of channel
         this.websocket = conn;
         conn.SetReceiver(this);

         this.dialog_complete_handler = handler;

         var fragment;

         await Fragment.load({
            name: "rootui5.browser.view.filedialog",
            controller: this,
            id: "FileDialogFragment"
         }).then(function (oFragment) {
            fragment = oFragment;
         });

         fragment.setModel(this.oModel);

         this.dialog = new Dialog({
            title: "Select name for saving file",
            contentWidth: "70%",
            contentHeight: "50%",
            resizable: true,
            draggable: true,
            content: fragment,
            beginButton: new Button({
               text: 'Cancel',
               press: this.dialogBtnCancelPress.bind(this)
            }),
            endButton: new Button({
               text: 'Ok',
               press: this.dialogBtnOkPress.bind(this)
            })
         });

         this.dialog.addStyleClass("sapUiSizeCompact");

         this.dialog.open();
      },

      /** Press Ok button id Dialog, send selected file name and wait if confirmation required */
      dialogBtnOkPress: function() {
         var fullname = this.getFullFileName();

         console.log('Sending DLGSELECT ' + fullname)

         this.websocket.Send("DLGSELECT:" + JSON.stringify(fullname));

         // if (typeof this.dialog_complete_handler == "function")
         //    this.dialog_complete_handler(true);

         // this.closeDialog();
      },

      /** Press Cancel button id Dialog */
      dialogBtnCancelPress: function() {

         this.websocket.Send("DLGNOSELECT");

         // if (typeof this.dialog_complete_handler == "function")
         //    this.dialog_complete_handler(false);

         // this.closeDialog();
      },

      /** Method to close dialog */
      closeDialog: function(result) {
         if ((result !== undefined) && (typeof this.dialog_complete_handler == "function"))
            this.dialog_complete_handler(result);

         delete this.dialog_complete_handler;

         if (this.websocket) {
            this.websocket.Close();
            delete this.websocket;
         }

         if (this.dialog) {
            this.dialog.close();
            this.dialog.destroy();
            delete this.dialog;
         }

      },

      onBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;
      }
   });

});
