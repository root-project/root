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

   var FileDialog = GuiPanelController.extend("rootui5.browser.controller.FileDialog", {

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

         this.kind = "None"; // not yet known
         this.oModel = new JSONModel({ dialogTitle: "Dialog Title", fileName: "", filesList: [{name:"first.txt", counter: 11}, {name:"second.txt", counter: 22}, {name:"third.xml", counter: 33}]});
         this.getView().setModel(this.oModel);

         var pthis = this;

         Fragment.load({name: "rootui5.browser.view.filedialog", controller: this, id: "FileDialogFragment"}).then(function (oFragment) {
            pthis.getView().addDependent(oFragment);

            pthis.getView().byId("dialogPage").addContent(oFragment);
            oFragment.setModel(pthis.oModel);
         });

         this.own_window = true;
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
         if (this.kind == "OpenFile")
            this.oModel.setProperty("/fileName", "");
         this.websocket.Send('CHPATH:' + JSON.stringify(path));
      },

      processInitMsg: function(msg) {
         var cfg = JSON.parse(msg);

         if (!this.dialog) {
            // when not an embedded dialog, update configuration from server
            this.kind = cfg.kind;
            this.oModel.setProperty("/dialogTitle", cfg.title);
            this.oModel.setProperty("/canEditName", (this.kind == "SaveAs") || (this.kind == "NewFile"));
         }

         if (cfg.fname)
            this.oModel.setProperty("/fileName", cfg.fname);

         this.updateBReadcrumbs(cfg.path);

         this.oModel.setProperty("/filesList", cfg.brepl.nodes);
      },

      /** Close file dialog */
      closeFileDialog: function(fname) {
         // add more logic when FileDialog embed into main window
         if (this.did_close) return;

         this.did_close = true;

         if (fname !== undefined)
            this._completeDialog(fname ? "onOk" : "onCancel", fname);

         if (this.websocket) {
            this.websocket.Close();
            delete this.websocket;
         }

         if (this.dialog) {
            this.dialog.close();
            this.dialog.destroy();
            delete this.dialog;
         }

         if (this.own_window && window)
            window.open('','_self').close();
      },

      OnWebsocketOpened: function(handle) {
         if (this.model)
            this.model.sendFirstRequest(this.websocket);
      },

      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         this.closeFileDialog();
      },

      /** Entry point for all data from server */
      OnWebsocketMsg: function(handle, msg) {

         if (typeof msg != "string")
            return console.error("Browser do not uses binary messages len = " + mgs.byteLength);

         console.log('GET DLG MGS: ', msg.substr(0,50));

         var mhdr = msg.split(":")[0];
         msg = msg.substr(mhdr.length+1);

         switch (mhdr) {
         case "INMSG":
            this.processInitMsg(msg);
            break;
         case "WORKPATH":
            this.updateBReadcrumbs(JSON.parse(msg));
            break;
         case "BREPL":   // browser reply
            var repl = JSON.parse(msg);
            this.oModel.setProperty("/filesList", repl.nodes);
            break;
         case "SELECT_CONFIRMED": // when selected file can be used for SaveAs operation
            this.closeFileDialog(msg);
            break;
         case "NEED_CONFIRM": // need confirmation warning
            this.showWarningDialog();
            break;
         case "NOSELECT_CONFIRMED":
            this.closeFileDialog("");
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

      /** Handler for List item press event */
      onItemPress: function(event) {
         var item = event.getParameters().listItem;
         if (!item) return;

         var ctxt = item.getBindingContext(),
             prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;

         // console.log('Property', prop);

         if (prop && (prop.icon == "sap-icon://folder-blank")) {
            if (this.kind == "OpenFile")
               this.oModel.setProperty("/fileName", "");
            return this.websocket.Send('CHDIR:' + item.getTitle()); // dialog send chdir
         }

         this.oModel.setProperty("/fileName", item.getTitle());
      },

      /** method used to complete dialog */
      _completeDialog: function(funcname, arg) {
         if (!this.dialog_args)
            return;

         if (!funcname) funcname = "onFailure";

         if (typeof this.dialog_args[funcname] == "function")
            this.dialog_args[funcname](arg);

         delete this.dialog_args;
      },

      /** @brief Start SaveAs dialog @private */

      _initDialog: async function(kind, args) {

         if (!args || typeof args != "object")
            return null;

         this.dialog_args = args || {};
         if (!args.websocket)
            return this._completeDialog("onFailure");

         var fname = args.filename || "";
         var p = Math.max(fname.lastIndexOf("/"), fname.lastIndexOf("\\"));
         if (p>0) fname = fname.substr(p+1);

         this.kind = kind; // not yet known
         this.oModel = new JSONModel({
                              canEditName: (this.kind == "SaveAs") || (this.kind == "NewFile"),
                              dialogTitle: args.title || "Title",
                              fileName: fname, // will be returned from the server, just for initialization
                              filesList: [{name:"first.txt", counter: 11}, {name:"second.txt", counter: 22}, {name:"third.xml", counter: 33}]
         });

         // create extra channel for the FileDialog
         this.websocket = args.websocket.CreateChannel();
         // assign ourself as receiver of all
         this.websocket.SetReceiver(this);

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
            title: "{/dialogTitle}",
            contentWidth: args.width || "70%",
            contentHeight: args.height || "50%",
            resizable: (args.resizable === undefined) ? true : args.resizable,
            draggable: (args.draggable === undefined) ? true : args.draggable,
            content: fragment,
            beginButton: new Button({
               text: 'Cancel',
               press: this.onCancelPress.bind(this)
            }),
            endButton: new Button({
               text: 'Ok',
               enabled: "{= ${/fileName} !== '' }",
               press: this.onOkPress.bind(this)
            })
         });

         this.dialog.addStyleClass("sapUiSizeCompact");

         this.dialog.setModel(this.oModel);

         this.dialog.open();

         args.websocket.Send("FILEDIALOG:" + JSON.stringify([ this.kind, args.filename,  this.websocket.getChannelId().toString() ]));

         return this;
      },

      /** Press Ok button id Dialog, send selected file name and wait if confirmation required */
      onOkPress: function() {
         var fullname = this.getFullFileName();

         if (this.websocket)
            this.websocket.Send("DLGSELECT:" + JSON.stringify(fullname));
         else
            this.closeFileDialog();
      },

      /** Press Cancel button Dialog */
      onCancelPress: function() {
         if (this.websocket)
            this.websocket.Send("DLGNOSELECT");
         else
            this.closeFileDialog();
      },

      onBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;
      }
   });

   /** Function to initiate SaveAs dialog from client side
    * Following arguments has to be specified:
    * args.websocket - current available connection, used to send "FILEDIALOG:" request
    * args.filename - initial file name in the dialog
    * args.title - title used for the dialog
    * args.onOk - handler when file is selected and "Ok" button is pressed
    * args.onCancel - handler when "Cancel" button is pressed
    * args.onFailure - handler when any failure appears, dialog will be closed afterwards
    */
   FileDialog.SaveAs = function(args) {
      var controller = new FileDialog();

      return controller._initDialog("SaveAs", args);
   };

   FileDialog.NewFile = function(args) {
      var controller = new FileDialog();

      return controller._initDialog("NewFile", args);
   };

   FileDialog.OpenFile = function(args) {
      var controller = new FileDialog();

      return controller._initDialog("OpenFile", args);
   };


   return FileDialog;

});
