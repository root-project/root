sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/model/json/JSONModel',
               'sap/ui/core/Fragment',
               'sap/ui/model/Filter',
               'sap/ui/model/FilterOperator',
               'sap/m/MessageBox'
],function(Controller, JSONModel, Fragment, Filter, FilterOperator, MessageBox) {

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
            fTreeName: "",
            fExprX: "", fExprY: "", fExprZ: "", fExprCut: "", fOption: "",
            fNumber: 0, fFirst: 0, fStep: 1, fLargerStep: 2, fTreeEntries: 100,
            fBranches: [ { fName: "px", fTitle: "px branch" }, { fName: "py", fTitle: "py branch" }, { fName: "pz", fTitle: "pz branch" } ]
         };
         this.cfg_model = new JSONModel(this.cfg);
         this.getView().setModel(this.cfg_model);

         this.byId("treeViewerPage").setShowHeader(this._embeded);
         this.byId("quitButton").setVisible(!this._embeded);

      },

      onWebsocketOpened: function(/*handle*/) {
      },

      onWebsocketClosed: function() {
         // when connection closed, close panel as well
         if (window && !this._embeded) window.close();
      },

      /** Entry point for all data from server */
      onWebsocketMsg: function(handle, msg /*, offset */) {

         // binary data can be send only as addition to draw message
         // here data can be placed in the queue and processed when all other prerequicities are done
         if (typeof msg != "string")
            return console.error(`TreeViewer does not uses binary messages len = ${mgs.byteLength}`);

         let p = msg.indexOf(":"), mhdr;
         if (p > 0) {
            mhdr = msg.slice(0, p);
            msg = msg.slice(p+1);
         } else {
            mhdr = msg;
         }

         // console.log(mhdr, msg.length, msg.substr(0,70), "...");

         switch (mhdr) {
            case "CFG":   // generic viewer configuration
               this.setCfg(this.jsroot.parse(msg)); // use jsroot.parse while refs are used
               break;
            case "PROGRESS":
               this.showProgess(parseFloat(msg));
               break;
            default:
               console.error(`Non recognized msg ${mhdr} len = ${msg.length}`);
         }
      },

      /** @summary processing viewer configuration */
      setCfg: function(cfg) {

         this.last_cfg = cfg;

         Object.assign(this.cfg, cfg);

         this.cfg_model.refresh();
      },

      onBeforeRendering: function() {
      },

      onAfterRendering: function() {
      },

      onBranchHelpRequest: function(oEvent) {
         let sInputValue = oEvent.getSource().getValue(),
             oView = this.getView();

         this.branchInputId = oEvent.getSource().getId();

         if (!this._pValueHelpDialog) {
            this._pValueHelpDialog = Fragment.load({
               id: oView.getId(),
               name: 'rootui5.tree.view.BranchHelpDialog',
               controller: this
            }).then(oDialog => {
               oView.addDependent(oDialog);
               return oDialog;
            });
         }
         this._pValueHelpDialog.then(oDialog =>{
            // Create a filter for the binding
            oDialog.getBinding('items').filter([new Filter('fName', FilterOperator.Contains, '')]);
            // Open ValueHelpDialog filtered by the input's value
            oDialog.open('');
         });
      },

      onBranchHelpSearch: function (oEvent) {
         let sValue = oEvent.getParameter('value'),
             oFilter = new Filter('fName', FilterOperator.Contains, sValue);

         oEvent.getSource().getBinding('items').filter([oFilter]);
      },

      onBranchHelpClose: function (oEvent) {
         let oSelectedItem = oEvent.getParameter('selectedItem');
         oEvent.getSource().getBinding("items").filter([]);
         if (oSelectedItem && this.branchInputId) {
            let old = this.byId(this.branchInputId).getValue();
            if (old && old[old.length-1] != ' ') old += ' ';
            this.byId(this.branchInputId).setValue(old + oSelectedItem.getTitle());
         }
         delete this.branchInputId;
      },

      performDraw: function() {
         let send = this.last_cfg;

         if (!send) return;

         if (!this.cfg.fExprX)
            return MessageBox.error('X expression not specified');

         Object.assign(send, this.cfg);

         send.fBranches = [];
         if (!send.fNumber)
            send.fNumber = 0;

         if (!send.fFirst)
            send.fFirst = 0;

         this.websocket.send("DRAW:"+JSON.stringify(send));
      },

      showProgess: function(val) {
        let pr = this.byId("draw_progress");
        pr.setVisible(true);
        pr.setPercentValue(val);

        if (val >= 100.)
           setTimeout(() => pr.setVisible(false), 2000);
      },

      /** @summary Reload configuration */
      onRealoadPress: function () {
         this.doReload();
      },

      doReload: function() {
         if (!this.standalone)
            this.websocket.send("GETCFG");
      },

      /** Quit ROOT session */
      onQuitRootPress: function() {
         if (!this.standalone)
            this.websocket.send("QUIT_ROOT");
      }

   });

});
