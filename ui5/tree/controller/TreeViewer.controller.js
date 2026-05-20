sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/model/json/JSONModel',
               'sap/ui/core/Fragment',
               'sap/ui/model/Filter',
               'sap/ui/model/FilterOperator',
               'sap/m/MessageBox'
],function(Controller, JSONModel, Fragment, Filter, FilterOperator, MessageBox) {

   'use strict';

   /** Tree viewer contoller
     * All TTree functionality is loaded after main ui5 rendering is performed */

   return Controller.extend('rootui5.tree.controller.TreeViewer', {
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
         this.standalone = this.websocket.isStandalone();

         if (!this.standalone && !this._embeded && this.websocket.addReloadKeyHandler)
            this.websocket.addReloadKeyHandler();

         this.cfg = {
            fTreeName: '',
            fExprX: '', fExprY: '', fExprZ: '', fExprCut: '', fOption: '',
            fNumber: 0, fFirst: 0, fStep: 1, fLargerStep: 2, fTreeEntries: 100,
            fBranches: []
         };
         this.cfg_model = new JSONModel(this.cfg);
         this.getView().setModel(this.cfg_model);

         this.byId('treeViewerPage').setShowHeader(this._embeded);
         this.byId('quitButton').setVisible(!this._embeded);

      },

      onWebsocketOpened: function(/*handle*/) {
      },

      onWebsocketClosed: function() {
         // when connection closed, close panel as well

         console.log('web socket closed', this._embeded);

         // if (window && !this._embeded) window.close();
      },

      /** Entry point for all data from server */
      onWebsocketMsg: function(handle, msg /*, offset */) {

         // binary data can be send only as addition to draw message
         // here data can be placed in the queue and processed when all other prerequicities are done
         if (typeof msg != 'string')
            return console.error(`TreeViewer does not uses binary messages len = ${mgs.byteLength}`);

         let p = msg.indexOf(':'), mhdr;
         if (p > 0) {
            mhdr = msg.slice(0, p);
            msg = msg.slice(p+1);
         } else {
            mhdr = msg;
         }

         // console.log(mhdr, msg.length, msg.substr(0,70), '...');

         switch (mhdr) {
            case 'CFG':   // generic viewer configuration
               this.setCfg(this.jsroot.parse(msg)); // use jsroot.parse while refs are used
               break;
            case 'PROGRESS':
               this.showProgess(parseFloat(msg));
               break;
            case 'SUGGEST':
               this.showSuggestedItem(msg);
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

         const inputs = ['input_x','input_y','input_z', 'input_cut'];

         inputs.forEach(id => {
            this.byId(id).$().find('input').focus(() => { this.lastFocus = id; });
         });
      },

      onBranchHelpRequest: function(oEvent) {

         this.branchInputId = oEvent.getSource().getId();

         if (!this._pValueHelpDialog) {
            this._pValueHelpDialog = Fragment.load({
               id: this.getView().getId(),
               name: 'rootui5.tree.view.BranchHelpDialog',
               controller: this
            }).then(oDialog => {
               this.getView().addDependent(oDialog);
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
         oEvent.getSource().getBinding('items').filter([]);
         if (oSelectedItem && this.branchInputId) {
            let old = this.byId(this.branchInputId).getValue();
            if (old &&  /[a-zA-Z\[\]._]/g.test(old[old.length-1])) old += ' + ';
            this.byId(this.branchInputId).setValue(old + oSelectedItem.getTitle());
         }
         delete this.branchInputId;
      },

      onPressClearBtn: function(oEvent) {
         let id = oEvent.getSource().getId();
         if (id.indexOf('clear_x') >= 0) {
            this.lastFocus = 'input_x';
            this.cfg.fExprX = '';
         } else if (id.indexOf('clear_y') >= 0) {
            this.cfg.fExprY = '';
            this.lastFocus = 'input_y';
         } else if (id.indexOf('clear_z') >= 0) {
            this.cfg.fExprZ = '';
            this.lastFocus = 'input_z';
         } else if (id.indexOf('clear_cut') >= 0) {
            this.cfg.fExprCut = '';
            this.lastFocus = 'input_cut';
         }

         this.cfg_model.refresh();
      },

      onPressClearWidget: function() {
         this.cfg.fExprX = this.cfg.fExprY = this.cfg.fExprZ = this.cfg.fExprCut = '';
         delete this.lastFocus;
         this.cfg_model.refresh();
      },

      onPressPerformDraw: function() {
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

         this.websocket.send('DRAW:'+JSON.stringify(send));
      },

      showProgess: function(val) {
        let pr = this.byId('draw_progress');
        pr.setVisible(true);
        pr.setPercentValue(val);

        if (val >= 100.)
           setTimeout(() => pr.setVisible(false), 2000);
      },

      /** @summary Show suggested by server branch/leaf name */
      showSuggestedItem: function(name) {
         let id = this.lastFocus || 'input_x',
             val = this.byId(id).getValue();

         this.byId(id).setValue(val ? val + ' + ' + name : name);
      },

      /** @summary Reload configuration */
      onRealoadPress: function () {
         this.doReload();
      },

      doReload: function() {
         if (!this.standalone)
            this.websocket.send('GETCFG');
      },

      /** Quit ROOT session */
      onQuitRootPress: function() {
         if (!this.standalone)
            this.websocket.send('QUIT_ROOT');
      }

   });

});
