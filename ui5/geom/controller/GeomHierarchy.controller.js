sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/m/Text',
               'sap/ui/table/Column',
               'rootui5/geom/model/GeomBrowserModel',
               'rootui5/geom/lib/ColorBox',
               'sap/ui/Device',
               'sap/ui/unified/Menu',
               'sap/ui/unified/MenuItem',
               'sap/ui/core/Popup',
               'sap/m/MessageToast'
], function(Controller,
            mText,
            tableColumn,
            GeomBrowserModel,
            GeomColorBox,
            Device,
            Menu,
            MenuItem,
            Popup,
            MessageToast) {

   "use strict";

   /** @summary Central geometry viewer contoller
    * @desc All TGeo functionality is loaded after main ui5 rendering is performed,
    * To start drawing, following stages should be completed:
    *    - ui5 element is rendered (onAfterRendering is called)
    *    - TGeo-related JSROOT functionality is loaded
    *    - RGeomDrawing object delivered from the server
    * Only after all this stages are completed, one could start to analyze  */

   return Controller.extend("rootui5.geom.controller.GeomHierarchy", {

      onInit: function () {

      },

      configure: function(args) {

         this.jsroot = args.jsroot;

         this.websocket = args.websocket; // special channel created from main conection

         this.viewer = args.viewer;

         this.websocket.setReceiver(this);

         // create model only for browser - no need for anybody else
         this.model = new GeomBrowserModel();

         this.model.useIndexSuffix = false;

         let t = this.byId("treeTable");

         t.setModel(this.model);

         t.setRowHeight(20);

         // let vis_selected_handler = this.visibilitySelected.bind(this);

         this.model.assignTreeTable(t);

         t.addColumn(new tableColumn('columnName', {
            label: 'Description',
            tooltip: 'Name of geometry nodes',
            template: new mText({text: "{name}", wrapping: false})
         }));

         if (args.show_columns) {
            //new mCheckBox({ enabled: true, visible: true, selected: "{node_visible}", select: vis_selected_handler }),
            t.setColumnHeaderVisible(true);
            t.addColumn(new tableColumn('columnColor', {
               label: 'Color',
               tooltip: 'Color of geometry volumes',
               width: '2rem',
               template: new GeomColorBox({color: "{_elem/color}", visible: "{= !!${_elem/color}}"})
            }));
            t.addColumn(new tableColumn('columnMaterial', {
               label: 'Material',
               tooltip: 'Material of the volumes',
               width: '6rem',
               template: new mText({text: "{_elem/material}", wrapping: false})
            }));
         }

         // catch re-rendering of the table to assign handlers
         t.addEventDelegate({
            onAfterRendering: function() { this.assignRowHandlers(); }
         }, this);

         this.model.setProperty('/hasViewer', this.viewer ? true : false);
      },

      /** @summary invoked when visibility checkbox clicked */
      visibilitySelected: function(oEvent) {
         let nodeid = this.getRowNodeId(oEvent.getSource());
         if (nodeid < 0) {
            console.error('Fail to identify nodeid');
            return;
         }

         let msg = "SETVI" + (oEvent.getParameter("selected") ? "1:" : "0:") + JSON.stringify(nodeid);

         // send info message to client to change visibility
         this.websocket.send(msg);
      },

      assignRowHandlers() {
         let rows = this.byId("treeTable").getRows();
         for (let k = 0; k < rows.length; ++k)
            rows[k].$().hover(this.onRowHover.bind(this, rows[k], true), this.onRowHover.bind(this, rows[k], false));
      },

      /** @brief Handler for mouse-hover event
       * @desc Used to highlight correspondent volume on geometry drawing */
      onRowHover(row, is_enter) {
         // property of current entry, not used now
         let ctxt = row.getBindingContext(),
             prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;

         this.viewer?.onRowHover(prop, is_enter);
      },

      /** @summary Return nodeid for the row */
      getRowNodeId(row) {
         let ctxt = row.getBindingContext(),
             ttt = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;
         return ttt && (ttt.id !== undefined) ? ttt.id : -1;
      },

      /** @summary Return arrys of ids for this row  */
      getRowIds(row) {
         let ctxt = row.getBindingContext();
         if (!ctxt) return null;

         let path = ctxt.getPath(), lastpos = 0, ids = [];

         while (lastpos>=0) {
            lastpos = path.indexOf("/childs", lastpos+1);

            let ttt = ctxt.getProperty(path.substr(0,lastpos));

            if (ttt?.id === undefined) {
               // it is not an error - sometime TableTree does not have displayed items
               // console.error('Fail to extract node id for path ' + path.substr(0,lastpos) + ' full path ' + ctxt.getPath());
               return null;
            }

            ids.push(ttt.id);
         }
         return ids;
      },

      /** @summary try to produce stack out of row path */
      getRowStack(row) {
         let ids = this.getRowIds(row);
         return ids ? this.viewer?.geo_clones.buildStackByIds(ids) : null;
      },

      /** @summary compare two paths to verify that both are the same
        * @returns 1000 if both are equivalent or maximal match length */
      comparePaths(path1, path2) {
         if (!path1) path1 = [];
         if (!path2) path2 = [];
         let len = Math.min(path1.length, path2.length);
         for (let i = 0; i < len; i++)
            if (path1[i] != path2[i])
               return i-1;

         return path1.length == path2.length ? 1000 : len;
      },

      /** @summary Highlights row with specified path */
      highlighRowWithPath(path) {
         let rows = this.byId("treeTable").getRows(), best_cmp = 0, best_indx = 0;

         for (let i = 0; i < rows.length; ++i) {
            rows[i].$().css("background-color", "");
            if (path && (path[0] != "OFF")) {
               let ctxt = rows[i].getBindingContext(),
                   prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;

               if (prop && prop.path) {
                  let cmp = this.comparePaths(prop.path, path);
                  if (cmp > best_cmp) { best_cmp = cmp; best_indx = i; }
               }
            }
         }

         if (best_cmp > 0)
            rows[best_indx].$().css("background-color", best_cmp == 1000 ? "yellow" : "lightgrey");
      },

      onWebsocketOpened( /*handle*/) {
         this.isConnected = true;
         this.model.sendFirstRequest(this.websocket);
      },

      onWebsocketClosed() {
         // when connection closed, close panel as well
         this.isConnected = false;
      },

      /** Entry point for all data from server */
      onWebsocketMsg(handle, msg /*, offset */) {

         // binary data can be send only as addition to draw message
         // here data can be placed in the queue and processed when all other prerequicities are done
         if (typeof msg != "string")
            return console.error("Geom hierarchy not uses binary messages len = " + mgs.byteLength);

         let mhdr = msg.slice(0,6);
         msg = msg.slice(6);

         // console.log(`RECV ${mhdr} len: ${msg.length} ${msg.slice(0,70)} ...`);

         switch (mhdr) {
         case "DESCR:":  // browser hierarchy
            this.parseDescription(msg, true);
            break;
         case "FESCR:":  // searching hierarchy
            this.parseDescription(msg, false);
            break;
         case "FDRAW:": // drawing of found nodes
            this.viewer?.checkDrawMsg("found", this.jsroot.parse(msg));
            break;
         case "BREPL:":   // browser reply
            if (this.model)
               this.model.processResponse(JSON.parse(msg));
            break;
         case "FOUND:":  // text message for found query
            this.showTextInBrowser(msg);
            this.viewer?.paintFoundNodes(null); // nothing can be shown
            break;
         case "MODIF:":
            this.modifyDescription(msg);
            break;
         default:
            console.error(`Non recognized msg ${mhdr} len=${msg.length}`);
         }
      },

      /** @summary Parse compact geometry description
       * @desc Used only to initialize hierarchy browser with full Tree,
       * later should be done differently */
      parseDescription(msg, is_original) {

         if (!this.model) return;

         let descr = JSON.parse(msg), br = this.byId("treeTable");

         br.setNoData("");
         br.setShowNoData(false);

         let topnode = this.model.buildTree(descr, is_original ? 1 : 999);
         if (this.standalone)
            this.fullModel = topnode;

         this.model.setFullModel(topnode);
      },

      /** TO BE CHANGED !!! When single node element is modified on the server side */
      modifyDescription(msg) {
         let arr = JSON.parse(msg);
         if (!arr || !this.viewer?.geo_clones) return;

         console.error('modifyDescription should be changed');
/*
         let can_refresh = true;

         for (let k=0;k<arr.length;++k) {
            let moditem = arr[k];

            this.geo.ClonedNodes.formatServerElement(moditem);

            let item = this.geo_clones.nodes[moditem.id];

            if (!item)
               return console.error('Fail to find item ' + moditem.id);

            item.vis = moditem.vis;
            item.matrix = moditem.matrix;

            let dnode = this.originalCache ? this.originalCache[moditem.id] : null;

            if (dnode) {
               // here we can modify only node which was changed
               dnode.title = moditem.name;
               dnode.node_visible = moditem.vis != 0;
            } else {
               can_refresh = false;
            }

            if (!moditem.vis && this.geo_painter)
               this.geo_painter.removeDrawnNode(moditem.id);
         }

         if (can_refresh) {
            this.model.refresh();
         } else {
            // rebuild complete tree for TreeBrowser
         }
*/
      },

      /** @summary Show special message insted of nodes hierarchy */
      showTextInBrowser(text) {
         let br = this.byId("treeTable");
         br.collapseAll();
         if (!text || (text === "RESET")) {
            br.setNoData("");
            br.setShowNoData(false);

            this.model.setNoData(false);
            this.model.refresh();

         } else {
            br.setNoData(text);
            br.setShowNoData(true);
            this.model.setNoData(true);
            this.model.refresh();
         }
      },

      /** @summary Show found nodes in the browser, used for offline */
      showFoundNodes(matches) {

         if (!this.viewer) return;

         let br = this.byId("treeTable"), nodes = [];
         for (let k = 0; k < matches.length; ++k)
            this.viewer.appendStackToTree(nodes, matches[k].stack, matches[k].color, matches[k].material);

         br.setNoData("");
         br.setShowNoData(false);
         this.model.setFullModel(nodes[0]);
      },

      onAfterMasterOpen() {
      },

      /** @summary method called from geom painter when specific node need to be activated in the browser
       * @desc Due to complex indexing in TreeTable it is not trivial to select special node */
      activateInTreeTable(itemnames, force) {

         if (!force || !itemnames || !this.model) return;

         let index = this.model.expandNodeByPath(itemnames[0]),
             tt = this.byId("treeTable");

         if ((index > 0) && tt) {
            tt.setFirstVisibleRow(Math.max(0, Math.round(index - tt.getVisibleRowCount()/2)));
            this.model.refresh(true);
         }
      },

      /** @summary Submit node search query to server, ignore in offline case */
      submitSearchQuery(query, from_handler) {

         if (!from_handler) {
            // do not submit immediately, but after very short timeout
            // if user types very fast - only last selection will be shown
            if (this.search_handler) clearTimeout(this.search_handler);
            this.search_handler = setTimeout(() => this.submitSearchQuery(query, true), 1000);
            return;
         }

         delete this.search_handler;

         if(query) {
            this.websocket.send("SEARCH:" + (query || ""));
         } else {
            this.viewer?.paintFoundNodes(null); // remove all search results
            this.doReload(false);
         }
      },

      /** @summary when new query entered in the seach field */
      onSearch(oEvt, direct) {
         let query = (typeof oEvt == 'string' && direct) ? oEvt : oEvt.getSource().getValue();
         if (!this.standalone) {
            this.submitSearchQuery(query);
         } else if (this.viewer) {
            let lst = this.viewer.findMatchesFromDraw(node => { return node.name.indexOf(query) == 0; });

            if (lst?.length) {
               this.showFoundNodes(lst);
               this.viewer?.paintFoundNodes(lst);
            } else {
               this.showTextInBrowser("No found nodes");
               this.viewer?.paintFoundNodes(null);
            }
         }
      },

      onCellClick(oEvent) {

         let tt = this.byId("treeTable"),
             first = tt.getFirstVisibleRow() || 0,
             rowindx = oEvent.getParameters().rowIndex - first,
             row = (rowindx >= 0) ? tt.getRows()[rowindx] : null,
             ctxt = row?.getBindingContext(),
             prop = ctxt?.getProperty(ctxt.getPath());

         if(prop && this.viewer?.isInfoPageActive())
            if (this.standalone) {
               this.viewer?.processInfoOffline(prop.path, prop.id);
            } else {
               this.viewer?.sendViewerRequest("INFO", { path: prop.path });
            }
      },

      onCellContextMenu(oEvent) {
         if (Device.support.touch) {
            return; //Do not use context menus on touch devices
         }

         let ctxt = oEvent.getParameter("rowBindingContext"),
             prop = ctxt?.getProperty(ctxt.getPath());

         //if (oEvent.getParameter("columnId") != this.getView().createId("productId")) {
         //   return; //Custom context menu for product id column only
         //}

         oEvent.preventDefault();

         // var oRowContext = oEvent.getParameter("rowBindingContext");

         if (!this._oIdContextMenu) {
            this._oIdContextMenu = new Menu();
            this.getView().addDependent(this._oIdContextMenu);
         }

         this._oIdContextMenu.destroyItems();
         this._oIdContextMenu.addItem(new MenuItem({
            text: 'Set as top',
            select: () => {
               if (this.standalone) {
                  MessageToast.show('Set as top not supported in standalone mode');
               } else {
                  this.viewer?.sendViewerRequest('SETTOP', { path: prop.path });
               }
            }
         }));

         let colid = oEvent.getParameter('columnId');

         this._oIdContextMenu.addItem(new MenuItem({
            text: 'Search for',
            select: () => {
               let value = prop._elem.name;
               if ((colid == 'columnMaterial') && prop._elem.material)
                  value = 'm:' + prop._elem.material;
               else if ((colid == 'columnColor') && prop._elem.color)
                  value = 'c:' + prop._elem.color;
               this.byId('searchNode').setValue(value);
               this.onSearch(value, true); // trigger search directly
            }
         }));

         //Open the menu on the cell
         let oCellDomRef = oEvent.getParameter("cellDomRef");
         this._oIdContextMenu.open(false, oCellDomRef, Popup.Dock.BeginTop, Popup.Dock.BeginBottom, oCellDomRef, "none none");
      },

      /** @summary Reload geometry description and base drawing, normally not required */
      onRealoadPress() {
         this.doReload(true);
      },

      doReload(force, only_this) {
         if (this.standalone) {
            this.showTextInBrowser();
            if (this.model)
               this.model.setFullModel(this.fullModel);
         } else if (this.model) {
            this.model.clearFullModel();
            this.model.reloadMainModel(force);
         }

         if (!only_this)
            this.viewer?.doReload(force, true);
      },

      onInfoPress() {
         this.viewer?.onInfoPress();
      }

   });

});
