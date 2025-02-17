sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/m/Text',
               'sap/ui/table/Column',
               'rootui5/geom/model/GeomBrowserModel',
               'rootui5/geom/lib/ColorBox',
               'sap/ui/Device',
               'sap/ui/unified/Menu',
               'sap/ui/unified/MenuItem',
               'sap/ui/core/Popup',
               'sap/ui/core/Icon',
               'sap/m/MessageToast',
               'sap/ui/layout/HorizontalLayout',
               'sap/m/CheckBox'
], function(Controller,
            mText,
            tableColumn,
            GeomBrowserModel,
            GeomColorBox,
            Device,
            Menu,
            MenuItem,
            Popup,
            Icon,
            MessageToast,
            HorizontalLayout,
            mCheckBox) {

   "use strict";

   /** @summary Central geometry viewer controller
    * @desc All TGeo functionality is loaded after main ui5 rendering is performed,
    * To start drawing, following stages should be completed:
    *    - ui5 element is rendered (onAfterRendering is called)
    *    - TGeo-related JSROOT functionality is loaded
    *    - RGeomDrawing object delivered from the server
    * Only after all this stages are completed, one could start to analyze  */

   return Controller.extend('rootui5.geom.controller.GeomHierarchy', {

      onInit() {
         let viewData = this.getView().getViewData();

         if (!viewData?.conn_handle?.getUserArgs('only_hierarchy')) return;

         // standalone running hierarchy only

         this.websocket = viewData.conn_handle;
         this.jsroot = viewData.jsroot;

         this.standalone = (this.websocket.kind == 'file');

         this.websocket.setReceiver(this);
         this.websocket.connect(viewData.conn_href);

         this._embeded = false;

         this.configureTable(true);
      },

      configure(args) {
         this.jsroot = args.jsroot;
         this.websocket = args.websocket; // special channel created from main connection
         this.viewer = args.viewer;
         this.standalone = args.standalone;
         this.websocket.setReceiver(this);
         this._embeded = true;

         this.configureTable(args.show_columns);
      },

      configureTable(show_columns) {

         // create model only for browser - no need for anybody else
         this.model = new GeomBrowserModel();

         this.model.useIndexSuffix = false;

         let t = this.byId("treeTable");

         t.setModel(this.model);

         t.setRowHeight(20);

         this.model.assignTreeTable(t);

         t.addColumn(new tableColumn('columnName', {
            label: 'Description',
            tooltip: 'Name of geometry nodes',
            autoResizable: true,
            width: show_columns ? '50%' : '100%',
            visible: true,
            tooltip: "{name}",
            template: new HorizontalLayout({
                  content: [
                     new Icon({ visible: '{top}', src: 'sap-icon://badge', tooltip: '{name} selected as top node' }).addStyleClass('sapUiTinyMarginEnd'),
                     new mText({ text: '{name}', tooltip: '{name}', wrapping: false })
                  ]
            })
         }));

         if (show_columns) {
            //new mCheckBox({ enabled: true, visible: true, selected: "{node_visible}", select: vis_selected_handler }),
            t.setColumnHeaderVisible(true);

            t.addColumn(new tableColumn('columnVis', {
               label: 'Visibility',
               tooltip: 'Visibility flags',
               autoResizable: true,
               visible: true,
               width: '20%',
               template: new HorizontalLayout({
                  content: [
                     new mCheckBox({ enabled: true, visible: true, selected: "{_node/visible}", select: evnt => this.changeVisibility(evnt), tooltip: '{name} logical node visibility' }),
                     new mCheckBox({ enabled: true, visible: true, selected: "{pvisible}", select: evnt => this.changeVisibility(evnt, true), tooltip: '{name} physical node visibility' })
                  ]
               })
            }));

            t.addColumn(new tableColumn('columnColor', {
               label: 'Color',
               tooltip: 'Color of geometry volumes',
               width: '10%',
               autoResizable: true,
               visible: true,
               template: new GeomColorBox({color: "{_node/color}", visible: "{= !!${_node/color}}"})
            }));
            t.addColumn(new tableColumn('columnMaterial', {
               label: 'Material',
               tooltip: 'Material of the volumes',
               width: '20%',
               autoResizable: true,
               visible: true,
               template: new mText({text: "{_node/material}", wrapping: false})
            }));
         }

         this._columnResized = 0;

         // catch re-rendering of the table to assign handlers
         t.addEventDelegate({
            onAfterRendering() {
               this.assignRowHandlers();
               if (this._columnResized < 1) return;
               this._columnResized = 0;
               let fullsz = 4;

               t.getColumns().forEach(col => {
                  if (col.getVisible()) fullsz += 4 + col.$().width();
               });

               this.viewer?.byId('geomViewerApp').getAggregation('_navMaster').setWidth(fullsz + 'px');
            }
         }, this);

         t.attachEvent("columnResize", {}, evnt => {
            this._columnResized++;
         }, this);

      },

      /** @summary invoked when visibility checkbox clicked */
      changeVisibility(oEvent, physical) {
         let row = oEvent.getSource(),
             flag = oEvent.getParameter('selected'),
             ctxt = row?.getBindingContext(),
             ttt = ctxt?.getProperty(ctxt?.getPath());

         if (!ttt?.path)
            return console.error('Fail to get path');

         if (this.standalone) {
            this.viewer?.changeNodeVisibilityOffline(ttt.path.join('/'), physical, flag);
         } else {
            let msg = '';

            if (physical) {
               msg = flag ? 'SHOW' : 'HIDE';
            } else {
               ttt.pvisible = flag;
               msg = "SETVI" + (flag ? '1' : '0');

               // all other rows referencing same node reset pvisible flag
               this.byId("treeTable").getRows().forEach(r => {
                  let r_ctxt = r.getBindingContext(),
                      r_ttt = r_ctxt?.getProperty(r_ctxt?.getPath());
                  if ((r_ttt?._node === ttt._node) && (r !== row))
                     r_ttt.pvisible = flag;
               });
            }

            msg += ':' + JSON.stringify(ttt.path);

            this.websocket.send(msg);
         }
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

         if (this.standalone) {
            this.viewer?.onRowHover(prop, is_enter);
         } else {
            let req = is_enter && prop?.path && prop?.isLeaf ? prop.path : [];
            return this.websocket.sendLast('hover', 200, 'HOVER:' + JSON.stringify(req));
         }
      },

      /** @summary Return nodeid for the row */
      getRowNodeId(row) {
         let ctxt = row.getBindingContext(),
             ttt = ctxt?.getProperty(ctxt?.getPath());

         return ttt?.id ?? -1;
      },

      /** @summary Return arrays of ids for this row  */
      getRowIds(row) {
         let ctxt = row.getBindingContext();
         if (!ctxt) return null;

         let path = ctxt.getPath(), lastpos = 0, ids = [];

         while (lastpos >= 0) {
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

      /** @summary Highlights row with specified path. Path is array of strings */
      highlighRowWithPath(path) {
         let rows = this.byId("treeTable").getRows(),
             best_cmp = 0, best_indx = 0,
             only_clear = (path[0] == '__OFF__');

         for (let i = 0; i < rows.length; ++i) {
            rows[i].$().css("background-color", "");
            if (!only_clear) {
               let ctxt = rows[i].getBindingContext(),
                   prop = ctxt?.getProperty(ctxt.getPath());

               if (prop?.path) {
                  let len = Math.min(prop.path.length, path.length), cmp = 0;
                  for (let i = 0; i < len; i++) {
                     if (prop.path[i] != path[i]) break;
                     cmp++;
                  }
                  if ((cmp == len) && (prop.path.length == path.length))
                     cmp = 1000; // full match

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
         if (window && !this._embeded) window.close();
         this.isConnected = false;
      },

      /** Entry point for all data from server */
      onWebsocketMsg(handle, msg /*, offset */) {

         // binary data can be send only as addition to draw message
         // here data can be placed in the queue and processed when all other prerequisites are done
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
         case "BREPL:":   // browser reply
            let bresp = JSON.parse(msg);

            let br = this.byId("treeTable")
            br?.setNoData("");
            br?.setShowNoData(false);

            this.model?.setNoData(false);
            this.model?.processResponse(bresp);
            break;
         case "FOUND:":  // text message for found query
            this.showTextInBrowser(msg);
            break;
         case "RELOAD":
            this.doReload(true);
            break;
         case "UPDATE":
            this.doReload(false);
            break;
         case "ACTIV:":
            this.activateInTreeTable(msg);
            break;
         case 'HIGHL:':
            this.highlighRowWithPath(JSON.parse(msg)); //
            break;
         case 'SETSR:':
            this.getView().byId("searchNode").setValue(msg);
            if (!msg)
               this.doReload(false);
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

         if (is_original) {
            let suffix = ':__SELECTED_STACK__:', p = msg.indexOf(suffix);
            if (p > 0) {
               this.selectedStack = JSON.parse(msg.slice(p+suffix.length));
               msg = msg.slice(0, p);
            } else {
               delete this.selectedStack;
            }

            suffix = ':__PHYSICAL_VISIBILITY__:';
            p = msg.indexOf(suffix);
            if (p > 0) {
               this.physVisibility = JSON.parse(msg.slice(p+suffix.length));
               msg = msg.slice(0, p);
            } else {
               delete this.physVisibility;
            }
         }

         let descr = JSON.parse(msg), br = this.byId("treeTable");

         br.setNoData("");
         br.setShowNoData(false);

         let topnode = this.model.buildTree(descr, is_original ? 1 : 999);
         this.model.hController = this;
         this.model.setFullModel(topnode);

         if (is_original) {
            this.fullModel = topnode;
            this.fullModelNodes = this.model.logicalNodes;
         }

      },

      /** @summary Returns stack by node path */
      getStackByPath(node, path) {
         if (!node || !path || path.length < 1 || (path[0] != node.name)) return null;
         let i = 1, stack = [];
         while (i < path.length) {
            let len = node.childs?.length ?? 0, found = false;
            for (let k = 0; k < len; ++k)
               if (node.childs[k].name == path[i]) {
                  stack.push(k);
                  node = node.childs[k];
                  i++;
                  found = true;
                  break;
               }
            if (!found) return null;
         }

         return stack;
      },

      /** @summary Get entry with physical node visibility */
      getPhysVisibilityEntry(path, force) {
         if ((!this.physVisibility && !force) || !this.fullModel)
            return;

         let stack = this.getStackByPath(this.fullModel, path);
         if (stack === null)
            return;

         let len = stack.length;

         for (let i = 0; i < this.physVisibility?.length; ++i) {
            let item = this.physVisibility[i], match = true;
            if (len != item.stack?.length) continue;
            for (let k = 0; match && (k < len); ++k)
               if (stack[k] != item.stack[k])
                  match = false;
            if (match)
               return item;
         }

         if (force) {
            if (!this.physVisibility)
               this.physVisibility = [];
            let item = { stack, visible: true };
            // TODO: one may add items in sort order to make it same as on server side
            this.physVisibility.push(item);
            return item;
         }
      },

      /** @summary Returns true if node identified by path is selected */
      getPhysTopNode(path) {
         if (!this.fullModel)
            return false;

         let stack = this.getStackByPath(this.fullModel, path);
         if (!stack)
            return !this.selectedStack;

         if (!this.selectedStack) return false;

         let len = this.selectedStack.length;
         if (len != stack.length)
            return false;

         for (let k = 0; k < len; ++k)
            if (stack[k] != this.selectedStack[k])
               return false;

         return true;
      },

      /** @summary Set top node by path */
      setPhysTopNode(path) {
         if (this.fullModel)
            this.selectedStack = this.getStackByPath(this.fullModel, path);
      },

      /** @summary Show special message instead of nodes hierarchy */
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
         let br = this.byId("treeTable"), nodes = [];
         for (let k = 0; k < matches.length; ++k)
            this.viewer?.appendStackToTree(nodes, matches[k].stack, matches[k].color, matches[k].material);

         br.setNoData("");
         br.setShowNoData(false);
         this.model.setFullModel(nodes[0]);
      },

      onAfterMasterOpen() {
      },

      /** @summary method called from geom painter when specific node need to be activated in the browser
       * @desc Due to complex indexing in TreeTable it is not trivial to select special node */
      activateInTreeTable(itemname) {
         if (!itemname || !this.model) return;

         let index = this.model.expandNodeByPath(itemname),
             tt = this.byId("treeTable");

         if ((index > 0) && tt) {
            tt.setFirstVisibleRow(Math.max(0, Math.round(index - tt.getVisibleRowCount()/2)));
            this.model.refresh(true);
         }
      },

      /** @summary Submit node search query to server, ignore in offline case */
      submitSearchQuery(query, from_handler, no_reload) {

         if (this.search_handler) {
             clearTimeout(this.search_handler);
             delete this.search_handler;
         }

         if (!from_handler) {
            // do not submit immediately, but after very short timeout
            // if user types very fast - only last selection will be shown
            this.search_handler = setTimeout(() => this.submitSearchQuery(query, true), 1000);
            return;
         }

         this.websocket.send('SEARCH:' + (query || ''));

         if(!query && !no_reload)
            this.doReload(false);
      },

      /** @summary when new query entered in the search field */
      onSearch(oEvt, direct) {

         let query = (typeof oEvt == 'string' && direct) ? oEvt : oEvt.getSource().getValue();

         if (!this.standalone) {
            this.submitSearchQuery(query);
         } else if (this.viewer) {
            let lst = this.viewer.findMatchesFromDraw(node => { return node.name.indexOf(query) == 0; });

            if (lst) {
               this.showFoundNodes(lst);
               this.viewer.paintFoundNodes(lst);
            } else {
               this.showTextInBrowser("No found nodes");
               this.viewer.paintFoundNodes(null);
            }
         }
      },

      /** @summary when click cell on tree table */
      onCellClick(oEvent) {
         let tt = this.byId("treeTable"),
             first = tt.getFirstVisibleRow() || 0,
             rowindx = oEvent.getParameters().rowIndex - first,
             row = (rowindx >= 0) ? tt.getRows()[rowindx] : null,
             ctxt = row?.getBindingContext(),
             prop = ctxt?.getProperty(ctxt.getPath());

         if (this.standalone) {
            this.viewer?.processInfoOffline(prop?.path, prop?.id);
         } else if (prop) {
            this.websocket.sendLast('click', 200, 'CLICK:' + JSON.stringify(prop.path));
         }
      },

      onCellContextMenu(oEvent) {
         if (Device.support.touch)
            return; //Do not use context menus on touch devices

         let ctxt = oEvent.getParameter('rowBindingContext'),
             colid = oEvent.getParameter('columnId'),
             prop = ctxt?.getProperty(ctxt.getPath());

         oEvent.preventDefault();

         if (!prop?._elem) return;

         if (!this._oIdContextMenu) {
            this._oIdContextMenu = new Menu();
            this.getView().addDependent(this._oIdContextMenu);
         }

         this._oIdContextMenu.destroyItems();
         if (!this.standalone)
            this._oIdContextMenu.addItem(new MenuItem({
               text: 'Set as top',
               select: () => {
                  this.setPhysTopNode(prop.path);
                  this.websocket.send('SETTOP:' + JSON.stringify(prop.path));

                  let len = this.model?.getLength() ?? 0;
                  for (let n = 0; n < len; ++n)
                     this.model?.setProperty(`/nodes/${n}/top`, false);
                  this.model?.setProperty(ctxt.getPath() + '/top', true);
               }
            }));

         let text = 'Search for ', value;
         if ((colid == 'columnMaterial') && prop._elem.material) {
            text += 'material';
            value = 'm:' + prop._elem.material;
         } else if ((colid == 'columnColor') && prop._elem.color) {
            text += 'color';
            value = 'c:' + prop._elem.color;
         } else {
            text += 'name';
            value = prop._elem.name;
         }

         this._oIdContextMenu.addItem(new MenuItem({
            text,
            select: () => {
               this.byId('searchNode').setValue(value);
               this.onSearch(value, true); // trigger search directly
            }
         }));

         this._oIdContextMenu.addItem(new MenuItem({
            startsSection: true,
            text: 'Show',
            tooltip: 'Show selected physical node',
            select: () => this.changeNodeVisibility('SHOW', prop.path)
         }));

         this._oIdContextMenu.addItem(new MenuItem({
            text: 'Hide',
            tooltip: 'Hide selected physical node',
            select: () => this.changeNodeVisibility('HIDE', prop.path)
         }));

         this._oIdContextMenu.addItem(new MenuItem({
            text: 'Reset',
            tooltip: 'Reset node show/hide settings',
            select: () => this.changeNodeVisibility('CLEAR', prop.path)
         }));

         this._oIdContextMenu.addItem(new MenuItem({
            text: 'Reset all',
            tooltip: 'Reset all individual show/hide settings',
            select: () => this.changeNodeVisibility('CLEARALL')
         }));

         //Open the menu on the cell
         let oCellDomRef = oEvent.getParameter("cellDomRef");
         this._oIdContextMenu.open(false, oCellDomRef, Popup.Dock.BeginTop, Popup.Dock.BeginBottom, oCellDomRef, "none none");
      },

      changeNodeVisibility(cmd, path) {
         if (this.standalone) {
            MessageToast.show('Change of node visibility in standalone mode not yet supported');
         } else {
            if (path !== undefined) cmd += ':' + JSON.stringify(path);
            this.websocket.send(cmd);
         }
      },

      /** @summary Reload geometry description and base drawing, normally not required */
      onRealoadPress() {
         this.doReload(true);
      },

      doReload(force) {
         // reassign logical nodes, used together with full model
         if (this.fullModelNodes && this.model)
            this.model.logicalNodes = this.fullModelNodes;

         if (this.standalone) {
            this.showTextInBrowser();
            this.model?.setFullModel(this.fullModel);
            this.viewer?.paintFoundNodes(null);
         } else {
            this.model?.clearFullModel();
            this.model?.reloadMainModel(force, true);

            let srch = this.byId('searchNode');
            if (srch.getValue()) {
               srch.setValue('');
               this.submitSearchQuery('', true, true);
            }
         }
      }

   });

});
