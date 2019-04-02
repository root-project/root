sap.ui.define(['sap/ui/core/Component',
               'sap/ui/core/mvc/Controller',
               'sap/ui/core/Control',
               'sap/ui/model/json/JSONModel',
               'sap/m/Text',
               'sap/m/CheckBox',
               'sap/ui/layout/Splitter',
               "sap/ui/core/ResizeHandler",
               "sap/ui/layout/HorizontalLayout",
               "sap/ui/table/Column",
               "rootui5/geom/model/BrowserModel"
],function(Component, Controller, CoreControl, JSONModel, mText, mCheckBox, Splitter, ResizeHandler,
           HorizontalLayout, tableColumn, BrowserModel) {

   "use strict";

   var geomColorBox = CoreControl.extend("rootui5.geom.controller.ColorBox", { // call the new Control type "my.ColorBox" and let it inherit from sap.ui.core.Control

      // the control API:
      metadata : {
         properties : {           // setter and getter are created behind the scenes, incl. data binding and type validation
            "color" : {type: "sap.ui.core.CSSColor", defaultValue: "#fff"} // you can give a default value and more
         }
      },

      // the part creating the HTML:
      renderer : function(oRm, oControl) { // static function, so use the given "oControl" instance instead of "this" in the renderer function
         // if (!oControl.getVisible()) return;

         oRm.write("<div");
         oRm.writeControlData(oControl);  // writes the Control ID and enables event handling - important!
         oRm.addStyle("background-color", oControl.getColor());  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.writeStyles();
         oRm.addClass("geomColorBox");      // add a CSS class for styles common to all control instances
         oRm.writeClasses();              // this call writes the above class plus enables support for Square.addStyleClass(...)
         oRm.write(">");
         oRm.write("</div>"); // no text content to render; close the tag
      }

/*
      // an event handler:
      onclick : function(evt) {   // is called when the Control's area is clicked - no further event registration required
         sap.ui.require([
            'sap/ui/unified/ColorPickerPopover'
         ], function (ColorPickerPopover) {
            if (!this.oColorPickerPopover) {
               this.oColorPickerPopover = new ColorPickerPopover({
                  change: this.handleChange.bind(this)
               });
            }
            this.oColorPickerPopover.setColorString(this.getColor());
            this.oColorPickerPopover.openBy(this);
         }.bind(this));
      },

      handleChange: function (oEvent) {
         var newColor = oEvent.getParameter("colorString");
         this.setColor(newColor);
         // TODO: fire a "change" event, in case the application needs to react explicitly when the color has changed
         // but when the color is bound via data binding, it will be updated also without this event
      }
*/

   });

   CoreControl.extend("eve.GeomDraw", {

      metadata : {
         properties : {           // setter and getter are created behind the scenes, incl. data binding and type validation
            "color" : {type: "sap.ui.core.CSSColor", defaultValue: "#fff"} // you can give a default value and more
         }
      },

      // the part creating the HTML:
      renderer : function(oRm, oControl) { // static function, so use the given "oControl" instance instead of "this" in the renderer function
         oRm.write("<div");
         oRm.writeControlData(oControl);  // writes the Control ID and enables event handling - important!
         // oRm.addStyle("background-color", oControl.getColor());  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("width", "100%");
         oRm.addStyle("height", "100%");
         oRm.addStyle("overflow", "hidden");
         oRm.writeStyles();
         oRm.writeClasses();              // this call writes the above class plus enables support for Square.addStyleClass(...)
         oRm.write(">");
         oRm.write("</div>"); // no text content to render; close the tag
      },

      onAfterRendering: function() {
         ResizeHandler.register(this, this.onResize.bind(this));
         this.geom_painter = null;
      },

      onResize: function() {
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 100); // minimal latency
      },

      onResizeTimeout: function() {
         delete this.resize_tmout;
         if (this.geo_painter)
            this.geo_painter.CheckResize();
      }
   });


   /** Central geometry viewer contoller
    * All TGeo functionality is loaded after main ui5 rendering is performed,
    * To start drawing, following stages should be completed:
    *    - ui5 element is rendered (onAfterRendering is called)
    *    - TGeo-related JSROOT functionality is loaded
    *    - REveGeomDrawing object delivered from the server
    * Only after all this stages are completed, one could start to analyze  */

   return Controller.extend("rootui5.geom.controller.GeomViewer", {
      onInit: function () {

         // this.websocket = this.getView().getViewData().conn_handle;

         this.websocket = Component.getOwnerComponentFor(this.getView()).getComponentData().conn_handle;

         this.websocket.SetReceiver(this);
         this.websocket.Connect();

         this.queue = []; // received draw messages

         // if true, most operations are performed locally without involving server
         this.standalone = this.websocket.kind == "file";

         this.data = { nodes: null };

         this.plainModel = false;

         if (this.plainModel) {
            this.model = new JSONModel(this.data);
            this.getView().setModel(this.model);
         } else {
            this.model = this.getView().getModel();
            console.log('USE PROVIDED MODEL', typeof this.model.sendFirstRequest)
         }


         // PART 2: instantiate Control and place it onto the page

         if (JSROOT.GetUrlOption('nobrowser') !== null) {
            // remove complete area - plain geometry drawing
            this.getView().byId("mainSplitter").removeAllContentAreas();
         } else {

            var t = this.getView().byId("treeTable");

            if (this.plainModel) {

               var vis_selected_handler = this.visibilitySelected.bind(this);

               t.addColumn(new tableColumn({
                  label: "Description",
                  template: new HorizontalLayout({
                     content: [
                        new mCheckBox({ enabled: true, visible: true, selected: "{node_visible}", select: vis_selected_handler }),
                        new geomColorBox({color:"{color}", visible: "{color_visible}" }),
                        new mText({text:"{title}", wrapping: false })
                     ]
                  })
                }));

            } else {
              /* t.addColumn(new tableColumn({
                  label: "Description",
                  template: new HorizontalLayout({
                     content: [
                        new mText({text:"{name}", wrapping: false })
                     ]
                  })
                }));
                */
            }

            // catch re-rendering of the table to assign handlers
            t.addEventDelegate({
               onAfterRendering: function() { this.assignRowHandlers(); }
            }, this);

         }

         // placeholder for geometry painter
         this.geomControl = new eve.GeomDraw({color:"#f00"});
         this.getView().byId("mainSplitter").addContentArea(this.geomControl);

         JSROOT.AssertPrerequisites("geom", function() {
            sap.ui.define(['rootui5/eve7/lib/EveElements'], function(EveElements) {
               this.creator = new EveElements();
               this.creator.useIndexAsIs = (JSROOT.GetUrlOption('useindx') !== null);
               this.checkRequestMsg();
            }.bind(this));
         }.bind(this));
      },

      /** invoked when visibility checkbox clicked */
      visibilitySelected: function(oEvent) {
         var nodeid = this.getRowNodeId(oEvent.getSource());
         if (nodeid<0) {
            console.error('Fail to identify nodeid');
            return;
         }

         var msg = "SETVI" + (oEvent.getParameter("selected") ? "1:" : "0:") + JSON.stringify(nodeid);

         // send info message to client to change visibility
         this.websocket.Send(msg);
      },

      assignRowHandlers: function() {
         var rows = this.getView().byId("treeTable").getRows();
         for (var k=0;k<rows.length;++k) {
            rows[k].$().hover(this.onRowHover.bind(this, rows[k], true), this.onRowHover.bind(this, rows[k], false));
         }
      },

      /** @brief function called then mouse-hover event over the row is invoked
       * @desc Used to highlight correspondent volume on geometry drawing */
      onRowHover: function(row, is_enter) {
         // property of current entry, not used now
         var ctxt = row.getBindingContext(),
             prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;

         var ids = (is_enter && prop && prop.end_node) ? this.getRowIds(row) : null;

         if (!this.standalone) {
            var req = ids ? JSON.stringify(ids) : "OFF";
            // avoid multiple time submitting same request
            if (this._last_hover_req === req) return;
            this._last_hover_req = req;
            return this.websocket.Send("HOVER:" + req);
         }

         if (this.geo_painter && this.geo_clones) {
            // remember current element with hover stack
            this._hover_stack = this.geo_clones.MakeStackByIds(ids);

            this.geo_painter.HighlightMesh(null, 0x00ff00, null, undefined, this._hover_stack, true);
         }
      },

      /** Return nodeid for the row */
      getRowNodeId: function(row) {
         var ctxt = row.getBindingContext();
         var ttt = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;
         return ttt && (ttt.id!==undefined) ? ttt.id : -1;
      },

      /** Return arrys of ids for this row  */
      getRowIds: function(row) {
         var ctxt = row.getBindingContext();
         if (!ctxt) return null;

         var path = ctxt.getPath(), lastpos = 0, ids = [];

         while (lastpos>=0) {
            lastpos = path.indexOf("/chlds", lastpos+1);

            var ttt = ctxt.getProperty(path.substr(0,lastpos));

            if (!ttt || (ttt.id===undefined)) {
               // it is not an error - sometime TableTree does not have displayed items
               // console.error('Fail to extract node id for path ' + path.substr(0,lastpos) + ' full path ' + ctxt.getPath());
               return null;
            }

            ids.push(ttt.id);
         }
         return ids;
      },

      /** try to produce stack out of row path */
      getRowStack: function(row) {
         var ids = this.getRowIds(row);
         return ids ? this.geo_clones.MakeStackByIds(ids) : null;
      },

      /** Callback from geo painter when mesh object is highlighted. Use for update of TreeTable */
      HighlightMesh: function(active_mesh, color, geo_object, geo_index, geo_stack) {
         if (!this.standalone) {
            var req = geo_stack ? JSON.stringify(geo_stack) : "OFF";
            // avoid multiple time submitting same request
            if (this._last_highlight_req === req) return;
            this._last_highlight_req = req;
            return this.websocket.Send("HIGHL:" + req);
         }

         if (this.geo_clones) {
            var ids = this.geo_clones.MakeIdsByStack(geo_stack);
            return this.highlighRowWithIds(ids);
         }

         // old code, can be removed soon

         var rows = this.getView().byId("treeTable").getRows(), best_cmp = 0, best_indx = 0;

         for (var i=0;i<rows.length;++i) {
            rows[i].$().css("background-color", "");
            if (geo_stack) {
               var cmp = JSROOT.GEO.CompareStacks(geo_stack, this.getRowStack(rows[i]));
               if (cmp > best_cmp) { best_cmp = cmp; best_indx = i; }
            }
         }

         if (best_cmp > 0)
            rows[best_indx].$().css("background-color", best_cmp == geo_stack.length ? "yellow" : "lightgrey");
      },

      /** compare row ids which id to be selected */
      compareIds: function(rowids, hids) {
         if (!rowids || !hids) return 0;
         var len = Math.min(rowids.length, hids.length);
         for (var k=0;k<len;++k)
            if (rowids[k] !== hids[k])
               return k-1;
         return len;
      },

      highlighRowWithIds: function(hids) {
         var rows = this.getView().byId("treeTable").getRows(), best_cmp = 0, best_indx = 0;

         for (var i=0;i<rows.length;++i) {
            rows[i].$().css("background-color", "");
            if (hids) {
               var rowids = this.getRowIds(rows[i]);
               var cmp = this.compareIds(rowids, hids);
               if (cmp > best_cmp) { best_cmp = cmp; best_indx = i; }
            }
         }

         if (best_cmp > 0)
            rows[best_indx].$().css("background-color", best_cmp == hids.length ? "yellow" : "lightgrey");
      },

      createGeoPainter: function(drawopt) {
         if (this.geo_painter) return;

         this.geo_painter = JSROOT.Painter.CreateGeoPainter(this.geomControl.getDomRef(), null, drawopt);
         this.geomControl.geo_painter = this.geo_painter;

         this.geo_painter.AddHighlightHandler(this);
         this.geo_painter.ActivateInBrowser = this.activateInTreeTable.bind(this);

         this.geo_painter.assignClones(this.geo_clones);
      },

      /** Extract shapes from binary data using appropriate draw message
       * Draw message is vector of REveGeomVisible objects, including info where shape is in raw data */
      extractRawShapes: function(draw_msg) {

         var nodes = null;

         // array for descriptors for each node
         // if array too large (>1M), use JS object while only ~1K nodes are expected to be used
         if (!this.geo_clones) {
            if (draw_msg.kind !== "draw") return false;
            nodes = (draw_msg.numnodes > 1e6) ? { length: draw_msg.numnodes } : new Array(draw_msg.numnodes); // array for all nodes
         }

         for (var cnt=0;cnt < draw_msg.nodes.length; ++cnt) {
            var node = draw_msg.nodes[cnt];
            this.formatNodeElement(node);
            if (nodes)
               nodes[node.id] = node;
            else
               this.geo_clones.updateNode(node);
         }

         if (!this.geo_clones) {
            this.geo_clones = new JSROOT.GEO.ClonedNodes(null, nodes);
            this.geo_clones.name_prefix = this.geo_clones.GetNodeName(0);
         }

         for (var cnt = 0; cnt < draw_msg.visibles.length; ++cnt) {
            var item = draw_msg.visibles[cnt], rd = item.ri;

            // entry may be provided without shape - it is ok
            if (!rd) continue;

            if (rd.server_shape) {
               item.server_shape = rd.server_shape;
               continue;
            }

            // reconstruct render data
            var off = draw_msg.offset + rd.rnr_offset;

            if (rd.vert_size) {
               rd.vtxBuff = new Float32Array(draw_msg.raw, off, rd.vert_size);
               off += rd.vert_size*4;
            }

            if (rd.norm_size) {
               rd.nrmBuff = new Float32Array(draw_msg.raw, off, rd.norm_size);
               off += rd.norm_size*4;
            }

            if (rd.index_size) {
               rd.idxBuff = new Uint32Array(draw_msg.raw, off, rd.index_size);
               off += rd.index_size*4;
            }

            // shape handle is similar to created in JSROOT.GeoPainter
            item.server_shape = rd.server_shape = { geom: this.creator.makeEveGeometry(rd), nfaces: (rd.index_size-2)/3, ready: true };
         }

         return true;
      },

      /** function to accumulate and process all drawings messages
       * if not all scripts are loaded, messages are quied and processed later */

      checkDrawMsg: function(kind, msg, _raw, _offset) {
         if (kind == "binary") {
            for (var k = 0; k < this.queue.length; ++k) {
               if (this.queue[k].binlen && !this.queue[k].raw) {
                  this.queue[k].raw = _raw;
                  this.queue[k].offset = _offset;
                  _raw = null;
                  break;
               }
            }

            if (_raw)
               return console.error("Did not process raw data " + _raw.byteLength + " offset " + _offset);
         } else if (kind) {
            if (!msg)
               return console.error("No message is provided for " + kind);

            msg.kind = kind;

            this.queue.push(msg);
         }

         if (!this.creator ||            // complete JSROOT/EVE7 TGeo functionality is loaded
            !this.queue.length ||        // drawing messages are created
            !this.renderingDone) return; // UI5 rendering is performed

         // first message in the queue still waiting for raw data
         if (this.queue[0].binlen && !this.queue[0].raw)
            return;

         // only from here we can start to analyze messages and create TGeo painter, clones objects and so on

         msg = this.queue.shift();

         switch (msg.kind) {
            case "draw":

               // keep for history
               this.last_draw_msg = msg;

               // here we should decode render data
               this.extractRawShapes(msg);

               // after clones are existing - ensure geo painter is there
               this.createGeoPainter(msg.drawopt);

               this.geo_painter.prepareObjectDraw(msg.visibles, "__geom_viewer_selection__");

               // TODO: handle completion of geo drawing

               // this is just start drawing, main work will be done asynchronous
               break;

            case "found":
               // only extend nodes and decode shapes
               if (this.extractRawShapes(msg))
                  this.paintFoundNodes(msg.visibles, true, msg.binlen > 0);
               break;

            case "append":
               this.extractRawShapes(msg);
               this.appendNodes(msg.visibles);
               break;
         }
      },

      OnWebsocketOpened: function(handle) {
         this.isConnected = true;

         // when connection established, checked if we can submit requested
         this.checkRequestMsg();

         if (!this.plainModel) {
            this.model.sendFirstRequest(this.websocket);
         }
      },


      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');

         if (window) window.close();

         this.isConnected = false;
      },

      /** Entry point for all data from server */
      OnWebsocketMsg: function(handle, msg, offset) {

         // binary data can be send only as addition to draw message
         // here data can be placed in the queue and processed when all other prerequicities are done
         if (typeof msg != "string")
            return this.checkDrawMsg("binary", null, msg, offset);

         var mhdr = msg.substr(0,6);
         msg = msg.substr(6);

         console.log(mhdr, msg.length, msg.substr(0,70), "...");

         switch (mhdr) {
         case "DESCR:":  // browser hierarchy
            if (this.plainModel)
               this.parseDescription(msg, true);
            break;
         case "FESCR:":  // searching hierarchy
            if (this.plainModel)
               this.parseDescription(msg, false);
            break;
         case "BREPL:":   // browser reply
            if (!this.plainModel)
               this.model.processResponse(JSON.parse(msg));
            break;
         case "FOUND:":  // text message for found query
            this.showTextInBrowser(msg);
            this.paintFoundNodes(null); // nothing can be shown
            break;
         case "MODIF:":
            this.modifyDescription(msg);
            break;
         case "GDRAW:":   // normal drawing of geometry
            this.checkDrawMsg("draw", JSROOT.parse(msg)); // use JSROOT.parse while refs are used
            break;
         case "FDRAW:":   // drawing of found nodes
            this.checkDrawMsg("found", JSROOT.parse(msg));
            break;
         case "APPND:":
            this.checkDrawMsg("append", JSROOT.parse(msg));
            break;
         case "HOVER:":
            this._hover_stack = (msg == "OFF") ? null : JSON.parse(msg);
            if (this.geo_painter)
               this.geo_painter.HighlightMesh(null, 0x00ff00, null, undefined, this._hover_stack, true);
            break;
         case "HIGHL:":
            if (this.plainModel)
               this.highlighRowWithIds((msg == "OFF") ? null : JSON.parse(msg));
            else
               console.warn("Not yet implemented!!!");
            break;
         default:
            console.error('Non recognized msg ' + mhdr + ' len=' + msg.length);
         }
      },

      /** Format REveGeomNode data to be able use it in list of clones */
      formatNodeElement: function(elem) {
         elem.kind = 2; // special element for geom viewer, used in TGeoPainter
         var m = elem.matr;
         delete elem.matr;
         if (!m || !m.length) return;

         if (m.length == 16) {
            elem.matrix = m;
         } else {
            var nm = elem.matrix = new Array(16);
            for (var k=0;k<16;++k) nm[k] = 0;
            nm[0] = nm[5] = nm[10] = nm[15] = 1;

            if (m.length == 3) {
               // translation martix
               nm[12] = m[0]; nm[13] = m[1]; nm[14] = m[2];
            } else if (m.length == 4) {
               // scale matrix
               nm[0] = m[0]; nm[5] = m[1]; nm[10] = m[2]; nm[15] = m[3];
            } else if (m.length == 9) {
               // rotation matrix
               nm[0] = m[0]; nm[4] = m[1]; nm[8]  = m[2];
               nm[1] = m[3]; nm[5] = m[4]; nm[9]  = m[5];
               nm[2] = m[6]; nm[6] = m[7]; nm[10] = m[8];
            } else {
               console.error('wrong number of elements in the matrix ' + m.length);
            }
         }
      },

      /** Parse compact geometry description,
       * Used only to initialize hierarchy browser with full Tree,
       * later should be done differently */
      parseDescription: function(msg, is_original) {

         if (!this.plainModel)
            return console.error('Not allowed to directly set hierarchy in such mode');

         var descr = JSON.parse(msg);

         this.buildTree(descr, is_original);

         this.byId("treeTable").expandToLevel(is_original ? 1 : 99);
      },

      /** TO BE CHANGED !!! When single node element is modified on the server side */
      modifyDescription: function(msg) {
         var arr = JSON.parse(msg), can_refresh = true;

         if (!arr || !this.geo_clones) return;

         for (var k=0;k<arr.length;++k) {
            var moditem = arr[k];

            this.formatNodeElement(moditem);

            var item = this.geo_clones.nodes[moditem.id];

            if (!item)
               return console.error('Fail to find item ' + moditem.id);

            item.vis = moditem.vis;
            item.matrix = moditem.matrix;

            var dnode = this.originalCache ? this.originalCache[moditem.id] : null;

            if (dnode) {
               // here we can modify only node which was changed
               dnode.title = moditem.name;
               dnode.color_visible = false;
               dnode.node_visible = moditem.vis != 0;
            } else {
               can_refresh = false;
            }

            if (!moditem.vis && this.geo_painter)
               this.geo_painter.RemoveDrawnNode(moditem.id);
         }

         if (can_refresh) {
            this.model.refresh();
         } else {
            // rebuild complete tree for TreeBrowser
            this.buildTree();
         }

      },

      buildTreeNode: function(nodes, cache, indx) {
         var tnode = cache[indx];
         if (tnode) return tnode;

         var node = nodes[indx];

         cache[indx] = tnode = { title: node.name, id: indx, color_visible: false, node_visible: node.vis != 0 };

         if (node.color) {
            tnode.color = "rgb(" + node.color + ")";
            tnode.color_visisble = true;
         }

         if (node.chlds && (node.chlds.length>0)) {
            tnode.chlds = [];
            for (var k=0;k<node.chlds.length;++k)
               tnode.chlds.push(this.buildTreeNode(nodes, cache, node.chlds[k]));
         } else {
            tnode.end_node = true;
         }

         return tnode;
      },

      /** Build complete tree of all existing nodes.
       * Produced structure can be very large, therefore later
       * one should move this functionality to the server */
      buildTree: function(nodes, is_original) {

         if (!nodes) return;

         var cache = [];
         this.data.nodes = [ this.buildTreeNode(nodes, cache, 0) ];

         if (is_original) {
            // remember nodes
            this.originalCache = cache;
            this.originalNodes = this.data.nodes;
         }

         this.model.refresh();
      },

      /** search main drawn nodes for matches */
      findMatchesFromDraw: function(func) {
         var matches = [];

         if (this.last_draw_msg && this.last_draw_msg.visibles && this.geo_clones)
            for (var k=0;k<this.last_draw_msg.visibles.length;++k) {
               var item = this.last_draw_msg.visibles[k];
               var res = this.geo_clones.ResolveStack(item.stack);
               if (func(res.node))
                  matches.push({ stack: item.stack, color: item.color });
            }

         return matches;
      },

      /** Show special message insted of nodes hierarchy */
      showTextInBrowser: function(text) {
         var br = this.byId("treeTable");
         br.collapseAll();
         if (!text || (text === "RESET")) {
            this.data.nodes = this.originalNodes || null;
         } else {
            this.data.nodes = [ { title: text } ];
         }
         this.model.refresh();
         br.expandToLevel(1);
      },

      /** Show found nodes in the browser */
      showFoundNodes: function(matches) {
         if (!matches)
            return this.showTextInBrowser();

         var nodes = [];
         for (var k=0;k<matches.length;++k)
             this.appendStackToTree(nodes, matches[k].stack, matches[k].color);
         this.data.nodes = [ nodes[0] ];

         this.model.refresh();
         if (matches.length < 100)
            this.byId("treeTable").expandToLevel(99);
      },

      /** Here one tries to append only given stack to the tree
        * used to build partial tree with visible objects
        * Used only in standalone mode */
      appendStackToTree: function(tnodes, stack, color) {
         var prnt = null, node = null;
         for (var i=-1;i<stack.length;++i) {
            var indx = (i<0) ? 0 : node.chlds[stack[i]];
            node = this.geo_clones.nodes[indx];
            var tnode = tnodes[indx];
            if (!tnode)
               tnodes[indx] = tnode = { title: node.name, id: indx, color_visible: false, node_visible: true };

            if (prnt) {
               if (!prnt.chlds) prnt.chlds = [];
               if (prnt.chlds.indexOf(tnode) < 0)
                  prnt.chlds.push(tnode);
            }
            prnt = tnode;
         }

         prnt.end_node = true;
         prnt.color = color ? "rgb(" + color + ")" : "";
         prnt.color_visible = prnt.color.length > 0;
      },

      /** Paint extra node - or remove them from painting */
      paintFoundNodes: function(visibles, append_more, with_binaries) {
         if (!this.geo_painter) return;

         if (append_more && (with_binaries || !visibles))
            this.geo_painter.appendMoreNodes(visibles || null);

         if (visibles && visibles.length && (visibles.length < 100)) {
            var dflt = Math.max(this.geo_painter.options.transparency, 0.98);
            this.geo_painter.changeGlobalTransparency(function(node) {
               if (node.stack)
                  for (var n=0;n<visibles.length;++n)
                     if (JSROOT.GEO.IsSameStack(node.stack, visibles[n].stack))
                        return 0;
               return dflt;
            });

         } else {
            this.geo_painter.changeGlobalTransparency();
         }
      },


      appendNodes: function(nodes) {
         if (this.geo_painter) this.geo_painter.prepareObjectDraw(nodes, "__geom_viewer_append__");
      },

      showMoreNodes: function(matches) {
         if (!this.geo_painter) return;
         this.geo_painter.appendMoreNodes(matches);
         if (this._hover_stack)
            this.geo_painter.HighlightMesh(null, 0x00ff00, null, undefined, this._hover_stack, true);
      },

      omBeforeRendering: function() {
         this.renderingDone = false;
      },

      onAfterRendering: function() {
         this.renderingDone = true;

         this.checkRequestMsg();
      },

      checkRequestMsg: function() {
         if (this.isConnected && this.renderingDone) {

            if (!this.ask_reload) {
               this.ask_reload = true;

               // THIS IS REQUIRED FOR RELOAD OF FULL HIERARCHY
               if (this.plainModel)
                  this.websocket.Send("RELOAD");
            }

            if (this.creator && !this.ask_getdraw) {
               this.websocket.Send("GETDRAW");
               this.ask_getdraw = true;
            }
         }
      },

      /** method called from geom painter when specific node need to be activated in the browser
       * Due to complex indexing in TreeTable it is not trivial to select special node */
      activateInTreeTable: function(itemnames, force) {
         if (!force || !itemnames) return;

         var stack = this.geo_clones.FindStackByName(itemnames[0]);
         if (!stack) return;

         function test_match(st) {
            if (!st || (st.length > stack.length)) return -1;
            for (var k=0;k<stack.length;++k) {
               if (k>=st.length) return k;
               if (stack[k] !== st[k]) return -1; // either stack matches completely or not at all
            }
            return stack.length;
         }

         // now start searching items

         var tt = this.getView().byId("treeTable"),
             rows = tt.getRows(), best_match = -1, best_row = 0;

         for (var i=0;i<rows.length;++i) {

            var rstack = this.getRowStack(rows[i]);
            var match = test_match(rstack);

            if (match > best_match) {
               best_match = match;
               best_row = rows[i].getIndex();
            }
         }

         // start from very beginning
         if (best_match < 0) {
            tt.collapseAll();
            best_match = 0;
            best_row = 0;
         }

         if (best_match < stack.length) {
            var ii = best_row;
            // item should remain as is, but all childs can be below limit
            tt.expand(ii);

            while (best_match < stack.length) {
               ii += stack[best_match++] + 1; // stack is index in child array, can use it here
               if (ii > tt.getFirstVisibleRow() + tt.getVisibleRowCount()) {
                  tt.setFirstVisibleRow(Math.max(0, ii - Math.round(tt.getVisibleRowCount()/2)));
               }
               tt.expand(ii);
            }
         }
      },

      /** Submit node search query to server, ignore in offline case */
      submitSearchQuery: function(query, from_handler) {

         if (!from_handler) {
            // do not submit immediately, but after very short timeout
            // if user types very fast - only last selection will be shown
            if (this.search_handler) clearTimeout(this.search_handler);
            this.search_handler = setTimeout(this.submitSearchQuery.bind(this, query, true), 1000);
            return;
         }

         delete this.search_handler;

         this.websocket.Send("SEARCH:" + (query || ""));
      },

      /** when new query entered in the seach field */
      onSearch : function(oEvt) {
         var query = oEvt.getSource().getValue();
         if (!this.standalone) {
            this.submitSearchQuery(query);
         } else if (query) {
            var lst = this.findMatchesFromDraw(function(node) {
               return node.name.indexOf(query)==0;
            });

            this.showFoundNodes(lst);
            this.paintFoundNodes(lst);

         } else {
            this.showTextInBrowser();
            this.paintFoundNodes(null);
         }
      },

      /** Reload geometry description and base drawing, normally not required */
      onRealoadPress: function (oEvent) {
         this.websocket.Send("RELOAD");
      },

      /** Quit ROOT session */
      onQuitRootPress: function(oEvent) {
         this.websocket.Send("QUIT_ROOT");
      }

   });

});
