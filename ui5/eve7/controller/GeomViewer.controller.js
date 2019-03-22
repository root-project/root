sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/core/Control',
               'sap/ui/model/json/JSONModel',
               'sap/m/Text',
               'sap/m/CheckBox',
               'sap/ui/layout/Splitter',
               "sap/ui/core/ResizeHandler",
               "sap/ui/layout/HorizontalLayout",
               "sap/ui/table/Column"
],function(Controller, CoreControl, JSONModel, mText, mCheckBox, Splitter, ResizeHandler,
           HorizontalLayout, tableColumn) {

   "use strict";

   var geomColorBox = CoreControl.extend("rootui5.eve7.controller.ColorBox", { // call the new Control type "my.ColorBox" and let it inherit from sap.ui.core.Control

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



   return Controller.extend("rootui5.eve7.controller.GeomViewer", {
      onInit: function () {

         this.websocket = this.getView().getViewData().conn_handle;

         this.websocket.SetReceiver(this);
         this.websocket.Connect();

         this.queue = []; // received draw messages

         this.data = { Nodes: null };

         this.model = new JSONModel(this.data);
         this.getView().setModel(this.model);

         // PART 2: instantiate Control and place it onto the page

         if (JSROOT.GetUrlOption('nobrowser') !== null) {
            // remove complete area - plain geometry drawing
            this.getView().byId("mainSplitter").removeAllContentAreas();
         } else {

            var t = this.getView().byId("treeTable");

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

      /** function called then mouse-hover event over the row is invoked
       * Used to highlight correspondent volume on geometry drawing
       */
      onRowHover: function(row, is_enter) {
         // property of current entry, not used now
         var ctxt = row.getBindingContext(),
             prop = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;

         // remember current element with hover stack
         this._hover_stack = (is_enter && prop && prop.end_node) ? this.getRowStack(row) : null;

         if (!this.geo_painter) return;

         var found_mesh = this.geo_painter.HighlightMesh(null, 0x00ff00, null, undefined, this._hover_stack, true);

         // request given stack
         if (this._hover_stack && !found_mesh)
            this.submitSearchQuery(this._hover_stack, true);
      },

      /** Return nodeid for the row */
      getRowNodeId: function(row) {
         var ctxt = row.getBindingContext();
         var ttt = ctxt ? ctxt.getProperty(ctxt.getPath()) : null;
         return ttt && (ttt.id!==undefined) ? ttt.id : -1;
      },

      /** try to produce stack out of row path */
      getRowStack: function(row) {
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

         return this.geo_clones.MakeStackByIds(ids);
      },

      /** Callback from geo painter when mesh object is highlighted. Use for update of TreeTable */
      HighlightMesh: function(active_mesh, color, geo_object, geo_index, geo_stack) {
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
      extractRawShapes: function(draw_msg, msg, offset) {

         var nodes = null;

         if (!this.geo_clones)
            nodes = new Array(draw_msg.numnodes); // array for all nodes

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
            var off = offset + rd.rnr_offset;

            if (rd.vert_size) {
               rd.vtxBuff = new Float32Array(msg, off, rd.vert_size);
               off += rd.vert_size*4;
            }

            if (rd.norm_size) {
               rd.nrmBuff = new Float32Array(msg, off, rd.norm_size);
               off += rd.norm_size*4;
            }

            if (rd.index_size) {
               rd.idxBuff = new Uint32Array(msg, off, rd.index_size);
               off += rd.index_size*4;
            }

            // shape handle is similar to created in JSROOT.GeoPainter
            item.server_shape = rd.server_shape = { geom: this.creator.makeEveGeometry(rd), nfaces: (rd.index_size-2)/3, ready: true };
         }
      },

      /** function to accumulate and process all drawings messages
       * if not all scripts are loaded, messages are quied and processed later */

      checkDrawMsg: function(kind, msg, _raw, _offset) {
         if (kind) {
            msg.kind = kind;
            msg.raw = _raw;
            msg.offset = _offset;

            this.queue.push(msg);
         }

         if (!this.creator ||             // complete JSROOT/EVE7 TGeo functionality is loaded
            (this.queue.length == 0) ||   // drawing messages are created
            !this.renderingDone) return;  // UI5 rendering is performed

         // only from here we can start to analyze messages and create TGeo painter, clones objects and so on


         msg = this.queue.shift();

         switch (msg.kind) {
            case "draw":
               // here we should decode render data
               this.extractRawShapes(msg, msg.raw, msg.offset);

               // after clones are existing - ensure geo painter is there
               this.createGeoPainter(msg.drawopt);

               this.geo_painter.prepareObjectDraw(msg.visibles, "__geom_viewer_selection__");

               // TODO: handle completion of geo drawing

               // this is just start drawing, main work will be done asynchronous
               break;

            case "found":
               this.extractRawShapes(msg, msg.raw, msg.offset);
               this.processSearchReply("BIN");
               break;

            case "append":
               this.extractRawShapes(msg, msg.raw, msg.offset);
               this.appendNodes(msg.visibles);
               break;

         }
      },

      OnWebsocketOpened: function(handle) {
         this.isConnected = true;

         // when connection established, checked if we can submit requested
         this.checkRequestMsg();
      },


      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');

         if (window) window.close();

         this.isConnected = false;
      },

      /** Entry point for all data from server */
      OnWebsocketMsg: function(handle, msg, offset) {

         if (typeof msg != "string") {

            if (this.draw_msg) {
               this.checkDrawMsg("draw", this.draw_msg, msg, offset);
               delete this.draw_msg;
            } else if (this.found_msg) {
               this.checkDrawMsg("found", this.found_msg, msg, offset);
               delete this.found_msg;
            } else if (this.append_msg) {
               this.checkDrawMsg("append", this.append_msg, msg, offset);
               delete this.append_msg;
            } else {
               console.error('not process binary data len=' + (msg ? msg.byteLength : 0))
            }

            return;
         }

         var mhdr = msg.substr(0,6);
         msg = msg.substr(6);

         // console.log(mhdr, msg.length, msg.substr(0,70), "...");

         switch (mhdr) {
         case "DESCR:":
            this.parseDescription(msg);
            break;
         case "MODIF:":
            this.modifyDescription(msg);
            break;
         case "GDRAW:":
            this.last_draw_msg = this.draw_msg = JSROOT.parse(msg); // use JSROOT.parse while refs are used
            break;
         case "APPND:":
            this.append_msg = JSROOT.parse(msg); // use JSROOT.parse while refs are used
            break;
         case "FOUND:":
            this.processSearchReply(msg, false);
            break;
         case "SHAPE:":
            this.processSearchReply(msg, true);
            break
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

      /** Parse and format base geometry description,
       * Used only to initialize hierarchy browser with full Tree,
       * later should be done differently */
      parseDescription: function(msg) {
         var descr = JSON.parse(msg);

         // var nodes = descr.fDesc;

         // we need to calculate matrixes here
         // for (var cnt = 0; cnt < nodes.length; ++cnt)
         //    this.formatNodeElement(nodes[cnt]);

         // var clones = new JSROOT.GEO.ClonedNodes(null, nodes);
         // clones.name_prefix = clones.GetNodeName(0);

         // this.assignClones(clones, descr.fDrawOptions);

         // this.draw_options = descr.fDrawOptions;

         this.buildTree(descr);
      },

      /** When single node element is modified on the server side */
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

      // here try to append only given stack to the tree
      // used to build partial tree with visible objects
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
      buildTree: function(nodes) {

         if (!nodes) return;

         this.originalCache = [];

         this.data.Nodes = [ this.buildTreeNode(nodes, this.originalCache, 0) ];

         this.originalNodes = this.data.Nodes;

         this.model.refresh();
      },

      /** search main drawn nodes for matches */
      findMatchesFromDraw: function(func) {
         var matches = [];

         if (this.last_draw_msg && this.last_draw_msg.visisbles && this.geo_clones)
            for (var k=0;k<this.last_draw_msg.visisbles.length;++k) {
               var item = this.last_draw_msg.visisbles[k];
               var res = this.geo_clones.ResolveStack(item.stack);
               if (func(res.node))
                  matches.push({ stack: item.stack, color: item.color });
            }

         return matches;
      },

      /** try to show selected nodes. With server may be provided shapes */
      showFoundNodes: function(matches, append_more, with_binaries) {

         if (typeof matches == "string") {
            this.byId("treeTable").collapseAll();
            this.data.Nodes = [ { title: matches } ];
            this.model.refresh();
            if (this.geo_painter) {
               if (append_more) this.geo_painter.appendMoreNodes(null);
               this.geo_painter.changeGlobalTransparency();
            }
            return;
         }

         // fully reset search selection
         if ((matches === null) || (matches === undefined)) {
            this.byId("treeTable").collapseAll();
            this.data.Nodes = this.originalNodes || null;
            this.model.refresh();
            this.byId("treeTable").expandToLevel(1);

            if (this.geo_painter) {
               if (append_more) this.geo_painter.appendMoreNodes(matches);
               this.geo_painter.changeGlobalTransparency();
            }
            return;
         }

         if (!matches || (matches.length == 0)) {
            this.data.Nodes = null;
            this.model.refresh();
         } else {
            var nodes = [];
            for (var k=0;k<matches.length;++k)
               this.appendStackToTree(nodes, matches[k].stack, matches[k].color);
            this.data.Nodes = [ nodes[0] ];
            this.model.refresh();
            if (matches.length < 100)
               this.byId("treeTable").expandToLevel(99);
         }

         if (this.geo_painter) {
            if (append_more && with_binaries) this.geo_painter.appendMoreNodes(matches);

            if ((matches.length>0) && (matches.length<100)) {
               var dflt = Math.max(this.geo_painter.options.transparency, 0.98);
               this.geo_painter.changeGlobalTransparency(function(node) {
                  if (node.stack)
                     for (var n=0;n<matches.length;++n)
                        if (JSROOT.GEO.IsSameStack(node.stack, matches[n].stack))
                           return 0;
                  return dflt;
               });
            } else {
               this.geo_painter.changeGlobalTransparency();
            }
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


      /** called when new portion of data received from server */
      processSearchReply: function(msg, is_shape) {
         // not waiting search - ignore any replies
         if (!this.waiting_search) return;

         var lst = [], has_binaries = false;

         if (msg == "BIN") {
            has_binaries = true;
            lst = this.found_msg; is_shape = this.found_shape;
            delete this.found_msg;
            delete this.found_shape;
         } else if (msg == "NO") {
            lst = "Not found";
         } else if (msg.substr(0,7) == "TOOMANY") {
            lst = "Too many " + msg.substr(8);
         } else {
            lst = JSROOT.parse(msg);
            for (var k=0;k<lst.length;++k)
               if (lst[k].ri) { // wait for binary render data
                  this.found_msg = lst;
                  this.found_shape = is_shape;
                  return;
               }
         }

         if (is_shape) {
            this.showMoreNodes(lst);
         } else {
            this.showFoundNodes(lst, true, has_binaries);
         }

         if (this.next_search) {
            this.websocket.Send(this.next_search);
            delete this.next_search;
         } else {
            this.waiting_search = false;
         }
      },

      /** Submit node search query to server, ignore in offline case */
      submitSearchQuery: function(query, from_handler) {

         // ignore query in file description mode
         if (this.websocket.kind == "file") return;

         if (!from_handler) {
            // do not submit immediately, but after very short timeout
            // if user types very fast - only last selection will be shown
            if (this.search_handler) clearTimeout(this.search_handler);
            this.search_handler = setTimeout(this.submitSearchQuery.bind(this, query, true), 1000);
            return;
         }

         delete this.search_handler;

         if (!query) {
            // if empty query specified - restore geometry drawing and ignore any possible reply from server
            this.waiting_search = false;
            delete this.next_search;
            this.showFoundNodes(null);
            return;
         }

         if (typeof query == "string")
            query = "SEARCH:" + query;
         else
            query = "GET:" + JSON.stringify(query);

         if (this.waiting_search) {
            // do not submit next search query when prvious not yet proceed
            this.next_search = query;
            return;
         }

         this.websocket.Send(query);
         this.waiting_search = true;
      },

      /** when new query entered in the seach field */
      onSearch : function(oEvt) {
         var query = oEvt.getSource().getValue();
         if (this.websocket.kind != "file") {
            this.submitSearchQuery(query);
         } else if (query) {
            this.showFoundNodes(
               this.findMatchesFromDraw(function(node) {
                  return node.name.indexOf(query)==0;
               })
            );
         } else {
            this.showFoundNodes(null);
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
