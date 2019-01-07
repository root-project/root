sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/core/Control',
               'sap/ui/model/json/JSONModel',
               'sap/ui/layout/Splitter',
               'sap/ui/layout/SplitterLayoutData',
               "sap/ui/core/ResizeHandler"
],function(Controller, CoreControl, JSONModel, Splitter, SplitterLayoutData, ResizeHandler) {
   "use strict";
   
   CoreControl.extend("eve.GeomDraw", { 

      // the control API:
      metadata : {
         properties : { // setter and getter are created behind the scenes, incl. data binding and type validation
            "color" : { type: "sap.ui.core.CSSColor", defaultValue: "#fff" } // you can give a default value and more
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
         //oRm.addClass("myColorBox");      // add a CSS class for styles common to all control instances
         oRm.writeClasses();              // this call writes the above class plus enables support for Square.addStyleClass(...)
         oRm.write(">"); 
         oRm.write("</div>"); // no text content to render; close the tag
      },

      // an event handler:
      onclick : function(evt) {   // is called when the Control's area is clicked - no further event registration required
      },
      
      onInit: function() {
      },

      onAfterRendering: function() {
         ResizeHandler.register(this, this.onResize.bind(this));
         this.did_rendering = true;
         this.geom_painter = null;
         
         this.geo_painter = JSROOT.Painter.CreateGeoPainter(this.getDomRef(), null, this.draw_options);
         
         if (this.geo_clones) 
            this.geo_painter.assignClones(this.geo_clones);
         
         if (this.geo_visisble) {
            this.geo_painter.prepareObjectDraw(this.geo_visisble,"__geom_viewer_selection__");
            delete this.geo_visisble;
         }
      },
      
      assignClones: function(clones, drawopt) {
         this.geo_clones = clones;
         this.draw_options = drawopt;

         if (this.geo_painter) {
            this.geo_painter.options = this.geo_painter.decodeOptions(drawopt);
            this.geo_painter.assignClones(clones);
         } 
      },
      
      startDrawing: function(visible) {
         if (this.geo_painter)
            this.geo_painter.prepareObjectDraw(visible,"__geom_viewer_selection__");
         else
            this.geo_visisble = visible;
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

   return Controller.extend("eve.GeomViewer", {
      onInit: function () {
         
         this.websocket = this.getView().getViewData().conn_handle;
         
         this.websocket.SetReceiver(this);
         this.websocket.Connect();
         
         this.data = { Nodes: null };
         
         this.descr = null; // description object from server 
         
         this.model = new JSONModel(this.data);
         this.getView().setModel(this.model);
         
         // PART 2: instantiate Control and place it onto the page

         this.geomControl = new eve.GeomDraw({color:"#f00"});
         
         this.creator = new JSROOT.EVE.EveElements();

         // catch re-rendering of the table to assign handlers 
         this.getView().byId("treeTable").addEventDelegate({
            onAfterRendering: function() { this.assignRowHandlers(); }
         }, this);
         
         //myControl.placeAt("content");

         // ok, add another instance...:
         //new my.ColorBox({color:"green"}).placeAt("content");
         
         this.getView().byId("mainSplitter").addContentArea(this.geomControl);
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
         // create stack and try highlight it if exists
         var stack = is_enter ? this.getRowStack(row) : null,
             ctxt = row.getBindingContext(),  
             prop = ctxt.getProperty(ctxt.getPath()),
             found_mesh = false;
         
         if (!this.geomControl || !this.geomControl.geo_painter) return;
         
         this._hover_stack = stack; // remember current element with hover stack
            
         found_mesh = this.geomControl.geo_painter.HighlightMesh(null, 0x00ff00, null, stack, true);
         
         if (stack && !found_mesh) {
            if (prop && prop.search_select && !prop.has_drawing) {
               console.log('Request stack', stack);
               this.submitSearchQuery(stack, true);
            }
         }
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
         
         return this.geomControl.geo_clones.MakeStackByIds(ids);
      },
      
      HighlightMesh: function(active_mesh, color, geo_object, geo_stack) {
         if (this.last_geo_stack === geo_stack) return;
         
         this.last_geo_stack = geo_stack;
         
         var rows = this.getView().byId("treeTable").getRows();
         
         for (var i=0;i<rows.length;++i) {
            var col = "";
            if (geo_stack && JSROOT.GEO.IsSameStack(this.getRowStack(rows[i]), geo_stack))
               col = "yellow";
            rows[i].$().css("background-color", col);
         }
      },
      
      /** Extract shapes from binary data using appropriate draw message 
       * Draw message is vector of REveGeomVisisble objects, including info where shape is in raw data */
      extractRawShapes: function(draw_msg, msg, offset) {
         
         for (var cnt=0;cnt < draw_msg.length;++cnt) {
            var item = draw_msg[cnt], rd = item.ri;

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
      
      /** Called when data comes via the websocket */
      OnWebsocketMsg: function(handle, msg, offset) {

         if (typeof msg != "string") {
            
            if (this.draw_msg) {
               // here we should decode render data
               
               this.extractRawShapes(this.draw_msg, msg, offset);

               // this is just start drawing, main work will be done asynchronous
               this.geomControl.startDrawing(this.draw_msg);
               
               delete this.draw_msg;
            } else if (this.found_msg) {
               
               this.extractRawShapes(this.found_msg, msg, offset);
              
               this.processSearchReply("FOUND:BIN");
               
               delete this.found_msg;
            }
            
            // console.log('ArrayBuffer size ',
            // msg.byteLength, 'offset', offset);
            return;
         }

         console.log("msg len=", msg.length, " txt:", msg.substr(0,70), "...");

         // this if first message
         if (!this.descr) {
            this.descr = JSROOT.parse(msg);
            
            // we need to calculate matrixes here
            var nodes = this.descr.fDesc;
            
            for (var cnt = 0; cnt < nodes.length; ++cnt) {
               var elem = nodes[cnt];
               elem.kind = 2; // special element
               var m = elem.matr;
               delete elem.matr;
               if (!m || !m.length) continue;
               
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
            }
            
            this.clones = new JSROOT.GEO.ClonedNodes(null, nodes);
            
            this.buildTree();
            
            this.geomControl.assignClones(this.clones, this.descr.fDrawOptions);
            
         } else if (msg.substr(0,5) == "DRAW:") {
            this.last_draw_msg = this.draw_msg = JSROOT.parse(msg.substr(5));
         } else if (msg.substr(0,6) == "FOUND:") {
            this.processSearchReply(msg, false); 
         } else if (msg.substr(0,6) == "SHAPE:") {
            this.processSearchReply(msg, true);
         }
      },
      
      buildTreeNode: function(indx) {
         var tnode = this.tree_nodes[indx];
         if (tnode) return tnode;

         var node = this.clones.nodes[indx];
         
         this.tree_nodes[indx] = tnode = { title: node.name, id: indx };
         
         if (node.chlds) {
            tnode.chlds = [];
            for (var k=0;k<node.chlds.length;++k) 
               tnode.chlds.push(this.buildTreeNode(node.chlds[k]));
         }
         
         return tnode;
      },
      
      // here try to append only given stack to the tree
      // used to build partial tree with visible objects
      appendStackToTree: function(tnodes, stack, has_drawing) {
         var prnt = null, node = null;
         for (var i=-1;i<stack.length;++i) {
            var indx = (i<0) ? 0 : node.chlds[stack[i]];
            node = this.clones.nodes[indx];
            var tnode = tnodes[indx];
            if (!tnode)
               tnodes[indx] = tnode = { title: node.name, id: indx };
            
            if (prnt) {
               if (!prnt.chlds) prnt.chlds = [];
               if (prnt.chlds.indexOf(tnode) < 0)
                  prnt.chlds.push(tnode);
            }
            prnt = tnode;
         }
         
         prnt.search_select = true;
         prnt.has_drawing = has_drawing;
      },
      
      buildTree: function() {
         if (!this.descr || !this.descr.fDesc) return;
         
         this.tree_nodes = [];
         
         this.data.Nodes = [ this.buildTreeNode(0) ];
         
         this.originalNodes = this.data.Nodes; 
         
         this.model.refresh();
      },
      
      /** search main drawn nodes for matches */ 
      findMatchesFromDraw: function(func) {
         var matches = [];
         
         if (this.last_draw_msg) 
            for (var k=0;k<this.last_draw_msg.length;++k) {
               var item = this.last_draw_msg[k];
               var res = this.clones.ResolveStack(item.stack);
               if (func(res.node)) 
                  matches.push({ stack: item.stack });
            }
         
         return matches;
      },
      
      /** try to show selected nodes. With server may be provided shapes */ 
      showFoundNodes: function(matches, append_more, with_binaries) {
         
         var gpainter = this.geomControl ? this.geomControl.geo_painter : null;
         
         if (typeof matches == "string") {
            this.byId("treeTable").collapseAll();
            this.data.Nodes = [ { title: matches } ];
            this.model.refresh();
            if (gpainter) {
               if (append_more) gpainter.appendMoreNodes(null);
               gpainter.changeGlobalTransparency();
            }
            return;
         }
         
         // fully reset search selection
         if ((matches === null) || (matches === undefined)) {
            this.byId("treeTable").collapseAll();
            this.data.Nodes = this.originalNodes || null;
            this.model.refresh();
            this.byId("treeTable").expandToLevel(1);
            
            if (gpainter) {
               if (append_more) gpainter.appendMoreNodes(matches);
               gpainter.changeGlobalTransparency();
            }
            return;
         }
         
         if (!matches || (matches.length == 0)) {
            this.data.Nodes = null;
            this.model.refresh();
         } else {
            var nodes = [];
            for (var k=0;k<matches.length;++k) 
               this.appendStackToTree(nodes, matches[k].stack, !!matches[k].ri);
            this.data.Nodes = [ nodes[0] ];
            this.model.refresh();
            if (matches.length < 100)
               this.byId("treeTable").expandToLevel(99);
         }
         
         if (gpainter) {
            if (append_more && with_binaries) gpainter.appendMoreNodes(matches);
            
            if ((matches.length>0) && (matches.length<100)) { 
               var dflt = Math.max(gpainter.options.transparency, 0.98);
               gpainter.changeGlobalTransparency(function(node) {
                  if (node.stack) 
                     for (var n=0;n<matches.length;++n)
                        if (JSROOT.GEO.IsSameStack(node.stack, matches[n].stack))
                           return 0;
                  return dflt;
               });
            } else {
               gpainter.changeGlobalTransparency();
            }
         }
      },

      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');
         
         if (window) window.close();
      },
      
      onAfterRendering: function() {
         if (this.geomControl && this.geomControl.geo_painter)
            this.geomControl.geo_painter.AddHighlightHandler(this);
      },
      

      onToolsMenuAction : function (oEvent) {

         var item = oEvent.getParameter("item");

         switch (item.getText()) {
            case "GED Editor": this.getView().byId("Summary").getController().toggleEditor(); break;
         }
      },
      
      showHelp : function(oEvent) {
         alert("User support: root-webgui@cern.ch");
      },
      
      showUserURL : function(oEvent) {
         sap.m.URLHelper.redirect("https://github.com/alja/jsroot/blob/dev/eve7.md", true);
      },
      
      processSearchReply: function(msg, is_shape) {
         // not waiting search - ignore any replies
         if (!this.waiting_search) return;
         
         msg = msg.substr(6);
         
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
            console.log('Process explicit shape for ', lst[0], has_binaries);  
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
      
      /** Submit node search query to server */ 
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
            query = "GETSHAPE:" + JSON.stringify(query);
         
         if (this.waiting_search) {
            // do not submit next search query when prvious not yet proceed
            this.next_search = query;
            return;
         }
         
         this.websocket.Send(query);
         this.waiting_search = true;
      },
      
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
      }
   });
});
