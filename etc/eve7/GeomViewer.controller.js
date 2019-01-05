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
         properties : {           // setter and getter are created behind the scenes, incl. data binding and type validation
            "color" : { type: "sap.ui.core.CSSColor", defaultValue: "#fff" } // you can give a default value and more
         }
      },

      // the part creating the HTML:
      renderer : function(oRm, oControl) { // static function, so use the given "oControl" instance instead of "this" in the renderer function
         oRm.write("<div"); 
         oRm.writeControlData(oControl);  // writes the Control ID and enables event handling - important!
         // oRm.addStyle("background-color", oControl.getColor());  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("width", "100%");  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("height", "100%");  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("overflow", "hidden");  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.writeStyles();
         oRm.addClass("myColorBox");      // add a CSS class for styles common to all control instances
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
            this.geo_painter.assignClones(clones);
         
         if (this.geo_visisble) {
            this.geo_painter.prepareObjectDraw(this.geo_visisble,"__geom_viewer_selection__");
            delete this.geo_visisble;
         }
      },
      
      assignClones: function(clones, drawopt) {
         if (this.geo_painter) {
            this.geo_painter.options = this.geo_painter.decodeOptions(drawopt);
            this.geo_painter.assignClones(clones);
         } else {
            this.geo_clones = clones;
            this.draw_options = drawopt;
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
      },

      handleChange: function (oEvent) {
         // var newColor = oEvent.getParameter("colorString");
         // this.setColor(newColor);
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
         
         //myControl.placeAt("content");

         // ok, add another instance...:
         //new my.ColorBox({color:"green"}).placeAt("content");
         
         this.getView().byId("mainSplitter").addContentArea(this.geomControl);
      },
      
      /** Called when data comes via the websocket */
      OnWebsocketMsg: function(handle, msg, offset) {

         if (typeof msg != "string") {
            
            if (this.draw_msg) {
               // here we should decode render data
               
               console.log('DRAWING cnt:', this.draw_msg.length, "BYNARY", msg.byteLength, offset);
               
               var rnr_cache = {};
               
               for (var cnt=0;cnt<this.draw_msg.length;++cnt) {
                  var rd = this.draw_msg[cnt];
                  
                  if (rd.rnr_offset<0) {
                     console.log('No binary data for elemet', cnt, rd);
                     continue;
                  }
                  
                  var cache = rnr_cache[rd.rnr_offset];
                  if (cache) {
                     rd.server_shape = cache.server_shape;
                     continue;
                  }

                  // reconstruct render data
                  var off = offset + rd.rnr_offset;
                  
                  var render_data = {}; // put render data in temporary object, only need to create mesh
                     
                  if (rd.vert_size) {
                     render_data.vtxBuff = new Float32Array(msg, off, rd.vert_size);
                     off += rd.vert_size*4;
                  }

                  if (rd.norm_size) {
                     render_data.nrmBuff = new Float32Array(msg, off, rd.norm_size);
                     off += rd.norm_size*4;
                  }

                  if (rd.index_size) {
                     render_data.idxBuff = new Uint32Array(msg, off, rd.index_size);
                     off += rd.index_size*4;
                  }
                   
                  // shape handle is similar to created in JSROOT.GeoPainter
                  rd.server_shape = { geom: this.creator.makeEveGeometry(render_data), nfaces: (rd.index_size-2) / 3, ready: true };
                   
                  rnr_cache[rd.rnr_offset] = rd;
               }

               this.geomControl.startDrawing(this.draw_msg);
               
               delete this.draw_msg;
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
            
         } else if (msg.indexOf("DRAW:") == 0) {
            this.draw_msg = JSROOT.parse(msg.substr(5));
         }
      },
      
      buildTreeNode: function(indx) {
         var tnode = this.tree_nodes[indx];
         if (tnode) return tnode;
         var node = this.clones.nodes[indx];
         
         this.tree_nodes[indx] = tnode = { title: node.name };
         
         if (node.chlds) {
            tnode.chlds = [];
            for (var k=0;k<node.chlds.length;++k)
               tnode.chlds.push(this.buildTreeNode(node.chlds[k]));
         }
         
         return tnode;
      },
      
      buildTree: function() {
         if (!this.descr || !this.descr.fDesc) return;
         
         this.tree_nodes = [];
         
         this.data.Nodes = [ this.buildTreeNode(0) ];
         
         this.model.refresh();
      },

      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');
         
         if (window) window.close();
      },
      
      onAfterRendering: function(){
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
      }
   });
});
