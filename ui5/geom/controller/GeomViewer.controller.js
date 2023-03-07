sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/model/json/JSONModel'
], function(Controller,
            JSONModel) {

   "use strict";

   /** @summary Central geometry viewer contoller
    * @desc All TGeo functionality is loaded after main ui5 rendering is performed,
    * To start drawing, following stages should be completed:
    *    - ui5 element is rendered (onAfterRendering is called)
    *    - TGeo-related JSROOT functionality is loaded
    *    - RGeomDrawing object delivered from the server
    * Only after all this stages are completed, one could start to analyze  */

   return Controller.extend("rootui5.geom.controller.GeomViewer", {

      onInit() {

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

         if (!this.standalone && !this._embeded && this.websocket.addReloadKeyHandler)
            this.websocket.addReloadKeyHandler();

         this.cfg = { standalone: this.websocket.kind == "file" };
         this.cfg_model = new JSONModel(this.cfg);
         this.getView().setModel(this.cfg_model);

         this.nobrowser = this.websocket.getUserArgs('nobrowser') || this.jsroot.decodeUrl().has('nobrowser');
         this.show_columns = !this.nobrowser && this.websocket.getUserArgs('show_columns') || this.jsroot.decodeUrl().has('show_columns');

         if (this.nobrowser) {
            // remove main area - plain geometry drawing
            // if master activated - immediately show control
            let app = this.byId("geomViewerApp");
            app.setMode(sap.m.SplitAppMode.HideMode);
            app.setInitialMaster(this.createId("geomControl"));
            app.removeMasterPage(this.byId("geomHierarchy"));
            this.byId("geomControl").setShowNavButton(false);
         } else {
            this.getView().byId('expandMaster').setVisible(this.show_columns);
         }

         Promise.all([import(this.jsroot.source_dir + 'modules/geom/geobase.mjs'), import(this.jsroot.source_dir + 'modules/geom/TGeoPainter.mjs')]).then(arr => {
            this.geo = Object.assign({}, arr[0], arr[1]);
            this.checkSendRequest();
         });
      },

      /** @summary Send RGeomRequest data to geometry viewer */
      sendViewerRequest(oper, args) {
         let req = { oper, path: [], stack: [] };
         Object.assign(req, args);
         this.websocket.send("GVREQ:" + JSON.stringify(req));
      },

      /** Process reply on RGeomRequest */
      processViewerReply(repl) {
         if (!repl?.oper)
            return false;

         if (repl.oper == "HOVER") {

            this._hover_stack = repl.stack || null;
            if (this.geo_painter)
               this.geo_painter.highlightMesh(null, 0x00ff00, null, undefined, this._hover_stack, true);

         } else if (repl.oper == "HIGHL") {

            this.highlighRowWithPath(repl.path);

         }
      },

      /** @brief Handler for mouse-hover event, provided from hierarchy
        * @desc Used to highlight correspondent volume on geometry drawing */
      onRowHover(prop, is_enter) {

         // ignore hover event when drawing not exists
         if (!this.isDrawPageActive()) return;

         if (!this.standalone) {
            let req = is_enter && prop && prop.path && prop.isLeaf ? prop.path : [ "OFF" ];
            // avoid multiple time submitting same request
            if (this.comparePaths(this._last_hover_req, req) === 1000) return;

            this._last_hover_req = req;
            return this.sendViewerRequest("HOVER", { path: req });
         }

         if (this.geo_painter && this.geo_clones) {
            let strpath = "";

            if (prop?.path && is_enter)
               strpath = prop.path.join("/");

            // remember current element with hover stack
            this._hover_stack = strpath ? this.geo_clones.findStackByName(strpath) : null;

            this.geo_painter.highlightMesh(null, 0x00ff00, null, undefined, this._hover_stack, true);
         }
      },


      /** @summary Callback from geo painter when mesh object is highlighted. Use for update of TreeTable */
      highlightMesh(active_mesh, color, geo_object, geo_index, geo_stack) {
         if (!this.standalone) {
            let req = geo_stack ? geo_stack : [];
            // avoid multiple time submitting same request
            if (this.geo.isSameStack(this._last_highlight_req, req)) return;
            this._last_highlight_req = req;
            return this.sendViewerRequest("HIGHL", { stack: req });
         }

         let hpath = "";

         if (this.geo_clones && geo_stack) {
            let info = this.geo_clones.resolveStack(geo_stack);
            if (info && info.name) hpath = info.name;
         }

         this.highlighRowWithPath(hpath.split("/"));
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
         if (this.nobrowser) return;

         this.byId('geomHierarchyPanel')?.getController()?.highlighRowWithPath(path);
      },

      createGeoPainter(drawopt) {

         if (this.geo_painter) {
            this.geo_painter.clearDrawings();
         } else {
            let geomDrawing = this.byId("geomDrawing");
            this.geo_painter = this.geo.createGeoPainter(geomDrawing.getDomRef(), null, drawopt);
            this.geo_painter.setMouseTmout(0);
            // this.geo_painter.setDepthMethod("dflt");
            this.geo_painter.ctrl.notoolbar = true;
            // this.geo_painter.showControlOptions = this.showControl.bind(this);

            this.geo_painter.setCompleteHandler(this.completeGeoDrawing.bind(this));

            this.geom_model = new JSONModel(this.geo_painter.ctrl);
            this.geo_painter.ctrl.cfg = {}; // dummy config until real config is received
            this.byId("geomControl").setModel(this.geom_model);
            geomDrawing.setGeomPainter(this.geo_painter);
            this.geo_painter.addHighlightHandler(this);
         }

         this.geo_painter.activateInBrowser = this.activateInTreeTable.bind(this);

         this.geo_painter.assignClones(this.geo_clones);
      },

      /** @summary Extract shapes from binary data using appropriate draw message
        * @desc Draw message is vector of REveGeomVisible objects, including info where shape is in raw data */
      extractRawShapes(draw_msg, recreate) {

         let nodes = null, old_gradpersegm = 0;

         // array for descriptors for each node
         // if array too large (>1M), use JS object while only ~1K nodes are expected to be used
         if (recreate) {
            if (draw_msg.kind !== "draw") return false;
            nodes = (draw_msg.numnodes > 1e6) ? { length: draw_msg.numnodes } : new Array(draw_msg.numnodes); // array for all nodes
         }

         draw_msg.nodes.forEach(node => {
            node = this.geo.ClonedNodes.formatServerElement(node);
            if (nodes)
               nodes[node.id] = node;
            else
               this.geo_clones.updateNode(node);
         });

         if (recreate) {
            this.geo_clones = new this.geo.ClonedNodes(null, nodes);
            this.geo_clones.name_prefix = this.geo_clones.getNodeName(0);
            // normally only need when making selection, not used in geo viewer
            // this.geo_clones.setMaxVisNodes(draw_msg.maxvisnodes);
            // this.geo_clones.setVisLevel(draw_msg.vislevel);
            // parameter need for visualization with transparency
            // TODO: provide from server
            this.geo_clones.maxdepth = 20;
         }

         let nsegm = 0;
         if (draw_msg.cfg)
            nsegm = draw_msg.cfg.nsegm;
         else if (this.geom_model)
            nsegm = this.geom_model.getProperty("/cfg/nsegm");

         if (nsegm) {
            old_gradpersegm = this.geo.geoCfg("GradPerSegm");
            this.geo.geoCfg("GradPerSegm", 360 / Math.max(nsegm,6));
         }

         for (let cnt = 0; cnt < draw_msg.visibles.length; ++cnt) {
            let item = draw_msg.visibles[cnt], rd = item.ri;

            // entry may be provided without shape - it is ok
            if (!rd) continue;

            item.server_shape = rd.server_shape =
               this.geo.createServerGeometry(rd, nsegm);
         }

         if (old_gradpersegm)
            this.geo.geoCfg("GradPerSegm", old_gradpersegm);

         return true;
      },

      /** @summary function to accumulate and process all drawings messages
       * @desc if not all scripts are loaded, messages are quied and processed later */
      checkDrawMsg(kind, msg) {
         if (kind) {
            if (!msg)
               return console.error(`No message is provided for ${kind}`);

            msg.kind = kind;

            this.queue.push(msg);
         }

         if (!this.geo ||                // complete JSROOT TGeo functionality is loaded
            !this.queue.length ||        // drawing messages are created
            !this.renderingDone) return; // UI5 rendering is performed

         // only from here we can start to analyze messages and create TGeo painter, clones objects and so on

         msg = this.queue.shift();

         switch (msg.kind) {
            case "draw":

               // keep for history
               this.last_draw_msg = msg;

               // here we should decode render data
               this.extractRawShapes(msg, true);

               // after clones are existing - ensure geo painter is there
               this.createGeoPainter(msg.cfg?.drawopt ?? '');

               // assign configuration to the control
               if (msg.cfg) {
                  this.geo_painter.ctrl.cfg = msg.cfg;
                  this.geo_painter.ctrl.show_config = true;
                  this.geom_model.refresh();
               }

               this.geo_painter.prepareObjectDraw(msg.visibles, "__geom_viewer_selection__");

               // TODO: handle completion of geo drawing

               // this is just start drawing, main work will be done asynchronous
               break;

            case "found":
               // only extend nodes and decode shapes
               if (this.extractRawShapes(msg))
                  this.paintFoundNodes(msg.visibles, true);
               break;

            case "append":
               this.extractRawShapes(msg);
               this.appendNodes(msg.visibles);
               break;
         }
      },

      completeGeoDrawing() {
         this.geom_model?.refresh();
      },

      onWebsocketOpened(/* handle */) {
         this.isConnected = true;

         // when connection established, checked if we can submit request
         this.checkSendRequest();
      },

      onWebsocketClosed() {
         // when connection closed, close panel as well
         if (window && !this._embeded) window.close();

         this.isConnected = false;
      },

      /** Entry point for all data from server */
      onWebsocketMsg(handle, msg /*, offset */) {

         // binary data can be send only as addition to draw message
         // here data can be placed in the queue and processed when all other prerequicities are done
         if (typeof msg != "string")
            return console.error("Geom viewer do not uses binary messages len = " + mgs.byteLength);

         let mhdr = msg.slice(0,6);
         msg = msg.slice(6);

         // console.log(`RECV ${mhdr} len: ${msg.length} ${msg.slice(0,70)} ...`);

         switch (mhdr) {
         case "MODIF:":
            this.modifyDescription(msg);
            break;
         case "GDRAW:":   // normal drawing of geometry
            this.checkDrawMsg("draw", this.jsroot.parse(msg)); // use jsroot.parse while refs are used
            break;
         case "FDRAW:": // drawing of found nodes
            this.checkDrawMsg("found", this.jsroot.parse(msg));
            break;
         case "APPND:":
            this.checkDrawMsg("append", this.jsroot.parse(msg));
            break;
         case "GVRPL:":
            this.processViewerReply(this.jsroot.parse(msg));
            break;
         case "NINFO:":
            this.provideNodeInfo(this.jsroot.parse(msg));
            break;
         case "CLRSCH":
            this.paintFoundNodes(null);
            break;
         case "DROPT:":
            this.applyDrawOptions(msg);
            break;
         case "IMAGE:":
            this.produceImage(msg);
            break;
         default:
            console.error('Non recognized msg ' + mhdr + ' len=' + msg.length);
         }
      },

      /** TO BE CHANGED !!! When single node element is modified on the server side */
      modifyDescription(msg) {
         let arr = JSON.parse(msg);
         if (!arr || !this.geo_clones) return;

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

      /** @summary search main drawn nodes for matches */
      findMatchesFromDraw(func) {
         let matches = [];

         if (this.last_draw_msg?.visibles && this.geo_clones)
            for (let k = 0; k < this.last_draw_msg.visibles.length; ++k) {
               let item = this.last_draw_msg.visibles[k];
               let res = this.geo_clones.resolveStack(item.stack);
               if (func(res.node))
                  matches.push({ stack: item.stack, color: item.color });
            }

         return matches;
      },

      /** @summary Here one tries to append only given stack to the tree
        * @desc used to build partial tree with visible objects
        * Used only in standalone mode */
      appendStackToTree(tnodes, stack, color, material) {
         let prnt = null, node = null, path = [];
         for (let i = -1; i < stack.length; ++i) {
            let indx = (i < 0) ? 0 : node.chlds[stack[i]];
            node = this.geo_clones.nodes[indx];
            path.push(node.name);
            let tnode = tnodes[indx];
            if (!tnode)
               tnodes[indx] = tnode = { name: node.name, path: path.slice(), id: indx, color: '', node_visible: true };

            if (prnt) {
               if (!prnt.childs) prnt.childs = [];
               if (prnt.childs.indexOf(tnode) < 0)
                  prnt.childs.push(tnode);
               prnt.nchilds = prnt.childs.length;
               prnt.expanded = true;
            }
            prnt = tnode;
         }

         prnt.end_node = true;
         prnt.color = color;
         prnt.material = material;
      },

      /** @summary Paint extra node - or remove them from painting */
      paintFoundNodes(visibles, append_more) {
         if (!this.geo_painter) return;

         if (append_more)
            this.geo_painter.appendMoreNodes(visibles || null);

         if (visibles?.length) {
            let dflt = Math.max(this.geo_painter.ctrl.transparency, 0.98), indx = 0;
            this.geo_painter.changedGlobalTransparency(node => {
               if (node.stack && (indx < visibles.length) &&  this.geo.isSameStack(node.stack, visibles[indx].stack)) {
                  indx++;
                  return 0;
               }
               return dflt;
            });

         } else {
            this.geo_painter.changedGlobalTransparency();
         }
      },

      appendNodes(nodes) {
         this.geo_painter?.prepareObjectDraw(nodes, "__geom_viewer_append__");
      },

      showMoreNodes(matches) {
         if (!this.geo_painter) return;
         this.geo_painter.appendMoreNodes(matches);
         if (this._hover_stack)
            this.geo_painter.highlightMesh(null, 0x00ff00, null, undefined, this._hover_stack, true);
      },

      onBeforeRendering() {
         this.renderingDone = false;
      },

      onAfterRendering() {
         this.renderingDone = true;

         this.checkSendRequest();
      },

      onAfterMasterOpen() {
      },

      checkSendRequest(force) {
         if (force) this.ask_getdraw = false;

         if (this.isConnected && !this.nobrowser && !this.send_channel) {
            let h = this.byId('geomHierarchyPanel');

            let websocket = this.websocket.createChannel();

            h.getController().configure({
               websocket,
               viewer: this,
               standalone: this.standalone,
               show_columns: this.show_columns,
               jsroot: this.jsroot
            });

            this.websocket.send("HCHANNEL:" + websocket.getChannelId());

            console.log('did send channel', websocket.getChannelId());
            this.send_channel = true;
         }

         if (this.isConnected && this.renderingDone) {

            if (this.geo && !this.ask_getdraw) {
               this.websocket.send("GETDRAW");
               this.ask_getdraw = true;
            }
         }
      },

      /** @summary method called from geom painter when specific node need to be activated in the browser
       * @desc Due to complex indexing in TreeTable it is not trivial to select special node */
      activateInTreeTable(itemnames, force) {
         if (this.nobrowser) return;
         this.byId('geomHierarchyPanel')?.getController().activateInTreeTable(itemnames, force);
      },

      /** when new draw options send from server */
      applyDrawOptions(opt) {
         if (!this.geo_painter) return;

         this.geo_painter.setAxesDraw(opt.indexOf("axis") >= 0);

         this.geo_painter.setAutoRotate(opt.indexOf("rotate") >= 0);
      },

      /** Try to provide as much info as possible offline */
      processInfoOffline(path, id) {
         let model = new JSONModel({ path: path, strpath: path.join("/")  });

         this.byId("geomInfo").setModel(model);

         if (this.geo_clones && path) {
            let stack = this.geo_clones.findStackByName(path.join("/"));

            let info = stack ? this.geo_clones.resolveStack(stack) : null;

            let build_shape = null;

            // this can be moved into GeoPainter later
            if (info && (info.id !== undefined) && this.geo_painter && this.geo_painter._draw_nodes) {
               for (let k = 0; k < this.geo_painter._draw_nodes.length; ++k) {
                  let item = this.geo_painter._draw_nodes[k];
                  if ((item.nodeid == info.id) && item.server_shape) {
                     build_shape = item.server_shape;
                     break;
                  }
               }
            }

            this.drawNodeShape(build_shape, true);
         }
      },

      /** @summary This is reply on INFO request */
      provideNodeInfo(info) {

         info.strpath = info.path.join("/"); // only for display

         let model = new JSONModel(info);

         this.byId("geomInfo").setModel(model);

         let server_shape = null;

         if (info.ri)
            server_shape = this.geo.createServerGeometry(info.ri, 0);

         this.drawNodeShape(server_shape, false);
      },

      drawNodeShape(server_shape, skip_cleanup) {

         let nodeDrawing = this.byId("nodeDrawing");

         nodeDrawing.setGeomPainter(null);

         this.node_painter_active = false;
         if (this.node_painter) {
            this.node_painter.clearDrawings();
            delete this.node_painter;
            delete this.node_model;
         }

         if (!server_shape) {
            this.byId("geomControl").setModel(this.geom_model);
            return;
         }

         this.node_painter = this.geo.createGeoPainter(nodeDrawing.getDomRef(), server_shape, "");
         this.node_painter.setMouseTmout(0);
         this.node_painter.ctrl.notoolbar = true;
         this.node_painter_active = true;

         // this.node_painter.setDepthMethod("dflt");
         this.node_model = new JSONModel(this.node_painter.ctrl);

         nodeDrawing.setGeomPainter(this.node_painter, skip_cleanup);
         this.byId("geomControl").setModel(this.node_model);
         this.node_painter.prepareObjectDraw(server_shape, "");
      },

      /** Save as png image */
      pressSaveButton() {
         this.produceImage("");
      },

      produceImage(name) {
         let painter = (this.node_painter_active && this.node_painter) ? this.node_painter : this.geo_painter;
         if (!painter) return;

         let dataUrl = painter.createSnapshot(this.standalone ? "geometry.png" : "asis");
         if (!dataUrl) return;
         let separ = dataUrl.indexOf("base64,");
         if ((separ>=0) && this.websocket && !this.standalone)
            this.websocket.send("IMAGE:" + name + "::" + dataUrl.substr(separ+7));
      },

      isDrawPageActive() {
         let app = this.byId("geomViewerApp"),
             curr = app ? app.getCurrentDetailPage() : null;
         return curr ? curr.getId() == this.createId("geomDraw") : false;
      },

      isInfoPageActive() {
         let app = this.byId("geomViewerApp"),
             curr = app ? app.getCurrentDetailPage() : null;
         return curr ? curr.getId() == this.createId("geomInfo") : false;
      },

      onInfoPress() {
         let app = this.byId("geomViewerApp"), ctrlmodel;

         if (this.isInfoPageActive()) {
            app.toDetail(this.createId("geomDraw"));
            this.node_painter_active = false;
            ctrlmodel = this.geom_model;
         } else {
            app.toDetail(this.createId("geomInfo"));
            this.node_painter_active = true;
            ctrlmodel = this.node_model;
         }

         if (ctrlmodel) {
            this.byId("geomControl").setModel(ctrlmodel);
            ctrlmodel.refresh();
         }

      },

      onExpandMaster() {
         const master = this.getView().byId('geomHierarchy').getParent();
         master.toggleStyleClass('masterExpanded');
         const expanded = master.hasStyleClass('masterExpanded');
         const btn = this.getView().byId('expandMaster');
         btn.setIcon(expanded ? "sap-icon://close-command-field" : "sap-icon://open-command-field");
      },

      /** Quit ROOT session */
      onQuitRootPress() {
         if (!this.standalone)
            this.websocket.send("QUIT_ROOT");
      },

      onPressMasterBack() {
         this.byId("geomViewerApp").backMaster();
      },

      onPressDetailBack() {
         this.byId("geomViewerApp").backDetail();
      },

      showControl() {
         this.byId("geomViewerApp").toMaster(this.createId("geomControl"));
      },

      sendConfig() {
         if (!this.standalone && this.geo_painter?.ctrl.cfg) {
            let cfg = this.geo_painter.ctrl.cfg;
            cfg.build_shapes = parseInt(cfg.build_shapes);
            this.websocket.send("CFG:" + this.jsroot.toJSON(cfg));
         }
      },

      /** @summary configuration handler changes,
        * @desc after short timeout send updated config to server  */
      configChanged() {
         if (this.config_tmout)
            clearTimeout(this.config_tmout);

         this.config_tmout = setTimeout(this.sendConfig.bind(this), 500);
      },

      processPainterChange(func, arg) {
         let painter = (this.node_painter_active && this.node_painter) ? this.node_painter : this.geo_painter;

         if (painter && (typeof painter[func] == 'function'))
            painter[func](arg);
      },

      lightChanged() {
         this.processPainterChange('changedLight');
      },

      sliderXchange() {
         this.processPainterChange('changedClipping', 0);
      },

      sliderYchange() {
         this.processPainterChange('changedClipping', 1);
      },

      sliderZchange() {
         this.processPainterChange('changedClipping', 2);
      },

      clipChanged() {
         this.processPainterChange('changedClipping', -1);
      },

      hightlightChanged() {
         this.processPainterChange('changedHighlight');
      },

      transparencyChange() {
         this.processPainterChange('changedGlobalTransparency');
      },

      wireframeChanged() {
         this.processPainterChange('changedWireFrame');
      },

      backgroundChanged(oEvent) {
         this.processPainterChange('changedBackground', oEvent.getParameter('value'));
      },

      axesChanged() {
         this.processPainterChange('changedAxes');
      },

      autorotateChanged() {
         this.processPainterChange('changedAutoRotate');
      },

      cameraReset() {
         this.processPainterChange('focusCamera');
      },

      depthTestChanged() {
         this.processPainterChange('changedDepthTest');
      },

      depthMethodChanged() {
         this.processPainterChange('changedDepthMethod');
      },

      sliderTransChange() {
         this.processPainterChange('changedTransformation');
      },

      pressTransReset() {
         this.processPainterChange('changedTransformation', 'reset');
      },

      pressReset() {
         this.processPainterChange('resetAdvanced');
         this.byId("geomControl").getModel().refresh();
      },

      ssaoChanged() {
         this.processPainterChange('changedSSAO');
      }
   });

});
