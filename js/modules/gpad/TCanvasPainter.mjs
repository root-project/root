import { BIT, settings, create, parse, toJSON, isBatchMode } from '../core.mjs';
import { select as d3_select } from '../d3.mjs';
import { closeCurrentWindow, showProgress, loadOpenui5, ToolbarIcons } from '../gui/utils.mjs';
import { GridDisplay, getHPainter } from '../gui/display.mjs';
import { cleanup, resize, selectActivePad } from '../base/ObjectPainter.mjs';
import { TAxisPainter } from './TAxisPainter.mjs';
import { TFramePainter } from './TFramePainter.mjs';
import { TPadPainter } from './TPadPainter.mjs';


/** @summary direct draw of TFrame object,
  * @desc pad or canvas should already exist
  * @private */
function directDrawTFrame(dom, obj, opt) {
   let fp = new TFramePainter(dom, obj);
   fp.addToPadPrimitives();
   if (opt == "3d") fp.mode3d = true;
   return fp.redraw();
}

const TCanvasStatusBits = {
   kShowEventStatus: BIT(15),
   kAutoExec: BIT(16),
   kMenuBar: BIT(17),
   kShowToolBar: BIT(18),
   kShowEditor: BIT(19),
   kMoveOpaque: BIT(20),
   kResizeOpaque: BIT(21),
   kIsGrayscale: BIT(22),
   kShowToolTips: BIT(23)
};

/**
  * @summary Painter for TCanvas object
  *
  * @private
  */

class TCanvasPainter extends TPadPainter {

   /** @summary Constructor */
   constructor(dom, canvas) {
      super(dom, canvas, true);
      this._websocket = null;
      this.tooltip_allowed = settings.Tooltip;
   }

   /** @summary Cleanup canvas painter */
   cleanup() {
      if (this._changed_layout)
         this.setLayoutKind('simple');
      delete this._changed_layout;
      super.cleanup();
   }

   /** @summary Returns layout kind */
   getLayoutKind() {
      let origin = this.selectDom('origin'),
         layout = origin.empty() ? "" : origin.property('layout');

      return layout || 'simple';
   }

   /** @summary Set canvas layout kind */
   setLayoutKind(kind, main_selector) {
      let origin = this.selectDom('origin');
      if (!origin.empty()) {
         if (!kind) kind = 'simple';
         origin.property('layout', kind);
         origin.property('layout_selector', (kind != 'simple') && main_selector ? main_selector : null);
         this._changed_layout = (kind !== 'simple'); // use in cleanup
      }
   }

   /** @summary Changes layout
     * @returns {Promise} indicating when finished */
   changeLayout(layout_kind, mainid) {
      let current = this.getLayoutKind();
      if (current == layout_kind)
         return Promise.resolve(true);

      let origin = this.selectDom('origin'),
          sidebar = origin.select('.side_panel'),
          main = this.selectDom(), lst = [];

      while (main.node().firstChild)
         lst.push(main.node().removeChild(main.node().firstChild));

      if (!sidebar.empty()) cleanup(sidebar.node());

      this.setLayoutKind("simple"); // restore defaults
      origin.html(""); // cleanup origin

      if (layout_kind == 'simple') {
         main = origin;
         for (let k = 0; k < lst.length; ++k)
            main.node().appendChild(lst[k]);
         this.setLayoutKind(layout_kind);
      } else {

         let grid = new GridDisplay(origin.node(), layout_kind);

         if (mainid == undefined)
            mainid = (layout_kind.indexOf("vert") == 0) ? 0 : 1;

         main = d3_select(grid.getGridFrame(mainid));
         sidebar = d3_select(grid.getGridFrame(1 - mainid));

         main.classed("central_panel", true).style('position', 'relative');
         sidebar.classed("side_panel", true).style('position', 'relative');

         // now append all childs to the new main
         for (let k = 0; k < lst.length; ++k)
            main.node().appendChild(lst[k]);

         this.setLayoutKind(layout_kind, ".central_panel");

         // remove reference to MDIDisplay, solves resize problem
         origin.property('mdi', null);
      }

      // resize main drawing and let draw extras
      resize(main.node());
      return Promise.resolve(true);
   }

   /** @summary Toggle projection
     * @returns {Promise} indicating when ready
     * @private */
   toggleProjection(kind) {
      delete this.proj_painter;

      if (kind) this.proj_painter = 1; // just indicator that drawing can be preformed

      if (this.showUI5ProjectionArea)
         return this.showUI5ProjectionArea(kind);

      let layout = 'simple', mainid;

      switch(kind) {
         case "X":
         case "bottom": layout = 'vert2_31'; mainid = 0; break;
         case "Y":
         case "left": layout = 'horiz2_13'; mainid = 1; break;
         case "top": layout = 'vert2_13'; mainid = 1; break;
         case "right": layout = 'horiz2_31'; mainid = 0; break;
      }

      return this.changeLayout(layout, mainid);
   }

   /** @summary Draw projection for specified histogram
     * @private */
   drawProjection(kind, hist, hopt) {

      if (!this.proj_painter) return; // ignore drawing if projection not configured

      if (hopt === undefined) hopt = "hist";

      if (this.proj_painter === 1) {

         let canv = create("TCanvas"),
             pad = this.pad,
             main = this.getFramePainter(), drawopt;

         if (kind == "X") {
            canv.fLeftMargin = pad.fLeftMargin;
            canv.fRightMargin = pad.fRightMargin;
            canv.fLogx = main.logx;
            canv.fUxmin = main.logx ? Math.log10(main.scale_xmin) : main.scale_xmin;
            canv.fUxmax = main.logx ? Math.log10(main.scale_xmax) : main.scale_xmax;
            drawopt = "fixframe";
         } else if (kind == "Y") {
            canv.fBottomMargin = pad.fBottomMargin;
            canv.fTopMargin = pad.fTopMargin;
            canv.fLogx = main.logy;
            canv.fUxmin = main.logy ? Math.log10(main.scale_ymin) : main.scale_ymin;
            canv.fUxmax = main.logy ? Math.log10(main.scale_ymax) : main.scale_ymax;
            drawopt = "rotate";
         }

         canv.fPrimitives.Add(hist, hopt);

         let promise = this.drawInUI5ProjectionArea
                       ? this.drawInUI5ProjectionArea(canv, drawopt)
                       : this.drawInSidePanel(canv, drawopt);

         promise.then(painter => { this.proj_painter = painter; });
      } else {
         let hp = this.proj_painter.getMainPainter();
         if (hp) hp.updateObject(hist, hopt);
         this.proj_painter.redrawPad();
      }
   }

   /** @summary Checks if canvas shown inside ui5 widget
     * @desc Function should be used only from the func which supposed to be replaced by ui5
     * @private */
   testUI5() {
      if (!this.use_openui) return false;
      console.warn("full ui5 should be used - not loaded yet? Please check!!");
      return true;
   }

   /** @summary Draw in side panel
     * @private */
   drawInSidePanel(canv, opt) {
      let side = this.selectDom('origin').select(".side_panel");
      if (side.empty()) return Promise.resolve(null);
      return this.drawObject(side.node(), canv, opt);
   }

   /** @summary Show message
     * @desc Used normally with web-based canvas and handled in ui5
     * @private */
   showMessage(msg) {
      if (!this.testUI5())
         showProgress(msg, 7000);
   }

   /** @summary Function called when canvas menu item Save is called */
   saveCanvasAsFile(fname) {
      let pnt = fname.indexOf(".");
      this.createImage(fname.slice(pnt+1))
          .then(res => this.sendWebsocket("SAVE:" + fname + ":" + res));
   }

   /** @summary Send command to server to save canvas with specified name
     * @desc Should be only used in web-based canvas
     * @private */
   sendSaveCommand(fname) {
      this.sendWebsocket("PRODUCE:" + fname);
   }

   /** @summary Submit menu request
     * @private */
   submitMenuRequest(painter, kind, reqid) {
      // only single request can be handled, no limit better in RCanvas
      return new Promise(resolveFunc => {
         this._getmenu_callback = resolveFunc;
         this.sendWebsocket('GETMENU:' + reqid); // request menu items for given painter
      });
   }

   /** @summary Submit object exec request
     * @private */
   submitExec(painter, exec, snapid) {
      if (this._readonly || !painter) return;

      if (!snapid) snapid = painter.snapid;
      if (!snapid || (typeof snapid != 'string')) return;

      this.sendWebsocket("OBJEXEC:" + snapid + ":" + exec);
   }

   /** @summary Send text message with web socket
     * @desc used for communication with server-side of web canvas
     * @private */
   sendWebsocket(msg) {
      if (!this._websocket) return;
      if (this._websocket.canSend())
         this._websocket.send(msg);
      else
         console.warn("DROP SEND: " + msg);
   }

   /** @summary Close websocket connection to canvas
     * @private */
   closeWebsocket(force) {
      if (this._websocket) {
         this._websocket.close(force);
         this._websocket.cleanup();
         delete this._websocket;
      }
   }

   /** @summary Use provided connection for the web canvas
     * @private */
   useWebsocket(handle) {
      this.closeWebsocket();

      this._websocket = handle;
      this._websocket.setReceiver(this);
      this._websocket.connect();
   }

   /** @summary Hanler for websocket open event
     * @private */
   onWebsocketOpened(/*handle*/) {
      // indicate that we are ready to recieve any following commands
   }

   /** @summary Hanler for websocket close event
     * @private */
   onWebsocketClosed(/*handle*/) {
      if (!this.embed_canvas)
         closeCurrentWindow();
   }

   /** @summary Handle websocket messages
     * @private */
   onWebsocketMsg(handle, msg) {
      console.log("GET MSG len:" + msg.length + " " + msg.slice(0,60));

      if (msg == "CLOSE") {
         this.onWebsocketClosed();
         this.closeWebsocket(true);
      } else if (msg.slice(0,6)=='SNAP6:') {
         // This is snapshot, produced with ROOT6

         let snap = parse(msg.slice(6));

         this.syncDraw(true).then(() => this.redrawPadSnap(snap)).then(() => {
            this.completeCanvasSnapDrawing();
            let ranges = this.getWebPadOptions(); // all data, including subpads
            if (ranges) ranges = ":" + ranges;
            handle.send("READY6:" + snap.fVersion + ranges); // send ready message back when drawing completed
            this.confirmDraw();
         });
      } else if (msg.slice(0,5)=='MENU:') {
         // this is menu with exact identifier for object
         let lst = parse(msg.slice(5));
         if (typeof this._getmenu_callback == 'function') {
            this._getmenu_callback(lst);
            delete this._getmenu_callback;
         }
      } else if (msg.slice(0,4)=='CMD:') {
         msg = msg.slice(4);
         let p1 = msg.indexOf(":"),
             cmdid = msg.slice(0,p1),
             cmd = msg.slice(p1+1),
             reply = "REPLY:" + cmdid + ":";
         if ((cmd == "SVG") || (cmd == "PNG") || (cmd == "JPEG")) {
            this.createImage(cmd.toLowerCase())
                .then(res => handle.send(reply + res));
         } else {
            console.log('Unrecognized command ' + cmd);
            handle.send(reply);
         }
      } else if ((msg.slice(0,7)=='DXPROJ:') || (msg.slice(0,7)=='DYPROJ:')) {
         let kind = msg[1],
             hist = parse(msg.slice(7));
         this.drawProjection(kind, hist);
      } else if (msg.slice(0,5)=='SHOW:') {
         let that = msg.slice(5),
             on = (that[that.length-1] == '1');
         this.showSection(that.slice(0,that.length-2), on);
      } else if (msg.slice(0,5) == "EDIT:") {
         let obj_painter = this.findSnap(msg.slice(5));
         console.log('GET EDIT ' + msg.slice(5) +  ' found ' + !!obj_painter);
         if (obj_painter)
            this.showSection("Editor", true)
                .then(() => this.producePadEvent("select", obj_painter.getPadPainter(), obj_painter));

      } else {
         console.log("unrecognized msg " + msg);
      }
   }

   /** @summary Handle pad button click event */
   clickPadButton(funcname, evnt) {
      if (funcname == "ToggleGed") return this.activateGed(this, null, "toggle");
      if (funcname == "ToggleStatus") return this.activateStatusBar("toggle");
      super.clickPadButton(funcname, evnt);
   }

   /** @summary Returns true if event status shown in the canvas */
   hasEventStatus() {
      if (this.testUI5())
         return false;
      if (this.brlayout)
         return this.brlayout.hasStatus();
      let hp = getHPainter();
      if (hp)
         return hp.hasStatusLine();
      return false;
   }

   /** @summary Show/toggle event status bar
     * @private */
   activateStatusBar(state) {
      if (this.testUI5()) return;
      if (this.brlayout) {
         this.brlayout.createStatusLine(23, state);
      } else {
         let hp = getHPainter();
         if (hp) hp.createStatusLine(23, state);
      }
      this.processChanges("sbits", this);
   }

   /** @summary Returns true if GED is present on the canvas */
   hasGed() {
      if (this.testUI5()) return false;
      return this.brlayout ? this.brlayout.hasContent() : false;
   }

   /** @summary Function used to de-activate GED
     * @private */
   removeGed() {
      if (this.testUI5()) return;

      this.registerForPadEvents(null);

      if (this.ged_view) {
         this.ged_view.getController().cleanupGed();
         this.ged_view.destroy();
         delete this.ged_view;
      }
      if (this.brlayout)
         this.brlayout.deleteContent();

      this.processChanges("sbits", this);
   }

   /** @summary Function used to activate GED
     * @returns {Promise} when GED is there
     * @private */
   activateGed(objpainter, kind, mode) {
      if (this.testUI5() || !this.brlayout)
         return Promise.resolve(false);

      if (this.brlayout.hasContent()) {
         if ((mode === "toggle") || (mode === false)) {
            this.removeGed();
         } else {
            let pp = objpainter ? objpainter.getPadPainter() : null;
            if (pp) pp.selectObjectPainter(objpainter);
         }

         return Promise.resolve(true);
      }

      if (mode === false)
         return Promise.resolve(false);

      let btns = this.brlayout.createBrowserBtns();

      ToolbarIcons.createSVG(btns, ToolbarIcons.diamand, 15, "toggle fix-pos mode")
                  .style("margin","3px").on("click", () => this.brlayout.toggleKind('fix'));

      ToolbarIcons.createSVG(btns, ToolbarIcons.circle, 15, "toggle float mode")
                  .style("margin","3px").on("click", () => this.brlayout.toggleKind('float'));

      ToolbarIcons.createSVG(btns, ToolbarIcons.cross, 15, "delete GED")
                  .style("margin","3px").on("click", () => this.removeGed());

      // be aware, that jsroot_browser_hierarchy required for flexible layout that element use full browser area
      this.brlayout.setBrowserContent("<div class='jsroot_browser_hierarchy' id='ged_placeholder'>Loading GED ...</div>");
      this.brlayout.setBrowserTitle("GED");
      this.brlayout.toggleBrowserKind(kind || "float");

      return new Promise(resolveFunc => {

         loadOpenui5().then(sap => {

            d3_select("#ged_placeholder").text("");

            sap.ui.define(["sap/ui/model/json/JSONModel", "sap/ui/core/mvc/XMLView"], (JSONModel,XMLView) => {

               let oModel = new JSONModel({ handle: null });

               XMLView.create({
                  viewName: "rootui5.canv.view.Ged"
               }).then(oGed => {

                  oGed.setModel(oModel);

                  oGed.placeAt("ged_placeholder");

                  this.ged_view = oGed;

                  // TODO: should be moved into Ged controller - it must be able to detect canvas painter itself
                  this.registerForPadEvents(oGed.getController().padEventsReceiver.bind(oGed.getController()));

                  let pp = objpainter ? objpainter.getPadPainter() : null;
                  if (pp) pp.selectObjectPainter(objpainter);

                  this.processChanges("sbits", this);

                  resolveFunc(true);
               });
            });
         });
      });
   }

   /** @summary Show section of canvas  like menu or editor */
   showSection(that, on) {
      if (this.testUI5())
         return Promise.resolve(false);

      console.log(`Show section ${that} flag = ${on}`);

      switch(that) {
         case "Menu": break;
         case "StatusBar": this.activateStatusBar(on); break;
         case "Editor": return this.activateGed(this, null, !!on);
         case "ToolBar": break;
         case "ToolTips": this.setTooltipAllowed(on); break;

      }
      return Promise.resolve(true);
   }

   /** @summary Complete handling of online canvas drawing
     * @private */
   completeCanvasSnapDrawing() {
      if (!this.pad) return;

      if (document && !this.embed_canvas && this._websocket)
         document.title = this.pad.fTitle;

      if (this._all_sections_showed) return;
      this._all_sections_showed = true;
      this.showSection("Menu", this.pad.TestBit(TCanvasStatusBits.kMenuBar));
      this.showSection("StatusBar", this.pad.TestBit(TCanvasStatusBits.kShowEventStatus));
      this.showSection("ToolBar", this.pad.TestBit(TCanvasStatusBits.kShowToolBar));
      this.showSection("Editor", this.pad.TestBit(TCanvasStatusBits.kShowEditor));
      this.showSection("ToolTips", this.pad.TestBit(TCanvasStatusBits.kShowToolTips) || this._highlight_connect);
   }

   /** @summary Handle highlight in canvas - delver information to server
     * @private */
   processHighlightConnect(hints) {
      if (!hints || hints.length == 0 || !this._highlight_connect ||
           !this._websocket || this.doingDraw() || !this._websocket.canSend(2)) return;

      let hint = hints[0] || hints[1];
      if (!hint || !hint.painter || !hint.painter.snapid || !hint.user_info) return;
      let pp = hint.painter.getPadPainter() || this;
      if (!pp.snapid) return;

      let arr = [pp.snapid, hint.painter.snapid, "0", "0"];

      if ((hint.user_info.binx !== undefined) && (hint.user_info.biny !== undefined)) {
         arr[2] = hint.user_info.binx.toString();
         arr[3] = hint.user_info.biny.toString();
      }  else if (hint.user_info.bin !== undefined) {
         arr[2] = hint.user_info.bin.toString();
      }

      let msg = JSON.stringify(arr);

      if (this._last_highlight_msg != msg) {
         this._last_highlight_msg = msg;
         this.sendWebsocket("HIGHLIGHT:" + msg);
      }
   }

   /** @summary Method informs that something was changed in the canvas
     * @desc used to update information on the server (when used with web6gui)
     * @private */
   processChanges(kind, painter, subelem) {
      // check if we could send at least one message more - for some meaningful actions
      if (!this._websocket || this._readonly || !this._websocket.canSend(2) || (typeof kind !== "string")) return;

      let msg = "";
      if (!painter) painter = this;
      switch (kind) {
         case "sbits":
            msg = "STATUSBITS:" + this.getStatusBits();
            break;
         case "frame": // when moving frame
         case "zoom":  // when changing zoom inside frame
            if (!painter.getWebPadOptions)
               painter = painter.getPadPainter();
            if (typeof painter.getWebPadOptions == "function")
               msg = "OPTIONS6:" + painter.getWebPadOptions("only_this");
            break;
         case "pave_moved":
            if (painter.fillWebObjectOptions) {
               let info = painter.fillWebObjectOptions();
               if (info) msg = "PRIMIT6:" + toJSON(info);
            }
            break;
         default:
            if ((kind.slice(0,5) == "exec:") && painter && painter.snapid) {
               console.log('Call exec', painter.snapid);

               msg = "PRIMIT6:" + toJSON({
                  _typename: "TWebObjectOptions",
                  snapid: painter.snapid.toString() + (subelem ? "#"+subelem : ""),
                  opt: kind.slice(5),
                  fcust: "exec",
                  fopt: []
               });
            } else {
               console.log("UNPROCESSED CHANGES", kind);
            }
      }

      if (msg) {
         console.log("Sending " + msg.length + "  " + msg.slice(0,40));
         this._websocket.send(msg);
      }
   }

   /** @summary Select active pad on the canvas */
   selectActivePad(pad_painter, obj_painter, click_pos) {
      if ((this.snapid === undefined) || !pad_painter) return; // only interactive canvas

      let arg = null, ischanged = false;

      if ((pad_painter.snapid !== undefined) && this._websocket)
         arg = { _typename: "TWebPadClick", padid: pad_painter.snapid.toString(), objid: "", x: -1, y: -1, dbl: false };

      if (!pad_painter.is_active_pad) {
         ischanged = true;
         this.forEachPainterInPad(pp => pp.drawActiveBorder(null, pp === pad_painter), "pads");
      }

      if (obj_painter && (obj_painter.snapid!==undefined) && arg) {
         ischanged = true;
         arg.objid = obj_painter.snapid.toString();
      }

      if (click_pos && arg) {
         ischanged = true;
         arg.x = Math.round(click_pos.x || 0);
         arg.y = Math.round(click_pos.y || 0);
         if (click_pos.dbl) arg.dbl = true;
      }

      if (arg && ischanged)
         this.sendWebsocket("PADCLICKED:" + toJSON(arg));
   }

   /** @summary Return actual TCanvas status bits  */
   getStatusBits() {
      let bits = 0;
      if (this.hasEventStatus()) bits |= TCanvasStatusBits.kShowEventStatus;
      if (this.hasGed()) bits |= TCanvasStatusBits.kShowEditor;
      if (this.isTooltipAllowed()) bits |= TCanvasStatusBits.kShowToolTips;
      if (this.use_openui) bits |= TCanvasStatusBits.kMenuBar;
      return bits;
   }

   /** @summary produce JSON for TCanvas, which can be used to display canvas once again */
   produceJSON() {

      let canv = this.getObject(),
          fill0 = (canv.fFillStyle == 0);

      if (fill0) canv.fFillStyle = 1001;

      if (!this.normal_canvas) {

         // fill list of primitives from painters
         this.forEachPainterInPad(p => {
            if (p.$secondary) return; // ignore all secoandry painters

            let subobj = p.getObject();
            if (subobj && subobj._typename)
               canv.fPrimitives.Add(subobj, p.getDrawOpt());
         }, "objects");
      }

      let res = toJSON(canv);

      if (fill0) canv.fFillStyle = 0;

      if (!this.normal_canvas)
         canv.fPrimitives.Clear();

      return res;
   }

   /** @summary draw TCanvas */
   static draw(dom, can, opt) {
      let nocanvas = !can;
      if (nocanvas) can = create("TCanvas");

      let painter = new TCanvasPainter(dom, can);
      painter.checkSpecialsInPrimitives(can);

      if (!nocanvas && can.fCw && can.fCh && !isBatchMode()) {
         let rect0 = painter.selectDom().node().getBoundingClientRect();
         if (!rect0.height && (rect0.width > 0.1*can.fCw)) {
            painter.selectDom().style("width", can.fCw+"px").style("height", can.fCh+"px");
            painter._fixed_size = true;
         }
      }

      painter.decodeOptions(opt);
      painter.normal_canvas = !nocanvas;
      painter.createCanvasSvg(0);

      painter.addPadButtons();

      if (nocanvas && opt.indexOf("noframe") < 0)
         directDrawTFrame(dom, null);

      // select global reference - required for keys handling
      selectActivePad({ pp: painter, active: true });

      return painter.drawPrimitives().then(() => {
         painter.showPadButtons();
         return painter;
      });
   }

} // class TCanvasPainter


/** @summary Ensure TCanvas and TFrame for the painter object
  * @param {Object} painter  - painter object to process
  * @param {string|boolean} frame_kind  - false for no frame or "3d" for special 3D mode
  * @desc Assign dom, creates TCanvas if necessary, add to list of pad painters */
function ensureTCanvas(painter, frame_kind) {
   if (!painter)
      return Promise.reject(Error('Painter not provided in ensureTCanvas'));

   // simple check - if canvas there, can use painter
   let noframe = (frame_kind === false) || (frame_kind == "3d") ? "noframe" : "",
       promise = painter.getCanvSvg().empty()
                 ? TCanvasPainter.draw(painter.getDom(), null, noframe)
                 : Promise.resolve(true);

   return promise.then(() => {
      if ((frame_kind !== false) &&  painter.getFrameSvg().select(".main_layer").empty() && !painter.getFramePainter())
         directDrawTFrame(painter.getDom(), null, frame_kind);

      painter.addToPadPrimitives();
      return painter;
   });
}

/** @summary draw TPad snapshot from TWebCanvas
  * @private */
function drawTPadSnapshot(dom, snap /*, opt*/) {
   let can = create("TCanvas"),
       painter = new TCanvasPainter(dom, can);
   painter.normal_canvas = false;
   painter.addPadButtons();

   return painter.syncDraw(true).then(() => painter.redrawPadSnap(snap)).then(() => {
      painter.confirmDraw();
      painter.showPadButtons();
      return painter;
   });
}

/** @summary draw TGaxis object
  * @private */
function drawTGaxis(dom, obj, opt) {
   let painter = new TAxisPainter(dom, obj, false);
   painter.disable_zooming = true;
   return ensureTCanvas(painter, false).then(() => {
      if (opt) painter.convertTo(opt);
      return painter.redraw();
   }).then(() => painter);
}

/** @summary draw TGaxis object
  * @private */
function drawTFrame(dom, obj, opt) {
   let fp = new TFramePainter(dom, obj);
   return ensureTCanvas(fp, false).then(() => {
      if (opt == "3d") fp.mode3d = true;
      return fp.redraw();
   });
}

export { ensureTCanvas, drawTPadSnapshot, drawTGaxis, drawTFrame,
         TPadPainter, TCanvasPainter };
