import { settings, create, parse, toJSON, loadScript, registerMethods, isBatchMode, isFunc, isStr, nsREX } from '../core.mjs';
import { select as d3_select, rgb as d3_rgb } from '../d3.mjs';
import { closeCurrentWindow, showProgress, loadOpenui5, ToolbarIcons, getColorExec } from '../gui/utils.mjs';
import { GridDisplay, getHPainter } from '../gui/display.mjs';
import { makeTranslate } from '../base/BasePainter.mjs';
import { selectActivePad, cleanup, resize, EAxisBits } from '../base/ObjectPainter.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';
import { RAxisPainter } from './RAxisPainter.mjs';
import { RFramePainter } from './RFramePainter.mjs';
import { RPadPainter } from './RPadPainter.mjs';
import { addDragHandler } from './TFramePainter.mjs';
import { WebWindowHandle } from '../webwindow.mjs';


/**
 * @summary Painter class for RCanvas
 *
 * @private
 */

class RCanvasPainter extends RPadPainter {

   /** @summary constructor */
   constructor(dom, canvas) {
      super(dom, canvas, true);
      this._websocket = null;
      this.tooltip_allowed = settings.Tooltip;
      this.v7canvas = true;
      if ((dom === null) && (canvas === null)) {
         // for web canvas details are important
         settings.SmallPad.width = 20;
         settings.SmallPad.height = 10;
      }
   }

   /** @summary Cleanup canvas painter */
   cleanup() {
      delete this._websocket;
      delete this._submreq;

     if (this._changed_layout)
         this.setLayoutKind('simple');
      delete this._changed_layout;

      super.cleanup();
   }

   /** @summary Returns layout kind */
   getLayoutKind() {
      const origin = this.selectDom('origin'),
         layout = origin.empty() ? '' : origin.property('layout');
      return layout || 'simple';
   }

   /** @summary Set canvas layout kind */
   setLayoutKind(kind, main_selector) {
      const origin = this.selectDom('origin');
      if (!origin.empty()) {
         if (!kind) kind = 'simple';
         origin.property('layout', kind);
         origin.property('layout_selector', (kind !== 'simple') && main_selector ? main_selector : null);
         this._changed_layout = (kind !== 'simple'); // use in cleanup
      }
   }

   /** @summary Changes layout
     * @return {Promise} indicating when finished */
   async changeLayout(layout_kind, mainid) {
      const current = this.getLayoutKind();
      if (current === layout_kind)
         return true;

      const origin = this.selectDom('origin'),
            sidebar2 = origin.select('.side_panel2'),
            lst = [];
      let sidebar = origin.select('.side_panel'),
          main = this.selectDom(), force;

      while (main.node().firstChild)
         lst.push(main.node().removeChild(main.node().firstChild));

      if (!sidebar.empty())
         cleanup(sidebar.node());
      if (!sidebar2.empty())
         cleanup(sidebar2.node());

      this.setLayoutKind('simple'); // restore defaults
      origin.html(''); // cleanup origin

      if (layout_kind === 'simple') {
         main = origin;
         for (let k = 0; k < lst.length; ++k)
            main.node().appendChild(lst[k]);
         this.setLayoutKind(layout_kind);
         force = true;
      } else {
         const grid = new GridDisplay(origin.node(), layout_kind);

         if (mainid === undefined)
            mainid = (layout_kind.indexOf('vert') === 0) ? 0 : 1;

         main = d3_select(grid.getGridFrame(mainid));
         main.classed('central_panel', true).style('position', 'relative');

         if (mainid === 2) {
            // left panel for Y
            sidebar = d3_select(grid.getGridFrame(0));
            sidebar.classed('side_panel2', true).style('position', 'relative');
            // bottom panel for X
            sidebar = d3_select(grid.getGridFrame(3));
            sidebar.classed('side_panel', true).style('position', 'relative');
         } else {
            sidebar = d3_select(grid.getGridFrame(1 - mainid));
            sidebar.classed('side_panel', true).style('position', 'relative');
         }

         // now append all childs to the new main
         for (let k = 0; k < lst.length; ++k)
            main.node().appendChild(lst[k]);

         this.setLayoutKind(layout_kind, '.central_panel');

         // remove reference to MDIDisplay, solves resize problem
         origin.property('mdi', null);
      }

      // resize main drawing and let draw extras
      resize(main.node(), force);
      return true;
   }

   /** @summary Toggle projection
     * @return {Promise} indicating when ready
     * @private */
   async toggleProjection(kind) {
      delete this.proj_painter;

      if (kind) this.proj_painter = { X: false, Y: false }; // just indicator that drawing can be preformed

      if (isFunc(this.showUI5ProjectionArea))
         return this.showUI5ProjectionArea(kind);

      let layout = 'simple', mainid;

      switch (kind) {
         case 'XY': layout = 'projxy'; mainid = 2; break;
         case 'X':
         case 'bottom': layout = 'vert2_31'; mainid = 0; break;
         case 'Y':
         case 'left': layout = 'horiz2_13'; mainid = 1; break;
         case 'top': layout = 'vert2_13'; mainid = 1; break;
         case 'right': layout = 'horiz2_31'; mainid = 0; break;
      }

      return this.changeLayout(layout, mainid);
   }

   /** @summary Draw projection for specified histogram
     * @private */
   async drawProjection(/* kind,hist,hopt */) {
      // dummy for the moment
      return false;
   }

   /** @summary Draw in side panel
     * @private */
   async drawInSidePanel(canv, opt, kind) {
      const sel = ((this.getLayoutKind() === 'projxy') && (kind === 'Y')) ? '.side_panel2' : '.side_panel',
            side = this.selectDom('origin').select(sel);
      return side.empty() ? null : this.drawObject(side.node(), canv, opt);
   }

   /** @summary Checks if canvas shown inside ui5 widget
     * @desc Function should be used only from the func which supposed to be replaced by ui5
     * @private */
   testUI5() {
      if (!this.use_openui) return false;
      console.warn('full ui5 should be used - not loaded yet? Please check!!');
      return true;
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
      const pnt = fname.indexOf('.');
      this.createImage(fname.slice(pnt+1))
          .then(res => this.sendWebsocket(`SAVE:${fname}:${res}`));
   }

   /** @summary Send command to server to save canvas with specified name
     * @desc Should be only used in web-based canvas
     * @private */
   sendSaveCommand(fname) {
      this.sendWebsocket('PRODUCE:' + fname);
   }

   /** @summary Send message via web socket
     * @private */
   sendWebsocket(msg) {
      if (this._websocket?.canSend()) {
         this._websocket.send(msg);
         return true;
      }

      return false;
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

   /** @summary set, test or reset timeout of specified name
     * @desc Used to prevent overloading of websocket for specific function */
   websocketTimeout(name, tm) {
      if (!this._websocket)
         return;
      if (!this._websocket._tmouts)
         this._websocket._tmouts = {};

      const handle = this._websocket._tmouts[name];
      if (tm === undefined)
         return handle !== undefined;

      if (tm === 'reset') {
         if (handle) { clearTimeout(handle); delete this._websocket._tmouts[name]; }
      } else if (!handle && Number.isInteger(tm))
         this._websocket._tmouts[name] = setTimeout(() => { delete this._websocket._tmouts[name]; }, tm);
   }

   /** @summary Hanler for websocket open event
     * @private */
   onWebsocketOpened(/* handle */) {
   }

   /** @summary Hanler for websocket close event
     * @private */
   onWebsocketClosed(/* handle */) {
      if (!this.embed_canvas)
         closeCurrentWindow();
   }

   /** @summary Hanler for websocket message
     * @private */
   onWebsocketMsg(handle, msg) {
      // console.log('GET_MSG ' + msg.slice(0,30));

      if (msg === 'CLOSE') {
         this.onWebsocketClosed();
         this.closeWebsocket(true);
      } else if (msg.slice(0, 5) === 'SNAP:') {
         msg = msg.slice(5);
         const p1 = msg.indexOf(':'),
             snapid = msg.slice(0, p1),
             snap = parse(msg.slice(p1+1));
         this.syncDraw(true)
             .then(() => {
                if (!this.snapid && snap?.fWinSize)
                   this.resizeBrowser(snap.fWinSize[0], snap.fWinSize[1]);
             }).then(() => this.redrawPadSnap(snap))
             .then(() => {
                 this.addPadInteractive();
                 handle.send(`SNAPDONE:${snapid}`); // send ready message back when drawing completed
                 this.confirmDraw();
              });
      } else if (msg.slice(0, 4) === 'JSON') {
         const obj = parse(msg.slice(4));
         // console.log('get JSON ', msg.length-4, obj._typename);
         this.redrawObject(obj);
      } else if (msg.slice(0, 9) === 'REPL_REQ:')
         this.processDrawableReply(msg.slice(9));
       else if (msg.slice(0, 4) === 'CMD:') {
         msg = msg.slice(4);
         const p1 = msg.indexOf(':'),
             cmdid = msg.slice(0, p1),
             cmd = msg.slice(p1+1),
             reply = `REPLY:${cmdid}:`;
         if ((cmd === 'SVG') || (cmd === 'PNG') || (cmd === 'JPEG')) {
            this.createImage(cmd.toLowerCase())
                .then(res => handle.send(reply + res));
         } else if (cmd.indexOf('ADDPANEL:') === 0) {
            const relative_path = cmd.slice(9);
            if (!isFunc(this.showUI5Panel))
               handle.send(reply + 'false');
             else {
               const conn = new WebWindowHandle(handle.kind);

               // set interim receiver until first message arrives
               conn.setReceiver({
                  cpainter: this,

                  onWebsocketOpened() {
                  },

                  onWebsocketMsg(panel_handle, msg) {
                     const panel_name = (msg.indexOf('SHOWPANEL:') === 0) ? msg.slice(10) : '';
                     this.cpainter.showUI5Panel(panel_name, panel_handle)
                                  .then(res => handle.send(reply + (res ? 'true' : 'false')));
                  },

                  onWebsocketClosed() {
                     // if connection failed,
                     handle.send(reply + 'false');
                  },

                  onWebsocketError() {
                     // if connection failed,
                     handle.send(reply + 'false');
                  }

               });

               let addr = handle.href;
               if (relative_path.indexOf('../') === 0) {
                  const ddd = addr.lastIndexOf('/', addr.length-2);
                  addr = addr.slice(0, ddd) + relative_path.slice(2);
               } else
                  addr += relative_path;

               // only when connection established, panel will be activated
               conn.connect(addr);
            }
         } else {
            console.log('Unrecognized command ' + cmd);
            handle.send(reply);
         }
      } else if ((msg.slice(0, 7) === 'DXPROJ:') || (msg.slice(0, 7) === 'DYPROJ:')) {
         const kind = msg[1],
             hist = parse(msg.slice(7));
         this.drawProjection(kind, hist);
      } else if (msg.slice(0, 5) === 'SHOW:') {
         const that = msg.slice(5),
             on = that[that.length-1] === '1';
         this.showSection(that.slice(0, that.length-2), on);
      } else
         console.log(`unrecognized msg len: ${msg.length} msg: ${msg.slice(0, 30)}`);
   }

   /** @summary Submit request to RDrawable object on server side */
   submitDrawableRequest(kind, req, painter, method) {
      if (!this._websocket || !req || !req._typename ||
          !painter.snapid || !isStr(painter.snapid)) return null;

      if (kind && method) {
         // if kind specified - check if such request already was submitted
         if (!painter._requests) painter._requests = {};

         const prevreq = painter._requests[kind];

         if (prevreq) {
            const tm = new Date().getTime();
            if (!prevreq._tm || (tm - prevreq._tm < 5000)) {
               prevreq._nextreq = req; // submit when got reply
               return false;
            }
            delete painter._requests[kind]; // let submit new request after timeout
         }

         painter._requests[kind] = req; // keep reference on the request
      }

      req.id = painter.snapid;

      if (method) {
         if (!this._nextreqid) this._nextreqid = 1;
         req.reqid = this._nextreqid++;
      } else
         req.reqid = 0; // request will not be replied


      const msg = JSON.stringify(req);

      if (req.reqid) {
         req._kind = kind;
         req._painter = painter;
         req._method = method;
         req._tm = new Date().getTime();

         if (!this._submreq) this._submreq = {};
         this._submreq[req.reqid] = req; // fast access to submitted requests
      }

      // console.log('Sending request ', msg.slice(0,60));

      this.sendWebsocket('REQ:' + msg);
      return req;
   }

   /** @summary Submit menu request
     * @private */
   async submitMenuRequest(painter, menukind, reqid) {
      return new Promise(resolveFunc => {
         this.submitDrawableRequest('', {
            _typename: `${nsREX}RDrawableMenuRequest`,
            menukind: menukind || '',
            menureqid: reqid // used to identify menu request
         }, painter, resolveFunc);
      });
   }

   /** @summary Submit executable command for given painter */
   submitExec(painter, exec, subelem) {
      // snapid is intentionally ignored - only painter.snapid has to be used
      if (!this._websocket) return;

      if (subelem && isStr(subelem)) {
         const len = subelem.length;
         if ((len > 2) && (subelem.indexOf('#x') === len - 2)) subelem = 'x'; else
         if ((len > 2) && (subelem.indexOf('#y') === len - 2)) subelem = 'y'; else
         if ((len > 2) && (subelem.indexOf('#z') === len - 2)) subelem = 'z';

         if ((subelem === 'x') || (subelem === 'y') || (subelem === 'z'))
            exec = subelem + 'axis#' + exec;
         else
            return console.log(`not recoginzed subelem ${subelem} in SubmitExec`);
       }

      this.submitDrawableRequest('', { _typename: `${nsREX}RDrawableExecRequest`, exec }, painter);
   }

   /** @summary Process reply from request to RDrawable */
   processDrawableReply(msg) {
      const reply = parse(msg);
      if (!reply || !reply.reqid || !this._submreq) return false;

      const req = this._submreq[reply.reqid];
      if (!req) return false;

      // remove reference first
      delete this._submreq[reply.reqid];

      // remove blocking reference for that kind
      if (req._kind && req._painter?._requests) {
         if (req._painter._requests[req._kind] === req)
            delete req._painter._requests[req._kind];
      }

      if (req._method)
         req._method(reply, req);

      // resubmit last request of that kind
      if (req._nextreq && !req._painter._requests[req._kind])
         this.submitDrawableRequest(req._kind, req._nextreq, req._painter, req._method);
   }

   /** @summary Show specified section in canvas */
   async showSection(that, on) {
      switch (that) {
         case 'Menu': break;
         case 'StatusBar': break;
         case 'Editor': break;
         case 'ToolBar': break;
         case 'ToolTips': this.setTooltipAllowed(on); break;
      }
      return true;
   }

   /** @summary Method informs that something was changed in the canvas
     * @desc used to update information on the server (when used with web6gui)
     * @private */
   processChanges(kind, painter, subelem) {
      // check if we could send at least one message more - for some meaningful actions
      if (!this._websocket || !this._websocket.canSend(2) || !isStr(kind)) return;

      const msg = '';
      if (!painter) painter = this;
      switch (kind) {
         case 'sbits':
            console.log('Status bits in RCanvas are changed - that to do?');
            break;
         case 'frame': // when moving frame
         case 'zoom':  // when changing zoom inside frame
            console.log('Frame moved or zoom is changed - that to do?');
            break;
         case 'pave_moved':
            console.log('TPave is moved inside RCanvas - that to do?');
            break;
         default:
            if ((kind.slice(0, 5) === 'exec:') && painter?.snapid)
               this.submitExec(painter, kind.slice(5), subelem);
             else
               console.log('UNPROCESSED CHANGES', kind);
      }

      if (msg)
         console.log('RCanvas::processChanges want to send  ' + msg.length + '  ' + msg.slice(0, 40));
   }

   /** @summary Handle pad button click event
     * @private */
   clickPadButton(funcname, evnt) {
      if (funcname === 'ToggleGed')
         return this.activateGed(this, null, 'toggle');
      if (funcname === 'ToggleStatus')
         return this.activateStatusBar('toggle');
      return super.clickPadButton(funcname, evnt);
   }

   /** @summary returns true when event status area exist for the canvas */
   hasEventStatus() {
      if (this.testUI5()) return false;
      if (this.brlayout)
         return this.brlayout.hasStatus();
      const hp = getHPainter();
      return hp ? hp.hasStatusLine() : false;
   }

   /** @summary Check if status bar can be toggled
     * @private */
   canStatusBar() {
      return this.testUI5() || this.brlayout || getHPainter();
   }

   /** @summary Show/toggle event status bar
     * @private */
   activateStatusBar(state) {
      if (this.testUI5())
         return;
      if (this.brlayout)
         this.brlayout.createStatusLine(23, state);
      else
         getHPainter()?.createStatusLine(23, state);

      this.processChanges('sbits', this);
   }

   /** @summary Show online canvas status
     * @private */
   showCanvasStatus(...msgs) {
      if (this.testUI5()) return;

      const br = this.brlayout || getHPainter()?.brlayout;

      br?.showStatus(...msgs);
   }

   /** @summary Returns true if GED is present on the canvas */
   hasGed() {
      if (this.testUI5()) return false;
      return this.brlayout?.hasContent() ?? false;
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
      this.brlayout?.deleteContent(true);
      this.processChanges('sbits', this);
   }

   /** @summary Get view data for ui5 panel
     * @private */
   getUi5PanelData(/* panel_name */) {
      return { jsroot: { settings, create, parse, toJSON, loadScript, EAxisBits, getColorExec } };
   }

   /** @summary Function used to activate GED
     * @return {Promise} when GED is there
     * @private */
   async activateGed(objpainter, kind, mode) {
      if (this.testUI5() || !this.brlayout)
         return false;

      if (this.brlayout.hasContent()) {
         if ((mode === 'toggle') || (mode === false))
            this.removeGed();
         else
            objpainter?.getPadPainter()?.selectObjectPainter(objpainter);

         return true;
      }

      if (mode === false)
         return false;

      const btns = this.brlayout.createBrowserBtns();

      ToolbarIcons.createSVG(btns, ToolbarIcons.diamand, 15, 'toggle fix-pos mode', 'browser')
                  .style('margin', '3px').on('click', () => this.brlayout.toggleKind('fix'));

      ToolbarIcons.createSVG(btns, ToolbarIcons.circle, 15, 'toggle float mode', 'browser')
                  .style('margin', '3px').on('click', () => this.brlayout.toggleKind('float'));

      ToolbarIcons.createSVG(btns, ToolbarIcons.cross, 15, 'delete GED', 'browser')
                  .style('margin', '3px').on('click', () => this.removeGed());

      // be aware, that jsroot_browser_hierarchy required for flexible layout that element use full browser area
      this.brlayout.setBrowserContent('<div class=\'jsroot_browser_hierarchy\' id=\'ged_placeholder\'>Loading GED ...</div>');
      this.brlayout.setBrowserTitle('GED');
      this.brlayout.toggleBrowserKind(kind || 'float');

      return new Promise(resolveFunc => {
         loadOpenui5.then(sap => {
            d3_select('#ged_placeholder').text('');

            sap.ui.require(['sap/ui/model/json/JSONModel', 'sap/ui/core/mvc/XMLView'], (JSONModel, XMLView) => {
               const oModel = new JSONModel({ handle: null });

               XMLView.create({
                  viewName: 'rootui5.canv.view.Ged',
                  viewData: this.getUi5PanelData('Ged')
               }).then(oGed => {
                  oGed.setModel(oModel);

                  oGed.placeAt('ged_placeholder');

                  this.ged_view = oGed;

                  // TODO: should be moved into Ged controller - it must be able to detect canvas painter itself
                  this.registerForPadEvents(oGed.getController().padEventsReceiver.bind(oGed.getController()));

                  objpainter?.getPadPainter()?.selectObjectPainter(objpainter);

                  this.processChanges('sbits', this);

                  resolveFunc(true);
               });
            });
         });
      });
   }

   /** @summary produce JSON for RCanvas, which can be used to display canvas once again
     * @private */
   produceJSON() {
      console.error('RCanvasPainter.produceJSON not yet implemented');
      return '';
   }

   /** @summary resize browser window to get requested canvas sizes */
   resizeBrowser(fullW, fullH) {
      if (!fullW || !fullH || this.isBatchMode() || this.embed_canvas || this.batch_mode)
         return;
      this._websocket?.resizeWindow(fullW, fullH);
   }

   /** @summary draw RCanvas object */
   static async draw(dom, can /*, opt */) {
      const nocanvas = !can;
      if (nocanvas)
         can = create(`${nsREX}RCanvas`);

      const painter = new RCanvasPainter(dom, can);
      painter.normal_canvas = !nocanvas;
      painter.createCanvasSvg(0);

      selectActivePad({ pp: painter, active: false });

      return painter.drawPrimitives().then(() => {
         painter.addPadInteractive();
         painter.addPadButtons();
         painter.showPadButtons();
         return painter;
      });
   }

} // class RCanvasPainter


/** @summary draw RPadSnapshot object
  * @private */
function drawRPadSnapshot(dom, snap /*, opt */) {
   const painter = new RCanvasPainter(dom, null);
   painter.normal_canvas = false;
   painter.batch_mode = isBatchMode();
   return painter.syncDraw(true).then(() => painter.redrawPadSnap(snap)).then(() => {
      painter.confirmDraw();
      painter.showPadButtons();
      return painter;
   });
}

/** @summary Ensure RCanvas and RFrame for the painter object
  * @param {Object} painter  - painter object to process
  * @param {string|boolean} frame_kind  - false for no frame or '3d' for special 3D mode
  * @desc Assigns DOM, creates and draw RCanvas and RFrame if necessary, add painter to pad list of painters
  * @return {Promise} for ready
  * @private */
async function ensureRCanvas(painter, frame_kind) {
   if (!painter)
      return Promise.reject(Error('Painter not provided in ensureRCanvas'));

   // simple check - if canvas there, can use painter
   const pr = painter.getCanvSvg().empty() ? RCanvasPainter.draw(painter.getDom(), null /* noframe */) : Promise.resolve(true);

   return pr.then(() => {
      if ((frame_kind !== false) && painter.getFrameSvg().selectChild('.main_layer').empty())
         return RFramePainter.draw(painter.getDom(), null, isStr(frame_kind) ? frame_kind : '');
   }).then(() => {
      painter.addToPadPrimitives();
      return painter;
   });
}


/** @summary Function used for direct draw of RFrameTitle
  * @private */
function drawRFrameTitle(reason, drag) {
   const fp = this.getFramePainter();
   if (!fp)
      return console.log('no frame painter - no title');

   const rect = fp.getFrameRect(),
         fx = rect.x,
         fy = rect.y,
         fw = rect.width,
         // fh           = rect.height,
         ph = this.getPadPainter().getPadHeight(),
         title = this.getObject(),
         title_width = fw,
         textFont = this.v7EvalFont('text', { size: 0.07, color: 'black', align: 22 });
   let title_margin = this.v7EvalLength('margin', ph, 0.02),
       title_height = this.v7EvalLength('height', ph, 0.05);

   if (reason === 'drag') {
      title_height = drag.height;
      title_margin = fy - drag.y - drag.height;
      const changes = {};
      this.v7AttrChange(changes, 'margin', title_margin / ph);
      this.v7AttrChange(changes, 'height', title_height / ph);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server
   }

   this.createG();

   makeTranslate(this.draw_g, fx, Math.round(fy-title_margin-title_height));

   const arg = { x: title_width/2, y: title_height/2, text: title.fText, latex: 1 };

   this.startTextDrawing(textFont, 'font');

   this.drawText(arg);

   return this.finishTextDrawing().then(() =>
      addDragHandler(this, { x: fx, y: Math.round(fy-title_margin-title_height), width: title_width, height: title_height,
                             minwidth: 20, minheight: 20, no_change_x: true, redraw: d => this.redraw('drag', d) })
   );
}

/// /////////////////////////////////////////////////////////////////////////////////////////

registerMethods(`${nsREX}RPalette`, {

   extractRColor(rcolor) {
     return rcolor.fColor || 'black';
   },

   getColor(indx) {
      return this.palette[indx];
   },

   getContourIndex(zc) {
      const cntr = this.fContour;
      let l = 0, r = cntr.length-1, mid;

      if (zc < cntr[0]) return -1;
      if (zc >= cntr[r]) return r-1;

      if (this.fCustomContour) {
         while (l < r-1) {
            mid = Math.round((l+r)/2);
            if (cntr[mid] > zc) r = mid; else l = mid;
         }
         return l;
      }

      // last color in palette starts from level cntr[r-1]
      return Math.floor((zc-cntr[0]) / (cntr[r-1] - cntr[0]) * (r-1));
   },

   getContourColor(zc) {
      const zindx = this.getContourIndex(zc);
      return (zindx < 0) ? '' : this.getColor(zindx);
   },

   getContour() {
      return this.fContour && (this.fContour.length > 1) ? this.fContour : null;
   },

   deleteContour() {
      delete this.fContour;
   },

   calcColor(value, entry1, entry2) {
      const dist = entry2.fOrdinal - entry1.fOrdinal,
          r1 = entry2.fOrdinal - value,
          r2 = value - entry1.fOrdinal;

      if (!this.fInterpolate || (dist <= 0))
         return (r1 < r2) ? entry2.fColor : entry1.fColor;

      // interpolate
      const col1 = d3_rgb(this.extractRColor(entry1.fColor)),
          col2 = d3_rgb(this.extractRColor(entry2.fColor)),
          color = d3_rgb(Math.round((col1.r*r1 + col2.r*r2)/dist),
                         Math.round((col1.g*r1 + col2.g*r2)/dist),
                         Math.round((col1.b*r1 + col2.b*r2)/dist));

      return color.toString();
   },

   createPaletteColors(len) {
      const arr = [];
      let indx = 0;

      while (arr.length < len) {
         const value = arr.length / (len-1),

          entry = this.fColors[indx];

         if ((Math.abs(entry.fOrdinal - value) < 0.0001) || (indx === this.fColors.length - 1)) {
            arr.push(this.extractRColor(entry.fColor));
            continue;
         }

         const next = this.fColors[indx+1];
         if (next.fOrdinal <= value)
            indx++;
         else
            arr.push(this.calcColor(value, entry, next));
      }

      return arr;
   },

   getColorOrdinal(value) {
      // extract color with ordinal value between 0 and 1
      if (!this.fColors)
         return 'black';
      if ((typeof value !== 'number') || (value < 0))
         value = 0;
      else if (value > 1)
         value = 1;

      // TODO: implement better way to find index

      let entry, next = this.fColors[0];
      for (let indx = 0; indx < this.fColors.length-1; ++indx) {
         entry = next;

         if (Math.abs(entry.fOrdinal - value) < 0.0001)
            return this.extractRColor(entry.fColor);

         next = this.fColors[indx+1];
         if (next.fOrdinal > value)
            return this.calcColor(value, entry, next);
      }

      return this.extractRColor(next.fColor);
   },

   setFullRange(min, max) {
      // set full z scale range, used in zooming
      this.full_min = min;
      this.full_max = max;
   },

   createContour(logz, nlevels, zmin, zmax, zminpositive) {
      this.fContour = [];
      delete this.fCustomContour;
      this.colzmin = zmin;
      this.colzmax = zmax;

      if (logz) {
         if (this.colzmax <= 0) this.colzmax = 1.0;
         if (this.colzmin <= 0) {
            if ((zminpositive === undefined) || (zminpositive <= 0))
               this.colzmin = 0.0001*this.colzmax;
            else
               this.colzmin = ((zminpositive < 3) || (zminpositive>100)) ? 0.3*zminpositive : 1;
         }
         if (this.colzmin >= this.colzmax)
            this.colzmin = 0.0001*this.colzmax;

         const logmin = Math.log(this.colzmin)/Math.log(10),
             logmax = Math.log(this.colzmax)/Math.log(10),
             dz = (logmax-logmin)/nlevels;
         this.fContour.push(this.colzmin);
         for (let level=1; level<nlevels; level++)
            this.fContour.push(Math.exp((logmin + dz*level)*Math.log(10)));
         this.fContour.push(this.colzmax);
         this.fCustomContour = true;
      } else {
         if ((this.colzmin === this.colzmax) && (this.colzmin !== 0)) {
            this.colzmax += 0.01*Math.abs(this.colzmax);
            this.colzmin -= 0.01*Math.abs(this.colzmin);
         }
         const dz = (this.colzmax-this.colzmin)/nlevels;
         for (let level=0; level<=nlevels; level++)
            this.fContour.push(this.colzmin + dz*level);
      }

      if (!this.palette || (this.palette.length !== nlevels))
         this.palette = this.createPaletteColors(nlevels);
   }

});

/** @summary draw RFont object
  * @private */
function drawRFont() {
   const font = this.getObject(),
         svg = this.getCanvSvg(),
         clname = 'custom_font_' + font.fFamily+font.fWeight+font.fStyle;
   let defs = svg.selectChild('.canvas_defs');

   if (defs.empty())
      defs = svg.insert('svg:defs', ':first-child').attr('class', 'canvas_defs');

   let entry = defs.selectChild('.' + clname);
   if (entry.empty()) {
      entry = defs.append('style')
                  .attr('type', 'text/css')
                  .attr('class', clname)
                  .text(`@font-face { font-family: "${font.fFamily}"; font-weight: ${font.fWeight ? font.fWeight : 'normal'}; font-style: ${font.fStyle ? font.fStyle : 'normal'}; src: ${font.fSrc}; }`);
      const p1 = font.fSrc.indexOf('base64,'),
            p2 = font.fSrc.lastIndexOf(' format(');
      if (p1 > 0 && p2 > p1) {
         const base64 = font.fSrc.slice(p1 + 7, p2 - 2),
               is_ttf = font.fSrc.indexOf('data:application/font-ttf') > 0;
         // TODO: for the moment only ttf format supported by jsPDF
         if (is_ttf)
            entry.property('$fonthandler', { name: font.fFamily, format: 'ttf', base64 });
      }
   }

   if (font.fDefault)
      this.getPadPainter()._dfltRFont = font;

   return true;
}

/** @summary draw RAxis object
  * @private */
function drawRAxis(dom, obj, opt) {
   const painter = new RAxisPainter(dom, obj, opt);
   painter.disable_zooming = true;
   return ensureRCanvas(painter, false)
           .then(() => painter.redraw())
           .then(() => painter);
}

/** @summary draw RFrame object
  * @private */
function drawRFrame(dom, obj, opt) {
   const p = new RFramePainter(dom, obj);
   if (opt === '3d') p.mode3d = true;
   return ensureRCanvas(p, false).then(() => p.redraw());
}

export { ensureRCanvas, drawRPadSnapshot,
         drawRFrameTitle, drawRFont, drawRAxis, drawRFrame,
         RObjectPainter, RPadPainter, RCanvasPainter };
