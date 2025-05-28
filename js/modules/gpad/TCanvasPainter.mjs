import { BIT, settings, gStyle, internals, browser, create, parse, toJSON, loadScript, isFunc, isStr, clTCanvas } from '../core.mjs';
import { select as d3_select } from '../d3.mjs';
import { closeCurrentWindow, showProgress, loadOpenui5, ToolbarIcons, getColorExec } from '../gui/utils.mjs';
import { GridDisplay, getHPainter } from '../gui/display.mjs';
import { cleanup, resize, selectActivePad, EAxisBits, getDomCanvasPainter } from '../base/ObjectPainter.mjs';
import { TFramePainter } from './TFramePainter.mjs';
import { TPadPainter, clTButton, createWebObjectOptions } from './TPadPainter.mjs';

const kShowEventStatus = BIT(15),
     // kAutoExec = BIT(16),
      kMenuBar = BIT(17),
      kShowToolBar = BIT(18),
      kShowEditor = BIT(19),
     // kMoveOpaque = BIT(20),
     // kResizeOpaque = BIT(21),
     // kIsGrayscale = BIT(22),
      kShowToolTips = BIT(23);

/** @summary direct draw of TFrame object,
  * @desc pad or canvas should already exist
  * @private */
function directDrawTFrame(dom, obj, opt) {
   const fp = new TFramePainter(dom, obj);
   fp.addToPadPrimitives();
   if (opt === '3d') fp.mode3d = true;
   return fp.redraw();
}

/**
  * @summary Painter for TCanvas object
  *
  * @private
  */

class TCanvasPainter extends TPadPainter {

   #websocket; // WebWindow handle used for communication with server
   #changed_layout; // modified layout
   #getmenu_callback;  // function called when menu items get from server
   #online_fixed_size; // when size fixed for online canvas
   #all_sections_showed; // set once after online canvas drawn
   #last_highlight_msg; // last highligh msg send to server

   /** @summary Constructor */
   constructor(dom, canvas, opt, kind = true) {
      super(dom, canvas, opt, kind);
      this.#websocket = null;
      this.tooltip_allowed = settings.Tooltip;
   }

   /** @summary Cleanup canvas painter */
   cleanup() {
      if (this.#changed_layout)
         this.setLayoutKind('simple');
      this.#changed_layout = undefined;
      super.cleanup();
   }

   /** @summary Returns canvas name */
   getCanvasName() { return this.getObjectName(); }

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
         this.#changed_layout = (kind !== 'simple'); // use in cleanup
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
   async drawProjection(kind, hist, hopt) {
      if (!this.proj_painter)
         return false; // ignore drawing if projection not configured

      if (hopt === undefined)
         hopt = 'hist';
      if (!kind) kind = 'X';

      if (!this.proj_painter[kind]) {
         this.proj_painter[kind] = 'init';

         const canv = create(clTCanvas),
               pad = this.getRootPad(),
               fp = this.getFramePainter();
         let drawopt;

         if (kind === 'X') {
            canv.fLeftMargin = pad.fLeftMargin;
            canv.fRightMargin = pad.fRightMargin;
            canv.fLogx = fp.logx;
            canv.fUxmin = fp.logx ? Math.log10(fp.scale_xmin) : fp.scale_xmin;
            canv.fUxmax = fp.logx ? Math.log10(fp.scale_xmax) : fp.scale_xmax;
            drawopt = 'fixframe';
         } else if (kind === 'Y') {
            canv.fBottomMargin = pad.fBottomMargin;
            canv.fTopMargin = pad.fTopMargin;
            canv.fLogx = fp.logy;
            canv.fUxmin = fp.logy ? Math.log10(fp.scale_ymin) : fp.scale_ymin;
            canv.fUxmax = fp.logy ? Math.log10(fp.scale_ymax) : fp.scale_ymax;
            drawopt = 'rotate';
         }

         canv.fPrimitives.Add(hist, hopt);

         const promise = isFunc(this.drawInUI5ProjectionArea)
                          ? this.drawInUI5ProjectionArea(canv, drawopt, kind)
                          : this.drawInSidePanel(canv, drawopt, kind);

         return promise.then(painter => { this.proj_painter[kind] = painter; return painter; });
      } else if (isStr(this.proj_painter[kind])) {
         console.log('Not ready with first painting', kind);
         return true;
      }

      this.proj_painter[kind].getMainPainter()?.updateObject(hist, hopt);
      return this.proj_painter[kind].redrawPad();
   }

   /** @summary Checks if canvas shown inside ui5 widget
     * @desc Function should be used only from the func which supposed to be replaced by ui5
     * @private */
   testUI5() {
      return this.use_openui ?? false;
   }

   /** @summary Draw in side panel
     * @private */
   async drawInSidePanel(canv, opt, kind) {
      const sel = ((this.getLayoutKind() === 'projxy') && (kind === 'Y')) ? '.side_panel2' : '.side_panel',
            side = this.selectDom('origin').select(sel);
      return side.empty() ? null : this.drawObject(side.node(), canv, opt);
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

   /** @summary Submit menu request
     * @private */
   async submitMenuRequest(_painter, _kind, reqid) {
      // only single request can be handled, no limit better in RCanvas
      return new Promise(resolveFunc => {
         this.#getmenu_callback = resolveFunc;
         this.sendWebsocket('GETMENU:' + reqid); // request menu items for given painter
      });
   }

   /** @summary Submit object exec request
     * @private */
   submitExec(painter, exec, snapid) {
      if (this.isReadonly() || !painter)
         return;

      if (!snapid)
         snapid = painter.getSnapId();
      if (snapid && isStr(snapid) && exec)
         return this.sendWebsocket(`OBJEXEC:${snapid}:${exec}`);
   }

   /** @summary Return assigned web socket
    * @private */
   getWebsocket() { return this.#websocket; }

   /** @summary Return true if message can be send via web socket
    * @private */
   canSendWebsocket(noper = 1) { return this.#websocket?.canSend(noper); }

   /** @summary Send text message with web socket
     * @desc used for communication with server-side of web canvas
     * @private */
   sendWebsocket(msg) {
      if (this.#websocket?.canSend()) {
         this.#websocket.send(msg);
         return true;
      }
      console.warn(`DROP SEND: ${msg}`);
      return false;
   }

   /** @summary Close websocket connection to canvas
     * @private */
   closeWebsocket(force) {
      if (this.#websocket) {
         this.#websocket.close(force);
         this.#websocket.cleanup();
         this.#websocket = undefined;
      }
   }

   /** @summary Use provided connection for the web canvas
     * @private */
   useWebsocket(handle) {
      this.closeWebsocket();

      this.#websocket = handle;
      this.#websocket.setReceiver(this);
      this.#websocket.connect();
   }

   /** @summary set, test or reset timeout of specified name
     * @desc Used to prevent overloading of websocket for specific function */
   websocketTimeout(name, tm) {
      if (!this.#websocket)
         return;
      if (!this.#websocket._tmouts)
         this.#websocket._tmouts = {};

      const handle = this.#websocket._tmouts[name];
      if (tm === undefined)
         return handle !== undefined;

      if (tm === 'reset') {
         if (handle) { clearTimeout(handle); delete this.#websocket._tmouts[name]; }
      } else if (!handle && Number.isInteger(tm))
         this.#websocket._tmouts[name] = setTimeout(() => { delete this.#websocket._tmouts[name]; }, tm);
   }

   /** @summary Handler for websocket open event
     * @private */
   onWebsocketOpened(/* handle */) {
      // indicate that we are ready to receive any following commands
   }

   /** @summary Handler for websocket close event
     * @private */
   onWebsocketClosed(/* handle */) {
      if (!this.embed_canvas)
         closeCurrentWindow();
   }

   /** @summary Handle websocket messages
     * @private */
   onWebsocketMsg(handle, msg) {
      // console.log(`GET len:${msg.length} msg:${msg.slice(0,60)}`);

      if (msg === 'CLOSE') {
         this.onWebsocketClosed();
         this.closeWebsocket(true);
      } else if (msg.slice(0, 6) === 'SNAP6:') {
         // This is snapshot, produced with TWebCanvas
         const p1 = msg.indexOf(':', 6),
               version = msg.slice(6, p1),
               snap = parse(msg.slice(p1+1));

         this.syncDraw(true)
             .then(() => {
                if (!this.getSnapId())
                   this.resizeBrowser(snap.fSnapshot.fWindowWidth, snap.fSnapshot.fWindowHeight);
                if (!this.getSnapId() && isFunc(this.setFixedCanvasSize))
                   this.#online_fixed_size = this.setFixedCanvasSize(snap.fSnapshot.fCw, snap.fSnapshot.fCh, snap.fFixedSize);
             })
             .then(() => this.redrawPadSnap(snap))
             .then(() => {
                this.completeCanvasSnapDrawing();
                let ranges = this.getWebPadOptions(); // all data, including sub-pads
                if (ranges) ranges = ':' + ranges;
                handle.send(`READY6:${version}${ranges}`); // send ready message back when drawing completed
                this.confirmDraw();
             }).catch(err => {
               if (isFunc(this.showConsoleError))
                  this.showConsoleError(err);
               else
                  console.log(err);
             });
      } else if (msg.slice(0, 5) === 'MENU:') {
         // this is menu with exact identifier for object
         const lst = parse(msg.slice(5));
         if (isFunc(this.#getmenu_callback)) {
            this.#getmenu_callback(lst);
            this.#getmenu_callback = undefined;
         }
      } else if (msg.slice(0, 4) === 'CMD:') {
         msg = msg.slice(4);
         const p1 = msg.indexOf(':'),
               cmdid = msg.slice(0, p1),
               cmd = msg.slice(p1+1),
               reply = `REPLY:${cmdid}:`;
         if ((cmd === 'SVG') || (cmd === 'PNG') || (cmd === 'JPEG') || (cmd === 'WEBP') || (cmd === 'PDF')) {
            this.createImage(cmd.toLowerCase())
                .then(res => handle.send(reply + res));
         } else {
            console.log(`Unrecognized command ${cmd}`);
            handle.send(reply);
         }
      } else if ((msg.slice(0, 7) === 'DXPROJ:') || (msg.slice(0, 7) === 'DYPROJ:')) {
         const kind = msg[1],
               hist = parse(msg.slice(7));
         this.websocketTimeout(`proj${kind}`, 'reset');
         this.drawProjection(kind, hist);
      } else if (msg.slice(0, 5) === 'CTRL:') {
         const ctrl = parse(msg.slice(5)) || {};
         let resized = false;
         if ((ctrl.title !== undefined) && (typeof document !== 'undefined'))
            document.title = ctrl.title;
         if (ctrl.x && ctrl.y && typeof window !== 'undefined') {
            window.moveTo(ctrl.x, ctrl.y);
            resized = true;
         }
         if (ctrl.w && ctrl.h) {
            this.resizeBrowser(Number.parseInt(ctrl.w), Number.parseInt(ctrl.h));
            resized = true;
         }
         if (ctrl.cw && ctrl.ch && isFunc(this.setFixedCanvasSize)) {
            this.#online_fixed_size = this.setFixedCanvasSize(Number.parseInt(ctrl.cw), Number.parseInt(ctrl.ch), true);
            resized = true;
         }
         const kinds = ['Menu', 'StatusBar', 'Editor', 'ToolBar', 'ToolTips'];
         kinds.forEach(kind => {
            if (ctrl[kind] !== undefined)
               this.showSection(kind, ctrl[kind] === '1');
         });

         if (ctrl.edit) {
            const obj_painter = this.findSnap(ctrl.edit);
            if (obj_painter) {
               this.showSection('Editor', true)
                   .then(() => this.producePadEvent('select', obj_painter.getPadPainter(), obj_painter));
            }
         }

         if (ctrl.winstate && typeof window !== 'undefined') {
            if (ctrl.winstate === 'iconify')
               window.blur();
            else
               window.focus();
         }

         if (resized)
            this.sendResized(true);
      } else
         console.log(`unrecognized msg ${msg}`);
   }

   /** @summary Send RESIZED message to client to inform about changes in canvas/window geometry
     * @private */
   sendResized(force) {
      const pad = this.getRootPad();
      if (!pad || (typeof window === 'undefined'))
         return;
      const cw = this.getPadWidth(), ch = this.getPadHeight(),
            wx = window.screenLeft, wy = window.screenTop,
            ww = window.outerWidth, wh = window.outerHeight,
            fixed = this.#online_fixed_size ? 1 : 0;
      if (!force) {
         force = (cw > 0) && (ch > 0) && ((pad.fCw !== cw) || (pad.fCh !== ch));
         if (force) {
            pad.fCw = cw;
            pad.fCh = ch;
         }
      }
      if (force)
         this.sendWebsocket(`RESIZED:${JSON.stringify([wx, wy, ww, wh, cw, ch, fixed])}`);
   }

   /** @summary Handle pad button click event */
   clickPadButton(funcname, evnt) {
      if (funcname === 'ToggleGed')
         return this.activateGed(this, null, 'toggle');
      if (funcname === 'ToggleStatus')
         return this.activateStatusBar('toggle');
      return super.clickPadButton(funcname, evnt);
   }

   /** @summary Returns true if event status shown in the canvas */
   hasEventStatus() {
      if (this.testUI5())
         return false;
      if (this.brlayout)
         return this.brlayout.hasStatus();
      return getHPainter()?.hasStatusLine() ?? false;
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
         loadOpenui5().then(sap => {
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

   /** @summary Show section of canvas  like menu or editor */
   async showSection(that, on) {
      if (this.testUI5())
         return false;

      switch (that) {
         case 'Menu': break;
         case 'StatusBar': this.activateStatusBar(on); break;
         case 'Editor': return this.activateGed(this, null, on);
         case 'ToolBar': break;
         case 'ToolTips': this.setTooltipAllowed(on); break;
      }
      return true;
   }

   /** @summary Send command to start fit panel code on the server
     * @private */
   startFitPanel(standalone) {
      if (!this.getWebsocket())
         return false;

      const new_conn = standalone ? null : this.getWebsocket().createChannel();

      this.sendWebsocket('FITPANEL:' + (standalone ? 'standalone' : new_conn.getChannelId()));

      return new_conn;
   }

   /** @summary Complete handling of online canvas drawing
     * @private */
   completeCanvasSnapDrawing() {
      const pad = this.getRootPad();
      if (!pad)
         return;

      this.addPadInteractive();

      if ((typeof document !== 'undefined') && !this.embed_canvas && this.getWebsocket())
         document.title = pad.fTitle;

      if (this.#all_sections_showed) return;
      this.#all_sections_showed = true;

      // used in Canvas.controller.js to avoid browser resize because of initial sections show/hide
      this._ignore_section_resize = true;

      this.showSection('Menu', pad.TestBit(kMenuBar));
      this.showSection('StatusBar', pad.TestBit(kShowEventStatus));
      this.showSection('ToolBar', pad.TestBit(kShowToolBar));
      this.showSection('Editor', pad.TestBit(kShowEditor));
      this.showSection('ToolTips', pad.TestBit(kShowToolTips) || this._highlight_connect);

      this._ignore_section_resize = false;
   }

   /** @summary Handle highlight in canvas - deliver information to server
     * @private */
   processHighlightConnect(hints) {
      if (!hints?.length || !this._highlight_connect ||
          this.doingDraw() || !this.canSendWebsocket(2)) return;

      const hint = hints[0] || hints[1];
      if (!hint || !hint.painter?.getSnapId() || !hint.user_info)
         return;
      const pp = hint.painter.getPadPainter() || this;
      if (!pp.getSnapId())
         return;

      const arr = [pp.getSnapId(), hint.painter.getSnapId(), '0', '0'];

      if ((hint.user_info.binx !== undefined) && (hint.user_info.biny !== undefined)) {
         arr[2] = hint.user_info.binx.toString();
         arr[3] = hint.user_info.biny.toString();
      } else if (hint.user_info.bin !== undefined)
         arr[2] = hint.user_info.bin.toString();


      const msg = JSON.stringify(arr);

      if (this.#last_highlight_msg !== msg) {
         this.#last_highlight_msg = msg;
         this.sendWebsocket(`HIGHLIGHT:${msg}`);
      }
   }

   /** @summary Method informs that something was changed in the canvas
     * @desc used to update information on the server (when used with web6gui)
     * @private */
   processChanges(kind, painter, subelem) {
      // check if we could send at least one message more - for some meaningful actions
      if (this.isReadonly() || !this.canSendWebsocket(2) || !isStr(kind))
         return;

      let msg = '';
      if (!painter) painter = this;
      switch (kind) {
         case 'sbits':
            msg = 'STATUSBITS:' + this.getStatusBits();
            break;
         case 'frame': // when changing frame
         case 'zoom':  // when changing zoom inside frame
            if (!isFunc(painter.getWebPadOptions))
               painter = painter.getPadPainter();
            if (isFunc(painter.getWebPadOptions))
               msg = 'OPTIONS6:' + painter.getWebPadOptions('only_this');
            break;
         case 'padpos': // when changing pad position
            msg = 'OPTIONS6:' + painter.getWebPadOptions('with_subpads');
            break;
         case 'drawopt':
            if (painter.getSnapId())
               msg = 'DRAWOPT:' + JSON.stringify([painter.getSnapId(), painter.getDrawOpt() || '']);
            break;
         case 'pave_moved': {
            const info = createWebObjectOptions(painter);
            if (info) msg = 'PRIMIT6:' + toJSON(info);
            break;
         }
         case 'logx':
         case 'logy':
         case 'logz': {
            const pp = painter.getPadPainter(),
                  pad = pp?.getRootPad();
            if (pp?.getSnapId() && pad) {
               const name = 'SetLog' + kind[3], value = pad['fLog' + kind[3]];
               painter = pp;
               kind = `exec:${name}(${value})`;
            }
            break;
         }
      }

      if (!msg && isFunc(painter?.getSnapId) && (kind.slice(0, 5) === 'exec:')) {
         const snapid = painter.getSnapId(subelem);
         if (snapid) {
            msg = 'PRIMIT6:' + toJSON({ _typename: 'TWebObjectOptions',
                     snapid, opt: kind.slice(5), fcust: 'exec', fopt: [] });
         }
      }

      if (msg) {
         // console.log(`Sending ${msg.length} ${msg.slice(0,40)}`);
         this.sendWebsocket(msg);
      } else
         console.log(`Unprocessed changes ${kind} for painter of ${painter?.getObject()?._typename} subelem ${subelem}`);
   }

   /** @summary Select active pad on the canvas */
   selectActivePad(pad_painter, obj_painter, click_pos) {
      if (!this.getSnapId() || !pad_painter)
         return; // only interactive canvas

      let arg = null, ischanged = false;
      const is_button = pad_painter.matchObjectType(clTButton);

      if (pad_painter.getSnapId() && this.getWebsocket())
         arg = { _typename: 'TWebPadClick', padid: pad_painter.getSnapId(), objid: '', x: -1, y: -1, dbl: false };

      if (!pad_painter.is_active_pad && !is_button) {
         ischanged = true;
         this.forEachPainterInPad(pp => pp.drawActiveBorder(null, pp === pad_painter), 'pads');
      }

      if (obj_painter?.hasSnapId() && arg) {
         ischanged = true;
         arg.objid = obj_painter.getSnapId();
      }

      if (click_pos && arg) {
         ischanged = true;
         arg.x = Math.round(click_pos.x || 0);
         arg.y = Math.round(click_pos.y || 0);
         if (click_pos.dbl) arg.dbl = true;
      }

      if (arg && (ischanged || is_button))
         this.sendWebsocket('PADCLICKED:' + toJSON(arg));
   }

   /** @summary Return actual TCanvas status bits  */
   getStatusBits() {
      let bits = 0;
      if (this.hasEventStatus()) bits |= kShowEventStatus;
      if (this.hasGed()) bits |= kShowEditor;
      if (this.isTooltipAllowed()) bits |= kShowToolTips;
      if (this.use_openui) bits |= kMenuBar;
      return bits;
   }

   /** @summary produce JSON for TCanvas, which can be used to display canvas once again */
   produceJSON(spacing) {
      const canv = this.getObject();

      if (canv._typename !== clTCanvas)
         return;

      const fill0 = (canv.fFillStyle === 0),
            axes = [], hists = [], prims = [];

      if (fill0)
         canv.fFillStyle = 1001;

      this.forEachPainterInPad(pp => {
         const pad = pp.getRootPad(true);
         if (pp.getNumPainters() && pad?.fPrimitives && !pad.fPrimitives.arr.length) {
            // create list of primitives when missing
            prims.push(pad.fPrimitives);
            pp.forEachPainterInPad(p => {
               // ignore all secondary painters
               if (p.isSecondary())
                  return;
               const subobj = p.getObject();
               if (subobj?._typename)
                  pad.fPrimitives.Add(subobj, p.getDrawOpt());
            }, 'objects');
         }

         const main = pp.getMainPainter(),
               fp = pp.getFramePainter();
         if (!isFunc(main?.getHisto) || !isFunc(main?.getDimension))
            return;

         // write selected range into TAxis properties
         const hist = main.getHisto(),
               ndim = main.getDimension();
         if (!hist?.fXaxis)
            return;

         const setAxisRange = (name, axis) => {
            if (fp?.zoomChangedInteractive(name)) {
               axes.push({ axis, f: axis.fFirst, l: axis.fLast, b: axis.fBits });
               axis.fFirst = main.getSelectIndex(name, 'left', 1);
               axis.fLast = main.getSelectIndex(name, 'right');
               axis.SetBit(EAxisBits.kAxisRange, (axis.fFirst > 0) || (axis.fLast < axis.fNbins));
            }
         };

         setAxisRange('x', hist.fXaxis);
         if (ndim > 1)
            setAxisRange('y', hist.fYaxis);
         if (ndim > 2)
            setAxisRange('z', hist.fZaxis);
         if ((ndim === 2) && fp?.zoomChangedInteractive('z')) {
            hists.push({ hist, min: hist.fMinimum, max: hist.fMaximum });
            hist.fMinimum = fp.zoom_zmin ?? fp.zmin;
            hist.fMaximum = fp.zoom_zmax ?? fp.zmax;
         }
      }, 'pads');

      const res = toJSON(canv, spacing);

      if (fill0)
         canv.fFillStyle = 0;

      axes.forEach(e => {
         e.axis.fFirst = e.f;
         e.axis.fLast = e.l;
         e.axis.fBits = e.b;
      });

      hists.forEach(e => {
         e.hist.fMinimum = e.min;
         e.hist.fMaximum = e.max;
      });

      prims.forEach(lst => lst.Clear());

      return res;
   }

   /** @summary resize browser window */
   resizeBrowser(fullW, fullH) {
      if (!fullW || !fullH || this.isBatchMode() || this.embed_canvas || this.batch_mode)
         return;

      // workaround for qt-based display where inner window size is used
      if (browser.qt6 && fullW > 100 && fullH > 60) {
         fullW -= 3;
         fullH -= 30;
      }

      this.getWebsocket()?.resizeWindow(fullW, fullH);
   }

   /** @summary draw TCanvas */
   static async draw(dom, can, opt) {
      const nocanvas = !can;
      if (nocanvas) can = create(clTCanvas);

      const painter = new TCanvasPainter(dom, can, opt, nocanvas ? 'auto' : true);
      painter.checkSpecialsInPrimitives(can, true);

      if (!nocanvas && can.fCw && can.fCh) {
         const d = painter.selectDom();
         let apply_size;
         if (!painter.isBatchMode()) {
            const rect0 = d.node().getBoundingClientRect();
            apply_size = !rect0.height && (rect0.width > 0.1*can.fCw);
         } else {
            const arg = d.property('_batch_use_canvsize');
            apply_size = arg || (arg === undefined);
         }
         if (apply_size) {
            d.style('width', can.fCw + 'px').style('height', can.fCh + 'px')
              .attr('width', can.fCw).attr('height', can.fCh);
            painter._setFixedSize(true);
         }
      }

      painter.createCanvasSvg(0);

      painter.addPadButtons();

      if (nocanvas && opt.indexOf('noframe') < 0)
         directDrawTFrame(painter, null);

      // select global reference - required for keys handling
      selectActivePad({ pp: painter, active: true });

      return painter.drawPrimitives().then(() => {
         painter.addPadInteractive();
         painter.showPadButtons();
         return painter;
      });
   }

} // class TCanvasPainter


/** @summary Ensure TCanvas and TFrame for the painter object
  * @param {Object} painter  - painter object to process
  * @param {string|boolean} frame_kind  - false for no frame or '3d' for special 3D mode
  * @desc Assign dom, creates TCanvas if necessary, add to list of pad painters */
async function ensureTCanvas(painter, frame_kind) {
   if (!painter)
      return Promise.reject(Error('Painter not provided in ensureTCanvas'));

   // simple check - if canvas there, can use painter
   const noframe = (frame_kind === false) || (frame_kind === '3d') ? 'noframe' : '',
         createCanv = () => {
            if ((noframe !== 'noframe') || !isFunc(painter.getUserRanges))
               return null;
            const ranges = painter.getUserRanges();
            if (!ranges)
               return null;
            const canv = create(clTCanvas),
                  dx = (ranges.maxx - ranges.minx) || 1,
                  dy = (ranges.maxy - ranges.miny) || 1;
            canv.fX1 = ranges.minx - dx * gStyle.fPadLeftMargin;
            canv.fX2 = ranges.maxx + dx * gStyle.fPadRightMargin;
            canv.fY1 = ranges.miny - dy * gStyle.fPadBottomMargin;
            canv.fY2 = ranges.maxy + dy * gStyle.fPadTopMargin;
            return canv;
         },
         pad_painter = painter.getPadPainter() || getDomCanvasPainter(painter.selectDom()),
         promise = pad_painter ? Promise.resolve(pad_painter) :
                     TCanvasPainter.draw(painter.getDom(), createCanv(), noframe);

   return promise.then(pp => {
      if ((frame_kind !== false) && pp.getFrameSvg().selectChild('.main_layer').empty() && !pp.getFramePainter())
         directDrawTFrame(pp, null, frame_kind);

      painter.addToPadPrimitives(pp);
      return painter;
   });
}

/** @summary draw TPad snapshot from TWebCanvas
  * @private */
async function drawTPadSnapshot(dom, snap, opt) {
   const can = create(clTCanvas),
         painter = new TCanvasPainter(dom, can, opt);
   painter.addPadButtons();

   return painter.syncDraw(true).then(() => painter.redrawPadSnap(snap)).then(() => {
      painter.confirmDraw();
      painter.showPadButtons();
      return painter;
   });
}

/** @summary draw TFrame object
  * @private */
async function drawTFrame(dom, obj, opt) {
   const fp = new TFramePainter(dom, obj);
   fp.mode3d = opt === '3d';
   return ensureTCanvas(fp, false).then(() => fp.redraw());
}

Object.assign(internals.jsroot, { ensureTCanvas, TPadPainter, TCanvasPainter });

export { ensureTCanvas, drawTPadSnapshot, drawTFrame, TPadPainter, TCanvasPainter };
