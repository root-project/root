import { gStyle, settings, constants, internals, addMethods, isPromise, isBatchMode } from '../core.mjs';
import { pointer as d3_pointer } from '../d3.mjs';
import { ColorPalette, addColor, getRootColors } from '../base/colors.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';
import { getElementRect, getAbsPosInCanvas, DrawOptions, compressSVG } from '../base/BasePainter.mjs';
import { selectActivePad, getActivePad } from '../base/ObjectPainter.mjs';
import { registerForResize } from '../gui/utils.mjs';
import { BrowserLayout } from '../gui/display.mjs';
import { createMenu, closeMenu } from '../gui/menu.mjs';
import { PadButtonsHandler } from './TPadPainter.mjs';


/**
 * @summary Painter class for RPad
 *
 * @private
 */

class RPadPainter extends RObjectPainter {

   /** @summary constructor */
   constructor(dom, pad, iscan) {
      super(dom, pad, "", "pad");
      this.pad = pad;
      this.iscan = iscan; // indicate if working with canvas
      this.this_pad_name = "";
      if (!this.iscan && (pad !== null)) {
         if (pad.fObjectID)
            this.this_pad_name = "pad" + pad.fObjectID; // use objectid as padname
         else
            this.this_pad_name = "ppp" + internals.id_counter++; // artificical name
      }
      this.painters = []; // complete list of all painters in the pad
      this.has_canvas = true;
      this.forEachPainter = this.forEachPainterInPad;
   }

   /** @summary Indicates that is not Root6 pad painter
    * @private */
   isRoot6() { return false; }

  /** @summary Returns SVG element for the pad itself
    * @private */
   svg_this_pad() {
      return this.getPadSvg(this.this_pad_name);
   }

   /** @summary Returns main painter on the pad
     * @desc Typically main painter is TH1/TH2 object which is drawing axes
    * @private */
   getMainPainter() {
      return this.main_painter_ref || null;
   }

   /** @summary Assign main painter on the pad
    * @private */
   setMainPainter(painter, force) {
      if (!this.main_painter_ref || force)
         this.main_painter_ref = painter;
   }

   /** @summary cleanup pad and all primitives inside */
   cleanup() {
      if (this._doing_draw)
         console.error('pad drawing is not completed when cleanup is called');

      this.painters.forEach(p => p.cleanup());

      let svg_p = this.svg_this_pad();
      if (!svg_p.empty()) {
         svg_p.property('pad_painter', null);
         if (!this.iscan) svg_p.remove();
      }

      delete this.main_painter_ref;
      delete this.frame_painter_ref;
      delete this.pads_cache;
      delete this._pad_x;
      delete this._pad_y;
      delete this._pad_width;
      delete this._pad_height;
      delete this._doing_draw;
      delete this._dfltRFont;

      this.painters = [];
      this.pad = null;
      this.draw_object = null;
      this.pad_frame = null;
      this.this_pad_name = undefined;
      this.has_canvas = false;

      selectActivePad({ pp: this, active: false });

      super.cleanup();
   }

   /** @summary Returns frame painter inside the pad
    * @private */
   getFramePainter() { return this.frame_painter_ref; }

   /** @summary get pad width */
   getPadWidth() { return this._pad_width || 0; }

   /** @summary get pad height */
   getPadHeight() { return this._pad_height || 0; }

   /** @summary get pad rect */
   getPadRect() {
      return {
         x: this._pad_x || 0,
         y: this._pad_y || 0,
         width: this.getPadWidth(),
         height: this.getPadHeight()
      }
   }

   /** @summary Returns frame coordiantes - also when frame is not drawn */
   getFrameRect() {
      let fp = this.getFramePainter();
      if (fp) return fp.getFrameRect();

      let w = this.getPadWidth(),
          h = this.getPadHeight(),
          rect = {};

      rect.szx = Math.round(0.5*w);
      rect.szy = Math.round(0.5*h);
      rect.width = 2*rect.szx;
      rect.height = 2*rect.szy;
      rect.x = Math.round(w/2 - rect.szx);
      rect.y = Math.round(h/2 - rect.szy);
      rect.hint_delta_x = rect.szx;
      rect.hint_delta_y = rect.szy;
      rect.transform = `translate(${rect.x},${rect.y})`;
      return rect;
   }

   /** @summary return RPad object */
   getRootPad(is_root6) {
      return (is_root6 === undefined) || !is_root6 ? this.pad : null;
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove
    * @private */
   cleanPrimitives(selector) {
      if (!selector || (typeof selector !== 'function')) return;

      for (let k = this.painters.length-1; k >= 0; --k)
         if (selector(this.painters[k])) {
            this.painters[k].cleanup();
            this.painters.splice(k, 1);
         }
   }

   /** @summary Try to find painter for specified object
     * @desc can be used to find painter for some special objects, registered as
     * histogram functions
     * @private */
   findPainterFor(selobj, selname, seltype) {
      return this.painters.find(p => {
         let pobj = p.getObject();
         if (!pobj) return;

         if (selobj && (pobj === selobj)) return true;
         if (!selname && !seltype) return;
         if (selname && (pobj.fName !== selname)) return;
         if (seltype && (pobj._typename !== seltype)) return;
         return true;
      });
   }

   /** @summary Returns palette associated with pad.
     * @desc Either from existing palette painter or just default palette */
   getHistPalette() {
      let pp = this.findPainterFor(undefined, undefined, "ROOT::Experimental::RPaletteDrawable");

      if (pp) return pp.getHistPalette();

      if (!this.fDfltPalette) {
         this.fDfltPalette = {
            _typename: "ROOT::Experimental::RPalette",
            fColors: [{ fOrdinal : 0,     fColor : { fColor : "rgb(53, 42, 135)" } },
                      { fOrdinal : 0.125, fColor : { fColor : "rgb(15, 92, 221)" } },
                      { fOrdinal : 0.25,  fColor : { fColor : "rgb(20, 129, 214)" } },
                      { fOrdinal : 0.375, fColor : { fColor : "rgb(6, 164, 202)" } },
                      { fOrdinal : 0.5,   fColor : { fColor : "rgb(46, 183, 164)" } },
                      { fOrdinal : 0.625, fColor : { fColor : "rgb(135, 191, 119)" } },
                      { fOrdinal : 0.75,  fColor : { fColor : "rgb(209, 187, 89)" } },
                      { fOrdinal : 0.875, fColor : { fColor : "rgb(254, 200, 50)" } },
                      { fOrdinal : 1,     fColor : { fColor : "rgb(249, 251, 14)" } }],
             fInterpolate: true,
             fNormalized: true
         };
         addMethods(this.fDfltPalette, "ROOT::Experimental::RPalette");
      }

      return this.fDfltPalette;
   }

   /** @summary Returns number of painters
     * @private */
   getNumPainters() { return this.painters.length; }

   /** @summary Call function for each painter in pad
     * @param {function} userfunc - function to call
     * @param {string} kind - "all" for all objects (default), "pads" only pads and subpads, "objects" only for object in current pad
     * @private */
   forEachPainterInPad(userfunc, kind) {
      if (!kind) kind = "all";
      if (kind != "objects") userfunc(this);
      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];
         if (typeof sub.forEachPainterInPad === 'function') {
            if (kind!="objects") sub.forEachPainterInPad(userfunc, kind);
         } else if (kind != "pads") userfunc(sub);
      }
   }

   /** @summary register for pad events receiver
     * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   registerForPadEvents(receiver) {
      this.pad_events_receiver = receiver;
   }

   /** @summary Generate pad events, normally handled by GED
     * @desc in pad painter, while pad may be drawn without canvas
     * @private */
   producePadEvent(_what, _padpainter, _painter, _position, _place) {
      if ((_what == "select") && (typeof this.selectActivePad == 'function'))
         this.selectActivePad(_padpainter, _painter, _position);

      if (this.pad_events_receiver)
         this.pad_events_receiver({ what: _what, padpainter:  _padpainter, painter: _painter, position: _position, place: _place });
   }

   /** @summary method redirect call to pad events receiver */
   selectObjectPainter(_painter, pos, _place) {

      let istoppad = (this.iscan || !this.has_canvas),
          canp = istoppad ? this : this.getCanvPainter();

      if (_painter === undefined) _painter = this;

      if (pos && !istoppad)
         pos = getAbsPosInCanvas(this.svg_this_pad(), pos);

      selectActivePad({ pp: this, active: true });

      canp.producePadEvent("select", this, _painter, pos, _place);
   }

   /** @summary Create SVG element for the canvas */
   createCanvasSvg(check_resize, new_size) {

      let factor = null, svg = null, lmt = 5, rect = null, btns, frect;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.getCanvSvg();

         if (svg.empty()) return false;

         factor = svg.property('height_factor');

         rect = this.testMainResize(check_resize, null, factor);

         if (!rect.changed) return false;

         if (!isBatchMode())
            btns = this.getLayerSvg("btns_layer", this.this_pad_name);

         frect = svg.select(".canvas_fillrect");

      } else {

         let render_to = this.selectDom();

         if (render_to.style('position') == 'static')
            render_to.style('position', 'relative');

         svg = render_to.append("svg")
             .attr("class", "jsroot root_canvas")
             .property('pad_painter', this) // this is custom property
             .property('current_pad', "") // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         this.setTopPainter(); //assign canvas as top painter of that element

         if (!isBatchMode() && !this.online_canvas)
            svg.append("svg:title").text("ROOT canvas");

         frect = svg.append("svg:path").attr("class","canvas_fillrect");
         if (!isBatchMode())
            frect.style("pointer-events", "visibleFill")
                 .on("dblclick", evnt => this.enlargePad(evnt))
                 .on("click", () => this.selectObjectPainter(this, null))
                 .on("mouseenter", () => this.showObjectStatus())
                 .on("contextmenu", settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);

         svg.append("svg:g").attr("class", "primitives_layer");
         svg.append("svg:g").attr("class", "info_layer");
         if (!isBatchMode())
            btns = svg.append("svg:g")
                      .attr("class","btns_layer")
                      .property('leftside', settings.ToolBarSide == 'left')
                      .property('vertical', settings.ToolBarVert);

         factor = 0.66;
         if (this.pad && this.pad.fWinSize[0] && this.pad.fWinSize[1]) {
            factor = this.pad.fWinSize[1] / this.pad.fWinSize[0];
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this._fixed_size) {
            render_to.style("overflow","auto");
            rect = { width: this.pad.fWinSize[0], height: this.pad.fWinSize[1] };
            if (!rect.width || !rect.height)
               rect = getElementRect(render_to);
         } else {
            rect = this.testMainResize(2, new_size, factor);
         }
      }

      this.createAttFill({ pattern: 1001, color: 0 });

      if ((rect.width <= lmt) || (rect.height <= lmt)) {
         svg.style("display", "none");
         console.warn(`Hide canvas while geometry too small w=${rect.width} h=${rect.height}`);
         rect.width = 200; rect.height = 100; // just to complete drawing
      } else {
         svg.style("display", null);
      }

      if (this._fixed_size) {
         svg.attr("x", 0)
            .attr("y", 0)
            .attr("width", rect.width)
            .attr("height", rect.height)
            .style("position", "absolute");
      } else {
        svg.attr("x", 0)
           .attr("y", 0)
           .style("width", "100%")
           .style("height", "100%")
           .style("position", "absolute")
           .style("left", 0)
           .style("top", 0)
           .style("right", 0)
           .style("bottom", 0);
      }

      svg.style("filter", settings.DarkMode ? "invert(100%)" : null);

      svg.attr("viewBox", `0 0 ${rect.width} ${rect.height}`)
         .attr("preserveAspectRatio", "none")  // we do not preserve relative ratio
         .property('height_factor', factor)
         .property('draw_x', 0)
         .property('draw_y', 0)
         .property('draw_width', rect.width)
         .property('draw_height', rect.height);

      this._pad_x = 0;
      this._pad_y = 0;
      this._pad_width = rect.width;
      this._pad_height = rect.height;

      frect.attr("d", `M0,0H${rect.width}V${rect.height}H0Z`)
           .call(this.fillatt.func);

      this._fast_drawing = settings.SmallPad && ((rect.width < settings.SmallPad.width) || (rect.height < settings.SmallPad.height));

      if (this.alignButtons && btns)
         this.alignButtons(btns, rect.width, rect.height);

      return true;
   }

   /** @summary Enlarge pad draw element when possible */
   enlargePad(evnt) {

      if (evnt) {
         evnt.preventDefault();
         evnt.stopPropagation();
      }

      let svg_can = this.getCanvSvg(),
          pad_enlarged = svg_can.property("pad_enlarged");

      if (this.iscan || !this.has_canvas || (!pad_enlarged && !this.hasObjectsToDraw() && !this.painters)) {
         if (this._fixed_size) return; // canvas cannot be enlarged in such mode
         if (!this.enlargeMain('toggle')) return;
         if (this.enlargeMain('state')=='off') svg_can.property("pad_enlarged", null);
      } else if (!pad_enlarged) {
         this.enlargeMain(true, true);
         svg_can.property("pad_enlarged", this.pad);
      } else if (pad_enlarged === this.pad) {
         this.enlargeMain(false);
         svg_can.property("pad_enlarged", null);
      } else {
         console.error('missmatch with pad double click events');
      }

      let was_fast = this._fast_drawing;

      this.checkResize(true);

      if (this._fast_drawing != was_fast)
         this.showPadButtons();
   }

   /** @summary Create SVG element for the pad
     * @returns true when pad is displayed and all its items should be redrawn */
   createPadSvg(only_resize) {

      if (!this.has_canvas) {
         this.createCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      let svg_parent = this.getPadSvg(this.pad_name), // this.pad_name MUST be here to select parent pad
          svg_can = this.getCanvSvg(),
          width = svg_parent.property("draw_width"),
          height = svg_parent.property("draw_height"),
          pad_enlarged = svg_can.property("pad_enlarged"),
          pad_visible = true,
          w = width, h = height, x = 0, y = 0,
          svg_pad = null, svg_rect = null, btns = null;

      if (this.pad && this.pad.fPos && this.pad.fSize) {
         x = Math.round(width * this.pad.fPos.fHoriz.fArr[0]);
         y = Math.round(height * this.pad.fPos.fVert.fArr[0]);
         w = Math.round(width * this.pad.fSize.fHoriz.fArr[0]);
         h = Math.round(height * this.pad.fSize.fVert.fArr[0]);
      }

      if (pad_enlarged) {
         pad_visible = false;
         if (pad_enlarged === this.pad)
            pad_visible = true;
         else
            this.forEachPainterInPad(pp => { if (pp.getObject() == pad_enlarged) pad_visible = true; }, "pads");

         if (pad_visible) { w = width; h = height; x = y = 0; }
      }

      if (only_resize) {
         svg_pad = this.svg_this_pad();
         svg_rect = svg_pad.select(".root_pad_border");
         if (!isBatchMode())
            btns = this.getLayerSvg("btns_layer", this.this_pad_name);
      } else {
         svg_pad = svg_parent.select(".primitives_layer")
             .append("svg:svg") // here was g before, svg used to blend all drawin outside
             .classed("__root_pad_" + this.this_pad_name, true)
             .attr("pad", this.this_pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this); // this is custom property

         if (!isBatchMode())
            svg_pad.append("svg:title").text("ROOT subpad");

         svg_rect = svg_pad.append("svg:path").attr("class", "root_pad_border");

         svg_pad.append("svg:g").attr("class","primitives_layer");
         if (!isBatchMode())
            btns = svg_pad.append("svg:g")
                          .attr("class","btns_layer")
                          .property('leftside', settings.ToolBarSide != 'left')
                          .property('vertical', settings.ToolBarVert);

         if (settings.ContextMenu)
            svg_rect.on("contextmenu", evnt => this.padContextMenu(evnt));

         if (!isBatchMode())
            svg_rect.style("pointer-events", "visibleFill") // get events also for not visible rect
                    .on("dblclick", evnt => this.enlargePad(evnt))
                    .on("click", () => this.selectObjectPainter(this, null))
                    .on("mouseenter", () => this.showObjectStatus());
      }

      this.createAttFill({ attr: this.pad });

      this.createAttLine({ attr: this.pad, color0: this.pad.fBorderMode == 0 ? 'none' : '' });

      svg_pad.style("display", pad_visible ? null : "none")
             .attr("viewBox", `0 0 ${w} ${h}`) // due to svg
             .attr("preserveAspectRatio", "none")   // due to svg, we do not preserve relative ratio
             .attr("x", x)    // due to svg
             .attr("y", y)   // due to svg
             .attr("width", w)    // due to svg
             .attr("height", h)   // due to svg
             .property('draw_x', x) // this is to make similar with canvas
             .property('draw_y', y)
             .property('draw_width', w)
             .property('draw_height', h);

      this._pad_x = x;
      this._pad_y = y;
      this._pad_width = w;
      this._pad_height = h;

      svg_rect.attr("d", `M0,0H${w}V${h}H0Z`)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      this._fast_drawing = settings.SmallPad && ((w < settings.SmallPad.width) || (h < settings.SmallPad.height));

       // special case of 3D canvas overlay
      if (svg_pad.property('can3d') === constants.Embed3D.Overlay)
          this.selectDom().select(".draw3d_" + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');

      if (this.alignButtons && btns) this.alignButtons(btns, w, h);

      return pad_visible;
   }

   /** @summary returns true if any objects beside sub-pads exists in the pad */
   hasObjectsToDraw() {
      let arr = this.pad ? this.pad.fPrimitives : null;
      return arr && arr.find(obj => obj._typename != "ROOT::Experimental::RPadDisplayItem") ? true : false;
   }

   /** @summary sync drawing/redrawing/resize of the pad
     * @param {string} kind - kind of draw operation, if true - always queued
     * @returns {Promise} when pad is ready for draw operation or false if operation already queued
     * @private */
   syncDraw(kind) {
      let entry = { kind : kind || "redraw" };
      if (this._doing_draw === undefined) {
         this._doing_draw = [ entry ];
         return Promise.resolve(true);
      }
      // if queued operation registered, ignore next calls, indx == 0 is running operation
      if ((entry.kind !== true) && (this._doing_draw.findIndex((e,i) => (i > 0) && (e.kind == entry.kind)) > 0))
         return false;
      this._doing_draw.push(entry);
      return new Promise(resolveFunc => {
         entry.func = resolveFunc;
      });
   }

   /** @summary confirms that drawing is completed, may trigger next drawing immediately
     * @private */
   confirmDraw() {
      if (this._doing_draw === undefined)
         return console.warn("failure, should not happen");
      this._doing_draw.shift();
      if (this._doing_draw.length == 0) {
         delete this._doing_draw;
      } else {
         let entry = this._doing_draw[0];
         if(entry.func) { entry.func(); delete entry.func; }
      }
   }

   /** @summary Draw single primitive */
   drawObject(dom, obj, opt) {
      console.log('Not possible to draw object without loading of draw.mjs');
      return Promise.resolve(null);
   }

   /** @summary Draw pad primitives
     * @private */
   drawPrimitives(indx) {

      if (indx === undefined) {
         if (this.iscan)
            this._start_tm = new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.length : 0;

         return this.syncDraw(true).then(() => this.drawPrimitives(0));
      }

      if (!this.pad || (indx >= this._num_primitives)) {

         this.confirmDraw();

         if (this._start_tm) {
            let spenttm = new Date().getTime() - this._start_tm;
            if (spenttm > 3000) console.log("Canvas drawing took " + (spenttm*1e-3).toFixed(2) + "s");
            delete this._start_tm;
         }

         return Promise.resolve();
      }

      // handle used to invoke callback only when necessary
      return this.drawObject(this.getDom(), this.pad.fPrimitives[indx], "").then(ppainter => {
         // mark painter as belonging to primitives
         if (ppainter && (typeof ppainter == 'object'))
            ppainter._primitive = true;

         return this.drawPrimitives(indx+1);
      });
   }

   /** @summary Process tooltip event in the pad
     * @private */
   processPadTooltipEvent(pnt) {
      let painters = [], hints = [];

      // first count - how many processors are there
      if (this.painters !== null)
         this.painters.forEach(obj => {
            if (typeof obj.processTooltipEvent == 'function') painters.push(obj);
         });

      if (pnt) pnt.nproc = painters.length;

      painters.forEach(obj => {
         let hint = obj.processTooltipEvent(pnt);
         if (!hint) hint = { user_info: null };
         hints.push(hint);
         if (pnt && pnt.painters) hint.painter = obj;
      });

      return hints;
   }

   /** @summary Changes canvas dark mode
     * @private */
   changeDarkMode(mode) {
      this.getCanvSvg().style("filter", (mode ?? settings.DarkMode)  ? "invert(100%)" : null);
   }

   /** @summary Fill pad context menu
     * @private */
   fillContextMenu(menu) {

      if (this.iscan)
         menu.add("header: RCanvas");
      else
         menu.add("header: RPad");

      menu.addchk(this.isTooltipAllowed(), "Show tooltips", () => this.setTooltipAllowed("toggle"));

      if (!this._websocket) {
         menu.addAttributesMenu(this);
         if (this.iscan)
            menu.addSettingsMenu(false, false, arg => {
               if (arg == "dark") this.changeDarkMode();
            });
      }

      menu.add("separator");

      if (typeof this.hasMenuBar == 'function' && typeof this.actiavteMenuBar == 'function')
         menu.addchk(this.hasMenuBar(), "Menu bar", flag => this.actiavteMenuBar(flag));

      if (typeof this.hasEventStatus == 'function' && typeof this.activateStatusBar == 'function')
         menu.addchk(this.hasEventStatus(), "Event status", () => this.activateStatusBar('toggle'));

      if (this.enlargeMain() || (this.has_canvas && this.hasObjectsToDraw()))
         menu.addchk((this.enlargeMain('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), () => this.enlargePad());

      let fname = this.this_pad_name || (this.iscan ? "canvas" : "pad");
      menu.add(`Save as ${fname}.png`, fname+".png", arg => this.saveAs("png", false, arg));
      menu.add(`Save as ${fname}.svg`, fname+".svg", arg => this.saveAs("svg", false, arg));

      return true;
   }

   /** @summary Show pad context menu
     * @private */
   padContextMenu(evnt) {
      if (evnt.stopPropagation) {
         let pos = d3_pointer(evnt, this.svg_this_pad().node());
         // this is normal event processing and not emulated jsroot event
         // for debug purposes keep original context menu for small region in top-left corner
         if ((pos.length==2) && (pos[0] >= 0) && (pos[0] < 10) && (pos[1] >= 0) && (pos[1] < 10)) return;

         evnt.stopPropagation(); // disable main context menu
         evnt.preventDefault();  // disable browser context menu

         let fp = this.getFramePainter();
         if (fp) fp.setLastEventPos();
      }

      createMenu(evnt, this).then(menu => {
         this.fillContextMenu(menu);
         return this.fillObjectExecMenu(menu);
      }).then(menu => menu.show());
   }

   /** @summary Redraw pad means redraw ourself
     * @returns {Promise} when redrawing ready */
   redrawPad(reason) {

      let sync_promise = this.syncDraw(reason);
      if (sync_promise === false) {
         console.log('Prevent RPad redrawing');
         return Promise.resolve(false);
      }

      let showsubitems = true;
      let redrawNext = indx => {
         while (indx < this.painters.length) {
            let sub = this.painters[indx++], res = 0;
            if (showsubitems || sub.this_pad_name)
               res = sub.redraw(reason);

            if (isPromise(res))
               return res.then(() => redrawNext(indx));
         }
         return Promise.resolve(true);
      };

      return sync_promise.then(() => {
         if (this.iscan) {
            this.createCanvasSvg(2);
         } else {
            showsubitems = this.createPadSvg(true);
         }
         return redrawNext(0);
      }).then(() => {
         if (getActivePad() === this) {
            let canp = this.getCanvPainter();
            if (canp) canp.producePadEvent("padredraw", this);
         }
         this.confirmDraw();
         return true;
      });
   }

   /** @summary redraw pad */
   redraw(reason) {
      return this.redrawPad(reason);
   }


   /** @summary Checks if pad should be redrawn by resize
     * @private */
   needRedrawByResize() {
      let elem = this.svg_this_pad();
      if (!elem.empty() && elem.property('can3d') === constants.Embed3D.Overlay) return true;

      for (let i = 0; i < this.painters.length; ++i)
         if (typeof this.painters[i].needRedrawByResize === 'function')
            if (this.painters[i].needRedrawByResize()) return true;

      return false;
   }

   /** @summary Check resize of canvas */
   checkCanvasResize(size, force) {

      if (!this.iscan && this.has_canvas) return false;

      let sync_promise = this.syncDraw("canvas_resize");
      if (sync_promise === false) return false;

      if ((size === true) || (size === false)) { force = size; size = null; }

      if (size && (typeof size === 'object') && size.force) force = true;

      if (!force) force = this.needRedrawByResize();

      let changed = false,
          redrawNext = indx => {
             if (!changed || (indx >= this.painters.length)) {
                this.confirmDraw();
                return changed;
             }

             let res = this.painters[indx].redraw(force ? "redraw" : "resize");
             if (!isPromise(res)) res = Promise.resolve();
             return res.then(() => redrawNext(indx+1));
          };

      return sync_promise.then(() => {

         changed = this.createCanvasSvg(force ? 2 : 1, size);

         // if canvas changed, redraw all its subitems.
         // If redrawing was forced for canvas, same applied for sub-elements
         return redrawNext(0);
      });
   }

   /** @summary update RPad object
     * @private */
   updateObject(obj) {
      if (!obj) return false;

      this.pad.fStyle = obj.fStyle;
      this.pad.fAttr = obj.fAttr;

      if (this.iscan) {
         this.pad.fTitle = obj.fTitle;
         this.pad.fWinSize = obj.fWinSize;
      } else {
         this.pad.fPos = obj.fPos;
         this.pad.fSize = obj.fSize;
      }

      return true;
   }


   /** @summary Add object painter to list of primitives
     * @private */
   addObjectPainter(objpainter, lst, indx) {
      if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
         // keep snap id in painter, will be used for the
         if (this.painters.indexOf(objpainter) < 0)
            this.painters.push(objpainter);
         objpainter.assignSnapId(lst[indx].fObjectID);
         if (!objpainter.rstyle) objpainter.rstyle = lst[indx].fStyle || this.rstyle;
      }
   }

   /** @summary Extract properties from TObjectDisplayItem */
   extractTObjectProp(snap) {
      if (snap.fColIndex && snap.fColValue) {
         let colors = this.root_colors || getRootColors();
         for (let k = 0; k < snap.fColIndex.length; ++k)
            colors[snap.fColIndex[k]] = snap.fColValue[k];
       }

      // painter used only for evaluation of attributes
      let pattr = new RObjectPainter(), obj = snap.fObject;
      pattr.assignObject(snap);
      pattr.csstype = snap.fCssType;
      pattr.rstyle = snap.fStyle;

      snap.fOption = pattr.v7EvalAttr("options", "");

      const extract_color = (member_name, attr_name) => {
         let col = pattr.v7EvalColor(attr_name, "");
         if (col) obj[member_name] = addColor(col, this.root_colors);
      };

      // handle TAttLine
      if ((obj.fLineColor !== undefined) && (obj.fLineWidth !== undefined) && (obj.fLineStyle !== undefined)) {
         extract_color("fLineColor", "line_color");
         obj.fLineWidth = pattr.v7EvalAttr("line_width", obj.fLineWidth);
         obj.fLineStyle = pattr.v7EvalAttr("line_style", obj.fLineStyle);
      }

      // handle TAttFill
      if ((obj.fFillColor !== undefined) && (obj.fFillStyle !== undefined)) {
         extract_color("fFillColor", "fill_color");
         obj.fFillStyle = pattr.v7EvalAttr("fill_style", obj.fFillStyle);
      }

      // handle TAttMarker
      if ((obj.fMarkerColor !== undefined) && (obj.fMarkerStyle !== undefined) && (obj.fMarkerSize !== undefined)) {
         extract_color("fMarkerColor", "marker_color");
         obj.fMarkerStyle = pattr.v7EvalAttr("marker_style", obj.fMarkerStyle);
         obj.fMarkerSize = pattr.v7EvalAttr("marker_size", obj.fMarkerSize);
      }

      // handle TAttText
      if ((obj.fTextColor !== undefined) && (obj.fTextAlign !== undefined) && (obj.fTextAngle !== undefined) && (obj.fTextSize !== undefined)) {
         extract_color("fTextColor", "text_color");
         obj.fTextAlign = pattr.v7EvalAttr("text_align", obj.fTextAlign);
         obj.fTextAngle = pattr.v7EvalAttr("text_angle", obj.fTextAngle);
         obj.fTextSize = pattr.v7EvalAttr("text_size", obj.fTextSize);
         // TODO: v7 font handling differs much from v6, ignore for the moment
      }
   }

   /** @summary Function called when drawing next snapshot from the list
     * @returns {Promise} with pad painter when ready
     * @private */
   drawNextSnap(lst, indx) {

      if (indx === undefined) {
         indx = -1;
         // flag used to prevent immediate pad redraw during first draw
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
         this._auto_color_cnt = 0;
      }

      delete this.next_rstyle;

      ++indx; // change to the next snap

      if (!lst || indx >= lst.length) {
         delete this._snaps_map;
         delete this._auto_color_cnt;
         return Promise.resolve(this);
      }

      let snap = lst[indx],
          snapid = snap.fObjectID,
          cnt = this._snaps_map[snapid],
          objpainter = null;

      if (cnt) cnt++; else cnt=1;
      this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

      // empty object, no need to do something, take next
      if (snap.fDummy) return this.drawNextSnap(lst, indx);

      // first appropriate painter for the object
      // if same object drawn twice, two painters will exists
      for (let k = 0; k<this.painters.length; ++k) {
         if (this.painters[k].snapid === snapid)
            if (--cnt === 0) { objpainter = this.painters[k]; break;  }
      }

      if (objpainter) {

         if (snap._typename == "ROOT::Experimental::RPadDisplayItem")  // subpad
            return objpainter.redrawPadSnap(snap).then(ppainter => {
               this.addObjectPainter(ppainter, lst, indx);
               return this.drawNextSnap(lst, indx);
            });

         if (snap._typename === "ROOT::Experimental::TObjectDisplayItem")
            this.extractTObjectProp(snap);

         let promise;

         if (objpainter.updateObject(snap.fDrawable || snap.fObject || snap, snap.fOption || ""))
            promise = objpainter.redraw();

         if (!isPromise(promise)) promise = Promise.resolve(true);

         return promise.then(() => this.drawNextSnap(lst, indx)); // call next
      }

      if (snap._typename == "ROOT::Experimental::RPadDisplayItem") { // subpad

         let subpad = snap; // not subpad, but just attributes

         let padpainter = new RPadPainter(this.getDom(), subpad, false);
         padpainter.decodeOptions("");
         padpainter.addToPadPrimitives(this.this_pad_name); // only set parent pad name
         padpainter.assignSnapId(snap.fObjectID);
         padpainter.rstyle = snap.fStyle;

         padpainter.createPadSvg();

         if (snap.fPrimitives && snap.fPrimitives.length > 0)
            padpainter.addPadButtons();

         // we select current pad, where all drawing is performed
         let prev_name = padpainter.selectCurrentPad(padpainter.this_pad_name);

         return padpainter.drawNextSnap(snap.fPrimitives).then(() => {
            padpainter.selectCurrentPad(prev_name);
            return this.drawNextSnap(lst, indx);
         });
      }

      // will be used in addToPadPrimitives to assign style to sub-painters
      this.next_rstyle = lst[indx].fStyle || this.rstyle;

      if (snap._typename === "ROOT::Experimental::TObjectDisplayItem") {

         // identifier used in RObjectDrawable
         const webSnapIds = { kNone: 0,  kObject: 1, kColors: 4, kStyle: 5, kPalette: 6 };

         if (snap.fKind == webSnapIds.kStyle) {
            Object.assign(gStyle, snap.fObject);
            return this.drawNextSnap(lst, indx);
         }

         if (snap.fKind == webSnapIds.kColors) {
            let ListOfColors = [], arr = snap.fObject.arr;
            for (let n = 0; n < arr.length; ++n) {
               let name = arr[n].fString, p = name.indexOf("=");
               if (p > 0)
                  ListOfColors[parseInt(name.slice(0,p))] = name.slice(p+1);
            }

            this.root_colors = ListOfColors;
            // set global list of colors
            // adoptRootColors(ListOfColors);
            return this.drawNextSnap(lst, indx);
         }

         if (snap.fKind == webSnapIds.kPalette) {
            let arr = snap.fObject.arr, palette = [];
            for (let n = 0; n < arr.length; ++n)
               palette[n] =  arr[n].fString;
            this.custom_palette = new ColorPalette(palette);
            return this.drawNextSnap(lst, indx);
         }

         if (!this.getFramePainter())
            return this.drawObject(this.getDom(), { _typename: "TFrame", $dummy: true }, "")
                       .then(() => this.drawNextSnap(lst, indx-1)); // call same object again

         this.extractTObjectProp(snap);
      }

      // TODO - fDrawable is v7, fObject from v6, maybe use same data member?
      return this.drawObject(this.getDom(), snap.fDrawable || snap.fObject || snap, snap.fOption || "").then(objpainter => {
         this.addObjectPainter(objpainter, lst, indx);
         return this.drawNextSnap(lst, indx);
      });
   }

   /** @summary Search painter with specified snapid, also sub-pads are checked
     * @private */
   findSnap(snapid, onlyid) {

      function check(checkid) {
         if (!checkid || (typeof checkid != 'string')) return false;
         if (checkid == snapid) return true;
         return onlyid && (checkid.length > snapid.length) &&
                (checkid.indexOf(snapid) == (checkid.length - snapid.length));
      }

      if (check(this.snapid)) return this;

      if (!this.painters) return null;

      for (let k=0;k<this.painters.length;++k) {
         let sub = this.painters[k];

         if (!onlyid && (typeof sub.findSnap === 'function'))
            sub = sub.findSnap(snapid);
         else if (!check(sub.snapid))
            sub = null;

         if (sub) return sub;
      }

      return null;
   }

   /** @summary Redraw pad snap
     * @desc Online version of drawing pad primitives
     * @returns {Promise} with pad painter*/
   redrawPadSnap(snap) {
      // for the pad/canvas display item contains list of primitives plus pad attributes

      if (!snap || !snap.fPrimitives)
         return Promise.resolve(this);

      // for the moment only window size attributes are provided
      // let padattr = { fCw: snap.fWinSize[0], fCh: snap.fWinSize[1], fTitle: snap.fTitle };

      // if canvas size not specified in batch mode, temporary use 900x700 size
      // if (this.batch_mode && this.iscan && (!padattr.fCw || !padattr.fCh)) { padattr.fCw = 900; padattr.fCh = 700; }

      if (this.iscan && this._websocket && snap.fTitle && !this.embed_canvas && (typeof document !== "undefined"))
         document.title = snap.fTitle;

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.assignSnapId(snap.fObjectID);

         this.draw_object = snap;
         this.pad = snap;

         if (this.batch_mode && this.iscan)
             this._fixed_size = true;

         let mainid = this.selectDom().attr("id");

         if (!this.batch_mode && !this.use_openui && !this.brlayout && mainid && (typeof mainid == "string")) {
            this.brlayout = new BrowserLayout(mainid, null, this);
            this.brlayout.create(mainid, true);
            this.setDom(this.brlayout.drawing_divid()); // need to create canvas
            registerForResize(this.brlayout);
         }

         this.createCanvasSvg(0);
         this.addPadButtons(true);

         return this.drawNextSnap(snap.fPrimitives);
      }

      // update only pad/canvas attributes
      this.updateObject(snap);

      // apply all changes in the object (pad or canvas)
      if (this.iscan) {
         this.createCanvasSvg(2);
      } else {
         this.createPadSvg(true);
      }

      let isanyfound = false, isanyremove = false;

      // find and remove painters which no longer exists in the list
      for (let k = 0;k < this.painters.length; ++k) {
         let sub = this.painters[k];
         if (sub.snapid===undefined) continue; // look only for painters with snapid

         snap.fPrimitives.forEach(prim => {
            if (sub && (prim.fObjectID === sub.snapid)) {
               sub = null; isanyfound = true;
            }
         });

         if (sub) {
            // remove painter which does not found in the list of snaps
            this.painters.splice(k--,1);
            sub.cleanup(); // cleanup such painter
            isanyremove = true;
            if (this.main_painter_ref === sub)
               delete this.main_painter_ref;
         }
      }

      if (isanyremove)
         delete this.pads_cache;

      if (!isanyfound) {
         let fp = this.getFramePainter();
         // cannot preserve ROOT6 frame - it must be recreated
         if (fp && fp.is_root6()) fp = null;
         for (let k = 0; k < this.painters.length; ++k)
             if (fp !== this.painters[k])
               this.painters[k].cleanup();
         this.painters = [];
         delete this.main_painter_ref;
         if (fp) {
            this.painters.push(fp);
            fp.cleanFrameDrawings();
            fp.redraw(); // need to create all layers again
         }
         if (this.removePadButtons) this.removePadButtons();
         this.addPadButtons(true);
      }

      let prev_name = this.selectCurrentPad(this.this_pad_name);

      return this.drawNextSnap(snap.fPrimitives).then(() => {
         this.selectCurrentPad(prev_name);

         if (getActivePad() === this) {
            let canp = this.getCanvPainter();
            if (canp) canp.producePadEvent("padredraw", this);
         }
         return this;
      });
   }

   /** @summary Create image for the pad
     * @desc Used with web-based canvas to create images for server side
     * @returns {Promise} with image data, coded with btoa() function
     * @private */
   createImage(format) {
      // use https://github.com/MrRio/jsPDF in the future here
      if (format == "pdf")
         return Promise.resolve(btoa("dummy PDF file"));

      if ((format == "png") || (format == "jpeg") || (format == "svg"))
         return this.produceImage(true, format).then(res => {
            if (!res || (format == "svg")) return res;
            let separ = res.indexOf("base64,");
            return (separ > 0) ? res.slice(separ+7) : "";
         });

      return Promise.resolve("");
   }

   /** @summary Show context menu for specified item
     * @private */
   itemContextMenu(name) {
       let rrr = this.svg_this_pad().node().getBoundingClientRect(),
           evnt = { clientX: rrr.left+10, clientY: rrr.top + 10 };

       // use timeout to avoid conflict with mouse click and automatic menu close
       if (name == "pad")
          return setTimeout(() => this.padContextMenu(evnt), 50);

       let selp = null, selkind;

       switch(name) {
          case "xaxis":
          case "yaxis":
          case "zaxis":
             selp = this.getMainPainter();
             selkind = name[0];
             break;
          case "frame":
             selp = this.getFramePainter();
             break;
          default: {
             let indx = parseInt(name);
             if (Number.isInteger(indx)) selp = this.painters[indx];
          }
       }

       if (!selp || (typeof selp.fillContextMenu !== 'function')) return;

       createMenu(evnt, selp).then(menu => {
          if (selp.fillContextMenu(menu, selkind))
             setTimeout(() => menu.show(), 50);
       });
   }

   /** @summary Save pad in specified format
     * @desc Used from context menu */
   saveAs(kind, full_canvas, filename) {
      if (!filename)
         filename = (this.this_pad_name || (this.iscan ? "canvas" : "pad")) + "." + kind;
      console.log('saveAs', kind, full_canvas, filename);

      this.produceImage(full_canvas, kind).then(imgdata => {
         if (!imgdata)
            return console.error(`Fail to produce image ${filename}`);

         let a = document.createElement('a');
         a.download = filename;
         a.href = (kind != "svg") ? imgdata : "data:image/svg+xml;charset=utf-8,"+encodeURIComponent(imgdata);
         document.body.appendChild(a);
         a.addEventListener("click", () => a.parentNode.removeChild(a));
         a.click();
      });
   }

   /** @summary Prodce image for the pad
     * @returns {Promise} with created image */
   produceImage(full_canvas, file_format) {

      let use_frame = (full_canvas === "frame"),
          elem = use_frame ? this.getFrameSvg(this.this_pad_name) : (full_canvas ? this.getCanvSvg() : this.svg_this_pad()),
          painter = (full_canvas && !use_frame) ? this.getCanvPainter() : this,
          items = []; // keep list of replaced elements, which should be moved back at the end

      if (elem.empty())
         return Promise.resolve("");

      if (!use_frame) // do not make transformations for the frame
      painter.forEachPainterInPad(pp => {

         let item = { prnt: pp.svg_this_pad() };
         items.push(item);

         // remove buttons from each subpad
         let btns = pp.getLayerSvg("btns_layer", this.this_pad_name);
         item.btns_node = btns.node();
         if (item.btns_node) {
            item.btns_prnt = item.btns_node.parentNode;
            item.btns_next = item.btns_node.nextSibling;
            btns.remove();
         }

         let main = pp.getFramePainter();
         if (!main || (typeof main.render3D !== 'function') || (typeof main.access3dKind != 'function')) return;

         let can3d = main.access3dKind();

         if ((can3d !== constants.Embed3D.Overlay) && (can3d !== constants.Embed3D.Embed)) return;

         let sz2 = main.getSizeFor3d(constants.Embed3D.Embed), // get size and position of DOM element as it will be embed
             canvas = main.renderer.domElement;

         main.render3D(0); // WebGL clears buffers, therefore we should render scene and convert immediately

         let dataUrl = canvas.toDataURL("image/png");

         // remove 3D drawings

         if (can3d === constants.Embed3D.Embed) {
            item.foreign = item.prnt.select("." + sz2.clname);
            item.foreign.remove();
         }

         let svg_frame = main.getFrameSvg();
         item.frame_node = svg_frame.node();
         if (item.frame_node) {
            item.frame_next = item.frame_node.nextSibling;
            svg_frame.remove();
         }

         // add svg image
         item.img = item.prnt.insert("image",".primitives_layer")     // create image object
                        .attr("x", sz2.x)
                        .attr("y", sz2.y)
                        .attr("width", canvas.width)
                        .attr("height", canvas.height)
                        .attr("href", dataUrl);

      }, "pads");

      const reEncode = data => {
         data = encodeURIComponent(data);
         data = data.replace(/%([0-9A-F]{2})/g, function(match, p1) {
           let c = String.fromCharCode('0x'+p1);
           return c === '%' ? '%25' : c;
         });
         return decodeURIComponent(data);
      }, reconstruct = () => {
         for (let k = 0; k < items.length; ++k) {
            let item = items[k];

            if (item.img)
               item.img.remove(); // delete embed image

            let prim = item.prnt.select(".primitives_layer");

            if (item.foreign) // reinsert foreign object
               item.prnt.node().insertBefore(item.foreign.node(), prim.node());

            if (item.frame_node) // reinsert frame as first in list of primitives
               prim.node().insertBefore(item.frame_node, item.frame_next);

            if (item.btns_node) // reinsert buttons
               item.btns_prnt.insertBefore(item.btns_node, item.btns_next);
         }
      };

      let width = elem.property('draw_width'), height = elem.property('draw_height');
      if (use_frame) {
         let fp = this.getFramePainter();
         width = fp.getFrameWidth();
         height = fp.getFrameHeight();
      }

      let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">${elem.node().innerHTML}</svg>`;

      if (internals.processSvgWorkarounds)
         svg = internals.processSvgWorkarounds(svg);

      svg = compressSVG(svg);

      if (file_format == "svg") {
         reconstruct();
         return Promise.resolve(svg); // return SVG file as is
      }

      let doctype = '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">',
          image = new Image();

      return new Promise(resolveFunc => {
         image.onload = function() {
            let canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            let context = canvas.getContext('2d');
            context.drawImage(image, 0, 0);

            reconstruct();

            resolveFunc(canvas.toDataURL('image/' + file_format));
         }

         image.onerror = function(arg) {
            console.log('IMAGE ERROR', arg);
            reconstruct();
            resolveFunc(null);
         }

         image.src = 'data:image/svg+xml;base64,' + window.btoa(reEncode(doctype + svg));
      });
   }

   /** @summary Process pad button click */
   clickPadButton(funcname, evnt) {

      if (funcname == "CanvasSnapShot")
         return this.saveAs("png", true);

      if (funcname == "enlargePad")
         return this.enlargePad();

      if (funcname == "PadSnapShot")
         return this.saveAs("png", false);

      if (funcname == "PadContextMenus") {

         if (evnt) {
            evnt.preventDefault();
            evnt.stopPropagation();
         }

         if (closeMenu()) return;

         createMenu(evnt, this).then(menu => {
            menu.add("header:Menus");

            if (this.iscan)
               menu.add("Canvas", "pad", this.itemContextMenu);
            else
               menu.add("Pad", "pad", this.itemContextMenu);

            if (this.getFramePainter())
               menu.add("Frame", "frame", this.itemContextMenu);

            let main = this.getMainPainter(); // here pad painter method

            if (main) {
               menu.add("X axis", "xaxis", this.itemContextMenu);
               menu.add("Y axis", "yaxis", this.itemContextMenu);
               if ((typeof main.getDimension === 'function') && (main.getDimension() > 1))
                  menu.add("Z axis", "zaxis", this.itemContextMenu);
            }

            if (this.painters && (this.painters.length>0)) {
               menu.add("separator");
               let shown = [];
               for (let n=0;n<this.painters.length;++n) {
                  let pp = this.painters[n];
                  let obj = pp ? pp.getObject() : null;
                  if (!obj || (shown.indexOf(obj)>=0)) continue;

                  let name = ('_typename' in obj) ? (obj._typename + "::") : "";
                  if ('fName' in obj) name += obj.fName;
                  if (name.length==0) name = "item" + n;
                  menu.add(name, n, this.itemContextMenu);
               }
            }

            menu.show();
         });

         return;
      }

      // click automatically goes to all sub-pads
      // if any painter indicates that processing completed, it returns true
      let done = false;

      for (let i = 0; i < this.painters.length; ++i) {
         let pp = this.painters[i];

         if (typeof pp.clickPadButton == 'function')
            pp.clickPadButton(funcname);

         if (!done && (typeof pp.clickButton == 'function'))
            done = pp.clickButton(funcname);
      }
   }

   /** @summary Add button to the pad
     * @private */
   addPadButton(_btn, _tooltip, _funcname, _keyname) {
      if (!settings.ToolBar || isBatchMode() || this.batch_mode) return;

      if (!this._buttons) this._buttons = [];
      // check if there are duplications

      for (let k=0;k<this._buttons.length;++k)
         if (this._buttons[k].funcname == _funcname) return;

      this._buttons.push({ btn: _btn, tooltip: _tooltip, funcname: _funcname, keyname: _keyname });

      let iscan = this.iscan || !this.has_canvas;
      if (!iscan && (_funcname.indexOf("Pad")!=0) && (_funcname !== "enlargePad")) {
         let cp = this.getCanvPainter();
         if (cp && (cp!==this)) cp.addPadButton(_btn, _tooltip, _funcname);
      }
   }

   /** @summary Add buttons for pad or canvas
     * @private */
   addPadButtons(is_online) {

      this.addPadButton("camera", "Create PNG", this.iscan ? "CanvasSnapShot" : "PadSnapShot", "Ctrl PrintScreen");

      if (settings.ContextMenu)
         this.addPadButton("question", "Access context menus", "PadContextMenus");

      let add_enlarge = !this.iscan && this.has_canvas && this.hasObjectsToDraw()

      if (add_enlarge || this.enlargeMain('verify'))
         this.addPadButton("circle", "Enlarge canvas", "enlargePad");

      if (is_online && this.brlayout) {
         this.addPadButton("diamand", "Toggle Ged", "ToggleGed");
         this.addPadButton("three_circles", "Toggle Status", "ToggleStatus");
      }

   }

   /** @summary Show pad buttons
     * @private */
   showPadButtons() {
      if (!this._buttons) return;

      PadButtonsHandler.assign(this);
      this.showPadButtons();
   }

   /** @summary Calculates RPadLength value */
   getPadLength(vertical, len, frame_painter) {
      let sign = vertical ? -1 : 1,
          rect, res,
          getV = (indx, dflt) => (indx < len.fArr.length) ? len.fArr[indx] : dflt,
          getRect = () => {
             if (!rect)
                rect = frame_painter ? frame_painter.getFrameRect() : this.getPadRect();
             return rect;
          };

      if (frame_painter) {
         let user = getV(2), func = vertical ? "gry" : "grx";
         if ((user !== undefined) && frame_painter[func])
            res = frame_painter[func](user);
      }

      if (res === undefined)
         res = vertical ? getRect().height : 0;

      let norm = getV(0, 0), pixel = getV(1, 0);

      res += sign*pixel;

      if (norm)
         res += sign * (vertical ? getRect().height : getRect().width) * norm;

      return Math.round(res);
   }


   /** @summary Calculates pad position for RPadPos values
     * @param {object} pos - instance of RPadPos
     * @param {object} frame_painter - if drawing will be performed inside frame, frame painter */
   getCoordinate(pos, frame_painter) {
      return {
         x: this.getPadLength(false, pos.fHoriz, frame_painter),
         y: this.getPadLength(true, pos.fVert, frame_painter)
      }
   }

   /** @summary Decode pad draw options */
   decodeOptions(opt) {
      let pad = this.getObject();
      if (!pad) return;

      let d = new DrawOptions(opt);

      if (!this.options) this.options = {};

      Object.assign(this.options, { GlobalColors: true, LocalColors: false, IgnorePalette: false, RotateFrame: false, FixFrame: false });

      if (d.check('NOCOLORS') || d.check('NOCOL')) this.options.GlobalColors = this.options.LocalColors = false;
      if (d.check('LCOLORS') || d.check('LCOL')) { this.options.GlobalColors = false; this.options.LocalColors = true; }
      if (d.check('NOPALETTE') || d.check('NOPAL')) this.options.IgnorePalette = true;
      if (d.check('ROTATE')) this.options.RotateFrame = true;
      if (d.check('FIXFRAME')) this.options.FixFrame = true;

      if (d.check('WHITE')) pad.fFillColor = 0;
      if (d.check('LOGX')) pad.fLogx = 1;
      if (d.check('LOGY')) pad.fLogy = 1;
      if (d.check('LOGZ')) pad.fLogz = 1;
      if (d.check('LOG')) pad.fLogx = pad.fLogy = pad.fLogz = 1;
      if (d.check('GRIDX')) pad.fGridx = 1;
      if (d.check('GRIDY')) pad.fGridy = 1;
      if (d.check('GRID')) pad.fGridx = pad.fGridy = 1;
      if (d.check('TICKX')) pad.fTickx = 1;
      if (d.check('TICKY')) pad.fTicky = 1;
      if (d.check('TICK')) pad.fTickx = pad.fTicky = 1;
   }

   /** @summary draw RPad object */
   static draw(dom, pad, opt) {
      let painter = new RPadPainter(dom, pad, false);
      painter.decodeOptions(opt);

      if (painter.getCanvSvg().empty()) {
         painter.has_canvas = false;
         painter.this_pad_name = "";
         painter.setTopPainter();
      } else {
         painter.addToPadPrimitives(painter.pad_name); // must be here due to pad painter
      }

      painter.createPadSvg();

      if (painter.matchObjectType("TPad") && (!painter.has_canvas || painter.hasObjectsToDraw())) {
         painter.addPadButtons();
      }

      // we select current pad, where all drawing is performed
      let prev_name = painter.has_canvas ? painter.selectCurrentPad(painter.this_pad_name) : undefined;

      selectActivePad({ pp: painter, active: false });

      // flag used to prevent immediate pad redraw during first draw
      return painter.drawPrimitives().then(() => {
         painter.showPadButtons();
         // we restore previous pad name
         painter.selectCurrentPad(prev_name);
         return painter;
      });
   }

} // class RPadPainter

export { RPadPainter };
