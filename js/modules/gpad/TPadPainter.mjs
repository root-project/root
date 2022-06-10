import { gStyle, settings, constants, internals,
         create, toJSON, isBatchMode, loadScript, injectCode, isPromise } from '../core.mjs';
import { color as d3_color, pointer as d3_pointer, select as d3_select } from '../d3.mjs';
import { ColorPalette, adoptRootColors, extendRootColors, getRGBfromTColor } from '../base/colors.mjs';
import { getElementRect, getAbsPosInCanvas, DrawOptions, compressSVG } from '../base/BasePainter.mjs';
import { ObjectPainter, selectActivePad, getActivePad  } from '../base/ObjectPainter.mjs';
import { TAttLineHandler } from '../base/TAttLineHandler.mjs';
import { createMenu, closeMenu } from '../gui/menu.mjs';
import { ToolbarIcons, registerForResize } from '../gui/utils.mjs';
import { BrowserLayout } from '../gui/display.mjs';


function getButtonSize(handler, fact) {
   return Math.round((fact || 1) * (handler.iscan || !handler.has_canvas ? 16 : 12));
}

function toggleButtonsVisibility(handler, action) {
   let group = handler.getLayerSvg("btns_layer", handler.this_pad_name),
       btn = group.select("[name='Toggle']");

   if (btn.empty()) return;

   let state = btn.property('buttons_state');

   if (btn.property('timout_handler')) {
      if (action!=='timeout') clearTimeout(btn.property('timout_handler'));
      btn.property('timout_handler', null);
   }

   let is_visible = false;
   switch(action) {
      case 'enable': is_visible = true; break;
      case 'enterbtn': return; // do nothing, just cleanup timeout
      case 'timeout': is_visible = false; break;
      case 'toggle':
         state = !state;
         btn.property('buttons_state', state);
         is_visible = state;
         break;
      case 'disable':
      case 'leavebtn':
         if (!state) btn.property('timout_handler', setTimeout(() => toggleButtonsVisibility(handler, 'timeout'), 1200));
         return;
   }

   group.selectAll('svg').each(function() {
      if (this===btn.node()) return;
      d3_select(this).style('display', is_visible ? "" : "none");
   });
}

let PadButtonsHandler = {

   alignButtons(btns, width, height) {
      let sz0 = getButtonSize(this, 1.25), nextx = (btns.property('nextx') || 0) + sz0, btns_x, btns_y;

      if (btns.property('vertical')) {
         btns_x = btns.property('leftside') ? 2 : (width - sz0);
         btns_y = height - nextx;
      } else {
         btns_x = btns.property('leftside') ? 2 : (width - nextx);
         btns_y = height - sz0;
      }

      btns.attr("transform","translate("+btns_x+","+btns_y+")");
   },

   findPadButton(keyname) {
      let group = this.getLayerSvg("btns_layer", this.this_pad_name), found_func = "";
      if (!group.empty())
         group.selectAll("svg").each(function() {
            if (d3_select(this).attr("key") === keyname)
               found_func = d3_select(this).attr("name");
         });

      return found_func;
   },

   removePadButtons() {
      let group = this.getLayerSvg("btns_layer", this.this_pad_name);
      if (!group.empty()) {
         group.selectAll("*").remove();
         group.property("nextx", null);
      }
   },

   showPadButtons() {
      let group = this.getLayerSvg("btns_layer", this.this_pad_name);
      if (group.empty()) return;

      // clean all previous buttons
      group.selectAll("*").remove();
      if (!this._buttons) return;

      let iscan = this.iscan || !this.has_canvas, ctrl,
          x = group.property('leftside') ? getButtonSize(this, 1.25) : 0, y = 0;

      if (this._fast_drawing) {
         ctrl = ToolbarIcons.createSVG(group, ToolbarIcons.circle, getButtonSize(this), "enlargePad");
         ctrl.attr("name", "Enlarge").attr("x", 0).attr("y", 0)
             .on("click", evnt => this.clickPadButton("enlargePad", evnt));
      } else {
         ctrl = ToolbarIcons.createSVG(group, ToolbarIcons.rect, getButtonSize(this), "Toggle tool buttons");

         ctrl.attr("name", "Toggle").attr("x", 0).attr("y", 0)
             .property("buttons_state", (settings.ToolBar!=='popup'))
             .on("click", () => toggleButtonsVisibility(this, 'toggle'))
             .on("mouseenter", () => toggleButtonsVisibility(this, 'enable'))
             .on("mouseleave", () => toggleButtonsVisibility(this, 'disable'));

         for (let k = 0; k < this._buttons.length; ++k) {
            let item = this._buttons[k];

            let btn = item.btn;
            if (typeof btn == 'string') btn = ToolbarIcons[btn];
            if (!btn) btn = ToolbarIcons.circle;

            let svg = ToolbarIcons.createSVG(group, btn, getButtonSize(this),
                        item.tooltip + (iscan ? "" : (" on pad " + this.this_pad_name)) + (item.keyname ? " (keyshortcut " + item.keyname + ")" : ""));

            if (group.property('vertical'))
                svg.attr("x", y).attr("y", x);
            else
               svg.attr("x", x).attr("y", y);

            svg.attr("name", item.funcname)
               .style('display', (ctrl.property("buttons_state") ? '' : 'none'))
               .on("mouseenter", () => toggleButtonsVisibility(this, 'enterbtn'))
               .on("mouseleave", () => toggleButtonsVisibility(this, 'leavebtn'));

            if (item.keyname) svg.attr("key", item.keyname);

            svg.on("click", evnt => this.clickPadButton(item.funcname, evnt));

            x += getButtonSize(this, 1.25);
         }
      }

      group.property("nextx", x);

      this.alignButtons(group, this.getPadWidth(), this.getPadHeight());

      if (group.property('vertical'))
         ctrl.attr("y", x);
      else if (!group.property('leftside'))
         ctrl.attr("x", x);
   },

   assign(painter) {
      Object.assign(painter, this);
   }

} // PadButtonsHandler



// identifier used in TWebCanvas painter
const webSnapIds = { kNone: 0,  kObject: 1, kSVG: 2, kSubPad: 3, kColors: 4, kStyle: 5 };

/**
  * @summary Painter for TPad object
  * @private
  */

class TPadPainter extends ObjectPainter {

   /** @summary constructor
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} pad - TPad object to draw
     * @param {boolean} [iscan] - if TCanvas object */
   constructor(dom, pad, iscan) {
      super(dom, pad);
      this.pad = pad;
      this.iscan = iscan; // indicate if working with canvas
      this.this_pad_name = "";
      if (!this.iscan && (pad !== null) && ('fName' in pad)) {
         this.this_pad_name = pad.fName.replace(" ", "_"); // avoid empty symbol in pad name
         let regexp = new RegExp("^[A-Za-z][A-Za-z0-9_]*$");
         if (!regexp.test(this.this_pad_name)) this.this_pad_name = 'jsroot_pad_' + internals.id_counter++;
      }
      this.painters = []; // complete list of all painters in the pad
      this.has_canvas = true;
      this.forEachPainter = this.forEachPainterInPad;
   }

   /** @summary Indicates that is is Root6 pad painter
    * @private */
   isRoot6() { return true; }

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
     * @desc Typically main painter is TH1/TH2 object which is drawing axes
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
      delete this.custom_palette;
      delete this._pad_x;
      delete this._pad_y;
      delete this._pad_width;
      delete this._pad_height;
      delete this._doing_draw;
      delete this._interactively_changed;

      this.painters = [];
      this.pad = null;
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

      if (this.pad) {
         rect.szx = Math.round(Math.max(0.0, 0.5 - Math.max(this.pad.fLeftMargin, this.pad.fRightMargin))*w);
         rect.szy = Math.round(Math.max(0.0, 0.5 - Math.max(this.pad.fBottomMargin, this.pad.fTopMargin))*h);
      } else {
         rect.szx = Math.round(0.5*w);
         rect.szy = Math.round(0.5*h);
      }

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
      return (is_root6 === undefined) || is_root6 ? this.pad : null;
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove */
   cleanPrimitives(selector) {
      if (!selector || (typeof selector !== 'function')) return;

      for (let k = this.painters.length-1; k >= 0; --k)
         if (selector(this.painters[k])) {
            this.painters[k].cleanup();
            this.painters.splice(k, 1);
         }
   }

  /** @summary returns custom palette associated with pad or top canvas
    * @private */
   getCustomPalette() {
      if (this.custom_palette)
         return this.custom_palette;

      let cp = this.getCanvPainter();
      return cp?.custom_palette;
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
            if (kind != "objects") sub.forEachPainterInPad(userfunc, kind);
         } else if (kind != "pads") {
            userfunc(sub);
         }
      }
   }

   /** @summary register for pad events receiver
     * @desc in pad painter, while pad may be drawn without canvas */
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

      if (canp) canp.producePadEvent("select", this, _painter, pos, _place);
   }

   /** @summary Draw pad active border
    * @private */
   drawActiveBorder(svg_rect, is_active) {
      if (is_active !== undefined) {
         if (this.is_active_pad === is_active) return;
         this.is_active_pad = is_active;
      }

      if (this.is_active_pad === undefined) return;

      if (!svg_rect)
         svg_rect = this.iscan ? this.getCanvSvg().select(".canvas_fillrect") :
                                 this.svg_this_pad().select(".root_pad_border");

      let lineatt = this.is_active_pad ? new TAttLineHandler({ style: 1, width: 1, color: "red" }) : this.lineatt;

      if (!lineatt) lineatt = new TAttLineHandler({ color: "none" });

      svg_rect.call(lineatt.func);
   }

   /** @summary Create SVG element for canvas */
   createCanvasSvg(check_resize, new_size) {

      let factor = null, svg = null, lmt = 5, rect = null, btns, info, frect;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.getCanvSvg();

         if (svg.empty()) return false;

         factor = svg.property('height_factor');

         rect = this.testMainResize(check_resize, null, factor);

         if (!rect.changed) return false;

         if (!isBatchMode())
            btns = this.getLayerSvg("btns_layer", this.this_pad_name);

         info = this.getLayerSvg("info_layer", this.this_pad_name);
         frect = svg.select(".canvas_fillrect");

      } else {

         let render_to = this.selectDom();

         if (render_to.style('position')=='static')
            render_to.style('position','relative');

         svg = render_to.append("svg")
             .attr("class", "jsroot root_canvas")
             .property('pad_painter', this) // this is custom property
             .property('current_pad', "") // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         this.setTopPainter(); //assign canvas as top painter of that element

         if (isBatchMode()) {
            svg.attr("xmlns", "http://www.w3.org/2000/svg");
            svg.attr("xmlns:xlink", "http://www.w3.org/1999/xlink");
         } else if (!this.online_canvas) {
            svg.append("svg:title").text("ROOT canvas");
         }

         if (!isBatchMode() || (this.pad.fFillStyle > 0))
            frect = svg.append("svg:path").attr("class","canvas_fillrect");

         if (!isBatchMode())
            frect.style("pointer-events", "visibleFill")
                 .on("dblclick", evnt => this.enlargePad(evnt))
                 .on("click", () => this.selectObjectPainter())
                 .on("mouseenter", () => this.showObjectStatus())
                 .on("contextmenu", settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);

         svg.append("svg:g").attr("class","primitives_layer");
         info = svg.append("svg:g").attr("class", "info_layer");
         if (!isBatchMode())
            btns = svg.append("svg:g")
                      .attr("class","btns_layer")
                      .property('leftside', settings.ToolBarSide == 'left')
                      .property('vertical', settings.ToolBarVert);

         factor = 0.66;
         if (this.pad?.fCw && this.pad?.fCh && (this.pad?.fCw > 0)) {
            factor = this.pad.fCh / this.pad.fCw;
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this._fixed_size) {
            render_to.style("overflow","auto");
            rect = { width: this.pad.fCw, height: this.pad.fCh };
            if (!rect.width || !rect.height)
               rect = getElementRect(render_to);
         } else {
            rect = this.testMainResize(2, new_size, factor);
         }
      }

      this.createAttFill({ attr: this.pad });

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

      svg.style("filter", settings.DarkMode || this.pad?.$dark ? "invert(100%)" : null);

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

      if (frect) {
         frect.attr("d", `M0,0H${rect.width}V${rect.height}H0Z`)
              .call(this.fillatt.func);
         this.drawActiveBorder(frect);
      }

      this._fast_drawing = settings.SmallPad && ((rect.width < settings.SmallPad.width) || (rect.height < settings.SmallPad.height));

      if (this.alignButtons && btns)
         this.alignButtons(btns, rect.width, rect.height);

      let dt = info.select(".canvas_date");
      if (!gStyle.fOptDate) {
         dt.remove();
      } else {
         if (dt.empty()) dt = info.append("text").attr("class", "canvas_date");
         let date = new Date(),
             posx = Math.round(rect.width * gStyle.fDateX),
             posy = Math.round(rect.height * (1 - gStyle.fDateY));
         if (!isBatchMode() && (posx < 25)) posx = 25;
         if (gStyle.fOptDate > 1) date.setTime(gStyle.fOptDate*1000);
         dt.attr("transform", `translate(${posx}, ${posy})`)
           .style("text-anchor", "start")
           .text(date.toLocaleString('en-GB'));
      }

      if (!gStyle.fOptFile || !this.getItemName())
         info.select(".canvas_item").remove();
      else
         this.drawItemNameOnCanvas(this.getItemName());

      return true;
   }

   /** @summary Draw item name on canvas if gStyle.fOptFile is configured
     * @private */
   drawItemNameOnCanvas(item_name) {
      let info = this.getLayerSvg("info_layer", this.this_pad_name),
          df = info.select(".canvas_item");
      if (!gStyle.fOptFile || !item_name) {
         df.remove();
      } else {
         if (df.empty()) df = info.append("text").attr("class", "canvas_item");
         let rect = this.getPadRect(),
             posx = Math.round(rect.width * (1 - gStyle.fDateX)),
             posy = Math.round(rect.height * (1 - gStyle.fDateY));
         df.attr("transform", `translate(${posx}, ${posy})`)
           .style("text-anchor", "end")
           .text(item_name);
      }
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

   /** @summary Create main SVG element for pad
     * @returns true when pad is displayed and all its items should be redrawn */
   createPadSvg(only_resize) {

      if (!this.has_canvas) {
         this.createCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      let svg_can = this.getCanvSvg(),
          width = svg_can.property("draw_width"),
          height = svg_can.property("draw_height"),
          pad_enlarged = svg_can.property("pad_enlarged"),
          pad_visible = !this.pad_draw_disabled && (!pad_enlarged || (pad_enlarged === this.pad)),
          w = Math.round(this.pad.fAbsWNDC * width),
          h = Math.round(this.pad.fAbsHNDC * height),
          x = Math.round(this.pad.fAbsXlowNDC * width),
          y = Math.round(height * (1 - this.pad.fAbsYlowNDC)) - h,
          svg_pad = null, svg_rect = null, btns = null;

      if (pad_enlarged === this.pad) { w = width; h = height; x = y = 0; }

      if (only_resize) {
         svg_pad = this.svg_this_pad();
         svg_rect = svg_pad.select(".root_pad_border");
         if (!isBatchMode())
            btns = this.getLayerSvg("btns_layer", this.this_pad_name);
      } else {
         svg_pad = svg_can.select(".primitives_layer")
             .append("svg:svg") // here was g before, svg used to blend all drawin outside
             .classed("__root_pad_" + this.this_pad_name, true)
             .attr("pad", this.this_pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this); // this is custom property

         if (!isBatchMode())
            svg_pad.append("svg:title").text("subpad " + this.this_pad_name);

         // need to check attributes directly while attributes objects will be created later
         if (!isBatchMode() || (this.pad.fFillStyle > 0) || ((this.pad.fLineStyle > 0) && (this.pad.fLineColor > 0)))
            svg_rect = svg_pad.append("svg:path").attr("class", "root_pad_border");

         if (!isBatchMode())
            svg_rect.style("pointer-events", "visibleFill") // get events also for not visible rect
                    .on("dblclick", evnt => this.enlargePad(evnt))
                    .on("click", () => this.selectObjectPainter())
                    .on("mouseenter", () => this.showObjectStatus())
                    .on("contextmenu", settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);

         svg_pad.append("svg:g").attr("class","primitives_layer");
         if (!isBatchMode())
            btns = svg_pad.append("svg:g")
                          .attr("class","btns_layer")
                          .property('leftside', settings.ToolBarSide != 'left')
                          .property('vertical', settings.ToolBarVert);
      }

      this.createAttFill({ attr: this.pad });
      this.createAttLine({ attr: this.pad, color0: this.pad.fBorderMode == 0 ? 'none' : '' });

      svg_pad.style("display", pad_visible ? null : "none")
             .attr("viewBox", `0 0 ${w} ${h}`) // due to svg
             .attr("preserveAspectRatio", "none")   // due to svg, we do not preserve relative ratio
             .attr("x", x)        // due to svg
             .attr("y", y)        // due to svg
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

      if (svg_rect) {
         svg_rect.attr("d", `M0,0H${w}V${h}H0Z`)
                 .call(this.fillatt.func)
                 .call(this.lineatt.func);

         this.drawActiveBorder(svg_rect);
      }

      this._fast_drawing = settings.SmallPad && ((w < settings.SmallPad.width) || (h < settings.SmallPad.height));

      // special case of 3D canvas overlay
      if (svg_pad.property('can3d') === constants.Embed3D.Overlay)
          this.selectDom().select(".draw3d_" + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');

      if (this.alignButtons && btns)
         this.alignButtons(btns, w, h);

      return pad_visible;
   }

   /** @summary Disable pad drawing
     * @desc Complete SVG element will be hidden */
   disablePadDrawing() {
      if (!this.pad_draw_disabled && this.has_canvas && !this.iscan) {
         this.pad_draw_disabled = true;
         this.createPadSvg(true);
      }
   }

   /** @summary Check if it is special object, which should be handled separately
     * @desc It can be TStyle or list of colors or palette object
     * @returns {boolean} tru if any */
   checkSpecial(obj) {

      if (!obj) return false;

      if (obj._typename == "TStyle") {
         Object.assign(gStyle, obj);
         return true;
      }

      if ((obj._typename == "TObjArray") && (obj.name == "ListOfColors")) {

         if (this.options && this.options.CreatePalette) {
            let arr = [];
            for (let n = obj.arr.length - this.options.CreatePalette; n<obj.arr.length; ++n) {
               let col = getRGBfromTColor(obj.arr[n]);
               if (!col) { console.log('Fail to create color for palette'); arr = null; break; }
               arr.push(col);
            }
            if (arr) this.custom_palette = new ColorPalette(arr);
         }

         if (!this.options || this.options.GlobalColors) // set global list of colors
            adoptRootColors(obj);

         // copy existing colors and extend with new values
         if (this.options && this.options.LocalColors)
            this.root_colors = extendRootColors(null, obj);
         return true;
      }

      if ((obj._typename == "TObjArray") && (obj.name == "CurrentColorPalette")) {
         let arr = [], missing = false;
         for (let n = 0; n < obj.arr.length; ++n) {
            let col = obj.arr[n];
            if (col && (col._typename == 'TColor')) {
               arr[n] = getRGBfromTColor(col);
            } else {
               console.log('Missing color with index ' + n); missing = true;
            }
         }
         if (!this.options || (!missing && !this.options.IgnorePalette))
            this.custom_palette = new ColorPalette(arr);
         return true;
      }

      return false;
   }

   /** @summary Check if special objects appears in primitives
     * @desc it could be list of colors or palette */
   checkSpecialsInPrimitives(can) {
      let lst = can?.fPrimitives;
      if (!lst) return;
      for (let i = 0; i < lst.arr.length; ++i) {
         if (this.checkSpecial(lst.arr[i])) {
            lst.arr.splice(i,1);
            lst.opt.splice(i,1);
            i--;
         }
      }
   }

   /** @summary try to find object by name in list of pad primitives
     * @desc used to find title drawing
     * @private */
   findInPrimitives(objname, objtype) {
      let arr = this.pad?.fPrimitives?.arr;

      return arr ? arr.find(obj => (obj.fName == objname) && (objtype ? (obj.typename == objtype) : true)) : null;
   }

   /** @summary Try to find painter for specified object
     * @desc can be used to find painter for some special objects, registered as
     * histogram functions
     * @param {object} selobj - object to which painter should be search, set null to ignore parameter
     * @param {string} [selname] - object name, set to null to ignore
     * @param {string} [seltype] - object type, set to null to ignore
     * @returns {object} - painter for specified object (if any)
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

   /** @summary Return true if any objects beside sub-pads exists in the pad */
   hasObjectsToDraw() {
      let arr = this.pad?.fPrimitives?.arr;
      return arr && arr.find(obj => obj._typename != "TPad") ? true : false;
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

   /** @summary indicates if painter performing objects draw
     * @private */
   doingDraw() {
      return this._doing_draw !== undefined;
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
   drawObject(/* dom, obj, opt */) {
      console.log('Not possible to draw object without loading of draw.mjs');
      return Promise.resolve(null);
   }

   /** @summary Draw pad primitives
     * @returns {Promise} when drawing completed
     * @private */
   drawPrimitives(indx) {

      if (indx === undefined) {
         if (this.iscan)
            this._start_tm = new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad?.fPrimitives?.arr?.length || 0;

         // sync to prevent immediate pad redraw during normal drawing sequence
         return this.syncDraw(true).then(() => this.drawPrimitives(0));
      }

      if (indx >= this._num_primitives) {
         if (this._start_tm) {
            let spenttm = new Date().getTime() - this._start_tm;
            if (spenttm > 1000) console.log(`Canvas ${this.pad?.fName || "---"} drawing took ${(spenttm*1e-3).toFixed(2)}s`);
            delete this._start_tm;
         }

         this.confirmDraw();
         return Promise.resolve();
      }

      // use of Promise should avoid large call-stack depth when many primitives are drawn
      return this.drawObject(this.getDom(), this.pad.fPrimitives.arr[indx], this.pad.fPrimitives.opt[indx]).then(op => {
         if (op && (typeof op == 'object'))
            op._primitive = true; // mark painter as belonging to primitives

         return this.drawPrimitives(indx+1);
      });
   }

   /** @summary Divide pad on subpads
     * @returns {Promise} when finished
     * @private */
   divide(nx, ny) {
      if (!ny) {
         let ndiv = nx;
         if (ndiv < 2) return Promise.resolve(this);
         nx = ny = Math.round(Math.sqrt(ndiv));
         if (nx*ny < ndiv) nx += 1;
      }

      if (nx*ny < 2) return Promise.resolve(this);

      let xmargin = 0.01, ymargin = 0.01,
          dy = 1/ny, dx = 1/nx, n = 0, subpads = [];
      for (let iy = 0; iy < ny; iy++) {
         let y2 = 1 - iy*dy - ymargin,
             y1 = y2 - dy + 2*ymargin;
         if (y1 < 0) y1 = 0;
         if (y1 > y2) continue;
         for (let ix = 0; ix < nx; ix++) {
            let x1 = ix*dx + xmargin,
                x2 = x1 +dx -2*xmargin;
            if (x1 > x2) continue;
            n++;
            let pad = create("TPad");
            pad.fName = pad.fTitle = this.pad.fName + "_" + n;
            pad.fNumber = n;
            if (!this.iscan) {
               pad.fAbsWNDC = (x2-x1) * this.pad.fAbsWNDC;
               pad.fAbsHNDC = (y2-y1) * this.pad.fAbsHNDC;
               pad.fAbsXlowNDC = this.pad.fAbsXlowNDC + x1 * this.pad.fAbsWNDC;
               pad.fAbsYlowNDC = this.pad.fAbsYlowNDC + y1 * this.pad.fAbsWNDC;
            } else {
               pad.fAbsWNDC = x2 - x1;
               pad.fAbsHNDC = y2 - y1;
               pad.fAbsXlowNDC = x1;
               pad.fAbsYlowNDC = y1;
            }

            subpads.push(pad);
         }
      }

      const drawNext = () => {
         if (subpads.length == 0)
            return Promise.resolve(this);
         return this.drawObject(this.getDom(), subpads.shift()).then(drawNext);
      };

      return drawNext();
   }

   /** @summary Return sub-pads painter, only direct childs are checked
     * @private */
   getSubPadPainter(n) {
      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];
         if (sub.pad && (typeof sub.forEachPainterInPad === 'function') && (sub.pad.fNumber === n)) return sub;
      }
      return null;
   }


   /** @summary Process tooltip event in the pad
     * @private */
   processPadTooltipEvent(pnt) {
      let painters = [], hints = [];

      // first count - how many processors are there
      if (this.painters !== null)
         this.painters.forEach(obj => {
            if (typeof obj.processTooltipEvent == 'function')
               painters.push(obj);
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
      this.getCanvSvg().style("filter", (mode ?? settings.DarkMode) ? "invert(100%)" : null);
   }

   /** @summary Fill pad context menu
     * @private */
   fillContextMenu(menu) {

      if (this.pad)
         menu.add("header:" + this.pad._typename + "::" + this.pad.fName);
      else
         menu.add("header:Canvas");

      menu.addchk(this.isTooltipAllowed(), "Show tooltips", () => this.setTooltipAllowed("toggle"));

      if (!this._websocket) {

         function SetPadField(arg) {
            this.pad[arg.slice(1)] = parseInt(arg[0]);
            this.interactiveRedraw("pad", arg.slice(1));
         }

         menu.addchk(this.pad.fGridx, 'Grid x', (this.pad.fGridx ? '0' : '1') + 'fGridx', SetPadField);
         menu.addchk(this.pad.fGridy, 'Grid y', (this.pad.fGridy ? '0' : '1') + 'fGridy', SetPadField);
         menu.add("sub:Ticks x");
         menu.addchk(this.pad.fTickx == 0, "normal", "0fTickx", SetPadField);
         menu.addchk(this.pad.fTickx == 1, "ticks on both sides", "1fTickx", SetPadField);
         menu.addchk(this.pad.fTickx == 2, "labels on both sides", "2fTickx", SetPadField);
         menu.add("endsub:");
         menu.add("sub:Ticks y");
         menu.addchk(this.pad.fTicky == 0, "normal", "0fTicky", SetPadField);
         menu.addchk(this.pad.fTicky == 1, "ticks on both sides", "1fTicky", SetPadField);
         menu.addchk(this.pad.fTicky == 2, "labels on both sides", "2fTicky", SetPadField);
         menu.add("endsub:");

         menu.addAttributesMenu(this);
         menu.add("Save to gStyle", function() {
            if (this.fillatt) this.fillatt.saveToStyle(this.iscan ? "fCanvasColor" : "fPadColor");
            gStyle.fPadGridX = this.pad.fGridX;
            gStyle.fPadGridY = this.pad.fGridX;
            gStyle.fPadTickX = this.pad.fTickx;
            gStyle.fPadTickY = this.pad.fTicky;
            gStyle.fOptLogx = this.pad.fLogx;
            gStyle.fOptLogy = this.pad.fLogy;
            gStyle.fOptLogz = this.pad.fLogz;
         }, "Store pad fill attributes, grid, tick and log scale settings to gStyle");

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
      menu.add(`Save as ${fname}.png`, fname+".png", arg => this.saveAs("png", this.iscan, arg));
      menu.add(`Save as ${fname}.svg`, fname+".svg", arg => this.saveAs("svg", this.iscan, arg));

      return true;
   }

   /** @summary Show pad context menu
     * @private */
   padContextMenu(evnt) {

      if (evnt.stopPropagation) { // this is normal event processing and not emulated jsroot event

         // for debug purposes keep original context menu for small region in top-left corner
         let pos = d3_pointer(evnt, this.svg_this_pad().node());

         if ((pos.length == 2) && (pos[0] >= 0) && (pos[0] < 10) && (pos[1] >= 0) && (pos[1] < 10)) return;

         evnt.stopPropagation(); // disable main context menu
         evnt.preventDefault();  // disable browser context menu

         let fp = this.getFramePainter();
         if (fp) fp.setLastEventPos();
      }

      createMenu(evnt, this).then(menu => {
         this.fillContextMenu(menu);
         return this.fillObjectExecMenu(menu, "");
      }).then(menu => menu.show());
   }

   /** @summary Redraw pad means redraw ourself
     * @returns {Promise} when redrawing ready */
   redrawPad(reason) {

      let sync_promise = this.syncDraw(reason);
      if (sync_promise === false) {
         console.log('Prevent redrawing', this.pad.fName);
         return Promise.resolve(false);
      }

      let showsubitems = true;
      const redrawNext = indx => {
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
         if (this.iscan)
            this.createCanvasSvg(2);
         else
            showsubitems = this.createPadSvg(true);
         return redrawNext(0);
      }).then(() => {
         this.confirmDraw();
         if (getActivePad() === this) {
            let canp = this.getCanvPainter();
            if (canp) canp.producePadEvent("padredraw", this);
         }
         return true;
      });
   }

   /** @summary redraw pad */
   redraw(reason) {
      // intentially do not return Promise to let re-draw sub-pads in parallel
      this.redrawPad(reason);
   }

   /** @summary Checks if pad should be redrawn by resize
     * @private */
   needRedrawByResize() {
      let elem = this.svg_this_pad();
      if (!elem.empty() && elem.property('can3d') === constants.Embed3D.Overlay) return true;

      return this.painters.findIndex(objp => {
         if (typeof objp.needRedrawByResize === 'function')
            return objp.needRedrawByResize();
      }) >= 0;
   }

   /** @summary Check resize of canvas
     * @returns {Promise} with result */
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

   /** @summary Update TPad object */
   updateObject(obj) {
      if (!obj) return false;

      this.pad.fBits = obj.fBits;
      this.pad.fTitle = obj.fTitle;

      this.pad.fGridx = obj.fGridx;
      this.pad.fGridy = obj.fGridy;
      this.pad.fTickx = obj.fTickx;
      this.pad.fTicky = obj.fTicky;
      this.pad.fLogx  = obj.fLogx;
      this.pad.fLogy  = obj.fLogy;
      this.pad.fLogz  = obj.fLogz;

      this.pad.fUxmin = obj.fUxmin;
      this.pad.fUxmax = obj.fUxmax;
      this.pad.fUymin = obj.fUymin;
      this.pad.fUymax = obj.fUymax;

      this.pad.fX1 = obj.fX1;
      this.pad.fX2 = obj.fX2;
      this.pad.fY1 = obj.fY1;
      this.pad.fY2 = obj.fY2;

      this.pad.fLeftMargin   = obj.fLeftMargin;
      this.pad.fRightMargin  = obj.fRightMargin;
      this.pad.fBottomMargin = obj.fBottomMargin;
      this.pad.fTopMargin    = obj.fTopMargin;

      this.pad.fFillColor = obj.fFillColor;
      this.pad.fFillStyle = obj.fFillStyle;
      this.pad.fLineColor = obj.fLineColor;
      this.pad.fLineStyle = obj.fLineStyle;
      this.pad.fLineWidth = obj.fLineWidth;

      this.pad.fPhi = obj.fPhi;
      this.pad.fTheta = obj.fTheta;

      if (this.iscan) this.checkSpecialsInPrimitives(obj);

      let fp = this.getFramePainter();
      if (fp) fp.updateAttributes(!fp.modified_NDC);

      if (!obj.fPrimitives) return false;

      let isany = false, p = 0;
      for (let n = 0; n < obj.fPrimitives.arr.length; ++n) {
         while (p < this.painters.length) {
            let pp = this.painters[p++];
            if (!pp._primitive) continue;
            if (pp.updateObject(obj.fPrimitives.arr[n])) isany = true;
            break;
         }
      }

      return isany;
   }

   /** @summary Add object painter to list of primitives
     * @private */
   addObjectPainter(objpainter, lst, indx) {
      if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
         // keep snap id in painter, will be used for the
         let pi = this.painters.indexOf(objpainter);
         if (pi < 0) this.painters.push(objpainter);

         if (typeof objpainter.setSnapId == 'function')
            objpainter.setSnapId(lst[indx].fObjectID);
         else
            objpainter.snapid = lst[indx].fObjectID;

         if (objpainter.$primary && (pi > 0) && this.painters[pi-1].$secondary) {
            this.painters[pi-1].snapid = objpainter.snapid + "#hist";
            console.log('ASSIGN SECONDARY HIST ID', this.painters[pi-1].snapid);
         }
      }
   }

   /** @summary Function called when drawing next snapshot from the list
     * @returns {Promise} for drawing of the snap
     * @private */
   drawNextSnap(lst, indx) {

      if (indx === undefined) {
         indx = -1;
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
      }

      ++indx; // change to the next snap

      if (!lst || (indx >= lst.length)) {
         delete this._snaps_map;
         return Promise.resolve(this);
      }

      let snap = lst[indx],
          snapid = snap.fObjectID,
          cnt = this._snaps_map[snapid],
          objpainter = null;

      if (cnt) cnt++; else cnt = 1;
      this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

      // first appropriate painter for the object
      // if same object drawn twice, two painters will exists
      for (let k = 0;  k < this.painters.length; ++k) {
         if (this.painters[k].snapid === snapid)
            if (--cnt === 0) { objpainter = this.painters[k]; break; }
      }

      if (objpainter) {

         if (snap.fKind === webSnapIds.kSubPad) // subpad
            return objpainter.redrawPadSnap(snap).then(() => this.drawNextSnap(lst, indx));

         let promise;

         if (snap.fKind === webSnapIds.kObject) { // object itself
            if (objpainter.updateObject(snap.fSnapshot, snap.fOption))
               promise = objpainter.redraw();
         } else if (snap.fKind === webSnapIds.kSVG) { // update SVG
            if (objpainter.updateObject(snap.fSnapshot))
               promise = objpainter.redraw();
         }

         if (!isPromise(promise)) promise = Promise.resolve(true);

         return promise.then(() => this.drawNextSnap(lst, indx)); // call next
      }

      // gStyle object
      if (snap.fKind === webSnapIds.kStyle) {
         Object.assign(gStyle, snap.fSnapshot);
         return this.drawNextSnap(lst, indx); // call next
      }

      // list of colors
      if (snap.fKind === webSnapIds.kColors) {

         let ListOfColors = [], arr = snap.fSnapshot.fOper.split(";");
         for (let n = 0; n < arr.length; ++n) {
            let name = arr[n], p = name.indexOf(":");
            if (p > 0) {
               ListOfColors[parseInt(name.slice(0,p))] = d3_color("rgb(" + name.slice(p+1) + ")").formatHex();
            } else {
               p = name.indexOf("=");
               ListOfColors[parseInt(name.slice(0,p))] = d3_color("rgba(" + name.slice(p+1) + ")").formatHex();
            }
         }

         // set global list of colors
         if (!this.options || this.options.GlobalColors)
            adoptRootColors(ListOfColors);

         // copy existing colors and extend with new values
         if (this.options && this.options.LocalColors)
            this.root_colors = extendRootColors(null, ListOfColors);

         // set palette
         if (snap.fSnapshot.fBuf && (!this.options || !this.options.IgnorePalette)) {
            let palette = [];
            for (let n = 0; n < snap.fSnapshot.fBuf.length; ++n)
               palette[n] = ListOfColors[Math.round(snap.fSnapshot.fBuf[n])];

            this.custom_palette = new ColorPalette(palette);
         }

         return this.drawNextSnap(lst, indx); // call next
      }

      if (snap.fKind === webSnapIds.kSubPad) { // subpad

         let subpad = snap.fSnapshot;

         subpad.fPrimitives = null; // clear primitives, they just because of I/O

         let padpainter = new TPadPainter(this.getDom(), subpad, false);
         padpainter.decodeOptions(snap.fOption);
         padpainter.addToPadPrimitives(this.this_pad_name);
         padpainter.snapid = snap.fObjectID;

         padpainter.createPadSvg();

         if (padpainter.matchObjectType("TPad") && (snap.fPrimitives.length > 0))
            padpainter.addPadButtons(true);

         // we select current pad, where all drawing is performed
         let prev_name = padpainter.selectCurrentPad(padpainter.this_pad_name);
         return padpainter.drawNextSnap(snap.fPrimitives).then(() => {
            padpainter.selectCurrentPad(prev_name);
            return this.drawNextSnap(lst, indx); // call next
         });
      }

      // here the case of normal drawing, will be handled in promise
      if ((snap.fKind === webSnapIds.kObject) || (snap.fKind === webSnapIds.kSVG))
         return this.drawObject(this.getDom(), snap.fSnapshot, snap.fOption).then(objpainter => {
            this.addObjectPainter(objpainter, lst, indx);
            return this.drawNextSnap(lst, indx);
         });

      return this.drawNextSnap(lst, indx);
   }

   /** @summary Return painter with specified id
     * @private */
   findSnap(snapid) {

      if (this.snapid === snapid) return this;

      if (!this.painters) return null;

      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];

         if (typeof sub.findSnap === 'function')
            sub = sub.findSnap(snapid);
         else if (sub.snapid !== snapid)
            sub = null;

         if (sub) return sub;
      }

      return null;
   }

   /** @summary Redraw pad snap
     * @desc Online version of drawing pad primitives
     * for the canvas snapshot contains list of objects
     * as first entry, graphical properties of canvas itself is provided
     * in ROOT6 it also includes primitives, but we ignore them
     * @returns {Promise} with pad painter when drawing completed
     * @private */
   redrawPadSnap(snap) {
      if (!snap || !snap.fPrimitives)
         return Promise.resolve(this);

      this.is_active_pad = !!snap.fActive; // enforce boolean flag
      this._readonly = (snap.fReadOnly === undefined) ? true : snap.fReadOnly; // readonly flag

      let first = snap.fSnapshot;
      first.fPrimitives = null; // primitives are not interesting, they are disabled in IO

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.snapid = snap.fObjectID;

         this.draw_object = first;
         this.pad = first;
         // this._fixed_size = true;

         // if canvas size not specified in batch mode, temporary use 900x700 size
         if (this.batch_mode && (!first.fCw || !first.fCh)) { first.fCw = 900; first.fCh = 700; }

         // case of ROOT7 with always dummy TPad as first entry
         if (!first.fCw || !first.fCh) this._fixed_size = false;

         let mainid = this.selectDom().attr("id");

         if (!this.batch_mode && !this.use_openui && !this.brlayout && mainid && (typeof mainid == "string")) {
            this.brlayout = new BrowserLayout(mainid, null, this);
            this.brlayout.create(mainid, true);
            // this.brlayout.toggleBrowserKind("float");
            this.setDom(this.brlayout.drawing_divid()); // need to create canvas
            registerForResize(this.brlayout);
         }

         this.createCanvasSvg(0);

         if (!this.batch_mode)
            this.addPadButtons(true);

         if (typeof snap.fHighlightConnect !== 'undefined')
            this._highlight_connect = snap.fHighlightConnect;

         let pr = Promise.resolve(true);

         if ((typeof snap.fScripts == "string") && snap.fScripts) {
            let src = "";

            if (snap.fScripts.indexOf("load:") == 0)
               src = snap.fScripts.slice(5).split(";");
            else if (snap.fScripts.indexOf("assert:") == 0)
               src = snap.fScripts.slice(7);

            pr = src ? loadScript(src) : injectCode(snap.fScripts);
         }

         return pr.then(() => this.drawNextSnap(snap.fPrimitives));
      }

      this.updateObject(first); // update only object attributes

      // apply all changes in the object (pad or canvas)
      if (this.iscan) {
         this.createCanvasSvg(2);
      } else {
         this.createPadSvg(true);
      }

      const MatchPrimitive = (painters, primitives, class_name, obj_name) => {
         let painter = painters.find(p => {
            if (p.snapid === undefined) return;
            if (!p.matchObjectType(class_name)) return;
            if (obj_name && (!p.getObject() || (p.getObject().fName !== obj_name))) return;
            return true;
         });
         if (!painter) return;
         let primitive = primitives.find(pr => {
            if ((pr.fKind !== 1) || !pr.fSnapshot || (pr.fSnapshot._typename !== class_name)) return;
            if (obj_name && (pr.fSnapshot.fName !== obj_name)) return;
            return true;
         });
         if (!primitive) return;

         // force painter to use new object id
         if (painter.snapid !== primitive.fObjectID)
            painter.snapid = primitive.fObjectID;
      };

      // check if frame or title was recreated, we could reassign handlers for them directly
      // while this is temporary objects, which can be recreated very often, try to catch such situation ourselfs
      MatchPrimitive(this.painters, snap.fPrimitives, "TFrame");
      MatchPrimitive(this.painters, snap.fPrimitives, "TPaveText", "title");

      let isanyfound = false, isanyremove = false;

      // find and remove painters which no longer exists in the list
      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];
         if ((sub.snapid===undefined) || sub.$secondary) continue; // look only for painters with snapid

         for (let i = 0; i < snap.fPrimitives.length; ++i)
            if (snap.fPrimitives[i].fObjectID === sub.snapid) { sub = null; isanyfound = true; break; }

         if (sub) {
            // console.log(`Remove painter ${k} from ${this.painters.length} class ${sub.getClassName()} ismain ${sub.isMainPainter()}`);
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
         // TODO: maybe just remove frame painter?
         let fp = this.getFramePainter();
         this.painters.forEach(objp => {
            if (fp !== objp) objp.cleanup();
         });
         delete this.main_painter_ref;
         this.painters = [];
         if (fp) {
            this.painters.push(fp);
            fp.cleanFrameDrawings();
            fp.redraw();
         }
         if (this.removePadButtons) this.removePadButtons();
         this.addPadButtons(true);
      }

      let prev_name = this.selectCurrentPad(this.this_pad_name);

      return this.drawNextSnap(snap.fPrimitives).then(() => {
         // redraw secondaries like stat box
         let promises = [];
         this.painters.forEach(sub => {
            if ((sub.snapid===undefined) || sub.$secondary) {
               let res = sub.redraw();
               if (isPromise(res)) promises.push(res);
            }
         });
         return Promise.all(promises);
      }).then(() => {
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
            if (!res || (format=="svg")) return res;
            let separ = res.indexOf("base64,");
            return (separ>0) ? res.slice(separ+7) : "";
         });

      return Promise.resolve("");
   }

   /** @summary Collects pad information for TWebCanvas
     * @desc need to update different states
     * @private */
   getWebPadOptions(arg) {
      let is_top = (arg === undefined), elem = null, scan_subpads = true;
      // no any options need to be collected in readonly mode
      if (is_top && this._readonly) return "";
      if (arg === "only_this") { is_top = true; scan_subpads = false; }
      if (is_top) arg = [];

      if (this.snapid) {
         elem = { _typename: "TWebPadOptions", snapid: this.snapid.toString(),
                  active: !!this.is_active_pad,
                  bits: 0, primitives: [],
                  logx: this.pad.fLogx, logy: this.pad.fLogy, logz: this.pad.fLogz,
                  gridx: this.pad.fGridx, gridy: this.pad.fGridy,
                  tickx: this.pad.fTickx, ticky: this.pad.fTicky,
                  mleft: this.pad.fLeftMargin, mright: this.pad.fRightMargin,
                  mtop: this.pad.fTopMargin, mbottom: this.pad.fBottomMargin,
                  zx1:0, zx2:0, zy1:0, zy2:0, zz1:0, zz2:0 };

         if (this.iscan) elem.bits = this.getStatusBits();

         if (this.getPadRanges(elem))
            arg.push(elem);
         else
            console.log('fail to get ranges for pad ' +  this.pad.fName);
      }

      this.painters.forEach(sub => {
         if (typeof sub.getWebPadOptions == "function") {
            if (scan_subpads) sub.getWebPadOptions(arg);
         } else if (sub.snapid) {
            let opt = { _typename: "TWebObjectOptions", snapid: sub.snapid.toString(), opt: sub.getDrawOpt(), fcust: "", fopt: [] };
            if (typeof sub.fillWebObjectOptions == "function")
               opt = sub.fillWebObjectOptions(opt);
            elem.primitives.push(opt);
         }
      });

      if (is_top) return toJSON(arg);
   }

   /** @summary returns actual ranges in the pad, which can be applied to the server
     * @private */
   getPadRanges(r) {

      if (!r) return false;

      let main = this.getFramePainter(),
          p = this.svg_this_pad();

      r.ranges = main && main.ranges_set ? true : false; // indicate that ranges are assigned

      r.ux1 = r.px1 = r.ranges ? main.scale_xmin : 0; // need to initialize for JSON reader
      r.uy1 = r.py1 = r.ranges ? main.scale_ymin : 0;
      r.ux2 = r.px2 = r.ranges ? main.scale_xmax : 0;
      r.uy2 = r.py2 = r.ranges ? main.scale_ymax : 0;

      if (main) {
         if (main.zoom_xmin !== main.zoom_xmax) {
            r.zx1 = main.zoom_xmin; r.zx2 = main.zoom_xmax;
         }

         if (main.zoom_ymin !== main.zoom_ymax) {
            r.zy1 = main.zoom_ymin; r.zy2 = main.zoom_ymax;
         }

         if (main.zoom_zmin !== main.zoom_zmax) {
            r.zz1 = main.zoom_zmin; r.zz2 = main.zoom_zmax;
         }
      }

      if (!r.ranges || p.empty()) return true;

      // calculate user range for full pad
      const same = x => x,
            direct_funcs = [same, Math.log10, x => Math.log10(x)/Math.log10(2)],
            revert_funcs = [same, x => Math.pow(10, x), x => Math.pow(2, x)],
            match = (v1, v0, range) => (Math.abs(v0-v1) < Math.abs(range)*1e-10) ? v0 : v1;

      let func = direct_funcs[main.logx],
          func2 = revert_funcs[main.logx],
          k = (func(main.scale_xmax) - func(main.scale_xmin))/p.property("draw_width"),
          x1 = func(main.scale_xmin) - k*p.property("draw_x"),
          x2 = x1 + k*p.property("draw_width");

      r.ux1 = match(func2(x1), r.ux1, r.px2-r.px1);
      r.ux2 = match(func2(x2), r.ux2, r.px2-r.px1);

      func = direct_funcs[main.logy];
      func2 = revert_funcs[main.logy];

      k = (func(main.scale_ymax) - func(main.scale_ymin))/p.property("draw_height");
      let y2 = func(main.scale_ymax) + k*p.property("draw_y"),
          y1 = y2 - k*p.property("draw_height");

      r.uy1 = match(func2(y1), r.uy1, r.py2-r.py1);
      r.uy2 = match(func2(y2), r.uy2, r.py2-r.py1);

      return true;
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
             selp = this.getFramePainter();
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
          items = [], // keep list of replaced elements, which should be moved back at the end
          active_pp = null;

      if (elem.empty())
         return Promise.resolve("");

      painter.forEachPainterInPad(pp => {

          if (pp.is_active_pad && !active_pp) {
             active_pp = pp;
             active_pp.drawActiveBorder(null, false);
          }

         if (use_frame) return; // do not make transformations for the frame

         let item = { prnt: pp.svg_this_pad() };
         items.push(item);

         // remove buttons from each subpad
         let btns = pp.getLayerSvg("btns_layer", pp.this_pad_name);
         item.btns_node = btns.node();
         if (item.btns_node) {
            item.btns_prnt = item.btns_node.parentNode;
            item.btns_next = item.btns_node.nextSibling;
            btns.remove();
         }

         let main = pp.getFramePainter();
         if (!main || (typeof main.render3D !== 'function') || (typeof main.access3dKind !== 'function')) return;

         let can3d = main.access3dKind();

         if ((can3d !== constants.Embed3D.Overlay) && (can3d !== constants.Embed3D.Embed)) return;

         let sz2 = main.getSizeFor3d(constants.Embed3D.Embed); // get size and position of DOM element as it will be embed

         let canvas = main.renderer.domElement;
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
         data = data.replace(/%([0-9A-F]{2})/g, (match, p1) => {
           let c = String.fromCharCode('0x'+p1);
           return c === '%' ? '%25' : c;
         });
         return decodeURIComponent(data);
      }, reconstruct = () => {
         // reactivate border
         if (active_pp)
            active_pp.drawActiveBorder(null, true);

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

            if (this.painters && (this.painters.length > 0)) {
               menu.add("separator");
               let shown = [];
               this.painters.forEach((pp,indx) => {
                  let obj = pp ? pp.getObject() : null;
                  if (!obj || (shown.indexOf(obj) >= 0)) return;
                  if (pp.$secondary) return;
                  let name = ('_typename' in obj) ? (obj._typename + "::") : "";
                  if ('fName' in obj) name += obj.fName;
                  if (!name.length) name = "item" + indx;
                  menu.add(name, indx, this.itemContextMenu);
               });
            }

            menu.show();
         });

         return;
      }

      // click automatically goes to all sub-pads
      // if any painter indicates that processing completed, it returns true
      let done = false;

      this.painters.forEach(pp => {
         if (typeof pp.clickPadButton == 'function')
            pp.clickPadButton(funcname);

         if (!done && (typeof pp.clickButton == 'function'))
            done = pp.clickButton(funcname);
      });
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

   /** @summary Show pad buttons
     * @private */
   showPadButtons() {
      if (!this._buttons) return;

       PadButtonsHandler.assign(this);
       this.showPadButtons();
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

   /** @summary Decode pad draw options
     * @private */
   decodeOptions(opt) {
      let pad = this.getObject();
      if (!pad) return;

      let d = new DrawOptions(opt);

      if (!this.options) this.options = {};

      Object.assign(this.options, { GlobalColors: true, LocalColors: false, CreatePalette: 0, IgnorePalette: false, RotateFrame: false, FixFrame: false });

      if (d.check('NOCOLORS') || d.check('NOCOL')) this.options.GlobalColors = this.options.LocalColors = false;
      if (d.check('LCOLORS') || d.check('LCOL')) { this.options.GlobalColors = false; this.options.LocalColors = true; }
      if (d.check('NOPALETTE') || d.check('NOPAL')) this.options.IgnorePalette = true;
      if (d.check('ROTATE')) this.options.RotateFrame = true;
      if (d.check('FIXFRAME')) this.options.FixFrame = true;

      if (d.check("CP",true)) this.options.CreatePalette = d.partAsInt(0,0);

      if (d.check("NOZOOMX")) this.options.NoZoomX = true;
      if (d.check("NOZOOMY")) this.options.NoZoomY = true;

      if (d.check("NOMARGINS")) pad.fLeftMargin = pad.fRightMargin = pad.fBottomMargin = pad.fTopMargin = 0;
      if (d.check('WHITE')) pad.fFillColor = 0;
      if (d.check('LOG2X')) { pad.fLogx = 2; pad.fUxmin = 0; pad.fUxmax = 1; pad.fX1 = 0; pad.fX2 = 1; }
      if (d.check('LOGX')) { pad.fLogx = 1; pad.fUxmin = 0; pad.fUxmax = 1; pad.fX1 = 0; pad.fX2 = 1; }
      if (d.check('LOG2Y')) { pad.fLogy = 2; pad.fUymin = 0; pad.fUymax = 1; pad.fY1 = 0; pad.fY2 = 1; }
      if (d.check('LOGY')) { pad.fLogy = 1; pad.fUymin = 0; pad.fUymax = 1; pad.fY1 = 0; pad.fY2 = 1; }
      if (d.check('LOG2Z')) pad.fLogz = 2;
      if (d.check('LOGZ')) pad.fLogz = 1;
      if (d.check('LOG2')) pad.fLogx = pad.fLogy = pad.fLogz = 2;
      if (d.check('LOG')) pad.fLogx = pad.fLogy = pad.fLogz = 1;
      if (d.check('GRIDX')) pad.fGridx = 1;
      if (d.check('GRIDY')) pad.fGridy = 1;
      if (d.check('GRID')) pad.fGridx = pad.fGridy = 1;
      if (d.check('TICKX')) pad.fTickx = 1;
      if (d.check('TICKY')) pad.fTicky = 1;
      if (d.check('TICK')) pad.fTickx = pad.fTicky = 1;
      if (d.check('OTX')) pad.$OTX = true;
      if (d.check('OTY')) pad.$OTY = true;
      if (d.check('CTX')) pad.$CTX = true;
      if (d.check('CTY')) pad.$CTY = true;
      if (d.check('RX')) pad.$RX = true;
      if (d.check('RY')) pad.$RY = true;

      this.storeDrawOpt(opt);
   }

   /** @summary draw TPad object */
   static draw(dom, pad, opt) {
      let painter = new TPadPainter(dom, pad, false);
      painter.decodeOptions(opt);

      if (painter.getCanvSvg().empty()) {
         // one can draw pad without canvas
         painter.has_canvas = false;
         painter.this_pad_name = "";
         painter.setTopPainter();
      } else {
         // pad painter will be registered in the canvas painters list
         painter.addToPadPrimitives(painter.pad_name);
      }

      painter.createPadSvg();

      if (painter.matchObjectType("TPad") && (!painter.has_canvas || painter.hasObjectsToDraw()))
         painter.addPadButtons();

      // we select current pad, where all drawing is performed
      let prev_name = painter.has_canvas ? painter.selectCurrentPad(painter.this_pad_name) : undefined;

      // set active pad
      selectActivePad({ pp: painter, active: true });

      // flag used to prevent immediate pad redraw during first draw
      return painter.drawPrimitives().then(() => {
         painter.showPadButtons();
         // we restore previous pad name
         painter.selectCurrentPad(prev_name);
         return painter;
      });
   };

} // class TPadPainter

export { TPadPainter, PadButtonsHandler };
