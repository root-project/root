import { gStyle, settings, browser, constants, internals, addMethods, isPromise, getPromise, postponePromise,
         isBatchMode, isObject, isFunc, isStr, clTFrame, nsREX, nsSVG, urlClassPrefix } from '../core.mjs';
import { select as d3_select } from '../d3.mjs';
import { ColorPalette, addColor, getRootColors, convertColor } from '../base/colors.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';
import { prSVG, getElementRect, getAbsPosInCanvas, DrawOptions, compressSVG, makeTranslate, svgToImage } from '../base/BasePainter.mjs';
import { selectActivePad, getActivePad, isPadPainter } from '../base/ObjectPainter.mjs';
import { registerForResize, saveFile } from '../gui/utils.mjs';
import { BrowserLayout, getHPainter } from '../gui/display.mjs';
import { createMenu, closeMenu } from '../gui/menu.mjs';
import { PadButtonsHandler, webSnapIds } from './TPadPainter.mjs';


/**
 * @summary Painter class for RPad
 *
 * @private
 */

class RPadPainter extends RObjectPainter {

   #iscan;      // is canvas flag
   #pad_name;   // name of the pad
   #pad;        // RPad object
   #painters;   // painters in the pad
   #pad_scale;  // scaling factor of the pad
   #pad_x;      // pad x coordinate
   #pad_y;      // pad y coordinate
   #pad_width;  // pad width
   #pad_height; // pad height
   #doing_draw; // drawing handles
   #custom_palette; // custom palette
   #frame_painter_ref; // frame painter
   #main_painter_ref; // main painter on the pad
   #num_primitives; // number of primitives
   #auto_color_cnt;  // counter for auto colors
   #fixed_size;  // fixed size flag
   #has_canvas;  // when top-level canvas exists
   #fast_drawing; // fast drawing mode
   #resize_tmout; // timeout handle for resize
   #start_draw_tm;  // time when start drawing primitives

   /** @summary constructor */
   constructor(dom, pad, opt, iscan, add_to_primitives) {
      super(dom, pad, '', 'pad');
      this.#pad = pad;
      this.#iscan = iscan; // indicate if working with canvas
      this.#pad_name = '';
      if (!iscan && pad) {
         if (pad.fObjectID)
            this.#pad_name = 'pad' + pad.fObjectID; // use objectid as pad name
         else
            this.#pad_name = 'ppp' + internals.id_counter++; // artificial name
      }
      this.#painters = []; // complete list of all painters in the pad
      this.#has_canvas = true;
      this.forEachPainter = this.forEachPainterInPad;

      const d = this.selectDom();
      if (!d.empty() && d.property('_batch_mode'))
         this.batch_mode = true;

      if (opt !== undefined)
         this.decodeOptions(opt);

      if (add_to_primitives) {
         if ((add_to_primitives !== 'webpad') && this.getCanvSvg().empty()) {
            this.#has_canvas = false;
            this.#pad_name = '';
            this.setTopPainter();
         } else
            this.addToPadPrimitives(); // must be here due to pad painter
      }
   }

   /** @summary Returns pad name
     * @protected */
   getPadName() { return this.#pad_name; }

   /** @summary Indicates that drawing runs in batch mode
     * @private */
   isBatchMode() {
      if (this.batch_mode !== undefined)
         return this.batch_mode;

      if (isBatchMode())
         return true;

      return this.isTopPad() ? false : this.getCanvPainter()?.isBatchMode();
   }

   /** @summary Indicates that is not Root6 pad painter
    * @private */
   isRoot6() { return false; }

   /** @summary Returns true if pad is editable */
   isEditable() { return true; }

   /** @summary Returns true if button */
   isButton() { return false; }

   /** @summary Returns true if it is canvas
    * @param {Boolean} [is_online = false] - if specified, checked if it is canvas with configured connection to server */
   isCanvas(is_online = false) {
      if (!this.#iscan)
         return false;
      if (is_online === true)
         return isFunc(this.getWebsocket) && this.getWebsocket();
      return isStr(is_online) ? this.#iscan === is_online : true;
   }

   /** @summary Returns true if it is canvas or top pad without canvas */
   isTopPad() { return this.isCanvas() || !this.#has_canvas; }

   /** @summary returns pad painter
     * @protected */
   getPadPainter() { return this.isTopPad() ? null : super.getPadPainter(); }

   /** @summary returns canvas painter
     * @protected */
   getCanvPainter(try_select) { return this.isTopPad() ? this : super.getCanvPainter(try_select); }

   /** @summary Canvas main svg element
     * @return {object} d3 selection with canvas svg
     * @protected */
   getCanvSvg() { return this.selectDom().select('.root_canvas'); }

   /** @summary Pad svg element
     * @return {object} d3 selection with pad svg
     * @protected */
   getPadSvg() {
      const c = this.getCanvSvg();
      if (!this.#pad_name || c.empty())
         return c;

      return c.select('.primitives_layer .__root_pad_' + this.#pad_name);
   }

   /** @summary Method selects immediate layer under canvas/pad main element
     * @param {string} name - layer name like 'primitives_layer', 'btns_layer', 'info_layer'
     * @protected */
   getLayerSvg(name) { return this.getPadSvg().selectChild('.' + name); }

   /** @summary Returns svg element for the frame in current pad
     * @protected */
   getFrameSvg() {
      const layer = this.getLayerSvg('primitives_layer');
      if (layer.empty()) return layer;
      let node = layer.node().firstChild;
      while (node) {
         const elem = d3_select(node);
         if (elem.classed('root_frame')) return elem;
         node = node.nextSibling;
      }
      return d3_select(null);
   }

   /** @summary Returns main painter on the pad
     * @desc Typically main painter is TH1/TH2 object which is drawing axes
     * @private */
   getMainPainter() { return this.#main_painter_ref || null; }

   /** @summary Assign main painter on the pad
    * @private */
   setMainPainter(painter, force) {
      if (!this.#main_painter_ref || force)
         this.#main_painter_ref = painter;
   }

   /** @summary cleanup pad and all primitives inside */
   cleanup() {
      if (this.#doing_draw)
         console.error('pad drawing is not completed when cleanup is called');

      this.#painters.forEach(p => p.cleanup());

      const svg_p = this.getPadSvg();
      if (!svg_p.empty()) {
         svg_p.property('pad_painter', null);
         if (!this.isCanvas()) svg_p.remove();
      }

      this.#main_painter_ref = undefined;
      this.#frame_painter_ref = undefined;
      this.#pad_x = this.#pad_y = this.#pad_width = this.#pad_height = undefined;
      this.#doing_draw = undefined;
      delete this._dfltRFont;

      this.#painters = [];
      this.#pad = undefined;
      this.assignObject(null);
      this.#pad_name = undefined;
      this.#has_canvas = false;

      selectActivePad({ pp: this, active: false });

      super.cleanup();
   }

   /** @summary Returns frame painter inside the pad
    * @private */
   getFramePainter() { return this.#frame_painter_ref; }

   /** @summary Assign actual frame painter
     * @private */
   setFramePainter(fp, on) {
      if (on)
         this.#frame_painter_ref = fp;
      else if (this.#frame_painter_ref === fp)
         this.#frame_painter_ref = undefined;
   }

   /** @summary get pad width */
   getPadWidth() { return this.#pad_width || 0; }

   /** @summary get pad height */
   getPadHeight() { return this.#pad_height || 0; }

   /** @summary get pad height */
   getPadScale() { return this.#pad_scale || 1; }

   /** @summary return pad log state x or y are allowed */
   getPadLog(/* name */) { return false; }

   /** @summary get pad rect */
   getPadRect() {
      return {
         x: this.#pad_x || 0,
         y: this.#pad_y || 0,
         width: this.getPadWidth(),
         height: this.getPadHeight()
      };
   }

   /** @summary Returns frame coordinates - also when frame is not drawn */
   getFrameRect() {
      const fp = this.getFramePainter();
      if (fp) return fp.getFrameRect();

      const w = this.getPadWidth(),
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
      rect.transform = makeTranslate(rect.x, rect.y) || '';
      return rect;
   }

   /** @summary return RPad object */
   getRootPad(is_root6) {
      return (is_root6 === undefined) || !is_root6 ? this.#pad : null;
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove
    * @private */
   cleanPrimitives(selector) {
      // remove all primitives
      if (selector === true)
         selector = () => true;

      if (!isFunc(selector))
         return false;

      let is_any = false;

      for (let k = this.#painters.length - 1; k >= 0; --k) {
         const subp = this.#painters[k];
         if (selector(subp)) {
            subp.cleanup();
            this.#painters.splice(k, 1);
            is_any = true;
         }
      }

      return is_any;
   }

   /** @summary Divide pad on sub-pads */
   async divide(/* nx, ny, use_existing */) {
      if (settings.Debug)
         console.warn('RPadPainter.divide not implemented');
      return this;
   }

   /** @summary Removes and cleanup specified primitive
     * @desc also secondary primitives will be removed
     * @return new index to continue loop or -111 if main painter removed
     * @private */
   removePrimitive(arg, clean_only_secondary) {
      let indx, prim = null;
      if (Number.isInteger(arg)) {
         indx = arg; prim = this.#painters[indx];
      } else {
         indx = this.#painters.indexOf(arg); prim = arg;
      }
      if (indx < 0)
         return indx;

      const arr = [];
      let resindx = indx - 1; // object removed itself
      arr.push(prim);
      this.#painters.splice(indx, 1);

      let len0 = 0;
      while (len0 < arr.length) {
         for (let k = this.#painters.length - 1; k >= 0; --k) {
            if (this.#painters[k].isSecondary(arr[len0])) {
               arr.push(this.#painters[k]);
               this.#painters.splice(k, 1);
               if (k <= indx) resindx--;
            }
         }
         len0++;
      }

      arr.forEach(painter => {
         if ((painter !== prim) || !clean_only_secondary)
            painter.cleanup();
         if (this.getMainPainter() === painter) {
            delete this.setMainPainter(undefined, true);
            resindx = -111;
         }
      });

      return resindx;
   }

   /** @summary try to find object by name in list of pad primitives
     * @desc used to find title drawing
     * @private */
   findInPrimitives(/* objname, objtype */) {
      if (settings.Debug)
         console.warn('findInPrimitives not implemented for RPad');
      return null;
   }

   /** @summary Try to find painter for specified object
     * @desc can be used to find painter for some special objects, registered as
     * histogram functions
     * @private */
   findPainterFor(selobj, selname, seltype) {
      return this.#painters.find(p => {
         const pobj = p.getObject();
         if (!pobj) return false;

         if (selobj && (pobj === selobj)) return true;
         if (!selname && !seltype) return false;
         if (selname && (pobj.fName !== selname)) return false;
         if (seltype && (pobj._typename !== seltype)) return false;
         return true;
      });
   }

   /** @summary Returns palette associated with pad.
     * @desc Either from existing palette painter or just default palette */
   getHistPalette() {
      const pp = this.findPainterFor(undefined, undefined, `${nsREX}RPaletteDrawable`);

      if (pp) return pp.getHistPalette();

      if (!this.fDfltPalette) {
         this.fDfltPalette = {
            _typename: `${nsREX}RPalette`,
            fColors: [{ fOrdinal: 0, fColor: { fColor: 'rgb(53, 42, 135)' } },
                      { fOrdinal: 0.125, fColor: { fColor: 'rgb(15, 92, 221)' } },
                      { fOrdinal: 0.25, fColor: { fColor: 'rgb(20, 129, 214)' } },
                      { fOrdinal: 0.375, fColor: { fColor: 'rgb(6, 164, 202)' } },
                      { fOrdinal: 0.5, fColor: { fColor: 'rgb(46, 183, 164)' } },
                      { fOrdinal: 0.625, fColor: { fColor: 'rgb(135, 191, 119)' } },
                      { fOrdinal: 0.75, fColor: { fColor: 'rgb(209, 187, 89)' } },
                      { fOrdinal: 0.875, fColor: { fColor: 'rgb(254, 200, 50)' } },
                      { fOrdinal: 1, fColor: { fColor: 'rgb(249, 251, 14)' } }],
             fInterpolate: true,
             fNormalized: true
         };
         addMethods(this.fDfltPalette, `${nsREX}RPalette`);
      }

      return this.fDfltPalette;
   }

   /** @summary Returns custom palette
     * @private */
   getCustomPalette(no_recursion) {
      return this.#custom_palette || (no_recursion ? null : this.getCanvPainter()?.getCustomPalette(true));
   }

   /** @summary Returns number of painters
     * @protected */
   getNumPainters() { return this.#painters.length; }

   /** @summary Add painter to pad list of painters
     * @protected */
   addToPrimitives(painter) {
      if (this.#painters.indexOf(painter) < 0)
         this.#painters.push(painter);
      return this;
   }

   /** @summary Call function for each painter in pad
     * @param {function} userfunc - function to call
     * @param {string} kind - 'all' for all objects (default), 'pads' only pads and sub-pads, 'objects' only for object in current pad
     * @private */
   forEachPainterInPad(userfunc, kind) {
      if (!kind)
         kind = 'all';
      if (kind !== 'objects') userfunc(this);
      for (let k = 0; k < this.#painters.length; ++k) {
         const sub = this.#painters[k];
         if (isFunc(sub.forEachPainterInPad)) {
            if (kind !== 'objects') sub.forEachPainterInPad(userfunc, kind);
         } else if (kind !== 'pads') userfunc(sub);
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
   producePadEvent(what, padpainter, painter, position) {
      if ((what === 'select') && isFunc(this.selectActivePad))
         this.selectActivePad(padpainter, painter, position);

      if (isFunc(this.pad_events_receiver))
         this.pad_events_receiver({ what, padpainter, painter, position });
   }

   /** @summary method redirect call to pad events receiver */
   selectObjectPainter(painter, pos) {
      const canp = this.isTopPad() ? this : this.getCanvPainter();

      if (painter === undefined)
         painter = this;

      if (pos && !this.isTopPad())
         pos = getAbsPosInCanvas(this.getPadSvg(), pos);

      selectActivePad({ pp: this, active: true });

      canp.producePadEvent('select', this, painter, pos);
   }

   /** @summary Set fast drawing property depending on the size
     * @private */
   setFastDrawing(w, h) {
      const was_fast = this.#fast_drawing;
      this.#fast_drawing = !this.hasSnapId() && settings.SmallPad && ((w < settings.SmallPad.width) || (h < settings.SmallPad.height));
      if (was_fast !== this.#fast_drawing)
         this.showPadButtons();
   }

   /** @summary Return fast drawing flag
   * @private */
   isFastDrawing() { return this.#fast_drawing; }

   /** @summary Returns true if canvas configured with grayscale
     * @private */
   isGrayscale() { return false; }

   /** @summary Set grayscale mode for the canvas
     * @private */
   setGrayscale(/* flag */) {
      console.error('grayscale mode not implemented for RCanvas');
   }

   /** @summary Returns true if default pad range is configured
     * @private */
   isDefaultPadRange() {
      return true;
   }

   /** @summary Create SVG element for the canvas */
   createCanvasSvg(check_resize, new_size) {
      const lmt = 5;
      let factor, svg, rect, btns, frect;

      if (check_resize > 0) {
         if (this.#fixed_size)
            return check_resize > 1; // flag used to force re-drawing of all sub-pads

         svg = this.getCanvSvg();
         if (svg.empty())
            return false;

         factor = svg.property('height_factor');

         rect = this.testMainResize(check_resize, null, factor);

         if (!rect.changed && (check_resize === 1))
            return false;

         if (!this.isBatchMode())
            btns = this.getLayerSvg('btns_layer');

         frect = svg.selectChild('.canvas_fillrect');
      } else {
         const render_to = this.selectDom();

         if (render_to.style('position') === 'static')
            render_to.style('position', 'relative');

         svg = render_to.append('svg')
             .attr('class', 'jsroot root_canvas')
             .property('pad_painter', this) // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         this.setTopPainter(); // assign canvas as top painter of that element

         if (!this.isBatchMode() && !this.online_canvas)
            svg.append('svg:title').text('ROOT canvas');

         if (!this.isBatchMode())
            svg.style('user-select', settings.UserSelect || null);

         frect = svg.append('svg:path').attr('class', 'canvas_fillrect');
         if (!this.isBatchMode()) {
            frect.style('pointer-events', 'visibleFill')
                 .on('dblclick', evnt => this.enlargePad(evnt, true))
                 .on('click', () => this.selectObjectPainter(this, null))
                 .on('mouseenter', () => this.showObjectStatus())
                 .on('contextmenu', settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);
         }

         svg.append('svg:g').attr('class', 'primitives_layer');
         svg.append('svg:g').attr('class', 'info_layer');
         if (!this.isBatchMode()) {
            btns = svg.append('svg:g')
                      .attr('class', 'btns_layer')
                      .property('leftside', settings.ToolBarSide === 'left')
                      .property('vertical', settings.ToolBarVert);
         }

         factor = 0.66;
         if (this.#pad?.fWinSize[0] && this.#pad.fWinSize[1]) {
            factor = this.#pad.fWinSize[1] / this.#pad.fWinSize[0];
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this.#fixed_size) {
            render_to.style('overflow', 'auto');
            rect = { width: this.#pad.fWinSize[0], height: this.#pad.fWinSize[1] };
            if (!rect.width || !rect.height)
               rect = getElementRect(render_to);
         } else
            rect = this.testMainResize(2, new_size, factor);
      }

      this.createAttFill({ pattern: 1001, color: 0 });

      if ((rect.width <= lmt) || (rect.height <= lmt)) {
         if (!this.hasSnapId()) {
            svg.style('display', 'none');
            console.warn(`Hide canvas while geometry too small w=${rect.width} h=${rect.height}`);
         }
         if (this.#pad_width && this.#pad_height) {
            // use last valid dimensions
            rect.width = this.#pad_width;
            rect.height = this.#pad_height;
         } else {
            // just to complete drawing.
            rect.width = 800;
            rect.height = 600;
         }
      } else
         svg.style('display', null);

      if (this.#fixed_size) {
         svg.attr('x', 0)
            .attr('y', 0)
            .attr('width', rect.width)
            .attr('height', rect.height)
            .style('position', 'absolute');
      } else {
        svg.attr('x', 0)
           .attr('y', 0)
           .style('width', '100%')
           .style('height', '100%')
           .style('position', 'absolute')
           .style('left', 0).style('top', 0).style('bottom', 0).style('right', 0);
      }

      svg.style('filter', settings.DarkMode ? 'invert(100%)' : null);

      svg.attr('viewBox', `0 0 ${rect.width} ${rect.height}`)
         .attr('preserveAspectRatio', 'none')  // we do not preserve relative ratio
         .property('height_factor', factor)
         .property('draw_x', 0)
         .property('draw_y', 0)
         .property('draw_width', rect.width)
         .property('draw_height', rect.height);

      this.#pad_x = 0;
      this.#pad_y = 0;
      this.#pad_width = rect.width;
      this.#pad_height = rect.height;

      frect.attr('d', `M0,0H${rect.width}V${rect.height}H0Z`)
           .call(this.fillatt.func);

      this.setFastDrawing(rect.width, rect.height);

      if (isFunc(this.alignButtons) && btns)
         this.alignButtons(btns, rect.width, rect.height);

      return true;
   }

   /** @summary Draw item name on canvas, dummy for RPad
     * @private */
   drawItemNameOnCanvas() {
   }

   /** @summary Enlarge pad draw element when possible */
   enlargePad(evnt, is_dblclick, is_escape) {
      evnt?.preventDefault();
      evnt?.stopPropagation();

      // ignore double click on online canvas itself for enlarge
      if (is_dblclick && this.isCanvas(true) && (this.enlargeMain('state') === 'off'))
         return;

      const svg_can = this.getCanvSvg(),
            pad_enlarged = svg_can.property('pad_enlarged');

      if (this.isTopPad() || (!pad_enlarged && !this.hasObjectsToDraw() && !this.#painters)) {
         if (this.#fixed_size) return; // canvas cannot be enlarged in such mode
         if (!this.enlargeMain(is_escape ? false : 'toggle')) return;
         if (this.enlargeMain('state') === 'off')
            svg_can.property('pad_enlarged', null);
         else
            selectActivePad({ pp: this, active: true });
      } else if (!pad_enlarged && !is_escape) {
         this.enlargeMain(true, true);
         svg_can.property('pad_enlarged', this.#pad);
         selectActivePad({ pp: this, active: true });
      } else if (pad_enlarged === this.#pad) {
         this.enlargeMain(false);
         svg_can.property('pad_enlarged', null);
      } else if (!is_escape && is_dblclick)
         console.error('missmatch with pad double click events');

      return this.checkResize(true);
   }

   /** @summary Create SVG element for the pad
     * @return true when pad is displayed and all its items should be redrawn */
   createPadSvg(only_resize) {
      if (this.isTopPad()) {
         this.createCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      const svg_parent = this.getPadPainter()?.getPadSvg(),
            svg_can = this.getCanvSvg(),
            width = svg_parent.property('draw_width'),
            height = svg_parent.property('draw_height'),
            pad_enlarged = svg_can.property('pad_enlarged');
      let pad_visible = true,
          w = width, h = height, x = 0, y = 0,
          svg_pad, svg_rect, btns = null;

      if (this.#pad?.fPos && this.#pad?.fSize) {
         x = Math.round(width * this.#pad.fPos.fHoriz.fArr[0]);
         y = Math.round(height * this.#pad.fPos.fVert.fArr[0]);
         w = Math.round(width * this.#pad.fSize.fHoriz.fArr[0]);
         h = Math.round(height * this.#pad.fSize.fVert.fArr[0]);
      }

      if (pad_enlarged) {
         pad_visible = false;
         if (pad_enlarged === this.#pad)
            pad_visible = true;
         else
            this.forEachPainterInPad(pp => { if (pp.getObject() === pad_enlarged) pad_visible = true; }, 'pads');

         if (pad_visible) { w = width; h = height; x = y = 0; }
      }

      if (only_resize) {
         svg_pad = this.getPadSvg();
         svg_rect = svg_pad.selectChild('.root_pad_border');
         if (!this.isBatchMode())
            btns = this.getLayerSvg('btns_layer');
         this.addPadInteractive(true);
      } else {
         svg_pad = svg_parent.selectChild('.primitives_layer')
             .append('svg:svg') // here was g before, svg used to blend all drawings outside
             .classed('__root_pad_' + this.#pad_name, true)
             .attr('pad', this.#pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this); // this is custom property

         if (!this.isBatchMode())
            svg_pad.append('svg:title').text('ROOT subpad');

         svg_rect = svg_pad.append('svg:path').attr('class', 'root_pad_border');

         svg_pad.append('svg:g').attr('class', 'primitives_layer');
         if (!this.isBatchMode()) {
            btns = svg_pad.append('svg:g')
                          .attr('class', 'btns_layer')
                          .property('leftside', settings.ToolBarSide !== 'left')
                          .property('vertical', settings.ToolBarVert);
         }

         if (settings.ContextMenu)
            svg_rect.on('contextmenu', evnt => this.padContextMenu(evnt));

         if (!this.isBatchMode()) {
            svg_rect.style('pointer-events', 'visibleFill') // get events also for not visible rect
                    .on('dblclick', evnt => this.enlargePad(evnt, true))
                    .on('click', () => this.selectObjectPainter(this, null))
                    .on('mouseenter', () => this.showObjectStatus());
         }
      }

      this.createAttFill({ attr: this.#pad });

      this.createAttLine({ attr: this.#pad, color0: this.#pad.fBorderMode === 0 ? 'none' : '' });

      svg_pad.style('display', pad_visible ? null : 'none')
             .attr('viewBox', `0 0 ${w} ${h}`) // due to svg
             .attr('preserveAspectRatio', 'none')   // due to svg, we do not preserve relative ratio
             .attr('x', x)    // due to svg
             .attr('y', y)   // due to svg
             .attr('width', w)    // due to svg
             .attr('height', h)   // due to svg
             .property('draw_x', x) // this is to make similar with canvas
             .property('draw_y', y)
             .property('draw_width', w)
             .property('draw_height', h);

      this.#pad_x = x;
      this.#pad_y = y;
      this.#pad_width = w;
      this.#pad_height = h;

      svg_rect.attr('d', `M0,0H${w}V${h}H0Z`)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      this.setFastDrawing(w, h);

      // special case of 3D canvas overlay
      if (svg_pad.property('can3d') === constants.Embed3D.Overlay) {
         this.selectDom().select('.draw3d_' + this.#pad_name)
             .style('display', pad_visible ? '' : 'none');
      }

      if (this.alignButtons && btns) this.alignButtons(btns, w, h);

      return pad_visible;
   }

   /** @summary Add pad interactive features like dragging and resize
    * @private */
   addPadInteractive(/* cleanup = false */) {
      if (isFunc(this.$userInteractive)) {
         this.$userInteractive();
         delete this.$userInteractive;
      }
      // if (this.isBatchMode())
      //   return;
   }

   /** @summary returns true if any objects beside sub-pads exists in the pad */
   hasObjectsToDraw() {
      return this.#pad?.fPrimitives?.find(obj => obj._typename !== `${nsREX}RPadDisplayItem`);
   }

   /** @summary sync drawing/redrawing/resize of the pad
     * @param {string} kind - kind of draw operation, if true - always queued
     * @return {Promise} when pad is ready for draw operation or false if operation already queued
     * @private */
   syncDraw(kind) {
      const entry = { kind: kind || 'redraw' };
      if (this.#doing_draw === undefined) {
         this.#doing_draw = [entry];
         return Promise.resolve(true);
      }
      // if queued operation registered, ignore next calls, indx === 0 is running operation
      if ((entry.kind !== true) && (this.#doing_draw.findIndex((e, i) => (i > 0) && (e.kind === entry.kind)) > 0))
         return false;
      this.#doing_draw.push(entry);
      return new Promise(resolveFunc => {
         entry.func = resolveFunc;
      });
   }

   /** @summary confirms that drawing is completed, may trigger next drawing immediately
     * @private */
   confirmDraw() {
      if (this.#doing_draw === undefined)
         return console.warn('failure, should not happen');
      this.#doing_draw.shift();
      if (!this.#doing_draw.length)
         this.#doing_draw = undefined;
      else {
         const entry = this.#doing_draw[0];
         if (entry.func) { entry.func(); delete entry.func; }
      }
   }

   /** @summary Draw single primitive */
   async drawObject(/* dom, obj, opt */) {
      console.log('Not possible to draw object without loading of draw.mjs');
      return null;
   }

   /** @summary Draw pad primitives
     * @private */
   async drawPrimitives(indx) {
      if (indx === undefined) {
         if (this.isCanvas())
            this.#start_draw_tm = new Date().getTime();

         // set number of primitives
         this.#num_primitives = this.#pad?.fPrimitives?.length ?? 0;

         return this.syncDraw(true).then(() => this.drawPrimitives(0));
      }

      if (!this.#pad || (indx >= this.#num_primitives)) {
         this.confirmDraw();

         if (this.#start_draw_tm) {
            const spenttm = new Date().getTime() - this.#start_draw_tm;
            if (spenttm > 3000) console.log(`Canvas drawing took ${(spenttm*1e-3).toFixed(2)}s`);
            this.#start_draw_tm = undefined;
         }

         return;
      }

      // handle used to invoke callback only when necessary
      return this.drawObject(this, this.#pad.fPrimitives[indx], '').then(op => {
         // mark painter as belonging to primitives
         if (isObject(op))
            op._primitive = true;

         return this.drawPrimitives(indx+1);
      });
   }

   /** @summary Provide autocolor
     * @private */
   getAutoColor() {
      const pal = this.getHistPalette(),
            cnt = this.#auto_color_cnt++,
            num = Math.max(this.#num_primitives - 1, 2);
      return pal?.getColorOrdinal((cnt % num) / num) ?? 'blue';
   }

   /** @summary Process tooltip event in the pad
     * @private */
   processPadTooltipEvent(pnt) {
      const painters = [], hints = [];

      // first count - how many processors are there
      this.#painters?.forEach(obj => {
         if (isFunc(obj.processTooltipEvent)) painters.push(obj);
      });

      if (pnt) pnt.nproc = painters.length;

      painters.forEach(obj => {
         const hint = obj.processTooltipEvent(pnt) || { user_info: null };
         hints.push(hint);
         if (pnt?.painters)
            hint.painter = obj;
      });

      return hints;
   }

   /** @summary Changes canvas dark mode
     * @private */
   changeDarkMode(mode) {
      this.getCanvSvg().style('filter', (mode ?? settings.DarkMode) ? 'invert(100%)' : null);
   }

   /** @summary Fill pad context menu
     * @private */
   fillContextMenu(menu) {
      const clname = this.isCanvas() ? 'RCanvas' : 'RPad';

      menu.header(clname, `${urlClassPrefix}ROOT_1_1Experimental_1_1${clname}.html`);

      menu.addchk(this.isTooltipAllowed(), 'Show tooltips', () => this.setTooltipAllowed('toggle'));

      if (!this.isCanvas(true)) {
         // if not online canvas
         menu.addAttributesMenu(this);
         if (this.isCanvas()) {
            menu.addSettingsMenu(false, false, arg => {
               if (arg === 'dark') this.changeDarkMode();
            });
         }
      }

      menu.separator();

      if (isFunc(this.hasMenuBar) && isFunc(this.actiavteMenuBar))
         menu.addchk(this.hasMenuBar(), 'Menu bar', flag => this.actiavteMenuBar(flag));

      if (isFunc(this.hasEventStatus) && isFunc(this.activateStatusBar) && isFunc(this.canStatusBar)) {
         if (this.canStatusBar())
            menu.addchk(this.hasEventStatus(), 'Event status', () => this.activateStatusBar('toggle'));
      }

      if (this.enlargeMain() || (!this.isTopPad() && this.hasObjectsToDraw()))
         menu.addchk((this.enlargeMain('state') === 'on'), 'Enlarge ' + (this.isCanvas() ? 'canvas' : 'pad'), () => this.enlargePad());

      const fname = this.#pad_name || (this.isCanvas() ? 'canvas' : 'pad');
      menu.sub('Save as');
      ['svg', 'png', 'jpeg', 'pdf', 'webp'].forEach(fmt => menu.add(`${fname}.${fmt}`, () => this.saveAs(fmt, this.isCanvas(), `${fname}.${fmt}`)));
      menu.endsub();

      return true;
   }

   /** @summary Show pad context menu
     * @private */
   padContextMenu(evnt) {
      if (evnt.stopPropagation) {
         // this is normal event processing and not emulated jsroot event

         evnt.stopPropagation(); // disable main context menu
         evnt.preventDefault();  // disable browser context menu

         this.getFramePainter()?.setLastEventPos();
      }

      createMenu(evnt, this).then(menu => {
         this.fillContextMenu(menu);
         return this.fillObjectExecMenu(menu);
      }).then(menu => menu.show());
   }

   /** @summary Redraw legend object
    * @desc Used when object attributes are changed to ensure that legend is up to date
    * @private */
   async redrawLegend() {}

   /** @summary Deliver mouse move or click event to the web canvas
     * @private */
   deliverWebCanvasEvent() {}

   /** @summary Redraw pad means redraw ourself
     * @return {Promise} when redrawing ready */
   async redrawPad(reason) {
      const sync_promise = this.syncDraw(reason);
      if (sync_promise === false) {
         console.log('Prevent RPad redrawing');
         return false;
      }

      let showsubitems = true;
      const redrawNext = indx => {
         while (indx < this.#painters.length) {
            const sub = this.#painters[indx++];
            let res = 0;
            if (showsubitems || isPadPainter(sub))
               res = sub.redraw(reason);

            if (isPromise(res))
               return res.then(() => redrawNext(indx));
         }
         return true;
      };

      return sync_promise.then(() => {
         if (this.isCanvas())
            this.createCanvasSvg(2);
          else
            showsubitems = this.createPadSvg(true);

         return redrawNext(0);
      }).then(() => {
         this.addPadInteractive();
         if (getActivePad() === this)
            this.getCanvPainter()?.producePadEvent('padredraw', this);
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
      const elem = this.getPadSvg();
      if (!elem.empty() && elem.property('can3d') === constants.Embed3D.Overlay) return true;

      for (let i = 0; i < this.#painters.length; ++i) {
         if (isFunc(this.#painters[i].needRedrawByResize))
            if (this.#painters[i].needRedrawByResize()) return true;
      }

      return false;
   }

   /** @summary Check resize of canvas */
   checkCanvasResize(size, force) {
      if (this._ignore_resize || !this.isTopPad())
         return false;

      const sync_promise = this.syncDraw('canvas_resize');
      if (sync_promise === false)
         return false;

      if ((size === true) || (size === false)) { force = size; size = null; }

      if (isObject(size) && size.force) force = true;

      if (!force) force = this.needRedrawByResize();

      let changed = false;
      const redrawNext = indx => {
         if (!changed || (indx >= this.#painters.length)) {
            this.confirmDraw();
            return changed;
         }

         return getPromise(this.#painters[indx].redraw(force ? 'redraw' : 'resize')).then(() => redrawNext(indx+1));
      };


      return sync_promise.then(() => {
         changed = this.createCanvasSvg(force ? 2 : 1, size);

         if (changed && this.isCanvas() && this.#pad && this.online_canvas && !this.embed_canvas && !this.isBatchMode()) {
            if (this.#resize_tmout)
               clearTimeout(this.#resize_tmout);
            this.#resize_tmout = setTimeout(() => {
               this.#resize_tmout = undefined;
               if (!this.#pad?.fWinSize)
                  return;
               const cw = this.getPadWidth(), ch = this.getPadHeight();
               if ((cw > 0) && (ch > 0) && ((this.#pad.fWinSize[0] !== cw) || (this.#pad.fWinSize[1] !== ch))) {
                  this.#pad.fWinSize[0] = cw;
                  this.#pad.fWinSize[1] = ch;
                  this.sendWebsocket(`RESIZED:[${cw},${ch}]`);
               }
            }, 1000); // long enough delay to prevent multiple occurrence
         }

         // if canvas changed, redraw all its subitems.
         // If redrawing was forced for canvas, same applied for sub-elements
         return redrawNext(0);
      });
   }

   /** @summary update RPad object
     * @private */
   updateObject(obj) {
      if (!obj)
         return false;

      this.#pad.fStyle = obj.fStyle;
      this.#pad.fAttr = obj.fAttr;

      if (this.isCanvas()) {
         this.#pad.fTitle = obj.fTitle;
         this.#pad.fWinSize = obj.fWinSize;
      } else {
         this.#pad.fPos = obj.fPos;
         this.#pad.fSize = obj.fSize;
      }

      return true;
   }

   /** @summary Add object painter to list of primitives
     * @private */
   addObjectPainter(objpainter, lst, indx) {
      if (objpainter && lst && lst[indx] && !objpainter.hasSnapId()) {
         // keep snap id in painter, will be used for the
         if (this.#painters.indexOf(objpainter) < 0)
            this.#painters.push(objpainter);
         objpainter.assignSnapId(lst[indx].fObjectID);
         if (!objpainter.rstyle)
            objpainter.rstyle = lst[indx].fStyle || this.rstyle;
      }
   }

   /** @summary Extract properties from TObjectDisplayItem */
   extractTObjectProp(snap) {
      if (snap.fColIndex && snap.fColValue) {
         const colors = this.getColors() || getRootColors();
         for (let k = 0; k < snap.fColIndex.length; ++k)
            colors[snap.fColIndex[k]] = convertColor(snap.fColValue[k]);
       }

      // painter used only for evaluation of attributes
      const pattr = new RObjectPainter(), obj = snap.fObject;
      pattr.assignObject(snap);
      pattr.csstype = snap.fCssType;
      pattr.rstyle = snap.fStyle;

      snap.fOption = pattr.v7EvalAttr('options', '');

      const extract_color = (member_name, attr_name) => {
         const col = pattr.v7EvalColor(attr_name, '');
         if (col) obj[member_name] = addColor(col, this.getColors());
      };

      // handle TAttLine
      if ((obj.fLineColor !== undefined) && (obj.fLineWidth !== undefined) && (obj.fLineStyle !== undefined)) {
         extract_color('fLineColor', 'line_color');
         obj.fLineWidth = pattr.v7EvalAttr('line_width', obj.fLineWidth);
         obj.fLineStyle = pattr.v7EvalAttr('line_style', obj.fLineStyle);
      }

      // handle TAttFill
      if ((obj.fFillColor !== undefined) && (obj.fFillStyle !== undefined)) {
         extract_color('fFillColor', 'fill_color');
         obj.fFillStyle = pattr.v7EvalAttr('fill_style', obj.fFillStyle);
      }

      // handle TAttMarker
      if ((obj.fMarkerColor !== undefined) && (obj.fMarkerStyle !== undefined) && (obj.fMarkerSize !== undefined)) {
         extract_color('fMarkerColor', 'marker_color');
         obj.fMarkerStyle = pattr.v7EvalAttr('marker_style', obj.fMarkerStyle);
         obj.fMarkerSize = pattr.v7EvalAttr('marker_size', obj.fMarkerSize);
      }

      // handle TAttText
      if ((obj.fTextColor !== undefined) && (obj.fTextAlign !== undefined) && (obj.fTextAngle !== undefined) && (obj.fTextSize !== undefined)) {
         extract_color('fTextColor', 'text_color');
         obj.fTextAlign = pattr.v7EvalAttr('text_align', obj.fTextAlign);
         obj.fTextAngle = pattr.v7EvalAttr('text_angle', obj.fTextAngle);
         obj.fTextSize = pattr.v7EvalAttr('text_size', obj.fTextSize);
         // TODO: v7 font handling differs much from v6, ignore for the moment
      }
   }

   /** @summary Function called when drawing next snapshot from the list
     * @return {Promise} with pad painter when ready
     * @private */
   async drawNextSnap(lst, pindx, indx) {
      if (indx === undefined) {
         indx = -1;
         // flag used to prevent immediate pad redraw during first draw
         this.#num_primitives = lst?.length ?? 0;
         this.#auto_color_cnt = 0;
      }

      delete this.next_rstyle;

      ++indx; // change to the next snap

      if (!lst || indx >= lst.length) {
         this.#auto_color_cnt = undefined;
         return this;
      }

      const snap = lst[indx], is_subpad = snap._typename === `${nsREX}RPadDisplayItem`;

      // empty object, no need to do something, take next
      if (snap.fDummy)
         return this.drawNextSnap(lst, pindx + 1, indx);

      if (snap._typename === `${nsREX}TObjectDisplayItem`) {
         // identifier used in TObjectDrawable

         if (snap.fKind === webSnapIds.kStyle) {
            Object.assign(gStyle, snap.fObject);
            return this.drawNextSnap(lst, pindx, indx);
         }

         if (snap.fKind === webSnapIds.kColors) {
            const colors = [], arr = snap.fObject.arr;
            for (let n = 0; n < arr.length; ++n) {
               const name = arr[n].fString, p = name.indexOf('=');
               if (p > 0)
                  colors[parseInt(name.slice(0, p))] = convertColor(name.slice(p+1));
            }

            this.setColors(colors);
            // set global list of colors
            // adoptRootColors(ListOfColors);
            return this.drawNextSnap(lst, pindx, indx);
         }

         if (snap.fKind === webSnapIds.kPalette) {
            const arr = snap.fObject.arr, palette = [];
            for (let n = 0; n < arr.length; ++n)
               palette[n] = arr[n].fString;
            this.#custom_palette = new ColorPalette(palette);
            return this.drawNextSnap(lst, pindx, indx);
         }

         if (snap.fKind === webSnapIds.kFont)
            return this.drawNextSnap(lst, pindx, indx);

         if (!this.getFramePainter()) {
            // draw dummy frame which is not provided by RCanvas
            return this.drawObject(this, { _typename: clTFrame, $dummy: true }, '')
                       .then(() => this.drawNextSnap(lst, pindx, indx - 1));
         }

         this.extractTObjectProp(snap);
      }

      // try to locate existing object painter, only allowed when redrawing pad snap
      let objpainter, promise;

      while ((pindx !== undefined) && (pindx < this.#painters.length)) {
         const subp = this.#painters[pindx++];

         if (subp.getSnapId() === snap.fObjectID) {
            objpainter = subp;
            break;
         } else if (subp.getSnapId() && !subp.isSecondary() && !is_subpad) {
            console.warn(`Mismatch in snapid between painter ${subp.getSnapId()} secondary: ${subp.isSecondary()} type: ${subp.getClassName()} and primitive ${snap.fObjectID} kind ${snap.fKind} type ${snap.fDrawable?._typename}`);
            break;
         }
      }

      if (objpainter) {
         if (is_subpad)
            promise = objpainter.redrawPadSnap(snap);
         else if (objpainter.updateObject(snap.fDrawable || snap.fObject || snap, snap.fOption || '', true))
            promise = objpainter.redraw();
      } else if (is_subpad) {
         const padpainter = new RPadPainter(this, snap, '', false, 'webpad');
         padpainter.assignSnapId(snap.fObjectID);
         padpainter.rstyle = snap.fStyle;

         padpainter.createPadSvg();

         if (snap.fPrimitives?.length)
            padpainter.addPadButtons();

         pindx++; // new painter will be add
         promise = padpainter.drawNextSnap(snap.fPrimitives).then(() => padpainter.addPadInteractive());
      } else {
         // will be used in addToPadPrimitives to assign style to sub-painters
         this.next_rstyle = snap.fStyle || this.rstyle;
         pindx++; // new painter will be add

         // TODO - fDrawable is v7, fObject from v6, maybe use same data member?
         promise = this.drawObject(this, snap.fDrawable || snap.fObject || snap, snap.fOption || '')
                       .then(objp => this.addObjectPainter(objp, lst, indx));
      };

      return getPromise(promise).then(() => this.drawNextSnap(lst, pindx, indx)); // call next
   }

   /** @summary Search painter with specified snapid, also sub-pads are checked
     * @private */
   findSnap(snapid, onlyid) {
      function check(checkid) {
         if (!checkid || !isStr(checkid)) return false;
         if (checkid === snapid) return true;
         return onlyid && (checkid.length > snapid.length) &&
                (checkid.indexOf(snapid) === (checkid.length - snapid.length));
      }

      if (check(this.getSnapId()))
         return this;

      if (!this.#painters)
         return null;

      for (let k=0; k < this.#painters.length; ++k) {
         let sub = this.#painters[k];

         if (!onlyid && isFunc(sub.findSnap))
            sub = sub.findSnap(snapid);
         else if (!check(sub.getSnapId()))
            sub = null;

         if (sub) return sub;
      }

      return null;
   }

   /** @summary Redraw pad snap
     * @desc Online version of drawing pad primitives
     * @return {Promise} with pad painter */
   async redrawPadSnap(snap) {
      // for the pad/canvas display item contains list of primitives plus pad attributes

      if (!snap || !snap.fPrimitives)
         return this;

      if (this.isCanvas(true) && snap.fTitle && !this.embed_canvas && (typeof document !== 'undefined'))
         document.title = snap.fTitle;

      if (!this.hasSnapId()) {
         // first time getting snap, create all gui elements first

         this.assignSnapId(snap.fObjectID);

         this.assignObject(snap);
         this.#pad = snap;

         if (this.isBatchMode() && this.isCanvas())
             this.#fixed_size = true;

         const mainid = this.selectDom().attr('id');

         if (!this.isBatchMode() && this.online_canvas && !this.use_openui && !this.brlayout && mainid && isStr(mainid) && !getHPainter()) {
            this.brlayout = new BrowserLayout(mainid, null, this);
            this.brlayout.create(mainid, true);
            this.setDom(this.brlayout.drawing_divid()); // need to create canvas
            registerForResize(this.brlayout);
         }

         this.createCanvasSvg(0);
         this.addPadButtons(true);

         return this.drawNextSnap(snap.fPrimitives).then(() => {
            if (isFunc(this.onCanvasUpdated))
               this.onCanvasUpdated(this);
            return this;
         });
      }

      // update only pad/canvas attributes
      this.updateObject(snap);

      // apply all changes in the object (pad or canvas)
      if (this.isCanvas())
         this.createCanvasSvg(2);
       else
         this.createPadSvg(true);

      let missmatch = false, i = 0, k = 0;

      // match painters with new list of primitives
      while (k < this.#painters.length) {
         const sub = this.#painters[k];

         // skip check secondary painters or painters without snapid
         // also frame painter will be excluded here
         if (!sub.hasSnapId() || sub.isSecondary()) {
            k++;
            continue; // look only for painters with snapid
         }

         if (i >= snap.fPrimitives.length)
            break;


         const prim = snap.fPrimitives[i];

         if (prim.fObjectID === sub.getSnapId()) {
            i++;
            k++;
         } else if (prim.fDummy || !prim.fObjectID || ((prim._typename === `${nsREX}TObjectDisplayItem`) && ((prim.fKind === webSnapIds.kStyle) || (prim.fKind === webSnapIds.kColors) || (prim.fKind === webSnapIds.kPalette) || (prim.fKind === webSnapIds.kFont)))) {
            // ignore primitives without snapid or which are not produce drawings
            i++;
         } else {
            missmatch = true;
            break;
         }
      }

      let cnt = 1000;
      // remove painters without primitives, limit number of checks
      while (!missmatch && (k < this.#painters.length) && (--cnt >= 0)) {
         if (this.removePrimitive(k) === -111)
            missmatch = true;
      }
      if (cnt < 0)
         missmatch = true;

      if (missmatch) {
         const old_painters = this.#painters;
         this.#painters = [];
         old_painters.forEach(objp => objp.cleanup());
         this.setMainPainter(undefined, true);
         if (isFunc(this.removePadButtons))
            this.removePadButtons();
         this.addPadButtons(true);
      }

      return this.drawNextSnap(snap.fPrimitives, missmatch ? undefined : 0).then(() => {
         this.addPadInteractive();
         if (getActivePad() === this)
            this.getCanvPainter()?.producePadEvent('padredraw', this);
         if (isFunc(this.onCanvasUpdated))
            this.onCanvasUpdated(this);
         return this;
      });
   }

   /** @summary Create image for the pad
     * @desc Used with web-based canvas to create images for server side
     * @return {Promise} with image data, coded with btoa() function
     * @private */
   async createImage(format) {
      if ((format === 'png') || (format === 'jpeg') || (format === 'svg') || (format === 'webp') || (format === 'pdf')) {
         return this.produceImage(true, format).then(res => {
            if (!res || (format === 'svg')) return res;
            const separ = res.indexOf('base64,');
            return (separ > 0) ? res.slice(separ+7) : '';
         });
      }

      return '';
   }

   /** @summary Show context menu for specified item
     * @private */
   itemContextMenu(name) {
      const rrr = this.getPadSvg().node().getBoundingClientRect(),
            evnt = { clientX: rrr.left+10, clientY: rrr.top + 10 };

      // use timeout to avoid conflict with mouse click and automatic menu close
      if (name === 'pad')
         return postponePromise(() => this.padContextMenu(evnt), 50);

      let selp = null, selkind;

      switch (name) {
         case 'xaxis':
         case 'yaxis':
         case 'zaxis':
            selp = this.getMainPainter();
            selkind = name[0];
            break;
         case 'frame':
            selp = this.getFramePainter();
            break;
         default: {
            const indx = parseInt(name);
            if (Number.isInteger(indx))
               selp = this.#painters[indx];
         }
      }

      if (!isFunc(selp?.fillContextMenu))
         return;

      return createMenu(evnt, selp).then(menu => {
         const offline_menu = selp.fillContextMenu(menu, selkind);
         if (offline_menu || selp.getSnapId())
            selp.fillObjectExecMenu(menu, selkind).then(() => postponePromise(() => menu.show(), 50));
      });
   }

   /** @summary Save pad in specified format
     * @desc Used from context menu */
   saveAs(kind, full_canvas, filename) {
      if (!filename)
         filename = (this.#pad_name || (this.isCanvas() ? 'canvas' : 'pad')) + '.' + kind;

      this.produceImage(full_canvas, kind).then(imgdata => {
         if (!imgdata)
            return console.error(`Fail to produce image ${filename}`);

         if ((browser.qt6 || browser.cef3) && this.getSnapId()) {
            console.warn(`sending file ${filename} to server`);
            let res = imgdata;
            if (kind !== 'svg') {
               const separ = res.indexOf('base64,');
               res = (separ > 0) ? res.slice(separ+7) : '';
            }
            if (res)
              this.getCanvPainter()?.sendWebsocket(`SAVE:${filename}:${res}`);
         } else
            saveFile(filename, (kind !== 'svg') ? imgdata : prSVG + encodeURIComponent(imgdata));
      });
   }

   /** @summary Search active pad
     * @return {Object} pad painter for active pad */
   findActivePad() { return null; }

   /** @summary Produce image for the pad
     * @return {Promise} with created image */
   async produceImage(full_canvas, file_format, args) {
      const use_frame = (full_canvas === 'frame'),
            elem = use_frame ? this.getFrameSvg() : (full_canvas ? this.getCanvSvg() : this.getPadSvg()),
            painter = (full_canvas && !use_frame) ? this.getCanvPainter() : this,
            items = []; // keep list of replaced elements, which should be moved back at the end

      if (elem.empty())
         return '';

      if (use_frame || !full_canvas) {
         const defs = this.getCanvSvg().selectChild('.canvas_defs');
         if (!defs.empty()) {
            items.push({ prnt: this.getCanvSvg(), defs });
            elem.node().insertBefore(defs.node(), elem.node().firstChild);
         }
      }

      if (!use_frame) {
         // do not make transformations for the frame
         painter.forEachPainterInPad(pp => {
            const item = { prnt: pp.getPadSvg() };
            items.push(item);

            // remove buttons from each sub-pad
            const btns = pp.getLayerSvg('btns_layer');
            item.btns_node = btns.node();
            if (item.btns_node) {
               item.btns_prnt = item.btns_node.parentNode;
               item.btns_next = item.btns_node.nextSibling;
               btns.remove();
            }

            const fp = pp.getFramePainter();
            if (!isFunc(fp?.access3dKind))
               return;

            const can3d = fp.access3dKind();
            if ((can3d !== constants.Embed3D.Overlay) && (can3d !== constants.Embed3D.Embed))
               return;

            const main = isFunc(fp.getRenderer) ? fp : fp.getMainPainter(),
                  canvas = isFunc(main.getRenderer) ? main.getRenderer()?.domElement : null;
            if (!isFunc(main?.render3D) || !isObject(canvas))
               return;

            const sz2 = fp.getSizeFor3d(constants.Embed3D.Embed); // get size and position of DOM element as it will be embed
            main.render3D(0); // WebGL clears buffers, therefore we should render scene and convert immediately
            const dataUrl = canvas.toDataURL('image/png');

            // remove 3D drawings
            if (can3d === constants.Embed3D.Embed) {
               item.foreign = item.prnt.select('.' + sz2.clname);
               item.foreign.remove();
            }

            const svg_frame = fp.getFrameSvg();
            item.frame_node = svg_frame.node();
            if (item.frame_node) {
               item.frame_next = item.frame_node.nextSibling;
               svg_frame.remove();
            }

            // add svg image
            item.img = item.prnt.insert('image', '.primitives_layer')     // create image object
                           .attr('x', sz2.x)
                           .attr('y', sz2.y)
                           .attr('width', canvas.width)
                           .attr('height', canvas.height)
                           .attr('href', dataUrl);
         }, 'pads');
      }

      let width = elem.property('draw_width'), height = elem.property('draw_height');
      if (use_frame) {
         const fp = this.getFramePainter();
         width = fp.getFrameWidth();
         height = fp.getFrameHeight();
      }

      const arg = (file_format === 'pdf')
         ? { node: elem.node(), width, height, reset_tranform: use_frame }
         : compressSVG(`<svg width="${width}" height="${height}" xmlns="${nsSVG}">${elem.node().innerHTML}</svg>`);

      return svgToImage(arg, file_format, args).then(res => {
         for (let k = 0; k < items.length; ++k) {
            const item = items[k];

            item.img?.remove(); // delete embed image

            const prim = item.prnt.selectChild('.primitives_layer');

            if (item.foreign) // reinsert foreign object
               item.prnt.node().insertBefore(item.foreign.node(), prim.node());

            if (item.frame_node) // reinsert frame as first in list of primitives
               prim.node().insertBefore(item.frame_node, item.frame_next);

            if (item.btns_node) // reinsert buttons
               item.btns_prnt.insertBefore(item.btns_node, item.btns_next);

            if (item.defs) // reinsert defs
               item.prnt.node().insertBefore(item.defs.node(), item.prnt.node().firstChild);
         }
         return res;
      });
   }

   /** @summary Process pad button click */
   clickPadButton(funcname, evnt) {
      if (funcname === 'CanvasSnapShot')
         return this.saveAs('png', true);

      if (funcname === 'enlargePad')
         return this.enlargePad();

      if (funcname === 'PadSnapShot')
         return this.saveAs('png', false);

      if (funcname === 'PadContextMenus') {
         evnt?.preventDefault();
         evnt?.stopPropagation();
         if (closeMenu()) return;

         return createMenu(evnt, this).then(menu => {
            menu.header('Menus');

            menu.add(this.isCanvas() ? 'Canvas' : 'Pad', 'pad', this.itemContextMenu);

            if (this.getFramePainter())
               menu.add('Frame', 'frame', this.itemContextMenu);

            const main = this.getMainPainter(); // here hist painter methods

            if (main) {
               menu.add('X axis', 'xaxis', this.itemContextMenu);
               menu.add('Y axis', 'yaxis', this.itemContextMenu);
               if (isFunc(main.getDimension) && (main.getDimension() > 1))
                  menu.add('Z axis', 'zaxis', this.itemContextMenu);
            }

            if (this.#painters?.length) {
               menu.separator();
               const shown = [];
               this.#painters.forEach((pp, indx) => {
                  const obj = pp?.getObject();
                  if (!obj || (shown.indexOf(obj) >= 0) || pp.isSecondary()) return;
                  let name = isFunc(pp.getClassName) ? pp.getClassName() : (obj._typename || '');
                  if (name) name += '::';
                  name += isFunc(pp.getObjectName) ? pp.getObjectName() : (obj.fName || `item${indx}`);
                  menu.add(name, indx, this.itemContextMenu);
                  shown.push(obj);
               });
            }

            menu.show();
         });
      }

      // click automatically goes to all sub-pads
      // if any painter indicates that processing completed, it returns true
      let done = false;
      const prs = [];

      for (let i = 0; i < this.#painters.length; ++i) {
         const pp = this.#painters[i];

         if (isFunc(pp.clickPadButton))
            prs.push(pp.clickPadButton(funcname, evnt));

         if (!done && isFunc(pp.clickButton)) {
            done = pp.clickButton(funcname);
            if (isPromise(done)) prs.push(done);
         }
      }

      return Promise.all(prs);
   }

   /** @summary Add button to the pad
     * @private */
   addPadButton(btn, tooltip, funcname, keyname) {
      if (!settings.ToolBar || this.isBatchMode())
         return;

      if (!this._buttons)
         this._buttons = [];
      // check if there are duplications

      for (let k = 0; k < this._buttons.length; ++k)
         if (this._buttons[k].funcname === funcname) return;

      this._buttons.push({ btn, tooltip, funcname, keyname });

      if (!this.isTopPad() && funcname.indexOf('Pad') && (funcname !== 'enlargePad')) {
         const cp = this.getCanvPainter();
         if (cp && (cp !== this)) cp.addPadButton(btn, tooltip, funcname);
      }
   }

   /** @summary Add buttons for pad or canvas
     * @private */
   addPadButtons(is_online) {
      this.addPadButton('camera', 'Create PNG', this.isCanvas() ? 'CanvasSnapShot' : 'PadSnapShot', 'Ctrl PrintScreen');

      if (settings.ContextMenu)
         this.addPadButton('question', 'Access context menus', 'PadContextMenus');

      const add_enlarge = !this.isTopPad() && this.hasObjectsToDraw();

      if (add_enlarge || this.enlargeMain('verify'))
         this.addPadButton('circle', 'Enlarge canvas', 'enlargePad');

      if (is_online && this.brlayout) {
         this.addPadButton('diamand', 'Toggle Ged', 'ToggleGed');
         this.addPadButton('three_circles', 'Toggle Status', 'ToggleStatus');
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
      let rect, res;
      const sign = vertical ? -1 : 1,
            getV = (indx, dflt) => (indx < len.fArr.length) ? len.fArr[indx] : dflt,
            getRect = () => {
               if (!rect)
                  rect = frame_painter ? frame_painter.getFrameRect() : this.getPadRect();
               return rect;
            };

      if (frame_painter) {
         const user = getV(2), func = vertical ? 'gry' : 'grx';
         if ((user !== undefined) && frame_painter[func])
            res = frame_painter[func](user);
      }

      if (res === undefined)
         res = vertical ? getRect().height : 0;

      const norm = getV(0, 0), pixel = getV(1, 0);

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
      };
   }

   /** @summary Decode pad draw options */
   decodeOptions(opt) {
      const pad = this.getObject();
      if (!pad) return;

      const d = new DrawOptions(opt),
            o = this.setOptions({ GlobalColors: true, LocalColors: false, IgnorePalette: false, RotateFrame: false, FixFrame: false });

      if (d.check('NOCOLORS') || d.check('NOCOL')) o.GlobalColors = o.LocalColors = false;
      if (d.check('LCOLORS') || d.check('LCOL')) { o.GlobalColors = false; o.LocalColors = true; }
      if (d.check('NOPALETTE') || d.check('NOPAL')) o.IgnorePalette = true;
      if (d.check('ROTATE')) o.RotateFrame = true;
      if (d.check('FIXFRAME')) o.FixFrame = true;

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
   static async draw(dom, pad, opt) {
      const painter = new RPadPainter(dom, pad, opt, false, true);

      painter.createPadSvg();

      if (painter.matchObjectType(nsREX + 'RPad') && (painter.isTopPad() || painter.hasObjectsToDraw()))
         painter.addPadButtons();

      selectActivePad({ pp: painter, active: false });

      // flag used to prevent immediate pad redraw during first draw
      return painter.drawPrimitives().then(() => {
         painter.addPadInteractive();
         painter.showPadButtons();
         return painter;
      });
   }

} // class RPadPainter

export { RPadPainter };
