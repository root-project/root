import { gStyle, settings, constants, browser, internals, BIT,
         create, toJSON, isBatchMode, loadScript, injectCode, isPromise, getPromise, postponePromise,
         isObject, isFunc, isStr,
         clTObjArray, clTPaveText, clTColor, clTPad, clTFrame, clTStyle, clTLegend, clTHStack, clTMultiGraph, clTLegendEntry, kTitle } from '../core.mjs';
import { select as d3_select, rgb as d3_rgb } from '../d3.mjs';
import { ColorPalette, adoptRootColors, getColorPalette, getGrayColors, extendRootColors, getRGBfromTColor, decodeWebCanvasColors } from '../base/colors.mjs';
import { getElementRect, getAbsPosInCanvas, DrawOptions, compressSVG, makeTranslate, convertDate, svgToImage } from '../base/BasePainter.mjs';
import { ObjectPainter, selectActivePad, getActivePad } from '../base/ObjectPainter.mjs';
import { TAttLineHandler } from '../base/TAttLineHandler.mjs';
import { addCustomFont } from '../base/FontHandler.mjs';
import { addDragHandler } from './TFramePainter.mjs';
import { createMenu, closeMenu } from '../gui/menu.mjs';
import { ToolbarIcons, registerForResize, saveFile } from '../gui/utils.mjs';
import { BrowserLayout, getHPainter } from '../gui/display.mjs';


const clTButton = 'TButton', kIsGrayscale = BIT(22);

function getButtonSize(handler, fact) {
   return Math.round((fact || 1) * (handler.iscan || !handler.has_canvas ? 16 : 12));
}

function toggleButtonsVisibility(handler, action, evnt) {
   evnt?.preventDefault();
   evnt?.stopPropagation();

   const group = handler.getLayerSvg('btns_layer', handler.this_pad_name),
         btn = group.select('[name=\'Toggle\']');

   if (btn.empty()) return;

   let state = btn.property('buttons_state');

   if (btn.property('timout_handler')) {
      if (action !== 'timeout') clearTimeout(btn.property('timout_handler'));
      btn.property('timout_handler', null);
   }

   let is_visible = false;
   switch (action) {
      case 'enable':
         is_visible = true;
         handler.btns_active_flag = true;
         break;
      case 'enterbtn':
         handler.btns_active_flag = true;
         return; // do nothing, just cleanup timeout
      case 'timeout': is_visible = false; break;
      case 'toggle':
         state = !state;
         btn.property('buttons_state', state);
         is_visible = state;
         break;
      case 'disable':
      case 'leavebtn':
         handler.btns_active_flag = false;
         if (!state)
            btn.property('timout_handler', setTimeout(() => toggleButtonsVisibility(handler, 'timeout'), 1200));
         return;
   }

   group.selectAll('svg').each(function() {
      if (this !== btn.node())
         d3_select(this).style('display', is_visible ? '' : 'none');
   });
}

const PadButtonsHandler = {

   alignButtons(btns, width, height) {
      const sz0 = getButtonSize(this, 1.25), nextx = (btns.property('nextx') || 0) + sz0;
      let btns_x, btns_y;

      if (btns.property('vertical')) {
         btns_x = btns.property('leftside') ? 2 : (width - sz0);
         btns_y = height - nextx;
      } else {
         btns_x = btns.property('leftside') ? 2 : (width - nextx);
         btns_y = height - sz0;
      }

      makeTranslate(btns, btns_x, btns_y);
   },

   findPadButton(keyname) {
      const group = this.getLayerSvg('btns_layer', this.this_pad_name);
      let found_func = '';
      if (!group.empty()) {
         group.selectAll('svg').each(function() {
            if (d3_select(this).attr('key') === keyname)
               found_func = d3_select(this).attr('name');
         });
      }
      return found_func;
   },

   removePadButtons() {
      const group = this.getLayerSvg('btns_layer', this.this_pad_name);
      if (!group.empty()) {
         group.selectAll('*').remove();
         group.property('nextx', null);
      }
   },

   showPadButtons() {
      const group = this.getLayerSvg('btns_layer', this.this_pad_name);
      if (group.empty()) return;

      // clean all previous buttons
      group.selectAll('*').remove();
      if (!this._buttons) return;

      const iscan = this.iscan || !this.has_canvas, y = 0;
      let ctrl, x = group.property('leftside') ? getButtonSize(this, 1.25) : 0;

      if (this._fast_drawing) {
         ctrl = ToolbarIcons.createSVG(group, ToolbarIcons.circle, getButtonSize(this), 'enlargePad', false)
                            .attr('name', 'Enlarge').attr('x', 0).attr('y', 0)
                            .on('click', evnt => this.clickPadButton('enlargePad', evnt));
      } else {
         ctrl = ToolbarIcons.createSVG(group, ToolbarIcons.rect, getButtonSize(this), 'Toggle tool buttons', false)
                            .attr('name', 'Toggle').attr('x', 0).attr('y', 0)
                            .property('buttons_state', (settings.ToolBar !== 'popup') || browser.touches)
                            .on('click', evnt => toggleButtonsVisibility(this, 'toggle', evnt));
         ctrl.node()._mouseenter = () => toggleButtonsVisibility(this, 'enable');
         ctrl.node()._mouseleave = () => toggleButtonsVisibility(this, 'disable');

         for (let k = 0; k < this._buttons.length; ++k) {
            const item = this._buttons[k];
            let btn = item.btn;

            if (isStr(btn))
               btn = ToolbarIcons[btn];
            if (!btn)
               btn = ToolbarIcons.circle;

            const svg = ToolbarIcons.createSVG(group, btn, getButtonSize(this),
                        item.tooltip + (iscan ? '' : (` on pad ${this.this_pad_name}`)) + (item.keyname ? ` (keyshortcut ${item.keyname})` : ''), false);

            if (group.property('vertical'))
                svg.attr('x', y).attr('y', x);
            else
               svg.attr('x', x).attr('y', y);

            svg.attr('name', item.funcname)
               .style('display', ctrl.property('buttons_state') ? '' : 'none')
               .attr('key', item.keyname || null)
               .on('click', evnt => this.clickPadButton(item.funcname, evnt));

            svg.node()._mouseenter = () => toggleButtonsVisibility(this, 'enterbtn');
            svg.node()._mouseleave = () => toggleButtonsVisibility(this, 'leavebtn');

            x += getButtonSize(this, 1.25);
         }
      }

      group.property('nextx', x);

      this.alignButtons(group, this.getPadWidth(), this.getPadHeight());

      if (group.property('vertical'))
         ctrl.attr('y', x);
      else if (!group.property('leftside'))
         ctrl.attr('x', x);
   },

   assign(painter) {
      Object.assign(painter, this);
   }

}, // PadButtonsHandler

// identifier used in TWebCanvas painter
webSnapIds = { kNone: 0, kObject: 1, kSVG: 2, kSubPad: 3, kColors: 4, kStyle: 5, kFont: 6 };


/** @summary Fill TWebObjectOptions for painter
  * @private */
function createWebObjectOptions(painter) {
   if (!painter?.snapid)
      return null;

   const obj = { _typename: 'TWebObjectOptions', snapid: painter.snapid.toString(), opt: painter.getDrawOpt(true), fcust: '', fopt: [] };
   if (isFunc(painter.fillWebObjectOptions))
      painter.fillWebObjectOptions(obj);
   return obj;
}


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
      this.this_pad_name = '';
      if (!this.iscan && pad?.fName) {
         this.this_pad_name = pad.fName.replace(' ', '_'); // avoid empty symbol in pad name
         const regexp = /^[A-Za-z][A-Za-z0-9_]*$/;
         if (!regexp.test(this.this_pad_name) || ((this.this_pad_name === 'button') && (pad._typename === clTButton)))
            this.this_pad_name = 'jsroot_pad_' + internals.id_counter++;
      }
      this.painters = []; // complete list of all painters in the pad
      this.has_canvas = true;
      this.forEachPainter = this.forEachPainterInPad;
      const d = this.selectDom();
      if (!d.empty() && d.property('_batch_mode'))
         this.batch_mode = true;
   }

   /** @summary Indicates that drawing runs in batch mode
     * @private */
   isBatchMode() {
      if (this.batch_mode !== undefined)
         return this.batch_mode;

      if (isBatchMode())
         return true;

      if (!this.iscan && this.has_canvas)
         return this.getCanvPainter()?.isBatchMode();

      return false;
   }

   /** @summary Indicates that is is Root6 pad painter
    * @private */
   isRoot6() { return true; }

   /** @summary Returns true if pad is editable */
   isEditable() {
      return this.pad?.fEditable ?? true;
   }

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

      const svg_p = this.svg_this_pad();
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
      delete this._snap_primitives;
      delete this._last_grayscale;
      delete this._custom_colors;
      delete this._custom_palette_indexes;
      delete this._custom_palette_colors;
      delete this.root_colors;

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
      };
   }

   /** @summary return pad log state x or y are allowed */
   getPadLog(name) {
      const pad = this.getRootPad();
      if (name === 'x')
         return pad?.fLogx;
      if (name === 'y')
         return pad?.fLogv ?? pad?.fLogy;
      return false;
   }

   /** @summary Returns frame coordiantes - also when frame is not drawn */
   getFrameRect() {
      const fp = this.getFramePainter();
      if (fp) return fp.getFrameRect();

      const w = this.getPadWidth(),
            h = this.getPadHeight(),
            rect = {};

      if (this.pad) {
         rect.szx = Math.round(Math.max(0, 0.5 - Math.max(this.pad.fLeftMargin, this.pad.fRightMargin))*w);
         rect.szy = Math.round(Math.max(0, 0.5 - Math.max(this.pad.fBottomMargin, this.pad.fTopMargin))*h);
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
      rect.transform = makeTranslate(rect.x, rect.y) || '';

      return rect;
   }

   /** @summary return RPad object */
   getRootPad(is_root6) {
      return (is_root6 === undefined) || is_root6 ? this.pad : null;
   }

   /** @summary Cleanup primitives from pad - selector lets define which painters to remove */
   cleanPrimitives(selector) {
      if (!isFunc(selector)) return;

      for (let k = this.painters.length-1; k >= 0; --k) {
         if (selector(this.painters[k])) {
            this.painters[k].cleanup();
            this.painters.splice(k, 1);
         }
      }
   }

   /** @summary Removes and cleanup specified primitive
     * @desc also secondary primitives will be removed
     * @return new index to continue loop or -111 if main painter removed
     * @private */
   removePrimitive(indx) {
      const prim = this.painters[indx], arr = [];
      let resindx = indx;
      for (let k = this.painters.length-1; k >= 0; --k) {
         if ((k === indx) || this.painters[k].isSecondary(prim)) {
            arr.push(this.painters[k]);
            this.painters.splice(k, 1);
            if (k <= indx) resindx--;
         }
      }

      arr.forEach(painter => {
         painter.cleanup();
         if (this.main_painter_ref === painter) {
            delete this.main_painter_ref;
            resindx = -111;
         }
      });

      return resindx;
   }

  /** @summary returns custom palette associated with pad or top canvas
    * @private */
   getCustomPalette() {
      return this.custom_palette || this.getCanvPainter()?.custom_palette;
   }

   /** @summary Returns number of painters
     * @private */
   getNumPainters() { return this.painters.length; }

   /** @summary Provides automatic color
    * @desc Uses ROOT colors palette if possible
    * @private */
   getAutoColor(numprimitives) {
      if (!numprimitives)
         numprimitives = this._num_primitives || 5;
      if (numprimitives < 2) numprimitives = 2;

      let indx = this._auto_color ?? 0;
      this._auto_color = (indx + 1) % numprimitives;
      if (indx >= numprimitives) indx = numprimitives - 1;

      const indexes = this._custom_palette_indexes || this.getCanvPainter()?._custom_palette_indexes;

      if (indexes?.length) {
         const p = Math.round(indx * (indexes.length - 3) / (numprimitives - 1));
         return indexes[p];
      }

      if (!this._auto_palette)
         this._auto_palette = getColorPalette(settings.Palette, this.isGrayscale());
      const palindx = Math.round(indx * (this._auto_palette.getLength()-3) / (numprimitives-1)),
            colvalue = this._auto_palette.getColor(palindx);

      return this.addColor(colvalue);
   }

   /** @summary Call function for each painter in pad
     * @param {function} userfunc - function to call
     * @param {string} kind - 'all' for all objects (default), 'pads' only pads and subpads, 'objects' only for object in current pad
     * @private */
   forEachPainterInPad(userfunc, kind) {
      if (!kind) kind = 'all';
      if (kind !== 'objects') userfunc(this);
      for (let k = 0; k < this.painters.length; ++k) {
         const sub = this.painters[k];
         if (isFunc(sub.forEachPainterInPad)) {
            if (kind !== 'objects') sub.forEachPainterInPad(userfunc, kind);
         } else if (kind !== 'pads')
            userfunc(sub);
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
   producePadEvent(what, padpainter, painter, position, place) {
      if ((what === 'select') && isFunc(this.selectActivePad))
         this.selectActivePad(padpainter, painter, position);

      if (isFunc(this.pad_events_receiver))
         this.pad_events_receiver({ what, padpainter, painter, position, place });
   }

   /** @summary method redirect call to pad events receiver */
   selectObjectPainter(painter, pos, place) {
      const istoppad = this.iscan || !this.has_canvas,
          canp = istoppad ? this : this.getCanvPainter();

      if (painter === undefined) painter = this;

      if (pos && !istoppad)
         pos = getAbsPosInCanvas(this.svg_this_pad(), pos);

      selectActivePad({ pp: this, active: true });

      canp?.producePadEvent('select', this, painter, pos, place);
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
         svg_rect = this.iscan ? this.getCanvSvg().selectChild('.canvas_fillrect') : this.svg_this_pad().selectChild('.root_pad_border');

      const cp = this.getCanvPainter();

      let lineatt = this.is_active_pad && cp?.highlight_gpad ? new TAttLineHandler({ style: 1, width: 1, color: 'red' }) : this.lineatt;

      if (!lineatt) lineatt = new TAttLineHandler({ color: 'none' });

      svg_rect.call(lineatt.func);
   }

   /** @summary Set fast drawing property depending on the size
     * @private */
   setFastDrawing(w, h) {
      const was_fast = this._fast_drawing;
      this._fast_drawing = settings.SmallPad && ((w < settings.SmallPad.width) || (h < settings.SmallPad.height));
      if (was_fast !== this._fast_drawing)
         this.showPadButtons();
   }

   /** @summary Returns true if canvas configured with grayscale
     * @private */
   isGrayscale() {
      if (!this.iscan) return false;
      return this.pad?.TestBit(kIsGrayscale) ?? false;
   }

   /** @summary Set grayscale mode for the canvas
     * @private */
   setGrayscale(flag) {
      if (!this.iscan) return;

      let changed = false;

      if (flag === undefined) {
         flag = this.pad?.TestBit(kIsGrayscale) ?? false;
         changed = (this._last_grayscale !== undefined) && (this._last_grayscale !== flag);
      } else if (flag !== this.pad?.TestBit(kIsGrayscale)) {
         this.pad?.InvertBit(kIsGrayscale);
         changed = true;
      }

      if (changed)
         this.forEachPainter(p => { delete p._color_palette; });

      this.root_colors = flag ? getGrayColors(this._custom_colors) : this._custom_colors;

      this._last_grayscale = flag;

      this.custom_palette = this._custom_palette_colors ? new ColorPalette(this._custom_palette_colors, flag) : null;
   }

   /** @summary Create SVG element for canvas */
   createCanvasSvg(check_resize, new_size) {
      const is_batch = this.isBatchMode(), lmt = 5;
      let factor = null, svg = null, rect = null, btns, info, frect;

      if (check_resize > 0) {
         if (this._fixed_size)
            return check_resize > 1; // flag used to force re-drawing of all subpads

         svg = this.getCanvSvg();
         if (svg.empty())
            return false;

         factor = svg.property('height_factor');

         rect = this.testMainResize(check_resize, null, factor);

         if (!rect.changed && (check_resize === 1))
            return false;

         if (!is_batch)
            btns = this.getLayerSvg('btns_layer', this.this_pad_name);

         info = this.getLayerSvg('info_layer', this.this_pad_name);
         frect = svg.selectChild('.canvas_fillrect');
      } else {
         const render_to = this.selectDom();

         if (render_to.style('position') === 'static')
            render_to.style('position', 'relative');

         svg = render_to.append('svg')
             .attr('class', 'jsroot root_canvas')
             .property('pad_painter', this) // this is custom property
             .property('current_pad', '') // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         this.setTopPainter(); // assign canvas as top painter of that element

         if (is_batch)
            svg.attr('xmlns', 'http://www.w3.org/2000/svg');
         else if (!this.online_canvas)
            svg.append('svg:title').text('ROOT canvas');

         if (!is_batch || (this.pad.fFillStyle > 0))
            frect = svg.append('svg:path').attr('class', 'canvas_fillrect');

         if (!is_batch) {
            frect.style('pointer-events', 'visibleFill')
                 .on('dblclick', evnt => this.enlargePad(evnt, true))
                 .on('click', () => this.selectObjectPainter())
                 .on('mouseenter', () => this.showObjectStatus())
                 .on('contextmenu', settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);
         }

         svg.append('svg:g').attr('class', 'primitives_layer');
         info = svg.append('svg:g').attr('class', 'info_layer');
         if (!is_batch) {
            btns = svg.append('svg:g')
                      .attr('class', 'btns_layer')
                      .property('leftside', settings.ToolBarSide === 'left')
                      .property('vertical', settings.ToolBarVert);
         }

         factor = 0.66;
         if (this.pad?.fCw && this.pad?.fCh && (this.pad?.fCw > 0)) {
            factor = this.pad.fCh / this.pad.fCw;
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this._fixed_size) {
            render_to.style('overflow', 'auto');
            rect = { width: this.pad.fCw, height: this.pad.fCh };
            if (!rect.width || !rect.height)
               rect = getElementRect(render_to);
         } else
            rect = this.testMainResize(2, new_size, factor);
      }

      this.setGrayscale();

      this.createAttFill({ attr: this.pad });

      if ((rect.width <= lmt) || (rect.height <= lmt)) {
         svg.style('display', 'none');
         console.warn(`Hide canvas while geometry too small w=${rect.width} h=${rect.height}`);
         if (this._pad_width && this._pad_height) {
            // use last valid dimensions
            rect.width = this._pad_width;
            rect.height = this._pad_height;
         } else {
            // just to complete drawing.
            rect.width = 800;
            rect.height = 600;
         }
      } else
         svg.style('display', null);

      svg.attr('x', 0).attr('y', 0).style('position', 'absolute');

      if (this._fixed_size)
         svg.attr('width', rect.width).attr('height', rect.height);
      else
         svg.style('width', '100%').style('height', '100%').style('left', 0).style('top', 0).style('bottom', 0).style('right', 0);

      svg.style('filter', settings.DarkMode || this.pad?.$dark ? 'invert(100%)' : null);

      svg.attr('viewBox', `0 0 ${rect.width} ${rect.height}`)
         .attr('preserveAspectRatio', 'none')  // we do not preserve relative ratio
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
         frect.attr('d', `M0,0H${rect.width}V${rect.height}H0Z`)
              .call(this.fillatt.func);
         this.drawActiveBorder(frect);
      }

      this.setFastDrawing(rect.width * (1 - this.pad.fLeftMargin - this.pad.fRightMargin), rect.height * (1 - this.pad.fBottomMargin - this.pad.fTopMargin));

      if (this.alignButtons && btns)
         this.alignButtons(btns, rect.width, rect.height);

      let dt = info.selectChild('.canvas_date');
      if (!gStyle.fOptDate)
         dt.remove();
       else {
         if (dt.empty())
             dt = info.append('text').attr('class', 'canvas_date');
         const posy = Math.round(rect.height * (1 - gStyle.fDateY)),
               date = new Date();
         let posx = Math.round(rect.width * gStyle.fDateX);
         if (!is_batch && (posx < 25))
            posx = 25;
         if (gStyle.fOptDate > 3)
            date.setTime(gStyle.fOptDate*1000);

         makeTranslate(dt, posx, posy)
            .style('text-anchor', 'start')
            .text(convertDate(date));
      }

      const iname = this.getItemName();
      if (iname)
         this.drawItemNameOnCanvas(iname);
      else if (!gStyle.fOptFile)
         info.selectChild('.canvas_item').remove();

      return true;
   }

   /** @summary Draw item name on canvas if gStyle.fOptFile is configured
     * @private */
   drawItemNameOnCanvas(item_name) {
      const info = this.getLayerSvg('info_layer', this.this_pad_name);
      let df = info.selectChild('.canvas_item');
      const fitem = getHPainter().findRootFileForItem(item_name),
            fname = (gStyle.fOptFile === 3) ? item_name : ((gStyle.fOptFile === 2) ? fitem?._fullurl : fitem?._name);

      if (!gStyle.fOptFile || !fname)
         df.remove();
       else {
         if (df.empty())
            df = info.append('text').attr('class', 'canvas_item');
         const rect = this.getPadRect();
         makeTranslate(df, Math.round(rect.width * (1 - gStyle.fDateX)), Math.round(rect.height * (1 - gStyle.fDateY)))
            .style('text-anchor', 'end')
            .text(fname);
      }
      if (((gStyle.fOptDate === 2) || (gStyle.fOptDate === 3)) && fitem?._file) {
         info.selectChild('.canvas_date')
             .text(convertDate(gStyle.fOptDate === 2 ? fitem._file.fDatimeC.getDate() : fitem._file.fDatimeM.getDate()));
      }
   }

   /** @summary Return true if this pad enlarged */
   isPadEnlarged() {
      if (this.iscan || !this.has_canvas)
         return this.enlargeMain('state') === 'on';
      return this.getCanvSvg().property('pad_enlarged') === this.pad;
   }

   /** @summary Enlarge pad draw element when possible */
   enlargePad(evnt, is_dblclick, is_escape) {
      evnt?.preventDefault();
      evnt?.stopPropagation();

      // ignore double click on canvas itself for enlarge
      if (is_dblclick && this._websocket && (this.enlargeMain('state') === 'off'))
         return;

      const svg_can = this.getCanvSvg(),
            pad_enlarged = svg_can.property('pad_enlarged');

      if (this.iscan || !this.has_canvas || (!pad_enlarged && !this.hasObjectsToDraw() && !this.painters)) {
         if (this._fixed_size) return; // canvas cannot be enlarged in such mode
         if (!this.enlargeMain(is_escape ? false : 'toggle')) return;
         if (this.enlargeMain('state') === 'off')
            svg_can.property('pad_enlarged', null);
         else
            selectActivePad({ pp: this, active: true });
      } else if (!pad_enlarged && !is_escape) {
         this.enlargeMain(true, true);
         svg_can.property('pad_enlarged', this.pad);
         selectActivePad({ pp: this, active: true });
      } else if (pad_enlarged === this.pad) {
         this.enlargeMain(false);
         svg_can.property('pad_enlarged', null);
      } else if (!is_escape && is_dblclick)
         console.error('missmatch with pad double click events');

      return this.checkResize(true);
   }

   /** @summary Create main SVG element for pad
     * @return true when pad is displayed and all its items should be redrawn */
   createPadSvg(only_resize) {
      if (!this.has_canvas) {
         this.createCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      const svg_can = this.getCanvSvg(),
            width = svg_can.property('draw_width'),
            height = svg_can.property('draw_height'),
            pad_enlarged = svg_can.property('pad_enlarged'),
            pad_visible = !this.pad_draw_disabled && (!pad_enlarged || (pad_enlarged === this.pad)),
            is_batch = this.isBatchMode();
      let w = Math.round(this.pad.fAbsWNDC * width),
          h = Math.round(this.pad.fAbsHNDC * height),
          x = Math.round(this.pad.fAbsXlowNDC * width),
          y = Math.round(height * (1 - this.pad.fAbsYlowNDC)) - h,
          svg_pad, svg_border, btns;

      if (pad_enlarged === this.pad) { w = width; h = height; x = y = 0; }

      if (only_resize) {
         svg_pad = this.svg_this_pad();
         svg_border = svg_pad.selectChild('.root_pad_border');
         if (!is_batch)
            btns = this.getLayerSvg('btns_layer', this.this_pad_name);
         this.addPadInteractive(true);
      } else {
         svg_pad = svg_can.selectChild('.primitives_layer')
             .append('svg:svg') // svg used to blend all drawings outside
             .classed('__root_pad_' + this.this_pad_name, true)
             .attr('pad', this.this_pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this); // this is custom property

         if (!is_batch)
            svg_pad.append('svg:title').text('subpad ' + this.this_pad_name);

         // need to check attributes directly while attributes objects will be created later
         if (!is_batch || (this.pad.fFillStyle > 0) || ((this.pad.fLineStyle > 0) && (this.pad.fLineColor > 0)))
            svg_border = svg_pad.append('svg:path').attr('class', 'root_pad_border');

         if (!is_batch) {
            svg_border.style('pointer-events', 'visibleFill') // get events also for not visible rect
                      .on('dblclick', evnt => this.enlargePad(evnt, true))
                      .on('click', () => this.selectObjectPainter())
                      .on('mouseenter', () => this.showObjectStatus())
                      .on('contextmenu', settings.ContextMenu ? evnt => this.padContextMenu(evnt) : null);
         }

         svg_pad.append('svg:g').attr('class', 'primitives_layer');
         if (!is_batch) {
            btns = svg_pad.append('svg:g')
                          .attr('class', 'btns_layer')
                          .property('leftside', settings.ToolBarSide !== 'left')
                          .property('vertical', settings.ToolBarVert);
         }
      }

      this.createAttFill({ attr: this.pad });
      this.createAttLine({ attr: this.pad, color0: !this.pad.fBorderMode ? 'none' : '' });

      svg_pad.style('display', pad_visible ? null : 'none')
             .attr('viewBox', `0 0 ${w} ${h}`) // due to svg
             .attr('preserveAspectRatio', 'none')   // due to svg, we do not preserve relative ratio
             .attr('x', x)        // due to svg
             .attr('y', y)        // due to svg
             .attr('width', w)    // due to svg
             .attr('height', h)   // due to svg
             .property('draw_x', x) // this is to make similar with canvas
             .property('draw_y', y)
             .property('draw_width', w)
             .property('draw_height', h);

      this._pad_x = x;
      this._pad_y = y;
      this._pad_width = w;
      this._pad_height = h;

      if (svg_border) {
         svg_border.attr('d', `M0,0H${w}V${h}H0Z`)
                   .call(this.fillatt.func)
                   .call(this.lineatt.func);
         this.drawActiveBorder(svg_border);

         let svg_border1 = svg_pad.selectChild('.root_pad_border1'),
             svg_border2 = svg_pad.selectChild('.root_pad_border2');

         if (this.pad.fBorderMode && this.pad.fBorderSize) {
            const pw = this.pad.fBorderSize, ph = this.pad.fBorderSize,
                side1 = `M0,0h${w}l${-pw},${ph}h${2*pw-w}v${h-2*ph}l${-pw},${ph}z`,
                side2 = `M${w},${h}v${-h}l${-pw},${ph}v${h-2*ph}h${2*pw-w}l${-pw},${ph}z`;

            if (svg_border2.empty())
               svg_border2 = svg_pad.insert('svg:path', '.primitives_layer').attr('class', 'root_pad_border2');
            if (svg_border1.empty())
               svg_border1 = svg_pad.insert('svg:path', '.primitives_layer').attr('class', 'root_pad_border1');

            svg_border1.attr('d', this.pad.fBorderMode > 0 ? side1 : side2)
                       .call(this.fillatt.func)
                       .style('fill', d3_rgb(this.fillatt.color).brighter(0.5).formatHex());
            svg_border2.attr('d', this.pad.fBorderMode > 0 ? side2 : side1)
                       .call(this.fillatt.func)
                       .style('fill', d3_rgb(this.fillatt.color).darker(0.5).formatHex());
         } else {
            svg_border1.remove();
            svg_border2.remove();
         }
      }

      this.setFastDrawing(w * (1 - this.pad.fLeftMargin-this.pad.fRightMargin), h * (1 - this.pad.fBottomMargin - this.pad.fTopMargin));

      // special case of 3D canvas overlay
      if (svg_pad.property('can3d') === constants.Embed3D.Overlay) {
         this.selectDom().select('.draw3d_' + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');
      }

      if (this.alignButtons && btns)
         this.alignButtons(btns, w, h);

      return pad_visible;
   }

   /** @summary Add pad interactive features like dragging and resize
     * @private */
   addPadInteractive(cleanup = false) {
      if (isFunc(this.$userInteractive)) {
         this.$userInteractive();
         delete this.$userInteractive;
      }

      if (this.isBatchMode() || this.iscan)
         return;

      const svg_can = this.getCanvSvg(),
            width = svg_can.property('draw_width'),
            height = svg_can.property('draw_height');

      addDragHandler(this, {
         cleanup, // do cleanup to let assign new handlers later on
         x: this._pad_x, y: this._pad_y, width: this._pad_width, height: this._pad_height, no_transform: true,
         only_resize: true, // !cleanup && (this._disable_dragging || this.getFramePainter()?.mode3d),
         is_disabled: kind => svg_can.property('pad_enlarged') || this.btns_active_flag ||
                             (kind === 'move' && (this._disable_dragging || this.getFramePainter()?.mode3d)),
         getDrawG: () => this.svg_this_pad(),
         pad_rect: { width, height },
         minwidth: 20, minheight: 20,
         move_resize: (_x, _y, _w, _h) => {
            const x0 = this.pad.fAbsXlowNDC,
                y0 = this.pad.fAbsYlowNDC,
                scale_w = _w / width / this.pad.fAbsWNDC,
                scale_h = _h / height / this.pad.fAbsHNDC,
                shift_x = _x / width - x0,
                shift_y = 1 - (_y + _h) / height - y0;
            this.forEachPainterInPad(p => {
               p.pad.fAbsXlowNDC += (p.pad.fAbsXlowNDC - x0) * (scale_w - 1) + shift_x;
               p.pad.fAbsYlowNDC += (p.pad.fAbsYlowNDC - y0) * (scale_h - 1) + shift_y;
               p.pad.fAbsWNDC *= scale_w;
               p.pad.fAbsHNDC *= scale_h;
            }, 'pads');
         },
         redraw: () => this.interactiveRedraw('pad', 'padpos')
      });
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
     * @return {boolean} true if any */
   checkSpecial(obj) {
      if (!obj) return false;

      if (obj._typename === clTStyle) {
         Object.assign(gStyle, obj);
         return true;
      }

      if ((obj._typename === clTObjArray) && (obj.name === 'ListOfColors')) {
         if (this.options?.CreatePalette) {
            let arr = [];
            for (let n = obj.arr.length - this.options.CreatePalette; n < obj.arr.length; ++n) {
               const col = getRGBfromTColor(obj.arr[n]);
               if (!col) { console.log('Fail to create color for palette'); arr = null; break; }
               arr.push(col);
            }
            if (arr) this.custom_palette = new ColorPalette(arr);
         }

         if (!this.options || this.options.GlobalColors) // set global list of colors
            adoptRootColors(obj);

         // copy existing colors and extend with new values
         this._custom_colors = this.options?.LocalColors ? extendRootColors(null, obj) : null;
         return true;
      }

      if ((obj._typename === clTObjArray) && (obj.name === 'CurrentColorPalette')) {
         const arr = [], indx = [];
         let missing = false;
         for (let n = 0; n < obj.arr.length; ++n) {
            const col = obj.arr[n];
            if (col?._typename === clTColor) {
               indx[n] = col.fNumber;
               arr[n] = getRGBfromTColor(col);
            } else {
               console.log(`Missing color with index ${n}`);
               missing = true;
            }
         }

         const apply = (!this.options || (!missing && !this.options.IgnorePalette));
         this._custom_palette_indexes = apply ? indx : null;
         this._custom_palette_colors = apply ? arr : null;

         return true;
      }

      return false;
   }

   /** @summary Check if special objects appears in primitives
     * @desc it could be list of colors or palette */
   checkSpecialsInPrimitives(can) {
      const lst = can?.fPrimitives;
      if (!lst) return;
      for (let i = 0; i < lst.arr?.length; ++i) {
         if (this.checkSpecial(lst.arr[i])) {
            lst.arr.splice(i, 1);
            lst.opt.splice(i, 1);
            i--;
         }
      }
   }

   /** @summary try to find object by name in list of pad primitives
     * @desc used to find title drawing
     * @private */
   findInPrimitives(objname, objtype) {
      const match = obj => obj && (obj?.fName === objname) && (objtype ? (obj?._typename === objtype) : true),
            snap = this._snap_primitives?.find(snap => match((snap.fKind === webSnapIds.kObject) ? snap.fSnapshot : null));
      if (snap) return snap.fSnapshot;

      return this.pad?.fPrimitives?.arr.find(match);
   }

   /** @summary Try to find painter for specified object
     * @desc can be used to find painter for some special objects, registered as
     * histogram functions
     * @param {object} selobj - object to which painter should be search, set null to ignore parameter
     * @param {string} [selname] - object name, set to null to ignore
     * @param {string} [seltype] - object type, set to null to ignore
     * @return {object} - painter for specified object (if any)
     * @private */
   findPainterFor(selobj, selname, seltype) {
      return this.painters.find(p => {
         const pobj = p.getObject();
         if (!pobj) return false;

         if (selobj && (pobj === selobj)) return true;
         if (!selname && !seltype) return false;
         if (selname && (pobj.fName !== selname)) return false;
         if (seltype && (pobj._typename !== seltype)) return false;
         return true;
      });
   }

   /** @summary Return true if any objects beside sub-pads exists in the pad */
   hasObjectsToDraw() {
      return this.pad?.fPrimitives?.arr?.find(obj => obj._typename !== clTPad);
   }

   /** @summary sync drawing/redrawing/resize of the pad
     * @param {string} kind - kind of draw operation, if true - always queued
     * @return {Promise} when pad is ready for draw operation or false if operation already queued
     * @private */
   syncDraw(kind) {
      const entry = { kind: kind || 'redraw' };
      if (this._doing_draw === undefined) {
         this._doing_draw = [entry];
         return Promise.resolve(true);
      }
      // if queued operation registered, ignore next calls, indx === 0 is running operation
      if ((entry.kind !== true) && (this._doing_draw.findIndex((e, i) => (i > 0) && (e.kind === entry.kind)) > 0))
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
         return console.warn('failure, should not happen');
      this._doing_draw.shift();
      if (this._doing_draw.length === 0)
         delete this._doing_draw;
       else {
         const entry = this._doing_draw[0];
         if (entry.func) { entry.func(); delete entry.func; }
      }
   }

   /** @summary Draw single primitive */
   async drawObject(/* dom, obj, opt */) {
      console.log('Not possible to draw object without loading of draw.mjs');
      return null;
   }

   /** @summary Draw pad primitives
     * @return {Promise} when drawing completed
     * @private */
   async drawPrimitives(indx) {
      if (indx === undefined) {
         if (this.iscan)
            this._start_tm = new Date().getTime();

         // set number of primitves
         this._num_primitives = this.pad?.fPrimitives?.arr?.length || 0;

         // sync to prevent immediate pad redraw during normal drawing sequence
         return this.syncDraw(true).then(() => this.drawPrimitives(0));
      }

      if (!this.pad || (indx >= this._num_primitives)) {
         if (this._start_tm) {
            const spenttm = new Date().getTime() - this._start_tm;
            if (spenttm > 1000) console.log(`Canvas ${this.pad?.fName || '---'} drawing took ${(spenttm*1e-3).toFixed(2)}s`);
            delete this._start_tm;
         }

         this.confirmDraw();
         return;
      }

      const obj = this.pad.fPrimitives.arr[indx];

      if (!obj || ((indx > 0) && (obj._typename === clTFrame) && this.getFramePainter()))
         return this.drawPrimitives(indx+1);

      // use of Promise should avoid large call-stack depth when many primitives are drawn
      return this.drawObject(this.getDom(), obj, this.pad.fPrimitives.opt[indx]).then(op => {
         if (isObject(op))
            op._primitive = true; // mark painter as belonging to primitives

         return this.drawPrimitives(indx+1);
      });
   }

   /** @summary Divide pad on subpads
     * @return {Promise} when finished
     * @private */
   async divide(nx, ny) {
      if (!this.pad.Divide(nx, ny))
         return this;

      const drawNext = indx => {
         if (indx >= this.pad.fPrimitives.arr.length)
            return this;
         return this.drawObject(this.getDom(), this.pad.fPrimitives.arr[indx]).then(() => drawNext(indx + 1));
      };

      return drawNext(0);
   }

   /** @summary Return sub-pads painter, only direct childs are checked
     * @private */
   getSubPadPainter(n) {
      for (let k = 0; k < this.painters.length; ++k) {
         const sub = this.painters[k];
         if (sub.pad && isFunc(sub.forEachPainterInPad) && (sub.pad.fNumber === n)) return sub;
      }
      return null;
   }


   /** @summary Process tooltip event in the pad
     * @private */
   processPadTooltipEvent(pnt) {
      const painters = [], hints = [];

      // first count - how many processors are there
      this.painters?.forEach(obj => {
         if (isFunc(obj.processTooltipEvent))
            painters.push(obj);
      });

      if (pnt) pnt.nproc = painters.length;

      painters.forEach(obj => {
         const hint = obj.processTooltipEvent(pnt) || { user_info: null };
         hints.push(hint);
         if (pnt?.painters) hint.painter = obj;
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
      if (this.pad)
         menu.add(`header:${this.pad._typename}::${this.pad.fName}`);
      else
         menu.add('header:Canvas');

      menu.addchk(this.isTooltipAllowed(), 'Show tooltips', () => this.setTooltipAllowed('toggle'));

      if (!this._websocket) {
         function SetPadField(arg) {
            this.pad[arg.slice(1)] = parseInt(arg[0]);
            this.interactiveRedraw('pad', arg.slice(1));
         }

         menu.addchk(this.pad?.fGridx, 'Grid x', (this.pad?.fGridx ? '0' : '1') + 'fGridx', SetPadField);
         menu.addchk(this.pad?.fGridy, 'Grid y', (this.pad?.fGridy ? '0' : '1') + 'fGridy', SetPadField);
         menu.add('sub:Ticks x');
         menu.addchk(this.pad?.fTickx === 0, 'normal', '0fTickx', SetPadField);
         menu.addchk(this.pad?.fTickx === 1, 'ticks on both sides', '1fTickx', SetPadField);
         menu.addchk(this.pad?.fTickx === 2, 'labels on both sides', '2fTickx', SetPadField);
         menu.add('endsub:');
         menu.add('sub:Ticks y');
         menu.addchk(this.pad?.fTicky === 0, 'normal', '0fTicky', SetPadField);
         menu.addchk(this.pad?.fTicky === 1, 'ticks on both sides', '1fTicky', SetPadField);
         menu.addchk(this.pad?.fTicky === 2, 'labels on both sides', '2fTicky', SetPadField);
         menu.add('endsub:');
         menu.addchk(this.pad?.fEditable, 'Editable', flag => { this.pad.fEditable = flag; this.interactiveRedraw('pad'); });
         if (this.iscan)
            menu.addchk(this.pad?.TestBit(kIsGrayscale), 'Gray scale', flag => { this.setGrayscale(flag); this.interactiveRedraw('pad'); });

         if (isFunc(this.drawObject))
            menu.add('Build legend', () => this.buildLegend());

         menu.addAttributesMenu(this);
         menu.add('Save to gStyle', () => {
            if (!this.pad) return;
            this.fillatt?.saveToStyle(this.iscan ? 'fCanvasColor' : 'fPadColor');
            gStyle.fPadGridX = this.pad.fGridx;
            gStyle.fPadGridY = this.pad.fGridy;
            gStyle.fPadTickX = this.pad.fTickx;
            gStyle.fPadTickY = this.pad.fTicky;
            gStyle.fOptLogx = this.pad.fLogx;
            gStyle.fOptLogy = this.pad.fLogy;
            gStyle.fOptLogz = this.pad.fLogz;
         }, 'Store pad fill attributes, grid, tick and log scale settings to gStyle');

         if (this.iscan) {
            menu.addSettingsMenu(false, false, arg => {
               if (arg === 'dark') this.changeDarkMode();
            });
         }
      }

      menu.add('separator');

      if (isFunc(this.hasMenuBar) && isFunc(this.actiavteMenuBar))
         menu.addchk(this.hasMenuBar(), 'Menu bar', flag => this.actiavteMenuBar(flag));

      if (isFunc(this.hasEventStatus) && isFunc(this.activateStatusBar) && isFunc(this.canStatusBar)) {
         if (this.canStatusBar())
            menu.addchk(this.hasEventStatus(), 'Event status', () => this.activateStatusBar('toggle'));
      }

      if (this.enlargeMain() || (this.has_canvas && this.hasObjectsToDraw()))
         menu.addchk(this.isPadEnlarged(), 'Enlarge ' + (this.iscan ? 'canvas' : 'pad'), () => this.enlargePad());

      const fname = this.this_pad_name || (this.iscan ? 'canvas' : 'pad');
      menu.add('sub:Save as');
      ['svg', 'png', 'jpeg', 'pdf', 'webp'].forEach(fmt => menu.add(`${fname}.${fmt}`, () => this.saveAs(fmt, this.iscan, `${fname}.${fmt}`)));
      menu.add('endsub:');

      return true;
   }

   /** @summary Show pad context menu
     * @private */
   async padContextMenu(evnt) {
      if (evnt.stopPropagation) {
         // this is normal event processing and not emulated jsroot event
         evnt.stopPropagation(); // disable main context menu
         evnt.preventDefault();  // disable browser context menu
         this.getFramePainter()?.setLastEventPos();
      }

      return createMenu(evnt, this).then(menu => {
         this.fillContextMenu(menu);
         return this.fillObjectExecMenu(menu, '');
      }).then(menu => menu.show());
   }

   /** @summary Redraw pad means redraw ourself
     * @return {Promise} when redrawing ready */
   async redrawPad(reason) {
      const sync_promise = this.syncDraw(reason);
      if (sync_promise === false) {
         console.log(`Prevent redrawing of ${this.pad.fName}`);
         return false;
      }

      let showsubitems = true;
      const redrawNext = indx => {
         while (indx < this.painters.length) {
            const sub = this.painters[indx++];
            let res = 0;
            if (showsubitems || sub.this_pad_name)
               res = sub.redraw(reason);

            if (isPromise(res))
               return res.then(() => redrawNext(indx));
         }
         return true;
      };

      return sync_promise.then(() => {
         if (this.iscan)
            this.createCanvasSvg(2);
         else
            showsubitems = this.createPadSvg(true);
         return redrawNext(0);
      }).then(() => {
         this.addPadInteractive();
         this.confirmDraw();
         if (getActivePad() === this)
            this.getCanvPainter()?.producePadEvent('padredraw', this);
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
      const elem = this.svg_this_pad();
      if (!elem.empty() && elem.property('can3d') === constants.Embed3D.Overlay) return true;

      return this.painters.findIndex(objp => {
         return isFunc(objp.needRedrawByResize) ? objp.needRedrawByResize() : false;
      }) >= 0;
   }

   /** @summary Check resize of canvas
     * @return {Promise} with result or false */
   checkCanvasResize(size, force) {
      if (this._ignore_resize)
         return false;

      if (!this.iscan && this.has_canvas) return false;

      const sync_promise = this.syncDraw('canvas_resize');
      if (sync_promise === false) return false;

      if ((size === true) || (size === false)) { force = size; size = null; }

      if (isObject(size) && size.force) force = true;

      if (!force) force = this.needRedrawByResize();

      let changed = false;
      const redrawNext = indx => {
         if (!changed || (indx >= this.painters.length)) {
            this.confirmDraw();
            return changed;
         }

         return getPromise(this.painters[indx].redraw(force ? 'redraw' : 'resize')).then(() => redrawNext(indx+1));
      };

      // return sync_promise.then(() => this.ensureBrowserSize(this.pad?.fCw, this.pad?.fCh)).then(() => {

      return sync_promise.then(() => {
         changed = this.createCanvasSvg(force ? 2 : 1, size);

         if (changed && this.iscan && this.pad && this.online_canvas && !this.embed_canvas && !this.isBatchMode()) {
            if (this._resize_tmout)
               clearTimeout(this._resize_tmout);
            this._resize_tmout = setTimeout(() => {
               delete this._resize_tmout;
               if (isFunc(this.sendResized))
                  this.sendResized();
            }, 1000); // long enough delay to prevent multiple occurence
         }

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
      this.pad.fLogx = obj.fLogx;
      this.pad.fLogy = obj.fLogy;
      this.pad.fLogz = obj.fLogz;

      this.pad.fUxmin = obj.fUxmin;
      this.pad.fUxmax = obj.fUxmax;
      this.pad.fUymin = obj.fUymin;
      this.pad.fUymax = obj.fUymax;

      this.pad.fX1 = obj.fX1;
      this.pad.fX2 = obj.fX2;
      this.pad.fY1 = obj.fY1;
      this.pad.fY2 = obj.fY2;

      this.pad.fLeftMargin = obj.fLeftMargin;
      this.pad.fRightMargin = obj.fRightMargin;
      this.pad.fBottomMargin = obj.fBottomMargin;
      this.pad.fTopMargin = obj.fTopMargin;

      this.pad.fFillColor = obj.fFillColor;
      this.pad.fFillStyle = obj.fFillStyle;
      this.pad.fLineColor = obj.fLineColor;
      this.pad.fLineStyle = obj.fLineStyle;
      this.pad.fLineWidth = obj.fLineWidth;

      this.pad.fPhi = obj.fPhi;
      this.pad.fTheta = obj.fTheta;
      this.pad.fEditable = obj.fEditable;

      if (this.iscan)
         this.checkSpecialsInPrimitives(obj);

      const fp = this.getFramePainter();
      if (fp) fp.updateAttributes(!fp.modified_NDC);

      if (!obj.fPrimitives) return false;

      let isany = false, p = 0;
      for (let n = 0; n < obj.fPrimitives.arr?.length; ++n) {
         while (p < this.painters.length) {
            const op = this.painters[p++];
            if (!op._primitive) continue;
            if (op.updateObject(obj.fPrimitives.arr[n], obj.fPrimitives.opt[n]))
               isany = true;
            break;
         }
      }

      return isany;
   }

   /** @summary add legend object to the pad and redraw it
     * @private */
   async buildLegend(x1, y1, x2, y2, title, opt) {
      const lp = this.findPainterFor(null, '', clTLegend);

      if (!lp && !isFunc(this.drawObject))
         return Promise.reject(Error('Not possible to build legend while module draw.mjs was not load'));

      const leg = lp?.getObject() ?? create(clTLegend),
            pad = this.getRootPad(true);

      leg.fPrimitives.Clear();

      for (let k = 0; k < this.painters.length; ++k) {
         const painter = this.painters[k],
               obj = painter.getObject();
         if (!obj || obj.fName === kTitle || obj.fName === 'stats' || painter.draw_content === false ||
              obj._typename === clTLegend || obj._typename === clTHStack || obj._typename === clTMultiGraph)
            continue;

         const entry = create(clTLegendEntry);
         entry.fObject = obj;
         entry.fLabel = painter.getItemName();
         if ((opt === 'all') || !entry.fLabel)
             entry.fLabel = obj.fName;
         entry.fOption = '';
         if (!entry.fLabel) continue;

         if (painter.lineatt?.used)
            entry.fOption += 'l';
         if (painter.fillatt?.used)
            entry.fOption += 'f';
         if (painter.markeratt?.used)
            entry.fOption += 'p';
         if (!entry.fOption)
            entry.fOption = 'l';

         leg.fPrimitives.Add(entry);
      }

      if (lp)
         return lp.redraw();

      const szx = 0.4;
      let szy = leg.fPrimitives.arr.length;
      // no entries - no need to draw legend
      if (!szy) return null;
      if (szy > 8) szy = 8;
      szy *= 0.1;

      if ((x1 === x2) || (y1 === y2)) {
         leg.fX1NDC = szx * pad.fLeftMargin + (1 - szx) * (1 - pad.fRightMargin);
         leg.fY1NDC = (1 - szy) * (1 - pad.fTopMargin) + szy * pad.fBottomMargin;
         leg.fX2NDC = 0.99 - pad.fRightMargin;
         leg.fY2NDC = 0.99 - pad.fTopMargin;
         if (opt === undefined) opt = 'autoplace';
      } else {
         leg.fX1NDC = x1;
         leg.fY1NDC = y1;
         leg.fX2NDC = x2;
         leg.fY2NDC = y2;
      }
      leg.fFillStyle = 1001;
      leg.fTitle = title ?? '';

      const prev_name = this.has_canvas ? this.selectCurrentPad(this.this_pad_name) : undefined;

      return this.drawObject(this.getDom(), leg, opt).then(p => {
         this.selectCurrentPad(prev_name);
         return p;
      });
   }

   /** @summary Add object painter to list of primitives
     * @private */
   addObjectPainter(objpainter, lst, indx) {
      if (objpainter && lst && lst[indx] && (objpainter.snapid === undefined)) {
         // keep snap id in painter, will be used for the
         if (this.painters.indexOf(objpainter) < 0)
            this.painters.push(objpainter);

         objpainter.snapid = lst[indx].fObjectID;
         const setSubSnaps = p => {
            if (!p._unique_painter_id) return;
            for (let k = 0; k < this.painters.length; ++k) {
               const sub = this.painters[k];
               if ((sub._main_painter_id === p._unique_painter_id) && sub._secondary_id) {
                  sub.snapid = p.snapid + '#' + sub._secondary_id;
                  setSubSnaps(sub);
               }
            }
         };

         setSubSnaps(objpainter);
      }
   }

   /** @summary Process snap with style
     * @private */
   processSnapStyle(snap) {
      Object.assign(gStyle, snap.fSnapshot);
   }

   /** @summary Process snap with colors
     * @private */
   processSnapColors(snap) {
      const ListOfColors = decodeWebCanvasColors(snap.fSnapshot.fOper);

      // set global list of colors
      if (!this.options || this.options.GlobalColors)
         adoptRootColors(ListOfColors);

      const greyscale = this.pad?.TestBit(kIsGrayscale) ?? false,
            colors = extendRootColors(null, ListOfColors, greyscale);

      // copy existing colors and extend with new values
      this._custom_colors = this.options?.LocalColors ? colors : null;

      // set palette
      if (snap.fSnapshot.fBuf && (!this.options || !this.options.IgnorePalette)) {
         const indexes = [], palette = [];
         for (let n = 0; n < snap.fSnapshot.fBuf.length; ++n) {
            indexes[n] = Math.round(snap.fSnapshot.fBuf[n]);
            palette[n] = colors[indexes[n]];
         }
         this._custom_palette_indexes = indexes;
         this._custom_palette_colors = palette;
         this.custom_palette = new ColorPalette(palette, greyscale);
      } else {
         delete this._custom_palette_indexes;
         delete this._custom_palette_colors;
         delete this.custom_palette;
      }
   }

   /** @summary Process snap with custom font
     * @private */
   processSnapFont(snap) {
      const arr = snap.fSnapshot.fOper.split(':');
      addCustomFont(Number.parseInt(arr[0]), arr[1], arr[2], arr[3]);
   }

   /** @summary Process special snaps like colors or style objects
     * @return {Promise} index where processing should start
     * @private */
   processSpecialSnaps(lst) {
      while (lst?.length) {
         const snap = lst[0];

         // gStyle object
         if (snap.fKind === webSnapIds.kStyle) {
            lst.shift();
            this.processSnapStyle(snap);
         } else if (snap.fKind === webSnapIds.kColors) {
            lst.shift();
            this.processSnapColors(snap);
         } else if (snap.fKind === webSnapIds.kFont) {
            lst.shift();
            this.processSnapFont(snap);
         } else
            break;
      }
   }

   /** @summary Function called when drawing next snapshot from the list
     * @return {Promise} for drawing of the snap
     * @private */
   async drawNextSnap(lst, indx) {
      if (indx === undefined) {
         indx = -1;
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
      }

      ++indx; // change to the next snap

      if (!lst || (indx >= lst.length)) {
         delete this._snaps_map;
         return this;
      }

      const snap = lst[indx];

      // gStyle object
      if (snap.fKind === webSnapIds.kStyle) {
         this.processSnapStyle(snap);
         return this.drawNextSnap(lst, indx); // call next
      }

      // list of colors
      if (snap.fKind === webSnapIds.kColors) {
         this.processSnapColors(snap);
         return this.drawNextSnap(lst, indx); // call next
      }

      const snapid = snap.fObjectID,
            is_frame = (snap.fKind === webSnapIds.kObject) && (snap.fSnapshot?._typename === clTFrame);
      let cnt = (this._snaps_map[snapid] || 0) + 1,
          objpainter = null;

      this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

      // first appropriate painter for the object
      // if same object drawn twice, two painters will exists
      for (let k = 0; k < this.painters.length; ++k) {
         const subp = this.painters[k];
         if (subp.snapid === snapid) {
            if (--cnt === 0) {
               objpainter = subp;
               break;
            }
         } else if (is_frame && !subp.snapid && (subp === this.getFramePainter())) {
            // workaround for the case when frame created afterwards by server
            subp.snapid = snapid;
            objpainter = subp;
            break;
         }
      }

      if (objpainter) {
         if (snap.fKind === webSnapIds.kSubPad) // subpad
            return objpainter.redrawPadSnap(snap).then(() => this.drawNextSnap(lst, indx));

         let promise;

         if (snap.fKind === webSnapIds.kObject) { // object itself
            if (objpainter.updateObject(snap.fSnapshot, snap.fOption, true))
               promise = objpainter.redraw();
         } else if (snap.fKind === webSnapIds.kSVG) { // update SVG
            if (objpainter.updateObject(snap.fSnapshot))
               promise = objpainter.redraw();
         }

         return getPromise(promise).then(() => this.drawNextSnap(lst, indx)); // call next
      }

      if (snap.fKind === webSnapIds.kSubPad) { // subpad
         const subpad = snap.fSnapshot;

         subpad.fPrimitives = null; // clear primitives, they just because of I/O

         const padpainter = new TPadPainter(this.getDom(), subpad, false);
         padpainter.decodeOptions(snap.fOption);
         padpainter.addToPadPrimitives(this.this_pad_name);
         padpainter.snapid = snap.fObjectID;
         padpainter.is_active_pad = !!snap.fActive; // enforce boolean flag
         padpainter._readonly = snap.fReadOnly ?? false; // readonly flag
         padpainter._snap_primitives = snap.fPrimitives; // keep list to be able find primitive
         padpainter._has_execs = snap.fHasExecs ?? false; // are there pad execs, enables some interactive features

         if (subpad.$disable_drawing)
            padpainter.pad_draw_disabled = true;

         padpainter.processSpecialSnaps(snap.fPrimitives); // need to process style and colors before creating graph elements

         padpainter.createPadSvg();

         if (padpainter.matchObjectType(clTPad) && (snap.fPrimitives.length > 0))
            padpainter.addPadButtons(true);

         // we select current pad, where all drawing is performed
         const prev_name = padpainter.selectCurrentPad(padpainter.this_pad_name);
         return padpainter.drawNextSnap(snap.fPrimitives).then(() => {
            padpainter.addPadInteractive();
            padpainter.selectCurrentPad(prev_name);
            return this.drawNextSnap(lst, indx); // call next
         });
      }

      // here the case of normal drawing, will be handled in promise
      if (((snap.fKind === webSnapIds.kObject) || (snap.fKind === webSnapIds.kSVG)) && (snap.fOption !== '__ignore_drawing__')) {
         return this.drawObject(this.getDom(), snap.fSnapshot, snap.fOption).then(objpainter => {
            this.addObjectPainter(objpainter, lst, indx);
            return this.drawNextSnap(lst, indx);
         });
      }

      return this.drawNextSnap(lst, indx);
   }

   /** @summary Return painter with specified id
     * @private */
   findSnap(snapid) {
      if (this.snapid === snapid)
         return this;

      if (!this.painters)
         return null;

      for (let k = 0; k < this.painters.length; ++k) {
         let sub = this.painters[k];

         if (isFunc(sub.findSnap))
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
     * @return {Promise} with pad painter when drawing completed
     * @private */
   async redrawPadSnap(snap) {
      if (!snap?.fPrimitives)
         return this;

      this.is_active_pad = !!snap.fActive; // enforce boolean flag
      this._readonly = snap.fReadOnly ?? false; // readonly flag
      this._snap_primitives = snap.fPrimitives; // keep list to be able find primitive
      this._has_execs = snap.fHasExecs ?? false; // are there pad execs, enables some interactive features

      const first = snap.fSnapshot;
      first.fPrimitives = null; // primitives are not interesting, they are disabled in IO

      // if there are execs in the pad, deliver events to the server
      this._deliver_webcanvas_events = first.fExecs?.arr?.length > 0;

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.snapid = snap.fObjectID;

         this.draw_object = this.pad = first; // first object is pad

         // this._fixed_size = true;

         // if canvas size not specified in batch mode, temporary use 900x700 size
         if (this.isBatchMode() && (!first.fCw || !first.fCh)) { first.fCw = 900; first.fCh = 700; }

         // case of ROOT7 with always dummy TPad as first entry
         if (!first.fCw || !first.fCh) this._fixed_size = false;

         const mainid = this.selectDom().attr('id');

         if (!this.isBatchMode() && !this.use_openui && !this.brlayout && mainid && isStr(mainid)) {
            this.brlayout = new BrowserLayout(mainid, null, this);
            this.brlayout.create(mainid, true);
            // this.brlayout.toggleBrowserKind('float');
            this.setDom(this.brlayout.drawing_divid()); // need to create canvas
            registerForResize(this.brlayout);
         }

         this.processSpecialSnaps(snap.fPrimitives);

         this.createCanvasSvg(0);

         if (!this.isBatchMode())
            this.addPadButtons(true);

         if (typeof snap.fHighlightConnect !== 'undefined')
            this._highlight_connect = snap.fHighlightConnect;

         let pr = Promise.resolve(true);

         if (isStr(snap.fScripts) && snap.fScripts) {
            let src = '';

            if (snap.fScripts.indexOf('load:') === 0)
               src = snap.fScripts.slice(5).split(';');
            else if (snap.fScripts.indexOf('assert:') === 0)
               src = snap.fScripts.slice(7);

            pr = src ? loadScript(src) : injectCode(snap.fScripts);
         }

         return pr.then(() => this.drawNextSnap(snap.fPrimitives));
      }

      this.updateObject(first); // update only object attributes

      // apply all changes in the object (pad or canvas)
      if (this.iscan)
         this.createCanvasSvg(2);
       else
         this.createPadSvg(true);

      const matchPrimitive = (painters, primitives, class_name, obj_name) => {
         const painter = painters.find(p => {
            if (p.snapid === undefined) return false;
            if (!p.matchObjectType(class_name)) return false;
            if (obj_name && (!p.getObject() || (p.getObject().fName !== obj_name))) return false;
            return true;
         });
         if (!painter) return;
         const primitive = primitives.find(pr => {
            if ((pr.fKind !== 1) || !pr.fSnapshot || (pr.fSnapshot._typename !== class_name)) return false;
            if (obj_name && (pr.fSnapshot.fName !== obj_name)) return false;
            return true;
         });
         if (!primitive) return;

         // force painter to use new object id
         if (painter.snapid !== primitive.fObjectID)
            painter.snapid = primitive.fObjectID;
      };

      // check if frame or title was recreated, we could reassign handlers for them directly
      // while this is temporary objects, which can be recreated very often, try to catch such situation ourselfs
      if (!snap.fWithoutPrimitives) {
         matchPrimitive(this.painters, snap.fPrimitives, clTFrame);
         matchPrimitive(this.painters, snap.fPrimitives, clTPaveText, kTitle);
      }

      let isanyfound = false, isanyremove = false;

      // find and remove painters which no longer exists in the list
      if (!snap.fWithoutPrimitives) {
         for (let k = 0; k < this.painters.length; ++k) {
            const sub = this.painters[k];

            // skip secondary painters or painters without snapid
            if (!isStr(sub.snapid) || sub.isSecondary()) continue; // look only for painters with snapid

            const prim = snap.fPrimitives.find(prim => (prim.fObjectID === sub.snapid && !prim.$checked));
            if (prim) {
               isanyfound = true;
               prim.$checked = true;
            } else {
               // remove painter which does not found in the list of snaps
               k = this.removePrimitive(k); // index modified
               isanyremove = true;
               if (k === -111) {
                  // main painter is removed - do full cleanup and redraw
                  isanyfound = false;
                  break;
               }
            }
         }
      }

      if (isanyremove)
         delete this.pads_cache;

      if (!isanyfound && !snap.fWithoutPrimitives) {
         // TODO: maybe just remove frame painter?
         const fp = this.getFramePainter(),
               old_painters = this.painters;
         this.painters = [];
         old_painters.forEach(objp => {
            if (fp !== objp) objp.cleanup();
         });
         delete this.main_painter_ref;
         if (fp) {
            this.painters.push(fp);
            fp.cleanFrameDrawings();
            fp.redraw();
         }
         if (isFunc(this.removePadButtons)) this.removePadButtons();
         this.addPadButtons(true);
      }

      const prev_name = this.selectCurrentPad(this.this_pad_name);

      return this.drawNextSnap(snap.fPrimitives).then(() => {
         this.addPadInteractive();
         this.selectCurrentPad(prev_name);
         if (getActivePad() === this)
            this.getCanvPainter()?.producePadEvent('padredraw', this);
         return this;
      });
   }

   /** @summary Deliver mouse move or click event to the web canvas
     * @private */
   deliverWebCanvasEvent(kind, x, y, hints) {
      if (!this._deliver_webcanvas_events || !this.is_active_pad || this.doingDraw() || x === undefined || y === undefined) return;
      const cp = this.getCanvPainter();
      if (!cp || !cp._websocket || !cp._websocket.canSend(2) || cp._readonly) return;

      let selobj_snapid = '';
      if (hints && hints[0] && hints[0].painter?.snapid)
         selobj_snapid = hints[0].painter.snapid.toString();

      const msg = JSON.stringify([this.snapid, kind, x.toString(), y.toString(), selobj_snapid]);

      cp.sendWebsocket(`EVENT:${msg}`);
   }

   /** @summary Create image for the pad
     * @desc Used with web-based canvas to create images for server side
     * @return {Promise} with image data, coded with btoa() function
     * @private */
   async createImage(format) {
      if ((format === 'png') || (format === 'jpeg') || (format === 'svg') || (format === 'pdf')) {
         return this.produceImage(true, format).then(res => {
            if (!res || (format === 'svg')) return res;
            const separ = res.indexOf('base64,');
            return (separ > 0) ? res.slice(separ+7) : '';
         });
      }

      return '';
   }

   /** @summary Collects pad information for TWebCanvas
     * @desc need to update different states
     * @private */
   getWebPadOptions(arg, cp) {
      let is_top = (arg === undefined), elem = null, scan_subpads = true;
      // no any options need to be collected in readonly mode
      if (is_top && this._readonly)
         return '';
      if (arg === 'only_this') {
         is_top = true;
         scan_subpads = false;
      } else if (arg === 'with_subpads') {
         is_top = true;
         scan_subpads = true;
      }
      if (is_top) arg = [];
      if (!cp) cp = this.iscan ? this : this.getCanvPainter();

      if (this.snapid) {
         elem = { _typename: 'TWebPadOptions', snapid: this.snapid.toString(),
                  active: !!this.is_active_pad,
                  cw: 0, ch: 0, w: [],
                  bits: 0, primitives: [],
                  logx: this.pad.fLogx, logy: this.pad.fLogy, logz: this.pad.fLogz,
                  gridx: this.pad.fGridx, gridy: this.pad.fGridy,
                  tickx: this.pad.fTickx, ticky: this.pad.fTicky,
                  mleft: this.pad.fLeftMargin, mright: this.pad.fRightMargin,
                  mtop: this.pad.fTopMargin, mbottom: this.pad.fBottomMargin,
                  xlow: 0, ylow: 0, xup: 1, yup: 1,
                  zx1: 0, zx2: 0, zy1: 0, zy2: 0, zz1: 0, zz2: 0 };

         if (this.iscan) {
            elem.bits = this.getStatusBits();
            elem.cw = this.getPadWidth();
            elem.ch = this.getPadHeight();
            elem.w = [window.screenLeft, window.screenTop, window.outerWidth, window.outerHeight];
         } else if (cp) {
            const cw = cp.getPadWidth(), ch = cp.getPadHeight(), rect = this.getPadRect();
            elem.cw = cw;
            elem.ch = ch;
            elem.xlow = rect.x / cw;
            elem.ylow = 1 - (rect.y + rect.height) / ch;
            elem.xup = elem.xlow + rect.width / cw;
            elem.yup = elem.ylow + rect.height / ch;
         }

         if (this.getPadRanges(elem))
            arg.push(elem);
         else
            console.log(`fail to get ranges for pad ${this.pad.fName}`);
      }

      this.painters.forEach(sub => {
         if (isFunc(sub.getWebPadOptions)) {
            if (scan_subpads) sub.getWebPadOptions(arg, cp);
         } else {
            const opt = createWebObjectOptions(sub);
            if (opt)
               elem.primitives.push(opt);
         }
      });

      if (is_top) return toJSON(arg);
   }

   /** @summary returns actual ranges in the pad, which can be applied to the server
     * @private */
   getPadRanges(r) {
      if (!r) return false;

      const main = this.getFramePainter(),
            p = this.svg_this_pad();

      r.ranges = main?.ranges_set ?? false; // indicate that ranges are assigned

      r.ux1 = r.px1 = r.ranges ? main.scale_xmin : 0; // need to initialize for JSON reader
      r.uy1 = r.py1 = r.ranges ? main.scale_ymin : 0;
      r.ux2 = r.px2 = r.ranges ? main.scale_xmax : 0;
      r.uy2 = r.py2 = r.ranges ? main.scale_ymax : 0;
      r.uz1 = r.ranges ? (main.scale_zmin ?? 0) : 0;
      r.uz2 = r.ranges ? (main.scale_zmax ?? 0) : 0;

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
      const func = (log, value, err) => {
         if (!log) return value;
         if (value <= 0) return err;
         value = Math.log10(value);
         if (log > 1) value = value/Math.log10(log);
         return value;
      }, frect = main.getFrameRect();

      r.ux1 = func(main.logx, r.ux1, 0);
      r.ux2 = func(main.logx, r.ux2, 1);

      let k = (r.ux2 - r.ux1)/(frect.width || 10);
      r.px1 = r.ux1 - k*frect.x;
      r.px2 = r.px1 + k*this.getPadWidth();

      r.uy1 = func(main.logy, r.uy1, 0);
      r.uy2 = func(main.logy, r.uy2, 1);

      k = (r.uy2 - r.uy1)/(frect.height || 10);
      r.py1 = r.uy1 - k*frect.y;
      r.py2 = r.py1 + k*this.getPadHeight();

      return true;
   }

   /** @summary Show context menu for specified item
     * @private */
   itemContextMenu(name) {
       const rrr = this.svg_this_pad().node().getBoundingClientRect(),
             evnt = { clientX: rrr.left + 10, clientY: rrr.top + 10 };

       // use timeout to avoid conflict with mouse click and automatic menu close
       if (name === 'pad')
          return postponePromise(() => this.padContextMenu(evnt), 50);

       let selp = null, selkind;

       switch (name) {
          case 'xaxis':
          case 'yaxis':
          case 'zaxis':
             selp = this.getFramePainter();
             selkind = name[0];
             break;
          case 'frame':
             selp = this.getFramePainter();
             break;
          default: {
             const indx = parseInt(name);
             if (Number.isInteger(indx))
                selp = this.painters[indx];
          }
       }

       if (!isFunc(selp?.fillContextMenu)) return;

       return createMenu(evnt, selp).then(menu => {
          const offline_menu = selp.fillContextMenu(menu, selkind);
          if (offline_menu || selp.snapid)
             return selp.fillObjectExecMenu(menu, selkind).then(() => postponePromise(() => menu.show(), 50));
       });
   }

   /** @summary Save pad as image
     * @param {string} kind - format of saved image like 'png', 'svg' or 'jpeg'
     * @param {boolean} full_canvas - does complete canvas (true) or only frame area (false) should be saved
     * @param {string} [filename] - name of the file which should be stored
     * @desc Normally used from context menu
     * @example
     * import { getElementCanvPainter } from 'https://root.cern/js/latest/modules/base/ObjectPainter.mjs';
     * let canvas_painter = getElementCanvPainter('drawing_div_id');
     * canvas_painter.saveAs('png', true, 'canvas.png'); */
   saveAs(kind, full_canvas, filename) {
      if (!filename)
         filename = (this.this_pad_name || (this.iscan ? 'canvas' : 'pad')) + '.' + kind;

      this.produceImage(full_canvas, kind).then(imgdata => {
         if (!imgdata)
            return console.error(`Fail to produce image ${filename}`);

         saveFile(filename, (kind !== 'svg') ? imgdata : 'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(imgdata));
      });
   }

   /** @summary Search active pad
     * @return {Object} pad painter for active pad */
   findActivePad() {
      let active_pp;
      this.forEachPainterInPad(pp => {
         if (pp.is_active_pad && !active_pp)
            active_pp = pp;
      }, 'pads');
      return active_pp;
   }

   /** @summary Prodce image for the pad
     * @return {Promise} with created image */
   async produceImage(full_canvas, file_format) {
      const use_frame = (full_canvas === 'frame'),
            elem = use_frame ? this.getFrameSvg(this.this_pad_name) : (full_canvas ? this.getCanvSvg() : this.svg_this_pad()),
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

      let active_pp = null;
      painter.forEachPainterInPad(pp => {
         if (pp.is_active_pad && !active_pp) {
            active_pp = pp;
            active_pp.drawActiveBorder(null, false);
         }

         if (use_frame) return; // do not make transformations for the frame

         const item = { prnt: pp.svg_this_pad() };
         items.push(item);

         // remove buttons from each subpad
         const btns = pp.getLayerSvg('btns_layer', pp.this_pad_name);
         item.btns_node = btns.node();
         if (item.btns_node) {
            item.btns_prnt = item.btns_node.parentNode;
            item.btns_next = item.btns_node.nextSibling;
            btns.remove();
         }

         const main = pp.getFramePainter();
         if (!isFunc(main?.render3D) || !isFunc(main?.access3dKind)) return;

         const can3d = main.access3dKind();
         if ((can3d !== constants.Embed3D.Overlay) && (can3d !== constants.Embed3D.Embed)) return;

         const sz2 = main.getSizeFor3d(constants.Embed3D.Embed), // get size and position of DOM element as it will be embed

         canvas = main.renderer.domElement;
         main.render3D(0); // WebGL clears buffers, therefore we should render scene and convert immediately
         const dataUrl = canvas.toDataURL('image/png');

         // remove 3D drawings
         if (can3d === constants.Embed3D.Embed) {
            item.foreign = item.prnt.select('.' + sz2.clname);
            item.foreign.remove();
         }

         const svg_frame = main.getFrameSvg();
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

      let width = elem.property('draw_width'), height = elem.property('draw_height');
      if (use_frame) {
         const fp = this.getFramePainter();
         width = fp.getFrameWidth();
         height = fp.getFrameHeight();
      }

      const arg = (file_format === 'pdf')
         ? { node: elem.node(), width, height, reset_tranform: use_frame }
         : compressSVG(`<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">${elem.node().innerHTML}</svg>`);

      return svgToImage(arg, file_format).then(res => {
         // reactivate border
         active_pp?.drawActiveBorder(null, true);

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
            menu.add('header:Menus');

            if (this.iscan)
               menu.add('Canvas', 'pad', this.itemContextMenu);
            else
               menu.add('Pad', 'pad', this.itemContextMenu);

            if (this.getFramePainter())
               menu.add('Frame', 'frame', this.itemContextMenu);

            const main = this.getMainPainter(); // here pad painter method

            if (main) {
               menu.add('X axis', 'xaxis', this.itemContextMenu);
               menu.add('Y axis', 'yaxis', this.itemContextMenu);
               if (isFunc(main.getDimension) && (main.getDimension() > 1))
                  menu.add('Z axis', 'zaxis', this.itemContextMenu);
            }

            if (this.painters?.length) {
               menu.add('separator');
               const shown = [];
               this.painters.forEach((pp, indx) => {
                  const obj = pp?.getObject();
                  if (!obj || (shown.indexOf(obj) >= 0)) return;
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

      for (let i = 0; i < this.painters.length; ++i) {
         const pp = this.painters[i];

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
      if (!settings.ToolBar || this.isBatchMode()) return;

      if (!this._buttons) this._buttons = [];
      // check if there are duplications

      for (let k = 0; k < this._buttons.length; ++k)
         if (this._buttons[k].funcname === funcname) return;

      this._buttons.push({ btn, tooltip, funcname, keyname });

      const iscan = this.iscan || !this.has_canvas;
      if (!iscan && (funcname.indexOf('Pad') !== 0) && (funcname !== 'enlargePad')) {
         const cp = this.getCanvPainter();
         if (cp && (cp !== this)) cp.addPadButton(btn, tooltip, funcname);
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
      this.addPadButton('camera', 'Create PNG', this.iscan ? 'CanvasSnapShot' : 'PadSnapShot', 'Ctrl PrintScreen');

      if (settings.ContextMenu)
         this.addPadButton('question', 'Access context menus', 'PadContextMenus');

      const add_enlarge = !this.iscan && this.has_canvas && this.hasObjectsToDraw();

      if (add_enlarge || this.enlargeMain('verify'))
         this.addPadButton('circle', 'Enlarge canvas', 'enlargePad');

      if (is_online && this.brlayout) {
         this.addPadButton('diamand', 'Toggle Ged', 'ToggleGed');
         this.addPadButton('three_circles', 'Toggle Status', 'ToggleStatus');
      }
   }

   /** @summary Decode pad draw options
     * @private */
   decodeOptions(opt) {
      const pad = this.getObject();
      if (!pad) return;

      const d = new DrawOptions(opt);

      if (!this.options) this.options = {};

      Object.assign(this.options, { GlobalColors: true, LocalColors: false, CreatePalette: 0, IgnorePalette: false, RotateFrame: false, FixFrame: false });

      if (d.check('NOCOLORS') || d.check('NOCOL')) this.options.GlobalColors = this.options.LocalColors = false;
      if (d.check('LCOLORS') || d.check('LCOL')) { this.options.GlobalColors = false; this.options.LocalColors = true; }
      if (d.check('NOPALETTE') || d.check('NOPAL')) this.options.IgnorePalette = true;
      if (d.check('ROTATE')) this.options.RotateFrame = true;
      if (d.check('FIXFRAME')) this.options.FixFrame = true;
      if (d.check('FIXSIZE') && this.iscan) this._fixed_size = true;

      if (d.check('CP', true)) this.options.CreatePalette = d.partAsInt(0, 0);

      if (d.check('NOZOOMX')) this.options.NoZoomX = true;
      if (d.check('NOZOOMY')) this.options.NoZoomY = true;
      if (d.check('GRAYSCALE') && !pad.TestBit(kIsGrayscale))
          pad.InvertBit(kIsGrayscale);

      function forEach(func, p) {
         if (!p) p = pad;
         func(p);
         const arr = p.fPrimitives?.arr || [];
         for (let i = 0; i < arr.length; ++i) {
            if (arr[i]._typename === clTPad)
               forEach(func, arr[i]);
         }
      }

      if (d.check('NOMARGINS')) forEach(p => { p.fLeftMargin = p.fRightMargin = p.fBottomMargin = p.fTopMargin = 0; });
      if (d.check('WHITE')) forEach(p => { p.fFillColor = 0; });
      if (d.check('LOG2X')) forEach(p => { p.fLogx = 2; p.fUxmin = 0; p.fUxmax = 1; p.fX1 = 0; p.fX2 = 1; });
      if (d.check('LOGX')) forEach(p => { p.fLogx = 1; p.fUxmin = 0; p.fUxmax = 1; p.fX1 = 0; p.fX2 = 1; });
      if (d.check('LOG2Y')) forEach(p => { p.fLogy = 2; p.fUymin = 0; p.fUymax = 1; p.fY1 = 0; p.fY2 = 1; });
      if (d.check('LOGY')) forEach(p => { p.fLogy = 1; p.fUymin = 0; p.fUymax = 1; p.fY1 = 0; p.fY2 = 1; });
      if (d.check('LOG2Z')) forEach(p => { p.fLogz = 2; });
      if (d.check('LOGZ')) forEach(p => { p.fLogz = 1; });
      if (d.check('LOGV')) forEach(p => { p.fLogv = 1; });
      if (d.check('LOG2')) forEach(p => { p.fLogx = p.fLogy = p.fLogz = 2; });
      if (d.check('LOG')) forEach(p => { p.fLogx = p.fLogy = p.fLogz = 1; });
      if (d.check('LNX')) forEach(p => { p.fLogx = 3; p.fUxmin = 0; p.fUxmax = 1; p.fX1 = 0; p.fX2 = 1; });
      if (d.check('LNY')) forEach(p => { p.fLogy = 3; p.fUymin = 0; p.fUymax = 1; p.fY1 = 0; p.fY2 = 1; });
      if (d.check('LN')) forEach(p => { p.fLogx = p.fLogy = p.fLogz = 3; });
      if (d.check('GRIDX')) forEach(p => { p.fGridx = 1; });
      if (d.check('GRIDY')) forEach(p => { p.fGridy = 1; });
      if (d.check('GRID')) forEach(p => { p.fGridx = p.fGridy = 1; });
      if (d.check('TICKX')) forEach(p => { p.fTickx = 1; });
      if (d.check('TICKY')) forEach(p => { p.fTicky = 1; });
      if (d.check('TICKZ')) forEach(p => { p.fTickz = 1; });
      if (d.check('TICK')) forEach(p => { p.fTickx = p.fTicky = 1; });
      if (d.check('OTX')) forEach(p => { p.$OTX = true; });
      if (d.check('OTY')) forEach(p => { p.$OTY = true; });
      if (d.check('CTX')) forEach(p => { p.$CTX = true; });
      if (d.check('CTY')) forEach(p => { p.$CTY = true; });
      if (d.check('RX')) forEach(p => { p.$RX = true; });
      if (d.check('RY')) forEach(p => { p.$RY = true; });

      this.storeDrawOpt(opt);
   }

   /** @summary draw TPad object */
   static async draw(dom, pad, opt) {
      const painter = new TPadPainter(dom, pad, false);
      painter.decodeOptions(opt);

      if (painter.getCanvSvg().empty()) {
         // one can draw pad without canvas
         painter.has_canvas = false;
         painter.this_pad_name = '';
         painter.setTopPainter();
      } else {
         // pad painter will be registered in the canvas painters list
         painter.addToPadPrimitives(painter.pad_name);
      }

      if (pad?.$disable_drawing)
         painter.pad_draw_disabled = true;

      painter.createPadSvg();

      if (painter.matchObjectType(clTPad) && (!painter.has_canvas || painter.hasObjectsToDraw()))
         painter.addPadButtons();

      // we select current pad, where all drawing is performed
      const prev_name = painter.has_canvas ? painter.selectCurrentPad(painter.this_pad_name) : undefined;

      // set active pad
      selectActivePad({ pp: painter, active: true });

      // flag used to prevent immediate pad redraw during first draw
      return painter.drawPrimitives().then(() => {
         painter.showPadButtons();
         painter.addPadInteractive();
         // we restore previous pad name
         painter.selectCurrentPad(prev_name);
         return painter;
      });
   }

} // class TPadPainter

export { TPadPainter, PadButtonsHandler, clTButton, kIsGrayscale, createWebObjectOptions };
