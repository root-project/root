import { pointer as d3_pointer } from '../d3.mjs';
import { settings, constants, internals, isNodeJs, isBatchMode, getPromise, BIT,
         getKindForType, clTObjString, clTAxis, isObject, isFunc, isStr, getDocument, urlClassPrefix } from '../core.mjs';
import { isPlainText, producePlainText, produceLatex, produceMathjax, typesetMathjax, approximateLabelWidth } from './latex.mjs';
import { getElementRect, BasePainter, makeTranslate } from './BasePainter.mjs';
import { TAttMarkerHandler } from './TAttMarkerHandler.mjs';
import { TAttFillHandler } from './TAttFillHandler.mjs';
import { TAttLineHandler } from './TAttLineHandler.mjs';
import { TAttTextHandler } from './TAttTextHandler.mjs';
import { FontHandler } from './FontHandler.mjs';
import { getRootColors } from './colors.mjs';


/** @summary returns true if pad painter @private */
function isPadPainter(p) {
   return isFunc(p?.getRootPad) && isFunc(p?.forEachPainterInPad);
}

/** @summary returns canvas painter from DOM element @private */
function getDomCanvasPainter(dom) {
   const elem = dom?.select('.root_canvas');
   return !elem || elem.empty() ? null : elem.property('pad_painter');
}

/**
 * @summary Painter class for ROOT objects
 *
 */

class ObjectPainter extends BasePainter {

   #draw_object;     // drawn object
   #draw_g;          // <g> element for object drawing
   #pad_painter_ref; // reference of pad painter
   #main_painter;    // WeakRef to main painter in the pad
   #primary_ref;     // reference of primary painter - if any
   #snapid;          // assigned online identifier
   #is_primary;      // if primary painter
   #secondary_id;    // id of this painter in relation to primary painter
   #options;         // current options object
   #options_store;   // stored draw options used to check changes
   #user_tooltip_handler; // configured user tooltip handler
   #user_tooltip_timeout; // timeout configured with tooltip handler
   #user_toottip_handle; // timeout handle processing user tooltip
   #user_context_menu; // function for user context menu
   #special_draw_area; // current special draw area like projection
   #root_colors;     // custom colors list
   #fillatt;         // fill attribute
   #lineatt;         // line attribute
   #markeratt;       // marker attribute
   #textatt;         // text attribute

   /** @summary constructor
     * @param {object|string} dom - dom element or identifier or pad painter
     * @param {object} obj - object to draw
     * @param {string} [opt] - object draw options */
   constructor(dom, obj, opt) {
      const pp = isPadPainter(dom) ? dom : null;

      super(pp?.getDom() ?? dom);

      this.setPadPainter(pp);

      this.#draw_g = undefined; // container for all drawn objects
      this.assignObject(obj);
      if (isStr(opt))
         this.#options = { original: opt };
   }

   /** @summary Assign object to the painter
     * @protected */
   assignObject(obj) { this.#draw_object = isObject(obj) ? obj : null; }

   /** @summary Returns drawn object */
   getObject() { return this.#draw_object; }

   /** @summary Assign new pad painter
     * @protected */
   setPadPainter(pp) { this.#pad_painter_ref = pp ? new WeakRef(pp) : undefined; }

   /** @summary returns pad painter where object is drawn
     * @protected */
   getPadPainter() { return this.#pad_painter_ref?.deref(); }

   /** @summary returns canvas painter
     * @protected */
   getCanvPainter() {
      let pp = this.getPadPainter();
      while (pp && !pp.isCanvas())
         pp = pp.getPadPainter();
      return pp;
   }

   /** @summary Indicates that drawing runs in batch mode
     * @private */
   isBatchMode() { return isBatchMode() ? true : (this.getCanvPainter()?.isBatchMode() ?? false); }

   /** @summary Assign snapid to the painter
    * @desc Identifier used to communicate with server side and identifies object on the server
    * @private */
   assignSnapId(id) { this.#snapid = id; }

   /** @summary Provides identifier on server for requested sub-element */
   getSnapId(subelem) { return !this.#snapid ? '' : (this.#snapid + (subelem ? '#' + subelem : '')); }

   /** @summary Returns true if snapid was assigned */
   hasSnapId() { return this.#snapid !== undefined; }

   /** @summary Generic method to cleanup painter.
     * @desc Remove object drawing and (in case of main painter) also main HTML components
     * @protected */
   cleanup() {
      this.removeG();

      let keep_origin = true;

      if (this.isMainPainter()) {
         const pp = this.getPadPainter();
         if (!pp || pp.isCanvas('auto'))
            keep_origin = false;
      }

      // cleanup all existing references
      this.#pad_painter_ref = undefined;
      this.#main_painter = null;
      this.#draw_object = null;
      this.#snapid = undefined;
      this.#is_primary = undefined;
      this.#primary_ref = undefined;
      this.#secondary_id = undefined;

      // remove attributes objects (if any)
      this.deleteAttr();
      this.#root_colors = undefined;
      this.#options = undefined;
      this.#options_store = undefined;

      // remove extra fields from v7 painters
      delete this.rstyle;
      delete this.csstype;

      super.cleanup(keep_origin);
   }

   /** @summary Returns drawn object name */
   getObjectName() { return this.getObject()?.fName ?? ''; }

   /** @summary Returns drawn object class name */
   getClassName() { return this.getObject()?._typename ?? ''; }

   /** @summary Checks if drawn object matches with provided typename
     * @param {string|object} arg - typename (or object with _typename member)
     * @protected */
   matchObjectType(arg) {
      const clname = this.getClassName();
      if (!arg || !clname)
         return false;
      if (isStr(arg))
         return arg === clname;
      if (isStr(arg._typename))
         return arg._typename === clname;
      return Boolean(clname.match(arg));
   }

   /** @summary Change item name
     * @desc When available, used for svg:title property
     * @private */
   setItemName(name, opt, hpainter) {
      super.setItemName(name, opt, hpainter);
      if (this._no_default_title || !name)
         return;
      const can = this.getCanvSvg();
      if (!can.empty()) can.select('title').text(name);
                   else this.selectDom().attr('title', name);
      const cp = this.getCanvPainter();
      if (cp && ((cp === this) || (this.isMainPainter() && (cp === this.getPadPainter()))))
         cp.drawItemNameOnCanvas(name);
   }

   /** @summary Create options and copy new args
     * @return options
     * @private */
   setOptions(new_options, as_is) {
      if (as_is)
         this.#options = new_options;
      else {
         if (!this.#options)
            this.#options = {};
         Object.assign(this.#options, new_options);
      }
      return this.#options;
   }

   /** @summary Return actual options */
   getOptions(as_is) {
      if (!as_is && !this.#options)
         this.#options = {};
      return this.#options;
   }

   /** @summary Emulate old options property */
   get options() { return this.getOptions(); };

   /** @summary Store actual options together with original string
     * @private */
   storeDrawOpt(original) {
      if (!this.#options)
         return;
      if (!original)
         original = '';
      const pp = original.indexOf(';;');
      if (pp >= 0)
         original = original.slice(0, pp);
      this.#options.original = original;
      this.#options_store = Object.assign({}, this.#options);
   }

   /** @summary Return dom argument for object drawing
    * @desc Can be used to draw other objects on same pad / same dom element
    * @protected */
   getDrawDom() { return this.getPadPainter() || this.getDom(); }

   /** @summary Return actual draw options as string
     * @param ignore_pad - do not include pad settings into histogram draw options
     * @desc if options are not modified - returns original string which was specified for object draw */
   getDrawOpt(ignore_pad) {
      if (!this.#options)
         return '';

      if (isFunc(this.#options.asString)) {
         let changed = false;
         const pp = this.getPadPainter();
         if (!this.#options_store || pp?.options._interactively_changed)
            changed = true;
         else {
            for (const k in this.#options_store) {
               if (this.#options[k] !== this.#options_store[k]) {
                  if ((k[0] !== '_') && (k[0] !== '$') && (k[0].toLowerCase() !== k[0]))
                     changed = true;
               }
            }
         }

         if (changed && isFunc(this.#options.asString))
            return this.#options.asString(this.isMainPainter(), ignore_pad ? null : pp?.getRootPad());
      }

      return this.#options.original || ''; // nothing better, return original draw option
   }

   /** @summary Returns array with supported draw options as configured in draw.mjs
     * @desc works via pad painter and only when module was loaded */
   getSupportedDrawOptions() {
      const pp = this.getPadPainter(),
            cl = this.getClassName();

      if (!cl || !isFunc(pp?.getObjectDrawSettings))
         return [];

      return pp.getObjectDrawSettings(getKindForType(cl), 'nosame')?.opts;
   }

   /** @summary Central place to update objects drawing
     * @param {object} obj - new version of object, values will be updated in original object
     * @param {string} [opt] - when specified, new draw options
     * @return {boolean|Promise} for object redraw
     * @desc Two actions typically done by redraw - update object content via {@link ObjectPainter#updateObject} and
      * then redraw correspondent pad via {@link ObjectPainter#redrawPad}. If possible one should redefine
      * only updateObject function and keep this function unchanged. But for some special painters this function is the
      * only way to control how object can be update while requested from the server
      * @protected */
   redrawObject(obj, opt) {
      if (!this.updateObject(obj, opt))
         return false;
      const doc = this.isBatchMode() ? null : getDocument(),
            current = doc?.body.style.cursor;
      if (doc)
         doc.body.style.cursor = 'wait';
      return this.redrawPad().then(res => {
         if (doc)
            doc.body.style.cursor = current;
         return res;
      });
   }

   /** @summary Generic method to update object content.
     * @desc Default implementation just copies first-level members to current object
     * @param {object} obj - object with new data
     * @param {string} [opt] - option which will be used for redrawing
     * @protected */
   updateObject(obj /* , opt */) {
      if (!this.matchObjectType(obj))
         return false;
      Object.assign(this.getObject(), obj);
      return true;
   }

   /** @summary Returns string with object hint
     * @desc It is either item name or object name or class name.
     * Such string typically used as object tooltip.
     * If result string larger than 20 symbols, it will be shorten. */
   getObjectHint() {
      const iname = this.getItemName();
      if (iname)
         return (iname.length > 20) ? '...' + iname.slice(iname.length - 17) : iname;
      return this.getObjectName() || this.getClassName() || '';
   }

   /** @summary Set colors list
    * @protected */
   setColors(lst) { this.#root_colors = lst; }

   /** @summary Return colors list
    * @protected */
   getColors(force) {
      if (!this.#root_colors && force)
         this.setColors(this.getCanvPainter()?.getColors() || getRootColors());
      return this.#root_colors;
   }

   /** @summary returns color from current list of colors
     * @desc First checks canvas painter and then just access global list of colors
     * @param {number} indx - color index
     * @return {string} with SVG color name or rgb()
     * @protected */
   getColor(indx) { return this.getColors(true)[indx]; }

   /** @summary Add color to list of colors
     * @desc Returned color index can be used as color number in all other draw functions
     * @return {number} new color index
     * @protected */
   addColor(color) {
      const lst = this.getColors(true),
            indx = lst.indexOf(color);
      if (indx >= 0)
         return indx;
      lst.push(color);
      return lst.length - 1;
   }

   /** @summary returns tooltip allowed flag
     * @desc If available, checks in canvas painter
     * @private */
   isTooltipAllowed() {
      const src = this.getCanvPainter() || this;
      return src.tooltip_allowed;
   }

   /** @summary change tooltip allowed flag
     * @param {boolean|string} [on = true] set tooltip allowed state or 'toggle'
     * @private */
   setTooltipAllowed(on = true) {
      const src = this.getCanvPainter() || this;
      src.tooltip_allowed = (on === 'toggle') ? !src.tooltip_allowed : on;
   }

   /** @summary Checks if draw elements were resized and drawing should be updated.
     * @desc Redirects to {@link TPadPainter#checkCanvasResize}
     * @private */
   checkResize(arg) {
      return this.getCanvPainter()?.checkCanvasResize(arg);
   }

   /** @summary removes <g> element with object drawing
     * @desc generic method to delete all graphical elements, associated with the painter
     * @protected */
   removeG() {
      this.#draw_g?.remove();
      this.#draw_g = undefined;
   }

   /** @summary Returns created <g> element used for object drawing
     * @desc Element should be created by {@link ObjectPainter#createG}
     * @protected */
   getG() { return this.#draw_g; }

   /** @summary introduce property for backward compatibility */
   get draw_g() { return this.#draw_g; }

   /** @summary Assign G element used for object drawing
     * @protected */
   setG(g) { this.#draw_g = g; return g; }

   /** @summary Append svg::path to G
     * @protected */
   appendPath(d) { return this.#draw_g.append('svg:path').attr('d', d); }

   /** @summary (re)creates svg:g element for object drawings
     * @desc either one attach svg:g to pad primitives (default)
     * or svg:g element created in specified frame layer ('main_layer' will be used when true specified)
     * @param {boolean|string} [frame_layer] - when specified, <g> element will be created inside frame layer, otherwise in the pad
     * @protected */
   createG(frame_layer, use_a = false) {
      let layer;

      const pp = this.getPadPainter();

      if (frame_layer === 'frame2d') {
         const fp = this.getFramePainter();
         frame_layer = fp && !fp.mode3d;
      }

      if (frame_layer) {
         const frame = pp.getFrameSvg();
         if (frame.empty()) {
            console.error('Not found frame to create g element inside');
            return frame;
         }
         if (!isStr(frame_layer)) frame_layer = 'main_layer';
         layer = frame.selectChild('.' + frame_layer);
      } else
         layer = pp.getLayerSvg('primitives_layer');

      if (this.#draw_g && this.#draw_g.node().parentNode !== layer.node()) {
         console.log('g element changes its layer!!');
         this.removeG();
      }

      if (this.#draw_g) {
         // clear all elements, keep g element on its place
         this.#draw_g.selectAll('*').remove();
      } else {
         this.#draw_g = layer.append(use_a ? 'svg:a' : 'svg:g');

         if (!frame_layer)
            layer.selectChildren('.most_upper_primitives').raise();
      }

      // set attributes for debugging, both should be there for opt out them later
      const clname = this.getClassName(), objname = this.getObjectName();
      if (objname || clname) {
         this.#draw_g.attr('objname', (objname || 'name').replace(/[^\w]/g, '_'))
                     .attr('objtype', (clname || 'type').replace(/[^\w]/g, '_'));
      }

      this.#draw_g.property('in_frame', Boolean(frame_layer)); // indicates coordinate system

      return this.#draw_g;
   }

   /** @summary Bring draw element to the front */
   bringToFront(check_online) {
      if (!this.#draw_g)
         return;
      const prnt = this.#draw_g.node().parentNode;
      prnt?.appendChild(this.#draw_g.node());

      if (check_online && this.getSnapId()) {
         const pp = this.getPadPainter();
         if (pp?.getSnapId())
            this.getCanvPainter()?.sendWebsocket('POPOBJ:'+JSON.stringify([pp.getSnapId(), this.getSnapId()]));
      }
   }

   /** @summary Assign is_primary flag
     * @private */
   setPrimary(flag = true) { this.#is_primary = flag; }

   /** @summary Return is_primary flag
     * @private */
   isPrimary() { return this.#is_primary; }

   /** @summary Assign secondary id
     * @private */
   setSecondaryId(primary, name) {
      primary.setPrimary(true); // mark as primary, used later
      this.#primary_ref = new WeakRef(primary);
      this.#secondary_id = name;
   }

   /** @summary Returns secondary id
     * @private */
   getSecondaryId() { return this.#secondary_id; }

   /** @summary Check if this is secondary painter
     * @desc if primary painter provided - check if this really main for this
     * @private */
   isSecondary(primary) {
      if (!this.#primary_ref)
         return false;
      return !isObject(primary) ? true : this.#primary_ref.deref() === primary;
   }

   /** @summary Return primary object
     * @private */
   getPrimary() { return this.#primary_ref?.deref(); }

   /** @summary Canvas main svg element
     * @return {object} d3 selection with canvas svg
     * @protected */
   getCanvSvg() { return this.selectDom().select('.root_canvas'); }

   /** @summary Method selects current pad name
     * @param {string} [new_name] - when specified, new current pad name will be configured
     * @return {string} previous selected pad or actual pad when new_name not specified
     * @private
     * @deprecated to be removed in v8 */
   selectCurrentPad() {
      console.warn('selectCurrentPad is deprecated, will be removed in v8');
      return '';
   }

   /** @summary Return functor, which can convert x and y coordinates into pixels, used for drawing in the pad
     * @desc X and Y coordinates can be converted by calling func.x(x) and func.y(y)
     * Only can be used for painting in the pad, means CreateG() should be called without arguments
     * @param {boolean} isndc - if NDC coordinates will be used
     * @param {boolean} [noround] - if set, return coordinates will not be rounded
     * @param {boolean} [use_frame_coordinates] - use frame coordinates even when drawing on the pad
     * @protected */
   getAxisToSvgFunc(isndc, nornd, use_frame_coordinates) {
      const func = { isndc, nornd },
            use_frame = this.getG()?.property('in_frame');
      if (use_frame || (use_frame_coordinates && !isndc))
         func.fp = this.getFramePainter();
      if (func.fp?.grx && func.fp?.gry) {
         func.x0 = (use_frame_coordinates && !isndc) ? func.fp.getFrameX() : 0;
         func.y0 = (use_frame_coordinates && !isndc) ? func.fp.getFrameY() : 0;
         if (nornd) {
            func.x = function(x) { return this.x0 + this.fp.grx(x); };
            func.y = function(y) { return this.y0 + this.fp.gry(y); };
         } else {
            func.x = function(x) { return this.x0 + Math.round(this.fp.grx(x)); };
            func.y = function(y) { return this.y0 + Math.round(this.fp.gry(y)); };
         }
      } else if (!use_frame) {
         const pp = this.getPadPainter();
         func.pad = isndc ? null : pp?.getRootPad(true); // need for NDC conversion
         func.padw = pp?.getPadWidth() ?? 10;
         func.x = function(value) {
            if (this.pad) {
               if (this.pad.fLogx)
                  value = (value > 0) ? Math.log10(value) : this.pad.fUxmin;
               value = (value - this.pad.fX1) / (this.pad.fX2 - this.pad.fX1);
            }
            value *= this.padw;
            return this.nornd ? value : Math.round(value);
         };
         func.padh = pp?.getPadHeight() ?? 10;
         func.y = function(value) {
            if (this.pad) {
               if (this.pad.fLogy)
                  value = (value > 0) ? Math.log10(value) : this.pad.fUymin;
               value = (value - this.pad.fY1) / (this.pad.fY2 - this.pad.fY1);
            }
            value = (1 - value) * this.padh;
            return this.nornd ? value : Math.round(value);
         };
      } else {
         console.error(`Problem to create functor for ${this.getClassName()}`);
         func.x = () => 0;
         func.y = () => 0;
      }
      return func;
   }

   /** @summary Converts x or y coordinate into pad SVG coordinates.
     * @desc Only can be used for painting in the pad, means CreateG() should be called without arguments
     * @param {string} axis - name like 'x' or 'y'
     * @param {number} value - axis value to convert.
     * @param {boolean} ndc - is value in NDC coordinates
     * @param {boolean} [noround] - skip rounding
     * @return {number} value of requested coordinates
     * @protected */
   axisToSvg(axis, value, ndc, noround) {
      const func = this.getAxisToSvgFunc(ndc, noround);
      return func[axis](value);
   }

   /** @summary Converts pad SVG x or y coordinates into axis values.
     * @desc Reverse transformation for {@link ObjectPainter#axisToSvg}
     * @param {string} axis - name like 'x' or 'y'
     * @param {number} coord - graphics coordinate.
     * @param {boolean} ndc - kind of return value
     * @return {number} value of requested coordinates
     * @protected */
   svgToAxis(axis, coord, ndc) {
      const use_frame = this.getG()?.property('in_frame');

      if (use_frame)
         return this.getFramePainter()?.revertAxis(axis, coord) ?? 0;

      const pp = this.getPadPainter(),
            pad = (ndc || !pp) ? null : pp.getRootPad(true);
      let value = !pp ? 0 : ((axis === 'y') ? (1 - coord / pp.getPadHeight()) : coord / pp.getPadWidth());

      if (pad) {
         if (axis === 'y') {
            value = pad.fY1 + value * (pad.fY2 - pad.fY1);
            if (pad.fLogy) value = Math.pow(10, value);
         } else {
            value = pad.fX1 + value * (pad.fX2 - pad.fX1);
            if (pad.fLogx) value = Math.pow(10, value);
         }
      }

      return value;
   }

   /** @summary Returns frame painter for current pad
     * @desc Pad has direct reference on frame if any
     * @protected */
   getFramePainter() {
      return this.getPadPainter()?.getFramePainter();
   }

   /** @summary Returns painter for main object on the pad.
     * @desc Typically it is first histogram drawn on the pad and which draws frame axes
     * But it also can be special use-case as TASImage or TGraphPolargram
     * @param {boolean} [not_store] - if true, prevent temporary storage of main painter reference
     * @protected */
   getMainPainter(not_store) {
      let res = this.#main_painter?.deref();
      if (!res) {
         const pp = this.getPadPainter();
         res = pp ? pp.getMainPainter() : this.getTopPainter();
         this.#main_painter = not_store || !res ? null : new WeakRef(res);
      }
      return res || null;
   }

   /** @summary Returns true if this is main painter
     * @protected */
   isMainPainter() { return this === this.getMainPainter(); }

   /** @summary Assign this as main painter on the pad
     * @desc Main painter typically responsible for axes drawing
     * Should not be used by pad/canvas painters, but rather by objects which are drawing axis
     * @protected */
   setAsMainPainter(force) {
      const pp = this.getPadPainter();
      if (!pp)
         this.setTopPainter(); // fallback on BasePainter method
      else
         pp.setMainPainter(this, force);
   }

   /** @summary Add painter to pad list of painters
     * @desc Normally called from {@link ensureTCanvas} function when new painter is created
     * @protected */
   addToPadPrimitives(pad_painter) {
      if (this.#pad_painter_ref)
         pad_painter = this.#pad_painter_ref.deref();
      else {
         if (!pad_painter)
            pad_painter = getDomCanvasPainter(this.selectDom()); // try to detect in DOM
         if (pad_painter)
            this.#pad_painter_ref = new WeakRef(pad_painter);
      }

      if (!pad_painter || (pad_painter === this))
         return null;

      return pad_painter.addToPrimitives(this);
   }

   /** @summary Creates marker attributes object
     * @desc Can be used to produce markers in painter.
     * See {@link TAttMarkerHandler} for more info.
     * Instance assigned as this.markeratt data member, recognized by GED editor
     * @param {object} args - either TAttMarker or see arguments of {@link TAttMarkerHandler}
     * @return {object} created handler
     * @protected */
   createAttMarker(args) {
      if (args === undefined)
         args = { attr: this.getObject() };
      else if (!isObject(args))
         args = { std: true };
      else if (args.fMarkerColor !== undefined && args.fMarkerStyle !== undefined && args.fMarkerSize !== undefined)
         args = { attr: args, std: false };

      if (args.std === undefined)
         args.std = true;
      if (args.painter === undefined)
         args.painter = this;

      let handler = args.std ? this.#markeratt : null;

      if (!handler)
         handler = new TAttMarkerHandler(args);
      else if (!handler.changed || args.force)
         handler.setArgs(args);

      if (args.std)
         this.#markeratt = handler;
      return handler;
   }

   /** @summary Creates line attributes object.
     * @desc Can be used to produce lines in painter.
     * See {@link TAttLineHandler} for more info.
     * Instance assigned as this.lineatt data member, recognized by GED editor
     * @param {object} args - either TAttLine or see constructor arguments of {@link TAttLineHandler}
     * @protected */
   createAttLine(args) {
      if (args === undefined)
         args = { attr: this.getObject() };
      else if (!isObject(args))
         args = { std: true };
      else if (args.fLineColor !== undefined && args.fLineStyle !== undefined && args.fLineWidth !== undefined)
         args = { attr: args, std: false };

      if (args.std === undefined)
         args.std = true;
      if (args.painter === undefined)
         args.painter = this;

      let handler = args.std ? this.#lineatt : null;

      if (!handler)
         handler = new TAttLineHandler(args);
      else if (!handler.changed || args.force)
         handler.setArgs(args);

      if (args.std)
         this.#lineatt = handler;
      return handler;
   }

   /** @summary Creates text attributes object.
     * @param {object} args - either TAttText or see constructor arguments of {@link TAttTextHandler}
     * @protected */
   createAttText(args) {
      if (args === undefined)
         args = { attr: this.getObject() };
      else if (!isObject(args))
         args = { std: true };
      else if (args.fTextFont !== undefined && args.fTextSize !== undefined && args.fTextColor !== undefined)
         args = { attr: args, std: false };

      if (args.std === undefined)
         args.std = true;
      if (args.painter === undefined)
         args.painter = this;

      let handler = args.std ? this.#textatt : null;

      if (!handler)
         handler = new TAttTextHandler(args);
      else if (!handler.changed || args.force)
         handler.setArgs(args);

      if (args.std)
         this.#textatt = handler;
      return handler;
   }

   /** @summary Creates fill attributes object.
     * @desc Method dedicated to create fill attributes, bound to canvas SVG
     * otherwise newly created patters will not be usable in the canvas
     * See {@link TAttFillHandler} for more info.
     * Instance assigned as this.fillatt data member, recognized by GED editors
     * @param {object} [args] - for special cases one can specify TAttFill as args or number of parameters
     * @param {boolean} [args.std = true] - this is standard fill attribute for object and should be used as this.fillatt
     * @param {object} [args.attr = null] - object, derived from TAttFill
     * @param {number} [args.pattern = undefined] - integer index of fill pattern
     * @param {number} [args.color = undefined] - integer index of fill color
     * @param {string} [args.color_as_svg = undefined] - color will be specified as SVG string, not as index from color palette
     * @param {number} [args.kind = undefined] - some special kind which is handled differently from normal patterns
     * @return created handle
     * @protected */
   createAttFill(args) {
      if (args === undefined)
         args = { attr: this.getObject() };
      else if (!isObject(args))
         args = { std: true };
      else if (args._typename && args.fFillColor !== undefined && args.fFillStyle !== undefined)
         args = { attr: args, std: false };

      if (args.std === undefined)
         args.std = true;
      if (args.painter === undefined)
         args.painter = this;

      let handler = args.std ? this.#fillatt : null;

      if (!args.svg)
         args.svg = this.getCanvSvg();

      if (!handler)
         handler = new TAttFillHandler(args);
      else if (!handler.changed || args.force)
         handler.setArgs(args);

      if (args.std)
         this.#fillatt = handler;
      return handler;
   }

   get fillatt() { return this.#fillatt; }
   get lineatt() { return this.#lineatt; }
   get markeratt() { return this.#markeratt; }
   get textatt() { return this.#textatt; }

   /** @summary call function for each painter in the pad
     * @desc Iterate over all known painters
     * @private */
   forEachPainter(userfunc, kind) {
      // iterate over all painters from pad list
      let pp = this.getPadPainter(), top = null;
      if (!pp) {
         top = this.getTopPainter();
         if (isPadPainter(top))
            pp = top;
      }
      if (pp)
         pp.forEachPainterInPad(userfunc, kind);
      else if (top && (kind !== 'pads'))
         userfunc(top);
   }

   /** @summary indicate that redraw was invoked via interactive action (like context menu or zooming)
     * @desc Use to catch such action by GED and by server-side
     * @return {Promise} when completed
     * @private */
   async interactiveRedraw(arg, info, subelem) {
      let reason, res;
      if (isStr(info) && info.indexOf('exec:'))
         reason = info;

      if (arg === 'pad')
         res = this.redrawPad(reason);
      else if (arg !== false)
         res = this.redraw(reason);

      return getPromise(res).then(() => {
         if (arg === 'attribute')
            return this.getPadPainter()?.redrawLegend();
      }).then(() => {
         // inform GED that something changes
         const canp = this.getCanvPainter();

         if (isFunc(canp?.producePadEvent))
            canp.producePadEvent('redraw', this.getPadPainter(), this, null, subelem);

         // inform server that draw options changes
         if (isFunc(canp?.processChanges))
            canp.processChanges(info, this, subelem);

         return this;
      });
   }

   /** @summary Redraw all objects in the current pad
     * @param {string} [reason] - like 'resize' or 'zoom'
     * @return {Promise} when pad redraw completed
     * @protected */
   async redrawPad(reason) {
      return this.getPadPainter()?.redrawPad(reason) ?? false;
   }

   /** @summary execute selected menu command, either locally or remotely
     * @private */
   executeMenuCommand(method) {
      if (method.fName === 'Inspect')
         // primitive inspector, keep it here
         return this.showInspector();

      return false;
   }

   /** @summary Invoke method for object via WebCanvas functionality
     * @desc Requires that painter marked with object identifier (this.#snapid) or identifier provided as second argument
     * Canvas painter should exists and in non-readonly mode
     * Execution string can look like 'Print()'.
     * Many methods call can be chained with 'Print();;Update();;Clear()'
     * @private */
   submitCanvExec(exec, snapid) {
      if (!exec || !isStr(exec))
         return;

      const canp = this.getCanvPainter();
      if (isFunc(canp?.submitExec))
         canp.submitExec(this, exec, snapid);
   }

   /** @summary remove all created draw attributes
     * @protected */
   deleteAttr(name) {
      if (!name || name === 'line')
         this.#lineatt = undefined;
      if (!name || name === 'fill')
         this.#fillatt = undefined;
      if (!name || name === 'marker')
         this.#markeratt = undefined;
      if (!name || name === 'text')
         this.#textatt = undefined;
   }

   /** @summary Show object in inspector for provided object
     * @protected */
   showInspector(/* opt */) {
      return false;
   }

   /** @summary Fill context menu for the object
     * @private */
   fillContextMenu(menu) {
      const cl = this.getClassName(),
            name = this.getObjectName(),
            p = cl.lastIndexOf('::'),
            cl0 = (p > 0) ? cl.slice(p+2) : cl,
            hdr = (cl0 && name) ? `${cl0}:${name}` : (cl0 || name || 'object'),
            url = cl ? `${urlClassPrefix}${cl.replaceAll('::', '_1_1')}.html` : '';

      menu.header(hdr, url);

      const size0 = menu.size();

      if (isFunc(this.fillContextMenuItems))
         this.fillContextMenuItems(menu);

      if ((menu.size() > size0) && this.showInspector('check'))
         menu.add('Inspect', this.showInspector);

      menu.addAttributesMenu(this);

      return menu.size() > size0;
   }

   /** @summary shows objects status
     * @desc Either used canvas painter method or globally assigned
     * When no parameters are specified, just basic object properties are shown
     * @private */
   showObjectStatus(name, title, info, info2) {
      let cp = this.getCanvPainter();

      if (!isFunc(cp?.showCanvasStatus))
         cp = null;

      if (!cp && !isFunc(internals.showStatus))
         return false;

      if (this.enlargeMain('state') === 'on')
         return false;

      if ((name === undefined) && (title === undefined)) {
         const obj = this.getObject();
         if (!obj) return;
         name = this.getItemName() || obj.fName;
         title = obj.fTitle || obj._typename;
         info = obj._typename;
      }

      if (cp)
         cp.showCanvasStatus(name, title, info, info2);
      else
         internals.showStatus(name, title, info, info2);
   }

   /** @summary Redraw object
     * @desc Basic method, should be reimplemented in all derived objects
     * for the case when drawing should be repeated
     * @abstract
     * @protected */
   redraw(/* reason */) {}

   /** @summary Start text drawing
     * @desc required before any text can be drawn
     * @param {number} font_face - font id as used in ROOT font attributes
     * @param {number} font_size - font size as used in ROOT font attributes
     * @param {object} [draw_g] - element where text drawn, by default using main object <g> element
     * @param {number} [max_font_size] - maximal font size, used when text can be scaled
     * @protected */
   startTextDrawing(font_face, font_size, draw_g, max_font_size, can_async) {
      if (!draw_g)
         draw_g = this.getG();
      if (!draw_g || draw_g.empty())
         return false;

      const font = (font_size === 'font') ? font_face : new FontHandler(font_face, font_size);
      if (can_async && font.needLoad())
         return font;

      font.setPainter(this); // may be required when custom font is used

      draw_g.call(font.func);

      draw_g.property('draw_text_completed', false) // indicate that draw operations submitted
            .property('all_args', []) // array of all submitted args, makes easier to analyze them
            .property('text_font', font)
            .property('text_factor', 0)
            .property('max_text_width', 0) // keep maximal text width, use it later
            .property('max_font_size', max_font_size)
            .property('_fast_drawing', this.getPadPainter()?.isFastDrawing() ?? false);

      if (draw_g.property('_fast_drawing'))
         draw_g.property('_font_too_small', (max_font_size && (max_font_size < 5)) || (font.size < 4));

      return true;
   }

   /** @summary Start async text drawing
    * @return {Promise} for loading of font if necessary
    * @private */
   async startTextDrawingAsync(font_face, font_size, draw_g, max_font_size) {
      const font = this.startTextDrawing(font_face, font_size, draw_g, max_font_size, true);
      if ((font === true) || (font === false))
         return font;
      return font.load().then(res => {
         if (!res)
            return false;

         return this.startTextDrawing(font, 'font', draw_g, max_font_size);
      });
   }

   /** @summary Apply scaling factor to all drawn text in the <g> element
     * @desc Can be applied at any time before finishTextDrawing is called - even in the postprocess callbacks of text draw
     * @param {number} factor - scaling factor
     * @param {object} [draw_g] - drawing element for the text
     * @protected */
   scaleTextDrawing(factor, draw_g) {
      if (!draw_g)
         draw_g = this.getG();
      if (!draw_g || draw_g.empty())
         return;
      if (factor && (factor > draw_g.property('text_factor')))
         draw_g.property('text_factor', factor);
   }

   /** @summary Analyze if all text draw operations are completed
     * @private */
   #checkAllTextDrawing(draw_g, resolveFunc, try_optimize) {
      const all_args = draw_g.property('all_args') || [];
      let missing = 0;

      all_args.forEach(arg => { if (!arg.ready) missing++; });

      if (missing > 0) {
         if (isFunc(resolveFunc)) {
            draw_g.node().textResolveFunc = resolveFunc;
            draw_g.node().try_optimize = try_optimize;
         }
         return;
      }

      draw_g.property('all_args', null); // clear all_args property

      // adjust font size (if there are normal text)
      const f = draw_g.property('text_factor'),
            font = draw_g.property('text_font'),
            max_sz = draw_g.property('max_font_size');
      let font_size = font.size, any_text = false, only_text = true;

      if ((f > 0) && ((f < 0.95) || (f > 1.05)))
         font.size = Math.max(1, Math.floor(font.size / f));

      if (max_sz && (font.size > max_sz))
         font.size = max_sz;

      if (font.size !== font_size) {
         draw_g.call(font.func);
         font_size = font.size;
      }

      all_args.forEach(arg => {
         if (arg.mj_node && arg.mj_func) {
            const svg = arg.mj_node.select('svg'); // MathJax svg
            arg.mj_func(this, arg.mj_node, svg, arg, font_size, f);
            delete arg.mj_node; // remove reference
            only_text = false;
         } else if (arg.txt_g)
            only_text = false;
      });

      if (!resolveFunc) {
         resolveFunc = draw_g.node().textResolveFunc;
         try_optimize = draw_g.node().try_optimize;
         delete draw_g.node().textResolveFunc;
         delete draw_g.node().try_optimize;
      }

      const optimize_arr = (try_optimize && only_text) ? [] : null;

      // now process text and latex drawings
      all_args.forEach(arg => {
         let txt, is_txt, scale = 1;
         if (arg.txt_node) {
            txt = arg.txt_node;
            delete arg.txt_node;
            is_txt = true;
            if (optimize_arr !== null) optimize_arr.push(txt);
         } else if (arg.txt_g) {
            txt = arg.txt_g;
            delete arg.txt_g;
            is_txt = false;
         } else
            return;

         txt.attr('visibility', null);

         any_text = true;

         if (arg.width) {
            // adjust x position when scale into specified rectangle
            if (arg.align[0] === 'middle')
               arg.x += arg.width / 2;
             else if (arg.align[0] === 'end')
                arg.x += arg.width;
         }

         if (arg.height) {
            if (arg.align[1].indexOf('bottom') === 0)
               arg.y += arg.height;
            else if (arg.align[1] === 'middle')
               arg.y += arg.height / 2;
         }

         let dx = 0, dy = 0;

         if (is_txt) {
            // handle simple text drawing

            if (isNodeJs()) {
               if (arg.scale && (f > 0)) { arg.box.width *= 1/f; arg.box.height *= 1/f; }
            } else if (!arg.plain && !arg.fast) {
               // exact box dimension only required when complex text was build
               arg.box = getElementRect(txt, 'bbox');
            }

            if (arg.plain) {
               txt.attr('text-anchor', arg.align[0]);
               if (arg.align[1] === 'top')
                  txt.attr('dy', '.8em');
               else if (arg.align[1] === 'middle') {
                  // if (isNodeJs()) txt.attr('dy', '.4em'); else // old workaround for node.js
                  txt.attr('dominant-baseline', 'middle');
               }
            } else {
               txt.attr('text-anchor', 'start');
               dx = ((arg.align[0] === 'middle') ? -0.5 : ((arg.align[0] === 'end') ? -1 : 0)) * arg.box.width;
               dy = ((arg.align[1] === 'top') ? (arg.top_shift || 1) : (arg.align[1] === 'middle') ? (arg.mid_shift || 0.5) : 0) * arg.box.height;
            }
         } else if (arg.text_rect) {
            // handle latex drawing
            const box = arg.text_rect;

            scale = (f > 0) && (Math.abs(1-f) > 0.01) ? 1/f : 1;

            dx = ((arg.align[0] === 'middle') ? -0.5 : ((arg.align[0] === 'end') ? -1 : 0)) * box.width * scale;

            if (arg.align[1] === 'top')
               dy = -box.y1*scale;
            else if (arg.align[1] === 'bottom')
               dy = -box.y2*scale;
            else if (arg.align[1] === 'middle')
               dy = -0.5*(box.y1 + box.y2)*scale;
         } else
            console.error('text rect not calcualted - please check code');

         if (!arg.rotate) {
            arg.x += dx;
            arg.y += dy;
            dx = dy = 0;
         }

         // use translate and then rotate to avoid complex sign calculations
         let trans = makeTranslate(Math.round(arg.x), Math.round(arg.y)) || '';
         const dtrans = makeTranslate(Math.round(dx), Math.round(dy)),
               append = aaa => { if (trans) trans += ' '; trans += aaa; };

         if (arg.rotate)
            append(`rotate(${Math.round(arg.rotate)})`);
         if (scale !== 1)
            append(`scale(${scale.toFixed(3)})`);
         if (dtrans)
            append(dtrans);
         if (trans)
            txt.attr('transform', trans);
      });


      // when no any normal text drawn - remove font attributes
      if (!any_text)
         font.clearFont(draw_g);

      if ((optimize_arr !== null) && (optimize_arr.length > 1)) {
         ['fill', 'text-anchor'].forEach(name => {
            let first = optimize_arr[0].attr(name);
            optimize_arr.forEach(txt_node => {
               const value = txt_node.attr(name);
               if (!value || (value !== first)) first = undefined;
            });
            if (first) {
               draw_g.attr(name, first);
               optimize_arr.forEach(txt_node => { txt_node.attr(name, null); });
            }
         });
      }

      // if specified, call resolve function
      if (resolveFunc) resolveFunc(this); // IMPORTANT - return painter, may use in draw methods
   }

   /** @summary Post-process plain text drawing
     * @private */
   #postprocessDrawText(arg, txt_node) {
      // complete rectangle with very rough size estimations
      arg.box = !isNodeJs() && !settings.ApproxTextSize && !arg.fast
                 ? getElementRect(txt_node, 'bbox')
                 : (arg.text_rect || { height: Math.round(1.15 * arg.font_size), width: approximateLabelWidth(arg.text, arg.font, arg.font_size) });

      txt_node.attr('visibility', 'hidden'); // hide elements until text drawing is finished

      if (arg.box.width > arg.draw_g.property('max_text_width'))
         arg.draw_g.property('max_text_width', arg.box.width);
      if (arg.scale)
         this.scaleTextDrawing(Math.max(1.05 * arg.box.width / arg.width, arg.box.height / arg.height), arg.draw_g);

      arg.result_width = arg.box.width;
      arg.result_height = arg.box.height;

      if (isFunc(arg.post_process))
         arg.post_process(this);

      return arg.box.width;
   }

   /** @summary Draw text
     * @desc The only legal way to draw text, support plain, latex and math text output
     * @param {object} arg - different text draw options
     * @param {string} arg.text - text to draw
     * @param {number} [arg.align = 12] - int value like 12 or 31
     * @param {string} [arg.align = undefined] - end;bottom
     * @param {number} [arg.x = 0] - x position
     * @param {number} [arg.y = 0] - y position
     * @param {number} [arg.width] - when specified, adjust font size in the specified box
     * @param {number} [arg.height] - when specified, adjust font size in the specified box
     * @param {boolean} [arg.scale = true] - scale into draw box when width and height parameters are specified
     * @param {number} [arg.latex] - 0 - plain text, 1 - normal TLatex, 2 - math
     * @param {string} [arg.color=black] - text color
     * @param {number} [arg.rotate] - rotation angle
     * @param {number} [arg.font_size] - fixed font size
     * @param {object} [arg.draw_g] - element where to place text, if not specified central draw_g container is used
     * @param {function} [arg.post_process] - optional function called when specified text is drawn
     * @protected */
   drawText(arg) {
      if (!arg.text)
         arg.text = '';

      arg.draw_g = arg.draw_g || this.getG();
      if (!arg.draw_g || arg.draw_g.empty())
         return;

      const font = arg.draw_g.property('text_font');
      arg.font = font; // use in latex conversion

      if (font) {
         arg.color = arg.color || font.color;
         arg.align = arg.align || font.align;
         arg.rotate = arg.rotate || font.angle;
      }

      let align = ['start', 'middle'];

      if (isStr(arg.align)) {
         align = arg.align.split(';');
         if (align.length === 1)
            align.push('middle');
      } else if (typeof arg.align === 'number') {
         if ((arg.align / 10) >= 3)
            align[0] = 'end';
         else if ((arg.align / 10) >= 2)
            align[0] = 'middle';
         if ((arg.align % 10) === 0)
            align[1] = 'bottom';
         else if ((arg.align % 10) === 1)
            align[1] = 'bottom-base';
         else if ((arg.align % 10) === 3)
            align[1] = 'top';
      } else if (isObject(arg.align) && (arg.align.length === 2))
         align = arg.align;

      if (arg.latex === undefined)
         arg.latex = 1; //  0: text, 1: latex, 2: math
      arg.align = align;
      arg.x = arg.x || 0;
      arg.y = arg.y || 0;
      if (arg.scale !== false)
         arg.scale = arg.width && arg.height && !arg.font_size;
      arg.width = arg.width || 0;
      arg.height = arg.height || 0;

      if (arg.draw_g.property('_fast_drawing')) {
         if (arg.scale) {
            // area too small - ignore such drawing
            if (arg.height < 4)
               return 0;
         } else if (arg.font_size) {
            // font size too small
            if (arg.font_size < 4)
               return 0;
         } else if (arg.draw_g.property('_font_too_small')) {
            // configure font is too small - ignore drawing
            return 0;
         }
      }

      // include drawing into list of all args
      arg.draw_g.property('all_args').push(arg);
      arg.ready = false; // indicates if drawing is ready for post-processing

      let use_mathjax = (arg.latex === 2);
      const cl = constants.Latex;

      if (arg.latex === 1) {
         use_mathjax = (settings.Latex === cl.AlwaysMathJax) ||
                       ((settings.Latex === cl.MathJax) && arg.text.match(/[#{\\]/g)) ||
                       arg.text.match(/[\\]/g);
      }

      if (!use_mathjax || arg.nomathjax) {
         arg.txt_node = arg.draw_g.append('svg:text');

         if (arg.color)
            arg.txt_node.attr('fill', arg.color);

         if (arg.font_size)
            arg.txt_node.attr('font-size', arg.font_size);
         else
            arg.font_size = font.size;

         arg.plain = !arg.latex || (settings.Latex === cl.Off) || (settings.Latex === cl.Symbols);

         arg.simple_latex = arg.latex && (settings.Latex === cl.Symbols);

         if (!arg.plain || arg.simple_latex || arg.font?.isSymbol) {
            if (arg.simple_latex || isPlainText(arg.text) || arg.plain) {
               arg.simple_latex = true;
               producePlainText(this, arg.txt_node, arg);
            } else {
               arg.txt_node.remove(); // just remove text node
               delete arg.txt_node;
               arg.txt_g = arg.draw_g.append('svg:g');
               produceLatex(this, arg.txt_g, arg);
            }
            arg.ready = true;
            this.#postprocessDrawText(arg, arg.txt_g || arg.txt_node);

            if (arg.draw_g.property('draw_text_completed'))
               this.#checkAllTextDrawing(arg.draw_g); // check if all other elements are completed
            return 0;
         }

         arg.plain = true;
         arg.txt_node.text(arg.text);
         arg.ready = true;

         return this.#postprocessDrawText(arg, arg.txt_node);
      }

      arg.mj_node = arg.draw_g.append('svg:g').attr('visibility', 'hidden'); // hide text until drawing is finished

      produceMathjax(this, arg.mj_node, arg).then(() => {
         arg.ready = true;
         if (arg.draw_g.property('draw_text_completed'))
            this.#checkAllTextDrawing(arg.draw_g);
      });

      return 0;
   }

   /** @summary Finish text drawing
     * @desc Should be called to complete all text drawing operations
     * @param {function} [draw_g] - <g> element for text drawing, default is getG()
     * @return {Promise} when text drawing completed
     * @protected */
   async finishTextDrawing(draw_g, try_optimize) {
      if (!draw_g)
         draw_g = this.getG();
      if (!draw_g || draw_g.empty())
         return false;

      draw_g.property('draw_text_completed', true); // mark that text drawing is completed

      return new Promise(resolveFunc => {
         this.#checkAllTextDrawing(draw_g, resolveFunc, try_optimize);
      });
   }

   /** @summary Configure user-defined context menu for the object
     * @desc fillmenu_func will be called when context menu is activated
     * Arguments fillmenu_func are (menu,kind)
     * First is menu object, second is object sub-element like axis 'x' or 'y'
     * Function should return promise with menu when items are filled
     * @param {function} fillmenu_func - function to fill custom context menu for object */
   configureUserContextMenu(fillmenu_func) {
      this.#user_context_menu = isFunc(fillmenu_func) ? fillmenu_func : undefined;
   }

   /** @summary Fill object menu in web canvas
     * @private */
   async fillObjectExecMenu(menu, kind) {
      if (isFunc(this.#user_context_menu))
         return this.#user_context_menu(menu, kind);

      const canvp = this.getCanvPainter();

      if (!this.getSnapId() || !canvp || canvp?.isReadonly() || !canvp?.getWebsocket())
         return menu;

      function doExecMenu(arg) {
         const execp = menu.exec_painter || this,
               cp = execp.getCanvPainter(),
               item = menu.exec_items[parseInt(arg)];

         if (!item?.fName) return;

         // this is special entry, produced by TWebMenuItem, which recognizes editor entries itself
         if (item.fExec === 'Show:Editor') {
            if (isFunc(cp?.activateGed))
               cp.activateGed(execp);
            return;
         }

         if (isFunc(cp?.executeObjectMethod) && cp.executeObjectMethod(execp, item, item.$execid))
            return;

         item.fClassName = execp.getClassName();
         if ((item.$execid.indexOf('#x') > 0) || (item.$execid.indexOf('#y') > 0) || (item.$execid.indexOf('#z') > 0))
            item.fClassName = clTAxis;

         if (execp.executeMenuCommand(item))
            return;

         if (!item.$execid)
            return;

         if (!item.fArgs) {
            return cp?.v7canvas ? cp.submitExec(execp, item.fExec, kind)
                                : execp.submitCanvExec(item.fExec, item.$execid);
         }

         menu.showMethodArgsDialog(item).then(args => {
            if (!args || execp.executeMenuCommand(item, args))
               return;

            const exec = item.fExec.slice(0, item.fExec.length - 1) + args + ')';
            if (cp?.v7canvas)
               cp.submitExec(execp, exec, kind);
            else
               cp?.sendWebsocket(`OBJEXEC:${item.$execid}:${exec}`);
         });
      }

      const doFillMenu = (_menu, _reqid, _resolveFunc, reply) => {
         // avoid multiple call of the callback after timeout
         if (menu._got_menu)
            return;
         menu._got_menu = true;

         if (reply && (_reqid !== reply.fId))
            console.error(`missmatch between request ${_reqid} and reply ${reply.fId} identifiers`);

         menu.exec_items = reply?.fItems;

         if (menu.exec_items?.length) {
            if (_menu.size() > 0)
               _menu.separator();

            let lastclname;

            for (let n = 0; n < menu.exec_items.length; ++n) {
               const item = menu.exec_items[n];
               item.$execid = reply.fId;
               item.$menu = menu;

               if (item.fClassName && lastclname && (lastclname !== item.fClassName)) {
                  _menu.endsub();
                  lastclname = '';
               }
               if (lastclname !== item.fClassName) {
                  lastclname = item.fClassName;
                  const p = lastclname.lastIndexOf('::'),
                        shortname = (p > 0) ? lastclname.slice(p+2) : lastclname;

                  _menu.sub(shortname.replace(/[<>]/g, '_'));
               }

               if ((item.fChecked === undefined) || (item.fChecked < 0))
                  _menu.add(item.fName, n, doExecMenu);
               else
                  _menu.addchk(item.fChecked, item.fName, n, doExecMenu);
            }

            if (lastclname)
               _menu.endsub();
         }

         _resolveFunc(_menu);
      },

      reqid = this.getSnapId(kind);

      menu._got_menu = false;

      // if menu painter differs from this, remember it for further usage
      if (menu.painter)
         menu.exec_painter = (menu.painter !== this) ? this : undefined;

      return new Promise(resolveFunc => {
         let did_resolve = false;

         function handleResolve(res) {
            if (did_resolve) return;
            did_resolve = true;
            resolveFunc(res);
         }

         // set timeout to avoid menu hanging
         setTimeout(() => doFillMenu(menu, reqid, handleResolve), 2000);

         canvp.submitMenuRequest(this, kind, reqid).then(lst => doFillMenu(menu, reqid, handleResolve, lst));
      });
   }

   /** @summary Configure user-defined tooltip handler
     * @desc Hook for the users to get tooltip information when mouse cursor moves over frame area
     * Handler function will be called every time when new data is selected
     * when mouse leave frame area, handler(null) will be called
     * @param {function} handler - function called when tooltip is produced
     * @param {number} [tmout = 100] - delay in ms before tooltip delivered */
   configureUserTooltipHandler(handler, tmout = 100) {
      if (!handler || !isFunc(handler)) {
         this.#user_tooltip_handler = undefined;
         this.#user_tooltip_timeout = undefined;
      } else {
         this.#user_tooltip_handler = handler;
         this.#user_tooltip_timeout = tmout;
      }
   }

    /** @summary Configure user-defined click handler
      * @desc Function will be called every time when frame click was performed
      * As argument, tooltip object with selected bins will be provided
      * If handler function returns true, default handling of click will be disabled
      * @param {function} handler - function called when mouse click is done */
   configureUserClickHandler(handler) {
      const fp = this.getFramePainter();
      if (isFunc(fp?.configureUserClickHandler))
         fp.configureUserClickHandler(handler);
   }

   /** @summary Configure user-defined dblclick handler
     * @desc Function will be called every time when double click was called
     * As argument, tooltip object with selected bins will be provided
     * If handler function returns true, default handling of dblclick (unzoom) will be disabled
     * @param {function} handler - function called when mouse double click is done */
   configureUserDblclickHandler(handler) {
      const fp = this.getFramePainter();
      if (isFunc(fp?.configureUserDblclickHandler))
         fp.configureUserDblclickHandler(handler);
   }

   /** @summary Check if user-defined tooltip function was configured
     * @return {boolean} flag is user tooltip handler was configured */
   hasUserTooltip() {
      return isFunc(this.#user_tooltip_handler);
   }

   /** @summary Provide tooltips data to user-defined function
     * @param {object} data - tooltip data
     * @private */
   provideUserTooltip(data) {
      if (!this.hasUserTooltip())
         return;

      if (this.#user_tooltip_timeout <= 0)
         return this.#user_tooltip_handler(data);

      if (this.#user_toottip_handle) {
         clearTimeout(this.#user_toottip_handle);
         this.#user_toottip_handle = undefined;
      }

      if (!data)
         return this.#user_tooltip_handler(data);

      // only after timeout user function will be called
      this.#user_toottip_handle = setTimeout(() => {
         this.#user_toottip_handle = undefined;
         if (this.#user_tooltip_handler)
            this.#user_tooltip_handler(data);
      }, this.#user_tooltip_timeout);
   }

   /** @summary Provide projection areas
     * @param kind - 'X', 'Y', 'XY' or ''
     * @private */
   async provideSpecialDrawArea(kind) {
      if (kind === this.#special_draw_area)
         return true;

      return this.getCanvPainter().toggleProjection(kind).then(() => {
         this.#special_draw_area = kind;
         return true;
      });
   }

   /** @summary Draw in special projection areas
     * @param obj - object to draw
     * @param opt - draw option
     * @param kind - '', 'X', 'Y'
     * @private */
   async drawInSpecialArea(obj, opt, kind) {
      const canp = this.getCanvPainter();
      if (this.#special_draw_area && isFunc(canp?.drawProjection))
         return canp.drawProjection(kind || this.#special_draw_area, obj, opt);

      return false;
   }

   /** @summary Get tooltip for painter and specified event position
     * @param {Object} evnt - object with clientX and clientY positions
     * @private */
   getToolTip(evnt) {
      if ((evnt?.clientX === undefined) || (evnt?.clientY === undefined))
         return null;

      const frame = this.getPadPainter()?.getFrameSvg();
      if (!frame || frame.empty())
         return null;
      const layer = frame.selectChild('.main_layer');
      if (layer.empty())
         return null;

      const pos = d3_pointer(evnt, layer.node()),
            pnt = { touch: false, x: pos[0], y: pos[1] };

      if (isFunc(this.extractToolTip))
         return this.extractToolTip(pnt);

      pnt.disabled = true;

      const res = isFunc(this.processTooltipEvent) ? this.processTooltipEvent(pnt) : null;

      return res?.user_info || res;
   }

} // class ObjectPainter


/** @summary Generic text drawing
  * @private */
function drawRawText(dom, txt /* , opt */) {
   const painter = new BasePainter(dom);
   painter.txt = txt;

   painter.redrawObject = async function(obj) {
      this.txt = obj;
      return this.drawText();
   };

   painter.drawText = async function() {
      let stxt = (this.txt._typename === clTObjString) ? this.txt.fString : this.txt.value;
      if (!isStr(stxt))
         stxt = '<undefined>';

      const mathjax = this.txt.mathjax || (settings.Latex === constants.Latex.AlwaysMathJax);

      if (!mathjax && !('as_is' in this.txt)) {
         const arr = stxt.split('\n');
         stxt = '';
         for (let i = 0; i < arr.length; ++i)
            stxt += `<pre style='margin:0'>${arr[i]}</pre>`;
      }

      const frame = this.selectDom();
      let main = frame.select('div');
      if (main.empty())
         main = frame.append('div').attr('style', 'max-width:100%;max-height:100%;overflow:auto');
      main.html(stxt);

      // (re) set painter to first child element, base painter not requires canvas
      this.setTopPainter();

      if (mathjax)
         typesetMathjax(frame.node());

      return this;
   };

   return painter.drawText();
}

/** @summary Returns canvas painter (if any) for specified DOM element
  * @param {string|object} dom - id or DOM element or pad painter
  * @private */
function getElementCanvPainter(dom) {
   return isPadPainter(dom) ? dom.getCanvPainter() : getDomCanvasPainter(new ObjectPainter(dom).selectDom());
}

/** @summary Returns pad painter (if any) for specified DOM element
  * @param {string|object} dom - id or DOM element or pad painter
  * @private */
function getElementPadPainter(dom) {
   return isPadPainter(dom) ? dom : new ObjectPainter(dom).getPadPainter();
}

/** @summary Returns main painter (if any) for specified HTML element - typically histogram painter
  * @param {string|object} dom - id or DOM element or pad painter
  * @private */
function getElementMainPainter(dom) {
   return isPadPainter(dom) ? dom.getMainPainter() : new ObjectPainter(dom).getMainPainter(true);
}

/** @summary Save object, drawn in specified element, as JSON.
  * @desc Normally it is TCanvas object with list of primitives
  * @param {string|object} dom - id of top div element or directly DOMElement
  * @return {string} produced JSON string */
function drawingJSON(dom) {
   return getElementCanvPainter(dom)?.produceJSON() || '';
}

let $active_pp = null;

/** @summary Set active pad painter
  * @desc Normally be used to handle key press events, which are global in the web browser
  * @param {object} args - functions arguments
  * @param {object} args.pp - pad painter
  * @param {boolean} [args.active] - is pad activated or not
  * @private */
function selectActivePad(args) {
   if (args.active) {
      $active_pp?.getFramePainter()?.setFrameActive(false);
      $active_pp = args.pp;
      $active_pp?.getFramePainter()?.setFrameActive(true);
   } else if ($active_pp === args.pp)
      $active_pp = null;
}

/** @summary Returns current active pad
  * @desc Should be used only for keyboard handling
  * @private */
function getActivePad() {
   return $active_pp;
}

/** @summary Check resize of drawn element
  * @param {string|object} dom - id or DOM element
  * @param {boolean|object} arg - options on how to resize
  * @desc As first argument dom one should use same argument as for the drawing
  * As second argument, one could specify 'true' value to force redrawing of
  * the element even after minimal resize
  * Or one just supply object with exact sizes like { width:300, height:200, force:true };
  * @example
  * import { resize } from 'https://root.cern/js/latest/modules/base/ObjectPainter.mjs';
  * resize('drawing', { width: 500, height: 200 });
  * resize(document.querySelector('#drawing'), true); */
function resize(dom, arg) {
   if (arg === true)
      arg = { force: true };
   else if (!isObject(arg))
      arg = null;
   let done = false;
   new ObjectPainter(dom).forEachPainter(painter => {
      if (!done && isFunc(painter.checkResize))
         done = painter.checkResize(arg);
   });
   return done;
}


/** @summary Safely remove all drawings from specified element
  * @param {string|object} dom - id or DOM element
  * @public
  * @example
  * import { cleanup } from 'https://root.cern/js/latest/modules/base/ObjectPainter.mjs';
  * cleanup('drawing');
  * cleanup(document.querySelector('#drawing')); */
function cleanup(dom) {
   const dummy = new ObjectPainter(dom), lst = [];
   dummy.forEachPainter(p => { if (lst.indexOf(p) < 0) lst.push(p); });
   lst.forEach(p => p.cleanup());
   dummy.selectDom().html('');
   return lst;
}

const EAxisBits = {
   kDecimals: BIT(7),
   kTickPlus: BIT(9),
   kTickMinus: BIT(10),
   kAxisRange: BIT(11),
   kCenterTitle: BIT(12),
   kCenterLabels: BIT(14),
   kRotateTitle: BIT(15),
   kPalette: BIT(16),
   kNoExponent: BIT(17),
   kLabelsHori: BIT(18),
   kLabelsVert: BIT(19),
   kLabelsDown: BIT(20),
   kLabelsUp: BIT(21),
   kIsInteger: BIT(22),
   kMoreLogLabels: BIT(23),
   kOppositeTitle: BIT(32) // artificial bit, not possible to set in ROOT
}, kAxisLabels = 'labels', kAxisNormal = 'normal', kAxisFunc = 'func', kAxisTime = 'time';

Object.assign(internals.jsroot, { ObjectPainter, cleanup, resize });

export { isPadPainter, getDomCanvasPainter, getElementPadPainter, getElementCanvPainter, getElementMainPainter, drawingJSON,
         selectActivePad, getActivePad, cleanup, resize, drawRawText,
         ObjectPainter, EAxisBits, kAxisLabels, kAxisNormal, kAxisFunc, kAxisTime };
