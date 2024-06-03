import { select as d3_select } from '../d3.mjs';
import { settings, internals, isNodeJs, isFunc, isStr, isObject, btoa_func, getDocument, source_dir, loadScript, httpRequest } from '../core.mjs';
import { detectFont, addCustomFont, getCustomFont, FontHandler } from './FontHandler.mjs';
import { approximateLabelWidth, replaceSymbolsInTextNode } from './latex.mjs';
import { getColor } from './colors.mjs';


/** @summary Returns visible rect of element
  * @param {object} elem - d3.select object with element
  * @param {string} [kind] - which size method is used
  * @desc kind = 'bbox' use getBBox, works only with SVG
  * kind = 'full' - full size of element, using getBoundingClientRect function
  * kind = 'nopadding' - excludes padding area
  * With node.js can use 'width' and 'height' attributes when provided in element
  * @private */
function getElementRect(elem, sizearg) {
   if (!elem || elem.empty())
      return { x: 0, y: 0, width: 0, height: 0 };

   if ((isNodeJs() && (sizearg !== 'bbox')) || elem.property('_batch_mode'))
      return { x: 0, y: 0, width: parseInt(elem.attr('width')), height: parseInt(elem.attr('height')) };

   const styleValue = name => {
      let value = elem.style(name);
      if (!value || !isStr(value)) return 0;
      value = parseFloat(value.replace('px', ''));
      return !Number.isFinite(value) ? 0 : Math.round(value);
   };

   let rect = elem.node().getBoundingClientRect();
   if ((sizearg === 'bbox') && (parseFloat(rect.width) > 0))
      rect = elem.node().getBBox();

   const res = { x: 0, y: 0, width: parseInt(rect.width), height: parseInt(rect.height) };
   if (rect.left !== undefined) {
      res.x = parseInt(rect.left);
      res.y = parseInt(rect.top);
   } else if (rect.x !== undefined) {
      res.x = parseInt(rect.x);
      res.y = parseInt(rect.y);
   }

   if ((sizearg === undefined) || (sizearg === 'nopadding')) {
      // this is size exclude padding area
      res.width -= styleValue('padding-left') + styleValue('padding-right');
      res.height -= styleValue('padding-top') + styleValue('padding-bottom');
   }

   return res;
}


/** @summary Calculate absolute position of provided element in canvas
  * @private */
function getAbsPosInCanvas(sel, pos) {
   if (!pos) return pos;

   while (!sel.empty() && !sel.classed('root_canvas')) {
      const cl = sel.attr('class');
      if (cl && ((cl.indexOf('root_frame') >= 0) || (cl.indexOf('__root_pad_') >= 0))) {
         pos.x += sel.property('draw_x') || 0;
         pos.y += sel.property('draw_y') || 0;
      }
      sel = d3_select(sel.node().parentNode);
   }
   return pos;
}


/** @summary Converts numeric value to string according to specified format.
  * @param {number} value - value to convert
  * @param {string} [fmt='6.4g'] - format can be like 5.4g or 4.2e or 6.4f
  * @param {boolean} [ret_fmt] - when true returns array with value and actual format like ['0.1','6.4f']
  * @return {string|Array} - converted value or array with value and actual format
  * @private */
function floatToString(value, fmt, ret_fmt) {
   if (!fmt) fmt = '6.4g';

   fmt = fmt.trim();
   const len = fmt.length;
   if (len < 2)
      return ret_fmt ? [value.toFixed(4), '6.4f'] : value.toFixed(4);
   const last = fmt[len-1];
   fmt = fmt.slice(0, len-1);
   let isexp, prec = fmt.indexOf('.');
   prec = (prec < 0) ? 4 : parseInt(fmt.slice(prec+1));
   if (!Number.isInteger(prec) || (prec <= 0)) prec = 4;

   let significance = false;
   if ((last === 'e') || (last === 'E')) isexp = true; else
   if (last === 'Q') { isexp = true; significance = true; } else
   if ((last === 'f') || (last === 'F')) isexp = false; else
   if (last === 'W') { isexp = false; significance = true; } else
   if ((last === 'g') || (last === 'G')) {
      const se = floatToString(value, fmt+'Q', true);
      let sg = floatToString(value, fmt+'W', true);
      if (se[0].length < sg[0].length) sg = se;
      return ret_fmt ? sg : sg[0];
   } else {
      isexp = false;
      prec = 4;
   }

   if (isexp) {
      // for exponential representation only one significant digit befor point
      if (significance) prec--;
      if (prec < 0) prec = 0;

      const se = value.toExponential(prec);
      return ret_fmt ? [se, `5.${prec}e`] : se;
   }

   let sg = value.toFixed(prec);

   if (significance) {
      // when using fixed representation, one could get 0
      if ((value !== 0) && (Number(sg) === 0) && (prec > 0)) {
         prec = 20; sg = value.toFixed(prec);
      }

      let l = 0;
      while ((l < sg.length) && (sg[l] === '0' || sg[l] === '-' || sg[l] === '.')) l++;

      let diff = sg.length - l - prec;
      if (sg.indexOf('.') > l) diff--;

      if (diff !== 0) {
         prec -= diff;
         if (prec < 0)
            prec = 0;
         else if (prec > 20)
            prec = 20;
         sg = value.toFixed(prec);
      }
   }

   return ret_fmt ? [sg, '5.'+prec+'f'] : sg;
}


/** @summary Draw options interpreter
  * @private */
class DrawOptions {

   constructor(opt) {
      this.opt = isStr(opt) ? opt.toUpperCase().trim() : '';
      this.part = '';
   }

   /** @summary Returns true if remaining options are empty or contain only seperators symbols. */
   empty() {
      if (this.opt.length === 0) return true;
      return this.opt.replace(/[ ;_,]/g, '').length === 0;
   }

   /** @summary Returns remaining part of the draw options. */
   remain() { return this.opt; }

   /** @summary Checks if given option exists */
   check(name, postpart) {
      const pos = this.opt.indexOf(name);
      if (pos < 0) return false;
      this.opt = this.opt.slice(0, pos) + this.opt.slice(pos + name.length);
      this.part = '';
      if (!postpart) return true;

      let pos2 = pos;
      while ((pos2 < this.opt.length) && (this.opt[pos2] !== ' ') && (this.opt[pos2] !== ',') && (this.opt[pos2] !== ';')) pos2++;
      if (pos2 > pos) {
         this.part = this.opt.slice(pos, pos2);
         this.opt = this.opt.slice(0, pos) + this.opt.slice(pos2);
      }

      if (postpart !== 'color')
         return true;

      this.color = this.partAsInt(1) - 1;
      if (this.color >= 0) return true;
      for (let col = 0; col < 8; ++col) {
         if (getColor(col).toUpperCase() === this.part) {
            this.color = col;
            return true;
         }
      }
      return false;
   }

   /** @summary Returns remaining part of found option as integer. */
   partAsInt(offset, dflt) {
      let mult = 1;
      const last = this.part ? this.part[this.part.length - 1] : '';
      if (last === 'K')
         mult = 1e3;
      else if (last === 'M')
         mult = 1e6;
      else if (last === 'G')
         mult = 1e9;
      let val = this.part.replace(/^\D+/g, '');
      val = val ? parseInt(val, 10) : Number.NaN;
      return !Number.isInteger(val) ? (dflt || 0) : mult*val + (offset || 0);
   }

   /** @summary Returns remaining part of found option as float. */
   partAsFloat(offset, dflt) {
      let val = this.part.replace(/^\D+/g, '');
      val = val ? parseFloat(val) : Number.NaN;
      return !Number.isFinite(val) ? (dflt || 0) : val + (offset || 0);
   }

} // class DrawOptions


/** @summary Simple random generator with controlled seed
  * @private */
class TRandom {

   constructor(i) {
      if (i !== undefined) this.seed(i);
   }

   /** @summary Seed simple random generator */
   seed(i) {
      i = Math.abs(i);
      if (i > 1e8)
         i = Math.abs(1e8 * Math.sin(i));
      else if (i < 1)
         i *= 1e8;
      this.m_w = Math.round(i);
      this.m_z = 987654321;
   }

   /** @summary Produce random value between 0 and 1 */
   random() {
      if (this.m_z === undefined) return Math.random();
      this.m_z = (36969 * (this.m_z & 65535) + (this.m_z >> 16)) & 0xffffffff;
      this.m_w = (18000 * (this.m_w & 65535) + (this.m_w >> 16)) & 0xffffffff;
      let result = ((this.m_z << 16) + this.m_w) & 0xffffffff;
      result /= 4294967296;
      return result + 0.5;
   }

} // class TRandom


/** @summary Build smooth SVG curve uzing Bezier
  * @desc Reuse code from https://stackoverflow.com/questions/62855310
  * @private */
function buildSvgCurve(p, args) {
   if (!args)
      args = {};
   if (!args.line)
      args.calc = true;
   else if (args.ndig === undefined)
      args.ndig = 0;

   let npnts = p.length;
   if (npnts < 3) args.line = true;

   args.t = args.t ?? 0.2;

   if ((args.ndig === undefined) || args.height) {
      args.maxy = p[0].gry;
      args.mindiff = 100;
      for (let i = 1; i < npnts; i++) {
         args.maxy = Math.max(args.maxy, p[i].gry);
         args.mindiff = Math.min(args.mindiff, Math.abs(p[i].grx - p[i-1].grx), Math.abs(p[i].gry - p[i-1].gry));
      }
      if (args.ndig === undefined)
         args.ndig = args.mindiff > 20 ? 0 : (args.mindiff > 5 ? 1 : 2);
   }

   const end_point = (pnt1, pnt2, sign) => {
      const len = Math.sqrt((pnt2.gry - pnt1.gry)**2 + (pnt2.grx - pnt1.grx)**2) * args.t,
            a2 = Math.atan2(pnt2.dgry, pnt2.dgrx),
            a1 = Math.atan2(sign*(pnt2.gry - pnt1.gry), sign*(pnt2.grx - pnt1.grx));

      pnt1.dgrx = len * Math.cos(2*a1 - a2);
      pnt1.dgry = len * Math.sin(2*a1 - a2);
   }, conv = val => {
      if (!args.ndig || (Math.round(val) === val))
         return val.toFixed(0);
      let s = val.toFixed(args.ndig), p = s.length-1;
      while (s[p] === '0') p--;
      if (s[p] === '.') p--;
      s = s.slice(0, p+1);
      return (s === '-0') ? '0' : s;
   };

   if (args.calc) {
      for (let i = 1; i < npnts - 1; i++) {
         p[i].dgrx = (p[i+1].grx - p[i-1].grx) * args.t;
         p[i].dgry = (p[i+1].gry - p[i-1].gry) * args.t;
      }

      if (npnts > 2) {
         end_point(p[0], p[1], 1);
         end_point(p[npnts - 1], p[npnts - 2], -1);
      } else if (p.length === 2) {
         p[0].dgrx = (p[1].grx - p[0].grx) * args.t;
         p[0].dgry = (p[1].gry - p[0].gry) * args.t;
         p[1].dgrx = -p[0].dgrx;
         p[1].dgry = -p[0].dgry;
      }
   }

   let path = `${args.cmd ?? 'M'}${conv(p[0].grx)},${conv(p[0].gry)}`;

   if (!args.line) {
      let i0 = 1;
      if (args.qubic) {
         npnts--; i0++;
         path += `Q${conv(p[1].grx-p[1].dgrx)},${conv(p[1].gry-p[1].dgry)},${conv(p[1].grx)},${conv(p[1].gry)}`;
      }
      path += `C${conv(p[i0-1].grx+p[i0-1].dgrx)},${conv(p[i0-1].gry+p[i0-1].dgry)},${conv(p[i0].grx-p[i0].dgrx)},${conv(p[i0].gry-p[i0].dgry)},${conv(p[i0].grx)},${conv(p[i0].gry)}`;

      // continue with simpler points
      for (let i = i0 + 1; i < npnts; i++)
         path += `S${conv(p[i].grx-p[i].dgrx)},${conv(p[i].gry-p[i].dgry)},${conv(p[i].grx)},${conv(p[i].gry)}`;

      if (args.qubic)
         path += `Q${conv(p[npnts].grx-p[npnts].dgrx)},${conv(p[npnts].gry-p[npnts].dgry)},${conv(p[npnts].grx)},${conv(p[npnts].gry)}`;
   } else if (npnts < 10000) {
      // build simple curve

      let acc_x = 0, acc_y = 0, currx = Math.round(p[0].grx), curry = Math.round(p[0].gry);

      const flush = () => {
         if (acc_x) { path += 'h' + acc_x; acc_x = 0; }
         if (acc_y) { path += 'v' + acc_y; acc_y = 0; }
      };

      for (let n = 1; n < npnts; ++n) {
         const bin = p[n],
               dx = Math.round(bin.grx) - currx,
               dy = Math.round(bin.gry) - curry;
         if (dx && dy) {
            flush();
            path += `l${dx},${dy}`;
         } else if (!dx && dy) {
            if ((acc_y === 0) || ((dy < 0) !== (acc_y < 0))) flush();
            acc_y += dy;
         } else if (dx && !dy) {
            if ((acc_x === 0) || ((dx < 0) !== (acc_x < 0))) flush();
            acc_x += dx;
         }
         currx += dx; curry += dy;
      }

      flush();
   } else {
      // build line with trying optimize many vertical moves
      let currx = Math.round(p[0].grx), curry = Math.round(p[0].gry),
          cminy = curry, cmaxy = curry, prevy = curry;

      for (let n = 1; n < npnts; ++n) {
         const bin = p[n],
               lastx = Math.round(bin.grx),
               lasty = Math.round(bin.gry),
               dx = lastx - currx;
         if (dx === 0) {
            // if X not change, just remember amplitude and
            cminy = Math.min(cminy, lasty);
            cmaxy = Math.max(cmaxy, lasty);
            prevy = lasty;
            continue;
         }

         if (cminy !== cmaxy) {
            if (cminy !== curry)
               path += `v${cminy-curry}`;
            path += `v${cmaxy-cminy}`;
            if (cmaxy !== prevy)
               path += `v${prevy-cmaxy}`;
            curry = prevy;
         }
         const dy = lasty - curry;
         if (dy)
            path += `l${dx},${dy}`;
         else
            path += `h${dx}`;
         currx = lastx; curry = lasty;
         prevy = cminy = cmaxy = lasty;
      }

      if (cminy !== cmaxy) {
         if (cminy !== curry)
            path += `v${cminy-curry}`;
         path += `v${cmaxy-cminy}`;
         if (cmaxy !== prevy)
            path += `v${prevy-cmaxy}`;
      }
   }

   if (args.height)
      args.close = `L${conv(p[p.length-1].grx)},${conv(Math.max(args.maxy, args.height))}H${conv(p[0].grx)}Z`;

   return path;
}

/** @summary Compress SVG code, produced from drawing
  * @desc removes extra info or empty elements
  * @private */
function compressSVG(svg) {
   svg = svg.replace(/url\(&quot;#(\w+)&quot;\)/g, 'url(#$1)')         // decode all URL
            .replace(/ class="\w*"/g, '')                              // remove all classes
            .replace(/ pad="\w*"/g, '')                                // remove all pad ids
            .replace(/ title=""/g, '')                                 // remove all empty titles
            .replace(/<g objname="\w*" objtype="\w*"/g, '<g')          // remove object ids
            .replace(/<g transform="translate\(\d+,\d+\)"><\/g>/g, '') // remove all empty groups with transform
            .replace(/<g transform="translate\(\d+,\d+\)" style="display: none;"><\/g>/g, '') // remove hidden title
            .replace(/<g><\/g>/g, '');                                 // remove all empty groups

   // remove all empty frame svgs, typically appears in 3D drawings, maybe should be improved in frame painter itself
   svg = svg.replace(/<svg x="0" y="0" overflow="hidden" width="\d+" height="\d+" viewBox="0 0 \d+ \d+"><\/svg>/g, '');

   return svg;
}


/**
 * @summary Base painter class
 *
 */

class BasePainter {

   /** @summary constructor
     * @param {object|string} [dom] - dom element or id of dom element */
   constructor(dom) {
      this.divid = null; // either id of DOM element or element itself
      if (dom) this.setDom(dom);
   }

   /** @summary Assign painter to specified DOM element
     * @param {string|object} elem - element ID or DOM Element
     * @desc Normally DOM element should be already assigned in constructor
     * @protected */
   setDom(elem) {
      if (elem !== undefined) {
         this.divid = elem;
         delete this._selected_main;
      }
   }

   /** @summary Returns assigned dom element */
   getDom() {
      return this.divid;
   }

   /** @summary Selects main HTML element assigned for drawing
     * @desc if main element was layouted, returns main element inside layout
     * @param {string} [is_direct] - if 'origin' specified, returns original element even if actual drawing moved to some other place
     * @return {object} d3.select object for main element for drawing */
   selectDom(is_direct) {
      if (!this.divid) return d3_select(null);

      let res = this._selected_main;
      if (!res) {
         if (isStr(this.divid)) {
            let id = this.divid;
            if (id[0] !== '#') id = '#' + id;
            res = d3_select(id);
            if (!res.empty()) this.divid = res.node();
         } else
            res = d3_select(this.divid);
         this._selected_main = res;
      }

      if (!res || res.empty() || (is_direct === 'origin')) return res;

      const use_enlarge = res.property('use_enlarge'),
            layout = res.property('layout') || 'simple',
            layout_selector = (layout === 'simple') ? '' : res.property('layout_selector');

      if (layout_selector)
         res = res.select(layout_selector);

      // one could redirect here
      if (!is_direct && !res.empty() && use_enlarge)
         res = d3_select(getDocument().getElementById('jsroot_enlarge_div'));

      return res;
   }

   /** @summary Access/change top painter
     * @private */
   _accessTopPainter(on) {
      const chld = this.selectDom().node()?.firstChild;
      if (!chld) return null;
      if (on === true)
         chld.painter = this;
      else if (on === false)
         delete chld.painter;
      return chld.painter;
   }

   /** @summary Set painter, stored in first child element
     * @desc Only make sense after first drawing is completed and any child element add to configured DOM
     * @protected */
   setTopPainter() {
      this._accessTopPainter(true);
   }

   /** @summary Return top painter set for the selected dom element
     * @protected */
   getTopPainter() {
      return this._accessTopPainter();
   }

   /** @summary Clear reference on top painter
     * @protected */
   clearTopPainter() {
      this._accessTopPainter(false);
   }

   /** @summary Generic method to cleanup painter
     * @desc Removes all visible elements and all internal data */
   cleanup(keep_origin) {
      this.clearTopPainter();
      const origin = this.selectDom('origin');
      if (!origin.empty() && !keep_origin) origin.html('');
      this.divid = null;
      delete this._selected_main;

      if (isFunc(this._hpainter?.removePainter))
         this._hpainter.removePainter(this);

      delete this._hitemname;
      delete this._hdrawopt;
      delete this._hpainter;
   }

   /** @summary Checks if draw elements were resized and drawing should be updated
     * @return {boolean} true if resize was detected
     * @protected
     * @abstract */
   checkResize(/* arg */) {}

   /** @summary Function checks if geometry of main div was changed.
     * @desc take into account enlarge state, used only in PadPainter class
     * @return size of area when main div is drawn
     * @private */
   testMainResize(check_level, new_size, height_factor) {
      const enlarge = this.enlargeMain('state'),
            origin = this.selectDom('origin'),
            main = this.selectDom(),
            lmt = 5; // minimal size

      if ((enlarge !== 'on') && new_size?.width && new_size?.height) {
         origin.style('width', new_size.width + 'px')
               .style('height', new_size.height + 'px');
      }

      const rect_origin = getElementRect(origin, true),
            can_resize = origin.attr('can_resize');
      let do_resize = false;

      if (can_resize === 'height')
         if (height_factor && Math.abs(rect_origin.width * height_factor - rect_origin.height) > 0.1 * rect_origin.width) do_resize = true;

      if (((rect_origin.height <= lmt) || (rect_origin.width <= lmt)) &&
         can_resize && can_resize !== 'false') do_resize = true;

      if (do_resize && (enlarge !== 'on')) {
         // if zero size and can_resize attribute set, change container size

         if (rect_origin.width > lmt) {
            height_factor = height_factor || 0.66;
            origin.style('height', Math.round(rect_origin.width * height_factor) + 'px');
         } else if (can_resize !== 'height')
            origin.style('width', '200px').style('height', '100px');
      }

      const rect = getElementRect(main),
            old_h = main.property('_jsroot_height'),
            old_w = main.property('_jsroot_width');

      rect.changed = false;

      if (old_h && old_w && (old_h > 0) && (old_w > 0)) {
         if ((old_h !== rect.height) || (old_w !== rect.width))
            rect.changed = (check_level > 1) || (rect.width / old_w < 0.99) || (rect.width / old_w > 1.01) || (rect.height / old_h < 0.99) || (rect.height / old_h > 1.01);
      } else
         rect.changed = true;

      if (rect.changed)
         main.property('_jsroot_height', rect.height).property('_jsroot_width', rect.width);

      // after change enlarge state always mark main element as resized
      if (origin.property('did_enlarge')) {
         rect.changed = true;
         origin.property('did_enlarge', false);
      }

      return rect;
   }

   /** @summary Try enlarge main drawing element to full HTML page.
     * @param {string|boolean} action  - defines that should be done
     * @desc Possible values for action parameter:
     *    - true - try to enlarge
     *    - false - revert enlarge state
     *    - 'toggle' - toggle enlarge state
     *    - 'state' - only returns current enlarge state
     *    - 'verify' - check if element can be enlarged
     * if action not specified, just return possibility to enlarge main div
     * @protected */
   enlargeMain(action, skip_warning) {
      const main = this.selectDom(true),
            origin = this.selectDom('origin'),
            doc = getDocument();

      if (main.empty() || !settings.CanEnlarge || (origin.property('can_enlarge') === false)) return false;

      if ((action === undefined) || (action === 'verify')) return true;

      const state = origin.property('use_enlarge') ? 'on' : 'off';

      if (action === 'state') return state;

      if (action === 'toggle') action = (state === 'off');

      let enlarge = d3_select(doc.getElementById('jsroot_enlarge_div'));

      if ((action === true) && (state !== 'on')) {
         if (!enlarge.empty()) return false;

         enlarge = d3_select(doc.body)
            .append('div')
            .attr('id', 'jsroot_enlarge_div')
            .attr('style', 'position: fixed; margin: 0px; border: 0px; padding: 0px; left: 1px; top: 1px; bottom: 1px; right: 1px; background: white; opacity: 0.95; z-index: 100; overflow: hidden;');

         const rect1 = getElementRect(main),
               rect2 = getElementRect(enlarge);

         // if new enlarge area not big enough, do not do it
         if ((rect2.width <= rect1.width) || (rect2.height <= rect1.height)) {
            if (rect2.width * rect2.height < rect1.width * rect1.height) {
               if (!skip_warning)
                  console.log(`Enlarged area ${rect2.width} x ${rect2.height} smaller then original drawing ${rect1.width} x ${rect1.height}`);
               enlarge.remove();
               return false;
            }
         }

         while (main.node().childNodes.length > 0)
            enlarge.node().appendChild(main.node().firstChild);

         origin.property('use_enlarge', true);
         origin.property('did_enlarge', true);
         return true;
      }
      if ((action === false) && (state !== 'off')) {
         while (enlarge.node() && enlarge.node().childNodes.length > 0)
            main.node().appendChild(enlarge.node().firstChild);

         enlarge.remove();
         origin.property('use_enlarge', false);
         origin.property('did_enlarge', true);
         return true;
      }

      return false;
   }

   /** @summary Set item name, associated with the painter
     * @desc Used by {@link HierarchyPainter}
     * @private */
   setItemName(name, opt, hpainter) {
      if (isStr(name))
         this._hitemname = name;
      else
         delete this._hitemname;
      // only upate draw option, never delete.
      if (isStr(opt))
         this._hdrawopt = opt;

      this._hpainter = hpainter;
   }

   /** @summary Returns assigned item name
     * @desc Used with {@link HierarchyPainter} to identify drawn item name */
   getItemName() { return this._hitemname ?? null; }

   /** @summary Returns assigned item draw option
     * @desc Used with {@link HierarchyPainter} to identify drawn item option */
   getItemDrawOpt() { return this._hdrawopt ?? ''; }

} // class BasePainter

/** @summary Load and initialize JSDOM from nodes
  * @return {Promise} with d3 selection for d3_body
  * @private */
async function _loadJSDOM() {
   return import('jsdom').then(handle => {
      if (!internals.nodejs_window) {
         internals.nodejs_window = (new handle.JSDOM('<!DOCTYPE html>hello')).window;
         internals.nodejs_document = internals.nodejs_window.document; // used with three.js
         internals.nodejs_body = d3_select(internals.nodejs_document).select('body'); // get d3 handle for body
      }

      return { JSDOM: handle.JSDOM, doc: internals.nodejs_document, body: internals.nodejs_body };
   });
}

/** @summary Return translate string for transform attribute of some svg element
  * @return string or null if x and y are zeros
  * @private */
function makeTranslate(g, x, y) {
   if (!isObject(g)) {
      y = x; x = g; g = null;
   }
   const res = y ? `translate(${x},${y})` : (x ? `translate(${x})` : null);
   return g ? g.attr('transform', res) : res;
}


/** @summary Configure special style used for highlight or dragging elements
  * @private */
function addHighlightStyle(elem, drag) {
   if (drag) {
      elem.style('stroke', 'steelblue')
          .style('fill-opacity', '0.1');
   } else {
      elem.style('stroke', '#4572A7')
          .style('fill', '#4572A7')
          .style('opacity', '0');
   }
}

/** @summary Create pdf for existing SVG element
  * @return {Promise} with produced PDF file as url string
  * @private */
async function svgToPDF(args, as_buffer) {
   const nodejs = isNodeJs();
   let _jspdf, _svg2pdf, need_symbols = false;

   const pr = nodejs
      ? import('../../scripts/jspdf.es.min.js').then(h1 => { _jspdf = h1; return import('../../scripts/svg2pdf.es.min.js'); }).then(h2 => { _svg2pdf = h2; })
      : loadScript(source_dir + 'scripts/jspdf.umd.min.js').then(() => loadScript(source_dir + 'scripts/svg2pdf.umd.min.js')).then(() => { _jspdf = globalThis.jspdf; _svg2pdf = globalThis.svg2pdf; }),
        restore_fonts = [], restore_dominant = [], restore_text = [],
        node_transform = args.node.getAttribute('transform'), custom_fonts = {};

   if (args.reset_tranform)
      args.node.removeAttribute('transform');

   return pr.then(() => {
      d3_select(args.node).selectAll('g').each(function() {
         if (this.hasAttribute('font-family')) {
            const name = this.getAttribute('font-family');
            if (name === 'Courier New') {
               this.setAttribute('font-family', 'courier');
               if (!args.can_modify) restore_fonts.push(this); // keep to restore it
            }
         }
      });

      d3_select(args.node).selectAll('text').each(function() {
         if (this.hasAttribute('dominant-baseline')) {
            this.setAttribute('dy', '.2em'); // slightly different as in plain text
            this.removeAttribute('dominant-baseline');
            if (!args.can_modify) restore_dominant.push(this); // keep to restore it
         } else if (args.can_modify && nodejs && this.getAttribute('dy') === '.4em')
            this.setAttribute('dy', '.2em'); // better allignment in PDF

         if (replaceSymbolsInTextNode(this)) {
            need_symbols = true;
            if (!args.can_modify) restore_text.push(this); // keep to restore it
         }
      });

      if (nodejs) {
         const doc = internals.nodejs_document;
         doc.oldFunc = doc.createElementNS;
         globalThis.document = doc;
         globalThis.CSSStyleSheet = internals.nodejs_window.CSSStyleSheet;
         globalThis.CSSStyleRule = internals.nodejs_window.CSSStyleRule;
         doc.createElementNS = function(ns, kind) {
            const res = doc.oldFunc(ns, kind);
            res.getBBox = function() {
               let width = 50, height = 10;
               if (this.tagName === 'text') {
                  // TODO: use jsDOC fonts for label width estimation
                  const font = detectFont(this);
                  width = approximateLabelWidth(this.textContent, font);
                  height = font.size;
               }

               return { x: 0, y: 0, width, height };
            };
            return res;
         };
      }

      // eslint-disable-next-line new-cap
      const doc = new _jspdf.jsPDF({
         orientation: 'landscape',
         unit: 'px',
         format: [args.width + 10, args.height + 10]
      });

      // add custom fonts to PDF document, only TTF format supported
      d3_select(args.node).selectAll('style').each(function() {
         const fh = this.$fonthandler;
         if (!fh || custom_fonts[fh.name] || (fh.format !== 'ttf')) return;
         const filename = fh.name.toLowerCase().replace(/\s/g, '') + '.ttf';
         doc.addFileToVFS(filename, fh.base64);
         doc.addFont(filename, fh.name, 'normal', 'normal', (fh.name === 'symbol') ? 'StandardEncoding' : 'Identity-H');
         custom_fonts[fh.name] = true;
      });

      let pr2 = Promise.resolve(true);

      if (need_symbols && !custom_fonts.symbol) {
         if (!getCustomFont('symbol')) {
            pr2 = nodejs
              ? import('fs').then(fs => {
                 const base64 = fs.readFileSync('../../fonts/symbol.ttf').toString('base64');
                 console.log('reading symbol.ttf', base64.length);
                 addCustomFont(25, 'symbol', 'ttf', base64);
              })
              : httpRequest(source_dir+'fonts/symbol.ttf', 'bin').then(buf => {
               const base64 = btoa_func(buf);
               addCustomFont(25, 'symbol', 'ttf', base64);
            });
         }

         pr2 = pr2.then(() => {
            const fh = getCustomFont('symbol'),
                  handler = new FontHandler(1242, 10);
            handler.name = 'symbol';
            handler.base64 = fh.base64;
            handler.addCustomFontToSvg(d3_select(args.node));
            doc.addFileToVFS('symbol.ttf', fh.base64);
            doc.addFont('symbol.ttf', 'symbol', 'normal', 'normal', 'StandardEncoding' /* 'WinAnsiEncoding' */);
         });
      }

      return pr2.then(() => _svg2pdf.svg2pdf(args.node, doc, { x: 5, y: 5, width: args.width, height: args.height }))
         .then(() => {
            if (args.reset_tranform && !args.can_modify && node_transform)
               args.node.setAttribute('transform', node_transform);

            restore_fonts.forEach(node => node.setAttribute('font-family', 'Courier New'));
            restore_dominant.forEach(node => {
               node.setAttribute('dominant-baseline', 'middle');
               node.removeAttribute('dy');
            });

            restore_text.forEach(node => { node.innerHTML = node.$originalHTML; });

            const res = as_buffer ? doc.output('arraybuffer') : doc.output('dataurlstring');
            if (nodejs) {
               globalThis.document = undefined;
               globalThis.CSSStyleSheet = undefined;
               globalThis.CSSStyleRule = undefined;
               internals.nodejs_document.createElementNS = internals.nodejs_document.oldFunc;
               if (as_buffer) return Buffer.from(res);
            }
             return res;
         });
   });
}


/** @summary Create image based on SVG
  * @param {string} svg - svg code of the image
  * @param {string} [image_format] - image format like 'png', 'jpeg' or 'webp'
  * @param {boolean} [as_buffer] - return Buffer object for image
  * @return {Promise} with produced image in base64 form or as Buffer (or canvas when no image_format specified)
  * @private */
async function svgToImage(svg, image_format, as_buffer) {
   if (image_format === 'svg')
      return svg;

   if (image_format === 'pdf')
      return svgToPDF(svg, as_buffer);

   // required with df104.py/df105.py example with RCanvas or any special symbols in TLatex
   const doctype = '<?xml version="1.0" standalone="no"?><!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">';
   svg = encodeURIComponent(doctype + svg);
   svg = svg.replace(/%([0-9A-F]{2})/g, (match, p1) => {
       const c = String.fromCharCode('0x'+p1);
       return c === '%' ? '%25' : c;
   });
   svg = decodeURIComponent(svg);

   const img_src = 'data:image/svg+xml;base64,' + btoa_func(svg);

   if (isNodeJs()) {
      return import('canvas').then(async handle => {
         return handle.default.loadImage(img_src).then(img => {
            const canvas = handle.default.createCanvas(img.width, img.height);

            canvas.getContext('2d').drawImage(img, 0, 0, img.width, img.height);

            if (as_buffer) return canvas.toBuffer('image/' + image_format);

            return image_format ? canvas.toDataURL('image/' + image_format) : canvas;
         });
      });
   }

   return new Promise(resolveFunc => {
      const image = document.createElement('img');

      image.onload = function() {
         const canvas = document.createElement('canvas');
         canvas.width = image.width;
         canvas.height = image.height;

         canvas.getContext('2d').drawImage(image, 0, 0);

         if (as_buffer && image_format)
            canvas.toBlob(blob => blob.arrayBuffer().then(resolveFunc), 'image/' + image_format);
         else
            resolveFunc(image_format ? canvas.toDataURL('image/' + image_format) : canvas);
      };
      image.onerror = function(arg) {
         console.log(`IMAGE ERROR ${arg}`);
         resolveFunc(null);
      };

      image.src = img_src;
   });
}

/** @summary Convert Date object into string used preconfigured time zone
 * @desc Time zone stored in settings.TimeZone */
function convertDate(dt) {
   let res = '';
   if (settings.TimeZone && isStr(settings.TimeZone)) {
     try {
        res = dt.toLocaleString('en-GB', { timeZone: settings.TimeZone });
     } catch (err) {
        res = '';
     }
   }
   return res || dt.toLocaleString('en-GB');
}

export { getElementRect, getAbsPosInCanvas, convertDate,
         DrawOptions, TRandom, floatToString, buildSvgCurve, compressSVG,
         BasePainter, _loadJSDOM, makeTranslate, addHighlightStyle, svgToImage };
