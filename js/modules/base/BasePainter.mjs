import { select as d3_select } from '../d3.mjs';
import { settings, internals, isNodeJs } from '../core.mjs';


/** @summary Returns visible rect of element
  * @param {object} elem - d3.select object with element
  * @param {string} [kind] - which size method is used
  * @desc kind = 'bbox' use getBBox, works only with SVG
  * kind = 'full' - full size of element, using getBoundingClientRect function
  * kind = 'nopadding' - excludes padding area
  * With node.js can use "width" and "height" attributes when provided in element
  * @private */
function getElementRect(elem, sizearg) {
   if (isNodeJs() && (sizearg != 'bbox'))
      return { x: 0, y: 0, width: parseInt(elem.attr("width")), height: parseInt(elem.attr("height")) };

   const styleValue = name => {
      let value = elem.style(name);
      if (!value || (typeof value !== 'string')) return 0;
      value = parseFloat(value.replace("px", ""));
      return !Number.isFinite(value) ? 0 : Math.round(value);
   };

   let rect = elem.node().getBoundingClientRect();
   if ((sizearg == 'bbox') && (parseFloat(rect.width) > 0))
      rect = elem.node().getBBox();

   let res = { x: 0, y: 0, width: parseInt(rect.width), height: parseInt(rect.height) };
   if (rect.left !== undefined) {
      res.x = parseInt(rect.left);
      res.y = parseInt(rect.top);
   } else if (rect.x !== undefined) {
      res.x = parseInt(rect.x);
      res.y = parseInt(rect.y);
   }

   if ((sizearg === undefined) || (sizearg == 'nopadding')) {
      // this is size exclude padding area
      res.width -= styleValue('padding-left') + styleValue('padding-right');
      res.height -= styleValue('padding-top') + styleValue('padding-bottom');
   }

   return res;
}


/** @summary Calculate absolute position of provided element in canvas
  * @private */
function getAbsPosInCanvas(sel, pos) {
   while (!sel.empty() && !sel.classed('root_canvas') && pos) {
      let cl = sel.attr("class");
      if (cl && ((cl.indexOf("root_frame") >= 0) || (cl.indexOf("__root_pad_") >= 0))) {
         pos.x += sel.property("draw_x") || 0;
         pos.y += sel.property("draw_y") || 0;
      }
      sel = d3_select(sel.node().parentNode);
   }
   return pos;
}


/** @summary Converts numeric value to string according to specified format.
  * @param {number} value - value to convert
  * @param {string} [fmt="6.4g"] - format can be like 5.4g or 4.2e or 6.4f
  * @param {boolean} [ret_fmt] - when true returns array with value and actual format like ["0.1","6.4f"]
  * @returns {string|Array} - converted value or array with value and actual format
  * @private */
function floatToString(value, fmt, ret_fmt) {
   if (!fmt) fmt = "6.4g";

   fmt = fmt.trim();
   let len = fmt.length;
   if (len < 2)
      return ret_fmt ? [value.toFixed(4), "6.4f"] : value.toFixed(4);
   let last = fmt[len-1];
   fmt = fmt.slice(0,len-1);
   let isexp, prec = fmt.indexOf(".");
   prec = (prec < 0) ? 4 : parseInt(fmt.slice(prec+1));
   if (!Number.isInteger(prec) || (prec <= 0)) prec = 4;

   let significance = false;
   if ((last=='e') || (last=='E')) { isexp = true; } else
   if (last=='Q') { isexp = true; significance = true; } else
   if ((last=='f') || (last=='F')) { isexp = false; } else
   if (last=='W') { isexp = false; significance = true; } else
   if ((last=='g') || (last=='G')) {
      let se = floatToString(value, fmt+'Q', true),
          sg = floatToString(value, fmt+'W', true);

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

      let se = value.toExponential(prec);

      return ret_fmt ? [se, '5.'+prec+'e'] : se;
   }

   let sg = value.toFixed(prec);

   if (significance) {

      // when using fixed representation, one could get 0.0
      if ((value!=0) && (Number(sg)==0.) && (prec>0)) {
         prec = 20; sg = value.toFixed(prec);
      }

      let l = 0;
      while ((l<sg.length) && (sg[l] == '0' || sg[l] == '-' || sg[l] == '.')) l++;

      let diff = sg.length - l - prec;
      if (sg.indexOf(".")>l) diff--;

      if (diff != 0) {
         prec-=diff;
         if (prec<0) prec = 0; else if (prec>20) prec = 20;
         sg = value.toFixed(prec);
      }
   }

   return ret_fmt ? [sg, '5.'+prec+'f'] : sg;
}


/** @summary Draw options interpreter
  * @private */
class DrawOptions {

   constructor(opt) {
      this.opt = opt && (typeof opt == "string") ? opt.toUpperCase().trim() : "";
      this.part = "";
   }

   /** @summary Returns true if remaining options are empty or contain only seperators symbols. */
   empty() {
      if (this.opt.length === 0) return true;
      return this.opt.replace(/[ ;_,]/g,"").length == 0;
   }

   /** @summary Returns remaining part of the draw options. */
   remain() { return this.opt; }

   /** @summary Checks if given option exists */
   check(name, postpart) {
      let pos = this.opt.indexOf(name);
      if (pos < 0) return false;
      this.opt = this.opt.slice(0, pos) + this.opt.slice(pos + name.length);
      this.part = "";
      if (!postpart) return true;

      let pos2 = pos;
      while ((pos2 < this.opt.length) && (this.opt[pos2] !== ' ') && (this.opt[pos2] !== ',') && (this.opt[pos2] !== ';')) pos2++;
      if (pos2 > pos) {
         this.part = this.opt.slice(pos, pos2);
         this.opt = this.opt.slice(0, pos) + this.opt.slice(pos2);
      }
      return true;
   }

   /** @summary Returns remaining part of found option as integer. */
   partAsInt(offset, dflt) {
      let val = this.part.replace(/^\D+/g, '');
      val = val ? parseInt(val, 10) : Number.NaN;
      return !Number.isInteger(val) ? (dflt || 0) : val + (offset || 0);
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
      if (i!==undefined) this.seed(i);
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


/** @summary Function used to provide svg:path for the smoothed curves.
  * @desc reuse code from d3.js. Used in TH1, TF1 and TGraph painters
  * @param {string} kind  should contain "bezier" or "line".
  * If first symbol "L", then it used to continue drawing
  * @private */
function buildSvgPath(kind, bins, height, ndig) {

   const smooth = kind.indexOf("bezier") >= 0;

   if (ndig === undefined) ndig = smooth ? 2 : 0;
   if (height === undefined) height = 0;

   const jsroot_d3_svg_lineSlope = (p0, p1) => (p1.gry - p0.gry) / (p1.grx - p0.grx),
         jsroot_d3_svg_lineFiniteDifferences = points => {
      let i = 0, j = points.length - 1, m = [], p0 = points[0], p1 = points[1], d = m[0] = jsroot_d3_svg_lineSlope(p0, p1);
      while (++i < j) {
         p0 = p1; p1 = points[i + 1];
         m[i] = (d + (d = jsroot_d3_svg_lineSlope(p0, p1))) / 2;
      }
      m[i] = d;
      return m;
   }, jsroot_d3_svg_lineMonotoneTangents = points => {
      let d, a, b, s, m = jsroot_d3_svg_lineFiniteDifferences(points), i = -1, j = points.length - 1;
      while (++i < j) {
         d = jsroot_d3_svg_lineSlope(points[i], points[i + 1]);
         if (Math.abs(d) < 1e-6) {
            m[i] = m[i + 1] = 0;
         } else {
            a = m[i] / d;
            b = m[i + 1] / d;
            s = a * a + b * b;
            if (s > 9) {
               s = d * 3 / Math.sqrt(s);
               m[i] = s * a;
               m[i + 1] = s * b;
            }
         }
      }
      i = -1;
      while (++i <= j) {
         s = (points[Math.min(j, i + 1)].grx - points[Math.max(0, i - 1)].grx) / (6 * (1 + m[i] * m[i]));
         points[i].dgrx = s || 0;
         points[i].dgry = m[i] * s || 0;
      }
   };

   let res = { path: "", close: "" }, bin = bins[0], maxy = Math.max(bin.gry, height + 5),
      currx = Math.round(bin.grx), curry = Math.round(bin.gry), dx, dy, npnts = bins.length;

   const conv = val => {
      let vvv = Math.round(val);
      if ((ndig == 0) || (vvv === val)) return vvv.toString();
      let str = val.toFixed(ndig);
      while ((str[str.length - 1] == '0') && (str.lastIndexOf(".") < str.length - 1))
         str = str.slice(0, str.length - 1);
      if (str[str.length - 1] == '.')
         str = str.slice(0, str.length - 1);
      if (str == "-0") str = "0";
      return str;
   };

   res.path = ((kind[0] == "L") ? "L" : "M") + conv(bin.grx) + "," + conv(bin.gry);

   // just calculate all deltas, can be used to build exclusion
   if (smooth || kind.indexOf('calc') >= 0)
      jsroot_d3_svg_lineMonotoneTangents(bins);

   if (smooth) {
      // build smoothed curve
      res.path += `C${conv(bin.grx+bin.dgrx)},${conv(bin.gry+bin.dgry)},`;
      for (let n = 1; n < npnts; ++n) {
         let prev = bin;
         bin = bins[n];
         if (n > 1) res.path += "S";
         res.path += `${conv(bin.grx - bin.dgrx)},${conv(bin.gry - bin.dgry)},${conv(bin.grx)},${conv(bin.gry)}`;
         maxy = Math.max(maxy, prev.gry);
      }
   } else if (npnts < 10000) {
      // build simple curve

      let acc_x = 0, acc_y = 0;

      const flush = () => {
         if (acc_x) { res.path += "h" + acc_x; acc_x = 0; }
         if (acc_y) { res.path += "v" + acc_y; acc_y = 0; }
      };

      for (let n = 1; n < npnts; ++n) {
         bin = bins[n];
         dx = Math.round(bin.grx) - currx;
         dy = Math.round(bin.gry) - curry;
         if (dx && dy) {
            flush();
            res.path += `l${dx},${dy}`;
         } else if (!dx && dy) {
            if ((acc_y === 0) || ((dy < 0) !== (acc_y < 0))) flush();
            acc_y += dy;
         } else if (dx && !dy) {
            if ((acc_x === 0) || ((dx < 0) !== (acc_x < 0))) flush();
            acc_x += dx;
         }
         currx += dx; curry += dy;
         maxy = Math.max(maxy, curry);
      }

      flush();

   } else {
      // build line with trying optimize many vertical moves
      let lastx, lasty, cminy = curry, cmaxy = curry, prevy = curry;
      for (let n = 1; n < npnts; ++n) {
         bin = bins[n];
         lastx = Math.round(bin.grx);
         lasty = Math.round(bin.gry);
         maxy = Math.max(maxy, lasty);
         dx = lastx - currx;
         if (dx === 0) {
            // if X not change, just remember amplitude and
            cminy = Math.min(cminy, lasty);
            cmaxy = Math.max(cmaxy, lasty);
            prevy = lasty;
            continue;
         }

         if (cminy !== cmaxy) {
            if (cminy != curry) res.path += "v" + (cminy - curry);
            res.path += "v" + (cmaxy - cminy);
            if (cmaxy != prevy) res.path += "v" + (prevy - cmaxy);
            curry = prevy;
         }
         dy = lasty - curry;
         if (dy)
            res.path += `l${dx},${dy}`;
         else
            res.path += "h" + dx;
         currx = lastx; curry = lasty;
         prevy = cminy = cmaxy = lasty;
      }

      if (cminy != cmaxy) {
         if (cminy != curry) res.path += "v" + (cminy - curry);
         res.path += "v" + (cmaxy - cminy);
         if (cmaxy != prevy) res.path += "v" + (prevy - cmaxy);
      }
   }

   if (height > 0)
      res.close = `L${conv(bin.grx)},${conv(maxy)}h${conv(bins[0].grx - bin.grx)}Z`;

   return res;
}

/** @summary Compress SVG code, produced from drawing
  * @desc removes extra info or empty elements
  * @private */
function compressSVG(svg) {

   svg = svg.replace(/url\(\&quot\;\#(\w+)\&quot\;\)/g, "url(#$1)")        // decode all URL
            .replace(/ class=\"\w*\"/g, "")                                // remove all classes
            .replace(/ pad=\"\w*\"/g, "")                                  // remove all pad ids
            .replace(/ title=\"\"/g, "")                                   // remove all empty titles
            .replace(/<g objname=\"\w*\" objtype=\"\w*\"/g, "<g")          // remove object ids
            .replace(/<g transform=\"translate\(\d+\,\d+\)\"><\/g>/g, "")  // remove all empty groups with transform
            .replace(/<g><\/g>/g, "");                                     // remove all empty groups

   // remove all empty frame svgs, typically appears in 3D drawings, maybe should be improved in frame painter itself
   svg = svg.replace(/<svg x=\"0\" y=\"0\" overflow=\"hidden\" width=\"\d+\" height=\"\d+\" viewBox=\"0 0 \d+ \d+\"><\/svg>/g, "")

   if (svg.indexOf("xlink:href") < 0)
      svg = svg.replace(/ xmlns:xlink=\"http:\/\/www.w3.org\/1999\/xlink\"/g, "");

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
     * @returns {object} d3.select object for main element for drawing */
   selectDom(is_direct) {

      if (!this.divid) return d3_select(null);

      let res = this._selected_main;
      if (!res) {
         if (typeof this.divid == "string") {
            let id = this.divid;
            if (id[0] != '#') id = "#" + id;
            res = d3_select(id);
            if (!res.empty()) this.divid = res.node();
         } else {
            res = d3_select(this.divid);
         }
         this._selected_main = res;
      }

      if (!res || res.empty() || (is_direct === 'origin')) return res;

      let use_enlarge = res.property('use_enlarge'),
          layout = res.property('layout') || 'simple',
          layout_selector = (layout == 'simple') ? "" : res.property('layout_selector');

      if (layout_selector) res = res.select(layout_selector);

      // one could redirect here
      if (!is_direct && !res.empty() && use_enlarge) res = d3_select("#jsroot_enlarge_div");

      return res;
   }

   /** @summary Access/change top painter
     * @private */
   _accessTopPainter(on) {
      let chld = this.selectDom().node()?.firstChild;
      if (!chld) return null;
      if (on === true) {
         chld.painter = this;
      } else if (on === false)
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
      let origin = this.selectDom('origin');
      if (!origin.empty() && !keep_origin) origin.html("");
      this.divid = null;
      delete this._selected_main;

      if (typeof this._hpainter?.removePainter === 'function')
         this._hpainter.removePainter(this);

      delete this._hitemname;
      delete this._hdrawopt;
      delete this._hpainter;
   }

   /** @summary Checks if draw elements were resized and drawing should be updated
     * @returns {boolean} true if resize was detected
     * @protected
     * @abstract */
   checkResize(/* arg */) {}

   /** @summary Function checks if geometry of main div was changed.
     * @desc take into account enlarge state, used only in PadPainter class
     * @returns size of area when main div is drawn
     * @private */
   testMainResize(check_level, new_size, height_factor) {

      let enlarge = this.enlargeMain('state'),
          main_origin = this.selectDom('origin'),
          main = this.selectDom(),
          lmt = 5; // minimal size

      if (enlarge !== 'on') {
         if (new_size && new_size.width && new_size.height)
            main_origin.style('width', new_size.width + "px")
               .style('height', new_size.height + "px");
      }

      let rect_origin = getElementRect(main_origin, true),
         can_resize = main_origin.attr('can_resize'),
         do_resize = false;

      if (can_resize == "height")
         if (height_factor && Math.abs(rect_origin.width * height_factor - rect_origin.height) > 0.1 * rect_origin.width) do_resize = true;

      if (((rect_origin.height <= lmt) || (rect_origin.width <= lmt)) &&
         can_resize && can_resize !== 'false') do_resize = true;

      if (do_resize && (enlarge !== 'on')) {
         // if zero size and can_resize attribute set, change container size

         if (rect_origin.width > lmt) {
            height_factor = height_factor || 0.66;
            main_origin.style('height', Math.round(rect_origin.width * height_factor) + 'px');
         } else if (can_resize !== 'height') {
            main_origin.style('width', '200px').style('height', '100px');
         }
      }

      let rect = getElementRect(main),
          old_h = main.property('draw_height'),
          old_w = main.property('draw_width');

      rect.changed = false;

      if (old_h && old_w && (old_h > 0) && (old_w > 0)) {
         if ((old_h !== rect.height) || (old_w !== rect.width))
            if ((check_level > 1) || (rect.width / old_w < 0.66) || (rect.width / old_w > 1.5) ||
               (rect.height / old_h < 0.66) && (rect.height / old_h > 1.5)) rect.changed = true;
      } else {
         rect.changed = true;
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

      let main = this.selectDom(true),
          origin = this.selectDom('origin');

      if (main.empty() || !settings.CanEnlarge || (origin.property('can_enlarge') === false)) return false;

      if ((action === undefined) || (action === 'verify')) return true;

      let state = origin.property('use_enlarge') ? "on" : "off";

      if (action === 'state') return state;

      if (action === 'toggle') action = (state === "off");

      let enlarge = d3_select("#jsroot_enlarge_div");

      if ((action === true) && (state !== "on")) {
         if (!enlarge.empty()) return false;

         enlarge = d3_select(document.body)
            .append("div")
            .attr("id", "jsroot_enlarge_div")
            .attr("style", "position: fixed; margin: 0px; border: 0px; padding: 0px; left: 1px; right: 1px; top: 1px; bottom: 1px; background: white; opacity: 0.95; z-index: 100; overflow: hidden;");

         let rect1 = getElementRect(main),
             rect2 = getElementRect(enlarge);

         // if new enlarge area not big enough, do not do it
         if ((rect2.width <= rect1.width) || (rect2.height <= rect1.height))
            if (rect2.width * rect2.height < rect1.width * rect1.height) {
               if (!skip_warning)
                  console.log(`Enlarged area ${rect2.width} x ${rect2.height} smaller then original drawing ${rect1.width} x ${rect1.height}`);
               enlarge.remove();
               return false;
            }

         while (main.node().childNodes.length > 0)
            enlarge.node().appendChild(main.node().firstChild);

         origin.property('use_enlarge', true);

         return true;
      }
      if ((action === false) && (state !== "off")) {

         while (enlarge.node() && enlarge.node().childNodes.length > 0)
            main.node().appendChild(enlarge.node().firstChild);

         enlarge.remove();
         origin.property('use_enlarge', false);
         return true;
      }

      return false;
   }

   /** @summary Set item name, associated with the painter
     * @desc Used by {@link HierarchyPainter}
     * @private */
   setItemName(name, opt, hpainter) {
      if (typeof name === 'string')
         this._hitemname = name;
      else
         delete this._hitemname;
      // only upate draw option, never delete.
      if (typeof opt === 'string') this._hdrawopt = opt;

      this._hpainter = hpainter;
   }

   /** @summary Returns assigned item name
     * @desc Used with {@link HierarchyPainter} to identify drawn item name */
   getItemName() { return this._hitemname ?? null; }

   /** @summary Returns assigned item draw option
     * @desc Used with {@link HierarchyPainter} to identify drawn item option */
   getItemDrawOpt() { return this._hdrawopt ?? ""; }

} // class BasePainter

/** @summary Load and initialize JSDOM from nodes
  * @returns {Promise} with d3 selection for d3_body
   * @private */
function _loadJSDOM() {
   return import("jsdom").then(handle => {

      if (!internals.nodejs_window) {
         internals.nodejs_window = (new handle.JSDOM("<!DOCTYPE html>hello")).window;
         internals.nodejs_document = internals.nodejs_window.document; // used with three.js
         internals.nodejs_body = d3_select(internals.nodejs_document).select('body'); //get d3 handle for body
      }

      return { JSDOM: handle.JSDOM, doc: internals.nodejs_document, body: internals.nodejs_body };
   });
}

export { getElementRect, getAbsPosInCanvas,
         DrawOptions, TRandom, floatToString, buildSvgPath, compressSVG,
         BasePainter, _loadJSDOM };
