/// @file JSRoot.painter.js
/// Baisc JavaScript ROOT painter classes

JSROOT.define(['d3'], (d3) => {

   "use strict";

   JSROOT.loadScript('$$$style/JSRoot.painter');

   if ((typeof d3 !== 'object') || !d3.version)
      console.error('Fail to detect d3.js');
   else if (d3.version[0] !== "7")
      console.error(`Unsupported d3.js version ${d3.version}, expected 7.3.0`);
   else if (d3.version !== '7.3.0')
      console.log(`Reuse existing d3.js version ${d3.version}, expected 7.3.0`);

   // ==========================================================================================

   /** @summary Draw options interpreter
     * @memberof JSROOT
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
         this.opt = this.opt.substr(0, pos) + this.opt.substr(pos + name.length);
         this.part = "";
         if (!postpart) return true;

         let pos2 = pos;
         while ((pos2 < this.opt.length) && (this.opt[pos2] !== ' ') && (this.opt[pos2] !== ',') && (this.opt[pos2] !== ';')) pos2++;
         if (pos2 > pos) {
            this.part = this.opt.substr(pos, pos2 - pos);
            this.opt = this.opt.substr(0, pos) + this.opt.substr(pos2);
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
   }

   // ============================================================================================

   /** @summary Simple random generator with controlled seed
     * @memberof JSROOT
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
   }

   // ============================================================================================

   /** @namespace
     * @summary Collection of Painter-related methods and classes
     * @alias JSROOT.Painter */
   let jsrp = {
      Coord: {
         kCARTESIAN: 1,
         kPOLAR: 2,
         kCYLINDRICAL: 3,
         kSPHERICAL: 4,
         kRAPIDITY: 5
      },
      root_colors: [],
      root_line_styles: ["", "", "3,3", "1,2",
         "3,4,1,4", "5,3,1,3", "5,3,1,3,1,3,1,3", "5,5",
         "5,3,1,3,1,3", "20,5", "20,10,1,10", "1,3"],
      root_markers: [
         0, 1, 2, 3, 4,           //  0..4
         5, 106, 107, 104, 1,     //  5..9
         1, 1, 1, 1, 1,           // 10..14
         1, 1, 1, 1, 1,           // 15..19
         104, 125, 126, 132, 4,   // 20..24
         25, 26, 27, 28, 130,     // 25..29
         30, 3, 32, 127, 128,     // 30..34
         35, 36, 37, 38, 137,     // 35..39
         40, 140, 42, 142, 44,    // 40..44
         144, 46, 146, 148, 149], // 45..49
      root_fonts: ['Arial', 'iTimes New Roman',
         'bTimes New Roman', 'biTimes New Roman', 'Arial',
         'oArial', 'bArial', 'boArial', 'Courier New',
         'oCourier New', 'bCourier New', 'boCourier New',
         'Symbol', 'Times New Roman', 'Wingdings', 'iSymbol',
         'Verdana', 'iVerdana', 'bVerdana', 'biVerdana'],
      // taken from symbols.html, counted only for letters and digits
    root_fonts_aver_width: [0.5778,0.5314,
         0.5809, 0.5540, 0.5778,
         0.5783,0.6034,0.6030,0.6003,
         0.6004,0.6003,0.6005,
         0.5521,0.5521,0.5664,0.5314,
         0.5664,0.5495,0.5748,0.5578]
   };

   /** @summary Check if object is a Promise
     * @memberof JSROOT.Painter
     * @private */
   function isPromise(obj) {
      return obj && (typeof obj == 'object') && (typeof obj.then == 'function');
   }

   /** @summary Covert value between 0 and 1 into hex, used for colors coding
     * @memberof JSROOT.Painter
     * @private */
   function toHex(num,scale) {
      let s = Math.round(num*(scale || 255)).toString(16);
      return s.length == 1 ? '0'+s : s;
   }

   jsrp.createMenu = function(evnt, handler, menuname) {
      document.body.style.cursor = 'wait';
      let show_evnt;
      // copy event values, otherwise they will gone after scripts loading
      if (evnt && (typeof evnt == "object"))
         if ((evnt.clientX !== undefined) && (evnt.clientY !== undefined))
            show_evnt = { clientX: evnt.clientX, clientY: evnt.clientY };
      return JSROOT.require(['menu']).then(() => {
         document.body.style.cursor = 'auto';
         return jsrp.createMenu(show_evnt, handler, menuname);
      });
   }

   jsrp.closeMenu = function(menuname) {
      JSROOT.require(['menu']).then(() => {
         jsrp.closeMenu(menuname);
      });
   }

   /** @summary Read style and settings from URL
     * @private */
   jsrp.readStyleFromURL = function(url) {
      let d = JSROOT.decodeUrl(url), g = JSROOT.gStyle, s = JSROOT.settings;

      if (d.has("optimize")) {
         s.OptimizeDraw = 2;
         let optimize = d.get("optimize");
         if (optimize) {
            optimize = parseInt(optimize);
            if (Number.isInteger(optimize)) s.OptimizeDraw = optimize;
         }
      }

      let inter = d.get("interactive");
      if (inter === "nomenu")
         s.ContextMenu = false;
      else if (inter !== undefined) {
         if (!inter || (inter == "1")) inter = "111111"; else
            if (inter == "0") inter = "000000";
         if (inter.length === 6) {
            switch(inter[0]) {
               case "0": g.ToolBar = false; break;
               case "1": g.ToolBar = 'popup'; break;
               case "2": g.ToolBar = true; break;
            }
            inter = inter.substr(1);
         }
         if (inter.length == 5) {
            s.Tooltip = parseInt(inter[0]);
            s.ContextMenu = (inter[1] != '0');
            s.Zooming = (inter[2] != '0');
            s.MoveResize = (inter[3] != '0');
            s.DragAndDrop = (inter[4] != '0');
         }
      }

      let tt = d.get("tooltip");
      if ((tt == "off") || (tt == "false") || (tt == "0"))
         s.Tooltip = false;
      else if (d.has("tooltip"))
         s.Tooltip = true;

      if (d.has("bootstrap") || d.has("bs"))
         s.Bootstrap = true;

      // s.Bootstrap = true;

      let mathjax = d.get("mathjax", null), latex = d.get("latex", null);

      if ((mathjax !== null) && (mathjax != "0") && (latex === null)) latex = "math";
      if (latex !== null)
         s.Latex = JSROOT.constants.Latex.fromString(latex);

      if (d.has("nomenu")) s.ContextMenu = false;
      if (d.has("noprogress")) s.ProgressBox = false;
      if (d.has("notouch")) JSROOT.browser.touches = false;
      if (d.has("adjframe")) s.CanAdjustFrame = true;

      let optstat = d.get("optstat"), optfit = d.get("optfit");
      if (optstat) g.fOptStat = parseInt(optstat);
      if (optfit) g.fOptFit = parseInt(optfit);
      g.fStatFormat = d.get("statfmt", g.fStatFormat);
      g.fFitFormat = d.get("fitfmt", g.fFitFormat);

      if (d.has("toolbar")) {
         let toolbar = d.get("toolbar", ""), val = null;
         if (toolbar.indexOf('popup') >= 0) val = 'popup';
         if (toolbar.indexOf('left') >= 0) { s.ToolBarSide = 'left'; val = 'popup'; }
         if (toolbar.indexOf('right') >= 0) { s.ToolBarSide = 'right'; val = 'popup'; }
         if (toolbar.indexOf('vert') >= 0) { s.ToolBarVert = true; val = 'popup'; }
         if (toolbar.indexOf('show') >= 0) val = true;
         s.ToolBar = val || ((toolbar.indexOf("0") < 0) && (toolbar.indexOf("false") < 0) && (toolbar.indexOf("off") < 0));
      }

      if (d.has("skipsi") || d.has("skipstreamerinfos"))
         s.SkipStreamerInfos = true;

      if (d.has("nodraggraphs"))
         s.DragGraphs = false;

      if (d.has("palette")) {
         let palette = parseInt(d.get("palette"));
         if (Number.isInteger(palette) && (palette > 0) && (palette < 113)) s.Palette = palette;
      }

      let render3d = d.get("render3d"), embed3d = d.get("embed3d"),
          geosegm = d.get("geosegm"), geocomp = d.get("geocomp");
      if (render3d) s.Render3D = JSROOT.constants.Render3D.fromString(render3d);
      if (embed3d) s.Embed3D = JSROOT.constants.Embed3D.fromString(embed3d);
      if (geosegm) s.GeoGradPerSegm = Math.max(2, parseInt(geosegm));
      if (geocomp) s.GeoCompressComp = (geocomp !== '0') && (geocomp !== 'false') && (geocomp !== 'off');

      if (d.has("hlimit")) s.HierarchyLimit = parseInt(d.get("hlimit"));
   }

   /** @summary Generates all root colors, used also in jstests to reset colors
     * @private */
   jsrp.createRootColors = function() {
      let colorMap = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', '#59d454', '#5954d9', 'white'];
      colorMap[110] = 'white';

      const moreCol = [
         { n: 11, s: 'c1b7ad4d4d4d6666668080809a9a9ab3b3b3cdcdcde6e6e6f3f3f3cdc8accdc8acc3c0a9bbb6a4b3a697b8a49cae9a8d9c8f83886657b1cfc885c3a48aa9a1839f8daebdc87b8f9a768a926983976e7b857d9ad280809caca6c0d4cf88dfbb88bd9f83c89a7dc08378cf5f61ac8f94a6787b946971d45a549300ff7b00ff6300ff4b00ff3300ff1b00ff0300ff0014ff002cff0044ff005cff0074ff008cff00a4ff00bcff00d4ff00ecff00fffd00ffe500ffcd00ffb500ff9d00ff8500ff6d00ff5500ff3d00ff2600ff0e0aff0022ff003aff0052ff006aff0082ff009aff00b1ff00c9ff00e1ff00f9ff00ffef00ffd700ffbf00ffa700ff8f00ff7700ff6000ff4800ff3000ff1800ff0000' },
         { n: 201, s: '5c5c5c7b7b7bb8b8b8d7d7d78a0f0fb81414ec4848f176760f8a0f14b81448ec4876f1760f0f8a1414b84848ec7676f18a8a0fb8b814ecec48f1f1768a0f8ab814b8ec48ecf176f10f8a8a14b8b848ecec76f1f1' },
         { n: 390, s: 'ffffcdffff9acdcd9affff66cdcd669a9a66ffff33cdcd339a9a33666633ffff00cdcd009a9a00666600333300' },
         { n: 406, s: 'cdffcd9aff9a9acd9a66ff6666cd66669a6633ff3333cd33339a3333663300ff0000cd00009a00006600003300' },
         { n: 422, s: 'cdffff9affff9acdcd66ffff66cdcd669a9a33ffff33cdcd339a9a33666600ffff00cdcd009a9a006666003333' },
         { n: 590, s: 'cdcdff9a9aff9a9acd6666ff6666cd66669a3333ff3333cd33339a3333660000ff0000cd00009a000066000033' },
         { n: 606, s: 'ffcdffff9affcd9acdff66ffcd66cd9a669aff33ffcd33cd9a339a663366ff00ffcd00cd9a009a660066330033' },
         { n: 622, s: 'ffcdcdff9a9acd9a9aff6666cd66669a6666ff3333cd33339a3333663333ff0000cd00009a0000660000330000' },
         { n: 791, s: 'ffcd9acd9a669a66339a6600cd9a33ffcd66ff9a00ffcd33cd9a00ffcd00ff9a33cd66006633009a3300cd6633ff9a66ff6600ff6633cd3300ff33009aff3366cd00336600339a0066cd339aff6666ff0066ff3333cd0033ff00cdff9a9acd66669a33669a009acd33cdff669aff00cdff339acd00cdff009affcd66cd9a339a66009a6633cd9a66ffcd00ff6633ffcd00cd9a00ffcd33ff9a00cd66006633009a3333cd6666ff9a00ff9a33ff6600cd3300ff339acdff669acd33669a00339a3366cd669aff0066ff3366ff0033cd0033ff339aff0066cd00336600669a339acd66cdff009aff33cdff009acd00cdffcd9aff9a66cd66339a66009a9a33cdcd66ff9a00ffcd33ff9a00cdcd00ff9a33ff6600cd33006633009a6633cd9a66ff6600ff6633ff3300cd3300ffff339acd00666600339a0033cd3366ff669aff0066ff3366cd0033ff0033ff9acdcd669a9a33669a0066cd339aff66cdff009acd009aff33cdff009a' },
         { n: 920, s: 'cdcdcd9a9a9a666666333333' }];

      moreCol.forEach(entry => {
         let s = entry.s;
         for (let n = 0; n < s.length; n += 6) {
            let num = entry.n + n / 6;
            colorMap[num] = '#' + s.substr(n,6);
         }
      });

      jsrp.root_colors = colorMap;
   }

   /** @summary Produces rgb code for TColor object
     * @private */
   jsrp.getRGBfromTColor = function(col) {
      if (!col || (col._typename != 'TColor')) return null;

      let rgb = '#' + toHex(col.fRed) + toHex(col.fGreen) + toHex(col.fBlue);
      if ((col.fAlpha !==undefined) && (col.fAlpha !== 1.))
         rgb += toHex(col.fAlpha);

      switch (rgb) {
         case '#ffffff': return 'white';
         case '#000000': return 'black';
         case '#ff0000': return 'red';
         case '#00ff00': return 'green';
         case '#0000ff': return 'blue';
         case '#ffff00': return 'yellow';
         case '#ff00ff': return 'magenta';
         case '#00ffff': return 'cyan';
      }
      return rgb;
   }

   /** @summary Add new colors from object array
     * @private */
   jsrp.extendRootColors = function(jsarr, objarr) {
      if (!jsarr) {
         jsarr = [];
         for (let n = 0; n < jsrp.root_colors.length; ++n)
            jsarr[n] = jsrp.root_colors[n];
      }

      if (!objarr) return jsarr;

      let rgb_array = objarr;
      if (objarr._typename && objarr.arr) {
         rgb_array = [];
         for (let n = 0; n < objarr.arr.length; ++n) {
            let col = objarr.arr[n];
            if (!col || (col._typename != 'TColor')) continue;

            if ((col.fNumber >= 0) && (col.fNumber <= 10000))
               rgb_array[col.fNumber] = jsrp.getRGBfromTColor(col);
         }
      }

      for (let n = 0; n < rgb_array.length; ++n)
         if (rgb_array[n] && (jsarr[n] != rgb_array[n]))
            jsarr[n] = rgb_array[n];

      return jsarr;
   }

   /** @ummary Set global list of colors.
     * @desc Either TObjArray of TColor instances or just plain array with rgb() code.
     * List of colors typically stored together with TCanvas primitives
     * @private */
   jsrp.adoptRootColors = function(objarr) {
      jsrp.extendRootColors(jsrp.root_colors, objarr);
   }

   /** @summary Return ROOT color by index
     * @desc Color numbering corresponds typical ROOT colors
     * @returns {String} with RGB color code or existing color name like 'cyan' */
   jsrp.getColor = function(indx) {
      return jsrp.root_colors[indx];
   }

   /** @summary Add new color
     * @param {string} rgb - color name or just string with rgb value
     * @param {array} [lst] - optional colors list, to which add colors
     * @returns {number} index of new color */
   jsrp.addColor = function(rgb, lst) {
      if (!lst) lst = jsrp.root_colors;
      let indx = lst.indexOf(rgb);
      if (indx >= 0) return indx;
      lst.push(rgb);
      return lst.length-1;
   }

   // =====================================================================

   /**
    * @summary Color palette handle
    *
    * @memberof JSROOT
    * @private
    */

   class ColorPalette {

      /** @summary constructor */
      constructor(arr) {
         this.palette = arr;
      }

      /** @summary Returns color index which correspond to contour index of provided length */
      calcColorIndex(i, len) {
         let plen = this.palette.length, theColor = Math.floor((i + 0.99) * plen / (len - 1));
         return (theColor > plen - 1) ? plen - 1 : theColor;
       }

      /** @summary Returns color with provided index */
      getColor(indx) { return this.palette[indx]; }

      /** @summary Returns number of colors in the palette */
      getLength() { return this.palette.length; }

      /** @summary Calculate color for given i and len */
      calcColor(i, len) { return this.getColor(this.calcColorIndex(i, len)); }

   }

   // =============================================================================

   /**
     * @summary Handle for marker attributes
     *
     * @memberof JSROOT
     * @private
     */

   class TAttMarkerHandler {

      /** @summary constructor
        * @param {object} args - attributes, see {@link JSROOT.TAttMarkerHandler#setArgs} for details */
      constructor(args) {
         this.x0 = this.y0 = 0;
         this.color = 'black';
         this.style = 1;
         this.size = 8;
         this.scale = 1;
         this.stroke = true;
         this.fill = true;
         this.marker = "";
         this.ndig = 0;
         this.used = true;
         this.changed = false;
         this.func = this.apply.bind(this);
         this.setArgs(args);
         this.changed = false;
      }

      /** @summary Set marker attributes.
        * @param {object} args - arguments can be
        * @param {object} args.attr - instance of TAttrMarker (or derived class) or
        * @param {string} args.color - color in HTML form like grb(1,4,5) or 'green'
        * @param {number} args.style - marker style
        * @param {number} args.size - marker size
        * @param {number} [args.refsize] - when specified and marker size < 1, marker size will be calculated relative to that size */
      setArgs(args) {
         if ((typeof args == 'object') && (typeof args.fMarkerStyle == 'number')) args = { attr: args };

         if (args.attr) {
            if (args.color === undefined)
               args.color = args.painter ? args.painter.getColor(args.attr.fMarkerColor) : jsrp.getColor(args.attr.fMarkerColor);
            if (!args.style || (args.style < 0)) args.style = args.attr.fMarkerStyle;
            if (!args.size) args.size = args.attr.fMarkerSize;
         }

         this.color = args.color;
         this.style = args.style;
         this.size = args.size;
         this.refsize = args.refsize;

         this._configure();
      }

      /** @summary Reset position, used for optimization of drawing of multiple markers
       * @private */
      resetPos() { this.lastx = this.lasty = null; }

      /** @summary Create marker path for given position.
        * @desc When drawing many elementary points, created path may depend from previously produced markers.
        * @param {number} x - first coordinate
        * @param {number} y - second coordinate
        * @returns {string} path string */
      create(x, y) {
         if (!this.optimized)
            return `M${(x + this.x0).toFixed(this.ndig)},${(y + this.y0).toFixed(this.ndig)}${this.marker}`;

         // use optimized handling with relative position
         let xx = Math.round(x), yy = Math.round(y), mv = `M${xx},${yy}`;
         if (this.lastx !== null) {
            if ((xx == this.lastx) && (yy == this.lasty)) {
               mv = ""; // pathological case, but let exclude it
            } else {
               let m2 = `m${xx-this.lastx},${yy - this.lasty}`;
               if (m2.length < mv.length) mv = m2;
            }
         }
         this.lastx = xx + 1; this.lasty = yy;
         return mv + "h1";
      }

      /** @summary Returns full size of marker */
      getFullSize() { return this.scale * this.size; }

      /** @summary Returns approximate length of produced marker string */
      getMarkerLength() { return this.marker ? this.marker.length : 10; }

      /** @summary Change marker attributes.
       *  @param {string} color - marker color
       *  @param {number} style - marker style
       *  @param {number} size - marker size */
      change(color, style, size) {
         this.changed = true;

         if (color !== undefined) this.color = color;
         if ((style !== undefined) && (style >= 0)) this.style = style;
         if (size !== undefined) this.size = size;

         this._configure();
      }

      /** @summary Prepare object to create marker
        * @private */
       _configure() {

         this.x0 = this.y0 = 0;

         if ((this.style === 1) || (this.style === 777)) {
            this.fill = false;
            this.marker = "h1";
            this.size = 1;
            this.optimized = true;
            this.resetPos();
            return true;
         }

         this.optimized = false;

         let marker_kind = jsrp.root_markers[this.style];
         if (marker_kind === undefined) marker_kind = 104;
         let shape = marker_kind % 100;

         this.fill = (marker_kind >= 100);

         this.scale = this.refsize || 8; // v7 defines refsize as 1 or pad height

         let size = this.getFullSize();

         this.ndig = (size > 7) ? 0 : ((size > 2) ? 1 : 2);
         if (shape == 30) this.ndig++; // increase precision for star
         let s1 = size.toFixed(this.ndig),
             s2 = (size/2).toFixed(this.ndig),
             s3 = (size/3).toFixed(this.ndig),
             s4 = (size/4).toFixed(this.ndig),
             s8 = (size/8).toFixed(this.ndig),
             s38 = (size*3/8).toFixed(this.ndig);

         switch (shape) {
            case 1: // dot
               this.marker = "h1";
               break;
            case 2: // plus
               this.y0 = -size / 2;
               this.marker = `v${s1}m-${s2},-${s2}h${s1}`;
               break;
            case 3: // asterisk
               this.x0 = this.y0 = -size / 2;
               this.marker = `l${s1},${s1}m0,-${s1}l-${s1},${s1}m0,-${s2}h${s1}m-${s2},-${s2}v${s1}`;
               break;
            case 4: // circle
               this.x0 = -parseFloat(s2);
               s1 = (parseFloat(s2) * 2).toFixed(this.ndig);
               this.marker = `a${s2},${s2},0,1,0,${s1},0a${s2},${s2},0,1,0,-${s1},0z`;
               break;
            case 5: // mult
               this.x0 = this.y0 = -size / 2;
               this.marker = `l${s1},${s1}m0,-${s1}l-${s1},${s1}`;
               break;
            case 6: // small dot
               this.x0 = -1;
               this.marker = "a1,1,0,1,0,2,0a1,1,0,1,0,-2,0z";
               break;
            case 7: // medium dot
               this.x0 = -1.5;
               this.marker = "a1.5,1.5,0,1,0,3,0a1.5,1.5,0,1,0,-3,0z";
               break;
            case 25: // square
               this.x0 = this.y0 = -size / 2;
               this.marker = `v${s1}h${s1}v-${s1}z`;
               break;
            case 26: // triangle-up
               this.y0 = -size / 2;
               this.marker = `l-${s2},${s1}h${s1}z`;
               break;
            case 27: // diamand
               this.y0 = -size / 2;
               this.marker = `l${s3},${s2}l-${s3},${s2}l-${s3},-${s2}z`;
               break;
            case 28: // cross
               this.x0 = this.y0 = size / 6;
               this.marker = `h${s3}v-${s3}h-${s3}v-${s3}h-${s3}v${s3}h-${s3}v${s3}h${s3}v${s3}h${s3}z`;
               break;
            case 30: // star
               this.y0 = -size / 2;
               let s56 = (size*5/6).toFixed(this.ndig), s58 = (size*5/8).toFixed(this.ndig);
               this.marker = `l${s3},${s1}l-${s56},-${s58}h${s1}l-${s56},${s58}z`;
               break;
            case 32: // triangle-down
               this.y0 = size / 2;
               this.marker = `l-${s2},-${s1}h${s1}z`;
               break;
            case 35:
               this.x0 = -size / 2;
               this.marker = `l${s2},${s2}l${s2},-${s2}l-${s2},-${s2}zh${s1}m-${s2},-${s2}v${s1}`;
               break;
            case 36:
               this.x0 = this.y0 = -size / 2;
               this.marker = `h${s1}v${s1}h-${s1}zl${s1},${s1}m0,-${s1}l-${s1},${s1}`;
               break;
            case 37:
               this.x0 = -size/2;
               this.marker = `h${s1}l-${s4},-${s2}l-${s2},${s1}h${s2}l-${s2},-${s1}z`;
               break;
            case 38:
               this.x0 = -size/4; this.y0 = -size/2;
               this.marker = `h${s2}l${s4},${s4}v${s2}l-${s4},${s4}h-${s2}l-${s4},-${s4}v-${s2}zm${s4},0v${s1}m-${s2},-${s2}h${s1}`;
               break;
            case 40:
               this.x0 = -size/4; this.y0 = -size/2;
               this.marker = `l${s2},${s1}l${s4},-${s4}l-${s1},-${s2}zm${s2},0l-${s2},${s1}l-${s4},-${s4}l${s1},-${s2}z`;
               break;
            case 42:
               this.y0 = -size/2;
               this.marker = `l${s8},${s38}l${s38},${s8}l-${s38},${s8}l-${s8},${s38}l-${s8},-${s38}l-${s38},-${s8}l${s38},-${s8}z`;
               break;
            case 44:
               this.x0 = -size/4; this.y0 = -size/2;
               this.marker = `h${s2}l-${s8},${s38}l${s38},-${s8}v${s2}l-${s38},-${s8}l${s8},${s38}h-${s2}l${s8},-${s38}l-${s38},${s8}v-${s2}l${s38},${s8}z`;
               break;
            case 46:
               this.x0 = -size/4; this.y0 = -size/2;
               this.marker = `l${s4},${s4}l${s4},-${s4}l${s4},${s4}l-${s4},${s4}l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}l-${s4},${s4}l-${s4},-${s4}l${s4},-${s4}l-${s4},-${s4}z`;
               break;
            case 48:
               this.x0 = -size/4; this.y0 = -size/2;
               this.marker = `l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}zm${s2},0l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}zm0,${s2}l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}zm-${s2},0l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}z`;
               break;
            case 49:
               this.x0 = -size/6; this.y0 = -size/2;
               this.marker = `h${s3}v${s3}h-${s3}zm${s3},${s3}h${s3}v${s3}h-${s3}zm-${s3},${s3}h${s3}v${s3}h-${s3}zm-${s3},-${s3}h${s3}v${s3}h-${s3}z`;
               break;
            default: // diamand
               this.y0 = -size / 2;
               this.marker = `l${s3},${s2}l-${s3},${s2}l-${s3},-${s2}z`;
               break;
         }

         return true;
      }

      /** @summary get stroke color */
      getStrokeColor() { return this.stroke ? this.color : "none"; }

      /** @summary get fill color */
      getFillColor() { return this.fill ? this.color : "none"; }

      /** @summary returns true if marker attributes will produce empty (invisible) output */
      empty() {
         return (this.color === 'none') || (!this.fill && !this.stroke);
      }

      /** @summary Apply marker styles to created element */
      apply(selection) {
         selection.style('stroke', this.stroke ? this.color : "none")
                  .style('fill', this.fill ? this.color : "none");
      }

      /** @summary Method used when color or pattern were changed with OpenUi5 widgets.
       * @private */
      verifyDirectChange(/* painter */) {
         this.change(this.color, parseInt(this.style), parseFloat(this.size));
      }

      /** @summary Create sample with marker in given SVG element
        * @param {selection} svg - SVG element
        * @param {number} width - width of sample SVG
        * @param {number} height - height of sample SVG
        * @private */
      createSample(svg, width, height) {
         this.resetPos();

         svg.append("path")
            .attr("d", this.create(width / 2, height / 2))
            .call(this.func);
      }

   }

   // =======================================================================

   /**
     * @summary Handle for line attributes
     *
     * @memberof JSROOT
     * @private
     */

   class TAttLineHandler {

      /** @summary constructor
        * @param {object} attr - attributes, see {@link JSROOT.TAttLineHandler#setArgs} */
      constructor(args) {
         this.func = this.apply.bind(this);
         this.used = true;
         if (args._typename && (args.fLineStyle !== undefined)) args = { attr: args };
         this.setArgs(args);
      }

      /** @summary Set line attributes.
        * @param {object} args - specify attributes by different ways
        * @param {object} args.attr - TAttLine object with appropriate data members or
        * @param {string} args.color - color in html like rgb(255,0,0) or "red" or "#ff0000"
        * @param {number} args.style - line style number
        * @param {number} args.width - line width */
      setArgs(args) {
         if (args.attr) {
            args.color = args.color0 || (args.painter ? args.painter.getColor(args.attr.fLineColor) : jsrp.getColor(args.attr.fLineColor));
            if (args.width === undefined) args.width = args.attr.fLineWidth;
            if (args.style === undefined) args.style = args.attr.fLineStyle;
         } else if (typeof args.color == 'string') {
            if ((args.color !== 'none') && !args.width) args.width = 1;
         } else if (typeof args.color == 'number') {
            args.color = args.painter ? args.painter.getColor(args.color) : jsrp.getColor(args.color);
         }

         if (args.width === undefined)
            args.width = (args.color && args.color != 'none') ? 1 : 0;

         this.color = (args.width === 0) ? 'none' : args.color;
         this.width = args.width;
         this.style = args.style;
         this.pattern = args.pattern || jsrp.root_line_styles[this.style] || null;

         if (args.can_excl) {
            this.excl_side = this.excl_width = 0;
            if (Math.abs(this.width) > 99) {
               // exclusion graph
               this.excl_side = (this.width < 0) ? -1 : 1;
               this.excl_width = Math.floor(this.width / 100) * 5;
               this.width = Math.abs(this.width % 100); // line width
            }
         }

         // if custom color number used, use lightgrey color to show lines
         if (!this.color && (this.width > 0))
            this.color = 'lightgrey';
      }

      /** @summary Change exclusion attributes */
      changeExcl(side, width) {
         if (width !== undefined)
            this.excl_width = width;
         if (side !== undefined) {
            this.excl_side = side;
            if ((this.excl_width === 0) && (this.excl_side !== 0)) this.excl_width = 20;
         }
         this.changed = true;
      }

      /** @summary returns true if line attribute is empty and will not be applied. */
      empty() { return this.color == 'none'; }

      /** @summary set border parameters, used for rect drawing */
      setBorder(rx, ry) {
         this.rx = rx;
         this.ry = ry;
         this.func = this.applyBorder.bind(this);
      }

      /** @summary Applies line attribute to selection.
        * @param {object} selection - d3.js selection */
      apply(selection) {
         this.used = true;
         if (this.empty())
            selection.style('stroke', null)
                     .style('stroke-width', null)
                     .style('stroke-dasharray', null);
         else
            selection.style('stroke', this.color)
                     .style('stroke-width', this.width)
                     .style('stroke-dasharray', this.pattern);
      }

      /** @summary Applies line and border attribute to selection.
        * @param {object} selection - d3.js selection */
      applyBorder(selection) {
         this.used = true;
         if (this.empty())
            selection.style('stroke', null)
                     .style('stroke-width', null)
                     .style('stroke-dasharray', null)
                     .attr("rx", null).attr("ry", null);
         else
            selection.style('stroke', this.color)
                     .style('stroke-width', this.width)
                     .style('stroke-dasharray', this.pattern)
                     .attr("rx", this.rx || null).attr("ry", this.ry || null);
      }

      /** @summary Change line attributes */
      change(color, width, style) {
         if (color !== undefined) this.color = color;
         if (width !== undefined) this.width = width;
         if (style !== undefined) {
            this.style = style;
            this.pattern = jsrp.root_line_styles[this.style] || null;
         }
         this.changed = true;
      }

      /** @summary Create sample element inside primitive SVG - used in context menu */
      createSample(svg, width, height) {
         svg.append("path")
            .attr("d", `M0,${height/2}h${width}`)
            .call(this.func);
      }
   }

   // =======================================================================

   /**
     * @summary Handle for fill attributes
     *
     * @memberof JSROOT
     * @private
     */

   class TAttFillHandler {
      /** @summary constructor
        * @param {object} args - arguments see {@link JSROOT.TAttFillHandler#setArgs} for more info
        * @param {number} [args.kind = 2] - 1 means object drawing where combination fillcolor==0 and fillstyle==1001 means no filling,  2 means all other objects where such combination is white-color filling */
      constructor(args) {
         this.color = "none";
         this.colorindx = 0;
         this.pattern = 0;
         this.used = true;
         this.kind = args.kind || 2;
         this.changed = false;
         this.func = this.apply.bind(this);
         this.setArgs(args);
         this.changed = false; // unset change property that
      }

      /** @summary Set fill style as arguments
        * @param {object} args - different arguments to set fill attributes
        * @param {object} [args.attr] - TAttFill object
        * @param {number} [args.color] - color id
        * @param {number} [args.pattern] - filll pattern id
        * @param {object} [args.svg] - SVG element to store newly created patterns
        * @param {string} [args.color_as_svg] - color in SVG format */
      setArgs(args) {
         if (args.attr && (typeof args.attr == 'object')) {
            if ((args.pattern === undefined) && (args.attr.fFillStyle !== undefined)) args.pattern = args.attr.fFillStyle;
            if ((args.color === undefined) && (args.attr.fFillColor !== undefined)) args.color = args.attr.fFillColor;
         }

         let was_changed = this.changed; // preserve changed state
         this.change(args.color, args.pattern, args.svg, args.color_as_svg, args.painter);
         this.changed = was_changed;
      }

      /** @summary Apply fill style to selection */
      apply(selection) {
         this.used = true;

         selection.style('fill', this.getFillColor());

         if ('opacity' in this)
            selection.style('opacity', this.opacity);

         if ('antialias' in this)
            selection.style('antialias', this.antialias);
      }

      /** @summary Returns fill color (or pattern url) */
      getFillColor() { return this.pattern_url || this.color; }

      /** @summary Returns fill color without pattern url.
        * @desc If empty, alternative color will be provided
        * @param {string} [altern] - alternative color which returned when fill color not exists
        * @private */
      getFillColorAlt(altern) { return this.color && (this.color != "none") ? this.color : altern; }

      /** @summary Returns true if color not specified or fill style not specified */
      empty() {
         let fill = this.getFillColor();
         return !fill || (fill == 'none');
      }

      /** @summary Returns true if fill attributes has real color */
      hasColor() {
         return this.color && (this.color != 'none');
      }

      /** @summary Set solid fill color as fill pattern
        * @param {string} col - solid color */
      setSolidColor(col) {
         delete this.pattern_url;
         this.color = col;
         this.pattern = 1001;
      }

      /** @summary Check if solid fill is used, also color can be checked
        * @param {string} [solid_color] - when specified, checks if fill color matches */
      isSolid(solid_color) {
         if (this.pattern !== 1001) return false;
         return !solid_color || (solid_color == this.color);
      }

      /** @summary Method used when color or pattern were changed with OpenUi5 widgets
        * @private */
      verifyDirectChange(painter) {
         if (typeof this.pattern == 'string')
            this.pattern = parseInt(this.pattern);
         if (!Number.isInteger(this.pattern)) this.pattern = 0;

         this.change(this.color, this.pattern, painter ? painter.getCanvSvg() : null, true, painter);
      }

      /** @summary Method to change fill attributes.
        * @param {number} color - color index
        * @param {number} pattern - pattern index
        * @param {selection} svg - top canvas element for pattern storages
        * @param {string} [color_as_svg] - when color is string, interpret as normal SVG color
        * @param {object} [painter] - when specified, used to extract color by index */
      change(color, pattern, svg, color_as_svg, painter) {
         delete this.pattern_url;
         this.changed = true;

         if ((color !== undefined) && Number.isInteger(parseInt(color)) && !color_as_svg)
            this.colorindx = parseInt(color);

         if ((pattern !== undefined) && Number.isInteger(parseInt(pattern))) {
            this.pattern = parseInt(pattern);
            delete this.opacity;
            delete this.antialias;
         }

         if ((this.pattern == 1000) && (this.colorindx === 0)) {
            this.pattern_url = 'white';
            return true;
         }

         if (this.pattern == 1000) this.pattern = 1001;

         if (this.pattern < 1001) {
            this.pattern_url = 'none';
            return true;
         }

         if (this.isSolid() && (this.colorindx === 0) && (this.kind === 1) && !color_as_svg) {
            this.pattern_url = 'none';
            return true;
         }

         let indx = this.colorindx;

         if (color_as_svg) {
            this.color = color;
            if (color != "none") indx = d3.color(color).hex().substr(1); // fictional index produced from color code
         } else {
            this.color = painter ? painter.getColor(indx) : jsrp.getColor(indx);
         }

         if (typeof this.color != 'string') this.color = "none";

         if (this.isSolid()) return true;

         if ((this.pattern >= 4000) && (this.pattern <= 4100)) {
            // special transparent colors (use for subpads)
            this.opacity = (this.pattern - 4000) / 100;
            return true;
         }

         if (!svg || svg.empty() || (this.pattern < 3000) || (this.color == "none")) return false;

         let id = "pat_" + this.pattern + "_" + indx,
            defs = svg.select('.canvas_defs');

         if (defs.empty())
            defs = svg.insert("svg:defs", ":first-child").attr("class", "canvas_defs");

         this.pattern_url = "url(#" + id + ")";
         this.antialias = false;

         if (!defs.select("." + id).empty())
            return true;

         let lines = "", lfill = null, fills = "", fills2 = "", w = 2, h = 2;

         switch (this.pattern) {
            case 3001: w = h = 2; fills = "M0,0h1v1h-1zM1,1h1v1h-1z"; break;
            case 3002: w = 4; h = 2; fills = "M1,0h1v1h-1zM3,1h1v1h-1z"; break;
            case 3003: w = h = 4; fills = "M2,1h1v1h-1zM0,3h1v1h-1z"; break;
            case 3004: w = h = 8; lines = "M8,0L0,8"; break;
            case 3005: w = h = 8; lines = "M0,0L8,8"; break;
            case 3006: w = h = 4; lines = "M1,0v4"; break;
            case 3007: w = h = 4; lines = "M0,1h4"; break;
            case 3008:
               w = h = 10;
               fills = "M0,3v-3h3ZM7,0h3v3ZM0,7v3h3ZM7,10h3v-3ZM5,2l3,3l-3,3l-3,-3Z";
               lines = "M0,3l5,5M3,10l5,-5M10,7l-5,-5M7,0l-5,5";
               break;
            case 3009: w = 12; h = 12; lines = "M0,0A6,6,0,0,0,12,0M6,6A6,6,0,0,0,12,12M6,6A6,6,0,0,1,0,12"; lfill = "none"; break;
            case 3010: w = h = 10; lines = "M0,2h10M0,7h10M2,0v2M7,2v5M2,7v3"; break; // bricks
            case 3011: w = 9; h = 18; lines = "M5,0v8M2,1l6,6M8,1l-6,6M9,9v8M6,10l3,3l-3,3M0,9v8M3,10l-3,3l3,3"; lfill = "none"; break;
            case 3012: w = 10; h = 20; lines = "M5,1A4,4,0,0,0,5,9A4,4,0,0,0,5,1M0,11A4,4,0,0,1,0,19M10,11A4,4,0,0,0,10,19"; lfill = "none"; break;
            case 3013: w = h = 7; lines = "M0,0L7,7M7,0L0,7"; lfill = "none"; break;
            case 3014: w = h = 16; lines = "M0,0h16v16h-16v-16M0,12h16M12,0v16M4,0v8M4,4h8M0,8h8M8,4v8"; lfill = "none"; break;
            case 3015: w = 6; h = 12; lines = "M2,1A2,2,0,0,0,2,5A2,2,0,0,0,2,1M0,7A2,2,0,0,1,0,11M6,7A2,2,0,0,0,6,11"; lfill = "none"; break;
            case 3016: w = 12; h = 7; lines = "M0,1A3,2,0,0,1,3,3A3,2,0,0,0,9,3A3,2,0,0,1,12,1"; lfill = "none"; break;
            case 3017: w = h = 4; lines = "M3,1l-2,2"; break;
            case 3018: w = h = 4; lines = "M1,1l2,2"; break;
            case 3019:
               w = h = 12;
               lines = "M1,6A5,5,0,0,0,11,6A5,5,0,0,0,1,6h-1h1A5,5,0,0,1,6,11v1v-1" +
                  "A5,5,0,0,1,11,6h1h-1A5,5,0,0,1,6,1v-1v1A5,5,0,0,1,1,6";
               lfill = "none";
               break;
            case 3020: w = 7; h = 12; lines = "M1,0A2,3,0,0,0,3,3A2,3,0,0,1,3,9A2,3,0,0,0,1,12"; lfill = "none"; break;
            case 3021: w = h = 8; lines = "M8,2h-2v4h-4v2M2,0v2h-2"; lfill = "none"; break; // left stairs
            case 3022: w = h = 8; lines = "M0,2h2v4h4v2M6,0v2h2"; lfill = "none"; break; // right stairs
            case 3023: w = h = 8; fills = "M4,0h4v4zM8,4v4h-4z"; fills2 = "M4,0L0,4L4,8L8,4Z"; break;
            case 3024: w = h = 16; fills = "M0,8v8h2v-8zM8,0v8h2v-8M4,14v2h12v-2z"; fills2 = "M0,2h8v6h4v-6h4v12h-12v-6h-4z"; break;
            case 3025: w = h = 18; fills = "M5,13v-8h8ZM18,0v18h-18l5,-5h8v-8Z"; break;
            default:
               if ((this.pattern > 3025) && (this.pattern < 3100)) {
                  // same as 3002, see TGX11.cxx, line 2234
                  w = 4; h = 2; fills = "M1,0h1v1h-1zM3,1h1v1h-1z"; break;
               }

               let code = this.pattern % 1000,
                  k = code % 10,
                  j = ((code - k) % 100) / 10,
                  i = (code - j * 10 - k) / 100;
               if (!i) break;

               let sz = i * 12, pos, step, x1, x2, y1, y2, max;  // axis distance between lines

               w = h = 6 * sz; // we use at least 6 steps

               const produce = (dy, swap) => {
                  pos = []; step = sz; y1 = 0; max = h;

                  // reduce step for smaller angles to keep normal distance approx same
                  if (Math.abs(dy) < 3) step = Math.round(sz / 12 * 9);
                  if (dy == 0) {
                     step = Math.round(sz / 12 * 8);
                     y1 = step / 2;
                  } else if (dy > 0) {
                     max -= step;
                  } else {
                     y1 = step;
                  }

                  while (y1 <= max) {
                     y2 = y1 + dy * step;
                     if (y2 < 0) {
                        x2 = Math.round(y1 / (y1 - y2) * w);
                        pos.push(0, y1, x2, 0);
                        pos.push(w, h - y1, w - x2, h);
                     } else if (y2 > h) {
                        x2 = Math.round((h - y1) / (y2 - y1) * w);
                        pos.push(0, y1, x2, h);
                        pos.push(w, h - y1, w - x2, 0);
                     } else {
                        pos.push(0, y1, w, y2);
                     }
                     y1 += step;
                  }
                  for (let k = 0; k < pos.length; k += 4) {
                     if (swap) { x1 = pos[k+1]; y1 = pos[k]; x2 = pos[k+3]; y2 = pos[k+2]; }
                          else { x1 = pos[k]; y1 = pos[k+1]; x2 = pos[k+2]; y2 = pos[k+3]; }
                      lines += `M${x1},${y1}`;
                      if (y2 == y1)
                         lines += `h${x2-x1}`;
                      else if (x2 == x1)
                         lines += `v${y2-y1}`;
                      else
                         lines += `L${x2},${y2}`;
                  }
               };

               switch (j) {
                  case 0: produce(0); break;
                  case 1: produce(1); break;
                  case 2: produce(2); break;
                  case 3: produce(3); break;
                  case 4: produce(6); break;
                  case 6: produce(3, true); break;
                  case 7: produce(2, true); break;
                  case 8: produce(1, true); break;
                  case 9: produce(0, true); break;
               }

               switch (k) {
                  case 0: if (j) produce(0); break;
                  case 1: produce(-1); break;
                  case 2: produce(-2); break;
                  case 3: produce(-3); break;
                  case 4: produce(-6); break;
                  case 6: produce(-3, true); break;
                  case 7: produce(-2, true); break;
                  case 8: produce(-1, true); break;
                  case 9: if (j != 9) produce(0, true); break;
               }

               break;
         }

         if (!fills && !lines) return false;

         let patt = defs.append('svg:pattern')
                        .attr("id", id).attr("class", id).attr("patternUnits", "userSpaceOnUse")
                        .attr("width", w).attr("height", h);

         if (fills2) {
            let col = d3.rgb(this.color);
            col.r = Math.round((col.r + 255) / 2); col.g = Math.round((col.g + 255) / 2); col.b = Math.round((col.b + 255) / 2);
            patt.append("svg:path").attr("d", fills2).style("fill", col);
         }
         if (fills) patt.append("svg:path").attr("d", fills).style("fill", this.color);
         if (lines) patt.append("svg:path").attr("d", lines).style('stroke', this.color).style("stroke-width", 1).style("fill", lfill);

         return true;
      }

      /** @summary Create sample of fill pattern inside SVG
       * @private */
      createSample(sample_svg, width, height) {
         // we need to create extra handle to change
         const sample = new TAttFillHandler({ svg: sample_svg, pattern: this.pattern, color: this.color, color_as_svg: true });

         sample_svg.append("path")
            .attr("d", `M0,0h${width}v${height}h${-width}z`)
            .call(sample.func);
      }
   }

   // ===========================================================================

   /**
    * @summary Helper class for font handling
    *
    * @memberof JSROOT
    * @private
    */

   class FontHandler {
      /** @summary constructor */
      constructor(fontIndex, size, scale, name, style, weight) {
         this.name = "Arial";
         this.style = null;
         this.weight = null;

         if (scale && (size < 1)) {
            size *= scale;
            this.scaled = true;
         }

         this.size = Math.round(size || 11);
         this.scale = scale;

         if (fontIndex !== null) {

            let indx = Math.floor(fontIndex / 10),
                fontName = jsrp.root_fonts[indx] || "Arial";

            while (fontName.length > 0) {
               if (fontName[0] === 'b')
                  this.weight = "bold";
               else if (fontName[0] === 'i')
                  this.style = "italic";
               else if (fontName[0] === 'o')
                  this.style = "oblique";
               else
                  break;
               fontName = fontName.substr(1);
            }

            this.name = fontName;
            this.aver_width = jsrp.root_fonts_aver_width[indx] || 0.55;
         } else {
            this.name = name;
            this.style = style || null;
            this.weight = weight || null;
            this.aver_width = this.weight ? 0.58 : 0.55;
         }

         if ((this.name == 'Symbol') || (this.name == 'Wingdings')) {
            this.isSymbol = this.name;
            this.name = "Times New Roman";
         } else {
            this.isSymbol = "";
         }

         this.func = this.setFont.bind(this);
      }

      /** @summary Assigns font-related attributes */
      setFont(selection, arg) {
         selection.attr("font-family", this.name);
         if (arg != 'without-size')
            selection.attr("font-size", this.size)
                     .attr("xml:space", "preserve");
         if (this.weight)
            selection.attr("font-weight", this.weight);
         if (this.style)
            selection.attr("font-style", this.style);
      }

      /** @summary Set font size (optional) */
      setSize(size) { this.size = Math.round(size); }

      /** @summary Set text color (optional) */
      setColor(color) { this.color = color; }

      /** @summary Set text align (optional) */
      setAlign(align) { this.align = align; }

      /** @summary Set text angle (optional) */
      setAngle(angle) { this.angle = angle; }

      /** @summary Allign angle to step raster, add optional offset */
      roundAngle(step, offset) {
         this.angle = parseInt(this.angle || 0);
         if (!Number.isInteger(this.angle)) this.angle = 0;
         this.angle = Math.round(this.angle/step) * step + (offset || 0);
         if (this.angle < 0)
            this.angle += 360;
         else if (this.angle >= 360)
            this.angle -= 360;
      }

      /** @summary Clears all font-related attributes */
      clearFont(selection) {
         selection.attr("font-family", null)
                  .attr("font-size", null)
                  .attr("xml:space", null)
                  .attr("font-weight", null)
                  .attr("font-style", null);
      }

      /** @summary Returns true in case of monospace font
        * @private */
      isMonospace() {
         let n = this.name.toLowerCase();
         return (n.indexOf("courier") == 0) || (n == "monospace") || (n == "monaco");
      }

      /** @summary Return full font declaration which can be set as font property like "12pt Arial bold"
        * @private */
      getFontHtml() {
         let res = Math.round(this.size) + "pt " + this.name;
         if (this.weight) res += " " + this.weight;
         if (this.style) res += " " + this.style;
         return res;
      }
   }

  // ===========================================================================

   /** @summary Tries to choose time format for provided time interval
     * @private */
   jsrp.chooseTimeFormat = function(awidth, ticks) {
      if (awidth < .5) return ticks ? "%S.%L" : "%H:%M:%S.%L";
      if (awidth < 30) return ticks ? "%Mm%S" : "%H:%M:%S";
      awidth /= 60; if (awidth < 30) return ticks ? "%Hh%M" : "%d/%m %H:%M";
      awidth /= 60; if (awidth < 12) return ticks ? "%d-%Hh" : "%d/%m/%y %Hh";
      awidth /= 24; if (awidth < 15.218425) return ticks ? "%d/%m" : "%d/%m/%y";
      awidth /= 30.43685; if (awidth < 6) return "%d/%m/%y";
      awidth /= 12; if (awidth < 2) return ticks ? "%m/%y" : "%d/%m/%y";
      return "%Y";
   }

   /** @summary Function used to provide svg:path for the smoothed curves.
     * @desc reuse code from d3.js. Used in TH1, TF1 and TGraph painters
     * @param {string} kind  should contain "bezier" or "line".
     * If first symbol "L", then it used to continue drawing
     * @private */
   jsrp.buildSvgPath = function(kind, bins, height, ndig) {

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
            str = str.substr(0, str.length - 1);
         if (str[str.length - 1] == '.')
            str = str.substr(0, str.length - 1);
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

   /** @summary Returns visible rect of element
     * @param {object} elem - d3.select object with element
     * @param {string} [kind] - which size method is used
     * @desc kind = 'bbox' use getBBox, works only with SVG
     * kind = 'full' - full size of element, using getBoundingClientRect function
     * kind = 'nopadding' - excludes padding area
     * With node.js can use "width" and "height" attributes when provided in element
     * @private */
   function getElementRect(elem, sizearg) {
      if (JSROOT.nodejs && (sizearg != 'bbox'))
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
   jsrp.getAbsPosInCanvas = (sel, pos) => {
      while (!sel.empty() && !sel.classed('root_canvas') && pos) {
         let cl = sel.attr("class");
         if (cl && ((cl.indexOf("root_frame") >= 0) || (cl.indexOf("__root_pad_") >= 0))) {
            pos.x += sel.property("draw_x") || 0;
            pos.y += sel.property("draw_y") || 0;
         }
         sel = d3.select(sel.node().parentNode);
      }
      return pos;
   }

   // ========================================================================================

   /**
    * @summary Base painter class in JSROOT
    *
    * @memberof JSROOT
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

         if (!this.divid) return d3.select(null);

         let res = this._selected_main;
         if (!res) {
            if (typeof this.divid == "string") {
               let id = this.divid;
               if (id[0] != '#') id = "#" + id;
               res = d3.select(id);
               if (!res.empty()) this.divid = res.node();
            } else {
               res = d3.select(this.divid);
            }
            this._selected_main = res;
         }

         if (!res || res.empty() || (is_direct === 'origin')) return res;

         let use_enlarge = res.property('use_enlarge'),
            layout = res.property('layout') || 'simple',
            layout_selector = (layout == 'simple') ? "" : res.property('layout_selector');

         if (layout_selector) res = res.select(layout_selector);

         // one could redirect here
         if (!is_direct && !res.empty() && use_enlarge) res = d3.select("#jsroot_enlarge_div");

         return res;
      }

      /** @summary Access/change top painter
        * @private */
      _accessTopPainter(on) {
         let main = this.selectDom().node(),
             chld = main ? main.firstChild : null;
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

         if (this._hpainter && typeof this._hpainter.removePainter === 'function')
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

         if (main.empty() || !JSROOT.settings.CanEnlarge || (origin.property('can_enlarge') === false)) return false;

         if (action === undefined) return true;

         if (action === 'verify') return true;

         let state = origin.property('use_enlarge') ? "on" : "off";

         if (action === 'state') return state;

         if (action === 'toggle') action = (state === "off");

         let enlarge = d3.select("#jsroot_enlarge_div");

         if ((action === true) && (state !== "on")) {
            if (!enlarge.empty()) return false;

            enlarge = d3.select(document.body)
               .append("div")
               .attr("id", "jsroot_enlarge_div");

            let rect1 = getElementRect(main),
                rect2 = getElementRect(enlarge);

            // if new enlarge area not big enough, do not do it
            if ((rect2.width <= rect1.width) || (rect2.height <= rect1.height))
               if (rect2.width * rect2.height < rect1.width * rect1.height) {
                  if (!skip_warning)
                     console.log('Enlarged area ' + rect2.width + "x" + rect2.height + ' smaller then original drawing ' + rect1.width + "x" + rect1.height);
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
        * @desc Used by {@link JSROOT.HierarchyPainter}
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
        * @desc Used with {@link JSROOT.HierarchyPainter} to identify drawn item name */
      getItemName() { return ('_hitemname' in this) ? this._hitemname : null; }

      /** @summary Returns assigned item draw option
        * @desc Used with {@link JSROOT.HierarchyPainter} to identify drawn item option */
      getItemDrawOpt() { return this._hdrawopt || ""; }

   } // class BasePainter

   // ==============================================================================


   /**
    * @summary Painter class for ROOT objects
    *
    * @memberof JSROOT
    */

   class ObjectPainter extends BasePainter {

      /** @summary constructor
        * @param {object|string} dom - dom element or identifier
        * @param {object} obj - object to draw
        * @param {string} [opt] - object draw options */
      constructor(dom, obj, opt) {
         super(dom);
         // this.draw_g = undefined; // container for all drawn objects
         // this._main_painter = undefined;  // main painter in the correspondent pad
         this.pad_name = dom ? this.selectCurrentPad() : ""; // name of pad where object is drawn
         this.assignObject(obj);
         if (typeof opt == "string")
            this.options = { original: opt };
      }

      /** @summary Assign object to the painter
        * @protected */
      assignObject(obj) {
         if (obj && (typeof obj == 'object'))
            this.draw_object = obj;
         else
            delete this.draw_object;
      }

      /** @summary Assigns pad name where element will be drawn
        * @desc Should happend before first draw of element is performed, only for special use case
        * @param {string} [pad_name] - on which subpad element should be draw, if not specified - use current
        * @protected */
      setPadName(pad_name) {
         this.pad_name = (typeof pad_name == 'string') ? pad_name : this.selectCurrentPad();
      }

      /** @summary Returns pad name where object is drawn */
      getPadName() {
         return this.pad_name || "";
      }

      /** @summary Assign snapid to the painter
       * @desc Identifier used to communicate with server side and identifies object on the server
       * @private */
      assignSnapId(id) { this.snapid = id; }

      /** @summary Generic method to cleanup painter.
        * @desc Remove object drawing and (in case of main painter) also main HTML components
        * @protected */
      cleanup() {

         this.removeG();

         let keep_origin = true;

         if (this.isMainPainter()) {
            let pp = this.getPadPainter();
            if (!pp || pp.normal_canvas === false) keep_origin = false;
         }

         // cleanup all existing references
         delete this.pad_name;
         delete this._main_painter;
         this.draw_object = null;
         delete this.snapid;

         // remove attributes objects (if any)
         delete this.fillatt;
         delete this.lineatt;
         delete this.markeratt;
         delete this.bins;
         delete this.root_colors;
         delete this.options;
         delete this.options_store;

         // remove extra fields from v7 painters
         delete this.rstyle;
         delete this.csstype;

         super.cleanup(keep_origin);
      }

      /** @summary Returns drawn object */
      getObject() { return this.draw_object; }

      /** @summary Returns drawn object class name */
      getClassName() {
         let obj = this.getObject(),
             clname = obj ? obj._typename : "";
         return clname || "";
      }

      /** @summary Checks if drawn object matches with provided typename
        * @param {string|object} arg - typename (or object with _typename member)
        * @protected */
      matchObjectType(arg) {
         if (!arg || !this.draw_object) return false;
         if (typeof arg === 'string') return (this.draw_object._typename === arg);
         if (arg._typename) return (this.draw_object._typename === arg._typename);
         return this.draw_object._typename.match(arg);
      }

      /** @summary Change item name
        * @desc When available, used for svg:title proprty
        * @private */
      setItemName(name, opt, hpainter) {
         super.setItemName(name, opt, hpainter);
         if (this.no_default_title || (name == "")) return;
         let can = this.getCanvSvg();
         if (!can.empty()) can.select("title").text(name);
                      else this.selectDom().attr("title", name);
      }

      /** @summary Store actual this.options together with original string
        * @private */
      storeDrawOpt(original) {
         if (!this.options) return;
         if (!original) original = "";
         let pp = original.indexOf(";;");
         if (pp >= 0) original = original.substr(0, pp);
         this.options.original = original;
         this.options_store = JSROOT.extend({}, this.options);
      }

      /** @summary Return actual draw options as string
        * @desc if options are not modified - returns original string which was specified for object draw */
      getDrawOpt() {
         if (!this.options) return "";

         if (typeof this.options.asString == "function") {
            let changed = false, pp = this.getPadPainter();
            if (!this.options_store || (pp && pp._interactively_changed)) {
               changed  = true;
            } else {
               for (let k in this.options)
                  if (this.options[k] !== this.options_store[k])
                     changed = true;
            }
            if (changed)
               return this.options.asString(this.isMainPainter(), pp ? pp.getRootPad() : null);
         }

         return this.options.original || ""; // nothing better, return original draw option
      }

      /** @summary Central place to update objects drawing
        * @param {object} obj - new version of object, values will be updated in original object
        * @param {string} [opt] - when specified, new draw options
        * @returns {boolean|Promise} for object redraw
        * @desc Two actions typically done by redraw - update object content via {@link JSROOT.ObjectPainter.updateObject} and
         * then redraw correspondent pad via {@link JSROOT.ObjectPainter.redrawPad}. If possible one should redefine
         * only updateObject function and keep this function unchanged. But for some special painters this function is the
         * only way to control how object can be update while requested from the server
         * @protected */
      redrawObject(obj, opt) {
         if (!this.updateObject(obj,opt)) return false;
         let current = document.body.style.cursor;
         document.body.style.cursor = 'wait';
         let res = this.redrawPad();
         document.body.style.cursor = current;
         return res || true;
      }

      /** @summary Generic method to update object content.
        * @desc Default implementation just copies first-level members to current object
        * @param {object} obj - object with new data
        * @param {string} [opt] - option which will be used for redrawing
        * @protected */
      updateObject(obj /*, opt */) {
         if (!this.matchObjectType(obj)) return false;
         JSROOT.extend(this.getObject(), obj);
         return true;
      }

      /** @summary Returns string with object hint
        * @desc It is either item name or object name or class name.
        * Such string typically used as object tooltip.
        * If result string larger than 20 symbols, it will be cutted. */
      getObjectHint() {
         let res = this.getItemName(), obj = this.getObject();
         if (!res) res = obj && obj.fName ? obj.fName : "";
         if (!res) res = this.getClassName();
         if (res.lenght > 20) res = res.substr(0, 17) + "...";
         return res;
      }

      /** @summary returns color from current list of colors
        * @desc First checks canvas painter and then just access global list of colors
        * @param {number} indx - color index
        * @returns {string} with SVG color name or rgb()
        * @protected */
      getColor(indx) {
         let jsarr = this.root_colors;

         if (!jsarr) {
            let pp = this.getCanvPainter();
            jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : jsrp.root_colors;
         }

         return jsarr[indx];
      }

      /** @summary Add color to list of colors
        * @desc Returned color index can be used as color number in all other draw functions
        * @returns {number} new color index
        * @protected */
      addColor(color) {
         let jsarr = this.root_colors;
         if (!jsarr) {
            let pp = this.getCanvPainter();
            jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : jsrp.root_colors;
         }
         let indx = jsarr.indexOf(color);
         if (indx >= 0) return indx;
         jsarr.push(color);
         return jsarr.length - 1;
      }

      /** @summary returns tooltip allowed flag
        * @desc If available, checks in canvas painter
        * @private */
      isTooltipAllowed() {
         let src = this.getCanvPainter() || this;
         return src.tooltip_allowed ? true : false;
      }

      /** @summary change tooltip allowed flag
        * @param {boolean|string} [on = true] set tooltip allowed state or 'toggle'
        * @private */
      setTooltipAllowed(on) {
         if (on === undefined) on = true;
         let src = this.getCanvPainter() || this;
         src.tooltip_allowed = (on == "toggle") ? !src.tooltip_allowed : on;
      }

      /** @summary Checks if draw elements were resized and drawing should be updated.
        * @desc Redirects to {@link JSROOT.TPadPainter.checkCanvasResize}
        * @private */
      checkResize(arg) {
         let p = this.getCanvPainter();
         if (!p) return false;

         // only canvas should be checked
         p.checkCanvasResize(arg);
         return true;
      }

      /** @summary removes <g> element with object drawing
        * @desc generic method to delete all graphical elements, associated with the painter
        * @protected */
      removeG() {
         if (this.draw_g) {
            this.draw_g.remove();
            delete this.draw_g;
         }
      }

      /** @summary Returns created <g> element used for object drawing
        * @desc Element should be created by {@link JSROOT.ObjectPainter.createG}
        * @protected */
      getG() { return this.draw_g; }

      /** @summary (re)creates svg:g element for object drawings
        * @desc either one attach svg:g to pad primitives (default)
        * or svg:g element created in specified frame layer ("main_layer" will be used when true specified)
        * @param {boolean|string} [frame_layer] - when specified, <g> element will be created inside frame layer, otherwise in the pad
        * @protected */
      createG(frame_layer) {

         let layer;

         if (frame_layer) {
            let frame = this.getFrameSvg();
            if (frame.empty()) {
               console.error('Not found frame to create g element inside');
               return frame;
            }
            if (typeof frame_layer != 'string') frame_layer = "main_layer";
            layer = frame.select("." + frame_layer);
         } else {
            layer = this.getLayerSvg("primitives_layer");
         }

         if (this.draw_g && this.draw_g.node().parentNode !== layer.node()) {
            console.log('g element changes its layer!!');
            this.removeG();
         }

         if (this.draw_g) {
            // clear all elements, keep g element on its place
            this.draw_g.selectAll('*').remove();
         } else {
            this.draw_g = layer.append("svg:g");

            if (!frame_layer)
               layer.selectChildren(".most_upper_primitives").raise();
         }

         // set attributes for debugging
         if (this.draw_object) {
            this.draw_g.attr('objname', (this.draw_object.fName || "name").replace(/[^\w]/g, '_'));
            this.draw_g.attr('objtype', (this.draw_object._typename || "type").replace(/[^\w]/g, '_'));
         }

         this.draw_g.property('in_frame', !!frame_layer); // indicates coordinate system

         return this.draw_g;
      }

      /** @summary Canvas main svg element
        * @returns {object} d3 selection with canvas svg
        * @protected */
      getCanvSvg() { return this.selectDom().select(".root_canvas"); }

      /** @summary Pad svg element
        * @param {string} [pad_name] - pad name to select, if not specified - pad where object is drawn
        * @returns {object} d3 selection with pad svg
        * @protected */
      getPadSvg(pad_name) {
         if (pad_name === undefined)
            pad_name = this.pad_name;

         let c = this.getCanvSvg();
         if (!pad_name || c.empty()) return c;

         let cp = c.property('pad_painter');
         if (cp && cp.pads_cache && cp.pads_cache[pad_name])
            return d3.select(cp.pads_cache[pad_name]);

         c = c.select(".primitives_layer .__root_pad_" + pad_name);
         if (cp) {
            if (!cp.pads_cache) cp.pads_cache = {};
            cp.pads_cache[pad_name] = c.node();
         }
         return c;
      }

      /** @summary Method selects immediate layer under canvas/pad main element
        * @param {string} name - layer name, exits "primitives_layer", "btns_layer", "info_layer"
        * @param {string} [pad_name] - pad name; current pad name  used by default
        * @protected */
      getLayerSvg(name, pad_name) {
         let svg = this.getPadSvg(pad_name);
         if (svg.empty()) return svg;

         if (name.indexOf("prim#") == 0) {
            svg = svg.select(".primitives_layer");
            name = name.substr(5);
         }

         let node = svg.node().firstChild;
         while (node) {
            let elem = d3.select(node);
            if (elem.classed(name)) return elem;
            node = node.nextSibling;
         }

         return d3.select(null);
      }

      /** @summary Method selects current pad name
        * @param {string} [new_name] - when specified, new current pad name will be configured
        * @returns {string} previous selected pad or actual pad when new_name not specified
        * @private */
      selectCurrentPad(new_name) {
         let svg = this.getCanvSvg();
         if (svg.empty()) return "";
         let curr = svg.property('current_pad');
         if (new_name !== undefined) svg.property('current_pad', new_name);
         return curr;
      }

      /** @summary returns pad painter
        * @param {string} [pad_name] pad name or use current pad by default
        * @protected */
      getPadPainter(pad_name) {
         let elem = this.getPadSvg(typeof pad_name == "string" ? pad_name : undefined);
         return elem.empty() ? null : elem.property('pad_painter');
      }

      /** @summary returns canvas painter
        * @protected */
      getCanvPainter() {
         let elem = this.getCanvSvg();
         return elem.empty() ? null : elem.property('pad_painter');
      }

      /** @summary Return functor, which can convert x and y coordinates into pixels, used for drawing in the pad
        * @desc X and Y coordinates can be converted by calling func.x(x) and func.y(y)
        * Only can be used for painting in the pad, means CreateG() should be called without arguments
        * @param {boolean} isndc - if NDC coordinates will be used
        * @param {boolean} [noround] - if set, return coordinates will not be rounded
        * @protected */
      getAxisToSvgFunc(isndc, nornd) {
         let func = { isndc: isndc, nornd: nornd },
             use_frame = this.draw_g && this.draw_g.property('in_frame');
         if (use_frame) func.main = this.getFramePainter();
         if (func.main && func.main.grx && func.main.gry) {
            if (nornd) {
               func.x = function(x) { return this.main.grx(x); }
               func.y = function(y) { return this.main.gry(y); }
            } else {
               func.x = function(x) { return Math.round(this.main.grx(x)); }
               func.y = function(y) { return Math.round(this.main.gry(y)); }
            }
         } else if (!use_frame) {
            let pp = this.getPadPainter();
            if (!isndc && pp) func.pad = pp.getRootPad(true); // need for NDC conversion
            func.padw = pp ? pp.getPadWidth() : 10;
            func.x = function(value) {
               if (this.pad) {
                  if (this.pad.fLogx)
                     value = (value > 0) ? Math.log10(value) : this.pad.fUxmin;
                  value = (value - this.pad.fX1) / (this.pad.fX2 - this.pad.fX1);
               }
               value *= this.padw;
               return this.nornd ? value : Math.round(value);
            }
            func.padh = pp ? pp.getPadHeight() : 10;
            func.y = function(value) {
               if (this.pad) {
                  if (this.pad.fLogy)
                     value = (value > 0) ? Math.log10(value) : this.pad.fUymin;
                  value = (value - this.pad.fY1) / (this.pad.fY2 - this.pad.fY1);
               }
               value = (1 - value) * this.padh;
               return this.nornd ? value : Math.round(value);
            }
         } else {
            console.error('Problem to create functor for', this.getClassName());
            func.x = () => 0;
            func.y = () => 0;

         }
         return func;
      }

      /** @summary Converts x or y coordinate into pad SVG coordinates.
        * @desc Only can be used for painting in the pad, means CreateG() should be called without arguments
        * @param {string} axis - name like "x" or "y"
        * @param {number} value - axis value to convert.
        * @param {boolean} ndc - is value in NDC coordinates
        * @param {boolean} [noround] - skip rounding
        * @returns {number} value of requested coordiantes
        * @protected */
      axisToSvg(axis, value, ndc, noround) {
         let func = this.getAxisToSvgFunc(ndc, noround);
         return func[axis](value);
      }

      /** @summary Converts pad SVG x or y coordinates into axis values.
        * @desc Reverse transformation for {@link JSROOT.ObjectPainter.axisToSvg}
        * @param {string} axis - name like "x" or "y"
        * @param {number} coord - graphics coordiante.
        * @param {boolean} ndc - kind of return value
        * @returns {number} value of requested coordiantes
        * @protected */
      svgToAxis(axis, coord, ndc) {
         let use_frame = this.draw_g && this.draw_g.property('in_frame');

         if (use_frame) {
            let main = this.getFramePainter();
            return main ? main.revertAxis(axis, coord) : 0;
         }

         let pp = this.getPadPainter(),
             value = (axis == "y") ? (1 - coord / pp.getPadHeight()) : coord / pp.getPadWidth(),
             pad = (ndc || !pp) ? null : pp.getRootPad(true);

         if (pad) {
            if (axis == "y") {
               value = pad.fY1 + value * (pad.fY2 - pad.fY1);
               if (pad.fLogy) value = Math.pow(10, value);
            } else {
               value = pad.fX1 + value * (pad.fX2 - pad.fX1);
               if (pad.fLogx) value = Math.pow(10, value);
            }
         }

         return value;
      }

      /** @summary Returns svg element for the frame in current pad
        * @protected */
      getFrameSvg() {
         let layer = this.getLayerSvg("primitives_layer");
         if (layer.empty()) return layer;
         let node = layer.node().firstChild;
         while (node) {
            let elem = d3.select(node);
            if (elem.classed("root_frame")) return elem;
            node = node.nextSibling;
         }
         return d3.select(null);
      }

      /** @summary Returns frame painter for current pad
        * @desc Pad has direct reference on frame if any
        * @protected */
      getFramePainter() {
         let pp = this.getPadPainter();
         return pp ? pp.getFramePainter() : null;
      }

      /** @summary Returns painter for main object on the pad.
        * @desc Typically it is first histogram drawn on the pad and which draws frame axes
        * But it also can be special usecase as TASImage or TGraphPolargram
        * @param {boolean} [not_store] - if true, prevent temporary storage of main painter reference
        * @protected */
      getMainPainter(not_store) {
         let res = this._main_painter;
         if (!res) {
            let pp = this.getPadPainter();
            if (!pp)
               res = this.getTopPainter();
            else
               res = pp.getMainPainter();
            if (!res) res = null;
            if (!not_store) this._main_painter = res;
         }
         return res;
      }

      /** @summary Returns true if this is main painter
        * @protected */
      isMainPainter() { return this === this.getMainPainter(); }

      /** @summary Assign this as main painter on the pad
        * @desc Main painter typically responsible for axes drawing
        * Should not be used by pad/canvas painters, but rather by objects which are drawing axis
        * @protected */
      setAsMainPainter(force) {
         let pp = this.getPadPainter();
         if (!pp)
            this.setTopPainter(); //fallback on BasePainter method
          else
            pp.setMainPainter(this, force);
      }

      /** @summary Add painter to pad list of painters
        * @param {string} [pad_name] - optional pad name where painter should be add
        * @desc Normally one should use {@link JSROOT.Painter.ensureTCanvas} to add painter to pad list of primitives
        * @protected */
      addToPadPrimitives(pad_name) {
         if (pad_name !== undefined) this.setPadName(pad_name);
         let pp = this.getPadPainter(pad_name); // important - pad_name must be here, otherwise PadPainter class confuses itself

         if (!pp || (pp === this)) return false;

         if (pp.painters.indexOf(this) < 0)
            pp.painters.push(this);

         if (!this.rstyle && pp.next_rstyle)
            this.rstyle = pp.next_rstyle;

         return true;
      }

      /** @summary Remove painter from pad list of painters
        * @protected */
      removeFromPadPrimitives() {
         let pp = this.getPadPainter();

         if (!pp || (pp === this)) return false;

         let k = pp.painters.indexOf(this);
         if (k >= 0) pp.painters.splice(k, 1);
         return true;
      }

      /** @summary Creates marker attributes object
        * @desc Can be used to produce markers in painter.
        * See {@link JSROOT.TAttMarkerHandler} for more info.
        * Instance assigned as this.markeratt data member, recognized by GED editor
        * @param {object} args - either TAttMarker or see arguments of {@link JSROOT.TAttMarkerHandler}
        * @returns {object} created handler
        * @protected */
      createAttMarker(args) {
         if (!args || (typeof args !== 'object')) args = { std: true }; else
            if (args.fMarkerColor !== undefined && args.fMarkerStyle !== undefined && args.fMarkerSize !== undefined) args = { attr: args, std: false };

         if (args.std === undefined) args.std = true;
         if (args.painter === undefined) args.painter = this;

         let handler = args.std ? this.markeratt : null;

         if (!handler)
            handler = new TAttMarkerHandler(args);
         else if (!handler.changed || args.force)
            handler.setArgs(args);

         if (args.std) this.markeratt = handler;

         // handler.used = false; // mark that line handler is not yet used
         return handler;
      }

      /** @summary Creates line attributes object.
        * @desc Can be used to produce lines in painter.
        * See {@link JSROOT.TAttLineHandler} for more info.
        * Instance assigned as this.lineatt data member, recognized by GED editor
        * @param {object} args - either TAttLine or see constructor arguments of {@link JSROOT.TAttLineHandler}
        * @protected */
      createAttLine(args) {
         if (!args || (typeof args !== 'object'))
            args = { std: true };
         else if (args.fLineColor !== undefined && args.fLineStyle !== undefined && args.fLineWidth !== undefined)
            args = { attr: args, std: false };

         if (args.std === undefined) args.std = true;
         if (args.painter === undefined) args.painter = this;

         let handler = args.std ? this.lineatt : null;

         if (!handler)
            handler = new TAttLineHandler(args);
         else if (!handler.changed || args.force)
            handler.setArgs(args);

         if (args.std) this.lineatt = handler;

         // handler.used = false; // mark that line handler is not yet used
         return handler;
      }

      /** @summary Creates fill attributes object.
        * @desc Method dedicated to create fill attributes, bound to canvas SVG
        * otherwise newly created patters will not be usable in the canvas
        * See {@link JSROOT.TAttFillHandler} for more info.
        * Instance assigned as this.fillatt data member, recognized by GED editors
        * @param {object} args - for special cases one can specify TAttFill as args or number of parameters
        * @param {boolean} [args.std = true] - this is standard fill attribute for object and should be used as this.fillatt
        * @param {object} [args.attr = null] - object, derived from TAttFill
        * @param {number} [args.pattern = undefined] - integer index of fill pattern
        * @param {number} [args.color = undefined] - integer index of fill color
        * @param {string} [args.color_as_svg = undefined] - color will be specified as SVG string, not as index from color palette
        * @param {number} [args.kind = undefined] - some special kind which is handled differently from normal patterns
        * @returns created handle
        * @protected */
      createAttFill(args) {
         if (!args || (typeof args !== 'object'))
            args = { std: true };
         else if (args._typename && args.fFillColor !== undefined && args.fFillStyle !== undefined)
            args = { attr: args, std: false };

         if (args.std === undefined) args.std = true;

         let handler = args.std ? this.fillatt : null;

         if (!args.svg) args.svg = this.getCanvSvg();
         if (args.painter === undefined) args.painter = this;

         if (!handler)
            handler = new TAttFillHandler(args);
         else if (!handler.changed || args.force)
            handler.setArgs(args);

         if (args.std) this.fillatt = handler;

         // handler.used = false; // mark that fill handler is not yet used

         return handler;
      }

      /** @summary call function for each painter in the pad
        * @desc Iterate over all known painters
        * @private */
      forEachPainter(userfunc, kind) {
         // iterate over all painters from pad list
         let pp = this.getPadPainter();
         if (pp) {
            pp.forEachPainterInPad(userfunc, kind);
         } else {
            let painter = this.getTopPainter();
            if (painter && (kind !== "pads")) userfunc(painter);
         }
      }

      /** @summary indicate that redraw was invoked via interactive action (like context menu or zooming)
        * @desc Use to catch such action by GED and by server-side
        * @returns {Promise} when completed
        * @private */
      interactiveRedraw(arg, info, subelem) {

         let reason, res;
         if ((typeof info == "string") && (info.indexOf("exec:") != 0)) reason = info;
         if (arg == "pad")
            res = this.redrawPad(reason);
         else if (arg !== false)
            res = this.redraw(reason);

         if (!isPromise(res)) res = Promise.resolve(false);

         return res.then(() => {
            // inform GED that something changes
            let canp = this.getCanvPainter();

            if (canp && (typeof canp.producePadEvent == 'function'))
               canp.producePadEvent("redraw", this.getPadPainter(), this, null, subelem);

            // inform server that drawopt changes
            if (canp && (typeof canp.processChanges == 'function'))
               canp.processChanges(info, this, subelem);

            return this;
         });
      }

      /** @summary Redraw all objects in the current pad
        * @param {string} [reason] - like 'resize' or 'zoom'
        * @returns {Promise} when pad redraw completed
        * @protected */
      redrawPad(reason) {
         let pp = this.getPadPainter();
         return pp ? pp.redrawPad(reason) : Promise.resolve(false);
      }

      /** @summary execute selected menu command, either locally or remotely
        * @private */
      executeMenuCommand(method) {

         if (method.fName == "Inspect") {
            // primitve inspector, keep it here
            this.showInspector();
            return true;
         }

         return false;
      }

      /** @summary Invoke method for object via WebCanvas functionality
        * @desc Requires that painter marked with object identifier (this.snapid) or identifier provided as second argument
        * Canvas painter should exists and in non-readonly mode
        * Execution string can look like "Print()".
        * Many methods call can be chained with "Print();;Update();;Clear()"
        * @private */
      submitCanvExec(exec, snapid) {
         if (!exec || (typeof exec != 'string')) return;

         let canp = this.getCanvPainter();
         if (canp && (typeof canp.submitExec == "function"))
            canp.submitExec(this, exec, snapid);
      }

      /** @summary remove all created draw attributes
        * @protected */
      deleteAttr() {
         delete this.lineatt;
         delete this.fillatt;
         delete this.markeratt;
      }

      /** @summary Show object in inspector for provided object
        * @protected */
      showInspector(obj) {
         let main = this.selectDom(),
            rect = getElementRect(main),
            w = Math.round(rect.width * 0.05) + "px",
            h = Math.round(rect.height * 0.05) + "px",
            id = "root_inspector_" + JSROOT._.id_counter++;

         main.append("div")
            .attr("id", id)
            .attr("class", "jsroot_inspector")
            .style('position', 'absolute')
            .style('top', h)
            .style('bottom', h)
            .style('left', w)
            .style('right', w);

         if (!obj || (typeof obj !== 'object') || !obj._typename)
            obj = this.getObject();

         JSROOT.draw(id, obj, 'inspect');
      }

      /** @summary Fill context menu for the object
        * @private */
      fillContextMenu(menu) {
         let title = this.getObjectHint();
         if (this.getObject() && ('_typename' in this.getObject()))
            title = this.getObject()._typename + "::" + title;

         menu.add("header:" + title);

         menu.addAttributesMenu(this);

         if (menu.size() > 0)
            menu.add('Inspect', this.showInspector);

         return menu.size() > 0;
      }

      /** @summary shows objects status
        * @desc Either used canvas painter method or globaly assigned
        * When no parameters are specified, just basic object properties are shown
        * @private */
      showObjectStatus(name, title, info, info2) {
         let cp = this.getCanvPainter();

         if (cp && (typeof cp.showCanvasStatus !== 'function')) cp = null;

         if (!cp && (typeof jsrp.showStatus !== 'function')) return false;

         if (this.enlargeMain('state') === 'on') return false;

         if ((name === undefined) && (title === undefined)) {
            let obj = this.getObject();
            if (!obj) return;
            name = this.getItemName() || obj.fName;
            title = obj.fTitle || obj._typename;
            info = obj._typename;
         }

         if (cp)
            cp.showCanvasStatus(name, title, info, info2);
         else
            jsrp.showStatus(name, title, info, info2);
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
        * @param {object} [draw_g] - element where text drawm, by default using main object <g> element
        * @param {number} [max_font_size] - maximal font size, used when text can be scaled
        * @protected */
      startTextDrawing(font_face, font_size, draw_g, max_font_size) {

         if (!draw_g) draw_g = this.draw_g;
         if (!draw_g || draw_g.empty()) return;

         let font = (font_size === 'font') ? font_face : new FontHandler(font_face, font_size),
             pp = this.getPadPainter();

         draw_g.call(font.func);

         draw_g.property('draw_text_completed', false) // indicate that draw operations submitted
               .property('all_args',[]) // array of all submitted args, makes easier to analyze them
               .property('text_font', font)
               .property('text_factor', 0.)
               .property('max_text_width', 0) // keep maximal text width, use it later
               .property('max_font_size', max_font_size)
               .property("_fast_drawing", pp ? pp._fast_drawing : false);

         if (draw_g.property("_fast_drawing"))
            draw_g.property("_font_too_small", (max_font_size && (max_font_size < 5)) || (font.size < 4));
      }

      /** @summary Apply scaling factor to all drawn text in the <g> element
        * @desc Can be applied at any time before finishTextDrawing is called - even in the postprocess callbacks of text draw
        * @param {number} factor - scaling factor
        * @param {object} [draw_g] - drawing element for the text
        * @protected */
      scaleTextDrawing(factor, draw_g) {
         if (!draw_g) draw_g = this.draw_g;
         if (!draw_g || draw_g.empty()) return;
         if (factor && (factor > draw_g.property('text_factor')))
            draw_g.property('text_factor', factor);
      }

      /** @summary Analyze if all text draw operations are completed
        * @private */
      _checkAllTextDrawing(draw_g, resolveFunc, try_optimize) {

         let all_args = draw_g.property('all_args'), missing = 0;
         if (!all_args) {
            console.log('Text drawing is finished - why calling _checkAllTextDrawing?????');
            all_args = [];
         }

         all_args.forEach(arg => { if (!arg.ready) missing++; });

         if (missing > 0) {
            if (typeof resolveFunc == 'function') {
               draw_g.node().textResolveFunc = resolveFunc;
               draw_g.node().try_optimize = try_optimize;
            }
            return;
         }

         draw_g.property('all_args', null); // clear all_args property

         // adjust font size (if there are normal text)
         let f = draw_g.property('text_factor'),
             font = draw_g.property('text_font'),
             max_sz = draw_g.property('max_font_size'),
             font_size = font.size, any_text = false, only_text = true;

         if ((f > 0) && ((f < 0.9) || (f > 1)))
            font.size = Math.floor(font.size / f);

         if (max_sz && (font.size > max_sz))
            font.size = max_sz;

         if (font.size != font_size) {
            draw_g.call(font.func);
            font_size = font.size;
         }

         all_args.forEach(arg => {
            if (arg.mj_node && arg.applyAttributesToMathJax) {
               let svg = arg.mj_node.select("svg"); // MathJax svg
               arg.applyAttributesToMathJax(this, arg.mj_node, svg, arg, font_size, f);
               delete arg.mj_node; // remove reference
               only_text = false;
            } else if (arg.txt_g) {
               only_text = false;
            }
         });

         if (!resolveFunc) {
            resolveFunc = draw_g.node().textResolveFunc;
            try_optimize = draw_g.node().try_optimize;
            delete draw_g.node().textResolveFunc;
            delete draw_g.node().try_optimize;
         }

         let optimize_arr = (try_optimize && only_text) ? [] : null;

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
            } else {
               return;
            }

            txt.attr('visibility', null);

            any_text = true;

            if (arg.width) {
               // adjust x position when scale into specified rectangle
               if (arg.align[0] == "middle")
                  arg.x += arg.width / 2;
                else if (arg.align[0] == "end")
                   arg.x += arg.width;
            }

            if (arg.height) {
               if (arg.align[1].indexOf('bottom') === 0)
                  arg.y += arg.height;
               else if (arg.align[1] == 'middle')
                  arg.y += arg.height / 2;
            }

            arg.dx = arg.dy = 0;

            if (is_txt) {

               // handle simple text drawing

               if (JSROOT.nodejs) {
                  if (arg.scale && (f > 0)) { arg.box.width *= 1/f; arg.box.height *= 1/f; }
               } else if (!arg.plain && !arg.fast) {
                  // exact box dimension only required when complex text was build
                  arg.box = getElementRect(txt, 'bbox');
               }

               if (arg.plain) {
                  txt.attr("text-anchor", arg.align[0]);
                  if (arg.align[1] == 'top')
                     txt.attr("dy", ".8em");
                  else if (arg.align[1] == 'middle') {
                     if (JSROOT.nodejs) txt.attr("dy", ".4em"); else txt.attr("dominant-baseline", "middle");
                  }
               } else {
                  txt.attr("text-anchor", "start");
                  arg.dx = ((arg.align[0] == "middle") ? -0.5 : ((arg.align[0] == "end") ? -1 : 0)) * arg.box.width;
                  arg.dy = ((arg.align[1] == 'top') ? (arg.top_shift || 1) : (arg.align[1] == 'middle') ? (arg.mid_shift || 0.5) : 0) * arg.box.height;
               }

            } else if (arg.text_rect) {

               // handle latext drawing
               let box = arg.text_rect;

               scale = (f > 0) && (Math.abs(1-f)>0.01) ? 1/f : 1;

               arg.dx = ((arg.align[0] == "middle") ? -0.5 : ((arg.align[0] == "end") ? -1 : 0)) * box.width * scale;

               if (arg.align[1] == 'top')
                  arg.dy = -box.y1*scale;
               else if (arg.align[1] == 'bottom')
                  arg.dy = -box.y2*scale;
               else if (arg.align[1] == 'middle')
                  arg.dy = -0.5*(box.y1 + box.y2)*scale;
            } else {
               console.error('text rect not calcualted - please check code');
            }

            if (!arg.rotate) { arg.x += arg.dx; arg.y += arg.dy; arg.dx = arg.dy = 0; }

            // use translate and then rotate to avoid complex sign calculations
            let trans = "";
            if (arg.y)
               trans = `translate(${Math.round(arg.x)},${Math.round(arg.y)})`;
            else if (arg.x)
               trans = `translate(${Math.round(arg.x)})`;
            if (arg.rotate)
               trans += ` rotate(${Math.round(arg.rotate)})`;
            if (scale !== 1)
               trans += ` scale(${scale.toFixed(3)})`;
            if (arg.dy)
               trans += ` translate(${Math.round(arg.dx)},${Math.round(arg.dy)})`;
            else if (arg.dx)
               trans += ` translate(${Math.round(arg.dx)})`;
            if (trans) txt.attr("transform", trans);
         });


         // when no any normal text drawn - remove font attributes
         if (!any_text)
            font.clearFont(draw_g);

         if ((optimize_arr !== null) && (optimize_arr.length > 1))
            ["fill", "text-anchor"].forEach(name => {
               let first = optimize_arr[0].attr(name);
               optimize_arr.forEach(txt_node => {
                  let value = txt_node.attr(name);
                  if (!value || (value !== first)) first = undefined;
               });
               if (first) {
                  draw_g.attr(name, first);
                  optimize_arr.forEach(txt_node => { txt_node.attr(name, null); });
               }
            });

         // if specified, call resolve function
         if (resolveFunc) resolveFunc(this); // IMPORTANT - return painter, may use in draw methods
      }

      /** @summary Post-process plain text drawing
        * @private */
      _postprocessDrawText(arg, txt_node) {
         // complete rectangle with very rougth size estimations
         arg.box = !JSROOT.nodejs && !JSROOT.settings.ApproxTextSize && !arg.fast ? getElementRect(txt_node, 'bbox') :
                  (arg.text_rect || { height: arg.font_size * 1.2, width: arg.text.length * arg.font_size * arg.font.aver_width });

         txt_node.attr('visibility', 'hidden'); // hide elements until text drawing is finished

         if (arg.box.width > arg.draw_g.property('max_text_width'))
            arg.draw_g.property('max_text_width', arg.box.width);
         if (arg.scale)
            this.scaleTextDrawing(Math.max(1.05 * arg.box.width / arg.width, 1. * arg.box.height / arg.height), arg.draw_g);

         arg.result_width = arg.box.width;
         arg.result_height = arg.box.height;

         if (typeof arg.post_process == 'function')
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
        * @param {number} [arg.latex] - 0 - plain text, 1 - normal TLatex, 2 - math
        * @param {string} [arg.color=black] - text color
        * @param {number} [arg.rotate] - rotaion angle
        * @param {number} [arg.font_size] - fixed font size
        * @param {object} [arg.draw_g] - element where to place text, if not specified central draw_g container is used
        * @param {function} [arg.post_process] - optional function called when specified text is drawn
        * @protected */
      drawText(arg) {

         if (!arg.text) arg.text = "";

         arg.draw_g = arg.draw_g || this.draw_g;
         if (!arg.draw_g || arg.draw_g.empty()) return;

         let font = arg.draw_g.property('text_font');
         arg.font = font; // use in latex conversion

         if (font) {
            if (font.color && !arg.color) arg.color = font.color;
            if (font.align && !arg.align) arg.align = font.align;
            if (font.angle && !arg.rotate) arg.rotate = font.angle;
         }

         let align = ['start', 'middle'];

         if (typeof arg.align == 'string') {
            align = arg.align.split(";");
            if (align.length == 1) align.push('middle');
         } else if (typeof arg.align == 'number') {
            if ((arg.align / 10) >= 3)
               align[0] = 'end';
            else if ((arg.align / 10) >= 2)
               align[0] = 'middle';
            if ((arg.align % 10) == 0)
               align[1] = 'bottom';
            else if ((arg.align % 10) == 1)
               align[1] = 'bottom-base';
            else if ((arg.align % 10) == 3)
               align[1] = 'top';
         } else if (arg.align && (typeof arg.align == 'object') && (arg.align.length == 2)) {
            align = arg.align;
         }

         if (arg.latex === undefined) arg.latex = 1; //  latex 0-text, 1-latex, 2-math
         arg.align = align;
         arg.x = arg.x || 0;
         arg.y = arg.y || 0;
         arg.scale = arg.width && arg.height && !arg.font_size;
         arg.width = arg.width || 0;
         arg.height = arg.height || 0;

         if (arg.draw_g.property("_fast_drawing")) {
            if (arg.scale) {
               // area too small - ignore such drawing
               if (arg.height < 4) return 0;
            } else if (arg.font_size) {
               // font size too small
               if (arg.font_size < 4) return 0;
            } else if (arg.draw_g.property("_font_too_small")) {
               // configure font is too small - ignore drawing
               return 0;
            }
         }

         // include drawing into list of all args
         arg.draw_g.property('all_args').push(arg);
         arg.ready = false; // indicates if drawing is ready for post-processing

         let use_mathjax = (arg.latex == 2);

         if (arg.latex === 1)
            use_mathjax = (JSROOT.settings.Latex == JSROOT.constants.Latex.AlwaysMathJax) ||
                          ((JSROOT.settings.Latex == JSROOT.constants.Latex.MathJax) && arg.text.match(/[#{\\]/g));

         if (!use_mathjax || arg.nomathjax) {

            arg.txt_node = arg.draw_g.append("svg:text");

            if (arg.color) arg.txt_node.attr("fill", arg.color);

            if (arg.font_size) arg.txt_node.attr("font-size", arg.font_size);
                          else arg.font_size = font.size;

            arg.plain = !arg.latex || (JSROOT.settings.Latex == JSROOT.constants.Latex.Off) || (JSROOT.settings.Latex == JSROOT.constants.Latex.Symbols);

            arg.simple_latex = arg.latex && (JSROOT.settings.Latex == JSROOT.constants.Latex.Symbols);

            if (!arg.plain || arg.simple_latex || (arg.font && arg.font.isSymbol)) {
               JSROOT.require(['latex']).then(ltx => {
                  if (arg.simple_latex || ltx.isPlainText(arg.text) || arg.plain) {
                     arg.simple_latex = true;
                     ltx.producePlainText(this, arg.txt_node, arg);
                  } else {
                     arg.txt_node.remove(); // just remove text node,
                     delete arg.txt_node;
                     arg.txt_g = arg.draw_g.append("svg:g");
                     ltx.produceLatex(this, arg.txt_g, arg);
                  }
                  arg.ready = true;
                  this._postprocessDrawText(arg, arg.txt_g || arg.txt_node);

                  if (arg.draw_g.property('draw_text_completed'))
                     this._checkAllTextDrawing(arg.draw_g); // check if all other elements are completed
               });
               return 0;
            }

            arg.plain = true;
            arg.txt_node.text(arg.text);
            arg.ready = true;

            return this._postprocessDrawText(arg, arg.txt_node);
         }

         arg.mj_node = arg.draw_g.append("svg:g")
                              .attr('visibility', 'hidden'); // hide text until drawing is finished

         JSROOT.require(['latex'])
               .then(ltx => ltx.produceMathjax(this, arg.mj_node, arg))
               .then(() => {
                  arg.ready = true;
                  if (arg.draw_g.property('draw_text_completed'))
                     this._checkAllTextDrawing(arg.draw_g);
               });

         return 0;
      }

      /** @summary Finish text drawing
        * @desc Should be called to complete all text drawing operations
        * @param {function} [draw_g] - <g> element for text drawing, this.draw_g used when not specified
        * @returns {Promise} when text drawing completed
        * @protected */
      finishTextDrawing(draw_g, try_optimize) {
         if (!draw_g) draw_g = this.draw_g;
         if (!draw_g || draw_g.empty())
            return Promise.resolve(false);

         draw_g.property('draw_text_completed', true); // mark that text drawing is completed

         return new Promise(resolveFunc => {
            this._checkAllTextDrawing(draw_g, resolveFunc, try_optimize);
         });
      }

      /** @summary Configure user-defined context menu for the object
        * @desc fillmenu_func will be called when context menu is actiavted
        * Arguments fillmenu_func are (menu,kind)
        * First is JSROOT menu object, second is object subelement like axis "x" or "y"
        * Function should return promise with menu when items are filled
        * @param {function} fillmenu_func - function to fill custom context menu for oabject */
      configureUserContextMenu(fillmenu_func) {

         if (!fillmenu_func || (typeof fillmenu_func !== 'function'))
            delete this._userContextMenuFunc;
         else
            this._userContextMenuFunc = fillmenu_func;
      }

      /** @summary Fill object menu in web canvas
        * @private */
      fillObjectExecMenu(menu, kind) {

         if (this._userContextMenuFunc)
            return this._userContextMenuFunc(menu, kind);

         let canvp = this.getCanvPainter();

         if (!this.snapid || !canvp || canvp._readonly || !canvp._websocket)
            return Promise.resolve(menu);

         function DoExecMenu(arg) {
            let execp = this.exec_painter || this,
               cp = execp.getCanvPainter(),
               item = execp.args_menu_items[parseInt(arg)];

            if (!item || !item.fName) return;

            // this is special entry, produced by TWebMenuItem, which recognizes editor entries itself
            if (item.fExec == "Show:Editor") {
               if (cp && (typeof cp.activateGed == 'function'))
                  cp.activateGed(execp);
               return;
            }

            if (cp && (typeof cp.executeObjectMethod == 'function'))
               if (cp.executeObjectMethod(execp, item, execp.args_menu_id)) return;

            if (execp.executeMenuCommand(item)) return;

            if (!execp.args_menu_id) return;

             if (!item.fArgs)
                return execp.submitCanvExec(item.fExec, execp.args_menu_id);

            item.fClassName = execp.getClassName();
            if ((execp.args_menu_id.indexOf("#x") > 0) || (execp.args_menu_id.indexOf("#y") > 0) || (execp.args_menu_id.indexOf("#z") > 0)) item.fClassName = "TAxis";

             menu.showMethodArgsDialog(item).then(args => {
                if (!args) return;
                if (execp.executeMenuCommand(item, args)) return;
                let exec = item.fExec.substr(0, item.fExec.length-1) + args + ')';
                if (cp) cp.sendWebsocket('OBJEXEC:' + execp.args_menu_id + ":" + exec);
            });
         }

         const DoFillMenu = (_menu, _reqid, _resolveFunc, reply) => {

            // avoid multiple call of the callback after timeout
            if (this._got_menu) return;
            this._got_menu = true;

            if (reply && (_reqid !== reply.fId))
               console.error('missmatch between request ' + _reqid + ' and reply ' + reply.fId + ' identifiers');

            let items = reply ? reply.fItems : null;

            if (items && items.length) {
               if (_menu.size() > 0)
                  _menu.add("separator");

               this.args_menu_items = items;
               this.args_menu_id = reply.fId;

               let lastclname;

               for (let n = 0; n < items.length; ++n) {
                  let item = items[n];

                  if (item.fClassName && lastclname && (lastclname != item.fClassName)) {
                     _menu.add("endsub:");
                     lastclname = "";
                  }
                  if (lastclname != item.fClassName) {
                     lastclname = item.fClassName;
                     let p = lastclname.lastIndexOf("::"),
                         shortname = (p > 0) ? lastclname.substr(p+2) : lastclname;

                     _menu.add("sub:" + shortname.replace(/[<>]/g,"_"));
                  }

                  if ((item.fChecked === undefined) || (item.fChecked < 0))
                     _menu.add(item.fName, n, DoExecMenu);
                  else
                     _menu.addchk(item.fChecked, item.fName, n, DoExecMenu);
               }

               if (lastclname) _menu.add("endsub:");
            }

            _resolveFunc(_menu);
         };

         let reqid = this.snapid;
         if (kind) reqid += "#" + kind; // use # to separate object id from member specifier like 'x' or 'z'

         this._got_menu = false;

         // if menu painter differs from this, remember it for further usage
         if (menu.painter)
            menu.painter.exec_painter = (menu.painter !== this) ? this : undefined;

         return new Promise(resolveFunc => {

            // set timeout to avoid menu hanging
            setTimeout(() => DoFillMenu(menu, reqid, resolveFunc), 2000);

            canvp.submitMenuRequest(this, kind, reqid).then(lst => DoFillMenu(menu, reqid, resolveFunc, lst));
         });
      }

      /** @summary Configure user-defined tooltip handler
        * @desc Hook for the users to get tooltip information when mouse cursor moves over frame area
        * Hanlder function will be called every time when new data is selected
        * when mouse leave frame area, handler(null) will be called
        * @param {function} handler - function called when tooltip is produced
        * @param {number} [tmout = 100] - delay in ms before tooltip delivered */
      configureUserTooltipHandler(handler, tmout) {
         if (!handler || (typeof handler !== 'function')) {
            delete this._user_tooltip_handler;
            delete this._user_tooltip_timeout;
         } else {
            this._user_tooltip_handler = handler;
            this._user_tooltip_timeout = tmout || 100;
         }
      }

       /** @summary Configure user-defined click handler
         * @desc Function will be called every time when frame click was perfromed
         * As argument, tooltip object with selected bins will be provided
         * If handler function returns true, default handling of click will be disabled
         * @param {function} handler - function called when mouse click is done */
      configureUserClickHandler(handler) {
         let fp = this.getFramePainter();
         if (fp && typeof fp.configureUserClickHandler == "function")
            fp.configureUserClickHandler(handler);
      }

      /** @summary Configure user-defined dblclick handler
        * @desc Function will be called every time when double click was called
        * As argument, tooltip object with selected bins will be provided
        * If handler function returns true, default handling of dblclick (unzoom) will be disabled
        * @param {function} handler - function called when mouse double click is done */
      configureUserDblclickHandler(handler) {
         let fp = this.getFramePainter();
         if (fp && typeof fp.configureUserDblclickHandler == "function")
            fp.configureUserDblclickHandler(handler);
      }

      /** @summary Check if user-defined tooltip function was configured
        * @returns {boolean} flag is user tooltip handler was configured */
      hasUserTooltip() {
         return typeof this._user_tooltip_handler == 'function';
      }

      /** @summary Provide tooltips data to user-defined function
        * @param {object} data - tooltip data
        * @private */
      provideUserTooltip(data) {

         if (!this.hasUserTooltip()) return;

         if (this._user_tooltip_timeout <= 0)
            return this._user_tooltip_handler(data);

         if (this._user_tooltip_handle) {
            clearTimeout(this._user_tooltip_handle);
            delete this._user_tooltip_handle;
         }

         if (!data)
            return this._user_tooltip_handler(data);

         let d = data;

         // only after timeout user function will be called
         this._user_tooltip_handle = setTimeout(() => {
            delete this._user_tooltip_handle;
            if (this._user_tooltip_handler) this._user_tooltip_handler(d);
         }, this._user_tooltip_timeout);
      }

      /** @summary Provide projection areas
        * @param kind - "X", "Y" or ""
        * @private */
      provideSpecialDrawArea(kind) {
         if (kind == this._special_draw_area)
            return Promise.resolve(true);

         return this.getCanvPainter().toggleProjection(kind).then(() => {
            this._special_draw_area = kind;
            return true;
         });
      }

      /** @summary Provide projection areas
        * @param kind - "X", "Y" or ""
        * @private */
      drawInSpecialArea(obj, opt) {
         let canp = this.getCanvPainter();
         if (!this._special_draw_area || !canp || typeof canp.drawProjection !== "function")
            return Promise.resolve(false);

         return canp.drawProjection(this._special_draw_area, obj, opt);
      }

      /** @summary Get tooltip for painter and specified event position
        * @param {Object} evnt - object wiith clientX and clientY positions
        * @private */
      getToolTip(evnt) {
         if (!evnt || (evnt.clientX === undefined) || (evnt.clientY === undefined)) return null;

         let frame = this.getFrameSvg();
         if (frame.empty()) return null;
         let layer = frame.select(".main_layer");
         if (layer.empty()) return null;

         let pos = d3.pointer(evnt, layer.node()),
             pnt = { touch: false, x: pos[0], y: pos[1] };

         if (typeof this.extractToolTip == 'function')
            return this.extractToolTip(pnt);

         pnt.disabled = true;

         let res = (typeof this.processTooltipEvent == 'function') ? this.processTooltipEvent(pnt) : null;

         return res && res.user_info ? res.user_info : res;
      }

   } // ObjectPainter

   // ===========================================================


   /**
     * @summary Base painter methods
     *
     * @private
     */

   JSROOT.AxisPainterMethods = {

      initAxisPainter() {
         this.name = "yaxis";
         this.kind = "normal";
         this.func = null;
         this.order = 0; // scaling order for axis labels

         this.full_min = 0;
         this.full_max = 1;
         this.scale_min = 0;
         this.scale_max = 1;
         this.ticks = []; // list of major ticks
      },

      /** @summary Cleanup axis painter */
      cleanupAxisPainter() {
         this.ticks = [];
         delete this.format;
         delete this.func;
         delete this.tfunc1;
         delete this.tfunc2;
         delete this.gr;
      },

      /** @summary Assign often used members of frame painter */
      assignFrameMembers(fp, axis) {
         fp["gr"+axis] = this.gr;                    // fp.grx
         fp["log"+axis] = this.log;                  // fp.logx
         fp["scale_"+axis+"min"] = this.scale_min;   // fp.scale_xmin
         fp["scale_"+axis+"max"] = this.scale_max;   // fp.scale_xmax
      },

      /** @summary Convert axis value into the Date object */
      convertDate(v) {
         return new Date(this.timeoffset + v*1000);
      },

      /** @summary Convert graphical point back into axis value */
      revertPoint(pnt) {
         let value = this.func.invert(pnt);
         return (this.kind == "time") ?  (value - this.timeoffset) / 1000 : value;
      },

      /** @summary Provide label for time axis */
      formatTime(d, asticks) {
         return asticks ? this.tfunc1(d) : this.tfunc2(d);
      },

      /** @summary Provide label for log axis */
      formatLog(d, asticks, fmt) {
         let val = parseFloat(d), rnd = Math.round(val);
         if (!asticks)
            return ((rnd === val) && (Math.abs(rnd)<1e9)) ? rnd.toString() : jsrp.floatToString(val, fmt || JSROOT.gStyle.fStatFormat);
         if (val <= 0) return null;
         let vlog = Math.log10(val), base = this.logbase;
         if (base !== 10) vlog = vlog / Math.log10(base);
         if (this.moreloglabels || (Math.abs(vlog - Math.round(vlog))<0.001)) {
            if (!this.noexp && (asticks != 2))
               return this.formatExp(base, Math.floor(vlog+0.01), val);

            return vlog < 0 ? val.toFixed(Math.round(-vlog+0.5)) : val.toFixed(0);
         }
         return null;
      },

      /** @summary Provide label for normal axis */
      formatNormal(d, asticks, fmt) {
         let val = parseFloat(d);
         if (asticks && this.order) val = val / Math.pow(10, this.order);

         if (val === Math.round(val))
            return Math.abs(val) < 1e9 ? val.toFixed(0) : val.toExponential(4);

         if (asticks) return (this.ndig>10) ? val.toExponential(this.ndig-11) : val.toFixed(this.ndig);

         return jsrp.floatToString(val, fmt || JSROOT.gStyle.fStatFormat);
      },

      /** @summary Provide label for exponential form */
      formatExp(base, order, value) {
         let res = "";
         if (value) {
            value = Math.round(value/Math.pow(base,order));
            if ((value!=0) && (value!=1)) res = value.toString() + (JSROOT.settings.Latex ? "#times" : "x");
         }
         if (Math.abs(base-Math.exp(1)) < 0.001)
            res += "e";
         else
            res += base.toString();
         if (JSROOT.settings.Latex > JSROOT.constants.Latex.Symbols)
            return res + "^{" + order + "}";
         const superscript_symbols = {
               '0': '\u2070', '1': '\xB9', '2': '\xB2', '3': '\xB3', '4': '\u2074', '5': '\u2075',
               '6': '\u2076', '7': '\u2077', '8': '\u2078', '9': '\u2079', '-': '\u207B'
            };
         let str = order.toString();
         for (let n = 0; n < str.length; ++n)
            res += superscript_symbols[str[n]];
         return res;
      },

      /** @summary Convert "raw" axis value into text */
      axisAsText(value, fmt) {
         if (this.kind == 'time')
            value = this.convertDate(value);
         if (this.format)
            return this.format(value, false, fmt);
         return value.toPrecision(4);
      },

      /** @summary Produce ticks for d3.scaleLog
        * @desc Fixing following problem, described [here]{@link https://stackoverflow.com/questions/64649793} */
      poduceLogTicks(func, number) {
         const linearArray = arr => {
            let sum1 = 0, sum2 = 0;
            for (let k = 1; k < arr.length; ++k) {
               let diff = (arr[k] - arr[k-1]);
               sum1 += diff;
               sum2 += diff*diff;
            }
            let mean = sum1/(arr.length-1),
                dev = sum2/(arr.length-1) - mean*mean;

            if (dev <= 0) return true;
            if (Math.abs(mean) < 1e-100) return false;
            return Math.sqrt(dev)/mean < 1e-6;
         };

         let arr = func.ticks(number);

         while ((number > 4) && linearArray(arr)) {
            number = Math.round(number*0.8);
            arr = func.ticks(number);
         }

         // if still linear array, try to sort out "bad" ticks
         if ((number < 5) && linearArray(arr) && this.logbase && (this.logbase != 10)) {
            let arr2 = [];
            arr.forEach(val => {
               let pow = Math.log10(val) / Math.log10(this.logbase);
               if (Math.abs(Math.round(pow) - pow) < 0.01) arr2.push(val);
            });
            if (arr2.length > 0) arr = arr2;
         }

         return arr;
      },

      /** @summary Produce axis ticks */
      produceTicks(ndiv, ndiv2) {
         if (!this.noticksopt) {
            let total = ndiv * (ndiv2 || 1);

            if (this.log) return this.poduceLogTicks(this.func, total);

            let dom = this.func.domain();

            const check = ticks => {
               if (ticks.length <= total) return true;
               if (ticks.length > total + 1) return false;
               return (ticks[0] === dom[0]) || (ticks[total] === dom[1]); // special case of N+1 ticks, but match any range
            };

            let res1 = this.func.ticks(total);
            if (ndiv2 || check(res1)) return res1;

            let res2 = this.func.ticks(Math.round(total * 0.7));
            return (res2.length > 2) && check(res2) ? res2 : res1;
         }

         let dom = this.func.domain(), ticks = [];
         if (ndiv2) ndiv = (ndiv-1) * ndiv2;
         for (let n = 0; n <= ndiv; ++n)
            ticks.push((dom[0]*(ndiv-n) + dom[1]*n)/ndiv);
         return ticks;
      },

      /** @summary Method analyze mouse wheel event and returns item with suggested zooming range */
      analyzeWheelEvent(evnt, dmin, item, test_ignore) {
         if (!item) item = {};

         let delta = 0, delta_left = 1, delta_right = 1;

         if ('dleft' in item) { delta_left = item.dleft; delta = 1; }
         if ('dright' in item) { delta_right = item.dright; delta = 1; }

         if (item.delta) {
            delta = item.delta;
         } else if (evnt) {
            delta = evnt.wheelDelta ? -evnt.wheelDelta : (evnt.deltaY || evnt.detail);
         }

         if (!delta || (test_ignore && item.ignore)) return;

         delta = (delta < 0) ? -0.2 : 0.2;
         delta_left *= delta;
         delta_right *= delta;

         let lmin = item.min = this.scale_min,
             lmax = item.max = this.scale_max,
             gmin = this.full_min,
             gmax = this.full_max;

         if ((item.min === item.max) && (delta < 0)) {
            item.min = gmin;
            item.max = gmax;
         }

         if (item.min >= item.max) return;

         if (item.reverse) dmin = 1 - dmin;

         if ((dmin > 0) && (dmin < 1)) {
            if (this.log) {
               let factor = (item.min>0) ? Math.log10(item.max/item.min) : 2;
               if (factor>10) factor = 10; else if (factor<0.01) factor = 0.01;
               item.min = item.min / Math.pow(10, factor*delta_left*dmin);
               item.max = item.max * Math.pow(10, factor*delta_right*(1-dmin));
            } else if ((delta_left === -delta_right) && !item.reverse) {
               // shift left/right, try to keep range constant
               let delta = (item.max - item.min) * delta_right * dmin;

               if ((Math.round(item.max) === item.max) && (Math.round(item.min) === item.min) && (Math.abs(delta) > 1)) delta = Math.round(delta);

               if (item.min + delta < gmin)
                  delta = gmin - item.min;
               else if (item.max + delta > gmax)
                  delta = gmax - item.max;

               if (delta != 0) {
                  item.min += delta;
                  item.max += delta;
                } else {
                  delete item.min;
                  delete item.max;
               }

            } else {
               let rx_left = (item.max - item.min), rx_right = rx_left;
               if (delta_left > 0) rx_left = 1.001 * rx_left / (1-delta_left);
               item.min += -delta_left*dmin*rx_left;
               if (delta_right > 0) rx_right = 1.001 * rx_right / (1-delta_right);
               item.max -= -delta_right*(1-dmin)*rx_right;
            }
            if (item.min >= item.max) {
               item.min = item.max = undefined;
            } else if (delta_left !== delta_right) {
               // extra check case when moving left or right
               if (((item.min < gmin) && (lmin === gmin)) ||
                   ((item.max > gmax) && (lmax === gmax)))
                      item.min = item.max = undefined;
            } else {
               if (item.min < gmin) item.min = gmin;
               if (item.max > gmax) item.max = gmax;
            }
         } else {
            item.min = item.max = undefined;
         }

         item.changed = ((item.min !== undefined) && (item.max !== undefined));

         return item;
      }

   }; // AxisPainterMethods

   // ===========================================================

   /** @summary Set active pad painter
     * @desc Normally be used to handle key press events, which are global in the web browser
     * @param {object} args - functions arguments
     * @param {object} args.pp - pad painter
     * @param {boolean} [args.active] - is pad activated or not
     * @private */
   jsrp.selectActivePad = function(args) {
      if (args.active) {
         let fp = this.$active_pp ? this.$active_pp.getFramePainter() : null;
         if (fp) fp.setFrameActive(false);

         this.$active_pp = args.pp;

         fp = this.$active_pp ? this.$active_pp.getFramePainter() : null;
         if (fp) fp.setFrameActive(true);
      } else if (this.$active_pp === args.pp) {
         delete this.$active_pp;
      }
   }

   /** @summary Returns current active pad
     * @desc Should be used only for keyboard handling
     * @private */
   jsrp.getActivePad = function() {
      return this.$active_pp;
   }

   // ================= painter of raw text ========================================

   /** @summary Generic text drawing
     * @private */
   jsrp.drawRawText = function(dom, txt /*, opt*/) {

      let painter = new BasePainter(dom);
      painter.txt = txt;

      painter.redrawObject = function(obj) {
         this.txt = obj;
         this.drawText();
         return true;
      }

      painter.drawText = function() {
         let txt = (this.txt._typename && (this.txt._typename == "TObjString")) ? this.txt.fString : this.txt.value;
         if (typeof txt != 'string') txt = "<undefined>";

         let mathjax = this.txt.mathjax || (JSROOT.settings.Latex == JSROOT.constants.Latex.AlwaysMathJax);

         if (!mathjax && !('as_is' in this.txt)) {
            let arr = txt.split("\n"); txt = "";
            for (let i = 0; i < arr.length; ++i)
               txt += "<pre style='margin:0'>" + arr[i] + "</pre>";
         }

         let frame = this.selectDom(),
            main = frame.select("div");
         if (main.empty())
            main = frame.append("div").attr('style', 'max-width:100%;max-height:100%;overflow:auto');
         main.html(txt);

         // (re) set painter to first child element, base painter not requires canvas
         this.setTopPainter();

         if (mathjax)
            return JSROOT.require('latex').then(ltx => { ltx.typesetMathjax(frame.node()); return this; });

         return Promise.resolve(this);
      }

      return painter.drawText();
   }

   /** @summary Register handle to react on window resize
     * @desc function used to react on browser window resize event
     * While many resize events could come in short time,
     * resize will be handled with delay after last resize event
     * @param {object|string} handle can be function or object with checkResize function or dom where painting was done
     * @param {number} [delay] - one could specify delay after which resize event will be handled
     * @protected */
   jsrp.registerForResize = function(handle, delay) {

      if (!handle || JSROOT.batch_mode || (typeof window == 'undefined')) return;

      let myInterval = null, myDelay = delay ? delay : 300;

      if (myDelay < 20) myDelay = 20;

      function ResizeTimer() {
         myInterval = null;

         document.body.style.cursor = 'wait';
         if (typeof handle == 'function')
            handle();
         else if (handle && (typeof handle == 'object') && (typeof handle.checkResize == 'function')) {
            handle.checkResize();
         } else {
            let node = new BasePainter(handle).selectDom();
            if (!node.empty()) {
               let mdi = node.property('mdi');
               if (mdi && typeof mdi.checkMDIResize == 'function') {
                  mdi.checkMDIResize();
               } else {
                  JSROOT.resize(node.node());
               }
            }
         }
         document.body.style.cursor = 'auto';
      }

      window.addEventListener('resize', () => {
         if (myInterval !== null) clearTimeout(myInterval);
         myInterval = setTimeout(ResizeTimer, myDelay);
      });
   }

   // list of registered draw functions
   let drawFuncs = { lst: [
      { name: "TCanvas", icon: "img_canvas", prereq: "gpad", class: "TCanvasPainter", opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" },
      { name: "TPad", icon: "img_canvas", prereq: "gpad", class: "TPadPainter", opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" },
      { name: "TSlider", icon: "img_canvas", prereq: "gpad", class: "TPadPainter" },
      { name: "TFrame", icon: "img_frame", prereq: "gpad", class: "TFramePainter" },
      { name: "TPave", icon: "img_pavetext", prereq: "hist", class: "TPavePainter" },
      { name: "TPaveText", icon: "img_pavetext", prereq: "hist", class: "TPavePainter" },
      { name: "TPavesText", icon: "img_pavetext", prereq: "hist", class: "TPavePainter" },
      { name: "TPaveStats", icon: "img_pavetext", prereq: "hist", class: "TPavePainter" },
      { name: "TPaveLabel", icon: "img_pavelabel", prereq: "hist", class: "TPavePainter" },
      { name: "TDiamond", icon: "img_pavelabel", prereq: "hist", class: "TPavePainter" },
      { name: "TLatex", icon: "img_text", prereq: "more", func: ".drawText", direct: true },
      { name: "TMathText", icon: "img_text", prereq: "more", func: ".drawText", direct: true },
      { name: "TText", icon: "img_text", prereq: "more", func: ".drawText", direct: true },
      { name: /^TH1/, icon: "img_histo1d", prereq: "hist", class: "TH1Painter", opt: ";hist;P;P0;E;E1;E2;E3;E4;E1X0;L;LF2;B;B1;A;TEXT;LEGO;same", ctrl: "l" },
      { name: "TProfile", icon: "img_profile", prereq: "hist", class: "TH1Painter", opt: ";E0;E1;E2;p;AH;hist" },
      { name: "TH2Poly", icon: "img_histo2d", prereq: "hist", class: "TH2Painter", opt: ";COL;COL0;COLZ;LCOL;LCOL0;LCOLZ;LEGO;TEXT;same", expand_item: "fBins", theonly: true },
      { name: "TProfile2Poly", sameas: "TH2Poly" },
      { name: "TH2PolyBin", icon: "img_histo2d", draw_field: "fPoly", draw_field_opt: "L" },
      { name: /^TH2/, icon: "img_histo2d", prereq: "hist", class: "TH2Painter", dflt: "col", opt: ";COL;COLZ;COL0;COL1;COL0Z;COL1Z;COLA;BOX;BOX1;PROJ;PROJX1;PROJX2;PROJX3;PROJY1;PROJY2;PROJY3;SCAT;TEXT;TEXTE;TEXTE0;CANDLE;CANDLE1;CANDLE2;CANDLE3;CANDLE4;CANDLE5;CANDLE6;CANDLEY1;CANDLEY2;CANDLEY3;CANDLEY4;CANDLEY5;CANDLEY6;VIOLIN;VIOLIN1;VIOLIN2;VIOLINY1;VIOLINY2;CONT;CONT1;CONT2;CONT3;CONT4;ARR;SURF;SURF1;SURF2;SURF4;SURF6;E;A;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same", ctrl: "lego" },
      { name: "TProfile2D", sameas: "TH2" },
      { name: /^TH3/, icon: 'img_histo3d', prereq: "hist3d", class: "TH3Painter", opt: ";SCAT;BOX;BOX2;BOX3;GLBOX1;GLBOX2;GLCOL" },
      { name: "THStack", icon: "img_histo1d", prereq: "hist", class: "THStackPainter", expand_item: "fHists", opt: "NOSTACK;HIST;E;PFC;PLC" },
      { name: "TPolyMarker3D", icon: 'img_histo3d', prereq: "base3d", func: ".drawPolyMarker3D", direct: true, frame: "3d" },
      { name: "TPolyLine3D", icon: 'img_graph', prereq: "base3d", func: ".drawPolyLine3D", direct: true, frame: "3d" },
      { name: "TGraphStruct" },
      { name: "TGraphNode" },
      { name: "TGraphEdge" },
      { name: "TGraphTime", icon: "img_graph", prereq: "more", class: "TGraphTimePainter", opt: "once;repeat;first", theonly: true },
      { name: "TGraph2D", icon: "img_graph", prereq: "hist3d", class: "TGraph2DPainter", opt: ";P;PCOL", theonly: true },
      { name: "TGraph2DErrors", icon: "img_graph", prereq: "hist3d", class: "TGraph2DPainter", opt: ";P;PCOL;ERR", theonly: true },
      { name: "TGraphPolargram", icon: "img_graph", prereq: "more", class: "TGraphPolargramPainter", theonly: true },
      { name: "TGraphPolar", icon: "img_graph", prereq: "more", class: "TGraphPolarPainter", opt: ";F;L;P;PE", theonly: true },
      { name: /^TGraph/, icon: "img_graph", prereq: "more", class: "TGraphPainter", opt: ";L;P" },
      { name: "TEfficiency", icon: "img_graph", prereq: "more", class: "TEfficiencyPainter", opt: ";AP" },
      { name: "TCutG", sameas: "TGraph" },
      { name: /^RooHist/, sameas: "TGraph" },
      { name: /^RooCurve/, sameas: "TGraph" },
      { name: "RooPlot", icon: "img_canvas", prereq: "more", func: ".drawRooPlot" },
      { name: "TRatioPlot", icon: "img_mgraph", prereq: "more", class: "TRatioPlotPainter", opt: "" },
      { name: "TMultiGraph", icon: "img_mgraph", prereq: "more", class: "TMultiGraphPainter", expand_item: "fGraphs" },
      { name: "TStreamerInfoList", icon: 'img_question', prereq: "hierarchy", func: ".drawStreamerInfo" },
      { name: "TPaletteAxis", icon: "img_colz", prereq: "hist", class: "TPavePainter" },
      { name: "TWebPainting", icon: "img_graph", prereq: "more", func: ".drawWebPainting" },
      { name: "TCanvasWebSnapshot", icon: "img_canvas", prereq: "gpad", func: ".drawTPadSnapshot" },
      { name: "TPadWebSnapshot", sameas: "TCanvasWebSnapshot" },
      { name: "kind:Text", icon: "img_text", func: jsrp.drawRawText },
      { name: "TObjString", icon: "img_text", func: jsrp.drawRawText },
      { name: "TF1", icon: "img_tf1", prereq: "math;more", class: "TF1Painter" },
      { name: "TF2", icon: "img_tf2", prereq: "math;hist", func: ".drawTF2" },
      { name: "TSpline3", icon: "img_tf1", prereq: "more", class: "TSplinePainter" },
      { name: "TSpline5", icon: "img_tf1", prereq: "more", class: "TSplinePainter" },
      { name: "TEllipse", icon: 'img_graph', prereq: "more", func: ".drawEllipse", direct: true },
      { name: "TArc", sameas: 'TEllipse' },
      { name: "TCrown", sameas: 'TEllipse' },
      { name: "TPie", icon: 'img_graph', prereq: "more", func: ".drawPie", direct: true },
      { name: "TPieSlice", icon: 'img_graph', dummy: true },
      { name: "TExec", icon: "img_graph", dummy: true },
      { name: "TLine", icon: 'img_graph', prereq: "more", func: ".drawLine", direct: true },
      { name: "TArrow", icon: 'img_graph', prereq: "more", func: ".drawArrow", direct: true },
      { name: "TPolyLine", icon: 'img_graph', prereq: "more", func: ".drawPolyLine", direct: true },
      { name: "TCurlyLine", sameas: 'TPolyLine' },
      { name: "TCurlyArc", sameas: 'TPolyLine' },
      { name: "TParallelCoord", icon: "img_graph", dummy: true },
      { name: "TGaxis", icon: "img_graph", prereq: "gpad", class: "TAxisPainter" },
      { name: "TLegend", icon: "img_pavelabel", prereq: "hist", class: "TPavePainter" },
      { name: "TBox", icon: 'img_graph', prereq: "more", func: ".drawBox", direct: true },
      { name: "TWbox", icon: 'img_graph', prereq: "more", func: ".drawBox", direct: true },
      { name: "TSliderBox", icon: 'img_graph', prereq: "more", func: ".drawBox", direct: true },
      { name: "TMarker", icon: 'img_graph', prereq: "more", func: ".drawMarker", direct: true },
      { name: "TPolyMarker", icon: 'img_graph', prereq: "more", func: ".drawPolyMarker", direct: true },
      { name: "TASImage", icon: 'img_mgraph', prereq: "more", class: "TASImagePainter", opt: ";z" },
      { name: "TJSImage", icon: 'img_mgraph', prereq: "more", func: ".drawJSImage", opt: ";scale;center" },
      { name: "TGeoVolume", icon: 'img_histo3d', prereq: "geom", class: "TGeoPainter", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;wire;no_screen;dflt", ctrl: "dflt" },
      { name: "TEveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", class: "TGeoPainter", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;wire;dflt", ctrl: "dflt" },
      { name: "ROOT::Experimental::REveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", class: "TGeoPainter", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;wire;dflt", ctrl: "dflt" },
      { name: "TGeoOverlap", icon: 'img_histo3d', prereq: "geom", expand: "JSROOT.GEO.expandObject", class: "TGeoPainter", opt: ";more;all;count;projx;projz;wire;dflt", dflt: "dflt", ctrl: "expand" },
      { name: "TGeoManager", icon: 'img_histo3d', prereq: "geom", expand: "JSROOT.GEO.expandObject", class: "TGeoPainter", opt: ";more;all;count;projx;projz;wire;tracks;no_screen;dflt", dflt: "expand", ctrl: "dflt" },
      { name: /^TGeo/, icon: 'img_histo3d', prereq: "geom", class: "TGeoPainter", expand: "JSROOT.GEO.expandObject", opt: ";more;all;axis;compa;count;projx;projz;wire;no_screen;dflt", dflt: "dflt", ctrl: "expand" },
      { name: "TAxis3D", icon: 'img_graph', prereq: "geom", func: ".drawAxis3D", direct: true },
      // these are not draw functions, but provide extra info about correspondent classes
      { name: "kind:Command", icon: "img_execute", execute: true },
      { name: "TFolder", icon: "img_folder", icon2: "img_folderopen", noinspect: true, prereq: "hierarchy", expand: ".folderHierarchy" },
      { name: "TTask", icon: "img_task", prereq: "hierarchy", expand: ".taskHierarchy", for_derived: true },
      { name: "TTree", icon: "img_tree", prereq: "tree", expand: 'JSROOT.treeHierarchy', func: 'JSROOT.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" },
      { name: "TNtuple", icon: "img_tree", prereq: "tree", expand: 'JSROOT.treeHierarchy', func: 'JSROOT.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" },
      { name: "TNtupleD", icon: "img_tree", prereq: "tree", expand: 'JSROOT.treeHierarchy', func: 'JSROOT.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" },
      { name: "TBranchFunc", icon: "img_leaf_method", prereq: "tree", func: 'JSROOT.drawTree', opt: ";dump", noinspect: true },
      { name: /^TBranch/, icon: "img_branch", prereq: "tree", func: 'JSROOT.drawTree', dflt: "expand", opt: ";dump", ctrl: "dump", shift: "inspect", ignore_online: true },
      { name: /^TLeaf/, icon: "img_leaf", prereq: "tree", noexpand: true, func: 'JSROOT.drawTree', opt: ";dump", ctrl: "dump", ignore_online: true },
      { name: "TList", icon: "img_list", prereq: "hierarchy", func: ".drawList", expand: ".listHierarchy", dflt: "expand" },
      { name: "THashList", sameas: "TList" },
      { name: "TObjArray", sameas: "TList" },
      { name: "TClonesArray", sameas: "TList" },
      { name: "TMap", sameas: "TList" },
      { name: "TColor", icon: "img_color" },
      { name: "TFile", icon: "img_file", noinspect: true },
      { name: "TMemFile", icon: "img_file", noinspect: true },
      { name: "TStyle", icon: "img_question", noexpand: true },
      { name: "Session", icon: "img_globe" },
      { name: "kind:TopFolder", icon: "img_base" },
      { name: "kind:Folder", icon: "img_folder", icon2: "img_folderopen", noinspect: true },
      { name: "ROOT::Experimental::RCanvas", icon: "img_canvas", prereq: "v7gpad", class: "RCanvasPainter", opt: "", expand_item: "fPrimitives" },
      { name: "ROOT::Experimental::RCanvasDisplayItem", icon: "img_canvas", prereq: "v7gpad", func: ".drawRPadSnapshot", opt: "", expand_item: "fPrimitives" }
   ], cache: {} };


   /** @summary Register draw function for the class
     * @desc List of supported draw options could be provided, separated  with ';'
     * @param {object} args - arguments
     * @param {string|regexp} args.name - class name or regexp pattern
     * @param {string} [args.prereq] - prerequicities to load before search for the draw function
     * @param {string} args.func - draw function name or just a function
     * @param {boolean} [args.direct] - if true, function is just Redraw() method of ObjectPainter
     * @param {string} [args.opt] - list of supported draw options (separated with semicolon) like "col;scat;"
     * @param {string} [args.icon] - icon name shown for the class in hierarchy browser
     * @param {string} [args.draw_field] - draw only data member from object, like fHistogram
     * @protected */
   jsrp.addDrawFunc = function(args) {
      drawFuncs.lst.push(args);
      return args;
   }

   /** @summary return draw handle for specified item kind
     * @desc kind could be ROOT.TH1I for ROOT classes or just
     * kind string like "Command" or "Text"
     * selector can be used to search for draw handle with specified option (string)
     * or just sequence id
     * @memberof JSROOT.Painter
     * @private */
   function getDrawHandle(kind, selector) {

      if (typeof kind != 'string') return null;
      if (selector === "") selector = null;

      let first = null;

      if ((selector === null) && (kind in drawFuncs.cache))
         return drawFuncs.cache[kind];

      let search = (kind.indexOf("ROOT.") == 0) ? kind.substr(5) : "kind:" + kind, counter = 0;
      for (let i = 0; i < drawFuncs.lst.length; ++i) {
         let h = drawFuncs.lst[i];
         if (typeof h.name == "string") {
            if (h.name != search) continue;
         } else {
            if (!search.match(h.name)) continue;
         }

         if (h.sameas !== undefined)
            return getDrawHandle("ROOT." + h.sameas, selector);

         if ((selector === null) || (selector === undefined)) {
            // store found handle in cache, can reuse later
            if (!(kind in drawFuncs.cache)) drawFuncs.cache[kind] = h;
            return h;
         } else if (typeof selector == 'string') {
            if (!first) first = h;
            // if drawoption specified, check it present in the list

            if (selector == "::expand") {
               if (('expand' in h) || ('expand_item' in h)) return h;
            } else
               if ('opt' in h) {
                  let opts = h.opt.split(';');
                  for (let j = 0; j < opts.length; ++j) opts[j] = opts[j].toLowerCase();
                  if (opts.indexOf(selector.toLowerCase()) >= 0) return h;
               }
         } else if (selector === counter) {
            return h;
         }
         ++counter;
      }

      return first;
   }

   /** @summary Scan streamer infos for derived classes
     * @desc Assign draw functions for such derived classes
     * @private */
   jsrp.addStreamerInfos = function(lst) {
      if (!lst) return;

      function CheckBaseClasses(si, lvl) {
         if (!si.fElements) return null;
         if (lvl > 10) return null; // protect against recursion

         for (let j = 0; j < si.fElements.arr.length; ++j) {
            // extract streamer info for each class member
            let element = si.fElements.arr[j];
            if (element.fTypeName !== 'BASE') continue;

            let handle = getDrawHandle("ROOT." + element.fName);
            if (handle && !handle.for_derived) handle = null;

            // now try find that base class of base in the list
            if (handle === null)
               for (let k = 0; k < lst.arr.length; ++k)
                  if (lst.arr[k].fName === element.fName) {
                     handle = CheckBaseClasses(lst.arr[k], lvl + 1);
                     break;
                  }

            if (handle && handle.for_derived) return handle;
         }
         return null;
      }

      for (let n = 0; n < lst.arr.length; ++n) {
         let si = lst.arr[n];
         if (getDrawHandle("ROOT." + si.fName) !== null) continue;

         let handle = CheckBaseClasses(si, 0);

         if (!handle) continue;

         let newhandle = JSROOT.extend({}, handle);
         // delete newhandle.for_derived; // should we disable?
         newhandle.name = si.fName;
         drawFuncs.lst.push(newhandle);
      }
   }

   /** @summary Provide draw settings for specified class or kind
     * @memberof JSROOT.Painter
     * @private */
   function getDrawSettings(kind, selector) {
      let res = { opts: null, inspect: false, expand: false, draw: false, handle: null };
      if (typeof kind != 'string') return res;
      let isany = false, noinspect = false, canexpand = false;
      if (typeof selector !== 'string') selector = "";

      for (let cnt = 0; cnt < 1000; ++cnt) {
         let h = getDrawHandle(kind, cnt);
         if (!h) break;
         if (!res.handle) res.handle = h;
         if (h.noinspect) noinspect = true;
         if (h.expand || h.expand_item || h.can_expand) canexpand = true;
         if (!h.func && !h.class) break;
         isany = true;
         if (!('opt' in h)) continue;
         let opts = h.opt.split(';');
         for (let i = 0; i < opts.length; ++i) {
            opts[i] = opts[i].toLowerCase();
            if ((selector.indexOf('nosame') >= 0) && (opts[i].indexOf('same') == 0)) continue;

            if (res.opts === null) res.opts = [];
            if (res.opts.indexOf(opts[i]) < 0) res.opts.push(opts[i]);
         }
         if (h.theonly) break;
      }

      if (selector.indexOf('noinspect') >= 0) noinspect = true;

      if (isany && (res.opts === null)) res.opts = [""];

      // if no any handle found, let inspect ROOT-based objects
      if (!isany && (kind.indexOf("ROOT.") == 0) && !noinspect) res.opts = [];

      if (!noinspect && res.opts)
         res.opts.push("inspect");

      res.inspect = !noinspect;
      res.expand = canexpand;
      res.draw = res.opts && (res.opts.length > 0);

      return res;
   }

   /** @summary Returns true if provided object class can be drawn
     * @param {string} classname - name of class to be tested
     * @private */
   jsrp.canDraw = function(classname) {
      return getDrawSettings("ROOT." + classname).opts !== null;
   }

   /** @summary Set default draw option for provided class */
   jsrp.setDefaultDrawOpt = function(classname, opt) {
      let handle = getDrawHandle("ROOT." + classname, 0);
      if (handle)
         handle.dflt = opt;
   }

   /** @summary Draw object in specified HTML element with given draw options.
     * @param {string|object} dom - id of div element to draw or directly DOMElement
     * @param {object} obj - object to draw, object type should be registered before in JSROOT
     * @param {string} opt - draw options separated by space, comma or semicolon
     * @returns {Promise} with painter object
     * @requires painter
     * @desc An extensive list of support draw options can be found on [JSROOT examples page]{@link https://root.cern/js/latest/examples.htm}
     * @example
     * JSROOT.openFile("https://root.cern/js/files/hsimple.root")
     *       .then(file => file.readObject("hpxpy;1"))
     *       .then(obj => JSROOT.draw("drawing", obj, "colz;logx;gridx;gridy")); */
   JSROOT.draw = function(dom, obj, opt) {
      if (!obj || (typeof obj !== 'object'))
         return Promise.reject(Error('not an object in JSROOT.draw'));

      if (opt == 'inspect')
         return JSROOT.require("hierarchy").then(() => jsrp.drawInspector(dom, obj));

      let handle, type_info;
      if ('_typename' in obj) {
         type_info = "type " + obj._typename;
         handle = getDrawHandle("ROOT." + obj._typename, opt);
      } else if ('_kind' in obj) {
         type_info = "kind " + obj._kind;
         handle = getDrawHandle(obj._kind, opt);
      } else
         return JSROOT.require("hierarchy").then(() => jsrp.drawInspector(dom, obj));

      // this is case of unsupported class, close it normally
      if (!handle)
         return Promise.reject(Error(`Object of ${type_info} cannot be shown with JSROOT.draw`));

      if (handle.dummy)
         return Promise.resolve(null);

      if (handle.draw_field && obj[handle.draw_field])
         return JSROOT.draw(dom, obj[handle.draw_field], opt || handle.draw_field_opt);

      if (!handle.func && !handle.direct && !handle.class) {
         if (opt && (opt.indexOf("same") >= 0)) {

            let main_painter = jsrp.getElementMainPainter(dom);

            if (main_painter && (typeof main_painter.performDrop === 'function'))
               return main_painter.performDrop(obj, "", null, opt);
         }

         return Promise.reject(Error(`Function not specified to draw object ${type_info}`));
      }

      function performDraw() {
         let promise;
         if (handle.direct == "v7") {
            let painter = new JSROOT.RObjectPainter(dom, obj, opt, handle.csstype);
            promise = jsrp.ensureRCanvas(painter, handle.frame || false).then(() => {
               painter.redraw = handle.func;
               let res = painter.redraw();
               if (!isPromise(res))
                  return painter;
               return res.then(() => painter);
            })
         } else if (handle.direct) {
            let painter = new ObjectPainter(dom, obj, opt);
            promise = jsrp.ensureTCanvas(painter, handle.frame || false).then(() => {
               painter.redraw = handle.func;
               let res = painter.redraw();
               if (!isPromise(res))
                  return painter;
               return res.then(() => painter);
            });
         } else {
            promise = handle.func(dom, obj, opt);
            if (!isPromise(promise)) promise = Promise.resolve(promise);
         }

         return promise.then(p => {
            if (!p)
               return Promise.reject(Error(`Fail to draw object ${type_info}`));

            if ((typeof p == 'object') && !p.options)
               p.options = { original: opt || "" }; // keep original draw options

             return p;
         });
      }

      if (typeof handle.func == 'function')
         return performDraw();

      let funcname, clname;
      if (typeof handle.func == 'string')
         funcname = handle.func;
      else if (typeof handle.class == 'string')
         clname = handle.class;
      else
         return Promise.reject(Error(`Draw function or class not specified to draw ${type_info}`));

      let prereq = handle.prereq || "";
      if (handle.direct == "v7")
         prereq += ";v7gpad";
      else if (handle.direct)
         prereq += ";gpad";
      if (handle.script)
         prereq += ";" + handle.script;

      if (!prereq)
         return Promise.reject(Error(`Prerequicities to load ${funcname} are not specified`));

      return JSROOT.require(prereq).then(() => {
         if (funcname) {
            let func = JSROOT.findFunction(funcname);
            if (!func)
               return Promise.reject(Error(`Fail to find function ${funcname} after loading ${prereq}`));
            handle.func = func;
         } else {
            let cl = JSROOT[clname];
            if (!cl || typeof cl.draw != 'function')
               return Promise.reject(Error(`Fail to find class JSROOT.${clname} after loading ${prereq}`));
            handle.class = cl;
            handle.func = cl.draw;
         }

         return performDraw();
      });
   }

   /** @summary Redraw object in specified HTML element with given draw options.
     * @param {string|object} dom - id of div element to draw or directly DOMElement
     * @param {object} obj - object to draw, object type should be registered before in JSROOT
     * @param {string} opt - draw options
     * @returns {Promise} with painter object
     * @requires painter
     * @desc If drawing was not done before, it will be performed with {@link JSROOT.draw}.
     * Otherwise drawing content will be updated */
   JSROOT.redraw = function(dom, obj, opt) {

      if (!obj || (typeof obj !== 'object'))
         return Promise.reject(Error('not an object in JSROOT.redraw'));

      let can_painter = jsrp.getElementCanvPainter(dom), handle, res_painter = null, redraw_res;
      if (obj._typename)
         handle = getDrawHandle("ROOT." + obj._typename);
      if (handle && handle.draw_field && obj[handle.draw_field])
         obj = obj[handle.draw_field];

      if (can_painter) {
         if (can_painter.matchObjectType(obj._typename)) {
            redraw_res = can_painter.redrawObject(obj, opt);
            if (redraw_res) res_painter = can_painter;
         } else {
            for (let i = 0; i < can_painter.painters.length; ++i) {
               let painter = can_painter.painters[i];
               if (painter.matchObjectType(obj._typename)) {
                  redraw_res = painter.redrawObject(obj, opt);
                  if (redraw_res) {
                     res_painter = painter;
                     break;
                  }
               }
            }
         }
      } else {
         let top = new BasePainter(dom).getTopPainter();
         // base painter do not have this method, if it there use it
         // it can be object painter here or can be specially introduce method to handling redraw!
         if (top && typeof top.redrawObject == 'function') {
            redraw_res = top.redrawObject(obj, opt);
            if (redraw_res) res_painter = top;
         }
      }

      if (res_painter) {
         if (!redraw_res || (typeof redraw_res != 'object') || !redraw_res.then)
            redraw_res = Promise.resolve(true);
         return redraw_res.then(() => res_painter);
      }

      JSROOT.cleanup(dom);

      return JSROOT.draw(dom, obj, opt);
   }

   /** @summary Save object, drawn in specified element, as JSON.
     * @desc Normally it is TCanvas object with list of primitives
     * @param {string|object} dom - id of top div element or directly DOMElement
     * @returns {string} produced JSON string */
   JSROOT.drawingJSON = function(dom) {
      let canp = jsrp.getElementCanvPainter(dom);
      return canp ? canp.produceJSON() : "";
   }

   /** @summary Compress SVG code, produced from JSROOT drawing
     * @desc removes extra info or empty elements
     * @private */
   jsrp.compressSVG = function(svg) {

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

   /** @summary Create SVG image for provided object.
     * @desc Function especially useful in Node.js environment to generate images for
     * supported ROOT classes
     * @param {object} args - contains different settings
     * @param {object} args.object - object for the drawing
     * @param {string} [args.option] - draw options
     * @param {number} [args.width = 1200] - image width
     * @param {number} [args.height = 800] - image height
     * @returns {Promise} with svg code */
   JSROOT.makeSVG = function(args) {

      if (!args) args = {};
      if (!args.object) return Promise.reject(Error("No object specified to generate SVG"));
      if (!args.width) args.width = 1200;
      if (!args.height) args.height = 800;

      function build(main) {

         main.attr("width", args.width).attr("height", args.height)
             .style("width", args.width + "px").style("height", args.height + "px");

         JSROOT._.svg_3ds = undefined;

         return JSROOT.draw(main.node(), args.object, args.option || "").then(() => {

            let has_workarounds = JSROOT._.svg_3ds && jsrp.processSvgWorkarounds;

            main.select('svg')
                .attr("xmlns", "http://www.w3.org/2000/svg")
                .attr("xmlns:xlink", "http://www.w3.org/1999/xlink")
                .attr("width", args.width)
                .attr("height", args.height)
                .attr("style", null).attr("class", null).attr("x", null).attr("y", null);

            function clear_element() {
               const elem = d3.select(this);
               if (elem.style('display')=="none") elem.remove();
            };

            // remove containers with display: none
            if (has_workarounds)
               main.selectAll('g.root_frame').each(clear_element);

            main.selectAll('svg').each(clear_element);

            let svg = main.html();

            if (has_workarounds)
               svg = jsrp.processSvgWorkarounds(svg);

            svg = jsrp.compressSVG(svg);

            JSROOT.cleanup(main.node());

            main.remove();

            return svg;
         });
      }

      if (!JSROOT.nodejs)
         return build(d3.select('body').append("div").style("visible", "hidden"));

      if (!JSROOT._.nodejs_document) {
         // use eval while old minifier is not able to parse newest Node.js syntax
         const { JSDOM } = require("jsdom");
         JSROOT._.nodejs_window = (new JSDOM("<!DOCTYPE html>hello")).window;
         JSROOT._.nodejs_document = JSROOT._.nodejs_window.document; // used with three.js
         JSROOT._.nodejs_window.d3 = d3.select(JSROOT._.nodejs_document); //get d3 into the dom
      }

      return build(JSROOT._.nodejs_window.d3.select('body').append('div'));
   }

   /** @summary Check resize of drawn element
     * @param {string|object} dom - id or DOM element
     * @param {boolean|object} arg - options on how to resize
     * @desc As first argument dom one should use same argument as for the drawing
     * As second argument, one could specify "true" value to force redrawing of
     * the element even after minimal resize
     * Or one just supply object with exact sizes like { width:300, height:200, force:true };
     * @example
     * JSROOT.resize("drawing", { width: 500, height: 200 } );
     * JSROOT.resize(document.querySelector("#drawing"), true); */
   JSROOT.resize = function(dom, arg) {
      if (arg === true)
         arg = { force: true };
      else if (typeof arg !== 'object')
         arg = null;
      let done = false;
      new ObjectPainter(dom).forEachPainter(painter => {
         if (!done && (typeof painter.checkResize == 'function'))
            done = painter.checkResize(arg);
      });
      return done;
   }

   /** @summary Returns canvas painter (if any) for specified HTML element
     * @param {string|object} dom - id or DOM element
     * @private */
   jsrp.getElementCanvPainter = function(dom) {
      return new ObjectPainter(dom).getCanvPainter();
   }

   /** @summary Returns main painter (if any) for specified HTML element - typically histogram painter
     * @param {string|object} dom - id or DOM element
     * @private */
   jsrp.getElementMainPainter = function(dom) {
      return new ObjectPainter(dom).getMainPainter(true);
   }

   /** @summary Safely remove all JSROOT drawings from specified element
     * @param {string|object} dom - id or DOM element
     * @requires painter
     * @example
     * JSROOT.cleanup("drawing");
     * JSROOT.cleanup(document.querySelector("#drawing")); */
   JSROOT.cleanup = function(dom) {
      let dummy = new ObjectPainter(dom), lst = [];
      dummy.forEachPainter(p => { if (lst.indexOf(p) < 0) lst.push(p); });
      lst.forEach(p => p.cleanup());
      dummy.selectDom().html("");
      return lst;
   }

   /** @summary Display progress message in the left bottom corner.
     * @desc Previous message will be overwritten
     * if no argument specified, any shown messages will be removed
     * @param {string} msg - message to display
     * @param {number} tmout - optional timeout in milliseconds, after message will disappear
     * @private */
   jsrp.showProgress = function(msg, tmout) {
      if (JSROOT.batch_mode || (typeof document === 'undefined')) return;
      let id = "jsroot_progressbox",
          box = d3.select("#" + id);

      if (!JSROOT.settings.ProgressBox) return box.remove();

      if ((arguments.length == 0) || !msg) {
         if ((tmout !== -1) || (!box.empty() && box.property("with_timeout"))) box.remove();
         return;
      }

      if (box.empty()) {
         box = d3.select(document.body).append("div").attr("id", id);
         box.append("p");
      }

      box.property("with_timeout", false);

      if (typeof msg === "string") {
         box.select("p").html(msg);
      } else {
         box.html("");
         box.node().appendChild(msg);
      }

      if (Number.isFinite(tmout) && (tmout > 0)) {
         box.property("with_timeout", true);
         setTimeout(() => jsrp.showProgress('', -1), tmout);
      }
   }

   /** @summary Converts numeric value to string according to specified format.
     * @param {number} value - value to convert
     * @param {string} [fmt="6.4g"] - format can be like 5.4g or 4.2e or 6.4f
     * @param {boolean} [ret_fmt] - when true returns array with value and actual format like ["0.1","6.4f"]
     * @returns {string|Array} - converted value or array with value and actual format */
   jsrp.floatToString = function(value, fmt, ret_fmt) {
      if (!fmt) fmt = "6.4g";

      fmt = fmt.trim();
      let len = fmt.length;
      if (len<2)
         return ret_fmt ? [value.toFixed(4), "6.4f"] : value.toFixed(4);
      let last = fmt[len-1];
      fmt = fmt.slice(0,len-1);
      let isexp, prec = fmt.indexOf(".");
      prec = (prec<0) ? 4 : parseInt(fmt.slice(prec+1));
      if (!Number.isInteger(prec) || (prec <=0)) prec = 4;

      let significance = false;
      if ((last=='e') || (last=='E')) { isexp = true; } else
      if (last=='Q') { isexp = true; significance = true; } else
      if ((last=='f') || (last=='F')) { isexp = false; } else
      if (last=='W') { isexp = false; significance = true; } else
      if ((last=='g') || (last=='G')) {
         let se = jsrp.floatToString(value, fmt+'Q', true),
             sg = jsrp.floatToString(value, fmt+'W', true);

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

   /** @summary Tries to close current browser tab
     * @desc Many browsers do not allow simple window.close() call,
     * therefore try several workarounds
     * @private */
   jsrp.closeCurrentWindow = function() {
      if (!window) return;
      window.close();
      window.open('', '_self').close();
   }

   jsrp.createRootColors();

   if (JSROOT.nodejs) jsrp.readStyleFromURL("?interactive=0&tooltip=0&nomenu&noprogress&notouch&toolbar=0&webgl=0");

   jsrp.getDrawHandle = getDrawHandle;
   jsrp.getDrawSettings = getDrawSettings;
   jsrp.getElementRect = getElementRect;
   jsrp.isPromise = isPromise;
   jsrp.toHex = toHex;

   JSROOT.TRandom = TRandom;
   JSROOT.DrawOptions = DrawOptions;
   JSROOT.ColorPalette = ColorPalette;
   JSROOT.TAttLineHandler = TAttLineHandler;
   JSROOT.TAttFillHandler = TAttFillHandler;
   JSROOT.TAttMarkerHandler = TAttMarkerHandler;
   JSROOT.FontHandler = FontHandler;
   JSROOT.BasePainter = BasePainter;
   JSROOT.ObjectPainter = ObjectPainter;

   JSROOT.Painter = jsrp;
   if (JSROOT.nodejs) module.exports = jsrp;

   return jsrp;

});
