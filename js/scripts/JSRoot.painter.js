/// @file JSRoot.painter.js
/// Baisc JavaScript ROOT painter classes

JSROOT.define(['d3'], (d3) => {

   "use strict";

   JSROOT.loadScript('$$$style/JSRoot.painter');

   if ((typeof d3 !== 'object') || !d3.version)
      console.error('Fail to detect d3.js');
   else if (d3.version[0] !== "6")
      console.error(`Unsupported d3.js version ${d3.version}, expected 6.1.1`);
   else if (d3.version !== '6.1.1')
      console.log(`Reuse existing d3.js version ${d3.version}, expected 6.1.1`);

   // ==========================================================================================

   /** @summary Draw options interpreter
     * @memberof JSROOT
     * @private */
   function DrawOptions(opt) {
      this.opt = opt && (typeof opt == "string") ? opt.toUpperCase().trim() : "";
      this.part = "";
   }

   /** @summary Returns true if remaining options are empty. */
   DrawOptions.prototype.empty = function() { return this.opt.length === 0; }

   /** @summary Returns remaining part of the draw options. */
   DrawOptions.prototype.remain = function() { return this.opt; }

   /** @summary Checks if given option exists */
   DrawOptions.prototype.check = function(name, postpart) {
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
   DrawOptions.prototype.partAsInt = function(offset, dflt) {
      let val = this.part.replace(/^\D+/g, '');
      val = val ? parseInt(val, 10) : Number.NaN;
      return isNaN(val) ? (dflt || 0) : val + (offset || 0);
   }

   /** @summary Returns remaining part of found option as float. */
   DrawOptions.prototype.partAsFloat = function(offset, dflt) {
      let val = this.part.replace(/^\D+/g, '');
      val = val ? parseFloat(val) : Number.NaN;
      return isNaN(val) ? (dflt || 0) : val + (offset || 0);
   }

   // ============================================================================================

   /** @namespace
     * @summary Collection of Painter-related methods and classes
     * @alias JSROOT.Painter
     * @private */
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
      root_markers: [0, 100, 8, 7, 0,  //  0..4
         9, 100, 100, 100, 100,  //  5..9
         100, 100, 100, 100, 100,  // 10..14
         100, 100, 100, 100, 100,  // 15..19
         100, 103, 105, 104, 0,  // 20..24
         3, 4, 2, 1, 106,  // 25..29
         6, 7, 5, 102, 101], // 30..34
      root_fonts: ['Arial', 'iTimes New Roman',
         'bTimes New Roman', 'biTimes New Roman', 'Arial',
         'oArial', 'bArial', 'boArial', 'Courier New',
         'oCourier New', 'bCourier New', 'boCourier New',
         'Symbol', 'Times New Roman', 'Wingdings', 'iSymbol', 'Verdana'],
      // taken from https://www.math.utah.edu/~beebe/fonts/afm-widths.html
      root_fonts_aver_width: [0.537, 0.510,
         0.535, 0.520, 0.537,
         0.54, 0.556, 0.56, 0.6,
         0.6, 0.6, 0.6,
         0.587, 0.514, 0.896, 0.587, 0.55]
   };

   // create menu, implemented in jquery part
   jsrp.createMenu = function(painter, evt) {
      document.body.style.cursor = 'wait';
      let show_evnt;
      // copy event values, otherwise they will gone after scripts loading
      if (evt && (typeof evt == "object"))
         if ((evt.clientX !== undefined) && (evt.clientY !== undefined))
            show_evnt = { clientX: evt.clientX, clientY: evt.clientY };
      return JSROOT.require(['jq2d']).then(() => {
         document.body.style.cursor = 'auto';
         return jsrp.createMenu(painter, show_evnt);
      });
   }

   // create menu, implemented in jquery part
   jsrp.closeMenu = function(menuname) {
      let x = document.getElementById(menuname || 'root_ctx_menu');
      if (x) { x.parentNode.removeChild(x); return true; }
      return false;
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
            if (!isNaN(optimize)) s.OptimizeDraw = optimize;
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

      let mathjax = d.get("mathjax", null), latex = d.get("latex", null);

      if ((mathjax !== null) && (mathjax != "0") && (latex === null)) latex = "math";
      if (latex !== null)
         s.Latex = JSROOT.constants.Latex.fromString(latex);

      if (d.has("nomenu")) s.ContextMenu = false;
      if (d.has("noprogress")) s.ProgressBox = false;
      if (d.has("notouch")) JSROOT.browser.touches = false;
      if (d.has("adjframe")) s.CanAdjustFrame = true;

      let optstat = d.get("optstat");
      if (optstat) g.fOptStat = parseInt(optstat);
      let optfit = d.get("optfit");
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

      if (d.has("palette")) {
         let palette = parseInt(d.get("palette"));
         if (!isNaN(palette) && (palette > 0) && (palette < 113)) s.Palette = palette;
      }

      let render3d = d.get("render3d");
      if (render3d)
         JSROOT.settings.Render3D = JSROOT.constants.Render3D.fromString(render3d);

      let embed3d = d.get("embed3d");
      if (embed3d)
         JSROOT.settings.Embed3D = JSROOT.constants.Embed3D.fromString(embed3d);

      let geosegm = d.get("geosegm");
      if (geosegm) s.GeoGradPerSegm = Math.max(2, parseInt(geosegm));
      let geocomp = d.get("geocomp");
      if (geocomp) s.GeoCompressComp = (geocomp !== '0') && (geocomp !== 'false') && (geocomp !== 'off');

      if (d.has("hlimit")) s.HierarchyLimit = parseInt(d.get("hlimit"));
   }

   /** @summary Generates all root colors, used also in jstests to reset colors
     * @private */
   jsrp.createRootColors = function() {
      let colorMap = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'rgb(89,212,84)', 'rgb(89,84,217)', 'white'];
      colorMap[110] = 'white';

      let moreCol = [
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
            colorMap[num] = 'rgb(' + parseInt("0x" + s.substr(n, 2)) + "," + parseInt("0x" + s.substr(n + 2, 2)) + "," + parseInt("0x" + s.substr(n + 4, 2)) + ")";
         }
      });

      jsrp.root_colors = colorMap;
   }

   /** @summary Produces rgb code for TColor object
     * @private */
   jsrp.getRGBfromTColor = function(col) {
      if (!col || (col._typename != 'TColor')) return null;
      let rgb = Math.round(col.fRed * 255) + "," + Math.round(col.fGreen * 255) + "," + Math.round(col.fBlue * 255);
      if ((col.fAlpha === undefined) || (col.fAlpha == 1.))
         rgb = "rgb(" + rgb + ")";
      else
         rgb = "rgba(" + rgb + "," + col.fAlpha.toFixed(3) + ")";

      switch (rgb) {
         case 'rgb(255,255,255)': return 'white';
         case 'rgb(0,0,0)': return 'black';
         case 'rgb(255,0,0)': return 'red';
         case 'rgb(0,255,0)': return 'green';
         case 'rgb(0,0,255)': return 'blue';
         case 'rgb(255,255,0)': return 'yellow';
         case 'rgb(255,0,255)': return 'magenta';
         case 'rgb(0,255,255)': return 'cyan';
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
     * @returns {number} index of new color */
   jsrp.addColor = function(rgb) {
      let indx = jsrp.root_colors.indexOf(rgb);
      if (indx >= 0) return indx;
      jsrp.root_colors.push(rgb);
      return jsrp.root_colors.length-1;
   }

   // =====================================================================

   /** Color palette handle  */

   function ColorPalette(arr) {
      this.palette = arr;
   }

      /** @summary Returns color index which correspond to contour index of provided length */
   ColorPalette.prototype.calcColorIndex = function(i, len) {
      let plen = this.palette.length, theColor = Math.floor((i + 0.99) * plen / (len - 1));
      return (theColor > plen - 1) ? plen - 1 : theColor;
    }

      /** @summary Returns color with provided index */
   ColorPalette.prototype.getColor = function(indx) { return this.palette[indx]; }

      /** @summary Returns number of colors in the palette */
   ColorPalette.prototype.getLength = function() { return this.palette.length; }

      /** @summary Calculate color for given i and len */
   ColorPalette.prototype.calcColor = function(i, len) { return this.getColor(this.calcColorIndex(i, len)); }

   // =============================================================================

   /**
     * @summary Handle for marker attributes
     *
     * @class
     * @memberof JSROOT
     * @param {object} args - different attributes, see {@link JSROOT.TAttMarkerHandler.SetArgs} for details
     * @private
     */

   function TAttMarkerHandler(args) {
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

      this.func = this.Apply.bind(this);

      this.SetArgs(args);

      this.changed = false;
   }

   /** @summary Set marker attributes.
    *
    * @param {object} args - arguments can be
    * @param {object} args.attr - instance of TAttrMarker (or derived class) or
    * @param {string} args.color - color in HTML form like grb(1,4,5) or 'green'
    * @param {number} args.style - marker style
    * @param {number} args.size - marker size
    */
   TAttMarkerHandler.prototype.SetArgs = function(args) {
      if ((typeof args == 'object') && (typeof args.fMarkerStyle == 'number')) args = { attr: args };

      if (args.attr) {
         if (args.color === undefined)
            args.color = args.painter ? args.painter.get_color(args.attr.fMarkerColor) : jsrp.getColor(args.attr.fMarkerColor);
         if (!args.style || (args.style < 0)) args.style = args.attr.fMarkerStyle;
         if (!args.size) args.size = args.attr.fMarkerSize;
      }

      this.Change(args.color, args.style, args.size);
   }

   /** @summary Reset position, used for optimization of drawing of multiple markers
    * @private */
   TAttMarkerHandler.prototype.reset_pos = function() { this.lastx = this.lasty = null; }

   /** @summary Create marker path for given position.
    *
    * @desc When drawing many elementary points, created path may depend from previously produced markers.
    *
    * @param {number} x - first coordinate
    * @param {number} y - second coordinate
    * @returns {string} path string
    */
   TAttMarkerHandler.prototype.create = function(x, y) {
      if (!this.optimized)
         return "M" + (x + this.x0).toFixed(this.ndig) + "," + (y + this.y0).toFixed(this.ndig) + this.marker;

      // use optimized handling with relative position
      let xx = Math.round(x), yy = Math.round(y), m1 = "M" + xx + "," + yy + "h1",
         m2 = (this.lastx === null) ? m1 : ("m" + (xx - this.lastx) + "," + (yy - this.lasty) + "h1");
      this.lastx = xx + 1; this.lasty = yy;
      return (m2.length < m1.length) ? m2 : m1;
   }

   /** @summary Returns full size of marker */
   TAttMarkerHandler.prototype.GetFullSize = function() { return this.scale * this.size; }

   /** @summary Returns approximate length of produced marker string */
   TAttMarkerHandler.prototype.MarkerLength = function() { return this.marker ? this.marker.length : 10; }

   /** @summary Change marker attributes.
    *
    *  @param {string} color - marker color
    *  @param {number} style - marker style
    *  @param {number} size - marker size
    */
   TAttMarkerHandler.prototype.Change = function(color, style, size) {
      this.changed = true;

      if (color !== undefined) this.color = color;
      if ((style !== undefined) && (style >= 0)) this.style = style;
      if (size !== undefined) this.size = size; else size = this.size;

      this.x0 = this.y0 = 0;

      if ((this.style === 1) || (this.style === 777)) {
         this.fill = false;
         this.marker = "h1";
         this.size = 1;
         this.optimized = true;
         this.reset_pos();
         return true;
      }

      this.optimized = false;

      let marker_kind = jsrp.root_markers[this.style];
      if (marker_kind === undefined) marker_kind = 100;
      let shape = marker_kind % 100;

      this.fill = (marker_kind >= 100);

      switch (this.style) {
         case 1: this.size = 1; this.scale = 1; break;
         case 6: this.size = 2; this.scale = 1; break;
         case 7: this.size = 3; this.scale = 1; break;
         default: this.size = size; this.scale = 8;
      }

      size = this.GetFullSize();

      this.ndig = (size > 7) ? 0 : ((size > 2) ? 1 : 2);
      if (shape == 6) this.ndig++;
      let half = (size / 2).toFixed(this.ndig), full = size.toFixed(this.ndig);

      switch (shape) {
         case 0: // circle
            this.x0 = -parseFloat(half);
            full = (parseFloat(half) * 2).toFixed(this.ndig);
            this.marker = "a" + half + "," + half + ",0,1,0," + full + ",0a" + half + "," + half + ",0,1,0,-" + full + ",0z";
            break;
         case 1: // cross
            let d = (size / 3).toFixed(this.ndig);
            this.x0 = this.y0 = size / 6;
            this.marker = "h" + d + "v-" + d + "h-" + d + "v-" + d + "h-" + d + "v" + d + "h-" + d + "v" + d + "h" + d + "v" + d + "h" + d + "z";
            break;
         case 2: // diamond
            this.x0 = -size / 2;
            this.marker = "l" + half + ",-" + half + "l" + half + "," + half + "l-" + half + "," + half + "z";
            break;
         case 3: // square
            this.x0 = this.y0 = -size / 2;
            this.marker = "v" + full + "h" + full + "v-" + full + "z";
            break;
         case 4: // triangle-up
            this.y0 = size / 2;
            this.marker = "l-" + half + ",-" + full + "h" + full + "z";
            break;
         case 5: // triangle-down
            this.y0 = -size / 2;
            this.marker = "l-" + half + "," + full + "h" + full + "z";
            break;
         case 6: // star
            this.y0 = -size / 2;
            this.marker = "l" + (size / 3).toFixed(this.ndig) + "," + full +
               "l-" + (5 / 6 * size).toFixed(this.ndig) + ",-" + (5 / 8 * size).toFixed(this.ndig) +
               "h" + full +
               "l-" + (5 / 6 * size).toFixed(this.ndig) + "," + (5 / 8 * size).toFixed(this.ndig) + "z";
            break;
         case 7: // asterisk
            this.x0 = this.y0 = -size / 2;
            this.marker = "l" + full + "," + full +
               "m0,-" + full + "l-" + full + "," + full +
               "m0,-" + half + "h" + full + "m-" + half + ",-" + half + "v" + full;
            break;
         case 8: // plus
            this.y0 = -size / 2;
            this.marker = "v" + full + "m-" + half + ",-" + half + "h" + full;
            break;
         case 9: // mult
            this.x0 = this.y0 = -size / 2;
            this.marker = "l" + full + "," + full + "m0,-" + full + "l-" + full + "," + full;
            break;
         default: // diamand
            this.x0 = -size / 2;
            this.marker = "l" + half + ",-" + half + "l" + half + "," + half + "l-" + half + "," + half + "z";
            break;
      }

      return true;
   }

   TAttMarkerHandler.prototype.getStrokeColor = function() { return this.stroke ? this.color : "none"; }

   TAttMarkerHandler.prototype.getFillColor = function() { return this.fill ? this.color : "none"; }

   /** @summary Apply marker styles to created element */
   TAttMarkerHandler.prototype.Apply = function(selection) {
      selection.style('stroke', this.stroke ? this.color : "none");
      selection.style('fill', this.fill ? this.color : "none");
   }

   /** @summary Method used when color or pattern were changed with OpenUi5 widgets.
    * @private */
   TAttMarkerHandler.prototype.verifyDirectChange = function(/* painter */) {
      this.Change(this.color, parseInt(this.style), parseFloat(this.size));
   }

   /** @summary Create sample with marker in given SVG element
    *
    * @param {selection} svg - SVG element
    * @param {number} width - width of sample SVG
    * @param {number} height - height of sample SVG
    * @private
    */
   TAttMarkerHandler.prototype.CreateSample = function(svg, width, height) {
      this.reset_pos();

      svg.append("path")
         .attr("d", this.create(width / 2, height / 2))
         .call(this.func);
   }

   // =======================================================================

   /**
     * @summary Handle for line attributes
     *
     * @class
     * @memberof JSROOT
     * @param {object} attr - TAttLine object
     * @private
     */

   function TAttLineHandler(args) {
      this.func = this.Apply.bind(this);
      this.used = true;
      if (args._typename && (args.fLineStyle !== undefined)) args = { attr: args };

      this.SetArgs(args);
   }

   /**
    * @summary Set line attributes.
    *
    * @param {object} args - specify attributes by different ways
    * @param {object} args.attr - TAttLine object with appropriate data members or
    * @param {string} args.color - color in html like rgb(10,0,0) or "red"
    * @param {number} args.style - line style number
    * @param {number} args.width - line width
    */
   TAttLineHandler.prototype.SetArgs = function(args) {
      if (args.attr) {
         args.color = args.color0 || (args.painter ? args.painter.get_color(args.attr.fLineColor) : jsrp.getColor(args.attr.fLineColor));
         if (args.width === undefined) args.width = args.attr.fLineWidth;
         args.style = args.attr.fLineStyle;
      } else if (typeof args.color == 'string') {
         if ((args.color !== 'none') && !args.width) args.width = 1;
      } else if (typeof args.color == 'number') {
         args.color = args.painter ? args.painter.get_color(args.color) : jsrp.getColor(args.color);
      }

      if (args.width === undefined)
         args.width = (args.color && args.color != 'none') ? 1 : 0;

      this.color = (args.width === 0) ? 'none' : args.color;
      this.width = args.width;
      this.style = args.style;

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

   /**
    * @summary Change exclusion attributes.
    * @private
    */
   TAttLineHandler.prototype.ChangeExcl = function(side, width) {
      if (width !== undefined) this.excl_width = width;
      if (side !== undefined) {
         this.excl_side = side;
         if ((this.excl_width === 0) && (this.excl_side !== 0)) this.excl_width = 20;
      }
      this.changed = true;
   }

   /** @returns true if line attribute is empty and will not be applied. */
   TAttLineHandler.prototype.empty = function() { return this.color == 'none'; }

   /**
    * @summary Applies line attribute to selection.
    *
    * @param {object} selection - d3.js selection
    */

   TAttLineHandler.prototype.Apply = function(selection) {
      this.used = true;
      if (this.empty())
         selection.style('stroke', null)
            .style('stroke-width', null)
            .style('stroke-dasharray', null);
      else
         selection.style('stroke', this.color)
            .style('stroke-width', this.width)
            .style('stroke-dasharray', jsrp.root_line_styles[this.style] || null);
   }

   /**
    * @summary Change line attributes
    * @private
    */
   TAttLineHandler.prototype.Change = function(color, width, style) {
      if (color !== undefined) this.color = color;
      if (width !== undefined) this.width = width;
      if (style !== undefined) this.style = style;
      this.changed = true;
   }

   /**
    * @summary Create sample element inside primitive SVG - used in context menu
    * @private
    */
   TAttLineHandler.prototype.CreateSample = function(svg, width, height) {
      svg.append("path")
         .attr("d", "M0," + height / 2 + "h" + width)
         .call(this.func);
   }

   // =======================================================================


   /**
     * @summary Handle for fill attributes
     *
     * @class
     * @memberof JSROOT
     * @param {object} args - different arguments to set fill attributes, see {@link JSROOT.TAttFillHandler.SetArgs} for more info
     * @param {number} [args.kind = 2] - 1 means object drawing where combination fillcolor==0 and fillstyle==1001 means no filling,  2 means all other objects where such combination is white-color filling
     * @private
     */

   function TAttFillHandler(args) {
      this.color = "none";
      this.colorindx = 0;
      this.pattern = 0;
      this.used = true;
      this.kind = args.kind || 2;
      this.changed = false;
      this.func = this.Apply.bind(this);
      this.SetArgs(args);
      this.changed = false; // unset change property that
   }

   /** @summary Set fill style as arguments
     * @param {object} args - different arguments to set fill attributes
     * @param {object} [args.attr] - TAttFill object
     * @param {number} [args.color] - color id
     * @param {number} [args.pattern] - filll pattern id
     * @param {object} [args.svg] - SVG element to store newly created patterns
     * @param {string} [args.color_as_svg] - color in SVG format */
   TAttFillHandler.prototype.SetArgs = function(args) {
      if (args.attr && (typeof args.attr == 'object')) {
         if ((args.pattern === undefined) && (args.attr.fFillStyle !== undefined)) args.pattern = args.attr.fFillStyle;
         if ((args.color === undefined) && (args.attr.fFillColor !== undefined)) args.color = args.attr.fFillColor;
      }
      this.Change(args.color, args.pattern, args.svg, args.color_as_svg, args.painter);
   }

   /** @summary Apply fill style to selection */
   TAttFillHandler.prototype.Apply = function(selection) {
      this.used = true;

      selection.style('fill', this.fillcolor());

      if ('opacity' in this)
         selection.style('opacity', this.opacity);

      if ('antialias' in this)
         selection.style('antialias', this.antialias);
   }

   /** @summary Returns fill color (or pattern url) */
   TAttFillHandler.prototype.fillcolor = function() { return this.pattern_url || this.color; }

   /** @summary Returns fill color without pattern url.
    *
    * @desc If empty, alternative color will be provided
    * @param {string} [altern] - alternative color which returned when fill color not exists
    * @private */
   TAttFillHandler.prototype.fillcoloralt = function(altern) { return this.color && (this.color != "none") ? this.color : altern; }

   /** @summary Returns true if color not specified or fill style not specified */
   TAttFillHandler.prototype.empty = function() {
      let fill = this.fillcolor();
      return !fill || (fill == 'none');
   }

   /** @summary Set solid fill color as fill pattern
    * @param {string} col - solid color */
   TAttFillHandler.prototype.SetSolidColor = function(col) {
      delete this.pattern_url;
      this.color = col;
      this.pattern = 1001;
   }

   /** @summary Check if solid fill is used, also color can be checked
    * @param {string} [solid_color = undefined] - when specified, checks if fill color matches */
   TAttFillHandler.prototype.isSolid = function(solid_color) {
      if (this.pattern !== 1001) return false;
      return !solid_color || solid_color == this.color;
   }

   /** @summary Method used when color or pattern were changed with OpenUi5 widgets
    * @private */
   TAttFillHandler.prototype.verifyDirectChange = function(painter) {
      if (typeof this.pattern == 'string') this.pattern = parseInt(this.pattern);
      if (isNaN(this.pattern)) this.pattern = 0;

      this.Change(this.color, this.pattern, painter ? painter.svg_canvas() : null, true, painter);
   }

   /** @summary Method to change fill attributes.
    *
    * @param {number} color - color index
    * @param {number} pattern - pattern index
    * @param {selection} svg - top canvas element for pattern storages
    * @param {string} [color_as_svg] - when color is string, interpret as normal SVG color
    * @param {object} [painter] - when specified, used to extract color by index
    */
   TAttFillHandler.prototype.Change = function(color, pattern, svg, color_as_svg, painter) {
      delete this.pattern_url;
      this.changed = true;

      if ((color !== undefined) && !isNaN(color) && !color_as_svg)
         this.colorindx = parseInt(color);

      if ((pattern !== undefined) && !isNaN(pattern)) {
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
         indx = 10000 + JSROOT._.id_counter++; // use fictional unique index far away from existing color indexes
      } else {
         this.color = painter ? painter.get_color(indx) : jsrp.getColor(indx);
      }

      if (typeof this.color != 'string') this.color = "none";

      if (this.isSolid()) return true;

      if ((this.pattern >= 4000) && (this.pattern <= 4100)) {
         // special transparent colors (use for subpads)
         this.opacity = (this.pattern - 4000) / 100;
         return true;
      }

      if (!svg || svg.empty() || (this.pattern < 3000)) return false;

      let id = "pat_" + this.pattern + "_" + indx,
         defs = svg.select('.canvas_defs');

      if (defs.empty())
         defs = svg.insert("svg:defs", ":first-child").attr("class", "canvas_defs");

      this.pattern_url = "url(#" + id + ")";
      this.antialias = false;

      if (!defs.select("." + id).empty()) {
         if (color_as_svg) console.log('find id in def', id);
         return true;
      }

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
               k = code % 10, j = ((code - k) % 100) / 10, i = (code - j * 10 - k) / 100;
            if (!i) break;

            let sz = i * 12;  // axis distance between lines

            w = h = 6 * sz; // we use at least 6 steps

            function produce(dy, swap) {
               let pos = [], step = sz, y1 = 0, y2, max = h;

               // reduce step for smaller angles to keep normal distance approx same
               if (Math.abs(dy) < 3) step = Math.round(sz / 12 * 9);
               if (dy == 0) { step = Math.round(sz / 12 * 8); y1 = step / 2; }
               else if (dy > 0) max -= step; else y1 = step;

               while (y1 <= max) {
                  y2 = y1 + dy * step;
                  if (y2 < 0) {
                     let x2 = Math.round(y1 / (y1 - y2) * w);
                     pos.push(0, y1, x2, 0);
                     pos.push(w, h - y1, w - x2, h);
                  } else if (y2 > h) {
                     let x2 = Math.round((h - y1) / (y2 - y1) * w);
                     pos.push(0, y1, x2, h);
                     pos.push(w, h - y1, w - x2, 0);
                  } else {
                     pos.push(0, y1, w, y2);
                  }
                  y1 += step;
               }
               for (let k = 0; k < pos.length; k += 4)
                  if (swap) lines += "M" + pos[k + 1] + "," + pos[k] + "L" + pos[k + 3] + "," + pos[k + 2];
                  else lines += "M" + pos[k] + "," + pos[k + 1] + "L" + pos[k + 2] + "," + pos[k + 3];
            }

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

      let patt = defs.append('svg:pattern').attr("id", id).attr("class", id).attr("patternUnits", "userSpaceOnUse")
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
   TAttFillHandler.prototype.CreateSample = function(sample_svg, width, height) {

      // we need to create extra handle to change
      let sample = new TAttFillHandler({ svg: sample_svg, pattern: this.pattern, color: this.color, color_as_svg: true });

      sample_svg.append("path")
         .attr("d", "M0,0h" + width + "v" + height + "h-" + width + "z")
         .call(sample.func);
   }

   // ===========================================================================

   /**
    * @summary Helper class for font handling
    *
    * @class
    * @memberof JSROOT
    * @private
    */

   function FontHandler(fontIndex, size) {
      this.name = "Arial";
      this.size = Math.round(size || 11);
      this.weight = null;
      this.style = null;

      let indx = Math.floor(fontIndex / 10),
          fontName = jsrp.root_fonts[indx] || "";

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

      if (fontName == 'Symbol')
         this.weight = this.style = null;

      this.name = fontName;
      this.aver_width = jsrp.root_fonts_aver_width[indx] || 0.55;

      this.func = this.setFont.bind(this);
   }

   /** @summary Assigns font-related attributes */
   FontHandler.prototype.setFont = function(selection, arg) {
      selection.attr("font-family", this.name);
      if (arg != 'without-size')
         selection.attr("font-size", this.size)
                  .attr("xml:space", "preserve");
      if (this.weight)
         selection.attr("font-weight", this.weight);
      if (this.style)
         selection.attr("font-style", this.style);
   }

   /** @summary Set text color (optional) */
   FontHandler.prototype.setColor = function(color) {
      this.color = color;
   }

   /** @summary Set text align (optional) */
   FontHandler.prototype.setAlign = function(align) {
      this.align = align;
   }

   /** @summary Set text angle (optional) */
   FontHandler.prototype.setAngle = function(angle) {
      this.angle = angle;
   }

   /** @summary Allign angle to step raster, add optional offset */
   FontHandler.prototype.roundAngle = function(step, offset) {
      this.angle = parseInt(this.angle || 0);
      if (isNaN(this.angle)) this.angle = 0;
      this.angle = Math.round(this.angle/step) * step + (offset || 0);
      if (this.angle < 0)
         this.angle += 360;
      else if (this.angle >= 360)
         this.angle -= 360;
   }

   /** @summary Clears all font-related attributes */
   FontHandler.prototype.clearFont = function(selection) {
      selection.attr("font-family", null)
               .attr("font-size", null)
               .attr("xml:space", null)
               .attr("font-weight", null)
               .attr("font-style", null);
   }

   /** @summary required for reasonable scaling of text in node.js
     * @returns approximate width of given label */
   FontHandler.prototype.approxTextWidth = function(label) { return label.length * this.size * this.aver_width; }

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

   /** @summary Returns time format
     * @param {TAxis} axis - TAxis object
     * @private */
   jsrp.getTimeFormat = function(axis) {
      let idF = axis.fTimeFormat.indexOf('%F');
      return (idF >= 0) ? axis.fTimeFormat.substr(0, idF) : axis.fTimeFormat;
   }

   /** @summary Return time offset value for given TAxis object
     * @private */
   jsrp.getTimeOffset = function(axis) {
      let dflt_time_offset = 788918400000;
      if (!axis) return dflt_time_offset;
      let idF = axis.fTimeFormat.indexOf('%F');
      if (idF < 0) return JSROOT.gStyle.fTimeOffset * 1000;
      let sof = axis.fTimeFormat.substr(idF + 2);
      // default string in axis offset
      if (sof.indexOf('1995-01-01 00:00:00s0') == 0) return dflt_time_offset;
      // special case, used from DABC painters
      if ((sof == "0") || (sof == "")) return 0;

      // decode time from ROOT string
      function next(separ, min, max) {
         let pos = sof.indexOf(separ);
         if (pos < 0) { pos = ""; return min; }
         let val = parseInt(sof.substr(0, pos));
         sof = sof.substr(pos + 1);
         if (isNaN(val) || (val < min) || (val > max)) { pos = ""; return min; }
         return val;
      }

      let year = next("-", 1970, 2300),
         month = next("-", 1, 12) - 1,
         day = next(" ", 1, 31),
         hour = next(":", 0, 23),
         min = next(":", 0, 59),
         sec = next("s", 0, 59),
         msec = next(" ", 0, 999);

      let dt = new Date(Date.UTC(year, month, day, hour, min, sec, msec));

      let offset = dt.getTime();

      // now also handle suffix like GMT or GMT -0600
      sof = sof.toUpperCase();

      if (sof.indexOf('GMT') == 0) {
         offset += dt.getTimezoneOffset() * 60000;
         sof = sof.substr(4).trim();
         if (sof.length > 3) {
            let p = 0, sign = 1000;
            if (sof[0] == '-') { p = 1; sign = -1000; }
            offset -= sign * (parseInt(sof.substr(p, 2)) * 3600 + parseInt(sof.substr(p + 2, 2)) * 60);
         }
      }

      return offset;
   }

   /** @summary Function used to provide svg:path for the smoothed curves.
    *
    * @desc reuse code from d3.js. Used in TH1, TF1 and TGraph painters
    * kind should contain "bezier" or "line".
    * If first symbol "L", then it used to continue drawing
    * @private */
   jsrp.BuildSvgPath = function(kind, bins, height, ndig) {

      let smooth = kind.indexOf("bezier") >= 0;

      if (ndig === undefined) ndig = smooth ? 2 : 0;
      if (height === undefined) height = 0;

      function jsroot_d3_svg_lineSlope(p0, p1) {
         return (p1.gry - p0.gry) / (p1.grx - p0.grx);
      }
      function jsroot_d3_svg_lineFiniteDifferences(points) {
         let i = 0, j = points.length - 1, m = [], p0 = points[0], p1 = points[1], d = m[0] = jsroot_d3_svg_lineSlope(p0, p1);
         while (++i < j) {
            m[i] = (d + (d = jsroot_d3_svg_lineSlope(p0 = p1, p1 = points[i + 1]))) / 2;
         }
         m[i] = d;
         return m;
      }
      function jsroot_d3_svg_lineMonotoneTangents(points) {
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
      }

      let res = { path: "", close: "" }, bin = bins[0], maxy = Math.max(bin.gry, height + 5),
         currx = Math.round(bin.grx), curry = Math.round(bin.gry), dx, dy, npnts = bins.length;

      function conv(val) {
         let vvv = Math.round(val);
         if ((ndig == 0) || (vvv === val)) return vvv.toString();
         let str = val.toFixed(ndig);
         while ((str[str.length - 1] == '0') && (str.lastIndexOf(".") < str.length - 1))
            str = str.substr(0, str.length - 1);
         if (str[str.length - 1] == '.')
            str = str.substr(0, str.length - 1);
         if (str == "-0") str = "0";
         return str;
      }

      res.path = ((kind[0] == "L") ? "L" : "M") + conv(bin.grx) + "," + conv(bin.gry);

      // just calculate all deltas, can be used to build exclusion
      if (smooth || kind.indexOf('calc') >= 0)
         jsroot_d3_svg_lineMonotoneTangents(bins);

      if (smooth) {
         // build smoothed curve
         res.path += "c" + conv(bin.dgrx) + "," + conv(bin.dgry) + ",";
         for (let n = 1; n < npnts; ++n) {
            let prev = bin;
            bin = bins[n];
            if (n > 1) res.path += "s";
            res.path += conv(bin.grx - bin.dgrx - prev.grx) + "," + conv(bin.gry - bin.dgry - prev.gry) + "," + conv(bin.grx - prev.grx) + "," + conv(bin.gry - prev.gry);
            maxy = Math.max(maxy, prev.gry);
         }
      } else if (npnts < 10000) {
         // build simple curve
         for (let n = 1; n < npnts; ++n) {
            bin = bins[n];
            dx = Math.round(bin.grx) - currx;
            dy = Math.round(bin.gry) - curry;
            if (dx && dy) res.path += "l" + dx + "," + dy;
            else if (!dx && dy) res.path += "v" + dy;
            else if (dx && !dy) res.path += "h" + dx;
            currx += dx; curry += dy;
            maxy = Math.max(maxy, curry);
         }
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
            if (dy) res.path += "l" + dx + "," + dy;
            else res.path += "h" + dx;
            currx = lastx; curry = lasty;
            prevy = cminy = cmaxy = lasty;
         }

         if (cminy != cmaxy) {
            if (cminy != curry) res.path += "v" + (cminy - curry);
            res.path += "v" + (cmaxy - cminy);
            if (cmaxy != prevy) res.path += "v" + (prevy - cmaxy);
            curry = prevy;
         }

      }

      if (height > 0)
         res.close = "L" + conv(bin.grx) + "," + conv(maxy) +
            "h" + conv(bins[0].grx - bin.grx) + "Z";

      return res;
   }

   // ========================================================================================

   /**
    * @summary Base painter class in JSROOT
    *
    * @class
    * @memberof JSROOT
    * @private
    */

   function BasePainter() {
      this.divid = null; // either id of element (preferable) or element itself
   }

   /** @summary Access painter reference, stored in first child element.
    *
    *    - on === true - set *this* as painter
    *    - on === false - delete painter reference
    *    - on === undefined - return painter
    *
    * @param {boolean} on - that to perfrom
    * @private
    */
   BasePainter.prototype.AccessTopPainter = function(on) {
      let main = this.select_main().node(),
         chld = main ? main.firstChild : null;
      if (!chld) return null;
      if (on === true) chld.painter = this; else
         if (on === false) delete chld.painter;
      return chld.painter;
   }

   /** @summary Generic method to cleanup painter */
   BasePainter.prototype.Cleanup = function(keep_origin) {

      let origin = this.select_main('origin');
      if (!origin.empty() && !keep_origin) origin.html("");
      if (this._changed_layout)
         this.set_layout_kind('simple');
      this.AccessTopPainter(false);
      this.divid = null;
      delete this._selected_main;

      if (this._hpainter && typeof this._hpainter.ClearPainter === 'function') this._hpainter.ClearPainter(this);

      delete this._changed_layout;
      delete this._hitemname;
      delete this._hdrawopt;
      delete this._hpainter;
   }

   /** @summary Function should be called by the painter when first drawing is completed
    * @private */
   BasePainter.prototype.DrawingReady = function(res_painter, res_value) {
      let res = (res_value === undefined) ? true : !!res_value;
      this._ready_called_ = res;
      if (this._ready_callbacks_ !== undefined) {
         let callbacks = (res ? this._ready_callbacks_ : this._reject_callbacks_) || [];
         if (!this._return_res_painter) res_painter = this;

         delete this._return_res_painter;
         delete this._ready_callbacks_;
         delete this._reject_callbacks_;

         while (callbacks.length)
            JSROOT.callBack(callbacks.shift(), res_painter);
      }
      return this;
   }

   /** @summary Function should be called when first drawing fails
    * @private */
   BasePainter.prototype.DrawingFail = function(res_painter) { return this.DrawingReady(res_painter, false); }

   /** @summary Call back will be called when painter ready with the drawing
    * @private */
   BasePainter.prototype.WhenReady = function(resolveFunc, rejectFunc) {
      if (typeof resolveFunc !== 'function') return;
      if ('_ready_called_' in this)
         return JSROOT.callBack(resolveFunc, this);
      if (!this._ready_callbacks_)
         this._ready_callbacks_ = [resolveFunc];
      else
         this._ready_callbacks_.push(resolveFunc);
      if (rejectFunc) {
         if (!this._reject_callbacks_)
            this._reject_callbacks_ = [rejectFunc];
         else
            this._reject_callbacks_.push(rejectFunc);
      }
   }

   /** @summary Create Promise object which will be completed when drawing is ready
    * @private */
   BasePainter.prototype.Promise = function(is_ready) {
      if (is_ready)
         this.DrawingReady(this);

      if (this._ready_called_)
         return Promise.resolve(this); // painting is done, we could return promise

      return new Promise((resolve, reject) => {
         this.WhenReady(resolve, reject);
      });
   }

   /** @summary Reset ready state - painter should again call DrawingReady to signal readyness
   * @private */
   BasePainter.prototype.ResetReady = function() {
      delete this._ready_called_;
      delete this._ready_callbacks_;
   }

   /** @summary Returns drawn object
    * @abstract */
   BasePainter.prototype.GetObject = function() {}

   /** @summary Returns true if type match with drawn object type
    * @param {string} typename - type name to check with
    * @returns {boolean} true if draw objects matches with provided type name
    * @abstract
    * @private */
   BasePainter.prototype.MatchObjectType = function(/* typename */) {}

   /** @summary Called to update drawn object content
    * @returns {boolean} true if update was performed
    * @abstract
    * @private */
   BasePainter.prototype.UpdateObject = function(/* obj */) {}

   /** @summary Redraw all objects in current pad
    * @param {string} reason - why redraw performed, can be "zoom" or empty ]
    * @abstract
    * @private */
   BasePainter.prototype.RedrawPad = function(/* reason */) {}

   /** @summary Updates object and readraw it
    * @param {object} obj - new version of object, values will be updated in original object
    * @returns {boolean} true if object updated and redrawn */
   BasePainter.prototype.RedrawObject = function(obj) {
      if (!this.UpdateObject(obj)) return false;
      let current = document.body.style.cursor;
      document.body.style.cursor = 'wait';
      this.RedrawPad();
      document.body.style.cursor = current;
      return true;
   }

   /** @summary Checks if draw elements were resized and drawing should be updated
    * @returns {boolean} true if resize was detected
    * @abstract
    * @private */
   BasePainter.prototype.CheckResize = function(/* arg */) {}

   /** @summary access to main HTML element used for drawing - typically <div> element
     * @desc if main element was layouted, returns main element inside layout
    * @param {string} is_direct - if 'origin' specified, returns original element even if actual drawing moved to some other place
    * @returns {object} d3.select for main element for drawing, defined with this.divid. */
   BasePainter.prototype.select_main = function(is_direct) {

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

   /** @summary Returns string with value of main element id attribute
   * @desc if main element does not have id, it will be generated */
   BasePainter.prototype.get_main_id = function() {
      let elem = this.select_main();
      if (elem.empty()) return "";
      let id = elem.attr("id");
      if (!id) {
         id = "jsroot_element_" + JSROOT._.id_counter++;
         elem.attr("id", id);
      }
      return id;
   }

   /** @summary Returns layout kind
    * @private */
   BasePainter.prototype.get_layout_kind = function() {
      let origin = this.select_main('origin'),
         layout = origin.empty() ? "" : origin.property('layout');

      return layout || 'simple';
   }

   /** @summary Set layout kind
    * @private */
   BasePainter.prototype.set_layout_kind = function(kind, main_selector) {
      // change layout settings
      let origin = this.select_main('origin');
      if (!origin.empty()) {
         if (!kind) kind = 'simple';
         origin.property('layout', kind);
         origin.property('layout_selector', (kind != 'simple') && main_selector ? main_selector : null);
         this._changed_layout = (kind !== 'simple'); // use in cleanup
      }
   }

   /** @summary Function checks if geometry of main div was changed.
    *
    * @desc returns size of area when main div is drawn
    * take into account enlarge state
    *
    * @private
    */
   BasePainter.prototype.check_main_resize = function(check_level, new_size, height_factor) {

      let enlarge = this.enlarge_main('state'),
         main_origin = this.select_main('origin'),
         main = this.select_main(),
         lmt = 5; // minimal size

      if (enlarge !== 'on') {
         if (new_size && new_size.width && new_size.height)
            main_origin.style('width', new_size.width + "px")
               .style('height', new_size.height + "px");
      }

      let rect_origin = this.get_visible_rect(main_origin, true),
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

      let rect = this.get_visible_rect(main),
         old_h = main.property('draw_height'), old_w = main.property('draw_width');

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
    *
    * @desc Possible values for parameter:
    *
    *    - true - try to enlarge
    *    - false - cancel enlarge state
    *    - 'toggle' - toggle enlarge state
    *    - 'state' - return current state
    *    - 'verify' - check if element can be enlarged
    *
    * if action not specified, just return possibility to enlarge main div
    *
    * @private
    */
   BasePainter.prototype.enlarge_main = function(action, skip_warning) {

      let main = this.select_main(true),
         origin = this.select_main('origin');

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

         let rect1 = this.get_visible_rect(main),
            rect2 = this.get_visible_rect(enlarge);

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

   /** @summary Return CSS value in given HTML element
    * @private */
   BasePainter.prototype.GetStyleValue = function(elem, name) {
      if (!elem || elem.empty()) return 0;
      let value = elem.style(name);
      if (!value || (typeof value !== 'string')) return 0;
      value = parseFloat(value.replace("px", ""));
      return isNaN(value) ? 0 : Math.round(value);
   }

   /** @summary Returns rect with width/height which correspond to the visible area of drawing region of element.
    * @private */
   BasePainter.prototype.get_visible_rect = function(elem, fullsize) {

      if (JSROOT.nodejs)
         return { width: parseInt(elem.attr("width")), height: parseInt(elem.attr("height")) };

      let rect = elem.node().getBoundingClientRect(),
         res = { width: Math.round(rect.width), height: Math.round(rect.height) };

      if (!fullsize) {
         // this is size exclude padding area
         res.width -= this.GetStyleValue(elem, 'padding-left') + this.GetStyleValue(elem, 'padding-right');
         res.height -= this.GetStyleValue(elem, 'padding-top') - this.GetStyleValue(elem, 'padding-bottom');
      }

      return res;
   }

   /** @summary Assign painter to specified element
    * @desc base painter does not creates canvas or frames
    * it registered in the first child element
    * @param {string|object} divid - element ID or DOM Element */
   BasePainter.prototype.SetDivId = function(divid) {
      if (divid !== undefined) {
         this.divid = divid;
         delete this._selected_main;
      }

      this.AccessTopPainter(true);
   }

   /** @summary Set item name, associated with the painter
    * @desc Used by {@link JSROOT.HiearchyPainter}
    * @private */
   BasePainter.prototype.SetItemName = function(name, opt, hpainter) {
      if (typeof name === 'string') this._hitemname = name;
      else delete this._hitemname;
      // only upate draw option, never delete. null specified when update drawing
      if (typeof opt === 'string') this._hdrawopt = opt;

      this._hpainter = hpainter;
   }

   /** @summary Returns assigned item name */
   BasePainter.prototype.GetItemName = function() { return ('_hitemname' in this) ? this._hitemname : null; }

   /** @summary Returns assigned item draw option
    * @private */
   BasePainter.prototype.GetItemDrawOpt = function() { return ('_hdrawopt' in this) ? this._hdrawopt : ""; }

   /** @summary Check if it makes sense to zoom inside specified axis range
    * @param {string} axis - name of axis like 'x', 'y', 'z'
    * @param {number} left - left axis range
    * @param {number} right - right axis range
    * @returns true is zooming makes sense
    * @abstract
    * @private
    */
   BasePainter.prototype.CanZoomIn = function(/* axis, left, right */) {}

   // ==============================================================================


   /**
    * @summary Painter class for ROOT objects
    *
    * @class
    * @memberof JSROOT
    * @extends ObjectPainter
    * @param {object} obj - object to draw
    * @param {string} [opt] - object draw options
    * @private
    */

   function ObjectPainter(obj, opt) {
      BasePainter.call(this);
      this.draw_g = null; // container for all drawn objects
      this.pad_name = ""; // name of pad where object is drawn
      this.main = null;  // main painter, received from pad
      if (typeof opt == "string") this.options = { original: opt };
      this.AssignObject(obj);
   }

   ObjectPainter.prototype = Object.create(BasePainter.prototype);

   /** @summary Assign object to the painter */
   ObjectPainter.prototype.AssignObject = function(obj) { this.draw_object = ((obj !== undefined) && (typeof obj == 'object')) ? obj : null; }

   /** @summary Assign snapid to the painter
    * @desc Identifier used to communicate with server side and identifies object on the server
    * @private */
   ObjectPainter.prototype.AssignSnapId = function(id) { this.snapid = id; }

   /** @summary Generic method to cleanup painter.
    * @desc Remove object drawing and in case of main painter - also main HTML components */
   ObjectPainter.prototype.Cleanup = function() {

      this.RemoveDrawG();

      let keep_origin = true;

      if (this.is_main_painter()) {
         let pp = this.pad_painter();
         if (!pp || pp.normal_canvas === false) keep_origin = false;
      }

      // cleanup all existing references
      this.pad_name = "";
      this.main = null;
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

      BasePainter.prototype.Cleanup.call(this, keep_origin);
   }

   /** @summary Returns drawn object */
   ObjectPainter.prototype.GetObject = function() { return this.draw_object; }

   /** @summary Returns drawn object class name */
   ObjectPainter.prototype.GetClassName = function() { return (this.draw_object ? this.draw_object._typename : "") || ""; }

   /** @summary Checks if drawn object matches with provided typename
    * @param {string} arg - typename
    * @param {string} arg._typename - if arg is object, use its typename */
   ObjectPainter.prototype.MatchObjectType = function(arg) {
      if (!arg || !this.draw_object) return false;
      if (typeof arg === 'string') return (this.draw_object._typename === arg);
      if (arg._typename) return (this.draw_object._typename === arg._typename);
      return this.draw_object._typename.match(arg);
   }

   /** @summary Change item name
    * @desc When available, used for svg:title proprty
    * @private */
   ObjectPainter.prototype.SetItemName = function(name, opt, hpainter) {
      BasePainter.prototype.SetItemName.call(this,name, opt, hpainter);
      if (this.no_default_title || (name == "")) return;
      let can = this.svg_canvas();
      if (!can.empty()) can.select("title").text(name);
                   else this.select_main().attr("title", name);
   }

   /** @summary Store actual options together with original string
    * @private */
   ObjectPainter.prototype.OptionsStore = function(original) {
      if (!this.options) return;
      if (!original) original = "";
      let pp = original.indexOf(";;");
      if (pp >= 0) original = original.substr(0, pp);
      this.options.original = original;
      this.options_store = JSROOT.extend({}, this.options);
   }

   /** @summary Checks if any draw options were changed
    * @private */
   ObjectPainter.prototype.OptionesChanged = function() {
      if (!this.options) return false;
      if (!this.options_store) return true;

      for (let k in this.options)
         if (this.options[k] !== this.options_store[k]) return true;

      return false;
   }

   /** @summary Return actual draw options as string
    * @private */
   ObjectPainter.prototype.OptionsAsString = function() {
      if (!this.options) return "";

      if (!this.OptionesChanged())
         return this.options.original || "";

      if (typeof this.options.asString == "function")
         return this.options.asString();

      return this.options.original || ""; // nothing better, return original draw option
   }

   /** @summary Generic method to update object content.
    * @desc Just copy all members from source object
    * @param {object} obj - object with new data */
   ObjectPainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;
      JSROOT.extend(this.GetObject(), obj);
      return true;
   }

   /** @summary Returns string which either item or object name.
    * @desc Such string can be used as tooltip. If result string larger than 20 symbols, it will be cutted.
    * @private */
   ObjectPainter.prototype.GetTipName = function(append) {
      let res = this.GetItemName(), obj = this.GetObject();
      if (!res) res = obj && obj.fName ? obj.fName : "";
      if (res.lenght > 20) res = res.substr(0, 17) + "...";
      if (res && append) res += append;
      return res;
   }

   /** @summary returns pad painter for specified pad
    * @private */
   ObjectPainter.prototype.pad_painter = function(pad_name) {
      let elem = this.svg_pad(typeof pad_name == "string" ? pad_name : undefined);
      return elem.empty() ? null : elem.property('pad_painter');
   }

   /** @summary returns canvas painter
    * @private */
   ObjectPainter.prototype.canv_painter = function() {
      let elem = this.svg_canvas();
      return elem.empty() ? null : elem.property('pad_painter');
   }

   /** @summary returns color from current list of colors
    * @private */
   ObjectPainter.prototype.get_color = function(indx) {
      let jsarr = this.root_colors;

      if (!jsarr) {
         let pp = this.canv_painter();
         jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : jsrp.root_colors;
      }

      return jsarr[indx];
   }

   /** @summary add color to list of colors
    * @private */
   ObjectPainter.prototype.add_color = function(color) {
      let jsarr = this.root_colors;
      if (!jsarr) {
         let pp = this.canv_painter();
         jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : jsrp.root_colors;
      }
      let indx = jsarr.indexOf(color);
      if (indx >= 0) return indx;
      jsarr.push(color);
      return jsarr.length - 1;
   }

   /** @summary returns tooltip allowed flag. Check canvas painter
    * @private */
   ObjectPainter.prototype.IsTooltipAllowed = function() {
      let src = this.canv_painter() || this;
      return src.tooltip_allowed ? true : false;
   }

   /** @summary returns tooltip allowed flag
    * @private */
   ObjectPainter.prototype.SetTooltipAllowed = function(on) {
      let src = this.canv_painter() || this;
      src.tooltip_allowed = (on == "toggle") ? !src.tooltip_allowed : on;
   }

   /** @summary returns custom palette for the object. If forced, will be created
    * @private */
   ObjectPainter.prototype.get_palette = function(force, palettedid) {
      if (!palettedid) {
         let pp = this.pad_painter();
         if (!pp) return null;
         if (pp.custom_palette) return pp.custom_palette;
      }

      let cp = this.canv_painter();
      if (!cp) return null;
      if (cp.custom_palette && !palettedid)
         return cp.custom_palette;

      if (force && jsrp.GetColorPalette)
         cp.custom_palette = jsrp.GetColorPalette(palettedid);

      return cp.custom_palette;
   }

   /** @summary Method called when interactively changes attribute in given class
    * @abstract
    * @private */
   ObjectPainter.prototype.AttributeChange = function(/* class_name, member_name, new_value */) {
      // only for objects in web canvas make sense to handle attributes changes from GED
      // console.log("Changed attribute class = " + class_name + " member = " + member_name + " value = " + new_value);
   }

   /** @summary Checks if draw elements were resized and drawing should be updated.
    * @desc Redirects to {@link JSROOT.TPadPainter.CheckCanvasResize}
    * @private */
   ObjectPainter.prototype.CheckResize = function(arg) {
      let p = this.canv_painter();
      if (!p) return false;

      // only canvas should be checked
      p.CheckCanvasResize(arg);
      return true;
   }

   /** @summary removes <g> element with object drawing
    * @desc generic method to delete all graphical elements, associated with painter */
   ObjectPainter.prototype.RemoveDrawG = function() {
      if (this.draw_g) {
         this.draw_g.remove();
         this.draw_g = null;
      }
   }

   /** @summary (re)creates svg:g element for object drawings
    *
    * @desc either one attach svg:g to pad list of primitives (default)
    * or svg:g element created in specified frame layer (default main_layer)
    * @param {string} [frame_layer=undefined] - when specified, <g> element will be created inside frame layer, otherwise in pad primitives list
    */
   ObjectPainter.prototype.CreateG = function(frame_layer) {
      if (this.draw_g) {
         // one should keep svg:g element on its place
         // d3.selectAll(this.draw_g.node().childNodes).remove();
         this.draw_g.selectAll('*').remove();
      } else if (frame_layer) {
         let frame = this.svg_frame();
         if (frame.empty()) return frame;
         if (typeof frame_layer != 'string') frame_layer = "main_layer";
         let layer = frame.select("." + frame_layer);
         if (layer.empty()) layer = frame.select(".main_layer");
         this.draw_g = layer.append("svg:g");
      } else {
         let layer = this.svg_layer("primitives_layer");
         this.draw_g = layer.append("svg:g");

         // layer.selectAll(".most_upper_primitives").raise();
         let up = [], chlds = layer.node().childNodes;
         for (let n = 0; n < chlds.length; ++n)
            if (d3.select(chlds[n]).classed("most_upper_primitives")) up.push(chlds[n]);

         up.forEach(top => { d3.select(top).raise(); });
      }

      // set attributes for debugging
      if (this.draw_object) {
         this.draw_g.attr('objname', encodeURI(this.draw_object.fName || "name"));
         this.draw_g.attr('objtype', encodeURI(this.draw_object._typename || "type"));
      }

      this.draw_g.property('in_frame', !!frame_layer); // indicates coordinate system

      return this.draw_g;
   }

   /** @summary This is main graphical SVG element, where all drawings are performed
    * @private */
   ObjectPainter.prototype.svg_canvas = function() { return this.select_main().select(".root_canvas"); }

   /** @summary This is SVG element, correspondent to current pad
    * @private */
   ObjectPainter.prototype.svg_pad = function(pad_name) {
      if (pad_name === undefined)
         pad_name = this.pad_name;

      let c = this.svg_canvas();
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
    * @private */
   ObjectPainter.prototype.svg_layer = function(name) {
      let svg = this.svg_pad();
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

   /** @summary Method returns current pad name
    * @param {string} [new_name] - when specified, new current pad name will be configured
    * @private */
   ObjectPainter.prototype.CurrentPadName = function(new_name) {
      let svg = this.svg_canvas();
      if (svg.empty()) return "";
      let curr = svg.property('current_pad');
      if (new_name !== undefined) svg.property('current_pad', new_name);
      return curr;
   }

   /** @summary Returns ROOT TPad object
    * @private */
   ObjectPainter.prototype.root_pad = function() {
      let pad_painter = this.pad_painter();
      return pad_painter ? pad_painter.pad : null;
   }

   /** @summary Converts x or y coordinate into SVG pad coordinates.
    *
    *  @param {string} axis - name like "x" or "y"
    *  @param {number} value - axis value to convert.
    *  @param {boolean} ndc - is value in NDC coordinates
    *  @param {boolean} noround - skip rounding
    *  @returns {number} value of requested coordiantes, rounded if kind.noround not specified
    *  @private */
   ObjectPainter.prototype.AxisToSvg = function(axis, value, ndc, noround) {
      let use_frame = this.draw_g && this.draw_g.property('in_frame'),
         main = use_frame ? this.frame_painter() : null;

      if (use_frame && main && main["gr" + axis]) {
         value = (axis == "y") ? main.gry(value) + (use_frame ? 0 : main.frame_y())
            : main.grx(value) + (use_frame ? 0 : main.frame_x());
      } else if (use_frame) {
         value = 0; // in principal error, while frame calculation requested
      } else {
         let pad = ndc ? null : this.root_pad();
         if (pad) {
            if (axis == "y") {
               if (pad.fLogy)
                  value = (value > 0) ? Math.log10(value) : pad.fUymin;
               value = (value - pad.fY1) / (pad.fY2 - pad.fY1);
            } else {
               if (pad.fLogx)
                  value = (value > 0) ? Math.log10(value) : pad.fUxmin;
               value = (value - pad.fX1) / (pad.fX2 - pad.fX1);
            }
         }
         value = (axis == "y") ? (1 - value) * this.pad_height() : value * this.pad_width();
      }

      return noround ? value : Math.round(value);
   }

   /** @summary Converts pad SVG x or y coordinates into axis values.
   *
   *  @param {string} axis - name like "x" or "y"
   *  @param {number} coord - graphics coordiante.
   *  @param {boolean} ndc - kind of return value
   *  @returns {number} value of requested coordiantes
   *  @private */
   ObjectPainter.prototype.SvgToAxis = function(axis, coord, ndc) {
      let use_frame = this.draw_g && this.draw_g.property('in_frame');

      if (use_frame) {
         let main = this.frame_painter();
         return main ? main.RevertAxis(axis, coord) : 0;
      }

      let value = (axis == "y") ? (1 - coord / this.pad_height()) : coord / this.pad_width();
      let pad = ndc ? null : this.root_pad();

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

   /** @summary Return functor, which can convert x and y coordinates into pixels, used for drawing
    * @desc Produce functor can convert x and y value by calling func.x(x) and func.y(y)
    * @param {boolean} isndc - if NDC coordinates will be used
    * @private */
   ObjectPainter.prototype.AxisToSvgFunc = function(isndc) {
      let func = { isndc: isndc }, use_frame = this.draw_g && this.draw_g.property('in_frame');
      if (use_frame) func.main = this.frame_painter();
      if (func.main && !isndc && func.main.grx && func.main.gry) {
         func.offx = func.main.frame_x();
         func.offy = func.main.frame_y();
         func.x = function(x) { return Math.round(this.main.grx(x) + this.offx); }
         func.y = function(y) { return Math.round(this.main.gry(y) + this.offy); }
      } else {
         if (!isndc) func.pad = this.root_pad(); // need for NDC conversion
         func.padh = this.pad_height();
         func.padw = this.pad_width();
         func.x = function(value) {
            if (this.pad) {
               if (this.pad.fLogx)
                  value = (value > 0) ? Math.log10(value) : this.pad.fUxmin;
               value = (value - this.pad.fX1) / (this.pad.fX2 - this.pad.fX1);
            }
            return Math.round(value * this.padw);
         }
         func.y = function(value) {
            if (this.pad) {
               if (this.pad.fLogy)
                  value = (value > 0) ? Math.log10(value) : this.pad.fUymin;
               value = (value - this.pad.fY1) / (this.pad.fY2 - this.pad.fY1);
            }
            return Math.round((1 - value) * this.padh);
         }
      }
      return func;
   }

   /** @summary Returns svg element for the frame in current pad.
    * @private */
   ObjectPainter.prototype.svg_frame = function() { return this.svg_layer("primitives_layer").select(".root_frame"); }

   /** @summary Returns pad width.
    * @private  */
   ObjectPainter.prototype.pad_width = function() {
      let res = this.svg_pad();
      res = res.empty() ? 0 : res.property("draw_width");
      return isNaN(res) ? 0 : res;
   }

   /** @summary Returns pad height
    * @param {string} [pad_name] - optional pad name, otherwise where object painter is drawn
    * @private */
   ObjectPainter.prototype.pad_height = function() {
      let res = this.svg_pad();
      res = res.empty() ? 0 : res.property("draw_height");
      return isNaN(res) ? 0 : res;
   }

   /** @summary Returns frame painter in current pad
    * @desc Pad has direct reference on frame if any
    * @private */
   ObjectPainter.prototype.frame_painter = function() {
      let pp = this.pad_painter();
      return pp ? pp.frame_painter() : null;
   }

   /** @summary Returns property of the frame painter
    * @private */
   ObjectPainter.prototype.frame_property = function(name) {
      let pp = this.frame_painter();
      return pp && pp[name] ? pp[name] : 0;
   }

   /** @summary Returns frame X coordinate relative to current pad */
   ObjectPainter.prototype.frame_x = function() { return this.frame_property("_frame_x"); }

   /** @summary Returns frame Y coordinate relative to current pad */
   ObjectPainter.prototype.frame_y = function() { return this.frame_property("_frame_y"); }

   /** @summary Returns frame width */
   ObjectPainter.prototype.frame_width = function() { return this.frame_property("_frame_width"); }

   /** @summary Returns frame height */
   ObjectPainter.prototype.frame_height = function() { return this.frame_property("_frame_height"); }

   /** @summary Returns main object painter on the pad.
    * @desc Normally this is first histogram drawn on the pad, which also draws all axes
    * @param {boolean} [not_store] - if true, prevent temporary store of main painter reference */
   ObjectPainter.prototype.main_painter = function(not_store) {
      let res = this.main;
      if (!res) {
         let svg_p = this.svg_pad();
         if (svg_p.empty()) {
            res = this.AccessTopPainter();
         } else {
            res = svg_p.property('mainpainter');
         }
         if (!res) res = null;
         if (!not_store) this.main = res;
      }
      return res;
   }

   /** @summary Returns true if this is main painter */
   ObjectPainter.prototype.is_main_painter = function() { return this === this.main_painter(); }

   /** @summary Assigns id of top element (normally div where drawing is done).
    *
    * @desc In some situations canvas may not exists - for instance object drawn as html, not as svg.
    * In such case the only painter will be assigned to the first element
    *
    * Following values of kind parameter are allowed:
    *   -  -1  only assign id, this painter not add to painters list
    *   -   0  normal painter (default)
    *   -   1  major objects like TH1/TH2 (required canvas with frame)
    *   -   2  if canvas missing, create it, but not set as main object
    *   -   3  if canvas and (or) frame missing, create them, but not set as main object
    *   -   4  major objects like TH3 (required canvas and frame in 3d mode)
    *   -   5  major objects like TGeoVolume (do not require canvas)
    *
    * @param {string|object} divid - id of div element or directly DOMElement
    * @param {number} [kind] - kind of object drawn with painter
    * @param {string} [pad_name] - when specified, subpad name used for object drawing
    * @private */
   ObjectPainter.prototype.SetDivId = function(divid, kind, pad_name) {

      if (divid !== undefined) {
         this.divid = divid;
         delete this._selected_main;
      }

      if (!kind || isNaN(kind)) kind = 0;

      // check if element really exists
      if ((kind >= 0) && this.select_main(true).empty()) {
         if (typeof divid == 'string') console.error('not found HTML element with id: ' + divid);
         else console.error('specified HTML element can not be selected with d3.select()');
         return false;
      }

      this.create_canvas = false;

      // SVG element where canvas is drawn
      let svg_c = this.svg_canvas();

      if (svg_c.empty() && (kind > 0) && (kind !== 5)) {
         if (typeof jsrp.drawCanvas == 'function')
             jsrp.drawCanvas(divid, null, ((kind == 2) || (kind == 4)) ? "noframe" : "");
         else
             return alert("Fail to draw TCanvas - please contact JSROOT developers");
         svg_c = this.svg_canvas();
         this.create_canvas = true;
      }

      if (svg_c.empty()) {
         if ((kind < 0) || (kind === 5) || this.iscan) return true;
         this.AccessTopPainter(true);
         return true;
      }

      // SVG element where current pad is drawn (can be canvas itself)
      this.pad_name = pad_name;
      if (this.pad_name === undefined)
         this.pad_name = this.CurrentPadName();

      if (kind < 0) return true;

      // create TFrame element if not exists
      if ((kind == 1) || (kind == 3) || (kind == 4))
         if (this.svg_frame().select(".main_layer").empty()) {
            if (typeof jsrp.drawFrame == 'function')
               jsrp.drawFrame(divid, null, (kind == 4) ? "3d" : "");
            if ((kind != 4) && this.svg_frame().empty())
               return alert("Fail to draw dummy TFrame");
         }

      let svg_p = this.svg_pad(this.pad_name); // important - padrent pad element accessed here
      if (svg_p.empty()) return true;

      let pp = svg_p.property('pad_painter');
      if (pp && (pp !== this)) {
         pp.painters.push(this);
         // workround to provide style for next object draing
         if (!this.rstyle && pp.next_rstyle)
            this.rstyle = pp.next_rstyle;
      }

      if (((kind === 1) || (kind === 4) || (kind === 5)) && !svg_p.property('mainpainter'))
         // when this is first main painter in the pad
         svg_p.property('mainpainter', this);

      return true;
   }

   /** @summary Calculate absolute position of provided selection.
    * @private */
   ObjectPainter.prototype.CalcAbsolutePosition = function(sel, pos) {
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

   /** @summary Creates marker attributes object
    *
    * @desc Can be used to produce markers in painter.
    * See {@link JSROOT.TAttMarkerHandler} for more info.
    * Instance assigned as this.markeratt data member, recognized by GED editor
    * @param {object} args - either TAttMarker or see arguments of {@link JSROOT.TAttMarkerHandler}
    * @returns created handler */
   ObjectPainter.prototype.createAttMarker = function(args) {
      if (!args || (typeof args !== 'object')) args = { std: true }; else
         if (args.fMarkerColor !== undefined && args.fMarkerStyle !== undefined && args.fMarkerSize !== undefined) args = { attr: args, std: false };

      if (args.std === undefined) args.std = true;
      if (args.painter === undefined) args.painter = this;

      let handler = args.std ? this.markeratt : null;

      if (!handler)
         handler = new TAttMarkerHandler(args);
      else if (!handler.changed || args.force)
         handler.SetArgs(args);

      if (args.std) this.markeratt = handler;

      // handler.used = false; // mark that line handler is not yet used
      return handler;
   }


   /** @summary Creates line attributes object.
   *
   * @desc Can be used to produce lines in painter.
   * See {@link JSROOT.TAttLineHandler} for more info.
   * Instance assigned as this.lineatt data member, recognized by GED editor
   * @param {object} args - either TAttLine or see constructor arguments of {@link JSROOT.TAttLineHandler} */
   ObjectPainter.prototype.createAttLine = function(args) {
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
         handler.SetArgs(args);

      if (args.std) this.lineatt = handler;

      // handler.used = false; // mark that line handler is not yet used
      return handler;
   }

   /** @summary Creates fill attributes object.
    *
    * @desc Method dedicated to create fill attributes, bound to canvas SVG
    * otherwise newly created patters will not be usable in the canvas
    * See {@link JSROOT.TAttFillHandler} for more info.
    * Instance assigned as this.fillatt data member, recognized by GED editor

    * @param {object} args - for special cases one can specify TAttFill as args or number of parameters
    * @param {boolean} [args.std = true] - this is standard fill attribute for object and should be used as this.fillatt
    * @param {object} [args.attr = null] - object, derived from TAttFill
    * @param {number} [args.pattern = undefined] - integer index of fill pattern
    * @param {number} [args.color = undefined] - integer index of fill color
    * @param {string} [args.color_as_svg = undefined] - color will be specified as SVG string, not as index from color palette
    * @param {number} [args.kind = undefined] - some special kind which is handled differently from normal patterns
    * @returns created handle
   */
   ObjectPainter.prototype.createAttFill = function(args) {
      if (!args || (typeof args !== 'object')) args = { std: true }; else
         if (args._typename && args.fFillColor !== undefined && args.fFillStyle !== undefined) args = { attr: args, std: false };

      if (args.std === undefined) args.std = true;

      let handler = args.std ? this.fillatt : null;

      if (!args.svg) args.svg = this.svg_canvas();
      if (args.painter === undefined) args.painter = this;

      if (!handler)
         handler = new TAttFillHandler(args);
      else if (!handler.changed || args.force)
         handler.SetArgs(args);

      if (args.std) this.fillatt = handler;

      // handler.used = false; // mark that fill handler is not yet used

      return handler;
   }

   /** @summary call function for each painter in the pad
    * @private */
   ObjectPainter.prototype.ForEachPainter = function(userfunc, kind) {
      // Iterate over all known painters

      // special case of the painter set as pointer of first child of main element
      let painter = this.AccessTopPainter();
      if (painter) {
         if (kind !== "pads") userfunc(painter);
         return;
      }

      // iterate over all painters from pad list
      let pp = this.pad_painter();
      if (pp) pp.ForEachPainterInPad(userfunc, kind);
   }

   /** @summary indicate that redraw was invoked via interactive action (like context menu or zooming)
    * @desc Use to catch such action by GED and by server-side
    * @private */
   ObjectPainter.prototype.InteractiveRedraw = function(arg, info, subelem) {

      let reason;
      if ((typeof info == "string") && (info.indexOf("exec:") != 0)) reason = info;

      if (arg == "pad") {
         this.RedrawPad(reason);
      } else if (arg == "axes") {
         let main = this.main_painter(true); // works for pad and any object drawn in the pad
         if (main && (typeof main.DrawAxes == 'function'))
            main.DrawAxes();
         else
            this.RedrawPad(reason);
      } else if (arg !== false) {
         this.Redraw(reason);
      }

      // inform GED that something changes
      let pp = this.pad_painter(), canp = this.canv_painter();

      if (canp && (typeof canp.PadEvent == 'function'))
         canp.PadEvent("redraw", pp, this, null, subelem);

      // inform server that drawopt changes
      if (canp && (typeof canp.ProcessChanges == 'function'))
         canp.ProcessChanges(info, this, subelem);
   }

   /** @summary Redraw all objects in correspondent pad */
   ObjectPainter.prototype.RedrawPad = function(reason) {
      let pad_painter = this.pad_painter();
      if (pad_painter) pad_painter.Redraw(reason);
   }

   /** @summary execute selected menu command, either locally or remotely
    * @private */
   ObjectPainter.prototype.ExecuteMenuCommand = function(method) {

      if (method.fName == "Inspect") {
         // primitve inspector, keep it here
         this.ShowInspector();
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
   ObjectPainter.prototype.WebCanvasExec = function(exec, snapid) {
      if (!exec || (typeof exec != 'string')) return;

      let canp = this.canv_painter();
      if (canp && (typeof canp.SubmitExec == "function"))
         canp.SubmitExec(this, exec, snapid);
   }

   /** @summary remove all created draw attributes
    * @private */
   ObjectPainter.prototype.DeleteAtt = function() {
      delete this.lineatt;
      delete this.fillatt;
      delete this.markeratt;
   }

   /** @summary Show object in inspector */
   ObjectPainter.prototype.ShowInspector = function(obj) {
      let main = this.select_main(),
         rect = this.get_visible_rect(main),
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
         obj = this.GetObject();

      JSROOT.draw(id, obj, 'inspect');
   }

   /** @summary Fill context menu for the object
    * @private */
   ObjectPainter.prototype.FillContextMenu = function(menu) {
      let title = this.GetTipName();
      if (this.GetObject() && ('_typename' in this.GetObject()))
         title = this.GetObject()._typename + "::" + title;

      menu.add("header:" + title);

      menu.AddAttributesMenu(this);

      if (menu.size() > 0)
         menu.add('Inspect', this.ShowInspector);

      return menu.size() > 0;
   }

   /** @summary Produce exec string for WebCanas to set color value
    * @desc Color can be id or string, but should belong to list of known colors
    * For higher color numbers TColor::GetColor(r,g,b) will be invoked to ensure color is exists
    * @private */
   ObjectPainter.prototype.GetColorExec = function(col, method) {
      let id = -1, arr = jsrp.root_colors;
      if (typeof col == "string") {
         if (!col || (col == "none")) id = 0; else
            for (let k = 1; k < arr.length; ++k)
               if (arr[k] == col) { id = k; break; }
         if ((id < 0) && (col.indexOf("rgb") == 0)) id = 9999;
      } else if (!isNaN(col) && arr[col]) {
         id = col;
         col = arr[id];
      }

      if (id < 0) return "";

      if (id >= 50) {
         // for higher color numbers ensure that such color exists
         let c = d3.color(col);
         id = "TColor::GetColor(" + c.r + "," + c.g + "," + c.b + ")";
      }

      return "exec:" + method + "(" + id + ")";
   }



   /** @summary returns function used to display object status
    * @private */
   ObjectPainter.prototype.GetShowStatusFunc = function() {
      // return function used to display object status
      // automatically disabled when drawing is enlarged - status line will be invisible

      let pp = this.canv_painter(), res = jsrp.ShowStatus;

      if (pp && (typeof pp.ShowCanvasStatus === 'function')) res = pp.ShowCanvasStatus.bind(pp);

      if (res && (this.enlarge_main('state') === 'on')) res = null;

      return res;
   }

   /** @summary shows objects status
    * @private */
   ObjectPainter.prototype.ShowObjectStatus = function() {
      // method called normally when mouse enter main object element

      let obj = this.GetObject(),
         status_func = this.GetShowStatusFunc();

      if (obj && status_func) status_func(this.GetItemName() || obj.fName, obj.fTitle || obj._typename, obj._typename);
   }


   /** @summary try to find object by name in list of pad primitives
    * @desc used to find title drawing
    * @private */
   ObjectPainter.prototype.FindInPrimitives = function(objname) {
      let painter = this.pad_painter();

      let arr = painter && painter.pad && painter.pad.fPrimitives ? painter.pad.fPrimitives.arr : null;

      if (arr && arr.length)
         for (let n = 0; n < arr.length; ++n) {
            let prim = arr[n];
            if (('fName' in prim) && (prim.fName === objname)) return prim;
         }

      return null;
   }

   /** @summary Try to find painter for specified object
    * @desc can be used to find painter for some special objects, registered as
    * histogram functions
    * @private */
   ObjectPainter.prototype.FindPainterFor = function(selobj, selname, seltype) {

      let painter = this.pad_painter();
      let painters = painter ? painter.painters : null;
      if (!painters) return null;

      for (let n = 0; n < painters.length; ++n) {
         let pobj = painters[n].GetObject();
         if (!pobj) continue;

         if (selobj && (pobj === selobj)) return painters[n];
         if (!selname && !seltype) continue;
         if (selname && (pobj.fName !== selname)) continue;
         if (seltype && (pobj._typename !== seltype)) continue;
         return painters[n];
      }

      return null;
   }

   /** @summary Remove painter from list of painters and cleanup all drawings */
   ObjectPainter.prototype.DeleteThis = function() {
      let pp = this.pad_painter();
      if (pp) {
         let k = pp.painters.indexOf(this);
         if (k >= 0) pp.painters.splice(k, 1);
      }

      this.Cleanup();
   }

   /** @summary Redraw object
    * @desc Basic method, should be reimplemented in all derived objects
    * for the case when drawing should be repeated
    * @abstract */
   ObjectPainter.prototype.Redraw = function(/* reason */) {}

   /** @summary Start text drawing
     * @desc required before any text can be drawn
     * @private */
   ObjectPainter.prototype.StartTextDrawing = function(font_face, font_size, draw_g, max_font_size) {

      if (!draw_g) draw_g = this.draw_g;

      let font = (font_size === 'font') ? font_face : new FontHandler(font_face, font_size);

      let pp = this.pad_painter();

      draw_g.call(font.func);

      draw_g.property('draw_text_completed', false) // indicate that draw operations submitted
            .property('all_args',[]) // array of all submitted args, makes easier to analyze them
            .property('text_font', font)
            .property('text_factor', 0.)
            .property('max_text_width', 0) // keep maximal text width, use it later
            .property('max_font_size', max_font_size)
            .property("_fast_drawing", pp && pp._fast_drawing);

      if (draw_g.property("_fast_drawing"))
         draw_g.property("_font_too_small", (max_font_size && (max_font_size < 5)) || (font.size < 4));
   }

   /** @summary function used to remember maximal text scaling factor
    * @private */
   ObjectPainter.prototype.TextScaleFactor = function(value, draw_g) {
      if (!draw_g) draw_g = this.draw_g;
      if (value && (value > draw_g.property('text_factor'))) draw_g.property('text_factor', value);
   }

   /** @summary getBBox does not work in mozilla when object is not displayed or not visible :(
    * getBoundingClientRect() returns wrong sizes for MathJax
    * are there good solution?
    * @private */
   ObjectPainter.prototype.GetBoundarySizes = function(elem) {
      if (!elem) { console.warn('empty node in GetBoundarySizes'); return { width: 0, height: 0 }; }
      let box = elem.getBoundingClientRect(); // works always, but returns sometimes results in ex values, which is difficult to use
      if (parseFloat(box.width) > 0) box = elem.getBBox(); // check that elements visible, request precise value
      let res = { width: parseInt(box.width), height: parseInt(box.height) };
      if ('left' in box) { res.x = parseInt(box.left); res.y = parseInt(box.right); } else
         if ('x' in box) { res.x = parseInt(box.x); res.y = parseInt(box.y); }
      return res;
   }

   /** @summary Finish text drawing
    * @returns {Promise} when done
    * @private */
   ObjectPainter.prototype.FinishTextPromise = function(draw_g) {
      return new Promise(resolveFunc => {
          this.FinishTextDrawing(draw_g, resolveFunc);
      });
   }

   /** @summary Finish text drawing
    * @desc Should be called to complete all text drawing operations
    * @param {function} call_ready - callback function
    * @private */
   ObjectPainter.prototype.FinishTextDrawing = function(draw_g, call_ready, checking_mathjax) {
      if (!draw_g) draw_g = this.draw_g;

      if (checking_mathjax) {
         if (!draw_g.property('draw_text_completed')) return;
      } else {
         draw_g.property('draw_text_completed', true); // mark that text drawing is completed
      }

      let all_args = draw_g.property('all_args'), missing = 0;
      if (!all_args) {
         console.log('Text drawing is finished - why?????');
         all_args = [];
      }

      all_args.forEach(arg => { if (!arg.ready) missing++; });

      if (missing > 0) {
         if (typeof call_ready == 'function')
            draw_g.node().text_callback = call_ready;
         return 0;
      }

      draw_g.property('all_args', null); // clear all_args property

      // adjust font size (if there are normal text)
      let f = draw_g.property('text_factor'),
          font = draw_g.property('text_font'),
          max_sz = draw_g.property('max_font_size'),
          font_size = font.size, any_text = false;

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
         }
      });

      // now hidden text after rescaling can be shown
      all_args.forEach(arg => {
         if (!arg.txt_node) return; // only normal text is processed
         any_text = true;
         let txt = arg.txt_node;
         delete arg.txt_node;
         txt.attr('visibility', null);


         if (JSROOT.nodejs) {
            if (arg.scale && (f > 0)) { arg.box.width = arg.box.width / f; arg.box.height = arg.box.height / f; }
         } else if (!arg.plain && !arg.fast) {
            // exact box dimension only required when complex text was build
            arg.box = this.GetBoundarySizes(txt.node());
         }

         // if (arg.text.length>20) console.log(arg.box, arg.align, arg.x, arg.y, 'plain', arg.plain, 'inside', arg.width, arg.height);

         if (arg.width) {
            // adjust x position when scale into specified rectangle
            if (arg.align[0] == "middle") arg.x += arg.width / 2; else
               if (arg.align[0] == "end") arg.x += arg.width;
         }

         arg.dx = arg.dy = 0;

         if (arg.plain) {
            txt.attr("text-anchor", arg.align[0]);
         } else {
            txt.attr("text-anchor", "start");
            arg.dx = ((arg.align[0] == "middle") ? -0.5 : ((arg.align[0] == "end") ? -1 : 0)) * arg.box.width;
         }

         if (arg.height) {
            if (arg.align[1].indexOf('bottom') === 0) arg.y += arg.height; else
               if (arg.align[1] == 'middle') arg.y += arg.height / 2;
         }

         if (arg.plain) {
            if (arg.align[1] == 'top') txt.attr("dy", ".8em"); else
               if (arg.align[1] == 'middle') {
                  if (JSROOT.nodejs) txt.attr("dy", ".4em"); else txt.attr("dominant-baseline", "middle");
               }
         } else {
            arg.dy = ((arg.align[1] == 'top') ? (arg.top_shift || 1) : (arg.align[1] == 'middle') ? (arg.mid_shift || 0.5) : 0) * arg.box.height;
         }

         if (!arg.rotate) { arg.x += arg.dx; arg.y += arg.dy; arg.dx = arg.dy = 0; }

         // use translate and then rotate to avoid complex sign calculations
         let trans = (arg.x || arg.y) ? "translate(" + Math.round(arg.x) + "," + Math.round(arg.y) + ")" : "";
         if (arg.rotate) trans += " rotate(" + Math.round(arg.rotate) + ")";
         if (arg.dx || arg.dy) trans += " translate(" + Math.round(arg.dx) + "," + Math.round(arg.dy) + ")";
         if (trans) txt.attr("transform", trans);
      });

      // when no any normal text drawn - remove font attributes
      if (!any_text)
         font.clearFont(draw_g);

      if (!call_ready) call_ready = draw_g.node().text_callback;
      draw_g.node().text_callback = null;

      // if specified, call ready function
      JSROOT.callBack(call_ready);
      return 0;
   }

   /** @summary Draw text
    *
    *  @param {object} arg - different text draw options
    *  @param {string} arg.text - text to draw
    *  @param {number} [arg.align = 12] - int value like 12 or 31
    *  @param {string} [arg.align = undefined] - end;bottom
    *  @param {number} [arg.x = 0] - x position
    *  @param {number} [arg.y = 0] - y position
    *  @param {number} [arg.width] - when specified, adjust font size in the specified box
    *  @param {number} [arg.height] - when specified, adjust font size in the specified box
    *  @param {number} arg.latex - 0 - plain text, 1 - normal TLatex, 2 - math
    *  @param {string} [arg.color=black] - text color
    *  @param {number} [arg.rotate] - rotaion angle
    *  @param {number} [arg.font_size] - fixed font size
    *  @param {object} [arg.draw_g] - element where to place text, if not specified central draw_g container is used
    */
   ObjectPainter.prototype.DrawText = function(arg) {

      if (!arg.text) arg.text = "";

      arg.draw_g = arg.draw_g || this.draw_g;

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
      } else if (arg.align && (typeof arg.align == 'object') && arg.align.length == 2) {
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

         if (!arg.plain || arg.simple_latex) {
            JSROOT.require(['latex']).then(ltx => {
               if (arg.simple_latex)
                  ltx.producePlainText(this, arg.txt_node, arg);
               else
                  ltx.produceLatex(this, arg.txt_node, arg);
               arg.ready = true;
               this.postprocessText(arg.txt_node, arg);
               this.FinishTextDrawing(arg.draw_g, null, true); // check if all other elements are completed
            });
            return 0;
         }

         arg.plain = true;
         arg.txt_node.text(arg.text);
         arg.ready = true;

         return this.postprocessText(arg.txt_node, arg);
      }

      arg.mj_node = arg.draw_g.append("svg:g")
                           .attr('visibility', 'hidden'); // hide text until drawing is finished

      JSROOT.require(['latex'])
            .then(ltx => ltx.produceMathjax(this, arg.mj_node, arg))
            .then(() => { arg.ready = true; this.FinishTextDrawing(arg.draw_g, null, true); });

      return 0;
   }

   /** @summary After normal SVG text is generated, check and recalculate some properties
     * @private */
   ObjectPainter.prototype.postprocessText = function(txt_node, arg) {
      // complete rectangle with very rougth size estimations

      arg.box = !JSROOT.nodejs && !JSROOT.settings.ApproxTextSize && !arg.fast ? this.GetBoundarySizes(txt_node.node()) :
               (arg.text_rect || { height: arg.font_size * 1.2, width: arg.font.approxTextWidth(arg.text) });

      txt_node.attr('visibility', 'hidden'); // hide elements until text drawing is finished

      if (arg.box.width > arg.draw_g.property('max_text_width')) arg.draw_g.property('max_text_width', arg.box.width);
      if (arg.scale) this.TextScaleFactor(1.05 * arg.box.width / arg.width, arg.draw_g);
      if (arg.scale) this.TextScaleFactor(1. * arg.box.height / arg.height, arg.draw_g);

      arg.result_width = arg.box.width;
      arg.result_height = arg.box.height;

      // in some cases
      if (typeof arg.post_process == 'function')
         arg.post_process(this);

      return arg.box.width;
   }

   // ===========================================================

   /** @summary Produce ticks for d3.scaleLog
     * @desc Fixing following problem, described [here]{@link https://stackoverflow.com/questions/64649793}
     * @private */
   jsrp.PoduceLogTicks = function(func, number) {
      function linearArray(arr) {
         let sum1 = 0, sum2 = 0;
         for (let k=1;k<arr.length;++k) {
            let diff = (arr[k] - arr[k-1]);
            sum1 += diff;
            sum2 += diff*diff;
         }
         let mean = sum1/(arr.length-1);
         let dev = sum2/(arr.length-1) - mean*mean;
         if (dev <= 0) return true;
         if (Math.abs(mean) < 1e-100) return false;
         return Math.sqrt(dev)/mean < 1e-10;
      }

      let arr = func.ticks(number);
      while ((number > 4) && linearArray(arr)) {
          number = Math.round(number*0.8);
          arr = func.ticks(number);
      }

      return arr;
   }


   /** @summary Set active pad painter
    *
    * @desc Normally be used to handle key press events, which are global in the web browser
    * @param {object} args - functions arguments
    * @param {object} args.pp - pad painter
    * @param {boolean} [args.active] - is pad activated or not
    * @private */
   jsrp.SelectActivePad = function(args) {
      if (args.active) {
         if (this.$active_pp && (typeof this.$active_pp.SetActive == 'function'))
            this.$active_pp.SetActive(false);

         this.$active_pp = args.pp;

         if (this.$active_pp && (typeof this.$active_pp.SetActive == 'function'))
            this.$active_pp.SetActive(true);
      } else if (this.$active_pp === args.pp) {
         delete this.$active_pp;
      }
   }

   /** @summary Returns current active pad
    * @desc Should be used only for keyboard handling
    * @private */

   jsrp.GetActivePad = function() {
      return this.$active_pp;
   }

   // =====================================================================

   JSROOT.EAxisBits = {
      kDecimals: JSROOT.BIT(7),
      kTickPlus: JSROOT.BIT(9),
      kTickMinus: JSROOT.BIT(10),
      kAxisRange: JSROOT.BIT(11),
      kCenterTitle: JSROOT.BIT(12),
      kCenterLabels: JSROOT.BIT(14),
      kRotateTitle: JSROOT.BIT(15),
      kPalette: JSROOT.BIT(16),
      kNoExponent: JSROOT.BIT(17),
      kLabelsHori: JSROOT.BIT(18),
      kLabelsVert: JSROOT.BIT(19),
      kLabelsDown: JSROOT.BIT(20),
      kLabelsUp: JSROOT.BIT(21),
      kIsInteger: JSROOT.BIT(22),
      kMoreLogLabels: JSROOT.BIT(23),
      kOppositeTitle: JSROOT.BIT(32) // atrificial bit, not possible to set in ROOT
   };

   // ================= painter of raw text ========================================

   /** @summary Generic text drawing
    * @private */
   jsrp.drawRawText = function(divid, txt /*, opt*/) {

      let painter = new BasePainter();
      painter.txt = txt;
      painter.SetDivId(divid);

      painter.RedrawObject = function(obj) {
         this.txt = obj;
         this.Draw();
         return true;
      }

      painter.Draw = function() {
         let txt = (this.txt._typename && (this.txt._typename == "TObjString")) ? this.txt.fString : this.txt.value;
         if (typeof txt != 'string') txt = "<undefined>";

         let mathjax = this.txt.mathjax || (JSROOT.settings.Latex == JSROOT.constants.Latex.AlwaysMathJax);

         if (!mathjax && !('as_is' in this.txt)) {
            let arr = txt.split("\n"); txt = "";
            for (let i = 0; i < arr.length; ++i)
               txt += "<pre style='margin:0'>" + arr[i] + "</pre>";
         }

         let frame = this.select_main(),
            main = frame.select("div");
         if (main.empty())
            main = frame.append("div").style('max-width', '100%').style('max-height', '100%').style('overflow', 'auto');
         main.html(txt);

         // (re) set painter to first child element
         this.SetDivId(this.divid);

         if (mathjax)
            return JSROOT.require('latex').then(ltx => ltx.typesetMathjax(frame.node()));
      }

      let promise = painter.Draw();

      return promise ? promise.then(() => painter) : painter.Promise(true);
   }

   /** @summary Register handle to react on window resize
    *
    * @desc function used to react on browser window resize event
    * While many resize events could come in short time,
    * resize will be handled with delay after last resize event
    * handle can be function or object with CheckResize function
    * one could specify delay after which resize event will be handled
    * @private */
   JSROOT.RegisterForResize = function(handle, delay) {

      if (!handle) return;

      let myInterval = null, myDelay = delay ? delay : 300;

      if (myDelay < 20) myDelay = 20;

      function ResizeTimer() {
         myInterval = null;

         document.body.style.cursor = 'wait';
         if (typeof handle == 'function') handle(); else
            if ((typeof handle == 'object') && (typeof handle.CheckResize == 'function')) handle.CheckResize(); else
               if (typeof handle == 'string') {
                  let node = d3.select('#' + handle);
                  if (!node.empty()) {
                     let mdi = node.property('mdi');
                     if (mdi) {
                        mdi.CheckMDIResize();
                     } else {
                        JSROOT.resize(node.node());
                     }
                  }
               }
         document.body.style.cursor = 'auto';
      }

      function ProcessResize() {
         if (myInterval !== null) clearTimeout(myInterval);
         myInterval = setTimeout(ResizeTimer, myDelay);
      }

      window.addEventListener('resize', ProcessResize);
   }

   // list of registered draw functions
   let drawFuncs = { lst: [
      { name: "TCanvas", icon: "img_canvas", prereq: "gpad", func: ".drawCanvas", opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" },
      { name: "TPad", icon: "img_canvas", prereq: "gpad", func: ".drawPad", opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" },
      { name: "TSlider", icon: "img_canvas", prereq: "gpad", func: ".drawPad" },
      { name: "TFrame", icon: "img_frame", prereq: "gpad", func: ".drawFrame" },
      { name: "TPave", icon: "img_pavetext", prereq: "hist", func: ".drawPave" },
      { name: "TPaveText", icon: "img_pavetext", prereq: "hist", func: ".drawPave" },
      { name: "TPavesText", icon: "img_pavetext", prereq: "hist", func: ".drawPave" },
      { name: "TPaveStats", icon: "img_pavetext", prereq: "hist", func: ".drawPave" },
      { name: "TPaveLabel", icon: "img_pavelabel", prereq: "hist", func: ".drawPave" },
      { name: "TDiamond", icon: "img_pavelabel", prereq: "hist", func: ".drawPave" },
      { name: "TLatex", icon: "img_text", prereq: "more", func: ".drawText", direct: true },
      { name: "TMathText", icon: "img_text", prereq: "more", func: ".drawText", direct: true },
      { name: "TText", icon: "img_text", prereq: "more", func: ".drawText", direct: true },
      { name: /^TH1/, icon: "img_histo1d", prereq: "hist", func: ".drawHistogram1D", opt: ";hist;P;P0;E;E1;E2;E3;E4;E1X0;L;LF2;B;B1;A;TEXT;LEGO;same", ctrl: "l" },
      { name: "TProfile", icon: "img_profile", prereq: "hist", func: ".drawHistogram1D", opt: ";E0;E1;E2;p;AH;hist" },
      { name: "TH2Poly", icon: "img_histo2d", prereq: "hist", func: ".drawHistogram2D", opt: ";COL;COL0;COLZ;LCOL;LCOL0;LCOLZ;LEGO;TEXT;same", expand_item: "fBins", theonly: true },
      { name: "TProfile2Poly", sameas: "TH2Poly" },
      { name: "TH2PolyBin", icon: "img_histo2d", draw_field: "fPoly" },
      { name: /^TH2/, icon: "img_histo2d", prereq: "hist", func: ".drawHistogram2D", opt: ";COL;COLZ;COL0;COL1;COL0Z;COL1Z;COLA;BOX;BOX1;PROJ;PROJX1;PROJX2;PROJX3;PROJY1;PROJY2;PROJY3;SCAT;TEXT;TEXTE;TEXTE0;CONT;CONT1;CONT2;CONT3;CONT4;ARR;SURF;SURF1;SURF2;SURF4;SURF6;E;A;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same", ctrl: "colz" },
      { name: "TProfile2D", sameas: "TH2" },
      { name: /^TH3/, icon: 'img_histo3d', prereq: "hist3d", func: ".drawHistogram3D", opt: ";SCAT;BOX;BOX2;BOX3;GLBOX1;GLBOX2;GLCOL" },
      { name: "THStack", icon: "img_histo1d", prereq: "hist", func: ".drawHStack", expand_item: "fHists", opt: "NOSTACK;HIST;E;PFC;PLC" },
      { name: "TPolyMarker3D", icon: 'img_histo3d', prereq: "hist3d", func: ".drawPolyMarker3D" },
      { name: "TPolyLine3D", icon: 'img_graph', prereq: "3d", func: ".drawPolyLine3D", direct: true },
      { name: "TGraphStruct" },
      { name: "TGraphNode" },
      { name: "TGraphEdge" },
      { name: "TGraphTime", icon: "img_graph", prereq: "more", func: ".drawGraphTime", opt: "once;repeat;first", theonly: true },
      { name: "TGraph2D", icon: "img_graph", prereq: "hist3d", func: ".drawGraph2D", opt: ";P;PCOL", theonly: true },
      { name: "TGraph2DErrors", icon: "img_graph", prereq: "hist3d", func: ".drawGraph2D", opt: ";P;PCOL;ERR", theonly: true },
      { name: "TGraphPolargram", icon: "img_graph", prereq: "more", func: ".drawGraphPolargram", theonly: true },
      { name: "TGraphPolar", icon: "img_graph", prereq: "more", func: ".drawGraphPolar", opt: ";F;L;P;PE", theonly: true },
      { name: /^TGraph/, icon: "img_graph", prereq: "more", func: ".drawGraph", opt: ";L;P" },
      { name: "TEfficiency", icon: "img_graph", prereq: "more", func: ".drawEfficiency", opt: ";AP" },
      { name: "TCutG", sameas: "TGraph" },
      { name: /^RooHist/, sameas: "TGraph" },
      { name: /^RooCurve/, sameas: "TGraph" },
      { name: "RooPlot", icon: "img_canvas", prereq: "more", func: ".drawRooPlot" },
      { name: "TMultiGraph", icon: "img_mgraph", prereq: "more", func: ".drawMultiGraph", expand_item: "fGraphs" },
      { name: "TStreamerInfoList", icon: 'img_question', prereq: "hierarchy", func: ".drawStreamerInfo" },
      { name: "TPaletteAxis", icon: "img_colz", prereq: "hist", func: ".drawPave" },
      { name: "TWebPainting", icon: "img_graph", prereq: "more", func: ".drawWebPainting" },
      { name: "TCanvasWebSnapshot", icon: "img_canvas", prereq: "gpad", func: ".drawPadSnapshot" },
      { name: "TPadWebSnapshot", sameas: "TCanvasWebSnapshot" },
      { name: "kind:Text", icon: "img_text", func: jsrp.drawRawText },
      { name: "TObjString", icon: "img_text", func: jsrp.drawRawText },
      { name: "TF1", icon: "img_tf1", prereq: "math;more", func: ".drawFunction" },
      { name: "TF2", icon: "img_tf2", prereq: "math;hist", func: ".drawTF2" },
      { name: "TSpline3", icon: "img_tf1", prereq: "more", func: ".drawSpline" },
      { name: "TSpline5", icon: "img_tf1", prereq: "more", func: ".drawSpline" },
      { name: "TEllipse", icon: 'img_graph', prereq: "more", func: ".drawEllipse", direct: true },
      { name: "TArc", sameas: 'TEllipse' },
      { name: "TCrown", sameas: 'TEllipse' },
      { name: "TPie", icon: 'img_graph', prereq: "more", func: ".drawPie", direct: true },
      { name: "TPieSlice", icon: 'img_graph', dummy: true },
      { name: "TLine", icon: 'img_graph', prereq: "more", func: ".drawLine", direct: true },
      { name: "TArrow", icon: 'img_graph', prereq: "more", func: ".drawArrow", direct: true },
      { name: "TPolyLine", icon: 'img_graph', prereq: "more", func: ".drawPolyLine", direct: true },
      { name: "TCurlyLine", sameas: 'TPolyLine' },
      { name: "TCurlyArc", sameas: 'TPolyLine' },
      { name: "TGaxis", icon: "img_graph", prereq: "gpad", func: ".drawGaxis" },
      { name: "TLegend", icon: "img_pavelabel", prereq: "hist", func: ".drawPave" },
      { name: "TBox", icon: 'img_graph', prereq: "more", func: ".drawBox", direct: true },
      { name: "TWbox", icon: 'img_graph', prereq: "more", func: ".drawBox", direct: true },
      { name: "TSliderBox", icon: 'img_graph', prereq: "more", func: ".drawBox", direct: true },
      { name: "TAxis3D", prereq: "hist3d", func: ".drawAxis3D" },
      { name: "TMarker", icon: 'img_graph', prereq: "more", func: ".drawMarker", direct: true },
      { name: "TPolyMarker", icon: 'img_graph', prereq: "more", func: ".drawPolyMarker", direct: true },
      { name: "TASImage", icon: 'img_mgraph', prereq: "more", func: ".drawASImage" },
      { name: "TJSImage", icon: 'img_mgraph', prereq: "more", func: ".drawJSImage", opt: ";scale;center" },
      { name: "TGeoVolume", icon: 'img_histo3d', prereq: "geom", func: ".drawGeoObject", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;wire;dflt", ctrl: "dflt" },
      { name: "TEveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", func: ".drawGeoObject", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;wire;dflt", ctrl: "dflt" },
      { name: "ROOT::Experimental::REveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", func: ".drawGeoObject", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;wire;dflt", ctrl: "dflt" },
      { name: "TGeoOverlap", icon: 'img_histo3d', prereq: "geom", expand: "JSROOT.GEO.expandObject", func: ".drawGeoObject", opt: ";more;all;count;projx;projz;wire;dflt", dflt: "dflt", ctrl: "expand" },
      { name: "TGeoManager", icon: 'img_histo3d', prereq: "geom", expand: "JSROOT.GEO.expandObject", func: ".drawGeoObject", opt: ";more;all;count;projx;projz;wire;tracks;dflt", dflt: "expand", ctrl: "dflt" },
      { name: /^TGeo/, icon: 'img_histo3d', prereq: "geom", func: ".drawGeoObject", opt: ";more;all;axis;compa;count;projx;projz;wire;dflt", ctrl: "dflt" },
      // these are not draw functions, but provide extra info about correspondent classes
      { name: "kind:Command", icon: "img_execute", execute: true },
      { name: "TFolder", icon: "img_folder", icon2: "img_folderopen", noinspect: true, prereq: "hierarchy", expand: ".FolderHierarchy" },
      { name: "TTask", icon: "img_task", prereq: "hierarchy", expand: ".TaskHierarchy", for_derived: true },
      { name: "TTree", icon: "img_tree", prereq: "tree", expand: 'JSROOT.TreeHierarchy', func: 'JSROOT.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" },
      { name: "TNtuple", icon: "img_tree", prereq: "tree", expand: 'JSROOT.TreeHierarchy', func: 'JSROOT.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" },
      { name: "TNtupleD", icon: "img_tree", prereq: "tree", expand: 'JSROOT.TreeHierarchy', func: 'JSROOT.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" },
      { name: "TBranchFunc", icon: "img_leaf_method", prereq: "tree", func: 'JSROOT.drawTree', opt: ";dump", noinspect: true },
      { name: /^TBranch/, icon: "img_branch", prereq: "tree", func: 'JSROOT.drawTree', dflt: "expand", opt: ";dump", ctrl: "dump", shift: "inspect", ignore_online: true },
      { name: /^TLeaf/, icon: "img_leaf", prereq: "tree", noexpand: true, func: 'JSROOT.drawTree', opt: ";dump", ctrl: "dump", ignore_online: true },
      { name: "TList", icon: "img_list", prereq: "hierarchy", func: ".drawList", expand: ".ListHierarchy", dflt: "expand" },
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
      { name: "ROOT::Experimental::RCanvas", icon: "img_canvas", prereq: "v7gpad", func: "JSROOT.v7.drawCanvas", opt: "", expand_item: "fPrimitives" },
      { name: "ROOT::Experimental::RCanvasDisplayItem", icon: "img_canvas", prereq: "v7gpad", func: "JSROOT.v7.drawPadSnapshot", opt: "", expand_item: "fPrimitives" }
   ], cache: {} };


   /** @summary Register draw function for the class
    * @desc List of supported draw options could be provided, separated  with ';'
    * @param {object} args - arguments
    * @param {string|regexp} args.name - class name or regexp pattern
    * @param {string} [args.prereq] - prerequicities to load before search for the draw function
    * @param {string} args.func - name of draw function for the class or just a function
    * @param {boolean} [args.direct=false] - if true, function is just Redraw() method of ObjectPainter
    * @param {string} [args.opt] - list of supported draw options (separated with semicolon) like "col;scat;"
    * @param {string} [args.icon] - icon name shown for the class in hierarchy browser
    * @param {string} [args.draw_field] - draw only data member from object, like fHistogram
    * @private */
   JSROOT.addDrawFunc = function(args) {
      drawFuncs.lst.push(args);
      return args;
   }

   /** @summary return draw handle for specified item kind
     * @desc kind could be ROOT.TH1I for ROOT classes or just
     * kind string like "Command" or "Text"
     * selector can be used to search for draw handle with specified option (string)
     * or just sequence id
     * @private */
   JSROOT.getDrawHandle = function(kind, selector) {

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
            return JSROOT.getDrawHandle("ROOT." + h.sameas, selector);

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
   JSROOT.addStreamerInfos = function(lst) {
      if (!lst) return;

      function CheckBaseClasses(si, lvl) {
         if (!si.fElements) return null;
         if (lvl > 10) return null; // protect against recursion

         for (let j = 0; j < si.fElements.arr.length; ++j) {
            // extract streamer info for each class member
            let element = si.fElements.arr[j];
            if (element.fTypeName !== 'BASE') continue;

            let handle = JSROOT.getDrawHandle("ROOT." + element.fName);
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
         if (JSROOT.getDrawHandle("ROOT." + si.fName) !== null) continue;

         let handle = CheckBaseClasses(si, 0);

         if (!handle) continue;

         let newhandle = JSROOT.extend({}, handle);
         // delete newhandle.for_derived; // should we disable?
         newhandle.name = si.fName;
         drawFuncs.lst.push(newhandle);
      }
   }

   /** @summary Provide draw settings for specified class or kind
    * @private */
   JSROOT.getDrawSettings = function(kind, selector) {
      let res = { opts: null, inspect: false, expand: false, draw: false, handle: null };
      if (typeof kind != 'string') return res;
      let isany = false, noinspect = false, canexpand = false;
      if (typeof selector !== 'string') selector = "";

      for (let cnt = 0; cnt < 1000; ++cnt) {
         let h = JSROOT.getDrawHandle(kind, cnt);
         if (!h) break;
         if (!res.handle) res.handle = h;
         if (h.noinspect) noinspect = true;
         if (h.expand || h.expand_item || h.can_expand) canexpand = true;
         if (!('func' in h)) break;
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

   /** Returns array with supported draw options for the specified kind
    * @private */
   JSROOT.getDrawOptions = function(kind /*, selector*/) {
      return JSROOT.getDrawSettings(kind).opts;
   }

   /** @summary Returns true if provided object class can be drawn
    * @private */
   JSROOT.canDraw = function(classname) {
      return JSROOT.getDrawSettings("ROOT." + classname).opts !== null;
   }

   /** @summary Implementation of JSROOT.draw
     * @private */
   function jsroot_draw(divid, obj, opt) {
      if (!obj || (typeof obj !== 'object'))
         return Promise.reject(Error('not an object in JSROOT.draw'));

      if (opt == 'inspect')
         return JSROOT.require("hierarchy").then(() => jsrp.drawInspector(divid, obj));

      let handle;
      if ('_typename' in obj)
         handle = JSROOT.getDrawHandle("ROOT." + obj._typename, opt);
      else if ('_kind' in obj)
         handle = JSROOT.getDrawHandle(obj._kind, opt);
      else
         return JSROOT.require("hierarchy").then(() => jsrp.drawInspector(divid, obj));

      // this is case of unsupported class, close it normally
      if (!handle)
         return Promise.reject(Error(`Object of ${obj.kind ? obj.kind : obj._typename} cannot be shown with JSROOT.draw`));

      if (handle.dummy)
         return Promise.resolve(null);

      if (handle.draw_field && obj[handle.draw_field])
         return JSROOT.draw(divid, obj[handle.draw_field], opt);

      if (!handle.func) {
         if (opt && (opt.indexOf("same") >= 0)) {
            let main_painter = JSROOT.get_main_painter(divid);
            if (main_painter && (typeof main_painter.PerformDrop === 'function'))
               return main_painter.PerformDrop(obj, "", null, opt);
         }

         return Promise.reject(Error('Function not specified'));
      }

      return new Promise(function(resolveFunc, rejectFunc) {

         function isPromise(obj) {
            return obj && (typeof obj == 'object') && (typeof obj.then == 'function');
         }

         function completeDraw(painter) {
            if (isPromise(painter)) {
               painter.then(resolveFunc, rejectFunc);
            } else if (painter && (typeof painter == 'object') && (typeof painter.WhenReady == 'function'))
               painter.WhenReady(resolveFunc, rejectFunc);
            else if (painter)
               resolveFunc(painter);
            else
               rejectFunc(Error("fail to draw"));
         }

         let painter = null;

         function performDraw() {
            let promise;
            if (handle.direct) {
               painter = new ObjectPainter(obj, opt);
               painter.csstype = handle.csstype;
               painter.SetDivId(divid, 2);
               painter.Redraw = handle.func;
               let promise = painter.Redraw();
               if (!isPromise(promise)) {
                  painter.DrawingReady();
                  promise = undefined;
               }
            } else {
               painter = handle.func(divid, obj, opt);

               if (!isPromise(painter) && painter && !painter.options)
                  painter.options = { original: opt || "" };
            }

            completeDraw(promise || painter);
         }

         if (typeof handle.func == 'function')
            return performDraw();

         let funcname = handle.func;
         if (!funcname || (typeof funcname != "string"))
            return completeDraw(null);

         // try to find function without prerequisites
         let func = JSROOT.findFunction(funcname);
         if (func) {
            handle.func = func; // remember function once it is found
            return performDraw();
         }

        let prereq = handle.prereq || "";
        if (handle.script && (typeof handle.script == 'string'))
           prereq += ";" + handle.script;

         if (!prereq)
            return completeDraw(null);

         JSROOT.require(prereq).then(() => {
            let func = JSROOT.findFunction(funcname);
            if (!func) {
               alert('Fail to find function ' + funcname + ' after loading ' + prereq);
               return completeDraw(null);
            }

            handle.func = func; // remember function once it found

            performDraw();
         });
      }); // Promise
   }

   /**
    * @summary Draw object in specified HTML element with given draw options.
    *
    * @param {string|object} divid - id of div element to draw or directly DOMElement
    * @param {object} obj - object to draw, object type should be registered before in JSROOT
    * @param {string} opt - draw options separated by space, comma or semicolon
    * @param {function} [callback] - function called when drawing is completed, first argument is object painter instance
    * @returns {Promise} with painter object only if callback parameter is not specified
    *
    * @desc
    * An extensive list of support draw options can be found on [JSROOT examples page]{@link https://root.cern/js/latest/examples.htm}
    * Parameter ```callback``` kept only for backward compatibility and will be removed in JSROOT v6.2
    *
    * @example
    * JSROOT.openFile("https://root.cern/js/files/hsimple.root")
    *       .then(file => file.ReadObject("hpxpy;1"))
    *       .then(obj => JSROOT.draw("drawing", obj, "colz;logx;gridx;gridy"));
    *
    */

   JSROOT.draw = function(divid, obj, opt, callback) {
      let res = jsroot_draw(divid, obj, opt);
      if (!callback || (typeof callback != 'function')) return res;
      res.then(callback).catch(() => callback(null));
   }

   /**
    * @summary Redraw object in specified HTML element with given draw options.
    *
    * @desc If drawing was not drawn before, it will be performed with {@link JSROOT.draw}.
    * If drawing was already done, that content will be updated
    * @param {string|object} divid - id of div element to draw or directly DOMElement
    * @param {object} obj - object to draw, object type should be registered before in JSROOT
    * @param {string} opt - draw options
    * @param {function} [callback] - function called when drawing is completed, first argument will be object painter instance
    * @returns {Promise} with painter used only when callback parameter is not specified
    */
   JSROOT.redraw = function(divid, obj, opt, callback) {

      if (!obj || (typeof obj !== 'object'))
         return callback ? callback(null) : Promise.reject(Error('not an object in JSROOT.redraw'));

      let dummy = new ObjectPainter();
      dummy.SetDivId(divid, -1);
      let can_painter = dummy.canv_painter(), handle;
      if (obj._typename)
         handle = JSROOT.getDrawHandle("ROOT." + obj._typename);
      if (handle && handle.draw_field && obj[handle.draw_field])
         obj = obj[handle.draw_field];

      if (can_painter) {
         let res_painter = null;
         if (obj._typename === "TCanvas") {
            can_painter.RedrawObject(obj);
            res_painter = can_painter;
         } else {
            for (let i = 0; i < can_painter.painters.length; ++i) {
               let painter = can_painter.painters[i];
               if (painter.MatchObjectType(obj._typename))
                  if (painter.UpdateObject(obj, opt)) {
                     can_painter.RedrawPad();
                     res_painter = painter;
                     break;
                  }
            }
         }

         if (res_painter)
            return callback ? callback(res_painter) : Promise.resolve(res_painter);

         console.warn(`Cannot find painter to update object of type ${obj._typename}`);
      }

      JSROOT.cleanup(divid);

      return JSROOT.draw(divid, obj, opt, callback);
   }

   /** @summary Save object, drawn in specified element, as JSON.
    *
    * @desc Normally it is TCanvas object with list of primitives
    * @param {string|object} divid - id of top div element or directly DOMElement
    * @returns {string} produced JSON string */
   JSROOT.drawingJSON = function(divid) {
      let p = new ObjectPainter;
      p.SetDivId(divid, -1);

      let canp = p.canv_painter();
      return canp ? canp.ProduceJSON() : "";
   }

   /** @summary Compress SVG code, produced from JSROOT drawing
     * @desc removes extra info or empty elements
     * @private */
   jsrp.CompressSVG = function(svg) {

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
    *
    * @desc Function especially useful in Node.js environment to generate images for
    * supported ROOT classes
    *
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

         main.attr("width", args.width).attr("height", args.height);

         main.style("width", args.width + "px").style("height", args.height + "px");

         JSROOT._.svg_3ds = undefined;

         return JSROOT.draw(main.node(), args.object, args.option || "").then(() => {

            let has_workarounds = JSROOT._.svg_3ds && jsrp.ProcessSVGWorkarounds;

            main.select('svg')
                .attr("xmlns", "http://www.w3.org/2000/svg")
                .attr("xmlns:xlink", "http://www.w3.org/1999/xlink")
                .attr("width", args.width)
                .attr("height", args.height)
                .attr("style", null).attr("class", null).attr("x", null).attr("y", null);

            let svg = main.html();

            if (has_workarounds)
               svg = jsrp.ProcessSVGWorkarounds(svg);

            svg = jsrp.CompressSVG(svg);

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

   /**
    * @summary Check resize of drawn element
    *
    * @desc As first argument divid one should use same argument as for the drawing
    * As second argument, one could specify "true" value to force redrawing of
    * the element even after minimal resize of the element
    * Or one just supply object with exact sizes like { width:300, height:200, force:true };
    * @param {string|object} divid - id or DOM element
    * @param {boolean|object} arg - options on how to resize
    *
    * @example
    * JSROOT.resize("drawing", { width: 500, height: 200 } );
    * JSROOT.resize(document.querySelector("#drawing"), true);
    */
   JSROOT.resize = function(divid, arg) {
      if (arg === true) arg = { force: true }; else
         if (typeof arg !== 'object') arg = null;
      let done = false, dummy = new ObjectPainter();
      dummy.SetDivId(divid, -1);
      dummy.ForEachPainter(painter => {
         if (!done && (typeof painter.CheckResize == 'function'))
            done = painter.CheckResize(arg);
      });
      return done;
   }

   /** @summary Returns main painter object for specified HTML element - typically histogram painter
     * @param {string|object} divid - id or DOM element
     * @private */
   JSROOT.get_main_painter = function(divid) {
      let dummy = new JSROOT.ObjectPainter();
      dummy.SetDivId(divid, -1);
      return dummy.main_painter(true);
   }

   /** @summary Safely remove all JSROOT objects from specified element
     * @param {string|object} divid - id or DOM element
     * @example
     * JSROOT.cleanup("drawing");
     * JSROOT.cleanup(document.querySelector("#drawing")); */
   JSROOT.cleanup = function(divid) {
      let dummy = new ObjectPainter(), lst = [];
      dummy.SetDivId(divid, -1);
      dummy.ForEachPainter(painter => {
         if (lst.indexOf(painter) < 0) lst.push(painter);
      });
      for (let n = 0; n < lst.length; ++n) lst[n].Cleanup();
      dummy.select_main().html("");
      return lst;
   }

   /** @summary Display progress message in the left bottom corner.
    *  @desc Previous message will be overwritten
    * if no argument specified, any shown messages will be removed
    * @param {string} msg - message to display
    * @param {number} tmout - optional timeout in milliseconds, after message will disappear
    * @private */
   JSROOT.progress = function(msg, tmout) {
      if (JSROOT.BatchMode || (typeof document === 'undefined')) return;
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

      if (!isNaN(tmout) && (tmout > 0)) {
         box.property("with_timeout", true);
         setTimeout(JSROOT.progress.bind(JSROOT, '', -1), tmout);
      }
   }

   /** @summary Converts numeric value to string according to specified format.
    *
    * @param {number} value - value to convert
    * @param {strting} [fmt="6.4g"] - format can be like 5.4g or 4.2e or 6.4f
    * @param {boolean} [ret_fmt=false] - when true returns array with actual format
    * @returns {string|Array} - converted value or array with value and actual format
    * @private */
   JSROOT.FFormat = function(value, fmt, ret_fmt) {
      if (!fmt) fmt = "6.4g";

      fmt = fmt.trim();
      let len = fmt.length;
      if (len<2)
         return ret_fmt ? [value.toFixed(4), "6.4f"] : value.toFixed(4);
      let last = fmt[len-1];
      fmt = fmt.slice(0,len-1);
      let isexp, prec = fmt.indexOf(".");
      prec = (prec<0) ? 4 : parseInt(fmt.slice(prec+1));
      if (isNaN(prec) || (prec <=0)) prec = 4;

      let significance = false;
      if ((last=='e') || (last=='E')) { isexp = true; } else
      if (last=='Q') { isexp = true; significance = true; } else
      if ((last=='f') || (last=='F')) { isexp = false; } else
      if (last=='W') { isexp = false; significance = true; } else
      if ((last=='g') || (last=='G')) {
         let se = JSROOT.FFormat(value, fmt+'Q', true),
             sg = JSROOT.FFormat(value, fmt+'W', true);

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
     *
     * @desc Many browsers do not allow simple window.close() call,
     * therefore try several workarounds
     * @private */
   JSROOT.CloseCurrentWindow = function() {
      if (!window) return;
      window.close();
      window.open('', '_self').close();
   }

   jsrp.createRootColors();

   if (JSROOT.nodejs) jsrp.readStyleFromURL("?interactive=0&tooltip=0&nomenu&noprogress&notouch&toolbar=0&webgl=0");

   JSROOT.DrawOptions = DrawOptions;
   JSROOT.ColorPalette = ColorPalette;
   JSROOT.TAttLineHandler = TAttLineHandler;
   JSROOT.TAttFillHandler = TAttFillHandler;
   JSROOT.TAttMarkerHandler = TAttMarkerHandler;
   JSROOT.FontHandler = FontHandler;
   JSROOT.BasePainter = BasePainter;
   JSROOT.ObjectPainter = ObjectPainter;

   // Only for backward compatibility with v5, will be removed in later JSROOT versions
   JSROOT.TBasePainter = BasePainter;
   JSROOT.TObjectPainter = ObjectPainter;
   JSROOT.StoreJSON = JSROOT.drawingJSON;
   // end of compatibility mode

   JSROOT.Painter = jsrp;
   if (JSROOT.nodejs) module.exports = jsrp;

   return jsrp;

});
