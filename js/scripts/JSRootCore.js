/** @fileoverview Core methods of JavaScript ROOT
  * @namespace JSROOT */

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {

      var jsroot = factory({}),
          dir = jsroot.source_dir + "scripts/",
          ext = jsroot.source_min ? ".min" : "",
          norjs = (typeof requirejs=='undefined'),
          paths = {
            'd3'                   : dir+'d3.min',
            'jquery'               : dir+'jquery.min',
            'jquery-ui'            : dir+'jquery-ui.min',
            'jqueryui-mousewheel'  : dir+'jquery.mousewheel.min',
            'jqueryui-touch-punch' : dir+'touch-punch.min',
            'rawinflate'           : dir+'rawinflate.min',
            'MathJax'              : 'https://root.cern/js/mathjax/latest/MathJax.js?config=TeX-AMS-MML_SVG&amp;delayStartupUntil=configured',
            'dat.gui'              : dir+'dat.gui.min',
            'threejs'              : dir+'three.min',
            'threejs_all'          : dir+'three.extra.min',
            'JSRootCore'           : dir+'JSRootCore'+ext,
            'JSRootMath'           : dir+'JSRootMath'+ext,
            'JSRootIOEvolution'    : dir+'JSRootIOEvolution'+ext,
            'JSRootTree'           : dir+'JSRootTree'+ext,
            'JSRoot.openui5'       : dir+'JSRoot.openui5'+ext,
            'JSRootPainter'        : dir+'JSRootPainter'+ext,
            'JSRootPainter.v6'     : dir+'JSRootPainter.v6'+ext,
            'JSRootPainter.hist'   : dir+'JSRootPainter.hist'+ext,
            'JSRootPainter.hist3d' : dir+'JSRootPainter.hist3d'+ext,
            'JSRootPainter.more'   : dir+'JSRootPainter.more'+ext,
            'JSRootPainter.hierarchy' : dir+'JSRootPainter.hierarchy'+ext,
            'JSRootPainter.jquery' : dir+'JSRootPainter.jquery'+ext,
            'JSRootPainter.v7'     : dir+'JSRootPainter.v7'+ext,
            'JSRootPainter.v7hist' : dir+'JSRootPainter.v7hist'+ext,
            'JSRootPainter.v7more' : dir+'JSRootPainter.v7more'+ext,
            'JSRoot3DPainter'      : dir+'JSRoot3DPainter'+ext,
            'ThreeCSG'             : dir+'ThreeCSG'+ext,
            'JSRootGeoBase'        : dir+'JSRootGeoBase'+ext,
            'JSRootGeoPainter'     : dir+'JSRootGeoPainter'+ext
         };

      if (norjs) {
         // just define locations
         paths['MathJax'] = 'https://root.cern/js/mathjax/latest/MathJax.js?config=TeX-AMS-MML_SVG&amp;delayStartupUntil=configured';

         require({ paths: paths });
      } else {
         var cfg_paths;
         if ((requirejs.s!==undefined) && (requirejs.s.contexts !== undefined) && ((requirejs.s.contexts._!==undefined) &&
               requirejs.s.contexts._.config!==undefined)) cfg_paths = requirejs.s.contexts._.config.paths;
         else console.warn("Require.js paths changed - please contact JSROOT developers");

         // check if modules are already loaded
         for (var module in paths)
            if (requirejs.defined(module) || (cfg_paths && (module in cfg_paths)))
               delete paths[module];

         // configure all dependencies
         requirejs.config({
            paths: paths,
            shim: {
               'jqueryui-mousewheel': { deps: ['jquery-ui'] },
               'jqueryui-touch-punch': { deps: ['jquery-ui'] }
            }
         });
      }

      define( jsroot );

      if (norjs || !require.specified("JSRootCore"))
         define('JSRootCore', [], jsroot);

     if (norjs || !require.specified("jsroot"))
        define('jsroot', [], jsroot);

   } else if (typeof exports === 'object' /*&& typeof module !== 'undefined'*/) {
      // processing with Node.js or CommonJS

      //  mark JSROOT as used with Node.js
      exports.BatchMode = exports.nodejs = (typeof global==='object') && global.process && (Object.prototype.toString.call(global.process) === '[object process]');

      factory(exports);

   } else {

      if (typeof JSROOT != 'undefined')
         throw new Error("JSROOT is already defined", "JSRootCore.js");

      JSROOT = {};

      factory(JSROOT);
   }
} (function(JSROOT) {

   "use strict";

   JSROOT.version = "dev 19/06/2020";

   JSROOT.source_dir = "";
   JSROOT.source_min = false;
   JSROOT.source_fullpath = ""; // full name of source script
   JSROOT.bower_dir = null;     // when specified, use standard libs from bower location
   JSROOT.nocache = false;      // when specified, used as extra URL parameter to load JSROOT scripts
   JSROOT.wrong_http_response_handling = false; // when configured, try to handle wrong content-length response from server
   JSROOT.sources = ['core'];   // indicates which major sources were loaded

   JSROOT.id_counter = 1;       // avoid id value 0, starts from 1
   if (JSROOT.BatchMode === undefined)
      JSROOT.BatchMode = false; // when true, disables all kind of interactive features

   //openuicfg // DO NOT DELETE, used to configure openui5 usage like JSROOT.openui5src = "nojsroot";

   // JSROOT.use_full_libs = true;

   JSROOT.touches = false;
   JSROOT.key_handling = true;  // enable/disable key press handling in JSROOT

   JSROOT.browser = { isOpera: false, isFirefox: true, isSafari: false, isChrome: false, isIE: false, isWin: false };

   if ((typeof document !== "undefined") && (typeof window !== "undefined")) {
      var scripts = document.getElementsByTagName('script');
      for (var n = 0; n < scripts.length; ++n) {
         if (!scripts[n].src || (typeof scripts[n].src !== 'string')) continue;

         var pos = scripts[n].src.indexOf("scripts/JSRootCore.");
         if (pos<0) continue;

         JSROOT.source_dir = scripts[n].src.substr(0, pos);
         JSROOT.source_min = scripts[n].src.indexOf("scripts/JSRootCore.min.js") >= 0;
         JSROOT.source_fullpath = scripts[n].src;

         if ((console!==undefined) && (typeof console.log == 'function'))
            console.log("Set JSROOT.source_dir to " + JSROOT.source_dir + ", " + JSROOT.version);
         break;
      }

      JSROOT.touches = ('ontouchend' in document); // identify if touch events are supported
      JSROOT.browser.isOpera = !!window.opera || navigator.userAgent.indexOf(' OPR/') >= 0;
      JSROOT.browser.isFirefox = typeof InstallTrigger !== 'undefined';
      JSROOT.browser.isSafari = Object.prototype.toString.call(window.HTMLElement).indexOf('Constructor') > 0;
      JSROOT.browser.isChrome = !!window.chrome && !JSROOT.browser.isOpera;
      JSROOT.browser.isIE = !!document.documentMode;
      JSROOT.browser.isWin = navigator.platform.indexOf('Win') >= 0;
      JSROOT.browser.isChromeHeadless = navigator.userAgent.indexOf('HeadlessChrome') >= 0;
   }

   JSROOT.browser.isWebKit = JSROOT.browser.isChrome || JSROOT.browser.isSafari || JSROOT.browser.isOpera;

   // default draw styles, can be changed after loading of JSRootCore.js
   // this style also can be changed providing style=itemname in the URL
   JSROOT.gStyle = {
         Tooltip: 1, // 0 - off, 1 - on
         TooltipAnimation: 500, // time in msec for appearance of tooltips, 0 - no animation
         ContextMenu: true,
         Zooming: true,  // global zooming flag, enable/disable any kind of interactive zooming
         ZoomMouse: true,  // Zooming with the mouse events
         ZoomWheel: true,  // Zooming with mouse wheel
         ZoomTouch: true,  // Zooming with the touch devices
         MoveResize: true,   // enable move and resize of elements like statbox, title, pave, colz
         DragAndDrop: true,  // enables drag and drop functionality
         ToolBar: 'popup',  // show additional tool buttons on the canvas, false - disabled, true - enabled, 'popup' - only toggle button
         ToolBarSide: 'left', // 'left' left-bottom corner on canvas, 'right' - right-bottom corner on canvas, opposite on sub-pads
         ToolBarVert: false,  // display tool bar vertical (default false)
         CanEnlarge: true,  // if drawing inside particular div can be enlarged on full window
         CanAdjustFrame: false,  // if frame position can be adjusted to let show axis or colz labels
         ApproxTextSize: false,  // calculation of text size consumes time and can be skipped to improve performance (but with side effects on text adjustments)
         OptimizeDraw: 1, // drawing optimization: 0 - disabled, 1 - only for large (>5000 1d bins, >50 2d bins) histograms, 2 - always
         AutoStat: true,
         FrameNDC: { fX1NDC: 0.07, fY1NDC: 0.12, fX2NDC: 0.95, fY2NDC: 0.88 },
         Palette: 57,
         Latex: 2,    // 0 - never, 1 - only latex symbols, 2 - normal TLatex processing (default), 3 - use MathJax for complex case, 4 - use MathJax always
         // MathJax : 0,  // deprecated, will be supported till JSROOT 6.0, use Latex variable  0 - never, 1 - only for complex cases, 2 - always
         ProgressBox: true,  // show progress box
         Embed3DinSVG: 2,  // 0 - no embed, only 3D plot, 1 - overlay over SVG (IE/WebKit), 2 - embed into SVG (only Firefox)
         ImageSVG: !JSROOT.nodejs, // when producing SVG images, use <image> elements to insert 3D drawings from three.js,
                                   // To enable on nodejs, one should call "npm install canvas"
         NoWebGL: false, // if true, WebGL will be disabled
         GeoGradPerSegm: 6, // amount of grads per segment in TGeo spherical shapes like tube
         GeoCompressComp: true, // if one should compress faces after creation of composite shape,
         IgnoreUrlOptions: false, // if true, ignore all kind of URL options in the browser URL
         HierarchyLimit: 250,   // how many items shown on one level of hierarchy
         SmallPad: { width: 150, height: 100 },   // size of pad, where many features will be deactivated like text draw or zooming

         // XValuesFormat : "6.4g",   // custom format for all X values
         // YValuesFormat : "6.4g",   // custom format for all Y values
         // ZValuesFormat : "6.4g",   // custom format for all Z values

         // these are TStyle attributes, which can be changed via URL 'style' parameter or delivered by TWebCanvas

         fOptLogx: 0,
         fOptLogy: 0,
         fOptLogz: 0,
         fOptDate: 0,
         fOptFile: 0,
         fOptTitle: 1,
         fPadBottomMargin: 0.1,
         fPadTopMargin: 0.1,
         fPadLeftMargin: 0.1,
         fPadRightMargin: 0.1,
         fPadGridX: false,
         fPadGridY: false,
         fPadTickX: 0,
         fPadTickY: 0,
         fStatColor: 0,
         fStatTextColor: 1,
         fStatBorderSize: 1,
         fStatFont: 42,
         fStatFontSize: 0,
         fStatStyle: 1001,
         fStatFormat: "6.4g",
         fStatX: 0.98,
         fStatY: 0.935,
         fStatW: 0.2,
         fStatH: 0.16,
         fTitleAlign: 23,
         fTitleColor: 0,
         fTitleTextColor: 1,
         fTitleBorderSize: 0,
         fTitleFont: 42,
         fTitleFontSize: 0.05,
         fTitleStyle: 0,
         fTitleX: 0.5,
         fTitleY: 0.995,
         fTitleW: 0,
         fTitleH: 0,
         fFitFormat: "5.4g",
         fOptStat: 1111,
         fOptFit: 0,
         fNumberContours: 20,
         fGridColor: 0,
         fGridStyle: 3,
         fGridWidth: 1,
         fFrameFillColor: 0,
         fFrameFillStyle: 1001,
         fFrameLineColor: 1,
         fFrameLineWidth: 1,
         fFrameLineStyle: 1,
         fFrameBorderSize: 1,
         fFrameBorderMode: 0,
         fEndErrorSize: 2,   // size in pixels of end error for E1 draw options
         fErrorX: 0.5,   // X size of the error marks for the histogram drawings
         fHistMinimumZero: false,   // when true, BAR and LEGO drawing using base = 0
         fPaintTextFormat : "g",
         fTimeOffset : 788918400 // UTC time at 01/01/95
      };

   /** Generate mask for given bit
    *
    * @param {number} n bit number
    * @returns {Number} produced make
    * @private */
   JSROOT.BIT = function(n) { return 1 << (n); }

   /** TH1 status bits
    * @private */
   JSROOT.TH1StatusBits = {
         kNoStats       : JSROOT.BIT(9),  // don't draw stats box
         kUserContour   : JSROOT.BIT(10), // user specified contour levels
         kCanRebin      : JSROOT.BIT(11), // can rebin axis
         kLogX          : JSROOT.BIT(15), // X-axis in log scale
         kIsZoomed      : JSROOT.BIT(16), // bit set when zooming on Y axis
         kNoTitle       : JSROOT.BIT(17), // don't draw the histogram title
         kIsAverage     : JSROOT.BIT(18)  // Bin contents are average (used by Add)
   };

   /** Wrapper for console.log, let redirect output to specified div element
    * @private */
   JSROOT.console = function(value, divid) {
      if ((typeof divid == 'string') && document.getElementById(divid))
         document.getElementById(divid).innerHTML = value;
      else
      if ((typeof console != 'undefined') && (typeof console.log == 'function'))
         console.log(value);
   }

   /** @summary Wrapper for alert, throws Error in Node.js
    * @private */
   JSROOT.alert = function(msg) {
      if (this.nodeis) throw new Error(msg);
      if (typeof alert === 'function') alert(msg);
      else JSROOT.console('ALERT: ' + msg);
   }

   /**
    * @summary Seed simple random generator
    *
    * @private
    * @param {number} i seed value
    */
   JSROOT.seed = function(i) {
      i = Math.abs(i);
      if (i > 1e8) i = Math.abs(1e8 * Math.sin(i)); else
      if (i < 1) i*=1e8;
      this.m_w = Math.round(i);
      this.m_z = 987654321;
   }

   /**
    * @summary Simple random generator
    *
    * @desc Works like Math.random(), but with configurable seed - see {@link JSROOT.seed}
    * @private
    * @returns {number} random value between 0 (inclusive) and 1.0 (exclusive)
    */
   JSROOT.random = function() {
      if (this.m_z===undefined) return Math.random();
      this.m_z = (36969 * (this.m_z & 65535) + (this.m_z >> 16)) & 0xffffffff;
      this.m_w = (18000 * (this.m_w & 65535) + (this.m_w >> 16)) & 0xffffffff;
      var result = ((this.m_z << 16) + this.m_w) & 0xffffffff;
      result /= 4294967296;
      return result + 0.5;
   }

   /** @summary Should be used to reintroduce objects references, produced by TBufferJSON.
    *
    * @desc Replace all references inside object, object should not be null
    * Idea of the code taken from JSON-R code, found on
    * https://github.com/graniteds/jsonr
    * Only unref part was used, arrays are not accounted as objects
    * @param {object} obj  object where references will be replaced
    * @returns {object} same object with replaced references
    * @private */
   JSROOT.JSONR_unref = function(obj) {

      var map = [], newfmt = undefined;

      function unref_value(value) {
         if ((value===null) || (value===undefined)) return;

         if (typeof value === 'string') {
            if (newfmt || (value.length < 6) || (value.indexOf("$ref:") !== 0)) return;
            var ref = parseInt(value.substr(5));
            if (isNaN(ref) || (ref < 0) || (ref >= map.length)) return;
            newfmt = false;
            return map[ref];
         }

         if (typeof value !== 'object') return;

         var i, k, res, proto = Object.prototype.toString.apply(value);

         // scan array - it can contain other objects
         if ((proto.indexOf('[object')==0) && (proto.indexOf('Array]')>0)) {
             for (i = 0; i < value.length; ++i) {
                res = unref_value(value[i]);
                if (res!==undefined) value[i] = res;
             }
             return;
         }

         var ks = Object.keys(value), len = ks.length;

         if ((newfmt!==false) && (len===1) && (ks[0]==='$ref')) {
            var ref = parseInt(value['$ref']);
            if (isNaN(ref) || (ref < 0) || (ref >= map.length)) return;
            newfmt = true;
            return map[ref];
         }

         if ((newfmt!==false) && (len>1) && (ks[0]==='$arr') && (ks[1]==='len')) {
            // this is ROOT-coded array
            var arr = null, dflt = (value.$arr==="Bool") ? false : 0;
            switch (value.$arr) {
               case "Int8" : arr = new Int8Array(value.len); break;
               case "Uint8" : arr = new Uint8Array(value.len); break;
               case "Int16" : arr = new Int16Array(value.len); break;
               case "Uint16" : arr = new Uint16Array(value.len); break;
               case "Int32" : arr = new Int32Array(value.len); break;
               case "Uint32" : arr = new Uint32Array(value.len); break;
               case "Float32" : arr = new Float32Array(value.len); break;
               case "Int64" :
               case "Uint64" :
               case "Float64" : arr = new Float64Array(value.len); break;
               default : arr = new Array(value.len); break;
            }
            for (var k=0;k<value.len;++k) arr[k] = dflt;

            if (value.b !== undefined) {
               // base64 coding
               var buf = atob(value.b);

               if (arr.buffer) {
                  var dv = new DataView(arr.buffer, value.o || 0),
                      len = Math.min(buf.length, dv.byteLength);
                  for (var k=0; k<len; ++k)
                     dv.setUint8(k, buf.charCodeAt(k));
               } else {
                  throw new Error('base64 coding supported only for native arrays with binary data');
               }
            } else {
               // compressed coding
               var nkey = 2, p = 0;
               while (nkey<len) {
                  if (ks[nkey][0]=="p") p = value[ks[nkey++]]; // position
                  if (ks[nkey][0]!=='v') throw new Error('Unexpected member ' + ks[nkey] + ' in array decoding');
                  var v = value[ks[nkey++]]; // value
                  if (typeof v === 'object') {
                     for (var k=0;k<v.length;++k) arr[p++] = v[k];
                  } else {
                     arr[p++] = v;
                     if ((nkey<len) && (ks[nkey][0]=='n')) {
                        var cnt = value[ks[nkey++]]; // counter
                        while (--cnt) arr[p++] = v;
                     }
                  }
               }
            }

            return arr;
         }

         if ((newfmt!==false) && (len===3) && (ks[0]==='$pair') && (ks[1]==='first') && (ks[2]==='second')) {
            newfmt = true;
            var f1 = unref_value(value.first),
                s1 = unref_value(value.second);
            if (f1!==undefined) value.first = f1;
            if (s1!==undefined) value.second = s1;
            value._typename = value['$pair'];
            delete value['$pair'];
            return; // pair object is not counted in the objects map
         }

         // debug code, can be commented out later
         if (map.indexOf(value) >= 0) {
            JSROOT.console('should never happen - object already in the map');
            return;
         }

         // add object to object map
         map.push(value);

         // add methods to all objects, where _typename is specified
         if ('_typename' in value) JSROOT.addMethods(value);

         for (k = 0; k < len; ++k) {
            i = ks[k];
            res = unref_value(value[i]);
            if (res!==undefined) value[i] = res;
         }
      }

      unref_value(obj);

      return obj;
   }

   JSROOT.debug = 0;

   /** @summary Just copies (not clone) all fields from source to the target object
    * @desc This is simple replacement of jQuery.extend method
    * @private */
   JSROOT.extend = function(tgt, src) {
      if ((src === null) || (typeof src !== 'object')) return tgt;
      if ((tgt === null) || (typeof tgt !== 'object')) tgt = {};

      for (var k in src)
         tgt[k] = src[k];

      return tgt;
   }

   /** @summary Make deep clone of the object, including all sub-objects
    * @private */
   JSROOT.clone = function(src, map, nofunc) {
      if (src === null) return null;

      if (!map) {
         map = { obj:[], clones:[], nofunc: nofunc };
      } else {
         var i = map.obj.indexOf(src);
         if (i>=0) return map.clones[i];
      }

      var proto = Object.prototype.toString.apply(src);

      // process normal array
      if (proto === '[object Array]') {
         var tgt = [];
         map.obj.push(src);
         map.clones.push(tgt);
         for (var i = 0; i < src.length; ++i)
            if (typeof src[i] === 'object')
               tgt.push(JSROOT.clone(src[i], map));
            else
               tgt.push(src[i]);

         return tgt;
      }

      // process typed array
      if ((proto.indexOf('[object ') == 0) && (proto.indexOf('Array]') == proto.length-6)) {
         var tgt = [];
         map.obj.push(src);
         map.clones.push(tgt);
         for (var i = 0; i < src.length; ++i)
            tgt.push(src[i]);

         return tgt;
      }

      var tgt = {};
      map.obj.push(src);
      map.clones.push(tgt);

      for (var k in src) {
         if (typeof src[k] === 'object')
            tgt[k] = JSROOT.clone(src[k], map);
         else
         if (!map.nofunc || (typeof src[k]!=='function'))
            tgt[k] = src[k];
      }

      return tgt;
   }

   /**
    * @summary Clear all functions from the contained objects
    *
    * Only such objects can be cloned when transfer to Worker or converted into JSON
    * @param {object} src  object where functions will be removed
    * @returns {object} same object after all functions are removed
    * @private
    */
   JSROOT.clear_func = function(src, map) {
      if (src === null) return src;

      var proto = Object.prototype.toString.apply(src);

      if (proto === '[object Array]') {
         for (var n=0;n<src.length;n++)
            if (typeof src[n] === 'object')
               JSROOT.clear_func(src[n], map);
         return src;
      }

      if ((proto.indexOf('[object ') == 0) && (proto.indexOf('Array]') == proto.length-6)) return src;

      if (!map) map = [];
      var nomap = (map.length == 0);
      if ('__clean_func__' in src) return src;

      map.push(src);
      src['__clean_func__'] = true;

      for (var k in src) {
         if (typeof src[k] === 'object')
            JSROOT.clear_func(src[k], map);
         else
         if (typeof src[k] === 'function') delete src[k];
      }

      if (nomap)
         for (var n=0;n<map.length;++n)
            delete map[n]['__clean_func__'];

      return src;
   }

   /**
    * @summary Parse JSON code produced with TBufferJSON.
    *
    * @param {string} json string to parse
    * @return {object|null} returns parsed object
    */
   JSROOT.parse = function(json) {
      if (!json) return null;
      var obj = JSON.parse(json);
      return obj ? this.JSONR_unref(obj) : obj;
   }

   /**
    * @summary Parse multi.json request results
    * @desc Method should be used to parse JSON code, produced by multi.json request of THttpServer
    *
    * @param {string} json string to parse
    * @return {Array|null} returns array of parsed elements
    */
   JSROOT.parse_multi = function(json) {
      if (!json) return null;
      var arr = JSON.parse(json);
      if (arr && arr.length)
         for (var i=0;i<arr.length;++i)
            arr[i] = this.JSONR_unref(arr[i]);
      return arr;
   }

   /**
    * @summary Method converts JavaScript object into ROOT-like JSON
    *
    * @desc Produced JSON can be used in JSROOT.parse() again
    * When performed properly, JSON can be used in TBufferJSON to read data back with C++
    */
   JSROOT.toJSON = function(obj) {
      if (!obj || typeof obj !== 'object') return "";

      var map = []; // map of stored objects

      function copy_value(value) {
         if (typeof value === "function") return undefined;

         if ((value===undefined) || (value===null) || (typeof value !== 'object')) return value;

         var proto = Object.prototype.toString.apply(value);

         // typed array need to be converted into normal array, otherwise looks strange
         if ((proto.indexOf('[object ') == 0) && (proto.indexOf('Array]') == proto.length-6)) {
            var arr = new Array(value.length)
            for (var i = 0; i < value.length; ++i)
               arr[i] = copy_value(value[i]);
            return arr;
         }

         // this is how reference is code
         var refid = map.indexOf(value);
         if (refid >= 0) return { $ref: refid };

         var ks = Object.keys(value), len = ks.length, tgt = {};

         if ((len == 3) && (ks[0]==='$pair') && (ks[1]==='first') && (ks[2]==='second')) {
            // special handling of pair objects which does not included into objects map
            tgt.$pair = value.$pair;
            tgt.first = copy_value(value.first);
            tgt.second = copy_value(value.second);
            return tgt;
         }

         map.push(value);

         for (var k = 0; k < len; ++k) {
            var name = ks[k];
            tgt[name] = copy_value(value[name]);
         }

         return tgt;
      }

      var tgt = copy_value(obj);

      return JSON.stringify(tgt);
   }

   /**
    * @summary Analyzes document.URL and extracts options after '?' mark
    *
    * @desc Following options supported ?opt1&opt2=3
    * In case of opt1 empty string will be returned, in case of opt2 '3'
    * If option not found, null is returned (or default value value is provided)
    *
    * @param {string} opt option to search
    * @param {string} full URL with options, document.URL will be used when not specified
    * @returns {string|null} found value
    * @private
    */
   JSROOT.GetUrlOption = function(opt, url, dflt) {

      if (dflt === undefined) dflt = null;
      if ((opt===null) || (typeof opt != 'string') || (opt.length==0)) return dflt;

      if (!url) {
         if (JSROOT.gStyle.IgnoreUrlOptions || (typeof document === 'undefined')) return dflt;
         url = document.URL;
      }

      var pos = url.indexOf("?"), nquotes;
      if (pos<0) return dflt;
      url = decodeURI(url.slice(pos+1));
      pos = url.lastIndexOf("#");
      if (pos>=0) url = url.substr(0,pos);

      while (url.length>0) {

         if (url==opt) return "";

         // try to correctly handle quotes in the URL
         pos = 0; nquotes = 0;
         while ((pos < url.length) && ((nquotes!==0) || (url[pos]!=="&"))) {
            switch (url[pos]) {
               case "'": if (nquotes>=0) nquotes = (nquotes+1)%2; break;
               case '"': if (nquotes<=0) nquotes = (nquotes-1)%2; break;
            }
            pos++;
         }

         if (url.indexOf(opt) == 0) {
            if (url[opt.length]=="&") return "";

            if (url[opt.length]==="=") {
               url = url.slice(opt.length+1, pos);
               if (((url[0]==="'") || (url[0]==='"')) && (url[0]===url[url.length-1])) url = url.substr(1, url.length-2);
               return url;
            }
         }

         url = url.substr(pos+1);
      }
      return dflt;
   }

   /**
    * @summary Parse string value as array.
    *
    * @desc It could be just simple string:  "value" or
    * array with or without string quotes:  [element], ['elem1',elem2]
    *
    * @private
    */
   JSROOT.ParseAsArray = function(val) {

      var res = [];

      if (typeof val != 'string') return res;

      val = val.trim();
      if (val=="") return res;

      // return as array with single element
      if ((val.length<2) || (val[0]!='[') || (val[val.length-1]!=']')) {
         res.push(val); return res;
      }

      // try to split ourself, checking quotes and brackets
      var nbr = 0, nquotes = 0, ndouble = 0, last = 1;

      for (var indx = 1; indx < val.length; ++indx) {
         if (nquotes > 0) {
            if (val[indx]==="'") nquotes--;
            continue;
         }
         if (ndouble > 0) {
            if (val[indx]==='"') ndouble--;
            continue;
         }
         switch (val[indx]) {
            case "'": nquotes++; break;
            case '"': ndouble++; break;
            case "[": nbr++; break;
            case "]": if (indx < val.length - 1) { nbr--; break; }
            case ",":
               if (nbr === 0) {
                  var sub =  val.substring(last, indx).trim();
                  if ((sub.length>1) && (sub[0]==sub[sub.length-1]) && ((sub[0]=='"') || (sub[0]=="'")))
                     sub = sub.substr(1, sub.length-2);
                  res.push(sub);
                  last = indx+1;
               }
               break;
         }
      }

      if (res.length === 0)
         res.push(val.substr(1, val.length-2).trim());

      return res;
   }

   /**
    * @summary Special handling of URL options to produce array.
    *
    * @desc If normal option is specified ...?opt=abc, than array with single element will be created
    * one could specify normal JSON array ...?opts=['item1','item2']
    * but also one could skip quotes ...?opts=[item1,item2]
    * @private
    */
   JSROOT.GetUrlOptionAsArray = function(opt, url) {

      var res = [];

      while (opt.length>0) {
         var separ = opt.indexOf(";");
         var part = (separ>0) ? opt.substr(0, separ) : opt;

         if (separ>0) opt = opt.substr(separ+1); else opt = "";

         var canarray = true;
         if (part[0]=='#') { part = part.substr(1); canarray = false; }

         var val = this.GetUrlOption(part, url, null);

         if (canarray) res = res.concat(JSROOT.ParseAsArray(val));
                  else if (val!==null) res.push(val);
      }
      return res;
   }

   /**
    * @summary Find function with given name.
    *
    * @desc Function name may include several namespaces like 'JSROOT.Painter.drawFrame'
    *
    * @private
    */
   JSROOT.findFunction = function(name) {
      if (typeof name === 'function') return name;
      if (typeof name !== 'string') return null;
      var names = name.split('.'), elem = null;
      if (typeof window === 'object') elem = window;
      if (names[0]==='JSROOT') { elem = this; names.shift(); }

      for (var n=0;elem && (n<names.length);++n)
         elem = elem[names[n]];

      return (typeof elem == 'function') ? elem : null;
   }

   /**
    * @summary Generic method to invoke callback function.
    *
    * @param {object|function} func either normal function or container like
    * { obj: object_pointer, func: name of method to call }
    * @param arg1 first optional argument of callback
    * @param arg2 second optional argument of callback
    *
    * @private
    */
   JSROOT.CallBack = function(func, arg1, arg2) {

      if (typeof func == 'string') func = JSROOT.findFunction(func);

      if (!func) return;

      if (typeof func == 'function') return func(arg1,arg2);

      if (typeof func != 'object') return;

      if (('obj' in func) && ('func' in func) &&
         (typeof func.obj == 'object') && (typeof func.func == 'string') &&
         (typeof func.obj[func.func] == 'function')) {
             return func.obj[func.func](arg1, arg2);
      }
   }

   /**
    * @summary Create asynchronous XMLHttpRequest object.
    *
    * @desc One should call req.send() to submit request
    * kind of the request can be:
    *
    *    - "bin" - abstract binary data, result as string
    *    - "buf" - abstract binary data, result as ArrayBuffer (default)
    *    - "text" - returns req.responseText
    *    - "object" - returns JSROOT.parse(req.responseText)
    *    - "multi" - returns correctly parsed multi.json request
    *    - "xml" - returns req.responseXML
    *    - "head" - returns request itself, uses "HEAD" method
    *
    * Result will be returned to the callback function.
    * Request will be set as *this* pointer in the callback.
    * If failed, request returns null
    *
    * @param {string} url - URL for the request
    * @param {string} kind - kind of requested data
    * @param {function} user_call_back - called when request is completed
    * @returns {object} XMLHttpRequest object
    *
    * @example
    * JSROOT.NewHttpRequest("https://root.cern/js/files/thstack.json.gz", "object",
    *                       function(res) {
    *     if (res) console.log('Retrieve object', res._typename);
    *         else console.error('Fail to get object');
    * }).send();
    */

   JSROOT.NewHttpRequest = function(url, kind, user_call_back) {

      var xhr = JSROOT.nodejs ? new (require("xhr2"))() : new XMLHttpRequest();

      xhr.http_callback = (typeof user_call_back == 'function') ? user_call_back.bind(xhr) : function() {};

      if (!kind) kind = "buf";

      var method = "GET", async = true, p = kind.indexOf(";sync");
      if (p>0) { kind = kind.substr(0,p); async = false; }
      if (kind === "head") method = "HEAD"; else
      if ((kind === "post") || (kind === "multi") || (kind === "posttext")) method = "POST";

      xhr.kind = kind;

      if (JSROOT.wrong_http_response_handling && (method == "GET") && (typeof xhr.addEventListener === 'function'))
         xhr.addEventListener("progress", function(oEvent) {
            if (oEvent.lengthComputable && this.expected_size && (oEvent.loaded > this.expected_size)) {
               this.did_abort = true;
               this.abort();
               console.warn('Server sends more bytes ' + oEvent.loaded + ' than expected ' + this.expected_size + '. Abort I/O operation');
               this.http_callback(null);
            }
         }.bind(xhr));

      xhr.onreadystatechange = function() {

         if (this.did_abort) return;

         if ((this.readyState === 2) && this.expected_size) {
            var len = parseInt(this.getResponseHeader("Content-Length"));
            if (!isNaN(len) && (len>this.expected_size) && !JSROOT.wrong_http_response_handling) {
               this.did_abort = true;
               this.abort();
               console.warn('Server response size ' + len + ' larger than expected ' + this.expected_size + '. Abort I/O operation');
               return this.http_callback(null);
            }
         }

         if (this.readyState != 4) return;

         if ((this.status != 200) && (this.status != 206) && !JSROOT.browser.qt5 &&
             // in these special cases browsers not always set status
             !((this.status == 0) && ((url.indexOf("file://")==0) || (url.indexOf("blob:")==0)))) {
            return this.http_callback(null);
         }

         if (this.nodejs_checkzip && (this.getResponseHeader("content-encoding") == "gzip")) {
            // special handling of gzipped JSON objects in Node.js
            var zlib = require('zlib'),
                str = zlib.unzipSync(Buffer.from(this.response));
            return this.http_callback(JSROOT.parse(str));
         }

         switch(this.kind) {
            case "xml": return this.http_callback(this.responseXML);
            case "posttext":
            case "text": return this.http_callback(this.responseText);
            case "object": return this.http_callback(JSROOT.parse(this.responseText));
            case "multi": return this.http_callback(JSROOT.parse_multi(this.responseText));
            case "head": return this.http_callback(this);
         }

         // if no response type is supported, return as text (most probably, will fail)
         if (this.responseType === undefined)
            return this.http_callback(this.responseText);

         if ((this.kind == "bin") && ('byteLength' in this.response)) {
            // if string representation in requested - provide it

            var filecontent = "", u8Arr = new Uint8Array(this.response);
            for (var i = 0; i < u8Arr.length; ++i)
               filecontent += String.fromCharCode(u8Arr[i]);

            return this.http_callback(filecontent);
         }

         this.http_callback(this.response);
      }

      xhr.open(method, url, async);

      if ((kind == "bin") || (kind == "buf")) xhr.responseType = 'arraybuffer';

      if (JSROOT.nodejs && (method == "GET") && (kind === "object") && (url.indexOf('.json.gz')>0)) {
         xhr.nodejs_checkzip = true;
         xhr.responseType = 'arraybuffer';
      }

      return xhr;
   }

   /**
    * @summary Dynamic script loader
    *
    * @desc One could specify list of scripts or style files, separated by semicolon ';'
    * one can prepend file name with '$$$' - than file will be loaded from JSROOT location
    * This location can be set by JSROOT.source_dir or it will be detected automatically
    * by the position of JSRootCore.js file, which must be loaded by normal methods:
    * <script type="text/javascript" src="scripts/JSRootCore.js"></script>
    * When all scripts are loaded, callback function will be called
    *
    * @private
    */
   JSROOT.loadScript = function(urllist, callback, debugout, from_previous) {

      delete JSROOT.complete_script_load;

      if (from_previous) {
         if (debugout)
            document.getElementById(debugout).innerHTML = "";
         else
            JSROOT.progress();

         if (!urllist) return JSROOT.CallBack(callback);
      }

      if (!urllist) return JSROOT.CallBack(callback);

      var filename = urllist, separ = filename.indexOf(";"),
          isrootjs = false, isbower = false;

      if (separ>0) {
         filename = filename.substr(0, separ);
         urllist = urllist.substr(separ+1);
      } else {
         urllist = "";
      }

      var completeLoad = JSROOT.loadScript.bind(JSROOT, urllist, callback, debugout, true);

      if (filename.indexOf('&&&scripts/')===0) {
         isrootjs = true;
         filename = filename.slice(3);
         if (JSROOT.use_full_libs) filename = "libs/" + filename.slice(8, filename.length-7) + ".js";
      } else if (filename.indexOf("$$$")===0) {
         isrootjs = true;
         filename = filename.slice(3);
         if ((filename.indexOf("style/")==0) && JSROOT.source_min &&
             (filename.lastIndexOf('.css')==filename.length-4) &&
             (filename.indexOf('.min.css')<0))
            filename = filename.slice(0, filename.length-4) + '.min.css';
      } else if (filename.indexOf("###")===0) {
         isbower = true;
         filename = filename.slice(3);
      }

      if (JSROOT.nodejs) {
         if ((filename.indexOf("scripts/")===0) && (filename.indexOf(".js")>0)) {
            console.log('load', filename);
            require("." + filename.substr(7));
         }
         return completeLoad();
      }

      var isstyle = filename.indexOf('.css') > 0;

      if (isstyle) {
         var styles = document.getElementsByTagName('link');
         for (var n = 0; n < styles.length; ++n) {
            if (!styles[n].href || (styles[n].type !== 'text/css') || (styles[n].rel !== 'stylesheet')) continue;

            if (styles[n].href.indexOf(filename)>=0) return completeLoad();
         }

      } else {
         var scripts = document.getElementsByTagName('script');

         for (var n = 0; n < scripts.length; ++n) {
            var src = scripts[n].src;
            if (!src) continue;

            if ((src.indexOf(filename)>=0) && (src.indexOf("load=")<0))
               // avoid wrong decision when script name is specified as more argument
               return completeLoad();
         }
      }

      if (isrootjs && JSROOT.source_dir) filename = JSROOT.source_dir + filename; else
      if (isbower && (JSROOT.bower_dir!==null)) filename = JSROOT.bower_dir + filename;

      var element = null;

      if (debugout)
         document.getElementById(debugout).innerHTML = "loading " + filename + " ...";
      else
         JSROOT.progress("loading " + filename + " ...");

      if (JSROOT.nocache && isrootjs && (filename.indexOf("?")<0))
         filename += "?stamp=" + JSROOT.nocache;

      if (isstyle) {
         element = document.createElement("link");
         element.setAttribute("rel", "stylesheet");
         element.setAttribute("type", "text/css");
         element.setAttribute("href", filename);
      } else {
         element = document.createElement("script");
         element.setAttribute('type', "text/javascript");
         element.setAttribute('src', filename);
      }

      JSROOT.complete_script_load = completeLoad;

      if (element.readyState) { // Internet Explorer specific
         element.onreadystatechange = function() {
            if (element.readyState == "loaded" || element.readyState == "complete") {
               element.onreadystatechange = null;
               if (JSROOT.complete_script_load) JSROOT.complete_script_load();
            }
         }
      } else { // Other browsers
         element.onload = function() {
            element.onload = null;
            if (JSROOT.complete_script_load) JSROOT.complete_script_load();
         }
      }

      document.getElementsByTagName("head")[0].appendChild(element);
   }

   /** @summary Load JSROOT functionality.
    *
    * @desc As first argument, required components should be specifed:
    *
    *    - 'io'     TFile functionality
    *    - 'tree'   TTree support
    *    - '2d'     basic 2d graphic (TCanvas/TPad/TFrame)
    *    - '3d'     basic 3d graphic (three.js)
    *    - 'hist'   histograms 2d graphic
    *    - 'hist3d' histograms 3d graphic
    *    - 'more2d' extra 2d graphic (TGraph, TF1)
    *    - 'v7'     ROOT v7 graphics
    *    - 'v7hist' ROOT v7 histograms
    *    - 'v7more' ROOT v7 special classes
    *    - 'math'   some methods from TMath class
    *    - 'jq'     jQuery and jQuery-ui
    *    - 'hierarchy' hierarchy browser
    *    - 'jq2d'   jQuery-dependent part of hierarchy
    *    - 'openui5' OpenUI5 and related functionality
    *    - 'geom'    TGeo support
    *    - 'simple'  for basic user interface
    *    - 'load:<path/script.js>' list of user-specific scripts at the end of kind string
    *
    * One could combine several compopnents, separating them by semicolon.
    * Depending of available components, either require.js or plain script loading will be used
    *
    * @param {string} kind - modules to load
    * @param {function} callback - called when all specified modules are loaded
    *
    * @example
    * JSROOT.AssertPrerequisites("io;tree", function() {
    *    var selector = new JSROOT.TSelector;
    * });
    */

   JSROOT.AssertPrerequisites = function(kind, callback, debugout) {
      // one could specify kind of requirements

      var jsroot = JSROOT;

      if (jsroot.doing_assert === undefined) jsroot.doing_assert = [];
      if (jsroot.ready_modules === undefined) jsroot.ready_modules = [];

      if (!kind || (typeof kind !== 'string'))
         return jsroot.CallBack(callback);

      if (kind === '__next__') {
         if (jsroot.doing_assert.length==0) return;
         var req = jsroot.doing_assert[0];
         if (req.running) return;
         kind = req._kind;
         callback = req._callback;
         debugout = req._debug;
      } else {
         jsroot.doing_assert.push({_kind:kind, _callback:callback, _debug: debugout});
         if (jsroot.doing_assert.length > 1) return;
      }

      function normal_callback() {
         var req = jsroot.doing_assert.shift();
         for (var n=0;n<req.modules.length;++n)
            jsroot.ready_modules.push(req.modules[n]);
         jsroot.CallBack(req._callback);
         jsroot.AssertPrerequisites('__next__');
      }

      jsroot.doing_assert[0].running = true;

      if (kind[kind.length-1]!=";") kind+=";";

      var ext = jsroot.source_min ? ".min" : "",
          need_jquery = false,
          use_require = (typeof define === "function") && define.amd,
          use_bower = jsroot.bower_dir!==null,
          mainfiles = "",
          extrafiles = "", // scripts for direct loading
          modules = [],  // modules used for require.js
          load_callback = normal_callback;

      if ((kind.indexOf('io;')>=0) || (kind.indexOf('tree;')>=0))
         if (jsroot.sources.indexOf("io")<0) {
            mainfiles += "&&&scripts/rawinflate.min.js;" +
                         "$$$scripts/JSRootIOEvolution" + ext + ".js;";
            modules.push('rawinflate', 'JSRootIOEvolution');
         }

      if ((kind.indexOf('math;')>=0) || (kind.indexOf('tree;')>=0) || (kind.indexOf('more2d;')>=0))
         if (jsroot.sources.indexOf("math")<0) {
            mainfiles += '$$$scripts/JSRootMath' + ext + ".js;";
            modules.push('JSRootMath');
         }

      if (kind.indexOf('tree;')>=0)
         if (jsroot.sources.indexOf("tree")<0) {
            mainfiles += "$$$scripts/JSRootTree" + ext + ".js;";
            modules.push('JSRootTree');
         }

      if ((kind.indexOf('2d;')>=0) || (kind.indexOf('v6;')>=0) || (kind.indexOf('v7;')>=0) ||
          (kind.indexOf("3d;")>=0) || (kind.indexOf("geom;")>=0)) {
          if (!use_require && (typeof d3 != 'object') && (jsroot._test_d3_ === undefined)) {
             mainfiles += use_bower ? '###d3/d3.min.js;' : '&&&scripts/d3.min.js;';
             jsroot._test_d3_ = null;
         }
         if (jsroot.sources.indexOf("2d") < 0) {
            modules.push('JSRootPainter');
            mainfiles += '$$$scripts/JSRootPainter' + ext + ".js;";
            extrafiles += '$$$style/JSRootPainter' + ext + '.css;';
         }
         if ((jsroot.sources.indexOf("v6") < 0) && (kind.indexOf('v7;') < 0)) {
            mainfiles += '$$$scripts/JSRootPainter.v6' + ext + ".js;";
            modules.push('JSRootPainter.v6');
         }
      }

      if (kind.indexOf('jq;')>=0) need_jquery = true;

      if (((kind.indexOf('hist;')>=0) || (kind.indexOf('hist3d;')>=0)) && (jsroot.sources.indexOf("hist")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.hist' + ext + ".js;";
         modules.push('JSRootPainter.hist');
      }

      if ((kind.indexOf('v6;')>=0) && (jsroot.sources.indexOf("v6")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.v6' + ext + ".js;";
         modules.push('JSRootPainter.v6');
      }

      if ((kind.indexOf('v7;')>=0) && (jsroot.sources.indexOf("v7")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.v7' + ext + ".js;";
         modules.push('JSRootPainter.v7');
      }

      if ((kind.indexOf('v7hist;')>=0) && (jsroot.sources.indexOf("v7hist")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.v7hist' + ext + ".js;";
         modules.push('JSRootPainter.v7hist');
      }

      if ((kind.indexOf('v7more;')>=0) && (jsroot.sources.indexOf("v7more")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.v7more' + ext + ".js;";
         modules.push('JSRootPainter.v7more');
      }

      if ((kind.indexOf('more2d;')>=0) && (jsroot.sources.indexOf("more2d")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.more' + ext + ".js;";
         modules.push('JSRootPainter.more');
      }

      if (((kind.indexOf('hierarchy;')>=0) || (kind.indexOf('jq2d;')>=0)) && (jsroot.sources.indexOf("hierarchy")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.hierarchy' + ext + ".js;";
         modules.push('JSRootPainter.hierarchy');
      }

      if ((kind.indexOf('jq2d;')>=0) && (jsroot.sources.indexOf("jq2d")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.jquery' + ext + ".js;";
         modules.push('JSRootPainter.jquery');
         need_jquery = true;
      }

      if ((kind.indexOf('openui5;')>=0) && (jsroot.sources.indexOf("openui5")<0)) {
         mainfiles += '$$$scripts/JSRoot.openui5' + ext + ".js;";
         modules.push('JSRoot.openui5');
         need_jquery = true;
      }

      if (((kind.indexOf("3d;")>=0) || (kind.indexOf("geom;")>=0)) && (jsroot.sources.indexOf("3d")<0)) {
         mainfiles += (use_bower ? "###threejs/build/" : "&&&scripts/") + "three.min.js;" +
                       "&&&scripts/three.extra.min.js;";
         modules.push("threejs", "threejs_all");
         mainfiles += "$$$scripts/JSRoot3DPainter" + ext + ".js;";
         modules.push('JSRoot3DPainter');
      }

      if ((kind.indexOf('hist3d;')>=0) && (jsroot.sources.indexOf("hist3d")<0)) {
         mainfiles += '$$$scripts/JSRootPainter.hist3d' + ext + ".js;";
         modules.push('JSRootPainter.hist3d');
      }

      if (kind.indexOf("datgui;")>=0) {
         if (!JSROOT.nodejs && (typeof window !='undefined'))
            mainfiles += (use_bower ? "###dat.gui" : "&&&scripts") + "/dat.gui.min.js;";
         // console.log('extra loading module dat.gui');
         modules.push('dat.gui');
      }

      if ((kind.indexOf("geom;")>=0) && (jsroot.sources.indexOf("geom")<0)) {
         mainfiles += "$$$scripts/ThreeCSG" + ext + ".js;" +
                      "$$$scripts/JSRootGeoBase" + ext + ".js;" +
                      "$$$scripts/JSRootGeoPainter" + ext + ".js;";
         extrafiles += "$$$style/JSRootGeoPainter" + ext + ".css;";
         modules.push('ThreeCSG', 'JSRootGeoBase', 'JSRootGeoPainter');
      }

      if (kind.indexOf("mathjax;")>=0) {

         if (typeof MathJax == 'undefined') {
            mainfiles += (use_bower ? "###MathJax" : "https://root.cern/js/mathjax/latest") +
                          "/MathJax.js?config=TeX-AMS-MML_SVG&delayStartupUntil=configured;";
            modules.push('MathJax');

            load_callback = function() {
               MathJax.Hub.Config({ jax: ["input/TeX", "output/SVG"],
                  TeX: { extensions: ["color.js"] },
                  SVG: { mtextFontInherit: true, minScaleAdjust: 100, matchFontHeight: true, useFontCache: false } });

               MathJax.Hub.Register.StartupHook("SVG Jax Ready",function () {
                  var VARIANT = MathJax.OutputJax.SVG.FONTDATA.VARIANT;
                  VARIANT["normal"].fonts.unshift("MathJax_SansSerif");
                  VARIANT["bold"].fonts.unshift("MathJax_SansSerif-bold");
                  VARIANT["italic"].fonts.unshift("MathJax_SansSerif");
                  VARIANT["-tex-mathit"].fonts.unshift("MathJax_SansSerif");
               });

               MathJax.Hub.Configured();

               normal_callback();
            }
         }
      }

      if (kind.indexOf("simple;")>=0) {
         need_jquery = true;
      }

      if (need_jquery && !jsroot.load_jquery) {
         var has_jq = (typeof jQuery != 'undefined'), lst_jq = "";

         if (has_jq)
            jsroot.console('Reuse existing jQuery ' + jQuery.fn.jquery + ", required 3.1.1", debugout);
         else
            lst_jq += (use_bower ? "###jquery/dist" : "&&&scripts") + "/jquery.min.js;";
         if (has_jq && typeof $.ui != 'undefined') {
            jsroot.console('Reuse existing jQuery-ui ' + $.ui.version + ", required 1.12.1", debugout);
         } else {
            lst_jq += (use_bower ? "###jquery-ui" : "&&&scripts") + '/jquery-ui.min.js;';
            extrafiles += '$$$style/jquery-ui' + ext + '.css;';
         }

         if (jsroot.touches) {
            lst_jq += use_bower ? '###jqueryui-touch-punch/jquery.ui.touch-punch.min.js;' : '$$$scripts/touch-punch.min.js;';
            modules.push('jqueryui-touch-punch');
         }

         modules.splice(0, 0, 'jquery', 'jquery-ui', 'jqueryui-mousewheel');
         mainfiles = lst_jq + mainfiles;

         jsroot.load_jquery = true;
      }

      var pos = kind.indexOf("user:");
      if (pos<0) pos = kind.indexOf("load:");
      if (pos>=0) extrafiles += kind.slice(pos+5);

      // check if modules already loaded
      for (var n=modules.length-1;n>=0;--n)
         if (jsroot.ready_modules.indexOf(modules[n])>=0)
            modules.splice(n,1);

      // no modules means no main files
      if (modules.length===0) mainfiles = "";

      jsroot.doing_assert[0].modules = modules;

      if ((modules.length>0) && (typeof define === "function") && define.amd) {
         jsroot.console("loading " + JSON.stringify(modules) + " with require.js", debugout);
         require(modules, function() {
            jsroot.loadScript(extrafiles, load_callback, debugout);
         });
      } else {
         jsroot.loadScript(mainfiles + extrafiles, load_callback, debugout);
      }
   }

   // function can be used to open ROOT file, I/O functionality will be loaded when missing
   JSROOT.OpenFile = function(filename, callback) {
      JSROOT.AssertPrerequisites("io", function() {
         JSROOT.OpenFile(filename, callback);
      });
   }

   // function can be used to draw supported ROOT classes,
   // required functionality will be loaded automatically
   // if painter pointer required, one should load '2d' functionality itself
   // or use callback function which provides painter pointer as first argument
   // defined in JSRootPainter.js
   JSROOT.draw = function(divid, obj, opt, callback) {
      JSROOT.AssertPrerequisites("2d", function() {
         JSROOT.draw(divid, obj, opt, callback);
      });
   }

   // redraw object on given element
   // defined in JSRootPainter.js
   JSROOT.redraw = function(divid, obj, opt, callback) {
      JSROOT.AssertPrerequisites("2d", function() {
         JSROOT.redraw(divid, obj, opt, callback);
      });
   }

   // Create SVG, defined in JSRootPainter.js
   JSROOT.MakeSVG = function(args, callback) {
      JSROOT.AssertPrerequisites("2d", function() {
         JSROOT.MakeSVG(args, callback);
      });
   }

   /** @summary Method to build JSROOT GUI with browser
    * @private
    */
   JSROOT.BuildSimpleGUI = function(user_scripts, andThen) {
      if (typeof user_scripts == 'function') {
         andThen = user_scripts;
         user_scripts = null;
      }

      var debugout = null,
          nobrowser = JSROOT.GetUrlOption('nobrowser')!=null,
          requirements = "2d;hierarchy;",
          simplegui = document.getElementById('simpleGUI');

      if (JSROOT.GetUrlOption('libs')!==null) JSROOT.use_full_libs = true;

      if (simplegui) {
         debugout = 'simpleGUI';
         if (JSROOT.GetUrlOption('file') || JSROOT.GetUrlOption('files')) requirements += "io;";
         if (simplegui.getAttribute('nobrowser') && (simplegui.getAttribute('nobrowser')!="false")) nobrowser = true;
      } else if (document.getElementById('onlineGUI')) {
         debugout = 'onlineGUI';
      } else if (document.getElementById('drawGUI')) {
         debugout = 'drawGUI';
         nobrowser = true;
      } else {
         requirements += "io;";
      }

      if (user_scripts == 'check_existing_elements') {
         user_scripts = null;
         if (debugout == null) return;
      }

      if (!nobrowser) requirements += 'jq2d;';

      if (!user_scripts) user_scripts = JSROOT.GetUrlOption("autoload") || JSROOT.GetUrlOption("load");

      if (user_scripts) requirements += "load:" + user_scripts + ";";

      JSROOT.AssertPrerequisites(requirements, function() {
         JSROOT.CallBack(JSROOT.findFunction(nobrowser ? 'JSROOT.BuildNobrowserGUI' : 'JSROOT.BuildGUI'));
         JSROOT.CallBack(andThen);
      }, debugout);
   }

   /** @summary Create some ROOT classes
    *
    * @param {string} typename - ROOT class name
    * @example
    * var obj = JSROOT.Create("TNamed");
    * obj.fName = "name";
    * obj.fTitle = "title";
    */
   JSROOT.Create = function(typename, target) {
      var obj = target || {};

      switch (typename) {
         case 'TObject':
             JSROOT.extend(obj, { fUniqueID: 0, fBits: 0 });
             break;
         case 'TNamed':
            JSROOT.extend(obj, { fUniqueID: 0, fBits: 0, fName: "", fTitle: "" });
            break;
         case 'TList':
         case 'THashList':
            JSROOT.extend(obj, { name: typename, arr : [], opt : [] });
            break;
         case 'TAttAxis':
            JSROOT.extend(obj, { fNdivisions: 510, fAxisColor: 1,
                                 fLabelColor: 1, fLabelFont: 42, fLabelOffset: 0.005, fLabelSize: 0.035, fTickLength: 0.03,
                                 fTitleOffset: 1, fTitleSize: 0.035, fTitleColor: 1, fTitleFont : 42 });
            break;
         case 'TAxis':
            JSROOT.Create("TNamed", obj);
            JSROOT.Create("TAttAxis", obj);
            JSROOT.extend(obj, { fNbins: 1, fXmin: 0, fXmax: 1, fXbins : [], fFirst: 0, fLast: 0,
                                 fBits2: 0, fTimeDisplay: false, fTimeFormat: "", fLabels: null, fModLabs: null });
            break;
         case 'TAttLine':
            JSROOT.extend(obj, { fLineColor: 1, fLineStyle: 1, fLineWidth: 1 });
            break;
         case 'TAttFill':
            JSROOT.extend(obj, { fFillColor: 0, fFillStyle: 0 } );
            break;
         case 'TAttMarker':
            JSROOT.extend(obj, { fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1. });
            break;
         case 'TLine':
            JSROOT.Create("TObject", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.extend(obj, { fX1: 0, fX2: 1, fY1: 0, fY2: 1 });
            break;
         case 'TBox':
            JSROOT.Create("TObject", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttFill", obj);
            JSROOT.extend(obj, { fX1: 0, fX2: 1, fY1: 0, fY2: 1 });
            break;
         case 'TPave':
            JSROOT.Create("TBox", obj);
            JSROOT.extend(obj, { fX1NDC : 0., fY1NDC: 0, fX2NDC: 1, fY2NDC: 1,
                                 fBorderSize: 0, fInit: 1, fShadowColor: 1,
                                 fCornerRadius: 0, fOption: "blNDC", fName: "title" });
            break;
         case 'TAttText':
            JSROOT.extend(obj, { fTextAngle: 0, fTextSize: 0, fTextAlign: 22, fTextColor: 1, fTextFont: 42});
            break;
         case 'TPaveText':
            JSROOT.Create("TPave", obj);
            JSROOT.Create("TAttText", obj);
            JSROOT.extend(obj, { fLabel: "", fLongest: 27, fMargin: 0.05, fLines: JSROOT.Create("TList") });
            break;
         case 'TPaveStats':
            JSROOT.Create("TPaveText", obj);
            JSROOT.extend(obj, { fOptFit: 0, fOptStat: 0, fFitFormat: "", fStatFormat: "", fParent: null });
            break;
         case 'TLegend':
            JSROOT.Create("TPave", obj);
            JSROOT.Create("TAttText", obj);
            JSROOT.extend(obj, { fColumnSeparation: 0, fEntrySeparation: 0.1, fMargin: 0.25, fNColumns: 1, fPrimitives: JSROOT.Create("TList") });
            break;
         case 'TLegendEntry':
            JSROOT.Create("TObject", obj);
            JSROOT.Create("TAttText", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttFill", obj);
            JSROOT.Create("TAttMarker", obj);
            JSROOT.extend(obj, { fLabel: "", fObject: null, fOption: "" });
            break;
         case 'TText':
            JSROOT.Create("TNamed", obj);
            JSROOT.Create("TAttText", obj);
            JSROOT.extend(obj, { fLimitFactorSize: 3, fOriginSize: 0.04 });
            break;
         case 'TLatex':
            JSROOT.Create("TText", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.extend(obj, { fX: 0, fY: 0 });
            break;
         case 'TObjString':
            JSROOT.Create("TObject", obj);
            JSROOT.extend(obj, { fString: "" });
            break;
         case 'TH1':
            JSROOT.Create("TNamed", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttFill", obj);
            JSROOT.Create("TAttMarker", obj);

            JSROOT.extend(obj, {
               fBits: 8,
               fNcells: 0,
               fXaxis: JSROOT.Create("TAxis"),
               fYaxis: JSROOT.Create("TAxis"),
               fZaxis: JSROOT.Create("TAxis"),
               fBarOffset: 0, fBarWidth: 1000, fEntries: 0.,
               fTsumw: 0., fTsumw2: 0., fTsumwx: 0., fTsumwx2: 0.,
               fMaximum: -1111., fMinimum: -1111, fNormFactor: 0., fContour: [],
               fSumw2: [], fOption: "",
               fFunctions: JSROOT.Create("TList"),
               fBufferSize: 0, fBuffer: [], fBinStatErrOpt: 0, fStatOverflows: 2 });
            break;
         case 'TH1I':
         case 'TH1F':
         case 'TH1D':
         case 'TH1S':
         case 'TH1C':
            JSROOT.Create("TH1", obj);
            obj.fArray = [];
            break;
         case 'TH2':
            JSROOT.Create("TH1", obj);
            JSROOT.extend(obj, { fScalefactor: 1., fTsumwy: 0.,  fTsumwy2: 0, fTsumwxy: 0});
            break;
         case 'TH2I':
         case 'TH2F':
         case 'TH2D':
         case 'TH2S':
         case 'TH2C':
            JSROOT.Create("TH2", obj);
            obj.fArray = [];
            break;
         case 'TH3':
            JSROOT.Create("TH1", obj);
            JSROOT.extend(obj, { fTsumwy: 0.,  fTsumwy2: 0, fTsumwz: 0.,  fTsumwz2: 0, fTsumwxy: 0, fTsumwxz: 0, fTsumwyz: 0 });
            break;
         case 'TH3I':
         case 'TH3F':
         case 'TH3D':
         case 'TH3S':
         case 'TH3C':
            JSROOT.Create("TH3", obj);
            obj.fArray = [];
            break;
         case 'THStack':
            JSROOT.Create("TNamed", obj);
            JSROOT.extend(obj, { fHists: JSROOT.Create("TList"), fHistogram: null, fMaximum: -1111, fMinimum: -1111 });
            break;
         case 'TGraph':
            JSROOT.Create("TNamed", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttFill", obj);
            JSROOT.Create("TAttMarker", obj);
            JSROOT.extend(obj, { fFunctions: JSROOT.Create("TList"), fHistogram: null,
                                 fMaxSize: 0, fMaximum: -1111, fMinimum: -1111, fNpoints: 0, fX: [], fY: [] });
            break;
         case 'TGraphAsymmErrors':
            JSROOT.Create("TGraph", obj);
            JSROOT.extend(obj, { fEXlow: [], fEXhigh: [], fEYlow: [], fEYhigh: []});
            break;
         case 'TMultiGraph':
            JSROOT.Create("TNamed", obj);
            JSROOT.extend(obj, { fFunctions: JSROOT.Create("TList"), fGraphs: JSROOT.Create("TList"),
                                 fHistogram: null, fMaximum: -1111, fMinimum: -1111 });
            break;
         case 'TGraphPolargram':
            JSROOT.Create("TNamed", obj);
            JSROOT.Create("TAttText", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.extend(obj, { fRadian: true, fDegree: false, fGrad: false, fPolarLabelColor: 1, fRadialLabelColor: 1,
                                 fAxisAngle: 0, fPolarOffset: 0.04, fPolarTextSize: 0.04, fRadialOffset: 0.025, fRadialTextSize: 0.035,
                                 fRwrmin: 0, fRwrmax: 1, fRwtmin: 0, fRwtmax: 2*Math.PI, fTickpolarSize: 0.02,
                                 fPolarLabelFont: 62, fRadialLabelFont: 62, fCutRadial: 0, fNdivRad: 508, fNdivPol: 508 });
            break;
         case 'TPolyLine':
            JSROOT.Create("TObject", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttFill", obj);
            JSROOT.extend(obj, { fLastPoint: -1, fN: 0, fOption: "", fX: null, fY: null });
            break;
         case 'TGaxis':
            JSROOT.Create("TLine", obj);
            JSROOT.Create("TAttText", obj);
            JSROOT.extend(obj, { fChopt: "", fFunctionName: "", fGridLength: 0,
                                 fLabelColor: 1, fLabelFont: 42, fLabelOffset: 0.005, fLabelSize: 0.035,
                                 fName: "", fNdiv: 12, fTickSize: 0.02, fTimeFormat: "",
                                 fTitle: "", fTitleOffset: 1, fTitleSize: 0.035,
                                 fWmax: 100, fWmin: 0 });
            break;
         case 'TAttPad':
            JSROOT.extend(obj, { fLeftMargin: JSROOT.gStyle.fPadLeftMargin,
                                 fRightMargin: JSROOT.gStyle.fPadRightMargin,
                                 fBottomMargin: JSROOT.gStyle.fPadBottomMargin,
                                 fTopMargin: JSROOT.gStyle.fPadTopMargin,
                                 fXfile: 2, fYfile: 2, fAfile: 1, fXstat: 0.99, fYstat: 0.99, fAstat: 2,
                                 fFrameFillColor: JSROOT.gStyle.fFrameFillColor,
                                 fFrameFillStyle: JSROOT.gStyle.fFrameFillStyle,
                                 fFrameLineColor: JSROOT.gStyle.fFrameLineColor,
                                 fFrameLineWidth: JSROOT.gStyle.fFrameLineWidth,
                                 fFrameLineStyle: JSROOT.gStyle.fFrameLineStyle,
                                 fFrameBorderSize: JSROOT.gStyle.fFrameBorderSize,
                                 fFrameBorderMode: JSROOT.gStyle.fFrameBorderMode });
            break;
         case 'TPad':
            JSROOT.Create("TObject", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttFill", obj);
            JSROOT.Create("TAttPad", obj);
            JSROOT.extend(obj, { fX1: 0, fY1: 0, fX2: 1, fY2: 1, fXtoAbsPixelk: 1, fXtoPixelk: 1,
                                 fXtoPixel: 1, fYtoAbsPixelk: 1, fYtoPixelk: 1, fYtoPixel: 1,
                                 fUtoAbsPixelk: 1, fUtoPixelk: 1, fUtoPixel: 1, fVtoAbsPixelk: 1,
                                 fVtoPixelk: 1, fVtoPixel: 1, fAbsPixeltoXk: 1, fPixeltoXk: 1,
                                 fPixeltoX: 1, fAbsPixeltoYk: 1, fPixeltoYk: 1, fPixeltoY: 1,
                                 fXlowNDC: 0, fYlowNDC: 0, fXUpNDC: 0, fYUpNDC: 0, fWNDC: 1, fHNDC: 1,
                                 fAbsXlowNDC: 0, fAbsYlowNDC: 0, fAbsWNDC: 1, fAbsHNDC: 1,
                                 fUxmin: 0, fUymin: 0, fUxmax: 0, fUymax: 0, fTheta: 30, fPhi: 30, fAspectRatio: 0,
                                 fNumber: 0, fLogx: JSROOT.gStyle.fOptLogx, fLogy: JSROOT.gStyle.fOptLogy, fLogz: JSROOT.gStyle.fOptLogz,
                                 fTickx: JSROOT.gStyle.fPadTickX,
                                 fTicky: JSROOT.gStyle.fPadTickY,
                                 fPadPaint: 0, fCrosshair: 0, fCrosshairPos: 0, fBorderSize: 2,
                                 fBorderMode: 0, fModified: false,
                                 fGridx: JSROOT.gStyle.fPadGridX,
                                 fGridy: JSROOT.gStyle.fPadGridY,
                                 fAbsCoord: false, fEditable: true, fFixedAspectRatio: false,
                                 fPrimitives: JSROOT.Create("TList"), fExecs: null,
                                 fName: "pad", fTitle: "canvas" });

            break;
         case 'TAttCanvas':
            JSROOT.extend(obj, { fXBetween: 2, fYBetween: 2, fTitleFromTop: 1.2,
                                 fXdate: 0.2, fYdate: 0.3, fAdate: 1 });
            break;
         case 'TCanvas':
            JSROOT.Create("TPad", obj);
            JSROOT.extend(obj, { fNumPaletteColor: 0, fNextPaletteColor: 0, fDISPLAY: "$DISPLAY",
                                 fDoubleBuffer: 0, fRetained: true, fXsizeUser: 0,
                                 fYsizeUser: 0, fXsizeReal: 20, fYsizeReal: 10,
                                 fWindowTopX: 0, fWindowTopY: 0, fWindowWidth: 0, fWindowHeight: 0,
                                 fCw: 500, fCh: 300, fCatt: JSROOT.Create("TAttCanvas"),
                                 kMoveOpaque: true, kResizeOpaque: true, fHighLightColor: 5,
                                 fBatch: true, kShowEventStatus: false, kAutoExec: true, kMenuBar: true });
            break;
         case 'TGeoVolume':
            JSROOT.Create("TNamed", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttFill", obj);
            JSROOT.extend(obj, { fGeoAtt:0, fFinder: null, fMedium: null, fNodes: null, fNtotal: 0, fNumber: 0, fRefCount: 0, fShape: null, fVoxels: null });
            break;
         case 'TGeoNode':
            JSROOT.Create("TNamed", obj);
            JSROOT.extend(obj, { fGeoAtt:0, fMother: null, fNovlp: 0, fNumber: 0, fOverlaps: null, fVolume: null });
            break;
         case 'TGeoNodeMatrix':
            JSROOT.Create("TGeoNode", obj);
            JSROOT.extend(obj, { fMatrix: null });
            break;
         case 'TGeoTrack':
            JSROOT.Create("TObject", obj);
            JSROOT.Create("TAttLine", obj);
            JSROOT.Create("TAttMarker", obj);
            JSROOT.extend(obj, { fGeoAtt:0, fNpoints: 0, fPoints: [] });
            break;
      }

      obj._typename = typename;
      this.addMethods(obj);
      return obj;
   }

   /** @summary Create TList
    * @desc obsolete, use JSROOT.Create("TList") instead
    * @deprecated */
   JSROOT.CreateTList = function() { return JSROOT.Create("TList"); }

   /** @summary Create TAxis
    * @desc obsolete, use JSROOT.Create("TAxis") instead
    * @deprecated */
   JSROOT.CreateTAxis = function() { return JSROOT.Create("TAxis"); }

   /** @summary Create histogram object
    * @param {string} typename - histogram typename like TH1I or TH2F
    * @param {number} nbinsx - number of bins on X-axis
    * @param {number} [nbinsy] - number of bins on Y-axis (for 2D/3D histograms)
    * @param {number} [nbinsz] - number of bins on Z-axis (for 3D histograms)
    */
   JSROOT.CreateHistogram = function(typename, nbinsx, nbinsy, nbinsz) {
      // create histogram object of specified type
      // if bins numbers are specified, appropriate typed array will be created
      var histo = JSROOT.Create(typename);
      if (!histo.fXaxis || !histo.fYaxis || !histo.fZaxis) return null;
      histo.fName = "hist"; histo.fTitle = "title";
      if (nbinsx) JSROOT.extend(histo.fXaxis, { fNbins: nbinsx, fXmin: 0, fXmax: nbinsx });
      if (nbinsy) JSROOT.extend(histo.fYaxis, { fNbins: nbinsy, fXmin: 0, fXmax: nbinsy });
      if (nbinsz) JSROOT.extend(histo.fZaxis, { fNbins: nbinsz, fXmin: 0, fXmax: nbinsz });
      switch (parseInt(typename[2])) {
         case 1: if (nbinsx) histo.fNcells = nbinsx+2; break;
         case 2: if (nbinsx && nbinsy) histo.fNcells = (nbinsx+2) * (nbinsy+2); break;
         case 3: if (nbinsx && nbinsy && nbinsz) histo.fNcells = (nbinsx+2) * (nbinsy+2) * (nbinsz+2); break;
      }
      if (histo.fNcells > 0) {
         switch (typename[3]) {
            case "C" : histo.fArray = new Int8Array(histo.fNcells); break;
            case "S" : histo.fArray = new Int16Array(histo.fNcells); break;
            case "I" : histo.fArray = new Int32Array(histo.fNcells); break;
            case "F" : histo.fArray = new Float32Array(histo.fNcells); break;
            case "L" : histo.fArray = new Float64Array(histo.fNcells); break;
            case "D" : histo.fArray = new Float64Array(histo.fNcells); break;
            default: histo.fArray = new Array(histo.fNcells); break;
         }
         for (var i=0;i<histo.fNcells;++i) histo.fArray[i] = 0;
      }
      return histo;
   }

   /** @summary Create 1-d histogram
    * @desc obsolete, use JSROOT.CreateHistogram() instead
    * @deprecated */
   JSROOT.CreateTH1 = function(nbinsx) {
      return JSROOT.CreateHistogram("TH1I", nbinsx);
   }

   /** @summary Create 2-d histogram
    * @desc obsolete, use JSROOT.CreateHistogram() instead
    * @deprecated */
   JSROOT.CreateTH2 = function(nbinsx, nbinsy) {
      return JSROOT.CreateHistogram("TH2I", nbinsx, nbinsy);
   }

   /** @summary Create 3-d histogram
    * @desc obsolete, use JSROOT.CreateHistogram() instead
    * @deprecated */
   JSROOT.CreateTH3 = function(nbinsx, nbinsy, nbinsz) {
      return JSROOT.CreateHistogram("TH3I", nbinsx, nbinsy, nbinsz);
   }

   /** @summary Creates TPolyLine object
    * @param {number} npoints - number of points
    * @param {boolean} [use_int32] - use Int32Array type for points, default is Float32Array */
   JSROOT.CreateTPolyLine = function(npoints, use_int32) {
      var poly = JSROOT.Create("TPolyLine");
      if (npoints) {
         poly.fN = npoints;
         if (use_int32) {
            poly.fX = new Int32Array(npoints);
            poly.fY = new Int32Array(npoints);
         } else {
            poly.fX = new Float32Array(npoints);
            poly.fY = new Float32Array(npoints);
         }
      }

      return poly;
   }

   /** @summary Creates TGraph object
    * @param {number} npoints - number of points in TGraph
    * @param {array} [xpts] - array with X coordinates
    * @param {array} [ypts] - array with Y coordinates */
   JSROOT.CreateTGraph = function(npoints, xpts, ypts) {
      var graph = JSROOT.extend(JSROOT.Create("TGraph"), { fBits: 0x408, fName: "graph", fTitle: "title" });

      if (npoints>0) {
         graph.fMaxSize = graph.fNpoints = npoints;

         var usex = (typeof xpts == 'object') && (xpts.length === npoints);
         var usey = (typeof ypts == 'object') && (ypts.length === npoints);

         for (var i=0;i<npoints;++i) {
            graph.fX.push(usex ? xpts[i] : i/npoints);
            graph.fY.push(usey ? ypts[i] : i/npoints);
         }
      }

      return graph;
   }

   /** @summary Creates THStack object
    * @desc
    * As arguments one could specify any number of histograms objects
    * @example
    * var nbinsx = 20;
    * var h1 = JSROOT.CreateHistogram("TH1F", nbinsx);
    * var h2 = JSROOT.CreateHistogram("TH1F", nbinsx);
    * var h3 = JSROOT.CreateHistogram("TH1F", nbinsx);
    * var stack = JSROOT.CreateTHStack(h1, h2, h3);
    * */
   JSROOT.CreateTHStack = function() {
      var stack = JSROOT.Create("THStack");
      for(var i=0; i<arguments.length; ++i)
         stack.fHists.Add(arguments[i], "");
      return stack;
   }

   /** @summary Creates TMultiGraph object
    * @desc
    * As arguments one could specify any number of TGraph objects */
   JSROOT.CreateTMultiGraph = function() {
      var mgraph = JSROOT.Create("TMultiGraph");
      for(var i=0; i<arguments.length; ++i)
          mgraph.fGraphs.Add(arguments[i], "");
      return mgraph;
   }

   JSROOT.methodsCache = {}; // variable used to keep methods for known classes

   /** @summary Returns methods for given typename
    * @private
    */
   JSROOT.getMethods = function(typename, obj) {

      var m = JSROOT.methodsCache[typename],
          has_methods = (m!==undefined);

      if (!has_methods) m = {};

      // Due to binary I/O such TObject methods may not be set for derived classes
      // Therefore when methods requested for given object, check also that basic methods are there
      if ((typename=="TObject") || (typename=="TNamed") || (obj && (obj.fBits!==undefined)))
         if (m.TestBit === undefined) {
            m.TestBit = function (f) { return (this.fBits & f) != 0; };
            m.InvertBit = function (f) { this.fBits = this.fBits ^ (f & 0xffffff); };
         }

      if (has_methods) return m;

      if ((typename === 'TList') || (typename === 'THashList')) {
         m.Clear = function() {
            this.arr = [];
            this.opt = [];
         }
         m.Add = function(obj,opt) {
            this.arr.push(obj);
            this.opt.push((opt && typeof opt=='string') ? opt : "");
         }
         m.AddFirst = function(obj,opt) {
            this.arr.unshift(obj);
            this.opt.unshift((opt && typeof opt=='string') ? opt : "");
         }
         m.RemoveAt = function(indx) {
            this.arr.splice(indx, 1);
            this.opt.splice(indx, 1);
         }
      }

      if ((typename === "TPaveText") || (typename === "TPaveStats")) {
         m.AddText = function(txt) {
            // this.fLines.Add({ _typename: 'TLatex', fTitle: txt, fTextColor: 1 });
            var line = JSROOT.Create("TLatex");
            line.fTitle = txt;
            this.fLines.Add(line);
         }
         m.Clear = function() {
            this.fLines.Clear();
         }
      }

      if ((typename.indexOf("TF1") == 0) || (typename === "TF2")) {
         m.addFormula = function(obj) {
            if (!obj) return;
            if (this.formulas === undefined) this.formulas = [];
            this.formulas.push(obj);
         }

         m.evalPar = function(x, y) {
            if (! ('_func' in this) || (this._title !== this.fTitle)) {

              var _func = this.fTitle, isformula = false, pprefix = "[";
              if (_func === "gaus") _func = "gaus(0)";
              if (this.fFormula && typeof this.fFormula.fFormula == "string") {
                 if (this.fFormula.fFormula.indexOf("[](double*x,double*p)")==0) {
                    isformula = true; pprefix = "p[";
                    _func = this.fFormula.fFormula.substr(21);
                 } else {
                    _func = this.fFormula.fFormula;
                    pprefix = "[p";
                 }
                 if (this.fFormula.fClingParameters && this.fFormula.fParams) {
                    for (var i=0;i<this.fFormula.fParams.length;++i) {
                       var regex = new RegExp('(\\[' + this.fFormula.fParams[i].first + '\\])', 'g'),
                           parvalue = this.fFormula.fClingParameters[this.fFormula.fParams[i].second];
                       _func = _func.replace(regex, (parvalue < 0) ? "(" + parvalue + ")" : parvalue);
                    }
                 }
              }

              if ('formulas' in this)
                 for (var i=0;i<this.formulas.length;++i)
                    while (_func.indexOf(this.formulas[i].fName) >= 0)
                       _func = _func.replace(this.formulas[i].fName, this.formulas[i].fTitle);
              _func = _func.replace(/\b(abs)\b/g, 'TMath::Abs')
                           .replace(/TMath::Exp\(/g, 'Math.exp(')
                           .replace(/TMath::Abs\(/g, 'Math.abs(');
              if (typeof JSROOT.Math == 'object') {
                 this._math = JSROOT.Math;
                 _func = _func.replace(/TMath::Prob\(/g, 'this._math.Prob(')
                              .replace(/TMath::Gaus\(/g, 'this._math.Gaus(')
                              .replace(/TMath::BreitWigner\(/g, 'this._math.BreitWigner(')
                              .replace(/xygaus\(/g, 'this._math.gausxy(this, x, y, ')
                              .replace(/gaus\(/g, 'this._math.gaus(this, x, ')
                              .replace(/gausn\(/g, 'this._math.gausn(this, x, ')
                              .replace(/expo\(/g, 'this._math.expo(this, x, ')
                              .replace(/landau\(/g, 'this._math.landau(this, x, ')
                              .replace(/landaun\(/g, 'this._math.landaun(this, x, ')
                              .replace(/ROOT::Math::/g, 'this._math.');
              }
              for (var i=0;i<this.fNpar;++i) {
                 var parname = pprefix + i + "]";
                 while(_func.indexOf(parname) != -1)
                    _func = _func.replace(parname, '('+this.GetParValue(i)+')');
              }
              _func = _func.replace(/\b(sin)\b/gi, 'Math.sin')
                           .replace(/\b(cos)\b/gi, 'Math.cos')
                           .replace(/\b(tan)\b/gi, 'Math.tan')
                           .replace(/\b(exp)\b/gi, 'Math.exp')
                           .replace(/\b(pow)\b/gi, 'Math.pow')
                           .replace(/pi/g, 'Math.PI');
              for (var n=2;n<10;++n)
                 _func = _func.replace('x^'+n, 'Math.pow(x,'+n+')');

              if (isformula) {
                 _func = _func.replace(/x\[0\]/g,"x");
                 if (this._typename==="TF2") {
                    _func = _func.replace(/x\[1\]/g,"y");
                    this._func = new Function("x", "y", _func).bind(this);
                 } else {
                    this._func = new Function("x", _func).bind(this);
                 }
              } else
              if (this._typename==="TF2")
                 this._func = new Function("x", "y", "return " + _func).bind(this);
              else
                 this._func = new Function("x", "return " + _func).bind(this);

              this._title = this.fTitle;
            }

            return this._func(x, y);
         }
         m.GetParName = function(n) {
            if (this.fParams && this.fParams.fParNames) return this.fParams.fParNames[n];
            if (this.fFormula && this.fFormula.fParams) return this.fFormula.fParams[n].first;
            if (this.fNames && this.fNames[n]) return this.fNames[n];
            return "p"+n;
         }
         m.GetParValue = function(n) {
            if (this.fParams && this.fParams.fParameters) return this.fParams.fParameters[n];
            if (this.fFormula && this.fFormula.fClingParameters) return this.fFormula.fClingParameters[n];
            if (this.fParams) return this.fParams[n];
            return undefined;
         }
         m.GetParError = function(n) {
            return this.fParErrors ? this.fParErrors[n] : undefined;
         }
         m.GetNumPars = function() {
            return this.fNpar;
         }
      }

      if (((typename.indexOf("TGraph") == 0) || (typename == "TCutG")) && (typename != "TGraphPolargram") && (typename != "TGraphTime")) {
         // check if point inside figure specified by the TGraph
         m.IsInside = function(xp,yp) {
            var i, j = this.fNpoints - 1, x = this.fX, y = this.fY, oddNodes = false;

            for (i=0; i<this.fNpoints; ++i) {
               if ((y[i]<yp && y[j]>=yp) || (y[j]<yp && y[i]>=yp)) {
                  if (x[i]+(yp-y[i])/(y[j]-y[i])*(x[j]-x[i])<xp) {
                     oddNodes = !oddNodes;
                  }
               }
               j=i;
            }

            return oddNodes;
         };
      }

      if (typename.indexOf("TH1") == 0 ||
          typename.indexOf("TH2") == 0 ||
          typename.indexOf("TH3") == 0) {
         m.getBinError = function(bin) {
            //   -*-*-*-*-*Return value of error associated to bin number bin*-*-*-*-*
            //    if the sum of squares of weights has been defined (via Sumw2),
            //    this function returns the sqrt(sum of w2).
            //    otherwise it returns the sqrt(contents) for this bin.
            if (bin >= this.fNcells) bin = this.fNcells - 1;
            if (bin < 0) bin = 0;
            if (bin < this.fSumw2.length)
               return Math.sqrt(this.fSumw2[bin]);
            return Math.sqrt(Math.abs(this.fArray[bin]));
         };
         m.setBinContent = function(bin, content) {
            // Set bin content - only trivial case, without expansion
            this.fEntries++;
            this.fTsumw = 0;
            if ((bin>=0) && (bin<this.fArray.length))
               this.fArray[bin] = content;
         };
      }

      if (typename.indexOf("TH1") == 0) {
         m.getBin = function(x) { return x; }
         m.getBinContent = function(bin) { return this.fArray[bin]; }
         m.Fill = function(x, weight) {
            var axis = this.fXaxis,
                bin = 1 + Math.floor((x - axis.fXmin) / (axis.fXmax - axis.fXmin) * axis.fNbins);
            if (bin < 0) bin = 0; else
            if (bin > axis.fNbins + 1) bin = axis.fNbins + 1;
            this.fArray[bin] += ((weight===undefined) ? 1 : weight);
            this.fEntries++;
         }
      }

      if (typename.indexOf("TH2") == 0) {
         m.getBin = function(x, y) { return (x + (this.fXaxis.fNbins+2) * y); }
         m.getBinContent = function(x, y) { return this.fArray[this.getBin(x, y)]; }
         m.Fill = function(x, y, weight) {
            var axis1 = this.fXaxis, axis2 = this.fYaxis,
                bin1 = 1 + Math.floor((x - axis1.fXmin) / (axis1.fXmax - axis1.fXmin) * axis1.fNbins),
                bin2 = 1 + Math.floor((y - axis2.fXmin) / (axis2.fXmax - axis2.fXmin) * axis2.fNbins);
            if (bin1 < 0) bin1 = 0; else
            if (bin1 > axis1.fNbins + 1) bin1 = axis1.fNbins + 1;
            if (bin2 < 0) bin2 = 0; else
            if (bin2 > axis2.fNbins + 1) bin2 = axis2.fNbins + 1;
            this.fArray[bin1 + (axis1.fNbins+2)*bin2] += ((weight===undefined) ? 1 : weight);
            this.fEntries++;
         }
      }

      if (typename.indexOf("TH3") == 0) {
         m.getBin = function(x, y, z) { return (x + (this.fXaxis.fNbins+2) * (y + (this.fYaxis.fNbins+2) * z)); }
         m.getBinContent = function(x, y, z) { return this.fArray[this.getBin(x, y, z)]; };
         m.Fill = function(x, y, z, weight) {
            var axis1 = this.fXaxis, axis2 = this.fYaxis, axis3 = this.fZaxis,
                bin1 = 1 + Math.floor((x - axis1.fXmin) / (axis1.fXmax - axis1.fXmin) * axis1.fNbins),
                bin2 = 1 + Math.floor((y - axis2.fXmin) / (axis2.fXmax - axis2.fXmin) * axis2.fNbins),
                bin3 = 1 + Math.floor((z - axis3.fXmin) / (axis3.fXmax - axis3.fXmin) * axis3.fNbins);
            if (bin1 < 0) bin1 = 0; else
            if (bin1 > axis1.fNbins + 1) bin1 = axis1.fNbins + 1;
            if (bin2 < 0) bin2 = 0; else
            if (bin2 > axis2.fNbins + 1) bin2 = axis2.fNbins + 1;
            if (bin3 < 0) bin3 = 0; else
            if (bin3 > axis3.fNbins + 1) bin3 = axis3.fNbins + 1;
            this.fArray[bin1 + (axis1.fNbins+2)* (bin2+(axis2.fNbins+2)*bin3)] += ((weight===undefined) ? 1 : weight);
            this.fEntries++;
         }
      }

      if (typename.indexOf("TProfile") == 0) {
         if (typename.indexOf("TProfile2D") == 0) {
            m.getBin = function(x, y) { return (x + (this.fXaxis.fNbins+2) * y); }
            m.getBinContent = function(x, y) {
               var bin = this.getBin(x, y);
               if (bin < 0 || bin >= this.fNcells) return 0;
               if (this.fBinEntries[bin] < 1e-300) return 0;
               if (!this.fArray) return 0;
               return this.fArray[bin]/this.fBinEntries[bin];
            }
            m.getBinEntries = function(x, y) {
               var bin = this.getBin(x, y);
               if (bin < 0 || bin >= this.fNcells) return 0;
               return this.fBinEntries[bin];
            }
         }
         else {
            m.getBin = function(x) { return x; }
            m.getBinContent = function(bin) {
               if (bin < 0 || bin >= this.fNcells) return 0;
               if (this.fBinEntries[bin] < 1e-300) return 0;
               if (!this.fArray) return 0;
               return this.fArray[bin]/this.fBinEntries[bin];
            };
         }
         m.getBinEffectiveEntries = function(bin) {
            if (bin < 0 || bin >= this.fNcells) return 0;
            var sumOfWeights = this.fBinEntries[bin];
            if ( !this.fBinSumw2 || this.fBinSumw2.length != this.fNcells) {
               // this can happen  when reading an old file
               return sumOfWeights;
            }
            var sumOfWeightsSquare = this.fBinSumw2[bin];
            return (sumOfWeightsSquare > 0) ? sumOfWeights * sumOfWeights / sumOfWeightsSquare : 0;
         };
         m.getBinError = function(bin) {
            if (bin < 0 || bin >= this.fNcells) return 0;
            var cont = this.fArray[bin],               // sum of bin w *y
                sum  = this.fBinEntries[bin],          // sum of bin weights
                err2 = this.fSumw2[bin],               // sum of bin w * y^2
                neff = this.getBinEffectiveEntries(bin);  // (sum of w)^2 / (sum of w^2)
            if (sum < 1e-300) return 0;                  // for empty bins
            var EErrorType = { kERRORMEAN : 0, kERRORSPREAD : 1, kERRORSPREADI : 2, kERRORSPREADG : 3 };
            // case the values y are gaussian distributed y +/- sigma and w = 1/sigma^2
            if (this.fErrorMode === EErrorType.kERRORSPREADG)
               return 1.0/Math.sqrt(sum);
            // compute variance in y (eprim2) and standard deviation in y (eprim)
            var contsum = cont/sum, eprim = Math.sqrt(Math.abs(err2/sum - contsum*contsum));
            if (this.fErrorMode === EErrorType.kERRORSPREADI) {
               if (eprim != 0) return eprim/Math.sqrt(neff);
               // in case content y is an integer (so each my has an error +/- 1/sqrt(12)
               // when the std(y) is zero
               return 1.0/Math.sqrt(12*neff);
            }
            // if approximate compute the sums (of w, wy and wy2) using all the bins
            //  when the variance in y is zero
            // case option "S" return standard deviation in y
            if (this.fErrorMode === EErrorType.kERRORSPREAD) return eprim;
            // default case : fErrorMode = kERRORMEAN
            // return standard error on the mean of y
            return (eprim/Math.sqrt(neff));
         };
      }

      if (typename == "TAxis") {
         m.GetBinLowEdge = function(bin) {
            if (this.fNbins <= 0) return 0;
            if ((this.fXbins.length > 0) && (bin > 0) && (bin <= this.fNbins)) return this.fXbins[bin-1];
            return this.fXmin + (bin-1) * (this.fXmax - this.fXmin) / this.fNbins;
         }
         m.GetBinCenter = function(bin) {
            if (this.fNbins <= 0) return 0;
            if ((this.fXbins.length > 0) && (bin > 0) && (bin < this.fNbins)) return (this.fXbins[bin-1] + this.fXbins[bin])/2;
            return this.fXmin + (bin-0.5) * (this.fXmax - this.fXmin) / this.fNbins;
         }
      }

      if (typeof JSROOT.getMoreMethods == "function")
         JSROOT.getMoreMethods(m, typename, obj);

      JSROOT.methodsCache[typename] = m;
      return m;
   }

   /** @summary Add methods for specified type.
    * Will be automatically applied when decoding JSON string
    * @private */
   JSROOT.registerMethods = function(typename, m) {
      JSROOT.methodsCache[typename] = m;
   }

   /** @summary Returns true if object represents basic ROOT collections
    * @private */
   JSROOT.IsRootCollection = function(lst, typename) {
      if (lst && (typeof lst === 'object')) {
         if ((lst.$kind === "TList") || (lst.$kind === "TObjArray")) return true;
         if (!typename) typename = lst._typename;
      }
      if (!typename) return false;
      return (typename === 'TList') || (typename === 'THashList') || (typename === 'TMap') ||
             (typename === 'TObjArray') || (typename === 'TClonesArray');
   }

   /** @summary Adds specific methods to the object.
    *
    * JSROOT implements some basic methods for different ROOT classes.
    * @param {object} obj - object where methods are assigned
    * @param {string} typename - optional typename, if not specified, obj._typename will be used
    * @private
    */
   JSROOT.addMethods = function(obj, typename) {
      this.extend(obj, this.getMethods(typename || obj._typename, obj));
   }

   JSROOT.lastFFormat = "";

   /** @summary Converts numeric value to string according to specified format.
    *
    * @param {number} value - value to convert
    * @param {strting} fmt - format can be like 5.4g or 4.2e or 6.4f
    * @returns {string} - converted value
    * @private
    */
   JSROOT.FFormat = function(value, fmt) {
      if (!fmt) fmt = "6.4g";

      JSROOT.lastFFormat = "";

      fmt = fmt.trim();
      var len = fmt.length;
      if (len<2) return value.toFixed(4);
      var last = fmt[len-1];
      fmt = fmt.slice(0,len-1);
      var isexp = null;
      var prec = fmt.indexOf(".");
      if (prec<0) prec = 4; else prec = Number(fmt.slice(prec+1));
      if (isNaN(prec) || (prec<0) || (prec==null)) prec = 4;

      var significance = false;
      if ((last=='e') || (last=='E')) { isexp = true; } else
      if (last=='Q') { isexp = true; significance = true; } else
      if ((last=='f') || (last=='F')) { isexp = false; } else
      if (last=='W') { isexp = false; significance = true; } else
      if ((last=='g') || (last=='G')) {
         var se = JSROOT.FFormat(value, fmt+'Q'),
             _fmt = JSROOT.lastFFormat,
             sg = JSROOT.FFormat(value, fmt+'W');

         if (se.length < sg.length) {
            JSROOT.lastFFormat = _fmt;
            return se;
         }
         return sg;
      } else {
         isexp = false;
         prec = 4;
      }

      if (isexp) {
         // for exponential representation only one significant digit befor point
         if (significance) prec--;
         if (prec<0) prec = 0;

         JSROOT.lastFFormat = '5.'+prec+'e';

         return value.toExponential(prec);
      }

      var sg = value.toFixed(prec);

      if (significance) {

         // when using fixed representation, one could get 0.0
         if ((value!=0) && (Number(sg)==0.) && (prec>0)) {
            prec = 20; sg = value.toFixed(prec);
         }

         var l = 0;
         while ((l<sg.length) && (sg[l] == '0' || sg[l] == '-' || sg[l] == '.')) l++;

         var diff = sg.length - l - prec;
         if (sg.indexOf(".")>l) diff--;

         if (diff != 0) {
            prec-=diff;
            if (prec<0) prec = 0; else if (prec>20) prec = 20;
            sg = value.toFixed(prec);
         }
      }

      JSROOT.lastFFormat = '5.'+prec+'f';

      return sg;
   }

   /** @summary Implements log10
    * @private */
   JSROOT.log10 = function(n) {
      return Math.log(n) / Math.log(10);
   }

   // Dummy function, will be redefined when JSRootPainter is loaded
   JSROOT.progress = function(msg, tmout) {
      if ((msg !== undefined) && (typeof msg=="string")) JSROOT.console(msg);
   }

   // connect to the TWebWindow instance
   JSROOT.ConnectWebWindow = function(arg) {
      if (typeof arg == 'function') arg = { callback: arg };

      if (arg.openui5src) JSROOT.openui5src = arg.openui5src;
      if (arg.openui5libs) JSROOT.openui5libs = arg.openui5libs;
      if (arg.openui5theme) JSROOT.openui5theme = arg.openui5theme;
      JSROOT.AssertPrerequisites("2d;" + (arg && arg.prereq ? arg.prereq : ""), function() {
         if (arg && arg.prereq) delete arg.prereq;
         JSROOT.ConnectWebWindow(arg);
      }, (arg ? arg.prereq_logdiv : undefined));
   }

   /** Initialize JSROOT.
    * Called when script is loaded. Process URL parameters, supplied with JSRootCore.js script
    * @private
    */
   JSROOT.Initialize = function() {

      if (JSROOT.source_fullpath.length === 0) return this;

      function window_on_load(func) {
         if (func!=null) {
            if (document.attachEvent ? document.readyState === 'complete' : document.readyState !== 'loading')
               func();
            else
               window.onload = func;
         }
         return JSROOT;
      }

      var src = JSROOT.source_fullpath;

      if (JSROOT.GetUrlOption('nocache', src) != null) JSROOT.nocache = (new Date).getTime(); // use timestamp to overcome cache limitation
      if ((JSROOT.GetUrlOption('wrong_http_response', src) != null) || (JSROOT.GetUrlOption('wrong_http_response') != null))
         JSROOT.wrong_http_response_handling = true; // server may send wrong content length by partial requests, use other method to control this

      if (JSROOT.GetUrlOption('gui', src) !== null)
         return window_on_load( function() { JSROOT.BuildSimpleGUI(); } );

      if ( typeof define === "function" && define.amd )
         return window_on_load( function() { JSROOT.BuildSimpleGUI('check_existing_elements'); } );

      var prereq = "";
      if (JSROOT.GetUrlOption('io', src)!=null) prereq += "io;";
      if (JSROOT.GetUrlOption('tree', src)!=null) prereq += "tree;";
      if (JSROOT.GetUrlOption('2d', src)!=null) prereq += "2d;";
      if (JSROOT.GetUrlOption('v7', src)!=null) prereq += "v7;";
      if (JSROOT.GetUrlOption('hist', src)!=null) prereq += "2d;hist;";
      if (JSROOT.GetUrlOption('hierarchy', src)!=null) prereq += "2d;hierarchy;";
      if (JSROOT.GetUrlOption('jq2d', src)!=null) prereq += "2d;hierarchy;jq2d;";
      if (JSROOT.GetUrlOption('more2d', src)!=null) prereq += "more2d;";
      if (JSROOT.GetUrlOption('geom', src)!=null) prereq += "geom;";
      if (JSROOT.GetUrlOption('3d', src)!=null) prereq += "3d;";
      if (JSROOT.GetUrlOption('math', src)!=null) prereq += "math;";
      if (JSROOT.GetUrlOption('mathjax', src)!=null) prereq += "mathjax;";
      if (JSROOT.GetUrlOption('openui5', src)!=null) prereq += "openui5;";
      var user = JSROOT.GetUrlOption('load', src),
          onload = JSROOT.GetUrlOption('onload', src),
          bower = JSROOT.GetUrlOption('bower', src);

      if (user) prereq += "io;2d;load:" + user;
      if ((bower===null) && (JSROOT.source_dir.indexOf("bower_components/jsroot/")>=0)) bower = "";
      if (bower!==null) {
         if (bower.length>0) JSROOT.bower_dir = bower; else
            if (JSROOT.source_dir.indexOf("jsroot/") == JSROOT.source_dir.length - 7)
               JSROOT.bower_dir = JSROOT.source_dir.substr(0, JSROOT.source_dir.length - 7);
         if (JSROOT.bower_dir !== null) console.log("Set JSROOT.bower_dir to " + JSROOT.bower_dir);
      }

      if ((prereq.length>0) || (onload!=null))
         window_on_load(function() {
            if (prereq.length>0) JSROOT.AssertPrerequisites(prereq, onload); else
               if (onload!=null) {
                  onload = JSROOT.findFunction(onload);
                  if (typeof onload == 'function') onload();
               }
         });

      return this;
   }

   return JSROOT.Initialize();

}));

/// JSRootCore.js ends

