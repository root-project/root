
/** @summary version id
  * @desc For the JSROOT release the string in format "major.minor.patch" like "7.0.0" */
let version_id = "dev";

/** @summary version date
  * @desc Release date in format day/month/year like "19/11/2021" */
let version_date = "10/06/2022";

/** @summary version id and date
  * @desc Produced by concatenation of {@link version_id} and {@link version_date}
  * Like "7.0.0 14/04/2022" */
let version = version_id + " " + version_date;

/** @summary Location of JSROOT scripts
  * @desc Automatically detected and used to load other scripts or modules */
let source_dir = "";

let nodejs = !!((typeof process == 'object') && process.version && (typeof process.versions == 'object') && process.versions.node && process.versions.v8);

/** @summary internal data
  * @private */
let internals = {
   id_counter: 1          ///< unique id contner, starts from 1
};

//openuicfg // DO NOT DELETE, used to configure openui5 usage like internals.openui5src = "nojsroot";

const src = import.meta?.url;
if (src && (typeof src == "string")) {
   const pos = src.indexOf("modules/core.mjs");
   if (pos >= 0) {
      source_dir = src.slice(0, pos);
      console.log(`Set jsroot source_dir to ${source_dir}, ${version}`);
   } else {
      console.log(`jsroot bundle, ${version}`);
      internals.ignore_v6 = true;
   }
}

let batch_mode = nodejs;

/** @summary Indicates if running in batch mode */
function isBatchMode() { return batch_mode; }

/** @summary Set batch mode */
function setBatchMode(on) { batch_mode = !!on; }

/** @summary Indicates if running inside Node.js */
function isNodeJs() { return nodejs; }


let node_atob, node_xhr2;

///_begin_exclude_in_qt5web_
if(isNodeJs()) { node_atob = await import('atob').then(h => h.default); node_xhr2 = await import('xhr2').then(h => h.default); } /// cutNodeJs
///_end_exclude_in_qt5web_

let browser = { isOpera: false, isFirefox: true, isSafari: false, isChrome: false, isWin: false, touches: false  };

if ((typeof document !== "undefined") && (typeof window !== "undefined")) {
   browser.isOpera = !!window.opera || navigator.userAgent.indexOf(' OPR/') >= 0;
   browser.isFirefox = typeof InstallTrigger !== 'undefined';
   browser.isSafari = Object.prototype.toString.call(window.HTMLElement).indexOf('Constructor') > 0;
   browser.isChrome = !!window.chrome && !browser.isOpera;
   browser.isChromeHeadless = navigator.userAgent.indexOf('HeadlessChrome') >= 0;
   browser.chromeVersion = (browser.isChrome || browser.isChromeHeadless) ? parseInt(navigator.userAgent.match(/Chrom(?:e|ium)\/([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+)/)[1]) : 0;
   browser.isWin = navigator.platform.indexOf('Win') >= 0;
   browser.touches = ('ontouchend' in document); // identify if touch events are supported
}

/** @summary Check if prototype string match to array (typed on untyped)
  * @returns {Number} 0 - not array, 1 - regular array, 2 - typed array
  * @private */
function isArrayProto(proto) {
    if ((proto.length < 14) || (proto.indexOf('[object ') != 0)) return 0;
    let p = proto.indexOf('Array]');
    if ((p < 0) || (p != proto.length - 6)) return 0;
    // plain array has only "[object Array]", typed array type name inside
    return proto.length == 14 ? 1 : 2;
}

/** @desc Specialized JSROOT constants, used in {@link settings}
  * @namespace */
let constants = {
   /** @summary Kind of 3D rendering, used for {@link settings.Render3D}
     * @namespace */
   Render3D: {
      /** @summary Default 3D rendering, normally WebGL, if not supported - SVG */
      Default: 0,
      /** @summary Use normal WebGL rendering and place as interactive Canvas element on HTML page */
      WebGL: 1,
      /** @summary Use WebGL rendering, but convert into svg image, not interactive */
      WebGLImage: 2,
      /** @summary Use SVG rendering, slow, inprecise and not interactive, nor recommendet */
      SVG: 3,
      fromString: function(s) {
         if ((s === "webgl") || (s == "gl")) return this.WebGL;
         if (s === "img") return this.WebGLImage;
         if (s === "svg") return this.SVG;
         return this.Default;
      }
   },
   /** @summary Way to embed 3D into SVG, used for {@link settings.Embed3D}
     * @namespace */
   Embed3D: {
      /** @summary Do not embed 3D drawing, use complete space */
      NoEmbed: -1,
      /** @summary Default embeding mode - on Firefox and latest Chrome is real ```Embed```, on all other ```Overlay``` */
      Default: 0,
      /** @summary WebGL canvas not inserted into SVG, but just overlayed The only way how earlier Chrome browser can be used */
      Overlay: 1,
      /** @summary Really embed WebGL Canvas into SVG */
      Embed: 2,
      /** @summary Embeding, but when SVG rendering or SVG image converion is used */
      EmbedSVG: 3,
      /** @summary Convert string values into number  */
      fromString: function(s) {
         if (s === "embed") return this.Embed;
         if (s === "overlay") return this.Overlay;
         return this.Default;
      }
   },
   /** @summary How to use latex in text drawing, used for {@link settings.Latex}
     * @namespace */
   Latex: {
      /** @summary do not use Latex at all for text drawing */
      Off: 0,
      /** @summary convert only known latex symbols */
      Symbols: 1,
      /** @summary normal latex processing with svg */
      Normal: 2,
      /** @summary use MathJax for complex cases, otherwise simple SVG text */
      MathJax: 3,
      /** @summary always use MathJax for text rendering */
      AlwaysMathJax: 4,
      /** @summary Convert string values into number */
      fromString: function(s) {
         if (!s || (typeof s !== 'string'))
            return this.Normal;
         switch(s){
            case "off": return this.Off;
            case "symbols": return this.Symbols;
            case "normal":
            case "latex":
            case "exp":
            case "experimental": return this.Normal;
            case "MathJax":
            case "mathjax":
            case "math": return this.MathJax;
            case "AlwaysMathJax":
            case "alwaysmath":
            case "alwaysmathjax": return this.AlwaysMathJax;
         }
         let code = parseInt(s);
         return (Number.isInteger(code) && (code >= this.Off) && (code <= this.AlwaysMathJax)) ? code : this.Normal;
      }
   }
};

/** @desc Global JSROOT settings
  * @namespace */
let settings = {
   /** @summary Render of 3D drawing methods, see {@link constants.Render3D} for possible values */
   Render3D: constants.Render3D.Default,
   /** @summary 3D drawing methods in batch mode, see {@link constants.Render3D} for possible values */
   Render3DBatch: constants.Render3D.Default,
   /** @summary Way to embed 3D drawing in SVG, see {@link constants.Embed3D} for possible values */
   Embed3D: constants.Embed3D.Default,
   /** @summary Enable or disable tooltips, default on */
   Tooltip: true,
   /** @summary Time in msec for appearance of tooltips, 0 - no animation */
   TooltipAnimation: 500,
   /** @summary Enables context menu usage */
   ContextMenu: true,
   /** @summary Global zooming flag, enable/disable any kind of interactive zooming */
   Zooming: true,
   /** @summary Zooming with the mouse events */
   ZoomMouse: true,
   /** @summary Zooming with mouse wheel */
   ZoomWheel: true,
   /** @summary Zooming on touch devices */
   ZoomTouch: true,
   /** @summary Enables move and resize of elements like statbox, title, pave, colz  */
   MoveResize: true,
   /** @summary Configures keybord key press handling
     * @desc Can be disabled to prevent keys heandling in complex HTML layouts
     * @default true */
   HandleKeys: true,
   /** @summary enables drag and drop functionality */
   DragAndDrop: true,
   /** @summary Interactive dragging of TGraph points */
   DragGraphs: true,
   /** @summary Show progress box */
   ProgressBox: true,
   /** @summary Show additional tool buttons on the canvas, false - disabled, true - enabled, 'popup' - only toggle button */
   ToolBar: 'popup',
   /** @summary Position of toolbar 'left' left-bottom corner on canvas, 'right' - right-bottom corner on canvas, opposite on sub-pads */
   ToolBarSide: 'left',
   /** @summary display tool bar vertical (default false) */
   ToolBarVert: false,
   /** @summary if drawing inside particular div can be enlarged on full window */
   CanEnlarge: true,
   /** @summary if frame position can be adjusted to let show axis or colz labels */
   CanAdjustFrame: false,
   /** @summary calculation of text size consumes time and can be skipped to improve performance (but with side effects on text adjustments) */
   ApproxTextSize: false,
   /** @summary Histogram drawing optimization: 0 - disabled, 1 - only for large (>5000 1d bins, >50 2d bins) histograms, 2 - always */
   OptimizeDraw: 1,
   /** @summary Automatically create stats box, default on */
   AutoStat: true,
   /** @summary Default frame position in NFC */
   FrameNDC: { fX1NDC: 0.07, fY1NDC: 0.12, fX2NDC: 0.95, fY2NDC: 0.88 },
   /** @summary size of pad, where many features will be deactivated like text draw or zooming  */
   SmallPad: { width: 150, height: 100 },
   /** @summary Default color palette id  */
   Palette: 57,
   /** @summary Configures Latex usage, see {@link constants.Latex} for possible values */
   Latex: constants.Latex.Normal,
   /** @summary Grads per segment in TGeo spherical shapes like tube */
   GeoGradPerSegm: 6,
   /** @summary Enables faces compression after creation of composite shape  */
   GeoCompressComp: true,
   /** @summary if true, ignore all kind of URL options in the browser URL */
   IgnoreUrlOptions: false,
   /** @summary how many items shown on one level of hierarchy */
   HierarchyLimit: 250,
   /** @summary custom format for all X values, when not specified {@link gStyle.fStatFormat} is used */
   XValuesFormat: undefined,
   /** @summary custom format for all Y values, when not specified {@link gStyle.fStatFormat} is used */
   YValuesFormat: undefined,
   /** @summary custom format for all Z values, when not specified {@link gStyle.fStatFormat} is used */
   ZValuesFormat: undefined,
   /** @summary Let detect and solve problem when browser returns wrong content-length parameter
     * @desc See [jsroot#189]{@link https://github.com/root-project/jsroot/issues/189} for more info
     * Can be enabled by adding "wrong_http_response" parameter to URL when using JSROOT UI
     * @default false */
   HandleWrongHttpResponse: false,
   /** @summary Let tweak browser caching
     * @desc When specified, extra URL parameter like ```?stamp=unique_value``` append to each files loaded
     * In such case browser will be forced to load file content disregards of server cache settings
     * @default true */
   UseStamp: true,
   /** @summary Skip streamer infos from the GUI */
   SkipStreamerInfos: false,
   /** @summary Show only last cycle for objects in TFile */
   OnlyLastCycle: false,
   /** @summary Configures dark mode for the GUI */
   DarkMode: false
};


if (nodejs)
   Object.assign(settings, { ToolBar: false, Tooltip: 0, ContextMenu: false, Zooming: false, MoveResize: false, DragAndDrop: false, ProgressBox: false });


/** @namespace
  * @summary Insiance of TStyle object like in ROOT
  * @desc Includes default draw styles, can be changed after loading of JSRoot.core.js
  * or can be load from the file providing style=itemname in the URL
  * See [TStyle docu]{@link https://root.cern/doc/master/classTStyle.html} "Private attributes" section for more detailed info about each value */
let gStyle = {
   fName: "Modern",
   /** @summary Default log x scale */
   fOptLogx: 0,
   /** @summary Default log y scale */
   fOptLogy: 0,
   /** @summary Default log z scale */
   fOptLogz: 0,
   fOptDate: 0,
   fOptFile: 0,
   fDateX: 0.01,
   fDateY: 0.01,
   /** @summary Draw histogram title */
   fOptTitle: 1,
   /** @summary Canvas fill color */
   fCanvasColor: 0,
   /** @summary Pad fill color */
   fPadColor: 0,
   fPadBottomMargin: 0.1,
   fPadTopMargin: 0.1,
   fPadLeftMargin: 0.1,
   fPadRightMargin: 0.1,
   /** @summary TPad.fGridx default value */
   fPadGridX: false,
   /** @summary TPad.fGridy default value */
   fPadGridY: false,
   fPadTickX: 0,
   fPadTickY: 0,
   /** @summary fill color for stat box */
   fStatColor: 0,
   /** @summary fill style for stat box */
   fStatStyle: 1000,
   /** @summary text color in stat box */
   fStatTextColor: 1,
   /** @summary text size in stat box */
   fStatFontSize: 0,
   /** @summary stat text font */
   fStatFont: 42,
   /** @summary Stat border size */
   fStatBorderSize: 1,
   /** @summary Printing format for stats */
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
   /** @summary X position of top left corner of title box */
   fTitleX: 0.5,
   /** @summary Y position of top left corner of title box  */
   fTitleY: 0.995,
   /** @summary Width of title box */
   fTitleW: 0,
   /** @summary Height of title box */
   fTitleH: 0,
   /** @summary Printing format for fit parameters */
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
   /** @summary size in pixels of end error for E1 draw options */
   fEndErrorSize: 2,
   /** @summary X size of the error marks for the histogram drawings */
   fErrorX: 0.5,
   /** @summary when true, BAR and LEGO drawing using base = 0  */
   fHistMinimumZero: false,
   /** @summary Margin between histogram's top and pad's top */
   fHistTopMargin: 0.05,
   fHistFillColor: 0,
   fHistFillStyle: 1001,
   fHistLineColor: 602,
   fHistLineStyle: 1,
   fHistLineWidth: 1,
   /** @summary format for bin content */
   fPaintTextFormat: "g",
   /** @summary default time offset, UTC time at 01/01/95   */
   fTimeOffset: 788918400,
   fLegendBorderSize: 1,
   fLegendFont: 42,
   fLegendTextSize: 0,
   fLegendFillColor: 0,
   fHatchesLineWidth: 1,
   fHatchesSpacing: 1
};

/** @summary Method returns current document in use
  * @private */
function getDocument() {
   if (nodejs)
      return internals.nodejs_document;
   if (typeof document !== 'undefined')
      return document;
   if (typeof window == 'object')
      return window.document;
   return undefined;
}

/** @summary Inject javascript code
  * @desc Replacement for eval
  * @returns {Promise} when code is injected
  * @private */
function injectCode(code) {
   if (nodejs) {
      let name, fs;
      return import('tmp').then(tmp => {
         name = tmp.tmpNameSync() + ".js";
         return import('fs');
      }).then(_fs => {
         fs = _fs;
         fs.writeFileSync(name, code);
         return import("file://" + name);
      }).finally(() => fs.unlinkSync(name));
   }
   if (typeof document !== 'undefined') {

      // check if code already loaded - to avoid duplication
      let scripts = document.getElementsByTagName('script');
      for (let n = 0; n < scripts.length; ++n)
         if (scripts[n].innerHTML == code)
            return Promise.resolve(true);

      let promise = code.indexOf("JSROOT.require") >= 0 ? _ensureJSROOT() : Promise.resolve(true);

      return promise.then(() => {
         return new Promise(resolve => {
            let element = document.createElement("script");
            element.setAttribute("type", "text/javascript");
            element.innerHTML = code;
            document.head.appendChild(element);
            setTimeout(() => resolve(true), 10); // while onload event not fired, just postpone resolve
         });
      });
   }

   return Promise.resolve(false);
}

/** @summary Load script or CSS file into the browser
  * @param {String} url - script or css file URL (or array, in this case they all loaded secuentially)
  * @returns {Promise} */
function loadScript(url) {
   if (!url)
      return Promise.resolve(true);

   if ((typeof url == 'string') && (url.indexOf(";") >= 0))
      url = url.split(";");

   if (typeof url != 'string') {
      let scripts = url, loadNext = () => {
         if (!scripts.length) return true;
         return loadScript(scripts.shift()).then(loadNext, loadNext);
      };
      return loadNext();
   }

   if (url.indexOf("$$$")===0) {
      url = url.slice(3);
      if ((url.indexOf("style/") == 0) && (url.indexOf('.css') < 0))
         url += ".css";
      url = source_dir + url;
   }

   let element, isstyle = url.indexOf(".css") > 0;

   if (nodejs) {
      if (isstyle)
         return Promise.resolve(null);
      if ((url.indexOf("http:") == 0) || (url.indexOf("https:") == 0))
         return httpRequest(url, "text").then(code => injectCode(code));

      return import(url);
   }

   const match_url = src => {
      if (src == url) return true;
      let indx = src.indexOf(url);
      return (indx > 0) && (indx + url.length == src.length) && (src[indx-1] == "/");
   };

   if (isstyle) {
      let styles = document.getElementsByTagName('link');
      for (let n = 0; n < styles.length; ++n) {
         if (!styles[n].href || (styles[n].type !== 'text/css') || (styles[n].rel !== 'stylesheet')) continue;
         if (match_url(styles[n].href))
            return Promise.resolve(true);
      }

   } else {
      let scripts = document.getElementsByTagName('script');
      for (let n = 0; n < scripts.length; ++n)
         if (match_url(scripts[n].src))
            return Promise.resolve(true);
   }

   if (isstyle) {
      element = document.createElement("link");
      element.setAttribute("rel", "stylesheet");
      element.setAttribute("type", "text/css");
      element.setAttribute("href", url);
   } else {
      element = document.createElement("script");
      element.setAttribute("type", "text/javascript");
      element.setAttribute("src", url);
   }

   return new Promise((resolve, reject) => {
      element.onload = () => resolve(true);
      element.onerror = () => { element.remove(); reject(Error(`Fail to load ${url}`)); };
      document.head.appendChild(element);
   });
}

/** @summary Generate mask for given bit
  * @param {number} n bit number
  * @returns {Number} produced mask
  * @private */
function BIT(n) { return 1 << n; }

/** @summary Make deep clone of the object, including all sub-objects
  * @returns {object} cloned object
  * @private */
function clone(src, map, nofunc) {
   if (!src) return null;

   if (!map) {
      map = { obj: [], clones: [], nofunc: nofunc };
   } else {
      const i = map.obj.indexOf(src);
      if (i >= 0) return map.clones[i];
   }

   let arr_kind = isArrayProto(Object.prototype.toString.apply(src));

   // process normal array
   if (arr_kind == 1) {
      let tgt = [];
      map.obj.push(src);
      map.clones.push(tgt);
      for (let i = 0; i < src.length; ++i)
         if (typeof src[i] === 'object')
            tgt.push(clone(src[i], map));
         else
            tgt.push(src[i]);

      return tgt;
   }

   // process typed array
   if (arr_kind == 2) {
      let tgt = [];
      map.obj.push(src);
      map.clones.push(tgt);
      for (let i = 0; i < src.length; ++i)
         tgt.push(src[i]);

      return tgt;
   }

   let tgt = {};
   map.obj.push(src);
   map.clones.push(tgt);

   for (let k in src) {
      if (typeof src[k] === 'object')
         tgt[k] = clone(src[k], map);
      else if (!map.nofunc || (typeof src[k]!=='function'))
         tgt[k] = src[k];
   }

   return tgt;
}

// used very often - keep shortcut
const extend = Object.assign;

/** @summary Adds specific methods to the object.
  * @desc JSROOT implements some basic methods for different ROOT classes.
  * @param {object} obj - object where methods are assigned
  * @param {string} [typename] - optional typename, if not specified, obj._typename will be used
  * @private */
function addMethods(obj, typename) {
   extend(obj, getMethods(typename || obj._typename, obj));
}

/** @summary Should be used to parse JSON string produced with TBufferJSON class
  * @desc Replace all references inside object like { "$ref": "1" }
  * @param {object|string} json  object where references will be replaced
  * @returns {object} parsed object */
function parse(json) {

   if (!json) return null;

   let obj = (typeof json == 'string') ? JSON.parse(json) : json,
       map = [], newfmt = undefined;

   const unref_value = value => {
      if ((value===null) || (value===undefined)) return;

      if (typeof value === 'string') {
         if (newfmt || (value.length < 6) || (value.indexOf("$ref:") !== 0)) return;
         let ref = parseInt(value.slice(5));
         if (!Number.isInteger(ref) || (ref < 0) || (ref >= map.length)) return;
         newfmt = false;
         return map[ref];
      }

      if (typeof value !== 'object') return;

      let proto = Object.prototype.toString.apply(value);

      // scan array - it can contain other objects
      if (isArrayProto(proto) > 0) {
          for (let i = 0; i < value.length; ++i) {
             let res = unref_value(value[i]);
             if (res!==undefined) value[i] = res;
          }
          return;
      }

      let ks = Object.keys(value), len = ks.length;

      if ((newfmt!==false) && (len===1) && (ks[0]==='$ref')) {
         const ref = parseInt(value['$ref']);
         if (!Number.isInteger(ref) || (ref < 0) || (ref >= map.length)) return;
         newfmt = true;
         return map[ref];
      }

      if ((newfmt!==false) && (len>1) && (ks[0]==='$arr') && (ks[1]==='len')) {
         // this is ROOT-coded array
         let arr;
         switch (value.$arr) {
            case "Int8": arr = new Int8Array(value.len); break;
            case "Uint8": arr = new Uint8Array(value.len); break;
            case "Int16": arr = new Int16Array(value.len); break;
            case "Uint16": arr = new Uint16Array(value.len); break;
            case "Int32": arr = new Int32Array(value.len); break;
            case "Uint32": arr = new Uint32Array(value.len); break;
            case "Float32": arr = new Float32Array(value.len); break;
            case "Int64":
            case "Uint64":
            case "Float64": arr = new Float64Array(value.len); break;
            default: arr = new Array(value.len);
         }

         arr.fill((value.$arr==="Bool") ? false : 0);

         if (value.b !== undefined) {
            // base64 coding

            let atob_func = nodejs ? node_atob : window.atob;

            let buf = atob_func(value.b);

            if (arr.buffer) {
               let dv = new DataView(arr.buffer, value.o || 0),
                   len = Math.min(buf.length, dv.byteLength);
               for (let k = 0; k < len; ++k)
                  dv.setUint8(k, buf.charCodeAt(k));
            } else {
               throw new Error('base64 coding supported only for native arrays with binary data');
            }
         } else {
            // compressed coding
            let nkey = 2, p = 0;
            while (nkey<len) {
               if (ks[nkey][0]=="p") p = value[ks[nkey++]]; // position
               if (ks[nkey][0]!=='v') throw new Error('Unexpected member ' + ks[nkey] + ' in array decoding');
               let v = value[ks[nkey++]]; // value
               if (typeof v === 'object') {
                  for (let k = 0; k < v.length; ++k) arr[p++] = v[k];
               } else {
                  arr[p++] = v;
                  if ((nkey<len) && (ks[nkey][0]=='n')) {
                     let cnt = value[ks[nkey++]]; // counter
                     while (--cnt) arr[p++] = v;
                  }
               }
            }
         }

         return arr;
      }

      if ((newfmt!==false) && (len===3) && (ks[0]==='$pair') && (ks[1]==='first') && (ks[2]==='second')) {
         newfmt = true;
         let f1 = unref_value(value.first),
             s1 = unref_value(value.second);
         if (f1!==undefined) value.first = f1;
         if (s1!==undefined) value.second = s1;
         value._typename = value['$pair'];
         delete value['$pair'];
         return; // pair object is not counted in the objects map
      }

     // prevent endless loop
     if (map.indexOf(value) >= 0) return;

      // add object to object map
      map.push(value);

      // add methods to all objects, where _typename is specified
      if (value._typename) addMethods(value);

      for (let k = 0; k < len; ++k) {
         const i = ks[k],
              res = unref_value(value[i]);
         if (res !== undefined) value[i] = res;
      }
   };

   unref_value(obj);

   return obj;
}

/** @summary Parse response from multi.json request
  * @desc Method should be used to parse JSON code, produced by multi.json request of THttpServer
  * @param {string} json string to parse
  * @returns {Array} array of parsed elements */
function parseMulti(json) {
   if (!json) return null;
   let arr = parse(json);
   if (arr && arr.length)
      for (let i = 0; i < arr.length; ++i)
         arr[i] = parse(arr[i]);
   return arr;
}

/** @summary Method converts JavaScript object into ROOT-like JSON
  * @desc Produced JSON can be used in parse() again
  * When performed properly, JSON can be used in [TBufferJSON::fromJSON()]{@link https://root.cern/doc/master/classTBufferJSON.html#a2ecf0daacdad801e60b8093a404c897d} method to read data back with C++
  * @param {object} obj - JavaScript object to convert
  * @param {number} [spacing] - optional line spacing in JSON
  * @returns {string} produced JSON code */
function toJSON(obj, spacing) {
   if (!obj || typeof obj !== 'object') return "";

   let map = []; // map of stored objects

   const copy_value = value => {
      if (typeof value === "function") return undefined;

      if ((value===undefined) || (value===null) || (typeof value !== 'object')) return value;

      // typed array need to be converted into normal array, otherwise looks strange
      if (isArrayProto(Object.prototype.toString.apply(value)) > 0) {
         let arr = new Array(value.length);
         for (let i = 0; i < value.length; ++i)
            arr[i] = copy_value(value[i]);
         return arr;
      }

      // this is how reference is code
      let refid = map.indexOf(value);
      if (refid >= 0) return { $ref: refid };

      let ks = Object.keys(value), len = ks.length, tgt = {};

      if ((len == 3) && (ks[0]==='$pair') && (ks[1]==='first') && (ks[2]==='second')) {
         // special handling of pair objects which does not included into objects map
         tgt.$pair = value.$pair;
         tgt.first = copy_value(value.first);
         tgt.second = copy_value(value.second);
         return tgt;
      }

      map.push(value);

      for (let k = 0; k < len; ++k) {
         let name = ks[k];
         if (name && (name[0] != '$'))
            tgt[name] = copy_value(value[name]);
      }

      return tgt;
   };

   let tgt = copy_value(obj);

   return JSON.stringify(tgt, null, spacing);
}

/** @summary decodes URL options after '?' mark
  * @desc Following options supported ?opt1&opt2=3
  * @param {string} [url] URL string with options, document.URL will be used when not specified
  * @returns {Object} with ```.has(opt)``` and ```.get(opt,dflt)``` methods
  * @example
  * let d = decodeUrl("any?opt1&op2=3");
  * console.log(`Has opt1 ${d.has("opt1")}`);     // true
  * console.log(`Get opt1 ${d.get("opt1")}`);     // ""
  * console.log(`Get opt2 ${d.get("opt2")}`);     // "3"
  * console.log(`Get opt3 ${d.get("opt3","-")}`); // "-" */
function decodeUrl(url) {
   let res = {
      opts: {},
      has(opt) { return this.opts[opt] !== undefined; },
      get(opt,dflt) { let v = this.opts[opt]; return v!==undefined ? v : dflt; }
   };

   if (!url || (typeof url !== 'string')) {
      if (settings.IgnoreUrlOptions || (typeof document === 'undefined')) return res;
      url = document.URL;
   }
   res.url = url;

   let p1 = url.indexOf("?");
   if (p1 < 0) return res;
   url = decodeURI(url.slice(p1+1));

   while (url.length > 0) {

      // try to correctly handle quotes in the URL
      let pos = 0, nq = 0, eq = -1, firstq = -1;
      while ((pos < url.length) && ((nq!==0) || ((url[pos]!=="&") && (url[pos]!=="#")))) {
         switch (url[pos]) {
            case "'": if (nq >= 0) nq = (nq+1)%2; if (firstq < 0) firstq = pos; break;
            case '"': if (nq <= 0) nq = (nq-1)%2; if (firstq < 0) firstq = pos; break;
            case '=': if ((firstq < 0) && (eq < 0)) eq = pos; break;
         }
         pos++;
      }

      if ((eq < 0) && (firstq < 0)) {
         res.opts[url.slice(0,pos)] = "";
      } if (eq > 0) {
         let val = url.slice(eq+1, pos);
         if (((val[0]==="'") || (val[0]==='"')) && (val[0]===val[val.length-1])) val = val.slice(1, val.length-1);
         res.opts[url.slice(0,eq)] = val;
      }

      if ((pos >= url.length) || (url[pos] == '#')) break;

      url = url.slice(pos+1);
   }

   return res;
}

/** @summary Find function with given name
  * @private */
function findFunction(name) {
   if (typeof name === 'function') return name;
   if (typeof name !== 'string') return null;
   let names = name.split('.'), elem = globalThis;

   for (let n = 0; elem && (n < names.length); ++n)
      elem = elem[names[n]];

   return (typeof elem == 'function') ? elem : null;
}

/** @summary Method to create http request
  * @private */
function createHttpRequest(url, kind, user_accept_callback, user_reject_callback) {
   let xhr = nodejs ? new node_xhr2() : new XMLHttpRequest();

   xhr.http_callback = (typeof user_accept_callback == 'function') ? user_accept_callback.bind(xhr) : function() {};
   xhr.error_callback = (typeof user_reject_callback == 'function') ? user_reject_callback.bind(xhr) : function(err) { console.warn(err.message); this.http_callback(null); }.bind(xhr);

   if (!kind) kind = "buf";

   let method = "GET", is_async = true, p = kind.indexOf(";sync");
   if (p > 0) { kind = kind.slice(0,p); is_async = false; }
   switch (kind) {
      case "head": method = "HEAD"; break;
      case "posttext": method = "POST"; kind = "text"; break;
      case "postbuf":  method = "POST"; kind = "buf"; break;
      case "post":
      case "multi": method = "POST"; break;
   }

   xhr.kind = kind;

   if (settings.HandleWrongHttpResponse && (method == "GET") && (typeof xhr.addEventListener === 'function'))
      xhr.addEventListener("progress", function(oEvent) {
         if (oEvent.lengthComputable && this.expected_size && (oEvent.loaded > this.expected_size)) {
            this.did_abort = true;
            this.abort();
            this.error_callback(Error('Server sends more bytes ' + oEvent.loaded + ' than expected ' + this.expected_size + '. Abort I/O operation'), 598);
         }
      }.bind(xhr));

   xhr.onreadystatechange = function() {

      if (this.did_abort) return;

      if ((this.readyState === 2) && this.expected_size) {
         let len = parseInt(this.getResponseHeader("Content-Length"));
         if (Number.isInteger(len) && (len > this.expected_size) && !settings.HandleWrongHttpResponse) {
            this.did_abort = true;
            this.abort();
            return this.error_callback(Error('Server response size ' + len + ' larger than expected ' + this.expected_size + '. Abort I/O operation'), 599);
         }
      }

      if (this.readyState != 4) return;

      if ((this.status != 200) && (this.status != 206) && !browser.qt5 &&
          // in these special cases browsers not always set status
          !((this.status == 0) && ((url.indexOf("file://")==0) || (url.indexOf("blob:")==0)))) {
            return this.error_callback(Error('Fail to load url ' + url), this.status);
      }

      if (this.nodejs_checkzip && (this.getResponseHeader("content-encoding") == "gzip"))
         // special handling of gzipped JSON objects in Node.js
         return import('zlib').then(handle => {
             let res = handle.unzipSync(Buffer.from(this.response)),
                 obj = JSON.parse(res); // zlib returns Buffer, use JSON to parse it
            return this.http_callback(parse(obj));
         });

      switch(this.kind) {
         case "xml": return this.http_callback(this.responseXML);
         case "text": return this.http_callback(this.responseText);
         case "object": return this.http_callback(parse(this.responseText));
         case "multi": return this.http_callback(parseMulti(this.responseText));
         case "head": return this.http_callback(this);
      }

      // if no response type is supported, return as text (most probably, will fail)
      if (this.responseType === undefined)
         return this.http_callback(this.responseText);

      if ((this.kind == "bin") && ('byteLength' in this.response)) {
         // if string representation in requested - provide it

         let filecontent = "", u8Arr = new Uint8Array(this.response);
         for (let i = 0; i < u8Arr.length; ++i)
            filecontent += String.fromCharCode(u8Arr[i]);

         return this.http_callback(filecontent);
      }

      this.http_callback(this.response);
   };

   xhr.open(method, url, is_async);

   if ((kind == "bin") || (kind == "buf"))
      xhr.responseType = 'arraybuffer';

   if (nodejs && (method == "GET") && (kind === "object") && (url.indexOf('.json.gz') > 0)) {
      xhr.nodejs_checkzip = true;
      xhr.responseType = 'arraybuffer';
   }

   return xhr;
}

/** @summary Submit asynchronoues http request
  * @desc Following requests kind can be specified:
  *    - "bin" - abstract binary data, result as string
  *    - "buf" - abstract binary data, result as ArrayBuffer (default)
  *    - "text" - returns req.responseText
  *    - "object" - returns parse(req.responseText)
  *    - "multi" - returns correctly parsed multi.json request
  *    - "xml" - returns req.responseXML
  *    - "head" - returns request itself, uses "HEAD" request method
  *    - "post" - creates post request, submits req.send(post_data)
  *    - "postbuf" - creates post request, expectes binary data as response
  * @param {string} url - URL for the request
  * @param {string} kind - kind of requested data
  * @param {string} [post_data] - data submitted with post kind of request
  * @returns {Promise} Promise for requested data, result type depends from the kind
  * @example
  * httpRequest("https://root.cern/js/files/thstack.json.gz", "object")
  *       .then(obj => console.log(`Get object of type ${obj._typename}`))
  *       .catch(err => console.error(err.message)); */
function httpRequest(url, kind, post_data) {
   return new Promise((accept, reject) => {
      let xhr = createHttpRequest(url, kind, accept, reject);
      xhr.send(post_data || null);
   });
}

/** @summary Create some ROOT classes
  * @desc Supported classes: `TObject`, `TNamed`, `TList`, `TAxis`, `TLine`, `TText`, `TLatex`, `TPad`, `TCanvas`
  * @param {string} typename - ROOT class name
  * @example
  * import { create } from 'path_to_jsroot/modules/core.mjs';
  * let obj = create("TNamed");
  * obj.fName = "name";
  * obj.fTitle = "title"; */
function create(typename, target) {
   let obj = target || {};

   switch (typename) {
      case 'TObject':
          extend(obj, { fUniqueID: 0, fBits: 0 });
          break;
      case 'TNamed':
         extend(obj, { fUniqueID: 0, fBits: 0, fName: "", fTitle: "" });
         break;
      case 'TList':
      case 'THashList':
         extend(obj, { name: typename, arr: [], opt: [] });
         break;
      case 'TAttAxis':
         extend(obj, { fNdivisions: 510, fAxisColor: 1,
                       fLabelColor: 1, fLabelFont: 42, fLabelOffset: 0.005, fLabelSize: 0.035, fTickLength: 0.03,
                       fTitleOffset: 1, fTitleSize: 0.035, fTitleColor: 1, fTitleFont : 42 });
         break;
      case 'TAxis':
         create("TNamed", obj);
         create("TAttAxis", obj);
         extend(obj, { fNbins: 1, fXmin: 0, fXmax: 1, fXbins : [], fFirst: 0, fLast: 0,
                       fBits2: 0, fTimeDisplay: false, fTimeFormat: "", fLabels: null, fModLabs: null });
         break;
      case 'TAttLine':
         extend(obj, { fLineColor: 1, fLineStyle: 1, fLineWidth: 1 });
         break;
      case 'TAttFill':
         extend(obj, { fFillColor: 0, fFillStyle: 0 } );
         break;
      case 'TAttMarker':
         extend(obj, { fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1. });
         break;
      case 'TLine':
         create("TObject", obj);
         create("TAttLine", obj);
         extend(obj, { fX1: 0, fX2: 1, fY1: 0, fY2: 1 });
         break;
      case 'TBox':
         create("TObject", obj);
         create("TAttLine", obj);
         create("TAttFill", obj);
         extend(obj, { fX1: 0, fX2: 1, fY1: 0, fY2: 1 });
         break;
      case 'TPave':
         create("TBox", obj);
         extend(obj, { fX1NDC : 0., fY1NDC: 0, fX2NDC: 1, fY2NDC: 1,
                       fBorderSize: 0, fInit: 1, fShadowColor: 1,
                       fCornerRadius: 0, fOption: "brNDC", fName: "title" });
         break;
      case 'TAttText':
         extend(obj, { fTextAngle: 0, fTextSize: 0, fTextAlign: 22, fTextColor: 1, fTextFont: 42});
         break;
      case 'TPaveText':
         create("TPave", obj);
         create("TAttText", obj);
         extend(obj, { fLabel: "", fLongest: 27, fMargin: 0.05, fLines: create("TList") });
         break;
      case 'TPaveStats':
         create("TPaveText", obj);
         extend(obj, { fFillColor: gStyle.fStatColor, fFillStyle: gStyle.fStatStyle,
                       fTextFont: gStyle.fStatFont, fTextSize: gStyle.fStatFontSize, fTextColor: gStyle.fStatTextColor,
                       fBorderSize: gStyle.fStatBorderSize,
                       fOptFit: 0, fOptStat: 0, fFitFormat: "", fStatFormat: "", fParent: null });
         break;
      case 'TLegend':
         create("TPave", obj);
         create("TAttText", obj);
         extend(obj, { fColumnSeparation: 0, fEntrySeparation: 0.1, fMargin: 0.25, fNColumns: 1, fPrimitives: create("TList"),
                       fBorderSize: gStyle.fLegendBorderSize, fTextFont: gStyle.fLegendFont, fTextSize: gStyle.fLegendTextSize, fFillColor: gStyle.fLegendFillColor });
         break;
      case 'TLegendEntry':
         create("TObject", obj);
         create("TAttText", obj);
         create("TAttLine", obj);
         create("TAttFill", obj);
         create("TAttMarker", obj);
         extend(obj, { fLabel: "", fObject: null, fOption: "" });
         break;
      case 'TText':
         create("TNamed", obj);
         create("TAttText", obj);
         extend(obj, { fLimitFactorSize: 3, fOriginSize: 0.04 });
         break;
      case 'TLatex':
         create("TText", obj);
         create("TAttLine", obj);
         extend(obj, { fX: 0, fY: 0 });
         break;
      case 'TObjString':
         create("TObject", obj);
         extend(obj, { fString: "" });
         break;
      case 'TH1':
         create("TNamed", obj);
         create("TAttLine", obj);
         create("TAttFill", obj);
         create("TAttMarker", obj);
         extend(obj, { fBits: 8, fNcells: 0,
                       fXaxis: create("TAxis"), fYaxis: create("TAxis"), fZaxis: create("TAxis"),
                       fFillColor: gStyle.fHistFillColor, fFillStyle: gStyle.fHistFillStyle,
                       fLineColor: gStyle.fHistLineColor, fLineStyle: gStyle.fHistLineStyle, fLineWidth: gStyle.fHistLineWidth,
                       fBarOffset: 0, fBarWidth: 1000, fEntries: 0.,
                       fTsumw: 0., fTsumw2: 0., fTsumwx: 0., fTsumwx2: 0.,
                       fMaximum: -1111., fMinimum: -1111, fNormFactor: 0., fContour: [],
                       fSumw2: [], fOption: "", fFunctions: create("TList"),
                       fBufferSize: 0, fBuffer: [], fBinStatErrOpt: 0, fStatOverflows: 2 });
         break;
      case 'TH1I':
      case 'TH1L64':
      case 'TH1F':
      case 'TH1D':
      case 'TH1S':
      case 'TH1C':
         create("TH1", obj);
         obj.fArray = [];
         break;
      case 'TH2':
         create("TH1", obj);
         extend(obj, { fScalefactor: 1., fTsumwy: 0.,  fTsumwy2: 0, fTsumwxy: 0 });
         break;
      case 'TH2I':
      case 'TH2L64':
      case 'TH2F':
      case 'TH2D':
      case 'TH2S':
      case 'TH2C':
         create("TH2", obj);
         obj.fArray = [];
         break;
      case 'TH3':
         create("TH1", obj);
         extend(obj, { fTsumwy: 0.,  fTsumwy2: 0, fTsumwz: 0.,  fTsumwz2: 0, fTsumwxy: 0, fTsumwxz: 0, fTsumwyz: 0 });
         break;
      case 'TH3I':
      case 'TH3L64':
      case 'TH3F':
      case 'TH3D':
      case 'TH3S':
      case 'TH3C':
         create("TH3", obj);
         obj.fArray = [];
         break;
      case 'THStack':
         create("TNamed", obj);
         extend(obj, { fHists: create("TList"), fHistogram: null, fMaximum: -1111, fMinimum: -1111 });
         break;
      case 'TGraph':
         create("TNamed", obj);
         create("TAttLine", obj);
         create("TAttFill", obj);
         create("TAttMarker", obj);
         extend(obj, { fFunctions: create("TList"), fHistogram: null,
                       fMaxSize: 0, fMaximum: -1111, fMinimum: -1111, fNpoints: 0, fX: [], fY: [] });
         break;
      case 'TGraphAsymmErrors':
         create("TGraph", obj);
         extend(obj, { fEXlow: [], fEXhigh: [], fEYlow: [], fEYhigh: []});
         break;
      case 'TMultiGraph':
         create("TNamed", obj);
         extend(obj, { fFunctions: create("TList"), fGraphs: create("TList"),
                       fHistogram: null, fMaximum: -1111, fMinimum: -1111 });
         break;
      case 'TGraphPolargram':
         create("TNamed", obj);
         create("TAttText", obj);
         create("TAttLine", obj);
         extend(obj, { fRadian: true, fDegree: false, fGrad: false, fPolarLabelColor: 1, fRadialLabelColor: 1,
                       fAxisAngle: 0, fPolarOffset: 0.04, fPolarTextSize: 0.04, fRadialOffset: 0.025, fRadialTextSize: 0.035,
                       fRwrmin: 0, fRwrmax: 1, fRwtmin: 0, fRwtmax: 2*Math.PI, fTickpolarSize: 0.02,
                       fPolarLabelFont: 62, fRadialLabelFont: 62, fCutRadial: 0, fNdivRad: 508, fNdivPol: 508 });
         break;
      case 'TPolyLine':
         create("TObject", obj);
         create("TAttLine", obj);
         create("TAttFill", obj);
         extend(obj, { fLastPoint: -1, fN: 0, fOption: "", fX: null, fY: null });
         break;
      case 'TGaxis':
         create("TLine", obj);
         create("TAttText", obj);
         extend(obj, { fChopt: "", fFunctionName: "", fGridLength: 0,
                       fLabelColor: 1, fLabelFont: 42, fLabelOffset: 0.005, fLabelSize: 0.035,
                       fName: "", fNdiv: 12, fTickSize: 0.02, fTimeFormat: "",
                       fTitle: "", fTitleOffset: 1, fTitleSize: 0.035,
                       fWmax: 100, fWmin: 0 });
         break;
      case 'TAttPad':
         extend(obj, { fLeftMargin: gStyle.fPadLeftMargin,
                       fRightMargin: gStyle.fPadRightMargin,
                       fBottomMargin: gStyle.fPadBottomMargin,
                       fTopMargin: gStyle.fPadTopMargin,
                       fXfile: 2, fYfile: 2, fAfile: 1, fXstat: 0.99, fYstat: 0.99, fAstat: 2,
                       fFrameFillColor: gStyle.fFrameFillColor,
                       fFrameFillStyle: gStyle.fFrameFillStyle,
                       fFrameLineColor: gStyle.fFrameLineColor,
                       fFrameLineWidth: gStyle.fFrameLineWidth,
                       fFrameLineStyle: gStyle.fFrameLineStyle,
                       fFrameBorderSize: gStyle.fFrameBorderSize,
                       fFrameBorderMode: gStyle.fFrameBorderMode });
         break;
      case 'TPad':
         create("TObject", obj);
         create("TAttLine", obj);
         create("TAttFill", obj);
         create("TAttPad", obj);
         extend(obj, { fFillColor: gStyle.fPadColor, fFillStyle: 1001,
                       fX1: 0, fY1: 0, fX2: 1, fY2: 1, fXtoAbsPixelk: 1, fXtoPixelk: 1,
                       fXtoPixel: 1, fYtoAbsPixelk: 1, fYtoPixelk: 1, fYtoPixel: 1,
                       fUtoAbsPixelk: 1, fUtoPixelk: 1, fUtoPixel: 1, fVtoAbsPixelk: 1,
                       fVtoPixelk: 1, fVtoPixel: 1, fAbsPixeltoXk: 1, fPixeltoXk: 1,
                       fPixeltoX: 1, fAbsPixeltoYk: 1, fPixeltoYk: 1, fPixeltoY: 1,
                       fXlowNDC: 0, fYlowNDC: 0, fXUpNDC: 0, fYUpNDC: 0, fWNDC: 1, fHNDC: 1,
                       fAbsXlowNDC: 0, fAbsYlowNDC: 0, fAbsWNDC: 1, fAbsHNDC: 1,
                       fUxmin: 0, fUymin: 0, fUxmax: 0, fUymax: 0, fTheta: 30, fPhi: 30, fAspectRatio: 0,
                       fNumber: 0, fLogx: gStyle.fOptLogx, fLogy: gStyle.fOptLogy, fLogz: gStyle.fOptLogz,
                       fTickx: gStyle.fPadTickX, fTicky: gStyle.fPadTickY,
                       fPadPaint: 0, fCrosshair: 0, fCrosshairPos: 0, fBorderSize: 2,
                       fBorderMode: 0, fModified: false,
                       fGridx: gStyle.fPadGridX, fGridy: gStyle.fPadGridY,
                       fAbsCoord: false, fEditable: true, fFixedAspectRatio: false,
                       fPrimitives: create("TList"), fExecs: null,
                       fName: "pad", fTitle: "canvas" });

         break;
      case 'TAttCanvas':
         extend(obj, { fXBetween: 2, fYBetween: 2, fTitleFromTop: 1.2,
                       fXdate: 0.2, fYdate: 0.3, fAdate: 1 });
         break;
      case 'TCanvas':
         create("TPad", obj);
         extend(obj, { fFillColor: gStyle.fCanvasColor, fFillStyle: 1001,
                       fNumPaletteColor: 0, fNextPaletteColor: 0, fDISPLAY: "$DISPLAY",
                       fDoubleBuffer: 0, fRetained: true, fXsizeUser: 0,
                       fYsizeUser: 0, fXsizeReal: 20, fYsizeReal: 10,
                       fWindowTopX: 0, fWindowTopY: 0, fWindowWidth: 0, fWindowHeight: 0,
                       fCw: 500, fCh: 300, fCatt: create("TAttCanvas"),
                       kMoveOpaque: true, kResizeOpaque: true, fHighLightColor: 5,
                       fBatch: true, kShowEventStatus: false, kAutoExec: true, kMenuBar: true });
         break;
      case 'TGeoVolume':
         create("TNamed", obj);
         create("TAttLine", obj);
         create("TAttFill", obj);
         extend(obj, { fGeoAtt: 0, fFinder: null, fMedium: null, fNodes: null, fNtotal: 0, fNumber: 0, fRefCount: 0, fShape: null, fVoxels: null });
         break;
      case 'TGeoNode':
         create("TNamed", obj);
         extend(obj, { fGeoAtt: 0, fMother: null, fNovlp: 0, fNumber: 0, fOverlaps: null, fVolume: null });
         break;
      case 'TGeoNodeMatrix':
         create("TGeoNode", obj);
         extend(obj, { fMatrix: null });
         break;
      case 'TGeoTrack':
         create("TObject", obj);
         create("TAttLine", obj);
         create("TAttMarker", obj);
         extend(obj, { fGeoAtt: 0, fNpoints: 0, fPoints: [] });
         break;
   }

   obj._typename = typename;
   addMethods(obj, typename);
   return obj;
}

/** @summary Create histogram object of specified type
  * @param {string} typename - histogram typename like 'TH1I' or 'TH2F'
  * @param {number} nbinsx - number of bins on X-axis
  * @param {number} [nbinsy] - number of bins on Y-axis (for 2D/3D histograms)
  * @param {number} [nbinsz] - number of bins on Z-axis (for 3D histograms)
  * @returns {Object} created histogram object
  * @example
  * let h1 = createHistogram("TH1I", 20);
  * h1.fName = "Hist1";
  * h1.fTitle = "Histogram title";
  * h1.fXaxis.fTitle = "xaxis";
  * h1.fYaxis.fTitle = "yaxis";
  * h1.fXaxis.fLabelSize = 0.02; */
function createHistogram(typename, nbinsx, nbinsy, nbinsz) {
   let histo = create(typename);
   if (!histo.fXaxis || !histo.fYaxis || !histo.fZaxis) return null;
   histo.fName = "hist"; histo.fTitle = "title";
   if (nbinsx) extend(histo.fXaxis, { fNbins: nbinsx, fXmin: 0, fXmax: nbinsx });
   if (nbinsy) extend(histo.fYaxis, { fNbins: nbinsy, fXmin: 0, fXmax: nbinsy });
   if (nbinsz) extend(histo.fZaxis, { fNbins: nbinsz, fXmin: 0, fXmax: nbinsz });
   switch (parseInt(typename[2])) {
      case 1: if (nbinsx) histo.fNcells = nbinsx+2; break;
      case 2: if (nbinsx && nbinsy) histo.fNcells = (nbinsx+2) * (nbinsy+2); break;
      case 3: if (nbinsx && nbinsy && nbinsz) histo.fNcells = (nbinsx+2) * (nbinsy+2) * (nbinsz+2); break;
   }
   if (histo.fNcells > 0) {
      switch (typename[3]) {
         case "C": histo.fArray = new Int8Array(histo.fNcells); break;
         case "S": histo.fArray = new Int16Array(histo.fNcells); break;
         case "I": histo.fArray = new Int32Array(histo.fNcells); break;
         case "F": histo.fArray = new Float32Array(histo.fNcells); break;
         case "L":
         case "D": histo.fArray = new Float64Array(histo.fNcells); break;
         default: histo.fArray = new Array(histo.fNcells);
      }
      histo.fArray.fill(0);
   }
   return histo;
}

/** @summary Creates TPolyLine object
  * @param {number} npoints - number of points
  * @param {boolean} [use_int32] - use Int32Array type for points, default is Float32Array */
function createTPolyLine(npoints, use_int32) {
   let poly = create("TPolyLine");
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
function createTGraph(npoints, xpts, ypts) {
   let graph = extend(create("TGraph"), { fBits: 0x408, fName: "graph", fTitle: "title" });

   if (npoints > 0) {
      graph.fMaxSize = graph.fNpoints = npoints;

      const usex = (typeof xpts == 'object') && (xpts.length === npoints);
      const usey = (typeof ypts == 'object') && (ypts.length === npoints);

      for (let i = 0; i < npoints; ++i) {
         graph.fX.push(usex ? xpts[i] : i/npoints);
         graph.fY.push(usey ? ypts[i] : i/npoints);
      }
   }

   return graph;
}

/** @summary Creates THStack object
  * @desc As arguments one could specify any number of histograms objects
  * @example
  * let nbinsx = 20;
  * let h1 = createHistogram("TH1F", nbinsx);
  * let h2 = createHistogram("TH1F", nbinsx);
  * let h3 = createHistogram("TH1F", nbinsx);
  * let stack = createTHStack(h1, h2, h3); */
function createTHStack() {
   let stack = create("THStack");
   for (let i = 0; i < arguments.length; ++i)
      stack.fHists.Add(arguments[i], "");
   return stack;
}

/** @summary Creates TMultiGraph object
  * @desc As arguments one could specify any number of TGraph objects
  * @example
  * let gr1 = createTGraph(100);
  * let gr2 = createTGraph(100);
  * let gr3 = createTGraph(100);
  * let mgr = createTMultiGraph(gr1, gr2, gr3); */
function createTMultiGraph() {
   let mgraph = create("TMultiGraph");
   for (let i = 0; i < arguments.length; ++i)
       mgraph.fGraphs.Add(arguments[i], "");
   return mgraph;
}

/** @summary variable used to keep methods for known classes
  * @private */
const methodsCache = {};

/** @summary Returns methods for given typename
  * @private */
function getMethods(typename, obj) {

   let m = methodsCache[typename],
       has_methods = (m!==undefined);

   if (!has_methods) m = {};

   // Due to binary I/O such TObject methods may not be set for derived classes
   // Therefore when methods requested for given object, check also that basic methods are there
   if ((typename=="TObject") || (typename=="TNamed") || (obj && (obj.fBits!==undefined)))
      if (typeof m.TestBit === 'undefined') {
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
         let line = create("TLatex");
         line.fTitle = txt;
         line.fTextAlign = this.fTextAlign;
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

      m.GetParName = function(n) {
         if (this.fParams && this.fParams.fParNames) return this.fParams.fParNames[n];
         if (this.fFormula && this.fFormula.fParams) {
            for (let k=0;k<this.fFormula.fParams.length;++k)
               if(this.fFormula.fParams[k].second == n)
                  return this.fFormula.fParams[k].first;
         }
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
         let i = 0, j = this.fNpoints - 1, x = this.fX, y = this.fY, oddNodes = false;

         for (; i < this.fNpoints; ++i) {
            if ((y[i]<yp && y[j]>=yp) || (y[j]<yp && y[i]>=yp)) {
               if (x[i]+(yp-y[i])/(y[j]-y[i])*(x[j]-x[i])<xp) {
                  oddNodes = !oddNodes;
               }
            }
            j=i;
         }

         return oddNodes;
      }
   }

   if (typename.indexOf("TH1") == 0 || typename.indexOf("TH2") == 0 || typename.indexOf("TH3") == 0) {
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
      }
      m.setBinContent = function(bin, content) {
         // Set bin content - only trivial case, without expansion
         this.fEntries++;
         this.fTsumw = 0;
         if ((bin>=0) && (bin<this.fArray.length))
            this.fArray[bin] = content;
      }
   }

   if (typename.indexOf("TH1") == 0) {
      m.getBin = function(x) { return x; }
      m.getBinContent = function(bin) { return this.fArray[bin]; }
      m.Fill = function(x, weight) {
         let axis = this.fXaxis,
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
         let axis1 = this.fXaxis, axis2 = this.fYaxis,
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
      m.getBinContent = function(x, y, z) { return this.fArray[this.getBin(x, y, z)]; }
      m.Fill = function(x, y, z, weight) {
         let axis1 = this.fXaxis, axis2 = this.fYaxis, axis3 = this.fZaxis,
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
            let bin = this.getBin(x, y);
            if (bin < 0 || bin >= this.fNcells) return 0;
            if (this.fBinEntries[bin] < 1e-300) return 0;
            if (!this.fArray) return 0;
            return this.fArray[bin]/this.fBinEntries[bin];
         }
         m.getBinEntries = function(x, y) {
            let bin = this.getBin(x, y);
            if (bin < 0 || bin >= this.fNcells) return 0;
            return this.fBinEntries[bin];
         }
      } else {
         m.getBin = function(x) { return x; }
         m.getBinContent = function(bin) {
            if (bin < 0 || bin >= this.fNcells) return 0;
            if (this.fBinEntries[bin] < 1e-300) return 0;
            if (!this.fArray) return 0;
            return this.fArray[bin]/this.fBinEntries[bin];
         }
      }
      m.getBinEffectiveEntries = function(bin) {
         if (bin < 0 || bin >= this.fNcells) return 0;
         let sumOfWeights = this.fBinEntries[bin];
         if ( !this.fBinSumw2 || this.fBinSumw2.length != this.fNcells) {
            // this can happen  when reading an old file
            return sumOfWeights;
         }
         let sumOfWeightsSquare = this.fBinSumw2[bin];
         return (sumOfWeightsSquare > 0) ? sumOfWeights * sumOfWeights / sumOfWeightsSquare : 0;
      }
      m.getBinError = function(bin) {
         if (bin < 0 || bin >= this.fNcells) return 0;
         let cont = this.fArray[bin],               // sum of bin w *y
             sum  = this.fBinEntries[bin],          // sum of bin weights
             err2 = this.fSumw2[bin],               // sum of bin w * y^2
             neff = this.getBinEffectiveEntries(bin);  // (sum of w)^2 / (sum of w^2)
         if (sum < 1e-300) return 0;                  // for empty bins
         const EErrorType = { kERRORMEAN: 0, kERRORSPREAD: 1, kERRORSPREADI: 2, kERRORSPREADG: 3 };
         // case the values y are gaussian distributed y +/- sigma and w = 1/sigma^2
         if (this.fErrorMode === EErrorType.kERRORSPREADG)
            return 1.0/Math.sqrt(sum);
         // compute variance in y (eprim2) and standard deviation in y (eprim)
         let contsum = cont/sum, eprim = Math.sqrt(Math.abs(err2/sum - contsum*contsum));
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
         return eprim/Math.sqrt(neff);
      }
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

   if (typename.indexOf("ROOT::Math::LorentzVector") === 0) {
      m.Px = m.X = function() { return this.fCoordinates.Px(); }
      m.Py = m.Y = function() { return this.fCoordinates.Py(); }
      m.Pz = m.Z = function() { return this.fCoordinates.Pz(); }
      m.E = m.T = function() { return this.fCoordinates.E(); }
      m.M2 = function() { return this.fCoordinates.M2(); }
      m.M = function() { return this.fCoordinates.M(); }
      m.R = m.P = function() { return this.fCoordinates.R(); }
      m.P2 = function() { return this.P() * this.P(); }
      m.Pt = m.pt = function() { return Math.sqrt(this.P2()); }
      m.Phi = m.phi = function() { return Math.atan2(this.fCoordinates.Py(), this.fCoordinates.Px()); }
      m.Eta = m.eta = function() { return Math.atanh(this.Pz()/this.P()); }
   }

   if (typename.indexOf("ROOT::Math::PxPyPzE4D") === 0) {
      m.Px = m.X = function() { return this.fX; }
      m.Py = m.Y = function() { return this.fY; }
      m.Pz = m.Z = function() { return this.fZ; }
      m.E = m.T = function() { return this.fT; }
      m.P2 = function() { return this.fX*this.fX + this.fY*this.fY + this.fZ*this.fZ; }
      m.R = m.P = function() { return Math.sqrt(this.P2()); }
      m.Mag2 = m.M2 = function() { return this.fT*this.fT - this.fX*this.fX - this.fY*this.fY - this.fZ*this.fZ; }
      m.Mag = m.M = function() {
         let mm = this.M2();
         if (mm >= 0) return Math.sqrt(mm);
         return -Math.sqrt(-mm);
      }
      m.Perp2 = m.Pt2 = function() { return this.fX*this.fX + this.fY*this.fY;}
      m.Pt = m.pt = function() { return Math.sqrt(this.P2()); }
      m.Phi = m.phi = function() { return Math.atan2(this.fY, this.fX); }
      m.Eta = m.eta = function() { return Math.atanh(this.Pz/this.P()); }
   }

   methodsCache[typename] = m;
   return m;
}

/** @summary Add methods for specified type.
  * @desc Will be automatically applied when decoding JSON string
  * @private */
function registerMethods(typename, m) {
   methodsCache[typename] = m;
}

/** @summary Returns true if object represents basic ROOT collections
  * @desc Checks if type is TList or TObjArray or TClonesArray or TMap or THashList
  * @param {object} lst - object to check
  * @param {string} [typename] - or just typename to check
  * @private */
function isRootCollection(lst, typename) {
   if (lst && (typeof lst === 'object')) {
      if ((lst.$kind === "TList") || (lst.$kind === "TObjArray")) return true;
      if (!typename) typename = lst._typename;
   }
   if (!typename) return false;
   return (typename === 'TList') || (typename === 'THashList') || (typename === 'TMap') ||
          (typename === 'TObjArray') || (typename === 'TClonesArray');
}

/** @summary Check if object is a Promise
  * @private */
function isPromise(obj) {
   return obj && (typeof obj == 'object') && (typeof obj.then == 'function');
}

/** @summary Ensure global JSROOT and v6 support methods
  * @private */
function _ensureJSROOT() {
   let pr = globalThis.JSROOT ? Promise.resolve(true) : loadScript(source_dir + 'scripts/JSRoot.core.js');

   return pr.then(() => {
      if (globalThis.JSROOT?._complete_loading)
         return globalThis.JSROOT._complete_loading();
   }).then(() => globalThis.JSROOT);
}

export { version_id, version_date, version, source_dir, isNodeJs, isBatchMode, setBatchMode,
         browser, internals, constants, settings, gStyle,
         isArrayProto, getDocument, BIT, clone, addMethods, parse, parseMulti, toJSON,
         decodeUrl, findFunction, createHttpRequest, httpRequest, loadScript, injectCode,
         create, createHistogram, createTPolyLine, createTGraph, createTHStack, createTMultiGraph,
         getMethods, registerMethods, isRootCollection, isPromise, _ensureJSROOT };
