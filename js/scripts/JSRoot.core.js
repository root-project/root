(function (factory) {
typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
typeof define === 'function' && define.amd ? define(['exports'], factory) : factory({});

})((function (exports) {

'use strict';

let sync_promises = [], getHPainter, _openui5args = {},
    jsrp = null, geo = null; // old JSROOT.Painter and JSROOT.GEO handles

function _sync() {
   let arr;
   if (sync_promises) {
      arr = sync_promises;
      sync_promises = [];
   }

   return Promise.all(arr).then(() => {

      if (globalThis.JSROOT) {
         if (getHPainter) globalThis.JSROOT.hpainter = getHPainter();
         if (globalThis.JSROOT.batch_mode && globalThis.JSROOT.isBatchMode && !globalThis.JSROOT.isBatchMode())
            globalThis.JSROOT.setBatchMode(true);
      }

      return globalThis.JSROOT;
   });
}


function loadPainter() {
   if (jsrp) return Promise.resolve(jsrp);
   return Promise.all([import('../modules/d3.mjs'), import('../modules/draw.mjs'), import('../modules/base/colors.mjs'),
                       import('../modules/base/BasePainter.mjs'), import('../modules/base/ObjectPainter.mjs'),
                       import('../modules/base/TAttLineHandler.mjs'), import('../modules/gui/menu.mjs')]).then(res => {
      if (jsrp) return jsrp;
      globalThis.d3 = res[0]; // assign global d3
      jsrp = Object.assign({}, res[1], res[2], res[3], res[4], res[5], res[6]);
      globalThis.JSROOT.Painter = jsrp;
      globalThis.JSROOT.BasePainter = res[3].BasePainter;
      globalThis.JSROOT.ObjectPainter = res[4].ObjectPainter;
      return jsrp;
   });
}

exports.httpRequest = function(...args) {
   return _sync().then(() => import('../modules/core.mjs')).then(handle => handle.httpRequest(...args));
}

exports.loadScript = function(...args) {
   return _sync().then(() => import('../modules/core.mjs')).then(handle => handle.loadScript(...args));
}

exports.buildGUI = function(...args) {
   return _sync().then(() => import('../modules/gui.mjs')).then(handle => handle.buildGUI(...args));
}

exports.openFile = function(...args) {
   return _sync().then(() => import('../modules/io.mjs')).then(handle => handle.openFile(...args));
}


/** @summary Old v6 method to load JSROOT functionality
  * @desc
  * Following components can be specified
  *    - 'io'     TFile functionality
  *    - 'tree'   TTree support
  *    - 'painter' d3.js plus basic painting functions
  *    - 'geom'   TGeo support
  *    - 'math'   some methods from TMath class
  *    - 'hierarchy' hierarchy browser
  *    - 'openui5' OpenUI5 and related functionality
  * @param {Array|string} req - list of required components (as array or string separated by semicolon)
  * @returns {Promise} with array of requirements (or single element) */

function v6_require(need) {
   if (!need)
      return Promise.resolve(null);

   if (typeof need == "string") need = need.split(";");

   need.forEach((name,indx) => {
      if ((name.indexOf("load:")==0) || (name.indexOf("user:")==0))
         need[indx] = name.slice(5);
      else if (name == "2d")
         need[indx] = "painter";
      else if ((name == "jq2d") || (name == "jq"))
         need[indx] = "hierarchy";
      else if (name == "v6")
         need[indx] = "gpad";
      else if (name == "v7")
         need[indx] = "v7gpad";
   });

   let arr = [];

   need.forEach(name => {
      if (name == "hist")
         arr.push(Promise.all([import("../modules/hist/TH1Painter.mjs"), import("../modules/hist/TH2Painter.mjs"), import("../modules/hist/THStackPainter.mjs")]).then(arr => {
            // copy hist painter objects into JSROOT
            Object.assign(globalThis.JSROOT, arr[0], arr[1], arr[2]);
         }));
      else if (name == "more")
         arr.push(import("../modules/draw/more.mjs"));
      else if (name == "gpad")
         arr.push(loadPainter().then(() => Promise.all([import("../modules/gpad/TAxisPainter.mjs"), import("../modules/gpad/TPadPainter.mjs"), import("../modules/gpad/TCanvasPainter.mjs")])).then(arr => {
            // copy all classes
            Object.assign(globalThis.JSROOT, arr[0], arr[1], arr[2]);
            if (jsrp) jsrp.ensureTCanvas = arr[2].ensureTCanvas;
            return globalThis.JSROOT;
         }));
      else if (name == "v7gpad")
         arr.push(loadPainter().then(() => Promise.all([import("../modules/gpad/RAxisPainter.mjs"), import("../modules/gpad/RPadPainter.mjs"), import("../modules/gpad/RCanvasPainter.mjs")])).then(arr => {
            // copy all classes
            Object.assign(globalThis.JSROOT, arr[0], arr[1], arr[2]);
            if (jsrp) jsrp.ensureRCanvas = arr[2].ensureRCanvas;
            arr[1].RPadPainter.prototype.drawObject = globalThis.JSROOT.draw;
            arr[2].RPadPainter.prototype.getObjectDrawSettings = globalThis.JSROOT.getDrawSettings;
            return globalThis.JSROOT;
         }));
      else if (name == "io")
         arr.push(import("../modules/io.mjs"));
      else if (name == "tree")
         arr.push(import("../modules/tree.mjs"));
      else if (name == "geom")
         arr.push(geo ? Promise.resolve(geo) : loadPainter().then(() => Promise.all([import("../modules/geom/geobase.mjs"),
            import("../modules/geom/TGeoPainter.mjs"), import("../modules/base/base3d.mjs"), import("../modules/three.mjs")])).then(res => {

            if (geo) return geo;

            globalThis.JSROOT.GEO = geo = Object.assign({}, res[0], res[1]);
            globalThis.JSROOT.TGeoPainter = res[1].TGeoPainter;
            globalThis.JSROOT.Painter.createGeoPainter = res[1].createGeoPainter;
            globalThis.JSROOT.Painter.GeoDrawingControl = res[1].GeoDrawingControl;

            class myPoints extends res[2].PointsCreator {
               noPromise() { return true; }
            };
            globalThis.JSROOT.Painter.PointsCreator = myPoints;

            if (!globalThis.THREE)
               globalThis.THREE = Object.assign({}, res[3]); // copy methods, let add more

            return geo;
         }));
      else if (name == "math")
         arr.push(import("../modules/base/math.mjs"));
      else if (name == "latex")
         arr.push(import("../modules/base/latex.mjs"));
      else if (name == "painter")
         arr.push(loadPainter());
      else if (name == "hierarchy")
         arr.push(Promise.all([import("../modules/gui/HierarchyPainter.mjs"), import("../modules/draw/TTree.mjs")]).then(arr => {
            Object.assign(globalThis.JSROOT, arr[0], arr[1]);
            getHPainter = arr[0].getHPainter;
            globalThis.JSROOT.hpainter = getHPainter();
            return globalThis.JSROOT;
         }));
      else if (name == "openui5")
         arr.push(import("../modules/gui/utils.mjs").then(handle => handle.loadOpenui5(_openui5args)));
      else if (name == "interactive")
         arr.push(import("../modules/gui/utils.mjs").then(handle => { return { addMoveHandler: handle.addMoveHandler }; }));
      else if (name.indexOf(".js") >= 0)
         arr.push(import("../modules/core.mjs").then(handle => handle.loadScript(name)));
   });

   // need notify calling function when require is completed
   let notify;
   sync_promises.push(new Promise(func => { notify = func; }));

   return new Promise(resolveFunc => {
      Promise.all(arr).then(res => {
         resolveFunc(res.length == 1 ? res[0] : res);
         if (notify) notify(true);
      });
   });
}


exports.require = v6_require;

exports.define = function(req, factoryFunc) {
   let pr = new Promise(resolveFunc => {
       v6_require(req).then(arr => {
         if (req.length < 2) factoryFunc(arr)
                        else factoryFunc(...arr);
         resolveFunc(true);
       });
   });

   sync_promises.push(pr); // will wait until other PRs are finished
}

/// duplicate function here, used before loading any other functionality
exports.decodeUrl = function(url) {
   let res = {
      opts: {},
      has: function(opt) { return this.opts[opt] !== undefined; },
      get: function(opt,dflt) { let v = this.opts[opt]; return v!==undefined ? v : dflt; }
   };

   if (!url || (typeof url !== 'string')) {
      if (typeof document === 'undefined') return res;
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


exports.connectWebWindow = function(arg) {
   if (typeof arg == 'function')
      arg = { callback: arg };
   else if (!arg || (typeof arg != 'object'))
      arg = {};

   _openui5args = arg;

   let d = exports.decodeUrl();

   if (d.has("headless") && d.get("key")) {
      let is_chrome = false;
      if ((typeof document !== "undefined") && (typeof window !== "undefined"))
         is_chrome = (!!window.chrome && !browser.isOpera) || (navigator.userAgent.indexOf('HeadlessChrome') >= 0);
      if (is_chrome) {
         let element = document.createElement("script");
         element.setAttribute("type", "text/javascript");
         element.setAttribute("src", "root_batch_holder.js?key=" + d.get("key"));
         document.head.appendChild(element);
         arg.ignore_chrome_batch_holder = true;
      }
   }

   return _sync().then(() => {

      let prereq = "";
      if (arg.prereq) prereq = arg.prereq;
      if (arg.prereq2) prereq += ";" + arg.prereq2;

      if (!prereq) return;

      return v6_require(prereq).then(() => {
            delete arg.prereq;
            delete arg.prereq2;

            if (arg.prereq_logdiv && document) {
               let elem = document.getElementById(arg.prereq_logdiv);
               if (elem) elem.innerHTML = '';
               delete arg.prereq_logdiv;
            }
         });
   }).then(() => import('../modules/webwindow.mjs')).then(h => {
      globalThis.JSROOT.WebWindowHandle = h.WebWindowHandle;
      return h.connectWebWindow(arg);
   });
}


// try to define global JSROOT
if ((typeof globalThis !== "undefined") && !globalThis.JSROOT) {

   globalThis.JSROOT = exports;

   globalThis.JSROOT.extend = Object.assign;

   globalThis.JSROOT._complete_loading = _sync;

   let pr = Promise.all([import('../modules/core.mjs'), import('../modules/draw.mjs'), import('../modules/gui/HierarchyPainter.mjs')]).then(arr => {
      Object.assign(globalThis.JSROOT, arr[0], arr[1]);

      globalThis.JSROOT._ = arr[0].internals;

      globalThis.JSROOT.HierarchyPainter = arr[2].HierarchyPainter;
      getHPainter = arr[2].getHPainter;

      globalThis.JSROOT.hpainter = getHPainter();
   });

   sync_promises.push(pr);
}

}));
