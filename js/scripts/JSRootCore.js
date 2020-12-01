// old-style core JSROOT main script
// temporary introduced here to support older JSROOT application
// will be removed with JSROOT v 6.2

"use strict";

if ((typeof document === "undefined") || (typeof window === "undefined")) {
   throw Error("JSRootCore.js can only be used with web browser")
}

(function() {

   function loadCoreScript(path) {
      let element = document.createElement("script");
      element.setAttribute('type', "text/javascript");
      element.setAttribute('src', path);

      return new Promise(resolveFunc => {
         element.onload = function() {
            element.onload = null;
            resolveFunc();
         }
         document.getElementsByTagName("head")[0].appendChild(element);
      })
   }

   let source_fullpath = document.currentScript.src;
   let minified = source_fullpath.indexOf("JSRootCore.min.js") >= 0;
   let p0 = source_fullpath.indexOf("JSRootCore.");
   let path0 = (p0>=0) ? source_fullpath.substr(0, p0) : "./";
   let _redirects = [];

   loadCoreScript(path0 + "JSRoot.core" + (minified ? ".min.js" : ".js")).then(() => {

      if (!JSROOT)
         throw Error("Fail to load JSRoot.core.js script from " + path0);

      JSROOT.GetUrlOption = function(opt, url, dflt) {
         JSROOT.warnOnce('Using obsolete JSROOT.GetUrlOption, change to JSROOT.decodeUrl');
         return JSROOT.decodeUrl(url).get(opt, dflt === undefined ? null : dflt);
      }

      JSROOT.AssertPrerequisites = function(req, callback) {
         JSROOT.warnOnce('Using obsolete JSROOT.AssertPrerequisites, change to JSROOT.require');
         req = req.replace(/2d;v7;/g, "v7gpad;").replace(/2d;v6;/g, "gpad;").replace(/more2d;/g, 'more;').replace(/2d;/g, 'gpad;').replace(/;v6;v7/g, ";gpad;v7gpad");
         JSROOT.require(req).then(callback);
      }

      JSROOT.OpenFile = function(filename, callback) {
         JSROOT.warnOnce('Using obsolete JSROOT.OpenFile function, change to JSROOT.openFile');
         let res = JSROOT.openFile(filename);
         return !callback ? res : res.then(callback);
      }

      JSROOT.JSONR_unref = function(arg) {
         JSROOT.warnOnce('Using obsolete JSROOT.JSONR_unref function, change to JSROOT.parse');
         return JSROOT.parse(arg);
      }

      JSROOT.MakeSVG = function(args) {
         JSROOT.warnOnce('Using obsolete JSROOT.MakeSVG function, change to JSROOT.makeSVG');
         return JSROOT.makeSVG(args);
      }

      JSROOT.CallBack = function(func, arg1, arg2) {
         JSROOT.warnOnce('Using obsolete JSROOT.CallBack function, change to JSROOT.callBack');
         return JSROOT.callBack(func, arg1, arg2);
      }

      function window_on_load() {
         if (document.attachEvent ? document.readyState === 'complete' : document.readyState !== 'loading')
            return Promise.resolve(true);

         return new Promise(resolve => {
            window.onload = resolve;
         });
      }


      _redirects.forEach(args => {
         let name = args.shift();
         if (typeof name == 'string') {
            JSROOT[name](...args);
         } else {
            let resolve = name;
            name = args.shift();
            JSROOT[name](...args).then(resolve);
         }
      });


      let d = JSROOT.decodeUrl(source_fullpath);

      if (d.has('nocache')) JSROOT.nocache = (new Date).getTime(); // use timestamp to overcome cache limitation
      if (d.has('wrong_http_response') || JSROOT.decodeUrl().has('wrong_http_response'))
         JSROOT.wrong_http_response_handling = true; // server may send wrong content length by partial requests, use other method to control this
      if (d.has('nosap')) JSROOT._.sap = undefined; // let ignore sap loader even with openui5 loaded

      // in case of require.js one have to use timeout to decople loading
      if (JSROOT._.amd && d.has('gui'))
         return window_on_load().then(() => setTimeout(() => JSROOT.buildGUI('check_existing_elements'), 50));

      if (d.has('gui'))
         return window_on_load().then(() => JSROOT.buildGUI());

      let prereq = "", user = d.get('load'), onload = d.get('onload');
      if (d.has('io')) prereq += "io;";
      if (d.has('tree')) prereq += "tree;";
      if (d.has('2d')) prereq += "gpad;";
      if (d.has('v7')) prereq += "v7gpad;";
      if (d.has('hist')) prereq += "hist;";
      if (d.has('hierarchy')) prereq += "hierarchy;";
      if (d.has('jq2d')) prereq += "jq2d;";
      if (d.has('more2d')) prereq += "more;";
      if (d.has('geom')) prereq += "geom;";
      if (d.has('3d')) prereq += "base3d;";
      if (d.has('math')) prereq += "math;";
      if (d.has('mathjax')) prereq += "mathjax;";
      if (d.has('openui5')) prereq += "openui5;";

      if (user) { prereq += "io;gpad;"; user = user.split(";"); }

      if (prereq || onload || user)
         window_on_load().then(() => JSROOT.require(prereq))
                         .then(() => JSROOT.loadScript(user))
                         .then(() => JSROOT.callBack(onload));

   });

   let tmpJSROOT = { _workaround: true };

   tmpJSROOT.OpenFile = function(filename, call_back) {
      if (JSROOT !== tmpJSROOT)
         JSROOT.OpenFile(filename, call_back);
      else
         _redirects.push(["OpenFile", filename, call_back]);
   }

   tmpJSROOT.openFile = function(filename) {
      if (JSROOT !== tmpJSROOT)
         return JSROOT.openFile(filename);
      return new Promise(resolve => _redirects.push([resolve, "openFile", filename]));
   }

   tmpJSROOT.draw = function(divid, obj, opt, call_back) {
      if (JSROOT !== tmpJSROOT)
         return JSROOT.draw(divid, obj, opt).then(call_back);
     _redirects.push(["draw", divid, obj, opt, call_back]);
   }

   tmpJSROOT.redraw = function(divid, obj, opt, call_back) {
      if (JSROOT !== tmpJSROOT)
         return JSROOT.redraw(divid, obj, opt).then(call_back);
     _redirects.push(["redraw", divid, obj, opt, call_back]);
   }

   // until real JSROOT is loaded, provide minimal functions
   globalThis.JSROOT = tmpJSROOT;

})();