import { settings, gStyle, isBatchMode, source_dir } from '../core.mjs';
import { select as d3_select, pointer as d3_pointer, drag as d3_drag } from '../d3.mjs';
import { BasePainter } from '../base/BasePainter.mjs';
import { resize } from '../base/ObjectPainter.mjs';


/** @summary Display progress message in the left bottom corner.
  * @desc Previous message will be overwritten
  * if no argument specified, any shown messages will be removed
  * @param {string} msg - message to display
  * @param {number} tmout - optional timeout in milliseconds, after message will disappear
  * @private */
function showProgress(msg, tmout) {
   if (isBatchMode() || (typeof document === 'undefined')) return;
   let id = "jsroot_progressbox",
       box = d3_select("#" + id);

   if (!settings.ProgressBox)
      return box.remove();

   if ((arguments.length == 0) || !msg) {
      if ((tmout !== -1) || (!box.empty() && box.property("with_timeout"))) box.remove();
      return;
   }

   if (box.empty()) {
      box = d3_select(document.body)
              .append("div").attr("id", id)
              .attr("style", "position: fixed; min-width: 100px; height: auto; overflow: visible; z-index: 101; border: 1px solid #999; background: #F8F8F8; left: 10px; bottom: 10px;");
      box.append("p");
   }

   box.property("with_timeout", false);

   if (typeof msg === "string") {
      box.select("p").html(msg);
   } else {
      box.html("");
      box.node().appendChild(msg);
   }
   injectStyle("#jsroot_progressbox p { font-size: 10px; margin-left: 10px; margin-right: 10px; margin-top: 3px; margin-bottom: 3px; }", box.node());

   if (Number.isFinite(tmout) && (tmout > 0)) {
      box.property("with_timeout", true);
      setTimeout(() => showProgress('', -1), tmout);
   }
}

/** @summary Tries to close current browser tab
  * @desc Many browsers do not allow simple window.close() call,
  * therefore try several workarounds
  * @private */
function closeCurrentWindow() {
   if (!window) return;
   window.close();
   window.open('', '_self').close();
}


function tryOpenOpenUI(sources, args) {
   if (!sources || (sources.length == 0)) {
      if (args.rejectFunc) {
         args.rejectFunc(Error("openui5 was not possible to load"));
         args.rejectFunc = null;
      }
      return;
   }

   // where to take openui5 sources
   let src = sources.shift();

   if ((src.indexOf("roothandler") == 0) && (src.indexOf("://") < 0))
      src = src.replace(/\:\//g,"://");

   let element = document.createElement("script");
   element.setAttribute('type', "text/javascript");
   element.setAttribute('id', "sap-ui-bootstrap");
   // use nojQuery while we are already load jquery and jquery-ui, later one can use directly sap-ui-core.js

   // this is location of openui5 scripts when working with THttpServer or when scripts are installed inside JSROOT
   element.setAttribute('src', src + "resources/sap-ui-core.js"); // latest openui5 version

   element.setAttribute('data-sap-ui-libs', args.openui5libs ?? "sap.m, sap.ui.layout, sap.ui.unified, sap.ui.commons");

   element.setAttribute('data-sap-ui-theme', args.openui5theme || 'sap_belize');
   element.setAttribute('data-sap-ui-compatVersion', 'edge');
   // element.setAttribute('data-sap-ui-bindingSyntax', 'complex');

   element.setAttribute('data-sap-ui-preload', 'async'); // '' to disable Component-preload.js

   element.setAttribute('data-sap-ui-evt-oninit', "completeUI5Loading()");

   element.onerror = function() {
      // remove failed element
      element.parentNode.removeChild(element);
      // and try next
      tryOpenOpenUI(sources, args);
   }

   element.onload = function() {
      console.log('Load openui5 from ' + src);
   }

   document.head.appendChild(element);
}


// return Promise let loader wait before dependent source will be invoked

function loadOpenui5(args) {
   // very simple - openui5 was loaded before and will be used as is
   if (typeof sap == 'object')
      return Promise.resolve(sap);

   if (!args) args = {};

   let rootui5sys = source_dir.replace(/jsrootsys/g, "rootui5sys");

   if (rootui5sys == source_dir) {
      // if jsrootsys location not detected, try to guess it
      if (window.location.port && (window.location.pathname.indexOf("/win") >= 0) && (!args.openui5src || args.openui5src == 'nojsroot' || args.openui5src == 'jsroot'))
         rootui5sys = window.location.origin + window.location.pathname + "../rootui5sys/";
      else
         rootui5sys = undefined;
   }

   let openui5_sources = [],
       openui5_dflt = "https://openui5.hana.ondemand.com/1.98.0/",
       openui5_root = rootui5sys ? rootui5sys + "distribution/" : "";

   if (typeof args.openui5src == 'string') {
      switch (args.openui5src) {
         case "nodefault": openui5_dflt = ""; break;
         case "default": openui5_sources.push(openui5_dflt); openui5_dflt = ""; break;
         case "nojsroot": /* openui5_root = ""; */ break;
         case "jsroot": openui5_sources.push(openui5_root); openui5_root = ""; break;
         default: openui5_sources.push(args.openui5src); break;
      }
   }

   if (openui5_root && (openui5_sources.indexOf(openui5_root) < 0)) openui5_sources.push(openui5_root);
   if (openui5_dflt && (openui5_sources.indexOf(openui5_dflt) < 0)) openui5_sources.push(openui5_dflt);

   return new Promise((resolve, reject) => {

      args.resolveFunc = resolve;
      args.rejectFunc = reject;

      globalThis.completeUI5Loading = function() {
         sap.ui.loader.config({
            paths: {
               jsroot: source_dir,
               rootui5: rootui5sys
            }
         });

         if (args.resolveFunc) {
            args.resolveFunc(sap);
            args.resolveFunc = null;
         }
      };

      tryOpenOpenUI(openui5_sources, args);
   });

}

// some icons taken from http://uxrepo.com/
const ToolbarIcons = {
   camera: { path: 'M 152.00,304.00c0.00,57.438, 46.562,104.00, 104.00,104.00s 104.00-46.562, 104.00-104.00s-46.562-104.00-104.00-104.00S 152.00,246.562, 152.00,304.00z M 480.00,128.00L 368.00,128.00 c-8.00-32.00-16.00-64.00-48.00-64.00L 192.00,64.00 c-32.00,0.00-40.00,32.00-48.00,64.00L 32.00,128.00 c-17.60,0.00-32.00,14.40-32.00,32.00l0.00,288.00 c0.00,17.60, 14.40,32.00, 32.00,32.00l 448.00,0.00 c 17.60,0.00, 32.00-14.40, 32.00-32.00L 512.00,160.00 C 512.00,142.40, 497.60,128.00, 480.00,128.00z M 256.00,446.00c-78.425,0.00-142.00-63.574-142.00-142.00c0.00-78.425, 63.575-142.00, 142.00-142.00c 78.426,0.00, 142.00,63.575, 142.00,142.00 C 398.00,382.426, 334.427,446.00, 256.00,446.00z M 480.00,224.00l-64.00,0.00 l0.00-32.00 l 64.00,0.00 L 480.00,224.00 z' },
   disk: { path: 'M384,0H128H32C14.336,0,0,14.336,0,32v448c0,17.656,14.336,32,32,32h448c17.656,0,32-14.344,32-32V96L416,0H384z M352,160   V32h32v128c0,17.664-14.344,32-32,32H160c-17.664,0-32-14.336-32-32V32h128v128H352z M96,288c0-17.656,14.336-32,32-32h256   c17.656,0,32,14.344,32,32v192H96V288z' },
   question: { path: 'M256,512c141.375,0,256-114.625,256-256S397.375,0,256,0S0,114.625,0,256S114.625,512,256,512z M256,64   c63.719,0,128,36.484,128,118.016c0,47.453-23.531,84.516-69.891,110.016C300.672,299.422,288,314.047,288,320   c0,17.656-14.344,32-32,32c-17.664,0-32-14.344-32-32c0-40.609,37.25-71.938,59.266-84.031   C315.625,218.109,320,198.656,320,182.016C320,135.008,279.906,128,256,128c-30.812,0-64,20.227-64,64.672   c0,17.664-14.336,32-32,32s-32-14.336-32-32C128,109.086,193.953,64,256,64z M256,449.406c-18.211,0-32.961-14.75-32.961-32.969   c0-18.188,14.75-32.953,32.961-32.953c18.219,0,32.969,14.766,32.969,32.953C288.969,434.656,274.219,449.406,256,449.406z' },
   undo: { path: 'M450.159,48.042c8.791,9.032,16.983,18.898,24.59,29.604c7.594,10.706,14.146,22.207,19.668,34.489  c5.509,12.296,9.82,25.269,12.92,38.938c3.113,13.669,4.663,27.834,4.663,42.499c0,14.256-1.511,28.863-4.532,43.822  c-3.009,14.952-7.997,30.217-14.953,45.795c-6.955,15.577-16.202,31.52-27.755,47.826s-25.88,32.9-42.942,49.807  c-5.51,5.444-11.787,11.67-18.834,18.651c-7.033,6.98-14.496,14.366-22.39,22.168c-7.88,7.802-15.955,15.825-24.187,24.069  c-8.258,8.231-16.333,16.203-24.252,23.888c-18.3,18.13-37.354,37.016-57.191,56.65l-56.84-57.445  c19.596-19.472,38.54-38.279,56.84-56.41c7.75-7.685,15.772-15.604,24.108-23.757s16.438-16.163,24.33-24.057  c7.894-7.893,15.356-15.33,22.402-22.312c7.034-6.98,13.312-13.193,18.821-18.651c22.351-22.402,39.165-44.648,50.471-66.738  c11.279-22.09,16.932-43.567,16.932-64.446c0-15.785-3.217-31.005-9.638-45.671c-6.422-14.665-16.229-28.504-29.437-41.529  c-3.282-3.282-7.358-6.395-12.217-9.325c-4.871-2.938-10.381-5.503-16.516-7.697c-6.121-2.201-12.815-3.992-20.058-5.373  c-7.242-1.374-14.9-2.064-23.002-2.064c-8.218,0-16.802,0.834-25.788,2.507c-8.961,1.674-18.053,4.429-27.222,8.271  c-9.189,3.842-18.456,8.869-27.808,15.089c-9.358,6.219-18.521,13.819-27.502,22.793l-59.92,60.271l93.797,94.058H0V40.91  l93.27,91.597l60.181-60.532c13.376-15.018,27.222-27.248,41.536-36.697c14.308-9.443,28.608-16.776,42.89-21.992  c14.288-5.223,28.505-8.74,42.623-10.557C294.645,0.905,308.189,0,321.162,0c13.429,0,26.389,1.185,38.84,3.562  c12.478,2.377,24.2,5.718,35.192,10.029c11.006,4.311,21.126,9.404,30.374,15.265C434.79,34.724,442.995,41.119,450.159,48.042z' },
   arrow_right: { path: 'M30.796,226.318h377.533L294.938,339.682c-11.899,11.906-11.899,31.184,0,43.084c11.887,11.899,31.19,11.893,43.077,0  l165.393-165.386c5.725-5.712,8.924-13.453,8.924-21.539c0-8.092-3.213-15.84-8.924-21.551L338.016,8.925  C332.065,2.975,324.278,0,316.478,0c-7.802,0-15.603,2.968-21.539,8.918c-11.899,11.906-11.899,31.184,0,43.084l113.391,113.384  H30.796c-16.822,0-30.463,13.645-30.463,30.463C0.333,212.674,13.974,226.318,30.796,226.318z' },
   arrow_up: { path: 'M295.505,629.446V135.957l148.193,148.206c15.555,15.559,40.753,15.559,56.308,0c15.555-15.538,15.546-40.767,0-56.304  L283.83,11.662C276.372,4.204,266.236,0,255.68,0c-10.568,0-20.705,4.204-28.172,11.662L11.333,227.859  c-7.777,7.777-11.666,17.965-11.666,28.158c0,10.192,3.88,20.385,11.657,28.158c15.563,15.555,40.762,15.555,56.317,0  l148.201-148.219v493.489c0,21.993,17.837,39.82,39.82,39.82C277.669,669.267,295.505,651.439,295.505,629.446z' },
   arrow_diag: { path: 'M279.875,511.994c-1.292,0-2.607-0.102-3.924-0.312c-10.944-1.771-19.333-10.676-20.457-21.71L233.97,278.348  L22.345,256.823c-11.029-1.119-19.928-9.51-21.698-20.461c-1.776-10.944,4.031-21.716,14.145-26.262L477.792,2.149  c9.282-4.163,20.167-2.165,27.355,5.024c7.201,7.189,9.199,18.086,5.024,27.356L302.22,497.527  C298.224,506.426,289.397,511.994,279.875,511.994z M118.277,217.332l140.534,14.294c11.567,1.178,20.718,10.335,21.878,21.896  l14.294,140.519l144.09-320.792L118.277,217.332z' },
   auto_zoom: { path: 'M505.441,242.47l-78.303-78.291c-9.18-9.177-24.048-9.171-33.216,0c-9.169,9.172-9.169,24.045,0.006,33.217l38.193,38.188  H280.088V80.194l38.188,38.199c4.587,4.584,10.596,6.881,16.605,6.881c6.003,0,12.018-2.297,16.605-6.875  c9.174-9.172,9.174-24.039,0.011-33.217L273.219,6.881C268.803,2.471,262.834,0,256.596,0c-6.229,0-12.202,2.471-16.605,6.881  l-78.296,78.302c-9.178,9.172-9.178,24.045,0,33.217c9.177,9.171,24.051,9.171,33.21,0l38.205-38.205v155.4H80.521l38.2-38.188  c9.177-9.171,9.177-24.039,0.005-33.216c-9.171-9.172-24.039-9.178-33.216,0L7.208,242.464c-4.404,4.403-6.881,10.381-6.881,16.611  c0,6.227,2.477,12.207,6.881,16.61l78.302,78.291c4.587,4.581,10.599,6.875,16.605,6.875c6.006,0,12.023-2.294,16.61-6.881  c9.172-9.174,9.172-24.036-0.005-33.211l-38.205-38.199h152.593v152.063l-38.199-38.211c-9.171-9.18-24.039-9.18-33.216-0.022  c-9.178,9.18-9.178,24.059-0.006,33.222l78.284,78.302c4.41,4.404,10.382,6.881,16.611,6.881c6.233,0,12.208-2.477,16.611-6.881  l78.302-78.296c9.181-9.18,9.181-24.048,0-33.205c-9.174-9.174-24.054-9.174-33.21,0l-38.199,38.188v-152.04h152.051l-38.205,38.199  c-9.18,9.175-9.18,24.037-0.005,33.211c4.587,4.587,10.596,6.881,16.604,6.881c6.01,0,12.024-2.294,16.605-6.875l78.303-78.285  c4.403-4.403,6.887-10.378,6.887-16.611C512.328,252.851,509.845,246.873,505.441,242.47z' },
   statbox: {
      path: 'M28.782,56.902H483.88c15.707,0,28.451-12.74,28.451-28.451C512.331,12.741,499.599,0,483.885,0H28.782   C13.074,0,0.331,12.741,0.331,28.451C0.331,44.162,13.074,56.902,28.782,56.902z' +
         'M483.885,136.845H28.782c-15.708,0-28.451,12.741-28.451,28.451c0,15.711,12.744,28.451,28.451,28.451H483.88   c15.707,0,28.451-12.74,28.451-28.451C512.331,149.586,499.599,136.845,483.885,136.845z' +
         'M483.885,273.275H28.782c-15.708,0-28.451,12.731-28.451,28.452c0,15.707,12.744,28.451,28.451,28.451H483.88   c15.707,0,28.451-12.744,28.451-28.451C512.337,286.007,499.599,273.275,483.885,273.275z' +
         'M256.065,409.704H30.492c-15.708,0-28.451,12.731-28.451,28.451c0,15.707,12.744,28.451,28.451,28.451h225.585   c15.707,0,28.451-12.744,28.451-28.451C284.516,422.436,271.785,409.704,256.065,409.704z'
   },
   circle: { path: "M256,256 m-150,0 a150,150 0 1,0 300,0 a150,150 0 1,0 -300,0" },
   three_circles: { path: "M256,85 m-70,0 a70,70 0 1,0 140,0 a70,70 0 1,0 -140,0  M256,255 m-70,0 a70,70 0 1,0 140,0 a70,70 0 1,0 -140,0  M256,425 m-70,0 a70,70 0 1,0 140,0 a70,70 0 1,0 -140,0 " },
   diamand: { path: "M256,0L384,256L256,511L128,256z" },
   rect: { path: "M80,80h352v352h-352z" },
   cross: { path: "M80,40l176,176l176,-176l40,40l-176,176l176,176l-40,40l-176,-176l-176,176l-40,-40l176,-176l-176,-176z" },
   vrgoggles: { size: "245.82 141.73", path: 'M175.56,111.37c-22.52,0-40.77-18.84-40.77-42.07S153,27.24,175.56,27.24s40.77,18.84,40.77,42.07S198.08,111.37,175.56,111.37ZM26.84,69.31c0-23.23,18.25-42.07,40.77-42.07s40.77,18.84,40.77,42.07-18.26,42.07-40.77,42.07S26.84,92.54,26.84,69.31ZM27.27,0C11.54,0,0,12.34,0,28.58V110.9c0,16.24,11.54,30.83,27.27,30.83H99.57c2.17,0,4.19-1.83,5.4-3.7L116.47,118a8,8,0,0,1,12.52-.18l11.51,20.34c1.2,1.86,3.22,3.61,5.39,3.61h72.29c15.74,0,27.63-14.6,27.63-30.83V28.58C245.82,12.34,233.93,0,218.19,0H27.27Z' },
   th2colorz: { recs: [{ x: 128, y: 486, w: 256, h: 26, f: 'rgb(38,62,168)' }, { y: 461, f: 'rgb(22,82,205)' }, { y: 435, f: 'rgb(16,100,220)' }, { y: 410, f: 'rgb(18,114,217)' }, { y: 384, f: 'rgb(20,129,214)' }, { y: 358, f: 'rgb(14,143,209)' }, { y: 333, f: 'rgb(9,157,204)' }, { y: 307, f: 'rgb(13,167,195)' }, { y: 282, f: 'rgb(30,175,179)' }, { y: 256, f: 'rgb(46,183,164)' }, { y: 230, f: 'rgb(82,186,146)' }, { y: 205, f: 'rgb(116,189,129)' }, { y: 179, f: 'rgb(149,190,113)' }, { y: 154, f: 'rgb(179,189,101)' }, { y: 128, f: 'rgb(209,187,89)' }, { y: 102, f: 'rgb(226,192,75)' }, { y: 77, f: 'rgb(244,198,59)' }, { y: 51, f: 'rgb(253,210,43)' }, { y: 26, f: 'rgb(251,230,29)' }, { y: 0, f: 'rgb(249,249,15)' }] },
   th2color: { recs: [{x:0,y:256,w:13,h:39,f:'rgb(38,62,168)'},{x:13,y:371,w:39,h:39},{y:294,h:39},{y:256,h:39},{y:218,h:39},{x:51,y:410,w:39,h:39},{y:371,h:39},{y:333,h:39},{y:294},{y:256,h:39},{y:218,h:39},{y:179,h:39},{y:141,h:39},{y:102,h:39},{y:64},{x:90,y:448,w:39,h:39},{y:410},{y:371,h:39},{y:333,h:39,f:'rgb(22,82,205)'},{y:294},{y:256,h:39,f:'rgb(16,100,220)'},{y:218,h:39},{y:179,h:39,f:'rgb(22,82,205)'},{y:141,h:39},{y:102,h:39,f:'rgb(38,62,168)'},{y:64},{y:0,h:27},{x:128,y:448,w:39,h:39},{y:410},{y:371,h:39},{y:333,h:39,f:'rgb(22,82,205)'},{y:294,f:'rgb(20,129,214)'},{y:256,h:39,f:'rgb(9,157,204)'},{y:218,h:39,f:'rgb(14,143,209)'},{y:179,h:39,f:'rgb(20,129,214)'},{y:141,h:39,f:'rgb(16,100,220)'},{y:102,h:39,f:'rgb(22,82,205)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{y:0,h:27},{x:166,y:486,h:14},{y:448,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39,f:'rgb(20,129,214)'},{y:294,f:'rgb(82,186,146)'},{y:256,h:39,f:'rgb(179,189,101)'},{y:218,h:39,f:'rgb(116,189,129)'},{y:179,h:39,f:'rgb(82,186,146)'},{y:141,h:39,f:'rgb(14,143,209)'},{y:102,h:39,f:'rgb(16,100,220)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:205,y:486,w:39,h:14},{y:448,h:39},{y:410},{y:371,h:39,f:'rgb(16,100,220)'},{y:333,h:39,f:'rgb(9,157,204)'},{y:294,f:'rgb(149,190,113)'},{y:256,h:39,f:'rgb(244,198,59)'},{y:218,h:39},{y:179,h:39,f:'rgb(226,192,75)'},{y:141,h:39,f:'rgb(13,167,195)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(22,82,205)'},{y:26,h:39,f:'rgb(38,62,168)'},{x:243,y:448,w:39,h:39},{y:410},{y:371,h:39,f:'rgb(18,114,217)'},{y:333,h:39,f:'rgb(30,175,179)'},{y:294,f:'rgb(209,187,89)'},{y:256,h:39,f:'rgb(251,230,29)'},{y:218,h:39,f:'rgb(249,249,15)'},{y:179,h:39,f:'rgb(226,192,75)'},{y:141,h:39,f:'rgb(30,175,179)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:282,y:448,h:39},{y:410},{y:371,h:39,f:'rgb(18,114,217)'},{y:333,h:39,f:'rgb(14,143,209)'},{y:294,f:'rgb(149,190,113)'},{y:256,h:39,f:'rgb(226,192,75)'},{y:218,h:39,f:'rgb(244,198,59)'},{y:179,h:39,f:'rgb(149,190,113)'},{y:141,h:39,f:'rgb(9,157,204)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:320,y:448,w:39,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39,f:'rgb(20,129,214)'},{y:294,f:'rgb(46,183,164)'},{y:256,h:39},{y:218,h:39,f:'rgb(82,186,146)'},{y:179,h:39,f:'rgb(9,157,204)'},{y:141,h:39,f:'rgb(20,129,214)'},{y:102,h:39,f:'rgb(16,100,220)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:358,y:448,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39},{y:294,f:'rgb(16,100,220)'},{y:256,h:39,f:'rgb(20,129,214)'},{y:218,h:39,f:'rgb(14,143,209)'},{y:179,h:39,f:'rgb(18,114,217)'},{y:141,h:39,f:'rgb(22,82,205)'},{y:102,h:39,f:'rgb(38,62,168)'},{y:64},{y:26,h:39},{x:397,y:448,w:39,h:39},{y:371,h:39},{y:333,h:39},{y:294,f:'rgb(22,82,205)'},{y:256,h:39},{y:218,h:39},{y:179,h:39,f:'rgb(38,62,168)'},{y:141,h:39},{y:102,h:39},{y:64},{y:26,h:39},{x:435,y:410,h:39},{y:371,h:39},{y:333,h:39},{y:294},{y:256,h:39},{y:218,h:39},{y:179,h:39},{y:141,h:39},{y:102,h:39},{y:64},{x:474,y:256,h:39},{y:179,h:39}] },
   th2draw3d: {
      path: "M172.768,0H51.726C23.202,0,0.002,23.194,0.002,51.712v89.918c0,28.512,23.2,51.718,51.724,51.718h121.042   c28.518,0,51.724-23.2,51.724-51.718V51.712C224.486,23.194,201.286,0,172.768,0z M177.512,141.63c0,2.611-2.124,4.745-4.75,4.745   H51.726c-2.626,0-4.751-2.134-4.751-4.745V51.712c0-2.614,2.125-4.739,4.751-4.739h121.042c2.62,0,4.75,2.125,4.75,4.739 L177.512,141.63L177.512,141.63z "+
            "M460.293,0H339.237c-28.521,0-51.721,23.194-51.721,51.712v89.918c0,28.512,23.2,51.718,51.721,51.718h121.045   c28.521,0,51.721-23.2,51.721-51.718V51.712C512.002,23.194,488.802,0,460.293,0z M465.03,141.63c0,2.611-2.122,4.745-4.748,4.745   H339.237c-2.614,0-4.747-2.128-4.747-4.745V51.712c0-2.614,2.133-4.739,4.747-4.739h121.045c2.626,0,4.748,2.125,4.748,4.739 V141.63z "+
            "M172.768,256.149H51.726c-28.524,0-51.724,23.205-51.724,51.726v89.915c0,28.504,23.2,51.715,51.724,51.715h121.042   c28.518,0,51.724-23.199,51.724-51.715v-89.915C224.486,279.354,201.286,256.149,172.768,256.149z M177.512,397.784   c0,2.615-2.124,4.736-4.75,4.736H51.726c-2.626-0.006-4.751-2.121-4.751-4.736v-89.909c0-2.626,2.125-4.753,4.751-4.753h121.042 c2.62,0,4.75,2.116,4.75,4.753L177.512,397.784L177.512,397.784z "+
            "M460.293,256.149H339.237c-28.521,0-51.721,23.199-51.721,51.726v89.915c0,28.504,23.2,51.715,51.721,51.715h121.045   c28.521,0,51.721-23.199,51.721-51.715v-89.915C512.002,279.354,488.802,256.149,460.293,256.149z M465.03,397.784   c0,2.615-2.122,4.736-4.748,4.736H339.237c-2.614,0-4.747-2.121-4.747-4.736v-89.909c0-2.626,2.121-4.753,4.747-4.753h121.045 c2.615,0,4.748,2.116,4.748,4.753V397.784z"
   },

   createSVG(group, btn, size, title) {
      injectStyle('.jsroot_svg_toolbar_btn { fill: steelblue; cursor: pointer; opacity: 0.3; } .jsroot_svg_toolbar_btn:hover { opacity: 1.0; }', group.node());

      let svg = group.append("svg:svg")
                     .attr("class", "jsroot_svg_toolbar_btn")
                     .attr("width", size + "px")
                     .attr("height", size + "px")
                     .attr("viewBox", "0 0 512 512")
                     .style("overflow", "hidden");

      if ('recs' in btn) {
         let rec = {};
         for (let n = 0; n < btn.recs.length; ++n) {
            Object.assign(rec, btn.recs[n]);
            svg.append('rect').attr("x", rec.x).attr("y", rec.y)
               .attr("width", rec.w).attr("height", rec.h)
               .style("fill", rec.f);
         }
      } else {
         svg.append('svg:path').attr('d', btn.path);
      }

      //  special rect to correctly get mouse events for whole button area
      svg.append("svg:rect").attr("x", 0).attr("y", 0).attr("width", 512).attr("height", 512)
         .style('opacity', 0).style('fill', "none").style("pointer-events", "visibleFill")
         .append("svg:title").text(title);

      return svg;
   }

} // ToolbarIcons


/** @summary Register handle to react on window resize
  * @desc function used to react on browser window resize event
  * While many resize events could come in short time,
  * resize will be handled with delay after last resize event
  * @param {object|string} handle can be function or object with checkResize function or dom where painting was done
  * @param {number} [delay] - one could specify delay after which resize event will be handled
  * @protected */
function registerForResize(handle, delay) {

   if (!handle || isBatchMode() || (typeof window == 'undefined')) return;

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
               resize(node.node());
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

function detectRightButton(event) {
   if ('buttons' in event) return event.buttons === 2;
   if ('which' in event) return event.which === 3;
   if ('button' in event) return event.button === 2;
   return false;
}

/** @summary Add move handlers for drawn element
  * @private */
function addMoveHandler(painter, enabled) {

   if (enabled === undefined) enabled = true;

   if (!settings.MoveResize || isBatchMode() || !painter.draw_g) return;

   if (!enabled) {
      if (painter.draw_g.property("assigned_move")) {
         let drag_move = d3_drag().subject(Object);
         drag_move.on("start", null).on("drag", null).on("end", null);
         painter.draw_g
               .style("cursor", null)
               .property("assigned_move", null)
               .call(drag_move);
      }
      return;
   }

   if (painter.draw_g.property("assigned_move")) return;

   let drag_move = d3_drag().subject(Object),
      not_changed = true, move_disabled = false;

   drag_move
      .on("start", function(evnt) {
         move_disabled = this.moveEnabled ? !this.moveEnabled() : false;
         if (move_disabled) return;
         if (detectRightButton(evnt.sourceEvent)) return;
         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();
         let pos = d3_pointer(evnt, this.draw_g.node());
         not_changed = true;
         if (this.moveStart)
            this.moveStart(pos[0], pos[1]);
      }.bind(painter)).on("drag", function(evnt) {
         if (move_disabled) return;
         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();
         not_changed = false;
         if (this.moveDrag)
            this.moveDrag(evnt.dx, evnt.dy);
      }.bind(painter)).on("end", function(evnt) {
         if (move_disabled) return;
         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();
         if (this.moveEnd)
            this.moveEnd(not_changed);
         let pp = this.getPadPainter();
         if (pp) pp.selectObjectPainter(this);
      }.bind(painter));

   painter.draw_g
          .style("cursor", "move")
          .property("assigned_move", true)
          .call(drag_move);
}

/** @summary Inject style
  * @param {String} code - css string
  * @private */
function injectStyle(code, node, tag) {
   if (isBatchMode() || !code || (typeof document === 'undefined'))
      return true;

   let styles = (node || document).getElementsByTagName('style');
   for (let n = 0; n < styles.length; ++n) {
      if (tag && styles[n].getAttribute("tag") == tag) {
         styles[n].innerHTML = code;
         return true;
      }

      if (styles[n].innerHTML == code)
         return true;
   }

   let element = document.createElement("style");
   if (tag) element.setAttribute("tag", tag);
   element.innerHTML = code;
   (node || document.head).appendChild(element);
   return true;
}

/** @summary Select predefined style
  * @private */
function selectgStyle(name) {
   gStyle.fName = name;
   switch (name) {
      case "Modern": Object.assign(gStyle, {
         fFrameBorderMode: 0, fFrameFillColor: 0, fCanvasBorderMode: 0,
         fCanvasColor: 0, fPadBorderMode: 0, fPadColor: 0, fStatColor: 0,
         fTitleAlign: 23, fTitleX: 0.5, fTitleBorderSize: 0, fTitleColor: 0, fTitleStyle: 0,
         fOptStat: 1111, fStatY: 0.935,
         fLegendBorderSize: 1, fLegendFont: 42, fLegendTextSize: 0, fLegendFillColor: 0 }); break;
      case "Plain": Object.assign(gStyle, {
         fFrameBorderMode: 0, fCanvasBorderMode: 0, fPadBorderMode: 0,
         fPadColor: 0, fCanvasColor: 0,
         fTitleColor: 0, fTitleBorderSize: 0, fStatColor: 0, fStatBorderSize: 1, fLegendBorderSize: 1 }); break;
      case "Bold": Object.assign(gStyle, {
         fCanvasColor: 10, fCanvasBorderMode: 0,
         fFrameLineWidth: 3, fFrameFillColor: 10,
         fPadColor: 10, fPadTickX: 1, fPadTickY: 1, fPadBottomMargin: 0.15, fPadLeftMargin: 0.15,
         fTitleColor: 10, fTitleTextColor: 600, fStatColor: 10 }); break;
   }
}

function saveCookie(obj, expires, name) {
   let arg = (expires <= 0) ? "" : btoa(JSON.stringify(obj)),
       d = new Date();
   d.setTime((expires <= 0) ? 0 : d.getTime() + expires*24*60*60*1000);
   document.cookie = `${name}=${arg}; expires=${d.toUTCString()}; SameSite=None; Secure; path=/;`;
}

function readCookie(name) {
   if (typeof document == 'undefined') return null;
   let decodedCookie = decodeURIComponent(document.cookie),
       ca = decodedCookie.split(';');
   name += "=";
   for(let i = 0; i < ca.length; i++) {
      let c = ca[i];
      while (c.charAt(0) == ' ')
        c = c.substring(1);
      if (c.indexOf(name) == 0) {
         let s = JSON.parse(atob(c.substring(name.length, c.length)));

         return (s && typeof s == 'object') ? s : null;
      }
   }
   return null;
}

/** @summary Save JSROOT settings as specified cookie parameter
  * @param {Number} expires - days when cookie will be removed by browser, negative - delete immediately
  * @param {String} name - cookie parameter name
  * @private */
function saveSettings(expires = 365, name = "jsroot_settings") {
   saveCookie(settings, expires, name);
}

/** @summary Read JSROOT settings from specified cookie parameter
  * @param {Boolean} only_check - when true just checks if settings were stored before with provided name
  * @param {String} name - cookie parameter name
  * @private */
function readSettings(only_check = false, name = "jsroot_settings") {
   let s = readCookie(name);
   if (!s) return false;
   if (!only_check)
      Object.assign(settings, s);
   return true;
}

/** @summary Save JSROOT gStyle object as specified cookie parameter
  * @param {Number} expires - days when cookie will be removed by browser, negative - delete immediately
  * @param {String} name - cookie parameter name
  * @private */
function saveStyle(expires = 365, name = "jsroot_style") {
   saveCookie(gStyle, expires, name);
}

/** @summary Read JSROOT gStyle object specified cookie parameter
  * @param {Boolean} only_check - when true just checks if settings were stored before with provided name
  * @param {String} name - cookie parameter name
  * @private */
function readStyle(only_check = false, name = "jsroot_style") {
   let s = readCookie(name);
   if (!s) return false;
   if (!only_check)
      Object.assign(gStyle, s);
   return true;
}


export { showProgress, closeCurrentWindow, loadOpenui5, ToolbarIcons, registerForResize,
         detectRightButton, addMoveHandler, injectStyle,
         selectgStyle, saveSettings, readSettings, saveStyle, readStyle };
