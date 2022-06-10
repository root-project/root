import { decodeUrl, settings, constants, gStyle, internals, findFunction, parse } from './core.mjs';

import { select as d3_select } from './d3.mjs';

import { HierarchyPainter } from './gui/HierarchyPainter.mjs';

import { readSettings, readStyle } from './gui/utils.mjs';

/** @summary Read style and settings from URL
  * @private */
function readStyleFromURL(url) {

   // first try to read settings from coockies
   readSettings();
   readStyle();

   let d = decodeUrl(url);

   function get_bool(name, field) {
      if (d.has(name)) {
         let val = d.get(name);
         settings[field] = (val != "0") && (val != "false") && (val != "off");
      }
   }

   if (d.has("optimize")) {
      settings.OptimizeDraw = 2;
      let optimize = d.get("optimize");
      if (optimize) {
         optimize = parseInt(optimize);
         if (Number.isInteger(optimize)) settings.OptimizeDraw = optimize;
      }
   }

   get_bool("lastcycle", "OnlyLastCycle");
   get_bool("usestamp", "UseStamp");
   get_bool("dark", "DarkMode");

   if (d.has('wrong_http_response'))
      settings.HandleWrongHttpResponse = true;

   let inter = d.get("interactive");
   if (inter === "nomenu")
      settings.ContextMenu = false;
   else if (inter !== undefined) {
      if (!inter || (inter == "1"))
         inter = "111111";
      else if (inter == "0")
         inter = "000000";
      if (inter.length === 6) {
         switch(inter[0]) {
            case "0": settings.ToolBar = false; break;
            case "1": settings.ToolBar = 'popup'; break;
            case "2": settings.ToolBar = true; break;
         }
         inter = inter.slice(1);
      }
      if (inter.length == 5) {
         settings.Tooltip = parseInt(inter[0]);
         settings.ContextMenu = (inter[1] != '0');
         settings.Zooming = (inter[2] != '0');
         settings.MoveResize = (inter[3] != '0');
         settings.DragAndDrop = (inter[4] != '0');
      }
   }

   get_bool("tooltip", "Tooltip");

   if (d.has("bootstrap") || d.has("bs"))
      settings.Bootstrap = true;

   let mathjax = d.get("mathjax", null), latex = d.get("latex", null);

   if ((mathjax !== null) && (mathjax != "0") && (latex === null)) latex = "math";
   if (latex !== null)
      settings.Latex = constants.Latex.fromString(latex);

   if (d.has("nomenu")) settings.ContextMenu = false;
   if (d.has("noprogress")) settings.ProgressBox = false;
   if (d.has("notouch")) browser.touches = false;
   if (d.has("adjframe")) settings.CanAdjustFrame = true;

   if (d.has("toolbar")) {
      let toolbar = d.get("toolbar", ""), val = null;
      if (toolbar.indexOf('popup') >= 0) val = 'popup';
      if (toolbar.indexOf('left') >= 0) { settings.ToolBarSide = 'left'; val = 'popup'; }
      if (toolbar.indexOf('right') >= 0) { settings.ToolBarSide = 'right'; val = 'popup'; }
      if (toolbar.indexOf('vert') >= 0) { settings.ToolBarVert = true; val = 'popup'; }
      if (toolbar.indexOf('show') >= 0) val = true;
      settings.ToolBar = val || ((toolbar.indexOf("0") < 0) && (toolbar.indexOf("false") < 0) && (toolbar.indexOf("off") < 0));
   }

   get_bool("skipsi", "SkipStreamerInfos");
   get_bool("skipstreamerinfos", "SkipStreamerInfos");

   if (d.has("nodraggraphs"))
      settings.DragGraphs = false;

   if (d.has("palette")) {
      let palette = parseInt(d.get("palette"));
      if (Number.isInteger(palette) && (palette > 0) && (palette < 113)) settings.Palette = palette;
   }

   let render3d = d.get("render3d"), embed3d = d.get("embed3d"), geosegm = d.get("geosegm");
   if (render3d) settings.Render3D = constants.Render3D.fromString(render3d);
   if (embed3d) settings.Embed3D = constants.Embed3D.fromString(embed3d);
   if (geosegm) settings.GeoGradPerSegm = Math.max(2, parseInt(geosegm));
   get_bool("geocomp", "GeoCompressComp");

   if (d.has("hlimit")) settings.HierarchyLimit = parseInt(d.get("hlimit"));

   function get_int_style(name, field, dflt) {
      if (!d.has(name)) return;
      let val = d.get(name);
      if (!val || (val == "true") || (val == "on"))
         gStyle[field] = dflt;
      else if ((val == "false") || (val == "off"))
         gStyle[field] = 0;
      else
         gStyle[field] = parseInt(val);
   }

   if (d.has("histzero")) gStyle.fHistMinimumZero = true;
   if (d.has("histmargin")) gStyle.fHistTopMargin = parseFloat(d.get("histmargin"));
   get_int_style("optstat", "fOptStat", 1111);
   get_int_style("optfit", "fOptFit", 0);
   get_int_style("optdate", "fOptDate", 1);
   get_int_style("optfile", "fOptFile", 1);
   get_int_style("opttitle", "fOptTitle", 1);
   gStyle.fStatFormat = d.get("statfmt", gStyle.fStatFormat);
   gStyle.fFitFormat = d.get("fitfmt", gStyle.fFitFormat);
}


/** @summary Build main GUI
  * @desc Used in many HTML files to create JSROOT GUI elements
  * @param {String} gui_element - id of the `<div>` element
  * @param {String} gui_kind - either "online", "nobrowser", "draw"
  * @returns {Promise} with {@link HierarchyPainter} instance
  * @example
  * import { buildGUI } from '/path_to_jsroot/modules/gui.mjs';
  * buildGUI("guiDiv"); */
function buildGUI(gui_element, gui_kind = "") {
   let myDiv = (typeof gui_element == 'string') ? d3_select('#' + gui_element) : d3_select(gui_element);
   if (myDiv.empty())
      return Promise.reject(Error('no div for gui found'));

   myDiv.html(""); // clear element

   let d = decodeUrl(), online = (gui_kind == "online"), nobrowser = false, drawing = false;

   if (gui_kind == "draw") {
      online = drawing = nobrowser = true;
   } else if ((gui_kind == "nobrowser") || d.has("nobrowser") || (myDiv.attr("nobrowser") && myDiv.attr("nobrowser")!=="false")) {
      nobrowser = true;
   }

   if (myDiv.attr("ignoreurl") === "true")
      settings.IgnoreUrlOptions = true;

   readStyleFromURL();

   if (nobrowser) {
      let guisize = d.get("divsize");
      if (guisize) {
         guisize = guisize.split("x");
         if (guisize.length != 2) guisize = null;
      }

      if (guisize) {
         myDiv.style('position',"relative").style('width', guisize[0] + "px").style('height', guisize[1] + "px");
      } else {
         d3_select('html').style('height','100%');
         d3_select('body').style('min-height','100%').style('margin',0).style('overflow',"hidden");
         myDiv.style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('right',0).style('padding',1);
      }
   }

   let hpainter = new HierarchyPainter('root', null);

   if (online) hpainter.is_online = drawing ? "draw" : "online";
   if (drawing) hpainter.exclude_browser = true;
   hpainter.start_without_browser = nobrowser;

   return hpainter.startGUI(myDiv).then(() => {
      if (!nobrowser)
         return hpainter.initializeBrowser();
      if (!drawing)
         return;
      let obj = null, func = internals.getCachedObject || findFunction('GetCachedObject');
      if (typeof func == 'function')
         obj = parse(func());
      if (obj) hpainter._cached_draw_object = obj;
      let opt = d.get("opt", "");
      if (d.has("websocket")) opt+=";websocket";
      return hpainter.display("", opt);
   }).then(() => hpainter);
}

export { buildGUI, internals, readStyleFromURL, HierarchyPainter };
