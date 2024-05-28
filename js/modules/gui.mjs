import { decodeUrl, settings, constants, gStyle, internals, browser, findFunction, parse, isFunc, isStr, isObject } from './core.mjs';
import { select as d3_select } from './d3.mjs';
import { HierarchyPainter } from './gui/HierarchyPainter.mjs';
import { setStoragePrefix, readSettings, readStyle } from './gui/utils.mjs';


/** @summary Read style and settings from URL
  * @private */
function readStyleFromURL(url) {
   // first try to read settings from coockies
   const d = decodeUrl(url),
         prefix = d.get('storage_prefix');

   if (isStr(prefix) && prefix)
      setStoragePrefix(prefix);

   readSettings();
   readStyle();

   function get_bool(name, field, special) {
      if (d.has(name)) {
         const val = d.get(name);
         if (special && (val === special))
            settings[field] = special;
         else
            settings[field] = (val !== '0') && (val !== 'false') && (val !== 'off');
      }
   }

   if (d.has('optimize')) {
      settings.OptimizeDraw = 2;
      let optimize = d.get('optimize');
      if (optimize) {
         optimize = parseInt(optimize);
         if (Number.isInteger(optimize))
            settings.OptimizeDraw = optimize;
      }
   }

   get_bool('lastcycle', 'OnlyLastCycle');
   get_bool('usestamp', 'UseStamp');
   get_bool('dark', 'DarkMode');

   let mr = d.get('maxranges');
   if (mr) {
      mr = parseInt(mr);
      if (Number.isInteger(mr)) settings.MaxRanges = mr;
   }

   if (d.has('wrong_http_response'))
      settings.HandleWrongHttpResponse = true;

   if (d.has('prefer_saved_points'))
      settings.PreferSavedPoints = true;

   const tf1_style = d.get('tf1');
   if (tf1_style === 'curve')
      settings.FuncAsCurve = true;
   else if (tf1_style === 'line')
      settings.FuncAsCurve = false;

   if (d.has('with_credentials'))
      settings.WithCredentials = true;

   let inter = d.get('interactive');
   if (inter === 'nomenu')
      settings.ContextMenu = false;
   else if (inter !== undefined) {
      if (!inter || (inter === '1'))
         inter = '111111';
      else if (inter === '0')
         inter = '000000';
      if (inter.length === 6) {
         switch (inter[0]) {
            case '0': settings.ToolBar = false; break;
            case '1': settings.ToolBar = 'popup'; break;
            case '2': settings.ToolBar = true; break;
         }
         inter = inter.slice(1);
      }
      if (inter.length === 5) {
         settings.Tooltip = parseInt(inter[0]);
         settings.ContextMenu = (inter[1] !== '0');
         settings.Zooming = (inter[2] !== '0');
         settings.MoveResize = (inter[3] !== '0');
         settings.DragAndDrop = (inter[4] !== '0');
      }
   }

   get_bool('tooltip', 'Tooltip');

   const mathjax = d.get('mathjax', null);
   let latex = d.get('latex', null);
   if ((mathjax !== null) && (mathjax !== '0') && (latex === null))
      latex = 'math';
   if (latex !== null)
      settings.Latex = constants.Latex.fromString(latex);

   if (d.has('nomenu')) settings.ContextMenu = false;
   if (d.has('noprogress'))
      settings.ProgressBox = false;
   else
      get_bool('progress', 'ProgressBox', 'modal');

   if (d.has('notouch')) browser.touches = false;
   if (d.has('adjframe')) settings.CanAdjustFrame = true;

   const has_toolbar = d.has('toolbar');
   if (has_toolbar) {
      const toolbar = d.get('toolbar', '');
      let val = null;
      if (toolbar.indexOf('popup') >= 0) val = 'popup';
      if (toolbar.indexOf('left') >= 0) { settings.ToolBarSide = 'left'; val = 'popup'; }
      if (toolbar.indexOf('right') >= 0) { settings.ToolBarSide = 'right'; val = 'popup'; }
      if (toolbar.indexOf('vert') >= 0) { settings.ToolBarVert = true; val = 'popup'; }
      if (toolbar.indexOf('show') >= 0) val = true;
      settings.ToolBar = val || ((toolbar.indexOf('0') < 0) && (toolbar.indexOf('false') < 0) && (toolbar.indexOf('off') < 0));
   }

   get_bool('skipsi', 'SkipStreamerInfos');
   get_bool('skipstreamerinfos', 'SkipStreamerInfos');

   if (d.has('nodraggraphs'))
      settings.DragGraphs = false;

   if (d.has('palette')) {
      const palette = parseInt(d.get('palette'));
      if (Number.isInteger(palette) && (palette > 0) && (palette < 113)) settings.Palette = palette;
   }

   const render3d = d.get('render3d'), embed3d = d.get('embed3d'), geosegm = d.get('geosegm');
   if (render3d) settings.Render3D = constants.Render3D.fromString(render3d);
   if (embed3d) settings.Embed3D = constants.Embed3D.fromString(embed3d);
   if (geosegm) settings.GeoGradPerSegm = Math.max(2, parseInt(geosegm));
   get_bool('geocomp', 'GeoCompressComp');

   if (d.has('hlimit')) settings.HierarchyLimit = parseInt(d.get('hlimit'));

   function get_int_style(name, field, dflt) {
      if (!d.has(name)) return;
      const val = d.get(name);
      if (!val || (val === 'true') || (val === 'on'))
         gStyle[field] = dflt;
      else if ((val === 'false') || (val === 'off'))
         gStyle[field] = 0;
      else
         gStyle[field] = parseInt(val);
      return gStyle[field] !== 0;
   }
   function get_float_style(name, field) {
      if (!d.has(name)) return;
      const val = d.get(name),
            flt = Number.parseFloat(val);
      if (Number.isFinite(flt))
         gStyle[field] = flt;
   }

   if (d.has('histzero')) gStyle.fHistMinimumZero = true;
   if (d.has('histmargin')) gStyle.fHistTopMargin = parseFloat(d.get('histmargin'));
   get_int_style('optstat', 'fOptStat', 1111);
   get_int_style('optfit', 'fOptFit', 0);
   const has_date = get_int_style('optdate', 'fOptDate', 1),
         has_file = get_int_style('optfile', 'fOptFile', 1);
   if ((has_date || has_file) && !has_toolbar)
      settings.ToolBarVert = true;
   get_float_style('datex', 'fDateX');
   get_float_style('datey', 'fDateY');

   get_int_style('opttitle', 'fOptTitle', 1);
   if (d.has('utc'))
      settings.TimeZone = 'UTC';
   else if (d.has('timezone')) {
      settings.TimeZone = d.get('timezone');
      if ((settings.TimeZone === 'default') || (settings.TimeZone === 'dflt'))
         settings.TimeZone = '';
   }

   gStyle.fStatFormat = d.get('statfmt', gStyle.fStatFormat);
   gStyle.fFitFormat = d.get('fitfmt', gStyle.fFitFormat);
}


/** @summary Build main GUI
  * @desc Used in many HTML files to create JSROOT GUI elements
  * @param {String} gui_element - id of the `<div>` element
  * @param {String} gui_kind - either 'online', 'nobrowser', 'draw'
  * @return {Promise} with {@link HierarchyPainter} instance
  * @example
  * import { buildGUI } from 'https://root.cern/js/latest/modules/gui.mjs';
  * buildGUI('guiDiv'); */
async function buildGUI(gui_element, gui_kind = '') {
   const myDiv = d3_select(isStr(gui_element) ? `#${gui_element}` : gui_element);
   if (myDiv.empty())
      return Promise.reject(Error('no div for gui found'));

   myDiv.html(''); // clear element

   const d = decodeUrl();
   let online = (gui_kind === 'online'), nobrowser = false, drawing = false;

   if (gui_kind === 'draw')
      online = drawing = nobrowser = true;
   else if ((gui_kind === 'nobrowser') || d.has('nobrowser') || (myDiv.attr('nobrowser') && myDiv.attr('nobrowser') !== 'false'))
      nobrowser = true;

   if (myDiv.attr('ignoreurl') === 'true')
      settings.IgnoreUrlOptions = true;

   readStyleFromURL();

   if (nobrowser) {
      let guisize = d.get('divsize');
      if (guisize) {
         guisize = guisize.split('x');
         if (guisize.length !== 2) guisize = null;
      }

      if (guisize)
         myDiv.style('position', 'relative').style('width', guisize[0] + 'px').style('height', guisize[1] + 'px');
      else {
         d3_select('html').style('height', '100%');
         d3_select('body').style('min-height', '100%').style('margin', 0).style('overflow', 'hidden');
         myDiv.style('position', 'absolute').style('left', 0).style('top', 0).style('bottom', 0).style('right', 0).style('padding', '1px');
      }
   }

   const hpainter = new HierarchyPainter('root', null);
   if (online) hpainter.is_online = drawing ? 'draw' : 'online';
   if (drawing) hpainter.exclude_browser = true;
   hpainter.start_without_browser = nobrowser;

   return hpainter.startGUI(myDiv).then(() => {
      if (!nobrowser)
         return hpainter.initializeBrowser();
      if (!drawing)
         return;
      const func = internals.getCachedObject || findFunction('GetCachedObject'),
            obj = isFunc(func) ? parse(func()) : undefined;
      if (isObject(obj))
         hpainter._cached_draw_object = obj;
      let opt = d.get('opt', '');
      if (d.has('websocket'))
         opt += ';websocket';
      return hpainter.display('', opt);
   }).then(() => hpainter);
}

export { buildGUI, internals, readStyleFromURL, HierarchyPainter };
