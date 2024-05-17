import { select as d3_select } from './d3.mjs';
import { loadScript, findFunction, internals, getPromise, isNodeJs, isObject, isFunc, isStr, _ensureJSROOT,
         prROOT,
         clTObject, clTNamed, clTString, clTAttLine, clTAttFill, clTAttMarker, clTAttText,
         clTObjString, clTFile, clTList, clTHashList, clTMap, clTObjArray, clTClonesArray,
         clTPave, clTPaveText, clTPavesText, clTPaveStats, clTPaveLabel, clTPaveClass, clTDiamond, clTLegend, clTPaletteAxis,
         clTText, clTLine, clTBox, clTLatex, clTMathText, clTAnnotation, clTMultiGraph, clTH2, clTF1, clTF2, clTF3, clTH3,
         clTProfile, clTProfile2D, clTProfile3D, clTFrame,
         clTColor, clTHStack, clTGraph, clTGraph2DErrors, clTGraph2DAsymmErrors,
         clTGraphPolar, clTGraphPolargram, clTGraphTime, clTCutG, clTPolyLine, clTPolyLine3D, clTPolyMarker3D,
         clTPad, clTStyle, clTCanvas, clTGaxis, clTGeoVolume, kInspect, nsREX, atob_func } from './core.mjs';
import { clTStreamerInfoList } from './io.mjs';
import { clTBranchFunc } from './tree.mjs';
import { BasePainter, compressSVG, svgToImage, _loadJSDOM } from './base/BasePainter.mjs';
import { ObjectPainter, cleanup, drawRawText, getElementCanvPainter, getElementMainPainter } from './base/ObjectPainter.mjs';
import { TPadPainter, clTButton } from './gpad/TPadPainter.mjs';


async function import_more() { return import('./draw/more.mjs'); }

async function import_canvas() { return import('./gpad/TCanvasPainter.mjs'); }

async function import_tree() { return import('./draw/TTree.mjs'); }

async function import_h() { return import('./gui/HierarchyPainter.mjs'); }

async function import_geo() {
   return import('./geom/TGeoPainter.mjs').then(geo => {
      const handle = getDrawHandle(prROOT + 'TGeoVolumeAssembly');
      if (handle) handle.icon = 'img_geoassembly';
      return geo;
   });
}

const clTGraph2D = 'TGraph2D', clTH2Poly = 'TH2Poly', clTEllipse = 'TEllipse',
      clTSpline3 = 'TSpline3', clTTree = 'TTree', clTCanvasWebSnapshot = 'TCanvasWebSnapshot',
      fPrimitives = 'fPrimitives', fFunctions = 'fFunctions',

/** @summary list of registered draw functions
  * @private */
drawFuncs = { lst: [
   { name: clTCanvas, icon: 'img_canvas', class: () => import_canvas().then(h => h.TCanvasPainter), opt: ';grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz', expand_item: fPrimitives, noappend: true },
   { name: clTPad, icon: 'img_canvas', func: TPadPainter.draw, opt: ';grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz', expand_item: fPrimitives, noappend: true },
   { name: 'TSlider', icon: 'img_canvas', func: TPadPainter.draw },
   { name: clTButton, icon: 'img_canvas', func: TPadPainter.draw },
   { name: clTFrame, icon: 'img_frame', draw: () => import_canvas().then(h => h.drawTFrame) },
   { name: clTPave, icon: 'img_pavetext', class: () => import('./hist/TPavePainter.mjs').then(h => h.TPavePainter) },
   { name: clTPaveText, sameas: clTPave },
   { name: clTPavesText, sameas: clTPave },
   { name: clTPaveStats, sameas: clTPave },
   { name: clTPaveLabel, sameas: clTPave },
   { name: clTPaveClass, sameas: clTPave },
   { name: clTDiamond, sameas: clTPave },
   { name: clTLegend, icon: 'img_pavelabel', sameas: clTPave },
   { name: clTPaletteAxis, icon: 'img_colz', sameas: clTPave },
   { name: clTLatex, icon: 'img_text', draw: () => import_more().then(h => h.drawText), direct: true },
   { name: clTMathText, sameas: clTLatex },
   { name: clTText, sameas: clTLatex },
   { name: clTAnnotation, sameas: clTLatex },
   { name: /^TH1/, icon: 'img_histo1d', class: () => import('./hist/TH1Painter.mjs').then(h => h.TH1Painter), opt: ';hist;P;P0;E;E1;E2;E3;E4;E1X0;L;LF2;C;B;B1;A;TEXT;LEGO;same', ctrl: 'l', expand_item: fFunctions, for_derived: true },
   { name: clTProfile, icon: 'img_profile', class: () => import('./hist/TH1Painter.mjs').then(h => h.TH1Painter), opt: ';E0;E1;E2;p;AH;hist', expand_item: fFunctions },
   { name: clTH2Poly, icon: 'img_histo2d', class: () => import('./hist/TH2Painter.mjs').then(h => h.TH2Painter), opt: ';COL;COL0;COLZ;LCOL;LCOL0;LCOLZ;LEGO;TEXT;same', expand_item: 'fBins', theonly: true },
   { name: 'TProfile2Poly', sameas: clTH2Poly },
   { name: 'TH2PolyBin', icon: 'img_histo2d', draw_field: 'fPoly', draw_field_opt: 'L' },
   { name: /^TH2/, icon: 'img_histo2d', class: () => import('./hist/TH2Painter.mjs').then(h => h.TH2Painter), dflt: 'col', opt: ';COL;COLZ;COL0;COL1;COL0Z;COL1Z;COLA;BOX;BOX1;PROJ;PROJX1;PROJX2;PROJX3;PROJY1;PROJY2;PROJY3;SCAT;TEXT;TEXTE;TEXTE0;CANDLE;CANDLE1;CANDLE2;CANDLE3;CANDLE4;CANDLE5;CANDLE6;CANDLEY1;CANDLEY2;CANDLEY3;CANDLEY4;CANDLEY5;CANDLEY6;VIOLIN;VIOLIN1;VIOLIN2;VIOLINY1;VIOLINY2;CONT;CONT1;CONT2;CONT3;CONT4;ARR;SURF;SURF1;SURF2;SURF4;SURF6;E;A;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same', ctrl: 'lego', expand_item: fFunctions, for_derived: true },
   { name: clTProfile2D, sameas: clTH2 },
   { name: /^TH3/, icon: 'img_histo3d', class: () => import('./hist/TH3Painter.mjs').then(h => h.TH3Painter), opt: ';SCAT;BOX;BOX2;BOX3;GLBOX1;GLBOX2;GLCOL', expand_item: fFunctions, for_derived: true },
   { name: clTProfile3D, sameas: clTH3 },
   { name: clTHStack, icon: 'img_histo1d', class: () => import('./hist/THStackPainter.mjs').then(h => h.THStackPainter), expand_item: 'fHists', opt: 'NOSTACK;HIST;E;PFC;PLC' },
   { name: clTPolyMarker3D, icon: 'img_histo3d', draw: () => import('./draw/draw3d.mjs').then(h => h.drawPolyMarker3D), direct: true, frame: '3d' },
   { name: clTPolyLine3D, icon: 'img_graph', draw: () => import('./draw/draw3d.mjs').then(h => h.drawPolyLine3D), direct: true, frame: '3d' },
   { name: 'TGraphStruct' },
   { name: 'TGraphNode' },
   { name: 'TGraphEdge' },
   { name: clTGraphTime, icon: 'img_graph', class: () => import('./hist/TGraphTimePainter.mjs').then(h => h.TGraphTimePainter), opt: 'once;repeat;first', theonly: true },
   { name: clTGraph2D, icon: 'img_graph', class: () => import('./hist/TGraph2DPainter.mjs').then(h => h.TGraph2DPainter), opt: ';P;PCOL', theonly: true },
   { name: clTGraph2DErrors, sameas: clTGraph2D, opt: ';P;PCOL;ERR', theonly: true },
   { name: clTGraph2DAsymmErrors, sameas: clTGraph2D, opt: ';P;PCOL;ERR', theonly: true },
   { name: clTGraphPolargram, icon: 'img_graph', class: () => import('./draw/TGraphPolarPainter.mjs').then(h => h.TGraphPolargramPainter), theonly: true },
   { name: clTGraphPolar, icon: 'img_graph', class: () => import('./draw/TGraphPolarPainter.mjs').then(h => h.TGraphPolarPainter), opt: ';F;L;P;PE', theonly: true },
   { name: /^TGraph/, icon: 'img_graph', class: () => import('./hist2d/TGraphPainter.mjs').then(h => h.TGraphPainter), opt: ';L;P' },
   { name: 'TEfficiency', icon: 'img_graph', class: () => import('./hist/TEfficiencyPainter.mjs').then(h => h.TEfficiencyPainter), opt: ';AP' },
   { name: clTCutG, sameas: clTGraph },
   { name: /^RooHist/, sameas: clTGraph },
   { name: /^RooCurve/, sameas: clTGraph },
   { name: 'TScatter', icon: 'img_graph', class: () => import('./hist2d/TScatterPainter.mjs').then(h => h.TScatterPainter), opt: ';A' },
   { name: 'RooPlot', icon: 'img_canvas', func: drawRooPlot },
   { name: 'TRatioPlot', icon: 'img_mgraph', class: () => import('./draw/TRatioPlotPainter.mjs').then(h => h.TRatioPlotPainter), opt: '' },
   { name: clTMultiGraph, icon: 'img_mgraph', class: () => import('./hist/TMultiGraphPainter.mjs').then(h => h.TMultiGraphPainter), opt: ';l;p;3d', expand_item: 'fGraphs' },
   { name: clTStreamerInfoList, icon: 'img_question', draw: () => import_h().then(h => h.drawStreamerInfo) },
   { name: 'TWebPainting', icon: 'img_graph', class: () => import('./draw/TWebPaintingPainter.mjs').then(h => h.TWebPaintingPainter) },
   { name: clTCanvasWebSnapshot, icon: 'img_canvas', draw: () => import_canvas().then(h => h.drawTPadSnapshot) },
   { name: 'TPadWebSnapshot', sameas: clTCanvasWebSnapshot },
   { name: 'kind:Text', icon: 'img_text', func: drawRawText },
   { name: clTObjString, icon: 'img_text', func: drawRawText },
   { name: clTF1, icon: 'img_tf1', class: () => import('./hist/TF1Painter.mjs').then(h => h.TF1Painter), opt: ';L;C;FC;FL' },
   { name: clTF2, icon: 'img_tf2', class: () => import('./hist/TF2Painter.mjs').then(h => h.TF2Painter), opt: ';BOX;ARR;SURF;SURF1;SURF2;SURF4;SURF6;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same' },
   { name: clTF3, icon: 'img_histo3d', class: () => import('./hist/TF3Painter.mjs').then(h => h.TF3Painter), opt: ';SURF' },
   { name: clTSpline3, icon: 'img_tf1', class: () => import('./draw/TSplinePainter.mjs').then(h => h.TSplinePainter) },
   { name: 'TSpline5', sameas: clTSpline3 },
   { name: clTEllipse, icon: 'img_graph', draw: () => import_more().then(h => h.drawEllipse), direct: true },
   { name: 'TArc', sameas: clTEllipse },
   { name: 'TCrown', sameas: clTEllipse },
   { name: 'TPie', icon: 'img_graph', draw: () => import_more().then(h => h.drawPie), direct: true },
   { name: 'TPieSlice', icon: 'img_graph', dummy: true },
   { name: 'TExec', icon: 'img_graph', dummy: true },
   { name: clTLine, icon: 'img_graph', class: () => import('./draw/TLinePainter.mjs').then(h => h.TLinePainter) },
   { name: 'TArrow', icon: 'img_graph', class: () => import('./draw/TArrowPainter.mjs').then(h => h.TArrowPainter) },
   { name: clTPolyLine, icon: 'img_graph', draw: () => import_more().then(h => h.drawPolyLine), direct: true },
   { name: 'TCurlyLine', sameas: clTPolyLine },
   { name: 'TCurlyArc', sameas: clTPolyLine },
   { name: 'TParallelCoord', icon: 'img_graph', dummy: true },
   { name: clTGaxis, icon: 'img_graph', class: () => import('./draw/TGaxisPainter.mjs').then(h => h.TGaxisPainter) },
   { name: clTBox, icon: 'img_graph', draw: () => import_more().then(h => h.drawBox), direct: true },
   { name: 'TWbox', sameas: clTBox },
   { name: 'TSliderBox', sameas: clTBox },
   { name: 'TMarker', icon: 'img_graph', draw: () => import_more().then(h => h.drawMarker), direct: true },
   { name: 'TPolyMarker', icon: 'img_graph', draw: () => import_more().then(h => h.drawPolyMarker), direct: true },
   { name: 'TASImage', icon: 'img_mgraph', class: () => import('./draw/TASImagePainter.mjs').then(h => h.TASImagePainter), opt: ';z' },
   { name: 'TJSImage', icon: 'img_mgraph', draw: () => import_more().then(h => h.drawJSImage), opt: ';scale;center' },
   { name: clTGeoVolume, icon: 'img_histo3d', class: () => import_geo().then(h => h.TGeoPainter), get_expand: () => import_geo().then(h => h.expandGeoObject), opt: ';more;all;count;projx;projz;wire;no_screen;dflt', ctrl: 'dflt' },
   { name: 'TEveGeoShapeExtract', sameas: clTGeoVolume, opt: ';more;all;count;projx;projz;wire;dflt' },
   { name: nsREX+'REveGeoShapeExtract', sameas: clTGeoVolume, opt: ';more;all;count;projx;projz;wire;dflt' },
   { name: 'TGeoOverlap', sameas: clTGeoVolume, opt: ';more;all;count;projx;projz;wire;dflt', dflt: 'dflt', ctrl: 'expand' },
   { name: 'TGeoManager', sameas: clTGeoVolume, opt: ';more;all;count;projx;projz;wire;tracks;no_screen;dflt', dflt: 'expand', ctrl: 'dflt', noappend: true, exapnd_after_draw: true },
   { name: 'TGeoVolumeAssembly', sameas: clTGeoVolume, /* icon: 'img_geoassembly', */ opt: ';more;all;count' },
   { name: /^TGeo/, class: () => import_geo().then(h => h.TGeoPainter), get_expand: () => import_geo().then(h => h.expandGeoObject), opt: ';more;all;axis;compa;count;projx;projz;wire;no_screen;dflt', dflt: 'dflt', ctrl: 'expand' },
   { name: 'TAxis3D', icon: 'img_graph', draw: () => import_geo().then(h => h.drawAxis3D), direct: true },
   // these are not draw functions, but provide extra info about correspondent classes
   { name: 'kind:Command', icon: 'img_execute', execute: true },
   { name: 'TFolder', icon: 'img_folder', icon2: 'img_folderopen', noinspect: true, get_expand: () => import_h().then(h => h.folderHierarchy) },
   { name: 'TTask', icon: 'img_task', get_expand: () => import_h().then(h => h.taskHierarchy), for_derived: true },
   { name: clTTree, icon: 'img_tree', get_expand: () => import('./tree.mjs').then(h => h.treeHierarchy), draw: () => import_tree().then(h => h.drawTree), dflt: 'expand', opt: 'player;testio', shift: kInspect },
   { name: 'TNtuple', sameas: clTTree },
   { name: 'TNtupleD', sameas: clTTree },
   { name: clTBranchFunc, icon: 'img_leaf_method', draw: () => import_tree().then(h => h.drawTree), opt: ';dump', noinspect: true },
   { name: /^TBranch/, icon: 'img_branch', draw: () => import_tree().then(h => h.drawTree), dflt: 'expand', opt: ';dump', ctrl: 'dump', shift: kInspect, ignore_online: true, always_draw: true },
   { name: /^TLeaf/, icon: 'img_leaf', noexpand: true, draw: () => import_tree().then(h => h.drawTree), opt: ';dump', ctrl: 'dump', ignore_online: true, always_draw: true },
   { name: clTList, icon: 'img_list', draw: () => import_h().then(h => h.drawList), get_expand: () => import_h().then(h => h.listHierarchy), dflt: 'expand' },
   { name: clTHashList, sameas: clTList },
   { name: clTObjArray, sameas: clTList },
   { name: clTClonesArray, sameas: clTList },
   { name: clTMap, sameas: clTList },
   { name: clTColor, icon: 'img_color' },
   { name: clTFile, icon: 'img_file', noinspect: true },
   { name: 'TMemFile', icon: 'img_file', noinspect: true },
   { name: clTStyle, icon: 'img_question', noexpand: true },
   { name: 'Session', icon: 'img_globe' },
   { name: 'kind:TopFolder', icon: 'img_base' },
   { name: 'kind:Folder', icon: 'img_folder', icon2: 'img_folderopen', noinspect: true },
   { name: nsREX+'RCanvas', icon: 'img_canvas', class: () => init_v7().then(h => h.RCanvasPainter), opt: '', expand_item: fPrimitives },
   { name: nsREX+'RCanvasDisplayItem', icon: 'img_canvas', draw: () => init_v7().then(h => h.drawRPadSnapshot), opt: '', expand_item: fPrimitives },
   { name: nsREX+'RHist1Drawable', icon: 'img_histo1d', class: () => init_v7('rh1').then(h => h.RH1Painter), opt: '' },
   { name: nsREX+'RHist2Drawable', icon: 'img_histo2d', class: () => init_v7('rh2').then(h => h.RH2Painter), opt: '' },
   { name: nsREX+'RHist3Drawable', icon: 'img_histo3d', class: () => init_v7('rh3').then(h => h.RH3Painter), opt: '' },
   { name: nsREX+'RHistDisplayItem', icon: 'img_histo1d', draw: () => init_v7('rh3').then(h => h.drawHistDisplayItem), opt: '' },
   { name: nsREX+'RText', icon: 'img_text', draw: () => init_v7('more').then(h => h.drawText), opt: '', direct: 'v7', csstype: 'text' },
   { name: nsREX+'RFrameTitle', icon: 'img_text', draw: () => init_v7().then(h => h.drawRFrameTitle), opt: '', direct: 'v7', csstype: 'title' },
   { name: nsREX+'RPaletteDrawable', icon: 'img_text', class: () => init_v7('more').then(h => h.RPalettePainter), opt: '' },
   { name: nsREX+'RDisplayHistStat', icon: 'img_pavetext', class: () => init_v7('pave').then(h => h.RHistStatsPainter), opt: '' },
   { name: nsREX+'RLine', icon: 'img_graph', draw: () => init_v7('more').then(h => h.drawLine), opt: '', direct: 'v7', csstype: 'line' },
   { name: nsREX+'RBox', icon: 'img_graph', draw: () => init_v7('more').then(h => h.drawBox), opt: '', direct: 'v7', csstype: 'box' },
   { name: nsREX+'RMarker', icon: 'img_graph', draw: () => init_v7('more').then(h => h.drawMarker), opt: '', direct: 'v7', csstype: 'marker' },
   { name: nsREX+'RPave', icon: 'img_pavetext', class: () => init_v7('pave').then(h => h.RPavePainter), opt: '' },
   { name: nsREX+'RLegend', icon: 'img_graph', class: () => init_v7('pave').then(h => h.RLegendPainter), opt: '' },
   { name: nsREX+'RPaveText', icon: 'img_pavetext', class: () => init_v7('pave').then(h => h.RPaveTextPainter), opt: '' },
   { name: nsREX+'RFrame', icon: 'img_frame', draw: () => init_v7().then(h => h.drawRFrame), opt: '' },
   { name: nsREX+'RFont', icon: 'img_text', draw: () => init_v7().then(h => h.drawRFont), opt: '', direct: 'v7', csstype: 'font' },
   { name: nsREX+'RAxisDrawable', icon: 'img_frame', draw: () => init_v7().then(h => h.drawRAxis), opt: '' }
], cache: {} };


/** @summary Register draw function for the class
  * @desc List of supported draw options could be provided, separated  with ';'
  * @param {object} args - arguments
  * @param {string|regexp} args.name - class name or regexp pattern
  * @param {function} [args.func] - draw function
  * @param {function} [args.draw] - async function to load draw function
  * @param {function} [args.class] - async function to load painter class with static draw function
  * @param {boolean} [args.direct] - if true, function is just Redraw() method of ObjectPainter
  * @param {string} [args.opt] - list of supported draw options (separated with semicolon) like 'col;scat;'
  * @param {string} [args.icon] - icon name shown for the class in hierarchy browser
  * @param {string} [args.draw_field] - draw only data member from object, like fHistogram
  * @protected */
function addDrawFunc(args) {
   drawFuncs.lst.push(args);
   return args;
}

/** @summary return draw handle for specified item kind
  * @desc kind could be ROOT.TH1I for ROOT classes or just
  * kind string like 'Command' or 'Text'
  * selector can be used to search for draw handle with specified option (string)
  * or just sequence id
  * @private */
function getDrawHandle(kind, selector) {
   if (!isStr(kind)) return null;
   if (selector === '') selector = null;

   let first = null;

   if ((selector === null) && (kind in drawFuncs.cache))
      return drawFuncs.cache[kind];

   const search = (kind.indexOf(prROOT) === 0) ? kind.slice(5) : `kind:${kind}`;
   let counter = 0;
   for (let i = 0; i < drawFuncs.lst.length; ++i) {
      const h = drawFuncs.lst[i];
      if (isStr(h.name)) {
         if (h.name !== search) continue;
      } else if (!search.match(h.name))
         continue;

      if (h.sameas) {
         const hs = getDrawHandle(prROOT + h.sameas, selector);
         if (hs) {
            for (const key in hs) {
               if (h[key] === undefined)
                  h[key] = hs[key];
            }
            delete h.sameas;
         }
         return h;
      }

      if ((selector === null) || (selector === undefined)) {
         // store found handle in cache, can reuse later
         if (!(kind in drawFuncs.cache)) drawFuncs.cache[kind] = h;
         return h;
      } else if (isStr(selector)) {
         if (!first) first = h;
         // if drawoption specified, check it present in the list

         if (selector === '::expand') {
            if (('expand' in h) || ('expand_item' in h)) return h;
         } else if ('opt' in h) {
            const opts = h.opt.split(';');
            for (let j = 0; j < opts.length; ++j)
               opts[j] = opts[j].toLowerCase();
            if (opts.indexOf(selector.toLowerCase()) >= 0) return h;
         }
      } else if (selector === counter)
         return h;
      ++counter;
   }

   return first;
}

/** @summary Returns true if handle can be potentially drawn
  * @private */
function canDrawHandle(h) {
   if (isStr(h))
      h = getDrawHandle(h);
   if (!isObject(h)) return false;
   return h.func || h.class || h.draw || h.draw_field;
}

/** @summary Provide draw settings for specified class or kind
  * @private */
function getDrawSettings(kind, selector) {
   const res = { opts: null, inspect: false, expand: false, draw: false, handle: null };
   if (!isStr(kind)) return res;
   let isany = false, noinspect = false, canexpand = false;
   if (!isStr(selector)) selector = '';

   for (let cnt = 0; cnt < 1000; ++cnt) {
      const h = getDrawHandle(kind, cnt);
      if (!h) break;
      if (!res.handle) res.handle = h;
      if (h.noinspect) noinspect = true;
      if (h.noappend) res.noappend = true;
      if (h.expand || h.get_expand || h.expand_item || h.can_expand) canexpand = true;
      if (!h.func && !h.class && !h.draw) break;
      isany = true;
      if (!('opt' in h)) continue;
      const opts = h.opt.split(';');
      for (let i = 0; i < opts.length; ++i) {
         opts[i] = opts[i].toLowerCase();
         if (opts[i].indexOf('same') === 0) {
            res.has_same = true;
            if (selector.indexOf('nosame') >= 0) continue;
         }

         if (res.opts === null) res.opts = [];
         if (res.opts.indexOf(opts[i]) < 0) res.opts.push(opts[i]);
      }
      if (h.theonly) break;
   }

   if (selector.indexOf('noinspect') >= 0) noinspect = true;

   if (isany && (res.opts === null)) res.opts = [''];

   // if no any handle found, let inspect ROOT-based objects
   if (!isany && (kind.indexOf(prROOT) === 0) && !noinspect) res.opts = [];

   if (!noinspect && res.opts)
      res.opts.push(kInspect);

   res.inspect = !noinspect;
   res.expand = canexpand;
   res.draw = !!res.opts;

   return res;
}

/** @summary Set default draw option for provided class
  * @example
  import { setDefaultDrawOpt } from 'https://root.cern/js/latest/modules/draw.mjs';
  setDefaultDrawOpt('TH1', 'text');
  setDefaultDrawOpt('TH2', 'col');  */
function setDefaultDrawOpt(classname, opt) {
   const handle = getDrawHandle(prROOT + classname, 0);
   if (handle)
      handle.dflt = opt;
}

/** @summary Draw object in specified HTML element with given draw options.
  * @param {string|object} dom - id of div element to draw or directly DOMElement
  * @param {object} obj - object to draw, object type should be registered before with {@link addDrawFunc}
  * @param {string} opt - draw options separated by space, comma or semicolon
  * @return {Promise} with painter object
  * @public
  * @desc An extensive list of support draw options can be found on [examples page]{@link https://root.cern/js/latest/examples.htm}
  * @example
  * import { openFile } from 'https://root.cern/js/latest/modules/io.mjs';
  * import { draw } from 'https://root.cern/js/latest/modules/draw.mjs';
  * let file = await openFile('https://root.cern/js/files/hsimple.root');
  * let obj = await file.readObject('hpxpy;1');
  * await draw('drawing', obj, 'colz;logx;gridx;gridy'); */
async function draw(dom, obj, opt) {
   if (!isObject(obj))
      return Promise.reject(Error('not an object in draw call'));

   if (isStr(opt) && (opt.indexOf(kInspect) === 0))
      return import_h().then(h => h.drawInspector(dom, obj, opt));

   let handle, type_info;
   if ('_typename' in obj) {
      type_info = 'type ' + obj._typename;
      handle = getDrawHandle(prROOT + obj._typename, opt);
   } else if ('_kind' in obj) {
      type_info = 'kind ' + obj._kind;
      handle = getDrawHandle(obj._kind, opt);
   } else
      return import_h().then(h => h.drawInspector(dom, obj, opt));

   // this is case of unsupported class, close it normally
   if (!handle)
      return Promise.reject(Error(`Object of ${type_info} cannot be shown with draw`));

   if (handle.dummy)
      return null;

   if (handle.draw_field && obj[handle.draw_field])
      return draw(dom, obj[handle.draw_field], opt || handle.draw_field_opt);

   if (!canDrawHandle(handle)) {
      if (opt && (opt.indexOf('same') >= 0)) {
         const main_painter = getElementMainPainter(dom);

         if (isFunc(main_painter?.performDrop))
            return main_painter.performDrop(obj, '', null, opt);
      }

      return Promise.reject(Error(`Function not specified to draw object ${type_info}`));
   }

   function performDraw() {
      let promise, painter;
      if (handle.direct === 'v7') {
         promise = import('./gpad/RCanvasPainter.mjs').then(v7h => {
            painter = new v7h.RObjectPainter(dom, obj, opt, handle.csstype);
            painter.redraw = handle.func;
            return v7h.ensureRCanvas(painter, handle.frame || false);
         }).then(() => painter.redraw());
      } else if (handle.direct) {
         painter = new ObjectPainter(dom, obj, opt);
         painter.redraw = handle.func;
         promise = import_canvas().then(v6h => v6h.ensureTCanvas(painter, handle.frame || false))
                                  .then(() => painter.redraw());
      } else
         promise = getPromise(handle.func(dom, obj, opt));

      return promise.then(p => {
         if (!painter)
            painter = p;
         if (painter === false)
            return null;
         if (!painter)
             throw Error(`Fail to draw object ${type_info}`);
         if (isObject(painter) && !painter.options)
            painter.options = { original: opt || '' }; // keep original draw options
         return painter;
      });
   }

   if (isFunc(handle.func))
      return performDraw();

   let promise;

   if (isFunc(handle.class)) {
      // class coded as async function which returns class handle
      // simple extract class and access class.draw method
      promise = handle.class().then(cl => { handle.func = cl.draw; });
   } else if (isFunc(handle.draw)) {
      // draw function without special class
      promise = handle.draw().then(h => { handle.func = h; });
   } else if (!handle.func || !isStr(handle.func))
      return Promise.reject(Error(`Draw function or class not specified to draw ${type_info}`));
   else if (!handle.prereq && !handle.script)
      return Promise.reject(Error(`Prerequicities to load ${handle.func} are not specified`));
   else {
      const init_promise = internals.ignore_v6
         ? Promise.resolve(true)
         : _ensureJSROOT().then(v6 => {
         const pr = handle.prereq ? v6.require(handle.prereq) : Promise.resolve(true);
         return pr.then(() => {
            if (handle.script)
               return loadScript(handle.script);
         }).then(() => v6._complete_loading());
      });

      promise = init_promise.then(() => {
         const func = findFunction(handle.func);
         if (!isFunc(func))
            return Promise.reject(Error(`Fail to find function ${handle.func} after loading ${handle.prereq || handle.script}`));

         handle.func = func;
      });
   }

   return promise.then(() => performDraw());
}

/** @summary Redraw object in specified HTML element with given draw options.
  * @param {string|object} dom - id of div element to draw or directly DOMElement
  * @param {object} obj - object to draw, object type should be registered before with {@link addDrawFunc}
  * @param {string} opt - draw options
  * @return {Promise} with painter object
  * @desc If drawing was not done before, it will be performed with {@link draw}.
  * Otherwise drawing content will be updated
  * @public
  * @example
  * import { openFile } from 'https://root.cern/js/latest/modules/io.mjs';
  * import { draw, redraw } from 'https://root.cern/js/latest/modules/draw.mjs';
  * let file = await openFile('https://root.cern/js/files/hsimple.root');
  * let obj = await file.readObject('hpxpy;1');
  * await draw('drawing', obj, 'colz');
  * let cnt = 0;
  * setInterval(() => {
  *    obj.fTitle = `Next iteration ${cnt++}`;
  *    redraw('drawing', obj, 'colz');
  * }, 1000); */
async function redraw(dom, obj, opt) {
   if (!isObject(obj))
      return Promise.reject(Error('not an object in redraw'));

   const can_painter = getElementCanvPainter(dom);
   let handle, res_painter = null, redraw_res;
   if (obj._typename)
      handle = getDrawHandle(prROOT + obj._typename);
   if (handle?.draw_field && obj[handle.draw_field])
      obj = obj[handle.draw_field];

   if (can_painter) {
      if (can_painter.matchObjectType(obj._typename)) {
         redraw_res = can_painter.redrawObject(obj, opt);
         if (redraw_res) res_painter = can_painter;
      } else {
         for (let i = 0; i < can_painter.painters.length; ++i) {
            const painter = can_painter.painters[i];
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
      const top = new BasePainter(dom).getTopPainter();
      // base painter do not have this method, if it there use it
      // it can be object painter here or can be specially introduce method to handling redraw!
      if (isFunc(top?.redrawObject)) {
         redraw_res = top.redrawObject(obj, opt);
         if (redraw_res) res_painter = top;
      }
   }

   if (res_painter)
      return getPromise(redraw_res).then(() => res_painter);

   cleanup(dom);

   return draw(dom, obj, opt);
}

/** @summary Scan streamer infos for derived classes
  * @desc Assign draw functions for such derived classes
  * @private */
function addStreamerInfosForPainter(lst) {
   if (!lst) return;

   const basics = [clTObject, clTNamed, clTString, 'TCollection', clTAttLine, clTAttFill, clTAttMarker, clTAttText];

   function checkBaseClasses(si, lvl) {
      const element = si.fElements?.arr[0];
      if ((element?.fTypeName !== 'BASE') || (lvl > 4))
         return null;
      // exclude very basic classes
      if (basics.indexOf(element.fName) >= 0)
         return null;

      let handle = getDrawHandle(prROOT + element.fName);
      if (handle && !handle.for_derived)
            handle = null;

      // now try find that base class of base in the list
      if (handle === null) {
         for (let k = 0; k < lst.arr.length; ++k) {
            if (lst.arr[k].fName === element.fName) {
               handle = checkBaseClasses(lst.arr[k], lvl + 1);
               break;
            }
         }
      }

      return handle?.for_derived ? handle : null;
   }

   lst.arr.forEach(si => {
      if (getDrawHandle(prROOT + si.fName) !== null) return;

      const handle = checkBaseClasses(si, 0);
      if (handle) {
         const newhandle = Object.assign({}, handle);
         delete newhandle.for_derived; // should we disable?
         newhandle.name = si.fName;
         addDrawFunc(newhandle);
      }
   });
}

/** @summary Create SVG/PNG/JPEG image for provided object.
  * @desc Function especially useful in Node.js environment to generate images for
  * supported ROOT classes, but also can be used from web browser
  * @param {object} args - function settings
  * @param {object} args.object - object for the drawing
  * @param {string} [args.format = 'svg'] - image format like 'svg' (default), 'png' or 'jpeg'
  * @param {string} [args.option = ''] - draw options
  * @param {number} [args.width = 1200] - image width
  * @param {number} [args.height = 800] - image height
  * @param {boolean} [args.as_buffer = false] - returns image as Buffer instance, can store directly to file
  * @param {boolean} [args.use_canvas_size = false] - if configured used size stored in TCanvas object
  * @return {Promise} with image code - svg as is, png/jpeg as base64 string or buffer (if as_buffer) specified
  * @example
  * // how makeImage can be used in node.js
  * import { openFile, makeImage } from 'jsroot';
  * let file = await openFile('https://root.cern/js/files/hsimple.root');
  * let object = await file.readObject('hpxpy;1');
  * let png64 = await makeImage({ format: 'png', object, option: 'colz', width: 1200, height: 800 });
  * let pngbuf = await makeImage({ format: 'png', as_buffer: true, object, option: 'colz', width: 1200, height: 800 }); */
async function makeImage(args) {
   if (!args) args = {};

   if (!isObject(args.object))
      return Promise.reject(Error('No object specified to generate SVG'));
   if (!args.format)
      args.format = 'svg';
   if (!args.width)
      args.width = 1200;
   if (!args.height)
      args.height = 800;

   if (args.use_canvas_size && (args.object?._typename === clTCanvas) && args.object.fCw && args.object.fCh) {
      args.width = args.object?.fCw;
      args.height = args.object?.fCh;
   }

   async function build(main) {
      main.attr('width', args.width).attr('height', args.height)
          .style('width', args.width + 'px').style('height', args.height + 'px')
          .property('_batch_mode', true)
          .property('_batch_format', args.format !== 'svg' ? args.format : null);

      function complete(res) {
         cleanup(main.node());
         main.remove();
         return res;
      }

      return draw(main.node(), args.object, args.option || '').then(() => {
         if (args.format !== 'svg') {
            const only_img = main.select('svg').selectChild('image');
            if (!only_img.empty()) {
               const href = only_img.attr('href');

               if (args.as_buffer) {
                  const p = href.indexOf('base64,'),
                        str = atob_func(href.slice(p + 7)),
                        buf = new ArrayBuffer(str.length),
                        bufView = new Uint8Array(buf);
                  for (let i = 0; i < str.length; i++)
                     bufView[i] = str.charCodeAt(i);
                  return isNodeJs() ? Buffer.from(buf) : buf;
               }
               return href;
            }
         }

         main.select('svg')
             .attr('xmlns', 'http://www.w3.org/2000/svg')
             .attr('width', args.width)
             .attr('height', args.height)
             .attr('style', null).attr('class', null).attr('x', null).attr('y', null);

         function clear_element() {
            const elem = d3_select(this);
            if (elem.style('display') === 'none') elem.remove();
         }

         main.selectAll('g.root_frame').each(clear_element);
         main.selectAll('svg').each(clear_element);

         let svg;
         if (args.format === 'pdf')
            svg = { node: main.select('svg').node(), width: args.width, height: args.height, can_modify: true };
         else {
            svg = compressSVG(main.html());
            if (args.format === 'svg')
               return complete(svg);
         }

         return svgToImage(svg, args.format, args.as_buffer).then(complete);
      });
   }

   return isNodeJs()
          ? _loadJSDOM().then(handle => build(handle.body.append('div')))
          : build(d3_select('body').append('div').style('display', 'none'));
}


/** @summary Create SVG image for provided object.
  * @desc Function especially useful in Node.js environment to generate images for
  * supported ROOT classes
  * @param {object} args - function settings
  * @param {object} args.object - object for the drawing
  * @param {string} [args.option] - draw options
  * @param {number} [args.width = 1200] - image width
  * @param {number} [args.height = 800] - image height
  * @param {boolean} [args.use_canvas_size = false] - if configured used size stored in TCanvas object
  * @return {Promise} with svg code
  * @example
  * // how makeSVG can be used in node.js
  * import { openFile, makeSVG } from 'jsroot';
  * let file = await openFile('https://root.cern/js/files/hsimple.root');
  * let object = await file.readObject('hpxpy;1');
  * let svg = await makeSVG({ object, option: 'lego2,pal50', width: 1200, height: 800 }); */
async function makeSVG(args) {
   if (!args) args = {};
   args.format = 'svg';
   return makeImage(args);
}

internals.addDrawFunc = addDrawFunc;

function assignPadPainterDraw(PadPainterClass) {
   PadPainterClass.prototype.drawObject = (...args) =>
      draw(...args).catch(err => { console.log(`Error ${err?.message ?? err}  at ${err?.stack ?? 'uncknown place'}`); return null; });
   PadPainterClass.prototype.getObjectDrawSettings = getDrawSettings;
}

// only now one can draw primitives in the canvas
assignPadPainterDraw(TPadPainter);

// load v7 only by demand
async function init_v7(arg) {
   return import('./gpad/RCanvasPainter.mjs').then(h => {
      // only now one can draw primitives in the canvas
      assignPadPainterDraw(h.RPadPainter);
      switch (arg) {
         case 'more': return import('./draw/v7more.mjs');
         case 'pave': return import('./hist/RPavePainter.mjs');
         case 'rh1': return import('./hist/RH1Painter.mjs');
         case 'rh2': return import('./hist/RH2Painter.mjs');
         case 'rh3': return import('./hist/RH3Painter.mjs');
      }
      return h;
   });
}


// to avoid cross-dependnecy between io.mjs and draw.mjs
internals.addStreamerInfosForPainter = addStreamerInfosForPainter;

/** @summary Draw TRooPlot
  * @private */
async function drawRooPlot(dom, plot) {
   return draw(dom, plot._hist, 'hist').then(async hp => {
      const arr = [];
      for (let i = 0; i < plot._items.arr.length; ++i)
         arr.push(draw(dom, plot._items.arr[i], plot._items.opt[i]));
      return Promise.all(arr).then(() => hp);
   });
}

export { addDrawFunc, getDrawHandle, canDrawHandle, getDrawSettings, setDefaultDrawOpt,
         draw, redraw, cleanup, makeSVG, makeImage, drawRooPlot, assignPadPainterDraw };
