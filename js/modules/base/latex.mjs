/// latex parsing and rendering

import { loadScript, settings, isNodeJs } from '../core.mjs';

import { getElementRect, _loadJSDOM } from './BasePainter.mjs';

import { FontHandler } from './FontHandler.mjs';

const symbols_map = {
   // greek letters
   '#alpha': '\u03B1',
   '#beta': '\u03B2',
   '#chi': '\u03C7',
   '#delta': '\u03B4',
   '#digamma': '\u03DD',
   '#varepsilon': '\u03B5',
   '#phi': '\u03C6',
   '#gamma': '\u03B3',
   '#eta': '\u03B7',
   '#iota': '\u03B9',
   '#varphi': '\u03C6',
   '#kappa': '\u03BA',
   '#koppa': '\u03DF',
   '#sampi': '\u03E1',
   '#lambda': '\u03BB',
   '#mu': '\u03BC',
   '#nu': '\u03BD',
   '#omicron': '\u03BF',
   '#pi': '\u03C0',
   '#theta': '\u03B8',
   '#rho': '\u03C1',
   '#sigma': '\u03C3',
   '#stigma': '\u03DB',
   '#san': '\u03FB',
   '#sho': '\u03F8',
   '#tau': '\u03C4',
   '#upsilon': '\u03C5',
   '#varomega': '\u03D6',
   '#varcoppa': '\u03D9',
   '#omega': '\u03C9',
   '#xi': '\u03BE',
   '#psi': '\u03C8',
   '#zeta': '\u03B6',
   '#Alpha': '\u0391',
   '#Beta': '\u0392',
   '#Chi': '\u03A7',
   '#Delta': '\u0394',
   '#Digamma': '\u03DC',
   '#Epsilon': '\u0395',
   '#Phi': '\u03A6',
   '#Gamma': '\u0393',
   '#Eta': '\u0397',
   '#Iota': '\u0399',
   '#vartheta': '\u03D1',
   '#Kappa': '\u039A',
   '#Koppa': '\u03DE',
   '#varKoppa': '\u03D8',
   '#Sampi': '\u03E0',
   '#Lambda': '\u039B',
   '#Mu': '\u039C',
   '#Nu': '\u039D',
   '#Omicron': '\u039F',
   '#Pi': '\u03A0',
   '#Theta': '\u0398',
   '#Rho': '\u03A1',
   '#Sigma': '\u03A3',
   '#Stigma': '\u03DA',
   '#San': '\u03FA',
   '#Sho': '\u03F7',
   '#Tau': '\u03A4',
   '#Upsilon': '\u03A5',
   '#varsigma': '\u03C2',
   '#Omega': '\u03A9',
   '#Xi': '\u039E',
   '#Psi': '\u03A8',
   '#Zeta': '\u0396',
   '#varUpsilon': '\u03D2',
   '#epsilon': '\u03B5',
   '#P': '\u00B6',

   // only required for MathJax to provide correct replacement
   '#sqrt': '\u221A',
   '#bar': '',
   '#overline': '',
   '#underline': '',
   '#strike': '',

   // from TLatex tables #2 & #3
   '#leq': '\u2264',
   '#/': '\u2044',
   '#infty': '\u221E',
   '#voidb': '\u0192',
   '#club': '\u2663',
   '#diamond': '\u2666',
   '#heart': '\u2665',
   '#spade': '\u2660',
   '#leftrightarrow': '\u2194',
   '#leftarrow': '\u2190',
   '#uparrow': '\u2191',
   '#rightarrow': '\u2192',
   '#downarrow': '\u2193',
   '#circ': '\u02C6', // ^
   '#pm': '\xB1',
   '#doublequote': '\u2033',
   '#geq': '\u2265',
   '#times': '\xD7',
   '#propto': '\u221D',
   '#partial': '\u2202',
   '#bullet': '\u2022',
   '#divide': '\xF7',
   '#neq': '\u2260',
   '#equiv': '\u2261',
   '#approx': '\u2248', // should be \u2245 ?
   '#3dots': '\u2026',
   '#cbar': '\x7C',
   '#topbar': '\xAF',
   '#downleftarrow': '\u21B5',
   '#aleph': '\u2135',
   '#Jgothic': '\u2111',
   '#Rgothic': '\u211C',
   '#voidn': '\u2118',
   '#otimes': '\u2297',
   '#oplus': '\u2295',
   '#oslash': '\u2205',
   '#cap': '\u2229',
   '#cup': '\u222A',
   '#supseteq': '\u2287',
   '#supset': '\u2283',
   '#notsubset': '\u2284',
   '#subseteq': '\u2286',
   '#subset': '\u2282',
   '#int': '\u222B',
   '#in': '\u2208',
   '#notin': '\u2209',
   '#angle': '\u2220',
   '#nabla': '\u2207',
   '#oright': '\xAE',
   '#ocopyright': '\xA9',
   '#trademark': '\u2122',
   '#prod': '\u220F',
   '#surd': '\u221A',
   '#upoint': '\u02D9',
   '#corner': '\xAC',
   '#wedge': '\u2227',
   '#vee': '\u2228',
   '#Leftrightarrow': '\u21D4',
   '#Leftarrow': '\u21D0',
   '#Uparrow': '\u21D1',
   '#Rightarrow': '\u21D2',
   '#Downarrow': '\u21D3',
   '#LT': '\x3C',
   '#void1': '\xAE',
   '#copyright': '\xA9',
   '#void3': '\u2122',
   '#sum': '\u2211',
   '#arctop': '\u239B',
   '#lbar': '\u23B8',
   '#arcbottom': '\u239D',
   '#void8': '',
   '#bottombar': '\u230A',
   '#arcbar': '\u23A7',
   '#ltbar': '\u23A8',
   '#AA': '\u212B',
   '#aa': '\u00E5',
   '#void06': '',
   '#GT': '\x3E',
   '#forall': '\u2200',
   '#exists': '\u2203',
   '#vec': '',
   '#dot': '\u22C5',
   '#hat': '\xB7',
   '#ddot': '',
   '#acute': '',
   '#grave': '',
   '#check': '\u2713',
   '#tilde': '\u02DC',
   '#slash': '\u2044',
   '#hbar': '\u0127',
   '#box': '\u25FD',
   '#Box': '\u2610',
   '#parallel': '\u2225',
   '#perp': '\u22A5',
   '#odot': '\u2299',
   '#left': '',
   '#right': '',
   '{}': ''
};

/** @summary Create a single regex to detect any symbol to replace
  * @private */
const symbolsRegexCache = new RegExp('(' + Object.keys(symbols_map).join('|').replace(/\\\{/g, '{').replace(/\\\}/g, '}') + ')', 'g');

/** @summary Simple replacement of latex letters
  * @private */
const translateLaTeX = (str, more) => {
   while ((str.length > 2) && (str[0] == '{') && (str[str.length - 1] == '}'))
      str = str.slice(1, str.length - 1);

   return str.replace(symbolsRegexCache, ch => symbols_map[ch]).replace(/\{\}/g, "");
};

// array with relative width of base symbols from range 32..126
const base_symbols_width = [453,535,661,973,955,1448,1242,324,593,596,778,1011,431,570,468,492,947,885,947,947,947,947,947,947,947,947,511,495,980,1010,987,893,1624,1185,1147,1193,1216,1080,1028,1270,1274,531,910,1177,1004,1521,1252,1276,1111,1276,1164,1056,1073,1215,1159,1596,1150,1124,1065,540,591,540,837,874,572,929,972,879,973,901,569,967,973,453,458,903,453,1477,973,970,972,976,638,846,548,973,870,1285,884,864,835,656,430,656,1069];

const extra_symbols_width = {945:1002,946:996,967:917,948:953,949:834,966:1149,947:847,951:989,953:516,954:951,955:913,956:1003,957:862,959:967,960:1070,952:954,961:973,963:1017,964:797,965:944,982:1354,969:1359,958:803,968:1232,950:825,913:1194,914:1153,935:1162,916:1178,917:1086,934:1358,915:1016,919:1275,921:539,977:995,922:1189,923:1170,924:1523,925:1253,927:1281,928:1281,920:1285,929:1102,931:1041,932:1069,933:1135,962:848,937:1279,926:1092,936:1334,918:1067,978:1154,8730:986,8804:940,8260:476,8734:1453,402:811,9827:1170,9830:931,9829:1067,9824:965,8596:1768,8592:1761,8593:895,8594:1761,8595:895,710:695,177:955,8243:680,8805:947,215:995,8733:1124,8706:916,8226:626,247:977,8800:969,8801:1031,8776:976,8230:1552,175:883,8629:1454,8501:1095,8465:1002,8476:1490,8472:1493,8855:1417,8853:1417,8709:1205,8745:1276,8746:1404,8839:1426,8835:1426,8836:1426,8838:1426,8834:1426,8747:480,8712:1426,8713:1426,8736:1608,8711:1551,174:1339,169:1339,8482:1469,8719:1364,729:522,172:1033,8743:1383,8744:1383,8660:1768,8656:1496,8657:1447,8658:1496,8659:1447,8721:1182,9115:882,9144:1000,9117:882,8970:749,9127:1322,9128:1322,8491:1150,229:929,8704:1397,8707:1170,8901:524,183:519,10003:1477,732:692,295:984,9725:1780,9744:1581,8741:737,8869:1390,8857:1421};

/** @ummary Calculate approximate labels width
  * @private */
function approximateLabelWidth(label, font, fsize) {
   let len = label.length,
       symbol_width = (fsize || font.size) * font.aver_width;
   if (font.isMonospace())
      return len * symbol_width;

   let sum = 0;
   for (let i = 0; i < len; ++i) {
      let code = label.charCodeAt(i);
      if ((code >= 32) && (code<127))
         sum += base_symbols_width[code-32];
      else
         sum += extra_symbols_width[code] || 1000;
   }

   return sum/1000*symbol_width;
}

/** @summary array defines features supported by latex parser, used by both old and new parsers
  * @private */
const latex_features = [
   { name: "#it{" }, // italic
   { name: "#bf{" }, // bold
   { name: "#underline{", deco: "underline" }, // underline
   { name: "#overline{", deco: "overline" }, // overline
   { name: "#strike{", deco: "line-through" }, // line through
   { name: "#kern[", arg: 'float' }, // horizontal shift
   { name: "#lower[", arg: 'float' },  // vertical shift
   { name: "#scale[", arg: 'float' },  // font scale
   { name: "#color[", arg: 'int' },   // font color
   { name: "#font[", arg: 'int' },    // font face
   { name: "_{", low_up: "low" },  // subscript
   { name: "^{", low_up: "up" },   // superscript
   { name: "#bar{", deco: "overline" /* accent: "\u02C9" */ }, // "\u0305"
   { name: "#hat{", accent: "\u02C6", hasw: true }, // "\u0302"
   { name: "#check{", accent: "\u02C7", hasw: true }, // "\u030C"
   { name: "#acute{", accent: "\u02CA" }, // "\u0301"
   { name: "#grave{", accent: "\u02CB" }, // "\u0300"
   { name: "#dot{", accent: "\u02D9" }, // "\u0307"
   { name: "#ddot{", accent: "\u02BA", hasw: true }, // "\u0308"
   { name: "#tilde{", accent: "\u02DC", hasw: true }, // "\u0303"
   { name: "#slash{", accent: "\u2215" }, // "\u0337"
   { name: "#vec{", accent: "\u02ED", hasw: true }, // "\u0350" arrowhead
   { name: "#frac{", twolines: 'line' },
   { name: "#splitline{", twolines: true },
   { name: "#sqrt[", arg: 'int', sqrt: true }, // root with arbitrary power
   { name: "#sqrt{", sqrt: true },             // square root
   { name: "#sum", special: '\u2211', w: 0.8, h: 0.9 },
   { name: "#int", special: '\u222B', w: 0.3, h: 1.0 },
   { name: "#left[", right: "#right]", braces: "[]" },
   { name: "#left(", right: "#right)", braces: "()" },
   { name: "#left{", right: "#right}", braces: "{}" },
   { name: "#left|", right: "#right|", braces: "||" },
   { name: "#[]{", braces: "[]" },
   { name: "#(){", braces: "()" },
   { name: "#{}{", braces: "{}" },
   { name: "#||{", braces: "||" }
];

// taken from: https://sites.math.washington.edu/~marshall/cxseminar/symbol.htm, starts from 33
const symbolsMap = [0,8704,0,8707,0,0,8717,0,0,8727,0,0,8722,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8773,913,914,935,916,917,934,915,919,921,977,922,923,924,925,927,928,920,929,931,932,933,962,937,926,936,918,0,8756,0,8869,0,0,945,946,967,948,949,966,947,951,953,981,954,955,956,957,959,960,952,961,963,964,965,982,969,958,968,950,0,402,0,8764,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,978,8242,8804,8260,8734,0,9827,9830,9829,9824,8596,8592,8593,8594,8595,0,0,8243,8805,0,8733,8706,8729,0,8800,8801,8776,8230,0,0,8629,8501,8465,8476,8472,8855,8853,8709,8745,8746,8835,8839,8836,8834,8838,8712,8713,8736,8711,0,0,8482,8719,8730,8901,0,8743,8744,8660,8656,8657,8658,8659,9674,9001,0,0,8482,8721,0,0,0,0,0,0,0,0,0,0,8364,9002,8747,8992,0,8993];

// taken from http://www.alanwood.net/demos/wingdings.html, starts from 33
const wingdingsMap = [128393,9986,9985,128083,128365,128366,128367,128383,9990,128386,128387,128234,128235,128236,128237,128193,128194,128196,128463,128464,128452,8987,128430,128432,128434,128435,128436,128427,128428,9991,9997,128398,9996,128076,128077,128078,9756,9758,9757,9759,128400,9786,128528,9785,128163,9760,127987,127985,9992,9788,128167,10052,128326,10014,128328,10016,10017,9770,9775,2384,9784,9800,9801,9802,9803,9804,9805,9806,9807,9808,9809,9810,9811,128624,128629,9679,128318,9632,9633,128912,10065,10066,11047,10731,9670,10070,11045,8999,11193,8984,127989,127990,128630,128631,0,9450,9312,9313,9314,9315,9316,9317,9318,9319,9320,9321,9471,10102,10103,10104,10105,10106,10107,10108,10109,10110,10111,128610,128608,128609,128611,128606,128604,128605,128607,183,8226,9642,9898,128902,128904,9673,9678,128319,9642,9723,128962,10022,9733,10038,10036,10041,10037,11216,8982,10209,8977,11217,10026,10032,128336,128337,128338,128339,128340,128341,128342,128343,128344,128345,128346,128347,11184,11185,11186,11187,11188,11189,11190,11191,128618,128619,128597,128596,128599,128598,128592,128593,128594,128595,9003,8998,11160,11162,11161,11163,11144,11146,11145,11147,129128,129130,129129,129131,129132,129133,129135,129134,129144,129146,129145,129147,129148,129149,129151,129150,8678,8680,8679,8681,11012,8691,11008,11009,11011,11010,129196,129197,128502,10004,128503,128505];

function replaceSymbols(s, kind) {
   let res = "", m = kind == "Wingdings" ? wingdingsMap : symbolsMap;
   for (let k = 0; k < s.length; ++k) {
      let code = s.charCodeAt(k),
          new_code = (code > 32) ? m[code-33] : 0;
      res += String.fromCodePoint(new_code || code);
   }
   return res;
}

/** @summary Just add plain text to the SVG text elements
  * @private */
function producePlainText(painter, txt_node, arg) {
   arg.plain = true;
   if (arg.simple_latex)
      arg.text = translateLaTeX(arg.text); // replace latex symbols
   if (arg.font && arg.font.isSymbol)
      txt_node.text(replaceSymbols(arg.text, arg.font.isSymbol));
   else
      txt_node.text(arg.text);
}

/** @summary Check if plain text
  * @private */
function isPlainText(txt) {
   return !txt || ((txt.indexOf("#") < 0) && (txt.indexOf("{") < 0));
}

/** @ummary translate TLatex and draw inside provided g element
  * @desc use <text> together with normal <path> elements
  * @private */
function parseLatex(node, arg, label, curr) {

   let nelements = 0;

   const currG = () => { if (!curr.g) curr.g = node.append("svg:g"); return curr.g; };

   const shiftX = dx => { curr.x += Math.round(dx); };

   const extendPosition = (x1, y1, x2, y2) => {
      if (!curr.rect) {
         curr.rect = { x1: x1, y1: y1, x2: x2, y2: y2 };
      } else {
         curr.rect.x1 = Math.min(curr.rect.x1, x1);
         curr.rect.y1 = Math.min(curr.rect.y1, y1);
         curr.rect.x2 = Math.max(curr.rect.x2, x2);
         curr.rect.y2 = Math.max(curr.rect.y2, y2);
      }

      curr.rect.width = curr.rect.x2 - curr.rect.x1;
      curr.rect.height = curr.rect.y2 - curr.rect.y1;

      if (!curr.parent)
         arg.text_rect = curr.rect;
   };

   /** Position pos.g node which directly attached to curr.g and uses curr.g coordinates */
   const positionGNode = (pos, x, y, inside_gg) => {
      x = Math.round(x);
      y = Math.round(y);
      if (y)
         pos.g.attr('transform',`translate(${x},${y})`);
      else if (x)
         pos.g.attr('transform',`translate(${x})`);

      pos.rect.x1 += x;
      pos.rect.x2 += x;
      pos.rect.y1 += y;
      pos.rect.y2 += y;

      if (inside_gg)
         extendPosition(curr.x + pos.rect.x1, curr.y + pos.rect.y1, curr.x + pos.rect.x2, curr.y + pos.rect.y2);
      else
         extendPosition(pos.rect.x1, pos.rect.y1, pos.rect.x2, pos.rect.y2);
   };

   /** Create special sub-container for elements like sqrt or braces  */
   const createGG = () => {
      let gg = currG();

      // this is indicator that gg element will be the only one, one can use directly main container
      if ((nelements == 1) && !label && !curr.x && !curr.y) {
         return gg;
      }

      gg = gg.append("svg:g");

      if (curr.y)
         gg.attr('transform',`translate(${curr.x},${curr.y})`);
      else if (curr.x)
         gg.attr('transform',`translate(${curr.x})`);
      return gg;
   };

   const extractSubLabel = (check_first, lbrace, rbrace) => {
      let pos = 0, n = 1, err = false;
      if (!lbrace) lbrace = "{";
      if (!rbrace) rbrace = "}";

      const match = br => (pos + br.length <= label.length) && (label.slice(pos, pos+br.length) == br);

      if (check_first) {
         if(!match(lbrace)) err = true; else label = label.slice(lbrace.length);
      }

      while (!err && (n != 0) && (pos < label.length)) {
         if (match(lbrace)) { n++; pos += lbrace.length; } else
         if (match(rbrace)) { n--; pos += rbrace.length; } else pos++;
      }
      if ((n != 0) || err) {
         console.log(`mismatch with open ${lbrace} and closing ${rbrace} in ${label}`);
         return -1;
      }

      let sublabel = label.slice(0, pos - rbrace.length);

      label = label.slice(pos);

      return sublabel;
   };

   const createPath = (gg, dofill) => {
      return gg.append("svg:path")
               .style("stroke", dofill ? "none" : (curr.color || arg.color))
               .style("stroke-width", dofill ? null : Math.max(1, Math.round(curr.fsize*(curr.font.weight ? 0.1 : 0.07))))
               .style("fill", dofill ? (curr.color || arg.color) : "none");
   };

   const createSubPos = fscale => {
      return { lvl: curr.lvl + 1, x: 0, y: 0, fsize: curr.fsize*(fscale || 1), color: curr.color, font: curr.font, parent: curr, painter: curr.painter };
   };

   while (label) {

      let best = label.length, found = null;

      for (let n = 0; n < latex_features.length; ++n) {
         let pos = label.indexOf(latex_features[n].name);
         if ((pos >= 0) && (pos < best)) { best = pos; found = latex_features[n]; }
      }

      if (best > 0) {

         let alone = (best == label.length) && (nelements == 0) && !found;

         nelements++;

         let s = translateLaTeX(label.slice(0, best));
         if ((s.length > 0) || alone) {
            // if single text element created, place it directly in the node
            let g = curr.g || (alone ? node : currG()),
                elem = g.append("svg:text");

            if (alone && !curr.g) curr.g = elem;

            // apply font attributes only once, inherited by all other elements
            if (curr.ufont)
               curr.font.setFont(curr.g /*, 'without-size' */);

            if (curr.bold !== undefined)
               curr.g.attr('font-weight', curr.bold ? 'bold' : 'normal');

            if (curr.italic !== undefined)
               curr.g.attr('font-style', curr.italic ? 'italic' : 'normal');

            // set fill color directly to element
            elem.attr("fill", curr.color || arg.color || null);

            // set font size directly to element to avoid complex control
            if (curr.fisze !== curr.font.size)
               elem.attr("font-size", Math.round(curr.fsize));

            if (curr.font && curr.font.isSymbol)
               elem.text(replaceSymbols(s, curr.font.isSymbol));
            else
               elem.text(s);

            let rect = !isNodeJs() && !settings.ApproxTextSize && !arg.fast ? getElementRect(elem, 'nopadding') :
                          { height: curr.fsize * 1.2, width: approximateLabelWidth(s, curr.font, curr.fsize) };

            if (curr.x) elem.attr("x", curr.x);
            if (curr.y) elem.attr("y", curr.y);

            // values used for superscript
            curr.last_y1 = curr.y - rect.height*0.8;
            curr.last_y2 = curr.y + rect.height*0.2;

            extendPosition(curr.x, curr.last_y1, curr.x + rect.width, curr.last_y2);

            if (!alone) {
               shiftX(rect.width);
            } else if (curr.deco) {
               elem.attr('text-decoration', curr.deco);
               delete curr.deco; // inform that decoration was applied
            }
         }
      }

      if (!found) return true;

      // remove preceeding block and tag itself
      label = label.slice(best + found.name.length);

      nelements++;

      if (found.accent) {
         let sublabel = extractSubLabel();
         if (sublabel === -1) return false;

         let gg = createGG(),
             subpos = createSubPos();

         parseLatex(gg, arg, sublabel, subpos);

         let minw = curr.fsize*0.6, xpos = 0,
             w = subpos.rect.width,
             y1 = Math.round(subpos.rect.y1),
             dy2 = Math.round(curr.fsize*0.1), dy = dy2*2,
             dot = `a${dy2},${dy2},0,0,1,${dy},0a${dy2},${dy2},0,0,1,${-dy},0z`;

         // shift symbol when it is too small
         if (found.hasw && (w < minw)) {
            w = minw;
            xpos = (minw - subpos.rect.width) / 2;
         }

         let w5 = Math.round(w*0.5), w3 = Math.round(w*0.3), w2 = w5-w3, w8 = w5+w3;
         w = w5*2;

         positionGNode(subpos, xpos, 0, true);

         switch(found.name) {
            case "#check{": createPath(gg).attr("d",`M${w2},${y1-dy}L${w5},${y1}L${w8},${y1-dy}`); break;
            case "#acute{": createPath(gg).attr("d",`M${w5},${y1}l${dy},${-dy}`); break;
            case "#grave{": createPath(gg).attr("d",`M${w5},${y1}l${-dy},${-dy}`); break;
            case "#dot{": createPath(gg, true).attr("d",`M${w5-dy2},${y1}${dot}`); break;
            case "#ddot{": createPath(gg, true).attr("d",`M${w5-3*dy2},${y1}${dot} M${w5+dy2},${y1}${dot}`); break;
            case "#tilde{": createPath(gg).attr("d",`M${w2},${y1} a${w3},${dy},0,0,1,${w3},0 a${w3},${dy},0,0,0,${w3},0`); break;
            case "#slash{": createPath(gg).attr("d",`M${w},${y1}L0,${Math.round(subpos.rect.y2)}`); break;
            case "#vec{": createPath(gg).attr("d",`M${w2},${y1}H${w8}M${w8-dy},${y1-dy}l${dy},${dy}l${-dy},${dy}`); break;
            default: createPath(gg).attr("d",`M${w2},${y1}L${w5},${y1-dy}L${w8},${y1}`); // #hat{
         }

         shiftX(subpos.rect.width);

         continue;
      }

      if (found.twolines) {

         curr.twolines = true;

         let line1 = extractSubLabel(), line2 = extractSubLabel(true);
         if ((line1 === -1) || (line2 === -1)) return false;

         let gg = createGG(),
             fscale = (curr.parent && curr.parent.twolines) ? 0.7 : 1,
             subpos1 = createSubPos(fscale);

         parseLatex(gg, arg, line1, subpos1);

         let path = (found.twolines == 'line') ? createPath(gg) : null,
             subpos2 = createSubPos(fscale);

         parseLatex(gg, arg, line2, subpos2);

         let w = Math.max(subpos1.rect.width, subpos2.rect.width),
             dw = subpos1.rect.width - subpos2.rect.width,
             dy = -curr.fsize*0.35; // approximate position of middle line

         positionGNode(subpos1, (dw < 0 ? -dw/2 : 0), dy - subpos1.rect.y2, true);

         positionGNode(subpos2, (dw > 0 ? dw/2 : 0), dy - subpos2.rect.y1, true);

         if (path) path.attr("d", `M0,${Math.round(dy)}h${Math.round(w - curr.fsize*0.1)}`);

         shiftX(w);

         delete curr.twolines;

         continue;
      }

      const extractLowUp = name => {
         let res = {};
         if (name) {
            res[name] = extractSubLabel();
            if (res[name] === -1) return false;
         }
         while (label.length > 0) {
            if (label.charAt(0) == '_') {
               label = label.slice(1);
               res.low = !res.low ? extractSubLabel(true) : -1;
               if (res.low === -1) {
                  console.log(`error with ${found.name} low limit`);
                  return false;
               }
            } else if (label.charAt(0) == '^') {
               label = label.slice(1);
               res.up = !res.up ? extractSubLabel(true) : -1;
               if (res.up === -1) {
                  console.log(`error with ${found.name} upper limit`);
                  return false;
               }
            } else break;
         }
         return res;
      };

      if (found.low_up) {
         const subs = extractLowUp(found.low_up);
         if (!subs) return false;

         let pos_up, pos_low,
             x = curr.x, y1 = -curr.fsize, y2 = 0.25*curr.fsize, w1 = 0, w2 = 0;

         if (subs.up) {
            pos_up = createSubPos(0.6);
            parseLatex(currG(), arg, subs.up, pos_up);
         }

         if (subs.low) {
            pos_low = createSubPos(0.6);
            parseLatex(currG(), arg, subs.low, pos_low);
         }

         if ((curr.last_y1 !== undefined) && (curr.last_y2 !== undefined)) {
            y1 = curr.last_y1; y2 = curr.last_y2;
         }

         if (pos_up) {
            positionGNode(pos_up, x, y1 - pos_up.rect.y1 - curr.fsize*0.1);
            w1 = pos_up.rect.width;
         }

         if (pos_low) {
            positionGNode(pos_low, x, y2 - pos_low.rect.y2 + curr.fsize*0.1);
            w2 = pos_low.rect.width;
         }

         shiftX(Math.max(w1,w2));

         continue;
      }

      if (found.special) {
         // this is sum and integral, now make fix height, later can adjust to right-content size

         let subs = extractLowUp() || {},
             gg = createGG(), path = createPath(gg),
             h = Math.round(curr.fsize*1.7), w = Math.round(curr.fsize), r = Math.round(h*0.1), x_up, x_low;

         if (found.name == "#sum") {
            x_up = x_low = w/2;
            path.attr("d",`M${w},${Math.round(-0.75*h)}h${-w}l${Math.round(0.4*w)},${Math.round(0.3*h)}l${Math.round(-0.4*w)},${Math.round(0.7*h)}h${w}`);
         } else {
            x_up = 3*r; x_low = r;
            path.attr("d",`M0,${Math.round(0.25*h-r)}a${r},${r},0,0,0,${2*r},0v${2*r-h}a${r},${r},0,1,1,${2*r},0`);
            // path.attr('transform','skewX(-3)'); could use skewX for italic-like style
         }

         extendPosition(curr.x, curr.y - 0.6*h, curr.x + w, curr.y + 0.4*h);

         if (subs.low) {
            let subpos1 = createSubPos(0.6);
            parseLatex(gg, arg, subs.low, subpos1);
            positionGNode(subpos1, (x_low - subpos1.rect.width/2), 0.25*h - subpos1.rect.y1, true);
         }

         if (subs.up) {
            let subpos2 = createSubPos(0.6);
            parseLatex(gg, arg, subs.up, subpos2);
            positionGNode(subpos2, (x_up - subpos2.rect.width/2), -0.75*h - subpos2.rect.y2, true);
         }

         shiftX(w);

         continue;
      }

      if (found.braces) {
         let rbrace = found.right, lbrace = rbrace ? found.name : "{",
             sublabel = extractSubLabel(false, lbrace, rbrace),
             gg = createGG(),
             subpos = createSubPos(),
             path1 = createPath(gg);

         parseLatex(gg, arg, sublabel, subpos);

         let path2 = createPath(gg),
             w = Math.max(2, Math.round(curr.fsize*0.2)),
             r = subpos.rect, dy = Math.round(r.y2 - r.y1),
             r_y1 = Math.round(r.y1), r_width = Math.round(r.width);

         switch (found.braces) {
            case "||":
               path1.attr("d",`M${w},${r_y1}v${dy}`);
               path2.attr("d",`M${3*w+r_width},${r_y1}v${dy}`);
               break;
            case "[]":
               path1.attr("d",`M${2*w},${r_y1}h${-w}v${dy}h${w}`);
               path2.attr("d",`M${2*w+r_width},${r_y1}h${w}v${dy}h${-w}`);
               break;
            case "{}":
               path1.attr("d",`M${2*w},${r_y1} a${w},${w},0,0,0,${-w},${w} v${dy/2-2*w} a${w},${w},0,0,1,${-w},${w} a${w},${w},0,0,1,${w},${w} v${dy/2-2*w} a${w},${w},0,0,0,${w},${w}`);
               path2.attr("d",`M${2*w+r_width},${r_y1} a${w},${w},0,0,1,${w},${w} v${dy/2-2*w} a${w},${w},0,0,0,${w},${w} a${w},${w},0,0,0,${-w},${w} v${dy/2-2*w} a${w},${w},0,0,1,${-w},${w}`);
               break;
            default: // ()
               path1.attr("d",`M${w},${r_y1}a${4*dy},${4*dy},0,0,0,0,${dy}`);
               path2.attr("d",`M${3*w+r_width},${r_y1}a${4*dy},${4*dy},0,0,1,0,${dy}`);
         }

         positionGNode(subpos, 2*w, 0, true);

         extendPosition(curr.x, curr.y + r.y1, curr.x + 4*w + r.width, curr.y + r.y2);

         shiftX(4*w + r.width);

         // values used for superscript
         curr.last_y1 = r.y1;
         curr.last_y2 = r.y2;

         continue;
      }

      if (found.deco) {
         let sublabel = extractSubLabel(),
             gg = createGG(),
             subpos = createSubPos();

         subpos.deco = found.deco;

         parseLatex(gg, arg, sublabel, subpos);

         let r = subpos.rect;

         if (subpos.deco) {
            let path = createPath(gg), r_width = Math.round(r.width);
            switch(subpos.deco) {
               case "underline": path.attr("d",`M0,${Math.round(r.y2)}h${r_width}`); break;
               case "overline": path.attr("d",`M0,${Math.round(r.y1)}h${r_width}`); break;
               case "line-through": path.attr("d",`M0,${Math.round(0.45*r.y1+0.55*r.y2)}h${r_width}`); break;
            }
         }

         positionGNode(subpos, 0, 0, true);

         shiftX(r.width);

         continue;
      }

      if (found.name == "#bf{" || found.name == "#it{") {
         let sublabel = extractSubLabel();
         if (sublabel === -1) return false;

         let subpos = createSubPos();

         if (found.name == "#bf{")
            subpos.bold = !subpos.bold;
         else
            subpos.italic = !subpos.italic;

         parseLatex(currG(), arg, sublabel, subpos);

         positionGNode(subpos, curr.x, curr.y);

         shiftX(subpos.rect.width);

         continue;
      }

      let foundarg = 0;

      if (found.arg) {
         let pos = label.indexOf("]{");
         if (pos < 0) { console.log('missing argument for ', found.name); return false; }
         foundarg = label.slice(0, pos);
         if (found.arg == 'int') {
            foundarg = parseInt(foundarg);
            if (!Number.isInteger(foundarg)) { console.log('wrong int argument', label.slice(0, pos)); return false; }
         } else if (found.arg == 'float') {
            foundarg = parseFloat(foundarg);
            if (!Number.isFinite(foundarg)) { console.log('wrong float argument', label.slice(0, pos)); return false; }
         }
         label = label.slice(pos + 2);
      }

      if ((found.name == "#kern[") || (found.name == "#lower[")) {
         let sublabel = extractSubLabel();
         if (sublabel === -1) return false;

         let subpos = createSubPos();

         parseLatex(currG(), arg, sublabel, subpos);

         let shiftx = 0, shifty = 0;
         if (found.name == "kern[") shiftx = foundarg; else shifty = foundarg;

         positionGNode(subpos, curr.x + shiftx * subpos.rect.width, curr.y + shifty * subpos.rect.height);

         shiftX(subpos.rect.width * (shiftx > 0 ? 1 + foundarg : 1));

         continue;
      }

      if ((found.name == "#color[") || (found.name == "#scale[") || (found.name == "#font[")) {

         let sublabel = extractSubLabel();
         if (sublabel === -1) return false;

         let subpos = createSubPos();

         if (found.name == "#color[")
            subpos.color = curr.painter.getColor(foundarg);
         else if (found.name == "#font[") {
            subpos.font = new FontHandler(foundarg);
            subpos.ufont = true; // mark that custom font is applied
         } else
            subpos.fsize *= foundarg;

         parseLatex(currG(), arg, sublabel, subpos);

         positionGNode(subpos, curr.x, curr.y);

         shiftX(subpos.rect.width);

         continue;
      }

     if (found.sqrt) {
         let sublabel = extractSubLabel();
         if (sublabel === -1) return false;

         let gg = createGG(), subpos0, subpos = createSubPos();

         if (found.arg) {
            subpos0 = createSubPos(0.7);
            parseLatex(gg, arg, foundarg.toString(), subpos0);
         }

         // placeholder for the sqrt sign
         let path = createPath(gg);

         parseLatex(gg, arg, sublabel, subpos);

         let r = subpos.rect,
             h = Math.round(r.height),
             h1 = Math.round(r.height*0.1),
             w = Math.round(r.width), midy = Math.round((r.y1 + r.y2)/2),
             f2 = Math.round(curr.fsize*0.2), r_y2 = Math.round(r.y2);

         if (subpos0)
            positionGNode(subpos0, 0, midy - subpos0.fsize*0.3, true);

         path.attr("d", `M0,${midy}h${h1}l${h1},${r_y2-midy-f2}l${h1},${-h+f2}h${Math.round(h*0.2+w)}v${h1}`);

         positionGNode(subpos, h*0.4, 0, true);

         extendPosition(curr.x, curr.y + r.y1-curr.fsize*0.1, curr.x + w + h*0.6, curr.y + r.y2);

         shiftX(w + h*0.6);

         continue;
     }

   }

   return true;
}

/** @ummary translate TLatex and draw inside provided g element
  * @desc use <text> together with normal <path> elements
  * @private */
function produceLatex(painter, node, arg) {

   let pos = { lvl: 0, g: node, x: 0, y: 0, dx: 0, dy: -0.1, fsize: arg.font_size, font: arg.font, parent: null, painter: painter };

   return parseLatex(node, arg, arg.text, pos);
}

let _mj_loading;

/** @summary Load MathJax functionality,
  * @desc one need not only to load script but wait for initialization
  * @private */
function loadMathjax() {
   let loading = (_mj_loading !== undefined);

   if (!loading && (typeof globalThis.MathJax != "undefined"))
      return Promise.resolve(globalThis.MathJax);

   if (!loading) _mj_loading = [];

   let promise = new Promise(resolve => { _mj_loading ? _mj_loading.push(resolve) : resolve(globalThis.MathJax); });

   if (loading) return promise;

   let svg_config = {
       scale: 1,                      // global scaling factor for all expressions
       minScale: .5,                  // smallest scaling factor to use
       mtextInheritFont: false,       // true to make mtext elements use surrounding font
       merrorInheritFont: true,       // true to make merror text use surrounding font
       mathmlSpacing: false,          // true for MathML spacing rules, false for TeX rules
       skipAttributes: {},            // RFDa and other attributes NOT to copy to the output
       exFactor: .5,                  // default size of ex in em units
       displayAlign: 'center',        // default for indentalign when set to 'auto'
       displayIndent: '0',            // default for indentshift when set to 'auto'
       fontCache: 'local',            // or 'global' or 'none'
       localID: null,                 // ID to use for local font cache (for single equation processing)
       internalSpeechTitles: true,    // insert <title> tags with speech content
       titleID: 0                     // initial id number to use for aria-labeledby titles
   };

   if (!isNodeJs()) {
      window.MathJax = {
         options: {
            enableMenu: false
         },
         loader: {
            load: ['[tex]/color', '[tex]/upgreek', '[tex]/mathtools', '[tex]/physics']
         },
         tex: {
            packages: {'[+]': ['color', 'upgreek', 'mathtools', 'physics']}
         },
         svg: svg_config,
         startup: {
            ready: function() {
               MathJax.startup.defaultReady();
               let arr = _mj_loading;
               _mj_loading = undefined;
               arr.forEach(func => func(globalThis.MathJax));
            }
         }
      };

      return loadScript('../../mathjax/3.2.0/es5/tex-svg.js')
               .catch(() => loadScript('https://cdn.jsdelivr.net/npm/mathjax@3.2.0/es5/tex-svg.js'))
               .then(() => promise);
   }

   let myJSDOM;

   return _loadJSDOM().then(handle => { myJSDOM = handle.JSDOM; return import('mathjax'); }).then(mj => {

      // return Promise with mathjax loading
      mj.init({
         loader: {
            load: ['input/tex', 'output/svg', '[tex]/color', '[tex]/upgreek', '[tex]/mathtools', '[tex]/physics']
          },
          tex: {
             packages: {'[+]': ['color', 'upgreek', 'mathtools', 'physics']}
          },
          svg: svg_config,
          config: {
             JSDOM: myJSDOM
          },
          startup: {
             typeset: false,
             ready: function() {
                   MathJax.startup.registerConstructor('jsdomAdaptor', () => {
                      return new MathJax._.adaptors.HTMLAdaptor.HTMLAdaptor(new MathJax.config.config.JSDOM().window);
                   });
                   MathJax.startup.useAdaptor('jsdomAdaptor', true);
                   MathJax.startup.defaultReady();
                   let arr = _mj_loading;
                   _mj_loading = undefined;
                   arr.forEach(func => func(MathJax));
             }
          }
      });

      return promise;
   });

}

const math_symbols_map = {
      '#LT': "\\langle",
      '#GT': "\\rangle",
      '#club': "\\clubsuit",
      '#spade': "\\spadesuit",
      '#heart': "\\heartsuit",
      '#diamond': "\\diamondsuit",
      '#voidn': "\\wp",
      '#voidb': "f",
      '#copyright': "(c)",
      '#ocopyright': "(c)",
      '#trademark': "TM",
      '#void3': "TM",
      '#oright': "R",
      '#void1': "R",
      '#3dots': "\\ldots",
      '#lbar': "\\mid",
      '#void8': "\\mid",
      '#divide': "\\div",
      '#Jgothic': "\\Im",
      '#Rgothic': "\\Re",
      '#doublequote': "\"",
      '#plus': "+",
      '#minus': "-",
      '#\/': "/",
      '#upoint': ".",
      '#aa': "\\mathring{a}",
      '#AA': "\\mathring{A}",
      '#omicron': "o",
      '#Alpha': "A",
      '#Beta': "B",
      '#Epsilon': "E",
      '#Zeta': "Z",
      '#Eta': "H",
      '#Iota': "I",
      '#Kappa': "K",
      '#Mu': "M",
      '#Nu': "N",
      '#Omicron': "O",
      '#Rho': "P",
      '#Tau': "T",
      '#Chi': "X",
      '#varomega': "\\varpi",
      '#corner': "?",
      '#ltbar': "?",
      '#bottombar': "?",
      '#notsubset': "?",
      '#arcbottom': "?",
      '#cbar': "?",
      '#arctop': "?",
      '#topbar': "?",
      '#arcbar': "?",
      '#downleftarrow': "?",
      '#splitline': "\\genfrac{}{}{0pt}{}",
      '#it': "\\textit",
      '#bf': "\\textbf",
      '#frac': "\\frac",
      '#left{': "\\lbrace",
      '#right}': "\\rbrace",
      '#left\\[': "\\lbrack",
      '#right\\]': "\\rbrack",
      '#\\[\\]{': "\\lbrack",
      ' } ': "\\rbrack",
      '#\\[': "\\lbrack",
      '#\\]': "\\rbrack",
      '#{': "\\lbrace",
      '#}': "\\rbrace",
      ' ': "\\;"
 };

 const mathjax_remap = {
   'upDelta': "Updelta",
   'upGamma': "Upgamma",
   'upLambda': "Uplambda",
   'upOmega': "Upomega",
   'upPhi': "Upphi",
   'upPi': "Uppi",
   'upPsi': "Uppsi",
   'upSigma': "Upsigma",
   'upTheta': "Uptheta",
   'upUpsilon': "Upupsilon",
   'upXi': "Upxi",
   'notcong': "ncong",
   'notgeq': "ngeq",
   'notgr': "ngtr",
   'notless': "nless",
   'notleq': "nleq",
   'notsucc': "nsucc",
   'notprec': "nprec",
   'notsubseteq': "nsubseteq",
   'notsupseteq': "nsupseteq",
   'openclubsuit': "clubsuit",
   'openspadesuit': "spadesuit",
   'dasharrow': "dashrightarrow",
   'comp': "circ",
   'iiintop': "iiint",
   'iintop': "iint",
   'ointop': "oint"
 };

 const mathjax_unicode = {
   'Digamma': 0x3DC,
   'upDigamma': 0x3DC,
   'digamma': 0x3DD,
   'updigamma': 0x3DD,
   'Koppa': 0x3DE,
   'koppa': 0x3DF,
   'upkoppa': 0x3DF,
   'upKoppa': 0x3DE,
   'VarKoppa': 0x3D8,
   'upVarKoppa': 0x3D8,
   'varkoppa': 0x3D9,
   'upvarkoppa': 0x3D9,
   'varkappa': 0x3BA, // not found archaic kappa - use normal
   'upvarkappa': 0x3BA,
   'varbeta': 0x3D0, // not found archaic beta - use normal
   'upvarbeta': 0x3D0,
   'Sampi': 0x3E0,
   'upSampi': 0x3E0,
   'sampi': 0x3E1,
   'upsampi': 0x3E1,
   'Stigma': 0x3DA,
   'upStigma': 0x3DA,
   'stigma': 0x3DB,
   'upstigma': 0x3DB,
   'San': 0x3FA,
   'upSan': 0x3FA,
   'san': 0x3FB,
   'upsan': 0x3FB,
   'Sho': 0x3F7,
   'upSho': 0x3F7,
   'sho': 0x3F8,
   'upsho': 0x3F8,
   'P': 0xB6,
   'aa': 0xB0,
   'bulletdashcirc': 0x22B7,
   'circdashbullet': 0x22B6,
   'downuparrows': 0x21F5,
   'updownarrows': 0x21C5,
   'dashdownarrow': 0x21E3,
   'dashuparrow': 0x21E1,
   'complement': 0x2201,
   'dbar': 0x18C,
   'ddddot': 0x22EF,
   'dddot': 0x22EF,
   'ddots': 0x22F1,
   'defineequal': 0x225D,
   'defineeq': 0x225D,
   'downdownharpoons': 0x2965,
   'downupharpoons': 0x296F,
   'updownharpoons': 0x296E,
   'upupharpoons': 0x2963,
   'hateq': 0x2259,
   'ldbrack': 0x27E6,
   'rdbrack': 0x27E7,
   'leadsfrom': 0x219C,
   'leftsquigarrow': 0x21DC,
   'lightning': 0x2607,
   'napprox': 0x2249,
   'nasymp': 0x226D,
   'nequiv': 0x2262,
   'nsimeq': 0x2244,
   'nsubseteq': 0x2288,
   'nsubset': 0x2284,
   'notapprox': 0x2249,
   'notasymp': 0x226D,
   'notequiv': 0x2262,
   'notni': 0x220C,
   'notsimeq': 0x2244,
   'notsubseteq': 0x2288,
   'notsubset': 0x2284,
   'notsupseteq': 0x2289,
   'notsupset': 0x2285,
   'nsupset': 0x2285,
   'setdif': 0x2216,
   'simarrow': 0x2972,
   't': 0x2040,
   'u': 0x2C7,
   'v': 0x2C7,
   'undercurvearrowright': 0x293B,
   'updbar': 0x18C,
   'wwbar': 0x2015,
   'awointop': 0x2232,
   'awoint': 0x2233,
   'barintop': 0x2A1C,
   'barint': 0x2A1B,
   'cwintop': 0x2231, // no opposite direction, use same
   'cwint': 0x2231,
   'cwointop': 0x2233,
   'cwoint': 0x2232,
   'oiiintop': 0x2230,
   'oiiint': 0x2230,
   'oiintop': 0x222F,
   'oiint': 0x222F,
   'slashintop': 0x2A0F,
   'slashint': 0x2A0F
 };

const mathjax_asis = ['"', "'", "`", "=", "~"];

/** @summary Function translates ROOT TLatex into MathJax format
  * @private */
function translateMath(str, kind, color, painter) {

   if (kind != 2) {
      for (let x in math_symbols_map)
         str = str.replace(new RegExp(x, 'g'), math_symbols_map[x]);

      for (let x in symbols_map)
         if (x.length > 2)
            str = str.replace(new RegExp(x, 'g'), "\\" + x.slice(1));

      // replace all #color[]{} occurances
      let clean = "", first = true;
      while (str) {
         let p = str.indexOf("#color[");
         if ((p < 0) && first) { clean = str; break; }
         first = false;
         if (p != 0) {
            let norm = (p < 0) ? str : str.slice(0, p);
            clean += norm;
            if (p < 0) break;
         }

         str = str.slice(p + 7);
         p = str.indexOf("]{");
         if (p <= 0) break;
         let colindx = parseInt(str.slice(0, p));
         if (!Number.isInteger(colindx)) break;
         let col = painter.getColor(colindx), cnt = 1;
         str = str.slice(p + 2);
         p = -1;
         while (cnt && (++p < str.length)) {
            if (str[p] == '{') cnt++; else if (str[p] == '}') cnt--;
         }
         if (cnt != 0) break;

         let part = str.slice(0, p);
         str = str.slice(p + 1);
         if (part)
            clean += "\\color{" + col + '}{' + part + "}";
      }

      str = clean;
   } else {
      if (str == "\\^") str = "\\unicode{0x5E}";
      if (str == "\\vec") str = "\\unicode{0x2192}";
      str = str.replace(/\\\./g, "\\unicode{0x2E}").replace(/\\\^/g, "\\hat");
      for (let x in mathjax_unicode)
         str = str.replace(new RegExp(`\\\\\\b${x}\\b`, 'g'), `\\unicode{0x${mathjax_unicode[x].toString(16)}}`);
      for(let x in mathjax_asis)
         str = str.replace(new RegExp(`(\\\\${mathjax_asis[x]})`, 'g'), `\\unicode{0x${mathjax_asis[x].charCodeAt(0).toString(16)}}`);
      for (let x in mathjax_remap)
         str = str.replace(new RegExp(`\\\\\\b${x}\\b`, 'g'), `\\${mathjax_remap[x]}`);
   }

   if (typeof color != 'string') return str;

   // MathJax SVG converter use colors in normal form
   //if (color.indexOf("rgb(")>=0)
   //   color = color.replace(/rgb/g, "[RGB]")
   //                .replace(/\(/g, '{')
   //                .replace(/\)/g, '}');
   return "\\color{" + color + '}{' + str + "}";
}

/** @summary Workaround to fix size attributes in MathJax SVG
  * @private */
function repairMathJaxSvgSize(painter, mj_node, svg, arg) {
   let transform = value => {
      if (!value || (typeof value !== "string") || (value.length < 3)) return null;
      let p = value.indexOf("ex");
      if ((p < 0) || (p !== value.length - 2)) return null;
      value = parseFloat(value.slice(0, p));
      return Number.isFinite(value) ? value * arg.font.size * 0.5 : null;
   };

   let width = transform(svg.getAttribute("width")),
       height = transform(svg.getAttribute("height")),
       valign = svg.getAttribute("style");

   if (valign && (valign.length > 18) && valign.indexOf("vertical-align:") == 0) {
      let p = valign.indexOf("ex;");
      valign = ((p > 0) && (p == valign.length - 3)) ? transform(valign.slice(16, valign.length - 1)) : null;
   } else {
      valign = null;
   }

   width = (!width || (width <= 0.5)) ? 1 : Math.round(width);
   height = (!height || (height <= 0.5)) ? 1 : Math.round(height);

   svg.setAttribute("width", width);
   svg.setAttribute('height', height);
   svg.removeAttribute("style");

   if (!isNodeJs()) {
      let box = getElementRect(mj_node, 'bbox');
      width = 1.05 * box.width; height = 1.05 * box.height;
   }

   arg.valign = valign;

   if (arg.scale)
      painter.scaleTextDrawing(Math.max(width / arg.width, height / arg.height), arg.draw_g);
}

/** @summary Apply attributes to mathjax drawing
  * @private */
function applyAttributesToMathJax(painter, mj_node, svg, arg, font_size, svg_factor) {
   let mw = parseInt(svg.attr("width")),
       mh = parseInt(svg.attr("height"));

   if (Number.isInteger(mh) && Number.isInteger(mw)) {
      if (svg_factor > 0.) {
         mw = mw / svg_factor;
         mh = mh / svg_factor;
         svg.attr("width", Math.round(mw)).attr("height", Math.round(mh));
      }
   } else {
      let box = getElementRect(mj_node, 'bbox'); // sizes before rotation
      mw = box.width || mw || 100;
      mh = box.height || mh || 10;
   }

   if ((svg_factor > 0.) && arg.valign) arg.valign = arg.valign / svg_factor;

   if (arg.valign === null) arg.valign = (font_size - mh) / 2;

   let sign = { x: 1, y: 1 }, nx = "x", ny = "y";
   if (arg.rotate == 180) { sign.x = sign.y = -1; } else
      if ((arg.rotate == 270) || (arg.rotate == 90)) {
         sign.x = (arg.rotate == 270) ? -1 : 1;
         sign.y = -sign.x;
         nx = "y"; ny = "x"; // replace names to which align applied
      }

   if (arg.align[0] == 'middle') arg[nx] += sign.x * (arg.width - mw) / 2; else
      if (arg.align[0] == 'end') arg[nx] += sign.x * (arg.width - mw);

   if (arg.align[1] == 'middle') arg[ny] += sign.y * (arg.height - mh) / 2; else
      if (arg.align[1] == 'bottom') arg[ny] += sign.y * (arg.height - mh); else
         if (arg.align[1] == 'bottom-base') arg[ny] += sign.y * (arg.height - mh - arg.valign);

   let trans = `translate(${arg.x},${arg.y})`;
   if (arg.rotate) trans += ` rotate(${arg.rotate})`;

   mj_node.attr('transform', trans).attr('visibility', null);
}

/** @summary Produce text with MathJax
  * @private */
function produceMathjax(painter, mj_node, arg) {
   let mtext = translateMath(arg.text, arg.latex, arg.color, painter),
       options = { em: arg.font.size, ex: arg.font.size/2, family: arg.font.name, scale: 1, containerWidth: -1, lineWidth: 100000 };

   return loadMathjax()
          .then(() => MathJax.tex2svgPromise(mtext, options))
          .then(elem => {
              // when adding element to new node, it will be removed from original parent
              let svg = elem.querySelector('svg');

              mj_node.append(function() { return svg; });

              repairMathJaxSvgSize(painter, mj_node, svg, arg);

              arg.applyAttributesToMathJax = applyAttributesToMathJax;
              return true;
           });
}

/** @summary Just typeset HTML node with MathJax
  * @private */
function typesetMathjax(node) {
   return loadMathjax().then(() => MathJax.typesetPromise(node ? [node] : undefined));
}

export { symbols_map, translateLaTeX, producePlainText, isPlainText, produceLatex, loadMathjax, produceMathjax, typesetMathjax };

