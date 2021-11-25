/// @file JSRoot.latex.js
/// Latex / MathJax processing

JSROOT.define(['d3', 'painter'], (d3, jsrp) => {

   "use strict";

   let symbols_map = {
      // greek letters
      '#alpha': '\u03B1',
      '#beta': '\u03B2',
      '#chi': '\u03C7',
      '#delta': '\u03B4',
      '#varepsilon': '\u03B5',
      '#phi': '\u03C6',
      '#gamma': '\u03B3',
      '#eta': '\u03B7',
      '#iota': '\u03B9',
      '#varphi': '\u03C6',
      '#kappa': '\u03BA',
      '#lambda': '\u03BB',
      '#mu': '\u03BC',
      '#nu': '\u03BD',
      '#omicron': '\u03BF',
      '#pi': '\u03C0',
      '#theta': '\u03B8',
      '#rho': '\u03C1',
      '#sigma': '\u03C3',
      '#tau': '\u03C4',
      '#upsilon': '\u03C5',
      '#varomega': '\u03D6',
      '#omega': '\u03C9',
      '#xi': '\u03BE',
      '#psi': '\u03C8',
      '#zeta': '\u03B6',
      '#Alpha': '\u0391',
      '#Beta': '\u0392',
      '#Chi': '\u03A7',
      '#Delta': '\u0394',
      '#Epsilon': '\u0395',
      '#Phi': '\u03A6',
      '#Gamma': '\u0393',
      '#Eta': '\u0397',
      '#Iota': '\u0399',
      '#vartheta': '\u03D1',
      '#Kappa': '\u039A',
      '#Lambda': '\u039B',
      '#Mu': '\u039C',
      '#Nu': '\u039D',
      '#Omicron': '\u039F',
      '#Pi': '\u03A0',
      '#Theta': '\u0398',
      '#Rho': '\u03A1',
      '#Sigma': '\u03A3',
      '#Tau': '\u03A4',
      '#Upsilon': '\u03A5',
      '#varsigma': '\u03C2',
      '#Omega': '\u03A9',
      '#Xi': '\u039E',
      '#Psi': '\u03A8',
      '#Zeta': '\u0396',
      '#varUpsilon': '\u03D2',
      '#epsilon': '\u03B5',

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
         str = str.substr(1, str.length - 2);

      return str.replace(symbolsRegexCache, ch => symbols_map[ch]).replace(/\{\}/g, "");
   };

   /** @summary handle for latex processing
     * @alias Latex
     * @memberof JSROOT
     * @private */
   let ltx = {};

   // required only for symbols.html to generate list of scale for symbols
   ltx.symbols_map = symbols_map;


   // array with relative width of base symbols from range 32..126
   const base_symbols_width = [453,535,661,973,955,1448,1242,324,593,596,778,1011,431,570,468,492,947,885,947,947,947,947,947,947,947,947,511,495,980,1010,987,893,1624,1185,1147,1193,1216,1080,1028,1270,1274,531,910,1177,1004,1521,1252,1276,1111,1276,1164,1056,1073,1215,1159,1596,1150,1124,1065,540,591,540,837,874,572,929,972,879,973,901,569,967,973,453,458,903,453,1477,973,970,972,976,638,846,548,973,870,1285,884,864,835,656,430,656,1069];

   const extra_symbols_width = {945:1002,946:996,967:917,948:953,949:834,966:1149,947:847,951:989,953:516,954:951,955:913,956:1003,957:862,959:967,960:1070,952:954,961:973,963:1017,964:797,965:944,982:1354,969:1359,958:803,968:1232,950:825,913:1194,914:1153,935:1162,916:1178,917:1086,934:1358,915:1016,919:1275,921:539,977:995,922:1189,923:1170,924:1523,925:1253,927:1281,928:1281,920:1285,929:1102,931:1041,932:1069,933:1135,962:848,937:1279,926:1092,936:1334,918:1067,978:1154,8730:986,8804:940,8260:476,8734:1453,402:811,9827:1170,9830:931,9829:1067,9824:965,8596:1768,8592:1761,8593:895,8594:1761,8595:895,710:695,177:955,8243:680,8805:947,215:995,8733:1124,8706:916,8226:626,247:977,8800:969,8801:1031,8776:976,8230:1552,175:883,8629:1454,8501:1095,8465:1002,8476:1490,8472:1493,8855:1417,8853:1417,8709:1205,8745:1276,8746:1404,8839:1426,8835:1426,8836:1426,8838:1426,8834:1426,8747:480,8712:1426,8713:1426,8736:1608,8711:1551,174:1339,169:1339,8482:1469,8719:1364,729:522,172:1033,8743:1383,8744:1383,8660:1768,8656:1496,8657:1447,8658:1496,8659:1447,8721:1182,9115:882,9144:1000,9117:882,8970:749,9127:1322,9128:1322,8491:1150,229:929,8704:1397,8707:1170,8901:524,183:519,10003:1477,732:692,295:984,9725:1780,9744:1581,8741:737,8869:1390,8857:1421};

   /** @ummary Calculate approximate labels width
     * @private */
   const approximateLabelWidth = (label, font, fsize) => {
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
   };

   /** array defines features supported by latex parser, used by both old and new parsers */
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



   ltx.translateLaTeX = translateLaTeX;

   /** @summary Just add plain text to the SVG text elements
     * @private */
   ltx.producePlainText = function(painter, txt_node, arg) {
      arg.plain = true;
      if (arg.simple_latex)
         arg.text = translateLaTeX(arg.text); // replace latex symbols
      txt_node.text(arg.text);
   }

   /** @summary Check if plain text
     * @private */
   ltx.isPlainText = function(txt) {
      return !txt || ((txt.indexOf("#") < 0) && (txt.indexOf("{") < 0));
   }

   /** @ummary draw TLatex inside element
     * @desc attempt to implement subset of TLatex with plain SVG text and tspan elements
     * @deprecated
     * @private */
   ltx.produceOldLatex = function(painter, node, arg, label, curr) {

      if (!curr) {
         // initial dy = -0.1 is to move complete from very bottom line like with normal text drawing
         curr = { lvl: 0, x: 0, y: 0, dx: 0, dy: -0.1, fsize: arg.font_size, parent: null };
         label = arg.text;
         arg.mainnode = node;
      }

      const extend_pos = (pos, value) => {

         let dx1, dx2, dy1, dy2;

         if (typeof value == 'string') {
            if (!pos.rect) pos.rect = { x: pos.x, y: pos.y, height: 0, width: 0 };
            dx1 = -pos.x;
            pos.x += approximateLabelWidth(value, arg.font, pos.fsize); // value.length * arg.font.aver_width * pos.fsize;
            dx2 = pos.x;
            dy1 = -(pos.y - pos.fsize * 1.1);
            dy2 = pos.y + pos.fsize * 0.1;
         } else {
            if (!pos.rect) pos.rect = JSROOT.extend({}, value);
            dx1 = -value.x;
            dx2 = value.x + value.width;
            dy1 = -value.y;
            dy2 = value.y + value.height;
         }

         let rect = pos.rect;

         dx1 += rect.x;
         dx2 -= (rect.x + rect.width);
         dy1 += rect.y;
         dy2 -= (rect.y + rect.height);

         if (dx1 > 0) { rect.x -= dx1; rect.width += dx1; }
         if (dx2 > 0) rect.width += dx2;
         if (dy1 > 0) { rect.y -= dy1; rect.height += dy1; }
         if (dy2 > 0) rect.height += dy2;

         if (pos.parent) return extend_pos(pos.parent, rect);

         // calculate dimensions for the
         arg.text_rect = rect;

         let h = rect.height, mid = rect.y + rect.height / 2;

         if (h > 0) {
            arg.mid_shift = -mid / h || 0.001;        // relative shift to get latex middle at given point
            arg.top_shift = -rect.y / h || 0.001; // relative shift to get latex top at given point
         }
      };

      const makeem = value => {
         if (Math.abs(value) < 1e-2) return null; // very small values not needed, attribute will be removed
         if (value == Math.round(value)) return Math.round(value) + "em";
         let res = value.toFixed(2);
         if (res.indexOf("0.") == 0) res = res.substr(1); else
            if (res.indexOf("-0.") == 0) res = "-." + res.substr(3);
         if (res[res.length - 1] == '0') res = res.substr(0, res.length - 1);
         return res + "em";
      };

      const get_boundary = (element, approx_rect) => {
         // actually, it is workaround for getBBox() or getElementBounday,
         // which is not implemented for tspan element in Firefox

         if (JSROOT.nodejs || !element || element.empty())
            return approx_rect || { height: 0, width: 0 };

         let important = [], prnt = element.node();

         while (prnt && (prnt != arg.mainnode.node())) {
            important.push(prnt);
            prnt = prnt.parentNode;
         }

         element.selectAll('tspan').each(function() { important.push(this) });

         let tspans = arg.mainnode.selectAll('tspan');

         // this is just workaround to know that many elements are created and in Chrome we need to redo them once again
         if (tspans.size() > 3) arg.large_latex = true;

         tspans.each(function() { if (important.indexOf(this) < 0) d3.select(this).attr('display', 'none'); });

         let box = jsrp.getElementRect(arg.mainnode, 'bbox');

         tspans.each(function() { if (important.indexOf(this) < 0) d3.select(this).attr('display', null); });

         return box;
      };

      let isany = false, best, found, foundarg, pos, n, subnode, subnode1, subpos = null, prevsubpos = null;

      while (label) {

         best = label.length; found = null; foundarg = null;

         for (n = 0; n < latex_features.length; ++n) {
            pos = label.indexOf(latex_features[n].name);
            if ((pos >= 0) && (pos < best)) { best = pos; found = latex_features[n]; }
         }

         if (!found && !isany) {
            let s = translateLaTeX(label);
            if (!curr.lvl && (s == label)) {
               // nothing need to be done - can do plain svg text
               ltx.producePlainText(painter, node, arg);
               return true;
            }
            extend_pos(curr, s);

            if (curr.accent && (s.length == 1)) {
               let elem = node.append('svg:tspan').text(s),
                  rect = get_boundary(elem, { width: 10000 }),
                  w = Math.min(rect.width / curr.fsize, 0.5); // at maximum, 0.5 should be used

               node.append('svg:tspan').attr('dx', makeem(curr.dx - w)).attr('dy', makeem(curr.dy - 0.2)).text(curr.accent);
               curr.dy = 0.2; // compensate hat
               curr.dx = Math.max(0.2, w - 0.2); // extra horizontal gap
               curr.accent = false;
            } else {
               node.text(s);
            }
            return true;
         }

         if (best > 0) {
            let s = translateLaTeX(label.substr(0, best));
            if (s.length > 0) {
               extend_pos(curr, s);
               node.append('svg:tspan')
                  .attr('dx', makeem(curr.dx))
                  .attr('dy', makeem(curr.dy))
                  .text(s);
               curr.dx = curr.dy = 0;
            }
            subpos = null; // indicate that last element is plain
            delete curr.special; // and any special handling is also over
            delete curr.next_super_dy; // remove potential shift
         }

         if (!found) return true;

         // remove preceeding block and tag itself
         label = label.substr(best + found.name.length);

         subnode1 = subnode = node.append('svg:tspan');

         prevsubpos = subpos;

         subpos = { lvl: curr.lvl + 1, x: curr.x, y: curr.y, fsize: curr.fsize, dx: 0, dy: 0, parent: curr };

         isany = true;

         if (found.arg) {
            pos = label.indexOf("]{");
            if (pos < 0) { console.log('missing argument for ', found.name); return false; }
            foundarg = label.substr(0, pos);
            if (found.arg == 'int') {
               foundarg = parseInt(foundarg);
               if (!Number.isInteger(foundarg)) { console.log('wrong int argument', label.substr(0, pos)); return false; }
            } else if (found.arg == 'float') {
               foundarg = parseFloat(foundarg);
               if (!Number.isFinite(foundarg)) { console.log('wrong float argument', label.substr(0, pos)); return false; }
            }
            label = label.substr(pos + 2);
         }

         let nextdy = curr.dy, nextdx = curr.dx, trav = null,
            scale = 1, left_brace = "{", right_brace = "}"; // this will be applied to the next element

         curr.dy = curr.dx = 0; // relative shift for elements

         if (found.special) {
            subnode.attr('dx', makeem(nextdx)).attr('dy', makeem(nextdy)).text(found.special);
            nextdx = nextdy = 0;
            curr.special = found;

            let rect = get_boundary(subnode);
            if (rect.width && rect.height) {
               found.w = rect.width / curr.fsize;
               found.h = rect.height / curr.fsize - 0.1;
            }
            continue; // just create special node
         }

         if (found.braces) {
            // special handling of large braces
            subpos.left_cont = subnode.append('svg:tspan'); // container for left brace
            subpos.left = subpos.left_cont.append('svg:tspan').text(found.braces[0]);
            subnode1 = subnode.append('svg:tspan');
            subpos.left_rect = { y: curr.y - curr.fsize * 1.1, height: curr.fsize * 1.2, x: curr.x, width: curr.fsize * 0.6 };
            extend_pos(curr, subpos.left_rect);
            subpos.braces = found; // indicate braces handling
            if (found.right) {
               left_brace = found.name;
               right_brace = found.right;
            }
         } else if (found.deco) {
            subpos.deco = found.deco;
         } else if (found.accent) {
            subpos.accent = found.accent;
         } else
            switch (found.name) {
               case "#color[":
                  if (painter.getColor(foundarg))
                     subnode.attr('fill', painter.getColor(foundarg));
                  break;
               case "#kern[": // horizontal shift
                  nextdx += foundarg;
                  break;
               case "#lower[": // after vertical shift one need to compensate it back
                  curr.dy -= foundarg;
                  nextdy += foundarg;
                  break;
               case "scale[":
                  scale = foundarg;
                  break;
               case "#font[":
                  let subfont = new JSROOT.FontHandler(foundarg);
                  subfont.setFont(subnode, 'without-size');
                  break;
               case "#it{":
                  curr.italic = true;
                  trav = curr;
                  while (trav = trav.parent)
                     if (trav.italic !== undefined) {
                        curr.italic = !trav.italic;
                        break;
                     }
                  subnode.attr('font-style', curr.italic ? 'italic' : 'normal');
                  break;
               case "#bf{":
                  curr.bold = true;
                  trav = curr;
                  while (trav = trav.parent)
                     if (trav.bold !== undefined) {
                        curr.bold = !trav.bold;
                        break;
                     }
                  subnode.attr('font-weight', curr.bold ? 'bold' : 'normal');
                  break;
               case "#underline{":
                  subnode.attr('text-decoration', 'underline');
                  break;
               case "#overline{":
                  subnode.attr('text-decoration', 'overline');
                  break;
               case "_{":
                  scale = 0.6;
                  subpos.script = 'sub';

                  if (curr.special) {
                     curr.dx = curr.special.w;
                     curr.dy = -0.7;
                     nextdx -= curr.dx;
                     nextdy -= curr.dy;
                  } else {
                     nextdx += 0.1 * scale;
                     nextdy += 0.4 * scale;
                     subpos.y += 0.4 * subpos.fsize;
                     curr.dy = -0.4 * scale; // compensate vertical shift back

                     if (prevsubpos && (prevsubpos.script === 'super')) {
                        let rect = get_boundary(prevsubpos.node, prevsubpos.rect);
                        subpos.width_limit = rect.width;
                        nextdx -= (rect.width / subpos.fsize + 0.1) * scale;
                     }
                  }
                  break;
               case "^{":
                  scale = 0.6;
                  subpos.script = 'super';

                  if (curr.special) {
                     curr.dx = curr.special.w;
                     curr.dy = curr.special.h;
                     nextdx -= curr.dx;
                     nextdy -= curr.dy;
                  } else {

                     curr.dy = 0.6 * scale; // compensate vertical shift afterwards
                     if (curr.next_super_dy) curr.dy -= curr.next_super_dy;

                     nextdx += 0.1 * scale;
                     nextdy -= curr.dy;

                     subpos.y -= 0.4 * subpos.fsize;

                     if (prevsubpos && (prevsubpos.script === 'sub')) {
                        let rect = get_boundary(prevsubpos.node, prevsubpos.rect);
                        subpos.width_limit = rect.width;
                        nextdx -= (rect.width / subpos.fsize + 0.1) * scale;
                     }
                  }
                  break;
               case "#frac{":
               case "#splitline{":
                  subpos.first = subnode;
                  subpos.two_lines = true;
                  subpos.need_middle = (found.name == "#frac{");
                  subpos.x0 = subpos.x;
                  nextdy -= 0.6;
                  curr.dy = -0.6;
                  break;
               case "#sqrt{":
                  foundarg = 2;
               case "#sqrt[":
                  subpos.square_root = subnode.append('svg:tspan');
                  subpos.square_root.append('svg:tspan').text((foundarg == 3) ? '\u221B' : ((foundarg == 4) ? '\u221C' : '\u221A')); // unicode square, cubic and fourth root
                  subnode1 = subnode.append('svg:tspan');
                  subpos.sqrt_rect = { y: curr.y - curr.fsize * 1.1, height: curr.fsize * 1.2, x: 0, width: curr.fsize * 0.7 };
                  extend_pos(curr, subpos.sqrt_rect); // just dummy symbol instead of square root
                  break;
            }

         if (scale !== 1) {
            // handle centrally change of scale factor
            subnode.attr('font-size', Math.round(scale * 100) + '%');
            subpos.fsize *= scale;
            nextdx = nextdx / scale;
            nextdy = nextdy / scale;
         }

         if (curr.special && !subpos.script) delete curr.special;
         delete curr.next_super_dy;

         subpos.node = subnode; // remember node where sublement is build

         while (true) {
            // loop need to create two lines for #frac or #splitline
            // normally only one sub-element is created

            // moving cursor with the tspan
            subpos.x += nextdx * subpos.fsize;
            subpos.y += nextdy * subpos.fsize;

            subnode.attr('dx', makeem(nextdx)).attr('dy', makeem(nextdy));
            nextdx = nextdy = 0;

            pos = -1; n = 1;

            while ((n != 0) && (++pos < label.length)) {
               if (label.indexOf(left_brace, pos) === pos) n++; else
                  if (label.indexOf(right_brace, pos) === pos) n--;
            }

            if (n != 0) {
               console.log('mismatch with open ' + left_brace + ' and close ' + right_brace + ' braces in Latex', label);
               return false;
            }

            let sublabel = label.substr(0, pos);

            // if (subpos.square_root) sublabel = "#frac{a}{bc}";

            if (!ltx.produceOldLatex(painter, subnode1, arg, sublabel, subpos)) return false;

            // takeover current possition and deltas
            curr.x = subpos.x;
            curr.y = subpos.y;

            curr.dx += subpos.dx * subpos.fsize / curr.fsize;
            curr.dy += subpos.dy * subpos.fsize / curr.fsize;

            label = label.substr(pos + right_brace.length);

            if (subpos.width_limit) {
               // special handling for the case when created element does not reach its minimal width
               // use when super-script and subscript should be combined together

               let rect = get_boundary(subnode1, subpos.rect);
               if (rect.width < subpos.width_limit)
                  curr.dx += (subpos.width_limit - rect.width) / curr.fsize;
               delete subpos.width_limit;
            }

            if (curr.special) {
               // case over #sum or #integral one need to compensate width
               let rect = get_boundary(subnode1, subpos.rect);
               curr.dx -= rect.width / curr.fsize; // compensate width as much as we can
            }

            if (subpos.square_root) {
               // creating cap for square root
               // while overline symbol does not match with square root, use empty text with overline
               let sqrt_dy = 0, yscale = 1,
                   bs = get_boundary(subpos.square_root, subpos.sqrt_rect),
                   be = get_boundary(subnode1, subpos.rect);

               // we can compare y coordinates while both nodes (root and element) on the same level
               if ((be.height > bs.height) && (bs.height > 0)) {
                  yscale = be.height / bs.height * 1.3;
                  sqrt_dy = ((be.y + be.height) - (bs.y + bs.height)) / curr.fsize / yscale;
                  subpos.square_root.style('font-size', Math.round(100 * yscale) + '%').attr('dy', makeem(sqrt_dy));
               }

               // we taking into account only element width
               let len = be.width / subpos.fsize / yscale,
                   a = "", nn = Math.round(Math.max(len * 3, 2));
               while (nn--) a += '\u203E'; // unicode overline

               subpos.square_root.append('svg:tspan').attr("dy", makeem(0)).text(a);

               subpos.square_root.append('svg:tspan').attr("dy", makeem(-sqrt_dy)).attr("dx", makeem(-a.length / 3 - 0.2)).text('\u2009'); // unicode tiny space

               break;
            }

            if (subpos.deco) {

               // use text-decoration attribute when there are no extra elements inside
               if (subnode1.selectAll('tspan').size() == 0) {
                  subnode1.attr('text-decoration', subpos.deco);
                  break;
               }

               let be = get_boundary(subnode1, subpos.rect),
                  len = be.width / subpos.fsize, fact, dy, symb;
               switch (subpos.deco) {
                  case "underline": dy = 0.35; fact = 1.2; symb = '\uFF3F'; break; // '\u2014'; // underline
                  case "overline": dy = -0.35; fact = 3; symb = '\u203E'; break; // overline
                  default: dy = 0; fact = 1.8; symb = '\u23AF'; break;
               }
               let nn = Math.round(Math.max(len * fact, 1)), a = "";
               while (nn--) a += symb;

               subnode1.append('svg:tspan').attr("dx", makeem(-len - 0.2)).attr("dy", makeem(dy)).text(a);
               curr.dy -= dy;
               break;
            }

            if (subpos.braces) {
               // handling braces

               let bs = get_boundary(subpos.left_cont, subpos.left_rect),
                   be = get_boundary(subnode1, subpos.rect),
                   yscale = 1, brace_dy = 0;

               // console.log('braces height', bs.height, ' entry height', be.height);

               if (1.2 * bs.height < be.height) {
                  // make scaling
                  yscale = be.height / bs.height;
                  // brace_dy = ((be.y+be.height) - (bs.y+bs.height))/curr.fsize/yscale - 0.15;
                  brace_dy = 0;
                  subpos.left.style('font-size', Math.round(100 * yscale) + '%').attr('dy', makeem(brace_dy));
                  // unicode tiny space, used to return cursor on vertical position
                  subpos.left_cont.append('svg:tspan').attr("dx", makeem(-0.2))
                     .attr("dy", makeem(-brace_dy * yscale)).text('\u2009');
                  curr.next_super_dy = -0.3 * yscale; // special shift for next comming superscript
               }

               subpos.left_rect.y = curr.y;
               subpos.left_rect.height *= yscale;

               extend_pos(curr, subpos.left_rect); // just dummy symbol instead of right brace for accounting

               let right_cont = subnode.append('svg:tspan')
                  .attr("dx", makeem(curr.dx))
                  .attr("dy", makeem(curr.dy));

               curr.dx = curr.dy = 0;

               if (yscale != 1) right_cont.append('svg:tspan').attr("dx", makeem(-0.2)).text('\u2009'); // unicode tiny space if larger brace is used

               let right = right_cont.append('svg:tspan').text(subpos.braces.braces[1]);

               if (yscale != 1) {
                  right.style('font-size', Math.round(100 * yscale) + '%').attr('dy', makeem(brace_dy));
                  curr.dy = -brace_dy * yscale; // compensation of right brace
               }

               break;
            }

            if (subpos.first && subpos.second) {
               // when two lines created, adjust horizontal position and place divider if required

               let rect1 = get_boundary(subpos.first, subpos.rect1),
                   rect2 = get_boundary(subpos.second, subpos.rect),
                   l1 = rect1.width / subpos.fsize,
                   l2 = rect2.width / subpos.fsize,
                   l3 = Math.max(l2, l1);

               if (subpos.need_middle) {
                  // starting from content len 1.2 two -- will be inserted
                  l3 = Math.round(Math.max(l3, 1) + 0.3);
                  let a = "";
                  while (a.length < l3) a += '\u2014';
                  node.append('svg:tspan')
                     .attr("dx", makeem(-0.5 * (l3 + l2)))
                     .attr("dy", makeem(curr.dy - 0.2))
                     .text(a);
                  curr.dy = 0.2; // return to the normal level
                  curr.dx = 0.2; // extra spacing
               } else {
                  curr.dx = 0.2;
                  if (l2 < l1) curr.dx += 0.5 * (l1 - l2);
               }

               if (subpos.need_middle || arg.align[0] == 'middle') {
                  subpos.first.attr("dx", makeem(0.5 * (l3 - l1)));
                  subpos.second.attr("dx", makeem(-0.5 * (l2 + l1)));
               } else if (arg.align[0] == 'end') {
                  if (l1 < l2) subpos.first.attr("dx", makeem(l2 - l1));
                  subpos.second.attr("dx", makeem(-l2));
               } else {
                  subpos.second.attr("dx", makeem(-l1));
               }

               delete subpos.first;
               delete subpos.second;
            }

            if (!subpos.two_lines) break;

            if (label[0] != '{') {
               console.log('missing { for second line', label);
               return false;
            }

            label = label.substr(1);

            subnode = subnode1 = node.append('svg:tspan');

            subpos.two_lines = false;
            subpos.rect1 = subpos.rect; // remember first rect
            delete subpos.rect;     // reset rectangle calculations
            subpos.x = subpos.x0;   // it is used only for SVG, make it more realistic
            subpos.second = subnode;

            nextdy = curr.dy + 1.6;
            curr.dy = -0.4;
            subpos.dx = subpos.dy = 0; // reset variable
         }

      }

      return true;
   }

   /** @ummary translate TLatex and draw inside provided g element
     * @desc use <text> together with normal <path> elements
     * @private */
   const parseLatex = (node, arg, label, curr) => {

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

         const match = br => (pos + br.length <= label.length) && (label.substr(pos, br.length) == br);

         if (check_first) {
            if(!match(lbrace)) err = true; else label = label.substr(lbrace.length);
         }

         while (!err && (n != 0) && (pos < label.length)) {
            if (match(lbrace)) { n++; pos += lbrace.length; } else
            if (match(rbrace)) { n--; pos += rbrace.length; } else pos++;
         }
         if ((n != 0) || err) {
            console.log(`mismatch with open ${lbrace} and closing ${rbrace} in ${label}`);
            return -1;
         }

         let sublabel = label.substr(0, pos - rbrace.length);

         label = label.substr(pos);

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

            let s = translateLaTeX(label.substr(0, best));
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

               elem.text(s);

               let rect = !JSROOT.nodejs && !JSROOT.settings.ApproxTextSize && !arg.fast ? jsrp.getElementRect(elem, 'nopadding') :
                             { height: curr.fsize * 1.2, width: approximateLabelWidth(s, curr.font, curr.fsize) }

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
         label = label.substr(best + found.name.length);

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
                  label = label.substr(1);
                  res.low = !res.low ? extractSubLabel(true) : -1;
                  if (res.low === -1) {
                     console.log(`error with ${found.name} low limit`);
                     return false;
                  }
               } else if (label.charAt(0) == '^') {
                  label = label.substr(1);
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

            let subs = extractLowUp(found.low_up);
            if (!subs) return false;

            let gg = createGG();

            let path = createPath(gg), h = Math.round(curr.fsize*2), w = Math.round(curr.fsize), r = Math.round(h*0.1), x_up, x_low;

            if (found.name == "#sum") {
               x_up = x_low = w/2;
               path.attr("d",`M${w},${Math.round(-0.75*h)}h${-w}l${Math.round(0.4*w)},${Math.round(0.3*h)}l${Math.round(-0.4*w)},${Math.round(0.7*h)}h${w}`);
            } else {
               x_up = 3*r; x_low = r;
               path.attr("d",`M0,${Math.round(0.25*h-r)}a${r},${r},0,0,0,${2*r},0v${2*r-h}a${r},${r},0,1,1,${2*r},0`);
               // path.attr('transform','skewX(-3)'); could use skewX for italic-like style
            }

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
            foundarg = label.substr(0, pos);
            if (found.arg == 'int') {
               foundarg = parseInt(foundarg);
               if (!Number.isInteger(foundarg)) { console.log('wrong int argument', label.substr(0, pos)); return false; }
            } else if (found.arg == 'float') {
               foundarg = parseFloat(foundarg);
               if (!Number.isFinite(foundarg)) { console.log('wrong float argument', label.substr(0, pos)); return false; }
            }
            label = label.substr(pos + 2);
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
               subpos.font = new JSROOT.FontHandler(foundarg);
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
   ltx.produceLatex = function(painter, node, arg) {

      let pos = { lvl: 0, g: node, x: 0, y: 0, dx: 0, dy: -0.1, fsize: arg.font_size, font: arg.font, parent: null, painter: painter };

      return parseLatex(node, arg, arg.text, pos);
   }


   /** @summary Load MathJax functionality,
     * @desc one need not only to load script but wait for initialization
     * @private */
   ltx.loadMathjax = () => {
      let loading = (ltx._mj_loading !== undefined);

      if (!loading && (typeof MathJax != "undefined"))
         return Promise.resolve(MathJax);

      if (!loading) ltx._mj_loading = [];

      let promise = new Promise(resolve => { ltx._mj_loading ? ltx._mj_loading.push(resolve) : resolve(MathJax); });

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

      if (!JSROOT.nodejs)
         window.MathJax = {
            options: {
               enableMenu: false
            },
            loader: {
               load: ['[tex]/color']
            },
            tex: {
               packages: {'[+]': ['color']}
            },
            svg: svg_config,
            startup: {
               ready: function() {
                  MathJax.startup.defaultReady();
                  let arr = ltx._mj_loading;
                  delete ltx._mj_loading;
                  arr.forEach(func => func(MathJax));
               }
            }
         };

      JSROOT.require('mathjax').then(mj => {

         if (!JSROOT.nodejs) return; // no need for something else, handled with ready

         // return Promise with mathjax loading
         mj.init({
            loader: {
               load: ['input/tex', 'output/svg', '[tex]/color']
             },
             tex: {
                packages: {'[+]': ['color']}
             },
             svg: svg_config,
             config: {
                JSDOM: require('jsdom').JSDOM
             },
             startup: {
                typeset: false,
                ready: function() {
                      MathJax.startup.registerConstructor('jsdomAdaptor', () => {
                         return new MathJax._.adaptors.HTMLAdaptor.HTMLAdaptor(new MathJax.config.config.JSDOM().window);
                      });
                      MathJax.startup.useAdaptor('jsdomAdaptor', true);
                      MathJax.startup.defaultReady();
                      let arr = ltx._mj_loading;
                      delete ltx._mj_loading;
                      arr.forEach(func => func(MathJax));
                }
             }
         });
      });

      return promise;
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

   /** @summary Function translates ROOT TLatex into MathJax format
     * @private */
   const translateMath = (str, kind, color, painter) => {

      if (kind != 2) {
         for (let x in math_symbols_map)
            str = str.replace(new RegExp(x, 'g'), math_symbols_map[x]);

         for (let x in symbols_map)
            if (x.length > 2)
               str = str.replace(new RegExp(x, 'g'), "\\" + x.substr(1));

         // replace all #color[]{} occurances
         let clean = "", first = true;
         while (str) {
            let p = str.indexOf("#color[");
            if ((p < 0) && first) { clean = str; break; }
            first = false;
            if (p != 0) {
               let norm = (p < 0) ? str : str.substr(0, p);
               clean += norm;
               if (p < 0) break;
            }

            str = str.substr(p + 7);
            p = str.indexOf("]{");
            if (p <= 0) break;
            let colindx = parseInt(str.substr(0, p));
            if (!Number.isInteger(colindx)) break;
            let col = painter.getColor(colindx), cnt = 1;
            str = str.substr(p + 2);
            p = -1;
            while (cnt && (++p < str.length)) {
               if (str[p] == '{') cnt++; else if (str[p] == '}') cnt--;
            }
            if (cnt != 0) break;

            let part = str.substr(0, p);
            str = str.substr(p + 1);
            if (part)
               clean += "\\color{" + col + '}{' + part + "}";
         }

         str = clean;
      } else {
         str = str.replace(/\\\^/g, "\\hat");
      }

      if (typeof color != 'string') return str;

      // MathJax SVG converter use colors in normal form
      //if (color.indexOf("rgb(")>=0)
      //   color = color.replace(/rgb/g, "[RGB]")
      //                .replace(/\(/g, '{')
      //                .replace(/\)/g, '}');
      return "\\color{" + color + '}{' + str + "}";
   };

   /** @summary Workaround to fix size attributes in MathJax SVG
     * @private */
   function repairMathJaxSvgSize(painter, mj_node, svg, arg) {
      let transform = value => {
         if (!value || (typeof value !== "string") || (value.length < 3)) return null;
         let p = value.indexOf("ex");
         if ((p < 0) || (p !== value.length - 2)) return null;
         value = parseFloat(value.substr(0, p));
         return Number.isFinite(value) ? value * arg.font.size * 0.5 : null;
      };

      let width = transform(svg.attr("width")),
          height = transform(svg.attr("height")),
          valign = svg.attr("style");

      if (valign && (valign.length > 18) && valign.indexOf("vertical-align:") == 0) {
         let p = valign.indexOf("ex;");
         valign = ((p > 0) && (p == valign.length - 3)) ? transform(valign.substr(16, valign.length - 17)) : null;
      } else {
         valign = null;
      }

      width = (!width || (width <= 0.5)) ? 1 : Math.round(width);
      height = (!height || (height <= 0.5)) ? 1 : Math.round(height);

      svg.attr("width", width).attr('height', height).attr("style", null);

      if (!JSROOT.nodejs) {
         let box = jsrp.getElementRect(mj_node, 'bbox');
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
         let box = jsrp.getElementRect(mj_node, 'bbox'); // sizes before rotation
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

      let trans = "translate(" + arg.x + "," + arg.y + ")";
      if (arg.rotate) trans += " rotate(" + arg.rotate + ")";

      mj_node.attr('transform', trans).attr('visibility', null);
   }

   /** @summary Produce text with MathJax
     * @private */
   ltx.produceMathjax = function(painter, mj_node, arg) {
      let mtext = translateMath(arg.text, arg.latex, arg.color, painter),
          options = { em: arg.font.size, ex: arg.font.size/2, family: arg.font.name, scale: 1, containerWidth: -1, lineWidth: 100000 };

      return ltx.loadMathjax()
             .then(() => MathJax.tex2svgPromise(mtext, options))
             .then(elem => {
                 let svg = d3.select(elem).select("svg");
                 // when adding element to new node, it will be removed from original parent
                 mj_node.append(function() { return svg.node(); });

                 repairMathJaxSvgSize(painter, mj_node, svg, arg);

                 arg.applyAttributesToMathJax = applyAttributesToMathJax;
                 return true;
              });
   }

   /** @summary Just typeset HTML node with MathJax
     * @private */
   ltx.typesetMathjax = function(node) {
      return ltx.loadMathjax()
                .then(() => MathJax.typesetPromise(node ? [node] : undefined));
   }

   if (JSROOT.nodejs) module.exports = ltx;
   return ltx;

})
