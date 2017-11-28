/// @file JSRootPainter.js
/// JavaScript ROOT graphics

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootCore', 'd3'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
      var jsroot = require("./JSRootCore.js");
      factory(jsroot, require("./d3.min.js"));
      if (jsroot.nodejs) jsroot.Painter.readStyleFromURL("?interactive=0&tooltip=0&nomenu&noprogress&notouch&toolbar=0&webgl=0");
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.js');

      if (typeof d3 != 'object')
         throw new Error('d3 is not defined', 'JSRootPainter.js');

      if (typeof JSROOT.Painter == 'object')
         throw new Error('JSROOT.Painter already defined', 'JSRootPainter.js');

      factory(JSROOT, d3);
   }
} (function(JSROOT, d3) {

   "use strict";

   JSROOT.sources.push("2d");

   // do it here while require.js does not provide method to load css files
   if ( typeof define === "function" && define.amd )
      JSROOT.loadScript('$$$style/JSRootPainter.css');

   // list of user painters, called with arguments func(vis, obj, opt)
   JSROOT.DrawFuncs = {lst:[], cache:{}};

   // add draw function for the class
   // List of supported draw options could be provided, separated  with ';'
   // Several different draw functions for the same class or kind could be specified
   JSROOT.addDrawFunc = function(_name, _func, _opt) {
      if ((arguments.length == 1) && (typeof arguments[0] == 'object')) {
         JSROOT.DrawFuncs.lst.push(arguments[0]);
         return arguments[0];
      }
      var handle = { name:_name, func:_func, opt:_opt };
      JSROOT.DrawFuncs.lst.push(handle);
      return handle;
   }

    // icons taken from http://uxrepo.com/

   JSROOT.ToolbarIcons = {
      camera: { path: 'M 152.00,304.00c0.00,57.438, 46.562,104.00, 104.00,104.00s 104.00-46.562, 104.00-104.00s-46.562-104.00-104.00-104.00S 152.00,246.562, 152.00,304.00z M 480.00,128.00L 368.00,128.00 c-8.00-32.00-16.00-64.00-48.00-64.00L 192.00,64.00 c-32.00,0.00-40.00,32.00-48.00,64.00L 32.00,128.00 c-17.60,0.00-32.00,14.40-32.00,32.00l0.00,288.00 c0.00,17.60, 14.40,32.00, 32.00,32.00l 448.00,0.00 c 17.60,0.00, 32.00-14.40, 32.00-32.00L 512.00,160.00 C 512.00,142.40, 497.60,128.00, 480.00,128.00z M 256.00,446.00c-78.425,0.00-142.00-63.574-142.00-142.00c0.00-78.425, 63.575-142.00, 142.00-142.00c 78.426,0.00, 142.00,63.575, 142.00,142.00 C 398.00,382.426, 334.427,446.00, 256.00,446.00z M 480.00,224.00l-64.00,0.00 l0.00-32.00 l 64.00,0.00 L 480.00,224.00 z' },
      disk: { path: 'M384,0H128H32C14.336,0,0,14.336,0,32v448c0,17.656,14.336,32,32,32h448c17.656,0,32-14.344,32-32V96L416,0H384z M352,160   V32h32v128c0,17.664-14.344,32-32,32H160c-17.664,0-32-14.336-32-32V32h128v128H352z M96,288c0-17.656,14.336-32,32-32h256   c17.656,0,32,14.344,32,32v192H96V288z' },
      question: { path: 'M256,512c141.375,0,256-114.625,256-256S397.375,0,256,0S0,114.625,0,256S114.625,512,256,512z M256,64   c63.719,0,128,36.484,128,118.016c0,47.453-23.531,84.516-69.891,110.016C300.672,299.422,288,314.047,288,320   c0,17.656-14.344,32-32,32c-17.664,0-32-14.344-32-32c0-40.609,37.25-71.938,59.266-84.031   C315.625,218.109,320,198.656,320,182.016C320,135.008,279.906,128,256,128c-30.812,0-64,20.227-64,64.672   c0,17.664-14.336,32-32,32s-32-14.336-32-32C128,109.086,193.953,64,256,64z M256,449.406c-18.211,0-32.961-14.75-32.961-32.969   c0-18.188,14.75-32.953,32.961-32.953c18.219,0,32.969,14.766,32.969,32.953C288.969,434.656,274.219,449.406,256,449.406z' },
      undo: { path: 'M450.159,48.042c8.791,9.032,16.983,18.898,24.59,29.604c7.594,10.706,14.146,22.207,19.668,34.489  c5.509,12.296,9.82,25.269,12.92,38.938c3.113,13.669,4.663,27.834,4.663,42.499c0,14.256-1.511,28.863-4.532,43.822  c-3.009,14.952-7.997,30.217-14.953,45.795c-6.955,15.577-16.202,31.52-27.755,47.826s-25.88,32.9-42.942,49.807  c-5.51,5.444-11.787,11.67-18.834,18.651c-7.033,6.98-14.496,14.366-22.39,22.168c-7.88,7.802-15.955,15.825-24.187,24.069  c-8.258,8.231-16.333,16.203-24.252,23.888c-18.3,18.13-37.354,37.016-57.191,56.65l-56.84-57.445  c19.596-19.472,38.54-38.279,56.84-56.41c7.75-7.685,15.772-15.604,24.108-23.757s16.438-16.163,24.33-24.057  c7.894-7.893,15.356-15.33,22.402-22.312c7.034-6.98,13.312-13.193,18.821-18.651c22.351-22.402,39.165-44.648,50.471-66.738  c11.279-22.09,16.932-43.567,16.932-64.446c0-15.785-3.217-31.005-9.638-45.671c-6.422-14.665-16.229-28.504-29.437-41.529  c-3.282-3.282-7.358-6.395-12.217-9.325c-4.871-2.938-10.381-5.503-16.516-7.697c-6.121-2.201-12.815-3.992-20.058-5.373  c-7.242-1.374-14.9-2.064-23.002-2.064c-8.218,0-16.802,0.834-25.788,2.507c-8.961,1.674-18.053,4.429-27.222,8.271  c-9.189,3.842-18.456,8.869-27.808,15.089c-9.358,6.219-18.521,13.819-27.502,22.793l-59.92,60.271l93.797,94.058H0V40.91  l93.27,91.597l60.181-60.532c13.376-15.018,27.222-27.248,41.536-36.697c14.308-9.443,28.608-16.776,42.89-21.992  c14.288-5.223,28.505-8.74,42.623-10.557C294.645,0.905,308.189,0,321.162,0c13.429,0,26.389,1.185,38.84,3.562  c12.478,2.377,24.2,5.718,35.192,10.029c11.006,4.311,21.126,9.404,30.374,15.265C434.79,34.724,442.995,41.119,450.159,48.042z' },
      arrow_right : { path : 'M30.796,226.318h377.533L294.938,339.682c-11.899,11.906-11.899,31.184,0,43.084c11.887,11.899,31.19,11.893,43.077,0  l165.393-165.386c5.725-5.712,8.924-13.453,8.924-21.539c0-8.092-3.213-15.84-8.924-21.551L338.016,8.925  C332.065,2.975,324.278,0,316.478,0c-7.802,0-15.603,2.968-21.539,8.918c-11.899,11.906-11.899,31.184,0,43.084l113.391,113.384  H30.796c-16.822,0-30.463,13.645-30.463,30.463C0.333,212.674,13.974,226.318,30.796,226.318z' },
      arrow_up : { path : 'M295.505,629.446V135.957l148.193,148.206c15.555,15.559,40.753,15.559,56.308,0c15.555-15.538,15.546-40.767,0-56.304  L283.83,11.662C276.372,4.204,266.236,0,255.68,0c-10.568,0-20.705,4.204-28.172,11.662L11.333,227.859  c-7.777,7.777-11.666,17.965-11.666,28.158c0,10.192,3.88,20.385,11.657,28.158c15.563,15.555,40.762,15.555,56.317,0  l148.201-148.219v493.489c0,21.993,17.837,39.82,39.82,39.82C277.669,669.267,295.505,651.439,295.505,629.446z' },
      arrow_diag : { path : 'M279.875,511.994c-1.292,0-2.607-0.102-3.924-0.312c-10.944-1.771-19.333-10.676-20.457-21.71L233.97,278.348  L22.345,256.823c-11.029-1.119-19.928-9.51-21.698-20.461c-1.776-10.944,4.031-21.716,14.145-26.262L477.792,2.149  c9.282-4.163,20.167-2.165,27.355,5.024c7.201,7.189,9.199,18.086,5.024,27.356L302.22,497.527  C298.224,506.426,289.397,511.994,279.875,511.994z M118.277,217.332l140.534,14.294c11.567,1.178,20.718,10.335,21.878,21.896  l14.294,140.519l144.09-320.792L118.277,217.332z' },
      auto_zoom: { path : 'M505.441,242.47l-78.303-78.291c-9.18-9.177-24.048-9.171-33.216,0c-9.169,9.172-9.169,24.045,0.006,33.217l38.193,38.188  H280.088V80.194l38.188,38.199c4.587,4.584,10.596,6.881,16.605,6.881c6.003,0,12.018-2.297,16.605-6.875  c9.174-9.172,9.174-24.039,0.011-33.217L273.219,6.881C268.803,2.471,262.834,0,256.596,0c-6.229,0-12.202,2.471-16.605,6.881  l-78.296,78.302c-9.178,9.172-9.178,24.045,0,33.217c9.177,9.171,24.051,9.171,33.21,0l38.205-38.205v155.4H80.521l38.2-38.188  c9.177-9.171,9.177-24.039,0.005-33.216c-9.171-9.172-24.039-9.178-33.216,0L7.208,242.464c-4.404,4.403-6.881,10.381-6.881,16.611  c0,6.227,2.477,12.207,6.881,16.61l78.302,78.291c4.587,4.581,10.599,6.875,16.605,6.875c6.006,0,12.023-2.294,16.61-6.881  c9.172-9.174,9.172-24.036-0.005-33.211l-38.205-38.199h152.593v152.063l-38.199-38.211c-9.171-9.18-24.039-9.18-33.216-0.022  c-9.178,9.18-9.178,24.059-0.006,33.222l78.284,78.302c4.41,4.404,10.382,6.881,16.611,6.881c6.233,0,12.208-2.477,16.611-6.881  l78.302-78.296c9.181-9.18,9.181-24.048,0-33.205c-9.174-9.174-24.054-9.174-33.21,0l-38.199,38.188v-152.04h152.051l-38.205,38.199  c-9.18,9.175-9.18,24.037-0.005,33.211c4.587,4.587,10.596,6.881,16.604,6.881c6.01,0,12.024-2.294,16.605-6.875l78.303-78.285  c4.403-4.403,6.887-10.378,6.887-16.611C512.328,252.851,509.845,246.873,505.441,242.47z' },
      statbox : {
         path : 'M28.782,56.902H483.88c15.707,0,28.451-12.74,28.451-28.451C512.331,12.741,499.599,0,483.885,0H28.782   C13.074,0,0.331,12.741,0.331,28.451C0.331,44.162,13.074,56.902,28.782,56.902z' +
                'M483.885,136.845H28.782c-15.708,0-28.451,12.741-28.451,28.451c0,15.711,12.744,28.451,28.451,28.451H483.88   c15.707,0,28.451-12.74,28.451-28.451C512.331,149.586,499.599,136.845,483.885,136.845z' +
                'M483.885,273.275H28.782c-15.708,0-28.451,12.731-28.451,28.452c0,15.707,12.744,28.451,28.451,28.451H483.88   c15.707,0,28.451-12.744,28.451-28.451C512.337,286.007,499.599,273.275,483.885,273.275z' +
                'M256.065,409.704H30.492c-15.708,0-28.451,12.731-28.451,28.451c0,15.707,12.744,28.451,28.451,28.451h225.585   c15.707,0,28.451-12.744,28.451-28.451C284.516,422.436,271.785,409.704,256.065,409.704z'
      },
      circle: { path: "M256,256 m-150,0 a150,150 0 1,0 300,0 a150,150 0 1,0 -300,0" },
      three_circles: { path: "M256,85 m-70,0 a70,70 0 1,0 140,0 a70,70 0 1,0 -140,0  M256,255 m-70,0 a70,70 0 1,0 140,0 a70,70 0 1,0 -140,0  M256,425 m-70,0 a70,70 0 1,0 140,0 a70,70 0 1,0 -140,0 " },
      diamand: { path: "M256,0L384,256L256,511L128,256z" },
      rect: { path: "M80,80h352v352h-352z" },

      CreateSVG : function(group,btn,size,title) {
         var svg = group.append("svg:svg")
                     .attr("class", "svg_toolbar_btn")
                     .attr("width",size+"px")
                     .attr("height",size+"px")
                     .attr("viewBox", "0 0 512 512")
                     .style("overflow","hidden");

           if ('recs' in btn) {
              var rec = {};
              for (var n=0;n<btn.recs.length;++n) {
                 JSROOT.extend(rec, btn.recs[n]);
                 svg.append('rect').attr("x", rec.x).attr("y", rec.y)
                     .attr("width", rec.w).attr("height", rec.h)
                     .attr("fill", rec.f);
              }
           } else {
              svg.append('svg:path').attr('d',btn.path);
           }

           //  special rect to correctly get mouse events for whole button area
           svg.append("svg:rect").attr("x",0).attr("y",0).attr("width",512).attr("height",512)
              .style('opacity',0).style('fill',"none").style("pointer-events","visibleFill")
              .append("svg:title").text(title);

           return svg;
      }
   };

   // ==========================================================================================

   var DrawOptions = function(opt) {
      this.opt = opt && (typeof opt=="string") ? opt.toUpperCase().trim() : "";
      this.part = "";
   }

   /// returns true if remaining options are empty
   DrawOptions.prototype.empty = function() {
      return this.opt.length === 0;
   }

   /// returns remaining part of the draw options
   DrawOptions.prototype.remain = function() {
      return this.opt;
   }

   DrawOptions.prototype.check = function(name,postpart) {
      var pos = this.opt.indexOf(name);
      if (pos < 0) return false;
      this.opt = this.opt.substr(0, pos) + this.opt.substr(pos + name.length);
      this.part = "";
      if (!postpart) return true;

      var pos2 = pos;
      while ((pos2<this.opt.length) && (this.opt[pos2] !== ' ') && (this.opt[pos2] !== ',') && (this.opt[pos2] !== ';')) pos2++;
      if (pos2 > pos) {
         this.part = this.opt.substr(pos, pos2-pos);
         this.opt = this.opt.substr(0, pos) + this.opt.substr(pos2);
      }
      return true;
   }

   DrawOptions.prototype.partAsInt = function(offset, dflt) {
      var val = this.part.replace( /^\D+/g, '');
      val = val ? parseInt(val,10) : Number.NaN;
      return isNaN(val) ? (dflt || 0) : val + (offset || 0);
   }

   /** @class JSROOT.Painter Holder of different functions and classes for drawing */
   var Painter = {
         Coord: {
            kCARTESIAN : 1,
            kPOLAR : 2,
            kCYLINDRICAL : 3,
            kSPHERICAL : 4,
            kRAPIDITY : 5
         },
         root_colors: [],
         root_line_styles: ["", "", "3,3", "1,2",
                            "3,4,1,4", "5,3,1,3", "5,3,1,3,1,3,1,3", "5,5",
                            "5,3,1,3,1,3", "20,5", "20,10,1,10", "1,3"],
         root_markers: [  0, 100,   8,   7,   0,  //  0..4
                          9, 100, 100, 100, 100,  //  5..9
                        100, 100, 100, 100, 100,  // 10..14
                        100, 100, 100, 100, 100,  // 15..19
                        100, 103, 105, 104,   0,  // 20..24
                          3,   4,   2,   1, 106,  // 25..29
                          6,   7,   5, 102, 101], // 30..34
          root_fonts: ['Arial', 'iTimes New Roman',
                       'bTimes New Roman', 'biTimes New Roman', 'Arial',
                       'oArial', 'bArial', 'boArial', 'Courier New',
                       'oCourier New', 'bCourier New', 'boCourier New',
                       'Symbol', 'Times New Roman', 'Wingdings', 'iSymbol', 'Verdana'],
          // taken from https://www.math.utah.edu/~beebe/fonts/afm-widths.html
          root_fonts_aver_width: [ 0.537, 0.510,
                                   0.535, 0.520, 0.537,
                                   0.54, 0.556, 0.56, 0.6,
                                   0.6, 0.6, 0.6,
                                   0.587, 0.514, 0.896, 0.587, 0.55 ],
          symbols_map: {
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
                '#acute': '\acute',
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
          },
          math_symbols_map: {
                '#LT':"\\langle",
                '#GT':"\\rangle",
                '#club':"\\clubsuit",
                '#spade':"\\spadesuit",
                '#heart':"\\heartsuit",
                '#diamond':"\\diamondsuit",
                '#voidn':"\\wp",
                '#voidb':"f",
                '#copyright':"(c)",
                '#ocopyright':"(c)",
                '#trademark':"TM",
                '#void3':"TM",
                '#oright':"R",
                '#void1':"R",
                '#3dots':"\\ldots",
                '#lbar':"\\mid",
                '#void8':"\\mid",
                '#divide':"\\div",
                '#Jgothic':"\\Im",
                '#Rgothic':"\\Re",
                '#doublequote':"\"",
                '#plus':"+",
                '#minus':"-",
                '#\/':"/",
                '#upoint':".",
                '#aa':"\\mathring{a}",
                '#AA':"\\mathring{A}",
                '#omicron':"o",
                '#Alpha':"A",
                '#Beta':"B",
                '#Epsilon':"E",
                '#Zeta':"Z",
                '#Eta':"H",
                '#Iota':"I",
                '#Kappa':"K",
                '#Mu':"M",
                '#Nu':"N",
                '#Omicron':"O",
                '#Rho':"P",
                '#Tau':"T",
                '#Chi':"X",
                '#varomega':"\\varpi",
                '#corner':"?",
                '#ltbar':"?",
                '#bottombar':"?",
                '#notsubset':"?",
                '#arcbottom':"?",
                '#cbar':"?",
                '#arctop':"?",
                '#topbar':"?",
                '#arcbar':"?",
                '#downleftarrow':"?",
                '#splitline':"\\genfrac{}{}{0pt}{}",
                '#it':"\\textit",
                '#bf':"\\textbf",
                '#frac':"\\frac",
                '#left{':"\\lbrace",
                '#right}':"\\rbrace",
                '#left\\[':"\\lbrack",
                '#right\\]':"\\rbrack",
                '#\\[\\]{':"\\lbrack",
                ' } ':"\\rbrack",
                '#\\[':"\\lbrack",
                '#\\]':"\\rbrack",
                '#{':"\\lbrace",
                '#}':"\\rbrace",
                ' ':"\\;"
          }
   };

   JSROOT.Painter = Painter; // export here to avoid ambiguity

   Painter.createMenu = function(painter, maincallback) {
      // dummy functions, forward call to the jquery function
      document.body.style.cursor = 'wait';
      JSROOT.AssertPrerequisites('hierarchy;jq2d;', function() {
         document.body.style.cursor = 'auto';
         Painter.createMenu(painter, maincallback);
      });
   }

   Painter.closeMenu = function(menuname) {
      var x = document.getElementById(menuname || 'root_ctx_menu');
      if (x) { x.parentNode.removeChild(x); return true; }
      return false;
   }

   Painter.readStyleFromURL = function(url) {
      var optimize = JSROOT.GetUrlOption("optimize", url);
      if (optimize=="") JSROOT.gStyle.OptimizeDraw = 2; else
      if (optimize!==null) {
         JSROOT.gStyle.OptimizeDraw = parseInt(optimize);
         if (isNaN(JSROOT.gStyle.OptimizeDraw)) JSROOT.gStyle.OptimizeDraw = 2;
      }

      var inter = JSROOT.GetUrlOption("interactive", url);
      if ((inter=="") || (inter=="1")) inter = "11111"; else
      if (inter=="0") inter = "00000";
      if ((inter!==null) && (inter.length==5)) {
         JSROOT.gStyle.Tooltip =     parseInt(inter[0]);
         JSROOT.gStyle.ContextMenu = (inter[1] != '0');
         JSROOT.gStyle.Zooming  =    (inter[2] != '0');
         JSROOT.gStyle.MoveResize =  (inter[3] != '0');
         JSROOT.gStyle.DragAndDrop = (inter[4] != '0');
      }

      var tt = JSROOT.GetUrlOption("tooltip", url);
      if (tt !== null) JSROOT.gStyle.Tooltip = parseInt(tt);

      var mathjax = JSROOT.GetUrlOption("mathjax", url),
          latex = JSROOT.GetUrlOption("latex", url);

      if ((mathjax!==null) && (mathjax!="0") && (latex===null)) latex = "math";
      if (latex!==null) JSROOT.gStyle.Latex = latex; // decoding will be performed with the first text drawing

      if (JSROOT.GetUrlOption("nomenu", url)!=null) JSROOT.gStyle.ContextMenu = false;
      if (JSROOT.GetUrlOption("noprogress", url)!=null) JSROOT.gStyle.ProgressBox = false;
      if (JSROOT.GetUrlOption("notouch", url)!=null) JSROOT.touches = false;
      if (JSROOT.GetUrlOption("adjframe", url)!=null) JSROOT.gStyle.CanAdjustFrame = true;

      JSROOT.gStyle.fOptStat = JSROOT.GetUrlOption("optstat", url, JSROOT.gStyle.fOptStat);
      JSROOT.gStyle.fOptFit = JSROOT.GetUrlOption("optfit", url, JSROOT.gStyle.fOptFit);
      JSROOT.gStyle.fStatFormat = JSROOT.GetUrlOption("statfmt", url, JSROOT.gStyle.fStatFormat);
      JSROOT.gStyle.fFitFormat = JSROOT.GetUrlOption("fitfmt", url, JSROOT.gStyle.fFitFormat);

      var toolbar = JSROOT.GetUrlOption("toolbar", url);
      if (toolbar !== null)
         if (toolbar==='popup') JSROOT.gStyle.ToolBar = 'popup';
                           else JSROOT.gStyle.ToolBar = (toolbar !== "0") && (toolbar !== "false");

      var palette = JSROOT.GetUrlOption("palette", url);
      if (palette!==null) {
         palette = parseInt(palette);
         if (!isNaN(palette) && (palette>0) && (palette<113)) JSROOT.gStyle.Palette = palette;
      }

      var embed3d = JSROOT.GetUrlOption("embed3d", url);
      if (embed3d !== null) JSROOT.gStyle.Embed3DinSVG = parseInt(embed3d);

      var webgl = JSROOT.GetUrlOption("webgl", url);
      if ((webgl === "0") || (webgl === "false")) JSROOT.gStyle.NoWebGL = true; else
      if (webgl === "ie") JSROOT.gStyle.NoWebGL = !JSROOT.browser.isIE;

      var geosegm = JSROOT.GetUrlOption("geosegm", url);
      if (geosegm!==null) JSROOT.gStyle.GeoGradPerSegm = Math.max(2, parseInt(geosegm));
      var geocomp = JSROOT.GetUrlOption("geocomp", url);
      if (geocomp!==null) JSROOT.gStyle.GeoCompressComp = (geocomp!=='0') && (geocomp!=='false');
   }

   /** Function that generates all root colors */
   Painter.createRootColors = function() {
      var colorMap = ['white','black','red','green','blue','yellow','magenta','cyan','rgb(89,212,84)','rgb(89,84,217)', 'white'];
      colorMap[110] = 'white';

      var moreCol = [
        {col:11,str:'c1b7ad4d4d4d6666668080809a9a9ab3b3b3cdcdcde6e6e6f3f3f3cdc8accdc8acc3c0a9bbb6a4b3a697b8a49cae9a8d9c8f83886657b1cfc885c3a48aa9a1839f8daebdc87b8f9a768a926983976e7b857d9ad280809caca6c0d4cf88dfbb88bd9f83c89a7dc08378cf5f61ac8f94a6787b946971d45a549300ff7b00ff6300ff4b00ff3300ff1b00ff0300ff0014ff002cff0044ff005cff0074ff008cff00a4ff00bcff00d4ff00ecff00fffd00ffe500ffcd00ffb500ff9d00ff8500ff6d00ff5500ff3d00ff2600ff0e0aff0022ff003aff0052ff006aff0082ff009aff00b1ff00c9ff00e1ff00f9ff00ffef00ffd700ffbf00ffa700ff8f00ff7700ff6000ff4800ff3000ff1800ff0000'},
        {col:201,str:'5c5c5c7b7b7bb8b8b8d7d7d78a0f0fb81414ec4848f176760f8a0f14b81448ec4876f1760f0f8a1414b84848ec7676f18a8a0fb8b814ecec48f1f1768a0f8ab814b8ec48ecf176f10f8a8a14b8b848ecec76f1f1'},
        {col:390,str:'ffffcdffff9acdcd9affff66cdcd669a9a66ffff33cdcd339a9a33666633ffff00cdcd009a9a00666600333300'},
        {col:406,str:'cdffcd9aff9a9acd9a66ff6666cd66669a6633ff3333cd33339a3333663300ff0000cd00009a00006600003300'},
        {col:422,str:'cdffff9affff9acdcd66ffff66cdcd669a9a33ffff33cdcd339a9a33666600ffff00cdcd009a9a006666003333'},
        {col:590,str:'cdcdff9a9aff9a9acd6666ff6666cd66669a3333ff3333cd33339a3333660000ff0000cd00009a000066000033'},
        {col:606,str:'ffcdffff9affcd9acdff66ffcd66cd9a669aff33ffcd33cd9a339a663366ff00ffcd00cd9a009a660066330033'},
        {col:622,str:'ffcdcdff9a9acd9a9aff6666cd66669a6666ff3333cd33339a3333663333ff0000cd00009a0000660000330000'},
        {col:791,str:'ffcd9acd9a669a66339a6600cd9a33ffcd66ff9a00ffcd33cd9a00ffcd00ff9a33cd66006633009a3300cd6633ff9a66ff6600ff6633cd3300ff33009aff3366cd00336600339a0066cd339aff6666ff0066ff3333cd0033ff00cdff9a9acd66669a33669a009acd33cdff669aff00cdff339acd00cdff009affcd66cd9a339a66009a6633cd9a66ffcd00ff6633ffcd00cd9a00ffcd33ff9a00cd66006633009a3333cd6666ff9a00ff9a33ff6600cd3300ff339acdff669acd33669a00339a3366cd669aff0066ff3366ff0033cd0033ff339aff0066cd00336600669a339acd66cdff009aff33cdff009acd00cdffcd9aff9a66cd66339a66009a9a33cdcd66ff9a00ffcd33ff9a00cdcd00ff9a33ff6600cd33006633009a6633cd9a66ff6600ff6633ff3300cd3300ffff339acd00666600339a0033cd3366ff669aff0066ff3366cd0033ff0033ff9acdcd669a9a33669a0066cd339aff66cdff009acd009aff33cdff009a'},
        {col:920,str:'cdcdcd9a9a9a666666333333'}];

      for (var indx = 0; indx < moreCol.length; ++indx) {
         var entry = moreCol[indx];
         for (var n=0; n<entry.str.length; n+=6) {
            var num = parseInt(entry.col) + parseInt(n/6);
            colorMap[num] = 'rgb(' + parseInt("0x" +entry.str.slice(n,n+2)) + "," + parseInt("0x" + entry.str.slice(n+2,n+4)) + "," + parseInt("0x" + entry.str.slice(n+4,n+6)) + ")";
         }
      }

      Painter.root_colors = colorMap;
   }

   Painter.MakeColorRGB = function(col) {
      if ((col==null) || (col._typename != 'TColor')) return null;
      var rgb = Math.round(col.fRed*255) + "," + Math.round(col.fGreen*255) + "," + Math.round(col.fBlue*255);
      if ((col.fAlpha === undefined) || (col.fAlpha == 1.))
         rgb = "rgb(" + rgb + ")";
      else
         rgb = "rgba(" + rgb + "," + col.fAlpha.toFixed(3) + ")";

      switch (rgb) {
         case 'rgb(255,255,255)': rgb = 'white'; break;
         case 'rgb(0,0,0)': rgb = 'black'; break;
         case 'rgb(255,0,0)': rgb = 'red'; break;
         case 'rgb(0,255,0)': rgb = 'green'; break;
         case 'rgb(0,0,255)': rgb = 'blue'; break;
         case 'rgb(255,255,0)': rgb = 'yellow'; break;
         case 'rgb(255,0,255)': rgb = 'magenta'; break;
         case 'rgb(0,255,255)': rgb = 'cyan'; break;
      }
      return rgb;
   }

   // add new colors from object array
   Painter.extendRootColors = function(jsarr, objarr) {
      if (!objarr || !objarr.arr) return;

      for (var n = 0; n < objarr.arr.length; ++n) {
         var col = objarr.arr[n];
         if (!col || (col._typename != 'TColor')) continue;

         var num = col.fNumber;
         if ((num<0) || (num>10000)) continue;

         var rgb = Painter.MakeColorRGB(col);
         if (rgb && (jsarr[num] != rgb)) jsarr[num] = rgb;
      }
   }

   /// Use TObjArray of TColor instances, typically stored together with TCanvas primitives
   Painter.adoptRootColors = function(objarr) {
      Painter.extendRootColors(Painter.root_colors, objarr);
   }

   function ColorPalette(arr) {
      this.palette = arr;
   }

   /// returns color index which correspond to contour index of provided length
   ColorPalette.prototype.calcColorIndex = function(i,len) {
      var theColor = Math.floor((i+0.99)*this.palette.length/(len-1));
      if (theColor > this.palette.length-1) theColor = this.palette.length-1;
      return theColor;
   }

   /// returns color with provided index
   ColorPalette.prototype.getColor = function(indx) {
      return this.palette[indx];
   }

   /// returns number of colors in the palette
   ColorPalette.prototype.getLength = function() {
      return this.palette.length;
   }

   // calculate color for given i and len
   ColorPalette.prototype.calcColor = function(i,len) {
      var indx = this.calcColorIndex(i,len);
      return this.getColor(indx);
   }

   function TAttMarkerHandler(attmarker, style) {
      var marker_color = Painter.root_colors[attmarker.fMarkerColor];
      if (!style || (style<0)) style = attmarker.fMarkerStyle;

      this.x0 = this.y0 = 0;
      this.color = marker_color;
      this.style = style;
      this.size = 8;
      this.scale = 1;
      this.stroke = true;
      this.fill = true;
      this.marker = "";
      this.ndig = 0;
      this.used = true;
      this.changed = false;

      this.func = this.Apply.bind(this);

      this.Change(marker_color, style, attmarker.fMarkerSize);

      this.changed = false;
   }

   TAttMarkerHandler.prototype.reset_pos = function() {
      this.lastx = this.lasty = null;
   }

   TAttMarkerHandler.prototype.create = function(x,y) {
      return "M" + (x+this.x0).toFixed(this.ndig)+ "," + (y+this.y0).toFixed(this.ndig) + this.marker;
   }

   TAttMarkerHandler.prototype.GetFullSize = function() {
      return this.scale*this.size;
   }

   TAttMarkerHandler.prototype.Change = function(color, style, size) {
      this.changed = true;

      if (color!==undefined) this.color = color;
      if ((style!==undefined) && (style>=0)) this.style = style;
      if (size!==undefined) this.size = size; else size = this.size;

      this.x0 = this.y0 = 0;

      if ((this.style === 1) || (this.style === 777)) {
         this.fill = false;
         this.marker = "h1";
         this.size = 1;

         // use special create function to handle relative position movements
         this.create = function(x,y) {
            var xx = Math.round(x), yy = Math.round(y), m1 = "M"+xx+","+yy+"h1";
            var m2 = (this.lastx===null) ? m1 : ("m"+(xx-this.lastx)+","+(yy-this.lasty)+"h1");
            this.lastx = xx+1; this.lasty = yy;
            return (m2.length < m1.length) ? m2 : m1;
         }

         this.reset_pos();
         return true;
      }

      var marker_kind = Painter.root_markers[this.style];
      if (marker_kind === undefined) marker_kind = 100;
      var shape = marker_kind % 100;

      this.fill = (marker_kind>=100);

      switch(this.style) {
         case 1: this.size = 1; this.scale = 1; break;
         case 6: this.size = 2; this.scale = 1; break;
         case 7: this.size = 3; this.scale = 1; break;
         default: this.size = size; this.scale = 8;
      }

      size = this.GetFullSize();

      this.ndig = (size>7) ? 0 : ((size>2) ? 1 : 2);
      if (shape == 6) this.ndig++;
      var half = (size/2).toFixed(this.ndig), full = size.toFixed(this.ndig);

      switch(shape) {
         case 0: // circle
            this.x0 = -parseFloat(half);
            full = (parseFloat(half)*2).toFixed(this.ndig);
            this.marker = "a"+half+","+half+",0,1,0,"+full+",0a"+half+","+half+",0,1,0,-"+full+",0z";
            break;
         case 1: // cross
            var d = (size/3).toFixed(this.ndig);
            this.x0 = this.y0 = size/6;
            this.marker = "h"+d+"v-"+d+"h-"+d+"v-"+d+"h-"+d+"v"+d+"h-"+d+"v"+d+"h"+d+"v"+d+"h"+d+"z";
            break;
         case 2: // diamond
            this.x0 = -size/2;
            this.marker = "l"+half+",-"+half+"l"+half+","+half+"l-"+half+","+half + "z";
            break;
         case 3: // square
            this.x0 = this.y0 = -size/2;
            this.marker = "v"+full+"h"+full+"v-"+full+"z";
            break;
         case 4: // triangle-up
            this.y0 = size/2;
            this.marker = "l-"+half+",-"+full+"h"+full+"z";
            break;
         case 5: // triangle-down
            this.y0 = -size/2;
            this.marker = "l-"+half+","+full+"h"+full+"z";
            break;
         case 6: // star
            this.y0 = -size/2;
            this.marker = "l"  + (size/3).toFixed(this.ndig) + "," + full +
                          "l-" + (5/6*size).toFixed(this.ndig) + ",-" + (5/8*size).toFixed(this.ndig) +
                          "h"  + full +
                          "l-" + (5/6*size).toFixed(this.ndig) + "," + (5/8*size).toFixed(this.ndig) + "z";
            break;
         case 7: // asterisk
            this.x0 = this.y0 = -size/2;
            this.marker = "l"+full+","+full +
                         "m0,-"+full+"l-"+full+","+full+
                         "m0,-"+half+"h"+full+"m-"+half+",-"+half+"v"+full;
            break;
         case 8: // plus
            this.y0 = -size/2;
            this.marker = "v"+full+"m-"+half+",-"+half+"h"+full;
            break;
         case 9: // mult
            this.x0 = this.y0 = -size/2;
            this.marker = "l"+full+","+full + "m0,-"+full+"l-"+full+","+full;
            break;
         default: // diamand
            this.x0 = -size/2;
            this.marker = "l"+half+",-"+half+"l"+half+","+half+"l-"+half+","+half+"z";
            break;
      }

      return true;
   }

   TAttMarkerHandler.prototype.Apply = function(selection) {
      selection.style('stroke', this.stroke ? this.color : "none");
      selection.style('fill', this.fill ? this.color : "none");
   }

   function TAttLineHandler(attline, borderw, can_excl, direct_line_color) {
      var color = 'black', _width = 0, style = 0;
      if (typeof attline == 'string') {
         color = attline;
         if (color!=='none') _width = 1;
      } else
      if (typeof attline == 'object') {
         if ('fLineColor' in attline) color = direct_line_color || Painter.root_colors[attline.fLineColor];
         if ('fLineWidth' in attline) _width = attline.fLineWidth;
         if ('fLineStyle' in attline) style = attline.fLineStyle;
      } else
      if ((attline !== undefined) && !isNaN(attline)) {
         color = Painter.root_colors[attline];
         if (color) { _width = 1; style = 1; }
      }

      if (borderw!==undefined) _width = borderw;

      this.used = true; // can mark object if it used or not,
      this.color = (_width==0) ? 'none' : color;
      this.width = _width;
      this.style = style;
      if (can_excl) {
         this.excl_side = this.excl_width = 0;
         if (Math.abs(this.width) > 99) {
            // exclusion graph
            this.excl_side = (this.width < 0) ? -1 : 1;
            this.excl_width = Math.floor(this.width / 100) * 5;
            this.width = Math.abs(this.width % 100); // line width
         }
      }

      // if custom color number used, use lightgrey color to show lines
      if ((this.color === undefined) && (this.width > 0))
         this.color = 'lightgrey';

      this.func = this.Apply.bind(this);
   }

   TAttLineHandler.prototype.ChangeExcl = function(side,width) {
      if (width !== undefined) this.excl_width = width;
      if (side !== undefined) {
         this.excl_side = side;
         if ((this.excl_width===0) && (this.excl_side!==0)) this.excl_width = 20;
      }
      this.changed = true;
   }

   TAttLineHandler.prototype.Apply = function(selection) {
      this.used = true;
      if (this.color=='none') {
         selection.style('stroke',null).style('stroke-width',null).style('stroke-dasharray',null);
      } else {
         selection.style('stroke', this.color)
                  .style('stroke-width', this.width)
                  .style('stroke-dasharray', Painter.root_line_styles[this.style] || null);
      }
   }

   TAttLineHandler.prototype.Change = function(color, width, style) {
      if (color !== undefined) this.color = color;
      if (width !== undefined) this.width = width;
      if (style !== undefined) this.style = style;
      this.changed = true;
   }

   function TAttFillHandler(attfill, pattern, color, kind, main_svg) {
      // fill kind can be 1 or 2
      // 1 means object drawing where combination fillcolor==0 and fillstyle==1001 means no filling
      // 2 means all other objects where such combination is white-color filling
      // object painter required when special fill pattern should be registered in central canvas def section

      this.color = "none";
      this.colorindx = 0;
      this.pattern = 0;
      this.used = true;
      this.kind = (kind!==undefined) ? kind : 2;
      this.changed = false;
      this.func = this.Apply.bind(this);

      if (attfill && (typeof attfill == 'object')) {
         if ((pattern===undefined) && ('fFillStyle' in attfill)) pattern = attfill.fFillStyle;
         if ((color==undefined) && ('fFillColor' in attfill)) color = attfill.fFillColor;
      }

      this.Change(color, pattern, main_svg);
      this.changed = false; // unset change property that
   }

   TAttFillHandler.prototype.Apply = function(selection) {
      this.used = true;

      selection.style('fill', this.color);

      if ('opacity' in this)
         selection.style('opacity', this.opacity);

      if ('antialias' in this)
         selection.style('antialias', this.antialias);
   }

   TAttFillHandler.prototype.empty = function() {
      // return true if color not specified or fill style not specified
      return (this.color == 'none');
   }

   TAttFillHandler.prototype.isSolid = function() {
      return this.pattern === 1001;
   }

   // method used when color or pattern were changed with OpenUi5 widgets
   TAttFillHandler.prototype.verifyDirectChange = function(painter) {
      if ((this.color !== 'none') && (this.pattern < 1001)) this.color = 'none';
      var indx = this.colorindx;
      if (this.color !== 'none') {
         indx = JSROOT.Painter.root_colors.indexOf(this.color);
         if (indx<0) {
            indx = JSROOT.Painter.root_colors.length;
            JSROOT.Painter.root_colors.push(this.color);
         }
      }
      this.Change(indx, this.pattern, painter ? painter.svg_canvas() : null);
   }

   TAttFillHandler.prototype.Change = function(color, pattern, svg) {
      this.changed = true;

      if ((color !== undefined) && !isNaN(color))
         this.colorindx = color;

      if ((pattern !== undefined) && !isNaN(pattern)) {
         this.pattern = pattern;
         delete this.opacity;
         delete this.antialias;
      }

      if ((this.pattern == 1000) && (this.colorindx===0)) {
         this.color = 'white';
         return true;
      }

      if (this.pattern < 1001) {
         this.color = 'none';
         return true;
      }

      if (this.isSolid() && (this.colorindx===0) && (this.kind===1)) {
         this.color = 'none';
         return true;
      }

      this.color = JSROOT.Painter.root_colors[this.colorindx];
      if (typeof this.color != 'string') this.color = "none";

      if (this.isSolid()) return true;

      if ((this.pattern >= 4000) && (this.pattern <= 4100)) {
         // special transparent colors (use for subpads)
         this.opacity = (this.pattern - 4000)/100;
         return true;
      }

      if (!svg || svg.empty() || (this.pattern < 3000)) return false;

      var id = "pat_" + this.pattern + "_" + this.colorindx;

      var defs = svg.select('.canvas_defs');
      if (defs.empty())
         defs = svg.insert("svg:defs",":first-child").attr("class","canvas_defs");

      var line_color = this.color;
      this.color = "url(#" + id + ")";
      this.antialias = false;

      if (!defs.select("."+id).empty()) return true;

      var lines = "", lfill = null, fills = "", fills2 = "", w = 2, h = 2;

      switch (this.pattern) {
         case 3001: w = h = 2; fills = "M0,0h1v1h-1zM1,1h1v1h-1z"; break;
         case 3002: w = 4; h = 2; fills = "M1,0h1v1h-1zM3,1h1v1h-1z"; break;
         case 3003: w = h = 4; fills = "M2,1h1v1h-1zM0,3h1v1h-1z"; break;
         case 3004: w = h = 8; lines = "M8,0L0,8"; break;
         case 3005: w = h = 8; lines = "M0,0L8,8"; break;
         case 3006: w = h = 4; lines = "M1,0v4"; break;
         case 3007: w = h = 4; lines = "M0,1h4"; break;
         case 3008:
            w = h = 10;
            fills = "M0,3v-3h3ZM7,0h3v3ZM0,7v3h3ZM7,10h3v-3ZM5,2l3,3l-3,3l-3,-3Z";
            lines = "M0,3l5,5M3,10l5,-5M10,7l-5,-5M7,0l-5,5";
            break;
         case 3009: w = 12; h = 12; lines = "M0,0A6,6,0,0,0,12,0M6,6A6,6,0,0,0,12,12M6,6A6,6,0,0,1,0,12"; lfill = "none"; break;
         case 3010: w = h = 10; lines = "M0,2h10M0,7h10M2,0v2M7,2v5M2,7v3"; break; // bricks
         case 3011: w = 9; h = 18; lines = "M5,0v8M2,1l6,6M8,1l-6,6M9,9v8M6,10l3,3l-3,3M0,9v8M3,10l-3,3l3,3"; lfill = "none"; break;
         case 3012: w = 10; h = 20; lines = "M5,1A4,4,0,0,0,5,9A4,4,0,0,0,5,1M0,11A4,4,0,0,1,0,19M10,11A4,4,0,0,0,10,19"; lfill = "none"; break;
         case 3013: w = h = 7; lines = "M0,0L7,7M7,0L0,7"; lfill = "none"; break;
         case 3014: w = h = 16; lines = "M0,0h16v16h-16v-16M0,12h16M12,0v16M4,0v8M4,4h8M0,8h8M8,4v8"; lfill = "none"; break;
         case 3015: w = 6; h = 12; lines = "M2,1A2,2,0,0,0,2,5A2,2,0,0,0,2,1M0,7A2,2,0,0,1,0,11M6,7A2,2,0,0,0,6,11"; lfill = "none"; break;
         case 3016: w = 12; h = 7; lines = "M0,1A3,2,0,0,1,3,3A3,2,0,0,0,9,3A3,2,0,0,1,12,1"; lfill = "none"; break;
         case 3017: w = h = 4; lines = "M3,1l-2,2"; break;
         case 3018: w = h = 4; lines = "M1,1l2,2"; break;
         case 3019:
            w = h = 12;
            lines = "M1,6A5,5,0,0,0,11,6A5,5,0,0,0,1,6h-1h1A5,5,0,0,1,6,11v1v-1" +
                    "A5,5,0,0,1,11,6h1h-1A5,5,0,0,1,6,1v-1v1A5,5,0,0,1,1,6";
            lfill = "none";
            break;
         case 3020: w = 7; h = 12; lines = "M1,0A2,3,0,0,0,3,3A2,3,0,0,1,3,9A2,3,0,0,0,1,12"; lfill = "none"; break;
         case 3021: w = h = 8; lines = "M8,2h-2v4h-4v2M2,0v2h-2"; lfill = "none"; break; // left stairs
         case 3022: w = h = 8; lines = "M0,2h2v4h4v2M6,0v2h2"; lfill = "none"; break; // right stairs
         case 3023: w = h = 8; fills = "M4,0h4v4zM8,4v4h-4z"; fills2 = "M4,0L0,4L4,8L8,4Z"; break;
         case 3024: w = h = 16; fills = "M0,8v8h2v-8zM8,0v8h2v-8M4,14v2h12v-2z"; fills2 = "M0,2h8v6h4v-6h4v12h-12v-6h-4z"; break;
         case 3025: w = h = 18; fills = "M5,13v-8h8ZM18,0v18h-18l5,-5h8v-8Z"; break;
         default:
            var code = this.pattern % 1000,
                k = code % 10, j = ((code - k) % 100) / 10, i = (code - j*10 - k)/100;
            if (!i) break;

            var sz = i*12;  // axis distance between lines

            w = h = 6*sz; // we use at least 6 steps

            function produce(dy,swap) {
               var pos = [], step = sz, y1 = 0, y2, max = h;

               // reduce step for smaller angles to keep normal distance approx same
               if (Math.abs(dy)<3) step = Math.round(sz/12*9);
               if (dy==0) { step = Math.round(sz/12*8); y1 = step/2; }
               else if (dy>0) max -= step; else y1 = step;

               while(y1<=max) {
                  y2 = y1 + dy*step;
                  if (y2 < 0) {
                     var x2 = Math.round(y1/(y1-y2)*w);
                     pos.push(0,y1,x2,0);
                     pos.push(w,h-y1,w-x2,h);
                  } else if (y2 > h) {
                     var x2 = Math.round((h-y1)/(y2-y1)*w);
                     pos.push(0,y1,x2,h);
                     pos.push(w,h-y1,w-x2,0);
                  } else {
                     pos.push(0,y1,w,y2);
                  }
                  y1+=step;
               }
               for (var k=0;k<pos.length;k+=4)
                  if (swap) lines += "M"+pos[k+1]+","+pos[k]+"L"+pos[k+3]+","+pos[k+2];
                       else lines += "M"+pos[k]+","+pos[k+1]+"L"+pos[k+2]+","+pos[k+3];
            }

            switch (j) {
               case 0: produce(0); break;
               case 1: produce(1); break;
               case 2: produce(2); break;
               case 3: produce(3); break;
               case 4: produce(6); break;
               case 6: produce(3,true); break;
               case 7: produce(2,true); break;
               case 8: produce(1,true); break;
               case 9: produce(0,true); break;
            }

            switch (k) {
               case 0: if (j) produce(0); break;
               case 1: produce(-1); break;
               case 2: produce(-2); break;
               case 3: produce(-3); break;
               case 4: produce(-6); break;
               case 6: produce(-3,true); break;
               case 7: produce(-2,true); break;
               case 8: produce(-1,true); break;
               case 9: if (j!=9) produce(0,true); break;
            }

            break;
      }

      if (!fills && !lines) return false;

      var patt = defs.append('svg:pattern').attr("id",id).attr("class",id).attr("patternUnits","userSpaceOnUse")
                     .attr("width", w).attr("height", h);

      if (fills2) {
         var col = d3.rgb(line_color);
         col.r = Math.round((col.r+255)/2); col.g = Math.round((col.g+255)/2); col.b = Math.round((col.b+255)/2);
         patt.append("svg:path").attr("d", fills2).style("fill", col);
      }
      if (fills) patt.append("svg:path").attr("d", fills).style("fill", line_color);
      if (lines) patt.append("svg:path").attr("d", lines).style('stroke', line_color).style("stroke-width", 1).style("fill", lfill);

      return true;
   }

   /** Function returns the ready to use marker for drawing */
   Painter.createAttMarker = function(attmarker, style) {
      return new TAttMarkerHandler(attmarker, style);
   }

   Painter.createAttLine = function(attline, borderw, can_excl) {
      return new TAttLineHandler(attline, borderw, can_excl);
   }

   Painter.clearCuts = function(chopt) {
      /* decode string "chopt" and remove graphical cuts */
      var left = chopt.indexOf('['),
          right = chopt.indexOf(']');
      if ((left>=0) && (right>=0) && (left<right))
          for (var i = left; i <= right; ++i) chopt[i] = ' ';
      return chopt;
   }

   Painter.getFontDetails = function(fontIndex, size) {

      var res = { name: "Arial", size: Math.round(size || 11), weight: null, style: null },
          indx = Math.floor(fontIndex / 10),
          fontName = Painter.root_fonts[indx] || "";

      while (fontName.length > 0) {
         if (fontName[0]==='b') res.weight = "bold"; else
         if (fontName[0]==='i') res.style = "italic"; else
         if (fontName[0]==='o') res.style = "oblique"; else break;
         fontName = fontName.substr(1);
      }

      if (fontName == 'Symbol')
         res.weight = res.style = null;

      res.name = fontName;
      res.aver_width = Painter.root_fonts_aver_width[indx] || 0.55;

      res.setFont = function(selection, arg) {
         selection.attr("font-family", this.name);
         if (arg != 'without-size')
            selection.attr("font-size", this.size)
                     .attr("xml:space", "preserve");
         if (this.weight)
            selection.attr("font-weight", this.weight);
         if (this.style)
            selection.attr("font-style", this.style);
      }

      res.func = res.setFont.bind(res);

      return res;
   }

   Painter.chooseTimeFormat = function(awidth, ticks) {
      if (awidth < .5) return ticks ? "%S.%L" : "%M:%S.%L";
      if (awidth < 30) return ticks ? "%Mm%S" : "%H:%M:%S";
      awidth /= 60; if (awidth < 30) return ticks ? "%Hh%M" : "%d/%m %H:%M";
      awidth /= 60; if (awidth < 12) return ticks ? "%d-%Hh" : "%d/%m/%y %Hh";
      awidth /= 24; if (awidth < 15.218425) return ticks ? "%d/%m" : "%d/%m/%y";
      awidth /= 30.43685; if (awidth < 6) return "%d/%m/%y";
      awidth /= 12; if (awidth < 2) return ticks ? "%m/%y" : "%d/%m/%y";
      return "%Y";
   }

   Painter.getTimeFormat = function(axis) {
      var idF = axis.fTimeFormat.indexOf('%F');
      if (idF >= 0) return axis.fTimeFormat.substr(0, idF);
      return axis.fTimeFormat;
   }

   Painter.getTimeOffset = function(axis) {
      var idF = axis.fTimeFormat.indexOf('%F');
      if (idF < 0) return JSROOT.gStyle.fTimeOffset*1000;
      var sof = axis.fTimeFormat.substr(idF + 2);
      // default string in axis offset
      if (sof.indexOf('1995-01-01 00:00:00s0')==0) return 788918400000;
      // special case, used from DABC painters
      if ((sof == "0") || (sof == "")) return 0;

      // decode time from ROOT string
      function next(separ, min, max) {
         var pos = sof.indexOf(separ);
         if (pos < 0) { pos = ""; return min; }
         var val = parseInt(sof.substr(0,pos));
         sof = sof.substr(pos+1);
         if (isNaN(val) || (val<min) || (val>max)) { pos = ""; return min; }
         return val;
      }

      var year = next("-", 1970, 2300),
          month = next("-", 1, 12) - 1,
          day = next(" ", 1, 31),
          hour = next(":", 0, 23),
          min = next(":", 0, 59),
          sec = next("s", 0, 59),
          msec = next(" ", 0, 999);

      var dt = new Date(Date.UTC(year, month, day, hour, min, sec, msec));

      var offset = dt.getTime();

      // now also handle suffix like GMT or GMT -0600
      sof = sof.toUpperCase();

      if (sof.indexOf('GMT')==0) {
         offset += dt.getTimezoneOffset()*60000;
         sof = sof.substr(4).trim();
         if (sof.length > 3) {
            var p = 0, sign = 1000;
            if (sof[0]=='-') { p = 1; sign = -1000; }
            offset -= sign * (parseInt(sof.substr(p,2))*3600 + parseInt(sof.substr(p+2,2))*60);
         }
      }

      return offset;
   }

   Painter.translateLaTeX = function(str) {
      while ((str.length>2) && (str[0]=='{') && (str[str.length-1]=='}'))
         str = str.substr(1,str.length-2);

      // str = str.replace(/\^2/gi, '^{2}').replace(/\^3/gi,'^{3}');

      for (var i in this.symbols_map)
         str = str.replace(new RegExp(i,'g'), this.symbols_map[i]);

      return str;
   }

   Painter.approxTextWidth = function(font, label) {
      // returns approximate width of given label, required for reasonable scaling of text in node.js

      return label.length * font.size * font.aver_width;
   }

   Painter.isAnyLatex = function(str) {
      return (str.indexOf("#")>=0) || (str.indexOf("\\")>=0) || (str.indexOf("{")>=0);
   }

   Painter.translateMath = function(str, kind, color, painter) {
      // function translate ROOT TLatex into MathJax format

      if (kind != 2) {
         for (var x in Painter.math_symbols_map)
            str = str.replace(new RegExp(x,'g'), Painter.math_symbols_map[x]);

         for (var x in Painter.symbols_map)
            str = str.replace(new RegExp(x,'g'), "\\" + x.substr(1));

         // replace all #color[]{} occurances
         var clean = "", first = true;
         while (str) {
            var p = str.indexOf("#color[");
            if ((p<0) && first) { clean = str; break; }
            first = false;
            if (p!=0) {
               var norm = (p<0) ? str : str.substr(0, p);
               clean += norm;
               if (p<0) break;
            }

            str = str.substr(p+7);
            p = str.indexOf("]{");
            if (p<=0) break;
            var colindx = parseInt(str.substr(0,p));
            if (isNaN(colindx)) break;
            var col = painter.get_color(colindx);
            str = str.substr(p+2);
            p = str.indexOf("}");
            if (p<0) break;

            var part = str.substr(0,p);
            str = str.substr(p+1);
            if (part)
               clean += "\\color{" + col + '}{' + part + "}";
         }

         str = clean;
      } else {
         str = str.replace(/\\\^/g, "\\hat");
      }

      if (typeof color != 'string') return "\\(" + str + "\\)";

      // MathJax SVG converter use colors in normal form
      //if (color.indexOf("rgb(")>=0)
      //   color = color.replace(/rgb/g, "[RGB]")
      //                .replace(/\(/g, '{')
      //                .replace(/\)/g, '}');
      return "\\(\\color{" + color + '}' + str + "\\)";
   }

   Painter.BuildSvgPath = function(kind, bins, height, ndig) {
      // function used to provide svg:path for the smoothed curves
      // reuse code from d3.js. Used in TH1, TF1 and TGraph painters
      // kind should contain "bezier" or "line".
      // If first symbol "L", than it used to continue drawing

      var smooth = kind.indexOf("bezier") >= 0;

      if (ndig===undefined) ndig = smooth ? 2 : 0;
      if (height===undefined) height = 0;

      function jsroot_d3_svg_lineSlope(p0, p1) {
         return (p1.gry - p0.gry) / (p1.grx - p0.grx);
      }
      function jsroot_d3_svg_lineFiniteDifferences(points) {
         var i = 0, j = points.length - 1, m = [], p0 = points[0], p1 = points[1], d = m[0] = jsroot_d3_svg_lineSlope(p0, p1);
         while (++i < j) {
            m[i] = (d + (d = jsroot_d3_svg_lineSlope(p0 = p1, p1 = points[i + 1]))) / 2;
         }
         m[i] = d;
         return m;
      }
      function jsroot_d3_svg_lineMonotoneTangents(points) {
         var d, a, b, s, m = jsroot_d3_svg_lineFiniteDifferences(points), i = -1, j = points.length - 1;
         while (++i < j) {
            d = jsroot_d3_svg_lineSlope(points[i], points[i + 1]);
            if (Math.abs(d) < 1e-6) {
               m[i] = m[i + 1] = 0;
            } else {
               a = m[i] / d;
               b = m[i + 1] / d;
               s = a * a + b * b;
               if (s > 9) {
                  s = d * 3 / Math.sqrt(s);
                  m[i] = s * a;
                  m[i + 1] = s * b;
               }
            }
         }
         i = -1;
         while (++i <= j) {
            s = (points[Math.min(j, i + 1)].grx - points[Math.max(0, i - 1)].grx) / (6 * (1 + m[i] * m[i]));
            points[i].dgrx = s || 0;
            points[i].dgry = m[i]*s || 0;
         }
      }

      var res = {}, bin = bins[0], prev, maxy = Math.max(bin.gry, height+5),
          currx = Math.round(bin.grx), curry = Math.round(bin.gry), dx, dy;

      res.path = ((kind[0] == "L") ? "L" : "M") +
                  bin.grx.toFixed(ndig) + "," + bin.gry.toFixed(ndig);

      // just calculate all deltas, can be used to build exclusion
      if (smooth || kind.indexOf('calc')>=0)
         jsroot_d3_svg_lineMonotoneTangents(bins);

      if (smooth)
         res.path +=  "c" + bin.dgrx.toFixed(ndig) + "," + bin.dgry.toFixed(ndig) + ",";

      for(var n=1; n<bins.length; ++n) {
          prev = bin;
          bin = bins[n];
          if (smooth) {
             if (n > 1) res.path += "s";
             res.path += (bin.grx-bin.dgrx-prev.grx).toFixed(ndig) + "," + (bin.gry-bin.dgry-prev.gry).toFixed(ndig) + "," + (bin.grx-prev.grx).toFixed(ndig) + "," + (bin.gry-prev.gry).toFixed(ndig);
             maxy = Math.max(maxy, prev.gry);
          } else {
             dx = Math.round(bin.grx - currx);
             dy = Math.round(bin.gry - curry);
             res.path += "l" + dx + "," + dy;
             currx+=dx; curry+=dy;
             maxy = Math.max(maxy, curry);
          }
      }

      if (height>0)
         res.close = "L" + bin.grx.toFixed(ndig) +"," + maxy.toFixed(ndig) +
                     "L" + bins[0].grx.toFixed(ndig) +"," + maxy.toFixed(ndig) + "Z";

      return res;
   }

   // ==============================================================================

   function LongPollSocket(addr) {
      this.path = addr;
      this.connid = null;
      this.req = null;

      this.nextrequest("", "connect");
   }

   LongPollSocket.prototype.nextrequest = function(data, kind) {
      var url = this.path, sync = "";
      if (kind === "connect") {
         url+="?connect";
         this.connid = "connect";
      } else
         if (kind === "close") {
            if ((this.connid===null) || (this.connid==="close")) return;
            url+="?connection="+this.connid + "&close";
            this.connid = "close";
            if (JSROOT.browser.qt5) sync = ";sync"; // use sync mode to close qt5 webengine
         } else
            if ((this.connid===null) || (typeof this.connid!=='number')) {
               return console.error("No connection");
            } else {
               url+="?connection="+this.connid;
               if (kind==="dummy") url+="&dummy";
            }

      if (data) {
         // special workaround to avoid POST request, which is not supported in WebEngine
         var post = "&post=";
         for (var k=0;k<data.length;++k) post+=data.charCodeAt(k).toString(16);
         url += post;
      }

      var req = JSROOT.NewHttpRequest(url, "text" + sync, function(res) {
         if (res===null) res = this.response; // workaround for WebEngine - it does not handle content correctly
         if (this.handle.req === this) {
            this.handle.req = null; // get response for existing dummy request
            if (res == "<<nope>>") res = "";
         }
         this.handle.processreq(res);
      });

      req.handle = this;
      if (kind==="dummy") this.req = req; // remember last dummy request, wait for reply
      req.send();
   }

   LongPollSocket.prototype.processreq = function(res) {
      if (res===null) {
         if (typeof this.onerror === 'function') this.onerror("receive data with connid " + (this.connid || "---"));
         // if (typeof this.onclose === 'function') this.onclose();
         this.connid = null;
         return;
      }

      if (this.connid==="connect") {
         this.connid = parseInt(res);
         console.log('Get new longpoll connection with id ' + this.connid);
         if (typeof this.onopen == 'function') this.onopen();
      } else if (this.connid==="close") {
         if (typeof this.onclose == 'function') this.onclose();
         return;
      } else {
         if ((typeof this.onmessage==='function') && res)
            this.onmessage({ data: res });
      }
      if (!this.req) this.nextrequest("","dummy"); // send new poll request when necessary
   }

   LongPollSocket.prototype.send = function(str) {
      this.nextrequest(str);
   }

   LongPollSocket.prototype.close = function() {
      this.nextrequest("", "close");
   }

   function Cef3QuerySocket(addr) {
      // make very similar to longpoll
      // create persistent CEF requests which could be use from client application at eny time

      if (!window || !('cefQuery' in window)) return null;

      this.path = addr;
      this.connid = null;
      this.nextrequest("","connect");
   }

   Cef3QuerySocket.prototype.nextrequest = function(data, kind) {
      var req = { request: "", persistent: false };
      if (kind === "connect") {
         req.request = "connect";
         req.persistent = true; // this initial request will be used for all messages from the server
         this.connid = "connect";
      } else if (kind === "close") {
         if ((this.connid===null) || (this.connid==="close")) return;
         req.request = this.connid + '::close';
         this.connid = "close";
      } else if ((this.connid===null) || (typeof this.connid!=='number')) {
         return console.error("No connection");
      } else {
         req.request = this.connid + '::post';
         if (data) req.request += "::" + data;
      }

      if (!req.request) return console.error("No CEF request");

      req.request = this.path + "::" + req.request; // URL always preceed any command

      req.onSuccess = this.onSuccess.bind(this);
      req.onFailure = this.onFailure.bind(this);

      this.cefid = window.cefQuery(req); // equvalent to req.send

      return this;
   }

   Cef3QuerySocket.prototype.onFailure = function(error_code, error_message) {
      console.log("CEF_ERR: " + error_code);
      if (typeof this.onerror === 'function') this.onerror("failure with connid " + (this.connid || "---"));
      this.connid = null;
   }

   Cef3QuerySocket.prototype.onSuccess = function(response) {
      if (!response) return; // normal situation when request does not send any reply

      if (this.connid==="connect") {
         this.connid = parseInt(response);
         console.log('Get new CEF connection with id ' + this.connid);
         if (typeof this.onopen == 'function') this.onopen();
      } else if (this.connid==="close") {
         if (typeof this.onclose == 'function') this.onclose();
      } else {
         if ((typeof this.onmessage==='function') && response)
            this.onmessage({ data: response });
      }
   }

   Cef3QuerySocket.prototype.send = function(str) {
      this.nextrequest(str);
   }

   Cef3QuerySocket.prototype.close = function() {
      this.nextrequest("", "close");
      if (this.cefid) window.cefQueryCancel(this.cefid);
      delete this.cefid;
   }

   // ========================================================================================


   // client communication handle for TWebWindow

   function WebWindowHandle(socket_kind) {
      if (socket_kind=='cefquery' && (!window || !('cefQuery' in window))) socket_kind = 'longpoll';

      this.kind = socket_kind;
      this.state = 0;
      this.cansend = 10;
      this.ackn = 10;
   }

   /// Set object which hanldes different socket callbacks like OnWebsocketMsg, OnWebsocketOpened, OnWebsocketClosed
   WebWindowHandle.prototype.SetReceiver = function(obj) {
      this.receiver = obj;
   }

   WebWindowHandle.prototype.Cleanup = function() {
      delete this.receiver;
      this.Close(true);
   }

   WebWindowHandle.prototype.InvokeReceiver = function(method, arg) {
      if (this.receiver && (typeof this.receiver[method] == 'function'))
         this.receiver[method](this, arg);
   }

   WebWindowHandle.prototype.Close = function(force) {
      if (this.timerid) {
         clearTimeout(this.timerid);
         delete this.timerid;
      }

      if (this._websocket && this.state > 0) {
         this.state = force ? -1 : 0; // -1 prevent socket from reopening
         this._websocket.onclose = null; // hide normal handler
         this._websocket.close();
         delete this._websocket;
      }
   }

   WebWindowHandle.prototype.Send = function(msg, chid) {
      if (!this._websocket || (this.state<=0)) return false;

      if (isNaN(chid) || (chid===undefined)) chid = 1; // when not configured, channel 1 is used - main widget

      if (this.cansend <= 0) console.error('should be queued before sending');

      var prefix = this.ackn + ":" + this.cansend + ":" + chid + ":";
      this.ackn = 0;
      this.cansend--; // decrease number of allowed sebd packets

      this._websocket.send(prefix + msg);
      if (this.kind === "websocket") {
         if (this.timerid) clearTimeout(this.timerid);
         this.timerid = setTimeout(this.KeepAlive.bind(this), 10000);
      }
      return true;
   }

   WebWindowHandle.prototype.KeepAlive = function() {
      delete this.timerid;
      this.Send("KEEPALIVE", 0);
   }

   /// method opens relative path with the same kind of socket
   WebWindowHandle.prototype.CreateRelative = function(relative) {
      if (!relative || !this.kind || !this.href) return null;

      var handle = new WebWindowHandle(this.kind);
      console.log('Try to connect ', this.href + relative);
      handle.Connect(this.href + relative);
      return handle;
   }

   WebWindowHandle.prototype.Connect = function(href) {
      // create websocket for current object (canvas)
      // via websocket one recieved many extra information

      this.Close();

      var pthis = this;

      function retry_open(first_time) {

         if (pthis.state != 0) return;

         if (!first_time) console.log("try open websocket again");

         if (pthis._websocket) pthis._websocket.close();
         delete pthis._websocket;

         var path = window.location.href, conn = null;

         if (path && path.lastIndexOf("/")>0) path = path.substr(0, path.lastIndexOf("/")+1);
         if (!href) href = path;
         pthis.href = href;

         console.log('Opening web socket at ' + href);

         if (pthis.kind == 'cefquery') {
            if (href.indexOf("rootscheme://rootserver")==0) href = href.substr(23);
            console.log('configure cefquery ' + href);
            conn = new Cef3QuerySocket(href);
         } else if ((pthis.kind !== 'longpoll') && first_time) {
            href = href.replace("http://", "ws://").replace("https://", "wss://");
            href += "root.websocket";
            console.log('configure websocket ' + href);
            conn = new WebSocket(href);
         } else {
            href += "root.longpoll";
            console.log('configure longpoll ' + href);
            conn = new LongPollSocket(href);
         }

         if (!conn) return;

         pthis._websocket = conn;

         conn.onopen = function() {
            console.log('websocket initialized');
            pthis.state = 1;
            pthis.Send("READY", 0); // need to confirm connection
            pthis.InvokeReceiver('OnWebsocketOpened');
         }

         conn.onmessage = function(e) {
            var msg = e.data;
            if (typeof msg != 'string') return console.log("unsupported message kind: " + (typeof msg));

            var i1 = msg.indexOf(":"),
                credit = parseInt(msg.substr(0,i1)),
                i2 = msg.indexOf(":", i1+1),
                cansend = parseInt(msg.substr(i1+1,i2-i1)),
                i3 = msg.indexOf(":", i2+1),
                chid = parseInt(msg.substr(i2+1,i3-i2));

            // console.log('msg(20)', msg.substr(0,20), credit, cansend, chid, i3);

            pthis.ackn++;            // count number of received packets,
            pthis.cansend += credit; // how many packets client can send

            msg = msg.substr(i3+1);

            if (chid == 0) {
               console.log('GET chid=0 message', msg);
               if (msg == "CLOSE") {
                  pthis.Close(true); // force closing of socket
                  pthis.InvokeReceiver('OnWebsocketClosed');
               }
            } else {
               pthis.InvokeReceiver('OnWebsocketMsg', msg);
            }

            if (pthis.ackn > 7)
               pthis.Send('READY', 0); // send dummy message to server
         }

         conn.onclose = function() {
            delete pthis._websocket;
            if (pthis.state > 0) {
               console.log('websocket closed');
               pthis.state = 0;
               pthis.InvokeReceiver('OnWebsocketClosed');
            }
         }

         conn.onerror = function (err) {
            console.log("err "+err);
            pthis.InvokeReceiver('OnWebsocketError', err);
         }

         setTimeout(retry_open, 3000); // after 3 seconds try again

      } // retry_open

      retry_open(true); // call for the first time
   }

   // ========================================================================================

   function TBasePainter() {
      this.divid = null; // either id of element (preferable) or element itself
   }

   TBasePainter.prototype.AccessTopPainter = function(on) {
      // access painter in the first child element
      // on === true - set this as painter
      // on === false - delete painter
      // on === undefined - return painter
      var main = this.select_main().node(),
         chld = main ? main.firstChild : null;
      if (!chld) return null;
      if (on===true) chld.painter = this; else
      if (on===false) delete chld.painter;
      return chld.painter;
   }

   TBasePainter.prototype.Cleanup = function(keep_origin) {
      // generic method to cleanup painter

      var origin = this.select_main('origin');
      if (!origin.empty() && !keep_origin) origin.html("");
      this.set_layout_kind('simple');
      this.AccessTopPainter(false);
      this.divid = null;

      if (this._hpainter && typeof this._hpainter.ClearPainter === 'function') this._hpainter.ClearPainter(this);

      delete this._hitemname;
      delete this._hdrawopt;
      delete this._hpainter;
   }

   TBasePainter.prototype.DrawingReady = function(res_painter) {
      // function should be called by the painter when first drawing is completed

      this._ready_called_ = true;
      if (this._ready_callback_ !== undefined) {
         var callbacks = this._ready_callback_;
         if (!this._return_res_painter) res_painter = this;

         delete this._return_res_painter;
         delete this._ready_callback_;

         while (callbacks.length)
            JSROOT.CallBack(callbacks.shift(), res_painter);
      }
      return this;
   }

   TBasePainter.prototype.WhenReady = function(callback) {
      // call back will be called when painter ready with the drawing
      if (typeof callback !== 'function') return;
      if ('_ready_called_' in this) return JSROOT.CallBack(callback, this);
      if (this._ready_callback_ === undefined) this._ready_callback_ = [];
      this._ready_callback_.push(callback);
   }

   TBasePainter.prototype.GetObject = function() {
      return null;
   }

   TBasePainter.prototype.MatchObjectType = function(typ) {
      return false;
   }

   TBasePainter.prototype.UpdateObject = function(obj) {
      return false;
   }

   TBasePainter.prototype.RedrawPad = function(resize) {
   }

   TBasePainter.prototype.RedrawObject = function(obj) {
      if (!this.UpdateObject(obj)) return false;
      var current = document.body.style.cursor;
      document.body.style.cursor = 'wait';
      this.RedrawPad();
      document.body.style.cursor = current;
      return true;
   }

   TBasePainter.prototype.CheckResize = function(arg) {
      return false; // indicate if resize is processed
   }

   TBasePainter.prototype.select_main = function(is_direct) {
      // return d3.select for main element for drawing, defined with divid
      // if main element was layouted, returns main element inside layout

      if (!this.divid) return d3.select(null);
      var id = this.divid;
      if ((typeof id == "string") && (id[0]!='#')) id = "#" + id;
      var res = d3.select(id);
      if (res.empty() || (is_direct==='origin')) return res;

      var use_enlarge = res.property('use_enlarge'),
          layout = res.property('layout') || 'simple',
          layout_selector = (layout=='simple') ? "" : res.property('layout_selector');

      if (layout_selector) res = res.select(layout_selector);

      // one could redirect here
      if (!is_direct && !res.empty() && use_enlarge) res = d3.select("#jsroot_enlarge_div");

      return res;
   }

   TBasePainter.prototype.get_layout_kind = function() {
      var origin = this.select_main('origin'),
          layout = origin.empty() ? "" : origin.property('layout');

      return layout || 'simple';
   }

   TBasePainter.prototype.set_layout_kind = function(kind, main_selector) {
      // change layout settings
      var origin = this.select_main('origin');
      if (!origin.empty()) {
         if (!kind) kind = 'simple';
         origin.property('layout', kind);
         origin.property('layout_selector', (kind!='simple') && main_selector ? main_selector : null);
      }
   }

   TBasePainter.prototype.check_main_resize = function(check_level, new_size, height_factor) {
      // function checks if geometry of main div changed
      // returns size of area when main div is drawn
      // take into account enlarge state

      var enlarge = this.enlarge_main('state'),
          main_origin = this.select_main('origin'),
          main = this.select_main(),
          lmt = 5; // minimal size

      if (enlarge !== 'on') {
         if (new_size && new_size.width && new_size.height)
            main_origin.style('width',new_size.width+"px")
                       .style('height',new_size.height+"px");
      }

      var rect_origin = this.get_visible_rect(main_origin, true);

      var can_resize = main_origin.attr('can_resize'),
          do_resize = false;

      if (can_resize == "height")
         if (height_factor && Math.abs(rect_origin.width*height_factor - rect_origin.height) > 0.1*rect_origin.width) do_resize = true;

      if (((rect_origin.height <= lmt) || (rect_origin.width <= lmt)) &&
           can_resize && can_resize !== 'false') do_resize = true;

      if (do_resize && (enlarge !== 'on')) {
          // if zero size and can_resize attribute set, change container size

         if (rect_origin.width > lmt) {
            height_factor = height_factor || 0.66;
            main_origin.style('height', Math.round(rect_origin.width * height_factor)+'px');
         } else
         if (can_resize !== 'height') {
            main_origin.style('width', '200px').style('height', '100px');
         }
      }

      var rect = this.get_visible_rect(main),
          old_h = main.property('draw_height'), old_w = main.property('draw_width');

      rect.changed = false;

      if (old_h && old_w && (old_h>0) && (old_w>0)) {
         if ((old_h !== rect.height) || (old_w !== rect.width))
            if ((check_level>1) || (rect.width/old_w<0.66) || (rect.width/old_w>1.5) ||
                  (rect.height/old_h<0.66) && (rect.height/old_h>1.5)) rect.changed = true;
      } else {
         rect.changed = true;
      }

      return rect;
   }

   TBasePainter.prototype.enlarge_main = function(action) {
      // action can be:  true, false, 'toggle', 'state', 'verify'
      // if action not specified, just return possibility to enlarge main div

      var main = this.select_main(true),
          origin = this.select_main('origin');

      if (main.empty() || !JSROOT.gStyle.CanEnlarge || (origin.property('can_enlarge')===false)) return false;

      if (action===undefined) return true;

      if (action==='verify') return true;

      var state = origin.property('use_enlarge') ? "on" : "off";

      if (action === 'state') return state;

      if (action === 'toggle') action = (state==="off");

      var enlarge = d3.select("#jsroot_enlarge_div");

      if ((action === true) && (state!=="on")) {
         if (!enlarge.empty()) return false;

         enlarge = d3.select(document.body)
                       .append("div")
                       .attr("id","jsroot_enlarge_div");

         var rect1 = this.get_visible_rect(main),
             rect2 = this.get_visible_rect(enlarge);

         // if new enlarge area not big enough, do not do it
         if ((rect2.width <= rect1.width) || (rect2.height <= rect1.height))
            if (rect2.width*rect2.height < rect1.width*rect1.height) {
               console.log('Enlarged area ' + rect2.width+"x"+rect2.height + ' smaller then original drawing ' + rect1.width+"x"+rect1.height);
               enlarge.remove();
               return false;
            }

         while (main.node().childNodes.length > 0)
            enlarge.node().appendChild(main.node().firstChild);

         origin.property('use_enlarge', true);

         return true;
      }
      if ((action === false) && (state!=="off")) {

         while (enlarge.node() && enlarge.node().childNodes.length > 0)
            main.node().appendChild(enlarge.node().firstChild);

         enlarge.remove();
         origin.property('use_enlarge', false);
         return true;
      }

      return false;
   }

   TBasePainter.prototype.GetStyleValue = function(elem, name) {
      if (!elem || elem.empty()) return 0;
      var value = elem.style(name);
      if (!value || (typeof value !== 'string')) return 0;
      value = parseFloat(value.replace("px",""));
      return isNaN(value) ? 0 : Math.round(value);
   }

   TBasePainter.prototype.get_visible_rect = function(elem, fullsize) {
      // return rect with width/height which correspond to the visible area of drawing region

      if (JSROOT.nodejs)
         return { width : parseInt(elem.attr("width")), height: parseInt(elem.attr("height")) };

      var rect = elem.node().getBoundingClientRect(),
          res = { width: Math.round(rect.width), height: Math.round(rect.height) };

      if (!fullsize) {
         // this is size exclude padding area
         res.width -= this.GetStyleValue(elem,'padding-left') + this.GetStyleValue(elem,'padding-right');
         res.height -= this.GetStyleValue(elem,'padding-top') - this.GetStyleValue(elem,'padding-bottom');
      }

      return res;
   }

   TBasePainter.prototype.SetDivId = function(divid) {
      // base painter does not creates canvas or frames
      // it registered in the first child element
      if (arguments.length > 0)
         this.divid = divid;

      this.AccessTopPainter(true);
   }

   TBasePainter.prototype.SetItemName = function(name, opt, hpainter) {
      if (typeof name === 'string') this._hitemname = name;
                               else delete this._hitemname;
      // only upate draw option, never delete. null specified when update drawing
      if (typeof opt === 'string') this._hdrawopt = opt;

      this._hpainter = hpainter;
   }

   TBasePainter.prototype.GetItemName = function() {
      return ('_hitemname' in this) ? this._hitemname : null;
   }

   TBasePainter.prototype.GetItemDrawOpt = function() {
      return ('_hdrawopt' in this) ? this._hdrawopt : "";
   }

   TBasePainter.prototype.CanZoomIn = function(axis,left,right) {
      // check if it makes sense to zoom inside specified axis range
      return false;
   }

   // ==============================================================================

   function TObjectPainter(obj) {
      TBasePainter.call(this);
      this.draw_g = null; // container for all drawn objects
      this.pad_name = ""; // name of pad where object is drawn
      this.main = null;  // main painter, received from pad
      this.draw_object = ((obj!==undefined) && (typeof obj == 'object')) ? obj : null;
   }

   TObjectPainter.prototype = Object.create(TBasePainter.prototype);

   TObjectPainter.prototype.Cleanup = function() {
      // generic method to cleanup painters
      // first of all, remove object drawing and in case of main painter - also main HTML components

      this.RemoveDrawG();

      var keep_origin = true;

      if (this.is_main_painter()) {
         var pp = this.pad_painter(true);
         if (!pp || pp.normal_canvas === false) keep_origin = false;
      }

      // cleanup all existing references
      this.pad_name = "";
      this.main = null;
      this.draw_object = null;

      // remove attributes objects (if any)
      delete this.fillatt;
      delete this.lineatt;
      delete this.markeratt;
      delete this.bins;
      delete this._drawopt;
      delete this.root_colors;

      TBasePainter.prototype.Cleanup.call(this, keep_origin);
   }

   TObjectPainter.prototype.GetObject = function() {
      return this.draw_object;
   }

   TObjectPainter.prototype.GetClassName = function() {
      var res = this.draw_object ? this.draw_object._typename : "";
      return res || "";
   }

   TObjectPainter.prototype.MatchObjectType = function(arg) {
      if (!arg || !this.draw_object) return false;
      if (typeof arg === 'string') return (this.draw_object._typename === arg);
      if (arg._typename) return (this.draw_object._typename === arg._typename);
      return this.draw_object._typename.match(arg);
   }

   TObjectPainter.prototype.SetItemName = function(name, opt, hpainter) {
      TBasePainter.prototype.SetItemName.call(this, name, opt, hpainter);
      if (this.no_default_title || (name=="")) return;
      var can = this.svg_canvas();
      if (!can.empty()) can.select("title").text(name);
                   else this.select_main().attr("title", name);
   }

   TObjectPainter.prototype.UpdateObject = function(obj) {
      // generic method to update object
      // just copy all members from source object
      if (!this.MatchObjectType(obj)) return false;
      JSROOT.extend(this.GetObject(), obj);
      return true;
   }

   TObjectPainter.prototype.GetTipName = function(append) {
      var res = this.GetItemName(), obj = this.GetObject();
      if (res===null) res = "";
      if ((res.length === 0) && obj && obj.fName)
         res = this.GetObject().fName;
      if (res.lenght > 20) res = res.substr(0,17) + "...";
      if ((res.length > 0) && (append!==undefined)) res += append;
      return res;
   }

   TObjectPainter.prototype.pad_painter = function(active_pad) {
      var can = active_pad ? this.svg_pad() : this.svg_canvas();
      return can.empty() ? null : can.property('pad_painter');
   }

   TObjectPainter.prototype.get_color = function(indx) {
      var jsarr = this.root_colors;

      if (!jsarr) {
         var pp = this.pad_painter();
         jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : JSROOT.Painter.root_colors;
      }

      return jsarr[indx];
   }

   TObjectPainter.prototype.add_color = function(color) {
      var jsarr = this.root_colors;
      if (!jsarr) {
         var pp = this.pad_painter();
         jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : JSROOT.Painter.root_colors;
      }
      var indx = jsarr.indexOf(color);
      if (indx >= 0) return indx;
      jsarr.push(color);
      return jsarr.length-1;
   }

   TObjectPainter.prototype.CheckResize = function(arg) {
      // no painter - no resize
      var pad_painter = this.pad_painter();
      if (!pad_painter) return false;

      // only canvas should be checked
      pad_painter.CheckCanvasResize(arg);
      return true;
   }

   TObjectPainter.prototype.RemoveDrawG = function() {
      // generic method to delete all graphical elements, associated with painter
      if (this.draw_g != null) {
         this.draw_g.remove();
         this.draw_g = null;
      }
   }

   TObjectPainter.prototype.RecreateDrawG = function(usepad, layer) {
      // keep old function for a while - later
      console.warn("Obsolete RecreateDrawG is used, will be removed soon. Change to CreateG");
      return this.CreateG(usepad ? undefined : layer);
   }

   /** function (re)creates svg:g element used for specific object drawings
     *  either one attach svg:g to pad list of primitives (default)
     *  or svg:g element created in specified frame layer (default main_layer) */
   TObjectPainter.prototype.CreateG = function(frame_layer) {
      if (this.draw_g) {
         // one should keep svg:g element on its place
         // d3.selectAll(this.draw_g.node().childNodes).remove();
         this.draw_g.selectAll('*').remove();
      } else
      if (frame_layer) {
         var frame = this.svg_frame();
         if (frame.empty()) return frame;
         if (typeof frame_layer != 'string') frame_layer = "main_layer";
         var layer = frame.select("." + frame_layer);
         if (layer.empty()) layer = frame.select(".main_layer");
         this.draw_g = layer.append("svg:g");
      } else {
         var layer = this.svg_layer("primitives_layer");
         this.draw_g = layer.append("svg:g");

         // layer.selectAll(".most_upper_primitives").raise();
         var up = [], chlds = layer.node().childNodes;
         for (var n=0;n<chlds.length;++n)
            if (d3.select(chlds[n]).classed("most_upper_primitives")) up.push(chlds[n]);

         up.forEach(function(top) { d3.select(top).raise(); });
      }

      // set attributes for debugging
      if (this.draw_object) {
         this.draw_g.attr('objname', encodeURI(this.draw_object.fName || "name"));
         this.draw_g.attr('objtype', encodeURI(this.draw_object._typename || "type"));
      }

      return this.draw_g;
   }

   /** This is main graphical SVG element, where all drawings are performed */
   TObjectPainter.prototype.svg_canvas = function() {
      return this.select_main().select(".root_canvas");
   }

   /** This is SVG element, correspondent to current pad */
   TObjectPainter.prototype.svg_pad = function(pad_name) {
      var c = this.svg_canvas();
      if (pad_name === undefined) pad_name = this.pad_name;
      if (pad_name && !c.empty())
         c = c.select(".primitives_layer").select("[pad=" + pad_name + ']');
      return c;
   }

   /** Method selects immediate layer under canvas/pad main element */
   TObjectPainter.prototype.svg_layer = function(name, pad_name) {
      var svg = this.svg_pad(pad_name);
      if (svg.empty()) return svg;

      if (name.indexOf("prim#")==0) {
         svg = svg.select(".primitives_layer");
         name = name.substr(5);
      }

      var node = svg.node().firstChild;
      while (node!==null) {
         var elem = d3.select(node);
         if (elem.classed(name)) return elem;
         node = node.nextSibling;
      }

      return d3.select(null);
   }

   TObjectPainter.prototype.CurrentPadName = function(new_name) {
      var svg = this.svg_canvas();
      if (svg.empty()) return "";
      var curr = svg.property('current_pad');
      if (new_name !== undefined) svg.property('current_pad', new_name);
      return curr;
   }

   TObjectPainter.prototype.root_pad = function() {
      var pad_painter = this.pad_painter(true);
      return pad_painter ? pad_painter.pad : null;
   }

   /** Converts pad x or y coordinate into NDC value */
   TObjectPainter.prototype.ConvertToNDC = function(axis, value, isndc) {
      if (isndc) return value;
      var pad = this.root_pad();
      if (!pad) return value;

      if (axis=="y") {
         if (pad.fLogy)
            value = (value>0) ? JSROOT.log10(value) : pad.fUymin;
         return (value - pad.fY1) / (pad.fY2 - pad.fY1);
      }
      if (pad.fLogx)
         value = (value>0) ? JSROOT.log10(value) : pad.fUxmin;
      return (value - pad.fX1) / (pad.fX2 - pad.fX1);
   }

   /** Converts x or y coordinate into SVG pad coordinates,
    *  which could be used directly for drawing in the pad.
    *  Parameters: axis should be "x" or "y", value to convert.
    *  \par kind can be:
    *  undefined or false - this is coordinate inside frame
    *  true - when NDC coordinates are used
    *  "pad" - when pad coordinates relative to pad ranges are specified
    *  Always return rounded values */
   TObjectPainter.prototype.AxisToSvg = function(axis, value, kind) {
      var main = this.main_painter();
      if (main && !kind) {
         // this is frame coordinates
         value = (axis=="y") ? main.gry(value) + main.frame_y()
                             : main.grx(value) + main.frame_x();
      } else {
         if (kind !== true) value = this.ConvertToNDC(axis, value);
         value = (axis=="y") ? (1-value)*this.pad_height() : value*this.pad_width();
      }
      return Math.round(value);
   }

   /** This is SVG element with current frame */
   TObjectPainter.prototype.svg_frame = function(pad_name) {
      return this.svg_layer("primitives_layer", pad_name).select(".root_frame");
   }

   TObjectPainter.prototype.frame_painter = function() {
      var elem = this.svg_frame();
      var res = elem.empty() ? null : elem.property('frame_painter');
      return res ? res : null;
   }

   TObjectPainter.prototype.pad_width = function(pad_name) {
      var res = this.svg_pad(pad_name);
      res = res.empty() ? 0 : res.property("draw_width");
      return isNaN(res) ? 0 : res;
   }

   TObjectPainter.prototype.pad_height = function(pad_name) {
      var res = this.svg_pad(pad_name);
      res = res.empty() ? 0 : res.property("draw_height");
      return isNaN(res) ? 0 : res;
   }

   TObjectPainter.prototype.frame_property = function(name) {
      var res = this.svg_frame();
      if (res.empty()) return 0;
      res = res.property(name);
      return ((res===undefined) || isNaN(res)) ? 0 : res;
   }

   TObjectPainter.prototype.frame_x = function() {
      return this.frame_property("draw_x");
   }

   TObjectPainter.prototype.frame_y = function() {
      return this.frame_property("draw_y");
   }

   TObjectPainter.prototype.frame_width = function() {
      return this.frame_property("draw_width");
   }

   TObjectPainter.prototype.frame_height = function() {
      return this.frame_property("draw_height");
   }

   TObjectPainter.prototype.embed_3d = function() {
      // returns embed mode for 3D drawings (three.js) inside SVG
      // 0 - no embedding, 3D drawing take full size of canvas
      // 1 - no embedding, canvas placed over svg with proper size (resize problem may appear)
      // 2 - normall embedding via ForeginObject, works only with Firefox
      // 3 - embedding 3D drawing as SVG canvas, requires SVG renderer

      if (JSROOT.BatchMode) return 3;
      if (JSROOT.gStyle.Embed3DinSVG < 2) return JSROOT.gStyle.Embed3DinSVG;
      if (JSROOT.browser.isFirefox /*|| JSROOT.browser.isWebKit*/)
         return JSROOT.gStyle.Embed3DinSVG; // use specified mode
      return 1; // default is overlay
   }

   TObjectPainter.prototype.access_3d_kind = function(new_value) {

      var svg = this.svg_pad(this.this_pad_name);
      if (svg.empty()) return -1;

      // returns kind of currently created 3d canvas
      var kind = svg.property('can3d');
      if (new_value !== undefined) svg.property('can3d', new_value);
      return ((kind===null) || (kind===undefined)) ? -1 : kind;
   }

   TObjectPainter.prototype.size_for_3d = function(can3d) {
      // one uses frame sizes for the 3D drawing - like TH2/TH3 objects

      if (can3d === undefined) can3d = this.embed_3d();

      var pad = this.svg_pad(this.this_pad_name),
          clname = "draw3d_" + (this.this_pad_name || this.pad_name || 'canvas');

      if (pad.empty()) {
         // this is a case when object drawn without canvas

         var rect = this.get_visible_rect(this.select_main());

         if ((rect.height<10) && (rect.width>10)) {
            rect.height = Math.round(0.66*rect.width);
            this.select_main().style('height', rect.height + "px");
         }
         rect.x = 0; rect.y = 0; rect.clname = clname; rect.can3d = -1;
         return rect;
      }

      var elem = pad;
      if (can3d === 0) elem = this.svg_canvas();

      var size = { x: 0, y: 0, width: 100, height: 100, clname: clname, can3d: can3d };

      if (this.frame_painter()!==null) {
         elem = this.svg_frame();
         size.x = elem.property("draw_x");
         size.y = elem.property("draw_y");
      }

      size.width = elem.property("draw_width");
      size.height = elem.property("draw_height");

      if ((this.frame_painter()===null) && (can3d > 0)) {
         size.x = Math.round(size.x + size.width*JSROOT.gStyle.fPadLeftMargin);
         size.y = Math.round(size.y + size.height*JSROOT.gStyle.fPadTopMargin);
         size.width = Math.round(size.width*(1 - JSROOT.gStyle.fPadLeftMargin - JSROOT.gStyle.fPadRightMargin));
         size.height = Math.round(size.height*(1- JSROOT.gStyle.fPadTopMargin - JSROOT.gStyle.fPadBottomMargin));
      }

      var pw = this.pad_width(this.this_pad_name), x2 = pw - size.x - size.width,
          ph = this.pad_height(this.this_pad_name), y2 = ph - size.y - size.height;

      if ((x2 >= 0) && (y2 >= 0)) {
         // while 3D canvas uses area also for the axis labels, extend area relative to normal frame
         size.x = Math.round(size.x * 0.3);
         size.y = Math.round(size.y * 0.9);
         size.width = pw - size.x - Math.round(x2*0.3);
         size.height = ph - size.y - Math.round(y2*0.5);
      }

      if (can3d === 1)
         this.CalcAbsolutePosition(this.svg_pad(this.this_pad_name), size);

      return size;
   }

   TObjectPainter.prototype.clear_3d_canvas = function() {
      var can3d = this.access_3d_kind(null);
      if (can3d < 0) return;

      var size = this.size_for_3d(can3d);

      if (size.can3d === 0) {
         d3.select(this.svg_canvas().node().nextSibling).remove(); // remove html5 canvas
         this.svg_canvas().style('display', null); // show SVG canvas
      } else {
         if (this.svg_pad(this.this_pad_name).empty()) return;

         this.apply_3d_size(size).remove();

         this.svg_frame().style('display', null);  // clear display property
      }
   }

   TObjectPainter.prototype.add_3d_canvas = function(size, canv) {

      if (!canv || (size.can3d < -1)) return;

      if (size.can3d === -1) {
         // case when 3D object drawn without canvas

         var main = this.select_main().node();
         if (main !== null) {
            main.appendChild(canv);
            canv.painter = this;
         }

         return;
      }

      this.access_3d_kind(size.can3d);

      if (size.can3d === 0) {
         this.svg_canvas().style('display', 'none'); // hide SVG canvas

         this.svg_canvas().node().parentNode.appendChild(canv); // add directly
      } else {
         if (this.svg_pad(this.this_pad_name).empty()) return;

         // first hide normal frame
         this.svg_frame().style('display', 'none');

         var elem = this.apply_3d_size(size);

         elem.attr('title','').node().appendChild(canv);
      }
   }

   TObjectPainter.prototype.apply_3d_size = function(size, onlyget) {

      if (size.can3d < 0) return d3.select(null);

      var elem;

      if (size.can3d > 1) {

         elem = this.svg_layer(size.clname);

         // elem = layer.select("." + size.clname);
         if (onlyget) return elem;

         var svg = this.svg_pad();

         if (size.can3d === 3) {
            // this is SVG mode

            if (elem.empty())
               elem = svg.insert("g",".primitives_layer").attr("class", size.clname);

            elem.attr("transform", "translate(" + size.x + "," + size.y + ")");

         } else {

            if (elem.empty())
               elem = svg.insert("foreignObject",".primitives_layer").attr("class", size.clname);

            elem.attr('x', size.x)
                .attr('y', size.y)
                .attr('width', size.width)
                .attr('height', size.height)
                .attr('viewBox', "0 0 " + size.width + " " + size.height)
                .attr('preserveAspectRatio','xMidYMid');
         }

      } else {
         var prnt = this.svg_canvas().node().parentNode;

         elem = d3.select(prnt).select("." + size.clname);
         if (onlyget) return elem;

         // force redraw by resize
         this.svg_canvas().property('redraw_by_resize', true);

         if (elem.empty())
            elem = d3.select(prnt).append('div').attr("class", size.clname + " jsroot_noselect");

         // our position inside canvas, but to set 'absolute' position we should use
         // canvas element offset relative to first parent with non-static position
         // now try to use getBoundingClientRect - it should be more precise

         var pos0 = prnt.getBoundingClientRect();

         while (prnt) {
            if (prnt === document) { prnt = null; break; }
            try {
               if (getComputedStyle(prnt).position !== 'static') break;
            } catch(err) {
               break;
            }
            prnt = prnt.parentNode;
         }

         var pos1 = prnt ? prnt.getBoundingClientRect() : { top: 0, left: 0 };

         var offx = Math.round(pos0.left - pos1.left),
             offy = Math.round(pos0.top - pos1.top);

         elem.style('position','absolute').style('left',(size.x+offx)+'px').style('top',(size.y+offy)+'px').style('width',size.width+'px').style('height',size.height+'px');
      }

      return elem;
   }

   /** Returns main pad painter - normally TH1/TH2 painter, which draws all axis */
   TObjectPainter.prototype.main_painter = function(not_store, pad_name) {
      var res = this.main;
      if (!res) {
         var svg_p = this.svg_pad(pad_name);
         if (svg_p.empty()) {
            res = this.AccessTopPainter();
         } else {
            res = svg_p.property('mainpainter');
         }
         if (!res) res = null;
         if (!not_store) this.main = res;
      }
      return res;
   }

   TObjectPainter.prototype.is_main_painter = function() {
      return this === this.main_painter();
   }

   TObjectPainter.prototype.SetDivId = function(divid, is_main, pad_name) {
      // Assigns id of top element (normally <div></div> where drawing is done
      // is_main - -1 - not add to painters list,
      //            0 - normal painter (default),
      //            1 - major objects like TH1/TH2 (required canvas with frame)
      //            2 - if canvas missing, create it, but not set as main object
      //            3 - if canvas and (or) frame missing, create them, but not set as main object
      //            4 - major objects like TH3 (required canvas, but no frame)
      //            5 - major objects like TGeoVolume (do not require canvas)
      // pad_name - when specified, subpad name used for object drawin
      // In some situations canvas may not exists - for instance object drawn as html, not as svg.
      // In such case the only painter will be assigned to the first element

      if (divid !== undefined)
         this.divid = divid;

      if (!is_main) is_main = 0;

      this.create_canvas = false;

      // SVG element where canvas is drawn
      var svg_c = this.svg_canvas();

      if (svg_c.empty() && (is_main > 0) && (is_main!==5)) {
         JSROOT.Painter.drawCanvas(divid, null, ((is_main == 2) || (is_main == 4)) ? "noframe" : "");
         svg_c = this.svg_canvas();
         this.create_canvas = true;
      }

      if (svg_c.empty()) {
         if ((is_main < 0) || (is_main===5) || this.iscan) return;
         this.AccessTopPainter(true);
         return;
      }

      // SVG element where current pad is drawn (can be canvas itself)
      this.pad_name = pad_name;
      if (this.pad_name === undefined)
         this.pad_name = this.CurrentPadName();

      if (is_main < 0) return;

      // create TFrame element if not exists
      if (this.svg_frame().select(".main_layer").empty() && ((is_main == 1) || (is_main == 3))) {
         JSROOT.Painter.drawFrame(divid, null);
         if (this.svg_frame().empty()) return alert("Fail to draw dummy TFrame");
      }

      var svg_p = this.svg_pad();
      if (svg_p.empty()) return;

      if (svg_p.property('pad_painter') !== this)
         svg_p.property('pad_painter').painters.push(this);

      if (((is_main === 1) || (is_main === 4) || (is_main === 5)) && !svg_p.property('mainpainter'))
         // when this is first main painter in the pad
         svg_p.property('mainpainter', this);
   }

   TObjectPainter.prototype.CalcAbsolutePosition = function(sel, pos) {
      while (!sel.empty() && !sel.classed('root_canvas')) {
         if (sel.classed('root_frame') || sel.classed('root_pad')) {
           pos.x += sel.property("draw_x");
           pos.y += sel.property("draw_y");
         }
         sel = d3.select(sel.node().parentNode);
      }
      return pos;
   }

   TObjectPainter.prototype.createAttFill = function(attfill, pattern, color, kind) {
      return new TAttFillHandler(attfill, pattern, color, kind, this.svg_canvas());
   }

   TBasePainter.prototype.AttributeChange = function(class_name, member_name, new_value) {
      // function called when user interactively changes attribute in given class

      // console.log("Changed attribute", class_name, member_name, new_value);
   }

   TObjectPainter.prototype.ForEachPainter = function(userfunc) {
      // Iterate over all known painters

      // special case of the painter set as pointer of first child of main element
      var painter = this.AccessTopPainter();
      if (painter) return userfunc(painter);

      // iterate over all painters from pad list
      var pad_painter = this.pad_painter(true);
      if (pad_painter)
         pad_painter.ForEachPainterInPad(userfunc);
   }

   TObjectPainter.prototype.RedrawPad = function() {
      // call Redraw methods for each painter in the frame
      // if selobj specified, painter with selected object will be redrawn
      var pad_painter = this.pad_painter(true);
      if (pad_painter) pad_painter.Redraw();
   }

   TObjectPainter.prototype.SwitchTooltip = function(on) {
      var fp = this.frame_painter();
      if (fp) fp.ProcessTooltipEvent(null, on);
      // this is 3D control object
      if (this.control && (typeof this.control.SwitchTooltip == 'function'))
         this.control.SwitchTooltip(on);
   }

   TObjectPainter.prototype.AddDrag = function(callback) {
      if (!JSROOT.gStyle.MoveResize) return;

      var pthis = this, drag_rect = null;

      function detectRightButton(event) {
         if ('buttons' in event) return event.buttons === 2;
         else if ('which' in event) return event.which === 3;
         else if ('button' in event) return event.button === 2;
         return false;
      }

      function rect_width() { return Number(pthis.draw_g.attr("width")); }
      function rect_height() { return Number(pthis.draw_g.attr("height")); }

      function MakeResizeElements(group, width, height, handler) {
         function make(cursor,d) {
            var clname = "js_" + cursor.replace('-','_'),
                elem = group.select('.'+clname);
            if (elem.empty()) elem = group.append('path').classed(clname,true);
            elem.style('opacity', 0).style('cursor', cursor).attr('d',d);
            if (handler) elem.call(handler);
         }

         make("nw-resize", "M2,2h15v-5h-20v20h5Z");
         make("ne-resize", "M" + (width-2) + ",2h-15v-5h20v20h-5 Z");
         make("sw-resize", "M2," + (height-2) + "h15v5h-20v-20h5Z");
         make("se-resize", "M" + (width-2) + "," + (height-2) + "h-15v5h20v-20h-5Z");

         make("w-resize", "M-3,18h5v" + Math.max(0, height - 2*18) + "h-5Z");
         make("e-resize", "M" + (width+3) + ",18h-5v" + Math.max(0, height - 2*18) + "h5Z");
         make("n-resize", "M18,-3v5h" + Math.max(0, width - 2*18) + "v-5Z");
         make("s-resize", "M18," + (height+3) + "v-5h" + Math.max(0, width - 2*18) + "v5Z");
      }

      function complete_drag() {
         drag_rect.style("cursor", "auto");

         if (!pthis.draw_g) {
            drag_rect.remove();
            drag_rect = null;
            return false;
         }

         var oldx = Number(pthis.draw_g.attr("x")),
             oldy = Number(pthis.draw_g.attr("y")),
             newx = Number(drag_rect.attr("x")),
             newy = Number(drag_rect.attr("y")),
             newwidth = Number(drag_rect.attr("width")),
             newheight = Number(drag_rect.attr("height"));

         if (callback.minwidth && newwidth < callback.minwidth) newwidth = callback.minwidth;
         if (callback.minheight && newheight < callback.minheight) newheight = callback.minheight;

         var change_size = (newwidth !== rect_width()) || (newheight !== rect_height()),
             change_pos = (newx !== oldx) || (newy !== oldy);

         pthis.draw_g.attr('x', newx).attr('y', newy)
                     .attr("transform", "translate(" + newx + "," + newy + ")")
                     .attr('width', newwidth).attr('height', newheight);

         drag_rect.remove();
         drag_rect = null;

         pthis.SwitchTooltip(true);

         MakeResizeElements(pthis.draw_g, newwidth, newheight);

         if (change_size || change_pos) {
            if (change_size && ('resize' in callback)) callback.resize(newwidth, newheight);
            if (change_pos && ('move' in callback)) callback.move(newx, newy, newx - oldxx, newy-oldy);

            if (change_size || change_pos) {
               if ('obj' in callback) {
                  callback.obj.fX1NDC = newx / pthis.pad_width();
                  callback.obj.fX2NDC = (newx + newwidth)  / pthis.pad_width();
                  callback.obj.fY1NDC = 1 - (newy + newheight) / pthis.pad_height();
                  callback.obj.fY2NDC = 1 - newy / pthis.pad_height();
                  callback.obj.modified_NDC = true; // indicate that NDC was interactively changed, block in updated
               }
               if ('redraw' in callback) callback.redraw();
            }
         }

         return change_size || change_pos;
      }

      var prefix = "", drag_move, drag_resize;
      if (JSROOT._test_d3_ === 3) {
         prefix = "drag";
         drag_move = d3.behavior.drag().origin(Object);
         drag_resize = d3.behavior.drag().origin(Object);
      } else {
         drag_move = d3.drag().subject(Object);
         drag_resize = d3.drag().subject(Object);
      }

      drag_move
         .on(prefix+"start",  function() {
            if (detectRightButton(d3.event.sourceEvent)) return;

            JSROOT.Painter.closeMenu(); // close menu

            pthis.SwitchTooltip(false); // disable tooltip

            d3.event.sourceEvent.preventDefault();
            d3.event.sourceEvent.stopPropagation();

            var handle = {
               acc_x1: Number(pthis.draw_g.attr("x")),
               acc_y1: Number(pthis.draw_g.attr("y")),
               pad_w: pthis.pad_width() - rect_width(),
               pad_h: pthis.pad_height() - rect_height(),
               drag_tm: new Date()
            };

            drag_rect = d3.select(pthis.draw_g.node().parentNode).append("rect")
                 .classed("zoom", true)
                 .attr("x", handle.acc_x1)
                 .attr("y", handle.acc_y1)
                 .attr("width", rect_width())
                 .attr("height", rect_height())
                 .style("cursor", "move")
                 .style("pointer-events","none") // let forward double click to underlying elements
                 .property('drag_handle', handle);


          }).on("drag", function() {
               if (!drag_rect) return;

               d3.event.sourceEvent.preventDefault();
               d3.event.sourceEvent.stopPropagation();

               var handle = drag_rect.property('drag_handle');

               handle.acc_x1 += d3.event.dx;
               handle.acc_y1 += d3.event.dy;

               drag_rect.attr("x", Math.min( Math.max(handle.acc_x1, 0), handle.pad_w))
                        .attr("y", Math.min( Math.max(handle.acc_y1, 0), handle.pad_h));

          }).on(prefix+"end", function() {
               if (!drag_rect) return;

               d3.event.sourceEvent.preventDefault();

               var handle = drag_rect.property('drag_handle');

               if (complete_drag() === false) {
                  var spent = (new Date()).getTime() - handle.drag_tm.getTime();
                  if (callback.ctxmenu && (spent > 600)) {
                     var rrr = resize_se.node().getBoundingClientRect();
                     pthis.ShowContextMenu('main', { clientX: rrr.left, clientY: rrr.top } );
                  } else if (callback.canselect && (spent <= 600)) {
                     pthis.pad_painter().SelectObjectPainter(pthis);
                  }

               }
            });

      drag_resize
        .on(prefix+"start", function() {
           if (detectRightButton(d3.event.sourceEvent)) return;

           d3.event.sourceEvent.stopPropagation();
           d3.event.sourceEvent.preventDefault();

           pthis.SwitchTooltip(false); // disable tooltip

           var handle = {
              acc_x1: Number(pthis.draw_g.attr("x")),
              acc_y1: Number(pthis.draw_g.attr("y")),
              pad_w:  pthis.pad_width(),
              pad_h:  pthis.pad_height()
           };

           handle.acc_x2 = handle.acc_x1 + rect_width();
           handle.acc_y2 = handle.acc_y1 + rect_height();

           drag_rect = d3.select(pthis.draw_g.node().parentNode)
                         .append("rect")
                         .classed("zoom", true)
                         .style("cursor", d3.select(this).style("cursor"))
                         .attr("x", handle.acc_x1)
                         .attr("y", handle.acc_y1)
                         .attr("width", handle.acc_x2 - handle.acc_x1)
                         .attr("height", handle.acc_y2 - handle.acc_y1)
                         .property('drag_handle', handle);

         }).on("drag", function() {
            if (!drag_rect) return;

            d3.event.sourceEvent.preventDefault();
            d3.event.sourceEvent.stopPropagation();

            var handle = drag_rect.property('drag_handle'),
                dx = d3.event.dx, dy = d3.event.dy, elem = d3.select(this);

            if (elem.classed('js_nw_resize')) { handle.acc_x1 += dx; handle.acc_y1 += dy; }
            else if (elem.classed('js_ne_resize')) { handle.acc_x2 += dx; handle.acc_y1 += dy; }
            else if (elem.classed('js_sw_resize')) { handle.acc_x1 += dx; handle.acc_y2 += dy; }
            else if (elem.classed('js_se_resize')) { handle.acc_x2 += dx; handle.acc_y2 += dy; }
            else if (elem.classed('js_w_resize')) { handle.acc_x1 += dx; }
            else if (elem.classed('js_n_resize')) { handle.acc_y1 += dy; }
            else if (elem.classed('js_e_resize')) { handle.acc_x2 += dx; }
            else if (elem.classed('js_s_resize')) { handle.acc_y2 += dy; }

            var x1 = Math.max(0, handle.acc_x1), x2 = Math.min(handle.acc_x2, handle.pad_w),
                y1 = Math.max(0, handle.acc_y1), y2 = Math.min(handle.acc_y2, handle.pad_h);

            drag_rect.attr("x", x1).attr("y", y1).attr("width", Math.max(0, x2-x1)).attr("height", Math.max(0, y2-y1));

         }).on(prefix+"end", function() {
            if (!drag_rect) return;

            d3.event.sourceEvent.preventDefault();

            complete_drag();
         });

      if (!callback.only_resize)
         this.draw_g.style("cursor", "move").call(drag_move);

      MakeResizeElements(this.draw_g, rect_width(), rect_height(), drag_resize);
   }

   TObjectPainter.prototype.startTouchMenu = function(kind) {
      // method to let activate context menu via touch handler

      var arr = d3.touches(this.svg_frame().node());
      if (arr.length != 1) return;

      if (!kind || (kind=="")) kind = "main";
      var fld = "touch_" + kind;

      d3.event.preventDefault();
      d3.event.stopPropagation();

      this[fld] = { dt: new Date(), pos: arr[0] };

      this.svg_frame().on("touchcancel", this.endTouchMenu.bind(this, kind))
                      .on("touchend", this.endTouchMenu.bind(this, kind));
   }

   TObjectPainter.prototype.endTouchMenu = function(kind) {
      var fld = "touch_" + kind;

      if (! (fld in this)) return;

      d3.event.preventDefault();
      d3.event.stopPropagation();

      var diff = new Date().getTime() - this[fld].dt.getTime();

      this.svg_frame().on("touchcancel", null)
                      .on("touchend", null);

      if (diff>500) {
         var rect = this.svg_frame().node().getBoundingClientRect();
         this.ShowContextMenu(kind, { clientX: rect.left + this[fld].pos[0],
                                      clientY: rect.top + this[fld].pos[1] } );
      }

      delete this[fld];
   }

   TObjectPainter.prototype.AddColorMenuEntry = function(menu, name, value, set_func, fill_kind) {
      if (value === undefined) return;
      menu.add("sub:"+name, function() {
         // todo - use jqury dialog here
         var useid = (typeof value !== 'string');
         var col = prompt("Enter color " + (useid ? "(only id number)" : "(name or id)"), value);
         if (col == null) return;
         var id = parseInt(col);
         if (!isNaN(id) && (JSROOT.Painter.root_colors[id] !== undefined)) {
            col = JSROOT.Painter.root_colors[id];
         } else {
            if (useid) return;
         }
         set_func.bind(this)(useid ? id : col);
      });
      var useid = (typeof value !== 'string');
      for (var n=-1;n<11;++n) {
         if ((n<0) && useid) continue;
         if ((n==10) && (fill_kind!==1)) continue;
         var col = (n<0) ? 'none' : JSROOT.Painter.root_colors[n];
         if ((n==0) && (fill_kind==1)) col = 'none';
         var svg = "<svg width='100' height='18' style='margin:0px;background-color:" + col + "'><text x='4' y='12' style='font-size:12px' fill='" + (n==1 ? "white" : "black") + "'>"+col+"</text></svg>";
         menu.addchk((value == (useid ? n : col)), svg, (useid ? n : col), set_func);
      }
      menu.add("endsub:");
   }

   TObjectPainter.prototype.AddSizeMenuEntry = function(menu, name, min, max, step, value, set_func) {
      if (value === undefined) return;

      menu.add("sub:"+name, function() {
         // todo - use jqury dialog here
         var entry = value.toFixed(4);
         if (step>=0.1) entry = value.toFixed(2);
         if (step>=1) entry = value.toFixed(0);
         var val = prompt("Enter value of " + name, entry);
         if (val==null) return;
         var val = parseFloat(val);
         if (!isNaN(val)) set_func.bind(this)((step>=1) ? Math.round(val) : val);
      });
      for (var val=min;val<=max;val+=step) {
         var entry = val.toFixed(2);
         if (step>=0.1) entry = val.toFixed(1);
         if (step>=1) entry = val.toFixed(0);
         menu.addchk((Math.abs(value - val) < step/2), entry, val, set_func);
      }
      menu.add("endsub:");
   }

   TObjectPainter.prototype.ExecuteMenuCommand = function(method) {
      // execute selected menu command, either locally or remotely

      if (method.fName == "Inspect") {
         this.ShowInpsector();
         return true;
      }

      var canvp = this.pad_painter();
      if (!canvp) return false;

      if ((method.fName == "FitPanel") && canvp.ActivateFitPanel) {
         canvp.ActivateFitPanel(this);
         return true;
      }

      if (canvp.ActivateGed && ((method.fName == "DrawPanel") || (method.fName == "SetLineAttributes")
            || (method.fName == "SetFillAttributes") || (method.fName == "SetMarkerAttributes"))) {
         canvp.ActivateGed(this); // activate GED
         return true;
      }

      return false;
   }

   TObjectPainter.prototype.FillObjectExecMenu = function(menu, kind, call_back) {

      var canvp = this.pad_painter();

      if (!this.snapid || !canvp || !canvp._websocket || canvp._getmenu_callback)
         return JSROOT.CallBack(call_back);

      function DoExecMenu(arg) {
         var canvp = this.pad_painter(),
             item = this.args_menu_items[parseInt(arg)];

         if (!item || !item.fName) return;

         if (canvp.MethodsDialog && (item.fArgs!==undefined))
            return canvp.MethodsDialog(this, item, this.args_menu_id);

         if (this.ExecuteMenuCommand(item)) return;

         if (canvp._websocket && this.args_menu_id) {
            console.log('execute method ' + item.fExec + ' for object ' + this.args_menu_id);
            canvp.SendWebsocket('OBJEXEC:' + this.args_menu_id + ":" + item.fExec);
         }
      }

      function DoFillMenu(_menu, _reqid, _call_back, items, replyid) {

         // avoid multiple call of the callback after timeout
         if (!canvp._getmenu_callback) return;
         delete canvp._getmenu_callback;

         if (_reqid !== replyid)
            console.error('missmatch between request ' + _reqid + ' and reply ' + replyid + ' identifiers');

         if (items && items.length) {
            _menu.add("separator");
            _menu.add("sub:Online");

            this.args_menu_items = items;
            this.args_menu_id = replyid;

            var lastclname;

            for (var n=0;n<items.length;++n) {
               var item = items[n];

               if (item.fClassName && lastclname && (lastclname!=item.fClassName)) _menu.add("separator");
               lastclname = item.fClassName;

               if ((item.fChecked === undefined) || (item.fChecked < 0))
                  _menu.add(item.fName, n, DoExecMenu);
               else
                  _menu.addchk(item.fChecked, item.fName, n, DoExecMenu);
            }

            _menu.add("endsub:");
         }

         JSROOT.CallBack(_call_back);
      }

      var reqid = this.snapid;
      if (kind) reqid += "#" + kind; // use # to separate object id from member specifier like 'x' or 'z'

      canvp._getmenu_callback = DoFillMenu.bind(this, menu, reqid, call_back);

      canvp.SendWebsocket('GETMENU:' + reqid); // request menu items for given painter

      setTimeout(canvp._getmenu_callback, 2000); // set timeout to avoid menu hanging
   }

   TObjectPainter.prototype.DeleteAtt = function() {
      // remove all created draw attributes
      delete this.lineatt;
      delete this.fillatt;
      delete this.markeratt;
   }

   TObjectPainter.prototype.FillAttContextMenu = function(menu, preffix) {
      // this method used to fill entries for different attributes of the object
      // like TAttFill, TAttLine, ....
      // all menu call-backs need to be rebind, while menu can be used from other painter

      if (!preffix) preffix = "";

      if (this.lineatt && this.lineatt.used) {
         menu.add("sub:"+preffix+"Line att");
         this.AddSizeMenuEntry(menu, "width", 1, 10, 1, this.lineatt.width,
                               function(arg) { this.lineatt.Change(undefined, parseInt(arg)); this.Redraw(); }.bind(this));
         this.AddColorMenuEntry(menu, "color", this.lineatt.color,
                          function(arg) { this.lineatt.Change(arg); this.Redraw(); }.bind(this));
         menu.add("sub:style", function() {
            var id = prompt("Enter line style id (1-solid)", 1);
            if (id == null) return;
            id = parseInt(id);
            if (isNaN(id) || !JSROOT.Painter.root_line_styles[id]) return;
            this.lineatt.Change(undefined, undefined, id);
            this.Redraw();
         }.bind(this));
         for (var n=1;n<11;++n) {

            var dash = JSROOT.Painter.root_line_styles[n];

            var svg = "<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>" + n + "</text><line x1='30' y1='8' x2='100' y2='8' stroke='black' stroke-width='3' stroke-dasharray='" + dash + "'></line></svg>";

            menu.addchk((this.lineatt.style==n), svg, n, function(arg) { this.lineatt.Change(undefined, undefined, parseInt(arg)); this.Redraw(); }.bind(this));
         }
         menu.add("endsub:");
         menu.add("endsub:");

         if (('excl_side' in this.lineatt) && (this.lineatt.excl_side!==0))  {
            menu.add("sub:Exclusion");
            menu.add("sub:side");
            for (var side=-1;side<=1;++side)
               menu.addchk((this.lineatt.excl_side==side), side, side, function(arg) {
                  this.lineatt.ChangeExcl(parseInt(arg));
                  this.Redraw();
               }.bind(this));
            menu.add("endsub:");

            this.AddSizeMenuEntry(menu, "width", 10, 100, 10, this.lineatt.excl_width,
                  function(arg) { this.lineatt.ChangeExcl(undefined, parseInt(arg)); this.Redraw(); }.bind(this));

            menu.add("endsub:");
         }
      }

      if (this.fillatt && this.fillatt.used) {
         menu.add("sub:"+preffix+"Fill att");
         this.AddColorMenuEntry(menu, "color", this.fillatt.colorindx,
               function(arg) { this.fillatt.Change(parseInt(arg), undefined, this.svg_canvas()); this.Redraw(); }.bind(this), this.fillatt.kind);
         menu.add("sub:style", function() {
            var id = prompt("Enter fill style id (1001-solid, 3000..3010)", this.fillatt.pattern);
            if (id == null) return;
            id = parseInt(id);
            if (isNaN(id)) return;
            this.fillatt.Change(undefined, id, this.svg_canvas());
            this.Redraw();
         }.bind(this));

         var supported = [1, 1001, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3010, 3021, 3022];

         var clone = JSROOT.clone(this.fillatt);
         if (clone.colorindx<=0) clone.colorindx = 1;

         for (var n=0; n<supported.length; ++n) {

            clone.Change(undefined, supported[n], this.svg_canvas());

            var svg = "<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>" + supported[n].toString() + "</text><rect x='40' y='0' width='60' height='18' stroke='none' fill='" + clone.color + "'></rect></svg>";

            menu.addchk(this.fillatt.pattern == supported[n], svg, supported[n], function(arg) {
               this.fillatt.Change(undefined, parseInt(arg), this.svg_canvas());
               this.Redraw();
            }.bind(this));
         }
         menu.add("endsub:");
         menu.add("endsub:");
      }

      if (this.markeratt && this.markeratt.used) {
         menu.add("sub:"+preffix+"Marker att");
         this.AddColorMenuEntry(menu, "color", this.markeratt.color,
                   function(arg) { this.markeratt.Change(arg); this.Redraw(); }.bind(this));
         this.AddSizeMenuEntry(menu, "size", 0.5, 6, 0.5, this.markeratt.size,
               function(arg) { this.markeratt.Change(undefined, undefined, parseFloat(arg)); this.Redraw(); }.bind(this));

         menu.add("sub:style");
         var supported = [1,2,3,4,5,6,7,8,21,22,23,24,25,26,27,28,29,30,31,32,33,34];

         var clone = JSROOT.clone(this.markeratt);
         for (var n=0; n<supported.length; ++n) {
            clone.Change(undefined, supported[n], 1.7);
            clone.reset_pos();
            var svg = "<svg width='60' height='18'><text x='1' y='12' style='font-size:12px'>" + supported[n].toString() + "</text><path stroke='black' fill='" + (clone.fill ? "black" : "none") + "' d='" + clone.create(40,8) + "'></path></svg>";

            menu.addchk(this.markeratt.style == supported[n], svg, supported[n],
                     function(arg) { this.markeratt.Change(undefined, parseInt(arg)); this.Redraw(); }.bind(this));
         }
         menu.add("endsub:");
         menu.add("endsub:");
      }
   }

   TObjectPainter.prototype.TextAttContextMenu = function(menu, prefix) {
      // for the moment, text attributes accessed directly from objects

      var obj = this.GetObject();
      if (!obj || !('fTextColor' in obj)) return;

      menu.add("sub:" + (prefix ? prefix : "Text"));
      this.AddColorMenuEntry(menu, "color", obj.fTextColor,
            function(arg) { this.GetObject().fTextColor = parseInt(arg); this.Redraw(); }.bind(this));

      var align = [11, 12, 13, 21, 22, 23, 31, 32, 33],
          hnames = ['left', 'centered' , 'right'],
          vnames = ['bottom', 'centered', 'top'];

      menu.add("sub:align");
      for (var n=0; n<align.length; ++n) {
         menu.addchk(align[n] == obj.fTextAlign,
                  align[n], align[n],
                  // align[n].toString() + "_h:" + hnames[Math.floor(align[n]/10) - 1] + "_v:" + vnames[align[n]%10-1], align[n],
                  function(arg) { this.GetObject().fTextAlign = parseInt(arg); this.Redraw(); }.bind(this));
      }
      menu.add("endsub:");

      menu.add("sub:font");
      for (var n=1; n<16; ++n) {
         menu.addchk(n == Math.floor(obj.fTextFont/10), n, n,
                  function(arg) { this.GetObject().fTextFont = parseInt(arg)*10+2; this.Redraw(); }.bind(this));
      }
      menu.add("endsub:");

      menu.add("endsub:");
   }

   TObjectPainter.prototype.ShowInpsector = function() {
      JSROOT.draw(this.divid, this.GetObject(), 'inspect');
   }

   TObjectPainter.prototype.FillContextMenu = function(menu) {

      var title = this.GetTipName();
      if (this.GetObject() && ('_typename' in this.GetObject()))
         title = this.GetObject()._typename + "::" + title;

      menu.add("header:"+ title);

      this.FillAttContextMenu(menu);

      if (menu.size()>0) menu.add('Inspect', this.ShowInpsector);

      return menu.size() > 0;
   }

   TObjectPainter.prototype.GetShowStatusFunc = function() {
      // return function used to display object status
      // automatically disabled when drawing is enlarged - status line will be invisible

      var pp = this.pad_painter(), res = JSROOT.Painter.ShowStatus;

      if (pp && (typeof pp.ShowStatus === 'function')) res = pp.ShowStatus;

      if (res && (this.enlarge_main('state')==='on')) res = null;

      return res;
   }

   TObjectPainter.prototype.ShowObjectStatus = function() {
      // method called normally when mouse enter main object element

      var obj = this.GetObject(),
          status_func = this.GetShowStatusFunc();

      if (obj && status_func) status_func(this.GetItemName() || obj.fName, obj.fTitle || obj._typename, obj._typename);
   }


   TObjectPainter.prototype.FindInPrimitives = function(objname) {
      // try to find object by name in list of pad primitives
      // used to find title drawing

      var painter = this.pad_painter(true);
      if ((painter === null) || (painter.pad === null)) return null;

      if (painter.pad.fPrimitives !== null)
         for (var n=0;n<painter.pad.fPrimitives.arr.length;++n) {
            var prim = painter.pad.fPrimitives.arr[n];
            if (('fName' in prim) && (prim.fName === objname)) return prim;
         }

      return null;
   }

   TObjectPainter.prototype.FindPainterFor = function(selobj,selname,seltype) {
      // try to find painter for specified object
      // can be used to find painter for some special objects, registered as
      // histogram functions

      var painter = this.pad_painter(true);
      var painters = (painter === null) ? null : painter.painters;
      if (painters === null) return null;

      for (var n = 0; n < painters.length; ++n) {
         var pobj = painters[n].GetObject();
         if (!pobj) continue;

         if (selobj && (pobj === selobj)) return painters[n];
         if (!selname && !seltype) continue;
         if (selname && (pobj.fName !== selname)) continue;
         if (seltype && (pobj._typename !== seltype)) continue;
         return painters[n];
      }

      return null;
   }

   TObjectPainter.prototype.ConfigureUserTooltipCallback = function(call_back, user_timeout) {
      // hook for the users to get tooltip information when mouse cursor moves over frame area
      // call_back function will be called every time when new data is selected
      // when mouse leave frame area, call_back(null) will be called

      if ((call_back === undefined) || (typeof call_back !== 'function')) {
         delete this.UserTooltipCallback;
         return;
      }

      if (user_timeout===undefined) user_timeout = 500;

      this.UserTooltipCallback = call_back;
      this.UserTooltipTimeout = user_timeout;
   }

   TObjectPainter.prototype.IsUserTooltipCallback = function() {
      return typeof this.UserTooltipCallback == 'function';
   }

   TObjectPainter.prototype.ProvideUserTooltip = function(data) {

      if (!this.IsUserTooltipCallback()) return;

      if (this.UserTooltipTimeout <= 0)
         return this.UserTooltipCallback(data);

      if (typeof this.UserTooltipTHandle != 'undefined') {
         clearTimeout(this.UserTooltipTHandle);
         delete this.UserTooltipTHandle;
      }

      if (data==null)
         return this.UserTooltipCallback(data);

      this.UserTooltipTHandle = setTimeout(function(d) {
         // only after timeout user function will be called
         delete this.UserTooltipTHandle;
         this.UserTooltipCallback(d);
      }.bind(this, data), this.UserTooltipTimeout);
   }

   TObjectPainter.prototype.Redraw = function() {
      // basic method, should be reimplemented in all derived objects
      // for the case when drawing should be repeated
   }

   TObjectPainter.prototype.StartTextDrawing = function(font_face, font_size, draw_g, max_font_size) {
      // we need to preserve font to be able rescale at the end

      if (!draw_g) draw_g = this.draw_g;

      var font = (font_size==='font') ? font_face : JSROOT.Painter.getFontDetails(font_face, font_size);

      draw_g.call(font.func);

      draw_g.property('draw_text_completed', false)
            .property('text_font', font)
            .property('mathjax_use', false)
            .property('text_factor', 0.)
            .property('max_text_width', 0) // keep maximal text width, use it later
            .property('max_font_size', max_font_size);
   }

   TObjectPainter.prototype.TextScaleFactor = function(value, draw_g) {
      // function used to remember maximal text scaling factor
      if (!draw_g) draw_g = this.draw_g;
      if (value && (value > draw_g.property('text_factor'))) draw_g.property('text_factor', value);
   }

   TObjectPainter.prototype.GetBoundarySizes = function(elem) {
      // getBBox does not work in mozilla when object is not displayed or not visible :(
      // getBoundingClientRect() returns wrong sizes for MathJax
      // are there good solution?

      if (elem===null) { console.warn('empty node in GetBoundarySizes'); return { width:0, height:0 }; }
      var box = elem.getBoundingClientRect(); // works always, but returns sometimes results in ex values, which is difficult to use
      if (parseFloat(box.width) > 0) box = elem.getBBox(); // check that elements visible, request precise value
      var res = { width : parseInt(box.width), height : parseInt(box.height) };
      if ('left' in box) { res.x = parseInt(box.left); res.y = parseInt(box.right); } else
      if ('x' in box) { res.x = parseInt(box.x); res.y = parseInt(box.y); }
      return res;
   }

   TObjectPainter.prototype.FinishTextDrawing = function(draw_g, call_ready) {
      if (!draw_g) draw_g = this.draw_g;

      if (draw_g.property('draw_text_completed')) {
         JSROOT.CallBack(call_ready);
         return draw_g.property('max_text_width');
      }

      if (call_ready) draw_g.node().text_callback = call_ready;

      var svgs = null;

      if (draw_g.property('mathjax_use')) {

         var missing = 0;
         svgs = draw_g.selectAll(".math_svg");

         svgs.each(function() {
            var fo_g = d3.select(this);
            if (fo_g.node().parentNode !== draw_g.node()) return;
            if (fo_g.select("svg").empty()) missing++;
         });

         // is any svg missing we should wait until drawing is really finished
         if (missing) return;
      }

      //if (!svgs) svgs = draw_g.selectAll(".math_svg");

      //var missing = 0;
      //svgs.each(function() {
      //   var fo_g = d3.select(this);
      //   if (fo_g.node().parentNode !== draw_g.node()) return;
      //   var entry = fo_g.property('_element');
      //   if (d3.select(entry).select("svg").empty()) missing++;
      //});
      //if (missing) console.warn('STILL SVG MISSING', missing);

      // adjust font size (if there are normal text)
      var painter = this,
          svg_factor = 0,
          f = draw_g.property('text_factor'),
          font = draw_g.property('text_font'),
          font_size = font.size;

      if ((f>0) && ((f<0.9) || (f>1.))) {
         font.size = Math.floor(font.size/f);
         if (draw_g.property('max_font_size') && (font.size > draw_g.property('max_font_size')))
            font.size = draw_g.property('max_font_size');
         draw_g.call(font.func);
         font_size = font.size;
      }

      // first analyze all MathJax SVG and repair width/height attributes
      if (svgs)
      svgs.each(function() {
         var fo_g = d3.select(this);
         if (fo_g.node().parentNode !== draw_g.node()) return;

         var vvv = fo_g.select("svg");
         if (vvv.empty()) {
            console.log('MathJax SVG ouptut error');
            return;
         }

         function transform(value) {
            if (!value || (typeof value !== "string")) return null;
            if (value.indexOf("ex")!==value.length-2) return null;
            value = parseFloat(value.substr(0, value.length-2));
            return isNaN(value) ? null : value*font_size*0.5;
         }

         var width = transform(vvv.attr("width")),
             height = transform(vvv.attr("height")),
             valign = vvv.attr("style");

         if (valign && valign.indexOf("vertical-align:")==0 && valign.indexOf("ex;")==valign.length-3) {
            valign = transform(valign.substr(16, valign.length-17));
         } else {
            valign = null;
         }

         width = (!width || (width<=0.5)) ? 1 : Math.round(width);
         height = (!height || (height<=0.5)) ? 1 : Math.round(height);

         vvv.attr("width", width).attr('height', height).attr("style",null);

         if (!JSROOT.nodejs) {
            var box = painter.GetBoundarySizes(fo_g.node());
            width = 1.05*box.width; height = 1.05*box.height;
         }

         var arg = fo_g.property("_arg");

         arg.valign = valign;

         if (arg.scale)
            svg_factor = Math.max(svg_factor, width / arg.width, height / arg.height);
      });

      if (svgs)
      svgs.each(function() {
         var fo_g = d3.select(this);
         // only direct parent
         if (fo_g.node().parentNode !== draw_g.node()) return;

         var arg = fo_g.property("_arg"),
             m = fo_g.select("svg"), // MathJax svg
             mw = parseInt(m.attr("width")),
             mh = parseInt(m.attr("height"));

         if (!isNaN(mh) && !isNaN(mw)) {
            if (svg_factor > 0.) {
               mw = mw/svg_factor;
               mh = mh/svg_factor;
               m.attr("width", Math.round(mw)).attr("height", Math.round(mh));
            }
         } else {
            var box = painter.GetBoundarySizes(fo_g.node()); // sizes before rotation
            mw = box.width || mw || 100;
            mh = box.height || mh || 10;
         }

         if ((svg_factor > 0.) && arg.valign) arg.valign = arg.valign/svg_factor;

         if (arg.valign===null) arg.valign = (font_size - mh)/2;

         var sign = { x:1, y:1 }, nx = "x", ny = "y";
         if (arg.rotate == 180) { sign.x = sign.y = -1; } else
         if ((arg.rotate == 270) || (arg.rotate == 90)) {
            sign.x = (arg.rotate == 270) ? -1 : 1;
            sign.y = -sign.x;
            nx = "y"; ny = "x"; // replace names to which align applied
         }

         if (arg.align[0] == 'middle') arg[nx] += sign.x*(arg.width - mw)/2; else
         if (arg.align[0] == 'end')    arg[nx] += sign.x*(arg.width - mw);

         if (arg.align[1] == 'middle') arg[ny] += sign.y*(arg.height - mh)/2; else
         if (arg.align[1] == 'bottom') arg[ny] += sign.y*(arg.height - mh); else
         if (arg.align[1] == 'bottom-base') arg[ny] += sign.y*(arg.height - mh - arg.valign);

         var trans = "translate("+arg.x+","+arg.y+")";
         if (arg.rotate) trans += " rotate("+arg.rotate+")";

         fo_g.attr('transform', trans).attr('visibility', null).property('_arg',null);
      });

      // now hidden text after rescaling can be shown
      draw_g.selectAll('.hidden_text').attr('visibility', null).attr('class', null).each(function() {
         // case when scaling is changed and we can shift text position only after final text size is defined
         var txt = d3.select(this),
             arg = txt.property("_arg");

         txt.property("_arg", null);

         if (!arg) return;

         if (JSROOT.nodejs) {
            if (arg.scale && (f>0)) { arg.box.width = arg.box.width/f; arg.box.height = arg.box.height/f; }
         } else if (!arg.plain) {
            // exact box dimension only required when complex text was build
            arg.box = painter.GetBoundarySizes(txt.node());
         }

         // if (arg.text.length>20) console.log(arg.box, arg.align, arg.x, arg.y, 'plain', arg.plain, 'inside', arg.width, arg.height);

         if (arg.width) {
            // adjust x position when scale into specified rectangle
            if (arg.align[0]=="middle") arg.x += arg.width/2; else
            if (arg.align[0]=="end") arg.x += arg.width;
         }

         arg.dx = arg.dy = 0;

         if (arg.plain) {
            txt.attr("text-anchor", arg.align[0]);
         } else {
            txt.attr("text-anchor", "start");
            arg.dx = ((arg.align[0]=="middle") ? -0.5 : ((arg.align[0]=="end") ? -1 : 0)) * arg.box.width;
         }

         if (arg.height) {
            if (arg.align[1].indexOf('bottom')===0) arg.y += arg.height; else
            if (arg.align[1] == 'middle') arg.y += arg.height/2;
         }

         if (arg.plain) {
            if (arg.align[1] == 'top') txt.attr("dy", ".8em"); else
            if (arg.align[1] == 'middle') {
               if (JSROOT.browser.isIE || JSROOT.nodejs) txt.attr("dy", ".4em"); else txt.attr("dominant-baseline", "middle");
            }
         } else {
            arg.dy = ((arg.align[1] == 'top') ? (arg.top_shift || 1) : (arg.align[1] == 'middle') ? (arg.mid_shift || 0.5) : 0) * arg.box.height;
         }

         // if (arg.text.length>20) console.log(arg.x, arg.y, arg.dx, arg.dy);

         if (!arg.rotate) { arg.x += arg.dx; arg.y += arg.dy; arg.dx = arg.dy = 0; }

         // use translate and then rotate to avoid complex sign calculations
         var trans = (arg.x || arg.y) ? "translate("+Math.round(arg.x)+","+Math.round(arg.y)+")" : "";
         if (arg.rotate) trans += " rotate("+Math.round(arg.rotate)+")";
         if (arg.dx || arg.dy) trans += " translate("+Math.round(arg.dx)+","+Math.round(arg.dy)+")";
         if (trans) txt.attr("transform", trans);

         if (JSROOT.browser.isWebKit && draw_g.node().insertAdjacentHTML && arg.large_latex) {
            // this is workaround for sporadic placement problem in Chrome/Opera
            // Due to unclear reasons tspan elements placed wrongly
            // Full refresh of created elements (including text itself) solves problem
            var html = txt.node().outerHTML;
            txt.remove();
            draw_g.node().insertAdjacentHTML( 'beforeend', html );
         }
      });

      if (!call_ready) call_ready = draw_g.node().text_callback;
      draw_g.node().text_callback = null;

      draw_g.property('draw_text_completed', true);

      // if specified, call ready function
      JSROOT.CallBack(call_ready);

      return draw_g.property('max_text_width');
   }

   TObjectPainter.prototype.produceLatex = function(node, label, arg, curr) {
      // attempt to implement subset of TLatex with plain SVG text and tspan elements

      if (!curr) {
         // initial dy = -0.1 is to move complete from very bottom line like with normal text drawing
         curr = { lvl: 0, x: 0, y: 0, dx: 0, dy: -0.1, fsize: arg.font_size, parent: null };
         arg.mainnode = node.node();
      }

      function extend_pos(pos, value) {

         var dx1, dx2, dy1, dy2;

         if (typeof value == 'string') {
            if (!pos.rect) pos.rect = { x: pos.x, y: pos.y, height: 0, width: 0 };
            dx1 = -pos.x;
            pos.x += value.length * arg.font.aver_width * pos.fsize;
            dx2 = pos.x;
            dy1 = -(pos.y-pos.fsize*1.1);
            dy2 = pos.y + pos.fsize*0.1;
         } else {
            if (!pos.rect) pos.rect = JSROOT.extend({}, value);
            dx1 = -value.x;
            dx2 = value.x+value.width;
            dy1 = -value.y;
            dy2 = value.y+value.height;
         }

         var rect = pos.rect;

         dx1 += rect.x;
         dx2 -= (rect.x+rect.width);
         dy1 += rect.y;
         dy2 -= (rect.y+rect.height);

         if (dx1>0) { rect.x -= dx1; rect.width += dx1; }
         if (dx2>0) rect.width += dx2;
         if (dy1>0) { rect.y -= dy1; rect.height += dy1; }
         if (dy2>0) rect.height+=dy2;

         if (pos.parent) return extend_pos(pos.parent, rect)

         // calculate dimensions for the
         arg.text_rect = rect;

         var h = rect.height, mid = rect.y + rect.height/2;

         if (h>0) {
            arg.mid_shift = -mid/h || 0.001;        // relative shift to get latex middle at given point
            arg.top_shift = -rect.y/h || 0.001; // relative shift to get latex top at given point
         }
      }

      function makeem(value) {
         if (Math.abs(value)<1e-2) return null; // very small values not needed, attribute will be removed
         if (value==Math.round(value)) return Math.round(value) + "em";
         var res = value.toFixed(2);
         if (res.indexOf("0.")==0) res = res.substr(1); else
         if (res.indexOf("-0.")==0) res = "-." + res.substr(3);
         if (res[res.length-1]=='0') res = res.substr(0, res.length-1);
         return res+"em";
      }

      function get_boundary(painter, element, approx_rect) {
         // actually, it is workaround for getBBox() or getElementBounday,
         // which is not implemented for tspan element in Firefox

         if (JSROOT.nodejs || !element || element.empty())
            return approx_rect || { height: 0, width: 0 };

         var important = [], prnt = element.node();

         // if (element.node().getBBox && !JSROOT.browser.isFirefox) return element.node().getBBox();

         while (prnt && (prnt!=arg.mainnode)) {
            important.push(prnt);
            prnt = prnt.parentNode;
         }

         element.selectAll('tspan').each(function() { important.push(this) });

         var tspans = d3.select(arg.mainnode).selectAll('tspan');

         // this is just workaround to know that many elements are created and in Chrome we need to redo them once again
         if (tspans.size()>3)  arg.large_latex = true;

         tspans.each(function() { if (important.indexOf(this)<0) d3.select(this).attr('display', 'none'); });
         var box = painter.GetBoundarySizes(arg.mainnode);

         tspans.each(function() { if (important.indexOf(this)<0) d3.select(this).attr('display', null); });

         return box;
      }

      var features = [
          { name: "#it{" }, // italic
          { name: "#bf{" }, // bold
          { name: "kern[", arg: 'float' }, // horizontal shift
          { name: "lower[", arg: 'float' },  // vertical shift
          { name: "scale[", arg: 'float' },  // font scale
          { name: "#color[", arg: 'int' },
          { name: "#font[", arg: 'int' },
          { name: "_{" },  // subscript
          { name: "^{" },   // superscript
          { name: "#bar{", accent: "\u02C9" }, // "\u0305"
          { name: "#hat{", accent: "\u02C6" }, // "\u0302"
          { name: "#check{", accent: "\u02C7" }, // "\u030C"
          { name: "#acute{", accent: "\u02CA" }, // "\u0301"
          { name: "#grave{", accent: "\u02CB" }, // "\u0300"
          { name: "#dot{", accent: "\u02D9" }, // "\u0307"
          { name: "#ddot{", accent: "\u02BA" }, // "\u0308"
          { name: "#tilde{", accent: "\u02DC" }, // "\u0303"
          { name: "#slash{", accent: "\u2215" }, // "\u0337"
          { name: "#vec{", accent: "\u02ED" }, // "\u0350" arrowhead
          { name: "#frac{" },
          { name: "#splitline{" },
          { name: "#sqrt[", arg: 'int' }, // root with arbitrary power (now only 3 or 4)
          { name: "#sqrt{" },
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

      var isany = false, best, found, foundarg, pos, n, subnode, subnode1, subpos = null, prevsubpos = null;

      while (label) {

         best = label.length; found = null; foundarg = null;

         for(n=0;n<features.length;++n) {
            pos = label.indexOf(features[n].name);
            if ((pos>=0) && (pos<best)) { best = pos; found = features[n]; }
         }

         if (!found && !isany) {
            var s = JSROOT.Painter.translateLaTeX(label);
            if (!curr.lvl && (s==label)) return 0; // indicate that nothing found - plain string
            extend_pos(curr, s);

            if (curr.accent && (s.length==1)) {
               var elem = node.append('svg:tspan').text(s),
                   rect = get_boundary(this, elem, { width : 10000 }),
                   w = Math.min(rect.width/curr.fsize, 0.5); // at maximum, 0.5 should be used

               node.append('svg:tspan').attr('dx', makeem(curr.dx-w)).attr('dy', makeem(curr.dy-0.2)).text(curr.accent);
               curr.dy = 0.2;; // compensate hat
               curr.dx = Math.max(0.2, w-0.2); // extra horizontal gap
               curr.accent = false;
            } else {
               node.text(s);
            }
            return true;
         }

         if (best>0) {
            var s = JSROOT.Painter.translateLaTeX(label.substr(0,best));
            if (s.length>0) {
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

         subpos = { lvl: curr.lvl+1, x: curr.x, y: curr.y, fsize: curr.fsize, dx:0, dy: 0, parent: curr };

         isany = true;

         if (found.arg) {
            pos = label.indexOf("]{");
            if (pos < 0) { console.log('missing argument for ', found.name); return false; }
            foundarg = label.substr(0,pos);
            if (found.arg == 'int') {
               foundarg = parseInt(foundarg);
               if (isNaN(foundarg)) { console.log('wrong int argument', label.substr(0,pos)); return false; }
            } else if (found.arg == 'float') {
               foundarg = parseFloat(foundarg);
               if (isNaN(foundarg)) { console.log('wrong float argument', label.substr(0,pos)); return false; }
            }
            label = label.substr(pos + 2);
         }

         var nextdy = curr.dy, nextdx = curr.dx, trav = null,
             scale = 1, left_brace = "{", right_brace = "}"; // this will be applied to the next element

         curr.dy = curr.dx = 0; // relative shift for elements

         if (found.special) {
            subnode.attr('dx', makeem(nextdx)).attr('dy', makeem(nextdy)).text(found.special);
            nextdx = nextdy = 0;
            curr.special = found;

            var rect = get_boundary(this, subnode);
            if (rect.width && rect.height) {
               found.w = rect.width/curr.fsize;
               found.h = rect.height/curr.fsize-0.1;
            }
            continue; // just create special node
         }

         if (found.braces) {
            // special handling of large braces
            subpos.left_cont = subnode.append('svg:tspan'); // container for left brace
            subpos.left = subpos.left_cont.append('svg:tspan').text(found.braces[0]);
            subnode1 = subnode.append('svg:tspan');
            subpos.left_rect = { y: curr.y - curr.fsize*1.1, height: curr.fsize*1.2, x: curr.x, width: curr.fsize*0.6 };
            extend_pos(curr, subpos.left_rect);
            subpos.braces = found; // indicate braces handling
            if (found.right) {
               left_brace = found.name;
               right_brace = found.right;
            }
         } else if (found.accent) {
            subpos.accent = found.accent;
         } else
         switch(found.name) {
            case "#color[":
               if (this.get_color(foundarg))
                   subnode.attr('fill', this.get_color(foundarg));
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
              JSROOT.Painter.getFontDetails(foundarg).setFont(subnode,'without-size');
              break;
           case "#it{":
              curr.italic = true;
              trav = curr;
              while (trav = trav.parent)
                 if (trav.italic!==undefined) {
                    curr.italic = !trav.italic;
                    break;
                 }
              subnode.attr('font-style', curr.italic ? 'italic' : 'normal');
              break;
           case "#bf{":
              curr.bold = true;
              trav = curr;
              while (trav = trav.parent)
                 if (trav.bold!==undefined) {
                    curr.bold = !trav.bold;
                    break;
                 }
              subnode.attr('font-weight', curr.bold ? 'bold' : 'normal');
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
                 nextdx += 0.1*scale;
                 nextdy += 0.4*scale;
                 subpos.y += 0.4*subpos.fsize;
                 curr.dy = -0.4*scale; // compensate vertical shift back

                 if (prevsubpos && (prevsubpos.script === 'super')) {
                    var rect = get_boundary(this, prevsubpos.node, prevsubpos.rect);
                    subpos.width_limit = rect.width;
                    nextdx -= (rect.width/subpos.fsize+0.1)*scale;
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

                 curr.dy = 0.6*scale; // compensate vertical shift afterwards
                 if (curr.next_super_dy) curr.dy -= curr.next_super_dy;

                 nextdx += 0.1*scale;
                 nextdy -= curr.dy;

                 subpos.y -= 0.4*subpos.fsize;

                 if (prevsubpos && (prevsubpos.script === 'sub')) {
                    var rect = get_boundary(this, prevsubpos.node, prevsubpos.rect);
                    subpos.width_limit = rect.width;
                    nextdx -= (rect.width/subpos.fsize+0.1)*scale;
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
              subpos.square_root.append('svg:tspan').text((foundarg==3) ? '\u221B' : ((foundarg==4) ? '\u221C' : '\u221A')); // unicode square, cubic and fourth root
              subnode1 = subnode.append('svg:tspan');
              subpos.sqrt_rect = { y: curr.y - curr.fsize*1.1, height: curr.fsize*1.2, x: 0, width: curr.fsize*0.7 };
              extend_pos(curr, subpos.sqrt_rect); // just dummy symbol instead of square root
              break;
         }

         if (scale!==1) {
            // handle centrally change of scale factor
            subnode.attr('font-size', Math.round(scale*100)+'%');
            subpos.fsize *= scale;
            nextdx = nextdx/scale;
            nextdy = nextdy/scale;
         }

         if (curr.special && !subpos.script) delete curr.special;
         delete curr.next_super_dy;

         subpos.node = subnode; // remember node where sublement is build

         while (true) {
            // loop need to create two lines for #frac or #splitline
            // normally only one sub-element is created

            // moving cursor with the tspan
            subpos.x += nextdx*subpos.fsize;
            subpos.y += nextdy*subpos.fsize;

            subnode.attr('dx', makeem(nextdx)).attr('dy', makeem(nextdy));
            nextdx = nextdy = 0;

            pos = -1; n = 1;

            while ((n!=0) && (++pos < label.length)) {
               if (label.indexOf(left_brace, pos) === pos) n++; else
               if (label.indexOf(right_brace, pos) === pos) n--;
            }

            if (n!=0) {
               console.log('mismatch with open ' + left_brace + ' and close ' + right_brace + ' braces in Latex', label);
               return false;
            }

            var sublabel = label.substr(0,pos);

            // if (subpos.square_root) sublabel = "#frac{a}{bc}";

            if (!this.produceLatex(subnode1, sublabel, arg, subpos)) return false;

            // takeover current possition and deltas
            curr.x = subpos.x;
            curr.y = subpos.y;

            curr.dx += subpos.dx*subpos.fsize/curr.fsize;
            curr.dy += subpos.dy*subpos.fsize/curr.fsize;

            label = label.substr(pos+right_brace.length);

            if (subpos.width_limit) {
               // special handling for the case when created element does not reach its minimal width
               // use when super-script and subscript should be combined together

               var rect = get_boundary(this,  subnode1, subpos.rect);
               if (rect.width < subpos.width_limit)
                  curr.dx += (subpos.width_limit-rect.width)/curr.fsize;
               delete subpos.width_limit;
            }

            if (curr.special) {
               // case over #sum or #integral one need to compensate width
               var rect = get_boundary(this,  subnode1, subpos.rect);
               curr.dx -= rect.width/curr.fsize; // compensate width as much as we can
            }

            if (subpos.square_root) {
               // creating cap for square root
               // while overline symbol does not match with square root, use empty text with overline
               var len = 2, scale = 1, sqrt_dy = 0, yscale = 1,
                   bs = get_boundary(this, subpos.square_root, subpos.sqrt_rect),
                   be = get_boundary(this, subnode1, subpos.rect);

               // we can compare y coordinates while both nodes (root and element) on the same level
               if ((be.height > bs.height) && (bs.height > 0)) {
                  yscale = be.height/bs.height*1.2;
                  sqrt_dy = ((be.y+be.height) - (bs.y+bs.height))/curr.fsize/yscale;
                  subpos.square_root.style('font-size', Math.round(100*yscale)+'%').attr('dy', makeem(sqrt_dy));
               }

               // we taking into account only element width
               len = be.width / subpos.fsize / yscale;

               var a = "", nn = Math.round(Math.max(len*3,2));
               while (nn--) a += '\u203E'; // unicode overline

               subpos.square_root.append('svg:tspan').attr("dy", makeem(-0.25)).text(a);

               subpos.square_root.append('svg:tspan').attr("dy", makeem(0.25-sqrt_dy)).attr("dx", makeem(-a.length/3-0.2)).text('\u2009'); // unicode tiny space

               break;
            }

            if (subpos.braces) {
               // handling braces

               var bs = get_boundary(this, subpos.left_cont, subpos.left_rect),
                   be = get_boundary(this, subnode1, subpos.rect),
                   yscale = 1, brace_dy = 0;

               // console.log('braces height', bs.height, ' entry height', be.height);

               if (1.2*bs.height < be.height) {
                  // make scaling
                  yscale = be.height/bs.height;
                  // brace_dy = ((be.y+be.height) - (bs.y+bs.height))/curr.fsize/yscale - 0.15;
                  brace_dy = 0;
                  subpos.left.style('font-size', Math.round(100*yscale)+'%').attr('dy', makeem(brace_dy));
                  // unicode tiny space, used to return cursor on vertical position
                  subpos.left_cont.append('svg:tspan').attr("dx",makeem(-0.2))
                                                  .attr("dy", makeem(-brace_dy*yscale)).text('\u2009');
                  curr.next_super_dy = -0.3*yscale; // special shift for next comming superscript
               }

               subpos.left_rect.y = curr.y;
               subpos.left_rect.height *= yscale;

               extend_pos(curr, subpos.left_rect); // just dummy symbol instead of right brace for accounting

               var right_cont = subnode.append('svg:tspan')
                                       .attr("dx", makeem(curr.dx))
                                       .attr("dy", makeem(curr.dy));

               curr.dx = curr.dy = 0;

               if (yscale!=1) right_cont.append('svg:tspan').attr("dx",makeem(-0.2)).text('\u2009'); // unicode tiny space if larger brace is used

               var right = right_cont.append('svg:tspan').text(subpos.braces.braces[1]);

               if (yscale!=1) {
                  right.style('font-size', Math.round(100*yscale)+'%').attr('dy', makeem(brace_dy));
                  curr.dy = -brace_dy*yscale; // compensation of right brace
               }

               break;
            }

            if (subpos.first && subpos.second) {
               // when two lines created, adjust horizontal position and place divider if required

               var rect1 = get_boundary(this, subpos.first, subpos.rect1),
                   rect2 = get_boundary(this, subpos.second, subpos.rect),
                   l1 = rect1.width / subpos.fsize,
                   l2 = rect2.width / subpos.fsize,
                   l3 = Math.max(l2, l1);

               if (subpos.need_middle) {
                  // starting from content len 1.2 two -- will be inserted
                  l3 = Math.round(Math.max(l3,1)+0.3);
                  var a = "";
                  while (a.length < l3) a += '\u2014';
                  node.append('svg:tspan')
                       .attr("dx", makeem(-0.5*(l3+l2)))
                       .attr("dy", makeem(curr.dy-0.2))
                       .text(a);
                  curr.dy = 0.2; // return to the normal level
                  curr.dx = 0.2; // extra spacing
               } else {
                  curr.dx = 0.2;
                  if (l2<l1) curr.dx += 0.5*(l1-l2);
               }

               if (subpos.need_middle || arg.align[0]=='middle') {
                  subpos.first.attr("dx", makeem(0.5*(l3-l1)));
                  subpos.second.attr("dx", makeem(-0.5*(l2+l1)));
               } else if (arg.align[0]=='end') {
                  if (l1<l2) subpos.first.attr("dx", makeem(l2-l1));
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

   TObjectPainter.prototype.DrawText = function(arg) {
      // following arguments can be supplied
      //  align - either int value or text
      //  x,y - position
      //  width, height - dimension (optional)
      //  text - text to draw
      //  latex - 0 - plain text, 1 - normal TLatex, 2 - math
      //  color - text color
      //  rotate - rotaion angle (optional)
      //  font_size - fixed font size (optional)
      //  draw_g - element where to place text

      var label = arg.text || "",
          align = ['start', 'middle'];

      if (typeof arg.align == 'string') {
         align = arg.align.split(";");
         if (align.length==1) align.push('middle');
      } else if (typeof arg.align == 'number') {
         if ((arg.align / 10) >= 3) align[0] = 'end'; else
         if ((arg.align / 10) >= 2) align[0] = 'middle';
         if ((arg.align % 10) == 0) align[1] = 'bottom'; else
         if ((arg.align % 10) == 1) align[1] = 'bottom-base'; else
         if ((arg.align % 10) == 3) align[1] = 'top';
      }

      arg.draw_g = arg.draw_g || this.draw_g;
      if (arg.latex===undefined) arg.latex = 1; //  latex 0-text, 1-latex, 2-math
      arg.align = align;
      arg.x = arg.x || 0;
      arg.y = arg.y || 0;
      arg.scale = arg.width && arg.height && !arg.font_size;
      arg.width = arg.width || 0;
      arg.height = arg.height || 0;

      if (JSROOT.gStyle.MathJax !== undefined) {
         switch (JSROOT.gStyle.MathJax) {
            case 0: JSROOT.gStyle.Latex = 2; break;
            case 2: JSROOT.gStyle.Latex = 4; break;
            default: JSROOT.gStyle.Latex = 3;
         }
         delete JSROOT.gStyle.MathJax;
      }

      if (typeof JSROOT.gStyle.Latex == 'string') {
         switch (JSROOT.gStyle.Latex) {
            case "off": JSROOT.gStyle.Latex = 0; break;
            case "symbols": JSROOT.gStyle.Latex = 1; break;
            case "MathJax":
            case "mathjax":
            case "math":   JSROOT.gStyle.Latex = 3; break;
            case "AlwaysMathJax":
            case "alwaysmath":
            case "alwaysmathjax": JSROOT.gStyle.Latex = 4; break;
            default:
               var code = parseInt(JSROOT.gStyle.Latex);
               JSROOT.gStyle.Latex = (!isNaN(code) && (code>=0) && (code<=4)) ? code : 2;
         }
      }

      var font = arg.draw_g.property('text_font'),
          use_mathjax = (arg.latex == 2);

      if (arg.latex === 1)
         use_mathjax = (JSROOT.gStyle.Latex > 3) || ((JSROOT.gStyle.Latex == 3) && JSROOT.Painter.isAnyLatex(label));

      // only Firefox can correctly rotate incapsulated SVG, produced by MathJax
      // if (!use_normal_text && (h<0) && !JSROOT.browser.isFirefox) use_normal_text = true;

      if (!use_mathjax || arg.nomathjax) {

         var txt = arg.draw_g.append("svg:text");

         if (arg.color) txt.attr("fill", arg.color);

         if (arg.font_size) txt.attr("font-size", arg.font_size);
                       else arg.font_size = font.size;

         arg.font = font; // use in latex conversion

         arg.plain = !arg.latex || (JSROOT.gStyle.Latex < 2) || (this.produceLatex(txt, label, arg) === 0);

         if (arg.plain) {
            if (arg.latex && (JSROOT.gStyle.Latex == 1)) label = Painter.translateLaTeX(label); // replace latex symbols
            txt.text(label);
         }

         // complete rectangle with very rougth size estimations
         arg.box = JSROOT.nodejs ? (arg.text_rect || { height: arg.font_size*1.2, width: JSROOT.Painter.approxTextWidth(font, label) })
                                 : this.GetBoundarySizes(txt.node());

         // if (label.length>20) console.log('label', label, 'box', arg.box);

         txt.attr('class','hidden_text')
             .attr('visibility','hidden') // hide elements until text drawing is finished
             .property("_arg", arg);

         if (arg.box.width > arg.draw_g.property('max_text_width')) arg.draw_g.property('max_text_width', arg.box.width);
         if (arg.scale) this.TextScaleFactor(1.05*arg.box.width/arg.width, arg.draw_g);
         if (arg.scale) this.TextScaleFactor(1.*arg.box.height/arg.height, arg.draw_g);

         return arg.box.width;
      }

      var mtext = JSROOT.Painter.translateMath(label, arg.latex, arg.color, this),
          fo_g = arg.draw_g.append("svg:g")
                       .attr('class', 'math_svg')
                       .attr('visibility','hidden')
                       .property('_arg', arg);

      arg.draw_g.property('mathjax_use', true);  // one need to know that mathjax is used

      if (JSROOT.nodejs) {
         // special handling for Node.js

         if (!JSROOT.nodejs_mathjax) {
            JSROOT.nodejs_mathjax = require("mathjax-node");
            JSROOT.nodejs_mathjax.config({
               TeX: { extensions: ["color.js"] },
               SVG: { mtextFontInherit: true, minScaleAdjust: 100, matchFontHeight: true, useFontCache: false }
            });
            JSROOT.nodejs_mathjax.start();
         }

         if ((mtext.indexOf("\\(")==0) && (mtext.lastIndexOf("\\)")==mtext.length-2))
            mtext = mtext.substr(2,mtext.length-4);

         JSROOT.nodejs_mathjax.typeset({
            jsroot_painter: this,
            jsroot_drawg: arg.draw_g,
            jsroot_fog: fo_g,
            ex: font.size,
            math: mtext,
            useFontCache: false,
            useGlobalCache: false,
            format: "TeX", // "TeX", "inline-TeX", "MathML"
            svg: true //  svg:true,
          }, function (data, opt) {
             if (!data.errors) {
                opt.jsroot_fog.html(data.svg);
             } else {
                console.log('MathJax error', opt.math);
                opt.jsroot_fog.html("<svg></svg>");
             }
             opt.jsroot_painter.FinishTextDrawing(opt.jsroot_drawg);
          });

         return 0;
      }

      var element = document.createElement("p");

      d3.select(element).style('visibility',"hidden").style('overflow',"hidden").style('position',"absolute")
                        .style("font-size",font.size+'px').style("font-family",font.name)
                        .html('<mtext>' + mtext + '</mtext>');
      document.body.appendChild(element);

      fo_g.property('_element', element);

      var painter = this;

      JSROOT.AssertPrerequisites('mathjax', function() {

         MathJax.Hub.Typeset(element, ["FinishMathjax", painter, arg.draw_g, fo_g]);

         MathJax.Hub.Queue(["FinishMathjax", painter, arg.draw_g, fo_g]); // repeat once again, while Typeset not always invoke callback
      });

      return 0;
   }

   TObjectPainter.prototype.FinishMathjax = function(draw_g, fo_g, id) {
      // function should be called when processing of element is completed

      if (fo_g.node().parentNode !== draw_g.node()) return;

      var entry = fo_g.property('_element');
      if (!entry) return;

      var vvv = d3.select(entry).select("svg");

      if (vvv.empty()) {

         var merr = d3.select(entry).select("merror"); // indication of error

         if (merr.empty()) return; // not yet finished

         console.warn('MathJax error', merr.text());

         var arg = fo_g.property('_arg');

         if (arg && arg.latex!=2) {
            arg.nomathjax = true;
            fo_g.remove(); // delete special entry
            this.DrawText(arg);
         } else
            fo_g.append("svg").attr('width', Math.min(20, merr.text().length + 5) + 'ex')
                              .attr('height', '3ex')
                              .style('vertical-align','0ex')
                              .append("text")
                              .style('font-size','12px')
                              .style('fill','red')
                              .attr('x','0')
                              .attr('y','2ex')
                              .text("Err: " + merr.text());
      } else {
         vvv.remove();
         fo_g.append(function() { return vvv.node(); });
      }

      fo_g.property('_element', null);
      document.body.removeChild(entry);

      this.FinishTextDrawing(draw_g); // check if all other elements are completed
   }


   // ===========================================================

   function TooltipHandler(obj) {
      JSROOT.TObjectPainter.call(this, obj);
      this.tooltip_enabled = true;  // this is internally used flag to temporary disbale/enable tooltib
      this.tooltip_allowed = (JSROOT.gStyle.Tooltip > 0); // this is interactively changed property
   }

   TooltipHandler.prototype = Object.create(TObjectPainter.prototype);

   TooltipHandler.prototype.hints_layer = function() {
      // return layer where frame tooltips are shown
      // only canvas info_layer can be used while other pads can overlay

      var canp = this.pad_painter();
      return canp ? canp.svg_layer("info_layer") : d3.select(null);
   }

   TooltipHandler.prototype.IsTooltipShown = function() {
      // return true if tooltip is shown, use to prevent some other action
      if (!this.tooltip_allowed || !this.tooltip_enabled) return false;
      return ! (this.hints_layer().select(".objects_hints").empty());
   }

   TooltipHandler.prototype.ProcessTooltipEvent = function(pnt, enabled) {
      // make central function which let show selected hints for the object

      if (enabled !== undefined) this.tooltip_enabled = enabled;

      var hints = [], nhints = 0, maxlen = 0, lastcolor1 = 0, usecolor1 = false,
          textheight = 11, hmargin = 3, wmargin = 3, hstep = 1.2,
          frame_rect = this.GetFrameRect(),
          pad_width = this.pad_width(),
          pp = this.pad_painter(true),
          font = JSROOT.Painter.getFontDetails(160, textheight),
          status_func = this.GetShowStatusFunc(),
          disable_tootlips = !this.tooltip_allowed || !this.tooltip_enabled;

      if ((pnt === undefined) || (disable_tootlips && !status_func)) pnt = null;
      if (pnt && disable_tootlips) pnt.disabled = true; // indicate that highlighting is not required

      // collect tooltips from pad painter - it has list of all drawn objects
      if (pp) hints = pp.GetTooltips(pnt);

      if (pnt && pnt.touch) textheight = 15;

      for (var n=0; n < hints.length; ++n) {
         var hint = hints[n];
         if (!hint) continue;
         if (!hint.lines || (hint.lines.length===0)) {
            hints[n] = null; continue;
         }

         // check if fully duplicated hint already exists
         for (var k=0;k<n;++k) {
            var hprev = hints[k], diff = false;
            if (!hprev || (hprev.lines.length !== hint.lines.length)) continue;
            for (var l=0;l<hint.lines.length && !diff;++l)
               if (hprev.lines[l] !== hint.lines[l]) diff = true;
            if (!diff) { hints[n] = null; break; }
         }
         if (!hints[n]) continue;

         nhints++;

         for (var l=0;l<hint.lines.length;++l)
            maxlen = Math.max(maxlen, hint.lines[l].length);

         hint.height = Math.round(hint.lines.length*textheight*hstep + 2*hmargin - textheight*(hstep-1));

         if ((hint.color1!==undefined) && (hint.color1!=='none')) {
            if ((lastcolor1!==0) && (lastcolor1 !== hint.color1)) usecolor1 = true;
            lastcolor1 = hint.color1;
         }
      }

      var layer = this.hints_layer(),
          hintsg = layer.select(".objects_hints"); // group with all tooltips

      if (status_func) {
         var title = "", name = "", coordinates = "", info = "";
         if (pnt) coordinates = Math.round(pnt.x)+","+Math.round(pnt.y);
         var hint = null, best_dist2 = 1e10, best_hint = null;
         // try to select hint with exact match of the position when several hints available
         if (hints && hints.length>0)
            for (var k=0;k<hints.length;++k) {
               if (!hints[k]) continue;
               if (!hint) hint = hints[k];
               if (hints[k].exact && (!hint || !hint.exact)) { hint = hints[k]; break; }

               if (!pnt || (hints[k].x===undefined) || (hints[k].y===undefined)) continue;

               var dist2 = (pnt.x-hints[k].x)*(pnt.x-hints[k].x) + (pnt.y-hints[k].y)*(pnt.y-hints[k].y);
               if (dist2<best_dist2) { best_dist2 = dist2; best_hint = hints[k]; }
            }

         if ((!hint || !hint.exact) && (best_dist2 < 400)) hint = best_hint;

         if (hint) {
            name = (hint.lines && hint.lines.length>1) ? hint.lines[0] : hint.name;
            title = hint.title || "";
            info = hint.line;
            if (!info && hint.lines) info = hint.lines.slice(1).join(' ');
         }

         status_func(name, title, info, coordinates);
      }

      // end of closing tooltips
      if (!pnt || disable_tootlips || (hints.length===0) || (maxlen===0) || (nhints > 15)) {
         hintsg.remove();
         return;
      }

      // we need to set pointer-events=none for all elements while hints
      // placed in front of so-called interactive rect in frame, used to catch mouse events

      if (hintsg.empty())
         hintsg = layer.append("svg:g")
                       .attr("class", "objects_hints")
                       .style("pointer-events","none");

      var frame_shift = { x: 0, y: 0 }, trans = frame_rect.transform || "";
      if (!pp.iscan) {
         pp.CalcAbsolutePosition(this.svg_pad(), frame_shift);
         trans = "translate(" + frame_shift.x + "," + frame_shift.y + ") " + trans;
      }

      // copy transform attributes from frame itself
      hintsg.attr("transform", trans);

      hintsg.property("last_point", pnt);

      var viewmode = hintsg.property('viewmode') || "",
          actualw = 0, posx = pnt.x + frame_rect.hint_delta_x;

      if (nhints > 1) {
         // if there are many hints, place them left or right

         var bleft = 0.5, bright = 0.5;

         if (viewmode=="left") bright = 0.7; else
         if (viewmode=="right") bleft = 0.3;

         if (posx <= bleft*frame_rect.width) {
            viewmode = "left";
            posx = 20;
         } else if (posx >= bright*frame_rect.width) {
            viewmode = "right";
            posx = frame_rect.width - 60;
         } else {
            posx = hintsg.property('startx');
         }
      } else {
         viewmode = "single";
         posx += 15;
      }

      if (viewmode !== hintsg.property('viewmode')) {
         hintsg.property('viewmode', viewmode);
         hintsg.selectAll("*").remove();
      }

      var curry = 10, // normal y coordinate
          gapy = 10,  // y coordinate, taking into account all gaps
          gapminx = -1111, gapmaxx = -1111,
          minhinty = -frame_shift.y,
          maxhinty = this.pad_height("") - frame_rect.y - frame_shift.y;

      function FindPosInGap(y) {
         for (var n=0;(n<hints.length) && (y < maxhinty); ++n) {
            var hint = hints[n];
            if (!hint) continue;
            if ((hint.y>=y-5) && (hint.y <= y+hint.height+5)) {
               y = hint.y+10;
               n = -1;
            }
         }
         return y;
      }

      for (var n=0; n < hints.length; ++n) {
         var hint = hints[n],
             group = hintsg.select(".painter_hint_"+n);
         if (hint===null) {
            group.remove();
            continue;
         }

         var was_empty = group.empty(), dx = 0, dy = 0;

         if (was_empty)
            group = hintsg.append("svg:svg")
                          .attr("class", "painter_hint_"+n)
                          .attr('opacity', 0) // use attribute, not style to make animation with d3.transition()
                          .style('overflow','hidden')
                          .style("pointer-events","none");

         if (viewmode == "single") {
            curry = pnt.touch ? (pnt.y - hint.height - 5) : Math.min(pnt.y + 15, maxhinty - hint.height - 3) + frame_rect.hint_delta_y;
         } else {
            gapy = FindPosInGap(gapy);
            if ((gapminx === -1111) && (gapmaxx === -1111)) gapminx = gapmaxx = hint.x;
            gapminx = Math.min(gapminx, hint.x);
            gapmaxx = Math.min(gapmaxx, hint.x);
         }

         group.attr("x", posx)
              .attr("y", curry)
              .property("curry", curry)
              .property("gapy", gapy);

         curry += hint.height + 5;
         gapy += hint.height + 5;

         if (!was_empty)
            group.selectAll("*").remove();

         group.attr("width", 60)
              .attr("height", hint.height);

         var r = group.append("rect")
                      .attr("x",0)
                      .attr("y",0)
                      .attr("width", 60)
                      .attr("height", hint.height)
                      .attr("fill","lightgrey")
                      .style("pointer-events","none");

         if (nhints > 1) {
            var col = usecolor1 ? hint.color1 : hint.color2;
            if ((col !== undefined) && (col!=='none'))
               r.attr("stroke", col).attr("stroke-width", hint.exact ? 3 : 1);
         }

         if (hint.lines != null) {
            for (var l=0;l<hint.lines.length;l++)
               if (hint.lines[l]!==null) {
                  var txt = group.append("svg:text")
                                 .attr("text-anchor", "start")
                                 .attr("x", wmargin)
                                 .attr("y", hmargin + l*textheight*hstep)
                                 .attr("dy", ".8em")
                                 .attr("fill","black")
                                 .style("pointer-events","none")
                                 .call(font.func)
                                 .text(hint.lines[l]);

                  var box = this.GetBoundarySizes(txt.node());

                  actualw = Math.max(actualw, box.width);
               }
         }

         function translateFn() {
            // We only use 'd', but list d,i,a as params just to show can have them as params.
            // Code only really uses d and t.
            return function(d, i, a) {
               return function(t) {
                  return t < 0.8 ? "0" : (t-0.8)*5;
               };
            };
         }

         if (was_empty)
            if (JSROOT.gStyle.TooltipAnimation > 0)
               group.transition().duration(JSROOT.gStyle.TooltipAnimation).attrTween("opacity", translateFn());
            else
               group.attr('opacity',1);
      }

      actualw += 2*wmargin;

      var svgs = hintsg.selectAll("svg");

      if ((viewmode == "right") && (posx + actualw > frame_rect.width - 20)) {
         posx = frame_rect.width - actualw - 20;
         svgs.attr("x", posx);
      }

      if ((viewmode == "single") && (posx + actualw > pad_width - frame_rect.x) && (posx > actualw+20)) {
         posx -= (actualw + 20);
         svgs.attr("x", posx);
      }

      // if gap not very big, apply gapy coordinate to open view on the histogram
      if ((viewmode !== "single") && (gapy < maxhinty) && (gapy !== curry)) {
         if ((gapminx <= posx+actualw+5) && (gapmaxx >= posx-5))
            svgs.attr("y", function() { return d3.select(this).property('gapy'); });
      } else if ((viewmode !== 'single') && (curry > maxhinty)) {
         var shift = Math.max((maxhinty - curry - 10), minhinty);
         if (shift<0)
            svgs.attr("y", function() { return d3.select(this).property('curry') + shift; });
      }

      if (actualw > 10)
         svgs.attr("width",actualw)
             .select('rect').attr("width", actualw);

      hintsg.property('startx', posx);
   }

   // ===============================================


   function TFramePainter(tframe) {
      TooltipHandler.call(this, tframe);
   }

   TFramePainter.prototype = Object.create(TooltipHandler.prototype);

   TFramePainter.prototype.GetTipName = function(append) {
      var res = TooltipHandler.prototype.GetTipName.call(this) || "TFrame";
      if (append) res+=append;
      return res;
   }

   TFramePainter.prototype.Shrink = function(shrink_left, shrink_right) {
      this.fX1NDC += shrink_left;
      this.fX2NDC -= shrink_right;
   }

   TFramePainter.prototype.SetLastEventPos = function(pnt) {
      // set position of last context menu event, can be
      this.fLastEventPnt = pnt;
   }

   TFramePainter.prototype.GetLastEventPos = function() {
      // return position of last event
      return this.fLastEventPnt;
   }

   TFramePainter.prototype.UpdateAttributes = function(force) {
      var pad = this.root_pad(),
          tframe = this.GetObject();

      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {
         if (!pad) {
            JSROOT.extend(this, JSROOT.gStyle.FrameNDC);
         } else {
            JSROOT.extend(this, {
               fX1NDC: pad.fLeftMargin,
               fX2NDC: 1 - pad.fRightMargin,
               fY1NDC: pad.fBottomMargin,
               fY2NDC: 1 - pad.fTopMargin
            });
         }
      }

      if (this.fillatt === undefined) {
         if (tframe) this.fillatt = this.createAttFill(tframe);
         else if (pad) this.fillatt = pad.fFrameFillColor ? this.createAttFill(null, pad.fFrameFillStyle, pad.fFrameFillColor) : this.createAttFill(pad);
         else this.fillatt = this.createAttFill(null, 1001, 0);

         // force white color for the frame
         if (!tframe && (this.fillatt.color == 'none') && this.pad_painter(true) && this.pad_painter(true).iscan) {
            this.fillatt.color = 'white';
         }
      }

      if (this.lineatt === undefined)
         if (pad) this.lineatt = new TAttLineHandler({ fLineColor: pad.fFrameLineColor, fLineWidth: pad.fFrameLineWidth, fLineStyle: pad.fFrameLineStyle });
             else this.lineatt = new TAttLineHandler(tframe ? tframe : 'black');
   }

   TFramePainter.prototype.SizeChanged = function() {
      // function called at the end of resize of frame
      // One should apply changes to the pad

      var pad = this.root_pad(),
          main = this.main_painter();

      if (pad) {
         pad.fLeftMargin = this.fX1NDC;
         pad.fRightMargin = 1 - this.fX2NDC;
         pad.fBottomMargin = this.fY1NDC;
         pad.fTopMargin = 1 - this.fY2NDC;
         if (main) main.SetRootPadRange(pad);
      }

      this.RedrawPad();
   }

   TFramePainter.prototype.Cleanup = function() {
      if (this.draw_g) {
         this.draw_g.selectAll("*").remove();
         this.draw_g.on("mousedown", null)
                    .on("dblclick", null)
                    .on("wheel", null)
                    .on("contextmenu", null)
                    .property('interactive_set', null);
      }
      this.draw_g = null;
      TooltipHandler.prototype.Cleanup.call(this);
   }

   TFramePainter.prototype.Redraw = function() {

      // first update all attributes from objects
      this.UpdateAttributes();

      var width = this.pad_width(),
          height = this.pad_height(),
          lm = Math.round(width * this.fX1NDC),
          w = Math.round(width * (this.fX2NDC - this.fX1NDC)),
          tm = Math.round(height * (1 - this.fY2NDC)),
          h = Math.round(height * (this.fY2NDC - this.fY1NDC)),
          rotate = false, fixpos = false, pp = this.pad_painter();

      if (pp && pp.options) {
         if (pp.options.RotateFrame) rotate = true;
         if (pp.options.FixFrame) fixpos = true;
      }

      // this is svg:g object - container for every other items belonging to frame
      this.draw_g = this.svg_layer("primitives_layer").select(".root_frame");

      var top_rect, main_svg;

      if (this.draw_g.empty()) {

         var layer = this.svg_layer("primitives_layer");

         this.draw_g = layer.append("svg:g").attr("class", "root_frame");

         this.draw_g.append("svg:title").text("");

         top_rect = this.draw_g.append("svg:rect");

         // append for the moment three layers - for drawing and axis
         this.draw_g.append('svg:g').attr('class','grid_layer');

         main_svg = this.draw_g.append('svg:svg')
                           .attr('class','main_layer')
                           .attr("x", 0)
                           .attr("y", 0)
                           .attr('overflow', 'hidden');

         this.draw_g.append('svg:g').attr('class','axis_layer');
         this.draw_g.append('svg:g').attr('class','upper_layer');
      } else {
         top_rect = this.draw_g.select("rect");
         main_svg = this.draw_g.select(".main_layer");
      }

      var trans = "translate(" + lm + "," + tm + ")";
      if (rotate) {
         trans += " rotate(-90) " + "translate(" + -h + ",0)";
         var d = w; w = h; h = d;
      }

      this.draw_g.property('frame_painter', this) // simple way to access painter via frame container
                 .property('draw_x', lm)
                 .property('draw_y', tm)
                 .property('draw_width', w)
                 .property('draw_height', h)
                 .attr("transform", trans);

      top_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      main_svg.attr("width", w)
              .attr("height", h)
              .attr("viewBox", "0 0 " + w + " " + h);

      var tooltip_rect = this.draw_g.select(".interactive_rect");

      if (JSROOT.BatchMode) return tooltip_rect.remove();

      this.draw_g.attr("x", lm)
                 .attr("y", tm)
                 .attr("width", w)
                 .attr("height", h);

      if (!rotate && !fixpos)
         this.AddDrag({ obj: this, only_resize: true, minwidth: 20, minheight: 20,
                        redraw: this.SizeChanged.bind(this) });

      var painter = this;

      function MouseMoveEvent() {
         var pnt = d3.mouse(tooltip_rect.node());
         painter.ProcessTooltipEvent({ x: pnt[0], y: pnt[1], touch: false });
      }

      function MouseCloseEvent() {
         painter.ProcessTooltipEvent(null);
      }

      function TouchMoveEvent() {
         var pnt = d3.touches(tooltip_rect.node());
         if (!pnt || pnt.length !== 1) return painter.ProcessTooltipEvent(null);
         painter.ProcessTooltipEvent({ x: pnt[0][0], y: pnt[0][1], touch: true });
      }

      function TouchCloseEvent() {
         painter.ProcessTooltipEvent(null);
      }

      if (tooltip_rect.empty()) {
         tooltip_rect =
            this.draw_g
                .append("rect")
                .attr("class","interactive_rect")
                .style('opacity',0)
                .style('fill',"none")
                .style("pointer-events","visibleFill")
                .on('mouseenter', MouseMoveEvent)
                .on('mousemove', MouseMoveEvent)
                .on('mouseleave', MouseCloseEvent);

         if (JSROOT.touches)
            tooltip_rect.on("touchstart", TouchMoveEvent)
                        .on("touchmove", TouchMoveEvent)
                        .on("touchend", TouchCloseEvent)
                        .on("touchcancel", TouchCloseEvent);
      }

      tooltip_rect.attr("x", 0)
                  .attr("y", 0)
                  .attr("width", w)
                  .attr("height", h);

      var hintsg = this.hints_layer().select(".objects_hints");
      // if tooltips were visible before, try to reconstruct them after short timeout
      if (!hintsg.empty() && this.tooltip_allowed)
         setTimeout(this.ProcessTooltipEvent.bind(this, hintsg.property('last_point')), 10);
   }

   TFramePainter.prototype.FillContextMenu = function(menu) {
      // fill context menu for the frame
      // it could be appended to the histogram menus

      var main = this.main_painter(), alone = menu.size()==0, pad = this.root_pad();

      if (alone)
         menu.add("header:Frame");
      else
         menu.add("separator");

      if (main) {
         if (main.zoom_xmin !== main.zoom_xmax)
            menu.add("Unzoom X", main.Unzoom.bind(main,"x"));
         if (main.zoom_ymin !== main.zoom_ymax)
            menu.add("Unzoom Y", main.Unzoom.bind(main,"y"));
         if (main.zoom_zmin !== main.zoom_zmax)
            menu.add("Unzoom Z", main.Unzoom.bind(main,"z"));
         menu.add("Unzoom all", main.Unzoom.bind(main,"xyz"));

         if (pad) {
            menu.addchk(pad.fLogx, "SetLogx", main.ToggleLog.bind(main,"x"));

            menu.addchk(pad.fLogy, "SetLogy", main.ToggleLog.bind(main,"y"));

            if (main.Dimension() == 2)
               menu.addchk(pad.fLogz, "SetLogz", main.ToggleLog.bind(main,"z"));
         }
         menu.add("separator");
      }

      menu.addchk(this.tooltip_allowed, "Show tooltips", function() {
         var fp = this.frame_painter();
         if (fp) fp.tooltip_allowed = !fp.tooltip_allowed;
      });
      this.FillAttContextMenu(menu,alone ? "" : "Frame ");
      menu.add("separator");
      menu.add("Save as frame.png", function(arg) {
         var top = this.svg_frame();
         if (!top.empty())
            JSROOT.saveSvgAsPng(top.node(), { name: "frame.png" } );
      });

      return true;
   }

   TFramePainter.prototype.GetFrameRect = function() {
      // returns frame rectangle plus extra info for hint display

      return {
         x: this.frame_x(),
         y: this.frame_y(),
         width: this.frame_width(),
         height: this.frame_height(),
         transform: this.draw_g ? this.draw_g.attr("transform") : "",
         hint_delta_x: 0,
         hint_delta_y: 0
      }
   }

   TFramePainter.prototype.ProcessFrameClick = function(pnt) {
      // function called when frame is clicked and object selection can be performed
      // such event can be used to select

      var pp = this.pad_painter(true);
      if (!pp) return;

      pnt.painters = true; // provide painters reference in the hints
      pnt.disabled = true; // do not invoke graphics

      // collect tooltips from pad painter - it has list of all drawn objects
      var hints = pp.GetTooltips(pnt), exact = null;
      for (var k=0; (k<hints.length) && !exact; ++k)
         if (hints[k] && hints[k].exact) exact = hints[k];
      //if (exact) console.log('Click exact', pnt, exact.painter.GetTipName());
      //      else console.log('Click frame', pnt);

      pp.SelectObjectPainter(exact ? exact.painter : this, pnt);
   }

   Painter.drawFrame = function(divid, obj) {
      var p = new TFramePainter(obj);
      p.SetDivId(divid, 2);
      p.Redraw();
      return p.DrawingReady();
   }

   // ===========================================================================

   function TPadPainter(pad, iscan) {
      TObjectPainter.call(this, pad);
      this.pad = pad;
      this.iscan = iscan; // indicate if working with canvas
      this.this_pad_name = "";
      if (!this.iscan && (pad !== null) && ('fName' in pad)) {
         this.this_pad_name = pad.fName.replace(" ", "_"); // avoid empty symbol in pad name
         var regexp = new RegExp("^[A-Za-z][A-Za-z0-9_]*$");
         if (!regexp.test(this.this_pad_name)) this.this_pad_name = 'jsroot_pad_' + JSROOT.id_counter++;
      }
      this.painters = []; // complete list of all painters in the pad
      this.has_canvas = true;
   }

   TPadPainter.prototype = Object.create(TObjectPainter.prototype);

   TPadPainter.prototype.Cleanup = function() {
      // cleanup only pad itself, all child elements will be collected and cleanup separately

      for (var k=0;k<this.painters.length;++k)
         this.painters[k].Cleanup();

      var svg_p = this.svg_pad(this.this_pad_name);
      if (!svg_p.empty()) {
         svg_p.property('pad_painter', null);
         svg_p.property('mainpainter', null);
         if (!this.iscan) svg_p.remove();
      }

      this.painters = [];
      this.pad = null;
      this.this_pad_name = "";
      this.has_canvas = false;

      TObjectPainter.prototype.Cleanup.call(this);
   }

   TPadPainter.prototype.CleanPrimitives = function(selector) {
      if (!selector || (typeof selector !== 'function')) return;

      for (var k=this.painters.length-1;k>=0;--k) {
         var p = this.painters[k];
         if (selector(p)) {
            p.Cleanup();
            this.painters.splice(k--, 1);
         }
      }
   }

   TPadPainter.prototype.GetCurrentPrimitiveIndx = function() {
      return this._current_primitive_indx || 0;
   }

   TPadPainter.prototype.GetNumPrimitives = function() {
      return this._num_primitives || 1;
   }

   TPadPainter.prototype.ForEachPainterInPad = function(userfunc, onlypadpainters) {
      userfunc(this);
      for (var k = 0; k < this.painters.length; ++k) {
         var sub = this.painters[k];
         if (typeof sub.ForEachPainterInPad === 'function')
            sub.ForEachPainterInPad(userfunc, onlypadpainters);
         else if (!onlypadpainters) userfunc(sub);
      }
   }

   TPadPainter.prototype.ButtonSize = function(fact) {
      return Math.round((!fact ? 1 : fact) * (this.iscan || !this.has_canvas ? 16 : 12));
   }

   TPadPainter.prototype.IsTooltipAllowed = function() {
      var res = undefined;
      this.ForEachPainterInPad(function(fp) {
         if ((res===undefined) && (fp.tooltip_allowed!==undefined)) res = fp.tooltip_allowed;
      });
      return res !== undefined ? res : false;
   }

   TPadPainter.prototype.SetTooltipAllowed = function(on) {
      this.ForEachPainterInPad(function(fp) {
         if (fp.tooltip_allowed!==undefined) fp.tooltip_allowed = on;
      });
   }

   TPadPainter.prototype.SelectObjectPainter = function(painter) {
      // dummy function, redefined in the TCanvasPainter
   }

   TPadPainter.prototype.CreateCanvasSvg = function(check_resize, new_size) {

      var factor = null, svg = null, lmt = 5, rect = null;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.svg_canvas();

         if (svg.empty()) return false;

         factor = svg.property('height_factor');

         rect = this.check_main_resize(check_resize, null, factor);

         if (!rect.changed) return false;

      } else {

         var render_to = this.select_main();

         if (render_to.style('position')=='static')
            render_to.style('position','relative');

         svg = render_to.append("svg")
             .attr("class", "jsroot root_canvas")
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null) // this is custom property
             .property('current_pad', "") // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         svg.append("svg:title").text("ROOT canvas");
         var frect = svg.append("svg:rect").attr("class","canvas_fillrect")
                               .attr("x",0).attr("y",0);
         if (!JSROOT.BatchMode)
            frect.style("pointer-events", "visibleFill")
                 .on("dblclick", this.EnlargePad.bind(this))
                 .on("click", this.SelectObjectPainter.bind(this, this))
                 .on("mouseenter", this.ShowObjectStatus.bind(this));

         svg.append("svg:g").attr("class","primitives_layer");
         svg.append("svg:g").attr("class","info_layer");
         svg.append("svg:g").attr("class","btns_layer");

         if (JSROOT.gStyle.ContextMenu)
            svg.select(".canvas_fillrect").on("contextmenu", this.ShowContextMenu.bind(this));

         factor = 0.66;
         if (this.pad && this.pad.fCw && this.pad.fCh && (this.pad.fCw > 0)) {
            factor = this.pad.fCh / this.pad.fCw;
            if ((factor < 0.1) || (factor > 10)) factor = 0.66;
         }

         if (this._fixed_size) {
            render_to.style("overflow","auto");
            rect = { width: this.pad.fCw, height: this.pad.fCh };
         } else {
            rect = this.check_main_resize(2, new_size, factor);
         }
      }

      if (!this.fillatt || !this.fillatt.changed)
         this.fillatt = this.createAttFill(this.pad);

      if ((rect.width<=lmt) || (rect.height<=lmt)) {
         svg.style("display", "none");
         console.warn("Hide canvas while geometry too small w=",rect.width," h=",rect.height);
         rect.width = 200; rect.height = 100; // just to complete drawing
      } else {
         svg.style("display", null);
      }

      if (this._fixed_size) {
         svg.attr("x", 0)
            .attr("y", 0)
            .attr("width", rect.width)
            .attr("height", rect.height)
            .style("position", "absolute");
      } else {
        svg.attr("x", 0)
           .attr("y", 0)
           .style("width", "100%")
           .style("height", "100%")
           .style("position", "absolute")
           .style("left", 0)
           .style("top", 0)
           .style("right", 0)
           .style("bottom", 0);
      }

      // console.log('CANVAS SVG width = ' + rect.width + " height = " + rect.height);

      svg.attr("viewBox", "0 0 " + rect.width + " " + rect.height)
         .attr("preserveAspectRatio", "none")  // we do not preserve relative ratio
         .property('height_factor', factor)
         .property('draw_x', 0)
         .property('draw_y', 0)
         .property('draw_width', rect.width)
         .property('draw_height', rect.height);

      svg.select(".canvas_fillrect")
         .attr("width", rect.width)
         .attr("height", rect.height)
         .call(this.fillatt.func);

      this.svg_layer("btns_layer")
          .attr("transform","translate(2," + (rect.height - this.ButtonSize(1.25)) + ")")
          .attr("display", svg.property("pad_enlarged") ? "none" : null); // hide buttons when sub-pad is enlarged

      return true;
   }

   TPadPainter.prototype.EnlargePad = function() {

      if (d3.event) {
         d3.event.preventDefault();
         d3.event.stopPropagation();
      }

      var svg_can = this.svg_canvas(),
          pad_enlarged = svg_can.property("pad_enlarged");

      if (this.iscan || !this.has_canvas || (!pad_enlarged && !this.HasObjectsToDraw() && !this.painters)) {
         if (this._fixed_size) return; // canvas cannot be enlarged in such mode
         if (!this.enlarge_main('toggle')) return;
         if (this.enlarge_main('state')=='off') svg_can.property("pad_enlarged", null);
      } else {
         if (!pad_enlarged) {
            this.enlarge_main(true);
            svg_can.property("pad_enlarged", this.pad);
         } else
         if (pad_enlarged === this.pad) {
            this.enlarge_main(false);
            svg_can.property("pad_enlarged", null);
         } else {
            console.error('missmatch with pad double click events');
         }
      }

      this.CheckResize({ force: true });
   }

   TPadPainter.prototype.CreatePadSvg = function(only_resize) {
      // returns true when pad is displayed and all its items should be redrawn

      if (!this.has_canvas) {
         this.CreateCanvasSvg(only_resize ? 2 : 0);
         return true;
      }

      var svg_can = this.svg_canvas(),
          width = svg_can.property("draw_width"),
          height = svg_can.property("draw_height"),
          pad_enlarged = svg_can.property("pad_enlarged"),
          pad_visible = !pad_enlarged || (pad_enlarged === this.pad),
          w = Math.round(this.pad.fAbsWNDC * width),
          h = Math.round(this.pad.fAbsHNDC * height),
          x = Math.round(this.pad.fAbsXlowNDC * width),
          y = Math.round(height * (1 - this.pad.fAbsYlowNDC)) - h,
          svg_pad = null, svg_rect = null, btns = null;

      if (pad_enlarged === this.pad) { w = width; h = height; x = y = 0; }

      if (only_resize) {
         svg_pad = this.svg_pad(this.this_pad_name);
         svg_rect = svg_pad.select(".root_pad_border");
         btns = this.svg_layer("btns_layer", this.this_pad_name);
      } else {
         svg_pad = svg_can.select(".primitives_layer")
             .append("svg:svg") // here was g before, svg used to blend all drawin outside
             .attr("class", "root_pad")
             .attr("pad", this.this_pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null); // this is custom property
         svg_rect = svg_pad.append("svg:rect").attr("class", "root_pad_border");

         svg_pad.append("svg:g").attr("class","primitives_layer");
         btns = svg_pad.append("svg:g").attr("class","btns_layer");

         if (JSROOT.gStyle.ContextMenu)
            svg_rect.on("contextmenu", this.ShowContextMenu.bind(this));

         if (!JSROOT.BatchMode)
            svg_rect.attr("pointer-events", "visibleFill") // get events also for not visible rect
                    .on("dblclick", this.EnlargePad.bind(this))
                    .on("click", this.SelectObjectPainter.bind(this, this))
                    .on("mouseenter", this.ShowObjectStatus.bind(this));
      }

      if (!this.fillatt || !this.fillatt.changed)
         this.fillatt = this.createAttFill(this.pad);
      if (!this.lineatt || !this.lineatt.changed) {
         this.lineatt = new TAttLineHandler(this.pad);
         if (this.pad.fBorderMode == 0) this.lineatt.color = 'none';
      }

      svg_pad
              //.attr("transform", "translate(" + x + "," + y + ")") // is not handled for SVG
             .attr("display", pad_visible ? null : "none")
             .attr("viewBox", "0 0 " + w + " " + h) // due to svg
             .attr("preserveAspectRatio", "none")   // due to svg, we do not preserve relative ratio
             .attr("x", x)    // due to svg
             .attr("y", y)   // due to svg
             .attr("width", w)    // due to svg
             .attr("height", h)   // due to svg
             .property('draw_x', x) // this is to make similar with canvas
             .property('draw_y', y)
             .property('draw_width', w)
             .property('draw_height', h);

      svg_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .call(this.fillatt.func)
              .call(this.lineatt.func);

      if (svg_pad.property('can3d') === 1)
         // special case of 3D canvas overlay
          this.select_main()
              .select(".draw3d_" + this.this_pad_name)
              .style('display', pad_visible ? '' : 'none');

      btns.attr("transform","translate("+ (w - (btns.property('nextx') || 0) - this.ButtonSize(1.25)) + "," + (h - this.ButtonSize(1.25)) + ")");

      return pad_visible;
   }

   TPadPainter.prototype.CheckSpecial = function(obj) {

      if (!obj || (obj._typename!=="TObjArray")) return false;

      if (obj.name == "ListOfColors") {
         if (!this.options || this.options.GlobalColors) // set global list of colors
            Painter.adoptRootColors(obj);
         if (this.options && this.options.LocalColors) {
            // copy existing colors and extend with new values
            this.root_colors = [];
            for (var n=0;n<JSROOT.Painter.root_colors.length;++n)
               this.root_colors[n] = JSROOT.Painter.root_colors[n];
            Painter.extendRootColors(this.root_colors, obj);
         }
         return true;
      }

      if (obj.name == "CurrentColorPalette") {
         var arr = [], missing = false;
         for (var n = 0; n < obj.arr.length; ++n) {
            var col = obj.arr[n];
            if (col && (col._typename == 'TColor')) {
               arr[n] = Painter.MakeColorRGB(col);
            } else {
               console.log('Missing color with index ' + n); missing = true;
            }
         }
         if (!this.options || (!missing && !this.options.IgnorePalette)) this.CanvasPalette = new ColorPalette(arr);
         return true;
      }

      return false;
   }

   TPadPainter.prototype.CheckSpecialsInPrimitives = function(can) {
      var lst = can ? can.fPrimitives : null;
      if (!lst) return;
      for (var i = 0; i < lst.arr.length; ++i) {
         if (this.CheckSpecial(lst.arr[i])) {
            lst.arr.splice(i,1);
            lst.opt.splice(i,1);
            i--;
         }
      }
   }

   TPadPainter.prototype.RemovePrimitive = function(obj) {
      if (!this.pad || !this.pad.fPrimitives) return;
      var indx = this.pad.fPrimitives.arr.indexOf(obj);
      if (indx>=0) this.pad.fPrimitives.RemoveAt(indx);
   }

   TPadPainter.prototype.FindPrimitive = function(exact_obj, classname, name) {
      if (!this.pad || !this.pad.fPrimitives) return null;

      for (var i=0; i < this.pad.fPrimitives.arr.length; i++) {
         var obj = this.pad.fPrimitives.arr[i];

         if ((exact_obj!==null) && (obj !== exact_obj)) continue;

         if ((classname !== undefined) && (classname !== null))
            if (obj._typename !== classname) continue;

         if ((name !== undefined) && (name !== null))
            if (obj.fName !== name) continue;

         return obj;
      }

      return null;
   }

   TPadPainter.prototype.HasObjectsToDraw = function() {
      // return true if any objects beside sub-pads exists in the pad

      if (!this.pad || !this.pad.fPrimitives) return false;

      for (var n=0;n<this.pad.fPrimitives.arr.length;++n)
         if (this.pad.fPrimitives.arr[n] && this.pad.fPrimitives.arr[n]._typename != "TPad") return true;

      return false;
   }

   TPadPainter.prototype.DrawPrimitives = function(indx, callback, ppainter) {

      if (indx===0) {
         // flag used to prevent immediate pad redraw during normal drawing sequence
         this._doing_pad_draw = true;

         // set number of primitves
         this._num_primitives = this.pad && this.pad.fPrimitives ? this.pad.fPrimitives.arr.length : 0;
      }

      while (true) {
         if (ppainter) ppainter._primitive = true; // mark painter as belonging to primitives

         if (!this.pad || (indx >= this.pad.fPrimitives.arr.length)) {
            delete this._doing_pad_draw;
            return JSROOT.CallBack(callback);
         }

         // handle use to invoke callback only when necessary
         var handle = { func: this.DrawPrimitives.bind(this, indx+1, callback) };

         // set current index
         this._current_primitive_indx = indx;

         ppainter = JSROOT.draw(this.divid, this.pad.fPrimitives.arr[indx], this.pad.fPrimitives.opt[indx], handle);

         if (!handle.completed) return;
         indx++;
      }
   }

   TPadPainter.prototype.GetTooltips = function(pnt) {
      var painters = [], hints = [];

      // first count - how many processors are there
      if (this.painters !== null)
         this.painters.forEach(function(obj) {
            if ('ProcessTooltip' in obj) painters.push(obj);
         });

      if (pnt) pnt.nproc = painters.length;

      painters.forEach(function(obj) {
         var hint = obj.ProcessTooltip(pnt);
         hints.push(hint);
         if (hint && pnt && pnt.painters) hint.painter = obj;
      });

      return hints;
   }

   TPadPainter.prototype.FillContextMenu = function(menu) {

      if (this.pad)
         menu.add("header: " + this.pad._typename + "::" + this.pad.fName);
      else
         menu.add("header: Canvas");

      var tooltipon = this.IsTooltipAllowed();
      menu.addchk(tooltipon, "Show tooltips", this.SetTooltipAllowed.bind(this, !tooltipon));

      if (!this._websocket) {

         function ToggleGridField(arg) {
            this.pad[arg] = this.pad[arg] ? 0 : 1;
            var main = this.svg_pad(this.this_pad_name).property('mainpainter');
            if (main && (typeof main.DrawGrids == 'function')) main.DrawGrids();
         }

         function SetTickField(arg) {
            this.pad[arg.substr(1)] = parseInt(arg[0]);

            var main = this.svg_pad(this.this_pad_name).property('mainpainter');
            if (main && (typeof main.DrawAxes == 'function')) main.DrawAxes();
         }

         menu.addchk(this.pad.fGridx, 'Grid x', 'fGridx', ToggleGridField);
         menu.addchk(this.pad.fGridy, 'Grid y', 'fGridy', ToggleGridField);
         menu.add("sub:Ticks x");
         menu.addchk(this.pad.fTickx == 0, "normal", "0fTickx", SetTickField);
         menu.addchk(this.pad.fTickx == 1, "ticks on both sides", "1fTickx", SetTickField);
         menu.addchk(this.pad.fTickx == 2, "labels on both sides", "2fTickx", SetTickField);
         menu.add("endsub:");
         menu.add("sub:Ticks y");
         menu.addchk(this.pad.fTicky == 0, "normal", "0fTicky", SetTickField);
         menu.addchk(this.pad.fTicky == 1, "ticks on both sides", "1fTicky", SetTickField);
         menu.addchk(this.pad.fTicky == 2, "labels on both sides", "2fTicky", SetTickField);
         menu.add("endsub:");

         //menu.addchk(this.pad.fTickx, 'Tick x', 'fTickx', ToggleField);
         //menu.addchk(this.pad.fTicky, 'Tick y', 'fTicky', ToggleField);

         this.FillAttContextMenu(menu);
      }

      menu.add("separator");

      if (this.ToggleEventStatus)
         menu.addchk(this.HasEventStatus(), "Event status", this.ToggleEventStatus.bind(this));

      if (this.enlarge_main() || (this.has_canvas && this.HasObjectsToDraw()))
         menu.addchk((this.enlarge_main('state')=='on'), "Enlarge " + (this.iscan ? "canvas" : "pad"), this.EnlargePad.bind(this));

      var fname = this.this_pad_name;
      if (fname.length===0) fname = this.iscan ? "canvas" : "pad";
      fname += ".png";

      menu.add("Save as "+fname, fname, this.SaveAsPng.bind(this, false));

      return true;
   }

   TPadPainter.prototype.ShowContextMenu = function(evnt) {
      if (!evnt) {
         // for debug purposes keep original context menu for small region in top-left corner
         var pos = d3.mouse(this.svg_pad(this.this_pad_name).node());

         if (pos && (pos.length==2) && (pos[0]>0) && (pos[0]<10) && (pos[1]>0) && pos[1]<10) return;

         d3.event.stopPropagation(); // disable main context menu
         d3.event.preventDefault();  // disable browser context menu

         // one need to copy event, while after call back event may be changed
         evnt = d3.event;

         var fp = this.frame_painter();
         if (fp) fp.SetLastEventPos();
      }

      JSROOT.Painter.createMenu(this, function(menu) {

         menu.painter.FillContextMenu(menu);

         menu.painter.FillObjectExecMenu(menu, "", function() { menu.show(evnt); });
      }); // end menu creation
   }

   TPadPainter.prototype.Redraw = function(resize) {

      // prevent redrawing
      if (this._doing_pad_draw) return console.log('Prevent redrawing', this.pad.fName);

      var showsubitems = true;

      if (this.iscan) {
         this.CreateCanvasSvg(2);
      } else {
         showsubitems = this.CreatePadSvg(true);
      }

      // even sub-pad is not visible, we should redraw sub-sub-pads to hide them as well
      for (var i = 0; i < this.painters.length; ++i) {
         var sub = this.painters[i];
         if (showsubitems || sub.this_pad_name) sub.Redraw(resize);
      }
   }

   TPadPainter.prototype.NumDrawnSubpads = function() {
      if (this.painters === undefined) return 0;

      var num = 0;

      for (var i = 0; i < this.painters.length; ++i) {
         var obj = this.painters[i].GetObject();
         if (obj && (obj._typename === "TPad")) num++;
      }

      return num;
   }

   TPadPainter.prototype.RedrawByResize = function() {
      if (this.access_3d_kind() === 1) return true;

      for (var i = 0; i < this.painters.length; ++i)
         if (typeof this.painters[i].RedrawByResize === 'function')
            if (this.painters[i].RedrawByResize()) return true;

      return false;
   }

   TPadPainter.prototype.CheckCanvasResize = function(size, force) {

      if (!this.iscan && this.has_canvas) return false;

      if (size && (typeof size === 'object') && size.force) force = true;

      if (!force) force = this.RedrawByResize();

      var changed = this.CreateCanvasSvg(force ? 2 : 1, size);

      // if canvas changed, redraw all its subitems.
      // If redrawing was forced for canvas, same applied for sub-elements
      if (changed)
         for (var i = 0; i < this.painters.length; ++i)
            this.painters[i].Redraw(force ? false : true);

      return changed;
   }

   TPadPainter.prototype.UpdateObject = function(obj) {
      if (!obj) return false;

      this.pad.fBits = obj.fBits;
      this.pad.fTitle = obj.fTitle;

      this.pad.fGridx = obj.fGridx;
      this.pad.fGridy = obj.fGridy;
      this.pad.fTickx = obj.fTickx;
      this.pad.fTicky = obj.fTicky;
      this.pad.fLogx  = obj.fLogx;
      this.pad.fLogy  = obj.fLogy;
      this.pad.fLogz  = obj.fLogz;

      this.pad.fUxmin = obj.fUxmin;
      this.pad.fUxmax = obj.fUxmax;
      this.pad.fUymin = obj.fUymin;
      this.pad.fUymax = obj.fUymax;

      this.pad.fLeftMargin   = obj.fLeftMargin;
      this.pad.fRightMargin  = obj.fRightMargin;
      this.pad.fBottomMargin = obj.fBottomMargin
      this.pad.fTopMargin    = obj.fTopMargin;

      this.pad.fFillColor = obj.fFillColor;
      this.pad.fFillStyle = obj.fFillStyle;
      this.pad.fLineColor = obj.fLineColor;
      this.pad.fLineStyle = obj.fLineStyle;
      this.pad.fLineWidth = obj.fLineWidth;

      if (this.iscan) this.CheckSpecialsInPrimitives(obj);

      var fp = this.frame_painter();
      if (fp) fp.UpdateAttributes(!fp.modified_NDC);

      if (!obj.fPrimitives) return false;

      var isany = false, p = 0;
      for (var n = 0; n < obj.fPrimitives.arr.length; ++n) {
         while (p < this.painters.length) {
            var pp = this.painters[p++];
            if (!pp._primitive) continue;
            if (pp.UpdateObject(obj.fPrimitives.arr[n])) isany = true;
            break;
         }
      }

      return isany;
   }

   TPadPainter.prototype.DrawNextSnap = function(lst, indx, call_back, objpainter) {
      // function called when drawing next snapshot from the list
      // it is also used as callback for drawing of previous snap

      if (indx===0) {
         // flag used to prevent immediate pad redraw during first draw
         this._doing_pad_draw = true;
         this._snaps_map = {}; // to control how much snaps are drawn
         this._num_primitives = lst ? lst.length : 0;
      }

      while (true) {

         if (objpainter && lst && lst[indx] && objpainter.snapid === undefined) {
            // keep snap id in painter, will be used for the
            if (this.painters.indexOf(objpainter)<0) this.painters.push(objpainter);
            objpainter.snapid = lst[indx].fObjectID;
         }

         objpainter = null;

         ++indx; // change to the next snap

         if (!lst || indx >= lst.length) {
            delete this._doing_pad_draw;
            delete this._snaps_map;
            return JSROOT.CallBack(call_back, this);
         }

         var snap = lst[indx],
             snapid = snap.fObjectID,
             cnt = this._snaps_map[snapid];

         if (cnt) cnt++; else cnt=1;
         this._snaps_map[snapid] = cnt; // check how many objects with same snapid drawn, use them again

         this._current_primitive_indx = indx;

         // first appropriate painter for the object
         // if same object drawn twice, two painters will exists
         for (var k=0; k<this.painters.length; ++k) {
            if (this.painters[k].snapid === snapid)
               if (--cnt === 0) { objpainter = this.painters[k]; break;  }
         }

         // function which should be called when drawing of next item finished
         var draw_callback = this.DrawNextSnap.bind(this, lst, indx, call_back);

         if (objpainter) {

            if (snap.fKind === 1) { // object itself
               if (objpainter.UpdateObject(snap.fSnapshot, snap.fOption)) objpainter.Redraw();
               continue; // call next
            }

            if (snap.fKind === 2) { // update SVG
               if (objpainter.UpdateObject(snap.fSnapshot)) objpainter.Redraw();
               continue; // call next
            }

            if (snap.fKind === 3) { // subpad
               return objpainter.RedrawPadSnap(snap, draw_callback);
            }

            continue; // call next
         }

         if (snap.fKind === 4) { // specials like list of colors
            this.CheckSpecial(snap.fSnapshot);
            continue;
         }

         if (snap.fKind === 3) { // subpad

            if (snap.fPrimitives._typename) {
               alert("Problem in JSON I/O with primitves for sub-pad");
               snap.fPrimitives = [ snap.fPrimitives ];
            }

            var subpad = snap.fPrimitives[0].fSnapshot;

            subpad.fPrimitives = null; // clear primitives, they just because of I/O

            var padpainter = new TPadPainter(subpad, false);
            padpainter.DecodeOptions(snap.fPrimitives[0].fOption);
            padpainter.SetDivId(this.divid); // pad painter will be registered in the canvas painters list
            padpainter.snapid = snap.fObjectID;

            padpainter.CreatePadSvg();

            if (padpainter.MatchObjectType("TPad") && snap.fPrimitives.length > 1) {
               padpainter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "PadSnapShot");
               padpainter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge pad", "EnlargePad");

               if (JSROOT.gStyle.ContextMenu)
                  padpainter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");
            }

            // we select current pad, where all drawing is performed
            var prev_name = padpainter.CurrentPadName(padpainter.this_pad_name);
            padpainter.DrawNextSnap(snap.fPrimitives, 0, function() {
               padpainter.CurrentPadName(prev_name);
               draw_callback(padpainter);
            });
            return;
         }

         var handle = { func: draw_callback };

         // here the case of normal drawing, can be improved
         if (snap.fKind === 1)
            objpainter = JSROOT.draw(this.divid, snap.fSnapshot, snap.fOption, handle);

         if (snap.fKind === 2)
            objpainter = JSROOT.draw(this.divid, snap.fSnapshot, snap.fOption, handle);

         if (!handle.completed) return; // if callback will be invoked, break while loop
      }
   }

   TPadPainter.prototype.FindSnap = function(snapid) {

      if (this.snapid === snapid) return this;

      if (!this.painters) return null;

      for (var k=0;k<this.painters.length;++k) {
         var sub = this.painters[k];

         if (typeof sub.FindSnap === 'function') sub = sub.FindSnap(snapid);
         else if (sub.snapid !== snapid) sub = null;

         if (sub) return sub;
      }

      return null;
   }

   TPadPainter.prototype.RedrawPadSnap = function(snap, call_back) {
      // for the canvas snapshot contains list of objects
      // as first entry, graphical properties of canvas itself is provided
      // in ROOT6 it also includes primitives, but we ignore them

      if (!snap || !snap.fPrimitives) return;

      // VERY BAD, NEED TO BE FIXED IN TBufferJSON - should be fixed now in master
      // Should be fixed now in ROOT
      // if (snap.fPrimitives._typename) snap.fPrimitives = [ snap.fPrimitives ];

      var first = snap.fPrimitives[0].fSnapshot;
      first.fPrimitives = null; // primitives are not interesting, just cannot disable it in IO

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.snapid = snap.fPrimitives[0].fObjectID;

         this.draw_object = first;
         this.pad = first;
         // this._fixed_size = true;

         // if canvas size not specified in batch mode, temporary use 900x700 size
         if (this.batch_mode && (!first.fCw || !first.fCh)) { first.fCw = 900; first.fCh = 700; }

         // case of ROOT7 with always dummy TPad as first entry
         if (!first.fCw || !first.fCh) this._fixed_size = false;

         this.CreateCanvasSvg(0);
         this.SetDivId(this.divid);  // now add to painters list

         this.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "CanvasSnapShot", "Ctrl PrintScreen");
         if (JSROOT.gStyle.ContextMenu)
            this.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");

         if (this.enlarge_main('verify'))
            this.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge canvas", "EnlargePad");

         this.DrawNextSnap(snap.fPrimitives, 0, call_back);

         return;
      }

      this.UpdateObject(first); // update only object attributes

      // apply all changes in the object (pad or canvas)
      if (this.iscan) {
         this.CreateCanvasSvg(2);
      } else {
         this.CreatePadSvg(true);
      }

      var isanyfound = false;

      // find and remove painters which no longer exists in the list
      for (var k=0;k<this.painters.length;++k) {
         var sub = this.painters[k];
         if (sub.snapid===undefined) continue; // look only for painters with snapid

         for (var i=1;i<snap.fPrimitives.length;++i)
            if (snap.fPrimitives[i].fObjectID === sub.snapid) { sub = null; isanyfound = true; break; }

         if (sub) {
            // remove painter which does not found in the list of snaps
            this.painters.splice(k--,1);
            sub.Cleanup(); // cleanup such painter
         }
      }

      if (!isanyfound) {
         var svg_p = this.svg_pad(this.this_pad_name);
         if (svg_p && !svg_p.empty())
            svg_p.property('mainpainter', null);
         for (var k=0;k<this.painters.length;++k)
            this.painters[k].Cleanup();
         this.painters = [];
      }

      var padpainter = this,
          prev_name = padpainter.CurrentPadName(padpainter.this_pad_name);

      padpainter.DrawNextSnap(snap.fPrimitives, 0, function() {
         padpainter.CurrentPadName(prev_name);
         call_back(padpainter);
      });
   }

   TPadPainter.prototype.CreateImage = function(format, call_back) {
      if (format=="svg") {
         JSROOT.CallBack(call_back, btoa(this.CreateSvg()));
      } else if ((format=="png") || (format=="jpeg")) {
         this.ProduceImage(true, 'any.' + format, function(can) {
            var res = can.toDataURL('image/' + format),
                separ = res.indexOf("base64,");
            JSROOT.CallBack(call_back, (separ>0) ? res.substr(separ+7) : "");
         });
      } else {
         JSROOT.CallBack(call_back, "");
      }
   }

   TPadPainter.prototype.GetAllRanges = function() {
      var res = "";

      if (this.snapid) {
         res = this.GetPadRanges();
         if (res) res = "id=" + this.snapid + ":" + res + ";";
      }

      for (var k=0;k<this.painters.length;++k)
         if (typeof this.painters[k].GetAllRanges == "function")
            res += this.painters[k].GetAllRanges();

      return res;
   }

   TPadPainter.prototype.GetPadRanges = function() {
      // function returns actual ranges in the pad, which can be applied to the server
      var main = this.main_painter(true, this.this_pad_name),
          p = this.svg_pad(this.this_pad_name),
          f = this.svg_frame(this.this_pad_name);

      if (!main) return "";

      var res1 = main.scale_xmin + ":" +
                 main.scale_xmax + ":" +
                 main.scale_ymin + ":" +
                 main.scale_ymax;

      if (f.empty() || p.empty()) return res1 + ":" + res1;

      var res2 = "";

      // calculate user range for full pad
      var same = function(x) { return x; },
          exp10 = function(x) { return Math.pow(10, x); };

      var func = main.logx ? JSROOT.log10 : same,
          func2 = main.logx ? exp10 : same;

      var k = (func(main.scale_xmax) - func(main.scale_xmin))/f.property("draw_width");
      var x1 = func(main.scale_xmin) - k*f.property("draw_x");
      var x2 = x1 + k*p.property("draw_width");
      res2 += func2(x1) + ":" + func2(x2);

      func = main.logy ? JSROOT.log10 : same;
      func2 = main.logy ? exp10 : same;

      var k = (func(main.scale_ymax) - func(main.scale_ymin))/f.property("draw_height");
      var y2 = func(main.scale_ymax) + k*f.property("draw_y");
      var y1 = y2 - k*p.property("draw_height");
      res2 += ":" + func2(y1) + ":" + func2(y2);

      return res1 + ":" + res2;
   }

   TPadPainter.prototype.ItemContextMenu = function(name) {
       var rrr = this.svg_pad(this.this_pad_name).node().getBoundingClientRect();
       var evnt = { clientX: rrr.left+10, clientY: rrr.top + 10 };

       // use timeout to avoid conflict with mouse click and automatic menu close
       if (name=="pad")
          return setTimeout(this.ShowContextMenu.bind(this, evnt), 50);

       var selp = null, selkind;

       switch(name) {
          case "xaxis":
          case "yaxis":
          case "zaxis":
             selp = this.main_painter();
             selkind = name[0];
             break;
          case "frame":
             selp = this.frame_painter();
             break;
          default: {
             var indx = parseInt(name);
             if (!isNaN(indx)) selp = this.painters[indx];
          }
       }

       if (!selp || (typeof selp.FillContextMenu !== 'function')) return;

       JSROOT.Painter.createMenu(selp, function(menu) {
          if (selp.FillContextMenu(menu,selkind))
             setTimeout(menu.show.bind(menu, evnt), 50);
       });
   }

   TPadPainter.prototype.CreateSvg = function() {
      var main = this.svg_canvas(),
          svg = main.html();

      svg = svg.replace(/url\(\&quot\;\#(\w+)\&quot\;\)/g,"url(#$1)")        // decode all URL
               .replace(/ class=\"\w*\"/g,"")                                // remove all classes
               .replace(/<g transform=\"translate\(\d+\,\d+\)\"><\/g>/g,"")  // remove all empty groups with transform
               .replace(/<g><\/g>/g,"");                                     // remove all empty groups

      svg = '<svg xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none"' +
            ' viewBox="0 0 ' + main.property('draw_width') + ' ' + main.property('draw_height') + '"' +
            ' width="' + main.property('draw_width') + '"' +
            ' height="' + main.property('draw_height') + '">' + svg + '</svg>';

       return svg;
   }

   TPadPainter.prototype.SaveAsPng = function(full_canvas, filename) {
      if (!filename) {
         filename = this.this_pad_name;
         if (filename.length === 0) filename = this.iscan ? "canvas" : "pad";
         filename += ".png";
      }
      this.ProduceImage(full_canvas, filename)
   }

   TPadPainter.prototype.ProduceImage = function(full_canvas, filename, call_back) {

      var elem = full_canvas ? this.svg_canvas() : this.svg_pad(this.this_pad_name);

      if (elem.empty()) return;

      var painter = full_canvas ? this.pad_painter() : this;

      document.body.style.cursor = 'wait';

      painter.ForEachPainterInPad(function(pp) {

         var main = pp.main_painter(true, pp.this_pad_name);
         if (!main || (typeof main.Render3D !== 'function')) return;

         var can3d = main.access_3d_kind();
         if ((can3d !== 1) && (can3d !== 2)) return;

         var sz = main.size_for_3d(3); // get size for SVG canvas

         var svg3d = main.Render3D(-1111); // render SVG

         //var rrr = new THREE.SVGRenderer({precision:0});
         //rrr.setSize(sz.width, sz.height);
         //rrr.render(main.scene, main.camera);

          main
              .insert("g",".primitives_layer")             // create special group
              .attr("class","temp_saveaspng")
              .attr("transform", "translate(" + sz.x + "," + sz.y + ")")
              .node().appendChild(svg3d);      // add code
      }, true);

//      if (((can3d === 1) || (can3d === 2)) && main && main.Render3D) {
           // this was saving of image buffer from 3D render
//         var canvas = main.renderer.domElement;
//         main.Render3D(0); // WebGL clears buffers, therefore we should render scene and convert immediately
//         var dataUrl = canvas.toDataURL("image/png");
//         dataUrl.replace("image/png", "image/octet-stream");
//         var link = document.createElement('a');
//         if (typeof link.download === 'string') {
//            document.body.appendChild(link); //Firefox requires the link to be in the body
//            link.download = filename;
//            link.href = dataUrl;
//            link.click();
//            document.body.removeChild(link); //remove the link when done
//         }
//      } else


      var options = { name: filename, removeClass: "btns_layer" };
      if (call_back) options.result = "canvas";

      JSROOT.saveSvgAsPng(elem.node(), options, function(res) {

         if (res===null) console.warn('problem when produce image');

         elem.selectAll(".temp_saveaspng").remove();

         document.body.style.cursor = 'auto';

         if (call_back) JSROOT.CallBack(call_back, res);
      });

   }

   TPadPainter.prototype.PadButtonClick = function(funcname) {

      if (funcname == "CanvasSnapShot") return this.SaveAsPng(true);

      if (funcname == "EnlargePad") return this.EnlargePad();

      if (funcname == "PadSnapShot") return this.SaveAsPng(false);

      if (funcname == "PadContextMenus") {

         d3.event.preventDefault();
         d3.event.stopPropagation();

         if (JSROOT.Painter.closeMenu()) return;

         var pthis = this, evnt = d3.event;

         JSROOT.Painter.createMenu(pthis, function(menu) {
            menu.add("header:Menus");

            if (pthis.iscan)
               menu.add("Canvas", "pad", pthis.ItemContextMenu);
            else
               menu.add("Pad", "pad", pthis.ItemContextMenu);

            if (pthis.frame_painter())
               menu.add("Frame", "frame", pthis.ItemContextMenu);

            var main = pthis.main_painter();

            if (main) {
               menu.add("X axis", "xaxis", pthis.ItemContextMenu);
               menu.add("Y axis", "yaxis", pthis.ItemContextMenu);
               if ((typeof main.Dimension === 'function') && (main.Dimension() > 1))
                  menu.add("Z axis", "zaxis", pthis.ItemContextMenu);
            }

            if (pthis.painters && (pthis.painters.length>0)) {
               menu.add("separator");
               var shown = [];
               for (var n=0;n<pthis.painters.length;++n) {
                  var pp = pthis.painters[n];
                  var obj = pp ? pp.GetObject() : null;
                  if (!obj || (shown.indexOf(obj)>=0)) continue;

                  var name = ('_typename' in obj) ? (obj._typename + "::") : "";
                  if ('fName' in obj) name += obj.fName;
                  if (name.length==0) name = "item" + n;
                  menu.add(name, n, pthis.ItemContextMenu);
               }
            }

            menu.show(evnt);
         });

         return;
      }

      // click automatically goes to all sub-pads
      // if any painter indicates that processing completed, it returns true
      var done = false;

      for (var i = 0; i < this.painters.length; ++i) {
         var pp = this.painters[i];

         if (typeof pp.PadButtonClick == 'function')
            pp.PadButtonClick(funcname);

         if (!done && (typeof pp.ButtonClick == 'function'))
            done = pp.ButtonClick(funcname);
      }
   }

   TPadPainter.prototype.FindButton = function(keyname) {
      var group = this.svg_layer("btns_layer", this.this_pad_name);
      if (group.empty()) return;

      var found_func = "";

      group.selectAll("svg").each(function() {
         if (d3.select(this).attr("key") === keyname)
            found_func = d3.select(this).attr("name");
      });

      return found_func;

   }

   TPadPainter.prototype.toggleButtonsVisibility = function(action) {
      var group = this.svg_layer("btns_layer", this.this_pad_name),
          btn = group.select("[name='Toggle']");

      if (btn.empty()) return;

      var state = btn.property('buttons_state');

      if (btn.property('timout_handler')) {
         if (action!=='timeout') clearTimeout(btn.property('timout_handler'));
         btn.property('timout_handler', null);
      }

      var is_visible = false;
      switch(action) {
         case 'enable': is_visible = true; break;
         case 'enterbtn': return; // do nothing, just cleanup timeout
         case 'timeout': is_visible = false; break;
         case 'toggle':
            state = !state;
            btn.property('buttons_state', state);
            is_visible = state;
            break;
         case 'disable':
         case 'leavebtn':
            if (!state) btn.property('timout_handler', setTimeout(this.toggleButtonsVisibility.bind(this,'timeout'), 500));
            return;
      }

      group.selectAll('svg').each(function() {
         if (this===btn.node()) return;
         d3.select(this).style('display', is_visible ? "" : "none");
      });
   }

   TPadPainter.prototype.AddButton = function(btn, tooltip, funcname, keyname) {

      // do not add buttons when not allowed
      if (!JSROOT.gStyle.ToolBar) return;

      var group = this.svg_layer("btns_layer", this.this_pad_name);
      if (group.empty()) return;

      // avoid buttons with duplicate names
      if (!group.select("[name='" + funcname + "']").empty()) return;

      var iscan = this.iscan || !this.has_canvas, ctrl;

      var x = group.property("nextx");
      if (!x) {
         ctrl = JSROOT.ToolbarIcons.CreateSVG(group, JSROOT.ToolbarIcons.rect, this.ButtonSize(), "Toggle tool buttons");

         ctrl.attr("name", "Toggle").attr("x", 0).attr("y", 0).attr("normalx",0)
             .property("buttons_state", (JSROOT.gStyle.ToolBar!=='popup'))
             .on("click", this.toggleButtonsVisibility.bind(this, 'toggle'))
             .on("mouseenter", this.toggleButtonsVisibility.bind(this, 'enable'))
             .on("mouseleave", this.toggleButtonsVisibility.bind(this, 'disable'));

         x = iscan ? this.ButtonSize(1.25) : 0;
      } else {
         ctrl = group.select("[name='Toggle']");
      }

      var svg = JSROOT.ToolbarIcons.CreateSVG(group, btn, this.ButtonSize(),
            tooltip + (iscan ? "" : (" on pad " + this.this_pad_name)) + (keyname ? " (keyshortcut " + keyname + ")" : ""));

      svg.attr("name", funcname).attr("x", x).attr("y", 0).attr("normalx",x)
         .style('display', (ctrl.property("buttons_state") ? '' : 'none'))
         .on("mouseenter", this.toggleButtonsVisibility.bind(this, 'enterbtn'))
         .on("mouseleave", this.toggleButtonsVisibility.bind(this, 'leavebtn'));

      if (keyname) svg.attr("key", keyname);

      svg.on("click", this.PadButtonClick.bind(this, funcname));

      group.property("nextx", x + this.ButtonSize(1.25));

      if (!iscan) {
         group.attr("transform","translate("+ (this.pad_width(this.this_pad_name) - group.property('nextx') - this.ButtonSize(1.25)) + "," + (this.pad_height(this.this_pad_name)-this.ButtonSize(1.25)) + ")");
         ctrl.attr("x", group.property('nextx'));
      }

      if (!iscan && (funcname.indexOf("Pad")!=0) && (this.pad_painter()!==this) && (funcname !== "EnlargePad"))
         this.pad_painter().AddButton(btn, tooltip, funcname);
   }

   TPadPainter.prototype.DrawingReady = function(res_painter) {

      var main = this.main_painter();

      if (main && main.mode3d && typeof main.Render3D == 'function') main.Render3D(-2222);

      TBasePainter.prototype.DrawingReady.call(this, res_painter);
   }

   TPadPainter.prototype.DecodeOptions = function(opt) {
      var pad = this.GetObject();
      if (!pad) return;

      var d = new JSROOT.DrawOptions(opt);

      if (d.check('WEBSOCKET')) this.OpenWebsocket();

      this.options = { GlobalColors: true, LocalColors: false, IgnorePalette: false, RotateFrame: false, FixFrame: false };

      if (d.check('NOCOLORS') || d.check('NOCOL')) this.options.GlobalColors = this.options.LocalColors = false;
      if (d.check('LCOLORS') || d.check('LCOL')) { this.options.GlobalColors = false; this.options.LocalColors = true; }
      if (d.check('NOPALETTE') || d.check('NOPAL')) this.options.IgnorePalette = true;
      if (d.check('ROTATE')) this.options.RotateFrame = true;
      if (d.check('FIXFRAME')) this.options.FixFrame = true;

      if (d.check('WHITE')) pad.fFillColor = 0;
      if (d.check('LOGX')) pad.fLogx = 1;
      if (d.check('LOGY')) pad.fLogy = 1;
      if (d.check('LOGZ')) pad.fLogz = 1;
      if (d.check('LOG')) pad.fLogx = pad.fLogy = pad.fLogz = 1;
      if (d.check('GRIDX')) pad.fGridx = 1;
      if (d.check('GRIDY')) pad.fGridy = 1;
      if (d.check('GRID')) pad.fGridx = pad.fGridy = 1;
      if (d.check('TICKX')) pad.fTickx = 1;
      if (d.check('TICKY')) pad.fTicky = 1;
      if (d.check('TICK')) pad.fTickx = pad.fTicky = 1;
   }

   Painter.drawPad = function(divid, pad, opt) {
      var painter = new TPadPainter(pad, false);
      painter.DecodeOptions(opt);

      painter.SetDivId(divid); // pad painter will be registered in the canvas painters list

      if (painter.svg_canvas().empty()) {
         painter.has_canvas = false;
         painter.this_pad_name = "";
      }

      painter.CreatePadSvg();

      if (painter.MatchObjectType("TPad") && (!painter.has_canvas || painter.HasObjectsToDraw())) {
         painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "PadSnapShot");

         if ((painter.has_canvas && painter.HasObjectsToDraw()) || painter.enlarge_main('verify'))
            painter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge pad", "EnlargePad");

         if (JSROOT.gStyle.ContextMenu)
            painter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");
      }

      // we select current pad, where all drawing is performed
      var prev_name = painter.has_canvas ? painter.CurrentPadName(painter.this_pad_name) : undefined;

      // flag used to prevent immediate pad redraw during first draw
      painter.DrawPrimitives(0, function() {
         // we restore previous pad name
         painter.CurrentPadName(prev_name);
         painter.DrawingReady();
      });

      return painter;
   }

   // ==========================================================================================

   function TCanvasPainter(canvas) {
      // used for online canvas painter
      TPadPainter.call(this, canvas, true);
      this._websocket = null;
   }

   TCanvasPainter.prototype = Object.create(TPadPainter.prototype);

   TCanvasPainter.prototype.ChangeLayout = function(layout_kind, call_back) {
      var current = this.get_layout_kind();
      if (current == layout_kind) return JSROOT.CallBack(call_back, true);

      var origin = this.select_main('origin'),
          sidebar = origin.select('.side_panel'),
          main = this.select_main(), lst = [];

      while (main.node().firstChild)
         lst.push(main.node().removeChild(main.node().firstChild));

      if (!sidebar.empty()) JSROOT.cleanup(sidebar.node());

      this.set_layout_kind("simple"); // restore defaults
      origin.html(""); // cleanup origin

      if (layout_kind == 'simple') {
         main = origin;
         for (var k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);
         this.set_layout_kind(layout_kind);
         // JSROOT.resize(main.node());
         return JSROOT.CallBack(call_back, true);
      }

      var pthis = this;

      JSROOT.AssertPrerequisites("jq2d", function() {

         var grid = new JSROOT.GridDisplay(origin.node(), layout_kind);

         if (layout_kind.indexOf("vert")==0) {
            main = d3.select(grid.GetFrame(0));
            sidebar = d3.select(grid.GetFrame(1));
         } else {
            main = d3.select(grid.GetFrame(1));
            sidebar = d3.select(grid.GetFrame(0));
         }

         main.classed("central_panel", true).style('position','relative');
         sidebar.classed("side_panel", true).style('position','relative');

         // now append all childs to the new main
         for (var k=0;k<lst.length;++k)
            main.node().appendChild(lst[k]);

         pthis.set_layout_kind(layout_kind, ".central_panel");

         JSROOT.CallBack(call_back, true);
      });
   }

   TCanvasPainter.prototype.ToggleProjection = function(kind, call_back) {
      delete this.proj_painter;

      if (kind) this.proj_painter = 1; // just indicator that drawing can be preformed

      if (this.use_openui && this.ShowUI5ProjectionArea)
         return this.ShowUI5ProjectionArea(kind, call_back);

      var layout = 'simple';

      if (kind == "X") layout = 'vert2_31'; else
      if (kind == "Y") layout = 'horiz2_13';

      this.ChangeLayout(layout, call_back);
   }

   TCanvasPainter.prototype.DrawProjection = function(kind,hist) {
      if (!this.proj_painter) return; // ignore drawing if projection not configured

      if (this.proj_painter === 1) {

         var canv = JSROOT.Create("TCanvas"), pthis = this, pad = this.root_pad(), main = this.main_painter(), drawopt;

         if (kind == "X") {
            canv.fLeftMargin = pad.fLeftMargin;
            canv.fRightMargin = pad.fRightMargin;
            canv.fLogx = main.logx ? 1 : 0;
            canv.fUxmin = main.logx ? JSROOT.log10(main.scale_xmin) : main.scale_xmin;
            canv.fUxmax = main.logx ? JSROOT.log10(main.scale_xmax) : main.scale_xmax;
            drawopt = "fixframe";
         } else {
            canv.fBottomMargin = pad.fBottomMargin;
            canv.fTopMargin = pad.fTopMargin;
            canv.fLogx = main.logy ? 1 : 0;
            canv.fUxmin = main.logy ? JSROOT.log10(main.scale_ymin) : main.scale_ymin;
            canv.fUxmax = main.logy ? JSROOT.log10(main.scale_ymax) : main.scale_ymax;
            drawopt = "rotate";
         }

         canv.fPrimitives.Add(hist, "hist");

         if (this.use_openui && this.DrawInUI5ProjectionArea ) {
            // copy frame attributes
            this.DrawInUI5ProjectionArea(canv, drawopt, function(painter) { pthis.proj_painter = painter; })
         } else {
            this.DrawInSidePanel(canv, drawopt, function(painter) { pthis.proj_painter = painter; })
         }
      } else {
         var hp = this.proj_painter.main_painter();
         if (hp) hp.UpdateObject(hist, "hist");
         this.proj_painter.RedrawPad();
      }
   }


   TCanvasPainter.prototype.DrawInSidePanel = function(canv, opt, call_back) {
      var side = this.select_main('origin').select(".side_panel");
      if (side.empty()) return JSROOT.CallBack(call_back, null);
      JSROOT.draw(side.node(), canv, opt, call_back);
   }


   TCanvasPainter.prototype.ShowMessage = function(msg) {
      JSROOT.progress(msg, 7000);
   }

   /// function called when canvas menu item Save is called
   TCanvasPainter.prototype.SaveCanvasAsFile = function(fname) {
      var pthis = this, pnt = fname.indexOf(".");
      this.CreateImage(fname.substr(pnt+1), function(res) {
         pthis.SendWebsocket("SAVE:" + fname + ":" + res);
      })
   }

   TCanvasPainter.prototype.WindowBeforeUnloadHanlder = function() {
      // when window closed, close socket
      this.CloseWebsocket(true);
   }

   TCanvasPainter.prototype.SendWebsocket = function(msg, chid) {
      if (this._websocket)
         this._websocket.Send(msg, chid);
   }

   TCanvasPainter.prototype.CloseWebsocket = function(force) {
      if (this._websocket) {
         this._websocket.Close(force);
         this._websocket.Cleanup();
         delete this._websocket;
      }
   }

   TCanvasPainter.prototype.OpenWebsocket = function(socket_kind) {
      // create websocket for current object (canvas)
      // via websocket one recieved many extra information

      this.CloseWebsocket();

      this._websocket = new JSROOT.WebWindowHandle(socket_kind);
      this._websocket.SetReceiver(this);
      this._websocket.Connect();
   }

   TCanvasPainter.prototype.OnWebsocketOpened = function(handle) {
      // indicate that we are ready to recieve any following commands
   }

   TCanvasPainter.prototype.OnWebsocketClosed = function(handle) {
      if (window) window.close(); // close window when socket disapper
   }

   TCanvasPainter.prototype.OnWebsocketMsg = function(handle, msg) {

      if (msg == "CLOSE") {
         this.OnWebsocketClosed();
         this.CloseWebsocket(true);
      } else if (msg.substr(0,5)=='SNAP:') {
         msg = msg.substr(5);
         var p1 = msg.indexOf(":"),
             snapid = msg.substr(0,p1),
             snap = JSROOT.parse(msg.substr(p1+1));
         this.RedrawPadSnap(snap, function() {
            handle.Send("SNAPDONE:" + snapid); // send ready message back when drawing completed
         });
      } else if (msg.substr(0,6)=='SNAP6:') {
         // This is snapshot, produced with ROOT6, handled slighly different

         this.root6_canvas = true; // indicate that drawing of root6 canvas is peformed
         // if (!this.snap_cnt) this.snap_cnt = 1; else this.snap_cnt++;

         msg = msg.substr(6);
         var p1 = msg.indexOf(":"),
             snapid = msg.substr(0,p1),
             snap = JSROOT.parse(msg.substr(p1+1)),
             pthis = this;

         // console.log('Get SNAP6', this.snap_cnt);

         this.RedrawPadSnap(snap, function() {
            // console.log('Complete SNAP6', pthis.snap_cnt);
            pthis.CompeteCanvasSnapDrawing();
            var ranges = pthis.GetAllRanges();
            if (ranges) ranges = ":" + ranges;
            // if (ranges) console.log("ranges: " + ranges);
            handle.Send("RREADY:" + snapid + ranges); // send ready message back when drawing completed
         });

      } else if (msg.substr(0,4)=='JSON') {
         var obj = JSROOT.parse(msg.substr(4));
         // console.log("get JSON ", msg.length-4, obj._typename);
         this.RedrawObject(obj);

      } else if (msg.substr(0,5)=='MENU:') {
         // this is menu with exact identifier for object
         msg = msg.substr(5);
         var p1 = msg.indexOf(":"),
             menuid = msg.substr(0,p1),
             lst = JSROOT.parse(msg.substr(p1+1));
         // console.log("get MENUS ", typeof lst, 'nitems', lst.length, msg.length-4);
         if (typeof this._getmenu_callback == 'function')
            this._getmenu_callback(lst, menuid);
      } else if (msg.substr(0,4)=='CMD:') {
         msg = msg.substr(4);
         var p1 = msg.indexOf(":"),
             cmdid = msg.substr(0,p1),
             cmd = msg.substr(p1+1),
             reply = "REPLY:" + cmdid + ":";
         if ((cmd == "SVG") || (cmd == "PNG") || (cmd == "JPEG")) {
            this.CreateImage(cmd.toLowerCase(), function(res) {
               handle.Send(reply + res);
            });
         } else {
            console.log('Unrecognized command ' + cmd);
            handle.Send(reply);
         }
      } else if ((msg.substr(0,7)=='DXPROJ:') || (msg.substr(0,7)=='DYPROJ:')) {
         var kind = msg[1],
             hist = JSROOT.parse(msg.substr(7));
         this.DrawProjection(kind, hist);
      } else if (msg.substr(0,5)=='SHOW:') {
         var that = msg.substr(5),
             on = that[that.length-1] == '1';
         this.ShowSection(that.substr(0,that.length-2), on);
      } else {
         console.log("unrecognized msg " + msg);
      }
   }

   TCanvasPainter.prototype.ShowSection = function(that, on) {
      switch(that) {
         case "Menu": break;
         case "StatusBar": break;
         case "Editor": break;
         case "ToolBar": break;
         case "ToolTips": this.SetTooltipAllowed(on); break;
      }
   }

   JSROOT.TCanvasStatusBits = {
      kShowEventStatus  : JSROOT.BIT(15),
      kAutoExec         : JSROOT.BIT(16),
      kMenuBar          : JSROOT.BIT(17),
      kShowToolBar      : JSROOT.BIT(18),
      kShowEditor       : JSROOT.BIT(19),
      kMoveOpaque       : JSROOT.BIT(20),
      kResizeOpaque     : JSROOT.BIT(21),
      kIsGrayscale      : JSROOT.BIT(22),
      kShowToolTips     : JSROOT.BIT(23)
   };

   TCanvasPainter.prototype.CompeteCanvasSnapDrawing = function() {
      if (!this.pad) return;

      if (document) document.title = this.pad.fTitle;

      if (this._all_sections_showed) return;
      this._all_sections_showed = true;
      this.ShowSection("Menu", this.pad.TestBit(JSROOT.TCanvasStatusBits.kMenuBar));
      this.ShowSection("StatusBar", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowEventStatus));
      this.ShowSection("ToolBar", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowToolBar));
      this.ShowSection("Editor", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowEditor));
      this.ShowSection("ToolTips", this.pad.TestBit(JSROOT.TCanvasStatusBits.kShowToolTips));
   }

   TCanvasPainter.prototype.HasEventStatus = function() {
      return this.has_event_status;
   }

   Painter.drawCanvas = function(divid, can, opt) {
      var nocanvas = (can===null);
      if (nocanvas) can = JSROOT.Create("TCanvas");

      var painter = new TCanvasPainter(can);
      painter.DecodeOptions(opt);
      painter.normal_canvas = !nocanvas;

      painter.SetDivId(divid, -1); // just assign id
      painter.CheckSpecialsInPrimitives(can);
      painter.CreateCanvasSvg(0);
      painter.SetDivId(divid);  // now add to painters list

      painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "CanvasSnapShot", "Ctrl PrintScreen");
      if (JSROOT.gStyle.ContextMenu)
         painter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");

      if (painter.enlarge_main('verify'))
         painter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge canvas", "EnlargePad");

      if (nocanvas && opt.indexOf("noframe") < 0)
         JSROOT.Painter.drawFrame(divid, null);

      painter.DrawPrimitives(0, function() { painter.DrawingReady(); });
      return painter;
   }

   Painter.drawPadSnapshot = function(divid, snap, opt) {
      // just for debugging without running web canvas

      var can = JSROOT.Create("TCanvas");

      var painter = new TCanvasPainter(can);
      painter.normal_canvas = false;

      painter.SetDivId(divid, -1); // just assign id

      painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "CanvasSnapShot", "Ctrl PrintScreen");
      if (JSROOT.gStyle.ContextMenu)
         painter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");

      if (painter.enlarge_main('verify'))
         painter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge canvas", "EnlargePad");

      // JSROOT.Painter.drawFrame(divid, null);

      painter.RedrawPadSnap(snap, function() { painter.DrawingReady(); });

      return painter;
   }

   // ================= painter of raw text ========================================


   Painter.drawRawText = function(divid, txt, opt) {

      var painter = new TBasePainter();
      painter.txt = txt;
      painter.SetDivId(divid);

      painter.RedrawObject = function(obj) {
         this.txt = obj;
         this.Draw();
         return true;
      }

      painter.Draw = function() {
         var txt = this.txt.value;
         if (typeof txt != 'string') txt = "<undefined>";

         var mathjax = this.txt.mathjax || (JSROOT.gStyle.Latex == 4);

         if (!mathjax && !('as_is' in this.txt)) {
            var arr = txt.split("\n"); txt = "";
            for (var i = 0; i < arr.length; ++i)
               txt += "<pre>" + arr[i] + "</pre>";
         }

         var frame = this.select_main(),
              main = frame.select("div");
         if (main.empty())
            main = frame.append("div").style('max-width','100%').style('max-height','100%').style('overflow','auto');
         main.html(txt);

         // (re) set painter to first child element
         this.SetDivId(this.divid);

         if (mathjax)
            JSROOT.AssertPrerequisites('mathjax', function() {
               MathJax.Hub.Typeset(frame.node());
            });
      }

      painter.Draw();
      return painter.DrawingReady();
   }

   // =========================================================================

   JSROOT.RegisterForResize = function(handle, delay) {
      // function used to react on browser window resize event
      // While many resize events could come in short time,
      // resize will be handled with delay after last resize event
      // handle can be function or object with CheckResize function
      // one could specify delay after which resize event will be handled

      if (!handle) return;

      var myInterval = null, myDelay = delay ? delay : 300;

      if (myDelay < 20) myDelay = 20;

      function ResizeTimer() {
         myInterval = null;

         document.body.style.cursor = 'wait';
         if (typeof handle == 'function') handle(); else
         if ((typeof handle == 'object') && (typeof handle.CheckResize == 'function')) handle.CheckResize(); else
         if (typeof handle == 'string') {
            var node = d3.select('#'+handle);
            if (!node.empty()) {
               var mdi = node.property('mdi');
               if (mdi) {
                  mdi.CheckMDIResize();
               } else {
                  JSROOT.resize(node.node());
               }
            }
         }
         document.body.style.cursor = 'auto';
      }

      function ProcessResize() {
         if (myInterval !== null) clearTimeout(myInterval);
         myInterval = setTimeout(ResizeTimer, myDelay);
      }

      window.addEventListener('resize', ProcessResize);
   }

   JSROOT.addDrawFunc({ name: "TCanvas", icon: "img_canvas", func: JSROOT.Painter.drawCanvas, opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" });
   JSROOT.addDrawFunc({ name: "TPad", icon: "img_canvas", func: JSROOT.Painter.drawPad, opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" });
   JSROOT.addDrawFunc({ name: "TSlider", icon: "img_canvas", func: JSROOT.Painter.drawPad });
   JSROOT.addDrawFunc({ name: "TFrame", icon: "img_frame", func: JSROOT.Painter.drawFrame });
   JSROOT.addDrawFunc({ name: "TPaveText", icon: "img_pavetext", prereq: "hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TPavesText", icon: "img_pavetext", prereq: "hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TPaveStats", icon: "img_pavetext", prereq: "hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TPaveLabel", icon: "img_pavelabel", prereq: "hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TDiamond", icon: "img_pavelabel", prereq: "hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TLatex", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: "TMathText", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: "TText", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: /^TH1/, icon: "img_histo1d", prereq: "hist", func: "JSROOT.Painter.drawHistogram1D", opt:";hist;P;P0;E;E1;E2;E3;E4;E1X0;L;LF2;B;B1;A;TEXT;LEGO;same", ctrl: "l" });
   JSROOT.addDrawFunc({ name: "TProfile", icon: "img_profile", prereq: "hist", func: "JSROOT.Painter.drawHistogram1D", opt:";E0;E1;E2;p;AH;hist"});
   JSROOT.addDrawFunc({ name: "TH2Poly", icon: "img_histo2d", prereq: "hist", func: "JSROOT.Painter.drawHistogram2D", opt:";COL;COL0;COLZ;LCOL;LCOL0;LCOLZ;LEGO;same", expand_item: "fBins", theonly: true });
   JSROOT.addDrawFunc({ name: "TH2PolyBin", icon: "img_histo2d", draw_field: "fPoly" });
   JSROOT.addDrawFunc({ name: /^TH2/, icon: "img_histo2d", prereq: "hist", func: "JSROOT.Painter.drawHistogram2D", opt:";COL;COLZ;COL0;COL1;COL0Z;COL1Z;COLA;BOX;BOX1;SCAT;TEXT;CONT;CONT1;CONT2;CONT3;CONT4;ARR;SURF;SURF1;SURF2;SURF4;SURF6;E;A;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same", ctrl: "colz" });
   JSROOT.addDrawFunc({ name: "TProfile2D", sameas: "TH2" });
   JSROOT.addDrawFunc({ name: /^TH3/, icon: 'img_histo3d', prereq: "hist3d", func: "JSROOT.Painter.drawHistogram3D", opt:";SCAT;BOX;BOX2;BOX3;GLBOX1;GLBOX2;GLCOL" });
   JSROOT.addDrawFunc({ name: "THStack", icon: "img_histo1d", prereq: "hist", func: "JSROOT.Painter.drawHStack", expand_item: "fHists", opt: "PFC;PLC" });
   JSROOT.addDrawFunc({ name: "TPolyMarker3D", icon: 'img_histo3d', prereq: "hist3d", func: "JSROOT.Painter.drawPolyMarker3D" });
   JSROOT.addDrawFunc({ name: "TPolyLine3D", icon: 'img_graph', prereq: "3d", func: "JSROOT.Painter.drawPolyLine3D", direct: true });
   JSROOT.addDrawFunc({ name: "TGraphStruct" });
   JSROOT.addDrawFunc({ name: "TGraphNode" });
   JSROOT.addDrawFunc({ name: "TGraphEdge" });
   JSROOT.addDrawFunc({ name: "TGraphTime", icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraphTime", opt: "once;repeat;first", theonly: true });
   JSROOT.addDrawFunc({ name: "TGraph2D", icon:"img_graph", prereq: "hist3d", func: "JSROOT.Painter.drawGraph2D", opt: ";P;PCOL", theonly: true });
   JSROOT.addDrawFunc({ name: "TGraph2DErrors", icon:"img_graph", prereq: "hist3d", func: "JSROOT.Painter.drawGraph2D", opt: ";P;PCOL;ERR", theonly: true });
   JSROOT.addDrawFunc({ name: "TGraphPolargram", icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraphPolargram", theonly: true });
   JSROOT.addDrawFunc({ name: "TGraphPolar", icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraphPolar", opt: ";F;L;P;PE", theonly: true });
   JSROOT.addDrawFunc({ name: /^TGraph/, icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraph", opt: ";L;P" });
   JSROOT.addDrawFunc({ name: "TEfficiency", icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawEfficiency", opt: ";AP" });
   JSROOT.addDrawFunc({ name: "TCutG", sameas: "TGraph" });
   JSROOT.addDrawFunc({ name: /^RooHist/, sameas: "TGraph" });
   JSROOT.addDrawFunc({ name: /^RooCurve/, sameas: "TGraph" });
   JSROOT.addDrawFunc({ name: "RooPlot", icon: "img_canvas", prereq: "more2d", func: "JSROOT.Painter.drawRooPlot" });
   JSROOT.addDrawFunc({ name: "TMultiGraph", icon: "img_mgraph", prereq: "more2d", func: "JSROOT.Painter.drawMultiGraph", expand_item: "fGraphs" });
   JSROOT.addDrawFunc({ name: "TStreamerInfoList", icon: 'img_question', prereq: "hierarchy",  func: "JSROOT.Painter.drawStreamerInfo" });
   JSROOT.addDrawFunc({ name: "TPaletteAxis", icon: "img_colz", prereq: "hist", func: "JSROOT.Painter.drawPaletteAxis" });
   JSROOT.addDrawFunc({ name: "TWebPainting", icon: "img_graph", prereq: "more2d", func: "JSROOT.Painter.drawWebPainting" });
   JSROOT.addDrawFunc({ name: "TPadWebSnapshot", icon: "img_canvas", func: JSROOT.Painter.drawPadSnapshot });
   JSROOT.addDrawFunc({ name: "kind:Text", icon: "img_text", func: JSROOT.Painter.drawRawText });
   JSROOT.addDrawFunc({ name: "TF1", icon: "img_tf1", prereq: "math;more2d", func: "JSROOT.Painter.drawFunction" });
   JSROOT.addDrawFunc({ name: "TF2", icon: "img_tf2", prereq: "math;hist", func: "JSROOT.Painter.drawTF2" });
   JSROOT.addDrawFunc({ name: "TSpline3", icon: "img_tf1", prereq: "more2d", func: "JSROOT.Painter.drawSpline" });
   JSROOT.addDrawFunc({ name: "TSpline5", icon: "img_tf1", prereq: "more2d", func: "JSROOT.Painter.drawSpline" });
   JSROOT.addDrawFunc({ name: "TEllipse", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawEllipse", direct: true });
   JSROOT.addDrawFunc({ name: "TArc", sameas: 'TEllipse' });
   JSROOT.addDrawFunc({ name: "TCrown", sameas: 'TEllipse' });
   JSROOT.addDrawFunc({ name: "TPie", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawPie", direct: true });
   JSROOT.addDrawFunc({ name: "TLine", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawLine", direct: true });
   JSROOT.addDrawFunc({ name: "TArrow", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawArrow", direct: true });
   JSROOT.addDrawFunc({ name: "TPolyLine", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawPolyLine", direct: true });
   JSROOT.addDrawFunc({ name: "TCurlyLine", sameas: 'TPolyLine' });
   JSROOT.addDrawFunc({ name: "TCurlyArc", sameas: 'TPolyLine' });
   JSROOT.addDrawFunc({ name: "TGaxis", icon: "img_graph", prereq: "hist", func: "JSROOT.Painter.drawGaxis" });
   JSROOT.addDrawFunc({ name: "TLegend", icon: "img_pavelabel", prereq: "hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TWbox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TSliderBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TAxis3D", prereq: "hist3d", func: "JSROOT.Painter.drawAxis3D" });
   JSROOT.addDrawFunc({ name: "TMarker", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawMarker", direct: true });
   JSROOT.addDrawFunc({ name: "TPolyMarker", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawPolyMarker", direct: true });
   JSROOT.addDrawFunc({ name: "TGeoVolume", icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", expand: "JSROOT.GEO.expandObject", opt:";more;all;count;projx;projz;dflt", ctrl: "dflt" });
   JSROOT.addDrawFunc({ name: "TEveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;dflt", ctrl: "dflt"  });
   JSROOT.addDrawFunc({ name: "TGeoManager", icon: 'img_histo3d', prereq: "geom", expand: "JSROOT.GEO.expandObject", func: "JSROOT.Painter.drawGeoObject", opt: ";more;all;count;projx;projz;dflt", dflt: "expand", ctrl: "dflt" });
   JSROOT.addDrawFunc({ name: /^TGeo/, icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", opt: ";more;all;axis;compa;count;projx;projz;dflt", ctrl: "dflt" });
   // these are not draw functions, but provide extra info about correspondent classes
   JSROOT.addDrawFunc({ name: "kind:Command", icon: "img_execute", execute: true });
   JSROOT.addDrawFunc({ name: "TFolder", icon: "img_folder", icon2: "img_folderopen", noinspect: true, prereq: "hierarchy", expand: "JSROOT.Painter.FolderHierarchy" });
   JSROOT.addDrawFunc({ name: "TTask", icon: "img_task", prereq: "hierarchy", expand: "JSROOT.Painter.TaskHierarchy", for_derived: true });
   JSROOT.addDrawFunc({ name: "TTree", icon: "img_tree", prereq: "tree", expand: 'JSROOT.Painter.TreeHierarchy', func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" });
   JSROOT.addDrawFunc({ name: "TNtuple", icon: "img_tree", prereq: "tree", expand: 'JSROOT.Painter.TreeHierarchy', func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" });
   JSROOT.addDrawFunc({ name: "TNtupleD", icon: "img_tree", prereq: "tree", expand: 'JSROOT.Painter.TreeHierarchy', func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" });
   JSROOT.addDrawFunc({ name: "TBranchFunc", icon: "img_leaf_method", prereq: "tree", func: 'JSROOT.Painter.drawTree', opt: ";dump", noinspect: true });
   JSROOT.addDrawFunc({ name: /^TBranch/, icon: "img_branch", prereq: "tree", func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: ";dump", ctrl: "dump", shift: "inspect", ignore_online: true });
   JSROOT.addDrawFunc({ name: /^TLeaf/, icon: "img_leaf", prereq: "tree", noexpand: true, func: 'JSROOT.Painter.drawTree', opt: ";dump", ctrl: "dump", ignore_online: true });
   JSROOT.addDrawFunc({ name: "TList", icon: "img_list", prereq: "hierarchy", func: "JSROOT.Painter.drawList", expand: "JSROOT.Painter.ListHierarchy", dflt: "expand" });
   JSROOT.addDrawFunc({ name: "THashList", sameas: "TList" });
   JSROOT.addDrawFunc({ name: "TObjArray", sameas: "TList" });
   JSROOT.addDrawFunc({ name: "TClonesArray", sameas: "TList" });
   JSROOT.addDrawFunc({ name: "TMap", sameas: "TList" });
   JSROOT.addDrawFunc({ name: "TColor", icon: "img_color" });
   JSROOT.addDrawFunc({ name: "TFile", icon: "img_file", noinspect:true });
   JSROOT.addDrawFunc({ name: "TMemFile", icon: "img_file", noinspect:true });
   JSROOT.addDrawFunc({ name: "TStyle", icon: "img_question", noexpand:true });
   JSROOT.addDrawFunc({ name: "Session", icon: "img_globe" });
   JSROOT.addDrawFunc({ name: "kind:TopFolder", icon: "img_base" });
   JSROOT.addDrawFunc({ name: "kind:Folder", icon: "img_folder", icon2: "img_folderopen", noinspect:true });

   JSROOT.addDrawFunc({ name: "ROOT::Experimental::TCanvas", icon: "img_canvas", prereq: "v7", func: "JSROOT.v7.drawCanvas", opt: "", expand_item: "fPrimitives" });


   JSROOT.getDrawHandle = function(kind, selector) {
      // return draw handle for specified item kind
      // kind could be ROOT.TH1I for ROOT classes or just
      // kind string like "Command" or "Text"
      // selector can be used to search for draw handle with specified option (string)
      // or just sequence id

      if (typeof kind != 'string') return null;
      if (selector === "") selector = null;

      var first = null;

      if ((selector === null) && (kind in JSROOT.DrawFuncs.cache))
         return JSROOT.DrawFuncs.cache[kind];

      var search = (kind.indexOf("ROOT.")==0) ? kind.substr(5) : "kind:"+kind;

      var counter = 0;
      for (var i=0; i < JSROOT.DrawFuncs.lst.length; ++i) {
         var h = JSROOT.DrawFuncs.lst[i];
         if (typeof h.name == "string") {
            if (h.name != search) continue;
         } else {
            if (!search.match(h.name)) continue;
         }

         if (h.sameas !== undefined)
            return JSROOT.getDrawHandle("ROOT."+h.sameas, selector);

         if (selector==null) {
            // store found handle in cache, can reuse later
            if (!(kind in JSROOT.DrawFuncs.cache)) JSROOT.DrawFuncs.cache[kind] = h;
            return h;
         } else
         if (typeof selector == 'string') {
            if (!first) first = h;
            // if drawoption specified, check it present in the list

            if (selector == "::expand") {
               if (('expand' in h) || ('expand_item' in h)) return h;
            } else
            if ('opt' in h) {
               var opts = h.opt.split(';');
               for (var j=0; j < opts.length; ++j) opts[j] = opts[j].toLowerCase();
               if (opts.indexOf(selector.toLowerCase())>=0) return h;
            }
         } else {
            if (selector === counter) return h;
         }
         ++counter;
      }

      return first;
   }

   JSROOT.addStreamerInfos = function(lst) {
      if (lst === null) return;

      function CheckBaseClasses(si, lvl) {
         if (si.fElements == null) return null;
         if (lvl>10) return null; // protect against recursion

         for (var j=0; j<si.fElements.arr.length; ++j) {
            // extract streamer info for each class member
            var element = si.fElements.arr[j];
            if (element.fTypeName !== 'BASE') continue;

            var handle = JSROOT.getDrawHandle("ROOT." + element.fName);
            if (handle && !handle.for_derived) handle = null;

            // now try find that base class of base in the list
            if (handle === null)
               for (var k=0;k<lst.arr.length; ++k)
                  if (lst.arr[k].fName === element.fName) {
                     handle = CheckBaseClasses(lst.arr[k], lvl+1);
                     break;
                  }

            if (handle && handle.for_derived) return handle;
         }
         return null;
      }

      for (var n=0;n<lst.arr.length;++n) {
         var si = lst.arr[n];
         if (JSROOT.getDrawHandle("ROOT." + si.fName) !== null) continue;

         var handle = CheckBaseClasses(si, 0);

         if (!handle) continue;

         var newhandle = JSROOT.extend({}, handle);
         // delete newhandle.for_derived; // should we disable?
         newhandle.name = si.fName;
         JSROOT.DrawFuncs.lst.push(newhandle);
      }
   }

   JSROOT.getDrawSettings = function(kind, selector) {
      var res = { opts: null, inspect: false, expand: false, draw: false, handle: null };
      if (typeof kind != 'string') return res;
      var allopts = null, isany = false, noinspect = false, canexpand = false;
      if (typeof selector !== 'string') selector = "";

      for (var cnt=0;cnt<1000;++cnt) {
         var h = JSROOT.getDrawHandle(kind, cnt);
         if (!h) break;
         if (!res.handle) res.handle = h;
         if (h.noinspect) noinspect = true;
         if (h.expand || h.expand_item || h.can_expand) canexpand = true;
         if (!('func' in h)) break;
         isany = true;
         if (! ('opt' in h)) continue;
         var opts = h.opt.split(';');
         for (var i = 0; i < opts.length; ++i) {
            opts[i] = opts[i].toLowerCase();
            if ((selector.indexOf('nosame')>=0) && (opts[i].indexOf('same')==0)) continue;

            if (res.opts===null) res.opts = [];
            if (res.opts.indexOf(opts[i])<0) res.opts.push(opts[i]);
         }
         if (h.theonly) break;
      }

      if (selector.indexOf('noinspect')>=0) noinspect = true;

      if (isany && (res.opts===null)) res.opts = [""];

      // if no any handle found, let inspect ROOT-based objects
      if (!isany && (kind.indexOf("ROOT.")==0) && !noinspect) res.opts = [];

      if (!noinspect && res.opts)
         res.opts.push("inspect");

      res.inspect = !noinspect;
      res.expand = canexpand;
      res.draw = res.opts && (res.opts.length>0);

      return res;
   }

   // returns array with supported draw options for the specified class
   JSROOT.getDrawOptions = function(kind, selector) {
      return JSROOT.getDrawSettings(kind).opts;
   }

   JSROOT.canDraw = function(classname) {
      return JSROOT.getDrawSettings("ROOT." + classname).opts !== null;
   }

   /** @fn JSROOT.draw(divid, obj, opt, callback)
    * Draw object in specified HTML element with given draw options  */
   JSROOT.draw = function(divid, obj, opt, drawcallback) {

      var isdirectdraw = true; // indicates if extra callbacks (via AssertPrerequisites) was invoked to process

      function completeDraw(painter) {
         var callbackfunc = null, ishandle = false;
         if (typeof drawcallback == 'function') callbackfunc = drawcallback; else
            if (drawcallback && (typeof drawcallback == 'object') && (typeof drawcallback.func=='function')) {
               callbackfunc = drawcallback.func;
               ishandle = true;
            }

         if (ishandle && isdirectdraw) {
            // if there is no painter or drawing is already completed, return directly
            if (!painter || painter._ready_called_) { drawcallback.completed = true; return painter; }
         }

         if (painter && drawcallback && (typeof painter.WhenReady == 'function'))
            painter.WhenReady(callbackfunc);
         else
            JSROOT.CallBack(callbackfunc, painter);
         return painter;
      }

      if ((obj===null) || (typeof obj !== 'object')) return completeDraw(null);

      if (opt == 'inspect') {
         var func = JSROOT.findFunction("JSROOT.Painter.drawInspector");
         if (func) return completeDraw(func(divid, obj));
         JSROOT.AssertPrerequisites("hierarchy", function() {
            completeDraw(JSROOT.Painter.drawInspector(divid, obj));
         });
         return null;
      }

      var handle = null, painter = null;
      if ('_typename' in obj) handle = JSROOT.getDrawHandle("ROOT." + obj._typename, opt);
      else if ('_kind' in obj) handle = JSROOT.getDrawHandle(obj._kind, opt);

      if (!handle) return completeDraw(null);

      if (handle.draw_field && obj[handle.draw_field])
         return JSROOT.draw(divid, obj[handle.draw_field], opt, drawcallback);

      if (!handle.func) return completeDraw(null);

      function performDraw() {
         if (handle.direct) {
            painter = new TObjectPainter(obj);
            painter.SetDivId(divid, 2);
            painter.Redraw = handle.func;
            painter._drawopt = opt;
            painter.Redraw();
            painter.DrawingReady();
         } else {
            painter = handle.func(divid, obj, opt);
         }

         return completeDraw(painter);
      }

      if (typeof handle.func == 'function') return performDraw();

      var funcname = "", prereq = "";
      if (typeof handle.func == 'object') {
         if ('func' in handle.func) funcname = handle.func.func;
         if ('script' in handle.func) prereq = "user:" + handle.func.script;
      } else
      if (typeof handle.func == 'string') {
         funcname = handle.func;
         if (('prereq' in handle) && (typeof handle.prereq == 'string')) prereq = handle.prereq;
         if (('script' in handle) && (typeof handle.script == 'string')) prereq += ";user:" + handle.script;
      }

      if (funcname.length === 0) return completeDraw(null);

      // try to find function without prerequisites
      var func = JSROOT.findFunction(funcname);
      if (func) {
          handle.func = func; // remember function once it is found
          return performDraw();
      }

      if (prereq.length === 0) return completeDraw(null);

      isdirectdraw = false;

      JSROOT.AssertPrerequisites(prereq, function() {
         var func = JSROOT.findFunction(funcname);
         if (!func) {
            alert('Fail to find function ' + funcname + ' after loading ' + prereq);
            return completeDraw(null);
         }

         handle.func = func; // remember function once it found

         performDraw();
      });

      return painter;
   }

   /** @fn JSROOT.redraw(divid, obj, opt)
    * Redraw object in specified HTML element with given draw options
    * If drawing was not exists, it will be performed with JSROOT.draw.
    * If drawing was already done, that content will be updated */

   JSROOT.redraw = function(divid, obj, opt, callback) {
      if (!obj) return JSROOT.CallBack(callback, null);

      var dummy = new TObjectPainter();
      dummy.SetDivId(divid, -1);
      var can_painter = dummy.pad_painter();

      var handle = null;
      if (obj._typename) handle = JSROOT.getDrawHandle("ROOT." + obj._typename);
      if (handle && handle.draw_field && obj[handle.draw_field])
         obj = obj[handle.draw_field];

      if (can_painter) {
         if (obj._typename === "TCanvas") {
            can_painter.RedrawObject(obj);
            JSROOT.CallBack(callback, can_painter);
            return can_painter;
         }

         for (var i = 0; i < can_painter.painters.length; ++i) {
            var painter = can_painter.painters[i];
            if (painter.MatchObjectType(obj._typename))
               if (painter.UpdateObject(obj)) {
                  can_painter.RedrawPad();
                  JSROOT.CallBack(callback, painter);
                  return painter;
               }
         }
      }

      if (can_painter)
         JSROOT.console("Cannot find painter to update object of type " + obj._typename);

      JSROOT.cleanup(divid);

      return JSROOT.draw(divid, obj, opt, callback);
   }

   /** @fn JSROOT.MakeSVG(args, callback)
    * Create SVG for specified args.object and args.option
    * One could provide args.width and args.height as size options.
    * As callback argument one gets SVG code */
   JSROOT.MakeSVG = function(args, callback) {

      if (!args) args = {};

      if (!args.object) return JSROOT.CallBack(callback, null);

      if (!args.width) args.width = 1200;
      if (!args.height) args.height = 800;

      function build(main) {

         main.attr("width", args.width).attr("height", args.height);

         main.style("width", args.width+"px").style("height", args.height+"px");

         JSROOT.svg_workaround = undefined;

         JSROOT.draw(main.node(), args.object, args.option || "", function(painter) {

            main.select('svg').attr("xmlns", "http://www.w3.org/2000/svg")
                              .attr("width", args.width)
                              .attr("height", args.height)
                              .attr("style", "").attr("style", null)
                              .attr("class", null).attr("x", null).attr("y", null);

            var svg = main.html();

            if (JSROOT.svg_workaround) {
               for (var k=0;k<JSROOT.svg_workaround.length;++k)
                 svg = svg.replace('<path jsroot_svg_workaround="' + k + '"></path>', JSROOT.svg_workaround[k]);
               JSROOT.svg_workaround = undefined;
            }

            svg = svg.replace(/url\(\&quot\;\#(\w+)\&quot\;\)/g,"url(#$1)")        // decode all URL
                     .replace(/ class=\"\w*\"/g,"")                                // remove all classes
                     .replace(/<g transform=\"translate\(\d+\,\d+\)\"><\/g>/g,"")  // remove all empty groups with transform
                     .replace(/<g><\/g>/g,"");                                     // remove all empty groups

            main.remove();

            JSROOT.CallBack(callback, svg);
         });
      }

      if (!JSROOT.nodejs) {
         build(d3.select(window.document).append("div").style("visible", "hidden"));
      } else
      if (JSROOT.nodejs_document) {
         build(JSROOT.nodejs_window.d3.select('body').append('div'));
      } else {
         var jsdom = require('jsdom');
         jsdom.env({
            html:'',
            features:{ QuerySelector:true }, //you need query selector for D3 to work
            done:function(errors, window) {

               window.d3 = d3.select(window.document); //get d3 into the dom
               JSROOT.nodejs_window = window;
               JSROOT.nodejs_document = window.document; // used with three.js

               build(window.d3.select('body').append('div'));
            }});
      }
   }

   // Check resize of drawn element
   // As first argument divid one should use same argument as for the drawing
   // As second argument, one could specify "true" value to force redrawing of
   // the element even after minimal resize of the element
   // Or one just supply object with exact sizes like { width:300, height:200, force:true };

   JSROOT.resize = function(divid, arg) {
      if (arg === true) arg = { force: true }; else
      if (typeof arg !== 'object') arg = null;
      var dummy = new TObjectPainter(), done = false;
      dummy.SetDivId(divid, -1);
      dummy.ForEachPainter(function(painter) {
         if (!done && typeof painter.CheckResize == 'function')
            done = painter.CheckResize(arg);
      });
      return done;
   }

   // for compatibility, keep old name
   JSROOT.CheckElementResize = JSROOT.resize;

   // safely remove all JSROOT objects from specified element
   JSROOT.cleanup = function(divid) {
      var dummy = new TObjectPainter(), lst = [];
      dummy.SetDivId(divid, -1);
      dummy.ForEachPainter(function(painter) {
         if (lst.indexOf(painter) < 0) lst.push(painter);
      });
      for (var n=0;n<lst.length;++n) lst[n].Cleanup();
      dummy.select_main().html("");
      return lst;
   }

   // function to display progress message in the left bottom corner
   // previous message will be overwritten
   // if no argument specified, any shown messages will be removed
   JSROOT.progress = function(msg, tmout) {
      if (JSROOT.BatchMode || !document) return;
      var id = "jsroot_progressbox",
          box = d3.select("#"+id);

      if (!JSROOT.gStyle.ProgressBox) return box.remove();

      if ((arguments.length == 0) || !msg) {
         if ((tmout !== -1) || (!box.empty() && box.property("with_timeout"))) box.remove();
         return;
      }

      if (box.empty()) {
         box = d3.select(document.body)
                .append("div")
                .attr("id", id);
         box.append("p");
      }

      box.property("with_timeout", false);

      if (typeof msg === "string") {
         box.select("p").html(msg);
      } else {
         box.html("");
         box.node().appendChild(msg);
      }

      if (!isNaN(tmout) && (tmout>0)) {
         box.property("with_timeout", true);
         setTimeout(JSROOT.progress.bind(JSROOT,'',-1), tmout);
      }
   }

   Painter.createRootColors();

   JSROOT.LongPollSocket = LongPollSocket;
   JSROOT.Cef3QuerySocket = Cef3QuerySocket;
   JSROOT.WebWindowHandle = WebWindowHandle;
   JSROOT.DrawOptions = DrawOptions;
   JSROOT.ColorPalette = ColorPalette;
   JSROOT.TAttLineHandler = TAttLineHandler;
   JSROOT.TAttFillHandler = TAttFillHandler;
   JSROOT.TAttMarkerHandler = TAttMarkerHandler;
   JSROOT.TBasePainter = TBasePainter;
   JSROOT.TObjectPainter = TObjectPainter;
   JSROOT.TooltipHandler = TooltipHandler;
   JSROOT.TFramePainter = TFramePainter;
   JSROOT.TPadPainter = TPadPainter;
   JSROOT.TCanvasPainter = TCanvasPainter;

   return JSROOT;

}));
