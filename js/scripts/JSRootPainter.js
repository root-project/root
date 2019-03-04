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

   if (!JSROOT._test_d3_) {
      if ((typeof d3 == 'object') && d3.version && (d3.version[0]==="5"))  {
         if (d3.version !== '5.7.0')
            console.log('Reuse existing d3.js ' + d3.version + ", expected 5.7.0");
         JSROOT._test_d3_ = 5;
      } else if ((typeof d3 == 'object') && d3.version && (d3.version[0]==="4"))  {
         if (d3.version !== '4.4.4')
            console.warn('Try to use older d3.js ' + d3.version + ", expected 5.7.0");
         JSROOT._test_d3_ = 4;
      } else if ((typeof d3 == 'object') && d3.version && (d3.version[0]==="3")) {
         console.error("Very old d3.js " + d3.version + " found, please UPGRADE");
         d3.timeFormat = d3.time.format;
         d3.scaleTime = d3.time.scale;
         d3.scaleLog = d3.scale.log;
         d3.scaleLinear = d3.scale.linear;
         JSROOT._test_d3_ = 3;
      } else {
         console.error('Fail to identify d3.js version '  + (d3 ? d3.version : "???"));
      }
   }

   // list of user painters, called with arguments func(vis, obj, opt)
   JSROOT.DrawFuncs = { lst:[], cache:{} };

   /** @summary Register draw function for the class
    * @desc List of supported draw options could be provided, separated  with ';'
    * Several different draw functions for the same class or kind could be specified
    * @param {object} args - arguments
    * @param {string} args.name - class name
    * @param {string} [args.prereq] - prerequicities to load before search for the draw function
    * @param {string} args.func - name of draw function for the class
    * @param {string} [args.direct=false] - if true, function is just Redraw() method of TObjectPainter
    * @param {string} args.opt - list of supported draw options (separated with semicolon) like "col;scat;"
    * @param {string} [args.icon] - icon name shown for the class in hierarchy browser
    */
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
      arrow_right : { path: 'M30.796,226.318h377.533L294.938,339.682c-11.899,11.906-11.899,31.184,0,43.084c11.887,11.899,31.19,11.893,43.077,0  l165.393-165.386c5.725-5.712,8.924-13.453,8.924-21.539c0-8.092-3.213-15.84-8.924-21.551L338.016,8.925  C332.065,2.975,324.278,0,316.478,0c-7.802,0-15.603,2.968-21.539,8.918c-11.899,11.906-11.899,31.184,0,43.084l113.391,113.384  H30.796c-16.822,0-30.463,13.645-30.463,30.463C0.333,212.674,13.974,226.318,30.796,226.318z' },
      arrow_up : { path: 'M295.505,629.446V135.957l148.193,148.206c15.555,15.559,40.753,15.559,56.308,0c15.555-15.538,15.546-40.767,0-56.304  L283.83,11.662C276.372,4.204,266.236,0,255.68,0c-10.568,0-20.705,4.204-28.172,11.662L11.333,227.859  c-7.777,7.777-11.666,17.965-11.666,28.158c0,10.192,3.88,20.385,11.657,28.158c15.563,15.555,40.762,15.555,56.317,0  l148.201-148.219v493.489c0,21.993,17.837,39.82,39.82,39.82C277.669,669.267,295.505,651.439,295.505,629.446z' },
      arrow_diag : { path: 'M279.875,511.994c-1.292,0-2.607-0.102-3.924-0.312c-10.944-1.771-19.333-10.676-20.457-21.71L233.97,278.348  L22.345,256.823c-11.029-1.119-19.928-9.51-21.698-20.461c-1.776-10.944,4.031-21.716,14.145-26.262L477.792,2.149  c9.282-4.163,20.167-2.165,27.355,5.024c7.201,7.189,9.199,18.086,5.024,27.356L302.22,497.527  C298.224,506.426,289.397,511.994,279.875,511.994z M118.277,217.332l140.534,14.294c11.567,1.178,20.718,10.335,21.878,21.896  l14.294,140.519l144.09-320.792L118.277,217.332z' },
      auto_zoom: { path: 'M505.441,242.47l-78.303-78.291c-9.18-9.177-24.048-9.171-33.216,0c-9.169,9.172-9.169,24.045,0.006,33.217l38.193,38.188  H280.088V80.194l38.188,38.199c4.587,4.584,10.596,6.881,16.605,6.881c6.003,0,12.018-2.297,16.605-6.875  c9.174-9.172,9.174-24.039,0.011-33.217L273.219,6.881C268.803,2.471,262.834,0,256.596,0c-6.229,0-12.202,2.471-16.605,6.881  l-78.296,78.302c-9.178,9.172-9.178,24.045,0,33.217c9.177,9.171,24.051,9.171,33.21,0l38.205-38.205v155.4H80.521l38.2-38.188  c9.177-9.171,9.177-24.039,0.005-33.216c-9.171-9.172-24.039-9.178-33.216,0L7.208,242.464c-4.404,4.403-6.881,10.381-6.881,16.611  c0,6.227,2.477,12.207,6.881,16.61l78.302,78.291c4.587,4.581,10.599,6.875,16.605,6.875c6.006,0,12.023-2.294,16.61-6.881  c9.172-9.174,9.172-24.036-0.005-33.211l-38.205-38.199h152.593v152.063l-38.199-38.211c-9.171-9.18-24.039-9.18-33.216-0.022  c-9.178,9.18-9.178,24.059-0.006,33.222l78.284,78.302c4.41,4.404,10.382,6.881,16.611,6.881c6.233,0,12.208-2.477,16.611-6.881  l78.302-78.296c9.181-9.18,9.181-24.048,0-33.205c-9.174-9.174-24.054-9.174-33.21,0l-38.199,38.188v-152.04h152.051l-38.205,38.199  c-9.18,9.175-9.18,24.037-0.005,33.211c4.587,4.587,10.596,6.881,16.604,6.881c6.01,0,12.024-2.294,16.605-6.875l78.303-78.285  c4.403-4.403,6.887-10.378,6.887-16.611C512.328,252.851,509.845,246.873,505.441,242.47z' },
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
      cross: { path: "M80,40l176,176l176,-176l40,40l-176,176l176,176l-40,40l-176,-176l-176,176l-40,-40l176,-176l-176,-176z" },
      vrgoggles: { size: "245.82 141.73", path: 'M175.56,111.37c-22.52,0-40.77-18.84-40.77-42.07S153,27.24,175.56,27.24s40.77,18.84,40.77,42.07S198.08,111.37,175.56,111.37ZM26.84,69.31c0-23.23,18.25-42.07,40.77-42.07s40.77,18.84,40.77,42.07-18.26,42.07-40.77,42.07S26.84,92.54,26.84,69.31ZM27.27,0C11.54,0,0,12.34,0,28.58V110.9c0,16.24,11.54,30.83,27.27,30.83H99.57c2.17,0,4.19-1.83,5.4-3.7L116.47,118a8,8,0,0,1,12.52-.18l11.51,20.34c1.2,1.86,3.22,3.61,5.39,3.61h72.29c15.74,0,27.63-14.6,27.63-30.83V28.58C245.82,12.34,233.93,0,218.19,0H27.27Z'},
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

   /** @summary Draw options interpreter.
    * @constructor
    * @memberof JSROOT
    */
   var DrawOptions = function(opt) {
      this.opt = opt && (typeof opt=="string") ? opt.toUpperCase().trim() : "";
      this.part = "";
   }

   /** @summary Returns true if remaining options are empty. */
   DrawOptions.prototype.empty = function() {
      return this.opt.length === 0;
   }

   /** @summary Returns remaining part of the draw options. */
   DrawOptions.prototype.remain = function() {
      return this.opt;
   }

   /** @summary Checks if given option exists */
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

   /** @summary Returns remaining part of found option as integer. */
   DrawOptions.prototype.partAsInt = function(offset, dflt) {
      var val = this.part.replace( /^\D+/g, '');
      val = val ? parseInt(val,10) : Number.NaN;
      return isNaN(val) ? (dflt || 0) : val + (offset || 0);
   }

   // ============================================================================================

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

   Painter.convertSymbol = function(charactere) {
     // example: '#pi' will give '\u03A0'
     return Painter.symbols_map[charactere];
   }

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
      if (inter === "nomenu") JSROOT.gStyle.ContextMenu = false;
      else if (inter !== null) {
         if (!inter || (inter=="1")) inter = "111111"; else
         if (inter=="0") inter = "000000";
         if (inter.length === 6) {
            if (inter[0] == "0") JSROOT.gStyle.ToolBar = false; else
            if (inter[0] == "1") JSROOT.gStyle.ToolBar = 'popup'; else
            if (inter[0] == "2") JSROOT.gStyle.ToolBar = true;
            inter = inter.substr(1);
         }
         if (inter.length==5) {
            JSROOT.gStyle.Tooltip =     parseInt(inter[0]);
            JSROOT.gStyle.ContextMenu = (inter[1] != '0');
            JSROOT.gStyle.Zooming  =    (inter[2] != '0');
            JSROOT.gStyle.MoveResize =  (inter[3] != '0');
            JSROOT.gStyle.DragAndDrop = (inter[4] != '0');
         }
      }

      var tt = JSROOT.GetUrlOption("tooltip", url);
      if (tt !== null) JSROOT.gStyle.Tooltip = parseInt(tt);

      var mathjax = JSROOT.GetUrlOption("mathjax", url),
          latex = JSROOT.GetUrlOption("latex", url);

      if ((mathjax!==null) && (mathjax!="0") && (latex===null)) latex = "math";
      if (latex!==null) JSROOT.gStyle.Latex = latex; // decoding will be performed with the first text drawing

      if (JSROOT.GetUrlOption("nomenu", url)!==null) JSROOT.gStyle.ContextMenu = false;
      if (JSROOT.GetUrlOption("noprogress", url)!==null) JSROOT.gStyle.ProgressBox = false;
      if (JSROOT.GetUrlOption("notouch", url)!==null) JSROOT.touches = false;
      if (JSROOT.GetUrlOption("adjframe", url)!==null) JSROOT.gStyle.CanAdjustFrame = true;

      var optstat = JSROOT.GetUrlOption("optstat", url);
      if (optstat!==null) JSROOT.gStyle.fOptStat = parseInt(optstat);
      var optfit = JSROOT.GetUrlOption("optfit", url);
      if (optfit!==null) JSROOT.gStyle.fOptFit = parseInt(optfit);
      JSROOT.gStyle.fStatFormat = JSROOT.GetUrlOption("statfmt", url, JSROOT.gStyle.fStatFormat);
      JSROOT.gStyle.fFitFormat = JSROOT.GetUrlOption("fitfmt", url, JSROOT.gStyle.fFitFormat);

      var toolbar = JSROOT.GetUrlOption("toolbar", url);
      if (toolbar !== null) {
         var val = null;
         if (toolbar.indexOf('popup')>=0) val = 'popup';
         if (toolbar.indexOf('left')>=0) { JSROOT.gStyle.ToolBarSide = 'left'; val = 'popup'; }
         if (toolbar.indexOf('right')>=0) { JSROOT.gStyle.ToolBarSide = 'right'; val = 'popup'; }
         if (toolbar.indexOf('vert')>=0) { JSROOT.gStyle.ToolBarVert = true; val = 'popup'; }
         if (toolbar.indexOf('show')>=0) val = true;
         JSROOT.gStyle.ToolBar = val || ((toolbar.indexOf("0")<0) && (toolbar.indexOf("false")<0) && (toolbar.indexOf("off")<0));
      }

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

   /** Add new colors from object array. */
   Painter.extendRootColors = function(jsarr, objarr) {
      if (!jsarr) {
         jsarr = [];
         for (var n=0;n<this.root_colors.length;++n)
            jsarr[n] = this.root_colors[n];
      }

      if (!objarr) return jsarr;

      var rgb_array = objarr;
      if (objarr._typename && objarr.arr) {
         rgb_array = [];
         for (var n = 0; n < objarr.arr.length; ++n) {
            var col = objarr.arr[n];
            if (!col || (col._typename != 'TColor')) continue;

            if ((col.fNumber>=0) && (col.fNumber<=10000))
               rgb_array[col.fNumber] = Painter.MakeColorRGB(col);
         }
      }


      for (var n = 0; n < rgb_array.length; ++n)
         if (rgb_array[n] && (jsarr[n] != rgb_array[n]))
            jsarr[n] = rgb_array[n];

      return jsarr;
   }

   /** Set global list of colors.
    * Either TObjArray of TColor instances or just plain array with rgb() code.
    * List of colors typically stored together with TCanvas primitives
    * @private */
   Painter.adoptRootColors = function(objarr) {
      this.extendRootColors(this.root_colors, objarr);
   }

   // =====================================================================

   /**
    * Color palette handle.
    * @constructor
    * @memberof JSROOT
    * @private
    */

   function ColorPalette(arr) {
      this.palette = arr;
   }

   /** @summary Returns color index which correspond to contour index of provided length */
   ColorPalette.prototype.calcColorIndex = function(i,len) {
      var theColor = Math.floor((i+0.99)*this.palette.length/(len-1));
      if (theColor > this.palette.length-1) theColor = this.palette.length-1;
      return theColor;
   }

   /** @summary Returns color with provided index */
   ColorPalette.prototype.getColor = function(indx) {
      return this.palette[indx];
   }

   /** @summary Returns number of colors in the palette */
   ColorPalette.prototype.getLength = function() {
      return this.palette.length;
   }

   /** @summary Calculate color for given i and len */
   ColorPalette.prototype.calcColor = function(i,len) {
      var indx = this.calcColorIndex(i,len);
      return this.getColor(indx);
   }

   // =============================================================================

   /**
    * @summary Handle for marker attributes.
    * @constructor
    * @memberof JSROOT
    */

   function TAttMarkerHandler(args) {
      this.x0 = this.y0 = 0;
      this.color = 'black';
      this.style = 1;
      this.size = 8;
      this.scale = 1;
      this.stroke = true;
      this.fill = true;
      this.marker = "";
      this.ndig = 0;
      this.used = true;
      this.changed = false;

      this.func = this.Apply.bind(this);

      this.SetArgs(args);

      this.changed = false;
   }

   /** @summary Set marker attributes.
    *
    * @param {object} args - arguments can be
    * @param {object} args.attr - instance of TAttrMarker (or derived class) or
    * @param {string} args.color - color in HTML form like grb(1,4,5) or 'green'
    * @param {number} args.style - marker style
    * @param {number} args.size - marker size
    */
   TAttMarkerHandler.prototype.SetArgs = function(args) {
      if ((typeof args == 'object') && (typeof args.fMarkerStyle == 'number')) args = { attr: args };

      if (args.attr) {
         if (args.color === undefined) args.color = Painter.root_colors[args.attr.fMarkerColor];
         if (!args.style || (args.style<0)) args.style = args.attr.fMarkerStyle;
         if (!args.size) args.size = args.attr.fMarkerSize;
      }

      this.Change(args.color, args.style, args.size);
   }

   /** @summary Reset position, used for optimization of drawing of multiple markers
    * @private */
   TAttMarkerHandler.prototype.reset_pos = function() {
      this.lastx = this.lasty = null;
   }

   /** @summary Create marker path for given position.
    *
    * @desc When drawing many elementary points, created path may depend from previously produced markers.
    *
    * @param {number} x - first coordinate
    * @param {number} y - second coordinate
    * @returns {string} path string
    */
   TAttMarkerHandler.prototype.create = function(x,y) {
      if (!this.optimized)
         return "M" + (x+this.x0).toFixed(this.ndig)+ "," + (y+this.y0).toFixed(this.ndig) + this.marker;

      // use optimized handling with relative position
      var xx = Math.round(x), yy = Math.round(y), m1 = "M"+xx+","+yy+"h1",
          m2 = (this.lastx===null) ? m1 : ("m"+(xx-this.lastx)+","+(yy-this.lasty)+"h1");
      this.lastx = xx+1; this.lasty = yy;
      return (m2.length < m1.length) ? m2 : m1;
   }

   /** @summary Returns full size of marker */
   TAttMarkerHandler.prototype.GetFullSize = function() {
      return this.scale*this.size;
   }

   /** @summary Returns approximate length of produced marker string */
   TAttMarkerHandler.prototype.MarkerLength = function() {
      return this.marker ? this.marker.length : 10;
   }

   /** @summary Change marker attributes.
    *
    *  @param {string} color - marker color
    *  @param {number} style - marker style
    *  @param {number} size - marker size
    */
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
         this.optimized = true;
         this.reset_pos();
         return true;
      }

      this.optimized = false;

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

   /** @summary Apply marker styles to created element */
   TAttMarkerHandler.prototype.Apply = function(selection) {
      selection.style('stroke', this.stroke ? this.color : "none");
      selection.style('fill', this.fill ? this.color : "none");
   }

   /** Method used when color or pattern were changed with OpenUi5 widgets.
    * @private */
   TAttMarkerHandler.prototype.verifyDirectChange = function(painter) {
      this.Change(this.color, parseInt(this.style), parseFloat(this.size));
   }

   /** @summary Create sample with marker in given SVG element
    *
    * @param {selection} svg - SVG element
    * @param {number} width - width of sample SVG
    * @param {number} height - height of sample SVG
    * @private
    */
   TAttMarkerHandler.prototype.CreateSample = function(svg, width, height) {
      this.reset_pos();

      svg.append("path")
         .attr("d", this.create(width/2, height/2))
         .call(this.func);
   }

   // =======================================================================

   /**
    * Handle for line attributes.
    * @constructor
    * @memberof JSROOT
    */

   function TAttLineHandler(args) {
      this.func = this.Apply.bind(this);
      this.used = true;
      if (args._typename && (args.fLineStyle!==undefined)) args = { attr: args };

      this.SetArgs(args);
   }

   /**
    * @summary Set line attributes.
    *
    * @param {object} args - specify attributes by different ways
    * @param {object} args.attr - TAttLine object with appropriate data members or
    * @param {string} args.color - color in html like rgb(10,0,0) or "red"
    * @param {number} args.style - line style number
    * @param {number} args.width - line width
    */
   TAttLineHandler.prototype.SetArgs = function(args) {
      if (args.attr) {
         args.color = args.color0 || Painter.root_colors[args.attr.fLineColor];
         if (args.width===undefined) args.width = args.attr.fLineWidth;
         args.style = args.attr.fLineStyle;
      } else if (typeof args.color == 'string') {
         if ((args.color !== 'none') && !args.width) args.width = 1;
      } else if (typeof args.color == 'number') {
         args.color = Painter.root_colors[args.color];
      }

      if (args.width===undefined)
         args.width = (args.color && args.color!='none') ? 1 : 0;

      this.color = (args.width===0) ? 'none' : args.color;
      this.width = args.width;
      this.style = args.style;

      if (args.can_excl) {
         this.excl_side = this.excl_width = 0;
         if (Math.abs(this.width) > 99) {
            // exclusion graph
            this.excl_side = (this.width < 0) ? -1 : 1;
            this.excl_width = Math.floor(this.width / 100) * 5;
            this.width = Math.abs(this.width % 100); // line width
         }
      }

      // if custom color number used, use lightgrey color to show lines
      if (!this.color && (this.width > 0))
         this.color = 'lightgrey';
   }

   /**
    * @summary Change exclusion attributes.
    * @private
    */

   TAttLineHandler.prototype.ChangeExcl = function(side,width) {
      if (width !== undefined) this.excl_width = width;
      if (side !== undefined) {
         this.excl_side = side;
         if ((this.excl_width===0) && (this.excl_side!==0)) this.excl_width = 20;
      }
      this.changed = true;
   }

   /**
    * @summary Returns true if line attribute is empty and will not be applied.
    */

   TAttLineHandler.prototype.empty = function() {
      return this.color == 'none';
   }

   /**
    * @summary Applies line attribute to selection.
    *
    * @param {object} selection - d3.js selection
    */

   TAttLineHandler.prototype.Apply = function(selection) {
      this.used = true;
      if (this.empty())
         selection.style('stroke', null)
                  .style('stroke-width', null)
                  .style('stroke-dasharray', null);
      else
         selection.style('stroke', this.color)
                  .style('stroke-width', this.width)
                  .style('stroke-dasharray', Painter.root_line_styles[this.style] || null);
   }

   /**
    * @summary Change line attributes
    * @private
    */

   TAttLineHandler.prototype.Change = function(color, width, style) {
      if (color !== undefined) this.color = color;
      if (width !== undefined) this.width = width;
      if (style !== undefined) this.style = style;
      this.changed = true;
   }

   /**
    * Create sample element inside primitive SVG - used in context menu
    * @private
    */

   TAttLineHandler.prototype.CreateSample = function(svg, width, height) {
      svg.append("path")
         .attr("d","M0," + height/2+"h"+width)
         .call(this.func);
   }

   // =======================================================================


   /**
    * @summary Handle for fill attributes.
    * @constructor
    * @memberof JSROOT
    * @param {object} args - different arguments to set fill attributes
    * @param {number} [args.kind = 2] - 1 means object drawing where combination fillcolor==0 and fillstyle==1001 means no filling,  2 means all other objects where such combination is white-color filling
    */

   function TAttFillHandler(args) {
      this.color = "none";
      this.colorindx = 0;
      this.pattern = 0;
      this.used = true;
      this.kind = args.kind || 2;
      this.changed = false;
      this.func = this.Apply.bind(this);
      this.SetArgs(args);
      this.changed = false; // unset change property that
   }

   /** @summary Set fill style as arguments */
   TAttFillHandler.prototype.SetArgs = function(args) {
      if (args.attr && (typeof args.attr == 'object')) {
         if ((args.pattern===undefined) && (args.attr.fFillStyle!==undefined)) args.pattern = args.attr.fFillStyle;
         if ((args.color===undefined) && (args.attr.fFillColor!==undefined)) args.color = args.attr.fFillColor;
      }
      this.Change(args.color, args.pattern, args.svg, args.color_as_svg);
   }

   /** @summary Apply fill style to selection */
   TAttFillHandler.prototype.Apply = function(selection) {
      this.used = true;

      selection.style('fill', this.fillcolor());

      if ('opacity' in this)
         selection.style('opacity', this.opacity);

      if ('antialias' in this)
         selection.style('antialias', this.antialias);
   }

   /** @summary Returns fill color (or pattern url) */
   TAttFillHandler.prototype.fillcolor = function() {
      return this.pattern_url || this.color;
   }

   /** @summary Returns fill color without pattern url.
    *
    * @desc If empty, alternative color will be provided
    * @param {string} [altern=undefined] - alternative color which returned when fill color not exists
    * @private */
   TAttFillHandler.prototype.fillcoloralt = function(altern) {
      return this.color && (this.color!="none") ? this.color : altern;
   }

   /** @summary Returns true if color not specified or fill style not specified */
   TAttFillHandler.prototype.empty = function() {
      var fill = this.fillcolor();
      return !fill || (fill == 'none');
   }

   /** @summary Set solid fill color as fill pattern
    * @param {string} col - solid color */
   TAttFillHandler.prototype.SetSolidColor = function(col) {
      delete this.pattern_url;
      this.color = col;
      this.pattern = 1001;
   }

   /** @summary Check if solid fill is used, also color can be checked
    * @param {string} [solid_color = undefined] - when specified, checks if fill color matches */
   TAttFillHandler.prototype.isSolid = function(solid_color) {
      if (this.pattern !== 1001) return false;
      return !solid_color || solid_color==this.color;
   }

   /** @summary Method used when color or pattern were changed with OpenUi5 widgets
    * @private */
   TAttFillHandler.prototype.verifyDirectChange = function(painter) {
      if (typeof this.pattern == 'string') this.pattern = parseInt(this.pattern);
      if (isNaN(this.pattern)) this.pattern = 0;

      this.Change(this.color, this.pattern, painter ? painter.svg_canvas() : null, true);
   }

   /** @summary Method to change fill attributes.
    *
    * @param {number} color - color index
    * @param {number} pattern - pattern index
    * @param {selection} svg - top canvas element for pattern storages
    * @param {string} [color_as_svg = undefined] - color as HTML string index
    */
   TAttFillHandler.prototype.Change = function(color, pattern, svg, color_as_svg) {
      delete this.pattern_url;
      this.changed = true;

      if ((color !== undefined) && !isNaN(color) && !color_as_svg)
         this.colorindx = parseInt(color);

      if ((pattern !== undefined) && !isNaN(pattern)) {
         this.pattern = parseInt(pattern);
         delete this.opacity;
         delete this.antialias;
      }

      if ((this.pattern == 1000) && (this.colorindx === 0)) {
         this.pattern_url = 'white';
         return true;
      }

      if (this.pattern == 1000) this.pattern = 1001;

      if (this.pattern < 1001) {
         this.pattern_url = 'none';
         return true;
      }

      if (this.isSolid() && (this.colorindx===0) && (this.kind===1) && !color_as_svg) {
         this.pattern_url = 'none';
         return true;
      }

      var indx = this.colorindx;

      if (color_as_svg) {
         this.color = color;
         indx = 10000 + JSROOT.id_counter++; // use fictional unique index far away from existing color indexes
      } else {
         this.color = JSROOT.Painter.root_colors[indx];
      }

      if (typeof this.color != 'string') this.color = "none";

      if (this.isSolid()) return true;

      if ((this.pattern >= 4000) && (this.pattern <= 4100)) {
         // special transparent colors (use for subpads)
         this.opacity = (this.pattern - 4000)/100;
         return true;
      }

      if (!svg || svg.empty() || (this.pattern < 3000)) return false;

      var id = "pat_" + this.pattern + "_" + indx,
          defs = svg.select('.canvas_defs');

      if (defs.empty())
         defs = svg.insert("svg:defs",":first-child").attr("class","canvas_defs");

      this.pattern_url = "url(#" + id + ")";
      this.antialias = false;

      if (!defs.select("."+id).empty()) {
         if (color_as_svg) console.log('find id in def', id);
         return true;
      }

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
            if ((this.pattern>3025) && (this.pattern<3100)) {
               // same as 3002, see TGX11.cxx, line 2234
               w = 4; h = 2; fills = "M1,0h1v1h-1zM3,1h1v1h-1z"; break;
            }

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
         var col = d3.rgb(this.color);
         col.r = Math.round((col.r+255)/2); col.g = Math.round((col.g+255)/2); col.b = Math.round((col.b+255)/2);
         patt.append("svg:path").attr("d", fills2).style("fill", col);
      }
      if (fills) patt.append("svg:path").attr("d", fills).style("fill", this.color);
      if (lines) patt.append("svg:path").attr("d", lines).style('stroke', this.color).style("stroke-width", 1).style("fill", lfill);

      return true;
   }

   /** @summary Create sample of fill pattern inside SVG
    * @private */
   TAttFillHandler.prototype.CreateSample = function(sample_svg, width, height) {

      // we need to create extra handle to change
      var sample = new TAttFillHandler({ svg: sample_svg, pattern: this.pattern, color: this.color, color_as_svg: true });

      sample_svg.append("path")
                .attr("d","M0,0h" + width+"v"+height+"h-" + width + "z")
                .call(sample.func);
   }

   // ===========================================================================

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
      if (awidth < .5) return ticks ? "%S.%L" : "%H:%M:%S.%L";
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

      if (!Painter.symbolsRegexCache) {
        // Create a single regex to detect any symbol to replace
        Painter.symbolsRegexCache = new RegExp('(' + Object.keys(Painter.symbols_map).join('|').replace(/\{/g, '\{').replace(/\\}/g, '\\}') + ')', 'g');
      }

      str = str.replace(Painter.symbolsRegexCache, Painter.convertSymbol);

      str = str.replace(/\{\}/g, "");

      return str;
   }

   Painter.approxTextWidth = function(font, label) {
      // returns approximate width of given label, required for reasonable scaling of text in node.js

      return label.length * font.size * font.aver_width;
   }

   Painter.isAnyLatex = function(str) {
      return (str.indexOf("#")>=0) || (str.indexOf("\\")>=0) || (str.indexOf("{")>=0);
   }

   /** Function translates ROOT TLatex into MathJax format */
   Painter.translateMath = function(str, kind, color, painter) {

      if (kind != 2) {
         for (var x in Painter.math_symbols_map)
            str = str.replace(new RegExp(x,'g'), Painter.math_symbols_map[x]);

         for (var x in Painter.symbols_map)
            if (x.length > 2)
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
            var col = painter.get_color(colindx), cnt = 1;
            str = str.substr(p+2);
            p = -1;
            while (cnt && (++p<str.length)) {
               if (str[p]=='{') cnt++; else if (str[p]=='}') cnt--;
            }
            if (cnt!=0) break;

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
      return "\\(\\color{" + color + '}{' + str + "}\\)";
   }

   /** Function used to provide svg:path for the smoothed curves.
    *
    * reuse code from d3.js. Used in TH1, TF1 and TGraph painters
    * kind should contain "bezier" or "line".
    * If first symbol "L", then it used to continue drawing
    * @private
    */
   Painter.BuildSvgPath = function(kind, bins, height, ndig) {

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

      var res = { path: "", close: "" }, bin = bins[0], prev, maxy = Math.max(bin.gry, height+5),
          currx = Math.round(bin.grx), curry = Math.round(bin.gry), dx, dy, npnts = bins.length;

      function conv(val) {
         var vvv = Math.round(val);
         if ((ndig==0) || (vvv===val)) return vvv.toString();
         var str = val.toFixed(ndig);
         while ((str[str.length-1] == '0') && (str.lastIndexOf(".") < str.length-1))
            str = str.substr(0,str.length-1);
         if (str[str.length-1] == '.')
            str = str.substr(0,str.length-1);
         if (str == "-0") str = "0";
         return str;
      }

      res.path = ((kind[0] == "L") ? "L" : "M") + conv(bin.grx) + "," + conv(bin.gry);

      // just calculate all deltas, can be used to build exclusion
      if (smooth || kind.indexOf('calc')>=0)
         jsroot_d3_svg_lineMonotoneTangents(bins);

      if (smooth) {
         // build smoothed curve
         res.path += "c" + conv(bin.dgrx) + "," + conv(bin.dgry) + ",";
         for(var n=1; n<npnts; ++n) {
            var prev = bin;
            bin = bins[n];
            if (n > 1) res.path += "s";
            res.path += conv(bin.grx-bin.dgrx-prev.grx) + "," + conv(bin.gry-bin.dgry-prev.gry) + "," + conv(bin.grx-prev.grx) + "," + conv(bin.gry-prev.gry);
            maxy = Math.max(maxy, prev.gry);
         }
      } else if (npnts < 10000) {
         // build simple curve
         for(var n=1; n<npnts; ++n) {
            bin = bins[n];
            dx = Math.round(bin.grx) - currx;
            dy = Math.round(bin.gry) - curry;
            if (dx && dy) res.path += "l"+dx+","+dy;
            else if (!dx && dy) res.path += "v"+dy;
            else if (dx && !dy) res.path += "h"+dx;
            currx += dx; curry += dy;
            maxy = Math.max(maxy, curry);
         }
      } else {
         // build line with trying optimize many vertical moves
         var lastx, lasty, cminy = curry, cmaxy = curry, prevy = curry;
         for(var n=1; n<npnts; ++n) {
            bin = bins[n];
            lastx = Math.round(bin.grx);
            lasty = Math.round(bin.gry);
            maxy = Math.max(maxy, lasty);
            dx = lastx - currx;
            if (dx===0) {
               // if X not change, just remember amplitude and
               cminy = Math.min(cminy, lasty);
               cmaxy = Math.max(cmaxy, lasty);
               prevy = lasty;
               continue;
            }

            if (cminy !== cmaxy) {
               if (cminy != curry) res.path += "v" + (cminy-curry);
               res.path += "v" + (cmaxy-cminy);
               if (cmaxy != prevy) res.path += "v" + (prevy-cmaxy);
               curry = prevy;
            }
            dy = lasty - curry;
            if (dy) res.path += "l"+dx+","+dy;
               else res.path += "h"+dx;
            currx = lastx; curry = lasty;
            prevy = cminy = cmaxy = lasty;
         }

         if (cminy != cmaxy) {
            if (cminy != curry) res.path += "v"+(cminy-curry);
            res.path += "v"+(cmaxy-cminy);
            if (cmaxy != prevy) res.path += "v"+(prevy-cmaxy);
            curry = prevy;
         }

      }

      if (height>0)
         res.close = "L"+conv(bin.grx)+","+conv(maxy) +
                     "h"+conv(bins[0].grx-bin.grx) + "Z";

      return res;
   }

   // ==============================================================================

   function LongPollSocket(addr, _raw, _args) {
      this.path = addr;
      this.connid = null;
      this.req = null;
      this.raw = _raw;
      this.args = _args;

      this.nextrequest("", "connect");
   }

   LongPollSocket.prototype.nextrequest = function(data, kind) {
      var url = this.path, reqmode = "buf", post = null;
      if (kind === "connect") {
         url += this.raw ? "?raw_connect" : "?txt_connect";
         if (this.args) url += "&" + this.args;
         console.log('longpoll connect ' + url + ' raw = ' + this.raw);
         this.connid = "connect";
      } else if (kind === "close") {
         if ((this.connid===null) || (this.connid==="close")) return;
         url+="?connection="+this.connid + "&close";
         this.connid = "close";
         reqmode += ";sync"; // use sync mode to close connection before browser window closed
      } else if ((this.connid===null) || (typeof this.connid!=='number')) {
         return console.error("No connection");
      } else {
         url+="?connection="+this.connid;
         if (kind==="dummy") url+="&dummy";
      }

      if (data) {
         if (this.raw) {
            // special workaround to avoid POST request, use base64 coding
            url += "&post=" + btoa(data);
         } else {
            // send data with post request - most efficient way
            reqmode = "post";
            post = data;
         }
      }

      var req = JSROOT.NewHttpRequest(url, reqmode, function(res) {
         // this set to the request itself, res is response

         if (this.handle.req === this)
            this.handle.req = null; // get response for existing dummy request

         if (res === null)
            return this.handle.processreq(null);

         if (this.handle.raw) {
            // raw mode - all kind of reply data packed into binary buffer
            // first 4 bytes header "txt:" or "bin:"
            // after the "bin:" there is length of optional text argument like "bin:14  :optional_text"
            // and immedaitely after text binary data. Server sends binary data so, that offset should be multiple of 8

            var str = "", i = 0, u8Arr = new Uint8Array(res), offset = u8Arr.length;
            if (offset < 4) {
               console.error('longpoll got short message in raw mode ' + offset);
               return this.handle.processreq(null);
            }

            while(i<4) str += String.fromCharCode(u8Arr[i++]);
            if (str != "txt:") {
               str = "";
               while ((i<offset) && (String.fromCharCode(u8Arr[i]) != ':')) str += String.fromCharCode(u8Arr[i++]);
               ++i;
               offset = i + parseInt(str.trim());
            }

            str = "";
            while (i<offset) str += String.fromCharCode(u8Arr[i++]);

            if (str) {
               if (str == "<<nope>>") str = "";
               this.handle.processreq(str);
            }
            if (offset < u8Arr.length)
               this.handle.processreq(res, offset);
         } else if (this.getResponseHeader("Content-Type") == "application/x-binary") {
            // binary reply with optional header
            var extra_hdr = this.getResponseHeader("LongpollHeader");
            if (extra_hdr) this.handle.processreq(extra_hdr);
            this.handle.processreq(res, 0);
         } else {
            // text reply
            if (res && typeof res !== "string") {
               var str = "", u8Arr = new Uint8Array(res);
               for (var i = 0; i < u8Arr.length; ++i)
                  str += String.fromCharCode(u8Arr[i]);
               res = str;
            }
            if (res == "<<nope>>") res = "";
            this.handle.processreq(res);
         }
      });

      req.handle = this;
      if (kind==="dummy") this.req = req; // remember last dummy request, wait for reply
      req.send(post);
   }

   LongPollSocket.prototype.processreq = function(res, _offset) {
      if (res===null) {
         if (typeof this.onerror === 'function') this.onerror("receive data with connid " + (this.connid || "---"));
         // if (typeof this.onclose === 'function') this.onclose();
         this.connid = null;
         return;
      }

      if (this.connid==="connect") {
         if (!res) {
            this.connid = null;
            if (typeof this.onerror === 'function') this.onerror("connection rejected");
            return;
         }

         this.connid = parseInt(res);
         console.log('Get new longpoll connection with id ' + this.connid);
         if (typeof this.onopen == 'function') this.onopen();
      } else if (this.connid==="close") {
         if (typeof this.onclose == 'function') this.onclose();
         return;
      } else {
         if ((typeof this.onmessage==='function') && res)
            this.onmessage({ data: res, offset: _offset });
      }

      if (!this.req) this.nextrequest("","dummy"); // send new poll request when necessary
   }

   LongPollSocket.prototype.send = function(str) {
      this.nextrequest(str);
   }

   LongPollSocket.prototype.close = function() {
      this.nextrequest("", "close");
   }

   // ========================================================================================

   function FileDumpSocket(receiver) {
      this.receiver = receiver;
      this.protocol = [];
      this.cnt = 0;
      JSROOT.NewHttpRequest("protocol.json", "text", this.get_protocol.bind(this)).send();
   }

   FileDumpSocket.prototype.get_protocol = function(res) {
      if (!res) return;
      this.protocol = JSON.parse(res);
      if (typeof this.onopen == 'function') this.onopen();
      this.next_operation();
   }

   FileDumpSocket.prototype.send = function(str) {
      if (this.protocol[this.cnt] == "send") {
         this.cnt++;
         setTimeout(this.next_operation.bind(this),10);
      }
   }

   FileDumpSocket.prototype.close = function() {
   }

   FileDumpSocket.prototype.next_operation = function() {
      // when file request running - just ignore
      if (this.wait_for_file) return;
      var fname = this.protocol[this.cnt];
      if (!fname) return;
      if (fname == "send") return; // waiting for send
      // console.log("getting file", fname, "wait", this.wait_for_file);
      this.wait_for_file = true;
      JSROOT.NewHttpRequest(fname, (fname.indexOf(".bin") > 0 ? "buf" : "text"), this.get_file.bind(this, fname)).send();
      this.cnt++;
   }

   FileDumpSocket.prototype.get_file = function(fname, res) {
      // console.log('got file', fname, typeof res, !!res);
      this.wait_for_file = false;
      if (!res) return;
      if (this.receiver.ProvideData)
         this.receiver.ProvideData(res, 0);
      setTimeout(this.next_operation.bind(this),10);
   }

   // ========================================================================================


   /** Client communication handle for TWebWindow.
    *
    * Should be created with {@link JSROOT.ConnectWebWindow} function
    *
    * @constructor
    * @memberof JSROOT
    */
   function WebWindowHandle(socket_kind) {
      this.kind = socket_kind;
      this.state = 0;
      this.cansend = 10;
      this.ackn = 10;
   }

   /** Set callbacks reciever.
    *
    * Following function can be defined in receiver object:
    *    - OnWebsocketMsg
    *    - OnWebsocketOpened,
    *    - OnWebsocketClosed
    */
   WebWindowHandle.prototype.SetReceiver = function(obj) {
      this.receiver = obj;
   }

   /** Cleanup and close connection. */
   WebWindowHandle.prototype.Cleanup = function() {
      delete this.receiver;
      this.Close(true);
   }

   /** Invoke method in the receiver.
    * @private */
   WebWindowHandle.prototype.InvokeReceiver = function(method, arg, arg2) {
      if (this.receiver && (typeof this.receiver[method] == 'function'))
         this.receiver[method](this, arg, arg2);
   }

   /** Provide data for receiver. When no queue - do it directly.
    * @private */
   WebWindowHandle.prototype.ProvideData = function(_msg, _len) {
      if (!this.msgqueue || !this.msgqueue.length)
         return this.InvokeReceiver("OnWebsocketMsg", _msg, _len);
      this.msgqueue.push({ ready: true, msg: _msg, len: _len});
   }

   /** Reserve entry in queue for data, which is not yet decoded.
    * @private */
   WebWindowHandle.prototype.ReserveQueueItem = function() {
      if (!this.msgqueue) this.msgqueue = [];
      var item = { ready: false, msg: null, len: 0 };
      this.msgqueue.push(item);
      return item;
   }

   /** Reserver entry in queue for data, which is not yet decoded.
    * @private */
   WebWindowHandle.prototype.DoneItem = function(item, _msg, _len) {
      item.ready = true;
      item.msg = _msg;
      item.len = _len;
      if (this._loop_msgqueue) return;
      this._loop_msgqueue = true;
      while ((this.msgqueue.length > 0) && this.msgqueue[0].ready) {
         var front = this.msgqueue.shift();
         this.InvokeReceiver("OnWebsocketMsg", front.msg, front.len);
      }
      if (this.msgqueue.length == 0)
         delete this.msgqueue;
      delete this._loop_msgqueue;
   }

   /** Close connection. */
   WebWindowHandle.prototype.Close = function(force) {

      if (this.timerid) {
         clearTimeout(this.timerid);
         delete this.timerid;
      }

      if (this._websocket && (this.state > 0)) {
         this.state = force ? -1 : 0; // -1 prevent socket from reopening
         this._websocket.onclose = null; // hide normal handler
         this._websocket.close();
         delete this._websocket;
      }
   }

   /** Returns if one can send text message to server. Checks number of send credits */
   WebWindowHandle.prototype.CanSend = function(numsend) {
      return (this.cansend >= (numsend || 1));
   }

   /** Send text message via the connection. */
   WebWindowHandle.prototype.Send = function(msg, chid) {
      if (!this._websocket || (this.state<=0)) return false;

      if (isNaN(chid) || (chid===undefined)) chid = 1; // when not configured, channel 1 is used - main widget

      if (this.cansend <= 0) console.error('should be queued before sending cansend: ' + this.cansend);

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

   /** Send keepalive message.
    * @private */
   WebWindowHandle.prototype.KeepAlive = function() {
      delete this.timerid;
      this.Send("KEEPALIVE", 0);
   }

   /** Method opens relative path with the same kind of socket.
    * @private
    */
   WebWindowHandle.prototype.CreateRelative = function(relative) {
      if (!relative || !this.kind || !this.href) return null;

      var handle = new WebWindowHandle(this.kind);
      console.log('Try to connect ', this.href + relative);
      handle.Connect(this.href + relative);
      return handle;
   }

   /** Create configured socket for current object. */
   WebWindowHandle.prototype.Connect = function(href) {

      this.Close();

      var pthis = this, ntry = 0, args = (this.key ? ("key=" + this.key) : "");

      function retry_open(first_time) {

         if (pthis.state != 0) return;

         if (!first_time) console.log("try connect window again" + (new Date()).getTime());

         if (pthis._websocket) pthis._websocket.close();
         delete pthis._websocket;

         var conn = null;
         if (!href) {
            href = window.location.href;
            if (href && href.lastIndexOf("/")>0) href = href.substr(0, href.lastIndexOf("/")+1);
         }
         pthis.href = href;
         ntry++;

         if (first_time) console.log('Opening web socket at ' + href);

         if (ntry>2) JSROOT.progress("Trying to connect " + href);

         var path = href;

         if (pthis.kind == "file") {
            path += "root.filedump";
            conn = new FileDumpSocket(pthis);
            console.log('configure protocol log ' + path);
         } else if ((pthis.kind === 'websocket') && first_time) {
            path = path.replace("http://", "ws://").replace("https://", "wss://") + "root.websocket";
            if (args) path += "?" + args;
            console.log('configure websocket ' + path);
            conn = new WebSocket(path);
         } else {
            path += "root.longpoll";
            console.log('configure longpoll ' + path);
            conn = new LongPollSocket(path, (pthis.kind === 'rawlongpoll'), args);
         }

         if (!conn) return;

         pthis._websocket = conn;

         conn.onopen = function() {
            if (ntry > 2) JSROOT.progress();
            pthis.state = 1;

            var key = pthis.key || "";

            pthis.Send("READY=" + key, 0); // need to confirm connection
            pthis.InvokeReceiver('OnWebsocketOpened');
         }

         conn.onmessage = function(e) {
            var msg = e.data;

            if (pthis.next_binary) {
               delete pthis.next_binary;

               if (msg instanceof Blob) {
                  // this is case of websocket
                  // console.log('Get Blob object - convert to buffer array');
                  var reader = new FileReader, qitem = pthis.ReserveQueueItem();
                  reader.onload = function(event) {
                     // The file's text will be printed here
                     pthis.DoneItem(qitem, event.target.result, 0);
                  }
                  reader.readAsArrayBuffer(msg, e.offset || 0);
               } else {
                  // console.log('got array ' + (typeof msg) + ' len = ' + msg.byteLength);
                  // this is from CEF or LongPoll handler
                  pthis.ProvideData(msg, e.offset || 0);
               }

               return;
            }

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
            } else if (msg == "$$binary$$") {
               pthis.next_binary = true;
            } else if (msg == "$$nullbinary$$") {
               pthis.ProvideData(new ArrayBuffer(0), 0);
            } else {
               pthis.ProvideData(msg);
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
            console.log("websocket error " + err);
            if (pthis.state > 0) {
               pthis.InvokeReceiver('OnWebsocketError', err);
               pthis.state = 0;
            }
         }

         // only in interactive mode try to reconnect
         if (!JSROOT.BatchMode)
            setTimeout(retry_open, 3000); // after 3 seconds try again

      } // retry_open

      retry_open(true); // call for the first time
   }

   /** @summary Method used to initialize connection to web window.
    *
    * @param {object} arg - arguemnts
    * @param {string} [arg.prereq] - prerequicities, which should be loaded
    * @param {string} [arg.openui5src] - source of openui5, either URL like "https://openui5.hana.ondemand.com" or "jsroot" which provides its own reduced openui5 package
    * @param {string} [arg.openui5libs] - list of openui5 libraries loaded, default is "sap.m, sap.ui.layout, sap.ui.unified"
    * @param {string} [arg.socket_kind] - kind of connection longpoll|websocket, detected automatically from URL
    * @param {object} arg.receiver - instance of receiver for websocket events, allows to initiate connection immediately
    * @param {string} arg.first_recv - required prefix in the first message from TWebWindow, remain part of message will be returned as arg.first_msg
    * @param {string} [arg.prereq2] - second part of prerequcities, which is loaded parallel to connecting with WebWindow
    * @param {function} arg.callback - function which is called with WebWindowHandle or when establish connection and get first portion of data
    */

   JSROOT.ConnectWebWindow = function(arg) {
      if (typeof arg == 'function') arg = { callback: arg }; else
      if (!arg || (typeof arg != 'object')) arg = {};

      if (arg.prereq) {
         if (arg.openui5src) JSROOT.openui5src = arg.openui5src;
         if (arg.openui5libs) JSROOT.openui5libs = arg.openui5libs;
         return JSROOT.AssertPrerequisites(arg.prereq, function() {
            delete arg.prereq; JSROOT.ConnectWebWindow(arg);
         }, arg.prereq_logdiv);
      }

      // special hold script, prevents headless browser from too early exit
      if ((JSROOT.GetUrlOption("batch_mode")!==null) && JSROOT.GetUrlOption("key") && (JSROOT.browser.isChromeHeadless || JSROOT.browser.isChrome))
         JSROOT.loadScript("root_batch_holder.js?key=" + JSROOT.GetUrlOption("key"));

      if (!arg.platform)
         arg.platform = JSROOT.GetUrlOption("platform");

      if (arg.platform == "qt5") JSROOT.browser.qt5 = true; else
      if (arg.platform == "cef3") JSROOT.browser.cef3 = true;

      if (arg.batch === undefined)
         arg.batch = (JSROOT.GetUrlOption("batch_mode")!==null); //  && (JSROOT.browser.qt5 || JSROOT.browser.cef3 || JSROOT.browser.isChrome);

      if (arg.batch) JSROOT.BatchMode = true;

      if (!arg.socket_kind)
         arg.socket_kind = JSROOT.GetUrlOption("ws");

      if (!arg.socket_kind) {
         if (JSROOT.browser.qt5) arg.socket_kind = "rawlongpoll"; else
         if (JSROOT.browser.cef3) arg.socket_kind = "longpoll"; else arg.socket_kind = "websocket";
      }

      // only for debug purposes
      // arg.socket_kind = "longpoll";

      var handle = new WebWindowHandle(arg.socket_kind);

      if (window) {
         window.onbeforeunload = handle.Close.bind(handle, true);
         if (JSROOT.browser.qt5) window.onqt5unload = window.onbeforeunload;
      }

      if (arg.first_recv) {
         arg.receiver = {
            OnWebsocketOpened: function(handle) {
            },

            OnWebsocketMsg: function(handle, msg) {
                // console.log('Get message ' + msg + ' handle ' + !!handle);
                if (msg.indexOf(arg.first_recv)!=0)
                   return handle.Close();
                arg.first_msg = msg.substr(arg.first_recv.length);

                if (!arg.prereq2) JSROOT.CallBack(arg.callback, handle, arg);
            },

            OnWebsocketClosed: function(handle) {
               // when connection closed, close panel as well
               JSROOT.CloseCurrentWindow();
            }
         };
      }

      handle.key = JSROOT.GetUrlOption("key");

      if (!arg.receiver)
         return JSROOT.CallBack(arg.callback, handle, arg);

      // when receiver is exists, it handles itself callbacks
      handle.SetReceiver(arg.receiver);
      handle.Connect();

      if (arg.prereq2) {
         JSROOT.AssertPrerequisites(arg.prereq2, function() {
            delete arg.prereq2; // indicate that func is loaded
            if (!arg.first_recv || arg.first_msg) JSROOT.CallBack(arg.callback, handle, arg);
         });
      } else if (!arg.first_recv) {
         JSROOT.CallBack(arg.callback, handle, arg);
      }
   }

   // ========================================================================================

   /**
    * @summary Basic painter class.
    * @constructor
    * @memberof JSROOT
    */

   function TBasePainter() {
      this.divid = null; // either id of element (preferable) or element itself
   }

   /** @summary Access painter reference, stored in first child element.
    *
    *    - on === true - set *this* as painter
    *    - on === false - delete painter reference
    *    - on === undefined - return painter
    *
    * @param {boolean} on - that to perfrom
    * @private
    */
   TBasePainter.prototype.AccessTopPainter = function(on) {
      var main = this.select_main().node(),
         chld = main ? main.firstChild : null;
      if (!chld) return null;
      if (on===true) chld.painter = this; else
      if (on===false) delete chld.painter;
      return chld.painter;
   }

   /** @summary Generic method to cleanup painter */
   TBasePainter.prototype.Cleanup = function(keep_origin) {

      var origin = this.select_main('origin');
      if (!origin.empty() && !keep_origin) origin.html("");
      if (this._changed_layout)
         this.set_layout_kind('simple');
      this.AccessTopPainter(false);
      this.divid = null;
      delete this._selected_main;

      if (this._hpainter && typeof this._hpainter.ClearPainter === 'function') this._hpainter.ClearPainter(this);

      delete this._changed_layout;
      delete this._hitemname;
      delete this._hdrawopt;
      delete this._hpainter;
   }

   /** @summary Function should be called by the painter when first drawing is completed
    * @private */

   TBasePainter.prototype.DrawingReady = function(res_painter) {
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

   /** @summary Call back will be called when painter ready with the drawing
    * @private
    */
   TBasePainter.prototype.WhenReady = function(callback) {
      if (typeof callback !== 'function') return;
      if ('_ready_called_' in this) return JSROOT.CallBack(callback, this);
      if (this._ready_callback_ === undefined) this._ready_callback_ = [];
      this._ready_callback_.push(callback);
   }

   /** @summary Reset ready state - painter should again call DrawingReady to signal readyness
   * @private
   */
   TBasePainter.prototype.ResetReady = function() {
      delete this._ready_called_;
      delete this._ready_callback_;
   }

   /** @summary Returns drawn object
    *
    * @abstract
    */
   TBasePainter.prototype.GetObject = function() {
      return null;
   }

   /** @summary Returns true if type match with drawn object type
    * @abstract
    * @private
    */
   TBasePainter.prototype.MatchObjectType = function(typ) {
      return false;
   }

   /** @summary Called to update drawn object content
    * @abstract
    * @private */
   TBasePainter.prototype.UpdateObject = function(obj) {
      return false;
   }

   /** @summary Redraw all objects in current pad
    * @abstract
    * @private */
   TBasePainter.prototype.RedrawPad = function(resize) {
   }

   /** @summary Updates object and readraw it
    * @param {object} obj - new version of object, values will be updated in original object
    * @returns {boolean} true if object updated and redrawn */
   TBasePainter.prototype.RedrawObject = function(obj) {
      if (!this.UpdateObject(obj)) return false;
      var current = document.body.style.cursor;
      document.body.style.cursor = 'wait';
      this.RedrawPad();
      document.body.style.cursor = current;
      return true;
   }

   /** @summary Checks if draw elements were resized and drawing should be updated
    * @abstract
    * @private */
   TBasePainter.prototype.CheckResize = function(arg) {
      return false;
   }

   /** @summary Method called when interactively changes attribute in given class
    * @abstract
    * @private */
   TBasePainter.prototype.AttributeChange = function(class_name, member_name, new_value) {
      // function called when user interactively changes attribute in given class

      // console.log("Changed attribute", class_name, member_name, new_value);
   }

   /** @summary Returns d3.select for main element for drawing, defined with this.divid.
    *
    * @desc if main element was layouted, returns main element inside layout */
   TBasePainter.prototype.select_main = function(is_direct) {

      if (!this.divid) return d3.select(null);

      var res = this._selected_main;
      if (!res) {
         if (typeof this.divid == "string") {
            var id = this.divid;
            if (id[0]!='#') id = "#" + id;
            res = d3.select(id);
            if (!res.empty()) this.divid = res.node();
         } else {
            res = d3.select(this.divid);
         }
         this._selected_main = res;
      }

      if (!res || res.empty() || (is_direct==='origin')) return res;

      var use_enlarge = res.property('use_enlarge'),
          layout = res.property('layout') || 'simple',
          layout_selector = (layout=='simple') ? "" : res.property('layout_selector');

      if (layout_selector) res = res.select(layout_selector);

      // one could redirect here
      if (!is_direct && !res.empty() && use_enlarge) res = d3.select("#jsroot_enlarge_div");

      return res;
   }

   /** @summary Returns string with value of main element id attribute
   *
   * @desc if main element does not have id, it will be generated */
   TBasePainter.prototype.get_main_id = function() {
      var elem = this.select_main();
      if (elem.empty()) return "";
      var id = elem.attr("id");
      if (!id) {
         id = "jsroot_element_" + JSROOT.id_counter++;
         elem.attr("id", id);
      }
      return id;
   }

   /** @summary Returns layout kind
    * @private
    */
   TBasePainter.prototype.get_layout_kind = function() {
      var origin = this.select_main('origin'),
          layout = origin.empty() ? "" : origin.property('layout');

      return layout || 'simple';
   }

   /** @summary Set layout kind
    * @private
    */
   TBasePainter.prototype.set_layout_kind = function(kind, main_selector) {
      // change layout settings
      var origin = this.select_main('origin');
      if (!origin.empty()) {
         if (!kind) kind = 'simple';
         origin.property('layout', kind);
         origin.property('layout_selector', (kind!='simple') && main_selector ? main_selector : null);
         this._changed_layout = (kind !== 'simple'); // use in cleanup
      }
   }

   /** @summary Function checks if geometry of main div was changed.
    *
    * @desc returns size of area when main div is drawn
    * take into account enlarge state
    *
    * @private
    */
   TBasePainter.prototype.check_main_resize = function(check_level, new_size, height_factor) {

      var enlarge = this.enlarge_main('state'),
          main_origin = this.select_main('origin'),
          main = this.select_main(),
          lmt = 5; // minimal size

      if (enlarge !== 'on') {
         if (new_size && new_size.width && new_size.height)
            main_origin.style('width',new_size.width+"px")
                       .style('height',new_size.height+"px");
      }

      var rect_origin = this.get_visible_rect(main_origin, true),
          can_resize = main_origin.attr('can_resize'),
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

   /** @summary Try enlarge main drawing element to full HTML page.
    *
    * @desc Possible values for parameter:
    *
    *    - true - try to enlarge
    *    - false - cancel enlarge state
    *    - 'toggle' - toggle enlarge state
    *    - 'state' - return current state
    *    - 'verify' - check if element can be enlarged
    *
    * if action not specified, just return possibility to enlarge main div
    *
    * @private
    */
   TBasePainter.prototype.enlarge_main = function(action, skip_warning) {

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
               if (!skip_warning)
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

   /** @summary Return CSS value in given HTML element
    * @private */
   TBasePainter.prototype.GetStyleValue = function(elem, name) {
      if (!elem || elem.empty()) return 0;
      var value = elem.style(name);
      if (!value || (typeof value !== 'string')) return 0;
      value = parseFloat(value.replace("px",""));
      return isNaN(value) ? 0 : Math.round(value);
   }

   /** @summary Returns rect with width/height which correspond to the visible area of drawing region of element.
    * @private */
   TBasePainter.prototype.get_visible_rect = function(elem, fullsize) {

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

   /** @summary Assign painter to specified element
    *
    * @desc base painter does not creates canvas or frames
    * it registered in the first child element
    *
    * @param {string|object} divid - element ID or DOM Element
    */
   TBasePainter.prototype.SetDivId = function(divid) {
      if (divid !== undefined) {
         this.divid = divid;
         delete this._selected_main;
      }

      this.AccessTopPainter(true);
   }

   /** @summary Set item name, associated with the painter
    *
    * @desc Used by {@link JSROOT.HiearchyPainter}
    * @private
    */
   TBasePainter.prototype.SetItemName = function(name, opt, hpainter) {
      if (typeof name === 'string') this._hitemname = name;
                               else delete this._hitemname;
      // only upate draw option, never delete. null specified when update drawing
      if (typeof opt === 'string') this._hdrawopt = opt;

      this._hpainter = hpainter;
   }

   /** @summary Returns assigned item name
    * @private */
   TBasePainter.prototype.GetItemName = function() {
      return ('_hitemname' in this) ? this._hitemname : null;
   }

   /** @summary Returns assigned item draw option
    * @private */
   TBasePainter.prototype.GetItemDrawOpt = function() {
      return ('_hdrawopt' in this) ? this._hdrawopt : "";
   }

   /** @summary Check if it makes sense to zoom inside specified axis range
    *
    * @param {string} axis - name of axis like 'x', 'y', 'z'
    * @param {number} left - left axis range
    * @param {number} right - right axis range
    * @returns true is zooming makes sense
    * @abstract
    * @private
    */
   TBasePainter.prototype.CanZoomIn = function(axis,left,right) {
      return false;
   }

   // ==============================================================================

   /**
    * Basic painter for objects inside TCanvas/TPad.
    *
    * @constructor
    * @memberof JSROOT
    * @augments JSROOT.TBasePainter
    * @param {object} obj - object to draw
    */
   function TObjectPainter(obj, opt) {
      TBasePainter.call(this);
      this.draw_g = null; // container for all drawn objects
      this.pad_name = ""; // name of pad where object is drawn
      this.main = null;  // main painter, received from pad
      if (typeof opt == "string") this.options = { original: opt };
      this.AssignObject(obj);
   }

   TObjectPainter.prototype = Object.create(TBasePainter.prototype);

   TObjectPainter.prototype.AssignObject = function(obj) {
      this.draw_object = ((obj!==undefined) && (typeof obj == 'object')) ? obj : null;
   }

   /** @summary Generic method to cleanup painter.
    *
    * @desc Remove object drawing and in case of main painter - also main HTML components
    */
   TObjectPainter.prototype.Cleanup = function() {

      this.RemoveDrawG();

      var keep_origin = true;

      if (this.is_main_painter()) {
         var pp = this.pad_painter();
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
      delete this.root_colors;
      delete this.options;
      delete this.options_store;

      TBasePainter.prototype.Cleanup.call(this, keep_origin);
   }

   /** @summary Returns drawn object */
   TObjectPainter.prototype.GetObject = function() {
      return this.draw_object;
   }

   /** @summary Returns drawn object class name */
   TObjectPainter.prototype.GetClassName = function() {
      var res = this.draw_object ? this.draw_object._typename : "";
      return res || "";
   }

   /** @summary Checks if drawn object matches with provided typename
    *
    * @param {string} arg - typename
    * @param {string} arg._typename - if arg is object, use its typename
    */
   TObjectPainter.prototype.MatchObjectType = function(arg) {
      if (!arg || !this.draw_object) return false;
      if (typeof arg === 'string') return (this.draw_object._typename === arg);
      if (arg._typename) return (this.draw_object._typename === arg._typename);
      return this.draw_object._typename.match(arg);
   }

   /** @summary Changes item name.
    *
    * @desc When available, used for svg:title proprty
    * @private */
   TObjectPainter.prototype.SetItemName = function(name, opt, hpainter) {
      TBasePainter.prototype.SetItemName.call(this, name, opt, hpainter);
      if (this.no_default_title || (name=="")) return;
      var can = this.svg_canvas();
      if (!can.empty()) can.select("title").text(name);
                   else this.select_main().attr("title", name);
   }

   /** @summary Store actual options together with original string
    * @private */
   TObjectPainter.prototype.OptionsStore = function(original) {
      if (!this.options) return;
      this.options.original = original || "";
      this.options_store = JSROOT.extend({}, this.options);
   }

   /** @summary Checks if any draw options were changed
    *
    * @private
    */
   TObjectPainter.prototype.OptionesChanged = function() {
      if (!this.options) return false;
      if (!this.options_store) return true;

      for (var k in this.options)
         if (this.options[k] !== this.options_store[k]) return true;

      return false;
   }

   /** @summary Return actual draw options as string
    * @private
    */
   TObjectPainter.prototype.OptionsAsString = function() {
      if (!this.options) return "";

      if (!this.OptionesChanged())
         return this.options.original || "";

      if (typeof this.options.asString == "function")
         return this.options.asString();

      return this.options.original || ""; // nothing better, return original draw option
   }

   /** @summary Generic method to update object content.
    *
    * @desc Just copy all members from source object
    * @param {object} obj - object with new data
    */
   TObjectPainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;
      JSROOT.extend(this.GetObject(), obj);
      return true;
   }

   /** @summary Returns string which either item or object name.
    *
    * @desc Such string can be used as tooltip. If result string larger than 20 symbols, it will be cutted.
    * @private
    */
   TObjectPainter.prototype.GetTipName = function(append) {
      var res = this.GetItemName(), obj = this.GetObject();
      if (!res) res = obj && obj.fName ? obj.fName : "";
      if (res.lenght > 20) res = res.substr(0,17) + "...";
      if (res && append) res += append;
      return res;
   }

   /** @summary returns pad painter for specified pad
    * @private */
   TObjectPainter.prototype.pad_painter = function(pad_name) {
      var elem = this.svg_pad(typeof pad_name == "string" ? pad_name : undefined);
      return elem.empty() ? null : elem.property('pad_painter');
   }

   /** @summary returns canvas painter
    * @private */
   TObjectPainter.prototype.canv_painter = function() {
      var elem = this.svg_canvas();
      return elem.empty() ? null : elem.property('pad_painter');
   }

   /** @summary returns color from current list of colors
    * @private */
   TObjectPainter.prototype.get_color = function(indx) {
      var jsarr = this.root_colors;

      if (!jsarr) {
         var pp = this.canv_painter();
         jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : JSROOT.Painter.root_colors;
      }

      return jsarr[indx];
   }

   /** @summary add color to list of colors
    * @private */
   TObjectPainter.prototype.add_color = function(color) {
      var jsarr = this.root_colors;
      if (!jsarr) {
         var pp = this.canv_painter();
         jsarr = this.root_colors = (pp && pp.root_colors) ? pp.root_colors : JSROOT.Painter.root_colors;
      }
      var indx = jsarr.indexOf(color);
      if (indx >= 0) return indx;
      jsarr.push(color);
      return jsarr.length-1;
   }

   /** @summary returns tooltip allowed flag. Check canvas painter
    * @private */
   TObjectPainter.prototype.IsTooltipAllowed = function() {
      var src = this.canv_painter() || this;
      return src.tooltip_allowed ? true : false;
   }

   /** @summary returns tooltip allowed flag
    * @private */
   TObjectPainter.prototype.SetTooltipAllowed = function(on) {
      var src = this.canv_painter() || this;
      src.tooltip_allowed = (on == "toggle") ? !src.tooltip_allowed : on;
   }

   /** @summary returns custom palette for the object. If forced, will be created
    * @private */
   TObjectPainter.prototype.get_palette = function(force, palettedid) {
      if (!palettedid) {
         var pp = this.pad_painter();
         if (!pp) return null;
         if (pp.custom_palette) return pp.custom_palette;
      }

      var cp = this.canv_painter();
      if (!cp) return null;
      if (cp.custom_palette && !palettedid) return cp.custom_palette;

      if (force && JSROOT.Painter.GetColorPalette)
         cp.custom_palette = JSROOT.Painter.GetColorPalette(palettedid);

      return cp.custom_palette;
   }



   /** @summary Checks if draw elements were resized and drawing should be updated.
    *
    * @desc Redirects to {@link TPadPainter.CheckCanvasResize}
    * @private */
   TObjectPainter.prototype.CheckResize = function(arg) {
      var pad_painter = this.canv_painter();
      if (!pad_painter) return false;

      // only canvas should be checked
      pad_painter.CheckCanvasResize(arg);
      return true;
   }

   /** @summary removes <g> element with object drawing
    * @desc generic method to delete all graphical elements, associated with painter */
   TObjectPainter.prototype.RemoveDrawG = function() {
      if (this.draw_g) {
         this.draw_g.remove();
         this.draw_g = null;
      }
   }

   /** @summary recreates <g> element for object drawing
    * @desc obsolete function, will be removed soon
    * @private */
   TObjectPainter.prototype.RecreateDrawG = function(usepad, layer) {
      // keep old function for a while - later
      console.warn("Obsolete RecreateDrawG is used, will be removed soon. Change to CreateG");
      return this.CreateG(usepad ? undefined : layer);
   }

   /** @summary (re)creates svg:g element for object drawings
    *
    * @desc either one attach svg:g to pad list of primitives (default)
    * or svg:g element created in specified frame layer (default main_layer)
    * @param {string} [frame_layer=undefined] - when specified, <g> element will be created inside frame layer, otherwise in pad primitives list
    */
   TObjectPainter.prototype.CreateG = function(frame_layer) {
      if (this.draw_g) {
         // one should keep svg:g element on its place
         // d3.selectAll(this.draw_g.node().childNodes).remove();
         this.draw_g.selectAll('*').remove();
      } else if (frame_layer) {
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

   /** @summary This is main graphical SVG element, where all drawings are performed
    * @private */
   TObjectPainter.prototype.svg_canvas = function() {
      return this.select_main().select(".root_canvas");
   }

   /** @summary This is SVG element, correspondent to current pad
    * @private */
   TObjectPainter.prototype.svg_pad = function(pad_name) {
      if (pad_name === undefined) pad_name = this.pad_name;

      var c = this.svg_canvas();
      if (!pad_name || c.empty()) return c;

      var cp = c.property('pad_painter');
      if (cp.pads_cache && cp.pads_cache[pad_name])
         return d3.select(cp.pads_cache[pad_name]);

      c = c.select(".primitives_layer .__root_pad_" + pad_name);
      if (!cp.pads_cache) cp.pads_cache = {};
      cp.pads_cache[pad_name] = c.node();
      return c;
   }

   /** @summary Method selects immediate layer under canvas/pad main element
    * @private */
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

   /** @summary Method returns current pad name
    * @param {string} [new_name = undefined] - when specified, new current pad name will be configured
    * @private */
   TObjectPainter.prototype.CurrentPadName = function(new_name) {
      var svg = this.svg_canvas();
      if (svg.empty()) return "";
      var curr = svg.property('current_pad');
      if (new_name !== undefined) svg.property('current_pad', new_name);
      return curr;
   }

   /** @summary Returns ROOT TPad object
    * @private */
   TObjectPainter.prototype.root_pad = function() {
      var pad_painter = this.pad_painter();
      return pad_painter ? pad_painter.pad : null;
   }

   /** @summary Converts pad x or y coordinate into NDC value
    * @private */
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

   /** @summary Converts x or y coordinate into SVG pad or frame coordinates.
    *
    *  @param {string} axis - name like "x" or "y"
    *  @param {number} value - axis value to convert.
    *  @param {boolean|string} kind - false or undefined is coordinate inside frame, true - when NDC pad coordinates are used, "pad" - when pad coordinates relative to pad ranges are specified
    *  @returns {number} rounded value of requested coordiantes
    *  @private
    */
   TObjectPainter.prototype.AxisToSvg = function(axis, value, kind) {
      var main = this.frame_painter();
      if (main && !kind && main["gr"+axis]) {
         // this is frame coordinates
         value = (axis=="y") ? main.gry(value) + main.frame_y()
                             : main.grx(value) + main.frame_x();
      } else {
         if (kind !== true) value = this.ConvertToNDC(axis, value);
         value = (axis=="y") ? (1-value)*this.pad_height() : value*this.pad_width();
      }
      return Math.round(value);
   }

  /** @summary Return functor, which can convert x and y coordinates into pixels, used for drawing
   *
   * Produce functor can convert x and y value by calling func.x(x) and func.y(y)
   *  @param {boolean|string} kind - false or undefined is coordinate inside frame, true - when NDC pad coordinates are used, "pad" - when pad coordinates relative to pad ranges are specified
   *  @private
   */
  TObjectPainter.prototype.AxisToSvgFunc = function(kind) {
     var func = { kind: kind }, main = this.frame_painter();
     if (main && !kind && main.grx && main.gry) {
        func.main = main;
        func.offx = main.frame_x();
        func.offy = main.frame_y();
        func.x = function(x) { return Math.round(this.main.grx(x) + this.offx); }
        func.y = function(y) { return Math.round(this.main.gry(y) + this.offy); }
     } else {
        if (kind !== true) func.p = this; // need for NDC conversion
        func.padh = this.pad_height();
        func.padw = this.pad_width();
        func.x = function(x) { if (this.p) x = this.p.ConvertToNDC("x", x); return Math.round(x*this.padw); }
        func.y = function(y) { if (this.p) y = this.p.ConvertToNDC("y", y); return Math.round((1-y)*this.padh); }
     }
     return func;
  }


   /** @summary Returns svg element for the frame.
    *
    * @param {string} [pad_name = undefined] - optional pad name, otherwise where object painter is drawn
    * @private */
   TObjectPainter.prototype.svg_frame = function(pad_name) {
      return this.svg_layer("primitives_layer", pad_name).select(".root_frame");
   }

   /** @summary Returns pad width.
    *
    * @param {string} [pad_name = undefined] - optional pad name, otherwise where object painter is drawn
    * @private
    */
   TObjectPainter.prototype.pad_width = function(pad_name) {
      var res = this.svg_pad(pad_name);
      res = res.empty() ? 0 : res.property("draw_width");
      return isNaN(res) ? 0 : res;
   }

   /** @summary Returns pad height
    *
    * @param {string} [pad_name = undefined] - optional pad name, otherwise where object painter is drawn
    * @private
    */
   TObjectPainter.prototype.pad_height = function(pad_name) {
      var res = this.svg_pad(pad_name);
      res = res.empty() ? 0 : res.property("draw_height");
      return isNaN(res) ? 0 : res;
   }

   /** @summary Returns frame painter in current pad
    * @private */
   TObjectPainter.prototype.frame_painter = function() {
      var pp = this.pad_painter();
      return pp ? pp.frame_painter_ref : null;
   }

   /** @summary Returns property of the frame painter
    * @private */
   TObjectPainter.prototype.frame_property = function(name) {
      var pp = this.frame_painter();
      return pp && pp[name] ? pp[name] : 0;
   }

   /** @summary Returns frame X coordinate relative to current pad */
   TObjectPainter.prototype.frame_x = function() {
      return this.frame_property("_frame_x");
   }

   /** @summary Returns frame Y coordinate relative to current pad */
   TObjectPainter.prototype.frame_y = function() {
      return this.frame_property("_frame_y");
   }

   /** @summary Returns frame width */
   TObjectPainter.prototype.frame_width = function() {
      return this.frame_property("_frame_width");
   }

   /** @summary Returns frame height */
   TObjectPainter.prototype.frame_height = function() {
      return this.frame_property("_frame_height");
   }

   /** @summary Returns embed mode for 3D drawings (three.js) inside SVG.
    *
    *    - 0  no embedding, 3D drawing take full size of canvas
    *    - 1  no embedding, canvas placed over svg with proper size (resize problem may appear)
    *    - 2  normall embedding via ForeginObject, works only with Firefox
    *    - 3  embedding 3D drawing as SVG canvas, requires SVG renderer
    *    - 4  embed 3D drawing as <image> element
    *
    *  @private
    */
   TObjectPainter.prototype.embed_3d = function() {
      if (JSROOT.BatchMode) return 4; // in batch - directly create svg::image after rendering
      if (JSROOT.gStyle.Embed3DinSVG < 2) return JSROOT.gStyle.Embed3DinSVG;
      if (JSROOT.browser.isFirefox /*|| JSROOT.browser.isWebKit*/)
         return JSROOT.gStyle.Embed3DinSVG; // use specified mode
      return 1; // default is overlay
   }

   /** @summary Access current 3d mode
    *
    * @param {string} [new_value = undefined] - when specified, set new 3d mode
    * @private
    */
   TObjectPainter.prototype.access_3d_kind = function(new_value) {
      var svg = this.svg_pad(this.this_pad_name);
      if (svg.empty()) return -1;

      // returns kind of currently created 3d canvas
      var kind = svg.property('can3d');
      if (new_value !== undefined) svg.property('can3d', new_value);
      return ((kind===null) || (kind===undefined)) ? -1 : kind;
   }

   /** @summary Returns size which availble for 3D drawing.
    *
    * @desc One uses frame sizes for the 3D drawing - like TH2/TH3 objects
    * @private
    */
   TObjectPainter.prototype.size_for_3d = function(can3d) {

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

      var elem = pad, fp = this.frame_painter();
      if (can3d === 0) elem = this.svg_canvas();

      var size = { x: 0, y: 0, width: 100, height: 100, clname: clname, can3d: can3d };

      if (fp && !fp.mode3d) {
         elem = this.svg_frame();
         size.x = elem.property("draw_x");
         size.y = elem.property("draw_y");
      }

      size.width = elem.property("draw_width");
      size.height = elem.property("draw_height");

      if ((!fp || fp.mode3d) && (can3d > 0)) {
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

   /** @summary Clear all 3D drawings
    * @private */
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

   /** @summary Add 3D canvas
    * @private */
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

   /** @summary Apply size to 3D elements
    * @private */
   TObjectPainter.prototype.apply_3d_size = function(size, onlyget) {

      if (size.can3d < 0) return d3.select(null);

      var elem;

      if (size.can3d > 1) {

         elem = this.svg_layer(size.clname);

         // elem = layer.select("." + size.clname);
         if (onlyget) return elem;

         var svg = this.svg_pad();

         if ((size.can3d === 3) || (size.can3d === 4)) {
            // this is SVG mode or image mode - just create group to hold element

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

   /** @summary Returns main object painter on the pad.
    *
    * @desc Normally this is first histogram drawn on the pad, which also draws all axes
    * @param {boolean} [not_store = undefined] - if true, prevent temporary store of main painter reference
    * @param {string} [pad_name = undefined] - when specified, returns main painter from specified pad */
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

   /** @summary Returns true if this is main painter */
   TObjectPainter.prototype.is_main_painter = function() {
      return this === this.main_painter();
   }

   /** @summary Assigns id of top element (normally div where drawing is done).
    *
    * @desc In some situations canvas may not exists - for instance object drawn as html, not as svg.
    * In such case the only painter will be assigned to the first element
    *
    * Following value of is_main parameter is allowed:
    *    -1 - only assign id, this painter not add to painters list,
    *     0 - normal painter (default),
    *     1 - major objects like TH1/TH2 (required canvas with frame)
    *     2 - if canvas missing, create it, but not set as main object
    *     3 - if canvas and (or) frame missing, create them, but not set as main object
    *     4 - major objects like TH3 (required canvas and frame in 3d mode)
    *     5 - major objects like TGeoVolume (do not require canvas)
    *
    *  @param {string|object} divid - id of div element or directly DOMElement
    *  @param {number} [kind = 0] - kind of object drawn with painter
    *  @param {string} [pad_name = undefined] - when specified, subpad name used for object drawin
    */
   TObjectPainter.prototype.SetDivId = function(divid, is_main, pad_name) {

      if (divid !== undefined) {
         this.divid = divid;
         delete this._selected_main;
      }

      if (!is_main) is_main = 0;

      // check if element really exists
      if ((is_main >= 0) && this.select_main(true).empty()) {
         if (typeof divid == 'string') console.error('element with id ' + divid + ' not exists');
                                  else console.error('specified HTML element can not be selected with d3.select()');
         return;
      }

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
      if (this.svg_frame().select(".main_layer").empty() && ((is_main == 1) || (is_main == 3) || (is_main == 4))) {
         if (typeof JSROOT.Painter.drawFrame == 'function')
            JSROOT.Painter.drawFrame(divid, null, (is_main == 4) ? "3d" : "");
         if ((is_main != 4) && this.svg_frame().empty()) return alert("Fail to draw dummy TFrame");
      }

      var svg_p = this.svg_pad();
      if (svg_p.empty()) return;

      var pp = svg_p.property('pad_painter');
      if (pp && (pp !== this))
          pp.painters.push(this);

      if (((is_main === 1) || (is_main === 4) || (is_main === 5)) && !svg_p.property('mainpainter'))
         // when this is first main painter in the pad
         svg_p.property('mainpainter', this);
   }

   /** @summary Calculate absolute position of provided selection.
    * @private */
   TObjectPainter.prototype.CalcAbsolutePosition = function(sel, pos) {
      while (!sel.empty() && !sel.classed('root_canvas')) {
         var cl = sel.attr("class");
         if (cl && ((cl.indexOf("root_frame")>=0) || (cl.indexOf("__root_pad_")>=0))) {
            pos.x += sel.property("draw_x") || 0;
            pos.y += sel.property("draw_y") || 0;
         }
         sel = d3.select(sel.node().parentNode);
      }
      return pos;
   }

   /** @summary Creates marker attributes object.
    *
    * @desc Can be used to produce markers in painter.
    * See {@link JSROOT.TAttMarkerHandler} for more info.
    * Instance assigned as this.markeratt data member, recognized by GED editor
    * @param {object} args - either TAttMarker or see arguments of {@link JSROOT.TAttMarkerHandler}
    * @returns created handler
    */
   TObjectPainter.prototype.createAttMarker = function(args) {
      if (!args || (typeof args !== 'object')) args = { std: true }; else
      if (args.fMarkerColor!==undefined && args.fMarkerStyle!==undefined && args.fMarkerSize!==undefined) args = { attr: args, std: false };

      if (args.std === undefined) args.std = true;

      var handler = args.std ? this.markeratt : null;

      if (!handler) handler = new TAttMarkerHandler(args);
      else if (!handler.changed || args.force) handler.SetArgs(args);

      if (args.std) this.markeratt = handler;

      // handler.used = false; // mark that line handler is not yet used
      return handler;
   }


   /** @summary Creates line attributes object.
   *
   * @desc Can be used to produce lines in painter.
   * See {@link JSROOT.TAttLineHandler} for more info.
   * Instance assigned as this.lineatt data member, recognized by GED editor
   * @param {object} args - either TAttLine or see constructor arguments of {@link JSROOT.TAttLineHandler}
   */
   TObjectPainter.prototype.createAttLine = function(args) {
      if (!args || (typeof args !== 'object')) args = { std: true }; else
      if (args.fLineColor!==undefined && args.fLineStyle!==undefined && args.fLineWidth!==undefined) args = { attr: args, std: false };

      if (args.std === undefined) args.std = true;

      var handler = args.std ? this.lineatt : null;

      if (!handler) handler = new TAttLineHandler(args);
      else if (!handler.changed || args.force) handler.SetArgs(args);

      if (args.std) this.lineatt = handler;

      // handler.used = false; // mark that line handler is not yet used
      return handler;
   }

   /** @summary Creates fill attributes object.
    *
    * @desc Method dedicated to create fill attributes, bound to canvas SVG
    * otherwise newly created patters will not be usable in the canvas
    * See {@link JSROOT.TAttFillHandler} for more info.
    * Instance assigned as this.fillatt data member, recognized by GED editor

    * @param {object} args - for special cases one can specify TAttFill as args or number of parameters
    * @param {boolean} [args.std = true] - this is standard fill attribute for object and should be used as this.fillatt
    * @param {object} [args.attr = null] - object, derived from TAttFill
    * @param {number} [args.pattern = undefined] - integer index of fill pattern
    * @param {number} [args.color = undefined] - integer index of fill color
    * @param {string} [args.color_as_svg = undefined] - color will be specified as SVG string, not as index from color palette
    * @param {number} [args.kind = undefined] - some special kind which is handled differently from normal patterns
    * @returns created handle
   */
   TObjectPainter.prototype.createAttFill = function(args) {
      if (!args || (typeof args !== 'object')) args = { std: true }; else
      if (args._typename && args.fFillColor!==undefined && args.fFillStyle!==undefined) args = { attr: args, std: false };

      if (args.std === undefined) args.std = true;

      var handler = args.std ? this.fillatt : null;

      if (!args.svg) args.svg = this.svg_canvas();

      if (!handler) handler = new TAttFillHandler(args);
      else if (!handler.changed || args.force) handler.SetArgs(args);

      if (args.std) this.fillatt = handler;

      // handler.used = false; // mark that fill handler is not yet used

      return handler;
   }

   /** @summary call function for each painter in the pad
    * @private */
   TObjectPainter.prototype.ForEachPainter = function(userfunc, kind) {
      // Iterate over all known painters

      // special case of the painter set as pointer of first child of main element
      var painter = this.AccessTopPainter();
      if (painter) {
         if (kind !== "pads") userfunc(painter);
         return;
      }

      // iterate over all painters from pad list
      var pp = this.pad_painter();
      if (pp) pp.ForEachPainterInPad(userfunc, kind);
   }

   /** @summary indicate that redraw was invoked via interactive action (like context menu)
    * desc  use to catch such action by GED
    * @private */
   TObjectPainter.prototype.InteractiveRedraw = function(arg, info) {

      if (arg == "pad") this.RedrawPad(); else
      if (arg !== true) this.Redraw();

      // inform GED that something changes
      var pad_painter = this.pad_painter();
      if (pad_painter && pad_painter.InteractiveObjectRedraw)
         pad_painter.InteractiveObjectRedraw(this);

      // inform server that drawopt changes
      var canp = this.canv_painter();
      if (canp && canp.ProcessChanges)
         canp.ProcessChanges(info, this.pad_painter());
   }

   /** @summary Redraw all objects in correspondent pad */
   TObjectPainter.prototype.RedrawPad = function() {
      var pad_painter = this.pad_painter();
      if (pad_painter) pad_painter.Redraw();
   }

   /** @summary Switch tooltip mode in frame painter
    * @private */
   TObjectPainter.prototype.SwitchTooltip = function(on) {
      var fp = this.frame_painter();
      if (fp) fp.ProcessTooltipEvent(null, on);
      // this is 3D control object
      if (this.control && (typeof this.control.SwitchTooltip == 'function'))
         this.control.SwitchTooltip(on);
   }

   /** @summary Add drag interactive elements
    * @private */
   TObjectPainter.prototype.AddDrag = function(callback) {
      if (!JSROOT.gStyle.MoveResize) return;

      var pthis = this, drag_rect = null, pp = this.pad_painter();
      if (pp && pp._fast_drawing) return;

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
                     pthis.canv_painter().SelectObjectPainter(pthis);
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

   /** @summary Activate context menu via touch events
    * @private */
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

   /** @summary Close context menu, started via touch events
    * @private */
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

   /** @summary Add color selection menu entries
    * @private */
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

   /** @summary Add size selection menu entries
    * @private */
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

   /** @summary execute selected menu command, either locally or remotely
    * @private */
   TObjectPainter.prototype.ExecuteMenuCommand = function(method) {

      if (method.fName == "Inspect") {
         // primitve inspector, keep it here
         this.ShowInpsector();
         return true;
      }

      var canvp = this.canv_painter();
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

   /** @summary Fill object menu in web canvas
    * @private */
   TObjectPainter.prototype.FillObjectExecMenu = function(menu, kind, call_back) {

      var canvp = this.canv_painter();

      if (!this.snapid || !canvp || !canvp._websocket || canvp._getmenu_callback)
         return JSROOT.CallBack(call_back);

      function DoExecMenu(arg) {
         var execp = this.exec_painter || this,
             canvp = execp.canv_painter(),
             item = execp.args_menu_items[parseInt(arg)];

         if (!item || !item.fName) return;

         if ((item.fArgs!==undefined) && canvp.showMethodsDialog)
            return canvp.showMethodsDialog(execp, item, execp.args_menu_id);

         if ((item.fName == "Inspect") && canvp.showInspector)
            return canvp.showInspector(execp.GetObject());

         if (execp.ExecuteMenuCommand(item)) return;

         if (canvp._websocket && execp.args_menu_id) {
            console.log('execute method ' + item.fExec + ' for object ' + execp.args_menu_id);
            canvp.SendWebsocket('OBJEXEC:' + execp.args_menu_id + ":" + item.fExec);
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

      // if menu painter differs from this, remember it for further usage
      if (menu.painter)
         menu.painter.exec_painter = (menu.painter !== this) ? this : undefined;

      canvp._getmenu_callback = DoFillMenu.bind(this, menu, reqid, call_back);

      canvp.SendWebsocket('GETMENU:' + reqid); // request menu items for given painter

      setTimeout(canvp._getmenu_callback, 2000); // set timeout to avoid menu hanging
   }

   /** @summary remove all created draw attributes
    * @private */
   TObjectPainter.prototype.DeleteAtt = function() {
      delete this.lineatt;
      delete this.fillatt;
      delete this.markeratt;
   }

   /** @summary Fill context menu for graphical attributes
    * @private */
   TObjectPainter.prototype.FillAttContextMenu = function(menu, preffix) {
      // this method used to fill entries for different attributes of the object
      // like TAttFill, TAttLine, ....
      // all menu call-backs need to be rebind, while menu can be used from other painter

      if (!preffix) preffix = "";

      if (this.lineatt && this.lineatt.used) {
         menu.add("sub:"+preffix+"Line att");
         this.AddSizeMenuEntry(menu, "width", 1, 10, 1, this.lineatt.width,
                               function(arg) { this.lineatt.Change(undefined, parseInt(arg)); this.InteractiveRedraw(); }.bind(this));
         this.AddColorMenuEntry(menu, "color", this.lineatt.color,
                          function(arg) { this.lineatt.Change(arg); this.InteractiveRedraw(); }.bind(this));
         menu.add("sub:style", function() {
            var id = prompt("Enter line style id (1-solid)", 1);
            if (id == null) return;
            id = parseInt(id);
            if (isNaN(id) || !JSROOT.Painter.root_line_styles[id]) return;
            this.lineatt.Change(undefined, undefined, id);
            this.InteractiveRedraw();
         }.bind(this));
         for (var n=1;n<11;++n) {

            var dash = JSROOT.Painter.root_line_styles[n];

            var svg = "<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>" + n + "</text><line x1='30' y1='8' x2='100' y2='8' stroke='black' stroke-width='3' stroke-dasharray='" + dash + "'></line></svg>";

            menu.addchk((this.lineatt.style==n), svg, n, function(arg) { this.lineatt.Change(undefined, undefined, parseInt(arg)); this.InteractiveRedraw(); }.bind(this));
         }
         menu.add("endsub:");
         menu.add("endsub:");

         if (('excl_side' in this.lineatt) && (this.lineatt.excl_side!==0))  {
            menu.add("sub:Exclusion");
            menu.add("sub:side");
            for (var side=-1;side<=1;++side)
               menu.addchk((this.lineatt.excl_side==side), side, side, function(arg) {
                  this.lineatt.ChangeExcl(parseInt(arg));
                  this.InteractiveRedraw();
               }.bind(this));
            menu.add("endsub:");

            this.AddSizeMenuEntry(menu, "width", 10, 100, 10, this.lineatt.excl_width,
                  function(arg) { this.lineatt.ChangeExcl(undefined, parseInt(arg)); this.InteractiveRedraw(); }.bind(this));

            menu.add("endsub:");
         }
      }

      if (this.fillatt && this.fillatt.used) {
         menu.add("sub:"+preffix+"Fill att");
         this.AddColorMenuEntry(menu, "color", this.fillatt.colorindx,
               function(arg) { this.fillatt.Change(parseInt(arg), undefined, this.svg_canvas()); this.InteractiveRedraw(); }.bind(this), this.fillatt.kind);
         menu.add("sub:style", function() {
            var id = prompt("Enter fill style id (1001-solid, 3000..3010)", this.fillatt.pattern);
            if (id == null) return;
            id = parseInt(id);
            if (isNaN(id)) return;
            this.fillatt.Change(undefined, id, this.svg_canvas());
            this.InteractiveRedraw();
         }.bind(this));

         var supported = [1, 1001, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3010, 3021, 3022];

         for (var n=0; n<supported.length; ++n) {

            var sample = this.createAttFill({ std: false, pattern: supported[n], color: this.fillatt.colorindx || 1 }),
                svg = "<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>" + supported[n].toString() + "</text><rect x='40' y='0' width='60' height='18' stroke='none' fill='" + sample.fillcolor() + "'></rect></svg>";

            menu.addchk(this.fillatt.pattern == supported[n], svg, supported[n], function(arg) {
               this.fillatt.Change(undefined, parseInt(arg), this.svg_canvas());
               this.InteractiveRedraw();
            }.bind(this));
         }
         menu.add("endsub:");
         menu.add("endsub:");
      }

      if (this.markeratt && this.markeratt.used) {
         menu.add("sub:"+preffix+"Marker att");
         this.AddColorMenuEntry(menu, "color", this.markeratt.color,
                   function(arg) { this.markeratt.Change(arg); this.InteractiveRedraw(); }.bind(this));
         this.AddSizeMenuEntry(menu, "size", 0.5, 6, 0.5, this.markeratt.size,
               function(arg) { this.markeratt.Change(undefined, undefined, parseFloat(arg)); this.InteractiveRedraw(); }.bind(this));

         menu.add("sub:style");
         var supported = [1,2,3,4,5,6,7,8,21,22,23,24,25,26,27,28,29,30,31,32,33,34];

         for (var n=0; n<supported.length; ++n) {

            var clone = new TAttMarkerHandler({ style: supported[n], color: this.markeratt.color, size: 1.7 }),
                svg = "<svg width='60' height='18'><text x='1' y='12' style='font-size:12px'>" + supported[n].toString() + "</text><path stroke='black' fill='" + (clone.fill ? "black" : "none") + "' d='" + clone.create(40,8) + "'></path></svg>";

            menu.addchk(this.markeratt.style == supported[n], svg, supported[n],
                     function(arg) { this.markeratt.Change(undefined, parseInt(arg)); this.InteractiveRedraw(); }.bind(this));
         }
         menu.add("endsub:");
         menu.add("endsub:");
      }
   }

   /** @summary Fill context menu for text attributes
    * @private */
   TObjectPainter.prototype.TextAttContextMenu = function(menu, prefix) {
      // for the moment, text attributes accessed directly from objects

      var obj = this.GetObject();
      if (!obj || !('fTextColor' in obj)) return;

      menu.add("sub:" + (prefix ? prefix : "Text"));
      this.AddColorMenuEntry(menu, "color", obj.fTextColor,
            function(arg) { this.GetObject().fTextColor = parseInt(arg); this.InteractiveRedraw(); }.bind(this));

      var align = [11, 12, 13, 21, 22, 23, 31, 32, 33],
          hnames = ['left', 'centered' , 'right'],
          vnames = ['bottom', 'centered', 'top'];

      menu.add("sub:align");
      for (var n=0; n<align.length; ++n) {
         menu.addchk(align[n] == obj.fTextAlign,
                  align[n], align[n],
                  // align[n].toString() + "_h:" + hnames[Math.floor(align[n]/10) - 1] + "_v:" + vnames[align[n]%10-1], align[n],
                  function(arg) { this.GetObject().fTextAlign = parseInt(arg); this.InteractiveRedraw(); }.bind(this));
      }
      menu.add("endsub:");

      menu.add("sub:font");
      for (var n=1; n<16; ++n) {
         menu.addchk(n == Math.floor(obj.fTextFont/10), n, n,
                  function(arg) { this.GetObject().fTextFont = parseInt(arg)*10+2; this.InteractiveRedraw(); }.bind(this));
      }
      menu.add("endsub:");

      menu.add("endsub:");
   }

   /** @symmary Show object in inspector */
   TObjectPainter.prototype.ShowInpsector = function() {
      JSROOT.draw(this.divid, this.GetObject(), 'inspect');
   }

   /** @symmary Fill context menu for the object
    * @private */
   TObjectPainter.prototype.FillContextMenu = function(menu) {

      var title = this.GetTipName();
      if (this.GetObject() && ('_typename' in this.GetObject()))
         title = this.GetObject()._typename + "::" + title;

      menu.add("header:"+ title);

      this.FillAttContextMenu(menu);

      if (menu.size()>0) menu.add('Inspect', this.ShowInpsector);

      return menu.size() > 0;
   }

   /** @symmary returns function used to display object status
    * @private */
   TObjectPainter.prototype.GetShowStatusFunc = function() {
      // return function used to display object status
      // automatically disabled when drawing is enlarged - status line will be invisible

      var pp = this.canv_painter(), res = JSROOT.Painter.ShowStatus;

      if (pp && pp.use_openui && (typeof pp.fullShowStatus === 'function')) res = pp.fullShowStatus.bind(pp);

      if (res && (this.enlarge_main('state')==='on')) res = null;

      return res;
   }

   /** @symmary shows objects status
    * @private */
   TObjectPainter.prototype.ShowObjectStatus = function() {
      // method called normally when mouse enter main object element

      var obj = this.GetObject(),
          status_func = this.GetShowStatusFunc();

      if (obj && status_func) status_func(this.GetItemName() || obj.fName, obj.fTitle || obj._typename, obj._typename);
   }


   /** @summary try to find object by name in list of pad primitives
    * @desc used to find title drawing
    * @private */
   TObjectPainter.prototype.FindInPrimitives = function(objname) {

      var painter = this.pad_painter();
      if (!painter || !painter.pad) return null;

      if (painter.pad.fPrimitives)
         for (var n=0;n<painter.pad.fPrimitives.arr.length;++n) {
            var prim = painter.pad.fPrimitives.arr[n];
            if (('fName' in prim) && (prim.fName === objname)) return prim;
         }

      return null;
   }

   /** @summary Try to find painter for specified object
    * @desc can be used to find painter for some special objects, registered as
    * histogram functions
    * @private */
   TObjectPainter.prototype.FindPainterFor = function(selobj,selname,seltype) {

      var painter = this.pad_painter();
      var painters = painter ? painter.painters : null;
      if (!painters) return null;

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

   /** @summary Remove painter from list of painters and cleanup all drawings */
   TObjectPainter.prototype.DeleteThis = function() {
      var pp = this.pad_painter();
      if (pp) {
         var k = pp.painters.indexOf(this);
         if (k>=0) pp.painters.splice(k,1);
      }

      this.Cleanup();
   }

   /** @summary Configure user-defined tooltip callback
    *
    * @desc Hook for the users to get tooltip information when mouse cursor moves over frame area
    * call_back function will be called every time when new data is selected
    * when mouse leave frame area, call_back(null) will be called
    */

   TObjectPainter.prototype.ConfigureUserTooltipCallback = function(call_back, user_timeout) {

      if (!call_back || (typeof call_back !== 'function')) {
         delete this.UserTooltipCallback;
         delete this.UserTooltipTimeout;
         return;
      }

      if (user_timeout===undefined) user_timeout = 500;

      this.UserTooltipCallback = call_back;
      this.UserTooltipTimeout = user_timeout;
   }

   /** @summary Configure user-defined click handler
   *
   * @desc Function will be called every time when frame click was perfromed
   * As argument, tooltip object with selected bins will be provided
   * If handler function returns true, default handling of click will be disabled
   */

  TObjectPainter.prototype.ConfigureUserClickHandler = function(handler) {
     var fp = this.frame_painter();
     if (fp && typeof fp.ConfigureUserClickHandler == 'function')
        fp.ConfigureUserClickHandler(handler);
  }

   /** @summary Configure user-defined dblclick handler
   *
   * @desc Function will be called every time when double click was called
   * As argument, tooltip object with selected bins will be provided
   * If handler function returns true, default handling of dblclick (unzoom) will be disabled
   */

  TObjectPainter.prototype.ConfigureUserDblclickHandler = function(handler) {
     var fp = this.frame_painter();
     if (fp && typeof fp.ConfigureUserDblclickHandler == 'function')
        fp.ConfigureUserDblclickHandler(handler);
  }

   /** @summary Check if user-defined tooltip callback is configured
    * @returns {Boolean}
    * @private */
   TObjectPainter.prototype.IsUserTooltipCallback = function() {
      return typeof this.UserTooltipCallback == 'function';
   }

   /** @summary Provide tooltips data to user-defained function
    * @param {object} data - tooltip data
    * @private */
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

   /** @summary Redraw object
    *
    * @desc Basic method, should be reimplemented in all derived objects
    * for the case when drawing should be repeated
    * @abstract
    */

   TObjectPainter.prototype.Redraw = function() {
   }

   /** @summary Start text drawing
    *
    * @desc required before any text can be drawn
    */
   TObjectPainter.prototype.StartTextDrawing = function(font_face, font_size, draw_g, max_font_size) {
      // we need to preserve font to be able rescale at the end

      if (!draw_g) draw_g = this.draw_g;

      var font = (font_size==='font') ? font_face : JSROOT.Painter.getFontDetails(font_face, font_size);

      var pp = this.pad_painter();

      draw_g.call(font.func);

      draw_g.property('draw_text_completed', false)
            .property('text_font', font)
            .property('mathjax_use', false)
            .property('text_factor', 0.)
            .property('max_text_width', 0) // keep maximal text width, use it later
            .property('max_font_size', max_font_size)
            .property("_fast_drawing", pp && pp._fast_drawing);

      if (draw_g.property("_fast_drawing"))
         draw_g.property("_font_too_small", (max_font_size && (max_font_size<5)) || (font.size < 4));
   }

   /** @summary function used to remember maximal text scaling factor
    * @private */
   TObjectPainter.prototype.TextScaleFactor = function(value, draw_g) {
      if (!draw_g) draw_g = this.draw_g;
      if (value && (value > draw_g.property('text_factor'))) draw_g.property('text_factor', value);
   }

   /** @summary getBBox does not work in mozilla when object is not displayed or not visible :(
    * getBoundingClientRect() returns wrong sizes for MathJax
    * are there good solution?
    * @private */
   TObjectPainter.prototype.GetBoundarySizes = function(elem) {
      if (elem===null) { console.warn('empty node in GetBoundarySizes'); return { width:0, height:0 }; }
      var box = elem.getBoundingClientRect(); // works always, but returns sometimes results in ex values, which is difficult to use
      if (parseFloat(box.width) > 0) box = elem.getBBox(); // check that elements visible, request precise value
      var res = { width : parseInt(box.width), height : parseInt(box.height) };
      if ('left' in box) { res.x = parseInt(box.left); res.y = parseInt(box.right); } else
      if ('x' in box) { res.x = parseInt(box.x); res.y = parseInt(box.y); }
      return res;
   }

   /** @summary Finish text drawing
    *
    * @desc Should be called to complete all text drawing operations
    */
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
          max_sz = draw_g.property('max_font_size'),
          font_size = font.size;

      if ((f > 0) && ((f < 0.9) || (f > 1)))
         font.size = Math.floor(font.size/f);

      if (max_sz && (font.size > max_sz))
          font.size = max_sz;

      if (font.size != font_size) {
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
         } else if (!arg.plain && !arg.fast) {
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

   /** @ummary draw TLatex inside element
    *
    * @desc attempt to implement subset of TLatex with plain SVG text and tspan elements
    * @private
    */
   TObjectPainter.prototype.produceLatex = function(node, label, arg, curr) {

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

   /** @summary draw text
    *
    *  @param {object} arg - different text draw options
    *  @param {string} arg.text - text to draw
    *  @param {number} [arg.align = 12] - int value like 12 or 31
    *  @param {string} [arg.align = undefined] - end;bottom
    *  @param {number} [arg.x = 0] - x position
    *  @param {number} [arg.y = 0] - y position
    *  @param {number} [arg.width = undefined] - when specified, adjust font size in the specified box
    *  @param {number} [arg.height = undefined] - when specified, adjust font size in the specified box
    *  @param {number} arg.latex - 0 - plain text, 1 - normal TLatex, 2 - math
    *  @param {string} [arg.color=black] - text color
    *  @param {number} [arg.rotate = undefined] - rotaion angle
    *  @param {number} [arg.font_size = undefined] - fixed font size
    *  @param {object} [arg.draw_g = this.draw_g] - element where to place text, if not specified central painter container is used
    */
   TObjectPainter.prototype.DrawText = function(arg) {

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

      if (arg.draw_g.property("_fast_drawing")) {
         if (arg.scale) {
            // area too small - ignore such drawing
            if (arg.height < 4) return 0;
         } else if (arg.font_size) {
            // font size too small
            if (arg.font_size < 4) return 0;
         } else if (arg.draw_g.property("_font_too_small")) {
            // configure font is too small - ignore drawing
            return 0;
         }
      }

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
         arg.box = !JSROOT.nodejs && !JSROOT.gStyle.ApproxTextSize && !arg.fast ? this.GetBoundarySizes(txt.node()) :
                     (arg.text_rect || { height: arg.font_size*1.2, width: JSROOT.Painter.approxTextWidth(font, label) });

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

   /** @summary Finish MathJax drawing
    * @desc function should be called when processing of element is completed
    * @private
    */

   TObjectPainter.prototype.FinishMathjax = function(draw_g, fo_g, id) {

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

   /** @summary Set active pad painter
    *
    * @desc Should be used to handle key press events, which are global in the web browser
    *  @param {object} args - functions arguments
    *  @param {object} args.pp - pad painter
    *  @param {boolean} [args.active = false] - is pad activated or not
    * @private */
   Painter.SelectActivePad = function(args) {
      if (args.active) {
         if (this.$active_pp && (typeof this.$active_pp.SetActive == 'function'))
            this.$active_pp.SetActive(false);

         this.$active_pp = args.pp;

         if (this.$active_pp && (typeof this.$active_pp.SetActive == 'function'))
            this.$active_pp.SetActive(true);
      } else if (this.$active_pp === args.pp) {
         delete this.$active_pp;
      }
   }

   /** @summary Returns current active pad
    * @desc Should be used only for keyboard handling
    * @private */

   Painter.GetActivePad = function() {
      return this.$active_pp;
   }

   // =====================================================================

   function TooltipHandler(obj) {
      JSROOT.TObjectPainter.call(this, obj);
      this.tooltip_enabled = true;  // this is internally used flag to temporary disbale/enable tooltip
   }

   TooltipHandler.prototype = Object.create(TObjectPainter.prototype);

   TooltipHandler.prototype.hints_layer = function() {
      // return layer where frame tooltips are shown
      // only canvas info_layer can be used while other pads can overlay

      var pp = this.canv_painter();
      return pp ? pp.svg_layer("info_layer") : d3.select(null);
   }

   TooltipHandler.prototype.IsTooltipShown = function() {
      // return true if tooltip is shown, use to prevent some other action
      if (!this.tooltip_enabled || !this.IsTooltipAllowed()) return false;
      var hintsg = this.hints_layer().select(".objects_hints");
      return hintsg.empty() ? false : hintsg.property("hints_pad") == this.pad_name;
   }

   TooltipHandler.prototype.ProcessTooltipEvent = function(pnt, enabled) {
      // make central function which let show selected hints for the object

      if (enabled !== undefined) this.tooltip_enabled = enabled;

      if (pnt && pnt.handler) {
         // special use of interactive handler in the frame painter
         var rect = this.draw_g ? this.draw_g.select(".interactive_rect") : null;
         if (!rect || rect.empty()) {
            pnt = null; // disable
         } else if (pnt.touch) {
            var pos = d3.touches(rect.node());
            pnt = (pos && pos.length == 1) ? { touch: true, x: pos[0][0], y: pos[0][1] } : null;
         } else {
            var pos = d3.mouse(rect.node());
            pnt = { touch: false, x: pos[0], y: pos[1] };
         }
      }

      var hints = [], nhints = 0, maxlen = 0, lastcolor1 = 0, usecolor1 = false,
          textheight = 11, hmargin = 3, wmargin = 3, hstep = 1.2,
          frame_rect = this.GetFrameRect(),
          pad_width = this.pad_width(),
          pp = this.pad_painter(),
          font = JSROOT.Painter.getFontDetails(160, textheight),
          status_func = this.GetShowStatusFunc(),
          disable_tootlips = !this.IsTooltipAllowed() || !this.tooltip_enabled;

      if ((pnt === undefined) || (disable_tootlips && !status_func)) pnt = null;
      if (pnt && disable_tootlips) pnt.disabled = true; // indicate that highlighting is not required
      if (pnt) pnt.painters = true; // get also painter

      // collect tooltips from pad painter - it has list of all drawn objects
      if (pp) hints = pp.GetTooltips(pnt);

      if (pnt && pnt.touch) textheight = 15;

      for (var n = 0; n < hints.length; ++n) {
         var hint = hints[n];
         if (!hint) continue;

         if (hint.painter && (hint.user_info!==undefined))
            if (hint.painter.ProvideUserTooltip(hint.user_info));

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
         var title = "", name = "", info = "",
             hint = null, best_dist2 = 1e10, best_hint = null,
             coordinates = pnt ? Math.round(pnt.x)+","+Math.round(pnt.y) : "";
         // try to select hint with exact match of the position when several hints available
         for (var k=0; k < (hints ? hints.length : 0); ++k) {
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
      hintsg.attr("transform", trans)
            .property("last_point", pnt)
            .property("hints_pad", this.pad_name);

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

         for (var l=0; l < (hint.lines ? hint.lines.length : 0); l++)
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

   JSROOT.EAxisBits = {
         kTickPlus      : JSROOT.BIT(9),
         kTickMinus     : JSROOT.BIT(10),
         kAxisRange     : JSROOT.BIT(11),
         kCenterTitle   : JSROOT.BIT(12),
         kCenterLabels  : JSROOT.BIT(14),
         kRotateTitle   : JSROOT.BIT(15),
         kPalette       : JSROOT.BIT(16),
         kNoExponent    : JSROOT.BIT(17),
         kLabelsHori    : JSROOT.BIT(18),
         kLabelsVert    : JSROOT.BIT(19),
         kLabelsDown    : JSROOT.BIT(20),
         kLabelsUp      : JSROOT.BIT(21),
         kIsInteger     : JSROOT.BIT(22),
         kMoreLogLabels : JSROOT.BIT(23),
         kDecimals      : JSROOT.BIT(11)
   };

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
         var txt = (this.txt._typename && (this.txt._typename == "TObjString")) ? this.txt.fString : this.txt.value;
         if (typeof txt != 'string') txt = "<undefined>";

         var mathjax = this.txt.mathjax || (JSROOT.gStyle.Latex == 4);

         if (!mathjax && !('as_is' in this.txt)) {
            var arr = txt.split("\n"); txt = "";
            for (var i = 0; i < arr.length; ++i)
               txt += "<pre style='margin:0'>" + arr[i] + "</pre>";
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

   /** @summary Register handle to react on window resize
    *
    * @desc function used to react on browser window resize event
    * While many resize events could come in short time,
    * resize will be handled with delay after last resize event
    * handle can be function or object with CheckResize function
    * one could specify delay after which resize event will be handled
    * @private
    */
   JSROOT.RegisterForResize = function(handle, delay) {

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

   JSROOT.addDrawFunc({ name: "TCanvas", icon: "img_canvas", prereq: "v6", func: "JSROOT.Painter.drawCanvas", opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" });
   JSROOT.addDrawFunc({ name: "TPad", icon: "img_canvas", prereq: "v6", func: "JSROOT.Painter.drawPad", opt: ";grid;gridx;gridy;tick;tickx;ticky;log;logx;logy;logz", expand_item: "fPrimitives" });
   JSROOT.addDrawFunc({ name: "TSlider", icon: "img_canvas", prereq: "v6", func: "JSROOT.Painter.drawPad" });
   JSROOT.addDrawFunc({ name: "TFrame", icon: "img_frame", prereq: "v6", func: "JSROOT.Painter.drawFrame" });
   JSROOT.addDrawFunc({ name: "TPave", icon: "img_pavetext", prereq: "v6;hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TPaveText", icon: "img_pavetext", prereq: "v6;hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TPavesText", icon: "img_pavetext", prereq: "v6;hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TPaveStats", icon: "img_pavetext", prereq: "v6;hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TPaveLabel", icon: "img_pavelabel", prereq: "v6;hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TDiamond", icon: "img_pavelabel", prereq: "v6;hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TLatex", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: "TMathText", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: "TText", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: /^TH1/, icon: "img_histo1d", prereq: "v6;hist", func: "JSROOT.Painter.drawHistogram1D", opt:";hist;P;P0;E;E1;E2;E3;E4;E1X0;L;LF2;B;B1;A;TEXT;LEGO;same", ctrl: "l" });
   JSROOT.addDrawFunc({ name: "TProfile", icon: "img_profile", prereq: "v6;hist", func: "JSROOT.Painter.drawHistogram1D", opt:";E0;E1;E2;p;AH;hist"});
   JSROOT.addDrawFunc({ name: "TH2Poly", icon: "img_histo2d", prereq: "v6;hist", func: "JSROOT.Painter.drawHistogram2D", opt:";COL;COL0;COLZ;LCOL;LCOL0;LCOLZ;LEGO;same", expand_item: "fBins", theonly: true });
   JSROOT.addDrawFunc({ name: "TProfile2Poly", sameas: "TH2Poly" });
   JSROOT.addDrawFunc({ name: "TH2PolyBin", icon: "img_histo2d", draw_field: "fPoly" });
   JSROOT.addDrawFunc({ name: /^TH2/, icon: "img_histo2d", prereq: "v6;hist", func: "JSROOT.Painter.drawHistogram2D", opt:";COL;COLZ;COL0;COL1;COL0Z;COL1Z;COLA;BOX;BOX1;SCAT;TEXT;CONT;CONT1;CONT2;CONT3;CONT4;ARR;SURF;SURF1;SURF2;SURF4;SURF6;E;A;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same", ctrl: "colz" });
   JSROOT.addDrawFunc({ name: "TProfile2D", sameas: "TH2" });
   JSROOT.addDrawFunc({ name: /^TH3/, icon: 'img_histo3d', prereq: "v6;hist3d", func: "JSROOT.Painter.drawHistogram3D", opt:";SCAT;BOX;BOX2;BOX3;GLBOX1;GLBOX2;GLCOL" });
   JSROOT.addDrawFunc({ name: "THStack", icon: "img_histo1d", prereq: "v6;hist", func: "JSROOT.Painter.drawHStack", expand_item: "fHists", opt: "NOSTACK;HIST;E;PFC;PLC" });
   JSROOT.addDrawFunc({ name: "TPolyMarker3D", icon: 'img_histo3d', prereq: "v6;hist3d", func: "JSROOT.Painter.drawPolyMarker3D" });
   JSROOT.addDrawFunc({ name: "TPolyLine3D", icon: 'img_graph', prereq: "3d", func: "JSROOT.Painter.drawPolyLine3D", direct: true });
   JSROOT.addDrawFunc({ name: "TGraphStruct" });
   JSROOT.addDrawFunc({ name: "TGraphNode" });
   JSROOT.addDrawFunc({ name: "TGraphEdge" });
   JSROOT.addDrawFunc({ name: "TGraphTime", icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraphTime", opt: "once;repeat;first", theonly: true });
   JSROOT.addDrawFunc({ name: "TGraph2D", icon:"img_graph", prereq: "v6;hist3d", func: "JSROOT.Painter.drawGraph2D", opt: ";P;PCOL", theonly: true });
   JSROOT.addDrawFunc({ name: "TGraph2DErrors", icon:"img_graph", prereq: "v6;hist3d", func: "JSROOT.Painter.drawGraph2D", opt: ";P;PCOL;ERR", theonly: true });
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
   JSROOT.addDrawFunc({ name: "TPaletteAxis", icon: "img_colz", prereq: "v6;hist", func: "JSROOT.Painter.drawPaletteAxis" });
   JSROOT.addDrawFunc({ name: "TWebPainting", icon: "img_graph", prereq: "more2d", func: "JSROOT.Painter.drawWebPainting" });
   JSROOT.addDrawFunc({ name: "TPadWebSnapshot", icon: "img_canvas", prereq: "v6", func: "JSROOT.Painter.drawPadSnapshot" });
   JSROOT.addDrawFunc({ name: "kind:Text", icon: "img_text", func: JSROOT.Painter.drawRawText });
   JSROOT.addDrawFunc({ name: "TObjString", icon: "img_text", func: JSROOT.Painter.drawRawText });
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
   JSROOT.addDrawFunc({ name: "TGaxis", icon: "img_graph", prereq: "v6", func: "JSROOT.Painter.drawGaxis" });
   JSROOT.addDrawFunc({ name: "TLegend", icon: "img_pavelabel", prereq: "v6;hist", func: "JSROOT.Painter.drawPave" });
   JSROOT.addDrawFunc({ name: "TBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TWbox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TSliderBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TAxis3D", prereq: "v6;hist3d", func: "JSROOT.Painter.drawAxis3D" });
   JSROOT.addDrawFunc({ name: "TMarker", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawMarker", direct: true });
   JSROOT.addDrawFunc({ name: "TPolyMarker", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawPolyMarker", direct: true });
   JSROOT.addDrawFunc({ name: "TASImage", icon: 'img_mgraph', prereq: "more2d", func: "JSROOT.Painter.drawASImage" });
   JSROOT.addDrawFunc({ name: "TJSImage", icon: 'img_mgraph', prereq: "more2d", func: "JSROOT.Painter.drawJSImage", opt: ";scale;center" });
   JSROOT.addDrawFunc({ name: "TGeoVolume", icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", expand: "JSROOT.GEO.expandObject", opt:";more;all;count;projx;projz;dflt", ctrl: "dflt" });
   JSROOT.addDrawFunc({ name: "TEveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;dflt", ctrl: "dflt"  });
   JSROOT.addDrawFunc({ name: "ROOT::Experimental::TEveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count;projx;projz;dflt", ctrl: "dflt"  });
   JSROOT.addDrawFunc({ name: "TGeoManager", icon: 'img_histo3d', prereq: "geom", expand: "JSROOT.GEO.expandObject", func: "JSROOT.Painter.drawGeoObject", opt: ";more;all;count;projx;projz;dflt", dflt: "expand", ctrl: "dflt" });
   JSROOT.addDrawFunc({ name: /^TGeo/, icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", opt: ";more;all;axis;compa;count;projx;projz;dflt", ctrl: "dflt" });
   // these are not draw functions, but provide extra info about correspondent classes
   JSROOT.addDrawFunc({ name: "kind:Command", icon: "img_execute", execute: true });
   JSROOT.addDrawFunc({ name: "TFolder", icon: "img_folder", icon2: "img_folderopen", noinspect: true, prereq: "hierarchy", expand: "JSROOT.Painter.FolderHierarchy" });
   JSROOT.addDrawFunc({ name: "TTask", icon: "img_task", prereq: "hierarchy", expand: "JSROOT.Painter.TaskHierarchy", for_derived: true });
   JSROOT.addDrawFunc({ name: "TTree", icon: "img_tree", prereq: "tree;more2d", expand: 'JSROOT.Painter.TreeHierarchy', func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" });
   JSROOT.addDrawFunc({ name: "TNtuple", icon: "img_tree", prereq: "tree;more2d", expand: 'JSROOT.Painter.TreeHierarchy', func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" });
   JSROOT.addDrawFunc({ name: "TNtupleD", icon: "img_tree", prereq: "tree;more2d", expand: 'JSROOT.Painter.TreeHierarchy', func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: "player;testio", shift: "inspect" });
   JSROOT.addDrawFunc({ name: "TBranchFunc", icon: "img_leaf_method", prereq: "tree;more2d", func: 'JSROOT.Painter.drawTree', opt: ";dump", noinspect: true });
   JSROOT.addDrawFunc({ name: /^TBranch/, icon: "img_branch", prereq: "tree;more2d", func: 'JSROOT.Painter.drawTree', dflt: "expand", opt: ";dump", ctrl: "dump", shift: "inspect", ignore_online: true });
   JSROOT.addDrawFunc({ name: /^TLeaf/, icon: "img_leaf", prereq: "tree;more2d", noexpand: true, func: 'JSROOT.Painter.drawTree', opt: ";dump", ctrl: "dump", ignore_online: true });
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

      var search = (kind.indexOf("ROOT.")==0) ? kind.substr(5) : "kind:"+kind, counter = 0;
      for (var i=0; i < JSROOT.DrawFuncs.lst.length; ++i) {
         var h = JSROOT.DrawFuncs.lst[i];
         if (typeof h.name == "string") {
            if (h.name != search) continue;
         } else {
            if (!search.match(h.name)) continue;
         }

         if (h.sameas !== undefined)
            return JSROOT.getDrawHandle("ROOT."+h.sameas, selector);

         if ((selector === null) || (selector === undefined)) {
            // store found handle in cache, can reuse later
            if (!(kind in JSROOT.DrawFuncs.cache)) JSROOT.DrawFuncs.cache[kind] = h;
            return h;
         } else if (typeof selector == 'string') {
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
         } else if (selector === counter) {
            return h;
         }
         ++counter;
      }

      return first;
   }

   /** @summary Scan streamer infos for derived classes
    * @desc Assign draw functions for such derived classes
    * @private */
   JSROOT.addStreamerInfos = function(lst) {
      if (!lst) return;

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

   /** @summary Provide draw settings for specified class or kind
    * @private
    */
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

   /** Returns array with supported draw options for the specified kind
    * @private */
   JSROOT.getDrawOptions = function(kind, selector) {
      return JSROOT.getDrawSettings(kind).opts;
   }

   /** @summary Returns true if provided object class can be drawn
    * @private */
   JSROOT.canDraw = function(classname) {
      return JSROOT.getDrawSettings("ROOT." + classname).opts !== null;
   }

   /**
    * @summary Draw object in specified HTML element with given draw options.
    *
    * @param {string|object} divid - id of div element to draw or directly DOMElement
    * @param {object} obj - object to draw, object type should be registered before in JSROOT
    * @param {string} opt - draw options separated by space, comma or semicolon
    * @param {function} drawcallback - function called when drawing is completed, first argument is object painter instance
    *
    * @desc
    * A complete list of options can be found depending of the object's ROOT class to draw: {@link https://root.cern/js/latest/examples.htm}
    *
    * @example
    * var filename = "https://root.cern/js/files/hsimple.root";
    * JSROOT.OpenFile(filename, function(file) {
    *    file.ReadObject("hpxpy;1", function(obj) {
    *       JSROOT.draw("drawing", obj, "colz;logx;gridx;gridy");
    *    });
    * });
    *
    */
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

      if ((obj === null) || (typeof obj !== 'object')) return completeDraw(null);

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

      if (!handle.func) {
         if (opt && (opt.indexOf("same")>=0)) {
            var main_painter = JSROOT.GetMainPainter(divid);
            if (main_painter && (typeof main_painter.PerformDrop === 'function'))
               return main_painter.PerformDrop(obj, "", null, opt, completeDraw);
         }

         return completeDraw(null);
      }

      function performDraw() {
         if (handle.direct) {
            painter = new TObjectPainter(obj, opt);
            painter.SetDivId(divid, 2);
            painter.Redraw = handle.func;
            painter.Redraw();
            painter.DrawingReady();
         } else {
            painter = handle.func(divid, obj, opt);

            if (painter && !painter.options) painter.options = { original: opt || "" };
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

   /**
    * @summary Redraw object in specified HTML element with given draw options.
    *
    * @desc If drawing was not drawn before, it will be performed with {@link JSROOT.draw}.
    * If drawing was already done, that content will be updated
    * @param {string|object} divid - id of div element to draw or directly DOMElement
    * @param {object} obj - object to draw, object type should be registered before in JSROOT
    * @param {string} opt - draw options
    * @param {function} callback - function called when drawing is completed, first argument will be object painter instance
    */
   JSROOT.redraw = function(divid, obj, opt, callback) {
      if (!obj) return JSROOT.CallBack(callback, null);

      var dummy = new TObjectPainter();
      dummy.SetDivId(divid, -1);
      var can_painter = dummy.canv_painter();

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

   /** @summary Save object, drawn in specified element, as JSON.
    *
    * @desc Normally it is TCanvas object with list of primitives
    * @param {string|object} divid - id of top div element or directly DOMElement
    * @returns {string} produced JSON string
    */

   JSROOT.StoreJSON = function(divid) {
      var p = new TObjectPainter;
      p.SetDivId(divid,-1);

      var canp = p.canv_painter();
      return canp ? canp.ProduceJSON() : "";
   }


   /** @summary Create SVG image for provided object.
    *
    * @desc Function especially useful in Node.js environment to generate images for
    * supported ROOT classes
    *
    * @param {object} args - contains different settings
    * @param {object} args.object - object for the drawing
    * @param {string} [args.option] - draw options
    * @param {number} [args.width = 1200] - image width
    * @param {number} [args.height = 800] - image height
    * @param {function} callback called with svg code as string argument
    */
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

            var has_workarounds = JSROOT.Painter.ProcessSVGWorkarounds && JSROOT.svg_workaround;

            main.select('svg').attr("xmlns", "http://www.w3.org/2000/svg")
                              .attr("xmlns:xlink", "http://www.w3.org/1999/xlink")
                              .attr("width", args.width)
                              .attr("height", args.height)
                              .attr("style", null).attr("class", null).attr("x", null).attr("y", null);

            var svg = main.html();

            if (JSROOT.nodejs)
               svg = svg.replace(/xlink_href_nodejs=/g,"xlink:href=");

            if (has_workarounds)
               svg = JSROOT.Painter.ProcessSVGWorkarounds(svg);

            svg = svg.replace(/url\(\&quot\;\#(\w+)\&quot\;\)/g,"url(#$1)")        // decode all URL
                     .replace(/ class=\"\w*\"/g,"")                                // remove all classes
                     .replace(/<g transform=\"translate\(\d+\,\d+\)\"><\/g>/g,"")  // remove all empty groups with transform
                     .replace(/<g><\/g>/g,"");                                     // remove all empty groups

            if (svg.indexOf("xlink:href")<0)
               svg = svg.replace(/ xmlns:xlink=\"http:\/\/www.w3.org\/1999\/xlink\"/g,"");

            main.remove();

            JSROOT.CallBack(callback, svg);
         });
      }

      if (!JSROOT.nodejs) {
         build(d3.select(window.document).append("div").style("visible", "hidden"));
      } else if (JSROOT.nodejs_document) {
         build(JSROOT.nodejs_window.d3.select('body').append('div'));
      } else {
         // use eval while old minifier is not able to parse newest Node.js syntax
         eval('const { JSDOM } = require("jsdom"); JSROOT.nodejs_window = (new JSDOM("<!DOCTYPE html>hello")).window;');
         JSROOT.nodejs_document = JSROOT.nodejs_window.document; // used with three.js
         JSROOT.nodejs_window.d3 = d3.select(JSROOT.nodejs_document); //get d3 into the dom
         build(JSROOT.nodejs_window.d3.select('body').append('div'));
      }
   }

   /**
    * @summary Check resize of drawn element
    *
    * @desc As first argument divid one should use same argument as for the drawing
    * As second argument, one could specify "true" value to force redrawing of
    * the element even after minimal resize of the element
    * Or one just supply object with exact sizes like { width:300, height:200, force:true };
    * @param {string|object} divid - id or DOM element
    * @param {boolean|object} arg - options on how to resize
    *
    * @example
    * JSROOT.resize("drawing", { width: 500, height: 200 } );
    * JSROOT.resize(document.querySelector("#drawing"), true);
    */
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

   /**
    * For compatibility, see {@link JSROOT.resize}
    * @private
    */
   JSROOT.CheckElementResize = JSROOT.resize;

   /** @summary Returns main painter object for specified HTML element
    * @param {string|object} divid - id or DOM element
    */

   JSROOT.GetMainPainter = function(divid) {
      var dummy = new JSROOT.TObjectPainter();
      dummy.SetDivId(divid, -1);
      return dummy.main_painter(true);
   }

   /**
    * @summary Safely remove all JSROOT objects from specified element
    *
    * @param {string|object} divid - id or DOM element
    *
    * @example
    * JSROOT.cleanup("drawing");
    * JSROOT.cleanup(document.querySelector("#drawing"));
    */
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

   /** Display progress message in the left bottom corner.
    *
    * Previous message will be overwritten
    * if no argument specified, any shown messages will be removed
    * @param {string} msg - message to display
    * @param {number} tmout - optional timeout in milliseconds, after message will disappear
    * @private
    */
   JSROOT.progress = function(msg, tmout) {
      if (JSROOT.BatchMode || (typeof document === 'undefined')) return;
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

   /** Tries to close current browser tab
   *
   * Many browsers do not allow simple window.close() call,
   * therefore try several workarounds
   */

   JSROOT.CloseCurrentWindow = function() {
      if (!window) return;
      window.close();
      window.open('','_self').close();
   }

   Painter.createRootColors();

   JSROOT.LongPollSocket = LongPollSocket;
   JSROOT.WebWindowHandle = WebWindowHandle;
   JSROOT.DrawOptions = DrawOptions;
   JSROOT.ColorPalette = ColorPalette;
   JSROOT.TAttLineHandler = TAttLineHandler;
   JSROOT.TAttFillHandler = TAttFillHandler;
   JSROOT.TAttMarkerHandler = TAttMarkerHandler;
   JSROOT.TBasePainter = TBasePainter;
   JSROOT.TObjectPainter = TObjectPainter;
   JSROOT.TooltipHandler = TooltipHandler;

   return JSROOT;

}));
