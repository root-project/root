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


   JSROOT.DrawOptions = function(opt) {
      this.opt = opt && (typeof opt=="string") ? opt.toUpperCase().trim() : "";
      this.part = "";
   }

   JSROOT.DrawOptions.prototype.empty = function() {
      return this.opt.length === 0;
   }

   JSROOT.DrawOptions.prototype.check = function(name,postpart) {
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

   JSROOT.DrawOptions.prototype.partAsInt = function(offset, dflt) {
      var val = this.part.replace( /^\D+/g, '');
      val = val ? parseInt(val,10) : Number.NaN;
      return isNaN(val) ? (dflt || 0) : val + (offset || 0);
   }

   /**
    * @class JSROOT.Painter Holder of different functions and classes for drawing
    */
   JSROOT.Painter = {};

   JSROOT.Painter.createMenu = function(painter, maincallback) {
      // dummy functions, forward call to the jquery function
      document.body.style.cursor = 'wait';
      JSROOT.AssertPrerequisites('hierarchy;jq2d;openui5;', function() {
         document.body.style.cursor = 'auto';
         JSROOT.Painter.createMenu(painter, maincallback);
      });
   }

   JSROOT.Painter.closeMenu = function(menuname) {
      var x = document.getElementById(menuname || 'root_ctx_menu');
      if (x) { x.parentNode.removeChild(x); return true; }
      return false;
   }

   JSROOT.Painter.readStyleFromURL = function(url) {
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

      var mathjax = JSROOT.GetUrlOption("mathjax", url);
      if ((mathjax!==null) && (mathjax!="0")) JSROOT.gStyle.MathJax = 1;

      if (JSROOT.GetUrlOption("nomenu", url)!=null) JSROOT.gStyle.ContextMenu = false;
      if (JSROOT.GetUrlOption("noprogress", url)!=null) JSROOT.gStyle.ProgressBox = false;
      if (JSROOT.GetUrlOption("notouch", url)!=null) JSROOT.touches = false;

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

   JSROOT.Painter.Coord = {
      kCARTESIAN : 1,
      kPOLAR : 2,
      kCYLINDRICAL : 3,
      kSPHERICAL : 4,
      kRAPIDITY : 5
   }

   /** Function that generates all root colors */
   JSROOT.Painter.root_colors = [];

   JSROOT.Painter.createRootColors = function() {
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

      JSROOT.Painter.root_colors = colorMap;
   }

   JSROOT.Painter.MakeColorRGB = function(col) {
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

   JSROOT.Painter.adoptRootColors = function(objarr) {
      if (!objarr || !objarr.arr) return;

      for (var n = 0; n < objarr.arr.length; ++n) {
         var col = objarr.arr[n];
         if (!col || (col._typename != 'TColor')) continue;

         var num = col.fNumber;
         if ((num<0) || (num>4096)) continue;

         var rgb = JSROOT.Painter.MakeColorRGB(col);
         if (rgb == null) continue;

         if (JSROOT.Painter.root_colors[num] != rgb)
            JSROOT.Painter.root_colors[num] = rgb;
      }
   }

   JSROOT.Painter.root_line_styles = ["", "", "3,3", "1,2",
         "3,4,1,4", "5,3,1,3", "5,3,1,3,1,3,1,3", "5,5",
         "5,3,1,3,1,3", "20,5", "20,10,1,10", "1,3"];

   // Initialize ROOT markers
   JSROOT.Painter.root_markers =
         [ 0, 100,   8,   7,   0,  //  0..4
           9, 100, 100, 100, 100,  //  5..9
         100, 100, 100, 100, 100,  // 10..14
         100, 100, 100, 100, 100,  // 15..19
         100, 103, 105, 104,   0,  // 20..24
           3,   4,   2,   1, 106,  // 25..29
           6,   7,   5, 102, 101]; // 30..34

   /** Function returns the ready to use marker for drawing */
   JSROOT.Painter.createAttMarker = function(attmarker, style) {

      var marker_color = JSROOT.Painter.root_colors[attmarker.fMarkerColor];

      if (!style || (style<0)) style = attmarker.fMarkerStyle;

      var res = { x0: 0, y0: 0, color: marker_color, style: style, size: 8, scale: 1, stroke: true, fill: true, marker: "",  ndig: 0, used: true, changed: false };

      res.Change = function(color, style, size) {

         this.changed = true;

         if (color!==undefined) this.color = color;
         if ((style!==undefined) && (style>=0)) this.style = style;
         if (size!==undefined) this.size = size; else size = this.size;

         this.x0 = this.y0 = 0;

         this.reset_pos = function() {
            this.lastx = this.lasty = null;
         }

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

         var marker_kind = ((this.style>0) && (this.style<JSROOT.Painter.root_markers.length)) ? JSROOT.Painter.root_markers[this.style] : 100;
         var shape = marker_kind % 100;

         this.fill = (marker_kind>=100);

         switch(this.style) {
            case 1: this.size = 1; this.scale = 1; break;
            case 6: this.size = 2; this.scale = 1; break;
            case 7: this.size = 3; this.scale = 1; break;
            default: this.size = size; this.scale = 8;
         }

         size = this.size*this.scale;

         this.ndig = (size>7) ? 0 : ((size>2) ? 1 : 2);
         if (shape == 6) this.ndig++;
         var half = (size/2).toFixed(this.ndig), full = size.toFixed(this.ndig);

         switch(shape) {
         case 0: // circle
            this.x0 = -size/2;
            this.marker = "a"+half+","+half+" 0 1,0 "+full+",0a"+half+","+half+" 0 1,0 -"+full+",0z";
            break;
         case 1: // cross
            var d = (size/3).toFixed(res.ndig);
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
            this.marker = "l-"+ half+",-"+full+"h"+full+"z";
            break;
         case 5: // triangle-down
            this.y0 = -size/2;
            this.marker = "l-"+ half+","+full+"h"+full+"z";
            break;
         case 6: // star
            this.y0 = -size/2;
            this.marker = "l" + (size/3).toFixed(res.ndig)+","+full +
                         "l-"+ (5/6*size).toFixed(res.ndig) + ",-" + (5/8*size).toFixed(res.ndig) +
                         "h" + full +
                         "l-" + (5/6*size).toFixed(res.ndig) + "," + (5/8*size).toFixed(res.ndig) + "z";
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
            this.marker = "l"+half+",-"+half+"l"+half+","+half+"l-"+half+","+half + "z";
            break;
         }

         this.create = function(x,y) {
            return "M" + (x+this.x0).toFixed(this.ndig)+ "," + (y+this.y0).toFixed(this.ndig) + this.marker;
         }

         return true;
      }

      res.Apply = function(selection) {
         selection.style('stroke', this.stroke ? this.color : "none");
         selection.style('fill', this.fill ? this.color : "none");
      }

      res.func = res.Apply.bind(res);

      res.Change(marker_color, style, attmarker.fMarkerSize);

      res.changed = false;

      return res;
   }

   JSROOT.Painter.createAttLine = function(attline, borderw, can_excl) {

      var color = 'black', _width = 0, style = 0;
      if (typeof attline == 'string') {
         color = attline;
         if (color!=='none') _width = 1;
      } else
      if (typeof attline == 'object') {
         if ('fLineColor' in attline) color = JSROOT.Painter.root_colors[attline.fLineColor];
         if ('fLineWidth' in attline) _width = attline.fLineWidth;
         if ('fLineStyle' in attline) style = attline.fLineStyle;
      } else
      if ((attline!==undefined) && !isNaN(attline)) {
         color = JSROOT.Painter.root_colors[attline];
      }

      if (borderw!==undefined) _width = borderw;

      var line = {
          used: true, // can mark object if it used or not,
          color: color,
          width: _width,
          dash: JSROOT.Painter.root_line_styles[style]
      };

      if (_width==0) line.color = 'none';

      if (can_excl) {
         line.excl_side = 0;
         line.excl_width = 0;
         if (Math.abs(line.width) > 99) {
            // exclusion graph
            line.excl_side = (line.width < 0) ? -1 : 1;
            line.excl_width = Math.floor(line.width / 100) * 5;
            line.width = line.width % 100; // line width
         }

         line.ChangeExcl = function(side,width) {
            if (width !== undefined) this.excl_width = width;
            if (side !== undefined) {
               this.excl_side = side;
               if ((this.excl_width===0) && (this.excl_side!==0)) this.excl_width = 20;
            }
            this.changed = true;
         }
      }

      // if custom color number used, use lightgrey color to show lines
      if ((line.color === undefined) && (line.width>0))
         line.color = 'lightgrey';

      line.Apply = function(selection) {
         this.used = true;
         if (this.color=='none') {
            selection.style('stroke',null).style('stroke-width',null).style('stroke-dasharray',null);
         } else {
            selection.style('stroke',this.color).style('stroke-width',this.width);
            if (this.dash && (this.dash.length>0)) selection.style('stroke-dasharray',this.dash);
         }
      }

      line.Change = function(color, width, dash) {
         if (color !== undefined) this.color = color;
         if (width !== undefined) this.width = width;
         if (dash !== undefined) this.dash = dash;
         this.changed = true;
      }

      line.func = line.Apply.bind(line);

      return line;
   }

   JSROOT.Painter.clearCuts = function(chopt) {
      /* decode string "chopt" and remove graphical cuts */
      var left = chopt.indexOf('[');
      var right = chopt.indexOf(']');
      if ((left>=0) && (right>=0) && (left<right))
          for (var i = left; i <= right; ++i) chopt[i] = ' ';
      return chopt;
   }

   JSROOT.Painter.root_fonts = new Array('Arial', 'Times New Roman',
         'bTimes New Roman', 'biTimes New Roman', 'Arial',
         'oArial', 'bArial', 'boArial', 'Courier New',
         'oCourier New', 'bCourier New', 'boCourier New',
         'Symbol', 'Times New Roman', 'Wingdings', 'Symbol', 'Verdana');

   JSROOT.Painter.getFontDetails = function(fontIndex, size) {

      var res = { name: "Arial", size: Math.round(size || 11), weight: null, style: null },
          fontName = JSROOT.Painter.root_fonts[Math.floor(fontIndex / 10)] || "";

      while (fontName.length > 0) {
         if (fontName[0]==='b') res.weight = "bold"; else
         if (fontName[0]==='i') res.style = "italic"; else
         if (fontName[0]==='o') res.style = "oblique"; else break;
         fontName = fontName.substr(1);
      }

      if (fontName == 'Symbol')
         res.weight = res.style = null;

      res.name = fontName;

      res.SetFont = function(selection) {
         selection.attr("font-family", this.name)
                  .attr("font-size", this.size)
                  .attr("xml:space","preserve");
         if (this.weight)
            selection.attr("font-weight", this.weight);
         if (this.style)
            selection.attr("font-style", this.style);
      }

      res.asStyle = function(sz) {
         return (sz ? sz : this.size) + "px " + this.name;
      }

      res.stringWidth = function(svg, line) {
         /* compute the bounding box of a string by using temporary svg:text */
         var text = svg.append("svg:text")
                     .attr("xml:space","preserve")
                     .style("opacity", 0)
                     .text(line);
         this.SetFont(text);
         var w = text.node().getBBox().width;
         text.remove();
         return w;
      }

      res.func = res.SetFont.bind(res);

      return res;
   }

   JSROOT.Painter.chooseTimeFormat = function(awidth, ticks) {
      if (awidth < .5) return ticks ? "%S.%L" : "%M:%S.%L";
      if (awidth < 30) return ticks ? "%Mm%S" : "%H:%M:%S";
      awidth /= 60; if (awidth < 30) return ticks ? "%Hh%M" : "%d/%m %H:%M";
      awidth /= 60; if (awidth < 12) return ticks ? "%d-%Hh" : "%d/%m/%y %Hh";
      awidth /= 24; if (awidth < 15.218425) return ticks ? "%d/%m" : "%d/%m/%y";
      awidth /= 30.43685; if (awidth < 6) return "%d/%m/%y";
      awidth /= 12; if (awidth < 2) return ticks ? "%m/%y" : "%d/%m/%y";
      return "%Y";
   }

   JSROOT.Painter.getTimeFormat = function(axis) {
      var idF = axis.fTimeFormat.indexOf('%F');
      if (idF >= 0) return axis.fTimeFormat.substr(0, idF);
      return axis.fTimeFormat;
   }

   JSROOT.Painter.getTimeOffset = function(axis) {
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
      return dt.getTime();
   }

   JSROOT.Painter.superscript_symbols_map = {
       '1': '\xB9',
       '2': '\xB2',
       '3': '\xB3',
       'o': '\xBA',
       '0': '\u2070',
       'i': '\u2071',
       '4': '\u2074',
       '5': '\u2075',
       '6': '\u2076',
       '7': '\u2077',
       '8': '\u2078',
       '9': '\u2079',
       '+': '\u207A',
       '-': '\u207B',
       '=': '\u207C',
       '(': '\u207D',
       ')': '\u207E',
       'n': '\u207F',
       'a': '\xAA',
       'v': '\u2C7D',
       'h': '\u02B0',
       'j': '\u02B2',
       'r': '\u02B3',
       'w': '\u02B7',
       'y': '\u02B8',
       'l': '\u02E1',
       's': '\u02E2',
       'x': '\u02E3'
   }

   JSROOT.Painter.subscript_symbols_map = {
         '0': '\u2080',
         '1': '\u2081',
         '2': '\u2082',
         '3': '\u2083',
         '4': '\u2084',
         '5': '\u2085',
         '6': '\u2086',
         '7': '\u2087',
         '8': '\u2088',
         '9': '\u2089',
         '+': '\u208A',
         '-': '\u208B',
         '=': '\u208C',
         '(': '\u208D',
         ')': '\u208E',
         'a': '\u2090',
         'e': '\u2091',
         'o': '\u2092',
         'x': '\u2093',
         'ə': '\u2094',
         'h': '\u2095',
         'k': '\u2096',
         'l': '\u2097',
         'm': '\u2098',
         'n': '\u2099',
         'p': '\u209A',
         's': '\u209B',
         't': '\u209C',
         'j': '\u2C7C'
    }

   JSROOT.Painter.translateSuperscript = function(_exp) {
      var res = "";
      for (var n=0;n<_exp.length;++n)
         res += (this.superscript_symbols_map[_exp[n]] || _exp[n]);
      return res;
   }

   JSROOT.Painter.translateSubscript = function(_sub) {
      var res = "";
      for (var n=0;n<_sub.length;++n)
         res += (this.subscript_symbols_map[_sub[n]] || _sub[n]);
      return res;
   }

   JSROOT.Painter.formatExp = function(label) {
      var str = label.toLowerCase().replace('e+', 'x10@').replace('e-', 'x10@-'),
          pos = str.indexOf('@'),
          exp = JSROOT.Painter.translateSuperscript(str.substr(pos+1)),
          str = str.substr(0, pos);

      return ((str === "1x10") ? "10" : str) + exp;
   }

   JSROOT.Painter.symbols_map = {
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
      // math symbols

      '#sqrt': '\u221A',

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
      '#cbar': '\u007C',
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
      '#upoint': '\u22C5',
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
      '#arctop': '',
      '#lbar': '',
      '#arcbottom': '',
      '#void8': '',
      '#bottombar': '\u230A',
      '#arcbar': '',
      '#ltbar': '',
      '#AA': '\u212B',
      '#aa': '\u00E5',
      '#void06': '',
      '#GT': '\x3E',
      '#forall': '\u2200',
      '#exists': '\u2203',
      '#bar': '',
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
      '#box': '',
      '#Box': '',
      '#parallel': '',
      '#perp': '\u22A5',
      '#odot': '',
      '#left': '',
      '#right': ''
   };

   JSROOT.Painter.translateLaTeX = function(_string) {
      var str = _string, i;

      var lstr = str.match(/\^{(.*?)}/gi);
      if (lstr)
         for (i = 0; i < lstr.length; ++i)
            str = str.replace(lstr[i], JSROOT.Painter.translateSuperscript(lstr[i].substr(2, lstr[i].length-3)));

      lstr = str.match(/\_{(.*?)}/gi);
      if (lstr)
         for (i = 0; i < lstr.length; ++i)
            str = str.replace(lstr[i], JSROOT.Painter.translateSubscript(lstr[i].substr(2, lstr[i].length-3)));

      lstr = str.match(/\#sqrt{(.*?)}/gi);
      if (lstr)
         for (i = 0; i < lstr.length; ++i)
            str = str.replace(lstr[i], lstr[i].replace(' ', '').replace('#sqrt{', '#sqrt').replace('}', ''));

      for (i in JSROOT.Painter.symbols_map)
         str = str.replace(new RegExp(i,'g'), JSROOT.Painter.symbols_map[i]);

      // simple workaround for simple #splitline{first_line}{second_line}
      if ((str.indexOf("#splitline{")==0) && (str[str.length-1]=="}")) {
         var pos = str.indexOf("}{");
         if ((pos>0) && (pos === str.lastIndexOf("}{")))
            str = str.replace("}{", "\n ").slice(11, str.length-1)
      }

      return str.replace(/\^2/gi,'\xB2').replace(/\^3/gi,'\xB3');
   }

   JSROOT.Painter.isAnyLatex = function(str) {
      return (str.indexOf("#")>=0) || (str.indexOf("\\")>=0) || (str.indexOf("{")>=0);
   }

   JSROOT.Painter.math_symbols_map = {
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
   };

   JSROOT.Painter.translateMath = function(str, kind, color) {
      // function translate ROOT TLatex into MathJax format

      if (kind!=2) {
         for (var x in JSROOT.Painter.math_symbols_map)
            str = str.replace(new RegExp(x,'g'), JSROOT.Painter.math_symbols_map[x]);

         for (var x in JSROOT.Painter.symbols_map)
            str = str.replace(new RegExp(x,'g'), "\\" + x.substr(1));
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

   JSROOT.Painter.BuildSvgPath = function(kind, bins, height, ndig) {
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

   TBasePainter.prototype.Cleanup = function() {
      // generic method to cleanup painter

      this.layout_main('simple');
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
         if (!this._return_res_painter) res_painter = this;
                                   else delete this._return_res_painter;

         while (this._ready_callback_.length)
            JSROOT.CallBack(this._ready_callback_.shift(), res_painter);
         delete this._ready_callback_;
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
      // if main element was layout, returns main element inside layout

      if (!this.divid) return d3.select(null);
      var id = this.divid;
      if ((typeof id == "string") && (id[0]!='#')) id = "#" + id;
      var res = d3.select(id);
      if (res.empty() || (is_direct==='origin')) return res;

      var use_enlarge = res.property('use_enlarge'),
          layout = res.property('layout');

      if (layout && (layout !=="simple")) {
         switch(is_direct) {
            case 'header': res = res.select(".canvas_header"); break;
            case 'footer': res = res.select(".canvas_footer"); break;
            default: res = res.select(".canvas_main");
         }
      } else {
         if (typeof is_direct === 'string') return d3.select(null);
      }

      // one could redirect here
      if (!is_direct && !res.empty() && use_enlarge) res = d3.select("#jsroot_enlarge_div");

      return res;
   }

   TBasePainter.prototype.layout_main = function(kind) {

      kind = kind || "simple";

      // first extract all childs
      var origin = this.select_main('origin');
      if (origin.empty() || (origin.property('layout') === kind)) return false;

      var main = this.select_main(), lst = [];

      while (main.node().firstChild)
         lst.push(main.node().removeChild(main.node().firstChild));

      if (kind === "simple") {
         // simple layout - nothing inside
         origin.html("");
         main = origin;
      } else {

         // now create all necessary divs

         var maindiv = origin.html("")
                          .append("div")
                          .attr("class","jsroot")
                          .style('display','flex')
                          .style('flex-direction','column')
                          .style('width','100%')
                          .style('height','100%');

         var header = maindiv.append("div").attr('class','canvas_header').style('width','100%');

         main = maindiv.append("div")
                       .style('flex',1) // use all available vertical space in the parent div
                       .style('width','100%')
                       .style("position","relative") // one should use absolute position for
                       .attr("class", "canvas_main");

         var footer = maindiv.append("div").attr('class','canvas_footer').style('width','100%');
      }

      // now append all childs to the newmain
      for (var k=0;k<lst.length;++k)
         main.node().appendChild(lst[k]);

      origin.property('layout', kind);

      return lst.length > 0; // return true when layout changed and there are elements inside
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

      this.RemoveDrawG();

      // generic method to cleanup painters
      //if (this.is_main_painter())
      //   this.select_main().html("");

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

      TBasePainter.prototype.Cleanup.call(this);
   }

   TObjectPainter.prototype.GetObject = function() {
      return this.draw_object;
   }

   TObjectPainter.prototype.MatchObjectType = function(arg) {
      if ((arg === undefined) || (arg === null) || (this.draw_object===null)) return false;
      if (typeof arg === 'string') return this.draw_object._typename === arg;
      return (typeof arg === 'object') && (this.draw_object._typename === arg._typename);
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
      var res = this.GetItemName();
      if (res===null) res = "";
      if ((res.length === 0) && ('fName' in this.GetObject()))
         res = this.GetObject().fName;
      if (res.lenght > 20) res = res.substr(0,17)+"...";
      if ((res.length > 0) && (append!==undefined)) res += append;
      return res;
   }

   TObjectPainter.prototype.pad_painter = function(active_pad) {
      var can = active_pad ? this.svg_pad() : this.svg_canvas();
      return can.empty() ? null : can.property('pad_painter');
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

   /** function (re)creates svg:g element used for specific object drawings
     *  either one attached svg:g to pad (take_pad==true) or to the frame (take_pad==false)
     *  svg:g element can be attached to different layers */
   TObjectPainter.prototype.RecreateDrawG = function(take_pad, layer) {
      if (this.draw_g) {
         // one should keep svg:g element on its place
         // d3.selectAll(this.draw_g.node().childNodes).remove();
         this.draw_g.selectAll('*').remove();
      } else
      if (take_pad) {
         if (typeof layer != 'string') layer = "text_layer";
         if (layer[0] == ".") layer = layer.substr(1);
         this.draw_g = this.svg_layer(layer).append("svg:g");
      } else {
         if (typeof layer != 'string') layer = ".main_layer";
         if (layer[0] != ".") layer = "." + layer;
         this.draw_g = this.svg_frame().select(layer).append("svg:g");
      }

      // set attributes for debugging
      if (this.draw_object!==null) {
         this.draw_g.attr('objname', encodeURI(this.draw_object.fName || "name"));
         this.draw_g.attr('objtype', encodeURI(this.draw_object._typename || "type"));
      }

      return this.draw_g;
   }

   /** This is main graphical SVG element, where all Canvas drawing are performed */
   TObjectPainter.prototype.svg_canvas = function() {
      return this.select_main().select(".root_canvas");
   }

   /** This is SVG element, correspondent to current pad */
   TObjectPainter.prototype.svg_pad = function(pad_name) {
      var c = this.svg_canvas();
      if (pad_name === undefined) pad_name = this.pad_name;
      if (pad_name && !c.empty())
         c = c.select(".subpads_layer").select("[pad=" + pad_name + ']');
      return c;
   }

   /** Method selects immediate layer under canvas/pad main element */
   TObjectPainter.prototype.svg_layer = function(name, pad_name) {
      var svg = this.svg_pad(pad_name);
      if (svg.empty()) return svg;

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
    *  Parameters: axis should be "x" or "y", value to convert
    *  Always return rounded values */
   TObjectPainter.prototype.AxisToSvg = function(axis, value, isndc) {
      var main = this.main_painter();
      if (main && !isndc) {
         // this is frame coordinates
         value = (axis=="y") ? main.gry(value) + main.frame_y()
                             : main.grx(value) + main.frame_x();
      } else {
         if (!isndc) value = this.ConvertToNDC(axis, value);
         value = (axis=="y") ? (1-value)*this.pad_height() : value*this.pad_width();
      }
      return Math.round(value);
   }

   /** This is SVG element with current frame */
   TObjectPainter.prototype.svg_frame = function(pad_name) {
      return this.svg_pad(pad_name).select(".root_frame");
   }

   TObjectPainter.prototype.frame_painter = function() {
      var elem = this.svg_frame();
      var res = elem.empty() ? null : elem.property('frame_painter');
      return res ? res : null;
   }

   TObjectPainter.prototype.pad_width = function(pad_name) {
      var res = this.svg_pad(pad_name).property("draw_width");
      return isNaN(res) ? 0 : res;
   }

   TObjectPainter.prototype.pad_height = function(pad_name) {
      var res = this.svg_pad(pad_name).property("draw_height");
      return isNaN(res) ? 0 : res;
   }

   TObjectPainter.prototype.frame_x = function() {
      var res = this.svg_frame().property("draw_x");
      return isNaN(res) ? 0 : res;
   }

   TObjectPainter.prototype.frame_y = function() {
      var res = this.svg_frame().property("draw_y");
      return isNaN(res) ? 0 : res;
   }

   TObjectPainter.prototype.frame_width = function() {
      var res = this.svg_frame().property("draw_width");
      return isNaN(res) ? 0 : res;
   }

   TObjectPainter.prototype.frame_height = function() {
      var res = this.svg_frame().property("draw_height");
      return isNaN(res) ? 0 : res;
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

         var layer = this.svg_layer("special_layer");

         elem = layer.select("." + size.clname);
         if (onlyget) return elem;

         if (size.can3d === 3) {
            // this is SVG mode

            if (elem.empty())
               elem = layer.append("g").attr("class", size.clname);

            elem.attr("transform", "translate(" + size.x + "," + size.y + ")");

         } else {

            if (elem.empty())
               elem = layer.append("foreignObject").attr("class", size.clname);

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

      // fill kind can be 1 or 2
      // 1 means object drawing where combination fillcolor==0 and fillstyle==1001 means no filling
      // 2 means all other objects where such combination is white-color filling

      var fill = { color: "none", colorindx: 0, pattern: 0, used: true, kind: 2, changed: false };

      if (kind!==undefined) fill.kind = kind;

      fill.Apply = function(selection) {
         this.used = true;

         selection.style('fill', this.color);

         if ('opacity' in this)
            selection.style('opacity', this.opacity);

         if ('antialias' in this)
            selection.style('antialias', this.antialias);
      }
      fill.func = fill.Apply.bind(fill);

      fill.empty = function() {
         // return true if color not specified or fill style not specified
         return (this.color == 'none');
      };

      fill.Change = function(color, pattern, svg) {
         this.changed = true;

         if ((color !== undefined) && !isNaN(color))
            this.colorindx = color;

         if ((pattern !== undefined) && !isNaN(pattern)) {
            this.pattern = pattern;
            delete this.opacity;
            delete this.antialias;
         }

         if (this.pattern < 1001) {
            this.color = 'none';
            return true;
         }

         if ((this.pattern === 1001) && (this.colorindx===0) && (this.kind===1)) {
            this.color = 'none';
            return true;
         }

         this.color = JSROOT.Painter.root_colors[this.colorindx];
         if (typeof this.color != 'string') this.color = "none";

         if (this.pattern === 1001) return true;

         if ((this.pattern >= 4000) && (this.pattern <= 4100)) {
            // special transparent colors (use for subpads)
            this.opacity = (this.pattern - 4000)/100;
            return true;
         }

         if ((svg===undefined) || svg.empty() || (this.pattern < 3000) || (this.pattern > 3025)) return false;

         var id = "pat_" + this.pattern + "_" + this.colorindx;

         var defs = svg.select('.canvas_defs');
         if (defs.empty())
            defs = svg.insert("svg:defs",":first-child").attr("class","canvas_defs");

         var line_color = this.color;
         this.color = "url(#" + id + ")";
         this.antialias = false;

         if (!defs.select("."+id).empty()) return true;

         var patt = defs.append('svg:pattern').attr("id",id).attr("class",id).attr("patternUnits","userSpaceOnUse");

         switch (this.pattern) {
           case 3001:
             patt.attr("width", 2).attr("height", 2);
             patt.append('svg:rect').attr("x", 0).attr("y", 0).attr("width", 1).attr("height", 1);
             patt.append('svg:rect').attr("x", 1).attr("y", 1).attr("width", 1).attr("height", 1);
             break;
           case 3002:
             patt.attr("width", 4).attr("height", 2);
             patt.append('svg:rect').attr("x", 1).attr("y", 0).attr("width", 1).attr("height", 1);
             patt.append('svg:rect').attr("x", 3).attr("y", 1).attr("width", 1).attr("height", 1);
             break;
           case 3003:
             patt.attr("width", 4).attr("height", 4);
             patt.append('svg:rect').attr("x", 2).attr("y", 1).attr("width", 1).attr("height", 1);
             patt.append('svg:rect').attr("x", 0).attr("y", 3).attr("width", 1).attr("height", 1);
             break;
           case 3005:
             patt.attr("width", 8).attr("height", 8);
             patt.append("svg:line").attr("x1", 0).attr("y1", 0).attr("x2", 8).attr("y2", 8);
             break;
           case 3006:
             patt.attr("width", 4).attr("height", 4);
             patt.append("svg:line").attr("x1", 1).attr("y1", 0).attr("x2", 1).attr("y2", 3);
             break;
           case 3007:
             patt.attr("width", 4).attr("height", 4);
             patt.append("svg:line").attr("x1", 0).attr("y1", 1).attr("x2", 3).attr("y2", 1);
             break;
           case 3010: // bricks
             patt.attr("width", 10).attr("height", 10);
             patt.append("svg:line").attr("x1", 0).attr("y1", 2).attr("x2", 10).attr("y2", 2);
             patt.append("svg:line").attr("x1", 0).attr("y1", 7).attr("x2", 10).attr("y2", 7);
             patt.append("svg:line").attr("x1", 2).attr("y1", 0).attr("x2", 2).attr("y2", 2);
             patt.append("svg:line").attr("x1", 7).attr("y1", 2).attr("x2", 7).attr("y2", 7);
             patt.append("svg:line").attr("x1", 2).attr("y1", 7).attr("x2", 2).attr("y2", 10);
             break;
           case 3021: // stairs
           case 3022:
             patt.attr("width", 10).attr("height", 10);
             patt.append("svg:line").attr("x1", 0).attr("y1", 5).attr("x2", 5).attr("y2", 5);
             patt.append("svg:line").attr("x1", 5).attr("y1", 5).attr("x2", 5).attr("y2", 0);
             patt.append("svg:line").attr("x1", 5).attr("y1", 10).attr("x2", 10).attr("y2", 10);
             patt.append("svg:line").attr("x1", 10).attr("y1", 10).attr("x2", 10).attr("y2", 5);
             break;
           default: /* == 3004 */
             patt.attr("width", 8).attr("height", 8);
             patt.append("svg:line").attr("x1", 8).attr("y1", 0).attr("x2", 0).attr("y2", 8);
             break;
         }

         patt.selectAll('line').style('stroke',line_color).style("stroke-width",1);
         patt.selectAll('rect').style("fill",line_color);

         return true;
      }

      if ((attfill!==null) && (typeof attfill == 'object')) {
         if ('fFillStyle' in attfill) pattern = attfill.fFillStyle;
         if ('fFillColor' in attfill) color = attfill.fFillColor;
      }

      fill.Change(color, pattern, this.svg_canvas());

      fill.changed = false;

      return fill;
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

      var pthis = this;

      var rect_width = function() { return Number(pthis.draw_g.attr("width")); };
      var rect_height = function() { return Number(pthis.draw_g.attr("height")); };

      var acc_x = 0, acc_y = 0, pad_w = 1, pad_h = 1, drag_tm = null;

      function detectRightButton(event) {
         if ('buttons' in event) return event.buttons === 2;
         else if ('which' in event) return event.which === 3;
         else if ('button' in event) return event.button === 2;
         return false;
      }

      var resize_corner1 = this.draw_g.select('.resize_corner1');
      if (resize_corner1.empty())
         resize_corner1 = this.draw_g
                              .append("path")
                              .attr('class','resize_corner1')
                              .attr("d","M2,2 h15 v-5 h-20 v20 h5 Z");

      var resize_corner2 = this.draw_g.select('.resize_corner2');
      if (resize_corner2.empty())
         resize_corner2 = this.draw_g
                              .append("path")
                              .attr('class','resize_corner2')
                              .attr("d","M-2,-2 h-15 v5 h20 v-20 h-5 Z");

      resize_corner1.style('opacity',0).style('cursor',"nw-resize");

      resize_corner2.style('opacity',0).style('cursor',"se-resize")
                    .attr("transform", "translate(" + rect_width() + "," + rect_height() + ")");

      var drag_rect = null;

      function complete_drag() {
         drag_rect.style("cursor", "auto");

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

         resize_corner2.attr("transform", "translate(" + newwidth + "," + newheight + ")");

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

            acc_x = 0; acc_y = 0;
            pad_w = pthis.pad_width() - rect_width();
            pad_h = pthis.pad_height() - rect_height();

            drag_tm = new Date();

            drag_rect = d3.select(pthis.draw_g.node().parentNode).append("rect")
                 .classed("zoom", true)
                 .attr("x",  pthis.draw_g.attr("x"))
                 .attr("y", pthis.draw_g.attr("y"))
                 .attr("width", rect_width())
                 .attr("height", rect_height())
                 .style("cursor", "move")
                 .style("pointer-events","none"); // let forward double click to underlying elements
          }).on("drag", function() {
               if (drag_rect == null) return;

               d3.event.sourceEvent.preventDefault();

               var x = Number(drag_rect.attr("x")), y = Number(drag_rect.attr("y"));
               var dx = d3.event.dx, dy = d3.event.dy;

               if ((acc_x<0) && (dx>0)) { acc_x+=dx; dx=0; if (acc_x>0) { dx=acc_x; acc_x=0; }}
               if ((acc_x>0) && (dx<0)) { acc_x+=dx; dx=0; if (acc_x<0) { dx=acc_x; acc_x=0; }}
               if ((acc_y<0) && (dy>0)) { acc_y+=dy; dy=0; if (acc_y>0) { dy=acc_y; acc_y=0; }}
               if ((acc_y>0) && (dy<0)) { acc_y+=dy; dy=0; if (acc_y<0) { dy=acc_y; acc_y=0; }}

               if (x+dx<0) { acc_x+=(x+dx); x=0; } else
               if (x+dx>pad_w) { acc_x+=(x+dx-pad_w); x=pad_w; } else x+=dx;

               if (y+dy<0) { acc_y+=(y+dy); y = 0; } else
               if (y+dy>pad_h) { acc_y+=(y+dy-pad_h); y=pad_h; } else y+=dy;

               drag_rect.attr("x", x).attr("y", y);

               d3.event.sourceEvent.stopPropagation();
          }).on(prefix+"end", function() {
               if (drag_rect==null) return;

               d3.event.sourceEvent.preventDefault();

               if (complete_drag() === false)
                  if(callback['ctxmenu'] && ((new Date()).getTime() - drag_tm.getTime() > 600)) {
                     var rrr = resize_corner2.node().getBoundingClientRect();
                     pthis.ShowContextMenu('main', { clientX: rrr.left, clientY: rrr.top } );
                  }
            });

      drag_resize
        .on(prefix+"start", function() {
           if (detectRightButton(d3.event.sourceEvent)) return;

           d3.event.sourceEvent.stopPropagation();
           d3.event.sourceEvent.preventDefault();

           pthis.SwitchTooltip(false); // disable tooltip

           acc_x = 0; acc_y = 0;
           pad_w = pthis.pad_width();
           pad_h = pthis.pad_height();
           drag_rect = d3.select(pthis.draw_g.node().parentNode).append("rect")
                        .classed("zoom", true)
                        .attr("x", pthis.draw_g.attr("x"))
                        .attr("y", pthis.draw_g.attr("y"))
                        .attr("width", rect_width())
                        .attr("height", rect_height())
                        .style("cursor", d3.select(this).style("cursor"));
         }).on("drag", function() {
            if (drag_rect == null) return;

            d3.event.sourceEvent.preventDefault();

            var w = Number(drag_rect.attr("width")), h = Number(drag_rect.attr("height")),
                x = Number(drag_rect.attr("x")), y = Number(drag_rect.attr("y"));
            var dx = d3.event.dx, dy = d3.event.dy;
            if ((acc_x<0) && (dx>0)) { acc_x+=dx; dx=0; if (acc_x>0) { dx=acc_x; acc_x=0; }}
            if ((acc_x>0) && (dx<0)) { acc_x+=dx; dx=0; if (acc_x<0) { dx=acc_x; acc_x=0; }}
            if ((acc_y<0) && (dy>0)) { acc_y+=dy; dy=0; if (acc_y>0) { dy=acc_y; acc_y=0; }}
            if ((acc_y>0) && (dy<0)) { acc_y+=dy; dy=0; if (acc_y<0) { dy=acc_y; acc_y=0; }}

            if (d3.select(this).classed('resize_corner1')) {
               if (x+dx < 0) { acc_x += (x+dx); w += x; x = 0; } else
               if (w-dx < 0) { acc_x -= (w-dx); x += w; w = 0; } else { x+=dx; w-=dx; }
               if (y+dy < 0) { acc_y += (y+dy); h += y; y = 0; } else
               if (h-dy < 0) { acc_y -= (h-dy); y += h; h = 0; } else { y+=dy; h-=dy; }
            } else {
               if (x+w+dx > pad_w) { acc_x += (x+w+dx-pad_w); w = pad_w-x; } else
               if (w+dx < 0) { acc_x += (w+dx); w = 0; } else w += dx;
               if (y+h+dy > pad_h) { acc_y += (y+h+dy-pad_h); h = pad_h-y; } else
               if (h+dy < 0) { acc_y += (h+dy); h=0; } else h += dy;
            }

            drag_rect.attr("x", x).attr("y", y).attr("width", w).attr("height", h);

            d3.event.sourceEvent.stopPropagation();
         }).on(prefix+"end", function() {
            if (drag_rect == null) return;

            d3.event.sourceEvent.preventDefault();

            complete_drag();
         });

      if (!callback.only_resize)
         this.draw_g.style("cursor", "move").call(drag_move);

      resize_corner1.call(drag_resize);
      resize_corner2.call(drag_resize);
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


   JSROOT.LongPollSocket = function(addr) {

      this.path = addr;
      this.connid = null;
      this.req = null;

      this.nextrequest = function(data, kind) {
         var url = this.path;
         if (kind === "connect") {
            url+="?connect";
            this.connid = "connect";
         } else
         if (kind === "close") {
            if ((this.connid===null) || (this.connid==="close")) return;
            url+="?connection="+this.connid + "&close";
            this.connid = "close";
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

         var req = JSROOT.NewHttpRequest(url, "text", function(res) {
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

      this.processreq = function(res) {

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
         } else
         if (this.connid==="close") {
            if (typeof this.onclose == 'function') this.onclose();
            return;
         } else {
            if ((typeof this.onmessage==='function') && res)
               this.onmessage({ data: res });
         }
         if (!this.req) this.nextrequest("","dummy"); // send new poll request when necessary
      }

      this.send = function(str) { this.nextrequest(str); }

      this.close = function() { this.nextrequest("", "close"); }

      this.nextrequest("", "connect");

      return this;
   }

   JSROOT.Cef3QuerySocket = function(addr) {
      // make very similar to longpoll
      // create persistent CEF requests which could be use from client application at eny time

      if (!window || !('cefQuery' in window)) return null;

      this.path = addr;
      this.connid = null;

      this.nextrequest = function(data, kind) {
         var req = { request: "", persistent: false };
         if (kind === "connect") {
            req.request = "connect";
            req.persistent = true; // this initial request will be used for all messages from the server
            this.connid = "connect";
         } else
         if (kind === "close") {
            if ((this.connid===null) || (this.connid==="close")) return;
            req.request = this.connid + '::close';
            this.connid = "close";
         } else
         if ((this.connid===null) || (typeof this.connid!=='number')) {
            return console.error("No connection");
         } else {
            req.request = this.connid + '::post';
            if (data) req.request += "::" + data;
         }

         if (!req.request) return console.error("No CEF request");

         req.request = this.path + "::" + req.request; // URL always preceed any command

         req.onSuccess = this.onSuccess.bind(this);
         req.onFailure = this.onFailure.bind(this);

         window.cefQuery(req); // equvalent to req.send
      }

      this.onFailure = function(error_code, error_message) {
         console.log("CEF_ERR: " + error_code);
         if (typeof this.onerror === 'function') this.onerror("failure with connid " + (this.connid || "---"));
         this.connid = null;
      };

      this.onSuccess = function(response) {
         if (!response) return; // normal situation when request does not send any reply

         if (this.connid==="connect") {
            this.connid = parseInt(response);
            console.log('Get new CEF connection with id ' + this.connid);
            if (typeof this.onopen == 'function') this.onopen();
         } else
         if (this.connid==="close") {
            if (typeof this.onclose == 'function') this.onclose();
         } else {
            if ((typeof this.onmessage==='function') && response)
               this.onmessage({ data: response });
         }
      }

      this.send = function(str) { this.nextrequest(str); }

      this.close = function() { this.nextrequest("", "close"); }

      this.nextrequest("","connect");

      return this;
   }


   TObjectPainter.prototype.OpenWebsocket = function(socket_kind) {
      // create websocket for current object (canvas)
      // via websocket one recieved many extra information

      delete this._websocket;

      if (socket_kind=='cefquery' && (!window || !('cefQuery' in window))) socket_kind = 'longpoll';

      // this._websocket = conn;
      this._websocket_kind = socket_kind;
      this._websocket_opened = false;

      var pthis = this, sum1 = 0, sum2 = 0, cnt = 0;

      function retry_open(first_time) {

      if (pthis._websocket_opened) return;
      console.log("try open wensocket again");
      if (pthis._websocket) pthis._websocket.close();
      delete pthis._websocket;

      var path = window.location.href, conn = null;

      if (pthis._websocket_kind == 'cefquery') {
         var pos = path.indexOf("draw.htm");
         if (pos < 0) return;
         path = path.substr(0,pos);

         if (path.indexOf("rootscheme://rootserver")==0) path = path.substr(23);
         console.log('configure cefquery ' + path);
         conn = new JSROOT.Cef3QuerySocket(path);
      } else
      if ((pthis._websocket_kind !== 'longpoll') && first_time) {
         path = path.replace("http://", "ws://");
         path = path.replace("https://", "wss://");
         var pos = path.indexOf("draw.htm");
         if (pos < 0) return;
         path = path.substr(0,pos) + "root.websocket";
         console.log('configure websocket ' + path);
         conn = new WebSocket(path);
      } else {
         var pos = path.indexOf("draw.htm");
         if (pos < 0) return;
         path = path.substr(0,pos) + "root.longpoll";
         console.log('configure longpoll ' + path);
         conn = new JSROOT.LongPollSocket(path);
      }

      if (!conn) return;

      pthis._websocket = conn;

      conn.onopen = function() {
         console.log('websocket initialized');
         pthis._websocket_opened = true;
         conn.send('READY'); // indicate that we are ready to recieve JSON code (or any other big piece)
      }

      conn.onmessage = function (e) {
         var d = e.data;
         if (typeof d != 'string') return console.log("msg",d);

         if (d.substr(0,4)=='SNAP') {
            var snap = JSROOT.parse(d.substr(4));

            if (typeof pthis.RedrawPadSnap === 'function') {
               pthis.RedrawPadSnap(snap, function() {
                  var reply = pthis.GetAllRanges();
                  if (reply) console.log("ranges: " + reply);
                  conn.send(reply ? "RREADY:" + reply : "RREADY:" ); // send ready message back when drawing completed
               });
            } else {
               conn.send('READY'); // send ready message back
            }

         } else
         if (d.substr(0,4)=='JSON') {
            var obj = JSROOT.parse(d.substr(4));
            // console.log("get JSON ", d.length-4, obj._typename);
            var tm1 = new Date().getTime();
            pthis.RedrawObject(obj);
            var tm2 = new Date().getTime();
            sum1+=1;
            sum2+=(tm2-tm1);
            if (sum1>10) { console.log('Redraw ', Math.round(sum2/sum1)); sum1=sum2=0; }

            conn.send('READY'); // send ready message back
            // if (++cnt > 10) conn.close();

         } else
         if (d.substr(0,4)=='MENU') {
            var lst = JSROOT.parse(d.substr(4));
            console.log("get MENUS ", typeof lst, 'nitems', lst.length, d.length-4);
            conn.send('READY'); // send ready message back
            if (typeof pthis._getmenu_callback == 'function')
               pthis._getmenu_callback(lst);
         } else
         if (d.substr(0,4)=='SVG:') {
            var fname = d.substr(4);
            console.log('get request for SVG image ' + fname);

            var res = "<svg>anything_else</svg>";

            if (pthis.CreateSvg) res = pthis.CreateSvg();

            console.log('SVG size = ' + res.length);

            conn.send("DONESVG:" + fname + ":" + res);
         } else
         if (d.substr(0,4)=='PNG:') {
            var fname = d.substr(4);
            console.log('get request for PNG image ' + fname);

            pthis.ProduceImage(true, 'any.png', function(can) {
               var res = can.toDataURL('image/png');
               console.log('PNG size = ' + res.length);
               var separ = res.indexOf("base64,");
               if (separ>0)
                  conn.send("DONEPNG:" + fname + ":" + res.substr(separ+7));
               else
                  conn.send("DONEPNG:" + fname);
            });

         } else

         if (d.substr(0,7)=='GETIMG:') {
            // obsolete, can be removed

            console.log('d',d);

            d = d.substr(7);
            var p = d.indexOf(":"),
                id = d.substr(0,p),
                fname = d.substr(p+1);
            conn.send('READY'); // send ready message back

            console.log('GET REQUEST FOR SVG FILE', fname, id);

            var painter = pthis.FindSnap(id);
            if (!painter) console.log('not find snap ' + id);

         } else {
            if (d) console.log("unrecognized msg",d);
         }
      }

      conn.onclose = function() {
         console.log('websocket closed');
         delete pthis._websocket;
         if (pthis._websocket_opened) {
            pthis._websocket_opened = false;
            window.close(); // close window when socked disapper
         }
      }

      conn.onerror = function (err) {
         console.log("err "+err);
         // conn.close();
      }

      setTimeout(retry_open, 3000); // after 3 seconds try again

      } // retry_open

      retry_open(true); // after short timeout
      // retry_open(); // call for the first time
   }


   TObjectPainter.prototype.FillObjectExecMenu = function(menu, call_back) {

      var canvp = this.pad_painter();

      if (!this.snapid || !canvp || !canvp._websocket || canvp._getmenu_callback)
         return JSROOT.CallBack(call_back);

      function DoExecMenu(arg) {
         console.log('execute method ' + arg + ' for object ' + this.snapid);

         var canvp = this.pad_painter();

         if (canvp && canvp._websocket && this.snapid)
            canvp._websocket.send('OBJEXEC:' + this.snapid + ":" + arg);
      }

      function DoFillMenu(_menu, _call_back, items) {

         // avoid multiple call of the callback after timeout
         if (!canvp._getmenu_callback) return;
         delete canvp._getmenu_callback;

         if (items && items.length) {
            _menu.add("separator");
            _menu.add("sub:Online");

            for (var n=0;n<items.length;++n) {
               var item = items[n];
               if ((item.fChecked === undefined) || (item.fChecked < 0))
                  _menu.add(item.fName, item.fExec, DoExecMenu);
               else
                  _menu.addchk(item.fChecked, item.fName, item.fExec, DoExecMenu);
            }

            _menu.add("endsub:");
         }

         JSROOT.CallBack(_call_back);
      }


      canvp._getmenu_callback = DoFillMenu.bind(this, menu, call_back);

      canvp._websocket.send('GETMENU:' + this.snapid); // request menu items for given painter

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
            if (isNaN(id) || (JSROOT.Painter.root_line_styles[id] === undefined)) return;
            this.lineatt.Change(undefined, undefined, JSROOT.Painter.root_line_styles[id]);
            this.Redraw();
         }.bind(this));
         for (var n=1;n<11;++n) {
            var style = JSROOT.Painter.root_line_styles[n];

            var svg = "<svg width='100' height='18'><text x='1' y='12' style='font-size:12px'>" + n + "</text><line x1='30' y1='8' x2='100' y2='8' stroke='black' stroke-width='3' stroke-dasharray='" + style + "'></line></svg>";

            menu.addchk((this.lineatt.dash==style), svg, style, function(arg) { this.lineatt.Change(undefined, undefined, arg); this.Redraw(); }.bind(this));
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


   TObjectPainter.prototype.FillContextMenu = function(menu) {

      var title = this.GetTipName();
      if (this.GetObject() && ('_typename' in this.GetObject()))
         title = this.GetObject()._typename + "::" + title;

      menu.add("header:"+ title);

      this.FillAttContextMenu(menu);

      if (menu.size()>0)
         menu.add('Inspect', function() {
             JSROOT.draw(this.divid, this.GetObject(), 'inspect');
         });

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

      var font = JSROOT.Painter.getFontDetails(font_face, font_size);

      draw_g.call(font.func);

      draw_g.property('draw_text_completed', false)
            .property('text_font', font)
            .property('mathjax_use', false)
            .property('normaltext_use', false)
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
      } else {
         //if (!draw_g.property('normaltext_use') && JSROOT.browser.isFirefox && (font.size<20)) {
         //   // workaround for firefox, where mathjax has problem when font size too small
         //   font.size = 20;
         //   draw_g.call(font.func);
         //}
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

         fo_g.property('_valign', valign);

         if (!JSROOT.nodejs) {
            var box = painter.GetBoundarySizes(fo_g.node());
            width = 1.05*box.width; height = 1.05*box.height;
         }

         if (fo_g.property('_scale'))
            svg_factor = Math.max(svg_factor, width / fo_g.property('_width'), height / fo_g.property('_height'));
      });

      if (svgs)
      svgs.each(function() {
         var fo_g = d3.select(this);
         // only direct parent
         if (fo_g.node().parentNode !== draw_g.node()) return;

         var valign = fo_g.property('_valign'),
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

         if ((svg_factor > 0.) && valign) valign = valign/svg_factor;

         if (valign===null) valign = (font_size - mh)/2;

         var align = fo_g.property('_align'),
             rotate = fo_g.property('_rotate'),
             fo_w = fo_g.property('_width'),
             fo_h = fo_g.property('_height'),
             tr = { x: fo_g.property('_x'), y: fo_g.property('_y') };

         var sign = { x:1, y:1 }, nx = "x", ny = "y";
         if (rotate == 180) { sign.x = sign.y = -1; } else
         if ((rotate == 270) || (rotate == 90)) {
            sign.x = (rotate===270) ? -1 : 1;
            sign.y = -sign.x;
            nx = "y"; ny = "x"; // replace names to which align applied
         }

         if (!fo_g.property('_scale')) fo_w = fo_h = 0;

         if (align[0] == 'middle') tr[nx] += sign.x*(fo_w - mw)/2; else
         if (align[0] == 'end')    tr[nx] += sign.x*(fo_w - mw);

         if (align[1] == 'middle') tr[ny] += sign.y*(fo_h - mh)/2; else
         if (align[1] == 'bottom') tr[ny] += sign.y*(fo_h - mh); else
         if (align[1] == 'bottom-base') tr[ny] += sign.y*(fo_h - mh - valign);

         var trans = "translate("+tr.x+","+tr.y+")";
         if (rotate!==0) trans += " rotate("+rotate+",0,0)";

         fo_g.attr('transform', trans).attr('visibility', null);
      });

      // now hidden text after rescaling can be shown
      draw_g.selectAll('.hidden_text').attr('opacity', '1').classed('hidden_text',false);

      if (!call_ready) call_ready = draw_g.node().text_callback;
      draw_g.node().text_callback = null;

      draw_g.property('draw_text_completed', true);

      // if specified, call ready function
      JSROOT.CallBack(call_ready);

      return draw_g.property('max_text_width');
   }

   TObjectPainter.prototype.DrawText = function(align_arg, x, y, w, h, label, tcolor, latex_kind, draw_g) {

      if (!draw_g) draw_g = this.draw_g;
      var align;

      if (typeof align_arg == 'string') {
         align = align_arg.split(";");
         if (align.length==1) align.push('middle');
      } else {
         align = ['start', 'middle'];
         if ((align_arg / 10) >= 3) align[0] = 'end'; else
         if ((align_arg / 10) >= 2) align[0] = 'middle';
         if ((align_arg % 10) == 0) align[1] = 'bottom'; else
         if ((align_arg % 10) == 1) align[1] = 'bottom-base'; else
         if ((align_arg % 10) == 3) align[1] = 'top';
      }

      var scale = (w>0) && (h>0);

      if (latex_kind==null) latex_kind = 1;
      if (latex_kind<2)
         if (!JSROOT.Painter.isAnyLatex(label)) latex_kind = 0;

      var use_normal_text = ((JSROOT.gStyle.MathJax<1) && (latex_kind!==2)) || (latex_kind<1),
          font = draw_g.property('text_font');

      // only Firefox can correctly rotate incapsulated SVG, produced by MathJax
      // if (!use_normal_text && (h<0) && !JSROOT.browser.isFirefox) use_normal_text = true;

      if (use_normal_text) {
         if (latex_kind>0) label = JSROOT.Painter.translateLaTeX(label);

         var pos_x = x.toFixed(0), pos_y = y.toFixed(0), pos_dy = "", middleline = false;

         if (w>0) {
            // adjust x position when scale into specified rectangle
            if (align[0]=="middle") pos_x = (x+w*0.5).toFixed(0); else
            if (align[0]=="end") pos_x = (x+w).toFixed(0);
         }

         if (h>0) {
            if (align[1].indexOf('bottom')===0) pos_y = (y+h).toFixed(0); else
            if (align[1] == 'top') pos_dy = ".8em"; else {
               pos_y = (y + h/2 + 1).toFixed(0);
               if (JSROOT.browser.isIE) pos_dy = ".4em"; else middleline = true;
            }
         } else {
            if (align[1] == 'top') pos_dy = ".8em"; else
            if (align[1] == 'middle') {
               if (JSROOT.browser.isIE) pos_dy = ".4em"; else middleline = true;
            }
         }

         // use translate and then rotate to avoid complex sign calculations
         var trans = "translate("+pos_x+","+pos_y+")";
         if (!scale && (h<0)) trans += " rotate("+(-h)+",0,0)";

         var txt = draw_g.append("text")
                         .attr("text-anchor", align[0])
                         .attr("x", 0)
                         .attr("y", 0)
                         .attr("fill", tcolor ? tcolor : null)
                         .attr("transform", trans)
                         .text(label);
         if (pos_dy) txt.attr("dy", pos_dy);
         if (middleline) txt.attr("dominant-baseline", "middle");

         draw_g.property('normaltext_use', true);

         // workaround for Node.js - use primitive estimation of textbox size
         // later can be done with Node.js (via SVG) or with alternative implementation of jsdom
         var box = !JSROOT.nodejs ? this.GetBoundarySizes(txt.node()) :
                    { height: Math.round(font.size*1.2), width: Math.round(label.length*font.size*0.4) };

         if (scale) txt.classed('hidden_text',true).attr('opacity','0'); // hide rescale elements

         if (box.width > draw_g.property('max_text_width')) draw_g.property('max_text_width', box.width);
         if ((w>0) && scale) this.TextScaleFactor(1.05*box.width / w, draw_g);
         if ((h>0) && scale) this.TextScaleFactor(1.*box.height / h, draw_g);

         return box.width;
      }

      w = Math.round(w); h = Math.round(h);
      x = Math.round(x); y = Math.round(y);

      var rotate = 0;

      if (!scale && h<0) { rotate = Math.abs(h); h = 0; }

      var mtext = JSROOT.Painter.translateMath(label, latex_kind, tcolor),
          fo_g = draw_g.append("svg:g")
                       .attr('class', 'math_svg')
                       .attr('visibility','hidden')
                       .property('_x',x) // used for translation later
                       .property('_y',y)
                       .property('_width',w) // used to check scaling
                       .property('_height',h)
                       .property('_scale', scale)
                       .property('_rotate', rotate)
                       .property('_align', align);

      draw_g.property('mathjax_use', true);  // one need to know that mathjax is used

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
            jsroot_drawg: draw_g,
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

         MathJax.Hub.Typeset(element, ["FinishMathjax", painter, draw_g, fo_g]);

         MathJax.Hub.Queue(["FinishMathjax", painter, draw_g, fo_g]); // repeat once again, while Typeset not always invoke callback
      });

      return 0;
   }

   TObjectPainter.prototype.FinishMathjax = function(draw_g, fo_g, id) {
      // function should be called when processing of element is completed

      if (fo_g.node().parentNode !== draw_g.node()) return;
      var entry = fo_g.property('_element');
      if (!entry) return;

      var vvv = d3.select(entry).select("svg");
      if (vvv.empty()) return; // not yet finished

      fo_g.property('_element', null);

      vvv.remove();
      document.body.removeChild(entry);

      fo_g.append(function() { return vvv.node(); });

      this.FinishTextDrawing(draw_g); // check if all other elements are completed
   }


   // ===========================================================

   function TFramePainter(tframe) {
      TObjectPainter.call(this, tframe);
      this.tooltip_enabled = true;
      this.tooltip_allowed = (JSROOT.gStyle.Tooltip > 0);
   }

   TFramePainter.prototype = Object.create(TObjectPainter.prototype);

   TFramePainter.prototype.Shrink = function(shrink_left, shrink_right) {
      this.fX1NDC += shrink_left;
      this.fX2NDC -= shrink_right;
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
         if (tframe)
            this.fillatt = this.createAttFill(tframe);
         else
         if (pad)
            this.fillatt = this.createAttFill(null, pad.fFrameFillStyle, pad.fFrameFillColor);
         else
            this.fillatt = this.createAttFill(null, 1001, 0);

         // force white color for the frame
         // if (this.fillatt.color == 'none') this.fillatt.color = 'white';
      }

      if (this.lineatt === undefined)
         this.lineatt = JSROOT.Painter.createAttLine(tframe ? tframe : 'black');
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
      TObjectPainter.prototype.Cleanup.call(this);
   }

   TFramePainter.prototype.Redraw = function() {

      // first update all attributes from objects
      this.UpdateAttributes();

      var width = this.pad_width(),
          height = this.pad_height(),
          lm = Math.round(width * this.fX1NDC),
          w = Math.round(width * (this.fX2NDC - this.fX1NDC)),
          tm = Math.round(height * (1 - this.fY2NDC)),
          h = Math.round(height * (this.fY2NDC - this.fY1NDC));

      // this is svg:g object - container for every other items belonging to frame
      this.draw_g = this.svg_frame();
      if (this.draw_g.empty())
         return console.error('did not found frame layer');

      var top_rect = this.draw_g.select("rect"),
          main_svg = this.draw_g.select(".main_layer");

      if (main_svg.empty()) {
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
      }

      this.draw_g
     //        .attr("x", lm)
     //        .attr("y", tm)
     //        .attr("width", w)
     //        .attr("height", h)
             .property('frame_painter', this) // simple way to access painter via frame container
             .property('draw_x', lm)
             .property('draw_y', tm)
             .property('draw_width', w)
             .property('draw_height', h)
             .attr("transform", "translate(" + lm + "," + tm + ")");

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

      if ((JSROOT.gStyle.Tooltip === 0) || JSROOT.BatchMode)
         return tooltip_rect.remove();

      this.draw_g.attr("x", lm)
                 .attr("y", tm)
                 .attr("width", w)
                 .attr("height", h);

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
                .style('opacity',0).style('fill',"none").style("pointer-events","visibleFill")
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

      var hintsg = this.svg_layer("stat_layer").select(".objects_hints");
      // if tooltips were visible before, try to reconstruct them after short timeout
      if (!hintsg.empty() && (JSROOT.gStyle.Tooltip > 0))
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

   JSROOT.saveSvgAsPng = function(el, options, call_back) {
      JSROOT.AssertPrerequisites("savepng", function() {
         JSROOT.saveSvgAsPng(el, options, call_back);
      });
   }

   TFramePainter.prototype.IsTooltipShown = function() {
      // return true if tooltip is shown, use to prevent some other action
      if (JSROOT.gStyle.Tooltip < 1) return false;
      return ! (this.svg_layer("stat_layer").select(".objects_hints").empty());
   }

   TFramePainter.prototype.ProcessTooltipEvent = function(pnt, enabled) {

      if (enabled !== undefined) this.tooltip_enabled = enabled;

      var hints = [], nhints = 0, maxlen = 0, lastcolor1 = 0, usecolor1 = false,
          textheight = 11, hmargin = 3, wmargin = 3, hstep = 1.2,
          height = this.frame_height(),
          width = this.frame_width(),
          pad_width = this.pad_width(),
          frame_x = this.frame_x(),
          pp = this.pad_painter(true),
          maxhinty = this.pad_height() - this.draw_g.property('draw_y'),
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

      var layer = this.svg_layer("stat_layer"),
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

      // copy transform attributes from frame itself
      hintsg.attr("transform", this.draw_g.attr("transform"));

      hintsg.property("last_point", pnt);

      var viewmode = hintsg.property('viewmode');
      if (viewmode === undefined) viewmode = "";

      var actualw = 0, posx = pnt.x + 15;

      if (nhints > 1) {
         // if there are many hints, place them left or right

         var bleft = 0.5, bright = 0.5;

         if (viewmode=="left") bright = 0.7; else
         if (viewmode=="right") bleft = 0.3;

         if (pnt.x <= bleft*width) {
            viewmode = "left";
            posx = 20;
         } else
         if (pnt.x >= bright*width) {
            viewmode = "right";
            posx = width - 60;
         } else {
            posx = hintsg.property('startx');
         }
      } else {
         viewmode = "single";
      }

      if (viewmode !== hintsg.property('viewmode')) {
         hintsg.property('viewmode', viewmode);
         hintsg.selectAll("*").remove();
      }

      var curry = 10, // normal y coordinate
          gapy = 10,  // y coordinate, taking into account all gaps
          gapminx = -1111, gapmaxx = -1111;

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

         var was_empty = group.empty();

         if (was_empty)
            group = hintsg.append("svg:svg")
                          .attr("class", "painter_hint_"+n)
                          .attr('opacity',0) // use attribute, not style to make animation with d3.transition()
                          .style('overflow','hidden').style("pointer-events","none");

         if (viewmode == "single") {
            curry = pnt.touch ? (pnt.y - hint.height - 5) : Math.min(pnt.y + 15, maxhinty - hint.height - 3);
         } else {
            gapy = FindPosInGap(gapy);
            if ((gapminx === -1111) && (gapmaxx === -1111)) gapminx = gapmaxx = hint.x;
            gapminx = Math.min(gapminx, hint.x);
            gapmaxx = Math.min(gapmaxx, hint.x);
         }

         group.attr("x", posx)
              .attr("y", curry)
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

      if ((viewmode == "right") && (posx + actualw > width - 20)) {
         posx = width - actualw - 20;
         svgs.attr("x", posx);
      }

      if ((viewmode == "single") && (posx + actualw > pad_width - frame_x) && (posx > actualw+20)) {
         posx -= (actualw + 20);
         svgs.attr("x", posx);
      }

      // if gap not very big, apply gapy coordinate to open view on the histogram
      if ((viewmode !== "single") && (gapy < maxhinty) && (gapy !== curry))
         if ((gapminx <= posx+actualw+5) && (gapmaxx >= posx-5))
            svgs.attr("y", function() { return d3.select(this).property('gapy'); });

      if (actualw > 10)
         svgs.attr("width", actualw)
             .select('rect').attr("width", actualw);

      hintsg.property('startx', posx);
   }

   JSROOT.Painter.drawFrame = function(divid, obj) {
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

   TPadPainter.prototype.ForEachPainterInPad = function(userfunc, onlypadpainters) {

      userfunc(this);

      for (var k = 0; k < this.painters.length; ++k) {
         var sub =  this.painters[k];

         if (typeof sub.ForEachPainterInPad === 'function')
            sub.ForEachPainterInPad(userfunc, onlypadpainters);
         else
         if (!onlypadpainters) userfunc(sub);
      }
   }

   TPadPainter.prototype.ButtonSize = function(fact) {
      return Math.round((!fact ? 1 : fact) * (this.iscan || !this.has_canvas ? 16 : 12));
   }

   TPadPainter.prototype.ToggleEventStatus = function() {
      // when function called, jquery should be already loaded

      if (this.enlarge_main('state')==='on') return;

      this.has_event_status = !this.has_event_status;
      if (JSROOT.Painter.ShowStatus) this.has_event_status = false;

      var resized = this.layout_main(this.has_event_status || this._websocket ? "canvas" : "simple");

      var footer = this.select_main('footer');

      if (!this.has_event_status) {
         footer.html("");
         delete this.status_layout;
         delete this.ShowStatus;
         delete this.ShowStatusFunc;
      } else {

         this.status_layout = new JSROOT.GridDisplay(footer.node(), 'horizx4_1213');

         var frame_titles = ['object name','object title','mouse coordinates','object info'];
         for (var k=0;k<4;++k)
            d3.select(this.status_layout.GetFrame(k)).attr('title', frame_titles[k]).style('overflow','hidden')
            .append("label").attr("class","jsroot_status_label");

         this.ShowStatusFunc = function(name, title, info, coordinates) {
            if (!this.status_layout) return;
            $(this.status_layout.GetFrame(0)).children('label').text(name || "");
            $(this.status_layout.GetFrame(1)).children('label').text(title || "");
            $(this.status_layout.GetFrame(2)).children('label').text(coordinates || "");
            $(this.status_layout.GetFrame(3)).children('label').text(info || "");
         }

         this.ShowStatus = this.ShowStatusFunc.bind(this);

         this.ShowStatus("canvas","title","info","");
      }

      if (resized) this.CheckCanvasResize(); // redraw with resize
   }

   TPadPainter.prototype.ShowCanvasMenu = function(name) {

      d3.event.stopPropagation(); // disable main context menu
      d3.event.preventDefault();  // disable browser context menu

      var evnt = d3.event;

      function HandleClick(arg) {
         if (!this._websocket) return;
         console.log('click', arg);

         if (arg=="Interrupt") { this._websocket.send("GEXE:gROOT->SetInterrupt()"); }
         if (arg=="Quit ROOT") { this._websocket.send("GEXE:gApplication->Terminate(0)"); }
      }

      JSROOT.Painter.createMenu(this, function(menu) {

         switch(name) {
            case "File": {
               menu.add("Close canvas", HandleClick);
               menu.add("separator");
               menu.add("Save PNG", HandleClick);
               var ext = ["ps","eps","pdf","tex","gif","jpg","png","C","root"];
               menu.add("sub:Save");
               for (var k in ext) menu.add("canvas."+ext[k], HandleClick);
               menu.add("endsub:");
               menu.add("separator");
               menu.add("Interrupt", HandleClick);
               menu.add("separator");
               menu.add("Quit ROOT", HandleClick);
               break;
            }
            case "Edit":
               menu.add("Clear pad", HandleClick);
               menu.add("Clear canvas", HandleClick);
               break;
            case "View": {
               menu.addchk(menu.painter.has_event_status, "Event status", menu.painter.ToggleEventStatus.bind(menu.painter));
               var fp = menu.painter.frame_painter();
               menu.addchk(fp && fp.tooltip_allowed, "Tooltip info", function() { if (fp) fp.tooltip_allowed = !fp.tooltip_allowed; });
               break;
            }
            case "Options": {
               var main = menu.painter.main_painter();
               menu.addchk(main && main.ToggleStat('only-check'), "Statistic", function() { if (main) main.ToggleStat(); });
               menu.addchk(main && main.ToggleTitle('only-check'), "Histogram title",  function() { if (main) main.ToggleTitle(); });
               menu.addchk(main && main.ToggleStat('fitpar-check'), "Fit parameters", function() { if (main) main.ToggleStat('fitpar-toggle'); });
               break;
            }
            case "Tools":
               menu.add("Inspector", HandleClick);
               break;
            case "Help":
               menu.add("header:Basic help on...");
               menu.add("Canvas", HandleClick);
               menu.add("Menu", HandleClick);
               menu.add("Browser", HandleClick);
               menu.add("separator");
               menu.add("About ROOT", HandleClick);
               break;
         }
         if (menu.size()>0) menu.show(evnt);
      });
   }

   TPadPainter.prototype.CreateCanvasMenu = function() {

      if (this.enlarge_main('state')==='on') return;

      this.layout_main("canvas");

      var header = this.select_main('header');

      header.html("").style('background','lightgrey');

      var items = ['File','Edit','View','Options','Tools','Help'];
      var painter = this;
      for (var k in items) {
         var elem = header.append("p").attr("class","canvas_menu").text(items[k]);
         if (items[k]=='Help') elem.style('float','right');
         elem.on('click', this.ShowCanvasMenu.bind(this, items[k]));
      }
   }

   TPadPainter.prototype.CreateCanvasSvg = function(check_resize, new_size) {

      var factor = null, svg = null, lmt = 5, rect = null;

      if (check_resize > 0) {

         if (this._fixed_size) return (check_resize > 1); // flag used to force re-drawing of all subpads

         svg = this.svg_canvas();

         factor = svg.property('height_factor');

         rect = this.check_main_resize(check_resize, null, factor);

         if (!rect.changed) return false;

      } else {

         if (this._websocket)
            this.CreateCanvasMenu();

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
                 .on("mouseenter", this.ShowObjectStatus.bind(this))

         svg.append("svg:g").attr("class","root_frame");
         svg.append("svg:g").attr("class","subpads_layer");
         svg.append("svg:g").attr("class","special_layer");
         svg.append("svg:g").attr("class","text_layer");
         svg.append("svg:g").attr("class","stat_layer");
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
         this.fillatt = this.createAttFill(this.pad, 1001, 0);

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
         .attr("width",rect.width)
         .attr("height",rect.height)
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
         svg_pad = svg_can.select(".subpads_layer")
             .append("g")
             .attr("class", "root_pad")
             .attr("pad", this.this_pad_name) // set extra attribute  to mark pad name
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null); // this is custom property
         svg_rect = svg_pad.append("svg:rect").attr("class", "root_pad_border");

         svg_pad.append("svg:g").attr("class","root_frame");
         svg_pad.append("svg:g").attr("class","special_layer");
         svg_pad.append("svg:g").attr("class","text_layer");
         svg_pad.append("svg:g").attr("class","stat_layer");
         btns = svg_pad.append("svg:g").attr("class","btns_layer");

         if (JSROOT.gStyle.ContextMenu)
            svg_rect.on("contextmenu", this.ShowContextMenu.bind(this));

         if (!JSROOT.BatchMode)
            svg_rect.attr("pointer-events", "visibleFill") // get events also for not visible rect
                    .on("dblclick", this.EnlargePad.bind(this))
                    .on("mouseenter", this.ShowObjectStatus.bind(this));
      }

      if (!this.fillatt || !this.fillatt.changed)
         this.fillatt = this.createAttFill(this.pad, 1001, 0);
      if (!this.lineatt || !this.lineatt.changed) {
         this.lineatt = JSROOT.Painter.createAttLine(this.pad);
         if (this.pad.fBorderMode == 0) this.lineatt.color = 'none';
      }

      svg_pad.attr("transform", "translate(" + x + "," + y + ")")
             .attr("display", pad_visible ? null : "none")
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

   TPadPainter.prototype.CheckColors = function(can) {
      if (!can || !can.fPrimitives) return;

      for (var i = 0; i < can.fPrimitives.arr.length; ++i) {
         var obj = can.fPrimitives.arr[i];
         if (obj==null) continue;
         if ((obj._typename=="TObjArray") && (obj.name == "ListOfColors")) {
            JSROOT.Painter.adoptRootColors(obj);
            can.fPrimitives.arr.splice(i,1);
            can.fPrimitives.opt.splice(i,1);
            return;
         }
      }
   }

   TPadPainter.prototype.RemovePrimitive = function(obj) {
      if ((this.pad===null) || (this.pad.fPrimitives === null)) return;
      var indx = this.pad.fPrimitives.arr.indexOf(obj);
      if (indx>=0) this.pad.fPrimitives.RemoveAt(indx);
   }

   TPadPainter.prototype.FindPrimitive = function(exact_obj, classname, name) {
      if ((this.pad===null) || (this.pad.fPrimitives === null)) return null;

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

      if ((this.pad===null) || !this.pad.fPrimitives || (this.pad.fPrimitives.arr.length==0)) return false;

      for (var n=0;n<this.pad.fPrimitives.arr.length;++n)
         if (this.pad.fPrimitives.arr[n] && this.pad.fPrimitives.arr[n]._typename != "TPad") return true;

      return false;
   }

   TPadPainter.prototype.DrawPrimitive = function(indx, callback, ppainter) {
      if (ppainter) ppainter._primitive = true; // mark painter as belonging to primitives

      if (!this.pad || (indx >= this.pad.fPrimitives.arr.length))
         return JSROOT.CallBack(callback);

      JSROOT.draw(this.divid, this.pad.fPrimitives.arr[indx], this.pad.fPrimitives.opt[indx], this.DrawPrimitive.bind(this, indx+1, callback));
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
         if (hint && pnt.painters) hint.painter = obj;
      });

      return hints;
   }

   TPadPainter.prototype.FillContextMenu = function(menu) {

      if (this.pad)
         menu.add("header: " + this.pad._typename + "::" + this.pad.fName);
      else
         menu.add("header: Canvas");

      menu.addchk((JSROOT.gStyle.Tooltip > 0), "Enable tooltips (global)", function() {
         JSROOT.gStyle.Tooltip = (JSROOT.gStyle.Tooltip === 0) ? 1 : -JSROOT.gStyle.Tooltip;
         var can_painter = this;
         if (!this.iscan && this.has_canvas) can_painter = this.pad_painter();
         if (can_painter && can_painter.ForEachPainterInPad)
            can_painter.ForEachPainterInPad(function(fp) {
               if (fp.tooltip_allowed!==undefined) fp.tooltip_allowed = (JSROOT.gStyle.Tooltip > 0);
            });
      });

      if (!this._websocket) {

         function ToggleField(arg) {
            this.pad[arg] = this.pad[arg] ? 0 : 1;
            var main = this.svg_pad(this.this_pad_name).property('mainpainter');
            if (!main) return;

            if ((arg.indexOf('fGrid')==0) && (typeof main.DrawGrids == 'function'))
               return main.DrawGrids();

            if ((arg.indexOf('fTick')==0) && (typeof main.DrawAxes == 'function'))
               return main.DrawAxes();
         }

         menu.addchk(this.pad.fGridx, 'Grid x', 'fGridx', ToggleField);
         menu.addchk(this.pad.fGridy, 'Grid y', 'fGridy', ToggleField);
         menu.addchk(this.pad.fTickx, 'Tick x', 'fTickx', ToggleField);
         menu.addchk(this.pad.fTicky, 'Tick y', 'fTicky', ToggleField);

         this.FillAttContextMenu(menu);
      }

      menu.add("separator");

      menu.addchk(this.has_event_status, "Event status", this.ToggleEventStatus.bind(this));

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
      }

      JSROOT.Painter.createMenu(this, function(menu) {

         menu.painter.FillContextMenu(menu);

         menu.painter.FillObjectExecMenu(menu, function() { menu.show(evnt); });
      }); // end menu creation
   }

   TPadPainter.prototype.Redraw = function(resize) {

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
         if ((obj!==null) && (obj._typename === "TPad")) num++;
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

      if (this.iscan) this.CheckColors(obj);

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

      if (objpainter && lst && lst[indx] && (typeof objpainter.snapid === 'undefined')) {
         // keep snap id in painter, will be used for the
         if (this.painters.indexOf(objpainter)<0) this.painters.push(objpainter);
         objpainter.snapid = lst[indx].fObjectID;
      }

      ++indx; // change to the next snap

      if (!lst || indx >= lst.length) return JSROOT.CallBack(call_back, this);

      var snap = lst[indx], painter = null;

      // first find existing painter for the object
      for (var k=0; k<this.painters.length; ++k) {
         if (this.painters[k].snapid === snap.fObjectID) { painter = this.painters[k]; break;  }
      }

      // function which should be called when drawing of next item finished
      var draw_callback = this.DrawNextSnap.bind(this, lst, indx, call_back);

      if (painter) {

         if (snap.fKind === 1) { // object itself
            if (painter.UpdateObject(snap.fSnapshot)) painter.Redraw();
            return draw_callback(painter); // call next
         }

         if (snap.fKind === 2) { // update SVG
            if (painter.UpdateObject(snap.fSnapshot)) painter.Redraw();
            return draw_callback(painter); // call next
         }

         if (snap.fKind === 3) { // subpad
            return painter.RedrawPadSnap(snap, draw_callback);
         }

         return draw_callback(painter); // call next
      }

      if (snap.fKind === 3) { // subpad

         if (snap.fPrimitives._typename) snap.fPrimitives = [ snap.fPrimitives ];

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

         // console.log("Create new subpad", padpainter.this_pad_name);

         // we select current pad, where all drawing is performed
         var prev_name = padpainter.CurrentPadName(padpainter.this_pad_name);
         padpainter.DrawNextSnap(snap.fPrimitives, 0, function() {
            padpainter.CurrentPadName(prev_name);
            draw_callback(padpainter);
         });
         return;
      }

      // here the case of normal drawing, can be improved
      if (snap.fKind === 1) {
         var obj = snap.fSnapshot;
         if (obj) obj.$snapid = snap.fObjectID; // mark object itself, workaround for stats drawing

         // TODO: frame should be created in histogram painters
         if (obj._typename != "TFrame" && this.svg_frame().select(".main_layer").empty())
            JSROOT.Painter.drawFrame(this.divid, null);

         // console.log("drawing object", obj._typename);

         return JSROOT.draw(this.divid, obj, snap.fOption, draw_callback);
      }

      if (snap.fKind === 2)
         return JSROOT.draw(this.divid, snap.fSnapshot, snap.fOption, draw_callback);

      draw_callback(null);
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
      if (snap.fPrimitives._typename) snap.fPrimitives = [ snap.fPrimitives ];

      var first = snap.fPrimitives[0].fSnapshot;
      first.fPrimitives = null; // primitives are not interesting, just cannot disable in IO

      if (this.snapid === undefined) {
         // first time getting snap, create all gui elements first

         this.snapid = snap.fPrimitives[0].fObjectID;

         this.draw_object = first;
         this.pad = first;
         this._fixed_size = true;

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
         console.log("Clean everything", this.this_pad_name);
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

      // this.DrawNextSnap(snap.fPrimitives, 0, call_back, null); // update all snaps after each other

      // show we redraw all other painters without snapid?
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

         var svg = d3.select(svg3d);

         var layer = main.svg_layer("special_layer");
         group = layer.append("g")
                      .attr("class","temp_saveaspng")
                      .attr("transform", "translate(" + sz.x + "," + sz.y + ")");
         group.node().appendChild(svg3d);
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
         case 'timeout': isvisible = false; break;
         case 'toggle': {
            state = !state; btn.property('buttons_state', state);
            is_visible = state;
            break;
         }
         case 'disable':
         case 'leavebtn': {
            if (state) return;
            return btn.property('timout_handler', setTimeout(this.toggleButtonsVisibility.bind(this,'timeout'),500));
         }
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

   TPadPainter.prototype.DecodeOptions = function(opt) {
      var pad = this.GetObject();
      if (!pad) return;

      var d = new JSROOT.DrawOptions(opt);

      if (d.check('WEBSOCKET')) this.OpenWebsocket();

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

   JSROOT.Painter.drawCanvas = function(divid, can, opt) {
      var nocanvas = (can===null);
      if (nocanvas) can = JSROOT.Create("TCanvas");

      var painter = new TPadPainter(can, true);
      painter.DecodeOptions(opt);

      painter.SetDivId(divid, -1); // just assign id
      painter.CheckColors(can);
      painter.CreateCanvasSvg(0);
      painter.SetDivId(divid);  // now add to painters list

      painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "CanvasSnapShot", "Ctrl PrintScreen");
      if (JSROOT.gStyle.ContextMenu)
         painter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");

      if (painter.enlarge_main('verify'))
         painter.AddButton(JSROOT.ToolbarIcons.circle, "Enlarge canvas", "EnlargePad");

      if (nocanvas && opt.indexOf("noframe") < 0)
         JSROOT.Painter.drawFrame(divid, null);

      painter.DrawPrimitive(0, function() { painter.DrawingReady(); });
      return painter;
   }

   JSROOT.Painter.drawPad = function(divid, pad, opt) {
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

      var prev_name;

      if (painter.has_canvas)
         // we select current pad, where all drawing is performed
         prev_name = painter.CurrentPadName(painter.this_pad_name);

      painter.DrawPrimitive(0, function() {
         // we restore previous pad name
         painter.CurrentPadName(prev_name);
         painter.DrawingReady();
      });

      return painter;
   }

   // ================= painter of raw text ========================================


   JSROOT.Painter.drawRawText = function(divid, txt, opt) {

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

         var mathjax = this.txt.mathjax || (JSROOT.gStyle.MathJax>1);

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
   JSROOT.addDrawFunc({ name: "TPaveText", icon: "img_pavetext", prereq: "hist", func: "JSROOT.Painter.drawPaveText" });
   JSROOT.addDrawFunc({ name: "TPaveStats", icon: "img_pavetext", prereq: "hist", func: "JSROOT.Painter.drawPaveText" });
   JSROOT.addDrawFunc({ name: "TPaveLabel", icon: "img_pavelabel", prereq: "hist", func: "JSROOT.Painter.drawPaveText" });
   JSROOT.addDrawFunc({ name: "TLatex", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: "TMathText", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: "TText", icon: "img_text", prereq: "more2d", func: "JSROOT.Painter.drawText", direct: true });
   JSROOT.addDrawFunc({ name: /^TH1/, icon: "img_histo1d", prereq: "hist", func: "JSROOT.Painter.drawHistogram1D", opt:";hist;P;P0;E;E1;E2;E3;E4;E1X0;L;LF2;B;B1;TEXT;LEGO;same", ctrl: "l" });
   JSROOT.addDrawFunc({ name: "TProfile", icon: "img_profile", prereq: "hist", func: "JSROOT.Painter.drawHistogram1D", opt:";E0;E1;E2;p;hist"});
   JSROOT.addDrawFunc({ name: "TH2Poly", icon: "img_histo2d", prereq: "hist", func: "JSROOT.Painter.drawHistogram2D", opt:";COL;COL0;COLZ;LCOL;LCOL0;LCOLZ;LEGO;same", expand_item: "fBins", theonly: true });
   JSROOT.addDrawFunc({ name: "TH2PolyBin", icon: "img_histo2d", draw_field: "fPoly" });
   JSROOT.addDrawFunc({ name: /^TH2/, icon: "img_histo2d", prereq: "hist", func: "JSROOT.Painter.drawHistogram2D", opt:";COL;COLZ;COL0;COL1;COL0Z;COL1Z;BOX;BOX1;SCAT;TEXT;CONT;CONT1;CONT2;CONT3;CONT4;ARR;SURF;SURF1;SURF2;SURF4;SURF6;E;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same", ctrl: "colz" });
   JSROOT.addDrawFunc({ name: "TProfile2D", sameas: "TH2" });
   JSROOT.addDrawFunc({ name: /^TH3/, icon: 'img_histo3d', prereq: "hist3d", func: "JSROOT.Painter.drawHistogram3D", opt:";SCAT;BOX;BOX2;BOX3;GLBOX1;GLBOX2;GLCOL" });
   JSROOT.addDrawFunc({ name: "THStack", icon: "img_histo1d", prereq: "hist", func: "JSROOT.Painter.drawHStack", expand_item: "fHists" });
   JSROOT.addDrawFunc({ name: "TPolyMarker3D", icon: 'img_histo3d', prereq: "hist3d", func: "JSROOT.Painter.drawPolyMarker3D" });
   JSROOT.addDrawFunc({ name: "TGraphPolargram" }); // just dummy entry to avoid drawing of this object
   JSROOT.addDrawFunc({ name: "TGraph2D", icon:"img_graph", prereq: "hist3d", func: "JSROOT.Painter.drawGraph2D", opt:";P;PCOL"});
   JSROOT.addDrawFunc({ name: "TGraph2DErrors", icon:"img_graph", prereq: "hist3d", func: "JSROOT.Painter.drawGraph2D", opt:";P;PCOL;ERR"});
   JSROOT.addDrawFunc({ name: /^TGraph/, icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraph", opt:";L;P"});
   JSROOT.addDrawFunc({ name: "TEfficiency", icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawEfficiency", opt:";AP"});
   JSROOT.addDrawFunc({ name: "TCutG", sameas: "TGraph" });
   JSROOT.addDrawFunc({ name: /^RooHist/, sameas: "TGraph" });
   JSROOT.addDrawFunc({ name: /^RooCurve/, sameas: "TGraph" });
   JSROOT.addDrawFunc({ name: "RooPlot", icon: "img_canvas", prereq: "more2d", func: "JSROOT.Painter.drawRooPlot" });
   JSROOT.addDrawFunc({ name: "TMultiGraph", icon: "img_mgraph", prereq: "more2d", func: "JSROOT.Painter.drawMultiGraph", expand_item: "fGraphs" });
   JSROOT.addDrawFunc({ name: "TStreamerInfoList", icon: 'img_question', prereq: "hierarchy",  func: "JSROOT.Painter.drawStreamerInfo" });
   JSROOT.addDrawFunc({ name: "TPaletteAxis", icon: "img_colz", prereq: "hist", func: "JSROOT.Painter.drawPaletteAxis" });
   JSROOT.addDrawFunc({ name: "TWebPainting", icon: "img_graph", prereq: "more2d", func: "JSROOT.Painter.drawWebPainting" });
   JSROOT.addDrawFunc({ name: "kind:Text", icon: "img_text", func: JSROOT.Painter.drawRawText });
   JSROOT.addDrawFunc({ name: "TF1", icon: "img_tf1", prereq: "math;more2d", func: "JSROOT.Painter.drawFunction" });
   JSROOT.addDrawFunc({ name: "TF2", icon: "img_tf2", prereq: "math;more2d", func: "JSROOT.Painter.drawTF2" });
   JSROOT.addDrawFunc({ name: "TEllipse", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawEllipse", direct: true });
   JSROOT.addDrawFunc({ name: "TLine", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawLine", direct: true });
   JSROOT.addDrawFunc({ name: "TArrow", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawArrow", direct: true });
   JSROOT.addDrawFunc({ name: "TPolyLine", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawPolyLine", direct: true });
   JSROOT.addDrawFunc({ name: "TGaxis", icon: "img_graph", prereq: "hist", func: "JSROOT.Painter.drawGaxis" });
   JSROOT.addDrawFunc({ name: "TLegend", icon: "img_pavelabel", prereq: "hist", func: "JSROOT.Painter.drawLegend" });
   JSROOT.addDrawFunc({ name: "TBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TWbox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TSliderBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox", direct: true });
   JSROOT.addDrawFunc({ name: "TAxis3D", prereq: "hist3d", func: "JSROOT.Painter.drawAxis3D" });
   JSROOT.addDrawFunc({ name: "TMarker", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawMarker", direct: true });
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
   JSROOT.addDrawFunc({ name: "TList", icon: "img_list", prereq: "hierarchy", expand: "JSROOT.Painter.ListHierarchy", dflt: "expand" });
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
   JSROOT.draw = function(divid, obj, opt, callback) {

      function completeDraw(painter) {
         if (painter && callback && (typeof painter.WhenReady == 'function'))
            painter.WhenReady(callback);
         else
            JSROOT.CallBack(callback, painter);
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
         return JSROOT.draw(divid, obj[handle.draw_field], opt, callback);

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

   JSROOT.Painter.createRootColors();

   JSROOT.TBasePainter = TBasePainter;
   JSROOT.TObjectPainter = TObjectPainter;
   JSROOT.TFramePainter = TFramePainter;
   JSROOT.TPadPainter = TPadPainter;

   return JSROOT;

}));
