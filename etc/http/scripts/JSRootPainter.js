/// @file JSRootPainter.js
/// JavaScript ROOT graphics

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['JSRootCore', 'd3'], factory );
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
                }
   };

   JSROOT.Toolbar = function(container, buttons) {
      if ((container !== undefined) && (typeof container.append == 'function'))  {
         this.element = container.append("div").attr('class','jsroot');
         this.addButtons(buttons);
      }
   }

   JSROOT.Toolbar.prototype.addButtons = function(buttons) {
      var pthis = this;

      this.buttonsNames = [];
      buttons.forEach(function(buttonGroup) {
         var group = pthis.element.append('div').attr('class', 'toolbar-group');

         buttonGroup.forEach(function(buttonConfig) {
            var buttonName = buttonConfig.name;
            if (!buttonName) {
               throw new Error('must provide button \'name\' in button config');
            }
            if (pthis.buttonsNames.indexOf(buttonName) !== -1) {
               throw new Error('button name \'' + buttonName + '\' is taken');
            }
            pthis.buttonsNames.push(buttonName);

            pthis.createButton(group, buttonConfig);
         });
      });
   };


   JSROOT.Toolbar.prototype.createButton = function(group, config) {

      var title = config.title;
      if (title === undefined) title = config.name;

      if (typeof config.click !== 'function')
         throw new Error('must provide button \'click\' function in button config');

      var button = group.append('a')
                        .attr('class','toolbar-btn')
                        .attr('rel', 'tooltip')
                        .attr('data-title', title)
                        .on('click', config.click);

      this.createIcon(button, config.icon || JSROOT.ToolbarIcons.question);
   };

   JSROOT.Toolbar.prototype.createIcon = function(button, thisIcon) {
      var size = thisIcon.size || 512;
      var scale = thisIcon.scale || 1;

      var svg = button.append("svg:svg")
                      .attr('height', '1em')
                      .attr('width', '1em')
                      .attr('viewBox', [0, 0, size, size].join(' '))

      if ('recs' in thisIcon) {
          var rec = {};
          for (var n=0;n<thisIcon.recs.length;++n) {
             JSROOT.extend(rec, thisIcon.recs[n]);
             svg.append('rect').attr("x", rec.x).attr("y", rec.y)
                               .attr("width", rec.w).attr("height", rec.h)
                               .attr("fill", rec.f);
          }
       } else {
          var elem = svg.append('svg:path').attr('d',thisIcon.path);
          if (scale !== 1)
             elem.attr('transform', 'scale(' + scale + ' ' + scale +')');
       }
   };

   JSROOT.Toolbar.prototype.removeAllButtons = function() {
      this.element.remove();
   };

   /**
    * @class JSROOT.Painter Holder of different functions and classes for drawing
    */
   JSROOT.Painter = {};

   JSROOT.Painter.createMenu = function(maincallback, menuname) {
      // dummy functions, forward call to the jquery function
      document.body.style.cursor = 'wait';
      JSROOT.AssertPrerequisites('jq2d', function() {
         document.body.style.cursor = 'auto';
         JSROOT.Painter.createMenu(maincallback, menuname);
      });
   }

   JSROOT.Painter.closeMenu = function(menuname) {
      if (!menuname) menuname = 'root_ctx_menu';
      var x = document.getElementById(menuname);
      if (x) x.parentNode.removeChild(x);
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
         JSROOT.gStyle.Tooltip =     parseInt(inter.charAt(0));
         JSROOT.gStyle.ContextMenu = (inter.charAt(1) != '0');
         JSROOT.gStyle.Zooming  =    (inter.charAt(2) != '0');
         JSROOT.gStyle.MoveResize =  (inter.charAt(3) != '0');
         JSROOT.gStyle.DragAndDrop = (inter.charAt(4) != '0');
      }

      var tt = JSROOT.GetUrlOption("tooltip", url);
      if (tt !== null) JSROOT.gStyle.Tooltip = parseInt(tt);

      var mathjax = JSROOT.GetUrlOption("mathjax", url);
      if ((mathjax!==null) && (mathjax!="0")) JSROOT.gStyle.MathJax = 1;

      if (JSROOT.GetUrlOption("nomenu", url)!=null) JSROOT.gStyle.ContextMenu = false;
      if (JSROOT.GetUrlOption("noprogress", url)!=null) JSROOT.gStyle.ProgressBox = false;
      if (JSROOT.GetUrlOption("notouch", url)!=null) JSROOT.touches = false;

      JSROOT.gStyle.OptStat = JSROOT.GetUrlOption("optstat", url, JSROOT.gStyle.OptStat);
      JSROOT.gStyle.OptFit = JSROOT.GetUrlOption("optfit", url, JSROOT.gStyle.OptFit);
      JSROOT.gStyle.StatFormat = JSROOT.GetUrlOption("statfmt", url, JSROOT.gStyle.StatFormat);
      JSROOT.gStyle.FitFormat = JSROOT.GetUrlOption("fitfmt", url, JSROOT.gStyle.FitFormat);

      var toolbar = JSROOT.GetUrlOption("toolbar", url);
      if (toolbar !== null)
         JSROOT.gStyle.ToolBar = (toolbar !== "0") && (toolbar !== "false");

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
   JSROOT.Painter.root_colors = function() {
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
         for (var n=0; n < entry.str.length; n+=6) {
            var num = parseInt(entry.col) + parseInt(n/6);
            colorMap[num] = 'rgb(' + parseInt("0x" +entry.str.slice(n,n+2)) + "," + parseInt("0x" + entry.str.slice(n+2,n+4)) + "," + parseInt("0x" + entry.str.slice(n+4,n+6)) + ")";
         }
      }

      return colorMap;
   }();

   JSROOT.Painter.MakeColorRGB = function(col) {
      if ((col==null) || (col._typename != 'TColor')) return null;
      var rgb = Math.round(col.fRed*255) + "," + Math.round(col.fGreen*255) + "," + Math.round(col.fBlue*255);
      if ((col.fAlpha === undefined) || (col.fAlpha == 1.))
         rgb = "rgb(" + rgb + ")";
      else
         rgb = "rgba(" + rgb + "," + col.fAlpha.toFixed(3) + ")";

      switch (rgb) {
         case 'rgb(255,255,255)' : rgb = 'white'; break;
         case 'rgb(0,0,0)' : rgb = 'black'; break;
         case 'rgb(255,0,0)' : rgb = 'red'; break;
         case 'rgb(0,255,0)' : rgb = 'green'; break;
         case 'rgb(0,0,255)' : rgb = 'blue'; break;
         case 'rgb(255,255,0)' : rgb = 'yellow'; break;
         case 'rgb(255,0,255)' : rgb = 'magenta'; break;
         case 'rgb(0,255,255)' : rgb = 'cyan'; break;
      }
      return rgb;
   }

   JSROOT.Painter.adoptRootColors = function(objarr) {
      if (!objarr || !objarr.arr) return;

      for (var n = 0; n < objarr.arr.length; ++n) {
         var col = objarr.arr[n];
         if ((col==null) || (col._typename != 'TColor')) continue;

         var num = col.fNumber;
         if ((num<0) || (num>4096)) continue;

         var rgb = JSROOT.Painter.MakeColorRGB(col);
         if (rgb == null) continue;

         if (JSROOT.Painter.root_colors[num] != rgb)
            JSROOT.Painter.root_colors[num] = rgb;
      }
   }

   JSROOT.Painter.root_line_styles = new Array("", "", "3,3", "1,2",
         "3,4,1,4", "5,3,1,3", "5,3,1,3,1,3,1,3", "5,5",
         "5,3,1,3,1,3", "20,5", "20,10,1,10", "1,3");

   // Initialize ROOT markers
   JSROOT.Painter.root_markers = new Array(
           0, 100,   8,   7,   0,  //  0..4
           9, 100, 100, 100, 100,  //  5..9
         100, 100, 100, 100, 100,  // 10..14
         100, 100, 100, 100, 100,  // 15..19
         100, 103, 105, 104,   0,  // 20..24
           3,   4,   2,   1, 106,  // 25..29
           6,   7,   5, 102, 101); // 30..34

   /** Function returns the ready to use marker for drawing */
   JSROOT.Painter.createAttMarker = function(attmarker, style) {

      var marker_color = JSROOT.Painter.root_colors[attmarker.fMarkerColor];

      if ((style===null) || (style===undefined)) style = attmarker.fMarkerStyle;

      var res = { x0: 0, y0: 0, color: marker_color, style: style, size: 8, stroke: true, fill: true, marker: "",  ndig: 0, used: true, changed: false };

      res.Change = function(color, style, size) {

         this.changed = true;

         if (color!==undefined) this.color = color;
         if (style!==undefined) this.style = style;
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
            case 1: size = this.size = 1; break;
            case 6: size = this.size = 2; break;
            case 7: size = this.size = 3; break;
            default: this.size = size; size*=8;
         }

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

      var color = 0, _width = 0, style = 0;
      if (typeof attline == 'string') {
         if (attline=='black') { color = 1; _width = 1; } else
         if (attline=='none') { _width = 0; }
      } else
      if (typeof attline == 'object') {
         if ('fLineColor' in attline) color = attline.fLineColor;
         if ('fLineWidth' in attline) _width = attline.fLineWidth;
         if ('fLineStyle' in attline) style = attline.fLineStyle;
      } else
      if ((attline!==undefined) && !isNaN(attline)) {
         color = attline;
      }

      if (borderw!==undefined) _width = borderw;

      var line = {
          used: true, // can mark object if it used or not,
          color: JSROOT.Painter.root_colors[color],
          width: _width,
          dash: JSROOT.Painter.root_line_styles[style]
      };

      if ((_width==0) || (color==0)) line.color = 'none';

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
      if ((line.color === undefined) && (color>0))
         line.color = 'lightgrey';

      line.Apply = function(selection) {
         this.used = true;
         if (this.color=='none') {
            selection.style('stroke', null);
            selection.style('stroke-width', null);
            selection.style('stroke-dasharray', null);
         } else {
            selection.style('stroke', this.color);
            selection.style('stroke-width', this.width);
            if (this.dash && (this.dash.length>0))
               selection.style('stroke-dasharray', this.dash);
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

      var fontName = JSROOT.Painter.root_fonts[Math.floor(fontIndex / 10)];

      var res = { name: "Arial", size: 11, weight: null, style: null };

      if (size != undefined) res.size = Math.round(size);

      if (typeof fontName != 'string') fontName = "";

      while (fontName.length > 0) {
         if (fontName.charAt(0)==='b') res.weight = "bold"; else
         if (fontName.charAt(0)==='i') res.style = "italic"; else
         if (fontName.charAt(0)==='o') res.style = "oblique"; else break;
         fontName = fontName.substr(1);
      }

      if (fontName == 'Symbol') {
         res.weight = null;
         res.style = null;
      }

      res.name = fontName;

      res.SetFont = function(selection) {
         selection.attr("font-family", this.name)
                  .attr("font-size", this.size)
                  .attr("xml:space","preserve");
         if (this.weight!=null)
            selection.attr("font-weight", this.weight);
         if (this.style!=null)
            selection.attr("font-style", this.style);
      }

      res.asStyle = function(sz) {
         return ((sz!=null) ? sz : this.size) + "px " + this.name;
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
      if (idF < 0) return JSROOT.gStyle.TimeOffset;
      var sof = axis.fTimeFormat.substr(idF + 2);
      if (sof == '1995-01-01 00:00:00s0') return 788918400000;
      // special case, used from DABC painters
      if ((sof == "0") || (sof == "")) return 0;

      // decode time from ROOT string
      var dt = new Date(0);
      var pos = sof.indexOf("-"); dt.setFullYear(sof.substr(0,pos)); sof = sof.substr(pos+1);
      pos = sof.indexOf("-"); dt.setMonth(parseInt(sof.substr(0,pos))-1); sof = sof.substr(pos+1);
      pos = sof.indexOf(" "); dt.setDate(sof.substr(0,pos)); sof = sof.substr(pos+1);
      pos = sof.indexOf(":"); dt.setHours(sof.substr(0,pos)); sof = sof.substr(pos+1);
      pos = sof.indexOf(":"); dt.setMinutes(sof.substr(0,pos)); sof = sof.substr(pos+1);
      pos = sof.indexOf("s"); dt.setSeconds(sof.substr(0,pos));
      if (pos>0) { sof = sof.substr(pos+1); dt.setMilliseconds(sof); }
      return dt.getTime();
   }

   JSROOT.Painter.formatExp = function(label) {
      var str = label;
      if (parseFloat(str) == 1.0) return '1';
      if (parseFloat(str) == 10.0) return '10';
      var str = str.replace('e+', 'x10@');
      var str = str.replace('e-', 'x10@-');
      var _val = str.substring(0, str.indexOf('@'));
      var _exp = str.substr(str.indexOf('@'));
      _val = _val.replace('@', '');
      _exp = _exp.replace('@', '');
      var u, size = _exp.length;
      for (var j = 0; j < size; ++j) {
         var u, c  = _exp.charAt(j);
         if (c == '+') u = '\u207A'; else
         if (c == '-') u = '\u207B'; else {
            var e = parseInt(c);
            if (e == 1) u = String.fromCharCode(0xB9); else
            if (e > 1 && e < 4) u = String.fromCharCode(0xB0 + e); else
                                u = String.fromCharCode(0x2070 + e);
         }
         _exp = _exp.replace(c, u);
      }
      _val = _val.replace('1x', '');
      return _val + _exp;
   };

   JSROOT.Painter.translateExp = function(str) {
      var lstr = str.match(/\^{[0-9]*}/gi);
      if (lstr != null) {
         var symbol = '';
         for (var i = 0; i < lstr.length; ++i) {
            symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('^{', ''); // &sup
            symbol = symbol.replace('}', ''); // ;
            var size = symbol.length;
            for (var j = 0; j < size; ++j) {
               var c = symbol.charAt(j);
               var u, e = parseInt(c);
               if (e == 1) u = String.fromCharCode(0xB9);
               else if (e > 1 && e < 4) u = String.fromCharCode(0xB0 + e);
               else u = String.fromCharCode(0x2070 + e);
               symbol = symbol.replace(c, u);
            }
            str = str.replace(lstr[i], symbol);
         }
      }
      return str;
   };

   JSROOT.Painter.symbols_map = {
      // greek letters
      '#alpha' : '\u03B1',
      '#beta' : '\u03B2',
      '#chi' : '\u03C7',
      '#delta' : '\u03B4',
      '#varepsilon' : '\u03B5',
      '#phi' : '\u03C6',
      '#gamma' : '\u03B3',
      '#eta' : '\u03B7',
      '#iota' : '\u03B9',
      '#varphi' : '\u03C6',
      '#kappa' : '\u03BA',
      '#lambda' : '\u03BB',
      '#mu' : '\u03BC',
      '#nu' : '\u03BD',
      '#omicron' : '\u03BF',
      '#pi' : '\u03C0',
      '#theta' : '\u03B8',
      '#rho' : '\u03C1',
      '#sigma' : '\u03C3',
      '#tau' : '\u03C4',
      '#upsilon' : '\u03C5',
      '#varomega' : '\u03D6',
      '#omega' : '\u03C9',
      '#xi' : '\u03BE',
      '#psi' : '\u03C8',
      '#zeta' : '\u03B6',
      '#Alpha' : '\u0391',
      '#Beta' : '\u0392',
      '#Chi' : '\u03A7',
      '#Delta' : '\u0394',
      '#Epsilon' : '\u0395',
      '#Phi' : '\u03A6',
      '#Gamma' : '\u0393',
      '#Eta' : '\u0397',
      '#Iota' : '\u0399',
      '#vartheta' : '\u03D1',
      '#Kappa' : '\u039A',
      '#Lambda' : '\u039B',
      '#Mu' : '\u039C',
      '#Nu' : '\u039D',
      '#Omicron' : '\u039F',
      '#Pi' : '\u03A0',
      '#Theta' : '\u0398',
      '#Rho' : '\u03A1',
      '#Sigma' : '\u03A3',
      '#Tau' : '\u03A4',
      '#Upsilon' : '\u03A5',
      '#varsigma' : '\u03C2',
      '#Omega' : '\u03A9',
      '#Xi' : '\u039E',
      '#Psi' : '\u03A8',
      '#Zeta' : '\u0396',
      '#varUpsilon' : '\u03D2',
      '#epsilon' : '\u03B5',
      // math symbols

      '#sqrt' : '\u221A',

      // from TLatex tables #2 & #3
      '#leq' : '\u2264',
      '#/' : '\u2044',
      '#infty' : '\u221E',
      '#voidb' : '\u0192',
      '#club' : '\u2663',
      '#diamond' : '\u2666',
      '#heart' : '\u2665',
      '#spade' : '\u2660',
      '#leftrightarrow' : '\u2194',
      '#leftarrow' : '\u2190',
      '#uparrow' : '\u2191',
      '#rightarrow' : '\u2192',
      '#downarrow' : '\u2193',
      '#circ' : '\u02C6', // ^
      '#pm' : '\xB1',
      '#doublequote' : '\u2033',
      '#geq' : '\u2265',
      '#times' : '\xD7',
      '#propto' : '\u221D',
      '#partial' : '\u2202',
      '#bullet' : '\u2022',
      '#divide' : '\xF7',
      '#neq' : '\u2260',
      '#equiv' : '\u2261',
      '#approx' : '\u2248', // should be \u2245 ?
      '#3dots' : '\u2026',
      '#cbar' : '\u007C',
      '#topbar' : '\xAF',
      '#downleftarrow' : '\u21B5',
      '#aleph' : '\u2135',
      '#Jgothic' : '\u2111',
      '#Rgothic' : '\u211C',
      '#voidn' : '\u2118',
      '#otimes' : '\u2297',
      '#oplus' : '\u2295',
      '#oslash' : '\u2205',
      '#cap' : '\u2229',
      '#cup' : '\u222A',
      '#supseteq' : '\u2287',
      '#supset' : '\u2283',
      '#notsubset' : '\u2284',
      '#subseteq' : '\u2286',
      '#subset' : '\u2282',
      '#int' : '\u222B',
      '#in' : '\u2208',
      '#notin' : '\u2209',
      '#angle' : '\u2220',
      '#nabla' : '\u2207',
      '#oright' : '\xAE',
      '#ocopyright' : '\xA9',
      '#trademark' : '\u2122',
      '#prod' : '\u220F',
      '#surd' : '\u221A',
      '#upoint' : '\u22C5',
      '#corner' : '\xAC',
      '#wedge' : '\u2227',
      '#vee' : '\u2228',
      '#Leftrightarrow' : '\u21D4',
      '#Leftarrow' : '\u21D0',
      '#Uparrow' : '\u21D1',
      '#Rightarrow' : '\u21D2',
      '#Downarrow' : '\u21D3',
      '#LT' : '\x3C',
      '#void1' : '\xAE',
      '#copyright' : '\xA9',
      '#void3' : '\u2122',
      '#sum' : '\u2211',
      '#arctop' : '',
      '#lbar' : '',
      '#arcbottom' : '',
      '#void8' : '',
      '#bottombar' : '\u230A',
      '#arcbar' : '',
      '#ltbar' : '',
      '#AA' : '\u212B',
      '#aa' : '\u00E5',
      '#void06' : '',
      '#GT' : '\x3E',
      '#forall' : '\u2200',
      '#exists' : '\u2203',
      '#bar' : '',
      '#vec' : '',
      '#dot' : '\u22C5',
      '#hat' : '\xB7',
      '#ddot' : '',
      '#acute' : '\acute',
      '#grave' : '',
      '#check' : '\u2713',
      '#tilde' : '\u02DC',
      '#slash' : '\u2044',
      '#hbar' : '\u0127',
      '#box' : '',
      '#Box' : '',
      '#parallel' : '',
      '#perp' : '\u22A5',
      '#odot' : '',
      '#left' : '',
      '#right' : ''
   };

   JSROOT.Painter.translateLaTeX = function(string) {
      var str = string;
      str = this.translateExp(str);
      while (str.indexOf('^{o}') != -1)
         str = str.replace('^{o}', '\xBA');
      var lstr = str.match(/\#sqrt{(.*?)}/gi);
      if (lstr != null)
         for (var i = 0; i < lstr.length; ++i) {
            var symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('#sqrt{', '#sqrt');
            symbol = symbol.replace('}', '');
            str = str.replace(lstr[i], symbol);
         }
      lstr = str.match(/\_{(.*?)}/gi);
      if (lstr != null)
         for (var i = 0; i < lstr.length; ++i) {
            var symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('_{', ''); // &sub
            symbol = symbol.replace('}', ''); // ;
            str = str.replace(lstr[i], symbol);
         }
      lstr = str.match(/\^{(.*?)}/gi);
      if (lstr != null)
         for (i = 0; i < lstr.length; ++i) {
            var symbol = lstr[i].replace(' ', '');
            symbol = symbol.replace('^{', ''); // &sup
            symbol = symbol.replace('}', ''); // ;
            str = str.replace(lstr[i], symbol);
         }
      while (str.indexOf('#/') != -1)
         str = str.replace('#/', JSROOT.Painter.symbols_map['#/']);
      for ( var x in JSROOT.Painter.symbols_map) {
         while (str.indexOf(x) != -1)
            str = str.replace(x, JSROOT.Painter.symbols_map[x]);
      }

      // simple workaround for simple #splitline{first_line}{second_line}
      if ((str.indexOf("#splitline{")==0) && (str.charAt(str.length-1)=="}")) {
         var pos = str.indexOf("}{");
         if ((pos>0) && (pos == str.lastIndexOf("}{"))) {
            str = str.replace("}{", "\n ");
            str = str.slice(11, str.length-1);
         }
      }

      str = str.replace('^2','\xB2').replace('^3','\xB3');

      return str;
   }

   JSROOT.Painter.isAnyLatex = function(str) {
      return (str.indexOf("#")>=0) || (str.indexOf("\\")>=0) || (str.indexOf("{")>=0);
   }

   JSROOT.Painter.translateMath = function(str, kind, color) {
      // function translate ROOT TLatex into MathJax format

      if (kind!=2) {
         str = str.replace(/#LT/g, "\\langle");
         str = str.replace(/#GT/g, "\\rangle");
         str = str.replace(/#club/g, "\\clubsuit");
         str = str.replace(/#spade/g, "\\spadesuit");
         str = str.replace(/#heart/g, "\\heartsuit");
         str = str.replace(/#diamond/g, "\\diamondsuit");
         str = str.replace(/#voidn/g, "\\wp");
         str = str.replace(/#voidb/g, "f");
         str = str.replace(/#copyright/g, "(c)");
         str = str.replace(/#ocopyright/g, "(c)");
         str = str.replace(/#trademark/g, "TM");
         str = str.replace(/#void3/g, "TM");
         str = str.replace(/#oright/g, "R");
         str = str.replace(/#void1/g, "R");
         str = str.replace(/#3dots/g, "\\ldots");
         str = str.replace(/#lbar/g, "\\mid");
         str = str.replace(/#void8/g, "\\mid");
         str = str.replace(/#divide/g, "\\div");
         str = str.replace(/#Jgothic/g, "\\Im");
         str = str.replace(/#Rgothic/g, "\\Re");
         str = str.replace(/#doublequote/g, "\"");
         str = str.replace(/#plus/g, "+");

         str = str.replace(/#diamond/g, "\\diamondsuit");
         str = str.replace(/#voidn/g, "\\wp");
         str = str.replace(/#voidb/g, "f");
         str = str.replace(/#copyright/g, "(c)");
         str = str.replace(/#ocopyright/g, "(c)");
         str = str.replace(/#trademark/g, "TM");
         str = str.replace(/#void3/g, "TM");
         str = str.replace(/#oright/g, "R");
         str = str.replace(/#void1/g, "R");
         str = str.replace(/#3dots/g, "\\ldots");
         str = str.replace(/#lbar/g, "\\mid");
         str = str.replace(/#void8/g, "\\mid");
         str = str.replace(/#divide/g, "\\div");
         str = str.replace(/#Jgothic/g, "\\Im");
         str = str.replace(/#Rgothic/g, "\\Re");
         str = str.replace(/#doublequote/g, "\"");
         str = str.replace(/#plus/g, "+");
         str = str.replace(/#minus/g, "-");
         str = str.replace(/#\//g, "/");
         str = str.replace(/#upoint/g, ".");
         str = str.replace(/#aa/g, "\\mathring{a}");
         str = str.replace(/#AA/g, "\\mathring{A}");

         str = str.replace(/#omicron/g, "o");
         str = str.replace(/#Alpha/g, "A");
         str = str.replace(/#Beta/g, "B");
         str = str.replace(/#Epsilon/g, "E");
         str = str.replace(/#Zeta/g, "Z");
         str = str.replace(/#Eta/g, "H");
         str = str.replace(/#Iota/g, "I");
         str = str.replace(/#Kappa/g, "K");
         str = str.replace(/#Mu/g, "M");
         str = str.replace(/#Nu/g, "N");
         str = str.replace(/#Omicron/g, "O");
         str = str.replace(/#Rho/g, "P");
         str = str.replace(/#Tau/g, "T");
         str = str.replace(/#Chi/g, "X");
         str = str.replace(/#varomega/g, "\\varpi");

         str = str.replace(/#corner/g, "?");
         str = str.replace(/#ltbar/g, "?");
         str = str.replace(/#bottombar/g, "?");
         str = str.replace(/#notsubset/g, "?");
         str = str.replace(/#arcbottom/g, "?");
         str = str.replace(/#cbar/g, "?");
         str = str.replace(/#arctop/g, "?");
         str = str.replace(/#topbar/g, "?");
         str = str.replace(/#arcbar/g, "?");
         str = str.replace(/#downleftarrow/g, "?");
         str = str.replace(/#splitline/g, "\\genfrac{}{}{0pt}{}");
         str = str.replace(/#it/g, "\\textit");
         str = str.replace(/#bf/g, "\\textbf");

         str = str.replace(/#frac/g, "\\frac");
         //str = str.replace(/#left{/g, "\\left\\{");
         //str = str.replace(/#right}/g, "\\right\\}");
         str = str.replace(/#left{/g, "\\lbrace");
         str = str.replace(/#right}/g, "\\rbrace");
         str = str.replace(/#left\[/g, "\\lbrack");
         str = str.replace(/#right\]/g, "\\rbrack");
         //str = str.replace(/#left/g, "\\left");
         //str = str.replace(/#right/g, "\\right");
         // processing of #[] #{} should be done
         str = str.replace(/#\[\]{/g, "\\lbrack");
         str = str.replace(/ } /g, "\\rbrack");
         //str = str.replace(/#\[\]/g, "\\brack");
         //str = str.replace(/#{}/g, "\\brace");
         str = str.replace(/#\[/g, "\\lbrack");
         str = str.replace(/#\]/g, "\\rbrack");
         str = str.replace(/#{/g, "\\lbrace");
         str = str.replace(/#}/g, "\\rbrace");
         str = str.replace(/ /g, "\\;");

         for (var x in JSROOT.Painter.symbols_map) {
            var y = "\\" + x.substr(1);
            str = str.replace(new RegExp(x,'g'), y);
         }
      } else {
         str = str.replace(/\\\^/g, "\\hat");
      }

      if (typeof color != 'string') return "\\(" + str + "\\)";

      if (color.indexOf("rgb(")>=0)
         color = color.replace(/rgb/g, "[RGB]")
                      .replace(/\(/g, '{')
                      .replace(/\)/g, '}');
      return "\\(\\color{" + color + '}' + str + "\\)";
   }

   // ==============================================================================

   JSROOT.TBasePainter = function() {
      this.divid = null; // either id of element (preferable) or element itself
   }

   JSROOT.TBasePainter.prototype.Cleanup = function() {
      // generic method to cleanup painter
   }

   JSROOT.TBasePainter.prototype.DrawingReady = function() {
      // function should be called by the painter when first drawing is completed
      this._ready_called_ = true;
      if ('_ready_callback_' in this) {
         JSROOT.CallBack(this._ready_callback_, this);
         delete this._ready_callback_;
      }
      return this;
   }

   JSROOT.TBasePainter.prototype.WhenReady = function(callback) {
      // call back will be called when painter ready with the drawing
      if ('_ready_called_' in this) return JSROOT.CallBack(callback, this);
      this._ready_callback_ = callback;
   }

   JSROOT.TBasePainter.prototype.GetObject = function() {
      return null;
   }

   JSROOT.TBasePainter.prototype.MatchObjectType = function(typ) {
      return false;
   }

   JSROOT.TBasePainter.prototype.UpdateObject = function(obj) {
      return false;
   }

   JSROOT.TBasePainter.prototype.RedrawPad = function(resize) {
   }

   JSROOT.TBasePainter.prototype.RedrawObject = function(obj) {
      if (this.UpdateObject(obj)) {
         var current = document.body.style.cursor;
         document.body.style.cursor = 'wait';
         this.RedrawPad();
         document.body.style.cursor = current;
      }
   }

   JSROOT.TBasePainter.prototype.CheckResize = function(arg) {
      return false; // indicate if resize is processed
   }

   JSROOT.TBasePainter.prototype.select_main = function() {
      // return d3.select for main element, defined with divid
      if ((this.divid === null) || (this.divid === undefined)) return d3.select(null);
      if ((typeof this.divid == "string") &&
          (this.divid.charAt(0) != "#")) return d3.select("#" + this.divid);
      return d3.select(this.divid);
   }

   JSROOT.TBasePainter.prototype.main_visible_rect = function() {
      // return rect with width/height which correcpond to the visible area of drawing region

      var render_to = this.select_main();

      var rect = render_to.node().getBoundingClientRect();

      var res = {};

      // this is size where canvas should be rendered
      res.width = Math.round(rect.width - this.GetStyleValue(render_to, 'padding-left') - this.GetStyleValue(render_to, 'padding-right'));
      res.height = Math.round(rect.height - this.GetStyleValue(render_to, 'padding-top') - this.GetStyleValue(render_to, 'padding-bottom'));

      return res;
   }

   JSROOT.TBasePainter.prototype.SetDivId = function(divid) {
      // base painter does not creates canvas or frames
      // it registered in the first child element
      if (arguments.length > 0)
         this.divid = divid;
      var main = this.select_main();
      var chld = main.node() ? main.node().firstChild : null;
      if (chld) chld.painter = this;
   }

   JSROOT.TBasePainter.prototype.SetItemName = function(name, opt) {
      if (typeof name === 'string') this._hitemname = name;
                               else delete this._hitemname;
      // only upate draw option, never delete. null specified when update drawing
      if (typeof opt === 'string') this._hdrawopt = opt;
   }

   JSROOT.TBasePainter.prototype.GetItemName = function() {
      return ('_hitemname' in this) ? this._hitemname : null;
   }

   JSROOT.TBasePainter.prototype.GetItemDrawOpt = function() {
      return ('_hdrawopt' in this) ? this._hdrawopt : "";
   }

   JSROOT.TBasePainter.prototype.CanZoomIn = function(axis,left,right) {
      // check if it makes sense to zoom inside specified axis range
      return false;
   }

   JSROOT.TBasePainter.prototype.GetStyleValue = function(select, name) {
      var value = select.style(name);
      if (!value) return 0;
      value = parseFloat(value.replace("px",""));
      return isNaN(value) ? 0 : value;
   }

   // ==============================================================================

   JSROOT.TObjectPainter = function(obj) {
      JSROOT.TBasePainter.call(this);
      this.draw_g = null; // container for all drawn objects
      this.pad_name = ""; // name of pad where object is drawn
      this.main = null;  // main painter, received from pad
      this.draw_object = ((obj!==undefined) && (typeof obj == 'object')) ? obj : null;
   }

   JSROOT.TObjectPainter.prototype = Object.create(JSROOT.TBasePainter.prototype);

   JSROOT.TObjectPainter.prototype.GetObject = function() {
      return this.draw_object;
   }

   JSROOT.TObjectPainter.prototype.MatchObjectType = function(arg) {
      if ((arg === undefined) || (arg === null) || (this.draw_object===null)) return false;
      if (typeof arg === 'string') return this.draw_object._typename === arg;
      return (typeof arg === 'object') && (this.draw_object._typename === arg._typename);
   }

   JSROOT.TObjectPainter.prototype.SetItemName = function(name, opt) {
      JSROOT.TBasePainter.prototype.SetItemName.call(this, name, opt);
      if (this.no_default_title || (name=="")) return;
      var can = this.svg_canvas();
      if (!can.empty()) can.select("title").text(name);
                   else this.select_main().attr("title", name);
   }

   JSROOT.TObjectPainter.prototype.UpdateObject = function(obj) {
      // generic method to update object
      // just copy all members from source object
      if (!this.MatchObjectType(obj)) return false;
      JSROOT.extend(this.GetObject(), obj);
      return true;
   }

   JSROOT.TObjectPainter.prototype.GetTipName = function(append) {
      var res = this.GetItemName();
      if (res===null) res = "";
      if ((res.length === 0) && ('fName' in this.GetObject()))
         res = this.GetObject().fName;
      if (res.lenght > 20) res = res.substr(0,17)+"...";
      if ((res.length > 0) && (append!==undefined)) res += append;
      return res;
   }

   JSROOT.TObjectPainter.prototype.pad_painter = function(active_pad) {
      var can = active_pad ? this.svg_pad() : this.svg_canvas();
      return can.empty() ? null : can.property('pad_painter');
   }

   JSROOT.TObjectPainter.prototype.CheckResize = function(arg) {
      // no painter - no resize
      var pad_painter = this.pad_painter();
      if (!pad_painter) return false;

      // only canvas should be checked
      pad_painter.CheckCanvasResize(arg);
      return true;
   }

   JSROOT.TObjectPainter.prototype.RemoveDrawG = function() {
      // generic method to delete all graphical elements, associated with painter
      if (this.draw_g != null) {
         this.draw_g.remove();
         this.draw_g = null;
      }
   }

   /** function (re)creates svg:g element used for specific object drawings
     *  either one attached svg:g to pad (take_pad==true) or to the frame (take_pad==false)
     *  svg:g element can be attached to different layers */
   JSROOT.TObjectPainter.prototype.RecreateDrawG = function(take_pad, layer) {
      if (this.draw_g) {
         // one should keep svg:g element on its place
         d3.selectAll(this.draw_g.node().childNodes).remove();
      } else
      if (take_pad) {
         if (typeof layer != 'string') layer = "text_layer";
         if (layer.charAt(0) == ".") layer = layer.substr(1);
         this.draw_g = this.svg_layer(layer).append("svg:g");
      } else {
         if (typeof layer != 'string') layer = ".main_layer";
         if (layer.charAt(0) != ".") layer = "." + layer;
         this.draw_g = this.svg_frame().select(layer).append("svg:g");
      }

      // set attributes for debugging
      if (this.draw_object!==null) {
         this.draw_g.attr('objname', this.draw_object.fName);
         this.draw_g.attr('objtype', this.draw_object._typename);
      }

      return this.draw_g;
   }

   /** This is main graphical SVG element, where all Canvas drawing are performed */
   JSROOT.TObjectPainter.prototype.svg_canvas = function() {
      return this.select_main().select(".root_canvas");
   }

   /** This is SVG element, correspondent to current pad */
   JSROOT.TObjectPainter.prototype.svg_pad = function(pad_name) {
      var c = this.svg_canvas();
      if (pad_name === undefined) pad_name = this.pad_name;
      if ((pad_name.length > 0) && !c.empty())
         c = c.select(".subpads_layer").select("[pad=" + pad_name + ']');
      return c;
   }

   /** Method selects immediate layer under canvas/pad main element */
   JSROOT.TObjectPainter.prototype.svg_layer = function(name, pad_name) {
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

   JSROOT.TObjectPainter.prototype.CurrentPadName = function(new_name) {
      var svg = this.svg_canvas();
      if (svg.empty()) return "";
      var curr = svg.property('current_pad');
      if (new_name !== undefined) svg.property('current_pad', new_name);
      return curr;
   }

   JSROOT.TObjectPainter.prototype.root_pad = function() {
      var pad_painter = this.pad_painter(true);
      return pad_painter ? pad_painter.pad : null;
   }

   /** Converts pad x or y coordinate into NDC value */
   JSROOT.TObjectPainter.prototype.ConvertToNDC = function(axis, value, isndc) {
      var pad = this.root_pad();
      if (isndc == null) isndc = false;

      if (isndc || (pad==null)) return value;

      if (axis=="y") {
         if (pad.fLogy)
            value = (value>0) ? JSROOT.log10(value) : pad.fUymin;
         return (value - pad.fY1) / (pad.fY2 - pad.fY1);
      }
      if (pad.fLogx)
         value = (value>0) ? JSROOT.log10(value) : pad.fUxmin;
      return (value - pad.fX1) / (pad.fX2 - pad.fX1);
   }

   /** Converts x or y coordinate into SVG coordinates,
    *  which could be used directly for drawing.
    *  Parameters: axis should be "x" or "y", value to convert, is ndc should be used */
   JSROOT.TObjectPainter.prototype.AxisToSvg = function(axis, value, ndc) {
      var main = this.main_painter();
      if ((main!=null) && !ndc)
         return axis=="y" ? main.gry(value) : main.grx(value);
      if (!ndc) value = this.ConvertToNDC(axis, value);
      if (axis=="y") return (1-value)*this.pad_height();
      return value*this.pad_width();
   }

   JSROOT.TObjectPainter.prototype.PadToSvg = function(axis, value, ndc) {
      return this.AxisToSvg(axis,value,ndc);
   }

   /** This is SVG element with current frame */
   JSROOT.TObjectPainter.prototype.svg_frame = function() {
      return this.svg_pad().select(".root_frame");
   }

   JSROOT.TObjectPainter.prototype.frame_painter = function() {
      var res = this.svg_frame().property('frame_painter');
      return (res===undefined) ? null : res;
   }

   JSROOT.TObjectPainter.prototype.pad_width = function(pad_name) {
      var res = this.svg_pad(pad_name).property("draw_width");
      return isNaN(res) ? 0 : res;
   }

   JSROOT.TObjectPainter.prototype.pad_height = function(pad_name) {
      var res = this.svg_pad(pad_name).property("draw_height");
      return isNaN(res) ? 0 : res;
   }

   JSROOT.TObjectPainter.prototype.frame_x = function() {
      var res = parseInt(this.svg_frame().attr("x"));
      return isNaN(res) ? 0 : res;
   }

   JSROOT.TObjectPainter.prototype.frame_y = function() {
      var res = parseInt(this.svg_frame().attr("y"));
      return isNaN(res) ? 0 : res;
   }

   JSROOT.TObjectPainter.prototype.frame_width = function() {
      var res = parseInt(this.svg_frame().attr("width"));
      return isNaN(res) ? 0 : res;
   }

   JSROOT.TObjectPainter.prototype.frame_height = function() {
      var res = parseInt(this.svg_frame().attr("height"));
      return isNaN(res) ? 0 : res;
   }

   JSROOT.TObjectPainter.prototype.embed_3d = function() {
      // returns embed mode for 3D drawings (three.js) inside SVG
      // 0 - no embedding,  3D drawing take full size of canvas
      // 1 - no embedding, canvas placed over svg with proper size (resize problem may appear)
      // 2 - normall embedding via ForeginObject, works only with Firefox

      if (JSROOT.gStyle.Embed3DinSVG < 2) return JSROOT.gStyle.Embed3DinSVG;
      if (JSROOT.browser.isFirefox /*|| JSROOT.browser.isWebKit*/)
         return JSROOT.gStyle.Embed3DinSVG; // use specified mode
      return 1; // default is overlay
   }

   JSROOT.TObjectPainter.prototype.access_3d_kind = function(new_value) {

      var svg = this.svg_pad();
      if (svg.empty()) return -1;

      // returns kind of currently created 3d canvas
      var kind = svg.property('can3d');
      if (new_value !== undefined) svg.property('can3d', new_value);
      return ((kind===null) || (kind===undefined)) ? -1 : kind;
   }

   JSROOT.TObjectPainter.prototype.size_for_3d = function(can3d) {
      // one uses frame sizes for the 3D drawing - like TH2/TH3 objects

      if (can3d === undefined) can3d = this.embed_3d();

      var pad = this.svg_pad();

      var clname = this.pad_name;
      if (clname == '') clname = 'canvas';
      clname = "draw3d_" + clname;

      if (pad.empty()) {
         // this is a case when object drawn without canvas

         var rect = this.main_visible_rect();

         if ((rect.height<10) && (rect.width>10)) {
            rect.height = Math.round(0.66*rect.width);
            this.select_main().style('height', rect.height + "px");
         }
         rect.x = 0; rect.y = 0; rect.clname = clname; rect.can3d = -1;
         return rect;
      }

      var elem = pad;
      if (can3d == 0) elem = this.svg_canvas();

      var size = { x: 0, y: 0, width: 100, height:100, clname: clname, can3d: can3d };

      if (this.frame_painter()!==null) {
         elem = this.svg_frame();
         size.x = elem.property("draw_x");
         size.y = elem.property("draw_y");
      }

      size.width = elem.property("draw_width");
      size.height = elem.property("draw_height");

      if ((this.frame_painter()===null) && (can3d > 0)) {
         size.x = Math.round(size.x + size.width*0.1);
         size.y = Math.round(size.y + size.height*0.1);
         size.width = Math.round(size.width*0.8);
         size.height = Math.round(size.height*0.8);
      }

      if (can3d === 1)
         this.CalcAbsolutePosition(this.svg_pad(), size);

      return size;
   }


   JSROOT.TObjectPainter.prototype.clear_3d_canvas = function() {
      var can3d = this.access_3d_kind(null);
      if (can3d < 0) return;

      var size = this.size_for_3d(can3d);

      if (size.can3d === 0) {
         d3.select(this.svg_canvas().node().nextSibling).remove(); // remove html5 canvas
         this.svg_canvas().style('display', null); // show SVG canvas
      } else {
         if (this.svg_pad().empty()) return;

         this.apply_3d_size(size).remove();

         this.svg_frame().style('display', null);
      }
   }

   JSROOT.TObjectPainter.prototype.add_3d_canvas = function(size, canv) {

      if ((canv == null) || (size.can3d < -1)) return;

      if (size.can3d === -1) {
         // case when 3D object drawn without canvas

         var main = this.select_main().node();
         if (main !== null) {
            main.appendChild(canv);
            canv['painter'] = this;
         }

         return;
      }

      this.access_3d_kind(size.can3d);

      if (size.can3d === 0) {
         this.svg_canvas().style('display', 'none'); // hide SVG canvas

         this.svg_canvas().node().parentNode.appendChild(canv); // add directly
      } else {
         if (this.svg_pad().empty()) return;

         // first hide normal frame
         this.svg_frame().style('display', 'none');

         var elem = this.apply_3d_size(size);

         elem.attr('title','').node().appendChild(canv);
      }
   }

   JSROOT.TObjectPainter.prototype.apply_3d_size = function(size, onlyget) {

      if (size.can3d < 0) return d3.select(null);

      var elem;

      if (size.can3d > 1) {

         var layer = this.svg_layer("special_layer");

         elem = layer.select("." + size.clname);
         if (onlyget) return elem;

         if (elem.empty())
            elem = layer.append("foreignObject").attr("class", size.clname);

         elem.attr('x', size.x)
             .attr('y', size.y)
             .attr('width', size.width)
             .attr('height', size.height)
             .attr('viewBox', "0 0 " + size.width + " " + size.height)
             .attr('preserveAspectRatio','xMidYMid');

      } else {
         var prnt = this.svg_canvas().node().parentNode;

         elem = d3.select(prnt).select("." + size.clname);
         if (onlyget) return elem;

         // force redraw by resize
         this.svg_canvas().property('redraw_by_resize', true);

         if (elem.empty())
            elem = d3.select(prnt).append('div').attr("class", size.clname);

         // our position inside canvas, but to set 'absolute' position we should use
         // canvas element offset relative to first parent with position
         var offx = 0, offy = 0;
         while (prnt && !offx && !offy) {
            if (getComputedStyle(prnt).position !== 'static') break;
            offx += prnt.offsetLeft;
            offy += prnt.offsetTop;
            prnt = prnt.parentNode;
         }

         elem.style('position','absolute')
             .style('left', (size.x + offx) + 'px')
             .style('top', (size.y + offy) + 'px')
             .style('width', size.width + 'px')
             .style('height', size.height + 'px');
      }

      return elem;
   }


   /** Returns main pad painter - normally TH1/TH2 painter, which draws all axis */
   JSROOT.TObjectPainter.prototype.main_painter = function() {
      if (this.main === null) {
         var svg_p = this.svg_pad();
         if (!svg_p.empty()) {
            this.main = svg_p.property('mainpainter');
            if (this.main === undefined) this.main = null;
         }
      }
      return this.main;
   }

   JSROOT.TObjectPainter.prototype.is_main_painter = function() {
      return this === this.main_painter();
   }

   JSROOT.TObjectPainter.prototype.SetDivId = function(divid, is_main, pad_name) {
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

      if ((is_main === null) || (is_main === undefined)) is_main = 0;

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
         var main = this.select_main();
         if (main.node() && main.node().firstChild)
            main.node().firstChild.painter = this;
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

   JSROOT.TObjectPainter.prototype.CalcAbsolutePosition = function(sel, pos) {
      while (!sel.empty() && !sel.classed('root_canvas')) {
         if (sel.classed('root_frame') || sel.classed('root_pad')) {
           pos.x += sel.property("draw_x");
           pos.y += sel.property("draw_y");
         }
         sel = d3.select(sel.node().parentNode);
      }
      return pos;
   }


   JSROOT.TObjectPainter.prototype.createAttFill = function(attfill, pattern, color, kind) {

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

         var defs = svg.select('defs');
         if (defs.empty())
            defs = svg.insert("svg:defs",":first-child");

         var line_color = this.color;
         this.color = "url(#" + id + ")";
         this.antialias = false;

         if (!defs.select("."+id).empty()) return true;

         var patt = defs.append('svg:pattern').attr("id", id).attr("class",id).attr("patternUnits","userSpaceOnUse");

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

         patt.selectAll('line').style("stroke",line_color).style("stroke-width", 1);
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

   JSROOT.TObjectPainter.prototype.ForEachPainter = function(userfunc) {
      // Iterate over all known painters

      var main = this.select_main();
      var painter = (main.node() && main.node().firstChild) ? main.node().firstChild['painter'] : null;
      if (painter!=null) return userfunc(painter);

      var pad_painter = this.pad_painter(true);
      if (pad_painter == null) return;

      userfunc(pad_painter);
      if ('painters' in pad_painter)
         for (var k = 0; k < pad_painter.painters.length; ++k)
            userfunc(pad_painter.painters[k]);
   }

   JSROOT.TObjectPainter.prototype.Cleanup = function() {
      // generic method to cleanup painters
      this.select_main().html("");
   }

   JSROOT.TObjectPainter.prototype.RedrawPad = function() {
      // call Redraw methods for each painter in the frame
      // if selobj specified, painter with selected object will be redrawn
      var pad_painter = this.pad_painter(true);
      if (pad_painter) pad_painter.Redraw();
   }

   JSROOT.TObjectPainter.prototype.SwitchTooltip = function(on) {
      var fp = this.frame_painter();
      if (fp) fp.ProcessTooltipEvent(null, on);
   }

   JSROOT.TObjectPainter.prototype.AddDrag = function(callback) {
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

      resize_corner1.style("opacity", "0")
                    .style("cursor", "nw-resize");

      resize_corner2.style("opacity", "0")
                    .style("cursor", "se-resize")
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


         var change_size = (newwidth !== rect_width()) || (newheight !== rect_height());
         var change_pos = (newx !== oldx) || (newy !== oldy);

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

      var drag_move = d3.behavior.drag().origin(Object)
         .on("dragstart",  function() {
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
          }).on("dragend", function() {
               if (drag_rect==null) return;

               d3.event.sourceEvent.preventDefault();

               if (complete_drag() === false)
                  if(callback['ctxmenu'] && ((new Date()).getTime() - drag_tm.getTime() > 600)) {
                     var rrr = resize_corner2.node().getBoundingClientRect();
                     pthis.ShowContextMenu('main', { clientX: rrr.left, clientY: rrr.top } );
                  }
            });

      var drag_resize = d3.behavior.drag().origin(Object)
        .on( "dragstart", function() {
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
         }).on( "dragend", function() {
            if (drag_rect == null) return;

            d3.event.sourceEvent.preventDefault();

            complete_drag();
         });

      if (!('only_resize' in callback)) {
         this.draw_g.style("cursor", "move").call(drag_move);
      }

      resize_corner1.call(drag_resize);
      resize_corner2.call(drag_resize);
   }

   JSROOT.TObjectPainter.prototype.startTouchMenu = function(kind) {
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

   JSROOT.TObjectPainter.prototype.endTouchMenu = function(kind) {
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

   JSROOT.TObjectPainter.prototype.AddColorMenuEntry = function(menu, name, value, set_func, fill_kind) {
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

   JSROOT.TObjectPainter.prototype.AddSizeMenuEntry = function(menu, name, min, max, step, value, set_func) {
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


   JSROOT.TObjectPainter.prototype.OpenWebsocket = function() {
      // create websocket for current object (canvas)
      // via websocket one recieved many extra information

      delete this._websocket;

      var path = window.location.href;
      path = path.replace("http://", "ws://");
      path = path.replace("https://", "wss://");
      var pos = path.indexOf("draw.htm");
      if (pos < 0) return;

      path = path.substr(0,pos) + "root.websocket";

      console.log('open websocket ' + path);

      var conn = new WebSocket(path);

      this._websocket = conn;

      var pthis = this, sum1 = 0, sum2 = 0, cnt = 0;

      conn.onopen = function() {
         console.log('websocket initialized');
         conn.send('READY'); // indicate that we are ready to recieve JSON code (or any other big peace)
      }

      conn.onmessage = function (e) {
         var d = e.data;
         if (typeof d != 'string') return console.log("msg",d);

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
            console.log("get MENUS ", typeof lst, lst.length, d.length-4);
            conn.send('READY'); // send ready message back
            if (typeof pthis._getmenu_callback == 'function')
               pthis._getmenu_callback(lst);
         } else {
            console.log("msg",d);
         }
      }

      conn.onclose = function() {
         console.log('websocket closed');

         delete pthis._websocket;
      }

      conn.onerror = function (err) {
         console.log("err",err);
         conn.close();
      }
   }


   JSROOT.TObjectPainter.prototype.FillObjectExecMenu = function(menu, call_back) {

      if (!('_websocket' in this) || ('_getmenu_callback' in this))
         return JSROOT.CallBack(call_back);

      function DoExecMenu(arg) {
         console.log('execute method ' + arg);

         if (this._websocket)
            this._websocket.send('EXEC'+arg);
      }

      function DoFillMenu(_menu, _call_back, items) {

        // avoid multiple call of the callback after timeout
        if (!this._getmenu_callback) return;
        delete this._getmenu_callback;

        if (items && items.length) {
           _menu.add("separator");
           _menu.add("sub:Online");

           for (var n=0;n<items.length;++n) {
              var item = items[n];
              if ('chk' in item)
                 _menu.addchk(item.chk, item.name, item.exec, DoExecMenu);
              else
                 _menu.add(item.name, item.exec, DoExecMenu);
           }

           _menu.add("endsub:");
        }

        JSROOT.CallBack(_call_back);
     }

     this._getmenu_callback = DoFillMenu.bind(this, menu, call_back);

     this._websocket.send('GETMENU'); // request menu items

     setTimeout(this._getmenu_callback, 2000); // set timeout to avoid menu hanging
   }

   JSROOT.TObjectPainter.prototype.DeleteAtt = function() {
      // remove all created draw attributes
      delete this.lineatt;
      delete this.fillatt;
      delete this.markeratt;
   }


   JSROOT.TObjectPainter.prototype.FillAttContextMenu = function(menu, preffix) {
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

   JSROOT.TObjectPainter.prototype.TextAttContextMenu = function(menu, prefix) {
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


   JSROOT.TObjectPainter.prototype.FillContextMenu = function(menu) {

      var title = this.GetTipName();
      if (this.GetObject() && ('_typename' in this.GetObject()))
         title = this.GetObject()._typename + "::" + title;

      menu.add("header:"+ title);

      this.FillAttContextMenu(menu);

      return menu.size() > 0;
   }


   JSROOT.TObjectPainter.prototype.FindInPrimitives = function(objname) {
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

   JSROOT.TObjectPainter.prototype.FindPainterFor = function(selobj,selname,seltype) {
      // try to find painter for sepcified object
      // can be used to find painter for some special objects, registered as
      // histogram functions

      var painter = this.pad_painter(true);
      var painters = (painter === null) ? null : painter.painters;
      if (painters === null) return null;

      for (var n = 0; n < painters.length; ++n) {
         var pobj = painters[n].GetObject();
         if (!pobj) continue;

         if (selobj && (pobj === selobj)) return painters[n];

         if (selname && (pobj.fName === selname)) return painters[n];

         if (seltype && (pobj._typename === seltype)) return painters[n];
      }

      return null;
   }

   JSROOT.TObjectPainter.prototype.ConfigureUserTooltipCallback = function(call_back, user_timeout) {
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

   JSROOT.TObjectPainter.prototype.IsUserTooltipCallback = function() {
      return typeof this.UserTooltipCallback == 'function';
   }

   JSROOT.TObjectPainter.prototype.ProvideUserTooltip = function(data) {

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

   JSROOT.TObjectPainter.prototype.Redraw = function() {
      // basic method, should be reimplemented in all derived objects
      // for the case when drawing should be repeated, probably with different
      // options
   }

   JSROOT.TObjectPainter.prototype.StartTextDrawing = function(font_face, font_size, draw_g, max_font_size) {
      // we need to preserve font to be able rescle at the end

      if (!draw_g) draw_g = this.draw_g;

      var font = JSROOT.Painter.getFontDetails(font_face, font_size);

      draw_g.call(font.func);

      draw_g.property('text_font', font);
      draw_g.property('mathjax_use', false);
      draw_g.property('text_factor', 0.);
      draw_g.property('max_text_width', 0); // keep maximal text width, use it later
      draw_g.property('max_font_size', max_font_size);
   }

   JSROOT.TObjectPainter.prototype.TextScaleFactor = function(value, draw_g) {
      // function used to remember maximal text scaling factor
      if (!draw_g) draw_g = this.draw_g;
      if (value && (value > draw_g.property('text_factor'))) draw_g.property('text_factor', value);
   }

   JSROOT.TObjectPainter.prototype.GetBoundarySizes = function(elem) {
      // getBBox does not work in mozilla when object is not displayed or not visisble :(
      // getBoundingClientRect() returns wrong sizes for MathJax
      // are there good solution?

      if (elem===null) { console.warn('empty node in GetBoundarySizes'); return { width:0, height:0 }; }
      var box = elem.getBoundingClientRect(); // works always, but returns sometimes wrong results
      if (parseInt(box.width) > 0) box = elem.getBBox(); // check that elements visible, request precise value
      var res = { width : parseInt(box.width), height : parseInt(box.height) };
      if ('left' in box) { res.x = parseInt(box.left); res.y = parseInt(box.right); } else
      if ('x' in box) { res.x = parseInt(box.x); res.y = parseInt(box.y); }
      return res;
   }

   JSROOT.TObjectPainter.prototype.FinishTextDrawing = function(draw_g, call_ready) {
      if (!draw_g) draw_g = this.draw_g;
      var pthis = this;

      var svgs = null;

      if (draw_g.property('mathjax_use')) {
         draw_g.property('mathjax_use', false);

         var missing = false;
         svgs = draw_g.selectAll(".math_svg");

         svgs.each(function() {
            var fo_g = d3.select(this);
            if (fo_g.node().parentNode !== draw_g.node()) return;
            var entry = fo_g.property('_element');
            if (d3.select(entry).select("svg").empty()) missing = true;
         });

         // is any svg missing we should wait until drawing is really finished
         if (missing) {
            JSROOT.AssertPrerequisites('mathjax', function() {
               if (typeof MathJax == 'object')
                  MathJax.Hub.Queue(["FinishTextDrawing", pthis, draw_g, call_ready]);
            });
            return null;
         }
      }

      if (svgs==null) svgs = draw_g.selectAll(".math_svg");

      var painter = this, svg_factor = 0.;

      // adjust font size (if there are normal text)
      var f = draw_g.property('text_factor');
      if ((f>0) && ((f<0.9) || (f>1.))) {
         var font = draw_g.property('text_font');
         font.size = Math.floor(font.size/f);
         if (draw_g.property('max_font_size') && (font.size>draw_g.property('max_font_size')))
            font.size = draw_g.property('max_font_size');
         draw_g.call(font.func);
      }

      // first remove dummy divs and check scaling coefficient
      svgs.each(function() {
         var fo_g = d3.select(this);
         if (fo_g.node().parentNode !== draw_g.node()) return;
         var entry = fo_g.property('_element'),
             rotate = fo_g.property('_rotate');

         fo_g.property('_element', null);

         var vvv = d3.select(entry).select("svg");
         if (vvv.empty()) {
            JSROOT.console('MathJax SVG ouptut error');
            return;
         }

         vvv.remove();
         document.body.removeChild(entry);

         fo_g.append(function() { return vvv.node(); });

         if (fo_g.property('_scale')) {
            var box = painter.GetBoundarySizes(fo_g.node());
            svg_factor = Math.max(svg_factor, 1.05*box.width / fo_g.property('_width'),
                                              1.05*box.height / fo_g.property('_height'));
         }
      });

      svgs.each(function() {
         var fo_g = d3.select(this);
         // only direct parent
         if (fo_g.node().parentNode !== draw_g.node()) return;

         if (svg_factor > 0.) {
            var m = fo_g.select("svg"); // MathJax svg
            var mw = m.attr("width"), mh = m.attr("height");
            if ((typeof mw == 'string') && (typeof mh == 'string') && (mw.indexOf("ex") > 0) && (mh.indexOf("ex") > 0)) {
               mw = parseFloat(mw.substr(0,mw.length-2));
               mh = parseFloat(mh.substr(0,mh.length-2));
               if ((mw>0) && (mh>0)) {
                  m.attr("width", (mw/svg_factor).toFixed(2)+"ex");
                  m.attr("height", (mh/svg_factor).toFixed(2)+"ex");
               }
            } else {
               JSROOT.console('Fail to downscale MathJax output');
            }
         }

         var box = painter.GetBoundarySizes(fo_g.node()), // sizes before rotation
             align = fo_g.property('_align'),
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

         if (align[0] == 'middle') tr[nx] += sign.x*(fo_w - box.width)/2; else
         if (align[0] == 'end')    tr[nx] += sign.x*(fo_w - box.width);

         if (align[1] == 'middle') tr[ny] += sign.y*(fo_h - box.height)/2; else
         if (align[1] == 'bottom') tr[ny] += sign.y*(fo_h - box.height);

         var trans = "translate("+tr.x+","+tr.y+")";
         if (rotate!==0) trans += " rotate("+rotate+",0,0)";

         fo_g.attr('transform', trans).attr('visibility', null);
      });

      // now hidden text after rescaling can be shown
      draw_g.selectAll('.hidden_text').attr('opacity', '1').classed('hidden_text',false);

      // if specified, call ready function
      JSROOT.CallBack(call_ready);

      return draw_g.property('max_text_width');
   }

   JSROOT.TObjectPainter.prototype.DrawText = function(align_arg, x, y, w, h, label, tcolor, latex_kind, draw_g) {

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
         if ((align_arg % 10) == 1) align[1] = 'bottom'; else
         if ((align_arg % 10) == 3) align[1] = 'top';
      }

      var scale = (w>0) && (h>0);

      if (latex_kind==null) latex_kind = 1;
      if (latex_kind<2)
         if (!JSROOT.Painter.isAnyLatex(label)) latex_kind = 0;

      var use_normal_text = ((JSROOT.gStyle.MathJax<1) && (latex_kind!==2)) || (latex_kind<1);

      // only Firefox can correctly rotate incapsulated SVG, produced by MathJax
      // if (!use_normal_text && (h<0) && !JSROOT.browser.isFirefox) use_normal_text = true;

      if (use_normal_text) {
         if (latex_kind>0) label = JSROOT.Painter.translateLaTeX(label);

         var pos_x = x.toFixed(1), pos_y = y.toFixed(1), pos_dy = null, middleline = false;

         if (w>0) {
            // adjust x position when scale into specified rectangle
            if (align[0]=="middle") pos_x = (x+w*0.5).toFixed(1); else
            if (align[0]=="end") pos_x = (x+w).toFixed(1);
         }

         if (h>0) {
            if (align[1] == 'bottom') pos_y = (y + h).toFixed(1); else
            if (align[1] == 'top') pos_dy = ".8em"; else {
               pos_y = (y + h/2).toFixed(1);
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
                         .text(label)
         if (pos_dy!=null) txt.attr("dy", pos_dy);
         if (middleline) txt.attr("dominant-baseline", "middle");

         var box = this.GetBoundarySizes(txt.node());

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

      var fo_g = draw_g.append("svg:g")
                       .attr('class', 'math_svg')
                       .attr('visibility','hidden')
                       .property('_x',x) // used for translation later
                       .property('_y',y)
                       .property('_width',w) // used to check scaling
                       .property('_height',h)
                       .property('_scale', scale)
                       .property('_rotate', rotate)
                       .property('_align', align);

      var element = document.createElement("div");
      d3.select(element).style("visibility", "hidden")
                        .style("overflow", "hidden")
                        .style("position", "absolute")
                        .html(JSROOT.Painter.translateMath(label, latex_kind, tcolor));
      document.body.appendChild(element);

      draw_g.property('mathjax_use', true);  // one need to know that mathjax is used
      fo_g.property('_element', element);

      JSROOT.AssertPrerequisites('mathjax', function() {
         if (typeof MathJax == 'object')
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, element]);
      });

      return 0;
   }

   // ===========================================================

   JSROOT.TFramePainter = function(tframe) {
      JSROOT.TObjectPainter.call(this, tframe);
      this.tooltip_enabled = true;
      this.tooltip_allowed = (JSROOT.gStyle.Tooltip > 0);
   }

   JSROOT.TFramePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TFramePainter.prototype.Shrink = function(shrink_left, shrink_right) {
      this.fX1NDC += shrink_left;
      this.fX2NDC -= shrink_right;
   }

   JSROOT.TFramePainter.prototype.UpdateAttributes = function(force) {
      var root_pad = this.root_pad(),
          tframe = this.GetObject();

      if ((this.fX1NDC === undefined) || (force && !this.modified_NDC)) {
         if (!root_pad) {
            JSROOT.extend(this, JSROOT.gStyle.FrameNDC);
         } else {
            JSROOT.extend(this, {
               fX1NDC: root_pad.fLeftMargin,
               fX2NDC: 1 - root_pad.fRightMargin,
               fY1NDC: root_pad.fBottomMargin,
               fY2NDC: 1 - root_pad.fTopMargin
            });
         }
      }

      if (this.fillatt === undefined) {
         if (tframe)
            this.fillatt = this.createAttFill(tframe);
         else
         if (root_pad)
            this.fillatt = this.createAttFill(null, root_pad.fFrameFillStyle, root_pad.fFrameFillColor);
         else
            this.fillatt = this.createAttFill(null, 1001, 0);

         // force white color for the frame
         if (this.fillatt.color == 'none') this.fillatt.color = 'white';
      }

      if (this.lineatt === undefined)
         this.lineatt = JSROOT.Painter.createAttLine(tframe ? tframe : 'black');
   }

   JSROOT.TFramePainter.prototype.Redraw = function() {

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

      // simple way to access painter via frame container
      this.draw_g.property('frame_painter', this);

      this.draw_g.attr("x", lm)
             .attr("y", tm)
             .attr("width", w)
             .attr("height", h)
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

      this.AddDrag({ obj: this, only_resize: true, minwidth: 20, minheight: 20,
                     redraw: this.RedrawPad.bind(this) });

      var tooltip_rect = this.draw_g.select(".interactive_rect");

      if (JSROOT.gStyle.Tooltip === 0)
         return tooltip_rect.remove();

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
                .style("opacity","0")
                .style("fill","none")
                .style("pointer-events", "visibleFill")
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


   JSROOT.TFramePainter.prototype.FillContextMenu = function(menu) {
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
            JSROOT.AssertPrerequisites("savepng", function() {
               saveSvgAsPng(top.node(), "frame.png");
            });
      });

      return true;
   }


   JSROOT.TFramePainter.prototype.IsTooltipShown = function() {
      // return true if tooltip is shown, use to prevent some other action
      if (JSROOT.gStyle.Tooltip < 1) return false;
      return ! (this.svg_layer("stat_layer").select(".objects_hints").empty());
   }

   JSROOT.TFramePainter.prototype.ProcessTooltipEvent = function(pnt, enabled) {

      if (enabled !== undefined) this.tooltip_enabled = enabled;

      if ((pnt === undefined) || !this.tooltip_allowed || !this.tooltip_enabled) pnt = null;

      var hints = [], nhints = 0, maxlen = 0, lastcolor1 = 0, usecolor1 = false,
          textheight = 11, hmargin = 3, wmargin = 3, hstep = 1.2,
          height = this.frame_height(),
          width = this.frame_width(),
          pp = this.pad_painter(true),
          maxhinty = this.pad_height() - this.draw_g.property('draw_y'),
          font = JSROOT.Painter.getFontDetails(160, textheight);

      // collect tooltips from pad painter - it has list of all drawn objects
      if (pp!==null) hints = pp.GetTooltips(pnt);

      if (pnt && pnt.touch) textheight = 15;

      for (var n=0; n < hints.length; ++n) {
         var hint = hints[n];
         if (!hint) continue;

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

      // end of closing tooltips
      if ((pnt === null) || (hints.length===0) || (maxlen===0)) {
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

      var curry = 10, // normal y coordiante
          gapy = 10, // y coordiante, taking into account all gaps
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
         var hint = hints[n];
         var group = hintsg.select(".painter_hint_"+n);
         if (hint===null) {
            group.remove();
            continue;
         }

         var was_empty = group.empty();

         if (was_empty)
            group = hintsg.append("svg:svg")
                          .attr("class", "painter_hint_"+n)
                          .style('overflow','hidden')
                          .attr("opacity","0")
                          .style("pointer-events","none");

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
            group.transition().duration(500).attrTween("opacity", translateFn());
      }

      actualw += 2*wmargin;

      var svgs = hintsg.selectAll("svg");

      if ((viewmode == "right") && (posx + actualw > width - 20)) {
         posx = width - actualw - 20;
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
      var p = new JSROOT.TFramePainter(obj);
      p.SetDivId(divid, 2);
      p.Redraw();
      return p.DrawingReady();
   }

   // ============================================================

   // base class for all objects, derived from TPave
   JSROOT.TPavePainter = function(pave) {
      JSROOT.TObjectPainter.call(this, pave);
      this.Enabled = true;
      this.UseContextMenu = true;
      this.UseTextColor = false; // indicates if text color used, enabled menu entry
   }

   JSROOT.TPavePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TPavePainter.prototype.DrawPave = function(arg) {
      // this draw only basic TPave

      this.UseTextColor = false;

      if (!this.Enabled)
         return this.RemoveDrawG();

      var pt = this.GetObject();

      if (pt.fInit===0) {
         pt.fInit = 1;
         var pad = this.root_pad();
         if (pt.fOption.indexOf("NDC")>=0) {
            pt.fX1NDC = pt.fX1; pt.fX2NDC = pt.fX2;
            pt.fY1NDC = pt.fY1; pt.fY2NDC = pt.fY2;
         } else
         if (pad !== null) {
            if (pad.fLogx) {
               if (pt.fX1 > 0) pt.fX1 = JSROOT.log10(pt.fX1);
               if (pt.fX2 > 0) pt.fX2 = JSROOT.log10(pt.fX2);
            }
            if (pad.fLogy) {
               if (pt.fY1 > 0) pt.fY1 = JSROOT.log10(pt.fY1);
               if (pt.fY2 > 0) pt.fY2 = JSROOT.log10(pt.fY2);
            }
            pt.fX1NDC = (pt.fX1-pad.fX1) / (pad.fX2 - pad.fX1);
            pt.fY1NDC = (pt.fY1-pad.fY1) / (pad.fY2 - pad.fY1);
            pt.fX2NDC = (pt.fX2-pad.fX1) / (pad.fX2 - pad.fX1);
            pt.fY2NDC = (pt.fY2-pad.fY1) / (pad.fY2 - pad.fY1);
         } else {
            pt.fX1NDC = pt.fY1NDC = 0.1;
            pt.fX2NDC = pt.fY2NDC = 0.9;
         }
      }

      var pos_x = Math.round(pt.fX1NDC * this.pad_width()),
          pos_y = Math.round((1.0 - pt.fY2NDC) * this.pad_height()),
          width = Math.round((pt.fX2NDC - pt.fX1NDC) * this.pad_width()),
          height = Math.round((pt.fY2NDC - pt.fY1NDC) * this.pad_height()),
          lwidth = pt.fBorderSize;

      // container used to recalculate coordinates
      this.RecreateDrawG(true, this.IsStats() ? "stat_layer" : "text_layer");

      // position and size required only for drag functions
      this.draw_g
           .attr("x", pos_x)
           .attr("y", pos_y)
           .attr("width", width)
           .attr("height", height)
           .attr("transform", "translate(" + pos_x + "," + pos_y + ")");

      // add shadow decoration before main rect
      if ((lwidth > 1) && (pt.fShadowColor > 0))
         this.draw_g.append("svg:path")
             .attr("d","M" + width + "," + height +
                      " v" + (-height + lwidth) + " h" + lwidth +
                      " v" + height + " h" + (-width) +
                      " v" + (-lwidth) + " Z")
            .style("fill", JSROOT.Painter.root_colors[pt.fShadowColor])
            .style("stroke", JSROOT.Painter.root_colors[pt.fShadowColor])
            .style("stroke-width", "1px");

      if (this.lineatt === undefined)
         this.lineatt = JSROOT.Painter.createAttLine(pt, lwidth>0 ? 1 : 0);
      if (this.fillatt === undefined)
         this.fillatt = this.createAttFill(pt);

      this.draw_g.append("rect")
          .attr("x", 0)
          .attr("y", 0)
          .attr("width", width)
          .attr("height", height)
          .style("pointer-events", "visibleFill")
          .call(this.fillatt.func)
          .call(this.lineatt.func);

      if ('PaveDrawFunc' in this)
         this.PaveDrawFunc(width, height, arg);

      this.AddDrag({ obj: pt, minwidth: 10, minheight: 20,
                     redraw: this.DrawPave.bind(this),
                     ctxmenu: JSROOT.touches && JSROOT.gStyle.ContextMenu && this.UseContextMenu });

      if (this.UseContextMenu && JSROOT.gStyle.ContextMenu)
         this.draw_g.on("contextmenu", this.ShowContextMenu.bind(this) );
   }

   JSROOT.TPavePainter.prototype.DrawPaveLabel = function(width, height) {
      this.UseTextColor = true;

      var pave = this.GetObject();

      this.StartTextDrawing(pave.fTextFont, height/1.2);

      this.DrawText(pave.fTextAlign, 0, 0, width, height, pave.fLabel, JSROOT.Painter.root_colors[pave.fTextColor]);

      this.FinishTextDrawing();
   }

   JSROOT.TPavePainter.prototype.DrawPaveText = function(width, height, refill) {

      if (refill && this.IsStats()) this.FillStatistic();

      var pt = this.GetObject(),
          tcolor = JSROOT.Painter.root_colors[pt.fTextColor],
          lwidth = pt.fBorderSize,
          first_stat = 0,
          num_cols = 0,
          nlines = pt.fLines.arr.length,
          lines = [],
          maxlen = 0;

      // adjust font size
      for (var j = 0; j < nlines; ++j) {
         var line = pt.fLines.arr[j].fTitle;
         lines.push(line);
         if (j>0) maxlen = Math.max(maxlen, line.length);
         if (!this.IsStats() || (j == 0) || (line.indexOf('|') < 0)) continue;
         if (first_stat === 0) first_stat = j;
         var parts = line.split("|");
         if (parts.length > num_cols)
            num_cols = parts.length;
      }

      // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
      var stepy = height / nlines, has_head = false, margin_x = pt.fMargin * width;

      this.StartTextDrawing(pt.fTextFont, height/(nlines * 1.2));

      if (nlines == 1) {
         this.DrawText(pt.fTextAlign, 0, 0, width, height, lines[0], tcolor);
         this.UseTextColor = true;
      } else {
         for (var j = 0; j < nlines; ++j) {
            var posy = j*stepy,
               jcolor = JSROOT.Painter.root_colors[pt.fLines.arr[j].fTextColor];
            if ((pt.fLines.arr[j].fTextColor == 0) || (jcolor===undefined)) {
               jcolor = tcolor;
               this.UseTextColor = true;
            }

            if (this.IsStats()) {
               if ((first_stat > 0) && (j >= first_stat)) {
                  var parts = lines[j].split("|");
                  for (var n = 0; n < parts.length; ++n)
                     this.DrawText("middle",
                                    width * n / num_cols, posy,
                                    width/num_cols, stepy, parts[n], jcolor);
               } else if (lines[j].indexOf('=') < 0) {
                  if (j==0) {
                     has_head = true;
                     if (lines[j].length > maxlen + 5)
                        lines[j] = lines[j].substr(0,maxlen+2) + "...";
                  }
                  this.DrawText((j == 0) ? "middle" : "start",
                                 margin_x, posy, width-2*margin_x, stepy, lines[j], jcolor);
               } else {
                  var parts = lines[j].split("="), sumw = 0;
                  for (var n = 0; n < 2; ++n)
                     sumw += this.DrawText((n == 0) ? "start" : "end",
                                      margin_x, posy, width-2*margin_x, stepy, parts[n], jcolor);
                  this.TextScaleFactor(1.05*sumw/(width-2*margin_x), this.draw_g);
               }
            } else {
               this.DrawText(pt.fTextAlign, margin_x, posy, width-2*margin_x, stepy, lines[j], jcolor);
            }
         }
      }

      var maxtw = this.FinishTextDrawing();

      if ((lwidth > 0) && has_head) {
         this.draw_g.append("svg:line")
                    .attr("x1", 0)
                    .attr("y1", stepy.toFixed(1))
                    .attr("x2", width)
                    .attr("y2", stepy.toFixed(1))
                    .call(this.lineatt.func);
      }

      if ((first_stat > 0) && (num_cols > 1)) {
         for (var nrow = first_stat; nrow < nlines; ++nrow)
            this.draw_g.append("svg:line")
                       .attr("x1", 0)
                       .attr("y1", (nrow * stepy).toFixed(1))
                       .attr("x2", width)
                       .attr("y2", (nrow * stepy).toFixed(1))
                       .call(this.lineatt.func);

         for (var ncol = 0; ncol < num_cols - 1; ++ncol)
            this.draw_g.append("svg:line")
                        .attr("x1", (width / num_cols * (ncol + 1)).toFixed(1))
                        .attr("y1", (first_stat * stepy).toFixed(1))
                        .attr("x2", (width / num_cols * (ncol + 1)).toFixed(1))
                        .attr("y2", height)
                        .call(this.lineatt.func);
      }

      if ((pt.fLabel.length>0) && !this.IsStats()) {
         var x = Math.round(width*0.25),
             y = Math.round(-height*0.02),
             w = Math.round(width*0.5),
             h = Math.round(height*0.04);

         var lbl_g = this.draw_g.append("svg:g");

         lbl_g.append("rect")
               .attr("x", x)
               .attr("y", y)
               .attr("width", w)
               .attr("height", h)
               .call(this.fillatt.func)
               .call(this.lineatt.func);

         this.StartTextDrawing(pt.fTextFont, h/1.5, lbl_g);

         this.DrawText(22, x, y, w, h, pt.fLabel, tcolor, 1, lbl_g);

         this.FinishTextDrawing(lbl_g);

         this.UseTextColor = true;
      }
   }

   JSROOT.TPavePainter.prototype.Format = function(value, fmt) {
      // method used to convert value to string according specified format
      // format can be like 5.4g or 4.2e or 6.4f
      if (!fmt) fmt = "stat";

      var pave = this.GetObject();

      if (fmt=="stat") {
         fmt = pave.fStatFormat;
         if (!fmt) fmt = JSROOT.gStyle.StatFormat;
      } else
      if (fmt=="fit") {
         fmt = pave.fFitFormat;
         if (!fmt) fmt = JSROOT.gStyle.FitFormat;
      } else
      if (fmt=="entries") {
         if (value < 1e9) return value.toFixed(0);
         fmt = "14.7g";
      } else
      if (fmt=="last") {
         fmt = this.lastformat;
      }

      delete this.lastformat;

      if (!fmt) fmt = "6.4g";

      var res = JSROOT.FFormat(value, fmt);

      this.lastformat = JSROOT.lastFFormat;

      return res;
   }

   JSROOT.TPavePainter.prototype.ShowContextMenu = function(evnt) {
      if (!evnt) {
         d3.event.stopPropagation(); // disable main context menu
         d3.event.preventDefault();  // disable browser context menu

         // one need to copy event, while after call back event may be changed
         evnt = d3.event;
      }

      var pthis = this, pave = this.GetObject();

      JSROOT.Painter.createMenu(function(menu) {
         menu.painter = pthis; // set as this in callbacks
         menu.add("header: " + pave._typename + "::" + pave.fName);
         if (pthis.IsStats()) {

            menu.add("SetStatFormat", function() {
               var fmt = prompt("Enter StatFormat", pave.fStatFormat);
               if (fmt!=null) {
                  pave.fStatFormat = fmt;
                  pthis.Redraw();
               }
            });
            menu.add("SetFitFormat", function() {
               var fmt = prompt("Enter FitFormat", pave.fFitFormat);
               if (fmt!=null) {
                  pave.fFitFormat = fmt;
                  pthis.Redraw();
               }
            });
            menu.add("separator");
            menu.add("sub:SetOptStat", function() {
               // todo - use jqury dialog here
               var fmt = prompt("Enter OptStat", pave.fOptStat);
               if (fmt!=null) { pave.fOptStat = parseInt(fmt); pthis.Redraw(); }
            });
            function AddStatOpt(pos, name) {
               var opt = (pos<10) ? pave.fOptStat : pave.fOptFit;
               opt = parseInt(parseInt(opt) / parseInt(Math.pow(10,pos % 10))) % 10;
               menu.addchk(opt, name, opt * 100 + pos, function(arg) {
                  var newopt = (arg % 100 < 10) ? pave.fOptStat : pave.fOptFit;
                  var oldopt = parseInt(arg / 100);
                  newopt -= (oldopt>0 ? oldopt : -1) * parseInt(Math.pow(10, arg % 10));
                  if (arg % 100 < 10) pave.fOptStat = newopt;
                  else pave.fOptFit = newopt;
                  pthis.Redraw();
               });
            }

            AddStatOpt(0, "Histogram name");
            AddStatOpt(1, "Entries");
            AddStatOpt(2, "Mean");
            AddStatOpt(3, "Std Dev");
            AddStatOpt(4, "Underflow");
            AddStatOpt(5, "Overflow");
            AddStatOpt(6, "Integral");
            AddStatOpt(7, "Skewness");
            AddStatOpt(8, "Kurtosis");
            menu.add("endsub:");

            menu.add("sub:SetOptFit", function() {
               // todo - use jqury dialog here
               var fmt = prompt("Enter OptStat", pave.fOptFit);
               if (fmt!=null) { pave.fOptFit = parseInt(fmt); pthis.Redraw(); }
            });
            AddStatOpt(10, "Fit parameters");
            AddStatOpt(11, "Par errors");
            AddStatOpt(12, "Chi square / NDF");
            AddStatOpt(13, "Probability");
            menu.add("endsub:");

            menu.add("separator");
         }

         if (pthis.UseTextColor)
            pthis.TextAttContextMenu(menu);

         pthis.FillAttContextMenu(menu);

         menu.show(evnt);
      }); // end menu creation
   }

   JSROOT.TPavePainter.prototype.IsStats = function() {
      return this.MatchObjectType('TPaveStats');
   }

   JSROOT.TPavePainter.prototype.FillStatistic = function() {
      var pave = this.GetObject(), main = this.main_painter();

      if (pave.fName !== "stats") return false;
      if ((main===null) || !('FillStatistic' in main)) return false;

      // no need to refill statistic if histogram is dummy
      if (main.IsDummyHisto()) return true;

      var dostat = new Number(pave.fOptStat);
      var dofit = new Number(pave.fOptFit);
      if (!dostat) dostat = JSROOT.gStyle.OptStat;
      if (!dofit) dofit = JSROOT.gStyle.OptFit;

      // make empty at the beginning
      pave.Clear();

      // we take statistic from first painter
      main.FillStatistic(this, dostat, dofit);

      return true;
   }

   JSROOT.TPavePainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;

      var pave = this.GetObject();

      if (!('modified_NDC' in pave)) {
         // if position was not modified interactively, update from source object
         pave.fInit = obj.fInit;
         pave.fX1 = obj.fX1; pave.fX2 = obj.fX2;
         pave.fY1 = obj.fY1; pave.fY2 = obj.fY2;
         pave.fX1NDC = obj.fX1NDC; pave.fX2NDC = obj.fX2NDC;
         pave.fY1NDC = obj.fY1NDC; pave.fY2NDC = obj.fY2NDC;
      }

      if (obj._typename === 'TPaveText') {
         pave.fLines = JSROOT.clone(obj.fLines);
         return true;
      } else
      if (obj._typename === 'TPaveLabel') {
         pave.fLabel = obj.fLabel;
         return true;
      } else
      if (obj._typename === 'TPaveStats') {
         pave.fOptStat = obj.fOptStat;
         pave.fOptFit = obj.fOptFit;
         return true;
      }

      return false;
   }

   JSROOT.TPavePainter.prototype.Redraw = function() {
      // if pavetext artificially disabled, do not redraw it

      this.DrawPave(true);
   }

   JSROOT.Painter.drawPaveText = function(divid, pave, opt) {

      // one could force drawing of PaveText on specific sub-pad
      var onpad = ((typeof opt == 'string') && (opt.indexOf("onpad:")==0)) ? opt.substr(6) : undefined;

      var painter = new JSROOT.TPavePainter(pave);
      painter.SetDivId(divid, 2, onpad);

      switch (pave._typename) {
         case "TPaveLabel":
            painter.PaveDrawFunc = painter.DrawPaveLabel;
            break;
         case "TPaveStats":
         case "TPaveText":
            painter.PaveDrawFunc = painter.DrawPaveText;
            break;
      }

      painter.Redraw();

      return painter.DrawingReady();
   }

   // ===========================================================================

   JSROOT.TPadPainter = function(pad, iscan) {
      JSROOT.TObjectPainter.call(this, pad);
      this.pad = pad;
      this.iscan = iscan; // indicate if workign with canvas
      this.this_pad_name = "";
      if (!this.iscan && (pad !== null) && ('fName' in pad))
         this.this_pad_name = pad.fName.replace(" ", "_"); // avoid empty symbol in pad name
      this.painters = new Array; // complete list of all painters in the pad
      this.has_canvas = true;
   }

   JSROOT.TPadPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TPadPainter.prototype.ButtonSize = function(fact) {
      return Math.round((!fact ? 1 : fact) * (this.iscan || !this.has_canvas ? 16 : 12));
   }

   JSROOT.TPadPainter.prototype.CreateCanvasSvg = function(check_resize, new_size) {

      var render_to = this.select_main();

      var rect = this.main_visible_rect();

      // this is size where canvas should be rendered
      var w = rect.width, h = rect.height;

      if ((typeof new_size == 'object') && (new_size!==null) && ('width' in new_size) && ('height' in new_size)) {
         w = new_size.width;
         h = new_size.height;
      }

      var factor = null, svg = null;

      if (check_resize > 0) {

         svg = this.svg_canvas();

         var oldw = svg.property('draw_width'), oldh = svg.property('draw_height');

         if ((w<=0) && (h<=0)) {
            svg.attr("visibility", "hidden");
            return false;
         } else {
            svg.attr("visibility", "visible");
            svg.select(".canvas_fillrect")
               .call(this.fillatt.func);
         }

         if (check_resize == 1) {
            if ((oldw == w) && (oldh == h)) return false;
         }

         factor = svg.property('height_factor');

         if (factor != null) {
            // if canvas was resize when created, resize height also now
            h = Math.round(w * factor);
            render_to.style('height', h+'px');
         }

         if ((check_resize==1) && (oldw>0) && (oldh>0) && !svg.property('redraw_by_resize'))
            if ((w/oldw>0.66) && (w/oldw<1.5) && (h/oldh>0.66) && (h/oldh<1.5)) {
               // not significant change in actual sizes, keep as it is
               // let browser scale SVG without our help
               return false;
            }

      } else {

         if ((h < 10) && (w > 0)) {
            // set aspect ratio for the place, where object will be drawn

            factor = 0.66;

            // for TCanvas reconstruct ratio between width and height
            if ((this.pad!==null) && ('fCw' in this.pad) && ('fCh' in this.pad) && (this.pad.fCw > 0)) {
               factor = this.pad.fCh / this.pad.fCw;
               if ((factor < 0.1) || (factor > 10))
                  factor = 0.66;
            }

            h = Math.round(w * factor);

            render_to.style('height', h+'px');
         }

         svg = this.select_main()
             .append("svg")
             .attr("class", "jsroot root_canvas")
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null) // this is custom property
             .property('current_pad', "") // this is custom property
             .property('redraw_by_resize', false); // could be enabled to force redraw by each resize

         svg.append("svg:title").text("ROOT canvas");
         svg.append("svg:rect").attr("class","canvas_fillrect")
                               .attr("x",0).attr("y",0).style("pointer-events", "visibleFill");
         svg.append("svg:g").attr("class","root_frame");
         svg.append("svg:g").attr("class","subpads_layer");
         svg.append("svg:g").attr("class","special_layer");
         svg.append("svg:g").attr("class","text_layer");
         svg.append("svg:g").attr("class","stat_layer");
         svg.append("svg:g").attr("class","btns_layer");

         if (JSROOT.gStyle.ContextMenu)
            svg.select(".canvas_fillrect").on("contextmenu", this.ShowContextMenu.bind(this));
      }

      if (!this.fillatt || !this.fillatt.changed)
         this.fillatt = this.createAttFill(this.pad, 1001, 0);

      if ((w<=0) || (h<=0)) {
         svg.attr("visibility", "hidden");
         w = 200; h = 100; // just to complete drawing
      } else {
         svg.attr("visibility", "visible");
      }

      svg.attr("x", 0)
         .attr("y", 0)
         .attr("width", "100%")
         .attr("height", "100%")
         .attr("viewBox", "0 0 " + w + " " + h)
         .attr("preserveAspectRatio", "none")  // we do not preserve relative ratio
         .property('height_factor', factor)
         .property('draw_x', 0)
         .property('draw_y', 0)
         .property('draw_width', w)
         .property('draw_height', h);

      svg.select(".canvas_fillrect")
         .attr("width",w)
         .attr("height",h)
         .call(this.fillatt.func);

      this.svg_layer("btns_layer").attr("transform","translate(2," + (h-this.ButtonSize(1.25)) + ")");

      return true;
   }

   JSROOT.TPadPainter.prototype.CreatePadSvg = function(only_resize) {
      if (!this.has_canvas)
         return this.CreateCanvasSvg(only_resize ? 2 : 0);

      var width = this.svg_canvas().property("draw_width"),
          height = this.svg_canvas().property("draw_height"),
          w = Math.round(this.pad.fAbsWNDC * width),
          h = Math.round(this.pad.fAbsHNDC * height),
          x = Math.round(this.pad.fAbsXlowNDC * width),
          y = Math.round(height - this.pad.fAbsYlowNDC * height) - h;

      var svg_pad = null, svg_rect = null, btns = null;

      if (only_resize) {
         svg_pad = this.svg_pad(this.this_pad_name);
         svg_rect = svg_pad.select(".root_pad_border");
         btns = this.svg_layer("btns_layer", this.this_pad_name);
      } else {
         svg_pad = this.svg_canvas().select(".subpads_layer")
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
         btns = svg_pad.append("svg:g").attr("class","btns_layer").property('nextx', 0);

         if (JSROOT.gStyle.ContextMenu)
            svg_pad.select(".root_pad_border").on("contextmenu", this.ShowContextMenu.bind(this));

         if (!this.fillatt || !this.fillatt.changed)
            this.fillatt = this.createAttFill(this.pad, 1001, 0);
         if (!this.lineatt || !this.lineatt.changed)
            this.lineatt = JSROOT.Painter.createAttLine(this.pad)
         if (this.pad.fBorderMode == 0) this.lineatt.color = 'none';
      }

      svg_pad.attr("transform", "translate(" + x + "," + y + ")")
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

      btns.attr("transform","translate("+ (w- btns.property('nextx') - this.ButtonSize(0.25)) + "," + (h-this.ButtonSize(1.25)) + ")");
   }

   JSROOT.TPadPainter.prototype.CheckColors = function(can) {
      if (can==null) return;



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

   JSROOT.TPadPainter.prototype.RemovePrimitive = function(obj) {
      if ((this.pad===null) || (this.pad.fPrimitives === null)) return;
      var indx = this.pad.fPrimitives.arr.indexOf(obj);
      if (indx>=0) this.pad.fPrimitives.RemoveAt(indx);
   }

   JSROOT.TPadPainter.prototype.FindPrimitive = function(exact_obj, classname, name) {
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

   JSROOT.TPadPainter.prototype.HasObjectsToDraw = function() {
      // return true if any objects beside sub-pads exists in the pad

      if ((this.pad===null) || !this.pad.fPrimitives || (this.pad.fPrimitives.arr.length==0)) return false;

      for (var n=0;n<this.pad.fPrimitives.arr.length;++n)
         if (this.pad.fPrimitives.arr[n] && this.pad.fPrimitives.arr[n]._typename != "TPad") return true;

      return false;
   }

   JSROOT.TPadPainter.prototype.DrawPrimitive = function(indx, callback) {
      if ((this.pad===null) || (indx >= this.pad.fPrimitives.arr.length))
         return JSROOT.CallBack(callback);

      var pp = JSROOT.draw(this.divid, this.pad.fPrimitives.arr[indx],  this.pad.fPrimitives.opt[indx]);

      if (pp === null) return this.DrawPrimitive(indx+1, callback);

      pp._primitive = true; // mark painter as belonging to primitives
      pp.WhenReady(this.DrawPrimitive.bind(this, indx+1, callback));
   }


   JSROOT.TPadPainter.prototype.GetTooltips = function(pnt) {
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

   JSROOT.TPadPainter.prototype.FillContextMenu = function(menu) {

      if (this.pad)
         menu.add("header: " + this.pad._typename + "::" + this.pad.fName);
      else
         menu.add("header: Canvas");

      if (this.iscan)
         menu.addchk((JSROOT.gStyle.Tooltip > 0), "Show tooltips", function() {
            JSROOT.gStyle.Tooltip = (JSROOT.gStyle.Tooltip === 0) ? 1 : -JSROOT.gStyle.Tooltip;
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

      var file_name = "canvas.png";
      if (!this.iscan) file_name = this.this_pad_name + ".png";

      menu.add("Save as "+file_name, file_name, function(arg) {
         // todo - use jqury dialog here
         var top = this.svg_pad(this.this_pad_name);
         if (!top.empty())
            JSROOT.AssertPrerequisites("savepng", function() {
               console.log('create', arg);
               top.selectAll(".btns_layer").style("display","none");
               saveSvgAsPng(top.node(), arg);
               top.selectAll(".btns_layer").style("display","");
            });
      });

      return true;
   }

   JSROOT.TPadPainter.prototype.ShowContextMenu = function(evnt) {
      if (!evnt) {

         // for debug purposes keep original context menu for small region in top-left corner
         var pos = d3.mouse(this.svg_pad(this.this_pad_name).node());
         if (pos && (pos.length==2) && (pos[0]>0) && (pos[0]<10) && (pos[1]>0) && pos[1]<10) return;

         d3.event.stopPropagation(); // disable main context menu
         d3.event.preventDefault();  // disable browser context menu

         // one need to copy event, while after call back event may be changed
         evnt = d3.event;
      }

      var pthis = this;

      JSROOT.Painter.createMenu(function(menu) {
         menu.painter = pthis; // set as this in callbacks

         pthis.FillContextMenu(menu);

         pthis.FillObjectExecMenu(menu, function() { menu.show(evnt); });

      }); // end menu creation
   }

   JSROOT.TPadPainter.prototype.Redraw = function(resize) {
      if (this.iscan)
         this.CreateCanvasSvg(2);
      else
         this.CreatePadSvg(true);

      // at the moment canvas painter donot redraw its subitems
      for (var i = 0; i < this.painters.length; ++i)
         this.painters[i].Redraw(resize);
   }

   JSROOT.TPadPainter.prototype.NumDrawnSubpads = function() {
      if (this.painters === undefined) return 0;

      var num = 0;

      for (var i = 0; i < this.painters.length; ++i) {
         var obj = this.painters[i].GetObject();
         if ((obj!==null) && (obj._typename === "TPad")) num++;
      }

      return num;
   }

   JSROOT.TPadPainter.prototype.RedrawByResize = function() {
      if (this.access_3d_kind() === 1) return true;

      for (var i = 0; i < this.painters.length; ++i)
         if (typeof this.painters[i].RedrawByResize === 'function')
            if (this.painters[i].RedrawByResize()) return true;

      return false;
   }

   JSROOT.TPadPainter.prototype.CheckCanvasResize = function(size, force) {
      if (!this.iscan && this.has_canvas) return false;

      if (size && (typeof size === 'object') && size.force) force = true;

      if (!force) force = this.RedrawByResize();

      var changed = this.CreateCanvasSvg(force ? 2 : 1, size);

      // if canvas changed, redraw all its subitems
      if (changed)
         for (var i = 0; i < this.painters.length; ++i)
            this.painters[i].Redraw(true);

      return changed;
   }

   JSROOT.TPadPainter.prototype.UpdateObject = function(obj) {
      if (!obj || !('fPrimitives' in obj)) return false;

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

   JSROOT.TPadPainter.prototype.ItemContextMenu = function(name) {
       var rrr = this.svg_pad(this.this_pad_name).node().getBoundingClientRect();
       var evnt = { clientX: rrr.left+10, clientY: rrr.top + 10 };

       // use timeout to avoid conflict with mouse click and automatic menu close
       if (name=="pad")
          return setTimeout(this.ShowContextMenu.bind(this, evnt), 50);

       var selp = null, selkind;

       switch(name) {
          case "xaxis": selp = this.main_painter(); selkind = "x"; break;
          case "yaxis": selp = this.main_painter(); selkind = "y"; break;
          case "frame": selp = this.frame_painter(); break;
          default: {
             var indx = parseInt(name);
             if (!isNaN(indx)) selp = this.painters[indx];
          }
       }

       if (!selp || (typeof selp.FillContextMenu !== 'function')) return;

       JSROOT.Painter.createMenu(function(menu) {
          menu.painter = selp;
          if (selp.FillContextMenu(menu,selkind))
             setTimeout(menu.show.bind(menu, evnt), 50);
       });

   }

   JSROOT.TPadPainter.prototype.PadButtonClick = function(funcname) {

      var elem = null, filename = "";

      if (funcname == "CanvasSnapShot") {
         elem = this.svg_canvas();
         filename = (this.pad ? this.pad.fName : "jsroot_canvas") + ".png";
      } else
      if (funcname == "PadSnapShot") {
         elem = this.svg_pad(this.this_pad_name);
         filename = this.this_pad_name + ".png";
      }
      if ((elem!==null) && !elem.empty()) {
         var main = elem.property('mainpainter');

         if ((elem.property('can3d') === 1) && (main!==undefined) && (main.renderer!==undefined)) {
            var dataUrl = main.renderer.domElement.toDataURL("image/png");
            dataUrl.replace("image/png", "image/octet-stream");
            var link = document.createElement('a');
            if (typeof link.download === 'string') {
               document.body.appendChild(link); //Firefox requires the link to be in the body
               link.download = filename;
               link.href = dataUrl;
               link.click();
               document.body.removeChild(link); //remove the link when done
            }
         } else {
            JSROOT.AssertPrerequisites("savepng", function() {
               elem.selectAll(".btns_layer").style("display","none");
               saveSvgAsPng(elem.node(), filename);
               elem.selectAll(".btns_layer").style("display","");
            });
         }
         return;
      }

      if (funcname == "PadContextMenus") {

         var pthis = this, evnt = d3.event;

         d3.event.preventDefault();
         d3.event.stopPropagation();

         JSROOT.Painter.createMenu(function(menu) {
            menu.painter = pthis; // set as this in callbacks
            menu.add("header:Menus");

            if (pthis.iscan)
               menu.add("Canvas", "pad", pthis.ItemContextMenu);
            else
               menu.add("Pad", "pad", pthis.ItemContextMenu);

            if (pthis.frame_painter())
               menu.add("Frame", "frame", pthis.ItemContextMenu);

            if (pthis.main_painter()) {
               menu.add("X axis", "xaxis", pthis.ItemContextMenu);
               menu.add("Y axis", "yaxis", pthis.ItemContextMenu);
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

   JSROOT.TPadPainter.prototype.AddButton = function(btn, tooltip, funcname) {

      // do not add buttons when not allowed
      if (!JSROOT.gStyle.ToolBar) return;

      var group = this.svg_layer("btns_layer", this.this_pad_name);
      if (group.empty()) return;

      // avoid buttons with duplicate names
      if (!group.select("[name=" + funcname + ']').empty()) return;

      var x = group.property("nextx");
      if (x===undefined) x = 0;

      var iscan = this.iscan || !this.has_canvas;

      var svg = group.append("svg:svg")
                     .attr("class", "svg_toolbar_btn")
                     .attr("name", funcname)
                     .attr("x", x)
                     .attr("y", 0)
                     .attr("width",this.ButtonSize()+"px")
                     .attr("height",this.ButtonSize()+"px")
                     .attr("viewBox", "0 0 512 512")
                     .style("overflow","hidden");

      svg.append("svg:title").text(tooltip + (iscan ? "" : (" on pad " + this.this_pad_name)));

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

      // special rect to correctly get mouse events for whole button area
      svg.append("svg:rect").attr("x",0).attr("y",0).attr("width",512).attr("height",512)
         .style("opacity","0").style("fill","none").style("pointer-events", "visibleFill");

      svg.on("click", this.PadButtonClick.bind(this, funcname));

      group.property("nextx", x+this.ButtonSize(1.25));

      if (!iscan)
         group.attr("transform","translate("+ (this.pad_width(this.this_pad_name) - group.property('nextx')-this.ButtonSize(0.25)) + "," + (this.pad_height(this.this_pad_name)-this.ButtonSize(1.25)) + ")");

      if (!iscan && (funcname.indexOf("Pad")!=0) && (this.pad_painter()!==this))
         this.pad_painter().AddButton(btn, tooltip, funcname);
   }

   JSROOT.Painter.drawCanvas = function(divid, can, opt) {
      var nocanvas = (can===null);
      if (nocanvas) can = JSROOT.Create("TCanvas");

      var painter = new JSROOT.TPadPainter(can, true);
      if (opt && (opt.indexOf("white")>=0)) can.fFillColor = 0;

      painter.SetDivId(divid, -1); // just assign id
      painter.CheckColors(can);
      painter.CreateCanvasSvg(0);
      painter.SetDivId(divid);  // now add to painters list

      painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "CanvasSnapShot");
      if (JSROOT.gStyle.ContextMenu)
         painter.AddButton(JSROOT.ToolbarIcons.question, "Access context menus", "PadContextMenus");

      if (nocanvas && opt.indexOf("noframe") < 0)
         JSROOT.Painter.drawFrame(divid, null);

      painter.DrawPrimitive(0, function() { painter.DrawingReady(); });
      return painter;
   }

   JSROOT.Painter.drawPad = function(divid, pad, opt) {
      var painter = new JSROOT.TPadPainter(pad, false);

      if (pad && opt && (opt.indexOf("white")>=0)) pad.fFillColor = 0;

      painter.SetDivId(divid); // pad painter will be registered in the canvas painters list

      if (painter.svg_canvas().empty()) {
         painter.has_canvas = false;
         painter.this_pad_name = "";
      }

      painter.CreatePadSvg();

      if (painter.MatchObjectType("TPad") && (!painter.has_canvas || painter.HasObjectsToDraw())) {
         painter.AddButton(JSROOT.ToolbarIcons.camera, "Create PNG", "PadSnapShot");
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

   // =======================================================================

   JSROOT.TAxisPainter = function(axis, embedded) {
      JSROOT.TObjectPainter.call(this, axis);

      this.embedded = embedded; // indicate that painter embedded into the histo painter

      this.name = "yaxis";
      this.kind = "normal";
      this.func = null;
      this.order = 0; // scaling order for axis labels

      this.full_min = 0;
      this.full_max = 1;
      this.scale_min = 0;
      this.scale_max = 1;
      this.ticks = []; // list of major ticks
   }

   JSROOT.TAxisPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TAxisPainter.prototype.SetAxisConfig = function(name, kind, func, min, max, smin, smax) {
      this.name = name;
      this.kind = kind;
      this.func = func;

      this.full_min = min;
      this.full_max = max;
      this.scale_min = smin;
      this.scale_max = smax;
   }

   JSROOT.TAxisPainter.prototype.CreateFormatFuncs = function() {

      var axis = this.GetObject(),
          is_gaxis = (axis && axis._typename === 'TGaxis');

      delete this.format;// remove formatting func

      var ndiv = 508;
      if (axis !== null)
         ndiv = Math.max(is_gaxis ? axis.fNdiv : axis.fNdivisions, 4) ;

      this.nticks = ndiv % 100;
      this.nticks2 = (ndiv % 10000 - this.nticks) / 100;
      this.nticks3 = Math.floor(ndiv/10000);

      if (axis && !is_gaxis && (this.nticks > 7)) this.nticks = 7;

      var gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);
      if (gr_range<=0) gr_range = 100;

      if (this.kind == 'time') {
         if (this.nticks > 8) this.nticks = 8;

         var scale_range = this.scale_max - this.scale_min;

         var tf1 = JSROOT.Painter.getTimeFormat(axis);
         if ((tf1.length == 0) || (scale_range < 0.1 * (this.full_max - this.full_min)))
            tf1 = JSROOT.Painter.chooseTimeFormat(scale_range / this.nticks, true);
         var tf2 = JSROOT.Painter.chooseTimeFormat(scale_range / gr_range, false);

         this.tfunc1 = this.tfunc2 = d3.time.format(tf1);
         if (tf2!==tf1)
            this.tfunc2 = d3.time.format(tf2);

         this.format = function(d, asticks) {
            return asticks ? this.tfunc1(d) : this.tfunc2(d);
         }

      } else
      if (this.kind == 'log') {
         this.nticks2 = 1;
         this.noexp = axis ? axis.TestBit(JSROOT.EAxisBits.kNoExponent) : false;
         if ((this.scale_max < 300) && (this.scale_min > 0.3)) this.noexp = true;
         this.moreloglabels = axis ? axis.TestBit(JSROOT.EAxisBits.kMoreLogLabels) : false;

         this.format = function(d, asticks, notickexp) {

            var val = parseFloat(d);

            if (!asticks) {
               var rnd = Math.round(val);
               return ((rnd === val) && (Math.abs(rnd)<1e9)) ? rnd.toString() : val.toExponential(4);
            }

            if (val <= 0) return null;
            var vlog = JSROOT.log10(val);
            if (this.moreloglabels || (Math.abs(vlog - Math.round(vlog))<0.001)) {
               if (!this.noexp && !notickexp)
                  return JSROOT.Painter.formatExp(val.toExponential(0));
               else
               if (vlog<0)
                  return val.toFixed(Math.round(-vlog+0.5));
               else
                  return val.toFixed(0);
            }
            return null;
         }
      } else
      if (this.kind == 'labels') {
         this.nticks = 50; // for text output allow max 50 names
         var scale_range = this.scale_max - this.scale_min;
         if (this.nticks > scale_range)
            this.nticks = Math.round(scale_range);
         this.nticks2 = 1;

         this.axis = axis;

         this.format = function(d) {
            var indx = Math.round(parseInt(d)) + 1;
            if ((indx<1) || (indx>this.axis.fNbins)) return null;
            for (var i = 0; i < this.axis.fLabels.arr.length; ++i) {
               var tstr = this.axis.fLabels.arr[i];
               if (tstr.fUniqueID == indx) return tstr.fString;
            }
            return null;
         }
      } else {

         this.range = Math.abs(this.scale_max - this.scale_min);
         if (this.range <= 0)
            this.ndig = -3;
         else
            this.ndig = Math.round(JSROOT.log10(this.nticks / this.range) + 0.7);

         this.format = function(d, asticks) {
            var val = parseFloat(d), rnd = Math.round(val);
            if (asticks) {
               if (this.order===0) {
                  if (val === rnd) return rnd.toString();
                  if (Math.abs(val) < 1e-10 * this.range) return 0;
                  val = val.toFixed(this.ndig > 0 ? this.ndig : 0);
                  if ((typeof d == 'string') && (d.length <= val.length+1)) return d;
                  return val;
               }
               val = val / Math.pow(10, this.order);
               rnd = Math.round(val);
               if (val === rnd) return rnd.toString();
               return val.toFixed(this.ndig + this.order > 0 ? this.ndig + this.order : 0 );
            }

            if (val === rnd)
               return (Math.abs(rnd)<1e9) ? rnd.toString() : val.toExponential(4);
            return val.toFixed(this.ndig+2 > 0 ? this.ndig+2 : 0);
         }
      }
   }

   JSROOT.TAxisPainter.prototype.CreateTicks = function() {
      // function used to create array with minor/middle/major ticks

      var handle = { nminor: 0, nmiddle: 0, nmajor: 0, func: this.func };

      handle.minor = handle.middle = handle.major = this.func.ticks(this.nticks);

      if (this.nticks2 > 1) {
         handle.minor = handle.middle = this.func.ticks(handle.major.length * this.nticks2);

         var gr_range = Math.abs(this.func.range()[1] - this.func.range()[0]);

         // avoid black filling by middle-size
         if ((handle.middle.length <= handle.major.length) || (handle.middle.length > gr_range/3.5)) {
            handle.minor = handle.middle = handle.major;
         } else
         if ((this.nticks3 > 1) && (this.kind !== 'log'))  {
            handle.minor = this.func.ticks(handle.middle.length * this.nticks3);
            if ((handle.minor.length <= handle.middle.length) || (handle.minor.length > gr_range/1.7)) handle.minor = handle.middle;
         }
      }

      handle.reset = function() {
         this.nminor = this.nmiddle = this.nmajor = 0;
      }

      handle.next = function(doround) {
         if (this.nminor >= this.minor.length) return false;

         this.tick = this.minor[this.nminor++];
         this.grpos = this.func(this.tick);
         if (doround) this.grpos = Math.round(this.grpos);
         this.kind = 3;

         if ((this.nmiddle < this.middle.length) && (Math.abs(this.grpos - this.func(this.middle[this.nmiddle])) < 1)) {
            this.nmiddle++;
            this.kind = 2;
         }

         if ((this.nmajor < this.major.length) && (Math.abs(this.grpos - this.func(this.major[this.nmajor])) < 1) ) {
            this.nmajor++;
            this.kind = 1;
         }
         return true;
      }

      handle.last_major = function() {
         return (this.kind !== 1) ? false : this.nmajor == this.major.length;
      }

      handle.next_major_grpos = function() {
         if (this.nmajor >= this.major.length) return null;
         return this.func(this.major[this.nmajor]);
      }

      return handle;
   }

   JSROOT.TAxisPainter.prototype.IsCenterLabels = function() {
      if (this.kind === 'labels') return true;
      if (this.kind === 'log') return false;
      var axis = this.GetObject();
      return axis && axis.TestBit(JSROOT.EAxisBits.kCenterLabels);
   }

   JSROOT.TAxisPainter.prototype.DrawAxis = function(vertical, layer, w, h, transform, reverse, second_shift) {
      // function draw complete TAxis
      // later will be used to draw TGaxis

      var axis = this.GetObject(),
          is_gaxis = (axis && axis._typename === 'TGaxis'),
          side = (this.name === "zaxis") ? -1  : 1, both_sides = 0,
          axis_g = layer, tickSize = 10, scaling_size = 100, text_scaling_size = 100;

      this.vertical = vertical;

      // shift for second ticks set (if any)
      if (!second_shift) second_shift = 0;

      if (is_gaxis) {
         if (!this.lineatt) this.lineatt = JSROOT.Painter.createAttLine(axis);
         scaling_size = (vertical ? this.pad_width() : this.pad_height());
         text_scaling_size = Math.min(this.pad_width(), this.pad_height());
         tickSize = Math.round(axis.fTickSize * scaling_size);
      } else {
         if (!this.lineatt) this.lineatt = JSROOT.Painter.createAttLine(axis.fAxisColor, 1);
         scaling_size = (vertical ? w : h);
         tickSize = Math.round(axis.fTickLength * scaling_size);
         text_scaling_size = Math.min(w,h);
      }

      if (!is_gaxis || (this.name === "zaxis")) {
         axis_g = layer.select("." + this.name + "_container");
         if (axis_g.empty())
            axis_g = layer.append("svg:g").attr("class",this.name+"_container");
         else
            axis_g.selectAll("*").remove();
      } else {

         if ((axis.fChopt.indexOf("-")>=0) && (axis.fChopt.indexOf("+")<0)) side = -1; else
         if (vertical && axis.fChopt=="+L") side = -1; else
         if ((axis.fChopt.indexOf("-")>=0) && (axis.fChopt.indexOf("+")>=0)) { side = 1; both_sides = 1; }

         axis_g.append("svg:line")
               .attr("x1",0).attr("y1",0)
               .attr("x1",vertical ? 0 : w)
               .attr("y1", vertical ? h : 0)
               .call(this.lineatt.func);
      }

      if (transform !== undefined)
         axis_g.attr("transform", transform);

      this.CreateFormatFuncs();

      var center_lbls = this.IsCenterLabels(),
          res = "", res2 = "", lastpos = 0, lasth = 0, textscale = 1;

      // first draw ticks

      this.ticks = [];

      var handle = this.CreateTicks();

      while (handle.next(true)) {
         var h1 = Math.round(tickSize/4), h2 = 0;

         if (handle.kind < 3)
            h1 = Math.round(tickSize/2);

         if (handle.kind == 1) {
            // if not showing lables, not show large tick
            if (!('format' in this) || (this.format(handle.tick,true)!==null)) h1 = tickSize;
            this.ticks.push(handle.grpos); // keep graphical positions of major ticks
         }

         if (both_sides > 0) h2 = -h1; else
         if (side < 0) { h2 = -h1; h1 = 0; } else { h2 = 0; }

         if (res.length == 0) {
            res = vertical ? ("M"+h1+","+handle.grpos) : ("M"+handle.grpos+","+(-h1));
            res2 = vertical ? ("M"+(second_shift-h1)+","+handle.grpos) : ("M"+handle.grpos+","+(second_shift+h1));
         } else {
            res += vertical ? ("m"+(h1-lasth)+","+(handle.grpos-lastpos)) : ("m"+(handle.grpos-lastpos)+","+(lasth-h1));
            res2 += vertical ? ("m"+(lasth-h1)+","+(handle.grpos-lastpos)) : ("m"+(handle.grpos-lastpos)+","+(h1-lasth));
         }

         res += vertical ? ("h"+ (h2-h1)) : ("v"+ (h1-h2));
         res2 += vertical ? ("h"+ (h1-h2)) : ("v"+ (h2-h1));

         lastpos = handle.grpos;
         lasth = h2;
      }

      if (res.length > 0)
         axis_g.append("svg:path").attr("d", res).call(this.lineatt.func);

      if ((second_shift!==0) && (res2.length>0))
         axis_g.append("svg:path").attr("d", res2).call(this.lineatt.func);

      var last = vertical ? h : 0,
          labelsize = (axis.fLabelSize >= 1) ? axis.fLabelSize : Math.round(axis.fLabelSize * (is_gaxis ? this.pad_height() : h)),
          labelfont = JSROOT.Painter.getFontDetails(axis.fLabelFont, labelsize),
          label_color = JSROOT.Painter.root_colors[axis.fLabelColor],
          labeloffset = 3 + Math.round(axis.fLabelOffset * scaling_size),
          label_g = axis_g.append("svg:g")
                         .attr("class","axis_labels")
                         .call(labelfont.func);

      this.order = 0;
      if ((this.kind=="normal") && vertical && !axis.TestBit(JSROOT.EAxisBits.kNoExponent)) {
         var maxtick = Math.max(Math.abs(handle.major[0]),Math.abs(handle.major[handle.major.length-1]));
         for(var order=18;order>-18;order-=3) {
            if (order===0) continue;
            if ((order<0) && ((this.range>=0.1) || (maxtick>=1.))) break;
            var mult = Math.pow(10, order);
            if ((this.range > mult * 9.99999) || ((maxtick > mult*50) && (this.range > mult * 0.05))) {
               this.order = order;
               break;
            }
         }
      }

      for (var nmajor=0;nmajor<handle.major.length;++nmajor) {
         var pos = Math.round(this.func(handle.major[nmajor]));
         var lbl = this.format(handle.major[nmajor], true);
         if (lbl === null) continue;

         var t = label_g.append("svg:text").attr("fill", label_color).text(lbl);

         if (vertical)
            t.attr("x", -labeloffset*side)
             .attr("y", pos)
             .style("text-anchor", (side > 0) ? "end" : "start")
             .style("dominant-baseline", "middle");
         else
            t.attr("x", pos)
             .attr("y", 2+labeloffset*side  + both_sides*tickSize)
             .attr("dy", (side > 0) ? ".7em" : "-.3em")
             .style("text-anchor", "middle");

         var tsize = this.GetBoundarySizes(t.node());
         var space_before = (nmajor > 0) ? (pos - last) : (vertical ? h/2 : w/2);
         var space_after = (nmajor < handle.major.length-1) ? (Math.round(this.func(handle.major[nmajor+1])) - pos) : space_before;
         var space = Math.min(Math.abs(space_before), Math.abs(space_after));

         if (vertical) {

            if ((space > 0) && (tsize.height > 5) && (this.kind !== 'log'))
               textscale = Math.min(textscale, space / tsize.height);

            if (center_lbls) {
               // if position too far top, remove label
               if (pos + space_after/2 - textscale*tsize.height/2 < -10)
                  t.remove();
               else
                  t.attr("y", Math.round(pos + space_after/2));
            }

         } else {

            // test if label consume too much space
            if ((space > 0) && (tsize.width > 10) && (this.kind !== 'log'))
               textscale = Math.min(textscale, space / tsize.width);

            if (center_lbls) {
               // if position too far right, remove label
               if (pos + space_after/2 - textscale*tsize.width/2 > w - 10)
                  t.remove();
               else
                  t.attr("x", Math.round(pos + space_after/2));
            }
         }

         last = pos;
     }

     if (this.order!==0) {
        var val = Math.pow(10,this.order).toExponential(0);

        var t = label_g.append("svg:text").attr("fill", label_color).text('\xD7' + JSROOT.Painter.formatExp(val));

        if (vertical)
           t.attr("x", labeloffset)
            .attr("y", 0)
            .style("text-anchor", "start")
            .style("dominant-baseline", "middle")
            .attr("dy", "-.5em");
     }


     if ((textscale>0) && (textscale<1.)) {
        // rotate X lables if they are too big
        if ((textscale < 0.7) && !vertical && (side>0)) {
           label_g.selectAll("text").each(function() {
              var txt = d3.select(this), x = txt.attr("x"), y = txt.attr("y") - 5;

              txt.attr("transform", "translate(" + x + "," + y + ") rotate(25)")
                 .style("text-anchor", "start")
                 .attr("x",null).attr("y",null);
           });
           textscale *= 3.5;
        }
        // round to upper boundary for calculated value like 4.4
        labelfont.size = Math.floor(labelfont.size * textscale + 0.7);
        label_g.call(labelfont.func);
     }

     if (axis.fTitle.length > 0) {
         var title_g = axis_g.append("svg:g").attr("class", "axis_title"),
             title_fontsize = (axis.fTitleSize >= 1) ? axis.fTitleSize : Math.round(axis.fTitleSize * text_scaling_size),
             center = axis.TestBit(JSROOT.EAxisBits.kCenterTitle),
             rotate = axis.TestBit(JSROOT.EAxisBits.kRotateTitle) ? -1 : 1,
             title_color = JSROOT.Painter.root_colors[axis.fTitleColor];

         this.StartTextDrawing(axis.fTitleFont, title_fontsize, title_g);

         var myxor = ((rotate<0) && !reverse) || ((rotate>=0) && reverse);

         if (vertical) {
            var xoffset = -side*Math.round(labeloffset + (2-side/10) * axis.fTitleOffset*title_fontsize);

            if ((this.name == "zaxis") && is_gaxis && ('getBoundingClientRect' in axis_g.node())) {
               // special handling for color palette labels - draw them always on right side
               var rect = axis_g.node().getBoundingClientRect();
               if (xoffset < rect.width - tickSize) xoffset = Math.round(rect.width - tickSize);
            }

            this.DrawText((center ? "middle" : (myxor ? "begin" : "end" ))+ ";middle",
                           xoffset,
                           Math.round(center ? h/2 : (reverse ? h : 0)),
                           0, (rotate<0 ? -90 : -270),
                           axis.fTitle, title_color, 1, title_g);
         } else {
            this.DrawText((center ? 'middle' : (myxor ? 'begin' : 'end')) + ";middle",
                          Math.round(center ? w/2 : (reverse ? 0 : w)),
                          Math.round(side*(labeloffset + 1.9*title_fontsize*axis.fTitleOffset)),
                          0, (rotate<0 ? -180 : 0),
                          axis.fTitle, title_color, 1, title_g);
         }

         this.FinishTextDrawing(title_g);
     }


     if (JSROOT.gStyle.Zooming) {
        var r =  axis_g.append("svg:rect")
                       .attr("class", "axis_zoom")
                       .style("opacity", "0")
                       .style("cursor", "crosshair");

        if (vertical)
           r.attr("x", (side >0) ? (-2*labelfont.size - 3) : 3)
            .attr("y", 0)
            .attr("width", 2*labelfont.size + 3)
            .attr("height", h)
        else
           r.attr("x", 0).attr("y", 0)
            .attr("width", w).attr("height", labelfont.size + 3);
      }

      this.position = 0;

      if ('getBoundingClientRect' in axis_g.node()) {
         var rect1 = axis_g.node().getBoundingClientRect(),
             rect2 = this.svg_pad().node().getBoundingClientRect();

         this.position = rect1.left - rect2.left; // use to control left position of Y scale
      }
   }

   JSROOT.TAxisPainter.prototype.Redraw = function() {

      var gaxis = this.GetObject(),
          x1 = Math.round(this.AxisToSvg("x", gaxis.fX1)),
          y1 = Math.round(this.AxisToSvg("y", gaxis.fY1)),
          x2 = Math.round(this.AxisToSvg("x", gaxis.fX2)),
          y2 = Math.round(this.AxisToSvg("y", gaxis.fY2)),
          w = x2 - x1, h = y1 - y2;

      var vertical = w<5,
          kind = "normal",
          func = null,
          min = gaxis.fWmin,
          max = gaxis.fWmax,
          reverse = false;

      if (gaxis.fChopt.indexOf("G")>=0) {
         func = d3.scale.log();
         kind = "log";
      } else {
         func = d3.scale.linear();
      }

      func.domain([min, max]);

      if (vertical) {
         if (h > 0) {
            func.range([h,0]);
         } else {
            var d = y1; y1 = y2; y2 = d;
            h = -h; reverse = true;
            func.range([0,h]);
         }
      } else {
         if (w > 0) {
            func.range([0,w]);
         } else {
            var d = x1; x1 = x2; x2 = d;
            w = -w; reverse = true;
            func.range([w,0]);
         }
      }

      this.SetAxisConfig(vertical ? "yaxis" : "xaxis", kind, func, min, max, min, max);

      this.RecreateDrawG(true, "text_layer");

      this.DrawAxis(vertical, this.draw_g, w, h, "translate(" + x1 + "," + y2 +")", reverse);
   }


   JSROOT.drawGaxis = function(divid, obj, opt) {
      var painter = new JSROOT.TAxisPainter(obj, false);

      painter.SetDivId(divid);

      painter.Redraw();

      return painter.DrawingReady();
   }


   // =============================================================

   JSROOT.THistPainter = function(histo) {
      JSROOT.TObjectPainter.call(this, histo);
      this.histo = histo;
      this.shrink_frame_left = 0.;
      this.draw_content = true;
      this.nbinsx = 0;
      this.nbinsy = 0;
      this.x_kind = 'normal'; // 'normal', 'time', 'labels'
      this.y_kind = 'normal'; // 'normal', 'time', 'labels'
   }

   JSROOT.THistPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.THistPainter.prototype.IsDummyHisto = function() {
      return (this.histo==null) || !this.draw_content || (this.options.Axis>0);
   }

   JSROOT.THistPainter.prototype.IsTProfile = function() {
      return this.MatchObjectType('TProfile');
   }

   JSROOT.THistPainter.prototype.IsTH2Poly = function() {
      return this.histo && this.histo._typename.match(/^TH2Poly/);
   }

   JSROOT.THistPainter.prototype.Dimension = function() {
      if (!this.histo) return 0;
      if (this.histo._typename.indexOf("TH2")==0) return 2;
      if (this.histo._typename.indexOf("TH3")==0) return 3;
      return 1;
   }

   JSROOT.THistPainter.prototype.DecodeOptions = function(opt) {

      if ((opt == null) || (opt == "")) opt = this.histo.fOption;

      /* decode string 'opt' and fill the option structure */
      var hdim = this.Dimension();
      var option = {
         Axis: 0, Bar: 0, Curve: 0, Error: 0, Hist: 0, Line: 0,
         Mark: 0, Fill: 0, Same: 0, Scat: 0, Func: 0, Star: 0,
         Arrow: 0, Box: 0, Text: 0, Char: 0, Color: 0, Contour: 0,
         Lego: 0, Surf: 0, Off: 0, Tri: 0, Proj: 0, AxisPos: 0,
         Spec: 0, Pie: 0, List: 0, Zscale: 0, FrontBox: 1, BackBox: 1, Candle: "",
         System: JSROOT.Painter.Coord.kCARTESIAN,
         AutoColor : 0, NoStat : 0, AutoZoom : false,
         HighRes: 0, Zero: 0, Palette: 0, Optimize: JSROOT.gStyle.OptimizeDraw
      };
      // check for graphical cuts
      var chopt = JSROOT.Painter.clearCuts(opt.toUpperCase());

      var pad = this.root_pad();

      // if (hdim > 1) option.Scat = 1;  // default was scatter plot

      // use error plot only when any sumw2 bigger than 0
      if ((hdim===1) && (this.histo.fSumw2.length > 0))
         for (var n=0;n<this.histo.fSumw2.length;++n)
            if (this.histo.fSumw2[n] > 0) { option.Error = 2; break; }

      if (this.histo.fFunctions !== null) option.Func = 1;

      var i = chopt.indexOf('PAL');
      if (i>=0) {
         var i2 = i+3;
         while ((i2<chopt.length) && (chopt.charCodeAt(i2)>=48) && (chopt.charCodeAt(i2)<58)) ++i2;
         if (i2>i+3) {
            option.Palette = parseInt(chopt.substring(i+3,i2));
            chopt = chopt.replace(chopt.substring(i,i2),"");
         }
      }


      var part = ""; // used to return part of string

      function check(name,postpart) {
         var pos = chopt.indexOf(name);
         if (pos < 0) return false;
         chopt = chopt.substr(0, pos) + chopt.substr(pos + name.length);
         if (postpart) {
            part = "";
            var pos2 = pos;
            while ((pos2<chopt.length) && (chopt[pos2] !== ' ') && (chopt[pos2] !== ',') && (chopt[pos2] !== ';')) pos2++;
            if (pos2 > pos) {
               part = chopt.substr(pos, pos2-pos);
               chopt = chopt.substr(0, pos) + chopt.substr(pos2);
            }
         }
         return true;
      }


      if (check('NOOPTIMIZE')) option.Optimize = 0;
      if (check('OPTIMIZE')) option.Optimize = 2;

      if (check('AUTOCOL')) { option.AutoColor = 1; option.Hist = 1; }
      if (check('AUTOZOOM')) { option.AutoZoom = 1; option.Hist = 1; }

      if (check('NOSTAT')) option.NoStat = 1;

      if (check('LOGX')) pad.fLogx = 1;
      if (check('LOGY')) pad.fLogy = 1;
      if (check('LOGZ')) pad.fLogz = 1;
      if (check('GRIDX')) pad.fGridx = 1;
      if (check('GRIDY')) pad.fGridy = 1;
      if (check('TICKX')) pad.fTickx = 1;
      if (check('TICKY')) pad.fTicky = 1;

      var nch = chopt.length;
      if (!nch) option.Hist = 1;

      if (check('SPEC')) {
         option.Scat = 0;
         option.Spec = Math.max(1600, check('BF(') ? parseInt(chopt) : 0);
      }

      check('GL');

      if (check('X+')) option.AxisPos = 10;
      if (check('Y+')) option.AxisPos += 1;

      if ((option.AxisPos == 10 || option.AxisPos == 1) && (nch == 2))
         option.Hist = 1;

      if (option.AxisPos == 11 && (nch == 4))
         option.Hist = 1;

      if (check('SAMES')) {
         if (nch == 5) option.Hist = 1;
         option.Same = 2;
      }

      if (check('SAME')) {
         if (nch == 4) option.Hist = 1;
         option.Same = 1;
      }

      if (check('PIE')) option.Pie = 1;

      if (check('CANDLE', true)) option.Candle = part;

      if (check('LEGO', true)) {
         option.Scat = 0;
         option.Lego = 1;
         if (part.indexOf('0') >= 0) option.Zero = 1;
         if (part.indexOf('1') >= 0) option.Lego = 11;
         if (part.indexOf('2') >= 0) option.Lego = 12;
         if (part.indexOf('3') >= 0) option.Lego = 13;
         if (part.indexOf('4') >= 0) option.Lego = 14;
         if (part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (part.indexOf('BB') >= 0) option.BackBox = 0;
         if (part.indexOf('Z')>=0) option.Zscale = 1;
      }

      if (check('SURF', true)) {
         option.Scat = 0;
         option.Surf = 1;
         if (part.indexOf('FB') >= 0) { option.FrontBox = 0; part = part.replace('FB',''); }
         if (part.indexOf('BB') >= 0) { option.BackBox = 0; part = part.replace('BB',''); }
         if ((part.length>0) && !isNaN(parseInt(part))) option.Surf = 10 + parseInt(part);
      }

      if (check('TF3', true)) {
         if (part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (part.indexOf('BB') >= 0) option.BackBox = 0;
      }

      if (check('ISO', true)) {
         if (part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (part.indexOf('BB') >= 0) option.BackBox = 0;
      }

      if (check('LIST')) option.List = 1;

      if (check('CONT', true)) {
         if (hdim > 1) {
            option.Scat = 0;
            option.Contour = 1;
            if (!isNaN(parseInt(part))) option.Contour = 10 + parseInt(part);
         } else {
            option.Hist = 1;
         }
      }

      // decode bar/hbar option
      if (check('HBAR', true)) option.Bar = 20; else
      if (check('BAR', true)) option.Bar = 10;
      if (option.Bar > 0) {
         option.Hist = 0;
         if (!isNaN(parseInt(part))) option.Bar += parseInt(part);
      }

      if (check('ARR')) {
         if (hdim > 1) {
            option.Arrow = 1;
            option.Scat = 0;
         } else {
            option.Hist = 1;
         }
      }

      if (check('BOX1')) option.Box = 11; else
      if (check('BOX')) option.Box = 1;

      if (option.Box)
         if (hdim > 1) option.Scat = 0;
                  else option.Hist = 1;

      if (check('COL', true)) {
         option.Color = 1;

         if (part[0]=='0') option.Color = 111;
         if (part[0]=='1') option.Color = 1;
         if (part[0]=='2') option.Color = 2;
         if (part[0]=='3') option.Color = 3;

         if (part.indexOf('Z')>=0) option.Zscale = 1;
         if (hdim == 1) option.Hist = 1;
                   else option.Scat = 0;
      }

      if (check('CHAR')) { option.Char = 1; option.Scat = 0; }
      if (check('FUNC')) { option.Func = 2; option.Hist = 0; }
      if (check('HIST')) { option.Hist = 2; option.Func = 0; option.Error = 0; }
      if (check('AXIS')) option.Axis = 1;
      if (check('AXIG')) option.Axis = 2;

      if (check('TEXT', true)) {
         option.Text = 1;
         option.Scat = 0;

         var angle = parseInt(part);
         if (!isNaN(angle)) {
            if (angle < 0) angle = 0;
            if (angle > 90) angle = 90;
            option.Text = 1000 + angle;
         }
         if (part.indexOf('N')>=0 && this.IsTH2Poly())
            option.Text += 3000;
      }

      if (check('SCAT')) option.Scat = 1;
      if (check('POL')) option.System = JSROOT.Painter.Coord.kPOLAR;
      if (check('CYL')) option.System = JSROOT.Painter.Coord.kCYLINDRICAL;
      if (check('SPH')) option.System = JSROOT.Painter.Coord.kSPHERICAL;
      if (check('PSR')) option.System = JSROOT.Painter.Coord.kRAPIDITY;

      if (check('TRI', true)) {
         option.Scat = 0;
         option.Color = 0;
         option.Tri = 1;
         if (part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (part.indexOf('BB') >= 0) option.BackBox = 0;
         if (part.indexOf('ERR') >= 0) option.Error = 1;
      }
      if (check('AITOFF')) option.Proj = 1;
      if (check('MERCATOR')) option.Proj = 2;
      if (check('SINUSOIDAL')) option.Proj = 3;
      if (check('PARABOLIC')) option.Proj = 4;

      if (option.Proj > 0) { option.Scat = 0; option.Contour = 14; }

      if ((hdim==3) && check('FB')) option.FrontBox = 0;
      if ((hdim==3) && check('BB')) option.BackBox = 0;

      if (check('A')) option.Axis = -1;
      if (check('B')) { option.Bar = 1; option.Hist = -1; }
      if (check('C')) { option.Curve = 1; option.Hist = -1; }
      if (check('F')) option.Fill = 1;
      if (check('][')) { option.Off = 1; option.Hist = 1; }
      if (check('F2')) option.Fill = 2;
      if (check('L')) { option.Line = 1; option.Hist = -1; }

      if (check('P0')) { option.Mark = 10; option.Hist = -1; }
      if (check('P')) { option.Mark = 1; option.Hist = -1; }
      if (check('Z')) option.Zscale = 1;
      if (check('*')) option.Star = 1;
      if (check('H')) option.Hist = 2;

      if (this.IsTH2Poly()) {
         if (option.Fill + option.Line + option.Mark != 0) option.Scat = 0;
      }

      if (check('E', true)) {
         if (hdim == 1) {
            option.Error = 1;
            if ((part.length>0) && !isNaN(parseInt(part[0])))
               option.Error = 10 + parseInt(part[0]);
            if (part.indexOf('X0')>=0) {
               if (option.Error == 1) option.Error += 20;
               option.Error += 10;
            }
            if (option.Text && this.IsTProfile()) {
               option.Text += 2000;
               option.Error = 0;
            }
         } else {
            if (option.Error == 0) {
               option.Error = 100;
               option.Scat = 0;
            }
            if (option.Text) {
               option.Text += 2000;
               option.Error = 0;
            }
         }
      }
      if (check('9')) option.HighRes = 1;

      //if (option.Surf == 15)
      //   if (option.System == JSROOT.Painter.Coord.kPOLAR || option.System == JSROOT.Painter.Coord.kCARTESIAN)
      //      option.Surf = 13;

      return option;
   }

   JSROOT.THistPainter.prototype.GetAutoColor = function(col) {
      if (this.options.AutoColor<=0) return col;

      var id = this.options.AutoColor;
      this.options.AutoColor = id % 8 + 1;
      return JSROOT.Painter.root_colors[id];
   }

   JSROOT.THistPainter.prototype.ScanContent = function() {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values

      alert("HistPainter.prototype.ScanContent not implemented");
   }

   JSROOT.THistPainter.prototype.CheckPadRange = function() {

      if (!this.is_main_painter()) return;

      this.zoom_xmin = this.zoom_xmax = 0;
      this.zoom_ymin = this.zoom_ymax = 0;
      this.zoom_zmin = this.zoom_zmax = 0;

      var pad = this.root_pad();

      if (!pad || !('fUxmin' in pad) || this.create_canvas) return;

      var min = pad.fUxmin, max = pad.fUxmax, xaxis = this.histo.fXaxis;

      // first check that non-default values are there
      if ((this.Dimension() < 3) && ((min !== 0) || (max !== 1))) {
         if (pad.fLogx > 0) {
            min = Math.exp(min * Math.log(10));
            max = Math.exp(max * Math.log(10));
         }

         if (min !== xaxis.fXmin || max !== xaxis.fXmax)
            if (min >= xaxis.fXmin && max <= xaxis.fXmax) {
               // set zoom values if only inside range
               this.zoom_xmin = min;
               this.zoom_xmax = max;
            }
      }

      // apply selected user range if range not selected via pad
      if (xaxis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
         xaxis.InvertBit(JSROOT.EAxisBits.kAxisRange); // axis range is not used for main painter
         if ((this.zoom_xmin === this.zoom_xmax) && (xaxis.fFirst !== xaxis.fLast) &&
             ((xaxis.fFirst > 1) || (xaxis.fLast < xaxis.fNbins))) {
               this.zoom_xmin = xaxis.fFirst > 1 ? xaxis.GetBinLowEdge(xaxis.fFirst-1) : xaxis.fXmin;
               this.zoom_xmax = xaxis.fLast < xaxis.fNbins ? xaxis.GetBinLowEdge(xaxis.fLast) : xaxis.fXmax;
         }
      }

      min = pad.fUymin; max = pad.fUymax;

      if ((this.Dimension() == 2) && ((min !== 0) || (max !== 1))) {
         if (pad.fLogy > 0) {
            min = Math.exp(min * Math.log(10));
            max = Math.exp(max * Math.log(10));
         }

         if (min !== this.histo.fYaxis.fXmin || max !== this.histo.fYaxis.fXmax)
            if (min >= this.histo.fYaxis.fXmin && max <= this.histo.fYaxis.fXmax) {
               // set zoom values if only inside range
               this.zoom_ymin = min;
               this.zoom_ymax = max;
            }
      }
   }

   JSROOT.THistPainter.prototype.CheckHistDrawAttributes = function() {

      if (!this.fillatt || !this.fillatt.changed)
         this.fillatt = this.createAttFill(this.histo, undefined, undefined, 1);

      if (!this.lineatt || !this.lineatt.changed) {
         this.lineatt = JSROOT.Painter.createAttLine(this.histo);
         var main = this.main_painter();

         if (main) {
            var newcol = main.GetAutoColor(this.lineatt.color);
            if (newcol !== this.lineatt.color) { this.lineatt.color = newcol; this.lineatt.changed = true; }
         }
      }
   }

   JSROOT.THistPainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) {
         alert("JSROOT.THistPainter.UpdateObject - wrong class " + obj._typename + " expected " + this.histo._typename);
         return false;
      }

      // TODO: simple replace of object does not help - one can have different
      // complex relations between histo and stat box, histo and colz axis,
      // one could have THStack or TMultiGraph object
      // The only that could be done is update of content

      // this.histo = obj;

      var histo = this.GetObject();

      histo.fFillColor = obj.fFillColor;
      histo.fFillStyle = obj.fFillStyle;
      histo.fLineColor = obj.fLineColor;
      histo.fLineStyle = obj.fLineStyle;
      histo.fLineWidth = obj.fLineWidth;

      histo.fEntries = obj.fEntries;
      histo.fTsumw = obj.fTsumw;
      histo.fTsumwx = obj.fTsumwx;
      histo.fTsumwx2 = obj.fTsumwx2;
      if (this.Dimension() == 2) {
         histo.fTsumwy = obj.fTsumwy;
         histo.fTsumwy2 = obj.fTsumwy2;
         histo.fTsumwxy = obj.fTsumwxy;
      }
      histo.fArray = obj.fArray;
      histo.fNcells = obj.fNcells;
      histo.fTitle = obj.fTitle;
      histo.fMinimum = obj.fMinimum;
      histo.fMaximum = obj.fMaximum;
      histo.fXaxis.fNbins = obj.fXaxis.fNbins;
      if (!this.main_painter().zoom_changed_interactive) {
         histo.fXaxis.fXmin = obj.fXaxis.fXmin;
         histo.fXaxis.fXmax = obj.fXaxis.fXmax;
         histo.fXaxis.fFirst = obj.fXaxis.fFirst;
         histo.fXaxis.fLast = obj.fXaxis.fLast;
         histo.fXaxis.fBits = obj.fXaxis.fBits;
         histo.fYaxis.fXmin = obj.fYaxis.fXmin;
         histo.fYaxis.fXmax = obj.fYaxis.fXmax;
         histo.fYaxis.fFirst = obj.fYaxis.fFirst;
         histo.fYaxis.fLast = obj.fYaxis.fLast;
         histo.fYaxis.fBits = obj.fYaxis.fBits;
      }
      histo.fSumw2 = obj.fSumw2;

      if (this.IsTProfile()) {
         histo.fBinEntries = obj.fBinEntries;
      }

      if (obj.fFunctions)
         for (var n=0;n<obj.fFunctions.arr.length;++n) {
            var func = obj.fFunctions.arr[n];
            if ((func._typename == "TPaveStats") && (func.fName == 'stats')) {
               var funcpainter = this.FindPainterFor(null,'stats');
               if (funcpainter) funcpainter.UpdateObject(func);
            }
         }

      if (!this.zoom_changed_interactive) this.CheckPadRange();

      this.ScanContent();

      return true;
   }

   JSROOT.THistPainter.prototype.CreateAxisFuncs = function(with_y_axis, with_z_axis) {
      // here functions are defined to convert index to axis value and back
      // introduced to support non-equidistant bins

      this.xmin = this.histo.fXaxis.fXmin;
      this.xmax = this.histo.fXaxis.fXmax;

      if (this.histo.fXaxis.fXbins.length == this.nbinsx+1) {
         this.regularx = false;
         this.GetBinX = function(bin) {
            var indx = Math.round(bin);
            if (indx <= 0) return this.xmin;
            if (indx > this.nbinsx) this.xmax;
            if (indx==bin) return this.histo.fXaxis.fXbins[indx];
            var indx2 = (bin < indx) ? indx - 1 : indx + 1;
            return this.histo.fXaxis.fXbins[indx] * Math.abs(bin-indx2) + this.histo.fXaxis.fXbins[indx2] * Math.abs(bin-indx);
         };
         this.GetIndexX = function(x,add) {
            for (var k = 1; k < this.histo.fXaxis.fXbins.length; ++k)
               if (x < this.histo.fXaxis.fXbins[k]) return Math.floor(k-1+add);
            return this.nbinsx;
         };
      } else {
         this.regularx = true;
         this.binwidthx = (this.xmax - this.xmin);
         if (this.nbinsx > 0)
            this.binwidthx = this.binwidthx / this.nbinsx;

         this.GetBinX = function(bin) { return this.xmin + bin*this.binwidthx; };
         this.GetIndexX = function(x,add) { return Math.floor((x - this.xmin) / this.binwidthx + add); };
      }

      this.ymin = this.histo.fYaxis.fXmin;
      this.ymax = this.histo.fYaxis.fXmax;

      if (!with_y_axis || (this.nbinsy==0)) return;

      if (this.histo.fYaxis.fXbins.length == this.nbinsy+1) {
         this.regulary = false;
         this.GetBinY = function(bin) {
            var indx = Math.round(bin);
            if (indx <= 0) return this.ymin;
            if (indx > this.nbinsy) this.ymax;
            if (indx==bin) return this.histo.fYaxis.fXbins[indx];
            var indx2 = (bin < indx) ? indx - 1 : indx + 1;
            return this.histo.fYaxis.fXbins[indx] * Math.abs(bin-indx2) + this.histo.fYaxis.fXbins[indx2] * Math.abs(bin-indx);
         };
         this.GetIndexY = function(y,add) {
            for (var k = 1; k < this.histo.fYaxis.fXbins.length; ++k)
               if (y < this.histo.fYaxis.fXbins[k]) return Math.floor(k-1+add);
            return this.nbinsy;
         };
      } else {
         this.regulary = true;
         this.binwidthy = (this.ymax - this.ymin);
         if (this.nbinsy > 0)
            this.binwidthy = this.binwidthy / this.nbinsy;

         this.GetBinY = function(bin) { return this.ymin+bin*this.binwidthy; };
         this.GetIndexY = function(y,add) { return Math.floor((y - this.ymin) / this.binwidthy + add); };
      }

      if (!with_z_axis || (this.nbinsz==0)) return;

      if (this.histo.fZaxis.fXbins.length == this.nbinsz+1) {
         this.regularz = false;
         this.GetBinZ = function(bin) {
            var indx = Math.round(bin);
            if (indx <= 0) return this.zmin;
            if (indx > this.nbinsz) this.zmax;
            if (indx==bin) return this.histo.fZaxis.fXbins[indx];
            var indx2 = (bin < indx) ? indx - 1 : indx + 1;
            return this.histo.fZaxis.fXbins[indx] * Math.abs(bin-indx2) + this.histo.fZaxis.fXbins[indx2] * Math.abs(bin-indx);
         };
         this.GetIndexZ = function(z,add) {
            for (var k = 1; k < this.histo.fZaxis.fXbins.length; ++k)
               if (z < this.histo.fZaxis.fXbins[k]) return Math.floor(k-1+add);
            return this.nbinsz;
         };
      } else {
         this.regularz = true;
         this.binwidthz = (this.zmax - this.zmin);
         if (this.nbinsz > 0)
            this.binwidthz = this.binwidthz / this.nbinsz;

         this.GetBinZ = function(bin) { return this.zmin+bin*this.binwidthz; };
         this.GetIndexZ = function(z,add) { return Math.floor((z - this.zmin) / this.binwidthz + add); };
      }
   }


   JSROOT.THistPainter.prototype.CreateXY = function() {
      // here we create x,y objects which maps our physical coordnates into pixels
      // while only first painter really need such object, all others just reuse it
      // following functions are introduced
      //    this.GetBin[X/Y]  return bin coordinate
      //    this.Convert[X/Y]  converts root value in JS date when date scale is used
      //    this.[x,y]  these are d3.scale objects
      //    this.gr[x,y]  converts root scale into graphical value
      //    this.Revert[X/Y]  converts graphical coordinates to root scale value

      if (!this.is_main_painter()) {
         this.x = this.main_painter().x;
         this.y = this.main_painter().y;
         return;
      }

      this.swap_xy = false;
      if (this.options.Bar>=20) this.swap_xy = true;
      this.logx = this.logy = false;

      var w = this.frame_width(), h = this.frame_height(), pad = this.root_pad();

      if (this.histo.fXaxis.fTimeDisplay) {
         this.x_kind = 'time';
         this.timeoffsetx = JSROOT.Painter.getTimeOffset(this.histo.fXaxis);
         this.ConvertX = function(x) { return new Date(this.timeoffsetx + x*1000); };
         this.RevertX = function(grx) { return (this.x.invert(grx) - this.timeoffsetx) / 1000; };
      } else {
         this.x_kind = (this.histo.fXaxis.fLabels==null) ? 'normal' : 'labels';
         this.ConvertX = function(x) { return x; };
         this.RevertX = function(grx) { return this.x.invert(grx); };
      }

      this.scale_xmin = this.xmin;
      this.scale_xmax = this.xmax;
      if (this.zoom_xmin != this.zoom_xmax) {
         this.scale_xmin = this.zoom_xmin;
         this.scale_xmax = this.zoom_xmax;
      }
      if (this.x_kind == 'time') {
         this.x = d3.time.scale();
      } else
      if (this.swap_xy ? pad.fLogy : pad.fLogx) {
         this.logx = true;

         if (this.scale_xmax <= 0) this.scale_xmax = 0;

         if ((this.scale_xmin <= 0) && (this.nbinsx>0))
            for (var i=0;i<this.nbinsx;++i) {
               this.scale_xmin = Math.max(this.scale_xmin, this.GetBinX(i));
               if (this.scale_xmin>0) break;
            }

         if ((this.scale_xmin <= 0) || (this.scale_xmin >= this.scale_xmax)) {
            this.scale_xmin = this.scale_xmax * 0.0001;
         }

         this.x = d3.scale.log();
      } else {
         this.x = d3.scale.linear();
      }

      this.x.domain([this.ConvertX(this.scale_xmin), this.ConvertX(this.scale_xmax)])
            .range(this.swap_xy ? [ h, 0 ] : [ 0, w ]);

      if (this.x_kind == 'time') {
         // we emulate scale functionality
         this.grx = function(val) { return this.x(this.ConvertX(val)); }
      } else
      if (this.logx) {
         this.grx = function(val) { return (val < this.scale_xmin) ? (this.swap_xy ? this.x.range()[0]+5 : -5) : this.x(val); }
      } else {
         this.grx = this.x;
      }

      this.scale_ymin = this.ymin;
      this.scale_ymax = this.ymax;
      if (this.zoom_ymin != this.zoom_ymax) {
         this.scale_ymin = this.zoom_ymin;
         this.scale_ymax = this.zoom_ymax;
      }

      if (this.histo.fYaxis.fTimeDisplay) {
         this.y_kind = 'time';
         this.timeoffsety = JSROOT.Painter.getTimeOffset(this.histo.fYaxis);
         this.ConvertY = function(y) { return new Date(this.timeoffsety + y*1000); };
         this.RevertY = function(gry) { return (this.y.invert(gry) - this.timeoffsety) / 1000; };
      } else {
         this.y_kind = ((this.Dimension()==2) && (this.histo.fYaxis.fLabels!=null)) ? 'labels' : 'normal';
         this.ConvertY = function(y) { return y; };
         this.RevertY = function(gry) { return this.y.invert(gry); };
      }

      if (this.swap_xy ? pad.fLogx : pad.fLogy) {
         this.logy = true;
         if (this.scale_ymax <= 0)
            this.scale_ymax = 1;
         else
         if ((this.zoom_ymin === this.zoom_ymax) && (this.Dimension()==1))
            this.scale_ymax*=1.8;

         if ((this.scale_ymin <= 0) && (this.nbinsy>0))
            for (var i=0;i<this.nbinsy;++i) {
               this.scale_ymin = Math.max(this.scale_ymin, this.GetBinY(i));
               if (this.scale_ymin>0) break;
            }

         if ((this.scale_ymin <= 0) && ('ymin_nz' in this) && (this.ymin_nz > 0))
            this.scale_ymin = 0.3*this.ymin_nz;

         if ((this.scale_ymin <= 0) || (this.scale_ymin >= this.scale_ymax))
            this.scale_ymin = 0.000001 * this.scale_ymax;
         this.y = d3.scale.log();
      } else
      if (this.y_kind=='time') {
         this.y = d3.time.scale();
      } else {
         this.y = d3.scale.linear()
      }

      this.y.domain([ this.ConvertY(this.scale_ymin), this.ConvertY(this.scale_ymax) ])
            .range(this.swap_xy ? [ 0, w ] : [ h, 0 ]);

      if (this.y_kind=='time') {
         // we emulate scale functionality
         this.gry = function(val) { return this.y(this.ConvertY(val)); }
      } else
      if (this.logy) {
         // make protecttion for log
         this.gry = function(val) { return (val < this.scale_ymin) ? (this.swap_xy ? -5 : this.y.range()[0]+5) : this.y(val); }
      } else {
         this.gry = this.y;
      }
   }

   JSROOT.THistPainter.prototype.DrawGrids = function() {
      // grid can only be drawn by first painter
      if (!this.is_main_painter()) return;

      var layer = this.svg_frame().select(".grid_layer");

      layer.selectAll(".xgrid").remove();
      layer.selectAll(".ygrid").remove();

      var pad = this.root_pad(), h = this.frame_height(), w = this.frame_width(), grid;

      // add a grid on x axis, if the option is set
      if (pad && pad.fGridx && this.x_handle) {
         grid = "";
         for (var n=0;n<this.x_handle.ticks.length;++n)
            if (this.swap_xy)
               grid += "M0,"+this.x_handle.ticks[n]+"h"+w;
            else
               grid += "M"+this.x_handle.ticks[n]+",0v"+h;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "xgrid")
               .attr("d", grid)
               .style("stroke", "black")
               .style("stroke-width", 1)
               .style("stroke-dasharray", JSROOT.Painter.root_line_styles[11]);
      }

      // add a grid on y axis, if the option is set
      if (pad && pad.fGridy && this.y_handle) {
         grid = "";
         for (var n=0;n<this.y_handle.ticks.length;++n)
            if (this.swap_xy)
               grid += "M"+this.y_handle.ticks[n]+",0v"+h;
            else
               grid += "M0,"+this.y_handle.ticks[n]+"h"+w;

         if (grid.length > 0)
          layer.append("svg:path")
               .attr("class", "ygrid")
               .attr("d", grid)
               .style("stroke", "black")
               .style("stroke-width", 1)
               .style("stroke-dasharray", JSROOT.Painter.root_line_styles[11]);
      }
   }

   JSROOT.THistPainter.prototype.DrawBins = function() {
      alert("HistPainter.DrawBins not implemented");
   }

   JSROOT.THistPainter.prototype.AxisAsText = function(axis, value) {
      if (axis == "x") {
         if (this.x_kind == 'time')
            value = this.ConvertX(value);

         if (this.x_handle && ('format' in this.x_handle))
            return this.x_handle.format(value);

         return value.toPrecision(4);
      }

      if (axis == "y") {
         if (this.y_kind == 'time')
            value = this.ConvertY(value);

         if (this.y_handle && ('format' in this.y_handle))
            return this.y_handle.format(value);

         return value.toPrecision(4);
      }

      return value.toPrecision(4);
   }

   JSROOT.THistPainter.prototype.DrawAxes = function(shrink_forbidden) {
      // axes can be drawn only for main histogram

      if (!this.is_main_painter()) return;

      var layer = this.svg_frame().select(".axis_layer"),
          w = this.frame_width(),
          h = this.frame_height(),
          pad = this.root_pad();

      this.x_handle = new JSROOT.TAxisPainter(this.histo.fXaxis, true);
      this.x_handle.SetDivId(this.divid, -1);

      this.x_handle.SetAxisConfig("xaxis",
                                  (this.logx && (this.x_kind !== "time")) ? "log" : this.x_kind,
                                  this.x, this.xmin, this.xmax, this.scale_xmin, this.scale_xmax);

      this.y_handle = new JSROOT.TAxisPainter(this.histo.fYaxis, true);
      this.y_handle.SetDivId(this.divid, -1);

      this.y_handle.SetAxisConfig("yaxis",
                                  (this.logy && this.y_kind !== "time") ? "log" : this.y_kind,
                                  this.y, this.ymin, this.ymax, this.scale_ymin, this.scale_ymax);

      var draw_horiz = this.swap_xy ? this.y_handle : this.x_handle;
      var draw_vertical = this.swap_xy ? this.x_handle : this.y_handle;

      draw_horiz.DrawAxis(false, layer, w, h, "translate(0," + h + ")", false, pad.fTickx ? -h : 0);
      draw_vertical.DrawAxis(true, layer, w, h, undefined, false, pad.fTicky ? w : 0);

      if (shrink_forbidden) return;

      var shrink = 0., ypos = draw_vertical.position;

      if ((-0.2*w < ypos) && (ypos < 0)) {
         shrink = -ypos/w + 0.001;
         this.shrink_frame_left += shrink;
      } else
      if ((ypos>0) && (ypos<0.3*w) && (this.shrink_frame_left > 0) && (ypos/w > this.shrink_frame_left)) {
         shrink = -this.shrink_frame_left;
         this.shrink_frame_left = 0.;
      }

      if (shrink != 0) {
         this.frame_painter().Shrink(shrink, 0);
         this.frame_painter().Redraw();
         this.CreateXY();
         this.DrawAxes(true);
      }
   }

   JSROOT.THistPainter.prototype.DrawTitle = function() {

      // case when histogram drawn over other histogram (same option)
      if (!this.is_main_painter()) return;

      var tpainter = this.FindPainterFor(null, "title");
      var pavetext = (tpainter !== null) ? tpainter.GetObject() : null;
      if (pavetext === null) pavetext = this.FindInPrimitives("title");
      if ((pavetext !== null) && (pavetext._typename !== "TPaveText")) pavetext = null;

      var draw_title = !this.histo.TestBit(JSROOT.TH1StatusBits.kNoTitle);

      if (pavetext !== null) {
         pavetext.Clear();
         if (draw_title)
            pavetext.AddText(this.histo.fTitle);
      } else
      if (draw_title && !tpainter && (this.histo.fTitle.length > 0)) {
         pavetext = JSROOT.Create("TPaveText");

         JSROOT.extend(pavetext, { fName: "title", fX1NDC: 0.28, fY1NDC: 0.94, fX2NDC: 0.72, fY2NDC: 0.99 } );
         pavetext.AddText(this.histo.fTitle);

         JSROOT.Painter.drawPaveText(this.divid, pavetext);
      }
   }

   JSROOT.THistPainter.prototype.ToggleStat = function(arg) {

      var stat = this.FindStat(), statpainter = null;

      if (stat == null) {
         if (arg=='only-check') return false;
         // when statbox created first time, one need to draw it
         stat = this.CreateStat();
      } else {
         statpainter = this.FindPainterFor(stat);
      }

      if (arg=='only-check') return statpainter ? statpainter.Enabled : false;

      if (statpainter) {
         statpainter.Enabled = !statpainter.Enabled;
         // when stat box is drawed, it always can be draw individualy while it
         // should be last for colz RedrawPad is used
         statpainter.Redraw();
         return statpainter.Enabled;
      }

      JSROOT.draw(this.divid, stat, "onpad:" + this.pad_name);

      return true;
   }

   JSROOT.THistPainter.prototype.IsAxisZoomed = function(axis) {
      var obj = this.main_painter();
      if (obj == null) obj = this;
      if (axis === "x") return obj.zoom_xmin != obj.zoom_xmax;
      if (axis === "y") return obj.zoom_ymin != obj.zoom_ymax;
      return false;
   }

   JSROOT.THistPainter.prototype.GetSelectIndex = function(axis, size, add) {
      // be aware - here indexs starts from 0
      var indx = 0, obj = this.main_painter();
      if (obj == null) obj = this;
      var nbin = this['nbins'+axis];
      if (!nbin) nbin = 0;
      if (!add) add = 0;

      var func = 'GetIndex' + axis.toUpperCase(),
          min = obj['zoom_' + axis + 'min'],
          max = obj['zoom_' + axis + 'max'];

      if ((min != max) && (func in this)) {
         if (size == "left") {
            indx = this[func](min, add);
         } else {
            indx = this[func](max, add + 0.5);
         }
      } else {
         indx = (size == "left") ? 0 : nbin;
      }

      var taxis; // TAxis object of histogram, where user range can be stored
      if (this.histo) taxis  = this.histo["f" + axis.toUpperCase() + "axis"];
      if (taxis) {
         if ((taxis.fFirst === taxis.fLast) || !taxis.TestBit(JSROOT.EAxisBits.kAxisRange) ||
             ((taxis.fFirst<=1) && (taxis.fLast>=nbin))) taxis = undefined;
      }

      if (size == "left") {
         if (indx < 0) indx = 0;
         if (taxis && (taxis.fFirst>1) && (indx<taxis.fFirst)) indx = taxis.fFirst-1;
      } else {
         if (indx > nbin) indx = nbin;
         if (taxis && (taxis.fLast <= nbin) && (indx>taxis.fLast)) indx = taxis.fLast;
      }

      return indx;
   }

   JSROOT.THistPainter.prototype.FindStat = function() {
      if (this.histo.fFunctions !== null)
         for (var i = 0; i < this.histo.fFunctions.arr.length; ++i) {
            var func = this.histo.fFunctions.arr[i];

            if ((func._typename == 'TPaveStats') &&
                (func.fName == 'stats')) return func;
         }

      return null;
   }

   JSROOT.THistPainter.prototype.CreateStat = function() {

      if (!this.draw_content) return null;

      var stats = this.FindStat();
      if (stats != null) return stats;

      stats = JSROOT.Create('TPaveStats');
      JSROOT.extend(stats, { fName : 'stats',
                             fOptStat: JSROOT.gStyle.OptStat,
                             fOptFit: JSROOT.gStyle.OptFit,
                             fBorderSize : 1} );
      JSROOT.extend(stats, JSROOT.gStyle.StatNDC);
      JSROOT.extend(stats, JSROOT.gStyle.StatText);
      JSROOT.extend(stats, JSROOT.gStyle.StatFill);

      if (this.histo._typename.match(/^TProfile/) || this.histo._typename.match(/^TH2/))
         stats.fY1NDC = 0.67;

      stats.AddText(this.histo.fName);

      if (this.histo.fFunctions === null)
         this.histo.fFunctions = JSROOT.Create("TList");

      this.histo.fFunctions.Add(stats,"");

      return stats;
   }

   JSROOT.THistPainter.prototype.AddFunction = function(obj, asfirst) {
      var histo = this.GetObject();
      if (!histo || !obj) return;

      if (histo.fFunctions == null)
         histo.fFunctions = JSROOT.Create("TList");

      if (asfirst)
         histo.fFunctions.AddFirst(obj);
      else
         histo.fFunctions.Add(obj);

   }

   JSROOT.THistPainter.prototype.FindFunction = function(type_name) {
      var funcs = this.GetObject().fFunctions;
      if (funcs === null) return null;

      for (var i = 0; i < funcs.arr.length; ++i)
         if (funcs.arr[i]._typename === type_name) return funcs.arr[i];

      return null;
   }

   JSROOT.THistPainter.prototype.DrawNextFunction = function(indx, callback) {
      // method draws next function from the functions list

      if (this.options.Same || (this.histo.fFunctions === null) || (indx >= this.histo.fFunctions.arr.length))
         return JSROOT.CallBack(callback);

      var func = this.histo.fFunctions.arr[indx],
          opt = this.histo.fFunctions.opt[indx],
          do_draw = false,
          func_painter = this.FindPainterFor(func);

      // no need to do something if painter for object was already done
      // object will be redraw automatically
      if (func_painter === null) {
         if (func._typename === 'TPaveText' || func._typename === 'TPaveStats') {
            do_draw = !this.histo.TestBit(JSROOT.TH1StatusBits.kNoStats) && (this.options.NoStat!=1);
         } else
         if (func._typename === 'TF1') {
            do_draw = !func.TestBit(JSROOT.BIT(9));
         } else
            do_draw = (func._typename !== "TPaletteAxis");
      }
      //if (('CompleteDraw' in func_painter) && (typeof func_painter.CompleteDraw == 'function'))
      //   func_painter.CompleteDraw();

      if (do_draw) {
         var painter = JSROOT.draw(this.divid, func, opt);
         if (painter) return painter.WhenReady(this.DrawNextFunction.bind(this, indx+1, callback));
      }

      this.DrawNextFunction(indx+1, callback);
   }

   JSROOT.THistPainter.prototype.UnzoomUserRange = function(dox, doy, doz) {

      if (!this.histo) return false;

      function UnzoomTAxis(obj) {
         if (!obj) return false;
         if (!obj.TestBit(JSROOT.EAxisBits.kAxisRange)) return false;
         if (obj.fFirst === obj.fLast) return false;
         if ((obj.fFirst <= 1) && (obj.fLast >= obj.fNbins)) return false;
         obj.InvertBit(JSROOT.EAxisBits.kAxisRange);
         return true;
      }

      var res = false;

      if (dox) res |= UnzoomTAxis(this.histo.fXaxis);
      if (doy) res |= UnzoomTAxis(this.histo.fYaxis);
      if (doz) res |= UnzoomTAxis(this.histo.fZaxis);

      return res;
   }

   JSROOT.THistPainter.prototype.ToggleLog = function(axis) {
      var obj = this.main_painter(), pad = this.root_pad();
      if (!obj) obj = this;
      var curr = pad["fLog" + axis];
      // do not allow log scale for labels
      if (!curr && (this[axis+"_kind"] == "labels")) return;
      pad["fLog" + axis] = curr ? 0 : 1;
      obj.RedrawPad();
   }

   JSROOT.THistPainter.prototype.Zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {
      // function can be used for zooming into specified range
      // if both ranges for axis 0 (like xmin==xmax==0), axis will be unzoomed

      var main = this.main_painter(),
          zoom_x = (xmin !== xmax), zoom_y = (ymin !== ymax), zoom_z = (zmin !== zmax),
          unzoom_x = false, unzoom_y = false, unzoom_z = false;

      if (zoom_x) {
         var cnt = 0;
         if (xmin <= main.xmin) { xmin = main.xmin; cnt++; }
         if (xmax >= main.xmax) { xmax = main.xmax; cnt++; }
         if (cnt === 2) { zoom_x = false; unzoom_x = true; }
      } else {
         unzoom_x = (xmin === xmax) && (xmin === 0);
      }

      if (zoom_y) {
         var cnt = 0;
         if (ymin <= main.ymin) { ymin = main.ymin; cnt++; }
         if (ymax >= main.ymax) { ymax = main.ymax; cnt++; }
         if (cnt === 2) { zoom_y = false; unzoom_y = true; }
      } else {
         unzoom_y = (ymin === ymax) && (ymin === 0);
      }

      if (zoom_z) {
         var cnt = 0;
         if (zmin <= main.zmin) { zmin = main.zmin; cnt++; }
         if (zmax >= main.zmax) { zmax = main.zmax; cnt++; }
         if (cnt === 2) { zoom_z = false; unzoom_z = true; }
      } else {
         unzoom_z = (zmin === zmax) && (zmin === 0);
      }

      var changed = false;

      // first process zooming (if any)
      if (zoom_x || zoom_y || zoom_z)
         main.ForEachPainter(function(obj) {
            if (zoom_x && obj.CanZoomIn("x", xmin, xmax)) {
               main.zoom_xmin = xmin;
               main.zoom_xmax = xmax;
               changed = true;
               zoom_x = false;
            }
            if (zoom_y && obj.CanZoomIn("y", ymin, ymax)) {
               main.zoom_ymin = ymin;
               main.zoom_ymax = ymax;
               changed = true;
               zoom_y = false;
            }
            if (zoom_z && obj.CanZoomIn("z", zmin, zmax)) {
               main.zoom_zmin = zmin;
               main.zoom_zmax = zmax;
               changed = true;
               zoom_z = false;
            }
         });

      // and process unzoom, if any
      if (unzoom_x || unzoom_y || unzoom_z) {
         if (unzoom_x) {
            if (main.zoom_xmin !== main.zoom_xmax) changed = true;
            main.zoom_xmin = main.zoom_xmax = 0;
         }
         if (unzoom_y) {
            if (main.zoom_ymin !== main.zoom_ymax) changed = true;
            main.zoom_ymin = main.zoom_ymax = 0;
         }
         if (unzoom_z) {
            if (main.zoom_zmin !== main.zoom_zmax) changed = true;
            main.zoom_zmin = main.zoom_zmax = 0;
         }

         // first try to unzoom main painter - it could have user range specified
         if (!changed) {
            changed = main.UnzoomUserRange(unzoom_x, unzoom_y, unzoom_z);

            // than try to unzoom all overlapped objects
            var pp = this.pad_painter(true);
            if (pp && pp.painters)
            pp.painters.forEach(function(paint){
               if (paint && (paint!==main) && (typeof paint.UnzoomUserRange == 'function'))
                  if (paint.UnzoomUserRange(unzoom_x, unzoom_y, unzoom_z)) changed = true;
            });
         }
      }

      if (changed) this.RedrawPad();

      return changed;
   }

   JSROOT.THistPainter.prototype.Unzoom = function(dox, doy, doz) {
      if (typeof dox === 'undefined') { dox = true; doy = true; doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z")>=0; doy = dox.indexOf("y")>=0; dox = dox.indexOf("x")>=0; }

      if (dox || doy || dox) this.zoom_changed_interactive = true;

      return this.Zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                       doy ? 0 : undefined, doy ? 0 : undefined,
                       doz ? 0 : undefined, doz ? 0 : undefined);
   }

   JSROOT.THistPainter.prototype.clearInteractiveElements = function() {
      JSROOT.Painter.closeMenu();
      if (this.zoom_rect != null) { this.zoom_rect.remove(); this.zoom_rect = null; }
      this.zoom_kind = 0;

      // enable tooltip in frame painter
      this.SwitchTooltip(true);
   }

   JSROOT.THistPainter.prototype.mouseDoubleClick = function() {
      d3.event.preventDefault();
      var m = d3.mouse(this.svg_frame().node());
      this.clearInteractiveElements();
      var kind = "xyz";
      if (m[0] < 0) kind = "y"; else
      if (m[1] > this.frame_height()) kind = "x";
      this.Unzoom(kind);
   }

   JSROOT.THistPainter.prototype.startRectSel = function() {
      // ignore when touch selection is actiavated

      if (this.zoom_kind > 100) return;

      // ignore all events from non-left button
      if ((d3.event.which || d3.event.button) !== 1) return;

      d3.event.preventDefault();

      this.clearInteractiveElements();
      this.zoom_origin = d3.mouse(this.svg_frame().node());

      this.zoom_curr = [ Math.max(0, Math.min(this.frame_width(), this.zoom_origin[0])),
                         Math.max(0, Math.min(this.frame_height(), this.zoom_origin[1])) ];

      if (this.zoom_origin[0] < 0) {
         this.zoom_kind = 3; // only y
         this.zoom_origin[0] = 0;
         this.zoom_origin[1] = this.zoom_curr[1];
         this.zoom_curr[0] = this.frame_width();
         this.zoom_curr[1] += 1;
      } else if (this.zoom_origin[1] > this.frame_height()) {
         this.zoom_kind = 2; // only x
         this.zoom_origin[0] = this.zoom_curr[0];
         this.zoom_origin[1] = 0;
         this.zoom_curr[0] += 1;
         this.zoom_curr[1] = this.frame_height();
      } else {
         this.zoom_kind = 1; // x and y
         this.zoom_origin[0] = this.zoom_curr[0];
         this.zoom_origin[1] = this.zoom_curr[1];
      }

      d3.select(window).on("mousemove.zoomRect", this.moveRectSel.bind(this))
                       .on("mouseup.zoomRect", this.endRectSel.bind(this), true);

      this.zoom_rect = null;

      // disable tooltips in frame painter
      this.SwitchTooltip(false);

      d3.event.stopPropagation();
   }

   JSROOT.THistPainter.prototype.moveRectSel = function() {

      if ((this.zoom_kind == 0) || (this.zoom_kind > 100)) return;

      d3.event.preventDefault();
      var m = d3.mouse(this.svg_frame().node());

      m[0] = Math.max(0, Math.min(this.frame_width(), m[0]));
      m[1] = Math.max(0, Math.min(this.frame_height(), m[1]));

      switch (this.zoom_kind) {
         case 1: this.zoom_curr[0] = m[0]; this.zoom_curr[1] = m[1]; break;
         case 2: this.zoom_curr[0] = m[0]; break;
         case 3: this.zoom_curr[1] = m[1]; break;
      }

      if (this.zoom_rect===null)
         this.zoom_rect = this.svg_frame()
                              .append("rect")
                              .attr("class", "zoom")
                              .attr("pointer-events","none");

      this.zoom_rect.attr("x", Math.min(this.zoom_origin[0], this.zoom_curr[0]))
                    .attr("y", Math.min(this.zoom_origin[1], this.zoom_curr[1]))
                    .attr("width", Math.abs(this.zoom_curr[0] - this.zoom_origin[0]))
                    .attr("height", Math.abs(this.zoom_curr[1] - this.zoom_origin[1]));
   }

   JSROOT.THistPainter.prototype.endRectSel = function() {
      if ((this.zoom_kind == 0) || (this.zoom_kind > 100)) return;

      d3.event.preventDefault();

      d3.select(window).on("mousemove.zoomRect", null)
                       .on("mouseup.zoomRect", null);

      var m = d3.mouse(this.svg_frame().node());

      m[0] = Math.max(0, Math.min(this.frame_width(), m[0]));
      m[1] = Math.max(0, Math.min(this.frame_height(), m[1]));

      var changed = [true, true];

      switch (this.zoom_kind) {
         case 1: this.zoom_curr[0] = m[0]; this.zoom_curr[1] = m[1]; break;
         case 2: this.zoom_curr[0] = m[0]; changed[1] = false; break; // only X
         case 3: this.zoom_curr[1] = m[1]; changed[0] = false; break; // only Y
      }

      var xmin, xmax, ymin, ymax, isany = false,
          idx = this.swap_xy ? 1 : 0, idy = 1 - idx;

      if (changed[idx] && (Math.abs(this.zoom_curr[idx] - this.zoom_origin[idx]) > 10)) {
         xmin = Math.min(this.RevertX(this.zoom_origin[idx]), this.RevertX(this.zoom_curr[idx]));
         xmax = Math.max(this.RevertX(this.zoom_origin[idx]), this.RevertX(this.zoom_curr[idx]));
         isany = true;
      }

      if (changed[idy] && (Math.abs(this.zoom_curr[idy] - this.zoom_origin[idy]) > 10)) {
         ymin = Math.min(this.RevertY(this.zoom_origin[idy]), this.RevertY(this.zoom_curr[idy]));
         ymax = Math.max(this.RevertY(this.zoom_origin[idy]), this.RevertY(this.zoom_curr[idy]));
         isany = true;
      }

      this.clearInteractiveElements();

      if (isany) {
         this.zoom_changed_interactive = true;
         this.Zoom(xmin, xmax, ymin, ymax);
      }
   }

   JSROOT.THistPainter.prototype.startTouchZoom = function() {
      // in case when zooming was started, block any other kind of events
      if (this.zoom_kind != 0) {
         d3.event.preventDefault();
         d3.event.stopPropagation();
         return;
      }

      var arr = d3.touches(this.svg_frame().node());
      this.touch_cnt+=1;

      // normally double-touch will be handled
      // touch with single click used for context menu
      if (arr.length == 1) {
         // this is touch with single element

         var now = new Date();
         var diff = now.getTime() - this.last_touch.getTime();
         this.last_touch = now;

         if ((diff < 300) && (this.zoom_curr != null)
               && (Math.abs(this.zoom_curr[0] - arr[0][0]) < 30)
               && (Math.abs(this.zoom_curr[1] - arr[0][1]) < 30)) {

            d3.event.preventDefault();
            d3.event.stopPropagation();

            this.clearInteractiveElements();
            this.Unzoom("xyz");

            this.last_touch = new Date(0);

            this.svg_frame().on("touchcancel", null)
                            .on("touchend", null, true);
         } else
         if (JSROOT.gStyle.ContextMenu) {
            this.zoom_curr = arr[0];
            this.svg_frame().on("touchcancel", this.endTouchSel.bind(this))
                            .on("touchend", this.endTouchSel.bind(this));
            d3.event.preventDefault();
            d3.event.stopPropagation();
         }
      }

      if (arr.length != 2) return;

      d3.event.preventDefault();
      d3.event.stopPropagation();

      this.clearInteractiveElements();

      this.svg_frame().on("touchcancel", null)
                      .on("touchend", null);

      var pnt1 = arr[0], pnt2 = arr[1];

      this.zoom_curr = [ Math.min(pnt1[0], pnt2[0]), Math.min(pnt1[1], pnt2[1]) ];
      this.zoom_origin = [ Math.max(pnt1[0], pnt2[0]), Math.max(pnt1[1], pnt2[1]) ];

      if (this.zoom_curr[0] < 0) {
         this.zoom_kind = 103; // only y
         this.zoom_curr[0] = 0;
         this.zoom_origin[0] = this.frame_width();
      } else if (this.zoom_origin[1] > this.frame_height()) {
         this.zoom_kind = 102; // only x
         this.zoom_curr[1] = 0;
         this.zoom_origin[1] = this.frame_height();
      } else {
         this.zoom_kind = 101; // x and y
      }

      this.SwitchTooltip(false);

      this.zoom_rect = this.svg_frame().append("rect")
            .attr("class", "zoom")
            .attr("id", "zoomRect")
            .attr("x", this.zoom_curr[0])
            .attr("y", this.zoom_curr[1])
            .attr("width", this.zoom_origin[0] - this.zoom_curr[0])
            .attr("height", this.zoom_origin[1] - this.zoom_curr[1]);

      d3.select(window).on("touchmove.zoomRect", this.moveTouchSel.bind(this))
                       .on("touchcancel.zoomRect", this.endTouchSel.bind(this))
                       .on("touchend.zoomRect", this.endTouchSel.bind(this));
   }

   JSROOT.THistPainter.prototype.moveTouchSel = function() {
      if (this.zoom_kind < 100) return;

      d3.event.preventDefault();

      var arr = d3.touches(this.svg_frame().node());

      if (arr.length != 2)
         return this.clearInteractiveElements();

      var pnt1 = arr[0], pnt2 = arr[1];

      if (this.zoom_kind != 103) {
         this.zoom_curr[0] = Math.min(pnt1[0], pnt2[0]);
         this.zoom_origin[0] = Math.max(pnt1[0], pnt2[0]);
      }
      if (this.zoom_kind != 102) {
         this.zoom_curr[1] = Math.min(pnt1[1], pnt2[1]);
         this.zoom_origin[1] = Math.max(pnt1[1], pnt2[1]);
      }

      this.zoom_rect.attr("x", this.zoom_curr[0])
                     .attr("y", this.zoom_curr[1])
                     .attr("width", this.zoom_origin[0] - this.zoom_curr[0])
                     .attr("height", this.zoom_origin[1] - this.zoom_curr[1]);

      if ((this.zoom_origin[0] - this.zoom_curr[0] > 10)
           || (this.zoom_origin[1] - this.zoom_curr[1] > 10))
         this.SwitchTooltip(false);

      d3.event.stopPropagation();
   }

   JSROOT.THistPainter.prototype.endTouchSel = function() {

      this.svg_frame().on("touchcancel", null)
                      .on("touchend", null);

      if (this.zoom_kind === 0) {
         // special case - single touch can ends up with context menu

         d3.event.preventDefault();

         var now = new Date();

         var diff = now.getTime() - this.last_touch.getTime();

         if ((diff > 500) && (diff<2000) && !this.frame_painter().IsTooltipShown()) {
            this.ShowContextMenu('main', { clientX: this.zoom_curr[0], clientY: this.zoom_curr[1] });
            this.last_touch = new Date(0);
         } else {
            this.clearInteractiveElements();
         }
      }

      if (this.zoom_kind < 100) return;

      d3.event.preventDefault();
      d3.select(window).on("touchmove.zoomRect", null)
                       .on("touchend.zoomRect", null)
                       .on("touchcancel.zoomRect", null);

      var xmin, xmax, ymin, ymax, isany = false,
          xid = this.swap_xy ? 1 : 0, yid = 1 - xid,
          changed = [true, true];
      if (this.zoom_kind === 102) changed[1] = false;
      if (this.zoom_kind === 103) changed[0] = false;

      if (changed[xid] && (Math.abs(this.zoom_curr[xid] - this.zoom_origin[xid]) > 10)) {
         xmin = Math.min(this.RevertX(this.zoom_origin[xid]), this.RevertX(this.zoom_curr[xid]));
         xmax = Math.max(this.RevertX(this.zoom_origin[xid]), this.RevertX(this.zoom_curr[xid]));
         isany = true;
      }

      if (changed[yid] && (Math.abs(this.zoom_curr[yid] - this.zoom_origin[yid]) > 10)) {
         ymin = Math.min(this.RevertY(this.zoom_origin[yid]), this.RevertY(this.zoom_curr[yid]));
         ymax = Math.max(this.RevertY(this.zoom_origin[yid]), this.RevertY(this.zoom_curr[yid]));
         isany = true;
      }

      this.clearInteractiveElements();
      this.last_touch = new Date(0);

      if (isany) {
         this.zoom_changed_interactive = true;
         this.Zoom(xmin, xmax, ymin, ymax);
      }

      d3.event.stopPropagation();
   }

   JSROOT.THistPainter.prototype.mouseWheel = function() {
      d3.event.stopPropagation();

      var delta = 0;
      switch (d3.event.deltaMode) {
         case 0: delta = d3.event.deltaY / this.pad_height() * 2; break; // DOM_DELTA_PIXEL
         case 1: delta = d3.event.deltaY / this.pad_height() * 40; break; // DOM_DELTA_LINE
         case 2: delta = d3.event.deltaY / this.pad_height() * 200; break; // DOM_DELTA_PAGE
      }
      if (delta===0) return;

      d3.event.preventDefault();

      this.clearInteractiveElements();

      if (delta < -0.2) delta = -0.2; else if (delta>0.2) delta = 0.2;

      var itemx = { name: "x", ignore: false },
          itemy = { name: "y", ignore: (this.Dimension() == 1) },
          cur = d3.mouse(this.svg_frame().node()),
          painter = this;

      function ProcessAxis(item, dmin, ignore) {
         item.min = item.max = undefined;
         item.changed = false;
         if (ignore && item.ignore) return;

         item.min = painter["scale_" + item.name+"min"];
         item.max = painter["scale_" + item.name+"max"];
         if ((item.min === item.max) && (delta<0)) {
            item.min = painter[item.name+"min"];
            item.max = painter[item.name+"max"];
         }

         if (item.min >= item.max) return;

         if ((dmin>0) && (dmin<1)) {
            if (painter['log'+item.name]) {
               var factor = (item.min>0) ? JSROOT.log10(item.max/item.min) : 2;
               if (factor>10) factor = 10; else if (factor<1.5) factor = 1.5;
               item.min = item.min / Math.pow(factor, delta*dmin);
               item.max = item.max * Math.pow(factor, delta*(1-dmin));
            } else {
               var rx = (item.max - item.min);
               if (delta>0) rx = 1.001 * rx / (1-delta);
               item.min += -delta*dmin*rx;
               item.max -= -delta*(1-dmin)*rx;
            }
            if (item.min >= item.max) item.min = item.max = undefined;
         } else {
            item.min = item.max = undefined;
         }

         item.changed = ((item.min !== undefined) && (item.max !== undefined));
      }

      ProcessAxis(this.swap_xy ? itemy : itemx, cur[0] / this.frame_width(), (cur[1] <= this.frame_height()));

      ProcessAxis(this.swap_xy ? itemx : itemy, 1 - cur[1] / this.frame_height(), (cur[0] >= 0));

      this.Zoom(itemx.min, itemx.max, itemy.min, itemy.max);

      if (itemx.changed || itemy.changed) this.zoom_changed_interactive = true;
   }

   JSROOT.THistPainter.prototype.AddInteractive = function() {
      // only first painter in list allowed to add interactive functionality to the frame

      if ((!JSROOT.gStyle.Zooming && !JSROOT.gStyle.ContextMenu) || !this.is_main_painter()) return;

      var svg = this.svg_frame();

      if (svg.empty() || svg.property('interactive_set')) return;

      this.last_touch = new Date(0);
      this.zoom_kind = 0; // 0 - none, 1 - XY, 2 - only X, 3 - only Y, (+100 for touches)
      this.zoom_rect = null;
      this.zoom_origin = null;  // original point where zooming started
      this.zoom_curr = null;    // current point for zomming
      this.touch_cnt = 0;

      if (JSROOT.gStyle.Zooming) {
         svg.on("mousedown", this.startRectSel.bind(this) );
         svg.on("dblclick", this.mouseDoubleClick.bind(this) );
         svg.on("wheel", this.mouseWheel.bind(this) );
      }

      if (JSROOT.touches && (JSROOT.gStyle.Zooming || JSROOT.gStyle.ContextMenu))
         svg.on("touchstart", this.startTouchZoom.bind(this) );

      if (JSROOT.gStyle.ContextMenu) {
         if (JSROOT.touches) {
            svg.selectAll(".xaxis_container")
               .on("touchstart", this.startTouchMenu.bind(this,"x") );
            svg.selectAll(".yaxis_container")
                .on("touchstart", this.startTouchMenu.bind(this,"y") );
         }
         svg.on("contextmenu", this.ShowContextMenu.bind(this) );
         svg.selectAll(".xaxis_container")
             .on("contextmenu", this.ShowContextMenu.bind(this,"x"));
         svg.selectAll(".yaxis_container")
             .on("contextmenu", this.ShowContextMenu.bind(this,"y"));
      }

      svg.property('interactive_set', true);
   }

   JSROOT.THistPainter.prototype.ShowContextMenu = function(kind, evnt, obj) {
      // ignore context menu when touches zooming is ongoing
      if (('zoom_kind' in this) && (this.zoom_kind > 100)) return;

      // this is for debug purposes only, when context menu is where, close is and show normal menu
      //if (!evnt && !kind && document.getElementById('root_ctx_menu')) {
      //   var elem = document.getElementById('root_ctx_menu');
      //   elem.parentNode.removeChild(elem);
      //   return;
      //}

      var menu_painter = this, frame_corner = false; // object used to show context menu

      if (!evnt) {
         d3.event.preventDefault();
         d3.event.stopPropagation(); // disable main context menu
         evnt = d3.event;

         if (kind === undefined) {
            var ms = d3.mouse(this.svg_frame().node());
            var tch = d3.touches(this.svg_frame().node());
            var pnt = null, pp = this.pad_painter(true), fp = this.frame_painter(), sel = null;

            if (tch.length === 1) pnt = { x: tch[0][0], y: tch[0][1], touch: true }; else
            if (ms.length === 2) pnt = { x: ms[0], y: ms[1], touch: false };

            if ((pnt !== null) && (pp !== null)) {
               pnt.painters = true; // assign painter for every tooltip
               var hints = pp.GetTooltips(pnt);

               var bestdist = 1000;
               for (var n=0;n<hints.length;++n)
                  if (hints[n] && hints[n].menu) {
                     var dist = ('menu_dist' in hints[n]) ? hints[n].menu_dist : 7;
                     if (dist < bestdist) { sel = hints[n].painter; bestdist = dist; }
                  }
            }

            if (sel!==null) menu_painter = sel; else
            if (fp!==null) kind = "frame";

            if (pnt!==null) frame_corner = (pnt.x>0) && (pnt.x<20) && (pnt.y>0) && (pnt.y<20);
         }
      }

      // one need to copy event, while after call back event may be changed
      menu_painter.ctx_menu_evnt = evnt;

      JSROOT.Painter.createMenu(function(menu) {
         menu.painter = this; // in all menu callbacks painter will be 'this' pointer
         var domenu = this.FillContextMenu(menu, kind, obj);

         // fill frame menu by default - or append frame elements when actiavted in the frame corner
         if (fp && (!domenu || (frame_corner && (kind!=="frame"))))
            domenu = fp.FillContextMenu(menu);

         if (domenu) {
            // suppress any running zomming
            this.SwitchTooltip(false);
            menu.show(this.ctx_menu_evnt, this.SwitchTooltip.bind(this, true) );
         }

         delete this.ctx_menu_evnt; // delete temporary variable
      }.bind(menu_painter) );  // end menu creation
   }


   JSROOT.THistPainter.prototype.ChangeUserRange = function(arg) {
      var taxis = this.histo['f'+arg+"axis"];
      if (!taxis) return;

      var curr = "[1," + taxis.fNbins+"]";
      if (taxis.TestBit(JSROOT.EAxisBits.kAxisRange))
          curr = "[" +taxis.fFirst+"," + taxis.fLast+"]";

      var res = prompt("Enter user range for axis " + arg + " like [1," + taxis.fNbins + "]", curr);
      if (res==null) return;
      res = JSON.parse(res);

      if (!res || (res.length!=2) || isNaN(res[0]) || isNaN(res[1])) return;
      taxis.fFirst = parseInt(res[0]);
      taxis.fLast = parseInt(res[1]);

      var newflag = (taxis.fFirst < taxis.fLast) && (taxis.fFirst >= 1) && (taxis.fLast<=taxis.fNbins);
      if (newflag != taxis.TestBit(JSROOT.EAxisBits.kAxisRange))
         taxis.InvertBit(JSROOT.EAxisBits.kAxisRange);

      this.Redraw();
   }

   JSROOT.THistPainter.prototype.FillContextMenu = function(menu, kind, obj) {

      // when fill and show context menu, remove all zooming
      this.clearInteractiveElements();

      if ((kind=="x") || (kind=="y") || (kind=="z")) {
         var faxis = this.histo.fXaxis;
         if (kind=="y")  faxis = this.histo.fYaxis;  else
         if (kind=="z")  faxis = obj ? obj : this.histo.fZaxis;
         menu.add("header: " + kind.toUpperCase() + " axis");
         menu.add("Unzoom", this.Unzoom.bind(this, kind));
         menu.addchk(this.options["Log" + kind], "SetLog"+kind, this.ToggleLog.bind(this, kind) );
         menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kMoreLogLabels), "More log",
               function() { faxis.InvertBit(JSROOT.EAxisBits.kMoreLogLabels); this.RedrawPad(); });
         menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kNoExponent), "No exponent",
               function() { faxis.InvertBit(JSROOT.EAxisBits.kNoExponent); this.RedrawPad(); });

         if ((kind === "z") && (this.options.Zscale > 0))
            if (this.FillPaletteMenu) this.FillPaletteMenu(menu);

         if (faxis != null) {
            menu.add("sub:Labels");
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kCenterLabels), "Center",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kCenterLabels); this.RedrawPad(); });
            this.AddColorMenuEntry(menu, "Color", faxis.fLabelColor,
                  function(arg) { faxis.fLabelColor = parseInt(arg); this.RedrawPad(); });
            this.AddSizeMenuEntry(menu,"Offset", 0, 0.1, 0.01, faxis.fLabelOffset,
                  function(arg) { faxis.fLabelOffset = parseFloat(arg); this.RedrawPad(); } );
            this.AddSizeMenuEntry(menu,"Size", 0.02, 0.11, 0.01, faxis.fLabelSize,
                  function(arg) { faxis.fLabelSize = parseFloat(arg); this.RedrawPad(); } );
            menu.add("endsub:");
            menu.add("sub:Title");
            menu.add("SetTitle", function() {
               var t = prompt("Enter axis title", faxis.fTitle);
               if (t!==null) { faxis.fTitle = t; this.RedrawPad(); }
            });
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kCenterTitle), "Center",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kCenterTitle); this.RedrawPad(); });
            menu.addchk(faxis.TestBit(JSROOT.EAxisBits.kRotateTitle), "Rotate",
                  function() { faxis.InvertBit(JSROOT.EAxisBits.kRotateTitle); this.RedrawPad(); });
            this.AddColorMenuEntry(menu, "Color", faxis.fTitleColor,
                  function(arg) { faxis.fTitleColor = parseInt(arg); this.RedrawPad(); });
            this.AddSizeMenuEntry(menu,"Offset", 0, 3, 0.2, faxis.fTitleOffset,
                                  function(arg) { faxis.fTitleOffset = parseFloat(arg); this.RedrawPad(); } );
            this.AddSizeMenuEntry(menu,"Size", 0.02, 0.11, 0.01, faxis.fTitleSize,
                  function(arg) { faxis.fTitleSize = parseFloat(arg); this.RedrawPad(); } );
            menu.add("endsub:");
         }
         menu.add("sub:Ticks");
         this.AddColorMenuEntry(menu, "Color", faxis.fLineColor,
                     function(arg) { faxis.fLineColor = parseInt(arg); this.RedrawPad(); });
         this.AddColorMenuEntry(menu, "Color", faxis.fAxisColor,
                     function(arg) { faxis.fAxisColor = parseInt(arg); this.RedrawPad(); });
         this.AddSizeMenuEntry(menu,"Size", -0.05, 0.055, 0.01, faxis.fTickLength,
                   function(arg) { faxis.fTickLength = parseFloat(arg); this.RedrawPad(); } );
         menu.add("endsub:");
         return true;
      }

      if (kind == "frame") {
         var fp = this.frame_painter();
         if (fp) return fp.FillContextMenu(menu);
      }

      menu.add("header:"+ this.histo._typename + "::" + this.histo.fName);

      if (this.draw_content) {
         menu.addchk(this.ToggleStat('only-check'), "Show statbox", function() { this.ToggleStat(); });
         if (this.Dimension() == 1) {
            menu.add("User range X", "X", this.ChangeUserRange);
         } else {
            menu.add("sub:User ranges");
            menu.add("X", "X", this.ChangeUserRange);
            menu.add("Y", "Y", this.ChangeUserRange);
            if (this.Dimension() > 2)
               menu.add("Z", "Z", this.ChangeUserRange);
            menu.add("endsub:")
         }

         if (typeof this.FillHistContextMenu == 'function')
            this.FillHistContextMenu(menu);
      }

      if ((this.options.Lego > 0) || (this.Dimension() === 3)) {
         // menu for 3D drawings

         if (menu.size() > 0)
            menu.add("separator");

         menu.addchk(this.enable_tooltip, 'Show tooltips', function() {
            this.enable_tooltip = !this.enable_tooltip;
         });

         if (this.enable_tooltip)
            menu.addchk(this.enable_hightlight, 'Hightlight bins', function() {
               this.enable_hightlight = !this.enable_hightlight;
               if (!this.enable_hightlight && this.BinHighlight3D) this.BinHighlight3D(null);
            });

         menu.addchk(this.options.FrontBox, 'Front box', function() {
            this.options.FrontBox = !this.options.FrontBox;
            if (this.Render3D) this.Render3D();
         });
         menu.addchk(this.options.BackBox, 'Back box', function() {
            this.options.BackBox = !this.options.BackBox;
            if (this.Render3D) this.Render3D();
         });


         if (this.draw_content) {
            menu.addchk(this.options.Zero, 'Suppress zeros', function() {
               this.options.Zero = !this.options.Zero;
               this.RedrawPad();
            });

            if ((this.options.Lego==12) || (this.options.Lego==14)) {
               menu.addchk(this.options.Zscale, "Z scale", function() {
                  this.ToggleColz();
               });
               if (this.FillPaletteMenu) this.FillPaletteMenu(menu);
            }
         }

         if (this.control && typeof this.control.reset === 'function')
            menu.add('Reset camera', function() {
               this.control.reset();
            });
      }

      this.FillAttContextMenu(menu);

      return true;
   }

   JSROOT.THistPainter.prototype.ButtonClick = function(funcname) {
      if (!this.is_main_painter()) return false;
      switch(funcname) {
         case "ToggleZoom":
            if ((this.zoom_xmin !== this.zoom_xmax) || (this.zoom_ymin !== this.zoom_ymax) || (this.zoom_zmin !== this.zoom_zmax)) {
               this.Unzoom();
               return true;
            }
            if (this.draw_content && ('AutoZoom' in this) && (this.Dimension() < 3)) {
               this.AutoZoom();
               return true;
            }
            break;
         case "ToggleLogX": this.ToggleLog("x"); break;
         case "ToggleLogY": this.ToggleLog("y"); break;
         case "ToggleLogZ": this.ToggleLog("z"); break;
         case "ToggleStatBox": this.ToggleStat(); return true; break;
      }
      return false;
   }

   JSROOT.THistPainter.prototype.FillToolbar = function() {
      var pp = this.pad_painter(true);
      if (pp===null) return;

      pp.AddButton(JSROOT.ToolbarIcons.auto_zoom, 'Toggle between unzoom and autozoom-in', 'ToggleZoom');
      pp.AddButton(JSROOT.ToolbarIcons.arrow_right, "Toggle log x", "ToggleLogX");
      pp.AddButton(JSROOT.ToolbarIcons.arrow_up, "Toggle log y", "ToggleLogY");
      if (this.Dimension() > 1)
         pp.AddButton(JSROOT.ToolbarIcons.arrow_diag, "Toggle log z", "ToggleLogZ");
      if (this.draw_content)
         pp.AddButton(JSROOT.ToolbarIcons.statbox, 'Toggle stat box', "ToggleStatBox");
   }


   // ======= TH1 painter================================================

   JSROOT.TH1Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);
   }

   JSROOT.TH1Painter.prototype = Object.create(JSROOT.THistPainter.prototype);

   JSROOT.TH1Painter.prototype.ScanContent = function() {
      // from here we analyze object content
      // therefore code will be moved

      this.nbinsx = this.histo.fXaxis.fNbins;
      this.nbinsy = 0;

      this.CreateAxisFuncs(false);

      var hmin = 0, hmin_nz = 0, hmax = 0, hsum = 0, first = true,
          left = this.GetSelectIndex("x","left"),
          right = this.GetSelectIndex("x","right"),
          profile = this.IsTProfile(), value, err;

      for (var i = 0; i < this.nbinsx; ++i) {
         value = this.histo.getBinContent(i + 1);
         hsum += profile ? this.histo.fBinEntries[i + 1] : value;

         if ((i<left) || (i>=right)) continue;

         if (value > 0)
            if ((hmin_nz == 0) || (value<hmin_nz)) hmin_nz = value;
         if (first) {
            hmin = hmax = value;
            first = false;;
         }

         err = (this.options.Error > 0) ? this.histo.getBinError(i + 1) : 0;

         hmin = Math.min(hmin, value - err);
         hmax = Math.max(hmax, value + err);
      }

      // account overflow/underflow bins
      if (profile)
         hsum += this.histo.fBinEntries[0] + this.histo.fBinEntries[this.nbinsx + 1];
      else
         hsum += this.histo.getBinContent(0) + this.histo.getBinContent(this.nbinsx + 1);

      this.stat_entries = hsum;
      if (this.histo.fEntries>1) this.stat_entries = this.histo.fEntries;

      this.hmin = hmin;
      this.hmax = hmax;

      this.ymin_nz = hmin_nz; // value can be used to show optimal log scale

      if ((this.nbinsx == 0) || ((Math.abs(hmin) < 1e-300 && Math.abs(hmax) < 1e-300))) {
         this.draw_content = false;
         hmin = this.ymin;
         hmax = this.ymax;
      } else {
         this.draw_content = true;
      }

      if (hmin >= hmax) {
         if (hmin == 0) { this.ymin = 0; this.ymax = 1; } else
         if (hmin < 0) { this.ymin = 2 * hmin; this.ymax = 0; }
                  else { this.ymin = 0; this.ymax = hmin * 2; }
      } else {
         var dy = (hmax - hmin) * 0.05;
         this.ymin = hmin - dy;
         if ((this.ymin < 0) && (hmin >= 0)) this.ymin = 0;
         this.ymax = hmax + dy;
      }

      hmin = hmax = null;
      var set_zoom = false;
      if (this.histo.fMinimum != -1111) {
         hmin = this.histo.fMinimum;
         if (hmin < this.ymin)
            this.ymin = hmin;
         else
            set_zoom = true;
      }
      if (this.histo.fMaximum != -1111) {
         hmax = this.histo.fMaximum;
         if (hmax > this.ymax)
            this.ymax = hmax;
         else
            set_zoom = true;
      }

      if (set_zoom) {
         this.zoom_ymin = (hmin == null) ? this.ymin : hmin;
         this.zoom_ymax = (hmax == null) ? this.ymax : hmax;
      }

      // If no any draw options specified, do not try draw histogram
      if (!this.options.Bar && !this.options.Hist &&
          !this.options.Error && !this.options.Same && !this.options.Lego) {
         this.draw_content = false;
      }
      if (this.options.Axis > 0) { // Paint histogram axis only
         this.draw_content = false;
      }
   }

   JSROOT.TH1Painter.prototype.CountStat = function(cond) {
      var profile = this.IsTProfile();

      var stat_sumw = 0, stat_sumwx = 0, stat_sumwx2 = 0, stat_sumwy = 0, stat_sumwy2 = 0;

      var left = this.GetSelectIndex("x", "left");
      var right = this.GetSelectIndex("x", "right");

      var xx = 0, w = 0, xmax = null, wmax = null;

      for (var i = left; i < right; ++i) {
         xx = this.GetBinX(i+0.5);

         if ((cond!=null) && !cond(xx)) continue;

         if (profile) {
            w = this.histo.fBinEntries[i + 1];
            stat_sumwy += this.histo.fArray[i + 1];
            stat_sumwy2 += this.histo.fSumw2[i + 1];
         } else {
            w = this.histo.getBinContent(i + 1);
         }

         if ((xmax==null) || (w>wmax)) { xmax = xx; wmax = w; }

         stat_sumw += w;
         stat_sumwx += w * xx;
         stat_sumwx2 += w * xx * xx;
      }

      // when no range selection done, use original statistic from histogram
      if (!this.IsAxisZoomed("x") && (this.histo.fTsumw>0)) {
         stat_sumw = this.histo.fTsumw;
         stat_sumwx = this.histo.fTsumwx;
         stat_sumwx2 = this.histo.fTsumwx2;
      }

      var res = { meanx: 0, meany: 0, rmsx: 0, rmsy: 0, integral: stat_sumw, entries: this.stat_entries, xmax:0, wmax:0 };

      if (stat_sumw > 0) {
         res.meanx = stat_sumwx / stat_sumw;
         res.meany = stat_sumwy / stat_sumw;
         res.rmsx = Math.sqrt(stat_sumwx2 / stat_sumw - res.meanx * res.meanx);
         res.rmsy = Math.sqrt(stat_sumwy2 / stat_sumw - res.meany * res.meany);
      }

      if (xmax!=null) {
         res.xmax = xmax;
         res.wmax = wmax;
      }

      return res;
   }

   JSROOT.TH1Painter.prototype.FillStatistic = function(stat, dostat, dofit) {
      if (!this.histo) return false;

      var pave = stat.GetObject(),
          data = this.CountStat(),
          print_name = dostat % 10,
          print_entries = Math.floor(dostat / 10) % 10,
          print_mean = Math.floor(dostat / 100) % 10,
          print_rms = Math.floor(dostat / 1000) % 10,
          print_under = Math.floor(dostat / 10000) % 10,
          print_over = Math.floor(dostat / 100000) % 10,
          print_integral = Math.floor(dostat / 1000000) % 10,
          print_skew = Math.floor(dostat / 10000000) % 10,
          print_kurt = Math.floor(dostat / 100000000) % 10;

      if (print_name > 0)
         pave.AddText(this.histo.fName);

      if (this.IsTProfile()) {

         if (print_entries > 0)
            pave.AddText("Entries = " + stat.Format(data.entries,"entries"));

         if (print_mean > 0) {
            pave.AddText("Mean = " + stat.Format(data.meanx));
            pave.AddText("Mean y = " + stat.Format(data.meany));
         }

         if (print_rms > 0) {
            pave.AddText("Std Dev = " + stat.Format(data.rmsx));
            pave.AddText("Std Dev y = " + stat.Format(data.rmsy));
         }

      } else {

         if (print_entries > 0)
            pave.AddText("Entries = " + stat.Format(data.entries,"entries"));

         if (print_mean > 0) {
            pave.AddText("Mean = " + stat.Format(data.meanx));
         }

         if (print_rms > 0) {
            pave.AddText("Std Dev = " + stat.Format(data.rmsx));
         }

         if (print_under > 0) {
            var res = (this.histo.fArray.length > 0) ? this.histo.fArray[0] : 0;
            pave.AddText("Underflow = " + stat.Format(res));
         }

         if (print_over > 0) {
            var res = (this.histo.fArray.length > 0) ? this.histo.fArray[this.histo.fArray.length - 1] : 0;
            pave.AddText("Overflow = " + stat.Format(res));
         }

         if (print_integral > 0) {
            pave.AddText("Integral = " + stat.Format(data.integral,"entries"));
         }

         if (print_skew > 0)
            pave.AddText("Skew = <not avail>");

         if (print_kurt > 0)
            pave.AddText("Kurt = <not avail>");
      }

      if (dofit!=0) {
         var f1 = this.FindFunction('TF1');
         if (f1!=null) {
            var print_fval    = dofit%10;
            var print_ferrors = Math.floor(dofit/10) % 10;
            var print_fchi2   = Math.floor(dofit/100) % 10;
            var print_fprob   = Math.floor(dofit/1000) % 10;

            if (print_fchi2 > 0)
               pave.AddText("#chi^2 / ndf = " + stat.Format(f1.fChisquare,"fit") + " / " + f1.fNDF);
            if (print_fprob > 0)
               pave.AddText("Prob = "  + (('Math' in JSROOT) ? stat.Format(JSROOT.Math.Prob(f1.fChisquare, f1.fNDF)) : "<not avail>"));
            if (print_fval > 0) {
               for(var n=0;n<f1.fNpar;++n) {
                  var parname = f1.GetParName(n);
                  var parvalue = f1.GetParValue(n);
                  if (parvalue != null) parvalue = stat.Format(Number(parvalue),"fit");
                                 else  parvalue = "<not avail>";
                  var parerr = "";
                  if (f1.fParErrors!=null) {
                     parerr = stat.Format(f1.fParErrors[n],"last");
                     if ((Number(parerr)==0.0) && (f1.fParErrors[n]!=0.0)) parerr = stat.Format(f1.fParErrors[n],"4.2g");
                  }

                  if ((print_ferrors > 0) && (parerr.length > 0))
                     pave.AddText(parname + " = " + parvalue + " #pm " + parerr);
                  else
                     pave.AddText(parname + " = " + parvalue);
               }
            }
         }
      }

      // adjust the size of the stats box with the number of lines
      var nlines = pave.fLines.arr.length,
          stath = nlines * JSROOT.gStyle.StatFontSize;
      if ((stath <= 0) || (JSROOT.gStyle.StatFont % 10 === 3)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         pave.fY1NDC = 0.93 - stath;
         pave.fY2NDC = 0.93;
      }

      return true;
   }

   JSROOT.TH1Painter.prototype.DrawBars = function() {
      var width = this.frame_width(), height = this.frame_height();

      this.RecreateDrawG(false, "main_layer");

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          pmain = this.main_painter(),
          pad = this.root_pad(),
          pthis = this,
          i, x1, x2, grx1, grx2, y, gry, w;

      var bars = "", barsl = "", barsr = "", side = this.options.Bar % 10;

      if (side>4) side = 4;

      for (i = left; i < right; ++i) {
         x1 = this.GetBinX(i);
         x2 = this.GetBinX(i+1);

         if (pmain.logx && (x2 <= 0)) continue;

         grx1 = Math.round(pmain.grx(x1));
         grx2 = Math.round(pmain.grx(x2));

         y = this.histo.getBinContent(i+1);
         if (pmain.logy && (y < pmain.scale_ymin)) continue;
         gry = Math.round(pmain.gry(y));

         w = grx2 - grx1;
         grx1 += Math.round(this.histo.fBarOffset/1000*w);
         w = Math.round(this.histo.fBarWidth/1000*w);

         if (pmain.swap_xy)
            bars += "M0,"+ grx1 + "h" + gry + "v" + w + "h" + (-gry)+ "z";
         else
            bars += "M"+grx1+"," + gry + "h" + w + "v" + (height - gry) + "h" + (-w)+ "z";

         if (side > 0) {
            grx2 = grx1 + w;
            w = Math.round(w * side / 10);
            if (pmain.swap_xy) {
               barsl += "M0,"+ grx1 + "h" + gry + "v" + w + "h" + (-gry)+ "z";
               barsr += "M0,"+ grx2 + "h" + gry + "v" + (-w) + "h" + (-gry)+ "z";
            } else {
               barsl += "M"+grx1+","+gry + "h" + w + "v" + (height - gry) + "h" + (-w)+ "z";
               barsr += "M"+grx2+","+gry + "h" + (-w) + "v" + (height - gry) + "h" + w + "z";
            }
         }
      }

      if (bars.length > 0)
         this.draw_g.append("svg:path")
                    .attr("d", bars)
                    .call(this.fillatt.func);

      if (barsl.length > 0)
         this.draw_g.append("svg:path")
               .attr("d", barsl)
               .call(this.fillatt.func)
               .style("fill", d3.rgb(this.fillatt.color).brighter(0.5).toString());

      if (barsr.length > 0)
         this.draw_g.append("svg:path")
               .attr("d", barsr)
               .call(this.fillatt.func)
               .style("fill", d3.rgb(this.fillatt.color).darker(0.5).toString());
   }

   JSROOT.TH1Painter.prototype.DrawBins = function() {
      // new method, create svg:path expression ourself directly from histogram
      // all points will be used, compress expression when too large

      this.CheckHistDrawAttributes();

      if (this.options.Bar > 0)
         return this.DrawBars();

      var width = this.frame_width(), height = this.frame_height();

      if (!this.draw_content || (width<=0) || (height<=0))
         return this.RemoveDrawG();

      this.RecreateDrawG(false, "main_layer");

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 2),
          pmain = this.main_painter(),
          pad = this.root_pad(),
          pthis = this,
          res = "", lastbin = false,
          startx, currx, curry, x, grx, y, gry, curry_min, curry_max, prevy, prevx, i, besti,
          exclude_zero = (this.options.Error!==10) && (this.options.Mark!==10),
          show_errors = (this.options.Error > 0),
          show_markers = (this.options.Mark > 0),
          path_fill = null, path_err = null, path_marker = null,
          endx = "", endy = "", dend = 0, my, yerr1, yerr2, bincont, binerr, mx1, mx2, mpath = "";

      if (show_errors && !show_markers && (this.histo.fMarkerStyle > 1))
         show_markers = true;

      if (this.options.Error == 12) {
         if (this.fillatt.color=='none') show_markers = true;
                                    else path_fill = "";
      } else
      if (this.options.Error > 0) path_err = "";

      if (show_markers) {
         // draw markers also when e2 option was specified
         if (!this.markeratt)
            this.markeratt = JSROOT.Painter.createAttMarker(this.histo);
         if (this.markeratt.size > 0) {
            // simply use relative move from point, can optimize in the future
            path_marker = "";
            this.markeratt.reset_pos();
         } else {
            show_markers = false;
         }
      }

      // if there are too many points, exclude many vertical drawings at the same X position
      // instead define min and max value and made min-max drawing
      var use_minmax = ((right-left) > 3*width);

      if (this.options.Error == 11) {
         var lw = this.lineatt.width + JSROOT.gStyle.EndErrorSize;
         endx = "m0," + lw + "v-" + 2*lw + "m0," + lw;
         endy = "m" + lw + ",0h-" + 2*lw + "m" + lw + ",0";
         dend = Math.floor((this.lineatt.width-1)/2);
      }

      var draw_markers = show_errors || show_markers;

      if (draw_markers) use_minmax = true;

      for (i = left; i <= right; ++i) {

         x = this.GetBinX(i);

         if (this.logx && (x <= 0)) continue;

         grx = Math.round(pmain.grx(x));

         lastbin = (i === right);

         if (lastbin && (left<right)) {
            gry = curry;
         } else {
            y = this.histo.getBinContent(i+1);
            gry = Math.round(pmain.gry(y));
         }

         if (res.length === 0) {
            besti = i;
            prevx = startx = currx = grx;
            prevy = curry_min = curry_max = curry = gry;
            res = "M"+currx+","+curry;
         } else
         if (use_minmax) {
            if ((grx === currx) && !lastbin) {
               if (gry < curry_min) besti = i;
               curry_min = Math.min(curry_min, gry);
               curry_max = Math.max(curry_max, gry);
               curry = gry;
            } else {

               if (draw_markers) {
                  bincont = this.histo.getBinContent(besti+1);
                  if (!exclude_zero || (bincont!==0)) {

                     mx1 = Math.round(pmain.grx(this.GetBinX(besti)));
                     mx2 = Math.round(pmain.grx(this.GetBinX(besti+1)));
                     my = Math.round(pmain.gry(bincont));
                     yerr1 = yerr2 = 20;
                     if (show_errors) {
                        binerr = this.histo.getBinError(besti+1);
                        yerr1 = Math.round(my - pmain.gry(bincont + binerr)); // up
                        yerr2 = Math.round(pmain.gry(bincont - binerr) - my); // down
                     }

                     if ((my >= -yerr1) && (my <= height + yerr2)) {
                        if (path_fill !== null)
                           path_fill +="M" + mx1 +","+(my-yerr1) +
                                       "h" + (mx2-mx1) + "v" + (yerr1+yerr2+1) + "h-" + (mx2-mx1) + "z";
                        if (path_err !== null)
                           path_err +="M" + (mx1+dend) +","+ my + endx + "h" + (mx2-mx1-2*dend) + endx +
                                      "M" + Math.round((mx1+mx2)/2) +"," + (my-yerr1+dend) + endy + "v" + (yerr1+yerr2-2*dend) + endy;
                        if (path_marker !== null)
                           path_marker += this.markeratt.create((mx1+mx2)/2, my);
                     }
                  }
               }

               // when several points as same X differs, need complete logic
               if (!draw_markers && ((curry_min !== curry_max) || (prevy !== curry_min))) {

                  if (prevx !== currx)
                     res += "h"+(currx-prevx);

                  if (curry === curry_min) {
                     if (curry_max !== prevy)
                        res += "v" + (curry_max - prevy);
                     if (curry_min !== curry_max)
                        res += "v" + (curry_min - curry_max);
                  } else {
                     if (curry_min !== prevy)
                        res += "v" + (curry_min - prevy);
                     if (curry_max !== curry_min)
                        res += "v" + (curry_max - curry_min);
                     if (curry !== curry_max)
                       res += "v" + (curry - curry_max);
                  }

                  prevx = currx;
                  prevy = curry;
               }

               if (lastbin && (prevx !== grx))
                  res += "h"+(grx-prevx);

               besti = i;
               curry_min = curry_max = curry = gry;
               currx = grx;
            }
         } else
         if ((gry !== curry) || lastbin) {
            if (grx !== currx) res += "h"+(grx-currx);
            if (gry !== curry) res += "v"+(gry-curry);
            curry = gry;
            currx = grx;
         }
      }

      if ((this.fillatt.color !== 'none') && (res.length>0)) {
         var h0 = (height+3);
         if ((this.hmin>=0) && (pmain.gry(0) < height)) h0 = Math.round(pmain.gry(0));
         res += "L"+currx+","+h0 + "L"+startx+","+h0 + "Z";
      }

      if (draw_markers) {

         if ((path_fill !== null) && (path_fill.length > 0))
            this.draw_g.append("svg:path")
                       .attr("d", path_fill)
                       .call(this.fillatt.func);

         if ((path_err !== null) && (path_err.length > 0))
               this.draw_g.append("svg:path")
                   .attr("d", path_err)
                   .call(this.lineatt.func)

         if ((path_marker !== null) && (path_marker.length > 0))
            this.draw_g.append("svg:path")
                .attr("d", path_marker)
                .call(this.markeratt.func);

      } else
      if (res.length > 0) {
         this.draw_g.append("svg:path")
                    .attr("d", res)
                    .style("stroke-linejoin","miter")
                    .call(this.lineatt.func)
                    .call(this.fillatt.func);
      }
   }

   JSROOT.TH1Painter.prototype.GetBinTips = function(bin) {
      var tips = [], name = this.GetTipName(), pmain = this.main_painter();
      if (name.length>0) tips.push(name);

      var x1 = this.GetBinX(bin),
          x2 = this.GetBinX(bin+1),
          cont = this.histo.getBinContent(bin+1);

      if ((this.options.Error > 0) || (this.options.Mark > 0)) {
         tips.push("x = " + pmain.AxisAsText("x", (x1+x2)/2));
         tips.push("y = " + pmain.AxisAsText("y", cont));
         if (this.options.Error > 0) {
            tips.push("error x = " + ((x2 - x1) / 2).toPrecision(4));
            tips.push("error y = " + this.histo.getBinError(bin + 1).toPrecision(4));
         }
      } else {
         tips.push("bin = " + (bin+1));

         if (pmain.x_kind === 'labels')
            tips.push("x = " + pmain.AxisAsText("x", x1));
         else
         if (pmain.x_kind === 'time')
            tips.push("x = " + pmain.AxisAsText("x", (x1+x2)/2));
         else
            tips.push("x = [" + pmain.AxisAsText("x", x1) + ", " + pmain.AxisAsText("x", x2) + ")");

         if (cont === Math.round(cont))
            tips.push("entries = " + cont);
         else
            tips.push("entries = " + JSROOT.FFormat(cont, JSROOT.gStyle.StatFormat));
      }

      return tips;
   }

   JSROOT.TH1Painter.prototype.ProcessTooltip = function(pnt) {
      if ((pnt === null) || !this.draw_content || (this.options.Lego > 0)) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         this.ProvideUserTooltip(null);
         return null;
      }

      var width = this.frame_width(),
          height = this.frame_height(),
          pmain = this.main_painter(),
          pad = this.root_pad(),
          painter = this,
          findbin = null, show_rect = true,
          grx1, midx, grx2, gry1, midy, gry2,
          left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 2),
          l = left, r = right;

      function GetBinGrX(i) {
         var x1 = painter.GetBinX(i);
         if ((x1<0) && pmain.logx) return null;
         return pmain.grx(x1);
      }

      function GetBinGrY(i) {
         var y = painter.histo.getBinContent(i + 1);
         if (pmain.logy && (y < painter.scale_ymin))
            return pmain.swap_xy ? -1000 : 10*height;
         return Math.round(pmain.gry(y));
      }

      while (l < r-1) {
         var m = Math.round((l+r)*0.5);

         var xx = GetBinGrX(m);
         if (xx === null) { l = m; continue; }

         if (xx < pnt.x - 0.5) l = m; else
            if (xx > pnt.x + 0.5) r = m; else { l++; r--; }
      }

      findbin = r = l;
      grx1 = GetBinGrX(findbin);

      while ((l>left) && (GetBinGrX(l-1) > grx1 - 1.0)) --l;
      while ((r<right) && (GetBinGrX(r+1) < grx1 + 1.0)) ++r;

      if (l < r) {
         // many points can be assigned with the same cursor position
         // first try point around mouse y
         var best = height;
         for (var m=l;m<=r;m++) {
            var dist = Math.abs(GetBinGrY(m) - pnt.y);
            if (dist < best) { best = dist; findbin = m; }
         }

         // if best distance still too far from mouse position, just take from between
         if (best > height/10)
            findbin = Math.round(l + (r-l) / height * pnt.y);

         grx1 = GetBinGrX(findbin);
      }

      grx1 = Math.round(grx1);
      grx2 = Math.round(GetBinGrX(findbin+1));

      if (this.options.Bar > 0) {
         var w = grx2 - grx1;
         grx1 += Math.round(this.histo.fBarOffset/1000*w);
         grx2 = grx1 + Math.round(this.histo.fBarWidth/1000*w);
      }

      midx = Math.round((grx1+grx2)/2);

      midy = gry1 = gry2 = GetBinGrY(findbin);

      if ((this.options.Error > 0) || (this.options.Mark > 0))  {

         show_rect = true;

         var msize = 3;
         if (this.markeratt) msize = Math.max(msize, 2+Math.round(this.markeratt.size * 4));

         // show at least 6 pixels as tooltip rect
         if (grx2 - grx1 < 2*msize) { grx1 = midx-msize; grx2 = midx+msize; }

         if (this.options.Error > 0) {
            var cont = this.histo.getBinContent(findbin+1);
            var binerr = this.histo.getBinError(findbin+1);

            gry1 = Math.round(pmain.gry(cont + binerr)); // up
            gry2 = Math.round(pmain.gry(cont - binerr)); // down

            if ((cont==0) && this.IsTProfile()) findbin = null;
         }

         gry1 = Math.min(gry1, midy - msize);
         gry2 = Math.max(gry2, midy + msize);

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt.y<gry1) || (pnt.y>gry2)) findbin = null;

      } else {

         // if histogram alone, use old-style with rects
         // if there are too many points at pixel, use circle
         show_rect = (pnt.nproc === 1) && (right-left < width);

         if (show_rect) {
            // for mouse events mouse pointer should be under the curve
            if ((pnt.y < gry1) && !pnt.touch) findbin = null;

            gry2 = height;

            if ((this.fillatt.color !== 'none') && (this.hmin>=0)) {
               gry2 = Math.round(pmain.gry(0));
               if ((gry2 > height) || (gry2 <= gry1)) gry2 = height;
            }
         }
      }

      if (findbin!==null) {
         // if bin on boundary found, check that x position is ok
         if ((findbin === left) && (grx1 > pnt.x + 2))  findbin = null; else
         if ((findbin === right-1) && (grx2 < pnt.x - 2)) findbin = null; else
         // if bars option used check that bar is not match
         if ((pnt.x < grx1 - 2) || (pnt.x > grx2 + 2)) findbin = null;
      }


      var ttrect = this.draw_g.select(".tooltip_bin");

      if ((findbin === null) || ((gry2 <= 0) || (gry1 >= height))) {
         ttrect.remove();
         this.ProvideUserTooltip(null);
         return null;
      }

      var res = { x: midx, y: midy,
                  color1: this.lineatt.color, color2: this.fillatt.color,
                  lines: this.GetBinTips(findbin) };

      if (show_rect) {

         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:rect")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         res.changed = ttrect.property("current_bin") !== findbin;

         if (res.changed)
            ttrect.attr("x", grx1)
                  .attr("width", grx2-grx1)
                  .attr("y", gry1)
                  .attr("height", gry2-gry1)
                  .style("opacity", "0.3")
                  .property("current_bin", findbin);

         res.exact = (Math.abs(midy - pnt.y) <= 5) || ((pnt.y>=gry1) && (pnt.y<=gry2));

         res.menu = true; // one could show context menu
         // distance to middle point, use to decide which menu to activate
         res.menu_dist = Math.sqrt((midx-pnt.x)*(midx-pnt.x) + (midy-pnt.y)*(midy-pnt.y));

      } else {
         var radius = this.lineatt.width + 3;

         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:circle")
                                .attr("class","tooltip_bin")
                                .style("pointer-events","none")
                                .attr("r", radius)
                                .call(this.lineatt.func)
                                .call(this.fillatt.func);

         res.exact = (Math.abs(midx - pnt.x) <= radius) && (Math.abs(midy - pnt.y) <= radius);

         res.menu = res.exact; // show menu only when mouse pointer exactly over the histogram
         res.menu_dist = Math.sqrt((midx-pnt.x)*(midx-pnt.x) + (midy-pnt.y)*(midy-pnt.y));

         res.changed = ttrect.property("current_bin") !== findbin;

         if (res.changed)
            ttrect.attr("cx", midx)
                  .attr("cy", midy)
                  .property("current_bin", findbin);
      }

      if (this.IsUserTooltipCallback() && res.changed) {
         this.ProvideUserTooltip({ obj: this.histo,  name: this.histo.fName,
                                   bin: findbin, cont: this.histo.getBinContent(findbin+1),
                                   grx: midx, gry: midy });
      }

      return res;
   }


   JSROOT.TH1Painter.prototype.FillHistContextMenu = function(menu) {

      menu.add("Auto zoom-in", this.AutoZoom);

      menu.addDrawMenu("Draw with", ["hist", "p", "e", "e1", "pe2", "lego"], function(arg) {
         this.options = this.DecodeOptions(arg);

         // redraw all objects
         this.RedrawPad();

         // if (this.options.Lego == 0) this.AddInteractive();
      });
   }

   JSROOT.TH1Painter.prototype.AutoZoom = function() {
      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          dist = right - left;

      if (dist == 0) return;

      // first find minimum
      var min = this.histo.getBinContent(left + 1);
      for (var indx = left; indx < right; ++indx)
         if (this.histo.getBinContent(indx + 1) < min)
            min = this.histo.getBinContent(indx + 1);
      if (min>0) return; // if all points positive, no chance for autoscale

      while ((left < right) && (this.histo.getBinContent(left + 1) <= min)) ++left;
      while ((left < right) && (this.histo.getBinContent(right) <= min)) --right;

      if ((right - left < dist) && (left < right))
         this.Zoom(this.GetBinX(left), this.GetBinX(right));
   }

   JSROOT.TH1Painter.prototype.CanZoomIn = function(axis,min,max) {
      if ((axis=="x") && (this.GetIndexX(max,0.5) - this.GetIndexX(min,0) > 1)) return true;

      if ((axis=="y") && (Math.abs(max-min) > Math.abs(this.ymax-this.ymin)*1e-6)) return true;

      // check if it makes sense to zoom inside specified axis range
      return false;
   }

   JSROOT.TH1Painter.prototype.Draw2D = function(call_back) {
      if (typeof this.Create3DScene === 'function')
         this.Create3DScene(-1);
      if (typeof this.DrawColorPalette === 'function')
         this.DrawColorPalette(false);

      this.DrawAxes();
      this.DrawGrids();
      this.DrawBins();
      this.DrawTitle();
      this.AddInteractive();
      JSROOT.CallBack(call_back);
   }

   JSROOT.TH1Painter.prototype.Draw3D = function(call_back) {
      JSROOT.AssertPrerequisites('more2d;3d', function() {
         this.Create3DScene = JSROOT.Painter.HPainter_Create3DScene;
         this.Draw3D = JSROOT.Painter.TH1Painter_Draw3D;
         this.Draw3D(call_back);
      }.bind(this));
   }

   JSROOT.THistPainter.prototype.Get3DToolTip = function(indx) {
      var tip = { bin: indx }, arr;
      switch (this.Dimension()) {
         case 1:
            tip.ix = indx; tip.iy = 1;
            tip.value = this.histo.getBinContent(tip.ix);
            arr = this.GetBinTips(indx-1);
            break;
         case 2: {
            tip.ix = indx % (this.nbinsx + 2);
            tip.iy = (indx - tip.ix) / (this.nbinsx + 2);
            tip.value = this.histo.getBinContent(tip.ix, tip.iy);
            arr = this.GetBinTips(tip.ix-1, tip.iy-1);
            break;
         }
         case 3: {
            tip.ix = indx % (this.nbinsx+2);
            tip.iy = ((indx - tip.ix) / (this.nbinsx+2)) % (this.nbinsy+2);
            tip.iz = (indx - tip.ix - tip.iy * (this.nbinsx+2)) / (this.nbinsx+2) / (this.nbinsy+2);
            tip.value = this.GetObject().getBinContent(tip.ix, tip.iy, tip.iz);
            arr = this.GetBinTips(tip.ix-1, tip.iy-1, tip.iz-1);
            break;
         }
      }

      if (arr) {
         tip.info = arr[0];
         for (var n=1;n<arr.length;++n) tip.info +="<br/>"+arr[n];
      }
      return tip;
   }


   JSROOT.TH1Painter.prototype.Redraw = function(resize) {
      this.CreateXY();

      var func_name = (this.options.Lego > 0) ? "Draw3D" : "Draw2D";

      this[func_name](null, resize);
   }

   JSROOT.Painter.drawHistogram1D = function(divid, histo, opt) {
      // create painter and add it to canvas
      var painter = new JSROOT.TH1Painter(histo);

      painter.SetDivId(divid, 1);

      // here we deciding how histogram will look like and how will be shown
      painter.options = painter.DecodeOptions(opt);

      painter.CheckPadRange();

      painter.ScanContent();

      painter.CreateXY();

      if (JSROOT.gStyle.AutoStat && painter.create_canvas)
         painter.CreateStat();

      var func_name = (painter.options.Lego > 0) ? "Draw3D" : "Draw2D";

      painter[func_name](function() {
         painter.DrawNextFunction(0, function() {

            if (painter.options.Lego === 0) {
               // painter.AddInteractive();
               if (painter.options.AutoZoom) painter.AutoZoom();
            }

            painter.FillToolbar();
            painter.DrawingReady();
         });
      });

      return painter;
   }

   // =====================================================================================

   JSROOT.Painter.drawText = function(divid, text) {
      var painter = new JSROOT.TObjectPainter(text);
      painter.SetDivId(divid, 2);

      painter.Redraw = function() {
         var text = this.GetObject(),
             w = this.pad_width(), h = this.pad_height(),
             pos_x = text.fX, pos_y = text.fY,
             tcolor = JSROOT.Painter.root_colors[text.fTextColor],
             use_pad = true, latex_kind = 0, fact = 1.;

         if (text.TestBit(JSROOT.BIT(14))) {
            // NDC coordiantes
            pos_x = pos_x * w;
            pos_y = (1 - pos_y) * h;
         } else
         if (this.main_painter() !== null) {
            w = this.frame_width(); h = this.frame_height(); use_pad = false;
            pos_x = this.main_painter().grx(pos_x);
            pos_y = this.main_painter().gry(pos_y);
         } else
         if (this.root_pad() !== null) {
            pos_x = this.ConvertToNDC("x", pos_x) * w;
            pos_y = (1 - this.ConvertToNDC("y", pos_y)) * h;
         } else {
            text.fTextAlign = 22;
            pos_x = w/2;
            pos_y = h/2;
            if (text.fTextSize === 0) text.fTextSize = 0.05;
            if (text.fTextColor === 0) text.fTextColor = 1;
         }

         this.RecreateDrawG(use_pad, use_pad ? "text_layer" : "upper_layer");

         if (text._typename == 'TLatex') { latex_kind = 1; fact = 0.9; } else
         if (text._typename == 'TMathText') { latex_kind = 2; fact = 0.8; }

         this.StartTextDrawing(text.fTextFont, Math.round(text.fTextSize*Math.min(w,h)*fact));

         this.DrawText(text.fTextAlign, Math.round(pos_x), Math.round(pos_y), 0, 0, text.fTitle, tcolor, latex_kind);

         this.FinishTextDrawing();
      }

      painter.Redraw();
      return painter.DrawingReady();
   }

   // ================= painter of raw text ========================================

   JSROOT.RawTextPainter = function(txt) {
      JSROOT.TBasePainter.call(this);
      this.txt = txt;
      return this;
   }

   JSROOT.RawTextPainter.prototype = Object.create( JSROOT.TBasePainter.prototype );

   JSROOT.RawTextPainter.prototype.RedrawObject = function(obj) {
      this.txt = obj;
      this.Draw();
      return true;
   }

   JSROOT.RawTextPainter.prototype.Draw = function() {
      var txt = this.txt.value;
      if (txt==null) txt = "<undefined>";

      var mathjax = 'mathjax' in this.txt;

      if (!mathjax && !('as_is' in this.txt)) {
         var arr = txt.split("\n"); txt = "";
         for (var i = 0; i < arr.length; ++i)
            txt += "<pre>" + arr[i] + "</pre>";
      }

      var frame = this.select_main();
      var main = frame.select("div");
      if (main.empty())
         main = frame.append("div")
                     .style('max-width','100%')
                     .style('max-height','100%')
                     .style('overflow','auto');

      main.html(txt);

      // (re) set painter to first child element
      this.SetDivId(this.divid);

      if (mathjax) {
         if (this['loading_mathjax']) return;
         this['loading_mathjax'] = true;
         var painter = this;
         JSROOT.AssertPrerequisites('mathjax', function() {
            painter['loading_mathjax'] = false;
            if (typeof MathJax == 'object') {
               MathJax.Hub.Queue(["Typeset", MathJax.Hub, frame.node()]);
            }
         });
      }
   }

   JSROOT.Painter.drawRawText = function(divid, txt, opt) {
      var painter = new JSROOT.RawTextPainter(txt);
      painter.SetDivId(divid);
      painter.Draw();
      return painter.DrawingReady();
   }

   // ===================== hierarchy scanning functions ==================================

   JSROOT.Painter.FolderHierarchy = function(item, obj) {

      if ((obj==null) || !('fFolders' in obj) || (obj.fFolders==null)) return false;

      if (obj.fFolders.arr.length===0) { item._more = false; return true; }

      item._childs = [];

      for ( var i = 0; i < obj.fFolders.arr.length; ++i) {
         var chld = obj.fFolders.arr[i];
         item._childs.push( {
            _name : chld.fName,
            _kind : "ROOT." + chld._typename,
            _obj : chld
         });
      }
      return true;
   }

   JSROOT.Painter.TaskHierarchy = function(item, obj) {
      // function can be used for different derived classes
      // we show not only child tasks, but all complex data members

      if ((obj==null) || !('fTasks' in obj) || (obj.fTasks==null)) return false;

      JSROOT.Painter.ObjectHierarchy(item, obj, { exclude: ['fTasks', 'fName'] } );

      if ((obj.fTasks.arr.length===0) && (item._childs.length==0)) { item._more = false; return true; }

      // item._childs = [];

      for ( var i = 0; i < obj.fTasks.arr.length; ++i) {
         var chld = obj.fTasks.arr[i];
         item._childs.push( {
            _name : chld.fName,
            _kind : "ROOT." + chld._typename,
            _obj : chld
         });
      }
      return true;
   }


   JSROOT.Painter.ListHierarchy = function(folder, lst) {
      if (lst._typename != 'TList' && lst._typename != 'TObjArray' && lst._typename != 'TClonesArray') return false;

      if ((lst.arr === undefined) || (lst.arr.length === 0)) {
         folder._more = false;
         return true;
      }

      folder._childs = [];
      for ( var i = 0; i < lst.arr.length; ++i) {
         var obj = lst.arr[i];

         var item;

         if (obj===null) {
            item = {
               _name: i.toString(),
               _kind: "ROOT.NULL",
               _title: "NULL",
               _obj: null
            }
         } else {
           item = {
             _name : obj.fName,
             _kind : "ROOT." + obj._typename,
             _title : obj.fTitle,
             _obj : obj
           };
           // if name is integer value, it should match array index
           if ((item._name === undefined) ||
               (!isNaN(parseInt(item._name)) && (parseInt(item._name)!==i))
               || (lst.arr.indexOf(obj)<i))
                 item._name = i.toString();

           if (item._title === undefined)
              item._title = obj._typename ? item._kind : item._name;
         }

         folder._childs.push(item);
      }
      return true;
   }

   JSROOT.Painter.TreeHierarchy = function(node, obj) {
      if (obj._typename != 'TTree' && obj._typename != 'TNtuple') return false;

      node._childs = [];

      for ( var i = 0; i < obj.fBranches.arr.length; ++i) {
         var branch = obj.fBranches.arr[i];
         var nb_leaves = branch.fLeaves.arr.length;

         // display branch with only leaf as leaf
         if (nb_leaves == 1 && branch.fLeaves.arr[0].fName == branch.fName) nb_leaves = 0;

         var subitem = {
            _name : branch.fName,
            _kind : nb_leaves > 0 ? "ROOT.TBranch" : "ROOT.TLeafF"
         }

         node._childs.push(subitem);

         if (nb_leaves > 0) {
            subitem._childs = [];
            for (var j = 0; j < nb_leaves; ++j) {
               var leafitem = {
                  _name : branch.fLeaves.arr[j].fName,
                  _kind : "ROOT.TLeafF"
               }
               subitem._childs.push(leafitem);
            }
         }
      }

      return true;
   }

   JSROOT.Painter.KeysHierarchy = function(folder, keys, file, dirname) {

      if (keys === undefined) return false;

      folder._childs = [];

      for (var i = 0; i < keys.length; ++i) {
         var key = keys[i];

         var item = {
            _name : key.fName + ";" + key.fCycle,
            _cycle : key.fCycle,
            _kind : "ROOT." + key.fClassName,
            _title : key.fTitle,
            _keyname : key.fName,
            _readobj : null,
            _parent : folder
         };

         if ('fRealName' in key)
            item._realname = key.fRealName + ";" + key.fCycle;

         if (key.fClassName == 'TDirectory' || key.fClassName == 'TDirectoryFile') {
            var dir = null;
            if ((dirname!=null) && (file!=null)) dir = file.GetDir(dirname + key.fName);
            if (dir == null) {
               item._more = true;
               item._expand = function(node, obj) {
                  // one can get expand call from child objects - ignore them
                  return JSROOT.Painter.KeysHierarchy(node, obj.fKeys);
               }
            } else {
               // remove cycle number - we have already directory
               item._name = key.fName;
               JSROOT.Painter.KeysHierarchy(item, dir.fKeys, file, dirname + key.fName + "/");
            }
         } else
         if ((key.fClassName == 'TList') && (key.fName == 'StreamerInfo')) {
            item._name = 'StreamerInfo';
            item._kind = "ROOT.TStreamerInfoList";
            item._title = "List of streamer infos for binary I/O";
            item._readobj = file.fStreamerInfos;
         }

         folder._childs.push(item);
      }

      return true;
   }

   JSROOT.Painter.ObjectHierarchy = function(top, obj, args) {
      if ((top==null) || (obj==null)) return false;

      // check nosimple property in all parents
      var nosimple = true,  prnt = top;
      while (prnt) {
         if ('_nosimple' in prnt) { nosimple = prnt._nosimple; break; }
         prnt = prnt._parent;
      }

      top._childs = [];
      if (!('_obj' in top))
         top._obj = obj;
      else
      if (top._obj !== obj) alert('object missmatch');

      if (!('_title' in top) && ('_typename' in obj))
         top._title = "ROOT." + obj._typename;

      for (var key in obj) {
         if (key == '_typename') continue;
         var fld = obj[key];
         if (typeof fld == 'function') continue;
         if (args && args.exclude && (args.exclude.indexOf(key)>=0)) continue;

         var item = {
            _parent : top,
            _name : key
         };

         if (fld === null) {
            item._value = item._title = "null";
            if (!nosimple) top._childs.push(item);
            continue;
         }

         var proto = Object.prototype.toString.apply(fld);
         var simple  = false;

         if ((proto.lastIndexOf('Array]') == proto.length-6) && (proto.indexOf('[object')==0)) {
            item._title = item._kind + " len=" + fld.length;
            simple = (proto != '[object Array]');
            if (fld.length === 0) {
               item._value = "[ ]";
               item._more = false; // hpainter will not try to expand again
            } else {
               item._value = "[...]";
               item._more = true;
               item._expand = JSROOT.Painter.ObjectHierarchy;
               item._obj = fld;
            }
         } else
         if (typeof fld == 'object') {
            if ('_typename' in fld)
               item._kind = item._title = "ROOT." + fld._typename;

            // check if object already shown in hierarchy (circular dependency)
            var curr = top, inparent = false;
            while (curr && !inparent) {
               inparent = (curr._obj === fld);
               curr = curr._parent;
            }

            if (inparent) {
               item._value = "{ prnt }";
               simple = true;
            } else {
               item._obj = fld;
               item._value = "{ }";
               if (fld._typename == 'TColor') item._value = JSROOT.Painter.MakeColorRGB(fld);
            }
         } else
         if ((typeof fld == 'number') || (typeof fld == 'boolean')) {
            simple = true;
            if (key == 'fBits')
               item._value = "0x" + fld.toString(16);
            else
               item._value = fld.toString();
            item._vclass = 'h_value_num';
         } else
         if (typeof fld == 'string') {
            simple = true;
            item._value = '"' + fld + '"';
            item._vclass = 'h_value_str';
         } else {
            simple = true;
            alert('miss ' + key + '  ' + typeof fld);
         }

         if (!simple || !nosimple)
            top._childs.push(item);
      }
      return true;
   }

   // =========== painter of hierarchical structures =================================

   JSROOT.hpainter = null; // global pointer

   JSROOT.HierarchyPainter = function(name, frameid, backgr) {
      JSROOT.TBasePainter.call(this);
      this.name = name;
      this.h = null; // hierarchy
      this.with_icons = true;
      this.background = backgr;
      this.files_monitoring = (frameid == null); // by default files monitored when nobrowser option specified
      this.nobrowser = frameid === null;
      if (!this.nobrowser) this.SetDivId(frameid);

      // remember only very first instance
      if (JSROOT.hpainter == null)
         JSROOT.hpainter = this;
   }

   JSROOT.HierarchyPainter.prototype = Object.create(JSROOT.TBasePainter.prototype);

   JSROOT.HierarchyPainter.prototype.ToggleFloatBrowser = function(force_mode) {
      if (!this.nobrowser || !this.disp) return;

      var elem = d3.select("#"+this.disp.frameid);
      if (elem.empty()) return;

      var container = d3.select(elem.node().parentNode);

      var main = container.select('.float_browser');

      if (main.empty()) {
         if (force_mode === false) return;
         var div = container.append("div").attr("class","jsroot");
         main = div.append("div").attr("class","float_browser").style('left', '-320px');
         main.transition().delay(700).style('left', '5px');
         this.SetDivId(main.node());
         this.RefreshHtml();
      } else {
         if (main.style('left') == '5px') {
            if (force_mode !== true)
               main.transition().delay(700).style('left', '-320px');
         } else {
            if (force_mode !== false)
               main.transition().delay(700).style('left', '5px');
         }
      }
   }

   JSROOT.HierarchyPainter.prototype.Cleanup = function() {
      // clear drawing and browser
      this.clear(true);
   }

   JSROOT.HierarchyPainter.prototype.FileHierarchy = function(file) {
      var painter = this;

      var folder = {
         _name : file.fFileName,
         _kind : "ROOT.TFile",
         _file : file,
         _fullurl : file.fFullURL,
         _had_direct_read : false,
         // this is central get method, item or itemname can be used
         _get : function(item, itemname, callback) {

            var fff = this; // file item

            if (item && item._readobj)
               return JSROOT.CallBack(callback, item, item._readobj);

            if (item!=null) itemname = painter.itemFullName(item, fff);

            function ReadFileObject(file) {
               if (fff._file==null) fff._file = file;

               if (file == null) return JSROOT.CallBack(callback, item, null);

               file.ReadObject(itemname, function(obj) {

                  // if object was read even when item didnot exist try to reconstruct new hierarchy
                  if ((item==null) && (obj!=null)) {
                     // first try to found last read directory
                     var d = painter.Find({name:itemname, top:fff, last_exists:true, check_keys:true });
                     if ((d!=null) && ('last' in d) && (d.last!=fff)) {
                        // reconstruct only subdir hierarchy
                        var dir = file.GetDir(painter.itemFullName(d.last, fff));
                        if (dir) {
                           d.last._name = d.last._keyname;
                           var dirname = painter.itemFullName(d.last, fff);
                           JSROOT.Painter.KeysHierarchy(d.last, dir.fKeys, file, dirname + "/");
                        }
                     } else {
                        // reconstruct full file hierarchy
                        JSROOT.Painter.KeysHierarchy(fff, file.fKeys, file, "");
                     }
                     item = painter.Find({name:itemname, top: fff});
                  }

                  if (item!=null) {
                     item._readobj = obj;
                     // remove cycle number for objects supporting expand
                     if ('_expand' in item) item._name = item._keyname;
                  }

                  JSROOT.CallBack(callback, item, obj);
               });
            }

            if (fff._file != null) {
               ReadFileObject(fff._file);
            } else {
               // try to reopen ROOT file
               new JSROOT.TFile(fff._fullurl, ReadFileObject);
            }
         }
      };

      JSROOT.Painter.KeysHierarchy(folder, file.fKeys, file, "");

      return folder;
   }

   JSROOT.HierarchyPainter.prototype.ForEach = function(callback, top) {

      if (top==null) top = this.h;
      if ((top==null) || (typeof callback != 'function')) return;
      function each_item(item) {
         callback(item);
         if ('_childs' in item)
            for (var n = 0; n < item._childs.length; ++n) {
               item._childs[n]._parent = item;
               each_item(item._childs[n]);
            }
      }

      each_item(top);
   }

   JSROOT.HierarchyPainter.prototype.Find = function(arg) {
      // search item in the hierarchy
      // One could specify simply item name or object with following arguments
      //   name:  item to search
      //   force: specified elements will be created when not exists
      //   last_exists: when specified last parent element will be returned
      //   check_keys: check TFile keys with cycle suffix
      //   top:   element to start search from

      function find_in_hierarchy(top, fullname) {

         if (!fullname || (fullname.length == 0) || (top==null)) return top;

         var pos = -1;

         function process_child(child) {
            // set parent pointer when searching child
            child._parent = top;
            if ((pos + 1 == fullname.length) || (pos < 0)) return child;

            return find_in_hierarchy(child, fullname.substr(pos + 1));
         }

         do {
            // we try to find element with slashes inside
            pos = fullname.indexOf("/", pos + 1);

            var localname = (pos < 0) ? fullname : fullname.substr(0, pos);

            if (top._childs) {
               // first try to find direct matched item
               for (var i = 0; i < top._childs.length; ++i)
                  if (top._childs[i]._name == localname)
                     return process_child(top._childs[i]);

               // if allowed, try to found item with key
               if (arg.check_keys)
                  for (var i = 0; i < top._childs.length; ++i) {
                    if ((typeof top._childs[i]._keyname === 'string') &&
                         (top._childs[i]._keyname === localname))
                           return process_child(top._childs[i]);
                  }

               var allow_index = arg.allow_index;
               if ((localname[0] === '[') && (localname[localname.length-1] === ']') &&
                   !isNaN(parseInt(localname.substr(1,localname.length-2)))) {
                  allow_index = true;
                  localname = localname.substr(1,localname.length-2);
               }

               // when search for the elements it could be allowed to check index
               if (allow_index) {
                  var indx = parseInt(localname);
                  if (!isNaN(indx) && (indx>=0) && (indx<top._childs.length))
                     return process_child(top._childs[indx]);
               }
            }

            if (arg.force) {
               // if didnot found element with given name we just generate it
               if (top._childs === undefined) top._childs = [];
               var child = { _name: localname };
               top._childs.push(child);
               return process_child(child);
            }

         } while (pos > 0);

         return (arg.last_exists && top) ? { last: top, rest: fullname } : null;
      }

      var top = this.h;
      var itemname = "";

      if (typeof arg == 'string') { itemname = arg; arg = {}; } else
      if (typeof arg == 'object') { itemname = arg.name; if ('top' in arg) top = arg.top; } else
         return null;

      return find_in_hierarchy(top, itemname);
   }

   JSROOT.HierarchyPainter.prototype.itemFullName = function(node, uptoparent) {
      var res = "";

      while (node && ('_parent' in node)) {
         if (res.length > 0) res = "/" + res;
         res = node._name + res;
         node = node._parent;
         if (uptoparent && (node === uptoparent)) break;
      }

      return res;
   }

   JSROOT.HierarchyPainter.prototype.ExecuteCommand = function(itemname, callback) {
      // execute item marked as 'Command'
      // If command requires additional arguments, they could be specified as extra arguments
      // Or they will be requested interactive

      var hitem = this.Find(itemname);
      var url = itemname + "/cmd.json";
      var pthis = this;
      var d3node = d3.select((typeof callback == 'function') ? null : callback);

      if ('_numargs' in hitem)
         for (var n = 0; n < hitem._numargs; ++n) {
            var argname = "arg" + (n+1);
            var argvalue = null;
            if (n+2<arguments.length) argvalue = arguments[n+2];
            if ((argvalue==null) && (typeof callback == 'object'))
               argvalue = prompt("Input argument " + argname + " for command " + hitem._name,"");
            if (argvalue==null) return;
            url += ((n==0) ? "?" : "&") + argname + "=" + argvalue;
         }

      if (!d3node.empty()) {
         d3node.style('background','yellow');
         if (hitem && hitem._title) d3node.attr('title', "Executing " + hitem._title);
      }

      JSROOT.NewHttpRequest(url, 'text', function(res) {
         if (typeof callback == 'function') return callback(res);
         if (d3node.empty()) return;
         var col = ((res!=null) && (res!='false')) ? 'green' : 'red';
         if (hitem && hitem._title) d3node.attr('title', hitem._title + " lastres=" + res);
         d3node.style('background', col).transition().duration(2000).each("end", function() { d3node.style('background', ''); });
         if ((col == 'green') && ('_hreload' in hitem)) pthis.reload();
         if ((col == 'green') && ('_update_item' in hitem)) pthis.updateItems(hitem._update_item.split(";"));
      }).send();
   }

   JSROOT.HierarchyPainter.prototype.RefreshHtml = function(callback) {
      if (this.divid == null) return JSROOT.CallBack(callback);
      var hpainter = this;
      JSROOT.AssertPrerequisites('jq2d', function() {
          hpainter.RefreshHtml(callback);
      });
   }

   JSROOT.HierarchyPainter.prototype.toggle = function(status, h) {
      var hitem = (h==null) ? this.h : h;

      if (!('_childs' in hitem)) {
         if (!status || this.with_icons || ((typeof hitem._expand) !== 'function')) return;
         this.expand(this.itemFullName(hitem));
         if ('_childs' in hitem) hitem._isopen = true;
         return;
      }

      if (hitem != this.h)
         if (status)
            hitem._isopen = true;
         else
            delete hitem._isopen;

      for (var i=0; i < hitem._childs.length; ++i)
         this.toggle(status, hitem._childs[i]);

      if (h==null) this.RefreshHtml();
   }

   JSROOT.HierarchyPainter.prototype.get = function(arg, call_back, options) {
      // get object item with specified name
      // depending from provided option, same item can generate different object types

      var itemname = (typeof arg == 'object') ? arg.arg : arg;

      var item = this.Find( { name: itemname, allow_index: true } );

      // if item not found, try to find nearest parent which could allow us to get inside
      var d = (item!=null) ? null : this.Find({ name: itemname, last_exists: true, check_keys: true, allow_index: true });

      // if item not found, try to expand hierarchy central function
      // implements not process get in central method of hierarchy item (if exists)
      // if last_parent found, try to expand it
      if ((d !== null) && ('last' in d) && (d.last !== null)) {
         var hpainter = this;
         var parentname = this.itemFullName(d.last);

         // this is indication that expand does not give us better path to searched item
         if ((typeof arg == 'object') && ('rest' in arg))
            if ((arg.rest == d.rest) || (arg.rest.length <= d.rest.length))
               return JSROOT.CallBack(call_back);

         return this.expand(parentname, function(res) {
            if (!res) JSROOT.CallBack(call_back);
            var newparentname = hpainter.itemFullName(d.last);
            if (newparentname.length>0) newparentname+="/";
            hpainter.get( { arg: newparentname + d.rest, rest: d.rest }, call_back, options);
         });
      }

      if ((item !== null) && (typeof item._obj == 'object'))
         return JSROOT.CallBack(call_back, item, item._obj);

      // normally search _get method in the parent items
      var curr = item;
      while (curr != null) {
         if (('_get' in curr) && (typeof curr._get == 'function'))
            return curr._get(item, null, call_back, options);
         curr = ('_parent' in curr) ? curr._parent : null;
      }

      JSROOT.CallBack(call_back, item, null);
   }

   JSROOT.HierarchyPainter.prototype.draw = function(divid, obj, drawopt) {
      // just envelope, one should be able to redefine it for sub-classes
      return JSROOT.draw(divid, obj, drawopt);
   }

   JSROOT.HierarchyPainter.prototype.redraw = function(divid, obj, drawopt) {
      // just envelope, one should be able to redefine it for sub-classes
      return JSROOT.redraw(divid, obj, drawopt);
   }

   JSROOT.HierarchyPainter.prototype.player = function(itemname, option, call_back) {
      var item = this.Find(itemname);

      if (!item || !('_player' in item)) return JSROOT.CallBack(call_back, null);

      var hpainter = this;

      var prereq = ('_prereq' in item) ? item['_prereq'] : '';

      JSROOT.AssertPrerequisites(prereq, function() {

         var player_func = JSROOT.findFunction(item._player);
         if (player_func == null) return JSROOT.CallBack(call_back, null);

         hpainter.CreateDisplay(function(mdi) {
            var res = null;
            if (mdi) res = player_func(hpainter, itemname, option);
            JSROOT.CallBack(call_back, res);
         });
      });
   }

   JSROOT.HierarchyPainter.prototype.canDisplay = function(item, drawopt) {
      if (item == null) return false;
      if ('_player' in item) return true;
      if (drawopt == 'inspect') return true;
      var handle = JSROOT.getDrawHandle(item._kind, drawopt);
      return (handle!=null) && ('func' in handle);
   }

   JSROOT.HierarchyPainter.prototype.isItemDisplayed = function(itemname) {
      var mdi = this.GetDisplay();
      if (!mdi) return false;

      return mdi.FindFrame(itemname) !== null;

   }

   JSROOT.HierarchyPainter.prototype.display = function(itemname, drawopt, call_back) {
      var h = this,
          painter = null,
          updating = false,
          frame_name = itemname,
          marker = "::_display_on_frame_::",
          p = drawopt ? drawopt.indexOf(marker) : -1;

      if (p>=0) {
         frame_name = drawopt.substr(p + marker.length);
         drawopt = drawopt.substr(0, p);
      }

      function display_callback() {
         if (painter) painter.SetItemName(itemname, updating ? null : drawopt); // mark painter as created from hierarchy
         JSROOT.CallBack(call_back, painter, itemname);
      }

      h.CreateDisplay(function(mdi) {

         if (!mdi) return display_callback();

         var item = h.Find(itemname);

         if ((item!=null) && ('_player' in item))
            return h.player(itemname, drawopt, display_callback);

         updating = (typeof(drawopt)=='string') && (drawopt.indexOf("update:")==0);

         if (updating) {
            drawopt = drawopt.substr(7);
            if ((item==null) || ('_doing_update' in item)) return display_callback();
            item._doing_update = true;
         }

         if (item!=null) {
            if (!h.canDisplay(item, drawopt)) return display_callback();
         }

         var divid = "";
         if ((typeof(drawopt)=='string') && (drawopt.indexOf("divid:")>=0)) {
            var pos = drawopt.indexOf("divid:");
            divid = drawopt.slice(pos+6);
            drawopt = drawopt.slice(0, pos);
         }

         JSROOT.progress("Loading " + itemname);

         h.get(itemname, function(item, obj) {

            JSROOT.progress();

            if (updating && item) delete item['_doing_update'];
            if (obj==null) return display_callback();

            JSROOT.progress("Drawing " + itemname);

            if (divid.length > 0) {
               painter = updating ? h.redraw(divid, obj, drawopt) : h.draw(divid, obj, drawopt);
            } else {
               mdi.ForEachPainter(function(p, frame) {
                  if (p.GetItemName() != itemname) return;
                  // verify that object was drawn with same option as specified now (if any)
                  if (!updating && (drawopt!=null) && (p.GetItemDrawOpt()!=drawopt)) return;
                  painter = p;
                  mdi.ActivateFrame(frame);
                  painter.RedrawObject(obj);
               });
            }

            if (painter==null) {
               if (updating) {
                  JSROOT.console("something went wrong - did not found painter when doing update of " + itemname);
               } else {
                  var frame = mdi.FindFrame(frame_name, true);
                  d3.select(frame).html("");
                  mdi.ActivateFrame(frame);
                  painter = h.draw(d3.select(frame).attr("id"), obj, drawopt);
                  item._painter = painter;
                  if (JSROOT.gStyle.DragAndDrop)
                     h.enable_dropping(frame, itemname);
               }
            }

            JSROOT.progress();

            if (painter === null) return display_callback();

            painter.WhenReady(display_callback);

         }, drawopt);
      });
   }

   JSROOT.HierarchyPainter.prototype.enable_dragging = function(element, itemname) {
      // here is not defined - implemented with jquery
   }

   JSROOT.HierarchyPainter.prototype.enable_dropping = function(frame, itemname) {
      // here is not defined - implemented with jquery
   }

   JSROOT.HierarchyPainter.prototype.dropitem = function(itemname, divid, call_back) {
      var h = this;

      h.get(itemname, function(item, obj) {
         if (obj!=null) {
            var painter = h.draw(divid, obj, "same");
            if (painter) painter.WhenReady(function() { painter.SetItemName(itemname); });
         }

         JSROOT.CallBack(call_back);
      });

      return true;
   }

   JSROOT.HierarchyPainter.prototype.updateItems = function(items) {
      // argument is item name or array of string with items name
      // only already drawn items will be update with same draw option

      if ((this.disp == null) || (items==null)) return;

      var draw_items = [], draw_options = [];

      this.disp.ForEachPainter(function(p) {
         var itemname = p.GetItemName();
         if ((itemname==null) || (draw_items.indexOf(itemname)>=0)) return;
         if (typeof items == 'array') {
            if (items.indexOf(itemname) < 0) return;
         } else {
            if (items != itemname) return;
         }
         draw_items.push(itemname);
         draw_options.push("update:" + p.GetItemDrawOpt());
      }, true); // only visible panels are considered

      if (draw_items.length > 0)
         this.displayAll(draw_items, draw_options);
   }


   JSROOT.HierarchyPainter.prototype.updateAll = function(only_auto_items, only_items) {
      // method can be used to fetch new objects and update all existing drawings
      // if only_auto_items specified, only automatic items will be updated

      if (this.disp == null) return;

      var allitems = [], options = [], hpainter = this;

      // first collect items
      this.disp.ForEachPainter(function(p) {
         var itemname = p.GetItemName();
         var drawopt = p.GetItemDrawOpt();
         if ((itemname==null) || (allitems.indexOf(itemname)>=0)) return;

         var item = hpainter.Find(itemname);
         if ((item==null) || ('_not_monitor' in item) || ('_player' in item)) return;
         var forced = false;

         if ('_always_monitor' in item) {
            forced = true;
         } else {
            var handle = JSROOT.getDrawHandle(item._kind);
            if (handle && ('monitor' in handle)) {
               if ((handle.monitor===false) || (handle.monitor=='never')) return;
               if (handle.monitor==='always') forced = true;
            }
         }

         if (forced || !only_auto_items) {
            allitems.push(itemname);
            options.push("update:" + drawopt);
         }
      }, true); // only visible panels are considered

      var painter = this;

      // force all files to read again (normally in non-browser mode)
      if (this.files_monitoring && !only_auto_items)
         this.ForEachRootFile(function(item) {
            painter.ForEach(function(fitem) { delete fitem._readobj; }, item);
            delete item._file;
         });

      if (allitems.length > 0)
         this.displayAll(allitems, options);
   }

   JSROOT.HierarchyPainter.prototype.displayAll = function(items, options, call_back) {

      if ((items == null) || (items.length == 0)) return JSROOT.CallBack(call_back);

      var h = this;

      if (!options) options = [];
      while (options.length < items.length)
         options.push("");

      if ((options.length == 1) && (options[0] == "iotest")) {
         h.clear();
         d3.select("#" + h.disp_frameid).html("<h2>Start I/O test "+ ('IO' in JSROOT ? "Mode=" + JSROOT.IO.Mode : "") +  "</h2>")

         var tm0 = new Date();
         return h.get(items[0], function(item, obj) {
            var tm1 = new Date();
            d3.select("#" + h.disp_frameid).append("h2").html("Item " + items[0] + " reading time = " + (tm1.getTime() - tm0.getTime()) + "ms");
            return JSROOT.CallBack(call_back);
         });
      }

      var dropitems = new Array(items.length);

      // First of all check that items are exists, look for cycle extension and plus sign
      for (var i = 0; i < items.length; ++i) {
         dropitems[i] = null;

         var item = items[i], can_split = true;

         if ((item.length>1) && (item[0]=='\'') && (item[item.length-1]=='\'')) {
            items[i] = item.substr(1, item.length-2);
            can_split = false;
         }

         var elem = h.Find({ name: items[i], check_keys: true });
         if (elem) { items[i] = h.itemFullName(elem); continue; }

         if (can_split && (items[i].indexOf("+") > 0)) {
            dropitems[i] = items[i].split("+");
            items[i] = dropitems[i].shift();
            // allow to specify _same_ item in different file
            for (var j = 0; j < dropitems[i].length; ++j) {
               var pos = dropitems[i][j].indexOf("_same_");
               if ((pos>0) && (h.Find(dropitems[i][j])==null))
                  dropitems[i][j] = dropitems[i][j].substr(0,pos) + items[i].substr(pos);
            }
         }

         // also check if subsequent items has _same_, than use name from first item
         var pos = items[i].indexOf("_same_");
         if ((pos>0) && !h.Find(items[i]) && (i>0))
            items[i] = items[i].substr(0,pos) + items[0].substr(pos);
      }

      // now check that items can be displayed
      for (var n = items.length-1; n>=0; --n) {
         var hitem = h.Find(items[n]);
         if ((hitem==null) || h.canDisplay(hitem, options[n])) continue;
         // try to expand specified item
         h.expand(items[n]);
         items.splice(n, 1);
         options.splice(n, 1);
         dropitems.splice(n, 1);
      }

      if (items.length == 0) return JSROOT.CallBack(call_back);

      var frame_names = [];
      for (var n=0; n<items.length;++n) {
         var fname = items[n], k = 0;
         while (frame_names.indexOf(fname)>=0)
            fname = items[n] + "_" + k++;
         frame_names.push(fname);
      }

      h.CreateDisplay(function(mdi) {
         if (!mdi) return JSROOT.CallBack(call_back);

         // Than create empty frames for each item
         for (var i = 0; i < items.length; ++i)
            if (options[i].indexOf('update:')!==0) {
               mdi.CreateFrame(frame_names[i]);
               options[i] += "::_display_on_frame_::"+frame_names[i];
            }

         // We start display of all items parallel
         for (var i = 0; i < items.length; ++i)
            h.display(items[i], options[i], DisplayCallback.bind(this,i));

         function DisplayCallback(indx, painter, itemname) {
            items[indx] = null; // mark item as ready
            DropNextItem(indx, painter);
         }

         function DropNextItem(indx, painter) {
            if (painter && dropitems[indx] && (dropitems[indx].length>0))
               return h.dropitem(dropitems[indx].shift(), painter.divid, DropNextItem.bind(this, indx, painter));

            var isany = false;
            for (var cnt = 0; cnt < items.length; ++cnt)
               if (items[cnt]!==null) isany = true;

            // only when items drawn and all sub-items dropped, one could perform call-back
            if (!isany) JSROOT.CallBack(call_back);
         }

      });
   }

   JSROOT.HierarchyPainter.prototype.reload = function() {
      var hpainter = this;
      if ('_online' in this.h)
         this.OpenOnline(this.h._online, function() {
            hpainter.RefreshHtml();
         });
   }

   JSROOT.HierarchyPainter.prototype.UpdateTreeNode = function() {
      // dummy function, will be redefined when jquery part loaded
   }

   JSROOT.HierarchyPainter.prototype.actiavte = function(items, force) {
      // activate (select) specified item
      // if force specified, all required sub-levels will be opened

      if (typeof items == 'string') items = [ items ];

      var active = [],  // array of elements to activate
          painter = this, // painter itself
          update = []; // array of elements to update
      this.ForEach(function(item) { if (item._background) { active.push(item); delete item._background; } });

      function mark_active() {
         if (typeof painter.UpdateBackground !== 'function') return;

         for (var n=update.length-1;n>=0;--n)
            painter.UpdateTreeNode(update[n]);

         for (var n=0;n<active.length;++n)
            painter.UpdateBackground(active[n], force);
      }

      function find_next(itemname, prev_found) {
         if (itemname === undefined) {
            // extract next element
            if (items.length == 0) return mark_active();
            itemname = items.shift();
         }

         var hitem = painter.Find(itemname);

         if (!hitem) {
            var d = painter.Find({ name: itemname, last_exists: true, check_keys: true, allow_index: true });
            if (!d || !d.last) return find_next();
            d.now_found = painter.itemFullName(d.last);

            if (force) {

               // if after last expand no better solution found - skip it
               if ((prev_found!==undefined) && (d.now_found === prev_found)) return find_next();

               return painter.expand(d.now_found, function(res) {
                  if (!res) return find_next();
                  var newname = painter.itemFullName(d.last);
                  if (newname.length>0) newname+="/";
                  find_next(newname + d.rest, d.now_found);
               });
            }
            hitem = d.last;
         }

         if (hitem) {
            // check that item is visible (opened), otherwise should enable parent

            var prnt = hitem._parent;
            while (prnt) {
               if (!prnt._isopen) {
                  if (force) {
                     prnt._isopen = true;
                     if (update.indexOf(prnt)<0) update.push(prnt);
                  } else {
                     hitem = prnt; break;
                  }
               }
               prnt = prnt._parent;
            }


            hitem._background = 'grey';
            if (active.indexOf(hitem)<0) active.push(hitem);
         }

         find_next();
      }

      // use recursion
      find_next();
   }

   JSROOT.HierarchyPainter.prototype.expand = function(itemname, call_back, d3cont) {
      var hpainter = this, hitem = this.Find(itemname);

      if (!hitem && d3cont) return JSROOT.CallBack(call_back);

      function DoExpandItem(_item, _obj, _name) {
         if (!_name) _name = hpainter.itemFullName(_item);

         // try to use expand function
         if (_obj && _item && (typeof _item._expand === 'function')) {
            if (_item._expand(_item, _obj)) {
               _item._isopen = true;
               if (_item._parent && !_item._parent._isopen) {
                  _item._parent._isopen = true; // also show parent
                  hpainter.UpdateTreeNode(_item._parent);
               } else {
                  hpainter.UpdateTreeNode(_item, d3cont);
               }
               JSROOT.CallBack(call_back, _item);
               return true;
            }
         }

         if (!('_expand' in _item)) {
            var handle = JSROOT.getDrawHandle(_item._kind, "::expand");
            if (handle && ('expand' in handle)) {
               JSROOT.AssertPrerequisites(handle.prereq, function() {
                  _item._expand = JSROOT.findFunction(handle.expand);
                  if (!_item._expand)
                     delete _item._expand;
                  else
                     hpainter.expand(_name, call_back, d3cont);
               });
               return true;
            }
         }

         if (_obj && JSROOT.Painter.ObjectHierarchy(_item, _obj)) {
            _item._isopen = true;
            if (_item._parent && !_item._parent._isopen) {
               _item._parent._isopen = true; // also show parent
               hpainter.UpdateTreeNode(_item._parent);
            } else {
               hpainter.UpdateTreeNode(_item, d3cont);
            }
            JSROOT.CallBack(call_back, _item);
            return true;
         }

         return false;
      }

      if (hitem) {
         // item marked as it cannot be expanded
         if (('_more' in hitem) && !hitem._more) return JSROOT.CallBack(call_back);

         if (DoExpandItem(hitem, hitem._obj, itemname)) return;
      }

      JSROOT.progress("Loading " + itemname);

      this.get(itemname, function(item, obj) {

         JSROOT.progress();

         if (obj && DoExpandItem(item, obj)) return;

         JSROOT.CallBack(call_back);
      }, "hierarchy_expand" ); // indicate that we getting element for expand, can handle it differently

   }

   JSROOT.HierarchyPainter.prototype.GetTopOnlineItem = function(item) {
      if (item!=null) {
         while ((item!=null) && (!('_online' in item))) item = item._parent;
         return item;
      }

      if (this.h==null) return null;
      if ('_online' in this.h) return this.h;
      if ((this.h._childs!=null) && ('_online' in this.h._childs[0])) return this.h._childs[0];
      return null;
   }


   JSROOT.HierarchyPainter.prototype.ForEachJsonFile = function(call_back) {
      if (this.h==null) return;
      if ('_jsonfile' in this.h)
         return JSROOT.CallBack(call_back, this.h);

      if (this.h._childs!=null)
         for (var n = 0; n < this.h._childs.length; ++n) {
            var item = this.h._childs[n];
            if ('_jsonfile' in item) JSROOT.CallBack(call_back, item);
         }
   }

   JSROOT.HierarchyPainter.prototype.OpenJsonFile = function(filepath, call_back) {
      var isfileopened = false;
      this.ForEachJsonFile(function(item) { if (item._jsonfile==filepath) isfileopened = true; });
      if (isfileopened) return JSROOT.CallBack(call_back);

      var pthis = this;
      JSROOT.NewHttpRequest(filepath,'object', function(res) {
         if (res == null) return JSROOT.CallBack(call_back);
         var h1 = { _jsonfile: filepath, _kind: "ROOT." + res._typename, _jsontmp: res, _name: filepath.split("/").pop() };
         if ('fTitle' in res) h1._title = res.fTitle;
         h1._get = function(item,itemname,callback) {
            if ('_jsontmp' in item) {
               //var res = item._jsontmp;
               //delete item._jsontmp;
               return JSROOT.CallBack(callback, item, item._jsontmp);
            }
            JSROOT.NewHttpRequest(item._jsonfile, 'object', function(res) {
               item._jsontmp = res;
               return JSROOT.CallBack(callback, item, item._jsontmp);
            }).send(null);
         }
         if (pthis.h == null) pthis.h = h1; else
         if (pthis.h._kind == 'TopFolder') pthis.h._childs.push(h1); else {
            var h0 = pthis.h;
            var topname = ('_jsonfile' in h0) ? "Files" : "Items";
            pthis.h = { _name: topname, _kind: 'TopFolder', _childs : [h0, h1] };
         }

         pthis.RefreshHtml(call_back);
      }).send(null);
   }

   JSROOT.HierarchyPainter.prototype.ForEachRootFile = function(call_back) {
      if (this.h==null) return;
      if ((this.h._kind == "ROOT.TFile") && (this.h._file!=null))
         return JSROOT.CallBack(call_back, this.h);

      if (this.h._childs != null)
         for (var n = 0; n < this.h._childs.length; ++n) {
            var item = this.h._childs[n];
            if ((item._kind == 'ROOT.TFile') && ('_fullurl' in item))
               JSROOT.CallBack(call_back, item);
         }
   }

   JSROOT.HierarchyPainter.prototype.OpenRootFile = function(filepath, call_back) {
      // first check that file with such URL already opened

      var isfileopened = false;
      this.ForEachRootFile(function(item) { if (item._fullurl==filepath) isfileopened = true; });
      if (isfileopened) return JSROOT.CallBack(call_back);

      var pthis = this;

      JSROOT.OpenFile(filepath, function(file) {
         if (file == null) return JSROOT.CallBack(call_back);
         var h1 = pthis.FileHierarchy(file);
         h1._isopen = true;
         if (pthis.h == null) pthis.h = h1; else
            if (pthis.h._kind == 'TopFolder') pthis.h._childs.push(h1); else {
               var h0 = pthis.h;
               var topname = (h0._kind == "ROOT.TFile") ? "Files" : "Items";
               pthis.h = { _name: topname, _kind: 'TopFolder', _childs : [h0, h1] };
            }

         pthis.RefreshHtml(call_back);
      });
   }

   JSROOT.HierarchyPainter.prototype.GetFileProp = function(itemname) {
      var item = this.Find(itemname);
      if (item == null) return null;

      var subname = item._name;
      while (item._parent) {
         item = item._parent;
         if ('_file' in item)
            return { kind: "file", fileurl: item._file.fURL, itemname: subname };

         if ('_jsonfile' in item)
            return { kind: "json", fileurl: item._jsonfile, itemname: subname };

         subname = item._name + "/" + subname;
      }

      return null;
   }

   JSROOT.MarkAsStreamerInfo = function(h,item,obj) {
      // this function used on THttpServer to mark streamer infos list
      // as fictional TStreamerInfoList class, which has special draw function
      if ((obj!=null) && (obj._typename=='TList'))
         obj._typename = 'TStreamerInfoList';
   }

   JSROOT.HierarchyPainter.prototype.GetOnlineItemUrl = function(item) {
      // returns URL, which could be used to request item from the online server
      if ((item!=null) && (typeof item == "string")) item = this.Find(item);
      var top = this.GetTopOnlineItem(item);
      if (item==null) return null;

      var urlpath = this.itemFullName(item, top);
      if (top && ('_online' in top) && (top._online!="")) urlpath = top._online + urlpath;
      return urlpath;
   }

   JSROOT.HierarchyPainter.prototype.GetOnlineItem = function(item, itemname, callback, option) {
      // method used to request object from the http server

      var url = itemname, h_get = false, req = "", req_kind = "object", pthis = this, draw_handle = null;

      if (option === 'hierarchy_expand') { h_get = true; option = undefined; }

      if (item != null) {
         url = this.GetOnlineItemUrl(item);
         var func = null;
         if ('_kind' in item) draw_handle = JSROOT.getDrawHandle(item._kind);

         if (h_get) {
            req = 'h.json?compact=3';
            item._expand = JSROOT.Painter.OnlineHierarchy; // use proper expand function
         } else
         if ('_make_request' in item) {
            func = JSROOT.findFunction(item._make_request);
         } else
         if ((draw_handle!=null) && ('make_request' in draw_handle)) {
            func = draw_handle.make_request;
         }

         if (typeof func == 'function') {
            // ask to make request
            var dreq = func(pthis, item, url, option);
            // result can be simple string or object with req and kind fields
            if (dreq!=null)
               if (typeof dreq == 'string') req = dreq; else {
                  if ('req' in dreq) req = dreq.req;
                  if ('kind' in dreq) req_kind = dreq.kind;
               }
         }

         if ((req.length==0) && (item._kind.indexOf("ROOT.")!=0))
           req = 'item.json.gz?compact=3';
      }

      if ((itemname==null) && (item!=null) && ('_cached_draw_object' in this) && (req.length == 0)) {
         // special handling for drawGUI when cashed
         var obj = this._cached_draw_object;
         delete this._cached_draw_object;
         return JSROOT.CallBack(callback, item, obj);
      }

      if (req.length == 0) req = 'root.json.gz?compact=3';

      if (url.length > 0) url += "/";
      url += req;

      var itemreq = JSROOT.NewHttpRequest(url, req_kind, function(obj) {

         var func = null;

         if (!h_get && (item!=null) && ('_after_request' in item)) {
            func = JSROOT.findFunction(item._after_request);
         } else
         if ((draw_handle!=null) && ('after_request' in draw_handle))
            func = draw_handle.after_request;

         if (typeof func == 'function') {
            var res = func(pthis, item, obj, option, itemreq);
            if ((res!=null) && (typeof res == "object")) obj = res;
         }

         JSROOT.CallBack(callback, item, obj);
      });

      itemreq.send(null);
   }

   JSROOT.Painter.OnlineHierarchy = function(node, obj) {
      // central function for expand of all online items

      if ((obj != null) && (node != null) && ('_childs' in obj)) {

         for (var n=0;n<obj._childs.length;++n)
            if (obj._childs[n]._more || obj._childs[n]._childs)
               obj._childs[n]._expand = JSROOT.Painter.OnlineHierarchy;

         node._childs = obj._childs;
         obj._childs = null;
         return true;
      }

      return false;
   }

   JSROOT.HierarchyPainter.prototype.OpenOnline = function(server_address, user_callback) {
      var painter = this;

      function AdoptHierarchy(result) {
         painter.h = result;
         if (painter.h == null) return;

         if (('_title' in painter.h) && (painter.h._title!='')) document.title = painter.h._title;

         result._isopen = true;

         // mark top hierarchy as online data and
         painter.h._online = server_address;

         painter.h._get = function(item, itemname, callback, option) {
            painter.GetOnlineItem(item, itemname, callback, option);
         }

         painter.h._expand = JSROOT.Painter.OnlineHierarchy;

         var scripts = "", modules = "";
         painter.ForEach(function(item) {
            if ('_childs' in item) item._expand = JSROOT.Painter.OnlineHierarchy;

            if ('_autoload' in item) {
               var arr = item._autoload.split(";");
               for (var n = 0; n < arr.length; ++n)
                  if ((arr[n].length>3) &&
                      ((arr[n].lastIndexOf(".js")==arr[n].length-3) ||
                      (arr[n].lastIndexOf(".css")==arr[n].length-4))) {
                     if (scripts.indexOf(arr[n])<0) scripts+=arr[n]+";";
                  } else {
                     if (modules.indexOf(arr[n])<0) modules+=arr[n]+";";
                  }
            }
         });

         if (scripts.length > 0) scripts = "user:" + scripts;

         // use AssertPrerequisites, while it protect us from race conditions
         JSROOT.AssertPrerequisites(modules + scripts, function() {

            painter.ForEach(function(item) {
               if (!('_drawfunc' in item) || !('_kind' in item)) return;
               var typename = "kind:" + item._kind;
               if (item._kind.indexOf('ROOT.')==0) typename = item._kind.slice(5);
               var drawopt = item._drawopt;
               if (!JSROOT.canDraw(typename) || (drawopt!=null))
                  JSROOT.addDrawFunc({ name: typename, func: item._drawfunc, script: item._drawscript, opt: drawopt });
            });

            JSROOT.CallBack(user_callback, painter);
         });
      }

      if (!server_address) server_address = "";

      if (typeof server_address == 'object') {
         var h = server_address;
         server_address = "";
         return AdoptHierarchy(h);
      }

      JSROOT.NewHttpRequest(server_address + "h.json?compact=3", 'object', AdoptHierarchy).send(null);
   }

   JSROOT.HierarchyPainter.prototype.GetOnlineProp = function(itemname) {
      var item = this.Find(itemname);
      if (item == null) return null;

      var subname = item._name;
      while (item._parent != null) {
         item = item._parent;

         if ('_online' in item) {
            return {
               server : item._online,
               itemname : subname
            };
         }
         subname = item._name + "/" + subname;
      }

      return null;
   }

   JSROOT.HierarchyPainter.prototype.FillOnlineMenu = function(menu, onlineprop, itemname) {

      var painter = this;

      var node = this.Find(itemname);
      var opts = JSROOT.getDrawOptions(node._kind, 'nosame');
      var handle = JSROOT.getDrawHandle(node._kind);
      var root_type = ('_kind' in node) ? node._kind.indexOf("ROOT.") == 0 : false;

      if (opts != null)
         menu.addDrawMenu("Draw", opts, function(arg) { painter.display(itemname, arg); });

      if ((node['_childs'] == null) && (node['_more'] || root_type))
         menu.add("Expand", function() { painter.expand(itemname); });

      if (handle && ('execute' in handle))
         menu.add("Execute", function() { painter.ExecuteCommand(itemname, menu.tree_node); });

      var drawurl = onlineprop.server + onlineprop.itemname + "/draw.htm";
      var separ = "?";
      if (this.IsMonitoring()) {
         drawurl += separ + "monitoring=" + this.MonitoringInterval();
         separ = "&";
      }

      if (opts != null)
         menu.addDrawMenu("Draw in new window", opts, function(arg) { window.open(drawurl+separ+"opt=" +arg); });

      if ((opts!=null) && (opts.length > 0) && root_type)
         menu.addDrawMenu("Draw as png", opts, function(arg) {
            window.open(onlineprop.server + onlineprop.itemname + "/root.png?w=400&h=300&opt=" + arg);
         });

      if ('_player' in node)
         menu.add("Player", function() { painter.player(itemname); });
   }

   JSROOT.HierarchyPainter.prototype.Adopt = function(h) {
      this.h = h;
      this.RefreshHtml();
   }

   JSROOT.HierarchyPainter.prototype.SetMonitoring = function(val) {
      this._monitoring_on = false;
      this._monitoring_interval = 3000;

      val = (val === undefined) ? 0 : parseInt(val);

      if (!isNaN(val) && (val>0)) {
         this._monitoring_on = true;
         this._monitoring_interval = Math.max(100,val);
      }
   }

   JSROOT.HierarchyPainter.prototype.MonitoringInterval = function(val) {
      // returns interval
      return ('_monitoring_interval' in this) ? this._monitoring_interval : 3000;
   }

   JSROOT.HierarchyPainter.prototype.EnableMonitoring = function(on) {
      this._monitoring_on = on;
   }

   JSROOT.HierarchyPainter.prototype.IsMonitoring = function() {
      return this._monitoring_on;
   }

   JSROOT.HierarchyPainter.prototype.SetDisplay = function(layout, frameid) {
      if ((frameid==null) && (typeof layout == 'object')) {
         this.disp = layout;
         this.disp_kind = 'custom';
         this.disp_frameid = null;
      } else {
         this.disp_kind = layout;
         this.disp_frameid = frameid;
      }
   }

   JSROOT.HierarchyPainter.prototype.GetLayout = function() {
      return this.disp_kind;
   }

   JSROOT.HierarchyPainter.prototype.clear = function(withbrowser) {
      if (this.disp) {
         this.disp.Reset();
         delete this.disp;
      }

      this.ForEach(function(item) {
         delete item._painter; // remove reference on the painter
         // when only display cleared, try to clear all browser items
         if (!withbrowser && (typeof item.clear=='function')) item.clear();
      });

      if (withbrowser) {
         this.select_main().html("");
         delete this.h;
      }
   }

   JSROOT.HierarchyPainter.prototype.GetDisplay = function() {
      return ('disp' in this) ? this.disp : null;
   }

   JSROOT.HierarchyPainter.prototype.CreateDisplay = function(callback) {

      var h = this;

      if ('disp' in this) {
         if ((h.disp.NumDraw() > 0) || (h.disp_kind == "custom")) return JSROOT.CallBack(callback, h.disp);
         h.disp.Reset();
         delete h.disp;
      }

      // check that we can found frame where drawing should be done
      if (document.getElementById(this.disp_frameid) == null)
         return JSROOT.CallBack(callback, null);

      if (h.disp_kind == "simple")
         h.disp = new JSROOT.SimpleDisplay(h.disp_frameid);
      else
      if (h.disp_kind.search("grid") == 0)
         h.disp = new JSROOT.GridDisplay(h.disp_frameid, h.disp_kind);
      else
         return JSROOT.AssertPrerequisites('jq2d', function() { h.CreateDisplay(callback); });

      JSROOT.CallBack(callback, h.disp);
   }

   JSROOT.HierarchyPainter.prototype.updateOnOtherFrames = function(painter, obj) {
      // function should update object drawings for other painters
      var mdi = this.disp;
      if (mdi==null) return false;

      var isany = false;
      mdi.ForEachPainter(function(p, frame) {
         if ((p===painter) || (p.GetItemName() != painter.GetItemName())) return;
         mdi.ActivateFrame(frame);
         p.RedrawObject(obj);
         isany = true;
      });
      return isany;
   }

   JSROOT.HierarchyPainter.prototype.CheckResize = function(size) {
      if ('disp' in this)
         this.disp.CheckMDIResize(null, size);
   }

   JSROOT.HierarchyPainter.prototype.StartGUI = function(h0, call_back) {
      var hpainter = this,
          filesarr = JSROOT.GetUrlOptionAsArray("file;files"),
          jsonarr = JSROOT.GetUrlOptionAsArray("json"),
          filesdir = JSROOT.GetUrlOption("path"),
          expanditems = JSROOT.GetUrlOptionAsArray("expand");

      if (expanditems.length==0 && (JSROOT.GetUrlOption("expand")=="")) expanditems.push("");

      if (filesdir!=null) {
         for (var i=0;i<filesarr.length;++i) filesarr[i] = filesdir + filesarr[i];
         for (var i=0;i<jsonarr.length;++i) jsonarr[i] = filesdir + jsonarr[i];
      }

      var itemsarr = JSROOT.GetUrlOptionAsArray("item;items");
      if ((itemsarr.length==0) && JSROOT.GetUrlOption("item")=="") itemsarr.push("");

      var optionsarr = JSROOT.GetUrlOptionAsArray("opt;opts"),
          monitor = JSROOT.GetUrlOption("monitoring");

      if ((jsonarr.length==1) && (itemsarr.length==0) && (expanditems.length==0)) itemsarr.push("");

      if (!this.disp_kind) {
         var layout = JSROOT.GetUrlOption("layout");
         if ((typeof layout == "string") && (layout.length>0))
            this.disp_kind = layout;
         else
         switch (itemsarr.length) {
           case 0:
           case 1: this.disp_kind = 'simple'; break;
           case 2: this.disp_kind = 'grid 1x2'; break;
           default: this.disp_kind = 'flex';
         }
      }

      if (JSROOT.GetUrlOption('files_monitoring')!=null) this.files_monitoring = true;

      this.SetMonitoring(monitor);

      function OpenAllFiles() {
         if (jsonarr.length>0)
            hpainter.OpenJsonFile(jsonarr.shift(), OpenAllFiles);
         else if (filesarr.length>0)
            hpainter.OpenRootFile(filesarr.shift(), OpenAllFiles);
         else if (expanditems.length>0)
            hpainter.expand(expanditems.shift(), OpenAllFiles);
         else
            hpainter.displayAll(itemsarr, optionsarr, function() {
               hpainter.RefreshHtml();

               JSROOT.RegisterForResize(hpainter);

               // assign regular update only when monitoring specified (or when work with online application)
               if (h0 || hpainter.GetTopOnlineItem() || hpainter.IsMonitoring())
                  setInterval(function() { hpainter.updateAll(!hpainter.IsMonitoring()); }, hpainter.MonitoringInterval());

               JSROOT.CallBack(call_back);
           });
      }

      function AfterOnlineOpened() {
         // check if server enables monitoring
         if (('_monitoring' in hpainter.h) && (monitor==null)) {
            hpainter.SetMonitoring(hpainter.h._monitoring);
         }

         if (('_layout' in hpainter.h) && (layout==null)) {
            hpainter.disp_kind = hpainter.h._layout;
         }

         if (('_loadfile' in hpainter.h) && (filesarr.length==0)) {
            filesarr = JSROOT.ParseAsArray(hpainter.h._loadfile);
         }

         if (('_drawitem' in hpainter.h) && (itemsarr.length==0)) {
            itemsarr = JSROOT.ParseAsArray(hpainter.h._drawitem);
            optionsarr = JSROOT.ParseAsArray(hpainter.h['_drawopt']);
         }

         OpenAllFiles();
      }

      if (h0) hpainter.OpenOnline(h0, AfterOnlineOpened);
          else OpenAllFiles();
   }

   JSROOT.BuildNobrowserGUI = function() {
      var myDiv = d3.select('#simpleGUI');
      var online = false, drawing = false;

      if (myDiv.empty()) {
         online = true;
         myDiv = d3.select('#onlineGUI');
         if (myDiv.empty()) { myDiv = d3.select('#drawGUI'); drawing = true; }
         if (myDiv.empty()) return alert('no div for simple nobrowser gui found');
      }

      JSROOT.Painter.readStyleFromURL();

      d3.select('html').style('height','100%');
      d3.select('body').style({ 'min-height':'100%', 'margin':'0px', "overflow": "hidden"});

      myDiv.style({"position":"absolute", "left":"1px", "top" :"1px", "bottom" :"1px", "right": "1px"});

      var hpainter = new JSROOT.HierarchyPainter('root', null);
      hpainter.SetDisplay(JSROOT.GetUrlOption("layout", null, "simple"), myDiv.attr('id'));

      var h0 = null;
      if (online) {
         var func = JSROOT.findFunction('GetCachedHierarchy');
         if (typeof func == 'function') h0 = func();
         if (typeof h0 != 'object') h0 = "";
      }

      hpainter.StartGUI(h0, function() {
         if (!drawing) return;
         var func = JSROOT.findFunction('GetCachedObject');
         var obj = (typeof func == 'function') ? JSROOT.JSONR_unref(func()) : null;
         if (obj!=null) hpainter['_cached_draw_object'] = obj;
         var opt = JSROOT.GetUrlOption("opt");

         hpainter.display("", opt, function(obj_painter) {
            if (JSROOT.GetUrlOption("websocket")!==null)
               obj_painter.OpenWebsocket();
         });
      });
   }

   JSROOT.Painter.drawStreamerInfo = function(divid, lst) {
      var painter = new JSROOT.HierarchyPainter('sinfo', divid, 'white');

      painter.h = { _name : "StreamerInfo", _childs : [] };

      for ( var i = 0; i < lst.arr.length; ++i) {
         var entry = lst.arr[i]

         if (entry._typename == "TList") continue;

         if (typeof (entry.fName) == 'undefined') {
            JSROOT.console("strange element in StreamerInfo with type " + entry._typename);
            continue;
         }

         var item = {
            _name : entry.fName + ";" + entry.fClassVersion,
            _kind : "class " + entry.fName,
            _title : "class:" + entry.fName + ' version:' + entry.fClassVersion + ' checksum:' + entry.fCheckSum,
            _icon: "img_class",
            _childs : []
         };

         if (entry.fTitle != '') item._title += '  ' + entry.fTitle;

         painter.h._childs.push(item);

         if (typeof entry.fElements == 'undefined') continue;
         for ( var l = 0; l < entry.fElements.arr.length; ++l) {
            var elem = entry.fElements.arr[l];
            if ((elem == null) || (typeof (elem.fName) == 'undefined')) continue;
            var info = elem.fTypeName + " " + elem.fName + ";";
            if (elem.fTitle != '') info += " // " + elem.fTitle;
            item._childs.push({ _name : info, _title: elem.fTypeName, _kind:elem.fTypeName, _icon: (elem.fTypeName == 'BASE') ? "img_class" : "img_member" });
         }
         if (item._childs.length == 0) delete item._childs;
      }

      painter.select_main().style('overflow','auto');

      painter.RefreshHtml(function() {
         painter.SetDivId(divid);
         painter.DrawingReady();
      });

      return painter;
   }

   JSROOT.Painter.drawInspector = function(divid, obj) {
      var painter = new JSROOT.HierarchyPainter('inspector', divid, 'white');
      painter.default_by_click = "expand"; // that painter tries to do by default
      painter.with_icons = false;
      painter.h = { _name: "Object", _title: "ROOT." + obj._typename, _click_action: "expand", _nosimple: false };
      if ((typeof obj.fName === 'string') && (obj.fName.length>0))
         painter.h._name = obj.fName;

      painter.select_main().style('overflow','auto');

      JSROOT.Painter.ObjectHierarchy(painter.h, obj);
      painter.RefreshHtml(function() {
         painter.SetDivId(divid);
         painter.DrawingReady();
      });

      return painter;
   }

   // ================================================================

   // JSROOT.MDIDisplay - class to manage multiple document interface for drawings

   JSROOT.MDIDisplay = function(frameid) {
      this.frameid = frameid;
      d3.select("#"+this.frameid).property('mdi', this);
   }

   JSROOT.MDIDisplay.prototype.ForEachFrame = function(userfunc, only_visible) {
      // method dedicated to iterate over existing panels
      // provided userfunc is called with arguemnts (frame)

      console.warn("ForEachFrame not implemented in MDIDisplay");
   }

   JSROOT.MDIDisplay.prototype.ForEachPainter = function(userfunc, only_visible) {
      // method dedicated to iterate over existing panles
      // provided userfunc is called with arguemnts (painter, frame)

      this.ForEachFrame(function(frame) {
         var dummy = new JSROOT.TObjectPainter();
         dummy.SetDivId(d3.select(frame).attr('id'), -1);
         dummy.ForEachPainter(function(painter) { userfunc(painter, frame); });
      }, only_visible);
   }

   JSROOT.MDIDisplay.prototype.NumDraw = function() {
      var cnt = 0;
      this.ForEachFrame(function() { ++cnt; });
      return cnt;
   }

   JSROOT.MDIDisplay.prototype.FindFrame = function(searchtitle, force) {
      var found_frame = null;

      this.ForEachFrame(function(frame) {
         if (d3.select(frame).attr('frame_title') == searchtitle)
            found_frame = frame;
      });

      if ((found_frame == null) && force)
         found_frame = this.CreateFrame(searchtitle);

      return found_frame;
   }

   JSROOT.MDIDisplay.prototype.ActivateFrame = function(frame) {
      // do nothing by default
   }

   JSROOT.MDIDisplay.prototype.CheckMDIResize = function(only_frame_id, size) {
      // perform resize for each frame
      var resized_frame = null;

      this.ForEachPainter(function(painter, frame) {

         if ((only_frame_id !== null) && (d3.select(frame).attr('id') != only_frame_id)) return;

         if ((painter.GetItemName()!==null) && (typeof painter.CheckResize == 'function')) {
            // do not call resize for many painters on the same frame
            if (resized_frame === frame) return;
            painter.CheckResize(size);
            resized_frame = frame;
         }
      });
   }

   JSROOT.MDIDisplay.prototype.Reset = function() {

      this.ForEachFrame(function(frame) {
         JSROOT.cleanup(frame);
      });

      d3.select("#"+this.frameid).html("").property('mdi', null);
   }

   JSROOT.MDIDisplay.prototype.Draw = function(title, obj, drawopt) {
      // draw object with specified options
      if (!obj) return;

      if (!JSROOT.canDraw(obj._typename, drawopt)) return;

      var frame = this.FindFrame(title, true);

      this.ActivateFrame(frame);

      return JSROOT.redraw(d3.select(frame).attr("id"), obj, drawopt);
   }


   // ==================================================

   JSROOT.CustomDisplay = function() {
      JSROOT.MDIDisplay.call(this, "dummy");
      this.frames = {}; // array of configured frames
   }

   JSROOT.CustomDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.CustomDisplay.prototype.AddFrame = function(divid, itemname) {
      if (!(divid in this.frames)) this.frames[divid] = "";

      this.frames[divid] += (itemname + ";");
   }

   JSROOT.CustomDisplay.prototype.ForEachFrame = function(userfunc,  only_visible) {
      var ks = Object.keys(this.frames);
      for (var k = 0; k < ks.length; ++k) {
         var node = d3.select("#"+ks[k]);
         if (!node.empty())
            JSROOT.CallBack(userfunc, node.node());
      }
   }

   JSROOT.CustomDisplay.prototype.CreateFrame = function(title) {
      var ks = Object.keys(this.frames);
      for (var k = 0; k < ks.length; ++k) {
         var items = this.frames[ks[k]];
         if (items.indexOf(title+";")>=0)
            return d3.select("#"+ks[k]).node();
      }
      return null;
   }

   JSROOT.CustomDisplay.prototype.Reset = function() {
      JSROOT.MDIDisplay.prototype.Reset.call(this);
      this.ForEachFrame(function(frame) {
         d3.select(frame).html("");
      });
   }

   // ==================================================

   JSROOT.SimpleDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
   }

   JSROOT.SimpleDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.SimpleDisplay.prototype.ForEachFrame = function(userfunc,  only_visible) {
      var node = d3.select("#"+this.frameid + "_simple_display");
      if (!node.empty())
         JSROOT.CallBack(userfunc, node.node());
   }

   JSROOT.SimpleDisplay.prototype.CreateFrame = function(title) {

      JSROOT.cleanup(this.frameid+"_simple_display");

      return d3.select("#"+this.frameid)
               .html("")
               .append("div")
               .attr("id", this.frameid + "_simple_display")
               .style("width", "100%")
               .style("height", "100%")
               .style("overflow" ,"hidden")
               .attr("frame_title", title)
               .node();
   }

   JSROOT.SimpleDisplay.prototype.Reset = function() {
      JSROOT.MDIDisplay.prototype.Reset.call(this);
      // try to remove different properties from the div
      d3.select("#"+this.frameid).html("");
   }

   // ================================================

   JSROOT.GridDisplay = function(frameid, sizex, sizey) {
      // create grid display object
      // one could use followinf arguments
      // new JSROOT.GridDisplay('yourframeid','4x4');
      // new JSROOT.GridDisplay('yourframeid','3x2');
      // new JSROOT.GridDisplay('yourframeid', 3, 4);

      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0;
      if (typeof sizex == "string") {
         if (sizex.search("grid") == 0)
            sizex = sizex.slice(4).trim();

         var separ = sizex.search("x");

         if (separ > 0) {
            sizey = parseInt(sizex.slice(separ + 1));
            sizex = parseInt(sizex.slice(0, separ));
         } else {
            sizex = parseInt(sizex);
            sizey = sizex;
         }

         if (isNaN(sizex)) sizex = 3;
         if (isNaN(sizey)) sizey = 3;
      }

      if (!sizex) sizex = 3;
      if (!sizey) sizey = sizex;
      this.sizex = sizex;
      this.sizey = sizey;
   }

   JSROOT.GridDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.GridDisplay.prototype.NumGridFrames = function() {
      return this.sizex*this.sizey;
   }

   JSROOT.GridDisplay.prototype.IsSingle = function() {
      return (this.sizex == 1) && (this.sizey == 1);
   }

   JSROOT.GridDisplay.prototype.ForEachFrame = function(userfunc, only_visible) {
      for (var cnt = 0; cnt < this.sizex * this.sizey; ++cnt) {
         var elem = this.IsSingle() ? d3.select("#"+this.frameid) : d3.select("#" + this.frameid + "_grid_" + cnt);

         if (!elem.empty() && elem.attr('frame_title') != '')
            JSROOT.CallBack(userfunc, elem.node());
      }
   }

   JSROOT.GridDisplay.prototype.CreateFrame = function(title) {

      var main = d3.select("#" + this.frameid);
      if (main.empty()) return null;

      var drawid = this.frameid;

      if (!this.IsSingle()) {
         var topid = this.frameid + '_grid';
         if (d3.select("#" + topid).empty()) {
            var rect = main.node().getBoundingClientRect();
            var h = Math.floor(rect.height / this.sizey) - 1;
            var w = Math.floor(rect.width / this.sizex) - 1;

            var content = "<div style='width:100%; height:100%; margin:0; padding:0; border:0; overflow:hidden'>"+
                          "<table id='" + topid + "' style='width:100%; height:100%; table-layout:fixed; border-collapse: collapse;'>";
            var cnt = 0;
            for (var i = 0; i < this.sizey; ++i) {
               content += "<tr>";
               for (var j = 0; j < this.sizex; ++j)
                  content += "<td><div id='" + topid + "_" + cnt++ + "' class='grid_cell'></div></td>";
               content += "</tr>";
            }
            content += "</table></div>";

            main.html(content);
            main.selectAll('.grid_cell').style({ 'width':  w + 'px', 'height': h + 'px', 'overflow' : 'hidden'});
         }

         drawid = topid + "_" + this.cnt;
         if (++this.cnt >= this.sizex * this.sizey) this.cnt = 0;
      }

      JSROOT.cleanup(drawid);

      return d3.select("#" + drawid).html("").attr('frame_title', title).node();
   }

   JSROOT.GridDisplay.prototype.Reset = function() {
      JSROOT.MDIDisplay.prototype.Reset.call(this);
      if (this.IsSingle())
         d3.select("#" + this.frameid).attr('frame_title', null);
      this.cnt = 0;
   }

   JSROOT.GridDisplay.prototype.CheckMDIResize = function(frame_id, size) {

      if (!this.IsSingle()) {
         var main = d3.select("#" + this.frameid);
         var rect = main.node().getBoundingClientRect();
         var h = Math.floor(rect.height / this.sizey) - 1;
         var w = Math.floor(rect.width / this.sizex) - 1;
         main.selectAll('.grid_cell').style({ 'width':  w + 'px', 'height': h + 'px'});
      }

      JSROOT.MDIDisplay.prototype.CheckMDIResize.call(this, frame_id, size);
   }

   // =========================================================================

   JSROOT.RegisterForResize = function(handle, delay) {
      // function used to react on browser window resize event
      // While many resize events could come in short time,
      // resize will be handled with delay after last resize event
      // handle can be function or object with CheckResize function
      // one could specify delay after which resize event will be handled

      if ((handle===null) || (handle === undefined)) return;

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

   JSROOT.addDrawFunc({ name: "TCanvas", icon: "img_canvas", func: JSROOT.Painter.drawCanvas });
   JSROOT.addDrawFunc({ name: "TPad", icon: "img_canvas", func: JSROOT.Painter.drawPad });
   JSROOT.addDrawFunc({ name: "TSlider", icon: "img_canvas", func: JSROOT.Painter.drawPad });
   JSROOT.addDrawFunc({ name: "TFrame", icon: "img_frame", func: JSROOT.Painter.drawFrame });
   JSROOT.addDrawFunc({ name: "TPaveText", icon: "img_pavetext", func: JSROOT.Painter.drawPaveText });
   JSROOT.addDrawFunc({ name: "TPaveStats", icon: "img_pavetext", func: JSROOT.Painter.drawPaveText });
   JSROOT.addDrawFunc({ name: "TPaveLabel", icon: "img_pavelabel", func: JSROOT.Painter.drawPaveText });
   JSROOT.addDrawFunc({ name: "TLatex", icon:"img_text", func: JSROOT.Painter.drawText });
   JSROOT.addDrawFunc({ name: "TMathText", icon:"img_text", func: JSROOT.Painter.drawText });
   JSROOT.addDrawFunc({ name: "TText", icon:"img_text", func: JSROOT.Painter.drawText });
   JSROOT.addDrawFunc({ name: /^TH1/, icon: "img_histo1d", func: JSROOT.Painter.drawHistogram1D, opt:";hist;P;P0;E;E1;E2;same"});
   JSROOT.addDrawFunc({ name: "TProfile", icon: "img_profile", func: JSROOT.Painter.drawHistogram1D, opt:";E0;E1;E2;p;hist"});
   JSROOT.addDrawFunc({ name: "TH2Poly", icon: "img_histo2d" }); // not yet supported
   JSROOT.addDrawFunc({ name: /^TH2/, icon: "img_histo2d", prereq: "more2d", func: "JSROOT.Painter.drawHistogram2D", opt:";COL;COLZ;COL0Z;BOX;SCAT;TEXT;LEGO;LEGO0;LEGO1;LEGO2;LEGO3;LEGO4;same" });
   JSROOT.addDrawFunc({ name: /^TH3/, icon: 'img_histo3d', prereq: "3d", func: "JSROOT.Painter.drawHistogram3D" });
   JSROOT.addDrawFunc({ name: "THStack", prereq: "more2d", func: "JSROOT.Painter.drawHStack" });
   JSROOT.addDrawFunc({ name: "TPolyMarker3D", icon: 'img_histo3d', prereq: "3d", func: "JSROOT.Painter.drawPolyMarker3D" });
   JSROOT.addDrawFunc({ name: "TGraphPolargram" }); // just dummy entry to avoid drawing of this object
   JSROOT.addDrawFunc({ name: /^TGraph/, icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraph", opt:";L;P"});
   JSROOT.addDrawFunc({ name: "TCutG", icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraph", opt:";L;P"});
   JSROOT.addDrawFunc({ name: /^RooHist/, icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraph", opt:";L;P" });
   JSROOT.addDrawFunc({ name: /^RooCurve/, icon:"img_graph", prereq: "more2d", func: "JSROOT.Painter.drawGraph", opt:";L;P" });
   JSROOT.addDrawFunc({ name: "TMultiGraph", icon:"img_mgraph", prereq: "more2d", func: "JSROOT.Painter.drawMultiGraph" });
   JSROOT.addDrawFunc({ name: "TStreamerInfoList", icon:'img_question', func: JSROOT.Painter.drawStreamerInfo });
   JSROOT.addDrawFunc({ name: "TPaletteAxis", icon: "img_colz", prereq: "more2d", func: "JSROOT.Painter.drawPaletteAxis" });
   JSROOT.addDrawFunc({ name: "kind:Text", icon:"img_text", func: JSROOT.Painter.drawRawText });
   JSROOT.addDrawFunc({ name: "TF1", icon: "img_graph", prereq: "math;more2d", func: "JSROOT.Painter.drawFunction" });
   JSROOT.addDrawFunc({ name: "TEllipse", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawEllipse" });
   JSROOT.addDrawFunc({ name: "TLine", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawLine" });
   JSROOT.addDrawFunc({ name: "TArrow", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawArrow" });
   JSROOT.addDrawFunc({ name: "TGaxis", icon: "img_graph", func: JSROOT.drawGaxis });
   JSROOT.addDrawFunc({ name: "TLegend", icon: "img_pavelabel", prereq: "more2d", func: "JSROOT.Painter.drawLegend" });
   JSROOT.addDrawFunc({ name: "TBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox" });
   JSROOT.addDrawFunc({ name: "TWbox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox" });
   JSROOT.addDrawFunc({ name: "TSliderBox", icon: 'img_graph', prereq: "more2d", func: "JSROOT.Painter.drawBox" });
   JSROOT.addDrawFunc({ name: "TGeoVolume", icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", expand: "JSROOT.GEO.expandObject", opt:";more;all;count" });
   JSROOT.addDrawFunc({ name: "TEveGeoShapeExtract", icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", expand: "JSROOT.GEO.expandObject", opt: ";more;all;count"  });
   JSROOT.addDrawFunc({ name: "TGeoManager", icon: 'img_histo3d', prereq: "geom", expand: "JSROOT.GEO.expandObject", func: "JSROOT.Painter.drawGeoObject", opt: ";more;all;count", dflt: "expand" });
   JSROOT.addDrawFunc({ name: /^TGeo/, icon: 'img_histo3d', prereq: "geom", func: "JSROOT.Painter.drawGeoObject", opt: ";more;all;axis;compa;count" });
   // these are not draw functions, but provide extra info about correspondent classes
   JSROOT.addDrawFunc({ name: "kind:Command", icon: "img_execute", execute: true });
   JSROOT.addDrawFunc({ name: "TFolder", icon: "img_folder", icon2: "img_folderopen", noinspect: true, expand: JSROOT.Painter.FolderHierarchy });
   JSROOT.addDrawFunc({ name: "TTask", icon: "img_task", expand: JSROOT.Painter.TaskHierarchy, for_derived: true });
   JSROOT.addDrawFunc({ name: "TTree", icon: "img_tree", noinspect:true, expand: JSROOT.Painter.TreeHierarchy });
   JSROOT.addDrawFunc({ name: "TNtuple", icon: "img_tree", noinspect:true, expand: JSROOT.Painter.TreeHierarchy });
   JSROOT.addDrawFunc({ name: "TBranch", icon: "img_branch", noinspect:true });
   JSROOT.addDrawFunc({ name: /^TLeaf/, icon: "img_leaf", noinspect:true });
   JSROOT.addDrawFunc({ name: "TList", icon: "img_list", noinspect:true, expand: JSROOT.Painter.ListHierarchy });
   JSROOT.addDrawFunc({ name: "TObjArray", icon: "img_list", noinspect:true, expand: JSROOT.Painter.ListHierarchy });
   JSROOT.addDrawFunc({ name: "TClonesArray", icon: "img_list", noinspect:true, expand: JSROOT.Painter.ListHierarchy });
   JSROOT.addDrawFunc({ name: "TColor", icon: "img_color" });
   JSROOT.addDrawFunc({ name: "TFile", icon: "img_file", noinspect:true });
   JSROOT.addDrawFunc({ name: "TMemFile", icon: "img_file", noinspect:true });
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

         if (selector==null) {
            // store found handle in cache, can reuse later
            if (!(kind in JSROOT.DrawFuncs.cache)) JSROOT.DrawFuncs.cache[kind] = h;
            return h;
         } else
         if (typeof selector == 'string') {
            if (!first) first = h;
            // if drawoption specified, check it present in the list

            if (selector == "::expand") {
               if ('expand' in h) return h;
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

         // console.log('Duplicate handle ' + handle.name + ' for derived class ' + si.fName);

         var newhandle = JSROOT.extend({}, handle);
         // delete newhandle.for_derived; // should we disable?
         newhandle.name = si.fName;
         JSROOT.DrawFuncs.lst.push(newhandle);
      }
   }

   // returns array with supported draw options for the specified class
   JSROOT.getDrawOptions = function(kind, selector) {
      if (typeof kind != 'string') return null;
      var allopts = null, isany = false, noinspect = false;
      for (var cnt=0;cnt<1000;++cnt) {
         var h = JSROOT.getDrawHandle(kind, cnt);
         if (h==null) break;
         if (h.noinspect) noinspect = true;
         if (!('func' in h)) break;
         isany = true;
         if (! ('opt' in h)) continue;
         var opts = h.opt.split(';');
         for (var i = 0; i < opts.length; ++i) {
            opts[i] = opts[i].toLowerCase();
            if ((selector=='nosame') && (opts[i].indexOf('same')==0)) continue;

            if (allopts===null) allopts = [];
            if (allopts.indexOf(opts[i])<0) allopts.push(opts[i]);
         }
      }

      if (isany && (allopts===null)) allopts = [""];

      // if no any handle found, let inspect ROOT-based objects
      if (!isany && kind.indexOf("ROOT.")==0) allopts = [];

      if (!noinspect && allopts)
         allopts.push("inspect");

      return allopts;
   }

   JSROOT.canDraw = function(classname) {
      return JSROOT.getDrawOptions("ROOT." + classname) !== null;
   }

   /** @fn JSROOT.draw(divid, obj, opt)
    * Draw object in specified HTML element with given draw options  */

   JSROOT.draw = function(divid, obj, opt) {
      if ((obj===null) || (typeof obj !== 'object')) return null;

      if (opt == 'inspect')
         return JSROOT.Painter.drawInspector(divid, obj);

      var handle = null, painter = null;
      if ('_typename' in obj) handle = JSROOT.getDrawHandle("ROOT." + obj._typename, opt);
      else if ('_kind' in obj) handle = JSROOT.getDrawHandle(obj._kind, opt);

      if ((handle==null) || !('func' in handle)) return null;

      function performDraw() {
         if ((painter===null) && ('painter_kind' in handle))
            painter = (handle.painter_kind == "base") ? new JSROOT.TBasePainter() : new JSROOT.TObjectPainter(obj);

         if (painter==null) return handle.func(divid, obj, opt);

         return handle.func.bind(painter)(divid, obj, opt, painter);
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

      if (funcname.length === 0) return null;

      if (prereq.length >  0) {

         // special handling for painters, which should be loaded via extra scripts
         // such painter get extra last argument - pointer on dummy painter object
         if (handle.painter_kind === undefined)
            handle.painter_kind = (funcname.indexOf("JSROOT.Painter")==0) ? "object" : "base";

         painter = (handle.painter_kind == "base") ? new JSROOT.TBasePainter() : new JSROOT.TObjectPainter(obj);

         JSROOT.AssertPrerequisites(prereq, function() {
            var func = JSROOT.findFunction(funcname);
            if (!func) {
               alert('Fail to find function ' + funcname + ' after loading ' + prereq);
            } else {
               handle.func = func; // remember function once it found

               if (performDraw() !== painter)
                  alert('Painter function ' + funcname + ' do not follow rules of dynamicaly loaded painters');
            }
         });

         return painter;
      }

      // try to find function without prerequisisties
      var func = JSROOT.findFunction(funcname);
      if (func) {
          handle.func = func; // remember function once it is found
          return performDraw();
      }

      return null;
   }

   /** @fn JSROOT.redraw(divid, obj, opt)
    * Redraw object in specified HTML element with given draw options
    * If drawing was not exists, it will be performed with JSROOT.draw.
    * If drawing was already done, that content will be updated */

   JSROOT.redraw = function(divid, obj, opt) {
      if (obj==null) return;

      var dummy = new JSROOT.TObjectPainter();
      dummy.SetDivId(divid, -1);
      var can_painter = dummy.pad_painter();

      if (can_painter !== null) {
         if (obj._typename === "TCanvas") {
            can_painter.RedrawObject(obj);
            return can_painter;
         }

         for (var i = 0; i < can_painter.painters.length; ++i) {
            var painter = can_painter.painters[i];
            if (painter.MatchObjectType(obj._typename))
               if (painter.UpdateObject(obj)) {
                  can_painter.RedrawPad();
                  return painter;
               }
         }
      }

      if (can_painter)
         JSROOT.console("Cannot find painter to update object of type " + obj._typename);

      JSROOT.cleanup(divid);

      return JSROOT.draw(divid, obj, opt);
   }

   // Check resize of drawn element
   // As first argument divid one should use same argment as for the drawing
   // As second argument, one could specify "true" value to force redrawing of
   // the element even after minimal resize of the element
   // Or one just supply object with exact sizes like { width:300, height:200, force:true };

   JSROOT.resize = function(divid, arg) {

      if (arg === true) arg = { force: true }; else
      if (typeof arg !== 'object') arg = null;
      var dummy = new JSROOT.TObjectPainter(), done = false;
      dummy.SetDivId(divid, -1);
      dummy.ForEachPainter(function(painter) {
         if (!done && typeof painter.CheckResize == 'function')
            done = painter.CheckResize(arg);
      });
   }

   // for compatibility, keep old name
   JSROOT.CheckElementResize = JSROOT.resize;

   // safely remove all JSROOT objects from specified element
   JSROOT.cleanup = function(divid) {
      var dummy = new JSROOT.TObjectPainter();
      dummy.SetDivId(divid, -1);
      dummy.ForEachPainter(function(painter) {
         if (typeof painter.Cleanup === 'function')
            painter.Cleanup();
      });
      dummy.select_main().html("");
   }

   // function to display progress message in the left bottom corner
   // previous message will be overwritten
   // if no argument specified, any shown messages will be removed
   JSROOT.progress = function(msg) {
      var id = "jsroot_progressbox";
      var box = d3.select("#"+id);

      if (!JSROOT.gStyle.ProgressBox) return box.remove();

      if ((arguments.length == 0) || (msg === undefined) || (msg === null))
         return box.remove();

      if (box.empty()) {
         box = d3.select(document.body)
                .append("div")
                .attr("id", id);
         box.append("p");
      }

      box.select("p").html(msg);
   }

   return JSROOT;

}));
