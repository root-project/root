/// @file JSRootPainter.js
/// JavaScript ROOT graphics

(function() {

   if (typeof JSROOT != 'object') {
      var e1 = new Error('JSROOT is not defined');
      e1.source = 'JSRootPainter.js';
      throw e1;
   }

   if (typeof d3 != 'object') {
      var e1 = new Error('This extension requires d3.v3.js');
      e1.source = 'JSRootPainter.js';
      throw e1;
   }

   if (typeof JSROOT.Painter == 'object') {
      var e1 = new Error('JSROOT.Painter already defined');
      e1.source = 'JSRootPainter.js';
      throw e1;
   }

   // list of user painters, called with arguments painter(vis, obj, opt)
   JSROOT.fDrawFunc = new Array;

   JSROOT.addDrawFunc = function(_name, _func) {
      JSROOT.fDrawFunc.push({ name:_name, func:_func });
   }

   JSROOT.gStyle = {
      'Tooltip' : true, // tooltip on/off
      'OptimizeDraw' : false, // if true, drawing of 1-D histogram will be
                              // optimized to exclude too-many points
      'AutoStat' : true,
      'OptStat' : 1111,
      'StatNDC' : { fX1NDC : 0.78, fY1NDC: 0.75, fX2NDC: 0.98, fY2NDC: 0.91 },
      'StatText': { fTextAngle: 0, fTextSize: 9, fTextAlign: 12, fTextColor: 1, fTextFont: 42 },
      'StatFill': { fFillColor: 0, fFillStyle: 1001 },
      'TimeOffset' : 788918400000, // UTC time at 01/01/95
      'StatFormat' : function(v) { return (Math.abs(v) < 1e5) ? v.toFixed(5) : v.toExponential(7); },
      'StatEntriesFormat' : function(v) { return (Math.abs(v) < 1e7) ? v.toFixed(0) : v.toExponential(7); }
   };

   /**
    * @class JSROOT.Painter Holder of different functions and classes for drawing
    */
   JSROOT.Painter = {};

   /**
    * @fn menu JSROOT.Painter.createmenu(event, menuname) Creates popup menu
    */
   JSROOT.Painter.createmenu = function(event, menuname) {

      if (!menuname) menuname = "root_ctx_menu";

      var xMousePosition = event.clientX + window.pageXOffset;
      var yMousePosition = event.clientY + window.pageYOffset;

      var x = document.getElementById(menuname);
      if (x) x.parentNode.removeChild(x);

      var d = document.createElement('div');
      d.setAttribute('class', 'ctxmenu');
      d.setAttribute('id', menuname);
      document.body.appendChild(d);
      d.style.left = xMousePosition + "px";
      d.style.top = yMousePosition + "px";
      d.onmouseover = function(e) {
         this.style.cursor = 'pointer';
      }
      d.onclick = function(e) {
         var x = document.getElementById(menuname);
         if (x) x.parentNode.removeChild(x);
      }

      document.body.onclick = function(e) {
         var x = document.getElementById(menuname);
         if (x)
            x.parentNode.removeChild(x);
      }

      return d;
   }

   /**
    * @fn void JSROOT.Painter.menuitem(menu, txt, func) Add item into popup menu
    */
   JSROOT.Painter.menuitem = function(menu, txt, func) {
      var p = document.createElement('p');
      menu.appendChild(p);
      p.onclick = func;
      p.setAttribute('class', 'ctxline');
      p.innerHTML = txt;
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
      var colorMap = new Array('rgb(255, 255, 255)', 'rgb(0, 0, 0)',
            'rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)',
            'rgb(255, 255, 0)', 'rgb(255, 0, 255)', 'rgb(0, 255, 255)',
            'rgb(89, 211, 84)', 'rgb(89, 84, 216)', 'rgb(254, 254, 254)',
            'rgb(191, 181, 173)', 'rgb(76, 76, 76)', 'rgb(102, 102, 102)',
            'rgb(127, 127, 127)', 'rgb(153, 153, 153)', 'rgb(178, 178, 178)',
            'rgb(204, 204, 204)', 'rgb(229, 229, 229)', 'rgb(242, 242, 242)',
            'rgb(204, 198, 170)', 'rgb(204, 198, 170)', 'rgb(193, 191, 168)',
            'rgb(186, 181, 163)', 'rgb(178, 165, 150)', 'rgb(183, 163, 155)',
            'rgb(173, 153, 140)', 'rgb(155, 142, 130)', 'rgb(135, 102, 86)',
            'rgb(175, 206, 198)', 'rgb(132, 193, 163)', 'rgb(137, 168, 160)',
            'rgb(130, 158, 140)', 'rgb(173, 188, 198)', 'rgb(122, 142, 153)',
            'rgb(117, 137, 145)', 'rgb(104, 130, 150)', 'rgb(109, 122, 132)',
            'rgb(124, 153, 209)', 'rgb(127, 127, 155)', 'rgb(170, 165, 191)',
            'rgb(211, 206, 135)', 'rgb(221, 186, 135)', 'rgb(188, 158, 130)',
            'rgb(198, 153, 124)', 'rgb(191, 130, 119)', 'rgb(206, 94, 96)',
            'rgb(170, 142, 147)', 'rgb(165, 119, 122)', 'rgb(147, 104, 112)',
            'rgb(211, 89, 84)');

      var circleColors = [ 632, 416, 600, 400, 616, 432 ];

      var rectangleColors = [ 800, 820, 840, 860, 880, 900 ];

      var set1 = [ 255, 204, 204, 255, 153, 153, 204, 153, 153, 255, 102, 102,
            204, 102, 102, 153, 102, 102, 255, 51, 51, 204, 51, 51, 153, 51,
            51, 102, 51, 51, 255, 0, 0, 204, 0, 0, 153, 0, 0, 102, 0, 0, 51, 0,
            0 ];
      var set2 = [ 204, 255, 204, 153, 255, 153, 153, 204, 153, 102, 255, 102,
            102, 204, 102, 102, 153, 102, 51, 255, 51, 51, 204, 51, 51, 153,
            51, 51, 102, 51, 0, 255, 0, 0, 204, 0, 0, 153, 0, 0, 102, 0, 0, 51,
            0 ];
      var set3 = [ 204, 204, 255, 153, 153, 255, 153, 153, 204, 102, 102, 255,
            102, 102, 204, 102, 102, 153, 51, 51, 255, 51, 51, 204, 51, 51,
            153, 51, 51, 102, 0, 0, 255, 0, 0, 204, 0, 0, 153, 0, 0, 102, 0, 0,
            51 ];
      var set4 = [ 255, 255, 204, 255, 255, 153, 204, 204, 153, 255, 255, 102,
            204, 204, 102, 153, 153, 102, 255, 255, 51, 204, 204, 51, 153, 153,
            51, 102, 102, 51, 255, 255, 0, 204, 204, 0, 153, 153, 0, 102, 102,
            0, 51, 51, 0 ];
      var set5 = [ 255, 204, 255, 255, 153, 255, 204, 153, 204, 255, 102, 255,
            204, 102, 204, 153, 102, 153, 255, 51, 255, 204, 51, 204, 153, 51,
            153, 102, 51, 102, 255, 0, 255, 204, 0, 204, 153, 0, 153, 102, 0,
            102, 51, 0, 51 ];
      var set6 = [ 204, 255, 255, 153, 255, 255, 153, 204, 204, 102, 255, 255,
            102, 204, 204, 102, 153, 153, 51, 255, 255, 51, 204, 204, 51, 153,
            153, 51, 102, 102, 0, 255, 255, 0, 204, 204, 0, 153, 153, 0, 102,
            102, 0, 51, 51 ];

      var circleSets = new Array(set1, set2, set3, set4, set5, set6);

      var set7 = [ 255, 204, 153, 204, 153, 102, 153, 102, 51, 153, 102, 0,
            204, 153, 51, 255, 204, 102, 255, 153, 0, 255, 204, 51, 204, 153,
            0, 255, 204, 0, 255, 153, 51, 204, 102, 0, 102, 51, 0, 153, 51, 0,
            204, 102, 51, 255, 153, 102, 255, 102, 0, 255, 102, 51, 204, 51, 0,
            255, 51, 0 ];
      var set8 = [ 153, 255, 51, 102, 204, 0, 51, 102, 0, 51, 153, 0, 102, 204,
            51, 153, 255, 102, 102, 255, 0, 102, 255, 51, 51, 204, 0, 51, 255,
            0, 204, 255, 153, 153, 204, 102, 102, 153, 51, 102, 153, 0, 153,
            204, 51, 204, 255, 102, 153, 255, 0, 204, 255, 51, 153, 204, 0,
            204, 255, 0 ];
      var set9 = [ 153, 255, 204, 102, 204, 153, 51, 153, 102, 0, 153, 102, 51,
            204, 153, 102, 255, 204, 0, 255, 102, 51, 255, 204, 0, 204, 153, 0,
            255, 204, 51, 255, 153, 0, 204, 102, 0, 102, 51, 0, 153, 51, 51,
            204, 102, 102, 255, 153, 0, 255, 153, 51, 255, 102, 0, 204, 51, 0,
            255, 51 ];
      var set10 = [ 153, 204, 255, 102, 153, 204, 51, 102, 153, 0, 51, 153, 51,
            102, 204, 102, 153, 255, 0, 102, 255, 51, 102, 255, 0, 51, 204, 0,
            51, 255, 51, 153, 255, 0, 102, 204, 0, 51, 102, 0, 102, 153, 51,
            153, 204, 102, 204, 255, 0, 153, 255, 51, 204, 255, 0, 153, 204, 0,
            204, 255 ];
      var set11 = [ 204, 153, 255, 153, 102, 204, 102, 51, 153, 102, 0, 153,
            153, 51, 204, 204, 102, 255, 153, 0, 255, 204, 51, 255, 153, 0,
            204, 204, 0, 255, 153, 51, 255, 102, 0, 204, 51, 0, 102, 51, 0,
            153, 102, 51, 204, 153, 102, 255, 102, 0, 255, 102, 51, 255, 51, 0,
            204, 51, 0, 255 ];
      var set12 = [ 255, 51, 153, 204, 0, 102, 102, 0, 51, 153, 0, 51, 204, 51,
            102, 255, 102, 153, 255, 0, 102, 255, 51, 102, 204, 0, 51, 255, 0,
            51, 255, 153, 204, 204, 102, 153, 153, 51, 102, 153, 0, 102, 204,
            51, 153, 255, 102, 204, 255, 0, 153, 204, 0, 153, 255, 51, 204,
            255, 0, 153 ];

      var rectSets = new Array(set7, set8, set9, set10, set11, set12);

      /*
       * Define circle colors
       */
      for (var i = 0; i < 6; i++) {
         for (var j = 0; j < 15; j++) {
            var colorn = circleColors[i] + j - 10;
            colorMap[colorn] = 'rgb(' + circleSets[i][3 * j] + ', '
                  + circleSets[i][3 * j + 1] + ', ' + circleSets[i][3 * j + 2] + ')';
            colorn = rectangleColors[i] + j - 9;
            colorMap[colorn] = 'rgb(' + rectSets[i][3 * j] + ', '
                  + rectSets[i][3 * j + 1] + ', ' + rectSets[i][3 * j + 2] + ')';
         }
      }
      return colorMap;
   }();

   JSROOT.Painter.adoptRootColors = function(objarr) {
      if (!objarr || !objarr.arr) return;

      for (var n in objarr.arr) {
         var col = objarr.arr[n];
         if ((col==null) || (col['_typename'] != 'TColor')) continue;

         var num = col.fNumber;
         if ((num<0) || (num>4096)) continue;

         var rgb = "rgb(" + (col.fRed*255).toFixed(0) + ", " + (col.fGreen*255).toFixed(0) + ", " + (col.fBlue*255).toFixed(0) + ")";

         while (num>JSROOT.Painter.root_colors.length)
            JSROOT.Painter.root_colors.push(rgb);

         if (JSROOT.Painter.root_colors[num] != rgb) {
             // console.log("Replace color "+ num + " " + rgb);
            JSROOT.Painter.root_colors[num] = rgb;
         }
      }
   }

   JSROOT.Painter.root_line_styles = new Array("", "", "3, 3", "1, 2",
         "3, 4, 1, 4", "5, 3, 1, 3", "5, 3, 1, 3, 1, 3, 1, 3", "5, 5",
         "5, 3, 1, 3, 1, 3", "20, 5", "20, 10, 1, 10", "1, 2");

   // Initialize ROOT markers
   JSROOT.Painter.root_markers = new Array('fcircle', 'fcircle', 'fcross',
         'dcross', 'ocircle', 'gcross', 'fcircle', 'fcircle', 'fcircle',
         'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle',
         'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle', 'fcircle',
         'fsquare', 'ftriangle-up', 'ftriangle-down', 'ocircle', 'osquare',
         'otriangle-up', 'odiamond', 'ocross', 'fstar', 'ostar', 'dcross',
         'otriangle-down', 'fdiamond', 'fcross');

   /**
    * Function returns the SVG symbol type identifier for a given root matker
    * The result is an array with 3 elements: the first is the identifier of the
    * root marker in the SVG symbols the second is true if the shape is filled
    * and false if it is open the third is true if the shape should be rotated
    * The identifier will be 6 if the shape is a star or 7 if it is '*'
    */
   JSROOT.Painter.getRootMarker = function(i) {
      var marker = JSROOT.Painter.root_markers[i];

      var res = { shape: 0, toFill: true, toRotate: false };

      if (typeof (marker) != 'undefined') {
         switch (marker.charAt(0)) {
            case 'd': res.shape = 7; return res;
            case 'o': res.toFill = false; break;
            case 'g': res.toRotate = true; break;
         }

         switch (marker.substr(1)) {
           case "circle":  res.shape = 0; break;
           case "cross":   res.shape = 1; break;
           case "diamond": res.shape = 2; break;
           case "square":  res.shape = 3; break;
           case "triangle-up": res.shape = 4; break;
           case "triangle-down": res.shape = 5; break;
           case "star":    res.shape = 6; break;
         }
      }
      return res;
   }

   JSROOT.Painter.clearCuts = function(chopt) {
      /* decode string "chopt" and remove graphical cuts */
      var left = chopt.indexOf('[');
      if (left == -1)
         return chopt;
      var right = chopt.indexOf(']');
      if (right == -1)
         return chopt;
      var nch = right - left;
      if (nch < 2)
         return chopt;
      for (var i = 0; i <= nch; i++)
         chopt[left + i] = ' ';
      return chopt;
   }

   JSROOT.Painter.root_fonts = new Array('Arial', 'Times New Roman',
         'bold Times New Roman', 'bold italic Times New Roman', 'Arial',
         'oblique Arial', 'bold Arial', 'bold oblique Arial', 'Courier New',
         'oblique Courier New', 'bold Courier New', 'bold oblique Courier New',
         'Symbol', 'Times New Roman', 'Wingdings', 'Symbol');

   JSROOT.Painter.getFontDetails = function(fontIndex) {

      var fontName = JSROOT.Painter.root_fonts[Math.floor(fontIndex / 10)];

      var weight = null;
      var style = null;
      var name = "Arial";

      if (fontName == null)
         fontName = "";

      if (fontName.indexOf("bold") != -1) {
         weight = "bold";
         // The first 5 characters are removed because "bold " is always first
         // when it occurs
         fontName = fontName.substring(5, fontName.length);
      }
      if (fontName.charAt(0) == 'i') {
         style = "italic";
         fontName = fontName.substring(7, fontName.length);
      } else if (fontName.charAt(0) == 'o') {
         style = "oblique";
         fontName = fontName.substring(8, fontName.length);
      }
      if (name == 'Symbol') {
         weight = null;
         style = null;
      }
      return {
         'weight' : weight,
         'style' : style,
         'name' : fontName
      };
   }

   JSROOT.Painter.createFillPattern = function(svg, pattern, color) {
      // create fill pattern - only if they don't exists yet

      if ((pattern == 0) || (color == 0)) return "none";
      if ((pattern >= 4000) && (pattern <= 4100)) return "none";

      if ((pattern < 3000) || (pattern>3025)) return JSROOT.Painter.root_colors[color];

      var id = "pat" + pattern + "_" + color;

      if (svg.attr("id") != null)
         id = svg.attr("id") + "_" + id;

      if (document.getElementById(id) != null) return "url(#" + id + ")";

      var line_color = JSROOT.Painter.root_colors[color];

      switch (pattern) {
      case 3001:
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits","userSpaceOnUse")
               .attr("width", "3px").attr("height", "2px").style("stroke", line_color)
            .append('svg:rect')
               .attr("x", 0).attr("y", 0).attr("width", 1).attr("height", 1).style("stroke",line_color)
            .append('svg:rect')
               .attr("x", 2).attr("y", 0).attr("width", 1).attr("height", 1).style("stroke", line_color)
            .append('svg:rect')
               .attr("x", 1).attr("y", 1).attr("width", 1).attr("height", 1).style("stroke", line_color);
         break;
      case 3002:
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits", "userSpaceOnUse")
               .attr("width", "4px").attr("height", "2px").style("stroke", line_color)
            .append('svg:rect')
               .attr("x", 1).attr("y", 0).attr("width", 1).attr("height", 1).style("stroke", line_color)
            .append('svg:rect')
               .attr("x", 3).attr("y", 1).attr("width", 1).attr("height", 1).style("stroke", line_color);
         break;
      case 3003:
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits", "userSpaceOnUse")
               .attr("width", "4px").attr("height", "4px").style("stroke", line_color)
            .append('svg:rect')
               .attr("x", 2).attr("y", 1).attr("width", 1).attr("height", 1).style("stroke", line_color)
            .append('svg:rect')
               .attr("x", 0).attr("y", 3).attr("width", 1).attr("height", 1).style("stroke", line_color);
         break;
      case 3004:
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits", "userSpaceOnUse")
               .attr("width", "8px").attr("height", "8px").style("stroke", line_color)
            .append("svg:line")
               .attr("x1", 8).attr("y1", 0).attr("x2", 0).attr("y2", 8)
               .style("stroke",line_color).style("stroke-width", 1);
         break;
      case 3005:
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits", "userSpaceOnUse")
               .attr("width", "8px").attr("height", "8px").style("stroke", line_color)
            .append("svg:line")
               .attr("x1", 0).attr("y1", 0).attr("x2", 8).attr("y2", 8)
               .style("stroke",line_color).style("stroke-width", 1);
         break;
      case 3006:
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits", "userSpaceOnUse")
               .attr("width", "4px").attr("height", "4px").style("stroke", line_color)
            .append("svg:line")
               .attr("x1", 1).attr("y1", 0).attr("x2", 1).attr("y2", 3)
               .style("stroke",line_color).style("stroke-width", 1);
         break;
      case 3007:
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits","userSpaceOnUse")
               .attr("width", "4px").attr("height", "4px").style("stroke", line_color)
            .append("svg:line")
               .attr("x1", 0).attr("y1", 1).attr("x2", 3).attr("y2", 1)
               .style("stroke",line_color).style("stroke-width", 1);
         break;
      default: /* == 3004 */
         svg.append('svg:pattern')
               .attr("id", id).attr("patternUnits","userSpaceOnUse")
               .attr("width", "8px").attr("height", "8px").style("stroke", line_color)
            .append("svg:line")
               .attr("x1", 8).attr("y1", 0).attr("x2", 0).attr("y2", 8)
               .style("stroke",line_color).style("stroke-width", 1);
         break;
      }
      return "url(#" + id + ")";
   }

   JSROOT.Painter.padtoX = function(pad, x) {
      // Convert x from pad to X.
      if (pad['fLogx'] && x < 50)
         return Math.exp(2.302585092994 * x);
      return x;
   }

   JSROOT.Painter.padtoY = function(pad, y) {
      // Convert y from pad to Y.
      if (pad['fLogy'] && y < 50)
         return Math.exp(2.302585092994 * y);
      return y;
   }

   JSROOT.Painter.xtoPad = function(x, pad) {
      if (pad['fLogx']) {
         if (x > 0)
            x = JSROOT.Math.log10(x);
         else
            x = pad['fUxmin'];
      }
      return x;
   }

   JSROOT.Painter.moveChildToEnd = function(child) {
      if (!child) return;
      var prnt = child.node().parentNode;
      prnt.removeChild(child.node());
      prnt.appendChild(child.node());
   }


   JSROOT.Painter.ytoPad = function(y, pad) {
      if (pad['fLogy']) {
         if (y > 0)
            y = JSROOT.Math.log10(y);
         else
            y = pad['fUymin'];
      }
      return y;
   };

   /**
    * Converts an HSL color value to RGB. Conversion formula adapted from
    * http://en.wikipedia.org/wiki/HSL_color_space. Assumes h, s, and l are
    * contained in the set [0, 1] and returns r, g, and b in the set [0, 255].
    *
    * @param Number
    *           h The hue
    * @param Number
    *           s The saturation
    * @param Number
    *           l The lightness
    * @return Array The RGB representation
    */
   JSROOT.Painter.HLStoRGB = function(h, l, s) {
      var r, g, b;
      if (s < 1e-300) {
         r = g = b = l; // achromatic
      } else {
         function hue2rgb(p, q, t) {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
         }
         var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
         var p = 2 * l - q;
         r = hue2rgb(p, q, h + 1 / 3);
         g = hue2rgb(p, q, h);
         b = hue2rgb(p, q, h - 1 / 3);
      }
      return 'rgb(' + Math.round(r * 255) + ', ' + Math.round(g * 255) + ', ' + Math.round(b * 255) + ')';
   }

   JSROOT.Painter.chooseTimeFormat = function(range, nticks) {
      if (nticks < 1) nticks = 1;
      var awidth = range / nticks;
      var reasformat = 0;

      // code from TAxis::ChooseTimeFormat
      // width in seconds ?
      if (awidth >= .5) {
         reasformat = 1;
         // width in minutes ?
         if (awidth >= 30) {
            awidth /= 60;  reasformat = 2;
            // width in hours ?
            if (awidth >= 30) {
               awidth /= 60;   reasformat = 3;
               // width in days ?
               if (awidth >= 12) {
                  awidth /= 24; reasformat = 4;
                  // width in months ?
                  if (awidth >= 15.218425) {
                     awidth /= 30.43685; reasformat = 5;
                     // width in years ?
                     if (awidth >= 6) {
                        awidth /= 12; reasformat = 6;
                        if (awidth >= 2) {
                           awidth /= 12; reasformat = 7;
                        }
                     }
                  }
               }
            }
         }
      }

      switch (reasformat) {
        case 0: return "%S";
        case 1: return "%Mm%S";
        case 2: return "%Hh%M";
        case 3: return "%d-%Hh";
        case 4: return "%d/%m";
        case 5: return "%d/%m/%y";
        case 6: return "%d/%m/%y";
        case 7: return "%m/%y";
      }

      return "%Y";
   }

   JSROOT.Painter.getTimeFormat = function(axis) {
      var timeFormat = axis['fTimeFormat'];
      var idF = timeFormat.indexOf('%F');
      if (idF >= 0)
         return timeFormat.substr(0, idF);
      return timeFormat;
   }

   JSROOT.Painter.getTimeOffset = function(axis) {
      var timeFormat = axis['fTimeFormat'];

      var idF = timeFormat.indexOf('%F');

      if (idF >= 0) {
         var lnF = timeFormat.length;
         var stringtimeoffset = timeFormat.substr(idF + 2, lnF);
         for (var i = 0; i < 3; ++i)
            stringtimeoffset = stringtimeoffset.replace('-', '/');
         // special case, used from DABC painters
         if ((stringtimeoffset == "0") || (stringtimeoffset == "")) return 0;

         var stimeoffset = new Date(stringtimeoffset);
         var timeoffset = stimeoffset.getTime();
         var ids = stringtimeoffset.indexOf('s');
         if (ids >= 0) {
            var lns = stringtimeoffset.length;
            var sdp = stringtimeoffset.substr(ids + 1, lns);
            var dp = parseFloat(sdp);
            timeoffset += dp;
         }
         return timeoffset;
      }

      return JSROOT.gStyle['TimeOffset'];
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
      '#odot' : ''
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
      return str;
   }

   JSROOT.Painter.stringWidth = function(svg, line, font_size, fontDetails) {
      /* compute the bounding box of a string by using temporary svg:text */
      var text = svg.append("svg:text")
                .attr("class", "temp_text")
                .attr("xml:space","preserve")
                .attr("font-family", fontDetails['name'])
                .attr("font-weight", fontDetails['weight'])
                .attr("font-style", fontDetails['style'])
                .attr("font-size", font_size)
                .style("opacity", 0)
                .text(line);
      var w = text.node().getBBox().width;
      text.remove();
      return w;
   }

   // ==============================================================================

   JSROOT.TBasePainter = function() {
   }

   JSROOT.TBasePainter.prototype.Cleanup = function() {
      // generic method to cleanup painter
   }

   JSROOT.TBasePainter.prototype.GetObject = function() {
      return null;
   }

   JSROOT.TBasePainter.prototype.UpdateObject = function(obj) {
      return false;
   }

   JSROOT.TBasePainter.prototype.RedrawPad = function(resize) {
   }

   JSROOT.TBasePainter.prototype.RedrawObject = function(obj) {
      if (this.UpdateObject(obj))
         this.RedrawPad();
   }

   JSROOT.TBasePainter.prototype.RedrawFrame = function() {
      // obsolete, should not be used
      this.RedrawPad();
   }

   JSROOT.TBasePainter.prototype.CheckResize = function(force) {
   }

   // ==============================================================================

   JSROOT.TObjectPainter = function(obj) {
      JSROOT.TBasePainter.call(this);
      this.obj_typename = (obj!=null) && ('_typename' in obj) ? obj['_typename'] : "";
      this.draw_g = null; // container for all draw objects
      this.pad_name = ""; // name of pad where object is drawn
      this.main = null;  // main painter, received from pad
   }

   JSROOT.TObjectPainter.prototype = Object.create(JSROOT.TBasePainter.prototype);

   JSROOT.TObjectPainter.prototype.CheckResize = function(force) {
      // no canvas - no resize
      var can = this.svg_canvas();

      var pad_painter = can ? can['pad_painter'] : null;

      if (pad_painter) pad_painter.CheckCanvasResize();
   }

   JSROOT.TObjectPainter.prototype.RemoveDrawG = function() {
      // generic method to delete all graphical elements, associated with
      // painter may not work for all cases

      if (this.draw_g != null) {
         this.draw_g.remove();
         this.draw_g = null;
      }
   }

   JSROOT.TObjectPainter.prototype.RecreateDrawG = function(take_pad) {
      //this.RemoveDrawG();

      if (this.draw_g)
         this.draw_g.selectAll("*").remove();

      if (take_pad) {
         if (!this.draw_g)
            this.draw_g = this.svg_pad(true).append("svg:g");
      } else {
         var frame = this.svg_frame(true);

         var w = frame.attr("width");
         var h = frame.attr("height");

         if (!this.draw_g)
            this.draw_g = frame.append("svg");

         this.draw_g.attr("x", 0)
                    .attr("y", 0)
                    .attr("width",w)
                    .attr("height", h)
                    .attr("viewBox", "0 0 " + w + " " + h);
      }
   }

   /** This is main graphical SVG element, where all Canvas drawing are performed */
   JSROOT.TObjectPainter.prototype.svg_canvas = function(asselect) {
      var res = d3.select("#" + this.divid + " .root_canvas");
      return asselect ? res : res.node();
   }

   /** This is SVG element, correspondent to current pad */
   JSROOT.TObjectPainter.prototype.svg_pad = function(asselect) {
      var c = this.svg_canvas(true);
      if (this.pad_name != '')
         c = c.select("[pad=" + this.pad_name + ']');
      return asselect ? c : c.node();
   }

   JSROOT.TObjectPainter.prototype.root_pad = function() {
      var p = this.svg_pad();
      var pad_painter = p ? p['pad_painter'] : null;
      return pad_painter ? pad_painter.pad : null;
   }

   /** This is SVG element with current frame */
   JSROOT.TObjectPainter.prototype.svg_frame = function(asselect) {
      var f = this.svg_pad(true).select(".root_frame");
      return asselect ? f : f.node();
   }

   /** Returns main pad painter - normally TH1/TH2 painter, which draws all axis */
   JSROOT.TObjectPainter.prototype.main_painter = function() {
      if (!this.main) {
         var svg_p = this.svg_pad();
         if (svg_p) this.main = svg_p['mainpainter'];
      }
      return this.main;
   }

   JSROOT.TObjectPainter.prototype.is_main_painter = function() {
      return this == this.main_painter();
   }

   JSROOT.TObjectPainter.prototype.SetDivId = function(divid, is_main) {
      // assign all basic graphic elements like canvas, pad, frame
      // create canvas and frame if required
      // is_main - -1 - not add to painters list,
      //            0 - normal painter,
      //            1 - major objects like TH1/TH2

      this['divid'] = divid;

      if (is_main == null) is_main = 0;

      this['create_canvas'] = false;

      // SVG element where canvas is drawn
      var svg_c = this.svg_canvas();

      if ((svg_c==null) && (is_main>0)) {
         JSROOT.Painter.drawCanvas(divid, null);
         svg_c = this.svg_canvas();
         this['create_canvas'] = true;
      }

      if (svg_c == null) {
         if ((this.obj_typename!="TCanvas") && (is_main>=0))
            console.log("Canvas not exists when trying to draw " + this.obj_typename);
         return;
      }

      // SVG element where current pad is drawn (can be canvas itself)
      this.pad_name = svg_c['current_pad'];

      if (is_main < 0) return;

      // create TFrame element if not exists when
      if ((is_main > 0) && (this.svg_frame()==null)) {
         JSROOT.Painter.drawFrame(divid, null);
         if (this.svg_frame()==null) return alert("Fail to draw dummy TFrame");
         this['create_canvas'] = true;
      }

      var svg_p = this.svg_pad();
      svg_p['pad_painter'].painters.push(this);

      if ((is_main > 0) && (svg_p['mainpainter']==null))
         // when this is first main painter in the pad
         svg_p['mainpainter'] = this;
   }

   JSROOT.TObjectPainter.prototype.Cleanup = function() {
      // generic method to cleanup painters
      $("#" + this.divid).empty();
   }

   JSROOT.TObjectPainter.prototype.RedrawPad = function(resize) {
      // call Redraw methods for each painter in the frame
      // if selobj specified, painter with selected object will be redrawn

      var pad = this.svg_pad();

      var pad_painter = pad ? pad['pad_painter'] : null;

      if (pad_painter) pad_painter.Redraw(true);
   }

   JSROOT.TObjectPainter.prototype.RemoveDrag = function(id) {
      var drag_rect_name = id + "_drag_rect";
      var resize_rect_name = id + "_resize_rect";
      if (this[drag_rect_name]) {
         this[drag_rect_name].remove();
         this[drag_rect_name] = null;
      }
      if (this[resize_rect_name]) {
         this[resize_rect_name].remove();
         this[resize_rect_name] = null;
      }
   }

   JSROOT.TObjectPainter.prototype.AddDrag = function(id, main_rect, callback) {
      var pthis = this;

      var drag_rect_name = id + "_drag_rect";
      var resize_rect_name = id + "_resize_rect";

      var istitle = (main_rect.node().tagName == "text");

      var rect_width = function() {
         if (istitle)
            return main_rect.node().getBBox().width;
         else
            return Number(main_rect.attr("width"));
      }

      var rect_height = function() {
         if (istitle)
            return main_rect.node().getBBox().height;
         else
            return Number(main_rect.attr("height"));
      }

      var rect_x = function() {
         var x = Number(main_rect.attr("x"));
         if (istitle)
            x -= rect_width() / 2;
         return x;
      }

      var rect_y = function() {
         var y = Number(main_rect.attr("y"));
         if (istitle)
            y -= rect_height();
         return y;
      }

      var acc_x = 0, acc_y = 0, pad_w = 1, pad_h = 1;

      var drag_move = d3.behavior.drag().origin(Object)
         .on("dragstart",  function() {
            d3.event.sourceEvent.preventDefault();

            acc_x = 0; acc_y = 0;
            pad_w = Number(pthis.svg_pad(true).attr("width")) - rect_width();
            pad_h = Number(pthis.svg_pad(true).attr("height")) - rect_height();

            pthis[drag_rect_name] =
               pthis.svg_pad(true)
                 .append("rect")
                 .attr("class", "zoom")
                 .attr("id", drag_rect_name)
                 .attr("x",  rect_x())
                 .attr("y", rect_y())
                 .attr("width", rect_width())
                 .attr("height", rect_height())
                 .style("cursor", "move");
          }).on("drag", function() {
               d3.event.sourceEvent.preventDefault();

               var x = Number(pthis[drag_rect_name].attr("x"));
               var y = Number(pthis[drag_rect_name].attr("y"));
               var dx = d3.event.dx, dy = d3.event.dy;

               if (((acc_x<0) && (dx>0)) || ((acc_x>0) && (dx<0))) { acc_x += dx; dx = 0; }
               if (((acc_y<0) && (dy>0)) || ((acc_y>0) && (dy<0))) { acc_y += dy; dy = 0; }

               if ((x + dx < 0) || (x +dx > pad_w)) acc_x += dx; else x+=dx;
               if ((y+dy < 0) || (y+dy > pad_h)) acc_y += dy; else y += dy;

               pthis[drag_rect_name].attr("x", x);
               pthis[drag_rect_name].attr("y", y);

               JSROOT.Painter.moveChildToEnd(pthis[drag_rect_name]);

               d3.event.sourceEvent.stopPropagation();
          }).on("dragend", function() {
               d3.event.sourceEvent.preventDefault();

               pthis[drag_rect_name].style("cursor", "auto");

               var x = Number(pthis[drag_rect_name].attr("x"));
               var y = Number(pthis[drag_rect_name].attr("y"));

               var dx = x - rect_x();
               var dy = y - rect_y();

               pthis[drag_rect_name].remove();
               pthis[drag_rect_name] = null;

               if (istitle) {
                  main_rect.attr("x", x + rect_width() / 2).attr("y", y + rect_height());
               } else {
                  main_rect.attr("x", x).attr("y", y);
                  pthis[resize_rect_name]
                     .attr("x", x + rect_width() - 20)
                     .attr("y", y + rect_height() - 20);
               }

               callback.move(x, y, dx, dy);

               // do it after call-back - rectangle has correct coordinates
               if (istitle)
                  pthis[resize_rect_name]
                      .attr("x", rect_x() + rect_width() - 20)
                      .attr("y", rect_y() + rect_height() - 20);
            });

      var drag_resize = d3.behavior.drag().origin(Object)
        .on( "dragstart", function() {
           d3.event.sourceEvent.preventDefault();
           acc_x = 0; acc_y = 0;
           pad_w = Number(pthis.svg_pad(true).attr("width")) - rect_x();
           pad_h = Number(pthis.svg_pad(true).attr("height")) - rect_y();
           pthis[drag_rect_name] =
              pthis.svg_pad(true)
                .append("rect")
                .attr("class", "zoom")
                .attr("id", drag_rect_name)
                .attr("x",  rect_x())
                .attr("y", rect_y())
                .attr("width", rect_width())
                .attr("height", rect_height())
                .style("cursor", "se-resize");
           // main_rect.style("cursor", "move");
         }).on("drag", function() {
            d3.event.sourceEvent.preventDefault();

            var w = Number(pthis[drag_rect_name].attr("width"));
            var h = Number(pthis[drag_rect_name].attr("height"));
            var dx = d3.event.dx, dy = d3.event.dy;
            if ((acc_x>0) && (dx<0)) { acc_x += dx; dx = 0; }
            if ((acc_y>0) && (dy<0)) { acc_y += dy; dy = 0; }
            if (w+dx > pad_w) acc_x += dx; else w+=dx;
            if (h+dy > pad_h) acc_y += dy; else h+=dy;
            pthis[drag_rect_name].attr("width", w);
            pthis[drag_rect_name].attr("height", h);

            JSROOT.Painter.moveChildToEnd(pthis[drag_rect_name]);

            d3.event.sourceEvent.stopPropagation();
         }).on( "dragend", function() {
            d3.event.sourceEvent.preventDefault();
            pthis[drag_rect_name].style("cursor", "auto");

            var newwidth = Number(pthis[drag_rect_name].attr("width"));
            var newheight = Number(pthis[drag_rect_name].attr("height"));

            pthis[drag_rect_name].remove();
            pthis[drag_rect_name] = null;

            callback.resize(newwidth, newheight);

            // do it after call-back - rectangle has correct coordinates
            if (istitle)
               pthis[resize_rect_name]
                .attr("x", rect_x() + rect_width() - 20)
                .attr("y", rect_y() + rect_height() - 20);
         });

      if (this[resize_rect_name] == null) {

         main_rect.call(drag_move);

         this[resize_rect_name] =
             this.svg_pad(true).append("rect")
                            .attr("id", resize_rect_name)
                            .style("opacity", "0")
                            .style("cursor", "se-resize")
                            .call(drag_resize);
      } else {
         // ensure that small resize rect appears after large move rect
         var prnt = this[resize_rect_name].node().parentNode;

         // first move small resize rec before main_rect
         prnt.removeChild(this[resize_rect_name].node());
         prnt.insertBefore(this[resize_rect_name].node(), main_rect.node());
         // than swap them while big rect should be in the front
         prnt.removeChild(main_rect.node());
         prnt.insertBefore(main_rect.node(), this[resize_rect_name].node());
      }

      this[resize_rect_name]
            .attr("x", rect_x() + rect_width() - 20)
            .attr("y", rect_y() + rect_height() - 20)
            .attr("width", 20)
            .attr("height", 20);
   }

   JSROOT.TObjectPainter.prototype.FindPainterFor = function(selobj,selname) {
      // try to find painter for sepcified object
      // can be used to find painter for some special objects, registered as
      // histogram functions

      var painters = this.svg_pad() ? this.svg_pad().pad_painter.painters : null;
      if (painters == null) return null;

      for (var n in painters) {
         var pobj = painters[n].GetObject();
         if (pobj==null) continue;

         if (selobj && (pobj === selobj)) return painters[n];

         if (selname && ('fName' in pobj) && (pobj['fName']==selname)) return painters[n];
      }

      return null;
   }

   JSROOT.TObjectPainter.prototype.Redraw = function() {
      // basic method, should be reimplemented in all derived objects
      // for the case when drawing should be repeated, probably with different
      // options
   }

   // ===========================================================

   JSROOT.TFramePainter = function(tframe) {
      JSROOT.TObjectPainter.call(this, tframe);
      this.tframe = tframe;
   }

   JSROOT.TFramePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TFramePainter.prototype.GetObject = function() {
      return this.tframe;
   }

   JSROOT.TFramePainter.prototype.Shrink = function(shrink_left, shrink_right) {
      var ndc = this.svg_frame() ? this.svg_frame()['NDC'] : null;
      if (ndc) {
         ndc.x1 += shrink_left;
         ndc.x2 -= shrink_right;
      }
   }

   JSROOT.TFramePainter.prototype.DrawFrameSvg = function() {
      var width = Number(this.svg_pad(true).attr("width")),
          height = Number(this.svg_pad(true).attr("height"));
      var w = width, h = height;

      var ndc = this.svg_frame() ? this.svg_frame()['NDC'] : null;
      if (ndc == null) ndc = { x1 : 0.07, y1 : 0.12, x2 : 0.95, y2 : 0.88 };

      var root_pad = this.root_pad();

      var lm = width * ndc.x1;
      var rm = width * (1 - ndc.x2);
      var tm = height * ndc.y1;
      var bm = height * (1 - ndc.y2);

      var framecolor = 'white', bordermode = 0, bordersize = 0, linecolor = 'black', linestyle = 0, linewidth = 1;

      if (this.tframe) {
         bordermode = this.tframe['fBorderMode'];
         bordersize = this.tframe['fBorderSize'];
         linecolor = JSROOT.Painter.root_colors[this.tframe['fLineColor']];
         linestyle = this.tframe['fLineStyle'];
         linewidth = this.tframe['fLineWidth'];
         if (root_pad) {
            var xspan = width / Math.abs(root_pad['fX2'] - root_pad['fX1']);
            var yspan = height / Math.abs(root_pad['fY2'] - root_pad['fY1']);
            var px1 = (this.tframe['fX1'] - root_pad['fX1']) * xspan;
            var py1 = (this.tframe['fY1'] - root_pad['fY1']) * yspan;
            var px2 = (this.tframe['fX2'] - root_pad['fX1']) * xspan;
            var py2 = (this.tframe['fY2'] - root_pad['fY1']) * yspan;
            var pxl, pxt, pyl, pyt;
            if (px1 < px2) { pxl = px1; pxt = px2; }
                      else { pxl = px2; pxt = px1; }
            if (py1 < py2) { pyl = py1; pyt = py2; }
                      else { pyl = py2; pyt = py1; }
            lm = pxl;
            bm = pyl;
            w = pxt - pxl;
            h = pyt - pyl;
            tm = height - pyt;
            rm = width - pxt;
         } else {
            lm = this.tframe['fX1'] * width;
            tm = this.tframe['fY1'] * height;
            bm = (1.0 - this.tframe['fY2']) * height;
            rm = (1.0 - this.tframe['fX2'] + shrink_right) * width;
            w -= (lm + rm);
            h -= (tm + bm);
         }
         framecolor = JSROOT.Painter.root_colors[this.tframe['fFillColor']];
         if (this.tframe['fFillStyle'] > 4000 && this.tframe['fFillStyle'] < 4100)
            framecolor = 'none';
      } else {
         if (root_pad) {
            framecolor = JSROOT.Painter.root_colors[root_pad['fFrameFillColor']];
            if (root_pad['fFrameFillStyle'] > 4000 && root_pad['fFrameFillStyle'] < 4100)
               framecolor = 'none';
         }
         w -= (lm + rm);
         h -= (tm + bm);
      }
      if (typeof (framecolor) == 'undefined')
         framecolor = 'white';

      // this is svg:g object - container for every other items belonging to frame
      var frame_g = this.svg_pad(true).select(".root_frame");

      if (frame_g.node() == null)
         frame_g = this.svg_pad(true).append("svg:g").attr("class", "root_frame");

      // calculate actual NDC coordinates, use them to properly locate PALETTE
      frame_g.node()['NDC'] = {
         x1 : lm / width,
         x2 : (lm + w) / width,
         y1 : tm / height,
         y2 : (tm + h) / height
      };

      // simple workaround to access painter via frame container
      frame_g.node()['frame_painter'] = this;

      lm = Math.round(lm); tm = Math.round(tm);
      w = Math.round(w); h = Math.round(h);

      frame_g.attr("x", lm).attr("y", tm)
             .attr("width", w).attr("height", h)
             .attr("transform", "translate(" + lm + "," + tm + ")");

      var top_rect = frame_g.select("rect");
      if (top_rect.node() == null)
         top_rect = frame_g.append("svg:rect");

      top_rect.attr("x", 0).attr("y", 0)
              .attr("width", w).attr("height", h)
              .attr("fill", framecolor)
              .style("stroke", linecolor)
              .style("stroke-width", linewidth);
   }

   JSROOT.TFramePainter.prototype.Redraw = function() {
      this.DrawFrameSvg();
   }

   JSROOT.Painter.drawFrame = function(divid, obj) {
      var p = new JSROOT.TFramePainter(obj);
      p.SetDivId(divid);
      p.DrawFrameSvg();
      return p;
   }

   // =========================================================================

   JSROOT.TF1Painter = function(tf1) {
      JSROOT.TObjectPainter.call(this, tf1);
      this.tf1 = tf1;
   }

   JSROOT.TF1Painter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TF1Painter.prototype.GetObject = function() {
      return this.tf1;
   }

   JSROOT.TF1Painter.prototype.Redraw = function() {
      this.DrawBins();
   }

   JSROOT.TF1Painter.prototype.Eval = function(x) {
      return this.tf1.evalPar(x);
   }

   JSROOT.TF1Painter.prototype.CreateDummyHisto = function() {
      var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
      if (this.tf1['fNsave'] > 0) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         var nb_points = this.tf1['fNpx'];
         for (var i = 0; i < nb_points; ++i) {
            var h = this.tf1['fSave'][i];
            if ((i == 0) || (h > ymax))
               ymax = h;
            if ((i == 0) || (h < ymin))
               ymin = h;
         }
         xmin = this.tf1['fSave'][nb_points + 1];
         xmax = this.tf1['fSave'][nb_points + 2];
      } else {
         // we don't have the points, so let's try to interpret the function
         // use fNpfits instead of fNpx if possible (to use more points)
         if (this.tf1['fNpfits'] <= 103)
            this.tf1['fNpfits'] = 103;
         xmin = this.tf1['fXmin'];
         xmax = this.tf1['fXmax'];

         var nb_points = Math.max(this.tf1['fNpx'], this.tf1['fNpfits']);

         var binwidthx = (xmax - xmin) / nb_points;
         var left = -1, right = -1;
         for (var i = 0; i < nb_points; ++i) {
            var h = this.Eval(xmin + (i * binwidthx));
            if (isNaN(h)) continue;

            if (left < 0) {
               left = i;
               ymax = h;
               ymin = h;
            }
            if ((right < 0) || (right == i - 1))
               right = i;

            if (h > ymax)
               ymax = h;
            if (h < ymin)
               ymin = h;
         }

         if (left < right) {
            xmax = xmin + right * binwidthx;
            xmin = xmin + left * binwidthx;
         }
      }

      if (ymax > 0.0) ymax *= 1.05;
      if (ymin < 0.0) ymin *= 1.05;

      var histo = JSROOT.Create("TH1I");

      histo['fName'] = this.tf1['fName'] + "_hist";
      histo['fTitle'] = this.tf1['fTitle'];

      histo['fXaxis']['fXmin'] = xmin;
      histo['fXaxis']['fXmax'] = xmax;
      histo['fYaxis']['fXmin'] = ymin;
      histo['fYaxis']['fXmax'] = ymax;

      return histo;
   }

   JSROOT.TF1Painter.prototype.CreateBins = function() {

      var pthis = this;

      if (this.tf1['fNsave'] > 0) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         var nb_points = this.tf1['fNpx'];

         var xmin = this.tf1['fSave'][nb_points + 1];
         var xmax = this.tf1['fSave'][nb_points + 2];
         var binwidthx = (xmax - xmin) / nb_points;

         this['bins'] = d3.range(nb_points).map(function(p) {
            return {
               x : xmin + (p * binwidthx),
               y : pthis.tf1['fSave'][p]
            };
         });
         this['interpolate_method'] = 'monotone';
      } else {
         if (this.tf1['fNpfits'] <= 103)
            this.tf1['fNpfits'] = 333;
         var xmin = this.tf1['fXmin'];
         var xmax = this.tf1['fXmax'];
         var nb_points = Math.max(this.tf1['fNpx'], this.tf1['fNpfits']);
         var binwidthx = (xmax - xmin) / nb_points;
         this['bins'] = d3.range(nb_points).map(function(p) {
            var xx = xmin + (p * binwidthx);
            var yy = pthis.Eval(xx);
            if (isNaN(yy)) yy = 0;
            return {
               x : xx,
               y : yy
            };
         });
         this['interpolate_method'] = 'cardinal-open';
      }
   }

   JSROOT.TF1Painter.prototype.DrawBins = function() {
      var w = Number(this.svg_frame(true).attr("width")),
          h = Number(this.svg_frame(true).attr("height"));

      this.RecreateDrawG();

      var pthis = this;
      var x = this.main_painter().x;
      var y = this.main_painter().y;

      var linecolor = JSROOT.Painter.root_colors[this.tf1['fLineColor']];
      if ((this.tf1['fLineColor'] == 0) || (this.tf1['fLineWidth'] == 0)) linecolor = "none";

      var fillcolor = JSROOT.Painter.createFillPattern(this.svg_canvas(true), this.tf1['fFillStyle'], this.tf1['fFillColor']);

      var line = d3.svg.line()
                   .x(function(d) { return Math.round(x(d.x)); })
                   .y(function(d) { return Math.round(y(d.y)); })
                   .interpolate(this.interpolate_method);

      var area = d3.svg.area()
                  .x(function(d) { return Math.round(x(d.x)); })
                  .y1(h)
                  .y0(function(d) { return Math.round(y(d.y)); });

      if (linecolor != "none")
         this.draw_g.append("svg:path")
            .attr("class", "line")
            .attr("d",line(pthis.bins))
            .style("stroke", linecolor)
            .style("stroke-width", pthis.tf1['fLineWidth'])
            .style("stroke-dasharray", JSROOT.Painter.root_line_styles[pthis.tf1['fLineStyle']])
            .style("fill", "none");

      if (fillcolor != "none")
         this.draw_g.append("svg:path")
                .attr("class", "area")
                .attr("d",area(pthis.bins))
                .style("stroke", "none")
                .style("fill", fillcolor)
                .style("antialias", "false");

      // add tooltips
      if (JSROOT.gStyle.Tooltip)
         this.draw_g.selectAll()
                   .data(this.bins).enter()
                   .append("svg:circle")
                   .attr("cx", function(d) { return x(d.x); })
                   .attr("cy", function(d) { return y(d.y); })
                   .attr("r", 4)
                   .attr("opacity", 0)
                   .append("svg:title")
                   .text( function(d) { return "x = " + d.x.toPrecision(4) + " \ny = " + d.y.toPrecision(4); });
   }

   JSROOT.TF1Painter.prototype.UpdateObject = function(obj) {
      if (obj['_typename'] != this.tf1['_typename']) return false;
      // TODO: realy update object content
      this.tf1 = obj;
      this.CreateBins();
      return true;
   }

   JSROOT.Painter.drawFunction = function(divid, tf1) {
      var painter = new JSROOT.TF1Painter(tf1);

      painter.SetDivId(divid, -1);

      if (painter.main_painter() == null) {
         var histo = painter.CreateDummyHisto();
         JSROOT.Painter.drawHistogram1D(divid, histo);
      }

      painter.SetDivId(divid);

      painter.CreateBins();

      painter.DrawBins();

      return painter;
   }

   // =======================================================================

   JSROOT.TGraphPainter = function(graph) {
      JSROOT.TObjectPainter.call(this, graph);
      this.graph = graph;
      this.ownhisto = false; // indicate if graph histogram was drawn for axes
   }

   JSROOT.TGraphPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TGraphPainter.prototype.GetObject = function() {
      return this.graph;
   }

   JSROOT.TGraphPainter.prototype.Redraw = function() {
      this.DrawBins();
   }

   JSROOT.TGraphPainter.prototype.DecodeOptions = function(opt) {
      this.logx = false;
      this.logy = false;
      this.logz = false;
      this.gridx = false;
      this.gridy = false;
      this.draw_all = true;
      this.optionLine = 0;
      this.optionAxis = 0;
      this.optionCurve = 0;
      this.optionStar = 0;
      this.optionMark = 0;
      this.optionBar = 0;
      this.optionR = 0;
      this.optionOne = 0;
      this.optionE = 0;
      this.optionFill = 0;
      this.optionZ = 0;
      this.optionCurveFill = 0;
      this.draw_errors = true;
      this.optionNone = 0; // no any drawing
      this.opt = "LP";

      if ((opt != null) && (opt != "")) {
         this.opt = opt.toUpperCase();
         this.opt.replace('SAME', '');
      }

      if (this.opt.indexOf('L') != -1)
         this.optionLine = 1;
      if (this.opt.indexOf('A') != -1)
         this.optionAxis = 1;
      if (this.opt.indexOf('C') != -1)
         this.optionCurve = 1;
      if (this.opt.indexOf('*') != -1)
         this.optionStar = 1;
      if (this.opt.indexOf('P') != -1)
         this.optionMark = 1;
      if (this.opt.indexOf('B') != -1)
         this.optionBar = 1;
      if (this.opt.indexOf('R') != -1)
         this.optionR = 1;
      if (this.opt.indexOf('1') != -1)
         this.optionOne = 1;
      if (this.opt.indexOf('F') != -1)
         this.optionFill = 1;
      if (this.opt.indexOf('2') != -1 || this.opt.indexOf('3') != -1
            || this.opt.indexOf('4') != -1 || this.opt.indexOf('5') != -1)
         this.optionE = 1;

      // if no drawing option is selected and if opt<>' ' nothing is done.
      if (this.optionLine + this.optionFill + this.optionCurve + this.optionStar + this.optionMark + this.optionBar + this.optionE == 0) {
         if (this.opt.length == 0)
            this.optionLine = 1;
         else {
            this.optionNone = 1;
            return;
         }
      }
      if (this.optionStar)
         this.graph['fMarkerStyle'] = 3;

      if (this.optionCurve && this.optionFill) {
         this.optionCurveFill = 1;
         this.optionFill = 0;
      }

      var pad = this.root_pad();
      if (pad) {
         this.logx = pad['fLogx'];
         this.logy = pad['fLogy'];
         this.logz = pad['fLogz'];
         this.gridx = pad['fGridx'];
         this.gridy = pad['fGridy'];
      }

      this.xaxis_type = this.logx ? 'logarithmic' : 'linear';
      this.yaxis_type = this.logy ? 'logarithmic' : 'linear';

      if (this.graph['_typename'] == 'TGraph') {
         // check for axis scale format, and convert if required
         if (this.graph['fHistogram']['fXaxis']['fTimeDisplay']) {
            this.xaxis_type = 'datetime';
         }

         if (this.graph['fHistogram']['fYaxis']['fTimeDisplay']) {
            this.yaxis_type = 'datetime';
         }
      } else if (this.graph['_typename'] == 'TGraphErrors') {
         this.maxEX = d3.max(this.graph['fEX']);
         this.maxEY = d3.max(this.graph['fEY']);
         if (this.maxEX < 1.0e-300 && this.maxEY < 1.0e-300)
            this.draw_errors = false;
      }
      this.seriesType = 'scatter';
      if (this.optionBar == 1) this.seriesType = 'bar';
      this.showMarker = false;
      if (this.optionMark == 1 || this.optionStar == 1) this.showMarker = true;

      if (this.optionLine == 1 || this.optionCurve == 1 || this.optionFill == 1)
         this.seriesType = 'line';

      if (this.optionBar == 1) {
         this.binwidthx = (this.graph['fHistogram']['fXaxis']['fXmax'] -
                           this.graph['fHistogram']['fXaxis']['fXmin'])
                            / (this.graph['fNpoints'] - 1);
      }
   }

   JSROOT.TGraphPainter.prototype.CreateBins = function() {
      var pthis = this;

      var npoints = this.graph['fNpoints'];
      if ((this.graph._typename=="TCutG") && (npoints>3)) npoints--;

      this.bins = d3.range(npoints).map(
            function(p) {
               if (pthis.optionBar == 1) {
                  return {
                     x : pthis.graph['fX'][p] - (pthis.binwidthx / 2),
                     y : pthis.graph['fY'][p], // graph['fHistogram']['fXaxis']['fXmin'],
                     bw : pthis.binwidthx,
                     bh : pthis.graph['fY'][p]
                  }
               } else if (pthis.graph['_typename'] == 'TGraphErrors') {
                  return {
                     x : pthis.graph['fX'][p],
                     y : pthis.graph['fY'][p],
                     exlow : pthis.graph['fEX'][p],
                     exhigh : pthis.graph['fEX'][p],
                     eylow : pthis.graph['fEY'][p],
                     eyhigh : pthis.graph['fEY'][p]
                  };
               } else if (pthis.graph['_typename'] == 'TGraphAsymmErrors'
                     || pthis.graph['_typename'].match(/^RooHist/)) {
                  return {
                     x : pthis.graph['fX'][p],
                     y : pthis.graph['fY'][p],
                     exlow : pthis.graph['fEXlow'][p],
                     exhigh : pthis.graph['fEXhigh'][p],
                     eylow : pthis.graph['fEYlow'][p],
                     eyhigh : pthis.graph['fEYhigh'][p]
                  };
               } else {
                  return {
                     x : pthis.graph['fX'][p],
                     y : pthis.graph['fY'][p]
                  };
               }
            });

      this.bins_lw = this.graph['fLineWidth']; // line width

      this.exclusionGraph = false;
      if (this.bins_lw <= 99) return;

      // special handling of exclusion graphs

      this.exclusionGraph = true;

      var ec, ff = 1;

      var normx, normy;
      var n = this.graph['fNpoints'];
      var glw = this.graph['fLineWidth'],
           xo = new Array(n + 2),
           yo = new Array(n + 2),
           xt = new Array(n + 2),
           yt = new Array(n + 2),
           xf = new Array(2 * n + 2),
           yf = new Array(2 * n + 2);
      // negative value means another side of the line...
      if (glw > 32767) glw = 65536 - glw;
      this.bins_lw = glw % 100; // line width
      if (this.bins_lw > 0) this.optionLine = 1;
      ec = JSROOT.Painter.root_colors[this.graph['fFillColor']];
      ec = ec.replace('rgb', 'rgba');
      ec = ec.replace(')', ', 0.20)');

      var a, i, j, nf, wk = (glw / 100) * 0.005;
      if (this.graph['fLineWidth'] > 32767) wk *= -1;

      var w = Number(this.svg_frame(true).attr("width")),
          h = Number(this.svg_frame(true).attr("height"));

      var ratio = w / h;

      var xmin = this.main_painter().xmin, xmax = this.main_painter().xmax,
          ymin = this.main_painter().ymin, ymax = this.main_painter().ymax;
      for (i = 0; i < n; i++) {
         xo[i] = (this.graph['fX'][i] - xmin) / (xmax - xmin);
         yo[i] = (this.graph['fY'][i] - ymin) / (ymax - ymin);
         if (w > h)
            yo[i] = yo[i] / ratio;
         else if (h > w)
            xo[i] = xo[i] / ratio;
      }
      // The first part of the filled area is made of the graph points.
      // Make sure that two adjacent points are different.
      xf[0] = xo[0];
      yf[0] = yo[0];
      nf = 0;
      for (i = 1; i < n; i++) {
         if (xo[i] == xo[i - 1] && yo[i] == yo[i - 1])  continue;
         nf++;
         xf[nf] = xo[i];
         if (xf[i] == xf[i - 1])
            xf[i] += 0.000001; // add an epsilon to avoid exact vertical
                                 // lines.
         yf[nf] = yo[i];
      }
      // For each graph points a shifted points is computed to build up
      // the second part of the filled area. First and last points are
      // treated as special cases, outside of the loop.
      if (xf[1] == xf[0]) {
         a = Math.PI / 2.0;
      } else {
         a = Math.atan((yf[1] - yf[0]) / (xf[1] - xf[0]));
      }
      if (xf[0] <= xf[1]) {
         xt[0] = xf[0] - wk * Math.sin(a);
         yt[0] = yf[0] + wk * Math.cos(a);
      } else {
         xt[0] = xf[0] + wk * Math.sin(a);
         yt[0] = yf[0] - wk * Math.cos(a);
      }
      if (xf[nf] == xf[nf - 1]) {
         a = Math.PI / 2.0;
      } else {
         a = Math.atan((yf[nf] - yf[nf - 1]) / (xf[nf] - xf[nf - 1]));
      }
      if (xf[nf] >= xf[nf - 1]) {
         xt[nf] = xf[nf] - wk * Math.sin(a);
         yt[nf] = yf[nf] + wk * Math.cos(a);
      } else {
         xt[nf] = xf[nf] + wk * Math.sin(a);
         yt[nf] = yf[nf] - wk * Math.cos(a);
      }

      var a1, a2, a3, xi0, yi0, xi1, yi1, xi2, yi2;
      for (i = 1; i < nf; i++) {
         xi0 = xf[i];
         yi0 = yf[i];
         xi1 = xf[i + 1];
         yi1 = yf[i + 1];
         xi2 = xf[i - 1];
         yi2 = yf[i - 1];
         if (xi1 == xi0) {
            a1 = Math.PI / 2.0;
         } else {
            a1 = Math.atan((yi1 - yi0) / (xi1 - xi0));
         }
         if (xi1 < xi0)
            a1 = a1 + Math.PI;
         if (xi2 == xi0) {
            a2 = Math.PI / 2.0;
         } else {
            a2 = Math.atan((yi0 - yi2) / (xi0 - xi2));
         }
         if (xi0 < xi2)
            a2 = a2 + Math.PI;
         x1 = xi0 - wk * Math.sin(a1);
         y1 = yi0 + wk * Math.cos(a1);
         x2 = xi0 - wk * Math.sin(a2);
         y2 = yi0 + wk * Math.cos(a2);
         xm = (x1 + x2) * 0.5;
         ym = (y1 + y2) * 0.5;
         if (xm == xi0) {
            a3 = Math.PI / 2.0;
         } else {
            a3 = Math.atan((ym - yi0) / (xm - xi0));
         }
         x3 = xi0 - wk * Math.sin(a3 + (Math.PI / 2.0));
         y3 = yi0 + wk * Math.cos(a3 + (Math.PI / 2.0));
         // Rotate (x3,y3) by PI around (xi0,yi0) if it is not on the (xm,ym)
         // side.
         if ((xm - xi0) * (x3 - xi0) < 0 && (ym - yi0) * (y3 - yi0) < 0) {
            x3 = 2 * xi0 - x3;
            y3 = 2 * yi0 - y3;
         }
         if ((xm == x1) && (ym == y1)) {
            x3 = xm;
            y3 = ym;
         }
         xt[i] = x3;
         yt[i] = y3;
      }
      // Close the polygon if the first and last points are the same
      if (xf[nf] == xf[0] && yf[nf] == yf[0]) {
         xm = (xt[nf] + xt[0]) * 0.5;
         ym = (yt[nf] + yt[0]) * 0.5;
         if (xm == xf[0]) {
            a3 = Math.PI / 2.0;
         } else {
            a3 = Math.atan((ym - yf[0]) / (xm - xf[0]));
         }
         x3 = xf[0] + wk * Math.sin(a3 + (Math.PI / 2.0));
         y3 = yf[0] - wk * Math.cos(a3 + (Math.PI / 2.0));
         if ((xm - xf[0]) * (x3 - xf[0]) < 0 && (ym - yf[0]) * (y3 - yf[0]) < 0) {
            x3 = 2 * xf[0] - x3;
            y3 = 2 * yf[0] - y3;
         }
         xt[nf] = x3;
         xt[0] = x3;
         yt[nf] = y3;
         yt[0] = y3;
      }
      // Find the crossing segments and remove the useless ones
      var xc, yc, c1, b1, c2, b2;
      var cross = false;
      var nf2 = nf;
      for (i = nf2; i > 0; i--) {
         for (j = i - 1; j > 0; j--) {
            if (xt[i - 1] == xt[i] || xt[j - 1] == xt[j])
               continue;
            c1 = (yt[i - 1] - yt[i]) / (xt[i - 1] - xt[i]);
            b1 = yt[i] - c1 * xt[i];
            c2 = (yt[j - 1] - yt[j]) / (xt[j - 1] - xt[j]);
            b2 = yt[j] - c2 * xt[j];
            if (c1 != c2) {
               xc = (b2 - b1) / (c1 - c2);
               yc = c1 * xc + b1;
               if (xc > Math.min(xt[i], xt[i - 1])
                     && xc < Math.max(xt[i], xt[i - 1])
                     && xc > Math.min(xt[j], xt[j - 1])
                     && xc < Math.max(xt[j], xt[j - 1])
                     && yc > Math.min(yt[i], yt[i - 1])
                     && yc < Math.max(yt[i], yt[i - 1])
                     && yc > Math.min(yt[j], yt[j - 1])
                     && yc < Math.max(yt[j], yt[j - 1])) {
                  nf++;
                  xf[nf] = xt[i];
                  yf[nf] = yt[i];
                  nf++;
                  xf[nf] = xc;
                  yf[nf] = yc;
                  i = j;
                  cross = true;
                  break;
               } else {
                  continue;
               }
            } else {
               continue;
            }
         }
         if (!cross) {
            nf++;
            xf[nf] = xt[i];
            yf[nf] = yt[i];
         }
         cross = false;
      }
      nf++;
      xf[nf] = xt[0];
      yf[nf] = yt[0];
      nf++;
      for (i = 0; i < nf; i++) {
         if (w > h) {
            xf[i] = xmin + (xf[i] * (xmax - xmin));
            yf[i] = ymin + (yf[i] * (ymax - ymin)) * ratio;
         } else if (h > w) {
            xf[i] = xmin + (xf[i] * (xmax - xmin)) * ratio;
            yf[i] = ymin + (yf[i] * (ymax - ymin));
         } else {
            xf[i] = xmin + (xf[i] * (xmax - xmin));
            yf[i] = ymin + (yf[i] * (ymax - ymin));
         }
         if (this.logx && xf[i] <= 0.0)
            xf[i] = xmin;
         if (this.logy && yf[i] <= 0.0)
            yf[i] = ymin;
      }

      this.excl = d3.range(nf).map(function(p) { return { x : xf[p], y : yf[p] }; });

      this.excl_ec = ec;
      this.excl_ff = ff;

      /* some clean-up */
      xo.splice(0, xo.length);
      yo.splice(0, yo.length);
      xo = null;
      yo = null;
      xt.splice(0, xt.length);
      yt.splice(0, yt.length);
      xt = null;
      yt = null;
      xf.splice(0, xf.length);
      yf.splice(0, yf.length);
      xf = null;
      yf = null;
   }

   JSROOT.TGraphPainter.prototype.DrawBins = function() {

      var w = Number(this.svg_frame(true).attr("width")),
          h = Number(this.svg_frame(true).attr("height"));

      this.RecreateDrawG();

      var pthis = this;

      function TooltipText(d) {

         var res = "x = " + pthis.main_painter().AxisAsText("x", d.x) + "\n" +
                   "y = " + pthis.main_painter().AxisAsText("y", d.y);

         if (pthis.draw_errors  && ('exlow' in d) && ((d.exlow!=0) || (d.exhigh!=0)))
            res += "\nerror x = -" + pthis.main_painter().AxisAsText("x", d.exlow) +
                              "/+" + pthis.main_painter().AxisAsText("x", d.exhigh);

         if (pthis.draw_errors  && ('eylow' in d) && ((d.eylow!=0) || (d.eyhigh!=0)) )
            res += "\nerror y = -" + pthis.main_painter().AxisAsText("y", d.eylow) +
                              "/+" + pthis.main_painter().AxisAsText("y", d.eyhigh);

         return res;
      }

      var x = this.main_painter().x;
      var y = this.main_painter().y;
      var line = d3.svg.line()
                  .x(function(d) { return Math.round(x(d.x)); })
                  .y(function(d) { return Math.round(y(d.y)); });

      if (this.seriesType == 'bar') {
         var fillcolor = JSROOT.Painter.root_colors[this.graph['fFillColor']];
         if (typeof (fillcolor) == 'undefined') fillcolor = "rgb(204, 204, 204)";
         /* filled bar graph */
         var xdom = this.main_painter().x.domain();
         var xfactor = xdom[1] - xdom[0];
         this.draw_errors = false;

         var nodes = this.draw_g.selectAll("bar_graph")
                  .data(pthis.bins).enter()
                  .append("svg:rect")
                  .attr("x", function(d) { return x(d.x) })
                  .attr("y", function(d) { return y(d.y) })
                  .attr("width", function(d) { return (w / (xdom[1] - xdom[0])) - 1 })
                  .attr("height", function(d) { return y(d.y) - y(d.y + d.bh); })
                  .style("fill", fillcolor);

         if (JSROOT.gStyle.Tooltip)
            nodes.append("svg:title").text(function(d) { return "x = " + d.x.toPrecision(4) + " \nentries = " + d.y.toPrecision(4); });
      }
      if (this.exclusionGraph) {
         /* first draw exclusion area, and then the line */
         this.showMarker = false;
         if (this.graph['fFillStyle'] > 3000 && this.graph['fFillStyle'] <= 3025) {
            this.draw_g.append("svg:path")
                   .attr("d", line(pthis.excl))
                   .style("stroke", "none")
                   .style("stroke-width", pthis.excl_ff)
                   .style("fill", JSROOT.Painter.createFillPattern(this.svg_canvas(true), this.graph['fFillStyle'], this.graph['fFillColor']))
                   .style("antialias", "false");
         } else {
            this.draw_g.append("svg:path")
                   .attr("d", line(pthis.excl))
                   .style("stroke", "none")
                   .style("stroke-width", pthis.excl_ff)
                   .style("fill", pthis.excl_ec);
         }
      }

      if (this.seriesType == 'line') {

         var close_symbol = "";
         if (this.graph._typename=="TCutG") close_symbol = " Z";

         var line_color = "none", line_style = "none", fill_color = "none";

         if (this.optionLine == 1) {
            line_color = JSROOT.Painter.root_colors[this.graph['fLineColor']];
            line_style = JSROOT.Painter.root_line_styles[this.graph['fLineStyle']];
         }
         if (this.optionFill == 1)
            fill_color = JSROOT.Painter.createFillPattern(this.svg_canvas(true), this.graph['fFillStyle'], this.graph['fFillColor']);

         this.draw_g.append("svg:path")
               .attr("d", line(pthis.bins) + close_symbol)
               .attr("class", "draw_line")
               .style("stroke", line_color)
               .style("stroke-width", pthis.bins_lw)
               .style("stroke-dasharray", line_style)
               .style("fill", fill_color);

         // do not add tooltip for line, when we wants to add markers
         if (JSROOT.gStyle.Tooltip && !this.showMarker)
            this.draw_g.selectAll("draw_line")
                       .data(pthis.bins).enter()
                       .append("svg:circle")
                       .attr("cx", function(d) { return Math.round(x(d.x)); })
                       .attr("cy", function(d) { return Math.round(y(d.y)); })
                       .attr("r", 3)
                       .attr("opacity", 0)
                       .append("svg:title")
                       .text(TooltipText);
      }

      if (this.draw_errors)
         this.draw_errors = (this.graph['_typename'] == 'TGraphErrors' ||
                             this.graph['_typename'] == 'TGraphAsymmErrors' ||
                             this.graph['_typename'].match(/^RooHist/)) && !this.optionBar;

      var nodes = null;

      if (this.draw_errors || this.showMarker) {
         var draw_bins = new Array;
         for (var i in this.bins) {
            var pntx = x(this.bins[i].x);
            var pnty = y(this.bins[i].y);
            if ((pntx>=0) && (pntx<=w) && (pnty>=0) && (pnty<=h)) draw_bins.push(this.bins[i]);
         }
         // here are up to five elements are collected, try to group them
         nodes = this.draw_g.selectAll("g.node")
                     .data(draw_bins)
                     .enter()
                     .append("svg:g");
      }

      if (this.draw_errors) {
         // than doing filer append error bars
         nodes.filter(function(d) { return (d.exlow > 0) || (d.exhigh > 0); })
              .append("svg:line")
              .attr("x1", function(d) { return Math.round(x(d.x - d.exlow)); })
              .attr("y1", function(d) { return Math.round(y(d.y)); })
              .attr("x2", function(d) { return Math.round(x(d.x + d.exhigh)); })
              .attr("y2", function(d) { return Math.round(y(d.y)); })
              .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
              .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.exlow > 0); })
              .append("svg:line")
              .attr("y1", function(d) { return Math.round(y(d.y)) - 3; })
              .attr("x1", function(d) { return Math.round(x(d.x - d.exlow)); })
              .attr("y2", function(d) { return Math.round(y(d.y) + 3); })
              .attr("x2", function(d) { return Math.round(x(d.x - d.exlow)); })
              .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
              .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.exhigh > 0); })
              .append("svg:line")
              .attr("y1", function(d) { return Math.round(y(d.y)) - 3; })
              .attr("x1", function(d) { return Math.round(x(d.x + d.exhigh)); })
              .attr("y2", function(d) { return Math.round(y(d.y)) + 3; })
              .attr("x2", function(d) { return Math.round(x(d.x + d.exhigh)); })
              .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
              .style( "stroke-width", this.graph['fLineWidth']);

         // Add y-error indicators

         nodes.filter(function(d) { return (d.eylow > 0) || (d.eyhigh > 0); })
              .append("svg:line")
              .attr("x1", function(d) { return Math.round(x(d.x)); })
              .attr("y1", function(d) { return Math.round(y(d.y - d.eylow)); })
              .attr("x2", function(d) { return Math.round(x(d.x)); })
              .attr("y2", function(d) { return Math.round(y(d.y + d.eyhigh)); })
              .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
              .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.eylow > 0); })
              .append("svg:line")
              .attr("x1", function(d) { return Math.round(x(d.x)) - 3; })
              .attr("y1", function(d) { return Math.round(y(d.y - d.eylow)); })
              .attr("x2", function(d) { return Math.round(x(d.x)) + 3; })
              .attr("y2", function(d) { return Math.round(y(d.y - d.eylow)); })
              .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
              .style("stroke-width", this.graph['fLineWidth']);

         nodes.filter(function(d) { return (d.eyhigh > 0); })
              .append("svg:line")
              .attr("x1", function(d) { return Math.round(x(d.x)) - 3; })
              .attr("y1", function(d) { return Math.round(y(d.y + d.eyhigh)); })
              .attr("x2", function(d) { return Math.round(x(d.x)) + 3; })
              .attr("y2", function(d) { return Math.round(y(d.y + d.eyhigh)); })
              .style("stroke", JSROOT.Painter.root_colors[this.graph['fLineColor']])
              .style("stroke-width", this.graph['fLineWidth']);
      }

      if (this.showMarker) {
         /* Add markers */
         var info_marker = JSROOT.Painter.getRootMarker(this.graph['fMarkerStyle']);

         var markerSize = this.graph['fMarkerSize'];
         var markerScale = (info_marker.shape == 0) ? 32 : 64;
         var marker_color = JSROOT.Painter.root_colors[this.graph['fMarkerColor']];
         if (this.graph['fMarkerStyle'] == 1) markerScale = 1;

         var marker;

         switch (info_marker.shape) {
            case 6:
               marker = "M " + (-4 * markerSize) + " " + (-1 * markerSize) +
                       " L " + 4 * markerSize + " " + (-1 * markerSize) +
                       " L " + (-2.4 * markerSize) + " " + 4 * markerSize +
                       " L 0 " + (-4 * markerSize) +
                       " L " + 2.8 * markerSize + " " + 4  * markerSize + " z";
               break;
            case 7:
               marker = "M " + (-4 * markerSize) + " " + (-4 * markerSize) +
                      " L " + 4 * markerSize + " " + 4 * markerSize +
                      " M 0 " + (-4 * markerSize) + " 0 " + 4 * markerSize +
                      " M "  + 4 * markerSize + " " + (-4 * markerSize) +
                      " L " + (-4 * markerSize) + " " + 4 * markerSize +
                      " M " + (-4 * markerSize) + " 0 L " + 4 * markerSize + " 0";
               break;
            default:
               marker = d3.svg.symbol().type(d3.svg.symbolTypes[info_marker.shape]).size(markerSize * markerScale);
               break;
         }

         nodes.append("svg:path")
              .attr("transform", function(d) { return "translate(" + Math.round(x(d.x)) + " , " + Math.round(y(d.y)) + ")"; })
              .style("fill", info_marker['toFill'] ? marker_color  : "none")
              .style("stroke", marker_color)
              .attr("d", marker);

      }

      if (JSROOT.gStyle.Tooltip && nodes)
         nodes.append("svg:title").text(TooltipText);
   }

   JSROOT.TGraphPainter.prototype.UpdateObject = function(obj) {
      if (obj['_typename'] != this.graph['_typename'])
         return false;

      // if our own histogram was used as axis drawing, we need update histogram  as well
      if (this.ownhisto)
         this.main_painter().UpdateObject(obj['fHistogram']);

      // TODO: make real update of TGraph object content
      this.graph['fX'] = obj['fX'];
      this.graph['fY'] = obj['fY'];
      this.graph['fNpoints'] = obj['fNpoints'];
      this.CreateBins();
      return true;
   }

   JSROOT.Painter.drawGraph = function(divid, graph, opt) {

      var painter = new JSROOT.TGraphPainter(graph);
      painter.SetDivId(divid, -1); // just to get access to existing elements

      if (painter.main_painter() == null) {
         if (graph['fHistogram']==null) {
            alert("drawing first graph without fHistogram field, not (yet) supported");
            return null;
         }
         JSROOT.Painter.drawHistogram1D(divid, graph['fHistogram']);
         painter.ownhisto = true;
      }

      painter.SetDivId(divid);

      painter.DecodeOptions(opt);

      painter.CreateBins();

      painter.DrawBins();

      return painter;
   }

   // ============================================================

   JSROOT.TPavePainter = function(pave) {
      JSROOT.TObjectPainter.call(this, pave);
      this.pavetext = pave;
      this.Enabled = true;
      this.main_rect = null;
   }

   JSROOT.TPavePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TPavePainter.prototype.GetObject = function() {
      return this.pavetext;
   }

   JSROOT.TPavePainter.prototype.DrawPaveText = function() {
      var pavetext = this.pavetext;

      var w = Number(this.svg_pad(true).attr("width")),
          h = Number(this.svg_pad(true).attr("height"));

      var pos_x = Math.round(pavetext['fX1NDC'] * w);

      var pos_y = Math.round((1.0 - pavetext['fY1NDC']) * h);
      var width = Math.round(Math.abs(pavetext['fX2NDC'] - pavetext['fX1NDC']) * w);
      var height = Math.round(Math.abs(pavetext['fY2NDC'] - pavetext['fY1NDC']) * h);
      pos_y -= height;
      var nlines = pavetext['fLines'].arr.length;
      var font_size = Math.round(height / (nlines * 1.2));
      var fcolor = JSROOT.Painter.root_colors[pavetext['fFillColor']];
      var lcolor = JSROOT.Painter.root_colors[pavetext['fLineColor']];
      var tcolor = JSROOT.Painter.root_colors[pavetext['fTextColor']];
      var scolor = JSROOT.Painter.root_colors[pavetext['fShadowColor']];
      if (pavetext['fFillStyle'] == 0) fcolor = 'none';
      // align = 10*HorizontalAlign + VerticalAlign
      // 1=left adjusted, 2=centered, 3=right adjusted
      // 1=bottom adjusted, 2=centered, 3=top adjusted
      // "middle", "start", "end"

      var align = 'start', halign = Math.round(pavetext['fTextAlign'] / 10);
      var baseline = 'bottom', valign = pavetext['fTextAlign'] % 10;
      if (halign == 1) align = 'start';
      else if (halign == 2) align = 'middle';
      else if (halign == 3) align = 'end';
      if (valign == 1)  baseline = 'bottom';
      else if (valign == 2) baseline = 'middle';
      else if (valign == 3) baseline = 'top';

      var h_margin = Math.round(pavetext['fMargin'] * width); // horizontal margin
      var lmargin = 0;
      switch (halign) {
         case 1: lmargin = h_margin; break;
         case 2: lmargin = width / 2; break;
         case 3: lmargin = width - h_margin; break;
      }

      // for now ignore all align parameters, draw as is
      if (nlines > 1)
         lmargin = pavetext['fMargin'] * width / 2;

      var fontDetails = JSROOT.Painter.getFontDetails(pavetext['fTextFont']);
      var lwidth = pavetext['fBorderSize'] ? pavetext['fBorderSize'] : 0;

      var pthis = this;

      if (this.main_rect == null)
         this.main_rect = this.svg_pad(true).append("rect");

      this.main_rect
             .attr("x", pos_x)
             .attr("y", pos_y)
             .attr("width", width)
             .attr("height", height)
             .attr("fill", fcolor)
             .style("stroke-width", lwidth ? 1 : 0)
             .style("stroke", lcolor);

      // container used to recalculate coordinates
      this.RecreateDrawG(true);

      this.draw_g.attr("transform", "translate(" + pos_x + "," + pos_y + ")");

      var first_stat = 0, num_cols = 0;
      var maxlw = 0;
      var lines = new Array;

      // adjust font size
      for (var j = 0; j < nlines; ++j) {
         var line = JSROOT.Painter.translateLaTeX(pavetext['fLines'].arr[j]['fTitle']);
         lines.push(line);
         var lw = lmargin + JSROOT.Painter.stringWidth(this.svg_pad(true), line, font_size, fontDetails) + h_margin;
         if (lw > maxlw) maxlw = lw;
         if ((j == 0) || (line.indexOf('|') < 0)) continue;
         if (first_stat === 0) first_stat = j;
         var parts = line.split("|");
         if (parts.length > num_cols)
            num_cols = parts.length;
      }

      if (maxlw > width)
         font_size = Math.floor(font_size * (width / maxlw));

      // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
      var stepy = (height - 0.2*font_size) / nlines;

      if (nlines == 1) {
         this.draw_g.append("text")
              .attr("text-anchor", align)
              .attr("x",lmargin)
              .attr("y", (height / 2) + (font_size / 3))
              .attr("xml:space","preserve")
              .attr("font-family", fontDetails['name'])
              .attr("font-weight", fontDetails['weight'])
              .attr("font-style", fontDetails['style'])
              .attr("font-size", font_size)
              .attr("fill", tcolor)
              .text(lines[0]);
      } else {

         for (var j = 0; j < nlines; ++j) {
            var jcolor = JSROOT.Painter.root_colors[pavetext['fLines'].arr[j]['fTextColor']];
            if (pavetext['fLines'].arr[j]['fTextColor'] == 0) jcolor = tcolor;
            var posy = (j+0.5)*stepy + font_size*0.5 - 1;

            if (pavetext['_typename'] == 'TPaveStats') {
               if ((first_stat > 0) && (j >= first_stat)) {
                  var parts = lines[j].split("|");
                  for (var n = 0; n < parts.length; n++)
                     this.draw_g.append("text")
                           .attr("text-anchor", "middle")
                           .attr("x", width * (n + 0.5) / num_cols)
                           .attr("y", posy)
                           .attr("xml:space","preserve")
                           .attr("font-family", fontDetails['name'])
                           .attr("font-weight", fontDetails['weight'])
                           .attr("font-style", fontDetails['style'])
                           .attr("font-size", font_size)
                           .attr("fill", jcolor)
                           .text(parts[n]);
               } else if ((j == 0) || (lines[j].indexOf('=') < 0)) {
                  this.draw_g.append("text")
                        .attr("text-anchor", (j == 0) ? "middle" : "start")
                        .attr("x", ((j == 0) ? width / 2 : pavetext['fMargin'] * width))
                        .attr("y", posy)
                        .attr("xml:space","preserve")
                        .attr("font-family", fontDetails['name'])
                        .attr("font-weight", fontDetails['weight'])
                        .attr("font-style", fontDetails['style'])
                        .attr("font-size", font_size)
                        .attr("fill", jcolor)
                        .text(lines[j]);
               } else {
                  var parts = lines[j].split("=");
                  for (var n = 0; n < 2; n++)
                     this.draw_g.append("text")
                            .attr("text-anchor", (n == 0) ? "start" : "end")
                            .attr("x", (n == 0) ? pavetext['fMargin'] * width  : (1 - pavetext['fMargin']) * width)
                            .attr("y", posy)
                            .attr("xml:space","preserve")
                            .attr("font-family", fontDetails['name'])
                            .attr("font-weight", fontDetails['weight'])
                            .attr("font-style", fontDetails['style'])
                            .attr("font-size", font_size)
                            .attr("fill", jcolor)
                            .text(parts[n]);
               }
            } else {
               this.draw_g.append("text")
                      .attr("text-anchor", "start")
                      .attr("x", lmargin)
                      .attr("y", posy)
                      .attr("xml:space","preserve")
                      .attr("font-family", fontDetails['name'])
                      .attr("font-weight", fontDetails['weight'])
                      .attr("font-style", fontDetails['style'])
                      .attr("font-size", font_size)
                      .attr("fill", jcolor)
                      .text(lines[j]);
            }
         }
      }

      if (pavetext['fBorderSize'] && (pavetext['_typename'] == 'TPaveStats')) {
         this.draw_g.append("svg:line")
                    .attr("class", "pavedraw")
                    .attr("x1", 0)
                    .attr("y1", stepy)
                    .attr("x2", width)
                    .attr("y2", stepy)
                    .style("stroke", lcolor)
                    .style("stroke-width", lwidth ? 1 : 'none');
      }

      if ((first_stat > 0) && (num_cols > 1)) {
         for (var nrow = first_stat; nrow < nlines; nrow++)
            this.draw_g.append("svg:line")
                       .attr("x1", 0)
                       .attr("y1", nrow * stepy)
                       .attr("x2", width)
                       .attr("y2", nrow * stepy)
                       .style("stroke", lcolor)
                       .style("stroke-width", lwidth ? 1 : 'none');

         for (var ncol = 0; ncol < num_cols - 1; ncol++)
            this.draw_g.append("svg:line")
                        .attr("x1", width / num_cols * (ncol + 1))
                        .attr("y1", first_stat * stepy)
                        .attr("x2", width / num_cols * (ncol + 1))
                        .attr("y2", height)
                        .style("stroke", lcolor)
                        .style("stroke-width", lwidth ? 1 : 'none');
      }

      if (lwidth && lwidth > 1) {
         this.draw_g.append("svg:line")
                    .attr("x1", width + (lwidth / 2))
                    .attr("y1", lwidth + 1)
                    .attr("x2", width + (lwidth / 2))
                    .attr("y2", height + lwidth - 1)
                    .style("stroke", lcolor)
                    .style("stroke-width", lwidth);
         this.draw_g.append("svg:line")
                    .attr("x1", lwidth + 1)
                    .attr("y1", height + (lwidth / 2))
                    .attr("x2", width + lwidth - 1)
                    .attr("y2", height + (lwidth / 2))
                    .style("stroke", lcolor)
                    .style("stroke-width", lwidth);
      }

      // force main rect of the stat box be last item in the primitives to
      // kept it on the top - for instance when colz is created
      JSROOT.Painter.moveChildToEnd(this.main_rect);
      JSROOT.Painter.moveChildToEnd(this.draw_g);

      this.AddDrag("stat", this.main_rect, {
         move : function(x, y, dx, dy) {
            pthis.draw_g.attr("transform", "translate(" + x + "," + y + ")");

            pthis.pavetext['fX1NDC'] += dx / Number(pthis.svg_pad(true).attr("width"));
            pthis.pavetext['fX2NDC'] += dx / Number(pthis.svg_pad(true).attr("width"));
            pthis.pavetext['fY1NDC'] -= dy / Number(pthis.svg_pad(true).attr("height"));
            pthis.pavetext['fY2NDC'] -= dy / Number(pthis.svg_pad(true).attr("height"));
         },
         resize : function(width, height) {
            pthis.pavetext['fX2NDC'] = pthis.pavetext['fX1NDC'] + width  / Number(pthis.svg_pad(true).attr("width"));
            pthis.pavetext['fY1NDC'] = pthis.pavetext['fY2NDC'] - height / Number(pthis.svg_pad(true).attr("height"));

            pthis.DrawPaveText();
         }
      });
   }

   JSROOT.TPavePainter.prototype.AddLine = function(txt) {
      this.pavetext.AddText(txt);
      //this.pavetext['fLines'].arr.push({'fTitle' : txt, "fTextColor" : 1 });
   }

   JSROOT.TPavePainter.prototype.IsStats = function() {
      if (!this.pavetext) return false;
      return this.pavetext['fName'] == "stats";
   }

   JSROOT.TPavePainter.prototype.FillStatistic = function() {
      if (!this.IsStats()) return;

      var dostat = new Number(this.pavetext['fOptStat']);
      if (!dostat) dostat = new Number(JSROOT.gStyle.OptStat);

      // we take histogram from first painter
      if ('FillStatistic' in this.main_painter()) {

         // make empty at the beginning
         this.pavetext['fLines'].arr.length = 0;

         this.main_painter().FillStatistic(this, dostat);
      }
   }

   JSROOT.TPavePainter.prototype.UpdateObject = function(obj) {
      if (obj._typename != 'TPaveText') return false;
      this.pavetext['fLines'] = JSROOT.clone(obj['fLines']);
      return true;
   }

   JSROOT.TPavePainter.prototype.Redraw = function() {

      this.RemoveDrawG();

      // if pavetext artificially disabled, do not redraw it
      if (!this.Enabled) {
         this.RemoveDrag("stat");
         if (this.main_rect) {
            this.main_rect.remove();
            this.main_rect = null;
         }
         return;
      }

      this.FillStatistic();

      this.DrawPaveText();
   }

   JSROOT.Painter.drawPaveText = function(divid, pavetext) {
      if (pavetext['fX1NDC'] < 0.0 || pavetext['fY1NDC'] < 0.0 ||
          pavetext['fX1NDC'] > 1.0 || pavetext['fY1NDC'] > 1.0)
         return null;

      var painter = new JSROOT.TPavePainter(pavetext);

      painter.SetDivId(divid);

      // refill statistic in any case
      // if ('_AutoCreated' in pavetext)
      painter.FillStatistic();

      painter.DrawPaveText();

      return painter;
   }

   // ===========================================================================

   JSROOT.TPadPainter = function(pad, iscan) {
      JSROOT.TObjectPainter.call(this, pad);
      if (this.obj_typename=="") this.obj_typename = iscan ? "TCanvas" : "TPad";
      this.pad = pad;
      this.iscan = iscan; // indicate if workign with canvas
      this.painters = new Array; // complete list of all painters in the pad
      this.primitive_painters = new Array; // list of painters for primitives, used in update
   }

   JSROOT.TPadPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TPadPainter.prototype.CreateCanvasSvg = function(only_resize) {

      var render_to  = "#" + this.divid;

      var w = $(render_to).width(), h = $(render_to).height();

      var svg = null;

      if (only_resize) {
         svg = this.svg_canvas(true);
         if ((svg.property('last_width') == w) && (svg.property('last_height') == h)) return false;
      } else {

         if (h < 10) {
            // set aspect ratio for the place, where object will be drawn

            var factor = 0.66;

            // for TCanvas reconstruct ratio between width and height
            if ((this.pad!=null) && ('fCw' in this.pad) && ('fCh' in this.pad) && (this.pad['fCw'] > 0)) {
               factor = this.pad['fCh'] / this.pad['fCw'];
               if ((factor < 0.1) || (factor > 10))
                  factor = 0.66;
            }

            h = w * factor;

            $(render_to).height(h);
         }

         var fillcolor = 'white';
         d3.select(render_to).style("background-color", fillcolor);
         svg = d3.select(render_to)
          .append("svg")
          .attr("class", "root_canvas")
          .style("background-color", fillcolor)
          // .attr("pointer-events", "all")   // comment out while it hides mouse events
          .property('pad_painter', this) // this is custom property
          .property('mainpainter', null) // this is custom property
          .property('current_pad', "") // this is custom property
      }

      svg.attr("width", w)
         .attr("height", h)
         .attr("viewBox", "0 0 " + w + " " + h)
         .property('last_width', w)
         .property('last_height', h);

      return true;
   }


   JSROOT.TPadPainter.prototype.CreatePadSvg = function(only_resize) {
      var width = Number(this.svg_canvas(true).attr("width")),
          height = Number(this.svg_canvas(true).attr("height"));
      var x = Math.round(this.pad['fAbsXlowNDC'] * width);
      var y = Math.round(height - this.pad['fAbsYlowNDC'] * height);
      var w = Math.round(this.pad['fAbsWNDC'] * width);
      var h = Math.round(this.pad['fAbsHNDC'] * height);
      y -= h;

      var fillcolor = JSROOT.Painter.root_colors[this.pad['fFillColor']];
      if (this.pad['fFillStyle'] > 4000 && this.pad['fFillStyle'] < 4100)
         fillcolor = 'none';

      var border_width = this.pad['fLineWidth'];
      var border_color = JSROOT.Painter.root_colors[this.pad['fLineColor']];
      if (this.pad['fBorderMode'] == 0) {
         border_width = 0;
         border_color = 'none';
      }

      var svg_pad, svg_rect;

      if (only_resize)
         svg_pad = this.svg_pad(true);
      else
         svg_pad = this.svg_canvas(true).append("g")
             .attr("class", "root_pad")
             .attr("pad", this.pad['fName']) // set extra attribute  to mark pad name
             .property('pad_painter', this) // this is custom property
             .property('mainpainter', null); // this is custom property

      svg_pad.attr("width", w)
             .attr("height", h)
             .attr("viewBox", x + " " + y + " " + (x+w) + " " + (y+h))
             .attr("transform", "translate(" + x + "," + y + ")");

      if (only_resize)
         svg_rect = svg_pad.select(".root_pad_border");
      else
         svg_rect = svg_pad.append("svg:rect").attr("class", "root_pad_border");

      svg_rect.attr("x", 0)
              .attr("y", 0)
              .attr("width", w)
              .attr("height", h)
              .attr("fill", fillcolor)
              .style("stroke-width", border_width)
              .style("stroke", border_color);
   }

   JSROOT.TPadPainter.prototype.CheckColors = function(can) {
      if (can==null) return;
      for (var i in can.fPrimitives.arr) {
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

   JSROOT.TPadPainter.prototype.DrawPrimitives = function() {
      if (this.pad==null) return;

      for (var i in this.pad.fPrimitives.arr) {
         var pp = JSROOT.draw(this.divid, this.pad.fPrimitives.arr[i],  this.pad.fPrimitives.opt[i]);
         this.primitive_painters.push(pp);
      }
   }

   JSROOT.TPadPainter.prototype.Redraw = function(resize) {
      if (resize && !this.iscan) this.CreatePadSvg(true);

      // at the moment canvas painter donot redraw its subitems
      for (var i in this.painters)
         this.painters[i].Redraw(resize);
   }


   JSROOT.TPadPainter.prototype.CheckCanvasResize = function() {
      if (!this.iscan) return;

      var changed = this.CreateCanvasSvg(true);
      if (changed) this.Redraw(true);
   }


   JSROOT.TPadPainter.prototype.UpdateObject = function(obj) {

      if ((obj == null) || !('fPrimitives' in obj)) return false;

      if (this.iscan) this.CheckColors(obj);

      if (obj.fPrimitives.arr.length != this.pad.fPrimitives.arr.length) return false;

      var isany = false;
      var cnt = 0;

      for (var n in obj.fPrimitives.arr) {
         var sub = obj.fPrimitives.arr[n];

         if ((n >= this.primitive_painters.length) || (this.primitive_painters[n]==null)) {
            console.log("No paintrer for object " + sub._typename);
            continue;
         }

         if (this.primitive_painters[n].UpdateObject(sub)) isany = true;
      }

      return isany;
   }

   JSROOT.Painter.drawCanvas = function(divid, can) {
      var painter = new JSROOT.TPadPainter(can, true);
      painter.SetDivId(divid);
      painter.CreateCanvasSvg();

      if (can==null) {
         JSROOT.Painter.drawFrame(divid, null);
      } else {
         painter.CheckColors(can);
         painter.DrawPrimitives();
      }

      return painter;
   }

   JSROOT.Painter.drawPad = function(divid, pad) {

      var painter = new JSROOT.TPadPainter(pad, false);
      painter.SetDivId(divid); // pad painter will be registered in the canvas painters list

      painter.CreatePadSvg();

      painter.pad_name = pad['fName'];

      // we select current pad, where all drawing is performed
      var prev_name = painter.svg_canvas()['current_pad'];
      painter.svg_canvas()['current_pad'] = pad['fName'];

      painter.DrawPrimitives();

      // we restore previous pad name
      painter.svg_canvas()['current_pad'] = prev_name;

      return painter;
   }

   // ===========================================================================

   JSROOT.TColzPalettePainter = function(palette) {
      JSROOT.TObjectPainter.call(this, palette);
      this.palette = palette;
   }

   JSROOT.TColzPalettePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TColzPalettePainter.prototype.GetObject = function() {
      return this.palette;
   }

   JSROOT.TColzPalettePainter.prototype.DrawPalette = function() {
      var palette = this.palette;
      var axis = palette['fAxis'];

      var minbin = this.main_painter().minbin;
      var maxbin = this.main_painter().maxbin;
      var nbr1 = axis['fNdiv'] % 100;
      if (nbr1<=0) nbr1 = 8;

      var width = Number(this.svg_pad(true).attr("width")),
          height = Number(this.svg_pad(true).attr("height"));

      var s_height = Math.abs(palette['fY2NDC'] - palette['fY1NDC']) * height;

      var axisOffset = axis['fLabelOffset'] * width;
      var tickSize = axis['fTickSize'] * width;

      var z = d3.scale.linear().clamp(true).domain([ minbin, maxbin ]).range( [ s_height, 0 ]).nice();

      var axisFontDetails = JSROOT.Painter.getFontDetails(axis['fLabelFont']);
      var axisLabelFontSize = axis['fLabelSize'] * height;

      var pos_x = palette['fX1NDC'] * width;
      var pos_y = height - palette['fY1NDC'] * height;

      var s_width = Math.abs(palette['fX2NDC'] - palette['fX1NDC']) * width;
      pos_y -= s_height;

      // Draw palette pad
      this.RecreateDrawG(true);

      this.draw_g.attr("height", s_height)
                 .attr("width", s_width)
                 .attr("transform", "translate(" + pos_x + ", " + pos_y + ")");

      var paletteColors = this.main_painter().paletteColors;

      // Draw palette
      var rectHeight = s_height / paletteColors.length;
      this.draw_g.selectAll("colorRect")
                 .data(paletteColors)
                 .enter()
                 .append("svg:rect")
                 .attr("class", "colorRect")
                 .attr("x", 0)
                 .attr("y",  function(d, i) { return s_height - (i + 1) * rectHeight; })
                 .attr("width", s_width)
                 .attr("height", rectHeight)
                 .attr("fill", function(d) { return d; })
                 .attr("stroke", function(d) { return d; });
      /*
       * Build and draw axes
       */

      var z_axis = d3.svg.axis().scale(z)
                    .orient("right")
                    .tickPadding(axisOffset)
                    .tickSize(-tickSize, -tickSize / 2, 0)
                    .ticks(nbr1);

      var zax = this.draw_g.append("svg:g")
                   .attr("class", "zaxis")
                   .attr("transform", "translate(" + s_width + ", 0)")
                   .call(z_axis);

      zax.selectAll("text")
              .attr("font-size", axisLabelFontSize)
              .attr("font-weight", axisFontDetails['weight'])
              .attr("font-style", axisFontDetails['style'])
              .attr("font-family", axisFontDetails['name'])
              .attr("fill", JSROOT.Painter.root_colors[axis['fLabelColor']]);

      /*
       * Add palette axis title
       */
      var title = axis['fTitle'];
      if (title != "" && typeof (axis['fTitleFont']) != 'undefined') {
         axisFontDetails = JSROOT.Painter.getFontDetails(axis['fTitleFont']);
         var axisTitleFontSize = axis['fTitleSize'] * height;
         this.draw_g.append("text")
                .attr("class", "Z axis label")
                .attr("x", s_width + axisLabelFontSize)
                .attr("y", s_height)
                .attr("text-anchor", "end")
                .attr("font-family", axisFontDetails['name'])
                .attr("font-weight", axisFontDetails['weight'])
                .attr("font-style", axisFontDetails['style'])
                .attr("font-size", axisTitleFontSize).text(title);
      }

      if (this.main_rect == null) {
         this.main_rect = this.svg_pad(true)
                           .append("rect")
                           .style("opacity", "0");
      } else {
         // ensure that all color drawing inserted before move rect
         var prnt = this.main_rect.node().parentNode;
         prnt.removeChild(this.draw_g.node());
         prnt.insertBefore(this.draw_g.node(), this.main_rect.node());
      }

      this.main_rect.attr("x", pos_x)
                    .attr("y", pos_y)
                    .attr("width", s_width)
                    .attr("height", s_height);

      var pthis = this;

      this.AddDrag("colz", this.main_rect, {
         move : function(x, y, dx, dy) {

            pthis.draw_g.attr("transform", "translate(" + x + "," + y + ")");

            pthis.palette['fX1NDC'] += dx / Number(pthis.svg_pad(true).attr("width"));
            pthis.palette['fX2NDC'] += dx / Number(pthis.svg_pad(true).attr("width"));
            pthis.palette['fY1NDC'] -= dy / Number(pthis.svg_pad(true).attr("height"));
            pthis.palette['fY2NDC'] -= dy / Number(pthis.svg_pad(true).attr("height"));
         },
         resize : function(width, height) {
            pthis.palette['fX2NDC'] = pthis.palette['fX1NDC'] + width / Number(pthis.svg_pad(true).attr("width"));
            pthis.palette['fY1NDC'] = pthis.palette['fY2NDC'] - height / Number(pthis.svg_pad(true).attr("height"));

            pthis.RemoveDrawG();
            pthis.DrawPalette();
         }
      });
   }

   JSROOT.TColzPalettePainter.prototype.Redraw = function() {

      var enabled = true;

      if ('options' in this.main_painter())
         enabled = (this.main_painter().options.Zscale > 0) && (this.main_painter().options.Color > 0);

      if (enabled) {
         this.DrawPalette();
      } else {
         // if palette artificially disabled, do not redraw it
         this.RemoveDrawG();
         this.RemoveDrag("colz");
         if (this.main_rect) {
            this.main_rect.remove();
            this.main_rect = null;
         }
      }
   }

   JSROOT.Painter.drawPaletteAxis = function(divid, palette) {
      var painter = new JSROOT.TColzPalettePainter(palette);

      painter.SetDivId(divid);

      painter.DrawPalette();

      return painter;
   }

   // =============================================================

   JSROOT.THistPainter = function(histo) {
      JSROOT.TObjectPainter.call(this, histo);
      this.histo = histo;
      this.shrink_frame_left = 0.;
      this.draw_content = true;
   }

   JSROOT.THistPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.THistPainter.prototype.GetObject = function() {
      return this.histo;
   }

   JSROOT.THistPainter.prototype.IsTProfile = function() {
      return this.histo && this.histo['_typename'] == 'TProfile';
   }

   JSROOT.THistPainter.prototype.IsTH2Poly = function() {
      return this.histo && this.histo['_typename'].match(/^TH2Poly/);
   }

   JSROOT.THistPainter.prototype.Dimension = function() {
      if (!this.histo) return 0;
      if ('fDimension' in this.histo) return this.histo['fDimension'];
      if (this.histo['_typename'].match(/^TH2/)) return 2;
      if (this.histo['_typename'].match(/^TH3/)) return 3;
      return 1;
   }

   JSROOT.THistPainter.prototype.DecodeOptions = function(opt) {
      if ((opt == null) || (opt == "")) opt = this.histo['fOption'];

      /* decode string 'opt' and fill the option structure */
      var hdim = this.Dimension();
      var nch = opt.length;
      var option = {
         'Axis' : 0, 'Bar' : 0, 'Curve' : 0, 'Error' : 0, 'Hist' : 0, 'Line' : 0,
         'Mark' : 0, 'Fill' : 0, 'Same' : 0, 'Scat' : 0, 'Func' : 0, 'Star' : 0,
         'Arrow' : 0, 'Box' : 0, 'Text' : 0, 'Char' : 0, 'Color' : 0, 'Contour' : 0,
         'Lego' : 0, 'Surf' : 0, 'Off' : 0, 'Tri' : 0, 'Proj' : 0, 'AxisPos' : 0,
         'Spec' : 0, 'Pie' : 0, 'List' : 0, 'Zscale' : 0, 'FrontBox' : 1, 'BackBox' : 1,
         'System' : JSROOT.Painter.Coord.kCARTESIAN,
         'HighRes' : 0, 'Zero' : 0, 'Logx' : 0, 'Logy' : 0, 'Logz' : 0, 'Gridx' : 0, 'Gridy' : 0
      };
      // check for graphical cuts
      var chopt = opt.toUpperCase();
      chopt = JSROOT.Painter.clearCuts(chopt);
      if (hdim > 1) option.Scat = 1;
      if (!nch) option.Hist = 1;
      if (this.IsTProfile()) option.Error = 2;
      if ('fFunctions' in this.histo) option.Func = 1;

      var l = chopt.indexOf('SPEC');
      if (l != -1) {
         option.Scat = 0;
         chopt = chopt.replace('SPEC', '    ');
         var bs = 0;
         l = chopt.indexOf('BF(');
         if (l != -1) bs = parseInt(chopt)
         option.Spec = Math.max(1600, bs);
         return option;
      }
      l = chopt.indexOf('GL');
      if (l != -1)  chopt = chopt.replace('GL', '  ');
      l = chopt.indexOf('X+');
      if (l != -1) {
         option.AxisPos = 10;
         chopt = chopt.replace('X+', '  ');
      }
      l = chopt.indexOf('Y+');
      if (l != -1) {
         option.AxisPos += 1;
         chopt = chopt.replace('Y+', '  ');
      }
      if ((option.AxisPos == 10 || option.AxisPos == 1) && (nch == 2))
         option.Hist = 1;
      if (option.AxisPos == 11 && nch == 4)
         option.Hist = 1;
      l = chopt.indexOf('SAMES');
      if (l != -1) {
         if (nch == 5) option.Hist = 1;
         option.Same = 2;
         chopt = chopt.replace('SAMES', '     ');
      }
      l = chopt.indexOf('SAME');
      if (l != -1) {
         if (nch == 4) option.Hist = 1;
         option.Same = 1;
         chopt = chopt.replace('SAME', '    ');
      }
      l = chopt.indexOf('PIE');
      if (l != -1) {
         option.Pie = 1;
         chopt = chopt.replace('PIE', '   ');
      }
      l = chopt.indexOf('LEGO');
      if (l != -1) {
         option.Scat = 0;
         option.Lego = 1;
         chopt = chopt.replace('LEGO', '    ');
         if (chopt[l + 4] == '1') {
            option.Lego = 11;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '2') {
            option.Lego = 12;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '3') {
            option.Lego = 13;
            chopt[l + 4] = ' ';
         }
         l = chopt.indexOf('FB');
         if (l != -1) {
            option.FrontBox = 0;
            chopt = chopt.replace('FB', '  ');
         }
         l = chopt.indexOf('BB');
         if (l != -1) {
            option.BackBox = 0;
            chopt = chopt.replace('BB', '  ');
         }
         l = chopt.indexOf('0');
         if (l != -1) {
            option.Zero = 1;
            chopt = chopt.replace('0', ' ');
         }
      }
      l = chopt.indexOf('SURF');
      if (l != -1) {
         option.Scat = 0;
         option.Surf = 1;
         chopt = chopt.replace('SURF', '    ');
         if (chopt[l + 4] == '1') {
            option.Surf = 11;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '2') {
            option.Surf = 12;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '3') {
            option.Surf = 13;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '4') {
            option.Surf = 14;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '5') {
            option.Surf = 15;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '6') {
            option.Surf = 16;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '7') {
            option.Surf = 17;
            chopt[l + 4] = ' ';
         }
         l = chopt.indexOf('FB');
         if (l != -1) {
            option.FrontBox = 0;
            chopt = chopt.replace('FB', '  ');
         }
         l = chopt.indexOf('BB');
         if (l != -1) {
            option.BackBox = 0;
            chopt = chopt.replace('BB', '  ');
         }
      }
      l = chopt.indexOf('TF3');
      if (l != -1) {
         l = chopt.indexOf('FB');
         if (l != -1) {
            option.FrontBox = 0;
            chopt = chopt.replace('FB', '  ');
         }
         l = chopt.indexOf('BB');
         if (l != -1) {
            option.BackBox = 0;
            chopt = chopt.replace('BB', '  ');
         }
      }
      l = chopt.indexOf('ISO');
      if (l != -1) {
         l = chopt.indexOf('FB');
         if (l != -1) {
            option.FrontBox = 0;
            chopt = chopt.replace('FB', '  ');
         }
         l = chopt.indexOf('BB');
         if (l != -1) {
            option.BackBox = 0;
            chopt = chopt.replace('BB', '  ');
         }
      }
      l = chopt.indexOf('LIST');
      if (l != -1) {
         option.List = 1;
         chopt = chopt.replace('LIST', '  ');
      }
      l = chopt.indexOf('CONT');
      if (l != -1) {
         chopt = chopt.replace('CONT', '    ');
         if (hdim > 1) {
            option.Scat = 0;
            option.Contour = 1;
            if (chopt[l + 4] == '1') {
               option.Contour = 11;
               chopt[l + 4] = ' ';
            }
            if (chopt[l + 4] == '2') {
               option.Contour = 12;
               chopt[l + 4] = ' ';
            }
            if (chopt[l + 4] == '3') {
               option.Contour = 13;
               chopt[l + 4] = ' ';
            }
            if (chopt[l + 4] == '4') {
               option.Contour = 14;
               chopt[l + 4] = ' ';
            }
            if (chopt[l + 4] == '5') {
               option.Contour = 15;
               chopt[l + 4] = ' ';
            }
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('HBAR');
      if (l != -1) {
         option.Hist = 0;
         option.Bar = 20;
         chopt = chopt.replace('HBAR', '    ');
         if (chopt[l + 4] == '1') {
            option.Bar = 21;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '2') {
            option.Bar = 22;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '3') {
            option.Bar = 23;
            chopt[l + 4] = ' ';
         }
         if (chopt[l + 4] == '4') {
            option.Bar = 24;
            chopt[l + 4] = ' ';
         }
      }
      l = chopt.indexOf('BAR');
      if (l != -1) {
         option.Hist = 0;
         option.Bar = 10;
         chopt = chopt.replace('BAR', '   ');
         if (chopt[l + 3] == '1') {
            option.Bar = 11;
            chopt[l + 3] = ' ';
         }
         if (chopt[l + 3] == '2') {
            option.Bar = 12;
            chopt[l + 3] = ' ';
         }
         if (chopt[l + 3] == '3') {
            option.Bar = 13;
            chopt[l + 3] = ' ';
         }
         if (chopt[l + 3] == '4') {
            option.Bar = 14;
            chopt[l + 3] = ' ';
         }
      }
      l = chopt.indexOf('ARR');
      if (l != -1) {
         chopt = chopt.replace('ARR', '   ');
         if (hdim > 1) {
            option.Arrow = 1;
            option.Scat = 0;
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('BOX');
      if (l != -1) {
         chopt = chopt.replace('BOX', '   ');
         if (hdim > 1) {
            Hoption.Scat = 0;
            Hoption.Box = 1;
            if (chopt[l + 3] == '1') {
               option.Box = 11;
               chopt[l + 3] = ' ';
            }
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('COLZ');
      if (l != -1) {
         chopt = chopt.replace('COLZ', '');
         if (hdim > 1) {
            option.Color = 2;
            option.Scat = 0;
            option.Zscale = 1;
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('COL');
      if (l != -1) {
         chopt = chopt.replace('COL', '   ');
         if (hdim > 1) {
            option.Color = 1;
            option.Scat = 0;
         } else {
            option.Hist = 1;
         }
      }
      l = chopt.indexOf('CHAR');
      if (l != -1) {
         option.Char = 1;
         chopt = chopt.replace('CHAR', '    ');
         option.Scat = 0;
      }
      l = chopt.indexOf('FUNC');
      if (l != -1) {
         option.Func = 2;
         chopt = chopt.replace('FUNC', '    ');
         option.Hist = 0;
      }
      l = chopt.indexOf('HIST');
      if (l != -1) {
         option.Hist = 2;
         chopt = chopt.replace('HIST', '    ');
         option.Func = 0;
         option.Error = 0;
      }
      l = chopt.indexOf('AXIS');
      if (l != -1) {
         option.Axis = 1;
         chopt = chopt.replace('AXIS', '    ');
      }
      l = chopt.indexOf('AXIG');
      if (l != -1) {
         option.Axis = 2;
         chopt = chopt.replace('AXIG', '    ');
      }
      l = chopt.indexOf('SCAT');
      if (l != -1) {
         option.Scat = 1;
         chopt = chopt.replace('SCAT', '    ');
      }
      l = chopt.indexOf('TEXT');
      if (l != -1) {
         var angle = parseInt(chopt);
         if (!isNaN(angle)) {
            if (angle < 0)
               angle = 0;
            if (angle > 90)
               angle = 90;
            option.Text = 1000 + angle;
         } else {
            option.Text = 1;
         }
         chopt = chopt.replace('TEXT', '    ');
         l = chopt.indexOf('N');
         if (l != -1 && this.IsTH2Poly())
            option.Text += 3000;
         option.Scat = 0;
      }
      l = chopt.indexOf('POL');
      if (l != -1) {
         option.System = JSROOT.Painter.Coord.kPOLAR;
         chopt = chopt.replace('POL', '   ');
      }
      l = chopt.indexOf('CYL');
      if (l != -1) {
         option.System = JSROOT.Painter.Coord.kCYLINDRICAL;
         chopt = chopt.replace('CYL', '   ');
      }
      l = chopt.indexOf('SPH');
      if (l != -1) {
         option.System = JSROOT.Painter.Coord.kSPHERICAL;
         chopt = chopt.replace('SPH', '   ');
      }
      l = chopt.indexOf('PSR');
      if (l != -1) {
         option.System = JSROOT.Painter.Coord.kRAPIDITY;
         chopt = chopt.replace('PSR', '   ');
      }
      l = chopt.indexOf('TRI');
      if (l != -1) {
         option.Scat = 0;
         option.Color = 0;
         option.Tri = 1;
         chopt = chopt.replace('TRI', '   ');
         l = chopt.indexOf('FB');
         if (l != -1) {
            option.FrontBox = 0;
            chopt = chopt.replace('FB', '  ');
         }
         l = chopt.indexOf('BB');
         if (l != -1) {
            option.BackBox = 0;
            chopt = chopt.replace('BB', '  ');
         }
         l = chopt.indexOf('ERR');
         if (l != -1)
            chopt = chopt.replace('ERR', '   ');
      }
      l = chopt.indexOf('AITOFF');
      if (l != -1) {
         Hoption.Proj = 1;
         chopt = chopt.replace('AITOFF', '      '); // Aitoff projection
      }
      l = chopt.indexOf('MERCATOR');
      if (l != -1) {
         option.Proj = 2;
         chopt = chopt.replace('MERCATOR', '        '); // Mercator projection
      }
      l = chopt.indexOf('SINUSOIDAL');
      if (l != -1) {
         option.Proj = 3;
         chopt = chopt.replace('SINUSOIDAL', '         '); // Sinusoidal
                                                            // projection
      }
      l = chopt.indexOf('PARABOLIC');
      if (l != -1) {
         option.Proj = 4;
         chopt = chopt.replace('PARABOLIC', '         '); // Parabolic
                                                            // projection
      }
      if (option.Proj > 0) {
         option.Scat = 0;
         option.Contour = 14;
      }
      if (chopt.indexOf('A') != -1)
         option.Axis = -1;
      if (chopt.indexOf('B') != -1)
         option.Bar = 1;
      if (chopt.indexOf('C') != -1) {
         option.Curve = 1;
         option.Hist = -1;
      }
      if (chopt.indexOf('F') != -1)
         option.Fill = 1;
      if (chopt.indexOf('][') != -1) {
         option.Off = 1;
         option.Hist = 1;
      }
      if (chopt.indexOf('F2') != -1) option.Fill = 2;
      if (chopt.indexOf('L') != -1) {
         option.Line = 1;
         option.Hist = -1;
      }
      if (chopt.indexOf('P') != -1) {
         option.Mark = 1;
         option.Hist = -1;
      }
      if (chopt.indexOf('Z') != -1) option.Zscale = 1;
      if (chopt.indexOf('*') != -1) option.Star = 1;
      if (chopt.indexOf('H') != -1) option.Hist = 2;
      if (chopt.indexOf('P0') != -1) option.Mark = 10;
      if (this.IsTH2Poly()) {
         if (option.Fill + option.Line + option.Mark != 0) option.Scat = 0;
      }

      if (chopt.indexOf('E') != -1) {
         if (hdim == 1) {
            option.Error = 1;
            if (chopt.indexOf('E0') != -1) option.Error = 10;
            if (chopt.indexOf('E1') != -1) option.Error = 11;
            if (chopt.indexOf('E2') != -1) option.Error = 12;
            if (chopt.indexOf('E3') != -1) option.Error = 13;
            if (chopt.indexOf('E4') != -1) option.Error = 14;
            if (chopt.indexOf('E5') != -1) option.Error = 15;
            if (chopt.indexOf('E6') != -1) option.Error = 16;
            if (chopt.indexOf('X0') != -1) {
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
      if (chopt.indexOf('9') != -1) option.HighRes = 1;
      if (option.Surf == 15) {
         if (option.System == JSROOT.Painter.Coord.kPOLAR
               || option.System == JSROOT.Painter.Coord.kCARTESIAN) {
            option.Surf = 13;
            // Warning('MakeChopt','option SURF5 is not supported in Cartesian
            // and Polar modes');
         }
      }

      // Check options incompatibilities
      if (option.Bar == 1) option.Hist = -1;

      return option;
   }

   JSROOT.THistPainter.prototype.ScanContent = function() {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values

      alert("HistPainter.prototype.ScanContent not implemented");
   }

   JSROOT.THistPainter.prototype.CheckPadOptions = function() {

      var pad = this.root_pad();

      if (pad!=null) {
         // Copy options from current pad
         this.options.Logx = pad['fLogx'];
         this.options.Logy = pad['fLogy'];
         this.options.Logz = pad['fLogz'];
         this.options.Gridx = pad['fGridx'];
         this.options.Gridy = pad['fGridy'];
      }

      if (this.main_painter() !== this) return;

      this['zoom_xmin'] = 0;
      this['zoom_xmax'] = 0;
      this['zoom_xpad'] = true; // indicate that zooming specified from pad

      this['zoom_ymin'] = 0;
      this['zoom_ymax'] = 0;
      this['zoom_ypad'] = true; // indicate that zooming specified from pad

      if ((pad!=null) && ('fUxmin' in pad) && !this.create_canvas) {
         this['zoom_xmin'] = pad.fUxmin;
         this['zoom_xmax'] = pad.fUxmax;
         this['zoom_ymin'] = pad.fUymin;
         this['zoom_ymax'] = pad.fUymax;

         if (pad.fLogx > 0) {
            this['zoom_xmin'] = Math.exp(this['zoom_xmin'] * Math.log(10));
            this['zoom_xmax'] = Math.exp(this['zoom_xmax'] * Math.log(10));
         }

         if (pad.fLogy > 0) {
            this['zoom_ymin'] = Math.exp(this['zoom_ymin'] * Math.log(10));
            this['zoom_ymax'] = Math.exp(this['zoom_ymax'] * Math.log(10));
         }
      }
   }


   JSROOT.THistPainter.prototype.UpdateObject = function(obj) {
      if (obj['_typename'] != this.histo['_typename']) {
         alert("JSROOT.THistPainter.UpdateObject - wrong class " + obj['_typename'] + " expected " + this.histo['_typename']);
         return false;
      }

      // TODO: simple replace of object does not help - one can have different
      // complex relations between histo and stat box, histo and colz axis,
      // on could have THStack or TMultiGraph object
      // The only that could be done is update of content

      // this.histo = obj;

      this.histo['fArray'] = obj['fArray'];
      this.histo['fN'] = obj['fN'];
      this.histo['fTitle'] = obj['fTitle'];
      this.histo['fXaxis']['fNbins'] = obj['fXaxis']['fNbins'];
      this.histo['fXaxis']['fXmin'] = obj['fXaxis']['fXmin'];
      this.histo['fXaxis']['fXmax'] = obj['fXaxis']['fXmax'];
      this.histo['fYaxis']['fXmin'] = obj['fYaxis']['fXmin'];
      this.histo['fYaxis']['fXmax'] = obj['fYaxis']['fXmax'];

      if (this.IsTProfile()) {
         this.histo['fBinEntries'] = obj['fBinEntries'];
         this.histo['fSumw2'] = obj['fSumw2'];
      }

      this.ScanContent();

      return true;
   }

   JSROOT.THistPainter.prototype.CreateXY = function() {
      // here we create x,y objects which maps our physical coordnates into pixels
      // while only first painter really need such object, all others just reuse it

      if (!this.is_main_painter()) {
         this['x'] = this.main_painter()['x'];
         this['y'] = this.main_painter()['y'];
         return;
      }

      var w = Number(this.svg_frame(true).attr("width")),
          h = Number(this.svg_frame(true).attr("height"));

      this['scale_xmin'] = this.xmin;
      this['scale_xmax'] = this.xmax;
      if (this.zoom_xmin != this.zoom_xmax) {
         this['scale_xmin'] = this.zoom_xmin;
         this['scale_xmax'] = this.zoom_xmax;
      }

      if (this.options.Logx) {
         if (this.scale_xmax <= 0) this.scale_xmax = 0;
         if ((this.scale_xmin <= 0) || (this.scale_xmin >= this.scale_xmax))
            this.scale_xmin = this.scale_xmax * 0.0001;
         this['x'] = d3.scale.log().domain([ this.scale_xmin, this.scale_xmax ]).range([ 0, w ]); // .clamp(true);
      } else {
         this['x'] = d3.scale.linear().domain([ this.scale_xmin, this.scale_xmax ]).range([ 0, w ]);
      }

      this['scale_ymin'] = this.ymin;
      this['scale_ymax'] = this.ymax;

      if (this.zoom_ypad) {
         if (this.histo.fMinimum != -1111) this.zoom_ymin = this.histo.fMinimum;
         if (this.histo.fMaximum != -1111) this.zoom_ymax = this.histo.fMaximum;
         this['zoom_ypad'] = false;
      }

      if (this.zoom_ymin != this.zoom_ymax) {
         this['scale_ymin'] = this.zoom_ymin;
         this['scale_ymax'] = this.zoom_ymax;
      }

      if (this.options.Logy) {
         if (this.scale_ymax <= 0) this.scale_ymax = 1;
         if ((this.scale_ymin <= 0) || (this.scale_ymin >= this.scale_ymax))
            this.scale_ymin = 0.0001 * this.scale_ymax;
         this['y'] = d3.scale.log().domain([ this.scale_ymin, this.scale_ymax ]).range([ h, 0 ]); // .clamp(true);
      } else {
         this['y'] = d3.scale.linear().domain([ this.scale_ymin, this.scale_ymax ]).range([ h, 0 ]);
      }
   }

   JSROOT.THistPainter.prototype.DrawGrids = function() {
      // grid can only be drawn by first painter
      if (!this.is_main_painter()) return;

      this.svg_frame(true).selectAll(".gridLine").remove();
      /* add a grid on x axis, if the option is set */

      // add a grid on x axis, if the option is set
      if (this.options.Gridx) {

         var h = Number(this.svg_frame(true).attr("height"));

         this.svg_frame(true).selectAll("gridLine")
                .data(this.x.ticks(this.x_nticks)).enter()
                  .append("svg:line")
                  .attr("class", "gridLine")
                  .attr("x1", this.x)
                  .attr("y1", h)
                  .attr("x2", this.x)
                  .attr("y2",0)
                  .style("stroke", "black")
                  .style("stroke-width", this.histo['fLineWidth'])
                  .style("stroke-dasharray", JSROOT.Painter.root_line_styles[11]);
      }

      // add a grid on y axis, if the option is set
      if (this.options.Gridy) {

         var w = Number(this.svg_frame(true).attr("width"));

         this.svg_frame(true).selectAll("gridLine")
               .data(this.y.ticks(this.y_nticks)).enter()
                 .append("svg:line")
                 .attr("class", "gridLine")
                 .attr("x1", 0)
                 .attr("y1", this.y)
                 .attr("x2", w)
                 .attr("y2", this.y)
                 .style("stroke", "black")
                 .style("stroke-width", this.histo['fLineWidth'])
                 .style("stroke-dasharray", JSROOT.Painter.root_line_styles[11]);
      }
   }

   JSROOT.THistPainter.prototype.DrawBins = function() {
      alert("HistPainter.DrawBins not implemented");
   }

   JSROOT.THistPainter.prototype.AxisAsText = function(axis, value) {
      if (axis == "x") {
         // this is indication
         if ('dfx' in this) {
            return this.dfx(new Date(this.timeoffsetx + value * 1000));
         }

         if (Math.abs(value) < 1e-14)
            if (Math.abs(this.xmax - this.xmin) > 1e-5)
               value = 0;
         return value.toPrecision(4);
      }

      if (axis == "y") {
         if ('dfy' in this) {
            return this.dfy(new Date(this.timeoffsety + value * 1000));
         }
         if (Math.abs(value) < 1e-14)
            if (Math.abs(this.ymax - this.ymin) > 1e-5)
               value = 0;
         return value.toPrecision(4);
      }

      return value.toPrecision(4);
   }

   JSROOT.THistPainter.prototype.DrawAxes = function(shrink_forbidden) {
      // axes can be drawn only for main histogram

      if (!this.is_main_painter()) return;

      var w = Number(this.svg_frame(true).attr("width")),
          h = Number(this.svg_frame(true).attr("height"));
      var noexpx = this.histo['fXaxis'].TestBit(JSROOT.EAxisBits.kNoExponent);
      var noexpy = this.histo['fYaxis'].TestBit(JSROOT.EAxisBits.kNoExponent);
      var moreloglabelsx = this.histo['fXaxis'].TestBit(JSROOT.EAxisBits.kMoreLogLabels);
      var moreloglabelsy = this.histo['fYaxis'].TestBit(JSROOT.EAxisBits.kMoreLogLabels);
      if (this.histo['fXaxis']['fXmax'] < 100 && this.histo['fXaxis']['fXmax'] / this.histo['fXaxis']['fXmin'] < 100) noexpx = true;
      if (this.histo['fYaxis']['fXmax'] < 100 && this.histo['fYaxis']['fXmax'] / this.histo['fYaxis']['fXmin'] < 100) noexpy = true;

      var xax_g = this.svg_frame(true).selectAll(".xaxis_container");
      if (xax_g.node()==null)
         xax_g = this.svg_frame(true).append("svg:g").attr("class","xaxis_container");

      xax_g.selectAll("*").remove();
      xax_g.attr("transform", "translate(0," + h + ")");

      var yax_g = this.svg_frame(true).selectAll(".yaxis_container");
      if (yax_g.node()==null)yax_g = this.svg_frame(true).append("svg:g").attr("class", "yaxis_container");
      yax_g.selectAll("*").remove();

      var x_axis = null, y_axis = null;

      var ndivx = this.histo['fXaxis']['fNdivisions'];
      this['x_nticks'] = ndivx % 100; // used also to draw grids
      var n2ax = (ndivx % 10000 - this.x_nticks) / 100;
      var n3ax = ndivx / 10000;

      var ndivy = this.histo['fYaxis']['fNdivisions'];
      this['y_nticks'] = ndivy % 100; // used also to draw grids
      var n2ay = (ndivy % 10000 - this.y_nticks) / 100;
      var n3ay = ndivy / 10000;

      /* X-axis label */
      var label = JSROOT.Painter.translateLaTeX(this.histo['fXaxis']['fTitle']);
      var xAxisLabelOffset = 3 + (this.histo['fXaxis']['fLabelOffset'] * h);
      var xAxisLabelFontSize = Math.round(this.histo['fXaxis']['fLabelSize'] * h);

      if (label.length > 0) {
         var xAxisTitleFontSize = Math.round(this.histo['fXaxis']['fTitleSize'] * h);
         var xAxisFontDetails = JSROOT.Painter.getFontDetails(this.histo['fXaxis']['fTitleFont']);
         xax_g.append("text")
               .attr("class", "x_axis_label")
               .attr("x", w)
               .attr("y", xAxisLabelFontSize + xAxisLabelOffset * this.histo['fXaxis']['fTitleOffset'] + xAxisTitleFontSize)
               .attr("text-anchor", "end")
               .attr("font-family", xAxisFontDetails['name'])
               .attr("font-weight", xAxisFontDetails['weight'])
               .attr("font-style", xAxisFontDetails['style'])
               .attr("font-size", xAxisTitleFontSize)
               .text(label);
      }

      /* Y-axis label */
      label = JSROOT.Painter.translateLaTeX(this.histo['fYaxis']['fTitle']);

      var yAxisLabelOffset = 3 + (this.histo['fYaxis']['fLabelOffset'] * w);
      var yAxisLabelFontSize = Math.round(this.histo['fYaxis']['fLabelSize'] * h);

      if (label.length > 0) {
         var yAxisTitleFontSize = Math.round(this.histo['fYaxis']['fTitleSize'] * h);
         var yAxisFontDetails = JSROOT.Painter.getFontDetails(this.histo['fYaxis']['fTitleFont']);
         yax_g.append("text")
                .attr("class", "y_axis_label")
                .attr("x", 0)
                .attr("y", - yAxisLabelFontSize - yAxisTitleFontSize - yAxisLabelOffset * this.histo['fYaxis']['fTitleOffset'])
                .attr("font-family", yAxisFontDetails['name'])
                .attr("font-size", yAxisTitleFontSize)
                .attr("font-weight", yAxisFontDetails['weight'])
                .attr("font-style", yAxisFontDetails['style']).attr("fill", "black")
                .attr("text-anchor", "end")
                .text(label)
                .attr("transform", "rotate(270, 0, 0)");
      }

      var xAxisColor = this.histo['fXaxis']['fAxisColor'];
      var xDivLength = this.histo['fXaxis']['fTickLength'] * h;
      var yAxisColor = this.histo['fYaxis']['fAxisColor'];
      var yDivLength = this.histo['fYaxis']['fTickLength'] * w;

      var pthis = this;

      /*
       * Define the scales, according to the information from the pad
       */
      var xrange = this.xmax - this.xmin;
      if (this.histo['fXaxis']['fTimeDisplay']) {
         if (this.x_nticks > 8) this.x_nticks = 8;

         var timeformatx = JSROOT.Painter.getTimeFormat(this.histo['fXaxis']);

         this['timeoffsetx'] = JSROOT.Painter.getTimeOffset(this.histo['fXaxis']);

         var scale_xrange = this.scale_xmax - this.scale_xmin;

         if ((timeformatx.length == 0) || (scale_xrange < 0.1 * xrange)) {
            timeformatx = JSROOT.Painter.chooseTimeFormat(scale_xrange, this.x_nticks);
         }

         this['dfx'] = d3.time.format(timeformatx);

         x_axis = d3.svg.axis().scale(this.x).orient("bottom")
                    .tickPadding(xAxisLabelOffset)
                    .tickSize(-xDivLength, -xDivLength / 2, -xDivLength / 4)
                    .tickFormat(function(d) { return pthis.dfx(new Date(pthis.timeoffsetx + d * 1000)); })
                    .ticks(this.x_nticks);
      } else if (this.options.Logx) {
         x_axis = d3.svg.axis().scale(this.x).orient("bottom")
                     .tickPadding(xAxisLabelOffset)
                     .tickSize(-xDivLength, -xDivLength / 2, -xDivLength / 4)
                     .tickFormat(function(d) {
                        var val = parseFloat(d);
                        var vlog = Math.abs(JSROOT.Math.log10(val));
                        if (moreloglabelsx) {
                           if (vlog % 1 < 0.7 || vlog % 1 > 0.9999) {
                              if (noexpx)
                                 return val.toFixed();
                              else
                                 return JSROOT.Painter.formatExp(val.toExponential(0));
                           } else
                              return null;
                        } else {
                           if (vlog % 1 < 0.0001 || vlog % 1 > 0.9999) {
                              if (noexpx)
                                 return val.toFixed();
                              else
                                 return JSROOT.Painter.formatExp(val.toExponential(0));
                           } else
                              return null;
                        }
                     });
      } else {
         x_axis = d3.svg.axis()
                    .scale(this.x)
                    .orient("bottom")
                    .tickPadding(xAxisLabelOffset)
                    .tickSize(-xDivLength, -xDivLength / 2, -xDivLength / 4)
                    .tickFormat(function(d) {
                       // avoid rounding problems around 0
                        if ((Math.abs(d) < 1e-14) && (Math.abs(xrange) > 1e-5)) d = 0;
                        return parseFloat(d.toPrecision(12)); })
                     .ticks(this.x_nticks);
      }

      var yrange = this.ymax - this.ymin;
      if (this.histo['fYaxis']['fTimeDisplay']) {
         if (this.y_nticks > 8)  this.y_nticks = 8;
         var timeformaty = JSROOT.Painter.getTimeFormat(this.histo['fYaxis']);

         this['timeoffsety'] = JSROOT.Painter
               .getTimeOffset(this.histo['fYaxis']);

         var scale_yrange = this.scale_ymax - this.scale_ymin;

         if ((timeformaty.length == 0) || (scale_yrange < 0.1 * yrange))
            timeformaty = JSROOT.Painter.chooseTimeFormat(scale_yrange, this.y_nticks);

         this['dfy'] = d3.time.format(timeformaty);

         y_axis = d3.svg.axis()
                      .scale(this.y)
                      .orient("left")
                      .tickPadding(yAxisLabelOffset)
                      .tickSize(-yDivLength, -yDivLength / 2,-yDivLength / 4)
                      .tickFormat(function(d) { return pthis.dfy(new Date(pthis.timeoffsety + d * 1000)); })
                      .ticks(this.y_nticks);
      } else if (this.options.Logy) {
         y_axis = d3.svg.axis()
                    .scale(this.y)
                    .orient("left")
                    .tickPadding(yAxisLabelOffset)
                    .tickSize(-yDivLength, -yDivLength / 2, -yDivLength / 4)
                    .tickFormat(function(d) {
                       var val = parseFloat(d);
                       var vlog = Math.abs(JSROOT.Math.log10(val));
                       if (moreloglabelsy) {
                          if (vlog % 1 < 0.7 || vlog % 1 > 0.9999) {
                             if (noexpy)
                                return val.toFixed();
                             else
                                return JSROOT.Painter.formatExp(val.toExponential(0));
                          } else
                             return null;
                       } else {
                          if (vlog % 1 < 0.0001 || vlog % 1 > 0.9999) {
                             if (noexpy)
                                return val.toFixed();
                             else
                                return JSROOT.Painter.formatExp(val.toExponential(0));
                          } else
                             return null;
                       }
                    });
      } else {
         if (this.y_nticks >= 10) this.y_nticks -= 2;
         y_axis = d3.svg.axis()
                   .scale(this.y)
                   .orient("left")
                   .tickPadding(yAxisLabelOffset)
                   .tickSize(-yDivLength, -yDivLength / 2,-yDivLength / 4)
                   .tickFormat(function(d) {
                       if ((Math.abs(d) < 1e-14) && (Math.abs(yrange) > 1e-5)) d = 0;
                       return parseFloat(d.toPrecision(12)); })
                   .ticks(this.y_nticks);
      }

      xax_g.append("svg:g").attr("class", "xaxis").call(x_axis);

      // this is additional ticks, required in d3.v3
      if ((n2ax > 0) && !this.options.Logx) {
         var x_axis_sub =
             d3.svg.axis().scale(this.x).orient("bottom")
               .tickPadding(xAxisLabelOffset).innerTickSize(-xDivLength / 2)
               .tickFormat(function(d) { return; })
               .ticks(this.x.ticks(this.x_nticks).length * n2ax);

         xax_g.append("svg:g").attr("class", "xaxis").call(x_axis_sub);
      }

      yax_g.append("svg:g").attr("class", "yaxis").call(y_axis);

      // this is additional ticks, required in d3.v3
      if ((n2ay > 0) && !this.options.Logy) {
         var y_axis_sub = d3.svg.axis().scale(this.y).orient("left")
               .tickPadding(yAxisLabelOffset).innerTickSize(-yDivLength / 2)
               .tickFormat(function(d) {  return; })
               .ticks(this.y.ticks(this.y_nticks).length * n2ay);

         yax_g.append("svg:g").attr("class", "yaxis").call(y_axis_sub);
      }

      var xAxisLabelFontDetails = JSROOT.Painter.getFontDetails(this.histo['fXaxis']['fLabelFont']);
      var yAxisLabelFontDetails = JSROOT.Painter.getFontDetails(this.histo['fYaxis']['fLabelFont']);

      xax_g.selectAll("text")
            .attr("font-family", xAxisLabelFontDetails['name'])
            .attr("font-size", xAxisLabelFontSize)
            .attr("font-weight", xAxisLabelFontDetails['weight'])
            .attr("font-style", xAxisLabelFontDetails['style']);

      yax_g.selectAll("text")
            .attr("font-family", yAxisLabelFontDetails['name'])
            .attr("font-size", yAxisLabelFontSize)
            .attr("font-weight", yAxisLabelFontDetails['weight'])
            .attr("font-style",  yAxisLabelFontDetails['style']);

      // we will use such rect for zoom selection
      xax_g.append("svg:rect")
           .attr("class", "xaxis_zoom")
           .attr("x", 0)
           .attr("y", 0)
           .attr("width", w)
           .attr("height", xAxisLabelFontSize + 3)
           .style('opacity', "0");

      // we will use such rect for zoom selection
      yax_g.append("svg:rect")
           .attr("class", "yaxis_zoom")
           .attr("x",-2 * yAxisLabelFontSize - 3)
           .attr("y", 0)
           .attr("width", 2 * yAxisLabelFontSize + 3)
           .attr("height", h)
           .style('opacity', "0");

      if ((shrink_forbidden==null) && typeof yax_g.node()['getBoundingClientRect'] == 'function') {

         var rect1 = yax_g.node().getBoundingClientRect();
         var rect2 = this.svg_pad().getBoundingClientRect();
         var position = rect1.left - rect2.left;

         var shrink = 0.;

         if (position < 0) {
            shrink = -position/w + 0.001;
            this.shrink_frame_left += shrink;
         } else
         if ((this.shrink_frame_left > 0) && (position/w > this.shrink_frame_left)) {
            shrink = -this.shrink_frame_left;
            this.shrink_frame_left = 0.;
         }

         if (shrink != 0) {
            this.svg_frame()['frame_painter'].Shrink(shrink, 0);
            this.svg_frame()['frame_painter'].Redraw();
            this.CreateXY();
            this.DrawAxes(true);
         }
      }
   }

   JSROOT.THistPainter.prototype.DrawTitle = function() {

      var painter = this.FindPainterFor(null,"title");

      if (painter!=null) {
         painter.pavetext.Clear();
         painter.pavetext.AddText(this.histo['fTitle']);
      } else {

         var pavetext = JSROOT.Create("TPaveText");

         jQuery.extend(pavetext, { fName: "title",
                                   fX1NDC: 0.2809483, fY1NDC: 0.9339831,
                                   fX2NDC: 0.7190517, fY2NDC: 0.995});
         pavetext.AddText(this.histo['fTitle']);

         painter = JSROOT.Painter.drawPaveText(this.divid, pavetext);
      }
   }

   JSROOT.THistPainter.prototype.ToggleStat = function() {

      var stat = this.FindStat();

      if (stat == null) {

         // when statbox created first time, one need to draw it
         stat = this.CreateStat();

         this.Redraw();

         return;
      }

      var statpainter = this.FindPainterFor(stat);
      if (statpainter == null) {
         alert("Did not found painter for existing stat??");
         return;
      }

      statpainter.Enabled = !statpainter.Enabled;

      // when stat box is drawed, it always can be draw individualy while it
      // should be last for colz RedrawPad is used
      statpainter.Redraw();
   }

   JSROOT.THistPainter.prototype.GetSelectIndex = function(axis, size, add) {
      // be aware - here indexs starts from 0
      var indx = 0;
      var obj = this.main_painter();
      if (obj == null) obj = this;
      var nbin = 0;
      if (!add) add = 0;

      if (axis == "x") {
         nbin = this.nbinsx;
         if (obj.zoom_xmin != obj.zoom_xmax) {
            if (size == "left")
               indx = Math.floor((obj.zoom_xmin - this.xmin) / this.binwidthx + add);
            else
               indx = Math.round((obj.zoom_xmax - this.xmin) / this.binwidthx + 0.5 + add);
         } else {
            indx = (size == "left") ? 0 : nbin;
         }

      } else
      if (axis == "y") {
         nbin = this.nbinsy;
         if (obj.zoom_ymin != obj.zoom_ymax) {
            if (size == "left")
               indx = Math.floor((obj.zoom_ymin - this.ymin) / this.binwidthy + add);
            else
               indx = Math.round((obj.zoom_ymax - this.ymin) / this.binwidthy + 0.5 + add);
         } else {
            indx = (size == "left") ? 0 : nbin;
         }
      }

      if (size == "left") {
         if (indx < 0) indx = 0;
      } else {
         if (indx > nbin) indx = nbin;
      }

      return indx;
   }

   JSROOT.THistPainter.prototype.FindStat = function() {

      if ('fFunctions' in this.histo)
         for ( var i in this.histo.fFunctions.arr) {

            var func = this.histo.fFunctions.arr[i];

            if (func['_typename'] == 'TPaveText' || func['_typename'] == 'TPaveStats') {
               return func;
            }
         }

      return null;
   }

   JSROOT.THistPainter.prototype.CreateStat = function() {

      if (!this.draw_content) return null;
      if (this.FindStat() != null) return null;

      var stats = JSROOT.Create('TPaveStats');
      jQuery.extend(stats, { _AutoCreated: true,
                             fName : 'stats',
                             fOptStat: JSROOT.gStyle.OptStat,
                             fBorderSize : 1} );
      jQuery.extend(stats, JSROOT.gStyle.StatNDC);
      jQuery.extend(stats, JSROOT.gStyle.StatText);
      jQuery.extend(stats, JSROOT.gStyle.StatFill);

      if (this.histo['_typename'] && (this.histo['_typename'].match(/^TProfile/) || this.histo['_typename'].match(/^TH2/)))
         stats['fY1NDC'] = 0.67;

      stats.AddText(this.histo['fName']);

      if (!'fFunctions' in this.histo)
         this.histo['fFunctions'] = JSROOT.Create("TList");

      this.histo.fFunctions.arr.push(stats);

      return stats;
   }

   JSROOT.THistPainter.prototype.DrawFunctions = function() {

      // draw statistics box & other TPaveTexts, which are belongs to histogram
      // should be called once to create all painters, which are than updated separately

      if (!('fFunctions' in this.histo))  return;
      // if (this.options.Func == 0) return; // in some cases on need to disable
      // functions drawing

      var lastpainter = this;

      var kNotDraw = JSROOT.BIT(9); // don't draw the function (TF1) when in a TH1

      var EStatusBits = {
         kCanDelete : JSROOT.BIT(0), // if object in a list can be deleted
         kMustCleanup : JSROOT.BIT(3), // if object destructor must call RecursiveRemove()
         kObjInCanvas : JSROOT.BIT(3), // for backward compatibility only, use kMustCleanup
         kIsReferenced : JSROOT.BIT(4), // if object is referenced by a TRef or TRefArray
         kHasUUID : JSROOT.BIT(5), // if object has a TUUID (its fUniqueID=UUIDNumber)
         kCannotPick : JSROOT.BIT(6), // if object in a pad cannot be picked
         kNoContextMenu : JSROOT.BIT(8), // if object does not want context menu
         kInvalidObject : JSROOT.BIT(13)  // if object ctor succeeded but object should not be used
      }

      for ( var i in this.histo.fFunctions.arr) {

         var func = this.histo.fFunctions.arr[i];

         var funcpainter = this.FindPainterFor(func);

         // no need to do something if painter for object was already done
         // object will be redraw automatically
         if (funcpainter != null) continue;

         if (func['_typename'] == 'TPaveText' || func['_typename'] == 'TPaveStats') {
            funcpainter = JSROOT.Painter.drawPaveText(this.divid, func);
         } else

         if (func['_typename'] == 'TF1') {
            var is_pad = this.root_pad() != null;
            if ((!is_pad && !func.TestBit(kNotDraw))
                  || (is_pad && func.TestBit(EStatusBits.kObjInCanvas)))
               funcpainter = JSROOT.Painter.drawFunction(this.divid, func);
         } else

         if (func['_typename'] == 'TPaletteAxis') {
            funcpainter = JSROOT.Painter.drawPaletteAxis(this.divid, func);
         }
      }
   }

   JSROOT.THistPainter.prototype.Redraw = function() {
      this.CreateXY();
      this.DrawAxes();
      this.DrawGrids();
      this.DrawBins();
      if (this.create_canvas) this.DrawTitle();
   }

   JSROOT.THistPainter.prototype.Unzoom = function(dox, doy) {
      var obj = this.main_painter();
      if (!obj) obj = this;

      var changed = false;

      if (dox) {
         if (obj['zoom_xmin'] != obj['zoom_xmax']) changed = true;
         obj['zoom_xmin'] = 0;
         obj['zoom_xmax'] = 0;
      }
      if (doy) {
         if (obj['zoom_ymin'] != obj['zoom_ymax']) changed = true;
         obj['zoom_ymin'] = 0;
         obj['zoom_ymax'] = 0;
      }

      if (changed) this.RedrawPad();
   }

   JSROOT.THistPainter.prototype.Zoom = function(xmin, xmax, ymin, ymax) {
      var obj = this.main_painter();
      if (!obj) obj = this;

      var isany = false;

      if ((xmin != xmax) && (Math.abs(xmax-xmin) > obj.binwidthx*2.0)) {
         obj['zoom_xmin'] = xmin;
         obj['zoom_xmax'] = xmax;
         isany = true;
      }
      if ((ymin != ymax) && (Math.abs(ymax-ymin) > (('binwidthy' in obj) ? (obj.binwidthy*2.0) : Math.abs(obj.ymax-obj.ymin)*1e-6))) {
         obj['zoom_ymin'] = ymin;
         obj['zoom_ymax'] = ymax;
         isany = true;
      }

      if (isany) this.RedrawPad();
   }

   JSROOT.THistPainter.prototype.AddInteractive = function() {
      // only first painter in list allowed to add interactive functionality to the main pad
      if (!this.is_main_painter()) return;

      // if (!this.draw_content) return;

      var width = Number(this.svg_frame(true).attr("width")),
          height = Number(this.svg_frame(true).attr("height"));
      var e, origin, curr = null, rect = null;
      var lasttouch = new Date(0);

      var zoom_kind = 0; // 0 - none, 1 - XY, 2 - only X, 3 - only Y, (+100 for touches)

      var disable_tooltip = false;

      // var zoom = d3.behavior.zoom().x(this.x).y(this.y);

      var pthis = this;

      function closeAllExtras() {
         var x = document.getElementById('root_ctx_menu');
         if (x) x.parentNode.removeChild(x);
         if (rect != null) { rect.remove(); rect = null; }
         zoom_kind = 0;
         if (disable_tooltip) {
            JSROOT.gStyle.Tooltip = true;
            disable_tooltip = false;
         }
      }

      function showContextMenu() {

         d3.event.preventDefault();

         // ignore context menu when touches zooming is ongoing
         if (zoom_kind > 100) return;

         // suppress any running zomming
         closeAllExtras();

         var menu = JSROOT.Painter.createmenu(d3.event, 'root_ctx_menu');

         menu['painter'] = pthis;

         JSROOT.Painter.menuitem(menu, pthis.histo['fName']);
         JSROOT.Painter.menuitem(menu, "----------------");

         pthis.FillContextMenu(menu);
      }

      function startTouchSel() {

         // in case when zooming was started, block any other kind of events
         if (zoom_kind != 0) {
            d3.event.preventDefault();
            d3.event.stopPropagation();
            return;
         }

         // update frame dimensions while frame could be resized
         width = Number(pthis.svg_frame(true).attr("width"));
         height = Number(pthis.svg_frame(true).attr("height"));

         e = this;
         // var t = d3.event.changedTouches;
         var arr = d3.touches(e);

         // only double-touch will be handled
         if (arr.length == 1) {

            var now = new Date();
            var diff = now.getTime() - lasttouch.getTime();

            if ((diff < 300) && (curr != null)
                  && (Math.abs(curr[0] - arr[0][0]) < 30)
                  && (Math.abs(curr[1] - arr[0][1]) < 30)) {

               d3.event.preventDefault();
               d3.event.stopPropagation();

               closeAllExtras();
               pthis.Unzoom(true, true);
            } else {
               lasttouch = now;
               curr = arr[0];
            }
         }

         if (arr.length != 2) return;

         d3.event.preventDefault();

         closeAllExtras();

         var pnt1 = arr[0];
         var pnt2 = arr[1];

         curr = new Array; // minimum
         origin = new Array; // maximum

         curr.push(Math.min(pnt1[0], pnt2[0]));
         curr.push(Math.min(pnt1[1], pnt2[1]));
         origin.push(Math.max(pnt1[0], pnt2[0]));
         origin.push(Math.max(pnt1[1], pnt2[1]));

         if (curr[0] < 0) {
            zoom_kind = 103; // only y
            curr[0] = 0;
            origin[0] = width;
         } else if (origin[1] > height) {
            zoom_kind = 102; // only x
            curr[1] = 0;
            origin[1] = height;
         } else {
            zoom_kind = 101; // x and y
         }

         // d3.select("body").classed("noselect", true);
         // d3.select("body").style("-webkit-user-select", "none");

         rect = pthis.svg_frame(true).append("rect")
               .attr("class", "zoom")
               .attr("id", "zoomRect")
               .attr("x", curr[0])
               .attr("y", curr[1])
               .attr("width", origin[0] - curr[0])
               .attr("height", origin[1] - curr[1]);

         // pthis.svg_frame(true).on("dblclick", unZoom);

         d3.select(window).on("touchmove.zoomRect", moveTouchSel)
                          .on("touchcancel.zoomRect", endTouchSel)
                          .on("touchend.zoomRect", endTouchSel, true);
         d3.event.stopPropagation();
      }

      function moveTouchSel() {
         if (zoom_kind < 100) return;

         d3.event.preventDefault();

         // var t = d3.event.changedTouches;
         var arr = d3.touches(e);

         if (arr.length != 2) {
            closeAllExtras();
            zoom_kind = 0;
            return;
         }

         var pnt1 = arr[0];
         var pnt2 = arr[1];

         if (zoom_kind != 103) {
            curr[0] = Math.min(pnt1[0], pnt2[0]);
            origin[0] = Math.max(pnt1[0], pnt2[0]);
         }
         if (zoom_kind != 102) {
            curr[1] = Math.min(pnt1[1], pnt2[1]);
            origin[1] = Math.max(pnt1[1], pnt2[1]);
         }

         rect.attr("x", curr[0])
             .attr("y", curr[1])
             .attr("width", origin[0] - curr[0])
             .attr("height", origin[1] - curr[1]);

         if (JSROOT.gStyle.Tooltip && ((origin[0] - curr[0]>10) || (origin[1] - curr[1]>10))) {
            JSROOT.gStyle.Tooltip = false;
            disable_tooltip = true;
         }

         d3.event.stopPropagation();
      }

      function endTouchSel() {

         if (zoom_kind < 100) return;

         d3.event.preventDefault();
         d3.select(window).on("touchmove.zoomRect", null)
                          .on("touchend.zoomRect", null)
                          .on("touchcancel.zoomRect", null);
         d3.select("body").classed("noselect", false);

         var xmin = 0, xmax = 0, ymin = 0, ymax = 0;

         var isany = false;

         if ((zoom_kind != 103) && (Math.abs(curr[0] - origin[0]) > 10)) {
            xmin = Math.min(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            xmax = Math.max(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            isany = true;
         }

         if ((zoom_kind != 102) && (Math.abs(curr[1] - origin[1]) > 10)) {
            ymin = Math.min(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            ymax = Math.max(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            isany = true;
         }

         d3.select("body").style("-webkit-user-select", "auto");

         if (disable_tooltip)
            JSROOT.gStyle.Tooltip = true;

         rect.remove();
         rect = null;
         zoom_kind = 0;

         if (isany) pthis.Zoom(xmin, xmax, ymin, ymax);

         d3.event.stopPropagation();
      }

      function detectLeftButton(event) {
         if ('buttons' in event) return event.buttons === 1;
         else if ('which' in event) return event.which === 1;
         else return event.button === 1;
       }

      function startRectSel() {

         // use only left button
         // if (!detectLeftButton(d3.event)) return;

         // ignore when touch selection is actiavated
         if (zoom_kind > 100) return;

         d3.event.preventDefault();

         // update frame dimensions while frame could be resized
         width = Number(pthis.svg_frame(true).attr("width"));
         height = Number(pthis.svg_frame(true).attr("height"));

         closeAllExtras();

         e = this;
         origin = d3.mouse(e);

         curr = new Array;
         curr.push(Math.max(0, Math.min(width, origin[0])));
         curr.push(Math.max(0, Math.min(height, origin[1])));

         if (origin[0] < 0) {
            zoom_kind = 3; // only y
            origin[0] = 0;
            origin[1] = curr[1];
            curr[0] = width;
            curr[1] += 1;
         } else if (origin[1] > height) {
            zoom_kind = 2; // only x
            origin[0] = curr[0];
            origin[1] = 0;
            curr[0] += 1;
            curr[1] = height;
         } else {
            zoom_kind = 1; // x and y
            origin[0] = curr[0];
            origin[1] = curr[1];
         }

         // d3.select("body").classed("noselect", true);
         // d3.select("body").style("-webkit-user-select", "none");

         rect = pthis.svg_frame(true)
                .append("rect")
                .attr("class", "zoom")
                .attr("id", "zoomRect");

         pthis.svg_frame(true).on("dblclick", unZoom);

         d3.select(window).on("mousemove.zoomRect", moveRectSel)
                          .on("mouseup.zoomRect", endRectSel, true);

         d3.event.stopPropagation();
      }

      function unZoom() {
         d3.event.preventDefault();
         var m = d3.mouse(e);
         closeAllExtras();
         if (m[0] < 0) pthis.Unzoom(false, true); else
         if (m[1] > height) pthis.Unzoom(true, false); else {
            pthis.Unzoom(true, true);
            pthis.svg_frame(true).on("dblclick", null);
         }
      }

      function moveRectSel() {

         if ((zoom_kind == 0) || (zoom_kind > 100)) return;

         d3.event.preventDefault();
         var m = d3.mouse(e);

         m[0] = Math.max(0, Math.min(width, m[0]));
         m[1] = Math.max(0, Math.min(height, m[1]));

         switch (zoom_kind) {
            case 1: curr[0] = m[0]; curr[1] = m[1]; break;
            case 2: curr[0] = m[0]; break;
            case 3: curr[1] = m[1]; break;
         }

         rect.attr("x", Math.min(origin[0], curr[0]))
             .attr("y", Math.min(origin[1], curr[1]))
             .attr("width", Math.abs(curr[0] - origin[0]))
             .attr("height", Math.abs(curr[1] - origin[1]));

         if (JSROOT.gStyle.Tooltip && ((Math.abs(curr[0] - origin[0])>10) || (Math.abs(curr[1] - origin[1])>10))) {
            JSROOT.gStyle.Tooltip = false;
            disable_tooltip = true;
         }
      }

      function endRectSel() {
         if ((zoom_kind == 0) || (zoom_kind > 100)) return;

         d3.event.preventDefault();
         // d3.select(window).on("touchmove.zoomRect",
         // null).on("touchend.zoomRect", null);
         d3.select(window).on("mousemove.zoomRect", null)
                          .on("mouseup.zoomRect", null);
         d3.select("body").classed("noselect", false);

         var m = d3.mouse(e);

         m[0] = Math.max(0, Math.min(width, m[0]));
         m[1] = Math.max(0, Math.min(height, m[1]));

         switch (zoom_kind) {
            case 1: curr[0] = m[0]; curr[1] = m[1]; break;
            case 2: curr[0] = m[0]; break; // only X
            case 3: curr[1] = m[1]; break; // only Y
         }

         var xmin = 0, xmax = 0, ymin = 0, ymax = 0;

         var isany = false;

         if ((zoom_kind != 3) && (Math.abs(curr[0] - origin[0]) > 10)) {
            xmin = Math.min(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            xmax = Math.max(pthis.x.invert(origin[0]), pthis.x.invert(curr[0]));
            isany = true;
         }

         if ((zoom_kind != 2) && (Math.abs(curr[1] - origin[1]) > 10)) {
            ymin = Math.min(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            ymax = Math.max(pthis.y.invert(origin[1]), pthis.y.invert(curr[1]));
            isany = true;
         }

         d3.select("body").style("-webkit-user-select", "auto");

         if (disable_tooltip) {
            JSROOT.gStyle.Tooltip = true;
            disable_tooltip = false;
         }

         rect.remove();
         rect = null;
         zoom_kind = 0;

         if (isany) pthis.Zoom(xmin, xmax, ymin, ymax);
      }

      this.svg_frame(true).on("mousedown", startRectSel);
      this.svg_frame(true).on("touchstart", startTouchSel);
      this.svg_frame(true).on("contextmenu", showContextMenu);

   }

   JSROOT.THistPainter.prototype.FillContextMenu = function(menu) {
      JSROOT.Painter.menuitem(menu, "Unzoom X", function() {
         menu['painter'].Unzoom(true, false);
      });
      JSROOT.Painter.menuitem(menu, "Unzoom Y", function() {
         menu['painter'].Unzoom(false, true);
      });
      JSROOT.Painter.menuitem(menu, "Unzoom", function() {
         menu['painter'].Unzoom(true, true);
      });

      JSROOT.Painter.menuitem(menu, JSROOT.gStyle.Tooltip ? "Disable tooltip" : "Enable tooltip", function() {
         JSROOT.gStyle.Tooltip = !JSROOT.gStyle.Tooltip;
         console.log("Change tooltip " + JSROOT.gStyle.Tooltip);
         menu['painter'].RedrawPad();
      });

      if (this.options) {

         var item = this.options.Logx > 0 ? "Linear X" : "Log X";

         JSROOT.Painter.menuitem(menu, item, function() {
            menu['painter'].options.Logx = 1 - menu['painter'].options.Logx;
            menu['painter'].RedrawPad();
         });

         var item = this.options.Logy > 0 ? "Linear Y" : "Log Y";
         JSROOT.Painter.menuitem(menu, item, function() {
            menu['painter'].options.Logy = 1 - menu['painter'].options.Logy;
            menu['painter'].RedrawPad();
         });
      }
      if (this.draw_content)
         JSROOT.Painter.menuitem(menu, "Toggle stat", function() {
            menu['painter'].ToggleStat();
         });
   }

   // ======= TH1 painter================================================

   JSROOT.TH1Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);
   }

   JSROOT.TH1Painter.prototype = Object.create(JSROOT.THistPainter.prototype);

   JSROOT.TH1Painter.prototype.ScanContent = function() {

      // from here we analyze object content
      // therefore code will be moved
      this.fillcolor = JSROOT.Painter.createFillPattern(this.svg_canvas(true), this.histo['fFillStyle'], this.histo['fFillColor']);
      if (this.histo['fFillStyle'] >= 4000 && this.histo['fFillStyle'] <= 4100) this.fillcolor = 'none';

      this.linecolor = JSROOT.Painter.root_colors[this.histo['fLineColor']];
      if (this.histo['fLineColor'] == 0) this.linecolor = '#4572A7';

      var hmin = 0, hmax = 0, hsum = 0;

      var profile = this.IsTProfile();

      this.nbinsx = this.histo['fXaxis']['fNbins'];

      for (var i = 0; i < this.nbinsx; ++i) {
         var value = this.histo.getBinContent(i + 1);
         hsum += profile ? this.histo.fBinEntries[i + 1] : value;
         if (i == 0) hmin = hmax = value;
         if (value < hmin) hmin = value; else
         if (value > hmax) hmax = value;
      }

      // account overflow/underflow bins
      if (profile)
         hsum += this.histo.fBinEntries[0] + this.histo.fBinEntries[this.nbinsx + 1];
      else
         hsum += this.histo.getBinContent(0) + this.histo.getBinContent(this.nbinsx + 1);

      this.stat_entries = hsum;

      // used in CreateXY and tooltip providing
      this.xmin = this.histo['fXaxis']['fXmin'];
      this.xmax = this.histo['fXaxis']['fXmax'];

      this.binwidthx = (this.xmax - this.xmin);
      if (this.nbinsx > 0)
         this.binwidthx = this.binwidthx / this.nbinsx;

      this.ymin = this.histo['fYaxis']['fXmin'];
      this.ymax = this.histo['fYaxis']['fXmax'];

      if ((this.nbinsx == 0) || ((Math.abs(hmin) < 1e-300 && Math.abs(hmax) < 1e-300))) {
         if (this.histo['fMinimum'] != -1111) this.ymin = this.histo['fMinimum'];
         if (this.histo['fMaximum'] != -1111) this.ymax = this.histo['fMaximum'];
         this.draw_content = false;
      } else {
         if (this.histo['fMinimum'] != -1111) hmin = this.histo['fMinimum'];
         if (this.histo['fMaximum'] != -1111) hmax = this.histo['fMaximum'];
         if (hmin >= hmax) {
            if (hmin == 0) { this.ymax = 0; this.ymax = 1; } else
            if (hmin < 0) { this.ymin = 2 * hmin; this.ymax = 0; }
                     else { this.ymin = 0; this.ymax = hmin * 2; }
         } else {
            var dy = (hmax - hmin) * 0.1;
            this.ymin = hmin - dy;
            if ((this.ymin < 0) && (hmin >= 0)) this.ymin = 0;
            this.ymax = hmax + dy;
         }
         this.draw_content = true;
      }

      // If no any draw options specified, do not try draw histogram
      if (this.options.Bar == 0 && this.options.Hist == 0
            && this.options.Error == 0 && this.options.Same == 0) {
         this.draw_content = false;
      }
      if (this.options.Axis > 0) // Paint histogram axis only
         this.draw_content = false;
   }

   JSROOT.TH1Painter.prototype.CountStat = function(cond) {
      var profile = this.IsTProfile();

      var stat_sumw = 0, stat_sumwx = 0, stat_sumwx2 = 0, stat_sumwy = 0, stat_sumwy2 = 0;

      var left = this.GetSelectIndex("x", "left");
      var right = this.GetSelectIndex("x", "right");

      var xx = 0, w = 0, xmax = null, wmax = null;

      for (var i = left; i < right; i++) {
         xx = this.xmin + (i + 0.5) * this.binwidthx;

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

   JSROOT.TH1Painter.prototype.FillStatistic = function(stat, dostat) {
      if (!this.histo) return false;

      var data = this.CountStat();

      var print_name = Math.floor(dostat % 10);
      var print_entries = Math.floor(dostat / 10) % 10;
      var print_mean = Math.floor(dostat / 100) % 10;
      var print_rms = Math.floor(dostat / 1000) % 10;
      var print_under = Math.floor(dostat / 10000) % 10;
      var print_over = Math.floor(dostat / 100000) % 10;
      var print_integral = Math.floor(dostat / 1000000) % 10;
      var print_skew = Math.floor(dostat / 10000000) % 10;
      var print_kurt = Math.floor(dostat / 100000000) % 10;

      if (print_name > 0)
         stat.AddLine(this.histo['fName']);

      if (this.IsTProfile()) {

         if (print_entries > 0)
            stat.AddLine("Entries = " + JSROOT.gStyle.StatEntriesFormat(data.entries));

         if (print_mean > 0) {
            stat.AddLine("Mean = " + JSROOT.gStyle.StatFormat(data.meanx));
            stat.AddLine("Mean y = " + JSROOT.gStyle.StatFormat(data.meany));
         }

         if (print_rms > 0) {
            stat.AddLine("RMS = " + JSROOT.gStyle.StatFormat(data.rmsx));
            stat.AddLine("RMS y = " + JSROOT.gStyle.StatFormat(data.rmsy));
         }

      } else {

         if (print_entries > 0)
            stat.AddLine("Entries = " + JSROOT.gStyle.StatEntriesFormat(data.entries));

         if (print_mean > 0) {
            stat.AddLine("Mean = " + JSROOT.gStyle.StatFormat(data.meanx));
         }

         if (print_rms > 0) {
            stat.AddLine("RMS = " + JSROOT.gStyle.StatFormat(data.rmsx));
         }

         if (print_under > 0) {
            var res = 0;
            if (this.histo['fArray'].length > 0)
               res = this.histo['fArray'][0];
            stat.AddLine("Underflow = " + JSROOT.gStyle.StatFormat(res));
         }

         if (print_over > 0) {
            var res = 0;
            if (this.histo['fArray'].length > 0)
               res = this.histo['fArray'][this.histo['fArray'].length - 1];
            stat.AddLine("Overflow = " + JSROOT.gStyle.StatFormat(res));
         }

         if (print_integral > 0) {
            stat.AddLine("Integral = " + JSROOT.gStyle.StatEntriesFormat(data.integral));
         }

         if (print_skew > 0)
            stat.AddLine("Skew = not avail");

         if (print_kurt > 0)
            stat.AddLine("Kurt = not avail");
      }

      // adjust the size of the stats box with the number of lines
      var nlines = stat.pavetext['fLines'].arr.length;
      var stath = nlines * JSROOT.gStyle.StatFontSize;
      if (stath <= 0 || 3 == (JSROOT.gStyle.StatFont % 10)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         stat.pavetext['fY1NDC'] = 0.93 - stath;
         stat.pavetext['fY2NDC'] = 0.93;
      }

      return true;
   }

   JSROOT.TH1Painter.prototype.CreateDrawBins = function(width, height) {
      // method is called directly before bins must be drawn

      var left = this.GetSelectIndex("x", "left", -1);
      var right = this.GetSelectIndex("x", "right", 2);
      var stepi = 1;

      var draw_bins = new Array;

      // reduce number of drawn points - we define interval where two points
      // will be selected - max and min
      if ((this.nbinsx > 10000) || (JSROOT.gStyle.OptimizeDraw && (right - left > width)))
         while ((right - left) / stepi > width) stepi++;

      var x1, x2 = this.xmin + left * this.binwidthx;
      var grx1 = -1111, grx2 = -1111, gry;
      var profile = this.IsTProfile();

      var point = null;

      for (var i = left; i < right; i += stepi) {
         // if interval wider than specified range, make it shorter
         if ((stepi > 1) && (i + stepi > right)) stepi = (right - i);
         x1 = x2;
         x2 += stepi * this.binwidthx;

         if (this.options.Logx && (x1 <= 0)) continue;

         grx1 = grx2;
         grx2 = this.x(x2);
         if (grx1 < 0) grx1 = this.x(x1);

         var pmax = i, cont = this.histo.getBinContent(i + 1);

         for (var ii = 1; ii < stepi; ii++) {
            var ccc = this.histo.getBinContent(i + ii + 1);
            if (ccc > cont) {
               cont = ccc;
               pmax = i + ii;
            }
         }

         // exclude zero bins from profile drawings
         if (profile && (cont==0)) continue;

         if (this.options.Logy && (cont < this.scale_ymin))
            gry = height + 10;
         else
            gry = this.y(cont);

         point = { x : grx1, y : gry };

         if (this.options.Error > 0) {
            point['xerr'] = (grx2 - grx1) / 2;
            point['yerr'] = gry - this.y(cont + this.histo.getBinError(pmax + 1));
         }

         if (this.options.Error > 0) {
            point['x'] = (grx1 + grx2) / 2;
            point['tip'] = "x = " + this.AxisAsText("x", (x1 + x2)/2 ) + "\n" +
                           "y = " + this.AxisAsText("y", cont) + "\n" +
                           "error x = " + ((x2 - x1) / 2).toPrecision(4) + "\n" +
                           "error y = " + this.histo.getBinError(pmax + 1).toPrecision(4);
         } else {
            point['width'] = grx2 - grx1;

            point['tip'] = "bin = " + (pmax + 1) + "\n" +
                           "x = [" + this.AxisAsText("x", x1) + ", " + this.AxisAsText("x", x2) + "]\n" +
                           "entries = " + cont;
         }

         draw_bins.push(point);
      }

      // if we need to draw line or area, we need extra point for correct drawing
      if ((right == this.nbinsx) && (this.options.Error == 0) && (point!=null)) {
         var extrapoint = jQuery.extend(true, {}, point);
         extrapoint.x = grx2;
         draw_bins.push(extrapoint);
      }

      return draw_bins;
   }

   JSROOT.TH1Painter.prototype.DrawErrors = function(draw_bins) {
      var w = Number(this.svg_frame(true).attr("width")),
          h = Number(this.svg_frame(true).attr("height"));

      /* Add a panel for each data point */
      var info_marker = JSROOT.Painter.getRootMarker(this.histo['fMarkerStyle']);
      var marker_size = this.histo['fMarkerSize'] * 32;

      var line_width = this.histo['fLineWidth'];
      var line_color = JSROOT.Painter.root_colors[this.histo['fLineColor']];
      var marker_color = JSROOT.Painter.root_colors[this.histo['fMarkerColor']];

      if (this.histo['fMarkerStyle'] == 1) marker_size = 1;

      var marker = d3.svg.symbol().type(d3.svg.symbolTypes[info_marker.shape]).size(marker_size);

      var pthis = this;

      /* Draw x-error indicators */
      var xerr = this.draw_g.selectAll("error_x")
                 .data(draw_bins).enter()
                 .append("svg:line")
                 .attr("x1", function(d) { return d.x - d.xerr; })
                 .attr("y1", function(d) { return d.y; })
                 .attr("x2", function(d) { return d.x + d.xerr; })
                 .attr("y2", function(d) { return d.y; })
                 .style("stroke", line_color)
                 .style("stroke-width", line_width);

      if (this.options.Error == 11) {
         this.draw_g.selectAll("e1_x")
            .data(draw_bins).enter()
            .append("svg:line")
            .attr("y1", function(d) { return d.y - 3; })
            .attr("x1", function(d) { return d.x - d.xerr; })
            .attr("y2", function(d) { return d.y + 3; })
            .attr("x2", function(d) { return d.x - d.xerr; })
            .style("stroke", line_color)
            .style("stroke-width", line_width);
         this.draw_g.selectAll("e1_x")
            .data(draw_bins).enter()
            .append("svg:line")
            .attr("y1", function(d) { return d.y - 3; })
            .attr("x1", function(d) { return d.x + d.xerr; })
            .attr("y2", function(d) { return d.y + 3; })
            .attr("x2", function(d) { return d.x + d.xerr; })
            .style("stroke", line_color)
            .style("stroke-width", line_width);
      }

      /* Draw y-error indicators */
      var yerr = this.draw_g.selectAll("error_y")
                   .data(draw_bins).enter()
                   .append("svg:line")
                   .attr("x1", function(d) { return d.x; })
                   .attr("y1", function(d) { return d.y - d.yerr; })
                   .attr("x2", function(d) { return d.x; })
                   .attr("y2", function(d) { return d.y + d.yerr; })
                   .style("stroke", line_color).style("stroke-width", line_width);

      if (this.options.Error == 11) {
         this.draw_g.selectAll("e1_y")
             .data(draw_bins).enter()
             .append("svg:line")
             .attr("x1", function(d) { return d.x - 3; })
             .attr("y1", function(d) { return d.y - d.yerr; })
             .attr("x2", function(d) { return d.x + 3; })
             .attr("y2", function(d) { return d.y - d.yerr; })
             .style("stroke", line_color)
             .style("stroke-width", line_width);
         this.draw_g.selectAll("e1_y")
              .data(draw_bins).enter()
              .append("svg:line")
              .attr("x1", function(d) { return d.x - 3; })
              .attr("y1", function(d) { return d.y + d.yerr; })
              .attr("x2", function(d) { return d.x + 3; })
              .attr("y2", function(d) { return d.y + d.yerr; })
              .style("stroke", line_color)
              .style("stroke-width", line_width);
      }
      var marks = this.draw_g.selectAll("markers")
                    .data(draw_bins).enter()
                    .append("svg:path")
                    .attr("class", "marker")
                    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
                    .style("fill", marker_color)
                    .style("stroke", marker_color)
                    .attr("d", marker);

      if (JSROOT.gStyle.Tooltip) {
         marks.append("svg:title").text(function(d) { return d.tip; });
         xerr.append("svg:title").text(function(d) { return d.tip; });
         yerr.append("svg:title").text(function(d) { return d.tip; });
      }
   }

   JSROOT.TH1Painter.prototype.DrawBins = function() {

      var width = Number(this.svg_frame(true).attr("width")),
          height = Number(this.svg_frame(true).attr("height"));

      if (!this.draw_content || (width<=0) || (height<=0)) {
         this.RemoveDrawG();
         return;
      }

      var draw_bins = this.CreateDrawBins(width, height);

      this.RecreateDrawG();

      if (this.options.Error > 0)
         return this.DrawErrors(draw_bins);

      var pthis = this;

      if (this.fillcolor!='none') {

         // histogram filling
         var area = d3.svg.area()
                    .x(function(d) { return d.x; })
                    .y0(function(d) { return d.y; })
                    .y1(function(d) { return height; })
                    .interpolate("step-after");

         this.draw_g.append("svg:path")
                    .attr("d", area(draw_bins))
                    .style("stroke", this.linecolor)
                    .style("stroke-width", this.histo['fLineWidth'])
                    .style("fill", this.fillcolor)
                    .style("antialias", "false");
      } else {

         var line = d3.svg.line()
                          .x(function(d) { return d.x; })
                          .y(function(d) { return d.y; })
                          .interpolate("step-after");

         this.draw_g
               .append("svg:path")
               .attr("d", line(draw_bins))
               // to draw one bar, one need two points
               .style("stroke", this.linecolor)
               .style("stroke-width", this.histo['fLineWidth'])
               .style("fill", "none")
               .style("stroke-dasharray", this.histo['fLineStyle'] > 1 ? JSROOT.Painter.root_line_styles[this.histo['fLineStyle']] : null)
               .style("antialias", "false");
      }

      if (JSROOT.gStyle.Tooltip) {
         // TODO: limit number of tooltips by number of visible pixels
         this.draw_g.selectAll("selections")
                    .data(draw_bins).enter()
                    .append("svg:line")
                    .attr("x1", function(d) { return d.x + d.width / 2; })
                    .attr("y1", function(d) { return Math.max(0, d.y); })
                    .attr("x2", function(d) { return d.x + d.width / 2; })
                    .attr("y2", function(d) { return height; })
                    .attr("opacity", 0)
                    .style("stroke", "#4572A7")
                    .style("stroke-width", function(d) { return d.width; })
                    .on('mouseover', function() {
                        if (JSROOT.gStyle.Tooltip && (d3.select(this).style("opacity")=="0"))
                          d3.select(this).transition().duration(100).style("opacity", "0.3");
                     })
                     .on('mouseout', function() {
                        d3.select(this).transition().duration(100).style("opacity", "0");
                     })
                     .append("svg:title").text(function(d) { return d.tip; });
      }
   }

   JSROOT.TH1Painter.prototype.FillContextMenu = function(menu) {
      JSROOT.THistPainter.prototype.FillContextMenu.call(this, menu);
      if (this.draw_content)
         JSROOT.Painter.menuitem(menu, "Auto zoom-in", function() { menu['painter'].AutoZoom(); });
   }

   JSROOT.TH1Painter.prototype.AutoZoom = function() {
      var left = this.GetSelectIndex("x", "left", -1);
      var right = this.GetSelectIndex("x", "right", 1);

      var dist = (right - left);

      if (dist == 0)
         return;

      var min = this.histo.getBinContent(left + 1);

      // first find minimum
      for (var indx = left; indx < right; indx++)
         if (this.histo.getBinContent(indx + 1) < min)
            min = this.histo.getBinContent(indx + 1);

      while ((left < right) && (this.histo.getBinContent(left + 1) <= min)) left++;
      while ((left < right) && (this.histo.getBinContent(right) <= min)) right--;

      if ((right - left < dist) && (left < right))
         this.Zoom(this.xmin + left * this.binwidthx, this.xmin + right * this.binwidthx, 0, 0);
   }

   JSROOT.Painter.drawHistogram1D = function(divid, histo, opt) {

      // create painter and add it to canvas
      var painter = new JSROOT.TH1Painter(histo);
      painter.SetDivId(divid, 1);

      // here we deciding how histogram will look like and how will be shown
      painter.options = painter.DecodeOptions(opt);

      painter.CheckPadOptions();

      painter.ScanContent();

      painter.CreateXY();

      painter.DrawAxes();

      painter.DrawGrids();

      painter.DrawBins();

      if (painter.create_canvas) painter.DrawTitle();

      if (JSROOT.gStyle.AutoStat && painter.create_canvas) painter.CreateStat();

      painter.DrawFunctions();

      painter.AddInteractive();

      return painter;
   }

   // ==================== painter for TH2 histograms ==============================

   JSROOT.TH2Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);
      this.paletteColors = [];
   }

   JSROOT.TH2Painter.prototype = Object.create(JSROOT.THistPainter.prototype);

   JSROOT.TH2Painter.prototype.FillContextMenu = function(menu) {
      JSROOT.THistPainter.prototype.FillContextMenu.call(this, menu);
      JSROOT.Painter.menuitem(menu, "Auto zoom-in", function() { menu['painter'].AutoZoom(); });
      JSROOT.Painter.menuitem(menu, "Draw in 3D", function() { menu['painter'].Draw3D(); });
      JSROOT.Painter.menuitem(menu, "Toggle col", function() {
         menu['painter'].options.Color = 1 - menu['painter'].options.Color;
         menu['painter'].RedrawPad();
      });

      if (this.options.Color > 0)
         JSROOT.Painter.menuitem(menu, "Toggle colz", function() { menu['painter'].ToggleColz(); });
   }

   JSROOT.TH2Painter.prototype.FindPalette = function(remove) {
      if ('fFunctions' in this.histo)
         for ( var i in this.histo.fFunctions.arr) {
            var func = this.histo.fFunctions.arr[i];
            if (func['_typename'] != 'TPaletteAxis')
               continue;
            if (remove) {
               this.histo.fFunctions.arr.splice(i, 1);
               return null;
            }

            return func;
         }

      return null;
   }

   JSROOT.TH2Painter.prototype.ToggleColz = function() {
      if (this.FindPalette() == null) {
         var shrink = this.CreatePalette(0.04);
         this.svg_frame()['frame_painter'].Shrink(0, shrink);
         this.options.Zscale = 1;
         // one should draw palette
         JSROOT.Painter.drawPaletteAxis(this.divid, this.FindPalette());
      } else {
         if (this.options.Zscale > 0)
            this.options.Zscale = 0;
         else
            this.options.Zscale = 1;
      }

      this.RedrawPad();
   }

   JSROOT.TH2Painter.prototype.AutoZoom = function() {
      var i1 = this.GetSelectIndex("x", "left", -1);
      var i2 = this.GetSelectIndex("x", "right", 1);
      var j1 = this.GetSelectIndex("y", "left", -1);
      var j2 = this.GetSelectIndex("y", "right", 1);

      if ((i1 == i2) || (j1 == j2)) return;

      var min = this.histo.getBinContent(i1 + 1, j1 + 1);

      // first find minimum
      for (var i = i1; i < i2; i++)
         for (var j = j1; j < j2; j++)
            if (this.histo.getBinContent(i + 1, j + 1) < min)
               min = this.histo.getBinContent(i + 1, j + 1);

      var ileft = i2, iright = i1, jleft = j2, jright = j1;

      for (var i = i1; i < i2; i++)
         for (var j = j1; j < j2; j++)
            if (this.histo.getBinContent(i + 1, j + 1) > min) {
               if (i < ileft) ileft = i;
               if (i >= iright) iright = i + 1;
               if (j < jleft) jleft = j;
               if (j >= jright) jright = j + 1;
            }

      var xmin = 0, xmax = 0, ymin = 0, ymax = 0;

      if ((ileft > i1 || iright < i2) && (ileft < iright - 1)) {
         xmin = this.xmin + ileft * this.binwidthx;
         xmax = this.xmin + iright * this.binwidthx;
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = this.ymin + jleft * this.binwidthy;
         ymax = this.ymin + jright * this.binwidthy;
      }

      this.Zoom(xmin, xmax, ymin, ymax);
   }

   JSROOT.TH2Painter.prototype.CreatePalette = function(rel_width) {
      if (this.FindPalette() != null) return 0.;

      if (!rel_width || rel_width <= 0) rel_width = 0.04;

      var pal = {};
      pal['_typename'] = 'TPaletteAxis';
      pal['fName'] = 'palette';

      pal['_AutoCreated'] = true;

      pal['fX1NDC'] = this.svg_frame()['NDC'].x2 - rel_width;
      pal['fY1NDC'] = this.svg_frame()['NDC'].y1;
      pal['fX2NDC'] = this.svg_frame()['NDC'].x2;
      pal['fY2NDC'] = this.svg_frame()['NDC'].y2;
      pal['fInit'] = 1;
      pal['fShadowColor'] = 1;
      pal['fCorenerRadius'] = 0;
      pal['fResizing'] = false;
      pal['fBorderSize'] = 4;
      pal['fName'] = "TPave";
      pal['fOption'] = "br";
      pal['fLineColor'] = 1;
      pal['fLineSyle'] = 1;
      pal['fLineWidth'] = 1;
      pal['fFillColor'] = 1;
      pal['fFillSyle'] = 1;

      var axis = {};

      axis['_typename'] = 'TGaxis';
      axis['fTickSize'] = 0.03;
      axis['fLabelOffset'] = 0.005;
      axis['fLabelSize'] = 0.035;
      axis['fTitleOffset'] = 1;
      axis['fTitleSize'] = 0.035;
      axis['fNdiv'] = 8;
      axis['fLabelColor'] = 1;
      axis['fLabelFont'] = 42;
      axis['fChopt'] = "";
      axis['fName'] = "";
      axis['fTitle'] = "";
      axis['fTimeFormat'] = "";
      axis['fFunctionName'] = "";
      axis['fWmin'] = 0;
      axis['fWmax'] = 100;
      axis['fLineColor'] = 1;
      axis['fLineSyle'] = 1;
      axis['fLineWidth'] = 1;
      axis['fTextAngle'] = 0;
      axis['fTextSize'] = 0.04;
      axis['fTextAlign'] = 11;
      axis['fTextColor'] = 1;
      axis['fTextFont'] = 42;

      pal['fAxis'] = axis;

      if (!'fFunctions' in this.histo)
         this.histo['fFunctions'] = JSROOT.Create("TList");

      // place colz in the beginning, that stat box is always drawn on the top
      this.histo.fFunctions.arr.unshift(pal);

      // and at the end try to check how much place will be used by the labels
      // in the palette

      var width = Number(this.svg_frame(true).attr("width")),
          height = Number(this.svg_frame(true).attr("height"));

      var axisOffset = axis['fLabelOffset'] * width;
      var tickSize = axis['fTickSize'] * width;
      var axisFontDetails = JSROOT.Painter.getFontDetails(axis['fLabelFont']);
      var axisLabelFontSize = axis['fLabelSize'] * height;

      var ticks = d3.scale.linear().clamp(true)
                  .domain([ this.minbin, this.maxbin ])
                  .range([ height, 0 ]).nice().ticks(axis['fNdiv'] % 100);

      var maxlen = 0;
      for (var i in ticks) {
         var len = JSROOT.Painter.stringWidth(this.svg_frame(true), ticks[i], axisLabelFontSize, axisFontDetails);
         if (len > maxlen) maxlen = len;
      }

      var rel = (maxlen + axisOffset) / width;

      if (pal['fX2NDC'] + rel > 0.98) {
         var shift = pal['fX2NDC'] + rel - 0.98;

         pal['fX1NDC'] -= shift;
         pal['fX2NDC'] -= shift;
         rel_width += shift;
      }

      return rel_width + 0.01;
   }

   JSROOT.TH2Painter.prototype.ScanContent = function() {
      this.fillcolor = JSROOT.Painter.root_colors[this.histo['fFillColor']];
      this.linecolor = JSROOT.Painter.root_colors[this.histo['fLineColor']];

      // if (this.histo['fFillColor'] == 0) this.fillcolor = '#4572A7'; // why?
      if (this.histo['fLineColor'] == 0)
         this.linecolor = '#4572A7';

      this.nbinsx = this.histo['fXaxis']['fNbins'];
      this.nbinsy = this.histo['fYaxis']['fNbins'];

      // used in CreateXY method
      this.xmin = this.histo['fXaxis']['fXmin'];
      this.xmax = this.histo['fXaxis']['fXmax'];
      this.ymin = this.histo['fYaxis']['fXmin'];
      this.ymax = this.histo['fYaxis']['fXmax'];

      this.binwidthx = (this.xmax - this.xmin);
      if (this.nbinsx > 0)
         this.binwidthx = this.binwidthx / this.nbinsx;

      this.binwidthy = (this.ymax - this.ymin);
      if (this.nbinsy > 0)
         this.binwidthy = this.binwidthy / this.nbinsy

      this.gmaxbin = this.histo.getBinContent(1, 1);
      this.gminbin = this.gmaxbin; // global min/max, used at the moment in 3D drawing
      for (var i = 0; i < this.nbinsx; ++i) {
         for (var j = 0; j < this.nbinsy; ++j) {
            var bin_content = this.histo.getBinContent(i + 1, j + 1);
            if (bin_content < this.gminbin) this.gminbin = bin_content; else
            if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;
         }
      }

      // used to enable/disable stat box
      this.draw_content = this.gmaxbin > 0;
   }

   JSROOT.TH2Painter.prototype.CountStat = function(cond) {
      var stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0, stat_sumx2 = 0, stat_sumy2 = 0, stat_sumxy2 = 0;

      var res = { entries: 0, integral: 0, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix : [], xmax: 0, ymax:0, wmax: null };
      for (var n = 0; n < 9; n++) res.matrix.push(0);

      var xleft = this.GetSelectIndex("x", "left");
      var xright = this.GetSelectIndex("x", "right");

      var yleft = this.GetSelectIndex("y", "left");
      var yright = this.GetSelectIndex("y", "right");

      for (var xi = 0; xi <= this.nbinsx + 1; xi++) {
         var xside = (xi <= xleft) ? 0 : (xi > xright ? 2 : 1);
         var xx = this.xmin + (xi - 0.5) * this.binwidthx;

         for (var yi = 0; yi <= this.nbinsx + 1; yi++) {
            var yside = (yi <= yleft) ? 0 : (yi > yright ? 2 : 1);
            var yy = this.ymin + (yi - 0.5) * this.binwidthy;

            var zz = this.histo.getBinContent(xi, yi);

            res.entries += zz;

            res.matrix[yside * 3 + xside] += zz;

            if ((xside != 1) || (yside != 1)) continue;

            if ((cond!=null) && !cond(xx,yy)) continue;

            if ((res.wmax==null) || (zz>res.wmax)) { res.wmax = zz; res.xmax = xx; res.ymax = yy; }

            stat_sum0 += zz;
            stat_sumx1 += xx * zz;
            stat_sumy1 += yy * zz;
            stat_sumx2 += xx * xx * zz;
            stat_sumy2 += yy * yy * zz;
            stat_sumxy2 += xx * yy * zz;
         }
      }

      if (stat_sum0 > 0) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.rmsx = Math.sqrt(stat_sumx2 / stat_sum0 - res.meanx * res.meanx);
         res.rmsy = Math.sqrt(stat_sumy2 / stat_sum0 - res.meany * res.meany);
      }

      if (res.wmax==null) res.wmax = 0;
      res.integral = stat_sum0;

      return res;
   }

   JSROOT.TH2Painter.prototype.FillStatistic = function(stat, dostat) {
      if (!this.histo) return false;

      var data = this.CountStat();

      var print_name = Math.floor(dostat % 10);
      var print_entries = Math.floor(dostat / 10) % 10;
      var print_mean = Math.floor(dostat / 100) % 10;
      var print_rms = Math.floor(dostat / 1000) % 10;
      var print_under = Math.floor(dostat / 10000) % 10;
      var print_over = Math.floor(dostat / 100000) % 10;
      var print_integral = Math.floor(dostat / 1000000) % 10;
      var print_skew = Math.floor(dostat / 10000000) % 10;
      var print_kurt = Math.floor(dostat / 100000000) % 10;

      if (print_name > 0)
         stat.AddLine(this.histo['fName']);

      if (print_entries > 0)
         stat.AddLine("Entries = " + JSROOT.gStyle.StatEntriesFormat(data.entries));

      if (print_mean > 0) {
         stat.AddLine("Mean x = " + JSROOT.gStyle.StatFormat(data.meanx));
         stat.AddLine("Mean y = " + JSROOT.gStyle.StatFormat(data.meany));
      }

      if (print_rms > 0) {
         stat.AddLine("RMS x = " + JSROOT.gStyle.StatFormat(data.rmsx));
         stat.AddLine("RMS y = " + JSROOT.gStyle.StatFormat(data.rmsy));
      }

      if (print_integral > 0) {
         stat.AddLine("Integral = " + JSROOT.gStyle.StatEntriesFormat(data.matrix[4]));
      }

      if (print_skew > 0) {
         stat.AddLine("Skewness x = <undef>");
         stat.AddLine("Skewness y = <undef>");
      }

      if (print_kurt > 0)
         stat.AddLine("Kurt = <undef>");

      if ((print_under > 0) || (print_over > 0)) {
         var m = data.matrix;

         stat.AddLine("" + m[6].toFixed(0) + " | " + m[7].toFixed(0) + " | "  + m[7].toFixed(0));
         stat.AddLine("" + m[3].toFixed(0) + " | " + m[4].toFixed(0) + " | "  + m[5].toFixed(0));
         stat.AddLine("" + m[0].toFixed(0) + " | " + m[1].toFixed(0) + " | "  + m[2].toFixed(0));
      }

      // adjust the size of the stats box wrt the number of lines
      var nlines = stat.pavetext['fLines'].arr.length;
      var stath = nlines * JSROOT.gStyle.StatFontSize;
      if (stath <= 0 || 3 == (JSROOT.gStyle.StatFont % 10)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         stat.pavetext['fY1NDC'] = 0.93 - stath;
         stat.pavetext['fY2NDC'] = 0.93;
      }
      return true;
   }

   JSROOT.TH2Painter.prototype.getValueColor = function(zc) {
      var wmin = this.minbin, wmax = this.maxbin;
      var wlmin = wmin, wlmax = wmax;
      var ndivz = this.histo['fContour'].length;
      if (ndivz < 16)
         ndivz = 16;
      var scale = ndivz / (wlmax - wlmin);
      if (this.options.Logz) {
         if (wmin <= 0 && wmax > 0)
            wmin = Math.min(1.0, 0.001 * wmax);
         wlmin = Math.log(wmin) / Math.log(10);
         wlmax = Math.log(wmax) / Math.log(10);
      }

      if (this.paletteColors.length == 0) {
         var saturation = 1, lightness = 0.5, maxHue = 280, minHue = 0, maxPretty = 50;
         for (var i = 0; i < maxPretty; i++) {
            var hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty)) / 360.0;
            var rgbval = JSROOT.Painter.HLStoRGB(hue, lightness, saturation);
            this.paletteColors.push(rgbval);
         }
      }
      if (this.options.Logz) zc = Math.log(zc) / Math.log(10);
      if (zc < wlmin) zc = wlmin;
      var ncolors = this.paletteColors.length;
      var color = Math.round(0.01 + (zc - wlmin) * scale);
      var theColor = Math.round((color + 0.99) * ncolors / ndivz) - 1;
      var icol = theColor % ncolors;
      if (icol < 0) icol = 0;

      return this.paletteColors[icol];
   }

   JSROOT.TH2Painter.prototype.CreateDrawBins = function(w, h, coordinates_kind, tipkind) {
      var i1 = this.GetSelectIndex("x", "left", 0);
      var i2 = this.GetSelectIndex("x", "right", 0);
      var j1 = this.GetSelectIndex("y", "left", 0);
      var j2 = this.GetSelectIndex("y", "right", 0);

      var x1, y1, x2, y2, grx1, gry1, grx2, gry2, fillcol, shrx, shry, binz, point, wx ,wy;

      // first found min/max values in selected range
      this.maxbin = this.minbin = this.histo.getBinContent(i1 + 1, j1 + 1);
      for (var i = i1; i < i2; i++) {
         for (var j = j1; j < j2; j++) {
            binz = this.histo.getBinContent(i + 1, j + 1);
            if (binz>this.maxbin) this.maxbin = binz; else
            if (binz<this.minbin) this.minbin = binz;
         }
      }

      var xfactor = 1, yfactor = 1;
      if (coordinates_kind == 1) {
         xfactor = 0.5 * w / (i2 - i1) / (this.maxbin - this.minbin);
         yfactor = 0.5 * h / (j2 - j1) / (this.maxbin - this.minbin);
      }

      var local_bins = new Array;

      x2 = this.xmin + i1 * this.binwidthx;
      grx2 = -11111;
      for (var i = i1; i < i2; i++) {
         x1 = x2;
         x2 += this.binwidthx;

         if (this.options.Logx && (x1 <= 0)) continue;

         grx1 = grx2;
         if (grx1 < 0) grx1 = this.x(x1);
         grx2 = this.x(x2);

         y2 = this.ymin + j1 * this.binwidthy;
         gry2 = -1111;
         for (var j = j1; j < j2; j++) {
            y1 = y2;
            y2 += this.binwidthy;
            if (this.options.Logy && (y1 <= 0)) continue;
            gry1 = gry2;
            if (gry1 < 0) gry1 = this.y(y1);
            gry2 = this.y(y2);
            binz = this.histo.getBinContent(i + 1, j + 1);
            if (binz <= this.minbin) continue;

            switch (coordinates_kind) {
            case 0:
               point = {
                  x : grx1,
                  y : gry2,
                  width : grx2 - grx1 + 1,  // +1 to fill gaps between colored bins
                  height : gry1 - gry2 + 1,
                  stroke : "none",
                  fill : this.getValueColor(binz)
               }
               point['tipcolor'] = (point['fill'] == "black") ? "grey" : "black";
               break;

            case 1:
               shrx = xfactor * (this.maxbin - binz);
               shry = yfactor * (this.maxbin - binz);
               point = {
                  x : grx1 + shrx,
                  y : gry2 + shry,
                  width : grx2 - grx1 - 2 * shrx,
                  height : gry1 - gry2 - 2 * shry,
                  stroke : this.linecolor,
                  fill : this.fillcolor
               }
               point['tipcolor'] = (point['fill'] == "black") ? "grey" : "black";
               break;

            case 2:
               point = {
                  x : (x1 + x2) / 2,
                  y : (y1 + y2) / 2,
                  z : binz
               }
               break;
            }

            if (tipkind == 1)
               point['tip'] = "x = [" + this.AxisAsText("x", x1) + ", " + this.AxisAsText("x", x2) + "]\n" +
                              "y = [" + this.AxisAsText("y", y1) + ", " + this.AxisAsText("y", y2) + "]\n" +
                              "entries = " + binz;
            else if (tipkind == 2)
               point['tip'] = "x = " + this.AxisAsText("x", x1) + "\n" +
                              "y = " + this.AxisAsText("y", y1) + "\n" +
                              "entries = " + binz;

            local_bins.push(point);
         }
      }

      return local_bins;
   }

   JSROOT.TH2Painter.prototype.DrawBins = function() {

      this.RecreateDrawG();

      var w = Number(this.svg_frame(true).attr("width")),
          h = Number(this.svg_frame(true).attr("height"));

      // this.options.Scat =1;
      // this.histo['fMarkerStyle'] = 2;

      var draw_markers = (this.options.Scat > 0 && this.histo['fMarkerStyle'] > 1);
      var normal_coordinates = (this.options.Color > 0) || draw_markers;

      var tipkind = 0;
      if (JSROOT.gStyle.Tooltip) tipkind = draw_markers ? 2 : 1;

      var local_bins = this.CreateDrawBins(w, h, normal_coordinates ? 0 : 1, tipkind);

      if (draw_markers) {

         // Add markers
         var info_marker = JSROOT.Painter.getRootMarker(this.histo['fMarkerStyle']);
         var markerSize = this.histo['fMarkerSize'];
         var markerScale = (info_marker.shape == 0) ? 32 : 64;
         if (this.histo['fMarkerStyle'] == 1) markerScale = 1;

         var marker = null;

         switch (info_marker.shape) {
         case 6:
            marker = "M " + (-4 * markerSize) + " " + (-1 * markerSize) + " L "
                  + 4 * markerSize + " " + (-1 * markerSize) + " L "
                  + (-2.4 * markerSize) + " " + 4 * markerSize + " L 0 "
                  + (-4 * markerSize) + " L " + 2.8 * markerSize + " " + 4
                  * markerSize + " z";
            break;
         case 7:
            marker = "M " + (-4 * markerSize) + " " + (-4 * markerSize) + " L "
                  + 4 * markerSize + " " + 4 * markerSize + " M 0 "
                  + (-4 * markerSize) + " 0 " + 4 * markerSize + " M "
                  + 4 * markerSize + " " + (-4 * markerSize) + " L "
                  + (-4 * markerSize) + " " + 4 * markerSize + " M "
                  + (-4 * markerSize) + " 0 L " + 4 * markerSize + " 0";
            break;
         default:
            marker = d3.svg.symbol().type(d3.svg.symbolTypes[info_marker.shape]).size(markerSize * markerScale);
            break;
         }
         var markers =
            this.draw_g.selectAll(".marker")
                  .data(local_bins)
                  .enter().append("svg:path")
                  .attr("class", "marker")
                  .attr("transform", function(d) { return "translate(" + d.x.toFixed(1) + "," + d.y.toFixed(1) + ")" })
                  .style("fill", JSROOT.Painter.root_colors[this.histo['fMarkerColor']])
                  .style("stroke", JSROOT.Painter.root_colors[this.histo['fMarkerColor']])
                  .attr("d", marker);

         if (JSROOT.gStyle.Tooltip)
            markers.append("svg:title").text(function(d) { return d.tip; });
      } else {
         var drawn_bins = this.draw_g.selectAll(".bins")
                           .data(local_bins).enter()
                           .append("svg:rect")
                           .attr("class", "bins")
                           .attr("x", function(d) { return d.x.toFixed(1); })
                           .attr("y", function(d) { return d.y.toFixed(1); })
                           .attr("width", function(d) { return d.width.toFixed(1); })
                           .attr("height", function(d) { return d.height.toFixed(1); })
                           .style("stroke", function(d) { return d.stroke; })
                           .style("fill", function(d) {
                               this['f0'] = d.fill;
                               this['f1'] = d.tipcolor;
                               return d.fill;
                            });

         if (JSROOT.gStyle.Tooltip)
            drawn_bins
              .on('mouseover', function() {
                   if (JSROOT.gStyle.Tooltip)
                      d3.select(this).transition().duration(100).style("fill", this['f1']);
              })
              .on( 'mouseout', function() {
                   d3.select(this).transition().duration(100).style("fill", this['f0']);
              })
              .append("svg:title").text(function(d) { return d.tip; });
      }

      delete local_bins;
   }

   JSROOT.TH2Painter.prototype.Draw2D = function() {

      if (this.options.Lego>0) this.options.Lego = 0;

      if (this['done2d']) return;

      // check if we need to create palette
      if ((this.FindPalette() == null) && this.create_canvas && (this.options.Zscale > 0)) {
         // create pallette
         var shrink = this.CreatePalette(0.04);
         this.svg_frame()['frame_painter'].Shrink(0, shrink);
         this.svg_frame()['frame_painter'].Redraw();
         this.CreateXY();
      } else if (this.options.Zscale == 0) {
         // delete palette - it may appear there due to previous draw options
         this.FindPalette(true);
      }

      // check if we need to create statbox
      if (JSROOT.gStyle.AutoStat && this.create_canvas)
         this.CreateStat();

      this.DrawAxes();

      this.DrawGrids();

      this.DrawBins();

      if (this.create_canvas) this.DrawTitle();

      this.DrawFunctions();

      this.AddInteractive();

      this['done2d'] = true; // indicate that 2d drawing was once done
   }

   JSROOT.TH2Painter.prototype.Draw3D = function() {

      if (this.options.Lego<=0) this.options.Lego = 1;
      var painter = this;

      JSROOT.AssertPrerequisites('3d', function() {
         JSROOT.Painter.real_drawHistogram2D(painter);
      });
   }

   JSROOT.Painter.drawHistogram2D = function(divid, histo, opt) {

      // create painter and add it to canvas
      var painter = new JSROOT.TH2Painter(histo);

      painter.SetDivId(divid, 1);

      // here we deciding how histogram will look like and how will be shown
      painter.options = painter.DecodeOptions(opt);

      painter.CheckPadOptions();

      painter.ScanContent();

      painter.CreateXY();

      if (painter.options.Lego > 0)
         painter.Draw3D();
      else
         painter.Draw2D();

      return painter;
   }

   JSROOT.Painter.drawHistogram3D = function(divid, obj, opt) {
      JSROOT.AssertPrerequisites('3d', function() {
         JSROOT.Painter.real_drawHistogram3D(divid, obj, opt);
      });
   }

   // ====================================================================

   JSROOT.THStackPainter = function(stack) {
      JSROOT.TObjectPainter.call(this, stack);
      this.stack = stack;
      this.nostack = false;
      this.firstpainter = null;
      this.painters = new Array; // keep painters to be able update objects
   }

   JSROOT.THStackPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.THStackPainter.prototype.GetObject = function() {
      return this.stack;
   }

   JSROOT.THStackPainter.prototype.drawStack = function(opt) {

      var pad = this.root_pad();
      var histos = this.stack['fHists'];
      var nhists = histos.arr.length;

      if (opt == null) opt = "";
                  else opt = opt.toLowerCase();
      var lsame = false;
      if (opt.indexOf("same") != -1) {
         lsame = true;
         opt.replace("same", "");
      }
      // compute the min/max of each axis
      var i, h;
      var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
      for (var i = 0; i < nhists; ++i) {
         h = histos.arr[i];
         if (i == 0 || h['fXaxis']['fXmin'] < xmin)
            xmin = h['fXaxis']['fXmin'];
         if (i == 0 || h['fXaxis']['fXmax'] > xmax)
            xmax = h['fXaxis']['fXmax'];
         if (i == 0 || h['fYaxis']['fXmin'] < ymin)
            ymin = h['fYaxis']['fXmin'];
         if (i == 0 || h['fYaxis']['fXmax'] > ymax)
            ymax = h['fYaxis']['fXmax'];
      }
      this.nostack = opt.indexOf("nostack") == -1 ? false : true;
      if (!this.nostack)
         this.stack.buildStack();

      var themin, themax;
      if (this.stack['fMaximum'] == -1111) themax = this.stack.getMaximum(opt);
                                      else themax = this.stack['fMaximum'];
      if (this.stack['fMinimum'] == -1111) {
         themin = this.stack.getMinimum(opt);
         if (pad && pad['fLogy']) {
            if (themin > 0)
               themin *= .9;
            else
               themin = themax * 1.e-3;
         } else if (themin > 0)
            themin = 0;
      } else
         themin = this.stack['fMinimum'];
      if (!('fHistogram' in this.stack)) {
         h = this.stack['fHists'].arr[0];
         this.stack['fHistogram'] = JSROOT.Create("TH1I");
         this.stack['fHistogram']['fName'] = "unnamed";
         this.stack['fHistogram']['fXaxis'] = JSROOT.clone(h['fXaxis']);
         this.stack['fHistogram']['fYaxis'] = JSROOT.clone(h['fYaxis']);
         this.stack['fHistogram']['fXaxis']['fXmin'] = xmin;
         this.stack['fHistogram']['fXaxis']['fXmax'] = xmax;
         this.stack['fHistogram']['fYaxis']['fXmin'] = ymin;
         this.stack['fHistogram']['fYaxis']['fXmax'] = ymax;
      }
      this.stack['fHistogram']['fTitle'] = this.stack['fTitle'];
      // var histo = JSROOT.clone(stack['fHistogram']);
      var histo = this.stack['fHistogram'];
      if (!histo.TestBit(JSROOT.TH1StatusBits.kIsZoomed)) {
         if (this.nostack && this.stack['fMaximum'] != -1111)
            histo['fMaximum'] = this.stack['fMaximum'];
         else {
            if (pad && pad['fLogy'])
               histo['fMaximum'] = themax * (1 + 0.2 * JSROOT.Math.log10(themax / themin));
            else
               histo['fMaximum'] = 1.05 * themax;
         }
         if (this.nostack && this.stack['fMinimum'] != -1111)
            histo['fMinimum'] = this.stack['fMinimum'];
         else {
            if (pad && pad['fLogy'])
               histo['fMinimum'] = themin / (1 + 0.5 * JSROOT.Math.log10(themax / themin));
            else
               histo['fMinimum'] = themin;
         }
      }
      if (!lsame) {

         var hopt = histo['fOption'];
         if ((opt != "") && (hopt.indexOf(opt) == -1))
            hopt += opt;

         if (histo['_typename'].match(/^TH1/))
            this.firstpainter = JSROOT.Painter.drawHistogram1D(this.divid, histo, hopt);
         else
         if (histo['_typename'].match(/^TH2/))
            this.firstpainter = JSROOT.Painter.drawHistogram2D(this.divid, histo, hopt);

      }
      for (var i = 0; i < nhists; ++i) {
         if (this.nostack)
            h = histos.arr[i];
         else
            h = this.stack['fStack'].arr[nhists - i - 1];

         var hopt = h['fOption'];
         if ((opt != "") && (hopt.indexOf(opt) == -1)) hopt += opt;
         hopt += "same";

         if (h['_typename'].match(/^TH1/)) {
            var subpainter = JSROOT.Painter.drawHistogram1D(this.divid, h, hopt);
            this.painters.push(subpainter);
         }
      }
   }

   JSROOT.THStackPainter.prototype.UpdateObject = function(obj) {
      var isany = false;
      if (this.firstpainter)
         if (this.firstpainter.UpdateObject(obj['fHistogram'])) isany = true;

      var histos = obj['fHists'];
      var nhists = histos.arr.length;

      for (var i = 0; i < nhists; ++i) {
         var h = null;

         if (this.nostack)
            h = histos.arr[i];
         else
            h = obj['fStack'].arr[nhists - i - 1];

         if (this.painters[i].UpdateObject(h)) isany = true;
      }

      return isany;
   }


   JSROOT.Painter.drawHStack = function(divid, stack, opt) {
      // paint the list of histograms
      // By default, histograms are shown stacked.
      // -the first histogram is paint
      // -then the sum of the first and second, etc

      if (!'fHists' in stack) return;
      if (stack['fHists'].arr.length == 0) return;

      var painter = new JSROOT.THStackPainter(stack);
      painter.SetDivId(divid);

      painter.drawStack(opt);

      return painter
   }

   // ==============================================================================

   JSROOT.TLegendPainter = function(legend) {
      JSROOT.TObjectPainter.call(this, legend);
      this.legend = legend;
   }

   JSROOT.TLegendPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TLegendPainter.prototype.GetObject = function() {
      return this.legend;
   }

   JSROOT.TLegendPainter.prototype.drawLegend = function() {
      this.RecreateDrawG(true);

      var svg = this.svg_pad(true);
      var pave = this.legend;

      var x = 0, y = 0, w = 0, h = 0;
      if (pave['fInit'] == 0) {
         x = pave['fX1'] * Number(svg.attr("width"));
         y = Number(svg.attr("height")) - pave['fY1']
               * Number(svg.attr("height"));
         w = (pave['fX2'] - pave['fX1']) * Number(svg.attr("width"));
         h = (pave['fY2'] - pave['fY1']) * Number(svg.attr("height"));
      } else {
         x = pave['fX1NDC'] * Number(svg.attr("width"));
         y = Number(svg.attr("height")) - pave['fY1NDC']
               * Number(svg.attr("height"));
         w = (pave['fX2NDC'] - pave['fX1NDC']) * Number(svg.attr("width"));
         h = (pave['fY2NDC'] - pave['fY1NDC']) * Number(svg.attr("height"));
      }
      y -= h;
      var fillcolor = JSROOT.Painter.root_colors[pave['fFillColor']];
      var lcolor = JSROOT.Painter.root_colors[pave['fLineColor']];
      var lwidth = pave['fBorderSize'] ? pave['fBorderSize'] : 0;
      if (pave['fFillStyle'] > 4000 && pave['fFillStyle'] < 4100)
         fillcolor = 'none';

      var p = this.draw_g
                 .attr("width", w)
                 .attr("height", h)
                 .attr("transform", "translate(" + x + "," + y + ")");

      p.append("svg:rect")
           .attr("x", 0)
           .attr("y", 0)
           .attr("width", w)
           .attr("height", h)
           .attr("fill", fillcolor)
           .style("stroke-width", lwidth ? 1 : 0)
           .style("stroke", lcolor);

      var tcolor = JSROOT.Painter.root_colors[pave['fTextColor']];
      var tpos_x = pave['fMargin'] * w;
      var nlines = pave.fPrimitives.arr.length;
      var font_size = Math.round(h / (nlines * 1.5));
      // var font_size = Math.round(pave['fTextSize'] * svg.height());
      var fontDetails = JSROOT.Painter.getFontDetails(pave['fTextFont']);

      var max_len = 0, mul = 1.4;
      for (var j = 0; j < nlines; ++j) {
         var line = JSROOT.Painter.translateLaTeX(pave.fPrimitives.arr[j]['fLabel']);
         var lw = tpos_x  + JSROOT.Painter.stringWidth(svg, line, font_size, fontDetails);
         if (lw > max_len) max_len = lw;
      }
      if (max_len > w) {
         font_size = Math.floor(font_size * 0.95 * (w / max_len));
         mul *= 0.95 * (max_len / w);
      }
      var x1 = pave['fX1NDC'];
      var x2 = pave['fX2NDC'];
      var y1 = pave['fY1NDC'];
      var y2 = pave['fY2NDC'];
      var margin = pave['fMargin'] * (x2 - x1) / pave['fNColumns'];
      var yspace = (y2 - y1) / nlines;
      var ytext = y2 + 0.5 * yspace; // y-location of 0th entry
      var boxw = margin * 0.35;

      for (var i = 0; i < nlines; ++i) {
         var leg = pave.fPrimitives.arr[i];
         var lopt = leg['fOption'].toLowerCase();

         var string = leg['fLabel'];

         var pos_y = ((i + 1) * (font_size * mul)) - (font_size / 3);
         var tpos_y = (i + 1) * (font_size * mul);
         if (nlines == 1) {
            var pos_y = (h * 0.75) - (font_size / 3);
            var tpos_y = h * 0.75;
         }

         var line_color = JSROOT.Painter.root_colors[leg['fLineColor']];
         var line_width = leg['fLineWidth'];
         var line_style = JSROOT.Painter.root_line_styles[leg['fLineStyle']];

         var fill_color = leg['fFillColor'];
         var fill_style = leg['fFillStyle'];

         var marker_color = JSROOT.Painter.root_colors[leg['fMarkerColor']];
         var marker_size = leg['fMarkerSize'];
         var marker_style = leg['fMarkerStyle'];

         var mo = leg['fObject'];

         if ((mo != null) && (typeof mo == 'object')) {
            if ('fLineColor' in mo) {
               line_color = JSROOT.Painter.root_colors[mo['fLineColor']];
               line_width = mo['fLineWidth'];
               line_style = JSROOT.Painter.root_line_styles[mo['fLineStyle']];
            }
            if ('fFillColor' in mo) {
               fill_color = mo['fFillColor'];
               fill_style = mo['fFillStyle'];
            }
            if ('fMarkerColor' in mo) {
               marker_color = JSROOT.Painter.root_colors[mo['fMarkerColor']];
               marker_size = mo['fMarkerSize'];
               marker_style = mo['fMarkerStyle'];
            }
         }

         p.append("text")
              .attr("class", "text")
              .attr("text-anchor", "start")
              .attr("x", tpos_x)
              .attr("y", tpos_y)
              .attr("xml:space","preserve")
              .attr("font-weight", fontDetails['weight'])
              .attr("font-style", fontDetails['style'])
              .attr("font-family", fontDetails['name'])
              .attr("font-size", font_size)
              .attr("fill", tcolor).text(string);

         // Draw fill pattern (in a box)
         if (lopt.indexOf('f') != -1) {
            // box total height is yspace*0.7
            // define x,y as the center of the symbol for this entry
            var xsym = margin / 2;
            var ysym = ytext;
            var xf = new Array(4), yf = new Array(4);
            xf[0] = xsym - boxw;
            yf[0] = ysym - yspace * 0.35;
            xf[1] = xsym + boxw;
            yf[1] = yf[0];
            xf[2] = xf[1];
            yf[2] = ysym + yspace * 0.35;
            xf[3] = xf[0];
            yf[3] = yf[2];
            for (var j = 0; j < 4; j++) {
               xf[j] = xf[j] * Number(svg.attr("width"));
               yf[j] = yf[j] * Number(svg.attr("height"));
            }
            var ww = xf[1] - xf[0];
            var hh = yf[2] - yf[0];
            pos_y = pos_y - (hh / 2);
            var pos_x = (tpos_x / 2) - (ww / 2);

            var fill_color = JSROOT.Painter.createFillPattern(this.svg_canvas(true), fill_style, fill_color);

            p.append("svg:rect")
                   .attr("x", pos_x)
                   .attr("y", pos_y)
                   .attr("width", ww)
                   .attr("height", hh)
                   .style("fill", fill_color)
                   .style("stroke-width", line_width)
                   .style("stroke", line_color);
         }
         // Draw line
         if (lopt.indexOf('l') != -1) {

            // line total length (in x) is margin*0.8
            var line_length = (0.7 * pave['fMargin']) * w;
            var pos_x = (tpos_x - line_length) / 2;
            p.append("svg:line")
               .attr("x1", pos_x)
               .attr("y1", pos_y)
               .attr("x2", pos_x + line_length)
               .attr("y2", pos_y)
               .style("stroke", line_color)
               .style("stroke-width", line_width)
               .style("stroke-dasharray", line_style);
         }
         // Draw error only
         if (lopt.indexOf('e') != -1  && (lopt.indexOf('l') == -1 || lopt.indexOf('f') != -1)) {
         }
         // Draw Polymarker
         if (lopt.indexOf('p') != -1) {

            var line_length = (0.7 * pave['fMargin']) * w;
            var pos_x = tpos_x / 2;

            var info_marker = JSROOT.Painter.getRootMarker(marker_style);
            var markerScale = 65;
            if (marker_style == 1) markerScale = 1;

            var marker;

            switch (info_marker.shape) {
            case 6:
               marker = "M " + (-4 * marker_size) + " "
                     + (-1 * marker_size) + " L " + 4 * marker_size + " "
                     + (-1 * marker_size) + " L " + (-2.4 * marker_size) + " "
                     + 4 * marker_size + " L 0 " + (-4 * marker_size) + " L "
                     + 2.8 * marker_size + " " + 4 * marker_size + " z";
               break;
            case 7:
               marker = "M " + (-4 * marker_size) + " "
                     + (-4 * marker_size) + " L " + 4 * marker_size + " "
                     + 4 * marker_size + " M 0 " + (-4 * marker_size) + " 0 "
                     + 4 * marker_size + " M " + 4 * marker_size + " "
                     + (-4 * marker_size) + " L " + (-4 * marker_size) + " "
                     + 4 * marker_size + " M " + (-4 * marker_size) + " 0 L "
                     + 4 * marker_size + " 0";
               break;
            default:
               marker = d3.svg.symbol().type(d3.svg.symbolTypes[info_marker.shape]).size(marker_size * markerScale);
               break;
            }
            p.append("svg:path")
                .attr("transform", function(d) { return "translate(" + pos_x + "," + pos_y + ")"; })
                .style("fill", info_marker['toFill'] ? marker_color : "none")
                .style("stroke", marker_color).attr("d", marker);
         }
      }
      if (lwidth && lwidth > 1) {
         p.append("svg:line")
            .attr("x1", w + (lwidth / 2))
            .attr("y1", lwidth + 1)
            .attr("x2", w + (lwidth / 2))
            .attr("y2",  h + lwidth - 1)
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
         p.append("svg:line")
            .attr("x1", lwidth + 1)
            .attr("y1", h + (lwidth / 2))
            .attr("x2", w + lwidth - 1)
            .attr("y2", h + (lwidth / 2))
            .style("stroke", lcolor)
            .style("stroke-width", lwidth);
      }
   }

   JSROOT.TLegendPainter.prototype.Redraw = function() {
      this.drawLegend();
   }

   JSROOT.Painter.drawLegend = function(divid, obj, opt) {
      var painter = new JSROOT.TLegendPainter(obj);
      painter.SetDivId(divid);
      painter.Redraw();
      return painter;
   }

   // =============================================================

   JSROOT.TMultiGraphPainter = function(mgraph) {
      JSROOT.TObjectPainter.call(this, mgraph);
      this.mgraph = mgraph;
      this.firstpainter = null;
      this.painters = new Array; // keep painters to be able update objects
   }

   JSROOT.TMultiGraphPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TMultiGraphPainter.prototype.GetObject = function() {
      return this.mgraph;
   }

   JSROOT.TMultiGraphPainter.prototype.UpdateObject = function(obj) {

      if ((obj==null) || (obj['_typename'] != 'TMultiGraph')) return false;

      var histo = obj['fHistogram'];
      var graphs = obj['fGraphs'];

      var isany = false;
      if (this.firstpainter && histo)
         if (this.firstpainter.UpdateObject(histo)) isany = true;

      for (var i in graphs.arr) {
         if (i>=this.painters.length) break;
         if (this.painters[i].UpdateObject(graphs.arr[i])) isany = true;
      }

      return isany;
   }

   JSROOT.TMultiGraphPainter.prototype.drawMultiGraph = function(opt) {
      var maximum, minimum, rwxmin = 0, rwxmax = 0, rwymin = 0, rwymax = 0, uxmin = 0, uxmax = 0, dx, dy;
      var histo = this.mgraph['fHistogram'];
      var graphs = this.mgraph['fGraphs'];
      var scalex = 1, scaley = 1;
      var logx = false, logy = false, logz = false, gridx = false, gridy = false;
      var draw_all = true;

      var pad = this.root_pad();

      if (pad!=null) {
         rwxmin = pad.fUxmin;
         rwxmax = pad.fUxmax;
         rwymin = pad.fUymin;
         rwymax = pad.fUymax;
         logx = pad['fLogx'];
         logy = pad['fLogy'];
         logz = pad['fLogz'];
         gridx = pad['fGridx'];
         gridy = pad['fGridy'];
      }
      if (histo!=null) {
         minimum = histo['fYaxis']['fXmin'];
         maximum = histo['fYaxis']['fXmax'];
         if (pad) {
            uxmin = JSROOT.Painter.padtoX(pad, rwxmin);
            uxmax = JSROOT.Painter.padtoX(pad, rwxmax);
         }
      } else {
         for (var i = 0; i < graphs.arr.length; ++i) {
            var r = graphs.arr[i].ComputeRange();
            if ((i==0) || (r.xmin < rwxmin)) rwxmin = r.xmin;
            if ((i==0) || (r.ymin < rwymin)) rwymin = r.ymin;
            if ((i==0) || (r.xmax > rwxmax)) rwxmax = r.xmax;
            if ((i==0) || (r.ymax > rwymax)) rwymax = r.ymax;
         }
         if (rwxmin == rwxmax)
            rwxmax += 1.;
         if (rwymin == rwymax)
            rwymax += 1.;
         dx = 0.05 * (rwxmax - rwxmin);
         dy = 0.05 * (rwymax - rwymin);
         uxmin = rwxmin - dx;
         uxmax = rwxmax + dx;
         if (logy) {
            if (rwymin <= 0)
               rwymin = 0.001 * rwymax;
            minimum = rwymin / (1 + 0.5 * JSROOT.Math.log10(rwymax / rwymin));
            maximum = rwymax * (1 + 0.2 * JSROOT.Math.log10(rwymax / rwymin));
         } else {
            minimum = rwymin - dy;
            maximum = rwymax + dy;
         }
         if (minimum < 0 && rwymin >= 0)
            minimum = 0;
         if (maximum > 0 && rwymax <= 0)
            maximum = 0;
      }
      if (this.mgraph['fMinimum'] != -1111)
         rwymin = minimum = this.mgraph['fMinimum'];
      if (this.mgraph['fMaximum'] != -1111)
         rwymax = maximum = this.mgraph['fMaximum'];
      if (uxmin < 0 && rwxmin >= 0) {
         if (logx) uxmin = 0.9 * rwxmin;
         // else uxmin = 0;
      }
      if (uxmax > 0 && rwxmax <= 0) {
         if (logx) uxmax = 1.1 * rwxmax;
      }
      if (minimum < 0 && rwymin >= 0) {
         if (logy) minimum = 0.9 * rwymin;
      }
      if (maximum > 0 && rwymax <= 0) {
         if (logy) maximum = 1.1 * rwymax;
      }
      if (minimum <= 0 && logy)
         minimum = 0.001 * maximum;
      if (uxmin <= 0 && logx) {
         if (uxmax > 1000)
            uxmin = 1;
         else
            uxmin = 0.001 * uxmax;
      }
      rwymin = minimum;
      rwymax = maximum;
      if (histo!=null) {
         histo['fYaxis']['fXmin'] = rwymin;
         histo['fYaxis']['fXmax'] = rwymax;
      }

      // Create a temporary histogram to draw the axis (if necessary)
      if (!histo) {
         histo = JSROOT.Create("TH1I");
         histo['fXaxis']['fXmin'] = rwxmin;
         histo['fXaxis']['fXmax'] = rwxmax;
         histo['fYaxis']['fXmin'] = rwymin;
         histo['fYaxis']['fXmax'] = rwymax;
      }

      // histogram painter will be first in the pad, will define axis and
      // interactive actions
      this.firstpainter = JSROOT.Painter.drawHistogram1D(this.divid, histo);

      for ( var i in graphs.arr) {
         var subpainter = JSROOT.Painter.drawGraph(this.divid, graphs.arr[i]);
         this.painters.push(subpainter);
      }
   }

   JSROOT.Painter.drawMultiGraph = function(divid, mgraph, opt) {
      var painter = new JSROOT.TMultiGraphPainter(mgraph);
      painter.SetDivId(divid);
      painter.drawMultiGraph(opt);
      return painter;
   }

   // =====================================================================================

   JSROOT.TTextPainter = function(text) {
      JSROOT.TObjectPainter.call(this, text);
      this.text = text;
   }

   JSROOT.TTextPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TTextPainter.prototype.GetObject = function() {
      return this.text;
   }

   JSROOT.TTextPainter.prototype.drawPaveLabel = function() {

      this.RecreateDrawG(true);

      var pavelabel = this.text;

      var w = Number(this.svg_pad(true).attr("width")),
          h = Number(this.svg_pad(true).attr("height"));

      var pos_x = pavelabel['fX1NDC'] * w;
      var pos_y = (1.0 - pavelabel['fY1NDC']) * h;
      var width = Math.abs(pavelabel['fX2NDC'] - pavelabel['fX1NDC']) * w;
      var height = Math.abs(pavelabel['fY2NDC'] - pavelabel['fY1NDC']) * h;
      pos_y -= height;
      var font_size = Math.round(height / 1.9);
      var fcolor = JSROOT.Painter.root_colors[pavelabel['fFillColor']];
      var lcolor = JSROOT.Painter.root_colors[pavelabel['fLineColor']];
      var tcolor = JSROOT.Painter.root_colors[pavelabel['fTextColor']];
      var scolor = JSROOT.Painter.root_colors[pavelabel['fShadowColor']];
      if (pavelabel['fFillStyle'] == 0) fcolor = 'none';
      // align = 10*HorizontalAlign + VerticalAlign
      // 1=left adjusted, 2=centered, 3=right adjusted
      // 1=bottom adjusted, 2=centered, 3=top adjusted
      var align = 'start', halign = Math.round(pavelabel['fTextAlign'] / 10);
      var baseline = 'bottom', valign = pavelabel['fTextAlign'] % 10;
      if (halign == 1) align = 'start';
      else if (halign == 2) align = 'middle';
      else if (halign == 3) align = 'end';
      if (valign == 1) baseline = 'bottom';
      else if (valign == 2) baseline = 'middle';
      else if (valign == 3) baseline = 'top';
      var lmargin = 0;
      switch (halign) {
         case 1: lmargin = pavelabel['fMargin'] * width; break;
         case 2: lmargin = width / 2; break;
         case 3: lmargin = width - (pavelabel['fMargin'] * width); break;
      }
      var lwidth = pavelabel['fBorderSize'] ? pavelabel['fBorderSize'] : 0;
      var fontDetails = JSROOT.Painter.getFontDetails(pavelabel['fTextFont']);

      var pave = this.draw_g
                   .attr("width", width)
                   .attr("height", height)
                   .attr("transform", "translate(" + pos_x + "," + pos_y + ")");

      pave.append("svg:rect")
             .attr("x", 0)
             .attr("y", 0)
             .attr("width", width)
             .attr("height", height)
             .attr("fill", fcolor)
             .style("stroke-width", lwidth ? 1 : 0)
             .style("stroke", lcolor);

      var line = JSROOT.Painter.translateLaTeX(pavelabel['fLabel']);

      var lw = JSROOT.Painter.stringWidth(this.svg_pad(true), line, font_size, fontDetails);
      if (lw > width) font_size = Math.floor(font_size * (width / lw));

      pave.append("text")
             .attr("class", "text")
             .attr("text-anchor", align)
             .attr("x", lmargin)
             .attr("y", (height / 2) + (font_size / 3))
             .attr("xml:space","preserve")
             .attr("font-weight", fontDetails['weight'])
             .attr("font-style", fontDetails['style'])
             .attr("font-family", fontDetails['name'])
             .attr("font-size", font_size)
             .attr("fill", tcolor)
             .text(line);

      if (lwidth && lwidth > 1) {
         pave.append("svg:line")
               .attr("x1", width + (lwidth / 2))
               .attr("y1", lwidth + 1)
               .attr("x2", width + (lwidth / 2))
               .attr("y2", height + lwidth - 1)
               .style("stroke", lcolor)
               .style("stroke-width", lwidth);
         pave.append("svg:line")
               .attr("x1", lwidth + 1)
               .attr("y1", height + (lwidth / 2))
               .attr("x2", width + lwidth - 1)
               .attr("y2", height + (lwidth / 2))
               .style("stroke", lcolor)
               .style("stroke-width", lwidth);
      }
   }

   JSROOT.TTextPainter.prototype.drawText = function() {
      this.RecreateDrawG(true);

      var kTextNDC = JSROOT.BIT(14);

      var pad = this.root_pad();
      if (pad==null) {
         alert("Cannot draw text with real TPad object");
         return;
      }

      var i, w = Number(this.svg_pad(true).attr("width")),
             h = Number(this.svg_pad(true).attr("height"));
      var align = 'start', halign = Math.round(this.text['fTextAlign'] / 10);
      var baseline = 'bottom', valign = this.text['fTextAlign'] % 10;
      if (halign == 1) align = 'start';
      else if (halign == 2) align = 'middle';
      else if (halign == 3) align = 'end';
      if (valign == 1) baseline = 'bottom';
      else if (valign == 2) baseline = 'middle';
      else if (valign == 3) baseline = 'top';
      var lmargin = 0;
      switch (halign) {
         case 1: lmargin = this.text['fMargin'] * w; break;
         case 2: lmargin = w / 2; break;
         case 3: lmargin = w - (this.text['fMargin'] * w); break;
      }
      var font_size = Math.round(this.text['fTextSize'] * Math.min(w,h));
      var pos_x = 0, pos_y = 0;
      if (this.text.TestBit(kTextNDC)) {
         pos_x = pad['fX1'] + this.text['fX'] * (pad['fX2'] - pad['fX1']);
         pos_y = pad['fY1'] + this.text['fY'] * (pad['fY2'] - pad['fY1']);
      } else {
         pos_x = JSROOT.Painter.xtoPad(this.text['fX'], pad);
         pos_y = JSROOT.Painter.ytoPad(this.text['fY'], pad);
      }
      pos_x = ((Math.abs(pad['fX1']) + pos_x) / (pad['fX2'] - pad['fX1'])) * w;
      pos_y = (1 - ((Math.abs(pad['fY1']) + pos_y) / (pad['fY2'] - pad['fY1']))) * h;
      var tcolor = JSROOT.Painter.root_colors[this.text['fTextColor']];
      var fontDetails = JSROOT.Painter.getFontDetails(this.text['fTextFont']);

      var string = this.text['fTitle'];
      // translate the LaTeX symbols
      if (this.text['_typename'] == 'TLatex')
         string = JSROOT.Painter.translateLaTeX(string);

      this.draw_g.append("text")
              .attr("class", "text")
              .attr("x", pos_x)
              .attr("y", pos_y)
              .attr("font-family", fontDetails['name'])
              .attr("font-weight", fontDetails['weight'])
              .attr("font-style", fontDetails['style'])
              .attr("font-size", font_size)
              .attr("text-anchor", align)
              .attr("fill", tcolor)
              .text(string);
   }

   JSROOT.TTextPainter.prototype.UpdateObject = function(obj) {
      if (this.text['_typename'] != obj['_typename']) return false;
      if (this.text['_typename'] == 'TPaveLabel') {
         this.text['fLabel'] = obj['fLabel'];
      } else {
         this.text['fTitle'] = obj['fTitle'];
      }

      return true;
   }


   JSROOT.TTextPainter.prototype.Redraw = function() {
      if (this.text['_typename'] == 'TPaveLabel')
         this.drawPaveLabel();
      else
         this.drawText();
   }

   JSROOT.Painter.drawText = function(divid, text) {
      var painter = new JSROOT.TTextPainter(text);
      painter.SetDivId(divid);
      painter.Redraw();
      return painter;
   }

   JSROOT.Painter.drawStreamerInfo = function(divid, obj) {
      $("#" + divid).css({ overflow : 'auto' });
      var painter = new JSROOT.HierarchyPainter('sinfo', divid);
      painter.ShowStreamerInfo(obj);
      return painter;
   }

   // =========== painter of hierarchical structures =================================

   JSROOT.HList = [];

   JSROOT.DelHList = function(_name) {
      for ( var i in JSROOT.HList)
         if (JSROOT.HList[i].name == _name) {
            var old = JSROOT.HList[i];
            JSROOT.HList.splice(i, 1);
            delete old;
            return true;
         }
   }

   JSROOT.AddHList = function(_name, _h) {
      JSROOT.DelHList(_name);
      JSROOT.HList.push({
         name : _name,
         h : _h
      });
   }

   JSROOT.H = function(name) {
      for ( var i in JSROOT.HList)
         if (JSROOT.HList[i].name == name)
            return JSROOT.HList[i].h;
      return null;
   }

   JSROOT.HierarchyPainter = function(name, frameid) {
      JSROOT.TBasePainter.call(this);
      JSROOT.AddHList(name, this);
      this.name = name;
      this.frameid = frameid;
      this.h = null; // hierarchy
   }

   JSROOT.HierarchyPainter.prototype = Object.create(JSROOT.TBasePainter.prototype);

   JSROOT.HierarchyPainter.prototype.Cleanup = function() {
      JSROOT.DelHList(this.name);
   }

   JSROOT.HierarchyPainter.prototype.GlobalName = function(suffix) {
      var res = "JSROOT.H(\'" + this.name + "\')";
      if (suffix != null) res += suffix;
      return res;
   }

   JSROOT.HierarchyPainter.prototype.ListHierarchy = function(folder, lst) {
      folder['_childs'] = [];
      for ( var i in lst.arr) {
         var obj = lst.arr[i];
         var item = {
            _name : obj['fName'],
            _kind : "ROOT." + obj['_typename'],
            _readobj : obj
         };
         folder._childs.push(item);
      }
   }

   JSROOT.HierarchyPainter.prototype.StreamerInfoHierarchy = function(folder, lst) {
      folder['_childs'] = [];

      for ( var i in lst.arr) {
         var entry = lst.arr[i]

         if (typeof (entry['fName']) == 'undefined') {
            console.log("strange element in StreamerInfo with name " + entry['fName']);
            continue;
         }

         var item = {
            _name : entry['fName'],
            _kind : "",
            _childs : []
         };

         folder._childs.push(item);

         item._childs.push({ _name : 'Checksum: ' + entry['fCheckSum'] });
         item._childs.push({ _name : 'Class version: ' + entry['fClassVersion'] });
         if (entry['fTitle'] != '') item._childs.push({ _name : 'Title: ' + entry['fTitle'] });
         if (typeof entry['fElements'] == 'undefined') continue;
         for ( var l in entry['fElements']['arr']) {
            var elem = entry['fElements']['arr'][l];
            if ((elem == null) || (typeof (elem['fName']) == 'undefined')) continue;
            var info = elem['fTypeName'] + " " + elem['fName'] + ";";
            if (elem['fTitle'] != '') info += " // " + elem['fTitle'];
            item._childs.push({ _name : info });
         }
      }
   }

   JSROOT.HierarchyPainter.prototype.TreeHierarchy = function(node, obj) {
      node._childs = [];

      for ( var i in obj['fBranches'].arr) {
         var branch = obj['fBranches'].arr[i];
         var nb_leaves = branch['fLeaves'].arr.length;

         // display branch with only leaf as leaf
         if (nb_leaves == 1 && branch['fLeaves'].arr[0]['fName'] == branch['fName']) nb_leaves = 0;

         var subitem = {
            _name : branch['fName'],
            _kind : nb_leaves > 0 ? "ROOT.TBranch" : "ROOT.TLeafF"
         }

         node._childs.push(subitem);

         if (nb_leaves > 0) {
            subitem._childs = [];
            for (var j = 0; j < nb_leaves; ++j) {
               var leafitem = {
                  _name : branch['fLeaves'].arr[j]['fName'],
                  _kind : "ROOT.TLeafF"
               }
               subitem._childs.push(leafitem);
            }
         }
      }
   }

   JSROOT.HierarchyPainter.prototype.KeysHierarchy = function(folder, keys, file) {
      folder['_childs'] = [];

      var painter = this;
      for (var i in keys) {
         var key = keys[i];

         var item = {
            _name : key['fName'] + ";" + key['fCycle'],
            _kind : "ROOT." + key['fClassName'],
            _title : key['fTitle'],
            _keyname : key['fName'],
            _readobj : null
         };

         if ('fRealName' in key)
            item['_realname'] = key['fRealName'] + ";" + key['fCycle'];

         // console.log("key class = " + key['fClassName']);

         if ((key['fClassName'] == 'TTree' || key['fClassName'] == 'TNtuple')) {
            item["_more"] = true;

            item['_expand'] = function(node, obj) {
               painter.TreeHierarchy(node, obj);
               return true;
            }
         } else if (key['fClassName'] == 'TDirectory'  || key['fClassName'] == 'TDirectoryFile') {
            item["_more"] = true;
            item["_isdir"] = true;
            item['_expand'] = function(node, obj) {
               painter.KeysHierarchy(node, obj.fKeys);
               return true;
            }
         } else if ((key['fClassName'] == 'TList') && (key['fName'] == 'StreamerInfo') && (file != null)) {
            item['_name'] = 'StreamerInfo';
            item['_kind'] = "ROOT.TStreamerInfoList";
            item['_title'] = "List of streamer infos for binary I/O";
            item['_readobj'] = file.fStreamerInfos;
            item['_expand'] = function(node, obj) {
               painter.StreamerInfoHierarchy(node, obj);
               return true;
            }

         } else if (key['fClassName'] == 'TList'
               || key['fClassName'] == 'TObjArray'
               || key['fClassName'] == 'TClonesArray') {
                item["_more"] = true;
                item['_expand'] = function(node, obj) {
                   painter.ListHierarchy(node, obj);
                   return true;
                }
         }

         folder._childs.push(item);
      }
   }

   JSROOT.HierarchyPainter.prototype.FileHierarchy = function(file) {
      var painter = this;

      var folder = {
         _name : file.fFileName,
         _kind : "ROOT.TFile",
         _file : file,
         // this is normal get method, where item name is used
         _get : function(item, callback) {
            if ((this._file == null) || (item._readobj != null)) {
               if (typeof callback == 'function')
                  callback(item, item._readobj);
               return;
            }

            var fullname = painter.itemFullName(item, this);
            // var pos = fullname.lastIndexOf(";");
            // if (pos>0) fullname = fullname.slice(0, pos);

            this._file.ReadObject(fullname, function(obj) {
               item._readobj = obj;
               if ('_expand' in item)
                  item._name = item._keyname; // remove cycle number for
                                                // objects supporting expand
               if (typeof callback == 'function')
                  callback(item, obj);
            });
         },
         // this is alternative get method, where items may not exists (due to
         // missing/not-read subfolder)
         _getdirect : function(itemname, callback) {
            this._file.ReadObject(itemname, function(obj) {
               if (typeof callback == 'function')
                  callback(itemname, obj);
            });
         }
      };

      this.KeysHierarchy(folder, file.fKeys, file);

      return folder;
   }

   JSROOT.HierarchyPainter.prototype.Find = function(fullname, top, replace) {
      if (!top) top = this.h;

      if (fullname.length == 0) return top;

      var pos = -1;

      do {
         // we try to find element with slashes inside
         pos = fullname.indexOf("/", pos + 1);

         var localname = (pos < 0) ? fullname : fullname.substr(0, pos);

         for ( var i in top._childs)
            if (top._childs[i]._name == localname) {
               top._childs[i]['_parent'] = top; // set parent pointer when
                                                // searching child
               if ((pos + 1 == fullname.length) || (pos < 0)) {
                  if (replace != null)
                     top._childs[i] = replace;
                  return top._childs[i];
               }

               return this.Find(fullname.substr(pos + 1), top._childs[i],
                     replace);
            }
      } while (pos > 0);
      return null;
   }

   JSROOT.HierarchyPainter.prototype.itemFullName = function(node, uptoparent) {
      var res = "";

      while ('_parent' in node) {
         if (res.length > 0) res = "/" + res;
         res = node._name + res;
         node = node._parent;
         if ((uptoparent != null) && (node == uptoparent)) break;
      }

      return res;
   }

   JSROOT.HierarchyPainter.prototype.CheckCanDo = function(node) {
      var cando = { expand : false, display : false, scan : true, open : false,
                    img1 : "", img2 : "", html : "", ctxt : false };

      var kind = node["_kind"];
      if (kind == null) kind = "";

      cando.expand = ('_more' in node);

      if (node == this.h) {
         cando.ctxt = true;
      } else if (kind == "ROOT.Session") {
         cando.img1 = JSROOT.source_dir + 'img/globe.gif';
      } else if (kind.match(/^ROOT.TH1/)) {
         cando.img1 = JSROOT.source_dir + 'img/histo.png';
         cando.scan = false;
         cando.display = true;
      } else if (kind.match(/^ROOT.TH2/)) {
         cando.img1 = JSROOT.source_dir + 'img/histo2d.png';
         cando.scan = false;
         cando.display = true;
      } else if (kind.match(/^ROOT.TH3/)) {
         cando.img1 = JSROOT.source_dir + 'img/histo3d.png';
         cando.scan = false;
         cando.display = true;
      } else if (kind == "ROOT.TCanvas") {
         cando.img1 = JSROOT.source_dir + 'img/canvas.png';
         cando.display = true;
      } else if (kind == "ROOT.TProfile") {
         cando.img1 = JSROOT.source_dir + 'img/profile.png';
         cando.display = true;
      } else if (kind.match(/^ROOT.TGraph/) || (kind=="TCutG")) {
         cando.img1 = JSROOT.source_dir + 'img/graph.png';
         cando.display = true;
      } else if (kind == "ROOT.TF1") {
         cando.img1 = JSROOT.source_dir + 'img/graph.png';
         cando.display = true;
      } else if (kind == "ROOT.TTree")
         cando.img1 = JSROOT.source_dir + 'img/tree.png';
      else if (kind == "ROOT.TFolder") {
         cando.img1 = JSROOT.source_dir + 'img/folder.gif';
         cando.img2 = JSROOT.source_dir + 'img/folderopen.gif';
      } else if (kind == "ROOT.TNtuple")
         cando.img1 = JSROOT.source_dir + 'img/tree.png';
      else if (kind == "ROOT.TBranch")
         cando.img1 = JSROOT.source_dir + 'img/branch.png';
      else if (kind.match(/^ROOT.TLeaf/))
         cando.img1 = JSROOT.source_dir + 'img/leaf.png';
      else if (kind == "ROOT.TStreamerInfoList") {
         cando.img1 = JSROOT.source_dir + 'img/question.gif';
         cando.expand = false;
         cando.display = true;
      } else if ((kind.indexOf("ROOT.") == 0) && JSROOT.canDraw(kind.slice(5))) {
         cando.img1 = JSROOT.source_dir + 'img/histo.png';
         cando.scan = false;
         cando.display = true;
      }

      return cando;
   }

   JSROOT.HierarchyPainter.prototype.createNode = function(node, fullname, parent) {
      var nodename = node._name;

      var nodefullname = "";

      if (node != this.h) {
         nodefullname = nodename;
         if (fullname.length > 0)
            nodefullname = fullname + "/" + nodename;
         node['_parent'] = parent;
      }

      var cando = this.CheckCanDo(node);

      // console.log("add kind = " + kind + " name = " + node._name);

      if (!node._childs || !cando.scan) {
         if (cando.expand) {
            cando.html = "javascript: " + this.GlobalName() + ".expand(\'" + nodefullname + "\');";
            if (cando.img1.length == 0) {
               cando.img1 = JSROOT.source_dir + 'img/folder.gif';
               cando.img2 = JSROOT.source_dir + 'img/folderopen.gif';
            }
         } else
         if (cando.display) {
            cando.html = "javascript: " + this.GlobalName() + ".display(\'" + nodefullname + "\');";
         } else
         if (cando.open && (cando.html.length == 0))
            cando.html = nodefullname + "/";
      }

      if (cando.img2 == "")
         cando.img2 = cando.img1;

      node['_d'] = {
         name : nodename,
         url : cando.html,
         title : "",
         icon : cando.img1,
         iconOpen : cando.img2,
         _id : 0,     // id used in html
         _io : false, // is open
         _is : false, // is selected
         _ls : false, // last sibling
         _hc : false  // has childs
      };

      if ('_realname' in node)
         node['_d']['name'] = node['_realname'];

      if ('_title' in node)
         node['_d']['title'] = node['_title'];

      if ('_fullname' in node)
         node['_d']['title'] += ("  fullname: " + node['_fullname']);

      if (node['_d']['title'].length == 0)
         node['_d']['title'] = node['_d']['name'];

      if (parent && parent._childs && (parent._childs[parent._childs.length - 1] == node))
         node['_d']._ls = true;

      node['_d']._id = this.grid++;

      // allow context menu only for objects which can be displayed or for top-level item
      if (cando.display || cando.ctxt)
         node['_d']['ctxt'] = this.GlobalName() + ".contextmenu(this, event, \'" + nodefullname + "\')";

      if (cando.scan && ('_childs' in node)) {
         node['_d']._hc = true;
         for ( var i in node._childs)
            this.createNode(node._childs[i], nodefullname, node);
      }
   }

   JSROOT.HierarchyPainter.prototype.RefreshHtml = function(force) {
      if (this.frameid == null) return;
      var elem = document.getElementById(this.frameid);
      if (elem == null) return;

      if (this.h == null) { elem.innerHTML = "<h2>null</h2>"; return; }

      if (force && this.h._d != null) {
         delete this.h._d;
         this.h._d = null;
      }

      if (this.h._d == null) {
         this.grid = 0;
         this.createNode(this.h, "", null);
      }

      this['html'] = "<p>";
      this['html'] += "<a href=\"javascript: " + this.GlobalName() + ".toggle(true);\">open all</a>";
      this['html'] += "| <a href=\"javascript: " + this.GlobalName() + ".toggle(false);\">close all</a>";
      if ('_online' in this.h)
         this['html'] += "| <a href=\"javascript: " + this.GlobalName() + ".reload();\">reload</a>";
      if ('disp_kind' in this)
         this['html'] += "| <a href=\"javascript: " + this.GlobalName() + ".clear();\">clear</a>";

      this['html'] += "</p>";
      this['html'] += '<div class="dtree">'

      this.addItemHtml(this.h);
      this['html'] += '</div>';
      elem.innerHTML = this['html'];
   }

   JSROOT.HierarchyPainter.prototype.addItemHtml = function(hitem, onlyitem) {
      if (this.icon == null)
         this.icon = {
            root : JSROOT.source_dir + 'img/base.gif',
            folder : JSROOT.source_dir + 'img/folder.gif',
            folderOpen : JSROOT.source_dir + 'img/folderopen.gif',
            node : JSROOT.source_dir + 'img/page.gif',
            empty : JSROOT.source_dir + 'img/empty.gif',
            line : JSROOT.source_dir + 'img/line.gif',
            join : JSROOT.source_dir + 'img/join.gif',
            joinBottom : JSROOT.source_dir + 'img/joinbottom.gif',
            plus : JSROOT.source_dir + 'img/plus.gif',
            plusBottom : JSROOT.source_dir + 'img/plusbottom.gif',
            minus : JSROOT.source_dir + 'img/minus.gif',
            minusBottom : JSROOT.source_dir + 'img/minusbottom.gif',
            nlPlus : JSROOT.source_dir + 'img/nolines_plus.gif',
            nlMinus : JSROOT.source_dir + 'img/nolines_minus.gif'
         };

      var isroot = (hitem == this.h);

      var node = hitem._d;
      var idname = this.name + "_id_" + node._id;

      if (!onlyitem)
         this['html'] += '<div class="dTreeNode" id="z' + idname + '">';

      // build indent
      var sindent = "";
      var prnt = isroot ? null : hitem._parent;
      while ((prnt != null) && (prnt != this.h)) {
         sindent = '<img src="' + (!prnt._d._ls ? this.icon.line : this.icon.empty) + '" alt="" />' + sindent;
         prnt = prnt._parent;
      }
      this['html'] += sindent;

      var opencode = this.GlobalName() + ".open(\'" + this.itemFullName(hitem) + "\')";

      if (isroot) {
         // for root node no extra code
      } else
      if (node._hc) {
         this['html'] += '<a href="javascript: ' + opencode + '"><img src="';
         this['html'] += ((node._io) ? (node._ls ? this.icon.minusBottom : this.icon.minus)
                                     : (node._ls ? this.icon.plusBottom : this.icon.plus));
         this['html'] += '" alt="" /></a>';
      } else {
         this['html'] += '<img src="' + ((node._ls ? this.icon.joinBottom : this.icon.join)) + '" alt="" />';
      }

      // make node icon
      if (!node.icon)
         node.icon = isroot ? this.icon.root : ((node._hc) ? this.icon.folder : this.icon.node);
      if (!node.iconOpen)
         node.iconOpen = (node._hc) ? this.icon.folderOpen : this.icon.node;
      if (isroot) {
         node.icon = this.icon.root;
         node.iconOpen = this.icon.root;
      }

      this['html'] += '<img src="' + ((node._io) ? node.iconOpen : node.icon) + '" alt=""/>';

      if (node.url) {
         this['html'] += '<a class="' + (node._is ? 'nodeSel' : 'node') + '" href="' + node.url + '"';
      } else
      if (node._hc && !isroot) {
         this['html'] += '<a href="javascript: ' + opencode + '" class="node"';
      } else {
         this['html'] += '<a';
      }

      if (node.title) this['html'] += ' title="' + node.title + '"';
      if (node.ctxt) this['html'] += ' oncontextmenu="' + node.ctxt + '"';
      this['html'] += '>' + node.name + '</a>';

      if (onlyitem) return;

      this['html'] += '</div>';

      var childs_display = node._hc && (isroot || node._io);

      // place for childs

      this['html'] += '<div id="d' + idname + '" class="clip" style="display:' + (childs_display ? 'block' : 'none') + ';">';
      if (childs_display)
         for ( var i in hitem._childs)
            this.addItemHtml(hitem._childs[i]);
      this['html'] += '</div>';
   }

   JSROOT.HierarchyPainter.prototype.open = function(itemname) {
      var hitem = this.Find(itemname);
      if (hitem == null) return;
      this.setDNodeOpenStatus(hitem, !hitem._d._io);
   }

   JSROOT.HierarchyPainter.prototype.setDNodeOpenStatus = function(hitem, status, force) {
      if (hitem == null) return;

      var node = hitem._d;
      var idname = this.name + "_id_" + node._id;

      node._io = status;

      var zDiv = document.getElementById('z' + idname);
      var dDiv = document.getElementById('d' + idname);

      if (zDiv) {
         this['html'] = '';
         this.addItemHtml(hitem, true);
         zDiv.innerHTML = this['html'];
      }

      if (dDiv) {
         if (node._io && node._hc && (force || (dDiv.childNodes.length == 0))) {
            this['html'] = '';
            for ( var i in hitem._childs)
               this.addItemHtml(hitem._childs[i]);
            dDiv.innerHTML = this['html'];
         }
         dDiv.style.display = node._io ? 'block' : 'none';
      }
   }

   JSROOT.HierarchyPainter.prototype.toggle = function(status) {
      var painter = this;

      var toggleItem = function(hitem) {

         if (hitem != painter.h)
            hitem._d._io = status;

         if ('_childs' in hitem)
            for ( var i in hitem._childs)
               toggleItem(hitem._childs[i]);
      }

      toggleItem(this.h);

      this.RefreshHtml();
   }

   JSROOT.HierarchyPainter.prototype.get = function(itemname, callback, options) {
      var item = this.Find(itemname);

      // process get in central method of hierarchy item (if exists)
      if ((item == null) && ('_getdirect' in this.h))
         return this.h._getdirect(itemname, callback);

      // normally search _get method in the parent items
      var curr = item;
      while (curr != null) {
         if (('_get' in curr) && (typeof (curr._get) == 'function'))
            return curr._get(item, callback);
         curr = ('_parent' in curr) ? curr['_parent'] : null;
      }

      if (typeof callback == 'function')
         callback(item, null);
   }

   JSROOT.HierarchyPainter.prototype.display = function(itemname, options, call_back) {
      if (!this.CreateDisplay()) return;

      var mdi = this['disp'];

      this.get(itemname, function(item, obj) {
         var painter = mdi.Draw(itemname, obj, options);
         if (typeof call_back == 'function') call_back(painter);
      });
   }

   JSROOT.HierarchyPainter.prototype.displayAll = function(items, options) {
      if ((items == null) || (items.length == 0)) return;
      if (!this.CreateDisplay()) return;
      if (options == null) options = [];
      while (options.length < items.length)
         options.push("");

      var mdi = this['disp'];

      // First of all check that items are exists, look for cycle extension
      for (var i in items)
         if (!this.Find(items[i]) && this.Find(items[i] + ";1")) items[i] += ";1";

      // Than create empty frames for each item
      for (var i in items)
         mdi.CreateFrame(items[i]);

      // Display items
      for ( var i in items)
         this.display(items[i], options[i]);
   }

   JSROOT.HierarchyPainter.prototype.reload = function() {
      if ('_online' in this.h)
         this.OpenOnline(this.h['_online']);
   }

   JSROOT.HierarchyPainter.prototype.ExpandDtree = function(node) {
      var itemname = this.itemFullName(node);

      for ( var i in node._childs)
         this.createNode(node._childs[i], itemname, node);

      node._d._hc = true;
      node._d._io = true;
      node._d.url = "";
      node._d.name = node._name;

      this.setDNodeOpenStatus(node, true, true);
   }

   JSROOT.HierarchyPainter.prototype.expand = function(itemname) {
      var painter = this;

      var item0 = this.Find(itemname);
      if (item0==null) return;
      item0['_doing_expand'] = true;

      this.get(itemname, function(item, obj) {
         delete item0['_doing_expand'];
         if ((item == null) || (obj == null)) return;

         var curr = item;
         while (curr != null) {
            if (('_expand' in curr) && (typeof (curr['_expand']) == 'function')) {
                if (curr['_expand'](item, obj))
                   painter.ExpandDtree(item);
                return;
            }
            curr = ('_parent' in curr) ? curr['_parent'] : null;
         }
      });
   }

   JSROOT.HierarchyPainter.prototype.OpenRootFile = function(filepath, andThan) {
      var pthis = this;

      var f = new JSROOT.TFile(filepath, function(file) {
         if (file == null) return;
         // for the moment file is the only entry
         pthis.h = pthis.FileHierarchy(file);

         pthis.RefreshHtml();

         if (typeof andThan == 'function') andThan();
      });
   }

   JSROOT.HierarchyPainter.prototype.GetFileProp = function(itemname) {
      var item = this.Find(itemname);
      if (item == null) return null;

      var subname = item._name;
      while (item._parent != null) {
         item = item._parent;
         if ('_file' in item) {
            return {
               fileurl : item._file.fURL,
               itemname : subname
            };
         }
         subname = item._name + "/" + subname;
      }

      return null;
   }

   JSROOT.HierarchyPainter.prototype.CompleteOnline = function(ready_callback) {
      // method called at the moment when new description (h.json) is loaded
      // and before any graphical element is created
      // one can load extra scripts here or assign draw functions
      ready_callback();
   }

   JSROOT.HierarchyPainter.prototype.OpenOnline = function(server_address, user_callback) {
      if (!server_address) server_address = "";

      var painter = this;

      var req = JSROOT.NewHttpRequest(server_address + "h.json?compact=3", 'object', function(result) {
         painter.h = result;
         if (painter.h == null) return;

         // mark top hierarchy as online data and
         painter.h['_online'] = server_address;

         painter.h['_get'] = function(item, callback) {

            var url = painter.itemFullName(item);
            if (url.length > 0) url += "/";
            var h_get = ('_more' in item) || ('_doing_expand' in item);
            url += h_get ? 'h.json?compact=3' : 'root.json.gz?compact=3';

            var itemreq = JSROOT.NewHttpRequest(url, 'object', function(obj) {
               if ((obj != null) && !h_get && (item._name === "StreamerInfo")
                     && (obj['_typename'] === 'TList'))
                  obj['_typename'] = 'TStreamerInfoList';

               if (typeof callback == 'function')
                  callback(item, obj);
            });

            itemreq.send(null);
         }

         painter.h['_expand'] = function(node, obj) {
            // central function for all expand

            if ((obj != null) && (node != null) && ('_childs' in obj)) {
               node._childs = obj._childs;
               obj._childs = null;
               return true;
            }
            return false;
         }

         painter.CompleteOnline(function() {
            if (painter.h != null)
               painter.RefreshHtml(true);

            if (typeof user_callback == 'function')
               user_callback(painter);
         });
      });

      req.send(null);
   }

   JSROOT.HierarchyPainter.prototype.GetOnlineProp = function(itemname) {
      var item = this.Find(itemname);
      if (item == null) return null;

      var subname = item._name;
      while (item._parent != null) {
         item = item._parent;

         if ('_online' in item) {
            return {
               server : item['_online'],
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
      var cando = this.CheckCanDo(node);

      if (cando.display)
         JSROOT.Painter.menuitem(menu, "Draw", function() { painter.display(itemname); });

      if (cando.expand || cando.display)
         JSROOT.Painter.menuitem(menu, "Expand", function() { painter.expand(itemname); });

      var drawurl = onlineprop.server + onlineprop.itemname + "/draw.htm";
      if (this['_monitoring_on'])
         drawurl += "?monitoring=" + this['_monitoring_interval'];

      if (cando.display)
         JSROOT.Painter.menuitem(menu, "Draw in new window", function() { window.open(drawurl); });

      if (cando.display)
         JSROOT.Painter.menuitem(menu, "Draw as png", function() {
            window.open(onlineprop.server + onlineprop.itemname + "/root.png?w=400&h=300&opt=");
         });
   }

   JSROOT.HierarchyPainter.prototype.ShowStreamerInfo = function(sinfo) {
      this.h = { _name : "StreamerInfo" };
      this.StreamerInfoHierarchy(this.h, sinfo);
      this.RefreshHtml();
   }

   JSROOT.HierarchyPainter.prototype.Adopt = function(h) {
      this.h = h;
      this.RefreshHtml();
   }

   JSROOT.HierarchyPainter.prototype.contextmenu = function(element, event, itemname) {
      event.preventDefault();

      var onlineprop = this.GetOnlineProp(itemname);
      var fileprop = this.GetFileProp(itemname);

      var menu = JSROOT.Painter.createmenu(event);

      function qualifyURL(url) {
         function escapeHTML(s) {
            return s.split('&').join('&amp;').split('<').join('&lt;').split('"').join('&quot;');
         }
         var el = document.createElement('div');
         el.innerHTML = '<a href="' + escapeHTML(url) + '">x</a>';
         return el.firstChild.href;
      }

      var painter = this;

      if (itemname == "") {
         var addr = "";
         if ('_online' in this.h) {
            addr = "/?";
            if (this['_monitoring_on'])
               addr += "monitoring=" + this['_monitoring_interval'];
         } else if ('_file' in this.h) {
            addr = JSROOT.source_dir + "index.htm?";
            addr += "file=" + this.h['_file'].fURL;
         }

         if (this['disp_kind']) {
            if (addr.length > 2) addr += "&";
            addr += "layout=" + this['disp_kind'].replace(/ /g, "");
         }

         var items = [];

         if (this['disp'] != null)
            this['disp'].ForEach(function(panel, itemname, painter) {
               items.push(itemname);
            });

         if (items.length == 1) {
            if (addr.length > 2) addr += "&";
            addr += "item=" + items[0];
         } else if (items.length > 1) {
            if (addr.length > 2) addr += "&";
            addr += "items=" + JSON.stringify(items);
         }

         JSROOT.Painter.menuitem(menu, "Direct link", function() { window.open(addr); });
         JSROOT.Painter.menuitem(menu, "Only items", function() { window.open(addr + "&nobrowser"); });
      } else
      if (onlineprop != null) {
         this.FillOnlineMenu(menu, onlineprop, itemname);
      } else
      if (fileprop != null) {
         JSROOT.Painter.menuitem(menu, "Draw", function() { painter.display(itemname); });
         var filepath = qualifyURL(fileprop.fileurl);
         if (filepath.indexOf(JSROOT.source_dir) == 0)
            filepath = filepath.slice(JSROOT.source_dir.length);
         JSROOT.Painter.menuitem(menu, "Draw in new window", function() {
             window.open(JSROOT.source_dir + "index.htm?nobrowser&file=" + filepath + "&item=" + fileprop.itemname);
         });
      }

      JSROOT.Painter.menuitem(menu, "Close", function() {});

      return false;
   }

   JSROOT.HierarchyPainter.prototype.SetDisplay = function(kind, frameid) {
      this['disp_kind'] = kind;
      this['disp_frameid'] = frameid;
   }

   JSROOT.HierarchyPainter.prototype.clear = function() {
      if ('disp' in this)
         this['disp'].Reset();
   }

   JSROOT.HierarchyPainter.prototype.CreateDisplay = function(force) {
      if ('disp' in this) {
         if (!force && this['disp'].NumDraw() > 0) return true;
         this['disp'].Reset();
         delete this['disp'];
      }

      // check that we can found frame where drawing should be done
      if (document.getElementById(this['disp_frameid']) == null) return false;

      if (this['disp_kind'] == "tabs")
         this['disp'] = new JSROOT.TabsDisplay(this['disp_frameid']);
      else
      if (this['disp_kind'].search("grid") == 0)
         this['disp'] = new JSROOT.GridDisplay(this['disp_frameid'], this['disp_kind']);
      else
         this['disp'] = new JSROOT.CollapsibleDisplay(this['disp_frameid']);

      return true;
   }

   JSROOT.HierarchyPainter.prototype.CheckResize = function(force) {
      if ('disp' in this)
         this['disp'].CheckResize();
   }

   // ================================================================

   // JSROOT.MDIDisplay - class to manage multiple document interface for
   // drawings

   JSROOT.MDIDisplay = function(frameid) {
      this.frameid = frameid;
   }

   JSROOT.MDIDisplay.prototype.ForEach = function(userfunc, only_visible) {
      alert("ForEach not implemented");
   }

   JSROOT.MDIDisplay.prototype.NumDraw = function() {
      var cnt = 0;
      this.ForEach(function() { cnt++; });
      return cnt;
   }

   JSROOT.MDIDisplay.prototype.FindFrame = function(searchitemname) {
      var found_frame = null;

      this.ForEach(function(frame, itemname) {
         if (itemname == searchitemname)
            found_frame = frame;
      });

      return found_frame;
   }

   JSROOT.MDIDisplay.prototype.FindPainter = function(searchitemname) {
      var frame = this.FindFrame(searchitemname);
      //if (frame == null) return null;
      //return document.getElementById($(frame).attr('id'))['painter'];
      return frame ? $(frame).prop('painter') : null;
   }

   JSROOT.MDIDisplay.prototype.ActivateFrame = function(frame) {
      // do nothing by default
   }

   JSROOT.MDIDisplay.prototype.Draw = function(itemname, obj, drawopt) {
      // draw object with specified options
      if (!obj) return;

      var frame = this.FindFrame(itemname);

      if ((frame != null) && this.FindPainter(itemname)) {
         this.ActivateFrame(frame);
         return;
      }

      if (!JSROOT.canDraw(obj['_typename'], drawopt)) return;

      if (frame == null)
         frame = this.CreateFrame(itemname);

      this.ActivateFrame(frame);

      var painter = JSROOT.draw($(frame).attr("id"), obj, drawopt);

      this.SetPainterForFrame(frame, painter);

      return painter;
   }

   JSROOT.MDIDisplay.prototype.Redraw = function(itemname, obj, drawopt) {
      // (re)draw object with specified options
      // if object was not drawn before, normal draw will be performed

      var p = this.FindPainter(itemname);
      if (p==null) {
         p = this.Draw(itemname, obj, drawopt);
      } else {
         if (p.UpdateObject(obj)) p.RedrawPad();
      }

      return p;
   }


   JSROOT.MDIDisplay.prototype.SetPainterForFrame = function(frame, painter) {
      //var hid = $(frame).attr('id');
      //document.getElementById(hid)['painter'] = painter;
      $(frame).prop('painter', painter);
      this.ActivateFrame(frame);
   }

   JSROOT.MDIDisplay.prototype.CheckResize = function() {
      this.ForEach(function(panel, itemname, painter) {
         if ((painter != null) && (typeof painter['CheckResize'] == 'function'))
             painter.CheckResize();
      });
   }

   JSROOT.MDIDisplay.prototype.Reset = function() {
      this.ForEach(function(panel, itemname, painter) {
         if ((painter != null) && (typeof painter['Clenaup'] == 'function'))
            painter.Clenaup();
      });

      document.getElementById(this.frameid).innerHTML = '';
   }

   // ==================================================

   JSROOT.CloseCollapsible = function(e, el) {
      var sel = $(el)[0].textContent;
      if (typeof (sel) == 'undefined')
         return;
      sel.replace(' x', '');
      sel.replace(';', '');
      sel.replace(' ', '');
      $(el).next().andSelf().remove();
      e.stopPropagation();
   };

   JSROOT.CollapsibleDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0; // use to count newly created frames
   }

   JSROOT.CollapsibleDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.CollapsibleDisplay.prototype.ForEach = function(userfunc,  only_visible) {
      var topid = this.frameid + '_collapsible';

      if (document.getElementById(topid) == null) return;

      if (typeof userfunc != 'function') return;

      $('#' + topid).children().each(function() {

         if (!('itemname' in this)) return;

         // check if only visible specified
         if (only_visible && $(this).is(":hidden")) return;

         userfunc(this, this['itemname'], $(this).prop('painter'));
      });
   }

   JSROOT.CollapsibleDisplay.prototype.ActivateFrame = function(frame) {
      if ($(frame).is(":hidden")) {
         $(frame).prev().toggleClass("ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                 .find("> .ui-icon").toggleClass("ui-icon-triangle-1-e ui-icon-triangle-1-s").end()
                 .next().toggleClass("ui-accordion-content-active").slideDown(0);
      }
      $(frame).prev()[0].scrollIntoView();
   }

   JSROOT.CollapsibleDisplay.prototype.CreateFrame = function(itemname) {

      var topid = this.frameid + '_collapsible';

      if (document.getElementById(topid) == null)
         $("#right-div")
               .append(
                     '<div id="'
                           + topid
                           + '" class="ui-accordion ui-accordion-icons ui-widget ui-helper-reset" style="overflow:auto; overflow-y:scroll; height:100%"></div>');

      var hid = topid + "_sub" + this.cnt++;
      var uid = hid + "h";

      var entryInfo = "<h5 id=\"" + uid + "\"><a> " + itemname
            + "</a>&nbsp; </h5>\n";
      entryInfo += "<div id='" + hid + "'></div>\n";
      $("#" + topid).append(entryInfo);

      document.getElementById(hid)['itemname'] = itemname;

      $('#' + uid)
            .addClass(
                  "ui-accordion-header ui-helper-reset ui-state-default ui-corner-top ui-corner-bottom")
            .hover(function() {
               $(this).toggleClass("ui-state-hover");
            })
            .prepend('<span class="ui-icon ui-icon-triangle-1-e"></span>')
            .append(
                  '<button type="button" class="closeButton" title="close canvas" onclick="JSROOT.CloseCollapsible(event, \'#'
                        + uid
                        + '\')"><img src="'
                        + JSROOT.source_dir
                        + '/img/remove.gif"/></button>')
            .click(
                  function() {
                     $(this)
                           .toggleClass(
                                 "ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
                           .find("> .ui-icon").toggleClass(
                                 "ui-icon-triangle-1-e ui-icon-triangle-1-s")
                           .end().next().toggleClass(
                                 "ui-accordion-content-active").slideToggle(0);
                     return false;
                  })
            .next()
            .addClass(
                  "ui-accordion-content  ui-helper-reset ui-widget-content ui-corner-bottom")
            .hide();

      $('#' + uid)
            .toggleClass(
                  "ui-accordion-header-active ui-state-active ui-state-default ui-corner-bottom")
            .find("> .ui-icon").toggleClass(
                  "ui-icon-triangle-1-e ui-icon-triangle-1-s").end().next()
            .toggleClass("ui-accordion-content-active").slideToggle(0);

      // $('#'+uid)[0].scrollIntoView();

      return $("#" + hid);
   }

   // ================================================

   JSROOT.TabsDisplay = function(frameid) {
      JSROOT.MDIDisplay.call(this, frameid);
      this.cnt = 0;
   }

   JSROOT.TabsDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.TabsDisplay.prototype.ForEach = function(userfunc, only_visible) {
      var topid = this.frameid + '_tabs';

      if (document.getElementById(topid) == null) return;

      if (typeof userfunc != 'function') return;

      var cnt = -1;
      var active = $('#' + topid).tabs("option", "active");

      $('#' + topid).children().each(function() {
         // check if only_visible specified
         if (only_visible && (cnt++ != active)) return;

         if (!('itemname' in this)) return;

         userfunc(this, this['itemname'], $(this).prop('painter'));
      });
   }

   JSROOT.TabsDisplay.prototype.ActivateFrame = function(frame) {
      var cnt = 0, id = -1;
      this.ForEach(function(fr) {
         if (fr === frame)
            id = cnt;
         cnt++;
      });

      $('#' + this.frameid + "_tabs").tabs("option", "active", id);
   }

   JSROOT.TabsDisplay.prototype.CreateFrame = function(itemname) {
      var topid = this.frameid + '_tabs';

      var hid = topid + "_sub" + this.cnt++;

      var li = '<li><a href="#' + hid + '">'
            + itemname
            + '</a><span class="ui-icon ui-icon-close" role="presentation">Remove Tab</span></li>';
      var cont = '<div id="' + hid + '"></div>';

      if (document.getElementById(topid) == null) {
         $("#" + this.frameid).append('<div id="' + topid + '">' + ' <ul>' + li + ' </ul>' + cont + '</div>');

         var tabs = $("#" + topid).tabs({ heightStyle : "fill" });

         tabs.delegate("span.ui-icon-close", "click", function() {
            var panelId = $(this).closest("li").remove().attr("aria-controls");
            $("#" + panelId).remove();
            tabs.tabs("refresh");
         });
      } else {

         // var tabs = $("#tabs").tabs();

         $("#" + topid).find(".ui-tabs-nav").append(li);
         $("#" + topid).append(cont);
         $("#" + topid).tabs("refresh");
         $("#" + topid).tabs("option", "active", -1);
      }
      $('#' + hid).empty();
      document.getElementById(hid)['itemname'] = itemname;
      return $('#' + hid);
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

         if (sizex == NaN)
            sizex = 3;
         if (sizey == NaN)
            sizey = 3;
      }

      if (!sizex)
         sizex = 3;
      if (!sizey)
         sizey = sizex;
      this.sizex = sizex;
      this.sizey = sizey;
   }

   JSROOT.GridDisplay.prototype = Object.create(JSROOT.MDIDisplay.prototype);

   JSROOT.GridDisplay.prototype.ForEach = function(userfunc, only_visible) {
      var topid = this.frameid + '_grid';

      if (document.getElementById(topid) == null) return;

      if (typeof userfunc != 'function') return;

      for (var cnt = 0; cnt < this.sizex * this.sizey; cnt++) {
         var hid = topid + "_" + cnt;

         var elem = document.getElementById(hid);

         if ((elem == null) || !('itemname' in elem)) continue;

         userfunc($("#" + hid), elem['itemname'], elem['painter']);
      }
   }

   JSROOT.GridDisplay.prototype.CreateFrame = function(itemname) {

      var topid = this.frameid + '_grid';

      if (document.getElementById(topid) == null) {

         var precx = 100. / this.sizex;
         var precy = 100. / this.sizey;
         var h = $("#" + this.frameid).height() / this.sizey;
         var w = $("#" + this.frameid).width() / this.sizex;

         var content = "<table id='" + topid + "' style='width:100%; height:100%; table-layout:fixed'>";
         var cnt = 0;
         for (var i = 0; i < this.sizey; i++) {
            content += "<tr>";
            for (var j = 0; j < this.sizex; j++)
               content += "<td><div id='" + topid + "_" + cnt++ + "'></div></td>";
            content += "</tr>";
         }
         content += "</table>";

         $("#" + this.frameid).empty();
         $("#" + this.frameid).append(content);

         $("[id^=" + this.frameid + "_grid_]").height(h);
         $("[id^=" + this.frameid + "_grid_]").width(w);
      }

      var hid = topid + "_" + this.cnt;
      if (++this.cnt >= this.sizex * this.sizey) this.cnt = 0;

      $("#" + hid).empty();
      document.getElementById(hid)['itemname'] = itemname;
      document.getElementById(hid)['painter'] = null;

      return $('#' + hid);
   }

   JSROOT.GridDisplay.prototype.Reset = function() {
      JSROOT.MDIDisplay.prototype.Reset.call(this);
      this.cnt = 0;
   }

   JSROOT.GridDisplay.prototype.CheckResize = function() {
      var h = $("#" + this.frameid).height() / this.sizey;
      var w = $("#" + this.frameid).width() / this.sizex;

      // console.log("big width = " + $("#" + this.frameid).width() + " height =
      // " + $("#" + this.frameid).height());

      // set height for all table cells, it is not done automatically by browser
      // for (var cnt=0;cnt<this.sizex*this.sizey;cnt++)
      // $("#" + this.frameid + "_grid_"+cnt).height(h);

      // $("[id^=" + this.frameid + "_grid_]").height(h).width(w);

      $("[id^=" + this.frameid + "_grid_]").height(h);
      $("[id^=" + this.frameid + "_grid_]").width(w);

      JSROOT.MDIDisplay.prototype.CheckResize.call(this);
   }

   JSROOT.RegisterForResize = function(handle, delay) {
      // function used to react on browser window resize event
      // While many resize events could come in short time,
      // resize will be handled with delay after last resize event
      // handle can be function or object with CheckResize function
      // one could specify delay after which resize event will be handled

      var myInterval = null;
      var myCounter = -1;
      var myPeriod = delay ? delay / 2.5 : 100;
      if (myPeriod < 20) myPeriod = 20;

      function ResizeTimer() {
         if (myCounter < 0) return;
         myCounter += 1;
         if (myCounter < 3) return;

         if (myInterval != null) {
            clearInterval(myInterval);
            myInterval = null;
         }
         myCounter = -1;

         document.body.style.cursor = 'wait';

         if (typeof handle == 'function') handle();
         else if ((typeof handle == 'object') && (typeof handle['CheckResize'] == 'function'))
            handle.CheckResize();

         document.body.style.cursor = 'auto';
      }

      function ProcessResize() {
         if (myInterval == null) {
            myInterval = setInterval(ResizeTimer, myPeriod);
         }
         myCounter = 0;
      }

      window.addEventListener('resize', ProcessResize);
   }

   JSROOT.addDrawFunc("TCanvas", JSROOT.Painter.drawCanvas);
   JSROOT.addDrawFunc("TPad", JSROOT.Painter.drawPad);
   JSROOT.addDrawFunc("TFrame", JSROOT.Painter.drawFrame);
   JSROOT.addDrawFunc("TLegend", JSROOT.Painter.drawLegend);
   JSROOT.addDrawFunc("TPaveText", JSROOT.Painter.drawPaveText);
   JSROOT.addDrawFunc("TPaveStats", JSROOT.Painter.drawPaveText);
   JSROOT.addDrawFunc("TLatex", JSROOT.Painter.drawText);
   JSROOT.addDrawFunc("TText", JSROOT.Painter.drawText);
   JSROOT.addDrawFunc("TPaveLabel", JSROOT.Painter.drawText);
   JSROOT.addDrawFunc(/^TH1/, JSROOT.Painter.drawHistogram1D);
   JSROOT.addDrawFunc("TProfile", JSROOT.Painter.drawHistogram1D);
   JSROOT.addDrawFunc(/^TH2/, JSROOT.Painter.drawHistogram2D);
   JSROOT.addDrawFunc(/^TH3/, JSROOT.Painter.drawHistogram3D);
   JSROOT.addDrawFunc("THStack", JSROOT.Painter.drawHStack);
   JSROOT.addDrawFunc("TF1", JSROOT.Painter.drawFunction);
   JSROOT.addDrawFunc(/^TGraph/, JSROOT.Painter.drawGraph);
   JSROOT.addDrawFunc("TCutG", JSROOT.Painter.drawGraph);
   JSROOT.addDrawFunc(/^RooHist/, JSROOT.Painter.drawGraph);
   JSROOT.addDrawFunc(/^RooCurve/, JSROOT.Painter.drawGraph);
   JSROOT.addDrawFunc("TMultiGraph", JSROOT.Painter.drawMultiGraph);
   JSROOT.addDrawFunc("TStreamerInfoList", JSROOT.Painter.drawStreamerInfo);

   JSROOT.getDrawFunc = function(classname) {
      if (typeof classname != 'string') return null;

      for (var i in JSROOT.fDrawFunc) {
         if ((typeof JSROOT.fDrawFunc[i].name) === "string") {
            if (JSROOT.fDrawFunc[i].name == classname) return JSROOT.fDrawFunc[i].func;
         } else {
            if (classname.match(JSROOT.fDrawFunc[i].name)) return JSROOT.fDrawFunc[i].func;
         }
      }
      return null;
   }

   JSROOT.canDraw = function(classname) {
      return JSROOT.getDrawFunc(classname) != null;
   }

   /** @fn JSROOT.draw(divid, obj, opt)
    * Draw object in specified HTML element with given draw options  */

   JSROOT.draw = function(divid, obj, opt) {
      if ((typeof obj != 'object') || (!('_typename' in obj))) return null;

      var draw_func = JSROOT.getDrawFunc(obj['_typename']);

      if (draw_func==null) return null;

      return draw_func(divid, obj, opt);
   }

   /** @fn JSROOT.redraw(divid, obj, opt)
    * Redraw object in specified HTML element with given draw options
    * If drawing was not exists, it will be performed with JSROOT.draw.
    * If drawing was already done, that content will be updated */

   JSROOT.redraw = function(divid, obj, opt) {
      if (obj==null) return;

      var can = d3.select("#" + divid + " .root_canvas");
      var can_painter = can.node() ? can.node()['pad_painter'] : null;

      if (can_painter != null) {
         if (obj._typename=="TCanvas") {
            can_painter.RedrawObject(obj);
           return can_painter;
         }

         for (var i in can_painter.painters) {
            var obj0 = can_painter.painters[i].GetObject();

            if ((obj0 != null) && (obj0._typename == obj._typename))
               if (can_painter.painters[i].UpdateObject(obj)) {
                  can_painter.RedrawPad();
                  return can_painter.painters[i];
               }
         }
      }

      if (can_painter)
          console.log("Cannot find painter to update object of type " + obj._typename);

      $("#"+divid).empty();
      return JSROOT.draw(divid, obj, opt);
   }

})();

// JSRootPainter.js ends

// example of user code for streamer and painter

/*

 (function(){

 Amore_String_Streamer = function(buf, obj, prop, streamer) {
    console.log("read property " + prop + " of typename " + streamer[prop]['typename']);
    obj[prop] = buf.ReadTString();
 }

 Amore_Draw = function(divid, obj, opt) { // custom draw function.
    return JSROOT.draw(divid, obj['fVal'], opt);
 }

 JSROOT.addUserStreamer("amore::core::String_t", Amore_String_Streamer);

 JSROOT.addDrawFunc("amore::core::MonitorObjectHisto<TH1F>", Amore_Draw);

})();

*/

