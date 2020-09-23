/// @file JSRootPainter.hist.js
/// JavaScript ROOT graphics for histogram classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3'], factory );
   } else if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"), require("d3"));
   } else {
      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.js', 'JSRootPainter.hist.js');
      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.hist.js');
      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.hist.js');
      factory(JSROOT, d3);
   }
} (function(JSROOT, d3) {

   "use strict";

   JSROOT.sources.push("hist");

   JSROOT.ToolbarIcons.th2color = {
       recs: [{x:0,y:256,w:13,h:39,f:'rgb(38,62,168)'},{x:13,y:371,w:39,h:39},{y:294,h:39},{y:256,h:39},{y:218,h:39},{x:51,y:410,w:39,h:39},{y:371,h:39},{y:333,h:39},{y:294},{y:256,h:39},{y:218,h:39},{y:179,h:39},{y:141,h:39},{y:102,h:39},{y:64},{x:90,y:448,w:39,h:39},{y:410},{y:371,h:39},{y:333,h:39,f:'rgb(22,82,205)'},{y:294},{y:256,h:39,f:'rgb(16,100,220)'},{y:218,h:39},{y:179,h:39,f:'rgb(22,82,205)'},{y:141,h:39},{y:102,h:39,f:'rgb(38,62,168)'},{y:64},{y:0,h:27},{x:128,y:448,w:39,h:39},{y:410},{y:371,h:39},{y:333,h:39,f:'rgb(22,82,205)'},{y:294,f:'rgb(20,129,214)'},{y:256,h:39,f:'rgb(9,157,204)'},{y:218,h:39,f:'rgb(14,143,209)'},{y:179,h:39,f:'rgb(20,129,214)'},{y:141,h:39,f:'rgb(16,100,220)'},{y:102,h:39,f:'rgb(22,82,205)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{y:0,h:27},{x:166,y:486,h:14},{y:448,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39,f:'rgb(20,129,214)'},{y:294,f:'rgb(82,186,146)'},{y:256,h:39,f:'rgb(179,189,101)'},{y:218,h:39,f:'rgb(116,189,129)'},{y:179,h:39,f:'rgb(82,186,146)'},{y:141,h:39,f:'rgb(14,143,209)'},{y:102,h:39,f:'rgb(16,100,220)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:205,y:486,w:39,h:14},{y:448,h:39},{y:410},{y:371,h:39,f:'rgb(16,100,220)'},{y:333,h:39,f:'rgb(9,157,204)'},{y:294,f:'rgb(149,190,113)'},{y:256,h:39,f:'rgb(244,198,59)'},{y:218,h:39},{y:179,h:39,f:'rgb(226,192,75)'},{y:141,h:39,f:'rgb(13,167,195)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(22,82,205)'},{y:26,h:39,f:'rgb(38,62,168)'},{x:243,y:448,w:39,h:39},{y:410},{y:371,h:39,f:'rgb(18,114,217)'},{y:333,h:39,f:'rgb(30,175,179)'},{y:294,f:'rgb(209,187,89)'},{y:256,h:39,f:'rgb(251,230,29)'},{y:218,h:39,f:'rgb(249,249,15)'},{y:179,h:39,f:'rgb(226,192,75)'},{y:141,h:39,f:'rgb(30,175,179)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:282,y:448,h:39},{y:410},{y:371,h:39,f:'rgb(18,114,217)'},{y:333,h:39,f:'rgb(14,143,209)'},{y:294,f:'rgb(149,190,113)'},{y:256,h:39,f:'rgb(226,192,75)'},{y:218,h:39,f:'rgb(244,198,59)'},{y:179,h:39,f:'rgb(149,190,113)'},{y:141,h:39,f:'rgb(9,157,204)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:320,y:448,w:39,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39,f:'rgb(20,129,214)'},{y:294,f:'rgb(46,183,164)'},{y:256,h:39},{y:218,h:39,f:'rgb(82,186,146)'},{y:179,h:39,f:'rgb(9,157,204)'},{y:141,h:39,f:'rgb(20,129,214)'},{y:102,h:39,f:'rgb(16,100,220)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:358,y:448,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39},{y:294,f:'rgb(16,100,220)'},{y:256,h:39,f:'rgb(20,129,214)'},{y:218,h:39,f:'rgb(14,143,209)'},{y:179,h:39,f:'rgb(18,114,217)'},{y:141,h:39,f:'rgb(22,82,205)'},{y:102,h:39,f:'rgb(38,62,168)'},{y:64},{y:26,h:39},{x:397,y:448,w:39,h:39},{y:371,h:39},{y:333,h:39},{y:294,f:'rgb(22,82,205)'},{y:256,h:39},{y:218,h:39},{y:179,h:39,f:'rgb(38,62,168)'},{y:141,h:39},{y:102,h:39},{y:64},{y:26,h:39},{x:435,y:410,h:39},{y:371,h:39},{y:333,h:39},{y:294},{y:256,h:39},{y:218,h:39},{y:179,h:39},{y:141,h:39},{y:102,h:39},{y:64},{x:474,y:256,h:39},{y:179,h:39}]
   };

   JSROOT.ToolbarIcons.th2draw3d = {
      path: "M172.768,0H51.726C23.202,0,0.002,23.194,0.002,51.712v89.918c0,28.512,23.2,51.718,51.724,51.718h121.042   c28.518,0,51.724-23.2,51.724-51.718V51.712C224.486,23.194,201.286,0,172.768,0z M177.512,141.63c0,2.611-2.124,4.745-4.75,4.745   H51.726c-2.626,0-4.751-2.134-4.751-4.745V51.712c0-2.614,2.125-4.739,4.751-4.739h121.042c2.62,0,4.75,2.125,4.75,4.739 L177.512,141.63L177.512,141.63z "+
            "M460.293,0H339.237c-28.521,0-51.721,23.194-51.721,51.712v89.918c0,28.512,23.2,51.718,51.721,51.718h121.045   c28.521,0,51.721-23.2,51.721-51.718V51.712C512.002,23.194,488.802,0,460.293,0z M465.03,141.63c0,2.611-2.122,4.745-4.748,4.745   H339.237c-2.614,0-4.747-2.128-4.747-4.745V51.712c0-2.614,2.133-4.739,4.747-4.739h121.045c2.626,0,4.748,2.125,4.748,4.739 V141.63z "+
            "M172.768,256.149H51.726c-28.524,0-51.724,23.205-51.724,51.726v89.915c0,28.504,23.2,51.715,51.724,51.715h121.042   c28.518,0,51.724-23.199,51.724-51.715v-89.915C224.486,279.354,201.286,256.149,172.768,256.149z M177.512,397.784   c0,2.615-2.124,4.736-4.75,4.736H51.726c-2.626-0.006-4.751-2.121-4.751-4.736v-89.909c0-2.626,2.125-4.753,4.751-4.753h121.042 c2.62,0,4.75,2.116,4.75,4.753L177.512,397.784L177.512,397.784z "+
            "M460.293,256.149H339.237c-28.521,0-51.721,23.199-51.721,51.726v89.915c0,28.504,23.2,51.715,51.721,51.715h121.045   c28.521,0,51.721-23.199,51.721-51.715v-89.915C512.002,279.354,488.802,256.149,460.293,256.149z M465.03,397.784   c0,2.615-2.122,4.736-4.748,4.736H339.237c-2.614,0-4.747-2.121-4.747-4.736v-89.909c0-2.626,2.121-4.753,4.747-4.753h121.045 c2.615,0,4.748,2.116,4.748,4.753V397.784z"
   };

   JSROOT.Painter.CreateDefaultPalette = function() {

      function HLStoRGB(h, l, s) {
         var r, g, b;
         if (s < 1e-100) {
            r = g = b = l; // achromatic
         } else {
            function hue2rgb(p, q, t) {
               if (t < 0) t += 1;
               if (t > 1) t -= 1;
               if (t < 1 / 6) return p + (q - p) * 6 * t;
               if (t < 1 / 2) return q;
               if (t < 2 / 3) return p + (q - p) * (2/3 - t) * 6;
               return p;
            }
            var q = (l < 0.5) ? l * (1 + s) : l + s - l * s,
                p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
         }
         return 'rgb(' + Math.round(r*255) + ',' + Math.round(g*255) + ',' + Math.round(b*255) + ')';
      }

      var palette = [], saturation = 1, lightness = 0.5, maxHue = 280, minHue = 0, maxPretty = 50;
      for (var i = 0; i < maxPretty; ++i) {
         var hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty)) / 360,
             rgbval = HLStoRGB(hue, lightness, saturation);
         palette.push(rgbval);
      }
      return new JSROOT.ColorPalette(palette);
   }

   JSROOT.Painter.CreateGrayPalette = function() {
      var palette = [];
      for (var i = 0; i < 50; ++i) {
         var code = Math.round((i+2)/60*255);
         palette.push('rgb('+code+','+code+','+code+')');
      }
      return new JSROOT.ColorPalette(palette);
   }

   JSROOT.Painter.CreateGradientColorTable = function(Stops, Red, Green, Blue, NColors, alpha) {
      // skip all checks
       var palette = [];

       for (var g = 1; g < Stops.length; g++) {
          // create the colors...
          var nColorsGradient = parseInt(Math.floor(NColors*Stops[g]) - Math.floor(NColors*Stops[g-1]));
          for (var c = 0; c < nColorsGradient; c++) {
             var col = Math.round(Red[g-1] + c * (Red[g] - Red[g-1])/nColorsGradient) + "," +
                       Math.round(Green[g-1] + c * (Green[g] - Green[g-1])/ nColorsGradient) + "," +
                       Math.round(Blue[g-1] + c * (Blue[g] - Blue[g-1])/ nColorsGradient);
             palette.push("rgb("+col+")");
          }
       }

       return new JSROOT.ColorPalette(palette);
   }

   JSROOT.Painter.GetColorPalette = function(col,alfa) {
      col = col || JSROOT.gStyle.Palette;
      if ((col>0) && (col<10)) return JSROOT.Painter.CreateGrayPalette();
      if (col < 51) return JSROOT.Painter.CreateDefaultPalette();
      if (col > 113) col = 57;
      var rgb, stops = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1];
      switch(col) {
         // Deep Sea
         case 51: rgb = [[0,9,13,17,24,32,27,25,29],[0,0,0,2,37,74,113,160,221],[28,42,59,78,98,129,154,184,221]]; break;
         // Grey Scale
         case 52: rgb = [[0,32,64,96,128,160,192,224,255],[0,32,64,96,128,160,192,224,255],[0,32,64,96,128,160,192,224,255]]; break;
         // Dark Body Radiator
         case 53: rgb = [[0,45,99,156,212,230,237,234,242],[0,0,0,45,101,168,238,238,243],[0,1,1,3,9,8,11,95,230]]; break;
         // Two-color hue (dark blue through neutral gray to bright yellow)
         case 54: rgb = [[0,22,44,68,93,124,160,192,237],[0,16,41,67,93,125,162,194,241],[97,100,99,99,93,68,44,26,74]]; break;
         // Rain Bow
         case 55: rgb = [[0,5,15,35,102,196,208,199,110],[0,48,124,192,206,226,97,16,0],[99,142,198,201,90,22,13,8,2]]; break;
         // Inverted Dark Body Radiator
         case 56: rgb = [[242,234,237,230,212,156,99,45,0],[243,238,238,168,101,45,0,0,0],[230,95,11,8,9,3,1,1,0]]; break;
         // Bird (default, keep float for backward compat)
         case 57: rgb = [[ 53.091,15.096,19.89,5.916,45.951,135.1755,208.743,253.878,248.982],[42.432,91.7745,128.5455,163.6845,183.039,191.046,186.864,200.481,250.716],[134.9715,221.442,213.8175,201.807,163.8375,118.881,89.2245,50.184,13.7445]]; break;
         // Cubehelix
         case 58: rgb = [[0,24,2,54,176,236,202,194,255],[0,29,92,129,117,120,176,236,255],[0,68,80,34,57,172,252,245,255]]; break;
         // Green Red Violet
         case 59: rgb = [[13,23,25,63,76,104,137,161,206],[95,67,37,21,0,12,35,52,79],[4,3,2,6,11,22,49,98,208]]; break;
         // Blue Red Yellow
         case 60: rgb = [[0,61,89,122,143,160,185,204,231],[0,0,0,0,14,37,72,132,235],[0,140,224,144,4,5,6,9,13]]; break;
         // Ocean
         case 61: rgb = [[14,7,2,0,5,11,55,131,229],[105,56,26,1,42,74,131,171,229],[2,21,35,60,92,113,160,185,229]]; break;
         // Color Printable On Grey
         case 62: rgb = [[0,0,0,70,148,231,235,237,244],[0,0,0,0,0,69,67,216,244],[0,102,228,231,177,124,137,20,244]]; break;
         // Alpine
         case 63: rgb = [[50,56,63,68,93,121,165,192,241],[66,81,91,96,111,128,155,189,241],[97,91,75,65,77,103,143,167,217]]; break;
         // Aquamarine
         case 64: rgb = [[145,166,167,156,131,114,101,112,132],[158,178,179,181,163,154,144,152,159],[190,199,201,192,176,169,160,166,190]]; break;
         // Army
         case 65: rgb = [[93,91,99,108,130,125,132,155,174],[126,124,128,129,131,121,119,153,173],[103,94,87,85,80,85,107,120,146]]; break;
         // Atlantic
         case 66: rgb = [[24,40,69,90,104,114,120,132,103],[29,52,94,127,150,162,159,151,101],[29,52,96,132,162,181,184,186,131]]; break;
         // Aurora
         case 67: rgb = [[46,38,61,92,113,121,132,150,191],[46,36,40,69,110,135,131,92,34],[46,80,74,70,81,105,165,211,225]]; break;
         // Avocado
         case 68: rgb = [[0,4,12,30,52,101,142,190,237],[0,40,86,121,140,172,187,213,240],[0,9,14,18,21,23,27,35,101]]; break;
         // Beach
         case 69: rgb = [[198,206,206,211,198,181,161,171,244],[103,133,150,172,178,174,163,175,244],[49,54,55,66,91,130,184,224,244]]; break;
         // Black Body
         case 70: rgb = [[243,243,240,240,241,239,186,151,129],[0,46,99,149,194,220,183,166,147],[6,8,36,91,169,235,246,240,233]]; break;
         // Blue Green Yellow
         case 71: rgb = [[22,19,19,25,35,53,88,139,210],[0,32,69,108,135,159,183,198,215],[77,96,110,116,110,100,90,78,70]]; break;
         // Brown Cyan
         case 72: rgb = [[68,116,165,182,189,180,145,111,71],[37,82,135,178,204,225,221,202,147],[16,55,105,147,196,226,232,224,178]]; break;
         // CMYK
         case 73: rgb = [[61,99,136,181,213,225,198,136,24],[149,140,96,83,132,178,190,135,22],[214,203,168,135,110,100,111,113,22]]; break;
         // Candy
         case 74: rgb = [[76,120,156,183,197,180,162,154,140],[34,35,42,69,102,137,164,188,197],[ 64,69,78,105,142,177,205,217,198]]; break;
         // Cherry
         case 75: rgb = [[37,102,157,188,196,214,223,235,251],[37,29,25,37,67,91,132,185,251],[37,32,33,45,66,98,137,187,251]]; break;
         // Coffee
         case 76: rgb = [[79,100,119,137,153,172,192,205,250],[63,79,93,103,115,135,167,196,250],[51,59,66,61,62,70,110,160,250]]; break;
         // Dark Rain Bow
         case 77: rgb = [[43,44,50,66,125,172,178,155,157],[63,63,85,101,138,163,122,51,39],[121,101,58,44,47,55,57,44,43]]; break;
         // Dark Terrain
         case 78: rgb = [[0,41,62,79,90,87,99,140,228],[0,57,81,93,85,70,71,125,228],[95,91,91,82,60,43,44,112,228]]; break;
         // Fall
         case 79: rgb = [[49,59,72,88,114,141,176,205,222],[78,72,66,57,59,75,106,142,173],[ 78,55,46,40,39,39,40,41,47]]; break;
         // Fruit Punch
         case 80: rgb = [[243,222,201,185,165,158,166,187,219],[94,108,132,135,125,96,68,51,61],[7,9,12,19,45,89,118,146,118]]; break;
         // Fuchsia
         case 81: rgb = [[19,44,74,105,137,166,194,206,220],[19,28,40,55,82,110,159,181,220],[19,42,68,96,129,157,188,203,220]]; break;
         // Grey Yellow
         case 82: rgb = [[33,44,70,99,140,165,199,211,216],[ 38,50,76,105,140,165,191,189,167],[ 55,67,97,124,140,166,163,129,52]]; break;
         // Green Brown Terrain
         case 83: rgb = [[0,33,73,124,136,152,159,171,223],[0,43,92,124,134,126,121,144,223],[0,43,68,76,73,64,72,114,223]]; break;
         // Green Pink
         case 84: rgb = [[5,18,45,124,193,223,205,128,49],[48,134,207,230,193,113,28,0,7],[6,15,41,121,193,226,208,130,49]]; break;
         // Island
         case 85: rgb = [[180,106,104,135,164,188,189,165,144],[72,126,154,184,198,207,205,190,179],[41,120,158,188,194,181,145,100,62]]; break;
         // Lake
         case 86: rgb = [[57,72,94,117,136,154,174,192,215],[0,33,68,109,140,171,192,196,209],[116,137,173,201,200,201,203,190,187]]; break;
         // Light Temperature
         case 87: rgb = [[31,71,123,160,210,222,214,199,183],[40,117,171,211,231,220,190,132,65],[234,214,228,222,210,160,105,60,34]]; break;
         // Light Terrain
         case 88: rgb = [[123,108,109,126,154,172,188,196,218],[184,138,130,133,154,175,188,196,218],[208,130,109,99,110,122,150,171,218]]; break;
         // Mint
         case 89: rgb = [[105,106,122,143,159,172,176,181,207],[252,197,194,187,174,162,153,136,125],[146,133,144,155,163,167,166,162,174]]; break;
         // Neon
         case 90: rgb = [[171,141,145,152,154,159,163,158,177],[236,143,100,63,53,55,44,31,6],[59,48,46,44,42,54,82,112,179]]; break;
         // Pastel
         case 91: rgb = [[180,190,209,223,204,228,205,152,91],[93,125,147,172,181,224,233,198,158],[236,218,160,133,114,132,162,220,218]]; break;
         // Pearl
         case 92: rgb = [[225,183,162,135,115,111,119,145,211],[205,177,166,135,124,117,117,132,172],[186,165,155,135,126,130,150,178,226]]; break;
         // Pigeon
         case 93: rgb = [[39,43,59,63,80,116,153,177,223],[39,43,59,74,91,114,139,165,223],[ 39,50,59,70,85,115,151,176,223]]; break;
         // Plum
         case 94: rgb = [[0,38,60,76,84,89,101,128,204],[0,10,15,23,35,57,83,123,199],[0,11,22,40,63,86,97,94,85]]; break;
         // Red Blue
         case 95: rgb = [[94,112,141,165,167,140,91,49,27],[27,46,88,135,166,161,135,97,58],[42,52,81,106,139,158,155,137,116]]; break;
         // Rose
         case 96: rgb = [[30,49,79,117,135,151,146,138,147],[63,60,72,90,94,94,68,46,16],[18,28,41,56,62,63,50,36,21]]; break;
         // Rust
         case 97: rgb = [[0,30,63,101,143,152,169,187,230],[0,14,28,42,58,61,67,74,91],[39,26,21,18,15,14,14,13,13]]; break;
         // Sandy Terrain
         case 98: rgb = [[149,140,164,179,182,181,131,87,61],[62,70,107,136,144,138,117,87,74],[40,38,45,49,49,49,38,32,34]]; break;
         // Sienna
         case 99: rgb = [[99,112,148,165,179,182,183,183,208],[39,40,57,79,104,127,148,161,198],[15,16,18,33,51,79,103,129,177]]; break;
         // Solar
         case 100: rgb = [[99,116,154,174,200,196,201,201,230],[0,0,8,32,58,83,119,136,173],[5,6,7,9,9,14,17,19,24]]; break;
         // South West
         case 101: rgb = [[82,106,126,141,155,163,142,107,66],[ 62,44,69,107,135,152,149,132,119],[39,25,31,60,73,68,49,72,188]]; break;
         // Starry Night
         case 102: rgb = [[18,29,44,72,116,158,184,208,221],[27,46,71,105,146,177,189,190,183],[39,55,80,108,130,133,124,100,76]]; break;
         // Sunset
         case 103: rgb = [[0,48,119,173,212,224,228,228,245],[0,13,30,47,79,127,167,205,245],[0,68,75,43,16,22,55,128,245]]; break;
         // Temperature Map
         case 104: rgb = [[34,70,129,187,225,226,216,193,179],[48,91,147,194,226,229,196,110,12],[234,212,216,224,206,110,53,40,29]]; break;
         // Thermometer
         case 105: rgb = [[30,55,103,147,174,203,188,151,105],[0,65,138,182,187,175,121,53,9],[191,202,212,208,171,140,97,57,30]]; break;
         // Valentine
         case 106: rgb = [[112,97,113,125,138,159,178,188,225],[16,17,24,37,56,81,110,136,189],[38,35,46,59,78,103,130,152,201]]; break;
         // Visible Spectrum
         case 107: rgb = [[18,72,5,23,29,201,200,98,29],[0,0,43,167,211,117,0,0,0],[51,203,177,26,10,9,8,3,0]]; break;
         // Water Melon
         case 108: rgb = [[19,42,64,88,118,147,175,187,205],[19,55,89,125,154,169,161,129,70],[19,32,47,70,100,128,145,130,75]]; break;
         // Cool
         case 109: rgb = [[33,31,42,68,86,111,141,172,227],[255,175,145,106,88,55,15,0,0],[255,205,202,203,208,205,203,206,231]]; break;
         // Copper
         case 110: rgb = [[0,25,50,79,110,145,181,201,254],[0,16,30,46,63,82,101,124,179],[0,12,21,29,39,49,61,74,103]]; break;
         // Gist Earth
         case 111: rgb = [[0,13,30,44,72,120,156,200,247],[0,36,84,117,141,153,151,158,247],[0,94,100,82,56,66,76,131,247]]; break;
         // Viridis
         case 112: rgb = [[26,51,43,33,28,35,74,144,246],[9,24,55,87,118,150,180,200,222],[30,96,112,114,112,101,72,35,0]]; break;
         // Cividis
         case 113: rgb = [[0,5,65,97,124,156,189,224,255],[32,54,77,100,123,148,175,203,234],[77,110,107,111,120,119,111,94,70]]; break;
         default: return JSROOT.Painter.CreateDefaultPalette();
      }

      return JSROOT.Painter.CreateGradientColorTable(stops, rgb[0], rgb[1], rgb[2], 255, alfa);
   }

   // ============================================================

   // painter class for objects, derived from TPave
   function TPavePainter(pave) {
      JSROOT.TObjectPainter.call(this, pave);
      this.Enabled = true;
      this.UseContextMenu = true;
      this.UseTextColor = false; // indicates if text color used, enabled menu entry
      this.FirstRun = 1; // counter required to correctly complete drawing
      this.AssignFinishPave();
   }

   TPavePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TPavePainter.prototype.AssignFinishPave = function() {
      function func() {
         // function used to signal drawing ready, required when text drawing postponed due to mathjax
         if (this.FirstRun <= 0) return;
         this.FirstRun--;
         if (this.FirstRun!==0) return;
         delete this.FinishPave; // no need for that callback
         this.DrawingReady();
      }
      this.FinishPave = func.bind(this);
   }

   TPavePainter.prototype.DrawPave = function(arg) {
      // this draw only basic TPave

      this.UseTextColor = false;

      if (!this.Enabled)
         return this.RemoveDrawG();

      var pt = this.GetObject(), opt = pt.fOption.toUpperCase();

      if (pt.fInit===0) {
         this.stored = JSROOT.extend({}, pt); // store coordinates to use them when updating
         pt.fInit = 1;
         var pad = this.root_pad();

         if ((pt._typename == "TPaletteAxis") && !pt.fX1 && !pt.fX2 && !pt.fY1 && !pt.fY2) {
            var fp = this.frame_painter();
            if (fp) {
               pt.fX1NDC = fp.fX2NDC + 0.01;
               pt.fX2NDC = Math.min(0.96, fp.fX2NDC + 0.06);
               pt.fY1NDC = fp.fY1NDC;
               pt.fY2NDC = fp.fY2NDC;
            } else {
               pt.fX2NDC = 0.8;
               pt.fX1NDC = 0.9;
               pt.fY1NDC = 0.1;
               pt.fY2NDC = 0.9;
            }
         } else if (opt.indexOf("NDC")>=0) {
            pt.fX1NDC = pt.fX1; pt.fX2NDC = pt.fX2;
            pt.fY1NDC = pt.fY1; pt.fY2NDC = pt.fY2;
         } else if (pad) {
            if (pad.fLogx) {
               if (pt.fX1 > 0) pt.fX1 = JSROOT.log10(pt.fX1);
               if (pt.fX2 > 0) pt.fX2 = JSROOT.log10(pt.fX2);
            }
            if (pad.fLogy) {
               if (pt.fY1 > 0) pt.fY1 = JSROOT.log10(pt.fY1);
               if (pt.fY2 > 0) pt.fY2 = JSROOT.log10(pt.fY2);
            }
            pt.fX1NDC = (pt.fX1 - pad.fX1) / (pad.fX2 - pad.fX1);
            pt.fY1NDC = (pt.fY1 - pad.fY1) / (pad.fY2 - pad.fY1);
            pt.fX2NDC = (pt.fX2 - pad.fX1) / (pad.fX2 - pad.fX1);
            pt.fY2NDC = (pt.fY2 - pad.fY1) / (pad.fY2 - pad.fY1);
         } else {
            pt.fX1NDC = pt.fY1NDC = 0.1;
            pt.fX2NDC = pt.fY2NDC = 0.9;
         }

         if ((pt.fX1NDC == pt.fX2NDC) && (pt.fY1NDC == pt.fY2NDC) && (pt._typename == "TLegend")) {
            pt.fX1NDC = Math.max(pad ? pad.fLeftMargin : 0, pt.fX2NDC - 0.3);
            pt.fX2NDC = Math.min(pt.fX1NDC + 0.3, pad ? 1-pad.fRightMargin : 1);
            var h0 = Math.max(pt.fPrimitives ? pt.fPrimitives.arr.length*0.05 : 0, 0.2);
            pt.fY2NDC = Math.min(pad ? 1-pad.fTopMargin : 1, pt.fY1NDC + h0);
            pt.fY1NDC = Math.max(pt.fY2NDC - h0, pad ? pad.fBottomMargin : 0);
         }
      }

      var pos_x = Math.round(pt.fX1NDC * this.pad_width()),
          pos_y = Math.round((1.0 - pt.fY2NDC) * this.pad_height()),
          width = Math.round((pt.fX2NDC - pt.fX1NDC) * this.pad_width()),
          height = Math.round((pt.fY2NDC - pt.fY1NDC) * this.pad_height()),
          brd = pt.fBorderSize,
          dx = (opt.indexOf("L")>=0) ? -1 : ((opt.indexOf("R")>=0) ? 1 : 0),
          dy = (opt.indexOf("T")>=0) ? -1 : ((opt.indexOf("B")>=0) ? 1 : 0);

      // container used to recalculate coordinates
      this.CreateG();

      this.draw_g.attr("transform", "translate(" + pos_x + "," + pos_y + ")");

      //if (!this.lineatt)
      //   this.lineatt = new JSROOT.TAttLineHandler(pt, brd>0 ? 1 : 0);

      this.createAttLine({ attr: pt, width: (brd > 0) ? pt.fLineWidth : 0 });

      this.createAttFill({ attr: pt });

      if (pt._typename == "TDiamond") {
         var h2 = Math.round(height/2), w2 = Math.round(width/2),
             dpath = "l"+w2+",-"+h2 + "l"+w2+","+h2 + "l-"+w2+","+h2+"z";

         if ((brd > 1) && (pt.fShadowColor > 0) && (dx || dy) && !this.fillatt.empty())
            this.draw_g.append("svg:path")
                 .attr("d","M0,"+(h2+brd) + dpath)
                 .style("fill", this.get_color(pt.fShadowColor))
                 .style("stroke", this.get_color(pt.fShadowColor))
                 .style("stroke-width", "1px");

         this.draw_g.append("svg:path")
             .attr("d", "M0,"+h2 +dpath)
             .call(this.fillatt.func)
             .call(this.lineatt.func);

         var text_g = this.draw_g.append("svg:g")
                                 .attr("transform", "translate(" + Math.round(width/4) + "," + Math.round(height/4) + ")");

         this.DrawPaveText(w2, h2, arg, text_g);

         return;
      }

      // add shadow decoration before main rect
      if ((brd > 1) && (pt.fShadowColor > 0) && !pt.fNpaves && (dx || dy)) {
         var spath = "", scol = this.get_color(pt.fShadowColor);
         if (this.fillatt.empty()) {
            if ((dx<0) && (dy<0))
               spath = "M0,0v"+(height-brd)+"h-"+brd+"v-"+height+"h"+width+"v"+brd;
            else // ((dx<0) && (dy>0))
               spath = "M0,"+height+"v-"+(height-brd)+"h-"+brd+"v"+height+"h"+width+"v-"+brd;
         } else {
            // when main is filled, one also can use fill for shadow to avoid complexity
            spath = "M"+(dx*brd)+","+(dy*brd) + "v"+height + "h"+width + "v-"+height
         }
         this.draw_g.append("svg:path")
                    .attr("d", spath + "z")
                    .style("fill", scol)
                    .style("stroke", scol)
                    .style("stroke-width", "1px");
      }

      if (pt.fNpaves)
         for (var n = pt.fNpaves-1; n>0; --n)
            this.draw_g.append("svg:path")
               .attr("d", "M" + (dx*4*n) + ","+ (dy*4*n) + "h"+width + "v"+height + "h-"+width + "z")
               .call(this.fillatt.func)
               .call(this.lineatt.func);

      var rect =
         this.draw_g.append("svg:path")
          .attr("d", "M0,0h"+width + "v"+height + "h-"+width + "z")
          .call(this.fillatt.func)
          .call(this.lineatt.func);

      if ('PaveDrawFunc' in this)
         this.PaveDrawFunc(width, height, arg);

      if (JSROOT.BatchMode || (pt._typename=="TPave")) return;

      // here all kind of interactive settings

      rect.style("pointer-events", "visibleFill")
          .on("mouseenter", this.ShowObjectStatus.bind(this))

      // position and size required only for drag functions
      this.draw_g.attr("x", pos_x)
                 .attr("y", pos_y)
                 .attr("width", width)
                 .attr("height", height);

      this.AddDrag({ obj: pt, minwidth: 10, minheight: 20, canselect: true,
                     redraw: this.DragRedraw.bind(this),
                     ctxmenu: JSROOT.touches && JSROOT.gStyle.ContextMenu && this.UseContextMenu });

      if (this.UseContextMenu && JSROOT.gStyle.ContextMenu)
         this.draw_g.on("contextmenu", this.ShowContextMenu.bind(this));
   }

   TPavePainter.prototype.DragRedraw = function() {
      this.InteractiveRedraw(false, "pave_moved");
      this.DrawPave();
   }

   TPavePainter.prototype.FillWebObjectOptions = function(res) {
      if (!res) {
         if (!this.snapid) return null;
         res = { _typename: "TWebObjectOptions", snapid: this.snapid.toString(), opt: this.OptionsAsString(), fcust: "", fopt: [] };
      }

      var pave = this.GetObject();

      if (pave && pave.fInit) {
         res.fcust = "pave";
         res.fopt = [pave.fX1NDC,pave.fY1NDC,pave.fX2NDC,pave.fY2NDC];
      }

      return res;
   }

   TPavePainter.prototype.DrawPaveLabel = function(_width, _height) {
      this.UseTextColor = true;

      var pave = this.GetObject();

      this.StartTextDrawing(pave.fTextFont, _height/1.2);

      this.DrawText({ align: pave.fTextAlign, width: _width, height: _height, text: pave.fLabel, color: this.get_color(pave.fTextColor) });

      this.FinishTextDrawing(null, this.FinishPave);
   }

   TPavePainter.prototype.DrawPaveStats = function(width, height) {

      if (this.IsStats()) this.FillStatistic();

      var pt = this.GetObject(), lines = [],
          tcolor = this.get_color(pt.fTextColor),
          first_stat = 0, num_cols = 0, maxlen = 0;

      // now draw TLine and TBox objects
      for (var j=0;j<pt.fLines.arr.length;++j) {
         var entry = pt.fLines.arr[j];
         if ((entry._typename=="TText") || (entry._typename=="TLatex"))
            lines.push(entry.fTitle);
      }

      var nlines = lines.length;

      // adjust font size
      for (var j = 0; j < nlines; ++j) {
         var line = lines[j];
         if (j>0) maxlen = Math.max(maxlen, line.length);
         if ((j == 0) || (line.indexOf('|') < 0)) continue;
         if (first_stat === 0) first_stat = j;
         var parts = line.split("|");
         if (parts.length > num_cols)
            num_cols = parts.length;
      }

      // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
      var stepy = height / nlines, has_head = false, margin_x = pt.fMargin * width;

      this.StartTextDrawing(pt.fTextFont, height/(nlines * 1.2));

      this.UseTextColor = true;

      if (nlines == 1) {
         this.DrawText({ align: pt.fTextAlign, width: width, height: height, text: lines[0], color: tcolor, latex: 1 });
      } else
      for (var j = 0; j < nlines; ++j) {
         var posy = j*stepy;
         this.UseTextColor = true;

         if (first_stat && (j >= first_stat)) {
            var parts = lines[j].split("|");
            for (var n = 0; n < parts.length; ++n)
               this.DrawText({ align: "middle", x: width * n / num_cols, y: posy, latex: 0,
                               width: width/num_cols, height: stepy, text: parts[n], color: tcolor });
         } else if (lines[j].indexOf('=') < 0) {
            if (j==0) {
               has_head = true;
               if (lines[j].length > maxlen + 5)
                  lines[j] = lines[j].substr(0,maxlen+2) + "...";
            }
            this.DrawText({ align: (j == 0) ? "middle" : "start", x: margin_x, y: posy,
                            width: width-2*margin_x, height: stepy, text: lines[j], color: tcolor });
         } else {
            var parts = lines[j].split("="), sumw = 0;
            for (var n = 0; n < 2; ++n)
               sumw += this.DrawText({ align: (n == 0) ? "start" : "end", x: margin_x, y: posy,
                                       width: width-2*margin_x, height: stepy, text: parts[n], color: tcolor });
            this.TextScaleFactor(1.05*sumw/(width-2*margin_x), this.draw_g);
         }
      }

      var lpath = "";

      if ((pt.fBorderSize > 0) && has_head)
         lpath += "M0," + Math.round(stepy) + "h" + width;

      if ((first_stat > 0) && (num_cols > 1)) {
         for (var nrow = first_stat; nrow < nlines; ++nrow)
            lpath += "M0," + Math.round(nrow * stepy) + "h" + width;
         for (var ncol = 0; ncol < num_cols - 1; ++ncol)
            lpath += "M" + Math.round(width / num_cols * (ncol + 1)) + "," + Math.round(first_stat * stepy) + "V" + height;
      }

      if (lpath) this.draw_g.append("svg:path").attr("d",lpath).call(this.lineatt.func);

      this.FinishTextDrawing(undefined, this.FinishPave);

      this.draw_g.classed("most_upper_primitives", true); // this primitive will remain on top of list
   }

   TPavePainter.prototype.DrawPaveText = function(width, height, dummy_arg, text_g) {

      var pt = this.GetObject(),
          tcolor = this.get_color(pt.fTextColor),
          nlines = 0, lines = [],
          can_height = this.pad_height(),
          pp = this.pad_painter(),
          individual_positioning = false,
          draw_header = (pt.fLabel.length>0);

      if (draw_header) this.FirstRun++; // increment finish counter

      if (!text_g) text_g = this.draw_g;

      // first check how many text lines in the list
      for (var j=0;j<pt.fLines.arr.length;++j) {
         var entry = pt.fLines.arr[j];
         if ((entry._typename=="TText") || (entry._typename=="TLatex")) {
            nlines++; // count lines
            if ((entry.fX>0) || (entry.fY>0)) individual_positioning = true;
         }
      }

      var fast_draw = (nlines==1) && pp && pp._fast_drawing, nline = 0;

      // now draw TLine and TBox objects
      for (var j=0;j<pt.fLines.arr.length;++j) {
         var entry = pt.fLines.arr[j],
             ytext = (nlines>0) ? Math.round((1-(nline-0.5)/nlines)*height) : 0;
         switch (entry._typename) {
            case "TText":
            case "TLatex":
               nline++; // just count line number
               if (individual_positioning) {
                  // each line should be drawn and scaled separately

                  var lx = entry.fX, ly = entry.fY;

                  if ((lx>0) && (lx<1)) lx = Math.round(lx*width); else lx = pt.fMargin * width;
                  if ((ly>0) && (ly<1)) ly = Math.round((1-ly)*height); else ly = ytext;

                  var jcolor = entry.fTextColor ? this.get_color(entry.fTextColor) : "";
                  if (!jcolor) {
                     jcolor = tcolor;
                     this.UseTextColor = true;
                  }

                  this.StartTextDrawing(pt.fTextFont, (entry.fTextSize || pt.fTextSize) * can_height, text_g);

                  this.DrawText({ align: entry.fTextAlign || pt.fTextAlign, x: lx, y: ly, text: entry.fTitle, color: jcolor,
                                  latex: (entry._typename == "TText") ? 0 : 1,  draw_g: text_g, fast: fast_draw });

                  this.FinishTextDrawing(text_g, this.FinishPave);

                  this.FirstRun++;

               } else {
                  lines.push(entry); // make as before
               }
               break;
            case "TLine":
            case "TBox":
               var lx1 = entry.fX1, lx2 = entry.fX2,
                   ly1 = entry.fY1, ly2 = entry.fY2;
               if (lx1!==0) lx1 = Math.round(lx1*width);
               lx2 = lx2 ? Math.round(lx2*width) : width;
               ly1 = ly1 ? Math.round((1-ly1)*height) : ytext;
               ly2 = ly2 ? Math.round((1-ly2)*height) : ytext;
               if (entry._typename == "TLine") {
                  var lineatt = new JSROOT.TAttLineHandler(entry);
                  text_g.append("svg:line")
                        .attr("x1", lx1)
                        .attr("y1", ly1)
                        .attr("x2", lx2)
                        .attr("y2", ly2)
                        .call(lineatt.func);
               } else {
                  var fillatt = this.createAttFill(entry);

                  text_g.append("svg:rect")
                      .attr("x", lx1)
                      .attr("y", ly2)
                      .attr("width", lx2-lx1)
                      .attr("height", ly1-ly2)
                      .call(fillatt.func);
               }
               break;
         }
      }

      if (individual_positioning) {

         // we should call FinishPave
         if (this.FinishPave) this.FinishPave();

      } else {

         // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
         var stepy = height / nlines, has_head = false, margin_x = pt.fMargin * width, max_font_size = 0;

         // for single line (typically title) limit font size
         if ((nlines == 1) && (pt.fTextSize > 0)) {
            max_font_size = Math.round(pt.fTextSize*can_height);
            if (max_font_size < 3) max_font_size = 3;
         }

         this.StartTextDrawing(pt.fTextFont, height/(nlines * 1.2), text_g, max_font_size);

         for (var j = 0; j < nlines; ++j) {
            var arg = null, lj = lines[j];

            if (nlines == 1) {
               arg = { x:0, y:0, width: width, height: height };
            } else {
               arg = { x: margin_x, y: j*stepy, width: width-2*margin_x, height: stepy };
               if (lj.fTextColor) arg.color = this.get_color(lj.fTextColor);
               if (lj.fTextSize) arg.font_size = Math.round(lj.fTextSize*can_height);
            }

            arg.align = pt.fTextAlign;
            arg.draw_g = text_g;
            arg.latex = (lj._typename == "TText" ? 0 : 1);
            arg.text = lj.fTitle;
            arg.fast = fast_draw;
            if (!arg.color) { this.UseTextColor = true; arg.color = tcolor; }

            this.DrawText(arg);
         }

         this.FinishTextDrawing(text_g, this.FinishPave);
      }

      if (draw_header) {
         var x = Math.round(width*0.25),
             y = Math.round(-height*0.02),
             w = Math.round(width*0.5),
             h = Math.round(height*0.04),
             lbl_g = text_g.append("svg:g");

         lbl_g.append("svg:path")
               .attr("d", "M"+x+","+y + "h"+w + "v"+h + "h-"+w + "z")
               .call(this.fillatt.func)
               .call(this.lineatt.func);

         this.StartTextDrawing(pt.fTextFont, h/1.5, lbl_g);

         this.DrawText({ align: 22, x: x, y: y, width: w, height: h, text: pt.fLabel, color: tcolor, draw_g: lbl_g });

         this.FinishTextDrawing(lbl_g, this.FinishPave);

         this.UseTextColor = true;
      }
   }

   /** Method used to convert value to string according specified format
     * format can be like 5.4g or 4.2e or 6.4f
     * @private */
   TPavePainter.prototype.Format = function(value, fmt) {
      if (!fmt) fmt = "stat";

      var pave = this.GetObject();

      switch(fmt) {
         case "stat" : fmt = pave.fStatFormat || JSROOT.gStyle.fStatFormat; break;
         case "fit": fmt = pave.fFitFormat || JSROOT.gStyle.fFitFormat; break;
         case "entries": if ((Math.abs(value) < 1e9) && (Math.round(value) == value)) return value.toFixed(0); fmt = "14.7g"; break;
         case "last": fmt = this.lastformat; break;
      }

      delete this.lastformat;

      var res = JSROOT.FFormat(value, fmt || "6.4g");

      this.lastformat = JSROOT.lastFFormat;

      return res;
   }

   TPavePainter.prototype.DrawPaveLegend = function(w, h) {

      var legend = this.GetObject(),
          nlines = legend.fPrimitives.arr.length,
          ncols = legend.fNColumns,
          nrows = nlines;

      if (ncols<2) ncols = 1; else { while ((nrows-1)*ncols >= nlines) nrows--; }

      function isEmpty(entry) {
         return !entry.fObject && !entry.fOption && (!entry.fLabel || (entry.fLabel == " "));
      }

      if (ncols==1) {
         for (var i=0;i<nlines;++i)
            if (isEmpty(legend.fPrimitives.arr[i])) nrows--;
      }

      if (nrows<1) nrows = 1;

      var tcolor = this.get_color(legend.fTextColor),
          column_width = Math.round(w/ncols),
          padding_x = Math.round(0.03*w/ncols),
          padding_y = Math.round(0.03*h),
          step_y = (h - 2*padding_y)/nrows,
          font_size = 0.9*step_y,
          max_font_size = 0, // not limited in the beggining
          ph = this.pad_height(),
          fsize, any_opt = false, i = -1;

      if (legend.fTextSize && (ph*legend.fTextSize > 2) && (ph*legend.fTextSize < font_size))
         font_size = max_font_size = Math.round(ph*legend.fTextSize);

      this.StartTextDrawing(legend.fTextFont, font_size, this.draw_g, max_font_size);

      for (var ii = 0; ii < nlines; ++ii) {
         var leg = legend.fPrimitives.arr[ii];

         if (isEmpty(leg)) continue; // let discard empty entry

         if (ncols==1) ++i; else i = ii;

         var lopt = leg.fOption.toLowerCase(),
             icol = i % ncols, irow = (i - icol) / ncols,
             x0 = icol * column_width,
             tpos_x = x0 + Math.round(legend.fMargin*column_width),
             mid_x = Math.round((x0 + tpos_x)/2),
             pos_y = Math.round(irow*step_y + padding_y), // top corner
             mid_y = Math.round((irow+0.5)*step_y + padding_y), // center line
             o_fill = leg, o_marker = leg, o_line = leg,
             mo = leg.fObject,
             painter = null, isany = false;

         if ((mo !== null) && (typeof mo == 'object')) {
            if ('fLineColor' in mo) o_line = mo;
            if ('fFillColor' in mo) o_fill = mo;
            if ('fMarkerColor' in mo) o_marker = mo;

            painter = this.FindPainterFor(mo);
         }

         // Draw fill pattern (in a box)
         if (lopt.indexOf('f') != -1) {
            var fillatt = (painter && painter.fillatt) ? painter.fillatt : this.createAttFill(o_fill);

            // box total height is yspace*0.7
            // define x,y as the center of the symbol for this entry
            var rect = this.draw_g.append("svg:rect")
                           .attr("x", x0 + padding_x)
                           .attr("y", Math.round(pos_y+step_y*0.1))
                           .attr("width", tpos_x - 2*padding_x - x0)
                           .attr("height", Math.round(step_y*0.8))
                           .call(fillatt.func);
            if (!fillatt.empty()) isany = true;
            if ((lopt.indexOf('l') < 0 && lopt.indexOf('e') < 0) && (lopt.indexOf('p') < 0)) {
               var lineatt = (painter && painter.lineatt) ? painter.lineatt : new JSROOT.TAttLineHandler(o_line);
               rect.call(lineatt.func);
               if (lineatt.color !== 'none') isany = true;
            }
         }

         // Draw line (also when error specified)
         if (lopt.indexOf('l') != -1 || lopt.indexOf('e') != -1) {
            var lineatt = (painter && painter.lineatt) ? painter.lineatt : new JSROOT.TAttLineHandler(o_line);
            this.draw_g.append("svg:line")
               .attr("x1", x0 + padding_x)
               .attr("y1", mid_y)
               .attr("x2", tpos_x - padding_x)
               .attr("y2", mid_y)
               .call(lineatt.func);
            if (lineatt.color !== 'none') isany = true;
         }

         // Draw error
         if (lopt.indexOf('e') != -1) {
            var lineatt = (painter && painter.lineatt) ? painter.lineatt : new JSROOT.TAttLineHandler(o_line);
            this.draw_g.append("svg:line")
                .attr("x1", mid_x)
                .attr("y1", Math.round(pos_y+step_y*0.1))
                .attr("x2", mid_x)
                .attr("y2", Math.round(pos_y+step_y*0.9))
                .call(lineatt.func);
            if (lineatt.color !== 'none') isany = true;
         }

         // Draw Polymarker
         if (lopt.indexOf('p') != -1) {
            var marker = (painter && painter.markeratt) ? painter.markeratt : new JSROOT.TAttMarkerHandler(o_marker);
            this.draw_g
                .append("svg:path")
                .attr("d", marker.create((x0 + tpos_x)/2, mid_y))
                .call(marker.func);
            if (marker.color !== 'none') isany = true;
         }

         // special case - nothing draw, try to show rect with line attributes
         if (!isany && painter && painter.lineatt && (painter.lineatt.color !== 'none'))
            this.draw_g.append("svg:rect")
                       .attr("x", x0 + padding_x)
                       .attr("y", Math.round(pos_y+step_y*0.1))
                       .attr("width", tpos_x - 2*padding_x - x0)
                       .attr("height", Math.round(step_y*0.8))
                       .attr("fill", "none")
                       .call(painter.lineatt.func);

         var pos_x = tpos_x;
         if (lopt.length>0) any_opt = true;
                       else if (!any_opt) pos_x = x0 + padding_x;

         if (leg.fLabel)
            this.DrawText({ align: "start", x: pos_x, y: pos_y, width: x0+column_width-pos_x-padding_x, height: step_y, text: leg.fLabel, color: tcolor });
      }

      // rescale after all entries are shown
      this.FinishTextDrawing(this.draw_g, this.FinishPave);
   }

   TPavePainter.prototype.DrawPaletteAxis = function(s_width, s_height, arg) {

      var pthis = this,
          palette = this.GetObject(),
          axis = palette.fAxis,
          can_move = (typeof arg == "string") && (arg.indexOf('can_move') > 0),
          postpone_draw = (typeof arg == "string") && (arg.indexOf('postpone') > 0),
          nbr1 = axis.fNdiv % 100,
          pos_x = parseInt(this.draw_g.attr("x")), // pave position
          pos_y = parseInt(this.draw_g.attr("y")),
          width = this.pad_width(),
          height = this.pad_height(),
          axisOffset = axis.fLabelOffset * width,
          main = this.main_painter(),
          framep = this.frame_painter(),
          zmin = 0, zmax = 100,
          contour = main.fContour;

      if (nbr1<=0) nbr1 = 8;
      axis.fTickSize = 0.6 * s_width / width; // adjust axis ticks size

      if (contour && framep) {
         zmin = Math.min(contour[0], framep.zmin);
         zmax = Math.max(contour[contour.length-1], framep.zmax);
      } else if ((main.gmaxbin!==undefined) && (main.gminbin!==undefined)) {
         // this is case of TH2 (needs only for size adjustment)
         zmin = main.gminbin; zmax = main.gmaxbin;
      } else if ((main.hmin!==undefined) && (main.hmax!==undefined)) {
         // this is case of TH1
         zmin = main.hmin; zmax = main.hmax;
      }

      var z = null, z_kind = "normal";

      if (this.root_pad().fLogz) {
         z = d3.scaleLog();
         z_kind = "log";
      } else {
         z = d3.scaleLinear();
      }
      z.domain([zmin, zmax]).range([s_height,0]);

      this.draw_g.selectAll("rect").style("fill", 'white');

      if (!contour || postpone_draw)
         // we need such rect to correctly calculate size
         this.draw_g.append("svg:rect")
                    .attr("x", 0)
                    .attr("y",  0)
                    .attr("width", s_width)
                    .attr("height", s_height)
                    .style("fill", 'white');
      else
         for (var i=0;i<contour.length-1;++i) {
            var z0 = z(contour[i]),
                z1 = z(contour[i+1]),
                col = main.getContourColor((contour[i]+contour[i+1])/2);

            var r = this.draw_g.append("svg:rect")
                       .attr("x", 0)
                       .attr("y",  Math.round(z1))
                       .attr("width", s_width)
                       .attr("height", Math.round(z0) - Math.round(z1))
                       .style("fill", col)
                       .style("stroke", col)
                       .property("fill0", col)
                       .property("fill1", d3.rgb(col).darker(0.5).toString())

            if (this.IsTooltipAllowed())
               r.on('mouseover', function() {
                  d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill1'));
               }).on('mouseout', function() {
                  d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill0'));
               }).append("svg:title").text(contour[i].toFixed(2) + " - " + contour[i+1].toFixed(2));

            if (JSROOT.gStyle.Zooming)
               r.on("dblclick", function() { pthis.frame_painter().Unzoom("z"); });
         }


      this.z_handle.SetAxisConfig("zaxis", z_kind, z, zmin, zmax, zmin, zmax);

      this.z_handle.max_tick_size = Math.round(s_width*0.7);

      this.z_handle.DrawAxis(true, this.draw_g, s_width, s_height, "translate(" + s_width + ", 0)");

      if (can_move && ('getBoundingClientRect' in this.draw_g.node())) {
         var rect = this.draw_g.node().getBoundingClientRect();

         var shift = (pos_x + parseInt(rect.width)) - Math.round(0.995*width) + 3;

         if (shift > 0) {
            this.draw_g.attr("x", pos_x - shift).attr("y", pos_y)
                       .attr("transform", "translate(" + (pos_x-shift) + ", " + pos_y + ")");
            palette.fX1NDC -= shift/width;
            palette.fX2NDC -= shift/width;
         }
      }

      var evnt = null, doing_zoom = false, sel1 = 0, sel2 = 0, zoom_rect = null;

      function moveRectSel() {

         if (!doing_zoom) return;

         d3.event.preventDefault();
         var m = d3.mouse(evnt);

         if (m[1] < sel1) sel1 = m[1]; else sel2 = m[1];

         zoom_rect.attr("y", sel1)
                  .attr("height", Math.abs(sel2-sel1));
      }

      function endRectSel() {
         if (!doing_zoom) return;

         d3.event.preventDefault();
         d3.select(window).on("mousemove.colzoomRect", null)
                          .on("mouseup.colzoomRect", null);
         zoom_rect.remove();
         zoom_rect = null;
         doing_zoom = false;

         var zmin = Math.min(z.invert(sel1), z.invert(sel2)),
             zmax = Math.max(z.invert(sel1), z.invert(sel2));

         pthis.frame_painter().Zoom("z", zmin, zmax);
      }

      function startRectSel() {
         // ignore when touch selection is activated
         if (doing_zoom) return;
         doing_zoom = true;

         d3.event.preventDefault();

         evnt = this;
         var origin = d3.mouse(evnt);

         sel1 = sel2 = origin[1];

         zoom_rect = pthis.draw_g
                .append("svg:rect")
                .attr("class", "zoom")
                .attr("id", "colzoomRect")
                .attr("x", "0")
                .attr("width", s_width)
                .attr("y", sel1)
                .attr("height", 5);

         d3.select(window).on("mousemove.colzoomRect", moveRectSel)
                          .on("mouseup.colzoomRect", endRectSel, true);

         d3.event.stopPropagation();
      }

      if (JSROOT.gStyle.Zooming)
         this.draw_g.select(".axis_zoom")
                    .on("mousedown", startRectSel)
                    .on("dblclick", function() { pthis.frame_painter().Unzoom("z"); });

      if (this.FinishPave) this.FinishPave();
   }

   TPavePainter.prototype.FillContextMenu = function(menu) {
      var pave = this.GetObject();

      menu.add("header: " + pave._typename + "::" + pave.fName);
      if (this.IsStats()) {
         menu.add("Default position", function() {
            pave.fX2NDC = JSROOT.gStyle.fStatX;
            pave.fX1NDC = pave.fX2NDC - JSROOT.gStyle.fStatW;
            pave.fY2NDC = JSROOT.gStyle.fStatY;
            pave.fY1NDC = pave.fY2NDC - JSROOT.gStyle.fStatH;
            pave.fInit = 1;
            this.InteractiveRedraw(true, "pave_moved")
         });

         menu.add("SetStatFormat", function() {
            var fmt = prompt("Enter StatFormat", pave.fStatFormat);
            if (fmt) {
               pave.fStatFormat = fmt;
               this.InteractiveRedraw(true, 'exec:SetStatFormat("'+fmt+'")');
            }
         });
         menu.add("SetFitFormat", function() {
            var fmt = prompt("Enter FitFormat", pave.fFitFormat);
            if (fmt) {
               pave.fFitFormat = fmt;
               this.InteractiveRedraw(true, 'exec:SetFitFormat("'+fmt+'")');
            }
         });
         menu.add("separator");
         menu.add("sub:SetOptStat", function() {
            // todo - use jqury dialog here
            var fmt = prompt("Enter OptStat", pave.fOptStat);
            if (fmt) {
               fmt = parseInt(fmt);
               if (!isNaN(fmt) && (fmt>=0)) {
                  pave.fOptStat = parseInt(fmt);
                  this.InteractiveRedraw(true, "exec:SetOptStat("+fmt+")");
               }
            }
         });
         function AddStatOpt(pos, name) {
            var opt = (pos<10) ? pave.fOptStat : pave.fOptFit;
            opt = parseInt(parseInt(opt) / parseInt(Math.pow(10,pos % 10))) % 10;
            menu.addchk(opt, name, opt * 100 + pos, function(arg) {
               var newopt = (arg % 100 < 10) ? pave.fOptStat : pave.fOptFit;
               var oldopt = parseInt(arg / 100);
               newopt -= (oldopt>0 ? oldopt : -1) * parseInt(Math.pow(10, arg % 10));
               if (arg % 100 < 10) {
                  pave.fOptStat = newopt;
                  this.InteractiveRedraw(true, "exec:SetOptStat("+newopt+")");
               } else {
                  pave.fOptFit = newopt;
                  this.InteractiveRedraw(true, "exec:SetOptFit("+newopt+")");
               }
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
            if (fmt) {
               fmt = parseInt(fmt);
               if (!isNaN(fmt) && (fmt>=0)) {
                  pave.fOptFit = fmt;
                  this.InteractiveRedraw(true, "exec:SetOptFit("+fmt+")");
               }
            }
         });
         AddStatOpt(10, "Fit parameters");
         AddStatOpt(11, "Par errors");
         AddStatOpt(12, "Chi square / NDF");
         AddStatOpt(13, "Probability");
         menu.add("endsub:");

         menu.add("separator");
      } else if (pave.fName === "title")
         menu.add("Default position", function() {
            pave.fX1NDC = 0.28;
            pave.fY1NDC = 0.94;
            pave.fX2NDC = 0.72;
            pave.fY2NDC = 0.99;
            pave.fInit = 1;
            this.InteractiveRedraw(true, "pave_moved");
         });

      if (this.UseTextColor)
         this.TextAttContextMenu(menu);

      this.FillAttContextMenu(menu);

      if (menu.size() > 0)
         menu.add('Inspect', this.ShowInspector);

      return menu.size() > 0;
   }

   TPavePainter.prototype.ShowContextMenu = function(evnt) {

      // for color palette
      if (this.z_handle)
         return this.frame_painter().ShowContextMenu("z", evnt, this.GetObject().fAxis);

      if (!evnt) {
         d3.event.stopPropagation(); // disable main context menu
         d3.event.preventDefault();  // disable browser context menu
         evnt = d3.event;
      }

      JSROOT.Painter.createMenu(this, function(menu) {
         menu.painter.FillContextMenu(menu);

         menu.painter.FillObjectExecMenu(menu, "title", function() {
            menu.show();
        });
      }, evnt); // end menu creation
   }

   TPavePainter.prototype.IsStats = function() {
      return this.MatchObjectType('TPaveStats');
   }

   TPavePainter.prototype.ClearPave = function() {
      this.GetObject().Clear();
   }

   TPavePainter.prototype.AddText = function(txt) {
      this.GetObject().AddText(txt);
   }

   TPavePainter.prototype.FillFunctionStat = function(f1, dofit) {
      if (!dofit || !f1) return false;

      var print_fval    = dofit % 10,
          print_ferrors = Math.floor(dofit/10) % 10,
          print_fchi2   = Math.floor(dofit/100) % 10,
          print_fprob   = Math.floor(dofit/1000) % 10;

      if (print_fchi2 > 0)
         this.AddText("#chi^2 / ndf = " + this.Format(f1.fChisquare,"fit") + " / " + f1.fNDF);
      if (print_fprob > 0)
         this.AddText("Prob = "  + (('Math' in JSROOT) ? this.Format(JSROOT.Math.Prob(f1.fChisquare, f1.fNDF)) : "<not avail>"));
      if (print_fval > 0)
         for(var n=0;n<f1.GetNumPars();++n) {
            var parname = f1.GetParName(n), parvalue = f1.GetParValue(n), parerr = f1.GetParError(n);

            parvalue = (parvalue===undefined) ? "<not avail>" : this.Format(Number(parvalue),"fit");
            if (parerr !== undefined) {
               parerr = this.Format(parerr,"last");
               if ((Number(parerr)===0) && (f1.GetParError(n) != 0)) parerr = this.Format(f1.GetParError(n),"4.2g");
            }

            if ((print_ferrors > 0) && parerr)
               this.AddText(parname + " = " + parvalue + " #pm " + parerr);
            else
               this.AddText(parname + " = " + parvalue);
         }

      return true;
   }

   TPavePainter.prototype.FillStatistic = function() {

      var pp = this.pad_painter();
      if (pp && pp._fast_drawing) return false;

      var pave = this.GetObject(),
          main = pave.$main_painter || this.main_painter();

      if (pave.fName !== "stats") return false;
      if (!main || (typeof main.FillStatistic !== 'function')) return false;

      var dostat = parseInt(pave.fOptStat), dofit = parseInt(pave.fOptFit);
      if (isNaN(dostat)) dostat = JSROOT.gStyle.fOptStat;
      if (isNaN(dofit)) dofit = JSROOT.gStyle.fOptFit;

      // we take statistic from main painter
      if (!main.FillStatistic(this, dostat, dofit)) return false;

      // adjust the size of the stats box with the number of lines
      var nlines = pave.fLines.arr.length,
          stath = nlines * JSROOT.gStyle.StatFontSize;
      if ((stath <= 0) || (JSROOT.gStyle.StatFont % 10 === 3)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         pave.fY1NDC = pave.fY2NDC - stath;
      }

      return true;
   }

   TPavePainter.prototype.IsDummyPos = function(p) {
      if (!p) return true;

      return !p.fInit && !p.fX1 && !p.fX2 && !p.fY1 && !p.fY2 && !p.fX1NDC && !p.fX2NDC && !p.fY1NDC && !p.fY2NDC;
   }

   TPavePainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;

      var pave = this.GetObject();

      if (!pave.modified_NDC && !this.IsDummyPos(obj)) {
         // if position was not modified interactively, update from source object

         if (this.stored && !obj.fInit && (this.stored.fX1 == obj.fX1)
             && (this.stored.fX2 == obj.fX2) && (this.stored.fY1 == obj.fY1) && (this.stored.fY2 == obj.fY2)) {
            // case when source object not initialized and original coordinates are not changed
            // take over only modified NDC coordinate, used in tutorials/graphics/canvas.C
            if (this.stored.fX1NDC != obj.fX1NDC) pave.fX1NDC = obj.fX1NDC;
            if (this.stored.fX2NDC != obj.fX2NDC) pave.fX2NDC = obj.fX2NDC;
            if (this.stored.fY1NDC != obj.fY1NDC) pave.fY1NDC = obj.fY1NDC;
            if (this.stored.fY2NDC != obj.fY2NDC) pave.fY2NDC = obj.fY2NDC;
         } else {
            pave.fInit = obj.fInit;
            pave.fX1 = obj.fX1; pave.fX2 = obj.fX2;
            pave.fY1 = obj.fY1; pave.fY2 = obj.fY2;
            pave.fX1NDC = obj.fX1NDC; pave.fX2NDC = obj.fX2NDC;
            pave.fY1NDC = obj.fY1NDC; pave.fY2NDC = obj.fY2NDC;
         }

         this.stored = JSROOT.extend({}, obj); // store latest coordinates
      }

      pave.fOption = obj.fOption;
      pave.fBorderSize = obj.fBorderSize;

      switch (obj._typename) {
         case 'TPaveText':
            pave.fLines = JSROOT.clone(obj.fLines);
            return true;
         case 'TPavesText':
            pave.fLines = JSROOT.clone(obj.fLines);
            pave.fNpaves = obj.fNpaves;
            return true;
         case 'TPaveLabel':
            pave.fLabel = obj.fLabel;
            return true;
         case 'TPaveStats':
            pave.fOptStat = obj.fOptStat;
            pave.fOptFit = obj.fOptFit;
            return true;
         case 'TLegend':
            var oldprim = pave.fPrimitives;
            pave.fPrimitives = obj.fPrimitives;
            pave.fNColumns = obj.fNColumns;
            if (oldprim && oldprim.arr && pave.fPrimitives && pave.fPrimitives.arr && (oldprim.arr.length == pave.fPrimitives.arr.length)) {
               // try to sync object reference, new object does not displayed automatically
               // in ideal case one should use snapids in the entries
               for (var k=0;k<oldprim.arr.length;++k) {
                  var oldobj = oldprim.arr[k].fObject, newobj = pave.fPrimitives.arr[k].fObject;

                  if (oldobj && newobj && oldobj._typename == newobj._typename && oldobj.fName == newobj.fName)
                     pave.fPrimitives.arr[k].fObject = oldobj;
               }
            }
            return true;
         case 'TPaletteAxis':
            pave.fBorderSize = 1;
            pave.fShadowColor = 0;
            return true;
      }

      return false;
   }

   TPavePainter.prototype.Redraw = function() {
      this.DrawPave();
   }

   TPavePainter.prototype.Cleanup = function() {
      if (this.z_handle) {
         this.z_handle.Cleanup();
         delete this.z_handle;
      }

      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   function drawPave(divid, pave, opt) {
      // one could force drawing of PaveText on specific sub-pad
      var onpad;
      if ((typeof opt == 'string') && (opt.indexOf("onpad:")==0)) {
         onpad = opt.substr(6);
         opt = "";
      }

      var painter = new JSROOT.TPavePainter(pave);

      painter.SetDivId(divid, 2, onpad);

      if ((pave.fName === "title") && (pave._typename === "TPaveText")) {
         var tpainter = painter.FindPainterFor(null, "title");
         if (tpainter && (tpainter !== painter)) {
            tpainter.DeleteThis();
         } else if ((opt == "postitle") || painter.IsDummyPos(pave)) {
            var st = JSROOT.gStyle, fp = painter.frame_painter();
            if (st && fp) {
               var midx = st.fTitleX, y2 = st.fTitleY, w = st.fTitleW, h = st.fTitleH;
               if (!h && fp) h = (y2-fp.fY2NDC)*0.7;
               if (!w && fp) w = fp.fX2NDC - fp.fX1NDC;
               if (!h || isNaN(h) || (h<0)) h = 0.06;
               if (!w || isNaN(w) || (w<0)) w = 0.44;
               pave.fX1NDC = midx - w/2;
               pave.fY1NDC = y2 - h;
               pave.fX2NDC = midx + w/2;
               pave.fY2NDC = y2;
               pave.fInit = 1;
            }
         }
      } else if (pave._typename === "TPaletteAxis") {
         pave.fBorderSize = 1;
         pave.fShadowColor = 0;

         // check some default values of TGaxis object, otherwise axis will not be drawn
         if (pave.fAxis) {
            if (!pave.fAxis.fChopt) pave.fAxis.fChopt = "+";
            if (!pave.fAxis.fNdiv) pave.fAxis.fNdiv = 12;
            if (!pave.fAxis.fLabelOffset) pave.fAxis.fLabelOffset = 0.005;
         }

         painter.z_handle = new JSROOT.TAxisPainter(pave.fAxis, true);
         painter.z_handle.SetDivId(divid, -1);

         painter.UseContextMenu = true;
      }

      switch (pave._typename) {
         case "TPaveLabel":
            painter.PaveDrawFunc = painter.DrawPaveLabel;
            break;
         case "TPaveStats":
            painter.PaveDrawFunc = painter.DrawPaveStats;
            painter.$secondary = true; // indicates that painter created from others
            break;
         case "TPaveText":
         case "TPavesText":
         case "TDiamond":
            painter.PaveDrawFunc = painter.DrawPaveText;
            break;
         case "TLegend":
            painter.PaveDrawFunc = painter.DrawPaveLegend;
            break;
         case "TPaletteAxis":
            painter.PaveDrawFunc = painter.DrawPaletteAxis;
            break;
      }

      painter.DrawPave(opt);

      // drawing ready handled in special painters, if not exists - drawing is done
      return painter.PaveDrawFunc ? painter : painter.DrawingReady();
   }

   /** @summary Produce and draw TLegend object for the specified divid
    * @desc Should be called when all other objects are painted
    * Invoked when item "$legend" specified in JSROOT URL string
    * @private */
   function produceLegend(divid, opt) {
      var main_painter = JSROOT.GetMainPainter(divid);
      if (!main_painter) return;

      var pp = main_painter.pad_painter(),
          pad = main_painter.root_pad();
      if (!pp || !pad) return;

      var leg = JSROOT.Create("TLegend");

      for (var k=0;k<pp.painters.length;++k) {
         var painter = pp.painters[k],
             obj = painter.GetObject();

         if (!obj) continue;

         var entry = JSROOT.Create("TLegendEntry");
         entry.fObject = obj;
         entry.fLabel = (opt == "all") ? obj.fName : painter.GetItemName();
         entry.fOption = "";
         if (!entry.fLabel) continue;

         if (painter.lineatt && painter.lineatt.used) entry.fOption+="l";
         if (painter.fillatt && painter.fillatt.used) entry.fOption+="f";
         if (painter.markeratt && painter.markeratt.used) entry.fOption+="m";
         if (!entry.fOption) entry.fOption = "l";

         leg.fPrimitives.Add(entry);
      }

      // no entries - no need to draw legend
      var szx = 0.4, szy = leg.fPrimitives.arr.length;
      if (!szy) return;
      if (szy>8) szy = 8;
      szy *= 0.1;

      JSROOT.extend(leg, {
         fX1NDC: szx*pad.fLeftMargin + (1-szx)*(1-pad.fRightMargin),
         fY1NDC: (1-szy)*(1-pad.fTopMargin) + szy*pad.fBottomMargin,
         fX2NDC: 0.99-pad.fRightMargin,
         fY2NDC: 0.99-pad.fTopMargin
      });
      leg.fFillStyle = 1001;

      return drawPave(divid, leg);
   }

   // ==============================================================================

   function THistDrawOptions(opt) {
      this.Reset();
   }

   THistDrawOptions.prototype.Reset = function() {
      JSROOT.extend(this,
            { Axis: 0, RevX: false, RevY: false, Bar: false, BarStyle: 0, Curve: false,
              Hist: true, Line: false, Fill: false,
              Error: false, ErrorKind: -1, errorX: JSROOT.gStyle.fErrorX,
              Mark: false, Same: false, Scat: false, ScatCoef: 1., Func: true,
              Arrow: false, Box: false, BoxStyle: 0,
              Text: false, TextAngle: 0, TextKind: "", Char: 0, Color: false, Contour: 0,
              Lego: 0, Surf: 0, Off: 0, Tri: 0, Proj: 0, AxisPos: 0,
              Spec: false, Pie: false, List: false, Zscale: false, PadPalette: false, Candle: "",
              GLBox: 0, GLColor: false, Project: "",
              System: JSROOT.Painter.Coord.kCARTESIAN,
              AutoColor: false, NoStat: false, ForceStat: false, PadStats: false, PadTitle: false, AutoZoom: false,
              HighRes: 0, Zero: true, Palette: 0, BaseLine: false,
              Optimize: JSROOT.gStyle.OptimizeDraw, Mode3D: false,
              FrontBox: true, BackBox: true,
              _pmc: false, _plc: false, _pfc: false, need_fillcol: false,
              minimum: -1111, maximum: -1111, ymin:0, ymax:0 });
   }

   THistDrawOptions.prototype.Decode = function(opt, hdim, histo, pad, painter) {
      this.orginal = opt; // will be overwritten by OptionsStore call

      var d = new JSROOT.DrawOptions(opt), check3dbox = "";

      if ((hdim===1) && (histo.fSumw2.length > 0))
         for (var n=0;n<histo.fSumw2.length;++n)
            if (histo.fSumw2[n] > 0) { this.Error = true; this.Hist = false; this.Zero = false; break; }

      this.ndim = hdim || 1; // keep dimensions, used for now in GED

      this.PadStats = d.check("USE_PAD_STATS");
      this.PadPalette = d.check("USE_PAD_PALETTE");
      this.PadTitle = d.check("USE_PAD_TITLE");

      if (d.check('PAL', true)) this.Palette = d.partAsInt();
      // this is zooming of histo content
      if (d.check('MINIMUM:', true)) this.minimum = parseFloat(d.part); else this.minimum = histo.fMinimum;
      if (d.check('MAXIMUM:', true)) this.maximum = parseFloat(d.part); else this.maximum = histo.fMaximum;
      // this is actual range of data - used by graph drawing
      if (d.check('YMIN:', true)) this.ymin = parseFloat(d.part);
      if (d.check('YMAX:', true)) this.ymax = parseFloat(d.part);


      if (d.check('NOOPTIMIZE')) this.Optimize = 0;
      if (d.check('OPTIMIZE')) this.Optimize = 2;

      if (d.check('AUTOCOL')) this.AutoColor = true;
      if (d.check('AUTOZOOM')) this.AutoZoom = true;

      if (d.check('OPTSTAT',true)) this.optstat = d.partAsInt();
      if (d.check('OPTFIT',true)) this.optfit = d.partAsInt();

      if (d.check('NOSTAT')) this.NoStat = true;
      if (d.check('STAT')) this.ForceStat = true;

      if (d.check('NOTOOLTIP') && painter) painter.SetTooltipAllowed(false);
      if (d.check('TOOLTIP') && painter) painter.SetTooltipAllowed(true);

      if (d.check('LOGX') && pad) { pad.fLogx = 1; pad.fUxmin = 0; pad.fUxmax = 1; pad.fX1 = 0; pad.fX2 = 1; }
      if (d.check('LOGY') && pad) { pad.fLogy = 1; pad.fUymin = 0; pad.fUymax = 1; pad.fY1 = 0; pad.fY2 = 1; }
      if (d.check('LOGZ') && pad) pad.fLogz = 1;
      if (d.check('GRIDXY') && pad) pad.fGridx = pad.fGridy = 1;
      if (d.check('GRIDX') && pad) pad.fGridx = 1;
      if (d.check('GRIDY') && pad) pad.fGridy = 1;
      if (d.check('TICKXY') && pad) pad.fTickx = pad.fTicky = 1;
      if (d.check('TICKX') && pad) pad.fTickx = 1;
      if (d.check('TICKY') && pad) pad.fTicky = 1;

      if (d.check('FILL_', true)) {
         if (d.partAsInt(1)>0) this.histoFillColor = d.partAsInt(); else
         for (var col=0;col<8;++col)
            if (JSROOT.Painter.root_colors[col].toUpperCase() === d.part) this.histoFillColor = col;
      }
      if (d.check('LINE_', true)) {
         if (d.partAsInt(1)>0) this.histoLineColor = JSROOT.Painter.root_colors[d.partAsInt()]; else
         for (var col=0;col<8;++col)
            if (JSROOT.Painter.root_colors[col].toUpperCase() === d.part) this.histoLineColor = d.part;
      }

      if (d.check('X+')) this.AxisPos = 10;
      if (d.check('Y+')) this.AxisPos += 1;

      if (d.check('SAMES')) { this.Same = true; this.ForceStat = true; }
      if (d.check('SAME')) { this.Same = true; this.Func = true; }

      if (d.check('SPEC')) this.Spec = true; // not used

      if (d.check('BASE0') || d.check('MIN0')) this.BaseLine = 0; else
      if (JSROOT.gStyle.fHistMinimumZero) this.BaseLine = 0;

      if (d.check('PIE')) this.Pie = true; // not used

      if (d.check('CANDLE', true)) this.Candle = d.part;

      if (d.check('GLBOX',true)) this.GLBox = 10 + d.partAsInt();
      if (d.check('GLCOL')) this.GLColor = true;

      d.check('GL'); // suppress GL

      if (d.check('LEGO', true)) {
         this.Lego = 1;
         if (d.part.indexOf('0') >= 0) this.Zero = false;
         if (d.part.indexOf('1') >= 0) this.Lego = 11;
         if (d.part.indexOf('2') >= 0) this.Lego = 12;
         if (d.part.indexOf('3') >= 0) this.Lego = 13;
         if (d.part.indexOf('4') >= 0) this.Lego = 14;
         check3dbox = d.part;
         if (d.part.indexOf('Z') >= 0) this.Zscale = true;
      }

      if (d.check('SURF', true)) {
         this.Surf = d.partAsInt(10, 1);
         check3dbox = d.part;
         if (d.part.indexOf('Z')>=0) this.Zscale = true;
      }

      if (d.check('TF3', true)) check3dbox = d.part;

      if (d.check('ISO', true)) check3dbox = d.part;

      if (d.check('LIST')) this.List = true; // not used

      if (d.check('CONT', true) && (hdim>1)) {
         this.Contour = 1;
         if (d.part.indexOf('Z') >= 0) this.Zscale = true;
         if (d.part.indexOf('1') >= 0) this.Contour = 11; else
         if (d.part.indexOf('2') >= 0) this.Contour = 12; else
         if (d.part.indexOf('3') >= 0) this.Contour = 13; else
         if (d.part.indexOf('4') >= 0) this.Contour = 14;
      }

      // decode bar/hbar option
      if (d.check('HBAR', true)) this.BarStyle = 20; else
      if (d.check('BAR', true)) this.BarStyle = 10;
      if (this.BarStyle > 0) {
         this.Hist = false;
         this.need_fillcol = true;
         this.BarStyle += d.partAsInt();
      }

      if (d.check('ARR'))
         this.Arrow = true;

      if (d.check('BOX',true))
         this.BoxStyle = 10 + d.partAsInt();

      this.Box = this.BoxStyle > 0;

      if (d.check('COL')) this.Color = true;
      if (d.check('CHAR')) this.Char = 1;
      if (d.check('FUNC')) { this.Func = true; this.Hist = false; }
      if (d.check('AXIS')) this.Axis = 1;
      if (d.check('AXIG')) this.Axis = 2;

      if (d.check('TEXT', true)) {
         this.Text = true;
         this.Hist = false;
         this.TextAngle = Math.min(d.partAsInt(), 90);
         if (d.part.indexOf('N')>=0) this.TextKind = "N";
         if (d.part.indexOf('E0')>=0) this.TextLine = true;
         if (d.part.indexOf('E')>=0) this.TextKind = "E";
      }

      if (d.check('SCAT=', true)) {
         this.Scat = true;
         this.ScatCoef = parseFloat(d.part);
         if (isNaN(this.ScatCoef) || (this.ScatCoef<=0)) this.ScatCoef = 1.;
      }

      if (d.check('SCAT')) this.Scat = true;
      if (d.check('POL')) this.System = JSROOT.Painter.Coord.kPOLAR;
      if (d.check('CYL')) this.System = JSROOT.Painter.Coord.kCYLINDRICAL;
      if (d.check('SPH')) this.System = JSROOT.Painter.Coord.kSPHERICAL;
      if (d.check('PSR')) this.System = JSROOT.Painter.Coord.kRAPIDITY;

      if (d.check('TRI', true)) {
         this.Color = false;
         this.Tri = 1;
         check3dbox = d.part;
         if (d.part.indexOf('ERR') >= 0) this.Error = true;
      }

      if (d.check('AITOFF')) this.Proj = 1;
      if (d.check('MERCATOR')) this.Proj = 2;
      if (d.check('SINUSOIDAL')) this.Proj = 3;
      if (d.check('PARABOLIC')) this.Proj = 4;
      if (this.Proj > 0) this.Contour = 14;

      if (d.check('PROJX',true)) this.Project = "X" + d.partAsInt(0,1);
      if (d.check('PROJY',true)) this.Project = "Y" + d.partAsInt(0,1);

      if (check3dbox) {
         if (check3dbox.indexOf('FB') >= 0) this.FrontBox = false;
         if (check3dbox.indexOf('BB') >= 0) this.BackBox = false;
      }

      if ((hdim==3) && d.check('FB')) this.FrontBox = false;
      if ((hdim==3) && d.check('BB')) this.BackBox = false;

      this._pfc = d.check("PFC");
      this._plc = d.check("PLC") || this.AutoColor;
      this._pmc = d.check("PMC");

      if (d.check('L')) { this.Line = true; this.Hist = false; this.Error = false; }
      if (d.check('F')) { this.Fill = true; this.need_fillcol = true; }

      if (d.check('A')) this.Axis = -1;
      if ((this.Axis || this.Color) && d.check("RX")) this.RevX = true;
      if ((this.Axis || this.Color) && d.check("RY")) this.RevY = true;

      if (d.check('B1')) { this.BarStyle = 1; this.BaseLine = 0; this.Hist = false; this.need_fillcol = true; }
      if (d.check('B')) { this.BarStyle = 1; this.Hist = false; this.need_fillcol = true; }
      if (d.check('C')) { this.Curve = true; this.Hist = false; }
      if (d.check('][')) { this.Off = 1; this.Hist = true; }

      if (d.check('HIST')) { this.Hist = true; this.Func = true; this.Error = false; }

      this.Bar = (this.BarStyle > 0);

      delete this.MarkStyle; // remove mark style if any

      if (d.check('P0')) { this.Mark = true; this.Hist = false; this.Zero = true; }
      if (d.check('P')) { this.Mark = true; this.Hist = false; this.Zero = false; }
      if (d.check('Z')) this.Zscale = true;
      if (d.check('*')) { this.Mark = true; this.MarkStyle = 3; this.Hist = false; }
      if (d.check('H')) this.Hist = true;

      if (d.check('E', true)) {
         this.Error = true;
         if (hdim == 1) {
            this.Zero = false; // do not draw empty bins with errors
            this.Hist = false;
            if (!isNaN(parseInt(d.part[0]))) this.ErrorKind = parseInt(d.part[0]);
            if ((this.ErrorKind === 3) || (this.ErrorKind === 4)) this.need_fillcol = true;
            if (this.ErrorKind === 0) this.Zero = true; // enable drawing of empty bins
            if (d.part.indexOf('X0')>=0) this.errorX = 0;
         }
      }
      if (d.check('9')) this.HighRes = 1;
      if (d.check('0')) this.Zero = false;
      if (this.Color && d.check('1')) this.Zero = false;

      // flag identifies 3D drawing mode for histogram
      if ((this.Lego > 0) || (hdim == 3) ||
          ((this.Surf > 0) || this.Error && (hdim == 2))) this.Mode3D = true;

      //if (this.Surf == 15)
      //   if (this.System == JSROOT.Painter.Coord.kPOLAR || this.System == JSROOT.Painter.Coord.kCARTESIAN)
      //      this.Surf = 13;
   }

   // function should approx reconstruct draw options
   THistDrawOptions.prototype.asString = function(painter) {
      var fp = painter ? painter.frame_painter() : null;

      var res = "";
      if (this.Mode3D) {

         if (this.Lego) {
            res = "LEGO";
            if (!this.Zero) res += "0";
            if (this.Lego > 10) res += (this.Lego-10);
            if (this.Zscale) res+="Z";
         } else if (this.Surf) {
            res = "SURF" + (this.Surf-10);
            if (this.Zscale) res+="Z";
         }
         if (!this.FrontBox) res+="FB";
         if (!this.BackBox) res+="BB";

      } else {
         if (this.Scat) {
            res = "SCAT";
         } else if (this.Color) {
            res = "COL";
            if (!this.Zero) res+="0";
            if (this.Zscale) res+="Z";
            if (this.Axis < 0) res+="A";
         } else if (this.Contour) {
            res = "CONT";
            if (this.Contour > 10) res += (this.Contour-10);
            if (this.Zscale) res+="Z";
         } else if (this.Bar) {
            res = (this.BaseLine === false) ? "B" : "B1";
         } else if (this.Mark) {
            res = this.Zero ? "P0" : "P"; // here invert logic with 0
         } else if (this.Scat) {
            res = "SCAT";
         } else if (this.Error) {
            res = "E";
            if (this.ErrorKind>=0) res += this.ErrorKind;
         } else if (this.Line) {
            res += "L";
            if (this.Fill) res += "F";
         }

         if (this.Text) {
            res += "TEXT";
            if (this.TextAngle) res += this.TextAngle;
            res += this.TextKind;
         }

      }
      return res;
   }

   // ==============================================================================


   /**
    * @summary Basic painter for histogram classes
    *
    * @constructor
    * @memberof JSROOT
    * @arguments JSROOT.TObjectPainter
    * @param {object} histo - histogram object
    */

   function THistPainter(histo) {
      JSROOT.TObjectPainter.call(this, histo);
      this.draw_content = true;
      this.nbinsx = 0;
      this.nbinsy = 0;
      this.accept_drops = true; // indicate that one can drop other objects like doing Draw("same")
      this.mode3d = false;
      this.hist_painter_id = JSROOT.id_counter++; // assign unique identifier for hist painter
   }

   THistPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   THistPainter.prototype.GetHisto = function() {
      return this.GetObject();
   }

   /** Returns histogram axis @prviate */
   THistPainter.prototype.GetAxis = function(name) {
      var histo = this.GetObject();
      if (histo)
         switch(name) {
            case "x": return histo.fXaxis;
            case "y": return histo.fYaxis;
            case "z": return histo.fZaxis;
         }
      return null;
   }

   THistPainter.prototype.IsTProfile = function() {
      return this.MatchObjectType('TProfile');
   }

   THistPainter.prototype.IsTH1K = function() {
      return this.MatchObjectType('TH1K');
   }

   THistPainter.prototype.IsTH2Poly = function() {
      return this.MatchObjectType(/^TH2Poly/) || this.MatchObjectType(/^TProfile2Poly/);
   }

   THistPainter.prototype.Clear3DScene = function() {
      var fp = this.frame_painter();
      if (fp && typeof fp.Create3DScene === 'function')
         fp.Create3DScene(-1);
      this.mode3d = false;
   }

   /** Cleanup histogram painter */
   THistPainter.prototype.Cleanup = function() {

      // clear all 3D buffers
      this.Clear3DScene();

      delete this.fPalette;
      delete this.fContour;
      delete this.options;

      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   THistPainter.prototype.Dimension = function() {
      var histo = this.GetHisto();
      if (!histo) return 0;
      if (histo._typename.match(/^TH2/)) return 2;
      if (histo._typename.match(/^TProfile2D/)) return 2;
      if (histo._typename.match(/^TH3/)) return 3;
      if (this.IsTH2Poly()) return 2;
      return 1;
   }

   /** Decode options string opt and fill the option structure
    * @private */
   THistPainter.prototype.DecodeOptions = function(opt) {
      var histo = this.GetHisto(),
          hdim = this.Dimension(),
          pad = this.root_pad();

      if (!this.options)
         this.options = new THistDrawOptions;
      else
         this.options.Reset();

      this.options.Decode(opt || histo.fOption, hdim, histo, pad, this);

      this.OptionsStore(opt); // opt will be return as default draw option, used in webcanvas
   }

   THistPainter.prototype.CopyOptionsFrom = function(src) {
      if (src === this) return;
      var o = this.options, o0 = src.options;

      o.Mode3D = o0.Mode3D;
      o.Zero = o0.Zero;
      if (o0.Mode3D) {
         o.Lego = o0.Lego;
         o.Surf = o0.Surf;
      } else {
         o.Color = o0.Color;
         o.Contour = o0.Contour;
      }
   }

   /// copy draw options to all other histograms in the pad
   THistPainter.prototype.CopyOptionsToOthers = function() {
      var pthis = this;

      this.ForEachPainter(function(painter) {
         if (painter === pthis) return;
         if (typeof painter.CopyOptionsFrom == 'function')
            painter.CopyOptionsFrom(pthis);
      }, "objects");
   }

   THistPainter.prototype.ScanContent = function(when_axis_changed) {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values
      // if when_axis_changed === true specified, content will be scanned after axis zoom changed

      alert("HistPainter.prototype.ScanContent not implemented");
   }

   THistPainter.prototype.CheckPadRange = function(use_pad) {

      // actual work will be done when frame will draw axes
      if (this.is_main_painter())
         this.check_pad_range = use_pad ? "pad_range" : true;
   }

   THistPainter.prototype.CheckHistDrawAttributes = function() {

      var histo = this.GetHisto(),
          pp = this.pad_painter();

      if (pp && (this.options._pfc || this.options._plc || this.options._pmc)) {
         var icolor = pp.CreateAutoColor();
         if (this.options._pfc) { histo.fFillColor = icolor; delete this.fillatt; }
         if (this.options._plc) { histo.fLineColor = icolor; delete this.lineatt; }
         if (this.options._pmc) { histo.fMarkerColor = icolor; delete this.markeratt; }
         this.options._pfc = this.options._plc = this.options._pmc = false;
      }

      this.createAttFill({ attr: histo, color: this.options.histoFillColor, kind: 1 });

      this.createAttLine({ attr: histo, color0: this.options.histoLineColor });
   }

   /** @brief Update histogram object
    * @param obj - new histogram instance
    * @param opt - new drawing option (optional)
    * @returns {Boolean} - true if histogram was successfully updated */
   THistPainter.prototype.UpdateObject = function(obj, opt) {

      var histo = this.GetHisto(),
          fp = this.frame_painter();

      if (obj !== histo) {

         if (!this.MatchObjectType(obj)) return false;

         // simple replace of object does not help - one can have different
         // complex relations between histo and stat box, histo and colz axis,
         // one could have THStack or TMultiGraph object
         // The only that could be done is update of content

         // check only stats bit, later other settings can be monitored
         var statpainter = this.FindPainterFor(this.FindStat());
         if (histo.TestBit(JSROOT.TH1StatusBits.kNoStats) != obj.TestBit(JSROOT.TH1StatusBits.kNoStats)) {
            histo.fBits = obj.fBits;
            if (statpainter) statpainter.Enabled = !histo.TestBit(JSROOT.TH1StatusBits.kNoStats);
         }

         // special treatment for webcanvas - also name can be changed
         if (this.snapid !== undefined)
            histo.fName = obj.fName;

         histo.fFillColor = obj.fFillColor;
         histo.fFillStyle = obj.fFillStyle;
         histo.fLineColor = obj.fLineColor;
         histo.fLineStyle = obj.fLineStyle;
         histo.fLineWidth = obj.fLineWidth;

         histo.fEntries = obj.fEntries;
         histo.fTsumw = obj.fTsumw;
         histo.fTsumwx = obj.fTsumwx;
         histo.fTsumwx2 = obj.fTsumwx2;
         histo.fXaxis.fNbins = obj.fXaxis.fNbins;
         if (this.Dimension() > 1) {
            histo.fTsumwy = obj.fTsumwy;
            histo.fTsumwy2 = obj.fTsumwy2;
            histo.fTsumwxy = obj.fTsumwxy;
            histo.fYaxis.fNbins = obj.fYaxis.fNbins;
            if (this.Dimension() > 2) {
               histo.fTsumwz = obj.fTsumwz;
               histo.fTsumwz2 = obj.fTsumwz2;
               histo.fTsumwxz = obj.fTsumwxz;
               histo.fTsumwyz = obj.fTsumwyz;
               histo.fZaxis.fNbins = obj.fZaxis.fNbins;
            }
         }

         histo.fArray = obj.fArray;
         histo.fNcells = obj.fNcells;
         histo.fTitle = obj.fTitle;
         histo.fMinimum = obj.fMinimum;
         histo.fMaximum = obj.fMaximum;
         function CopyAxis(tgt,src) {
            tgt.fTitle = src.fTitle;
            tgt.fLabels = src.fLabels;
            tgt.fXmin = src.fXmin;
            tgt.fXmax = src.fXmax;
            tgt.fTimeDisplay = src.fTimeDisplay;
            tgt.fTimeFormat = src.fTimeFormat;
            // copy attributes
            tgt.fAxisColor = src.fAxisColor;
            tgt.fLabelColor = src.fLabelColor;
            tgt.fLabelFont = src.fLabelFont;
            tgt.fLabelOffset = src.fLabelOffset;
            tgt.fLabelSize = src.fLabelSize;
            tgt.fNdivisions = src.fNdivisions;
            tgt.fTickLength = src.fTickLength;
            tgt.fTitleColor = src.fTitleColor;
            tgt.fTitleFont = src.fTitleFont;
            tgt.fTitleOffset = src.fTitleOffset;
            tgt.fTitleSize = src.fTitleSize;
         }
         CopyAxis(histo.fXaxis, obj.fXaxis);
         CopyAxis(histo.fYaxis, obj.fYaxis);
         CopyAxis(histo.fZaxis, obj.fZaxis);

         if (this.snapid || !fp || !fp.zoom_changed_interactive) {
            function CopyZoom(tgt,src) {
               tgt.fFirst = src.fFirst;
               tgt.fLast = src.fLast;
               tgt.fBits = src.fBits;
            }
            CopyZoom(histo.fXaxis, obj.fXaxis);
            CopyZoom(histo.fYaxis, obj.fYaxis);
            CopyZoom(histo.fZaxis, obj.fZaxis);
         }
         histo.fSumw2 = obj.fSumw2;

         if (this.IsTProfile()) {
            histo.fBinEntries = obj.fBinEntries;
         } else if (this.IsTH1K()) {
            histo.fNIn = obj.fNIn;
            histo.fReady = false;
         } else if (this.IsTH2Poly()) {
            histo.fBins = obj.fBins;
         }

         if (this.options.Func) {

            var painters = [], newfuncs = [], pp = this.pad_painter(), pid = this.hist_painter_id;

            // find painters associated with histogram
            if (pp)
               pp.ForEachPainterInPad(function(objp) {
                  if (objp.child_painter_id === pid)
                     painters.push(objp);
               }, "objects");

            if (obj.fFunctions)
               for (var n=0;n<obj.fFunctions.arr.length;++n) {
                  var func = obj.fFunctions.arr[n];
                  if (!func || !func._typename) continue;
                  var funcpainter = null, func_indx = -1;

                  // try to find matching object in associated list of painters
                  for (var i=0;i<painters.length;++i)
                     if (painters[i].MatchObjectType(func._typename) && (painters[i].GetObject().fName === func.fName)) {
                        funcpainter = painters[i];
                        func_indx = i;
                        break;
                     }
                  // or just in generic list of painted objects
                  if (!funcpainter && func.fName)
                     funcpainter = this.FindPainterFor(null, func.fName, func._typename);

                  if (funcpainter) {
                     funcpainter.UpdateObject(func);
                     if (func_indx >= 0) painters.splice(func_indx, 1);
                  } else {
                     newfuncs.push(func);
                  }
               }

            if (statpainter) {
               var indx = painters.indexOf(statpainter);
               if (indx >= 0) painters.splice(indx, 1);
            }

            // remove all function which are not found in new list of primitives
            if (pp && (painters.length > 0))
               pp.CleanPrimitives(function(p) { return painters.indexOf(p) >= 0; });

            // plot new objects on the same pad - will works only for simple drawings already loaded
            if (pp && (newfuncs.length > 0)) {
               var prev_name = pp.has_canvas ? pp.CurrentPadName(pp.this_pad_name) : undefined;
               for (var k=0;k<newfuncs.length;++k)
                  JSROOT.draw(this.divid, newfuncs[k],"", function (painter) { painter.child_painter_id = pid; } )
               pp.CurrentPadName(prev_name);
            }
         }

         var changed_opt = (histo.fOption != obj.fOption);
         histo.fOption = obj.fOption;

         if (((opt !== undefined) && (this.options.original !== opt)) || changed_opt)
            this.DecodeOptions(opt || histo.fOption);
      }

      if (this.snapid || !fp || !fp.zoom_changed_interactive)
         this.CheckPadRange();

      this.ScanContent();

      this.histogram_updated = true; // indicate that object updated

      return true;
   }

   THistPainter.prototype.CreateAxisFuncs = function(with_y_axis, with_z_axis) {
      // here functions are defined to convert index to axis value and back
      // introduced to support non-equidistant bins

      var histo = this.GetHisto();

      function AssignFuncs(axis) {
         if (axis.fXbins.length >= axis.fNbins) {
            axis.regular = false;
            axis.GetBinCoord = function(bin) {
               var indx = Math.round(bin);
               if (indx <= 0) return this.fXmin;
               if (indx > this.fNbins) return this.fXmax;
               if (indx==bin) return this.fXbins[indx];
               var indx2 = (bin < indx) ? indx - 1 : indx + 1;
               return this.fXbins[indx] * Math.abs(bin-indx2) + this.fXbins[indx2] * Math.abs(bin-indx);
            };
            axis.FindBin = function(x,add) {
               for (var k = 1; k < this.fXbins.length; ++k)
                  if (x < this.fXbins[k]) return Math.floor(k-1+add);
               return this.fNbins;
            };
         } else {
            axis.regular = true;
            axis.binwidth = (axis.fXmax - axis.fXmin) / (axis.fNbins || 1);
            axis.GetBinCoord = function(bin) { return this.fXmin + bin*this.binwidth; };
            axis.FindBin = function(x,add) { return Math.floor((x - this.fXmin) / this.binwidth + add); };
         }
      }

      this.xmin = histo.fXaxis.fXmin;
      this.xmax = histo.fXaxis.fXmax;
      AssignFuncs(histo.fXaxis);

      this.ymin = histo.fYaxis.fXmin;
      this.ymax = histo.fYaxis.fXmax;

      if (!with_y_axis || (this.nbinsy==0)) return;

      AssignFuncs(histo.fYaxis);

      if (!with_z_axis || (this.nbinsz==0)) return;

      AssignFuncs(histo.fZaxis);
   }

   /** @summary Create x,y objects which maps user coordinates into pixels
    *  @desc Now moved into TFramePainter
    *  @private */
   THistPainter.prototype.CreateXY = function() {
      if (!this.is_main_painter()) return;

      var histo = this.GetHisto(),
          fp = this.frame_painter();

      if (!fp)
         return console.warn("histogram drawn without frame - not supported");

      // artifically add y range to display axes
      if (this.ymin === this.ymax) this.ymax += 1;

      fp.SetAxesRanges(histo.fXaxis, this.xmin, this.xmax, histo.fYaxis, this.ymin, this.ymax, histo.fZaxis, 0, 0);
      fp.CreateXY({ ndim: this.Dimension(),
                    check_pad_range: this.check_pad_range,
                    create_canvas: this.create_canvas,
                    zoom_ymin: this.zoom_ymin,
                    zoom_ymax: this.zoom_ymax,
                    ymin_nz: this.ymin_nz,
                    swap_xy: (this.options.BarStyle >= 20),
                    reverse_x: this.options.RevX,
                    reverse_y: this.options.RevY,
                    Proj: this.options.Proj,
                    extra_y_space: this.options.Text && (this.options.BarStyle > 0) });
      delete this.check_pad_range;
   }

   THistPainter.prototype.DrawBins = function() {
      alert("HistPainter.DrawBins not implemented");
   }

   THistPainter.prototype.DrawAxes = function() {
      // axes can be drawn only for main histogram

      if (!this.is_main_painter() || this.options.Same) return;

      var fp = this.frame_painter();
      if (!fp) return;

      fp.DrawAxes(false, this.options.Axis < 0, this.options.AxisPos, this.options.Zscale);
      fp.DrawGrids();
   }

   THistPainter.prototype.ToggleTitle = function(arg) {
      var histo = this.GetHisto();
      if (!this.is_main_painter() || !histo) return false;
      if (arg==='only-check') return !histo.TestBit(JSROOT.TH1StatusBits.kNoTitle);
      histo.InvertBit(JSROOT.TH1StatusBits.kNoTitle);
      this.DrawTitle();
   }

   THistPainter.prototype.DrawTitle = function() {

      // case when histogram drawn over other histogram (same option)
      if (!this.is_main_painter() || this.options.Same) return;

      var histo = this.GetHisto(), st = JSROOT.gStyle,
          tpainter = this.FindPainterFor(null, "title"),
          pt = tpainter ? tpainter.GetObject() : null;

      if (!pt) pt = this.FindInPrimitives("title");
      if (pt && (pt._typename !== "TPaveText")) pt = null;

      var draw_title = !histo.TestBit(JSROOT.TH1StatusBits.kNoTitle) && (st.fOptTitle > 0);

      // histo.fTitle = "#strike{testing} #overline{Title:} #overline{Title:_{X}} #underline{test}  #underline{test^{X}}";
      // histo.fTitle = "X-Y-#overline{V}_{#Phi}";

      if (pt) {
         pt.Clear();
         if (draw_title) pt.AddText(histo.fTitle);
         if (tpainter) tpainter.Redraw();
      } else if (draw_title && !tpainter && histo.fTitle && !this.options.PadTitle) {
         pt = JSROOT.Create("TPaveText");
         pt.fName = "title";
         pt.fTextFont = st.fTitleFont;
         pt.fTextSize = st.fTitleFontSize;
         pt.fTextColor = st.fTitleTextColor;
         pt.fTextAlign = st.fTitleAlign;
         pt.fFillColor = st.fTitleColor;
         pt.fFillStyle = st.fTitleStyle;
         pt.fBorderSize = st.fTitleBorderSize;

         pt.AddText(histo.fTitle);
         tpainter = JSROOT.Painter.drawPave(this.divid, pt, "postitle");
         if (tpainter) tpainter.$secondary = true;
      }
   }

   THistPainter.prototype.processTitleChange = function(arg) {

      var histo = this.GetHisto(),
          tpainter = this.FindPainterFor(null, "title");

      if (!histo || !tpainter) return null;

      if (arg==="check")
         return (!this.is_main_painter() || this.options.Same) ? null : histo;

      var pt = tpainter.GetObject();
      pt.Clear();
      pt.AddText(histo.fTitle);

      tpainter.Redraw();

      this.WebCanvasExec('SetTitle("' + histo.fTitle + '")');
   }

   THistPainter.prototype.UpdateStatWebCanvas = function() {
      if (!this.snapid) return;

      var stat = this.FindStat(),
          statpainter = this.FindPainterFor(stat);

      if (statpainter && !statpainter.snapid) statpainter.Redraw();
   }

   THistPainter.prototype.ToggleStat = function(arg) {

      var stat = this.FindStat(), statpainter = null;

      if (!arg) arg = "";

      if (!stat) {
         if (arg.indexOf('-check') > 0) return false;
         // when statbox created first time, one need to draw it
         stat = this.CreateStat(true);
      } else {
         statpainter = this.FindPainterFor(stat);
      }

      if (arg=='only-check') return statpainter ? statpainter.Enabled : false;

      if (arg=='fitpar-check') return stat ? stat.fOptFit : false;

      if (arg=='fitpar-toggle') {
         if (!stat) return false;
         stat.fOptFit = stat.fOptFit ? 0 : 1111; // for websocket command should be send to server
         if (statpainter) statpainter.Redraw();
         return true;
      }

      if (statpainter) {
         statpainter.Enabled = !statpainter.Enabled;
         // when stat box is drawn, it always can be drawn individually while it
         // should be last for colz RedrawPad is used
         statpainter.Redraw();
         return statpainter.Enabled;
      }

      JSROOT.draw(this.divid, stat, "onpad:" + this.pad_name);

      return true;
   }

   THistPainter.prototype.GetSelectIndex = function(axis, side, add) {
      // be aware - here indexes starts from 0
      var indx = 0,
          main = this.frame_painter(),
          nbin = this['nbins'+axis] || 0,
          taxis = this.GetAxis(axis),
          min = main ? main['zoom_' + axis + 'min'] : 0,
          max = main ? main['zoom_' + axis + 'max'] : 0;

      if ((min !== max) && taxis) {
         if (side == "left")
            indx = taxis.FindBin(min, add || 0);
         else
            indx = taxis.FindBin(max, (add || 0) + 0.5);
         if (indx<0) indx = 0; else if (indx>nbin) indx = nbin;
      } else {
         indx = (side == "left") ? 0 : nbin;
      }

      // TAxis object of histogram, where user range can be stored
      if (taxis) {
         if ((taxis.fFirst === taxis.fLast) || !taxis.TestBit(JSROOT.EAxisBits.kAxisRange) ||
             ((taxis.fFirst<=1) && (taxis.fLast>=nbin))) taxis = undefined;
      }

      if (side == "left") {
         if (indx < 0) indx = 0;
         if (taxis && (taxis.fFirst>1) && (indx<taxis.fFirst)) indx = taxis.fFirst-1;
      } else {
         if (indx > nbin) indx = nbin;
         if (taxis && (taxis.fLast <= nbin) && (indx>taxis.fLast)) indx = taxis.fLast;
      }

      return indx;
   }

   /** Find function in histogram list of functions
    * @private */
   THistPainter.prototype.FindFunction = function(type_name, obj_name) {
      var histo = this.GetObject(),
          funcs = histo && histo.fFunctions ? histo.fFunctions.arr : null;

      if (!funcs) return null;

      for (var i = 0; i < funcs.length; ++i) {
         if (obj_name && (funcs[i].fName !== obj_name)) continue;
         if (funcs[i]._typename === type_name) return funcs[i];
      }

      return null;
   }

   /** Find stats box - either in list of functions or as object of correspondent painter
    * @private */
   THistPainter.prototype.FindStat = function() {
      if (this.options.PadStats) {
         var p = this.FindPainterFor(null,"stats", "TPaveStats");
         return p ? p.GetObject() : null;
      }

      return this.FindFunction('TPaveStats', 'stats');
   }

   THistPainter.prototype.IgnoreStatsFill = function() {
      return !this.GetObject() || (!this.draw_content && !this.create_stats) || (this.options.Axis>0);
   }

   /** @summary Create stat box for histogram if required
    * @private
    */
   THistPainter.prototype.CreateStat = function(force) {

      var histo = this.GetHisto();

      if (this.options.PadStats || !histo) return null;

      if (!force && !this.options.ForceStat) {
         if (this.options.NoStat || histo.TestBit(JSROOT.TH1StatusBits.kNoStats) || !JSROOT.gStyle.AutoStat) return null;

         if (!this.draw_content || !this.is_main_painter()) return null;
      }

      var stats = this.FindStat(), st = JSROOT.gStyle,
          optstat = this.options.optstat, optfit = this.options.optfit;

      if (optstat !== undefined) {
         if (stats) stats.fOptStat = optstat;
         delete this.options.optstat;
      } else {
         optstat = histo.$custom_stat || st.fOptStat;
      }

      if (optfit !== undefined) {
         if (stats) stats.fOptFit = optfit;
         delete this.options.optfit;
      } else {
         optfit = st.fOptFit;
      }

      if (!stats && !optstat && !optfit) return null;

      this.create_stats = true;

      if (stats) return stats;

      stats = JSROOT.Create('TPaveStats');
      JSROOT.extend(stats, { fName: 'stats',
                             fOptStat: optstat,
                             fOptFit: optfit,
                             fBorderSize: 1 });

      stats.fX1NDC = st.fStatX - st.fStatW;
      stats.fY1NDC = st.fStatY - st.fStatH;
      stats.fX2NDC = st.fStatX;
      stats.fY2NDC = st.fStatY;

      stats.fFillColor = st.fStatColor;
      stats.fFillStyle = st.fStatStyle;

      stats.fTextAngle = 0;
      stats.fTextSize = st.fStatFontSize;
      stats.fTextAlign = 12;
      stats.fTextColor = st.fStatTextColor;
      stats.fTextFont = st.fStatFont;

      if (histo._typename.match(/^TProfile/) || histo._typename.match(/^TH2/))
         stats.fY1NDC = 0.67;

      stats.AddText(histo.fName);

      this.AddFunction(stats);

      return stats;
   }

   THistPainter.prototype.AddFunction = function(obj, asfirst) {
      var histo = this.GetObject();
      if (!histo || !obj) return;

      if (!histo.fFunctions)
         histo.fFunctions = JSROOT.Create("TList");

      if (asfirst)
         histo.fFunctions.AddFirst(obj);
      else
         histo.fFunctions.Add(obj);
   }

   /** @summary Method draws next function from the functions list @private */
   THistPainter.prototype.DrawNextFunction = function(indx, callback, painter) {

      if (painter && (typeof painter == "object"))
         painter.child_painter_id = this.hist_painter_id;

      var histo = this.GetHisto();
      if (!this.options.Func || !histo.fFunctions ||
          (indx >= histo.fFunctions.arr.length)) return JSROOT.CallBack(callback);

      var func = histo.fFunctions.arr[indx],
          opt = histo.fFunctions.opt[indx],
          do_draw = false,
          func_painter = this.FindPainterFor(func);

      // no need to do something if painter for object was already done
      // object will be redraw automatically
      if (func_painter === null) {
         if (func._typename === 'TPaveText' || func._typename === 'TPaveStats') {
            do_draw = !histo.TestBit(JSROOT.TH1StatusBits.kNoStats) && !this.options.NoStat;
         } else if (func._typename === 'TF1') {
            do_draw = !func.TestBit(JSROOT.BIT(9));
         } else {
            do_draw = (func._typename !== 'TPaletteAxis');
         }
      }

      if (do_draw)
         return JSROOT.draw(this.divid, func, opt, this.DrawNextFunction.bind(this, indx+1, callback));

      this.DrawNextFunction(indx+1, callback);
   }

   /** @summary Unzoom user range if any @private */
   THistPainter.prototype.UnzoomUserRange = function(dox, doy, doz) {

      var res = false, painter = this, histo = this.GetHisto();

      if (!histo) return false;

      function UnzoomTAxis(obj) {
         if (!obj) return false;
         if (!obj.TestBit(JSROOT.EAxisBits.kAxisRange)) return false;
         if (obj.fFirst === obj.fLast) return false;
         if ((obj.fFirst <= 1) && (obj.fLast >= obj.fNbins)) return false;
         obj.InvertBit(JSROOT.EAxisBits.kAxisRange);
         return true;
      }

      function UzoomMinMax(ndim, hist) {
         if (painter.Dimension()!==ndim) return false;
         if ((painter.options.minimum===-1111) && (painter.options.maximum===-1111)) return false;
         if (!painter.draw_content) return false; // if not drawing content, not change min/max
         painter.options.minimum = painter.options.maximum = -1111;
         painter.ScanContent(true); // to reset ymin/ymax
         return true;
      }

      if (dox && UnzoomTAxis(histo.fXaxis)) res = true;
      if (doy && (UnzoomTAxis(histo.fYaxis) || UzoomMinMax(1, histo))) res = true;
      if (doz && (UnzoomTAxis(histo.fZaxis) || UzoomMinMax(2, histo))) res = true;

      return res;
   }

   THistPainter.prototype.AllowDefaultYZooming = function() {
      // return true if default Y zooming should be enabled
      // it is typically for 2-Dim histograms or
      // when histogram not draw, defined by other painters

      if (this.Dimension()>1) return true;
      if (this.draw_content) return false;

      var pad_painter = this.pad_painter();
      if (pad_painter &&  pad_painter.painters)
         for (var k = 0; k < pad_painter.painters.length; ++k) {
            var subpainter = pad_painter.painters[k];
            if ((subpainter!==this) && subpainter.wheel_zoomy!==undefined)
               return subpainter.wheel_zoomy;
         }

      return false;
   }

   /** @summary Add different interactive handlers
    *
    * @desc only first (main) painter in list allowed to add interactive functionality
    * Most of interactivity now handled by frame
    * @private
    */
   THistPainter.prototype.AddInteractive = function() {

      if (this.is_main_painter()) {
         var fp = this.frame_painter();
         if (fp) fp.AddInteractive();
      }
   }

   THistPainter.prototype.ShowContextMenu = function(kind, evnt, obj) {

      // this is for debug purposes only, when context menu is where, close is and show normal menu
      //if (!evnt && !kind && document.getElementById('root_ctx_menu')) {
      //   var elem = document.getElementById('root_ctx_menu');
      //   elem.parentNode.removeChild(elem);
      //   return;
      //}

      var menu_painter = this, frame_corner = false, fp = null; // object used to show context menu

      if (!evnt) {
         d3.event.preventDefault();
         d3.event.stopPropagation(); // disable main context menu
         evnt = d3.event;

         if (kind === undefined) {
            var ms = d3.mouse(this.svg_frame().node()),
                tch = d3.touches(this.svg_frame().node()),
                pp = this.pad_painter(),
                pnt = null, sel = null;

            fp = this.frame_painter();

            if (tch.length === 1) pnt = { x: tch[0][0], y: tch[0][1], touch: true }; else
            if (ms.length === 2) pnt = { x: ms[0], y: ms[1], touch: false };

            if ((pnt !== null) && (pp !== null)) {
               pnt.painters = true; // assign painter for every tooltip
               var hints = pp.GetTooltips(pnt), bestdist = 1000;
               for (var n=0;n<hints.length;++n)
                  if (hints[n] && hints[n].menu) {
                     var dist = ('menu_dist' in hints[n]) ? hints[n].menu_dist : 7;
                     if (dist < bestdist) { sel = hints[n].painter; bestdist = dist; }
                  }
            }

            if (sel!==null) menu_painter = sel; else
            if (fp!==null) kind = "frame";

            if (pnt!==null) frame_corner = (pnt.x>0) && (pnt.x<20) && (pnt.y>0) && (pnt.y<20);

            if (fp) fp.SetLastEventPos(pnt);
         }
      }

      JSROOT.Painter.createMenu(menu_painter, function(menu) {
         var domenu = menu.painter.FillContextMenu(menu, kind, obj);

         // fill frame menu by default - or append frame elements when activated in the frame corner
         if (fp && (!domenu || (frame_corner && (kind!=="frame"))))
            domenu = fp.FillContextMenu(menu);

         if (domenu)
            menu.painter.FillObjectExecMenu(menu, kind, function() {
                // suppress any running zooming
                menu.painter.SwitchTooltip(false);
                menu.show(null, menu.painter.SwitchTooltip.bind(menu.painter, true));
            });

      }, evnt);  // end menu creation
   }

   /** @summary Invoke dialog to enter and modify user range @private */
   THistPainter.prototype.ChangeUserRange = function(arg) {
      var histo = this.GetHisto(),
          taxis = histo ? histo['f'+arg+"axis"] : null;
      if (!taxis) return;

      var curr = "[1," + taxis.fNbins+"]";
      if (taxis.TestBit(JSROOT.EAxisBits.kAxisRange))
          curr = "[" +taxis.fFirst+"," + taxis.fLast+"]";

      var res = prompt("Enter user range for axis " + arg + " like [1," + taxis.fNbins + "]", curr);
      if (!res) return;
      res = JSON.parse(res);

      if (!res || (res.length!=2) || isNaN(res[0]) || isNaN(res[1])) return;
      taxis.fFirst = parseInt(res[0]);
      taxis.fLast = parseInt(res[1]);

      var newflag = (taxis.fFirst < taxis.fLast) && (taxis.fFirst >= 1) && (taxis.fLast<=taxis.fNbins);
      if (newflag != taxis.TestBit(JSROOT.EAxisBits.kAxisRange))
         taxis.InvertBit(JSROOT.EAxisBits.kAxisRange);

      this.Redraw();
   }

   /** @summary Start dialog to modify range of axis where histogram values are displayed @private */
   THistPainter.prototype.ChangeValuesRange = function() {
      let curr;
      if ((this.options.minimum != -1111) && (this.options.maximum != -1111))
         curr = "[" + this.options.minimum + "," + this.options.maximum + "]";
      else
         curr = "[" + this.gminbin + "," + this.gmaxbin + "]";

      let res = prompt("Enter min/max hist values or empty string to reset", curr);
      res = res ? JSON.parse(res) : [];

      if (!res || (typeof res != "object") || (res.length!=2) || isNaN(res[0]) || isNaN(res[1])) {
         this.options.minimum = this.options.maximum = -1111;
      } else {
         this.options.minimum = res[0];
         this.options.maximum = res[1];
       }

       this.RedrawPad();
   }

   THistPainter.prototype.FillContextMenu = function(menu) {

      var histo = this.GetHisto(),
          fp = this.frame_painter();
      if (!histo) return;

      menu.add("header:"+ histo._typename + "::" + histo.fName);

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
            else
               menu.add("Values", this.ChangeValuesRange);
            menu.add("endsub:")
         }

         if (typeof this.FillHistContextMenu == 'function')
            this.FillHistContextMenu(menu);
      }

      if (this.options.Mode3D) {
         // menu for 3D drawings

         if (menu.size() > 0)
            menu.add("separator");

         var main = this.main_painter() || this;

         menu.addchk(main.IsTooltipAllowed(), 'Show tooltips', function() {
            main.SetTooltipAllowed("toggle");
         });

         menu.addchk(fp.enable_highlight, 'Highlight bins', function() {
            fp.enable_highlight = !fp.enable_highlight;
            if (!fp.enable_highlight && fp.BinHighlight3D && fp.mode3d) fp.BinHighlight3D(null);
         });

         if (fp && fp.Render3D) {
            menu.addchk(main.options.FrontBox, 'Front box', function() {
               main.options.FrontBox = !main.options.FrontBox;
               fp.Render3D();
            });
            menu.addchk(main.options.BackBox, 'Back box', function() {
               main.options.BackBox = !main.options.BackBox;
               fp.Render3D();
            });
         }

         if (this.draw_content) {
            menu.addchk(!this.options.Zero, 'Suppress zeros', function() {
               this.options.Zero = !this.options.Zero;
               this.InteractiveRedraw("pad");
            });

            if ((this.options.Lego==12) || (this.options.Lego==14)) {
               menu.addchk(this.options.Zscale, "Z scale", function() {
                  this.ToggleColz();
               });
               if (this.FillPaletteMenu) this.FillPaletteMenu(menu);
            }
         }

         if (main.control && typeof main.control.reset === 'function')
            menu.add('Reset camera', function() {
               main.control.reset();
            });
      }

      this.FillAttContextMenu(menu);

      if (this.histogram_updated && fp.zoom_changed_interactive)
         menu.add('Let update zoom', function() {
            fp.zoom_changed_interactive = 0;
         });

      return true;
   }

   THistPainter.prototype.ButtonClick = function(funcname) {
      // TODO: move to frame painter

      var fp = this.frame_painter();

      if (!this.is_main_painter() || !fp) return false;

      switch(funcname) {
         case "ToggleZoom":
            if ((fp.zoom_xmin !== fp.zoom_xmax) || (fp.zoom_ymin !== fp.zoom_ymax) || (fp.zoom_zmin !== fp.zoom_zmax)) {
               fp.Unzoom();
               fp.zoom_changed_interactive = 0;
               return true;
            }
            if (this.draw_content && (typeof this.AutoZoom === 'function')) {
               this.AutoZoom();
               return true;
            }
            break;
         case "ToggleLogX": fp.ToggleLog("x"); break;
         case "ToggleLogY": fp.ToggleLog("y"); break;
         case "ToggleLogZ": fp.ToggleLog("z"); break;
         case "ToggleStatBox": this.ToggleStat(); return true; break;
      }
      return false;
   }

   THistPainter.prototype.FillToolbar = function(not_shown) {
      var pp = this.pad_painter();
      if (!pp) return;

      pp.AddButton(JSROOT.ToolbarIcons.auto_zoom, 'Toggle between unzoom and autozoom-in', 'ToggleZoom', "Ctrl *");
      pp.AddButton(JSROOT.ToolbarIcons.arrow_right, "Toggle log x", "ToggleLogX", "PageDown");
      pp.AddButton(JSROOT.ToolbarIcons.arrow_up, "Toggle log y", "ToggleLogY", "PageUp");
      if (this.Dimension() > 1)
         pp.AddButton(JSROOT.ToolbarIcons.arrow_diag, "Toggle log z", "ToggleLogZ");
      if (this.draw_content)
         pp.AddButton(JSROOT.ToolbarIcons.statbox, 'Toggle stat box', "ToggleStatBox");
      if (!not_shown) pp.ShowButtons();
   }

   /** @summary Returns tooltip information for 3D drawings @private */
   THistPainter.prototype.Get3DToolTip = function(indx) {
      var histo = this.GetHisto(),
          tip = { bin: indx, name: histo.fName, title: histo.fTitle };
      switch (this.Dimension()) {
         case 1:
            tip.ix = indx; tip.iy = 1;
            tip.value = histo.getBinContent(tip.ix);
            tip.error = histo.getBinError(indx);
            tip.lines = this.GetBinTips(indx-1);
            break;
         case 2:
            tip.ix = indx % (this.nbinsx + 2);
            tip.iy = (indx - tip.ix) / (this.nbinsx + 2);
            tip.value = histo.getBinContent(tip.ix, tip.iy);
            tip.error = histo.getBinError(indx);
            tip.lines = this.GetBinTips(tip.ix-1, tip.iy-1);
            break;
         case 3:
            tip.ix = indx % (this.nbinsx+2);
            tip.iy = ((indx - tip.ix) / (this.nbinsx+2)) % (this.nbinsy+2);
            tip.iz = (indx - tip.ix - tip.iy * (this.nbinsx+2)) / (this.nbinsx+2) / (this.nbinsy+2);
            tip.value = histo.getBinContent(tip.ix, tip.iy, tip.iz);
            tip.error = histo.getBinError(indx);
            tip.lines = this.GetBinTips(tip.ix-1, tip.iy-1, tip.iz-1);
            break;
      }

      return tip;
   }

   THistPainter.prototype.CreateContour = function(nlevels, zmin, zmax, zminpositive) {

      if (nlevels<1) nlevels = JSROOT.gStyle.fNumberContours;
      this.fContour = [];
      this.colzmin = zmin;
      this.colzmax = zmax;

      if (this.root_pad().fLogz) {
         if (this.colzmax <= 0) this.colzmax = 1.;
         if (this.colzmin <= 0)
            if ((zminpositive===undefined) || (zminpositive <= 0))
               this.colzmin = 0.0001*this.colzmax;
            else
               this.colzmin = ((zminpositive < 3) || (zminpositive>100)) ? 0.3*zminpositive : 1;
         if (this.colzmin >= this.colzmax) this.colzmin = 0.0001*this.colzmax;

         var logmin = Math.log(this.colzmin)/Math.log(10),
             logmax = Math.log(this.colzmax)/Math.log(10),
             dz = (logmax-logmin)/nlevels;
         this.fContour.push(this.colzmin);
         for (var level=1; level<nlevels; level++)
            this.fContour.push(Math.exp((logmin + dz*level)*Math.log(10)));
         this.fContour.push(this.colzmax);
         this.fCustomContour = true;
      } else {
         if ((this.colzmin === this.colzmax) && (this.colzmin !== 0)) {
            this.colzmax += 0.01*Math.abs(this.colzmax);
            this.colzmin -= 0.01*Math.abs(this.colzmin);
         }
         var dz = (this.colzmax-this.colzmin)/nlevels;
         for (var level=0; level<=nlevels; level++)
            this.fContour.push(this.colzmin + dz*level);
      }

      var fp = this.frame_painter();
      if ((this.Dimension() < 3) && fp) {
         fp.zmin = this.colzmin;
         fp.zmax = this.colzmax;
      }

      return this.fContour;
   }

   THistPainter.prototype.GetContour = function() {
      if (this.fContour) return this.fContour;

      var main = this.main_painter(),
          fp = this.frame_painter();
      if (main && (main !== this) && main.fContour) {
         this.fContour = main.fContour;
         this.fCustomContour = main.fCustomContour;
         this.colzmin = main.colzmin;
         this.colzmax = main.colzmax;
         return this.fContour;
      }

      // if not initialized, first create contour array
      // difference from ROOT - fContour includes also last element with maxbin, which makes easier to build logz
      var histo = this.GetObject(), nlevels = JSROOT.gStyle.fNumberContours,
          zmin = this.minbin, zmax = this.maxbin, zminpos = this.minposbin;
      if (zmin === zmax) { zmin = this.gminbin; zmax = this.gmaxbin; zminpos = this.gminposbin }
      if (histo.fContour) nlevels = histo.fContour.length;
      if ((this.options.minimum !== -1111) && (this.options.maximum != -1111)) {
         zmin = this.options.minimum;
         zmax = this.options.maximum;
      }
      if (fp && (fp.zoom_zmin != fp.zoom_zmax)) {
         zmin = fp.zoom_zmin;
         zmax = fp.zoom_zmax;
      }

      if (histo.fContour && (histo.fContour.length>1) && histo.TestBit(JSROOT.TH1StatusBits.kUserContour)) {
         this.fContour = JSROOT.clone(histo.fContour);
         this.fCustomContour = true;
         this.colzmin = zmin;
         this.colzmax = zmax;
         if (zmax > this.fContour[this.fContour.length-1]) this.fContour.push(zmax);
         if ((this.Dimension()<3) && fp) {
            fp.zmin = this.colzmin;
            fp.zmax = this.colzmax;
         }
         return this.fContour;
      }

      this.fCustomContour = false;

      return this.CreateContour(nlevels, zmin, zmax, zminpos);
   }

   /// return index from contours array, which corresponds to the content value **zc**
   THistPainter.prototype.getContourIndex = function(zc) {

      var cntr = this.GetContour();

      if (this.fCustomContour) {
         var l = 0, r = cntr.length-1, mid;
         if (zc < cntr[0]) return -1;
         if (zc >= cntr[r]) return r;
         while (l < r-1) {
            mid = Math.round((l+r)/2);
            if (cntr[mid] > zc) r = mid; else l = mid;
         }
         return l;
      }

      // bins less than zmin not drawn
      if (zc < this.colzmin) return this.options.Zero ? -1 : 0;

      // if bin content exactly zmin, draw it when col0 specified or when content is positive
      if (zc===this.colzmin) return ((this.colzmin != 0) || !this.options.Zero || this.IsTH2Poly()) ? 0 : -1;

      return Math.floor(0.01+(zc-this.colzmin)*(cntr.length-1)/(this.colzmax-this.colzmin));
   }

   /// return color from the palette, which corresponds given controur value
   /// optionally one can return color index of the palette
   THistPainter.prototype.getContourColor = function(zc, asindx) {
      var zindx = this.getContourIndex(zc);
      if (zindx < 0) return null;

      var cntr = this.GetContour(),
          palette = this.GetPalette(),
          indx = palette.calcColorIndex(zindx, cntr.length);

      return asindx ? indx : palette.getColor(indx);
   }

   THistPainter.prototype.GetPalette = function(force) {
      if (!this.fPalette || force)
         this.fPalette = this.get_palette(true, this.options.Palette);
      return this.fPalette;
   }

   THistPainter.prototype.FillPaletteMenu = function(menu) {

      var curr = this.options.Palette, hpainter = this;
      if ((curr===null) || (curr===0)) curr = JSROOT.gStyle.Palette;

      function change(arg) {
         hpainter.options.Palette = parseInt(arg);
         hpainter.GetPalette(true);
         hpainter.Redraw(); // redraw histogram
      };

      function add(id, name, more) {
         menu.addchk((id===curr) || more, '<nobr>' + name + '</nobr>', id, change);
      };

      menu.add("sub:Palette");

      add(50, "ROOT 5", (curr>=10) && (curr<51));
      add(51, "Deep Sea");
      add(52, "Grayscale", (curr>0) && (curr<10));
      add(53, "Dark body radiator");
      add(54, "Two-color hue");
      add(55, "Rainbow");
      add(56, "Inverted dark body radiator");
      add(57, "Bird", (curr>113));
      add(58, "Cubehelix");
      add(59, "Green Red Violet");
      add(60, "Blue Red Yellow");
      add(61, "Ocean");
      add(62, "Color Printable On Grey");
      add(63, "Alpine");
      add(64, "Aquamarine");
      add(65, "Army");
      add(66, "Atlantic");

      menu.add("endsub:");
   }

   THistPainter.prototype.DrawColorPalette = function(enabled, postpone_draw, can_move) {
      // only when create new palette, one could change frame size

      if (!this.is_main_painter()) return null;

      var pal = this.FindFunction('TPaletteAxis'),
          pal_painter = this.FindPainterFor(pal);

      if (this._can_move_colz) { can_move = true; delete this._can_move_colz; }

      if (!pal_painter && !pal) {
         pal_painter = this.FindPainterFor(undefined, undefined, "TPaletteAxis");
         if (pal_painter) {
            pal = pal_painter.GetObject();
            // add to list of functions
            this.AddFunction(pal, true);
         }
      }

      if (!enabled) {
         if (pal_painter) {
            pal_painter.Enabled = false;
            pal_painter.RemoveDrawG(); // completely remove drawing without need to redraw complete pad
         }

         return null;
      }

      if (!pal) {

         if (this.options.PadPalette) return null;

         pal = JSROOT.Create('TPave');

         JSROOT.extend(pal, { _typename: "TPaletteAxis", fName: "TPave", fH: null, fAxis: JSROOT.Create('TGaxis'),
                               fX1NDC: 0.91, fX2NDC: 0.95, fY1NDC: 0.1, fY2NDC: 0.9, fInit: 1 } );

         var zaxis = this.GetHisto().fZaxis;

         JSROOT.extend(pal.fAxis, { fTitle: zaxis.fTitle, fTitleSize: zaxis.fTitleSize, fChopt: "+",
                                    fLineColor: zaxis.fAxisColor, fLineSyle: 1, fLineWidth: 1,
                                    fTextAngle: 0, fTextSize: zaxis.fLabelSize, fTextAlign: 11,
                                    fTextColor: zaxis.fLabelColor, fTextFont: zaxis.fLabelFont });

         // place colz in the beginning, that stat box is always drawn on the top
         this.AddFunction(pal, true);

         can_move = true;
      }

      var frame_painter = this.frame_painter();

      // keep palette width
      if (can_move && frame_painter) {
         pal.fX2NDC = frame_painter.fX2NDC + 0.01 + (pal.fX2NDC - pal.fX1NDC);
         pal.fX1NDC = frame_painter.fX2NDC + 0.01;
         pal.fY1NDC = frame_painter.fY1NDC;
         pal.fY2NDC = frame_painter.fY2NDC;
      }

      var arg = "";
      if (postpone_draw) arg+=";postpone";
      if (can_move && !this.do_redraw_palette) arg+=";can_move";

      if (!pal_painter) {
         // when histogram drawn on sub pad, let draw new axis object on the same pad
         var prev = this.CurrentPadName(this.pad_name);
         // CAUTION!!! This is very special place where return value of JSROOT.draw is allowed
         // while palette drawing is in same script. Normally callback should be used
         pal_painter = JSROOT.draw(this.divid, pal, arg);
         this.CurrentPadName(prev);
      } else {
         pal_painter.Enabled = true;
         pal_painter.DrawPave(arg);
      }

      // mark painter as secondary - not in list of TCanvas primitives
      pal_painter.$secondary = true;

      // make dummy redraw, palette will be updated only from histogram painter
      pal_painter.Redraw = function() {};

      // special code to adjust frame position to actual position of palette
      if (can_move && frame_painter && (pal.fX1NDC-0.005 < frame_painter.fX2NDC) && !this.do_redraw_palette) {

         this.do_redraw_palette = true;

         frame_painter.fX2NDC = pal.fX1NDC - 0.01;
         frame_painter.Redraw();
         // here we should redraw main object
         if (!postpone_draw) this.Redraw();

         delete this.do_redraw_palette;
      }

      return pal_painter;
   }

   THistPainter.prototype.ToggleColz = function() {
      var can_toggle = this.options.Mode3D ? (this.options.Lego === 12 || this.options.Lego === 14 || this.options.Surf === 11 || this.options.Surf === 12) :
                       this.options.Color || this.options.Contour;

      if (can_toggle) {
         this.options.Zscale = !this.options.Zscale;
         this.DrawColorPalette(this.options.Zscale, false, true);
      }
   }

   THistPainter.prototype.ToggleMode3D = function() {
      this.options.Mode3D = !this.options.Mode3D;

      if (this.options.Mode3D) {
         if (!this.options.Surf && !this.options.Lego && !this.options.Error) {
            if ((this.nbinsx>=50) || (this.nbinsy>=50))
               this.options.Lego = this.options.Color ? 14 : 13;
            else
               this.options.Lego = this.options.Color ? 12 : 1;

            this.options.Zero = false; // do not show zeros by default
         }
      }

      this.CopyOptionsToOthers();
      this.InteractiveRedraw("pad","drawopt");
   }

   THistPainter.prototype.PrepareColorDraw = function(args) {

      if (!args) args = { rounding: true, extra: 0, middle: 0 };

      if (args.extra === undefined) args.extra = 0;
      if (args.middle === undefined) args.middle = 0;

      var histo = this.GetHisto(),
          xaxis = histo.fXaxis, yaxis = histo.fYaxis,
          pmain = this.frame_painter(),
          hdim = this.Dimension(),
          i, j, x, y, binz, binarea,
          res = {
             i1: this.GetSelectIndex("x", "left", 0 - args.extra),
             i2: this.GetSelectIndex("x", "right", 1 + args.extra),
             j1: (hdim===1) ? 0 : this.GetSelectIndex("y", "left", 0 - args.extra),
             j2: (hdim===1) ? 1 : this.GetSelectIndex("y", "right", 1 + args.extra),
             min: 0, max: 0, sumz: 0, xbar1: 0, xbar2: 1, ybar1: 0, ybar2: 1
          };

      res.grx = new Float32Array(res.i2+1);
      res.gry = new Float32Array(res.j2+1);

      if (typeof histo.fBarOffset == 'number' && typeof histo.fBarWidth == 'number'
           && (histo.fBarOffset || histo.fBarWidth !== 1000)) {
             if (histo.fBarOffset <= 1000) {
                res.xbar1 = res.ybar1 = 0.001*histo.fBarOffset;
             } else if (histo.fBarOffset <= 3000) {
                res.xbar1 = 0.001*(histo.fBarOffset-2000);
             } else if (histo.fBarOffset <= 5000) {
                res.ybar1 = 0.001*(histo.fBarOffset-4000);
             }

             if (histo.fBarWidth <= 1000) {
                res.xbar2 = Math.min(1., res.xbar1 + 0.001*histo.fBarWidth);
                res.ybar2 = Math.min(1., res.ybar1 + 0.001*histo.fBarWidth);
             } else if (histo.fBarWidth <= 3000) {
                res.xbar2 = Math.min(1., res.xbar1 + 0.001*(histo.fBarWidth-2000));
                // res.ybar2 = res.ybar1 + 1;
             } else if (histo.fBarWidth <= 5000) {
                // res.xbar2 = res.xbar1 + 1;
                res.ybar2 = Math.min(1., res.ybar1 + 0.001*(histo.fBarWidth-4000));
             }
         }

      if (args.original) {
         res.original = true;
         res.origx = new Float32Array(res.i2+1);
         res.origy = new Float32Array(res.j2+1);
      }

      if (args.pixel_density) args.rounding = true;

      if (!pmain) {
         console.warn("cannot draw histogram without frame");
         return res;
      }

       // calculate graphical coordinates in advance
      for (i = res.i1; i <= res.i2; ++i) {
         x = xaxis.GetBinCoord(i + args.middle);
         if (pmain.logx && (x <= 0)) { res.i1 = i+1; continue; }
         if (res.origx) res.origx[i] = x;
         res.grx[i] = pmain.grx(x);
         if (args.rounding) res.grx[i] = Math.round(res.grx[i]);

         if (args.use3d) {
            if (res.grx[i] < -pmain.size_xy3d) { res.i1 = i; res.grx[i] = -pmain.size_xy3d; }
            if (res.grx[i] > pmain.size_xy3d) { res.i2 = i; res.grx[i] = pmain.size_xy3d; }
         }
      }

      if (hdim===1) {
         res.gry[0] = pmain.gry(0);
         res.gry[1] = pmain.gry(1);
      } else
      for (j = res.j1; j <= res.j2; ++j) {
         y = yaxis.GetBinCoord(j + args.middle);
         if (pmain.logy && (y <= 0)) { res.j1 = j+1; continue; }
         if (res.origy) res.origy[j] = y;
         res.gry[j] = pmain.gry(y);
         if (args.rounding) res.gry[j] = Math.round(res.gry[j]);

         if (args.use3d) {
            if (res.gry[j] < -pmain.size_xy3d) { res.j1 = j; res.gry[j] = -pmain.size_xy3d; }
            if (res.gry[j] > pmain.size_xy3d) { res.j2 = j; res.gry[j] = pmain.size_xy3d; }
         }
      }

      //  find min/max values in selected range

      binz = histo.getBinContent(res.i1 + 1, res.j1 + 1);
      this.maxbin = this.minbin = this.minposbin = null;

      for (i = res.i1; i < res.i2; ++i) {
         for (j = res.j1; j < res.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            res.sumz += binz;
            if (args.pixel_density) {
               binarea = (res.grx[i+1]-res.grx[i])*(res.gry[j]-res.gry[j+1]);
               if (binarea <= 0) continue;
               res.max = Math.max(res.max, binz);
               if ((binz>0) && ((binz<res.min) || (res.min===0))) res.min = binz;
               binz = binz/binarea;
            }
            if (this.maxbin===null) {
               this.maxbin = this.minbin = binz;
            } else {
               this.maxbin = Math.max(this.maxbin, binz);
               this.minbin = Math.min(this.minbin, binz);
            }
            if (binz > 0)
               if ((this.minposbin===null) || (binz<this.minposbin)) this.minposbin = binz;
         }
      }

      // force recalculation of z levels
      this.fContour = null;
      this.fCustomContour = false;

      return res;
   }

   // ========================================================================

   /**
    * @summary Painter for TH1 classes
    *
    * @constructor
    * @memberof JSROOT
    * @arguments JSROOT.THistPainter
    * @param {object} histo - histogram object
    */

   function TH1Painter(histo) {
      THistPainter.call(this, histo);
   }

   TH1Painter.prototype = Object.create(THistPainter.prototype);

   TH1Painter.prototype.ConvertTH1K = function() {
      var histo = this.GetObject();

      if (histo.fReady) return;

      var arr = histo.fArray, entries = histo.fEntries; // array of values
      histo.fNcells = histo.fXaxis.fNbins + 2;
      histo.fArray = new Float64Array(histo.fNcells);
      for (var n=0;n<histo.fNcells;++n) histo.fArray[n] = 0;
      for (var n=0;n<histo.fNIn;++n) histo.Fill(arr[n]);
      histo.fReady = true;
      histo.fEntries = entries;
   }

   /** @summary Scan content of 1-D histogram
    * @desc Detect min/max values for x and y axis
    * @param when_axis_changed - true when only zooming was changed, some checks may be skipped */
   TH1Painter.prototype.ScanContent = function(when_axis_changed) {

      if (when_axis_changed && !this.nbinsx) when_axis_changed = false;

      if (this.IsTH1K()) this.ConvertTH1K();

      var histo = this.GetHisto();

      if (!when_axis_changed) {
         this.nbinsx = histo.fXaxis.fNbins;
         this.nbinsy = 0;
         this.CreateAxisFuncs(false);
      }

      var left = this.GetSelectIndex("x", "left"),
          right = this.GetSelectIndex("x", "right");

      if (when_axis_changed) {
         if ((left === this.scan_xleft) && (right === this.scan_xright)) return;
      }

      // Paint histogram axis only
      this.draw_content = !(this.options.Axis > 0);

      this.scan_xleft = left;
      this.scan_xright = right;

      var hmin = 0, hmin_nz = 0, hmax = 0, hsum = 0, first = true,
          profile = this.IsTProfile(), value, err;

      for (var i = 0; i < this.nbinsx; ++i) {
         value = histo.getBinContent(i + 1);
         hsum += profile ? histo.fBinEntries[i + 1] : value;

         if ((i<left) || (i>=right)) continue;

         if ((value > 0) && ((hmin_nz == 0) || (value < hmin_nz))) hmin_nz = value;

         if (first) {
            hmin = hmax = value;
            first = false;
         }

         err = this.options.Error ? histo.getBinError(i + 1) : 0;

         hmin = Math.min(hmin, value - err);
         hmax = Math.max(hmax, value + err);
      }

      // account overflow/underflow bins
      if (profile)
         hsum += histo.fBinEntries[0] + histo.fBinEntries[this.nbinsx + 1];
      else
         hsum += histo.getBinContent(0) + histo.getBinContent(this.nbinsx + 1);

      this.stat_entries = hsum;
      if (histo.fEntries > 1) this.stat_entries = histo.fEntries;

      this.hmin = hmin;
      this.hmax = hmax;

      this.ymin_nz = hmin_nz; // value can be used to show optimal log scale

      if ((this.nbinsx == 0) || ((Math.abs(hmin) < 1e-300) && (Math.abs(hmax) < 1e-300)))
         this.draw_content = false;

      var set_zoom = false, set_zoom2 = false;

      if (this.draw_content) {
         if (hmin >= hmax) {
            if (hmin == 0) { this.ymin = 0; this.ymax = 1; }
            else if (hmin < 0) { this.ymin = 2 * hmin; this.ymax = 0; }
            else { this.ymin = 0; this.ymax = hmin * 2; }
         } else {
            var dy = (hmax - hmin) * 0.05;
            this.ymin = hmin - dy;
            if ((this.ymin < 0) && (hmin >= 0)) this.ymin = 0;
            this.ymax = hmax + dy;
         }
      } else if (this.options.ymin !== this.options.ymax) {
         this.ymin = this.options.ymin;
         this.ymax = this.options.ymax;
         set_zoom2 = true;
      }

      hmin = this.options.minimum;
      hmax = this.options.maximum;

      if ((hmin === hmax) && (hmin !== -1111)) {
         if (hmin < 0) {
            hmin *= 2; hmax = 0;
         } else {
            hmin = 0; hmax*=2; if (!hmax) hmax = 1;
         }
      }

      if ((hmin != -1111) && (hmax != -1111) && !this.draw_content && !set_zoom2) {
         this.ymin = hmin;
         this.ymax = hmax;
      } else {
         if (hmin != -1111) {
            if (hmin < this.ymin) this.ymin = hmin; else set_zoom = true;
         }
         if (hmax != -1111) {
            if (hmax > this.ymax) this.ymax = hmax; else set_zoom = true;
         }
      }

      if (!when_axis_changed) {
         if (set_zoom && (this.draw_content || set_zoom2)) {
            this.zoom_ymin = (hmin == -1111) ? this.ymin : hmin;
            this.zoom_ymax = (hmax == -1111) ? this.ymax : hmax;
         } else {
            delete this.zoom_ymin;
            delete this.zoom_ymax;
         }
      }

      // used in AllowDefaultYZooming
      if (this.Dimension() > 1) this.wheel_zoomy = true; else
      if (this.draw_content) this.wheel_zoomy = false;
   }

   /** @summary Count histogram statistic @private */
   TH1Painter.prototype.CountStat = function(cond) {
      var profile = this.IsTProfile(),
          histo = this.GetHisto(), xaxis = histo.fXaxis,
          left = this.GetSelectIndex("x", "left"),
          right = this.GetSelectIndex("x", "right"),
          stat_sumw = 0, stat_sumwx = 0, stat_sumwx2 = 0, stat_sumwy = 0, stat_sumwy2 = 0,
          i, xx = 0, w = 0, xmax = null, wmax = null,
          fp = this.frame_painter(),
          res = { name: histo.fName, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, integral: 0, entries: this.stat_entries, xmax:0, wmax:0 };

      for (i = left; i < right; ++i) {
         xx = xaxis.GetBinCoord(i + 0.5);

         if (cond && !cond(xx)) continue;

         if (profile) {
            w = histo.fBinEntries[i + 1];
            stat_sumwy += histo.fArray[i + 1];
            stat_sumwy2 += histo.fSumw2[i + 1];
         } else {
            w = histo.getBinContent(i + 1);
         }

         if ((xmax===null) || (w>wmax)) { xmax = xx; wmax = w; }

         stat_sumw += w;
         stat_sumwx += w * xx;
         stat_sumwx2 += w * xx * xx;
      }

      // when no range selection done, use original statistic from histogram
      if (!fp.IsAxisZoomed("x") && (histo.fTsumw>0)) {
         stat_sumw = histo.fTsumw;
         stat_sumwx = histo.fTsumwx;
         stat_sumwx2 = histo.fTsumwx2;
      }

      res.integral = stat_sumw;

      if (stat_sumw > 0) {
         res.meanx = stat_sumwx / stat_sumw;
         res.meany = stat_sumwy / stat_sumw;
         res.rmsx = Math.sqrt(Math.abs(stat_sumwx2 / stat_sumw - res.meanx * res.meanx));
         res.rmsy = Math.sqrt(Math.abs(stat_sumwy2 / stat_sumw - res.meany * res.meany));
      }

      if (xmax!==null) {
         res.xmax = xmax;
         res.wmax = wmax;
      }

      return res;
   }

   /** @summary Fill stat box @private */
   TH1Painter.prototype.FillStatistic = function(stat, dostat, dofit) {

      // no need to refill statistic if histogram is dummy
      if (this.IgnoreStatsFill()) return false;

      var histo = this.GetHisto(),
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

      // make empty at the beginning
      stat.ClearPave();

      if (print_name > 0)
         stat.AddText(data.name);

      if (this.IsTProfile()) {

         if (print_entries > 0)
            stat.AddText("Entries = " + stat.Format(data.entries,"entries"));

         if (print_mean > 0) {
            stat.AddText("Mean = " + stat.Format(data.meanx));
            stat.AddText("Mean y = " + stat.Format(data.meany));
         }

         if (print_rms > 0) {
            stat.AddText("Std Dev = " + stat.Format(data.rmsx));
            stat.AddText("Std Dev y = " + stat.Format(data.rmsy));
         }

      } else {

         if (print_entries > 0)
            stat.AddText("Entries = " + stat.Format(data.entries, "entries"));

         if (print_mean > 0)
            stat.AddText("Mean = " + stat.Format(data.meanx));

         if (print_rms > 0)
            stat.AddText("Std Dev = " + stat.Format(data.rmsx));

         if (print_under > 0)
            stat.AddText("Underflow = " + stat.Format((histo.fArray.length > 0) ? histo.fArray[0] : 0, "entries"));

         if (print_over > 0)
            stat.AddText("Overflow = " + stat.Format((histo.fArray.length > 0) ? histo.fArray[histo.fArray.length - 1] : 0, "entries"));

         if (print_integral > 0)
            stat.AddText("Integral = " + stat.Format(data.integral, "entries"));

         if (print_skew > 0)
            stat.AddText("Skew = <not avail>");

         if (print_kurt > 0)
            stat.AddText("Kurt = <not avail>");
      }

      if (dofit) stat.FillFunctionStat(this.FindFunction('TF1'), dofit);

      return true;
   }

   /** @summary Draw histogram as bars @private */
   TH1Painter.prototype.DrawBars = function(width, height) {

      this.CreateG(true);

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          pmain = this.frame_painter(),
          pad = this.root_pad(),
          histo = this.GetHisto(), xaxis = histo.fXaxis,
          pthis = this,
          show_text = this.options.Text, text_col, text_angle, text_size,
          i, x1, x2, grx1, grx2, y, gry1, gry2, w,
          bars = "", barsl = "", barsr = "",
          side = (this.options.BarStyle > 10) ? this.options.BarStyle % 10 : 0;

      if (side>4) side = 4;
      gry2 = pmain.swap_xy ? 0 : height;
      if ((this.options.BaseLine !== false) && !isNaN(this.options.BaseLine))
         if (this.options.BaseLine >= pmain.scale_ymin)
            gry2 = Math.round(pmain.gry(this.options.BaseLine));

      if (show_text) {
         text_col = this.get_color(histo.fMarkerColor);
         text_angle = -1*this.options.TextAngle;
         text_size = 20;

         if ((histo.fMarkerSize!==1) && text_angle)
            text_size = 0.02*height*histo.fMarkerSize;

         this.StartTextDrawing(42, text_size, this.draw_g, text_size);
      }


      for (i = left; i < right; ++i) {
         x1 = xaxis.GetBinLowEdge(i+1);
         x2 = xaxis.GetBinLowEdge(i+2);

         if (pmain.logx && (x2 <= 0)) continue;

         grx1 = Math.round(pmain.grx(x1));
         grx2 = Math.round(pmain.grx(x2));

         y = histo.getBinContent(i+1);
         if (pmain.logy && (y < pmain.scale_ymin)) continue;
         gry1 = Math.round(pmain.gry(y));

         w = grx2 - grx1;
         grx1 += Math.round(histo.fBarOffset/1000*w);
         w = Math.round(histo.fBarWidth/1000*w);

         if (pmain.swap_xy)
            bars += "M"+gry2+","+grx1 + "h"+(gry1-gry2) + "v"+w + "h"+(gry2-gry1) + "z";
         else
            bars += "M"+grx1+","+gry1 + "h"+w + "v"+(gry2-gry1) + "h"+(-w)+ "z";

         if (side > 0) {
            grx2 = grx1 + w;
            w = Math.round(w * side / 10);
            if (pmain.swap_xy) {
               barsl += "M"+gry2+","+grx1 + "h"+(gry1-gry2) + "v" + w + "h"+(gry2-gry1) + "z";
               barsr += "M"+gry2+","+grx2 + "h"+(gry1-gry2) + "v" + (-w) + "h"+(gry2-gry1) + "z";
            } else {
               barsl += "M"+grx1+","+gry1 + "h"+w + "v"+(gry2-gry1) + "h"+(-w)+ "z";
               barsr += "M"+grx2+","+gry1 + "h"+(-w) + "v"+(gry2-gry1) + "h"+w + "z";
            }
         }

         if (show_text && y) {
            var lbl = (y === Math.round(y)) ? y.toString() : JSROOT.FFormat(y, JSROOT.gStyle.fPaintTextFormat);

            if (pmain.swap_xy)
               this.DrawText({ align: 12, x: Math.round(gry1 + text_size/2), y: Math.round(grx1+0.1), height: Math.round(w*0.8), text: lbl, color: text_col, latex: 0 });
            else if (text_angle)
               this.DrawText({ align: 12, x: grx1+w/2, y: Math.round(gry1 - 2 - text_size/5), width: 0, height: 0, rotate: text_angle, text: lbl, color: text_col, latex: 0 });
            else
               this.DrawText({ align: 22, x: Math.round(grx1 + w*0.1), y: Math.round(gry1-2-text_size), width: Math.round(w*0.8), height: text_size, text: lbl, color: text_col, latex: 0 });
         }
      }

      if (bars)
         this.draw_g.append("svg:path")
                    .attr("d", bars)
                    .call(this.fillatt.func);

      if (barsl)
         this.draw_g.append("svg:path")
               .attr("d", barsl)
               .call(this.fillatt.func)
               .style("fill", d3.rgb(this.fillatt.color).brighter(0.5).toString());

      if (barsr)
         this.draw_g.append("svg:path")
               .attr("d", barsr)
               .call(this.fillatt.func)
               .style("fill", d3.rgb(this.fillatt.color).darker(0.5).toString());

      if (show_text)
         this.FinishTextDrawing(this.draw_g);
   }

   TH1Painter.prototype.DrawFilledErrors = function(width, height) {
      this.CreateG(true);

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          pmain = this.frame_painter(),
          histo = this.GetHisto(), xaxis = histo.fXaxis,
          i, x, grx, y, yerr, gry1, gry2,
          bins1 = [], bins2 = [];

      for (i = left; i < right; ++i) {
         x = xaxis.GetBinCoord(i+0.5);
         if (pmain.logx && (x <= 0)) continue;
         grx = Math.round(pmain.grx(x));

         y = histo.getBinContent(i+1);
         yerr = histo.getBinError(i+1);
         if (pmain.logy && (y-yerr < pmain.scale_ymin)) continue;

         gry1 = Math.round(pmain.gry(y + yerr));
         gry2 = Math.round(pmain.gry(y - yerr));

         bins1.push({ grx:grx, gry: gry1 });
         bins2.unshift({ grx:grx, gry: gry2 });
      }

      var kind = (this.options.ErrorKind === 4) ? "bezier" : "line",
          path1 = JSROOT.Painter.BuildSvgPath(kind, bins1),
          path2 = JSROOT.Painter.BuildSvgPath("L"+kind, bins2);

      this.draw_g.append("svg:path")
                 .attr("d", path1.path + path2.path + "Z")
                 .style("stroke", "none")
                 .call(this.fillatt.func);
   }

   TH1Painter.prototype.DrawBins = function() {
      // new method, create svg:path expression ourself directly from histogram
      // all points will be used, compress expression when too large

      this.CheckHistDrawAttributes();

      var width = this.frame_width(), height = this.frame_height();

      if (!this.draw_content || (width<=0) || (height<=0))
          return this.RemoveDrawG();

      if (this.options.Bar)
         return this.DrawBars(width, height);

      if ((this.options.ErrorKind === 3) || (this.options.ErrorKind === 4))
         return this.DrawFilledErrors(width, height);

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 2),
          pmain = this.frame_painter(),
          pad = this.root_pad(),
          histo = this.GetHisto(),
          xaxis = histo.fXaxis,
          pthis = this,
          res = "", lastbin = false,
          startx, currx, curry, x, grx, y, gry, curry_min, curry_max, prevy, prevx, i, bestimin, bestimax,
          exclude_zero = !this.options.Zero,
          show_errors = this.options.Error,
          show_markers = this.options.Mark,
          show_line = this.options.Line || this.options.Curve,
          show_text = this.options.Text,
          text_profile = show_text && (this.options.TextKind == "E") && this.IsTProfile() && histo.fBinEntries,
          path_fill = null, path_err = null, path_marker = null, path_line = null,
          do_marker = false, do_err = false,
          endx = "", endy = "", dend = 0, my, yerr1, yerr2, bincont, binerr, mx1, mx2, midx, mmx1, mmx2,
          mpath = "", text_col, text_angle, text_size;

      if (show_errors && !show_markers && (histo.fMarkerStyle > 1))
         show_markers = true;

      if (this.options.ErrorKind === 2) {
         if (this.fillatt.empty()) show_markers = true;
                              else path_fill = "";
      } else if (this.options.Error) {
         path_err = "";
         do_err = true;
      }

      if (show_line) path_line = "";

      if (show_markers) {
         // draw markers also when e2 option was specified
         this.createAttMarker({ attr: histo, style: this.options.MarkStyle }); // when style not configured, it will be ignored
         if (this.markeratt.size > 0) {
            // simply use relative move from point, can optimize in the future
            path_marker = "";
            do_marker = true;
            this.markeratt.reset_pos();
         } else {
            show_markers = false;
         }
      }

      var draw_markers = show_errors || show_markers,
          draw_any_but_hist = draw_markers || show_text || show_line,
          draw_hist = this.options.Hist && (!this.lineatt.empty() || !this.fillatt.empty());

      if (!draw_hist && !draw_any_but_hist)
         return this.RemoveDrawG();

      this.CreateG(true);

      if (show_text) {
         text_col = this.get_color(histo.fMarkerColor);
         text_angle = -1*this.options.TextAngle;
         text_size = 20;

         if ((histo.fMarkerSize!==1) && text_angle)
            text_size = 0.02*height*histo.fMarkerSize;

         if (!text_angle && !this.options.TextKind) {
             var space = width / (right - left + 1);
             if (space < 3 * text_size) {
                text_angle = 270;
                text_size = Math.round(space*0.7);
             }
         }

         this.StartTextDrawing(42, text_size, this.draw_g, text_size);
      }

      // if there are too many points, exclude many vertical drawings at the same X position
      // instead define min and max value and made min-max drawing
      var use_minmax = ((right-left) > 3*width);

      if (this.options.ErrorKind === 1) {
         var lw = this.lineatt.width + JSROOT.gStyle.fEndErrorSize;
         endx = "m0," + lw + "v-" + 2*lw + "m0," + lw;
         endy = "m" + lw + ",0h-" + 2*lw + "m" + lw + ",0";
         dend = Math.floor((this.lineatt.width-1)/2);
      }

      if (draw_any_but_hist) use_minmax = true;

      // just to get correct values for the specified bin
      function extract_bin(bin) {
         bincont = histo.getBinContent(bin+1);
         if (exclude_zero && (bincont===0)) return false;
         mx1 = Math.round(pmain.grx(xaxis.GetBinLowEdge(bin+1)));
         mx2 = Math.round(pmain.grx(xaxis.GetBinLowEdge(bin+2)));
         midx = Math.round((mx1+mx2)/2);
         my = Math.round(pmain.gry(bincont));
         yerr1 = yerr2 = 20;
         if (show_errors) {
            binerr = histo.getBinError(bin+1);
            yerr1 = Math.round(my - pmain.gry(bincont + binerr)); // up
            yerr2 = Math.round(pmain.gry(bincont - binerr) - my); // down
         }
         return true;
      }

      function draw_errbin(bin) {
         if (pthis.options.errorX > 0) {
            mmx1 = Math.round(midx - (mx2-mx1)*pthis.options.errorX);
            mmx2 = Math.round(midx + (mx2-mx1)*pthis.options.errorX);
            path_err += "M" + (mmx1+dend) +","+ my + endx + "h" + (mmx2-mmx1-2*dend) + endx;
         }
         path_err += "M" + midx +"," + (my-yerr1+dend) + endy + "v" + (yerr1+yerr2-2*dend) + endy;
      }

      function draw_bin(bin) {
         if (extract_bin(bin)) {
            if (show_text) {
               var cont = text_profile ? histo.fBinEntries[bin+1] : bincont;

               if (cont!==0) {
                  var lbl = (cont === Math.round(cont)) ? cont.toString() : JSROOT.FFormat(cont, JSROOT.gStyle.fPaintTextFormat);

                  if (text_angle)
                     pthis.DrawText({ align: 12, x: midx, y: Math.round(my - 2 - text_size/5), width: 0, height: 0, rotate: text_angle, text: lbl, color: text_col, latex: 0 });
                  else
                     pthis.DrawText({ align: 22, x: Math.round(mx1 + (mx2-mx1)*0.1), y: Math.round(my-2-text_size), width: Math.round((mx2-mx1)*0.8), height: text_size, text: lbl, color: text_col, latex: 0 });
               }
            }

            if (show_line && (path_line !== null))
               path_line += ((path_line.length===0) ? "M" : "L") + midx + "," + my;

            if (draw_markers) {
               if ((my >= -yerr1) && (my <= height + yerr2)) {
                  if (path_fill !== null)
                     path_fill += "M" + mx1 +","+(my-yerr1) +
                                  "h" + (mx2-mx1) + "v" + (yerr1+yerr2+1) + "h-" + (mx2-mx1) + "z";
                  if ((path_marker !== null) && do_marker)
                     path_marker += pthis.markeratt.create(midx, my);
                  if ((path_err !== null) && do_err)
                     draw_errbin(bin);
               }
            }
         }
      }

      // check if we should draw markers or error marks directly, skipping optimization
      if (do_marker || do_err)
         if (!JSROOT.gStyle.OptimizeDraw || ((right-left<50000) && (JSROOT.gStyle.OptimizeDraw==1))) {
            for (i = left; i < right; ++i) {
               if (extract_bin(i)) {
                  if (path_marker !== null)
                     path_marker += pthis.markeratt.create(midx, my);
                  if (path_err !== null)
                     draw_errbin(i);
               }
            }
            do_err = do_marker = false;
         }


      for (i = left; i <= right; ++i) {

         x = xaxis.GetBinLowEdge(i+1);

         if (this.logx && (x <= 0)) continue;

         grx = Math.round(pmain.grx(x));

         lastbin = (i === right);

         if (lastbin && (left<right)) {
            gry = curry;
         } else {
            y = histo.getBinContent(i+1);
            gry = Math.round(pmain.gry(y));
         }

         if (res.length === 0) {
            bestimin = bestimax = i;
            prevx = startx = currx = grx;
            prevy = curry_min = curry_max = curry = gry;
            res = "M"+currx+","+curry;
         } else if (use_minmax) {
            if ((grx === currx) && !lastbin) {
               if (gry < curry_min) bestimax = i; else
               if (gry > curry_max) bestimin = i;

               curry_min = Math.min(curry_min, gry);
               curry_max = Math.max(curry_max, gry);
               curry = gry;
            } else {

               if (draw_any_but_hist) {
                  if (bestimin === bestimax) { draw_bin(bestimin); } else
                  if (bestimin < bestimax) { draw_bin(bestimin); draw_bin(bestimax); } else {
                     draw_bin(bestimax); draw_bin(bestimin);
                  }
               }

               // when several points at same X differs, need complete logic
               if (draw_hist && ((curry_min !== curry_max) || (prevy !== curry_min))) {

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

               bestimin = bestimax = i;
               curry_min = curry_max = curry = gry;
               currx = grx;
            }
            // end of use_minmax
         } else if ((gry !== curry) || lastbin) {
            if (grx !== currx) res += "h"+(grx-currx);
            if (gry !== curry) res += "v"+(gry-curry);
            curry = gry;
            currx = grx;
         }
      }

      var h0 = height + 3, gry0 = Math.round(pmain.gry(0));
      if (gry0 <= 0) h0 = -3; else if (gry0 < height) h0 = gry0;
      var close_path = "L"+currx+","+h0 + "L"+startx+","+h0 + "Z";

      if (draw_markers || show_line) {
         if ((path_fill !== null) && (path_fill.length > 0))
            this.draw_g.append("svg:path")
                       .attr("d", path_fill)
                       .call(this.fillatt.func);

         if ((path_err !== null) && (path_err.length > 0))
               this.draw_g.append("svg:path")
                   .attr("d", path_err)
                   .call(this.lineatt.func);

         if ((path_line !== null) && (path_line.length > 0)) {
            if (!this.fillatt.empty())
               this.draw_g.append("svg:path")
                     .attr("d", (this.options.Fill ? path_line : res) + close_path)
                     .attr("stroke", "none")
                     .call(this.fillatt.func);

            this.draw_g.append("svg:path")
                   .attr("d", path_line)
                   .attr("fill", "none")
                   .call(this.lineatt.func);
         }

         if ((path_marker !== null) && (path_marker.length > 0))
            this.draw_g.append("svg:path")
                .attr("d", path_marker)
                .call(this.markeratt.func);
      }

      if ((res.length > 0) && draw_hist) {
         if (!this.fillatt.empty())
            res += close_path;
         this.draw_g.append("svg:path")
                    .attr("d", res)
                    .style("stroke-linejoin","miter")
                    .call(this.lineatt.func)
                    .call(this.fillatt.func);
      }

      if (show_text)
         this.FinishTextDrawing(this.draw_g);

   }

   TH1Painter.prototype.GetBinTips = function(bin) {
      var tips = [],
          name = this.GetTipName(),
          pmain = this.frame_painter(),
          histo = this.GetHisto(),
          x1 = histo.fXaxis.GetBinLowEdge(bin+1),
          x2 = histo.fXaxis.GetBinLowEdge(bin+2),
          cont = histo.getBinContent(bin+1),
          xlbl = "", xnormal = false;

      if (name.length>0) tips.push(name);

      if (pmain.x_kind === 'labels') xlbl = pmain.AxisAsText("x", x1); else
      if (pmain.x_kind === 'time') xlbl = pmain.AxisAsText("x", (x1+x2)/2); else
       { xnormal = true; xlbl = "[" + pmain.AxisAsText("x", x1) + ", " + pmain.AxisAsText("x", x2) + ")"; }

      if (this.options.Error || this.options.Mark) {
         tips.push("x = " + xlbl);
         tips.push("y = " + pmain.AxisAsText("y", cont));
         if (this.options.Error) {
            if (xnormal) tips.push("error x = " + ((x2 - x1) / 2).toPrecision(4));
            tips.push("error y = " + histo.getBinError(bin + 1).toPrecision(4));
         }
      } else {
         tips.push("bin = " + (bin+1));
         tips.push("x = " + xlbl);
         if (histo['$baseh']) cont -= histo['$baseh'].getBinContent(bin+1);
         if (cont === Math.round(cont))
            tips.push("entries = " + cont);
         else
            tips.push("entries = " + JSROOT.FFormat(cont, JSROOT.gStyle.fStatFormat));
      }

      return tips;
   }

   /** @summary Process tooltip event @private */
   TH1Painter.prototype.ProcessTooltip = function(pnt) {
      if ((pnt === null) || !this.draw_content || !this.draw_g || this.options.Mode3D) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      var width = this.frame_width(),
          height = this.frame_height(),
          pmain = this.frame_painter(),
          pad = this.root_pad(),
          painter = this,
          histo = this.GetHisto(),
          findbin = null, show_rect = true,
          grx1, midx, grx2, gry1, midy, gry2, gapx = 2,
          left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 2),
          l = left, r = right, pnt_x = pnt.x, pnt_y = pnt.y;

      function GetBinGrX(i) {
         var xx = histo.fXaxis.GetBinLowEdge(i+1);
         return (pmain.logx && (xx<=0)) ? null : pmain.grx(xx);
      }

      function GetBinGrY(i) {
         var yy = histo.getBinContent(i + 1);
         if (pmain.logy && (yy < pmain.scale_ymin))
            return pmain.swap_xy ? -1000 : 10*height;
         return Math.round(pmain.gry(yy));
      }

      if (pmain.swap_xy) {
         var d = pnt.x; pnt_x = pnt_y; pnt_y = d;
         d = height; height = width; width = d;
      }

      while (l < r-1) {
         var m = Math.round((l+r)*0.5), xx = GetBinGrX(m);
         if ((xx === null) || (xx < pnt_x - 0.5)) {
            if (pmain.swap_xy) r = m; else l = m;
         } else if (xx > pnt_x + 0.5) {
            if (pmain.swap_xy) l = m; else r = m;
         } else { l++; r--; }
      }

      findbin = r = l;
      grx1 = GetBinGrX(findbin);

      if (pmain.swap_xy) {
         while ((l>left) && (GetBinGrX(l-1) < grx1 + 2)) --l;
         while ((r<right) && (GetBinGrX(r+1) > grx1 - 2)) ++r;
      } else {
         while ((l>left) && (GetBinGrX(l-1) > grx1 - 2)) --l;
         while ((r<right) && (GetBinGrX(r+1) < grx1 + 2)) ++r;
      }

      if (l < r) {
         // many points can be assigned with the same cursor position
         // first try point around mouse y
         var best = height;
         for (var m=l;m<=r;m++) {
            var dist = Math.abs(GetBinGrY(m) - pnt_y);
            if (dist < best) { best = dist; findbin = m; }
         }

         // if best distance still too far from mouse position, just take from between
         if (best > height/10)
            findbin = Math.round(l + (r-l) / height * pnt_y);

         grx1 = GetBinGrX(findbin);
      }

      grx1 = Math.round(grx1);
      grx2 = Math.round(GetBinGrX(findbin+1));

      if (this.options.Bar) {
         var w = grx2 - grx1;
         grx1 += Math.round(histo.fBarOffset/1000*w);
         grx2 = grx1 + Math.round(histo.fBarWidth/1000*w);
      }

      if (grx1 > grx2) { var d = grx1; grx1 = grx2; grx2 = d; }

      midx = Math.round((grx1+grx2)/2);

      midy = gry1 = gry2 = GetBinGrY(findbin);

      if (this.options.Bar) {
         show_rect = true;

         gapx = 0;

         gry1 = Math.round(pmain.gry(((this.options.BaseLine!==false) && (this.options.BaseLine > pmain.scale_ymin)) ? this.options.BaseLine : pmain.scale_ymin));

         if (gry1 > gry2) { var d = gry1; gry1 = gry2; gry2 = d; }

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt_y<gry1) || (pnt_y>gry2)) findbin = null;

      } else if (this.options.Error || this.options.Mark || this.options.Line || this.options.Curve)  {

         show_rect = true;

         var msize = 3;
         if (this.markeratt) msize = Math.max(msize, this.markeratt.GetFullSize());

         if (this.options.Error) {
            var cont = histo.getBinContent(findbin+1),
                binerr = histo.getBinError(findbin+1);

            gry1 = Math.round(pmain.gry(cont + binerr)); // up
            gry2 = Math.round(pmain.gry(cont - binerr)); // down

            if ((cont==0) && this.IsTProfile()) findbin = null;

            var dx = (grx2-grx1)*this.options.errorX;
            grx1 = Math.round(midx - dx);
            grx2 = Math.round(midx + dx);
         }

         // show at least 6 pixels as tooltip rect
         if (grx2 - grx1 < 2*msize) { grx1 = midx-msize; grx2 = midx+msize; }

         gry1 = Math.min(gry1, midy - msize);
         gry2 = Math.max(gry2, midy + msize);

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt_y<gry1) || (pnt_y>gry2)) findbin = null;

      } else {

         // if histogram alone, use old-style with rects
         // if there are too many points at pixel, use circle
         show_rect = (pnt.nproc === 1) && (right-left < width);

         if (show_rect) {
            gry2 = height;

            if (!this.fillatt.empty()) {
               gry2 = Math.round(pmain.gry(0));
               if (gry2 < 0) gry2 = 0; else if (gry2 > height) gry2 = height;
               if (gry2 < gry1) { var d = gry1; gry1 = gry2; gry2 = d; }
            }

            // for mouse events pointer should be between y1 and y2
            if (((pnt.y < gry1) || (pnt.y > gry2)) && !pnt.touch) findbin = null;
         }
      }

      if (findbin!==null) {
         // if bin on boundary found, check that x position is ok
         if ((findbin === left) && (grx1 > pnt_x + gapx))  findbin = null; else
         if ((findbin === right-1) && (grx2 < pnt_x - gapx)) findbin = null; else
         // if bars option used check that bar is not match
         if ((pnt_x < grx1 - gapx) || (pnt_x > grx2 + gapx)) findbin = null; else
         // exclude empty bin if empty bins suppressed
         if (!this.options.Zero && (histo.getBinContent(findbin+1)===0)) findbin = null;
      }

      var ttrect = this.draw_g.select(".tooltip_bin");

      if ((findbin === null) || ((gry2 <= 0) || (gry1 >= height))) {
         ttrect.remove();
         return null;
      }

      var res = { name: histo.fName, title: histo.fTitle,
                  x: midx, y: midy, exact: true,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.fillcoloralt('blue') : 'blue',
                  lines: this.GetBinTips(findbin) };

      if (pnt.disabled) {
         // case when tooltip should not highlight bin

         ttrect.remove();
         res.changed = true;
      } else
      if (show_rect) {

         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:rect")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         res.changed = ttrect.property("current_bin") !== findbin;

         if (res.changed)
            ttrect.attr("x", pmain.swap_xy ? gry1 : grx1)
                  .attr("width", pmain.swap_xy ? gry2-gry1 : grx2-grx1)
                  .attr("y", pmain.swap_xy ? grx1 : gry1)
                  .attr("height", pmain.swap_xy ? grx2-grx1 : gry2-gry1)
                  .style("opacity", "0.3")
                  .property("current_bin", findbin);

         res.exact = (Math.abs(midy - pnt_y) <= 5) || ((pnt_y>=gry1) && (pnt_y<=gry2));

         res.menu = true; // one could show context menu
         // distance to middle point, use to decide which menu to activate
         res.menu_dist = Math.sqrt((midx-pnt_x)*(midx-pnt_x) + (midy-pnt_y)*(midy-pnt_y));

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

      if (res.changed)
         res.user_info = { obj: histo,  name: histo.fName,
                           bin: findbin, cont: histo.getBinContent(findbin+1),
                           grx: midx, gry: midy };

      return res;
   }

   TH1Painter.prototype.FillHistContextMenu = function(menu) {

      menu.add("Auto zoom-in", this.AutoZoom);

      var sett = JSROOT.getDrawSettings("ROOT." + this.GetObject()._typename, 'nosame');

      menu.addDrawMenu("Draw with", sett.opts, function(arg) {
         if (arg==='inspect')
            return this.ShowInspector();

         this.DecodeOptions(arg);

         if (this.options.need_fillcol && this.fillatt && this.fillatt.empty())
            this.fillatt.Change(5,1001);

         // redraw all objects in pad, inform dependent objects
         this.InteractiveRedraw("pad", "drawopt");
      });
   }

   TH1Painter.prototype.AutoZoom = function() {
      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          dist = right - left, histo = this.GetHisto();

      if ((dist == 0) || !histo) return;

      // first find minimum
      var min = histo.getBinContent(left + 1);
      for (var indx = left; indx < right; ++indx)
         min = Math.min(min, histo.getBinContent(indx+1));
      if (min > 0) return; // if all points positive, no chance for autoscale

      while ((left < right) && (histo.getBinContent(left+1) <= min)) ++left;
      while ((left < right) && (histo.getBinContent(right) <= min)) --right;

      // if singular bin
      if ((left === right-1) && (left > 2) && (right < this.nbinsx-2)) {
         --left; ++right;
      }

      if ((right - left < dist) && (left < right))
         this.frame_painter().Zoom(histo.fXaxis.GetBinLowEdge(left+1), histo.fXaxis.GetBinLowEdge(right+1));
   }

   TH1Painter.prototype.CanZoomIn = function(axis,min,max) {
      var histo = this.GetHisto();

      if ((axis=="x") && histo && (histo.fXaxis.FindBin(max,0.5) - histo.fXaxis.FindBin(min,0) > 1)) return true;

      if ((axis=="y") && (Math.abs(max-min) > Math.abs(this.ymax-this.ymin)*1e-6)) return true;

      // check if it makes sense to zoom inside specified axis range
      return false;
   }

   TH1Painter.prototype.CallDrawFunc = function(callback, reason) {

      var main = this.main_painter(),
          fp = this.frame_painter();

     if ((main !== this) && fp && (fp.mode3d !== this.options.Mode3D))
        this.CopyOptionsFrom(main);

      var funcname = this.options.Mode3D ? "Draw3D" : "Draw2D";

      this[funcname](callback, reason);
   }

   TH1Painter.prototype.Draw2D = function(call_back, reason) {
      this.Clear3DScene();
      this.mode3d = false;

      this.ScanContent(true);

      this.CreateXY();

      if (typeof this.DrawColorPalette === 'function')
         this.DrawColorPalette(false);

      this.DrawAxes();
      this.DrawBins();
      this.DrawTitle();
      this.UpdateStatWebCanvas();
      this.AddInteractive();
      JSROOT.CallBack(call_back);
   }

   TH1Painter.prototype.Draw3D = function(call_back, reason) {
      this.mode3d = true;
      JSROOT.AssertPrerequisites('hist3d', function() {
         this.Draw3D(call_back, reason);
      }.bind(this));
   }

   TH1Painter.prototype.Redraw = function(reason) {
      this.CallDrawFunc(null, reason);
   }

   function drawHistogram1D(divid, histo, opt) {
      // create painter and add it to canvas
      var painter = new TH1Painter(histo);

      if (!painter.SetDivId(divid, 1)) return null;

      // here we deciding how histogram will look like and how will be shown
      painter.DecodeOptions(opt);

      painter.CheckPadRange(!painter.options.Mode3D);

      painter.ScanContent();

      painter.CreateStat(); // only when required

      painter.CallDrawFunc(function() {
         painter.DrawNextFunction(0, function() {
            if (!painter.options.Mode3D && painter.options.AutoZoom) painter.AutoZoom();
            painter.FillToolbar();
            painter.DrawingReady();
         });
      });

      return painter;
   }

   // ========================================================================

   /**
    * @summary Painter for TH2 classes
    *
    * @constructor
    * @memberof JSROOT
    * @arguments JSROOT.THistPainter
    * @param {object} histo - histogram object
    */

   function TH2Painter(histo) {
      THistPainter.call(this, histo);
      this.fContour = null; // contour levels
      this.fCustomContour = false; // are this user-defined levels (can be irregular)
      this.fPalette = null;
      this.wheel_zoomy = true;
   }

   TH2Painter.prototype = Object.create(THistPainter.prototype);

   TH2Painter.prototype.Cleanup = function() {
      delete this.fCustomContour;
      delete this.tt_handle;

      THistPainter.prototype.Cleanup.call(this);
   }

   TH2Painter.prototype.ToggleProjection = function(kind, width) {

      if ((kind=="Projections") || (kind=="Off")) kind = "";

      if ((typeof kind == 'string') && (kind.length>1)) {
          width = parseInt(kind.substr(1));
          kind = kind[0];
      }

      if (!width) width = 1;

      if (kind && (this.is_projection==kind)) {
         if (this.projection_width === width) {
            kind = "";
         } else {
            this.projection_width = width;
            return;
         }
      }

      delete this.proj_hist;

      var new_proj = (this.is_projection === kind) ? "" : kind;
      this.is_projection = ""; // disable projection redraw until callback
      this.projection_width = width;

      var canp = this.canv_painter();
      if (canp) canp.ToggleProjection(new_proj, this.RedrawProjection.bind(this, "toggling", new_proj));
   }

   TH2Painter.prototype.RedrawProjection = function(ii1, ii2, jj1, jj2) {
      if (ii1 === "toggling") {
         this.is_projection = ii2;
         ii1 = ii2 = undefined;
      }
      if (!this.is_projection) return;

      if (jj2 == undefined) {
         if (!this.tt_handle) return;
         ii1 = Math.round((this.tt_handle.i1 + this.tt_handle.i2)/2); ii2 = ii1+1;
         jj1 = Math.round((this.tt_handle.j1 + this.tt_handle.j2)/2); jj2 = jj1+1;
      }

      var canp = this.canv_painter(), histo = this.GetHisto();

      if (canp && !canp._readonly && (this.snapid !== undefined)) {
         // this is when projection should be created on the server side
         var exec = "EXECANDSEND:D" + this.is_projection + "PROJ:" + this.snapid + ":";
         if (this.is_projection == "X")
            exec += 'ProjectionX("_projx",' + (jj1+1) + ',' + jj2 + ',"")';
         else
            exec += 'ProjectionY("_projy",' + (ii1+1) + ',' + ii2 + ',"")';
         canp.SendWebsocket(exec);
         return;
      }

      if (!this.proj_hist) {
         if (this.is_projection == "X") {
            this.proj_hist = JSROOT.CreateHistogram("TH1D", this.nbinsx);
            JSROOT.extend(this.proj_hist.fXaxis, histo.fXaxis);
            this.proj_hist.fName = "xproj";
            this.proj_hist.fTitle = "X projection";
         } else {
            this.proj_hist = JSROOT.CreateHistogram("TH1D", this.nbinsy);
            JSROOT.extend(this.proj_hist.fXaxis, histo.fYaxis);
            this.proj_hist.fName = "yproj";
            this.proj_hist.fTitle = "Y projection";
         }
      }

      if (this.is_projection == "X") {
         for (var i=0;i<this.nbinsx;++i) {
            var sum=0;
            for (var j=jj1;j<jj2;++j) sum += histo.getBinContent(i+1,j+1);
            this.proj_hist.setBinContent(i+1, sum);
         }
      } else {
         for (var j=0;j<this.nbinsy;++j) {
            var sum = 0;
            for (var i=ii1;i<ii2;++i) sum += histo.getBinContent(i+1,j+1);
            this.proj_hist.setBinContent(j+1, sum);
         }
      }

      return canp.DrawProjection(this.is_projection, this.proj_hist);
   }

   TH2Painter.prototype.ExecuteMenuCommand = function(method, args) {
      if (THistPainter.prototype.ExecuteMenuCommand.call(this,method, args)) return true;

      if ((method.fName == 'SetShowProjectionX') || (method.fName == 'SetShowProjectionY')) {
         this.ToggleProjection(method.fName[17], args && parseInt(args) ? parseInt(args) : 1);
         return true;
      }

      return false;
   }

   TH2Painter.prototype.FillHistContextMenu = function(menu) {
      // painter automatically bind to menu callbacks

      if (!this.IsTH2Poly()) {
         menu.add("sub:Projections", this.ToggleProjection);
         var kind = this.is_projection || "";
         if (kind) kind += this.projection_width;
         var kinds = ["X1", "X2", "X3", "X5", "X10", "Y1", "Y2", "Y3", "Y5", "Y10"];
         if (this.is_projection) kinds.push("Off");
         for (var k=0;k<kinds.length;++k)
            menu.addchk(kind==kinds[k], kinds[k], kinds[k], this.ToggleProjection);
         menu.add("endsub:");

         menu.add("Auto zoom-in", this.AutoZoom);
      }

      var sett = JSROOT.getDrawSettings("ROOT." + this.GetObject()._typename, 'nosame');

      menu.addDrawMenu("Draw with", sett.opts, function(arg) {
         if (arg==='inspect')
            return this.ShowInspector();
         this.DecodeOptions(arg);
         this.InteractiveRedraw("pad", "drawopt");
      });

      if (this.options.Color)
         this.FillPaletteMenu(menu);
   }

   TH2Painter.prototype.ButtonClick = function(funcname) {
      if (THistPainter.prototype.ButtonClick.call(this, funcname)) return true;

      if (this !== this.main_painter()) return false;

      switch(funcname) {
         case "ToggleColor": this.ToggleColor(); break;
         case "ToggleColorZ": this.ToggleColz(); break;
         case "Toggle3D": this.ToggleMode3D(); break;
         default: return false;
      }

      // all methods here should not be processed further
      return true;
   }

   TH2Painter.prototype.FillToolbar = function() {
      THistPainter.prototype.FillToolbar.call(this, true);

      var pp = this.pad_painter();
      if (!pp) return;

      if (!this.IsTH2Poly())
         pp.AddButton(JSROOT.ToolbarIcons.th2color, "Toggle color", "ToggleColor");
      pp.AddButton(JSROOT.ToolbarIcons.th2colorz, "Toggle color palette", "ToggleColorZ");
      pp.AddButton(JSROOT.ToolbarIcons.th2draw3d, "Toggle 3D mode", "Toggle3D");
      pp.ShowButtons();
   }

   TH2Painter.prototype.ToggleColor = function() {

      if (this.options.Mode3D) {
         this.options.Mode3D = false;
         this.options.Color = true;
      } else {
         this.options.Color = !this.options.Color;
      }

      this._can_move_colz = true; // indicate that next redraw can move Z scale

      this.CopyOptionsToOthers();

      this.RedrawPad();

      // this.DrawColorPalette(this.options.Color && this.options.Zscale);
   }

   TH2Painter.prototype.AutoZoom = function() {
      if (this.IsTH2Poly()) return; // not implemented

      var i1 = this.GetSelectIndex("x", "left", -1),
          i2 = this.GetSelectIndex("x", "right", 1),
          j1 = this.GetSelectIndex("y", "left", -1),
          j2 = this.GetSelectIndex("y", "right", 1),
          i,j, histo = this.GetObject();

      if ((i1 == i2) || (j1 == j2)) return;

      // first find minimum
      var min = histo.getBinContent(i1 + 1, j1 + 1);
      for (i = i1; i < i2; ++i)
         for (j = j1; j < j2; ++j)
            min = Math.min(min, histo.getBinContent(i+1, j+1));
      if (min > 0) return; // if all points positive, no chance for autoscale

      var ileft = i2, iright = i1, jleft = j2, jright = j1;

      for (i = i1; i < i2; ++i)
         for (j = j1; j < j2; ++j)
            if (histo.getBinContent(i + 1, j + 1) > min) {
               if (i < ileft) ileft = i;
               if (i >= iright) iright = i + 1;
               if (j < jleft) jleft = j;
               if (j >= jright) jright = j + 1;
            }

      var xmin, xmax, ymin, ymax, isany = false;

      if ((ileft === iright-1) && (ileft > i1+1) && (iright < i2-1)) { ileft--; iright++; }
      if ((jleft === jright-1) && (jleft > j1+1) && (jright < j2-1)) { jleft--; jright++; }

      if ((ileft > i1 || iright < i2) && (ileft < iright - 1)) {
         xmin = histo.fXaxis.GetBinLowEdge(ileft+1);
         xmax = histo.fXaxis.GetBinLowEdge(iright+1);
         isany = true;
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = histo.fYaxis.GetBinLowEdge(jleft+1);
         ymax = histo.fYaxis.GetBinLowEdge(jright+1);
         isany = true;
      }

      if (isany)
         this.frame_painter().Zoom(xmin, xmax, ymin, ymax);
   }

   TH2Painter.prototype.ScanContent = function(when_axis_changed) {

      // no need to rescan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy) return;

      var i, j, histo = this.GetObject();

      this.nbinsx = histo.fXaxis.fNbins;
      this.nbinsy = histo.fYaxis.fNbins;

      // used in CreateXY method

      this.CreateAxisFuncs(true);

      if (this.IsTH2Poly()) {
         this.gminposbin = null;
         this.gminbin = this.gmaxbin = 0;

         for (var n=0, len=histo.fBins.arr.length; n<len; ++n) {
            var bin_content = histo.fBins.arr[n].fContent;
            if (n===0) this.gminbin = this.gmaxbin = bin_content;

            if (bin_content < this.gminbin) this.gminbin = bin_content; else
               if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;

            if (bin_content > 0)
               if ((this.gminposbin===null) || (this.gminposbin > bin_content)) this.gminposbin = bin_content;
         }
      } else {
         // global min/max, used at the moment in 3D drawing
         this.gminbin = this.gmaxbin = histo.getBinContent(1, 1);
         this.gminposbin = null;
         for (i = 0; i < this.nbinsx; ++i) {
            for (j = 0; j < this.nbinsy; ++j) {
               var bin_content = histo.getBinContent(i+1, j+1);
               if (bin_content < this.gminbin) this.gminbin = bin_content; else
                  if (bin_content > this.gmaxbin) this.gmaxbin = bin_content;
               if (bin_content > 0)
                  if ((this.gminposbin===null) || (this.gminposbin > bin_content)) this.gminposbin = bin_content;
            }
         }
      }

      // this value used for logz scale drawing
      if (this.gminposbin === null) this.gminposbin = this.gmaxbin*1e-4;

      if (this.options.Axis > 0) { // Paint histogram axis only
         this.draw_content = false;
      } else {
         this.draw_content = (this.gmaxbin > 0);
         if (!this.draw_content  && this.options.Zero && this.IsTH2Poly()) {
            this.draw_content = true;
            this.options.Line = 1;
         }
      }
   }

   TH2Painter.prototype.CountStat = function(cond) {
      var histo = this.GetHisto(), xaxis = histo.fXaxis, yaxis = histo.fYaxis,
          stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumx2 = 0, stat_sumy2 = 0, stat_sumxy = 0,
          xside, yside, xx, yy, zz,
          fp = this.frame_painter(),
          res = { name: histo.fName, entries: 0, integral: 0, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix: [0,0,0,0,0,0,0,0,0], xmax: 0, ymax:0, wmax: null };

      if (this.IsTH2Poly()) {

         var len = histo.fBins.arr.length, i, bin, n, gr, ngr, numgraphs, numpoints,
             pmain = this.frame_painter();

         for (i=0;i<len;++i) {
            bin = histo.fBins.arr[i];

            xside = 1; yside = 1;

            if (bin.fXmin > pmain.scale_xmax) xside = 2; else
            if (bin.fXmax < pmain.scale_xmin) xside = 0;
            if (bin.fYmin > pmain.scale_ymax) yside = 2; else
            if (bin.fYmax < pmain.scale_ymin) yside = 0;

            xx = yy = numpoints = 0;
            gr = bin.fPoly; numgraphs = 1;
            if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

            for (ngr=0;ngr<numgraphs;++ngr) {
               if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];

               for (n=0;n<gr.fNpoints;++n) {
                  ++numpoints;
                  xx += gr.fX[n];
                  yy += gr.fY[n];
               }
            }

            if (numpoints > 1) {
               xx = xx / numpoints;
               yy = yy / numpoints;
            }

            zz = bin.fContent;

            res.entries += zz;

            res.matrix[yside * 3 + xside] += zz;

            if ((xside != 1) || (yside != 1)) continue;

            if (cond && !cond(xx,yy)) continue;

            if ((res.wmax===null) || (zz>res.wmax)) { res.wmax = zz; res.xmax = xx; res.ymax = yy; }

            stat_sum0 += zz;
            stat_sumx1 += xx * zz;
            stat_sumy1 += yy * zz;
            stat_sumx2 += xx * xx * zz;
            stat_sumy2 += yy * yy * zz;
            stat_sumxy += xx * yy * zz;
         }
      } else {
         var xleft = this.GetSelectIndex("x", "left"),
             xright = this.GetSelectIndex("x", "right"),
             yleft = this.GetSelectIndex("y", "left"),
             yright = this.GetSelectIndex("y", "right"),
             xi, yi;

         for (xi = 0; xi <= this.nbinsx + 1; ++xi) {
            xside = (xi <= xleft) ? 0 : (xi > xright ? 2 : 1);
            xx = xaxis.GetBinCoord(xi - 0.5);

            for (yi = 0; yi <= this.nbinsy + 1; ++yi) {
               yside = (yi <= yleft) ? 0 : (yi > yright ? 2 : 1);
               yy = yaxis.GetBinCoord(yi - 0.5);

               zz = histo.getBinContent(xi, yi);

               res.entries += zz;

               res.matrix[yside * 3 + xside] += zz;

               if ((xside != 1) || (yside != 1)) continue;

               if ((cond!=null) && !cond(xx,yy)) continue;

               if ((res.wmax===null) || (zz>res.wmax)) { res.wmax = zz; res.xmax = xx; res.ymax = yy; }

               stat_sum0 += zz;
               stat_sumx1 += xx * zz;
               stat_sumy1 += yy * zz;
               stat_sumx2 += xx * xx * zz;
               stat_sumy2 += yy * yy * zz;
               stat_sumxy += xx * yy * zz;
            }
         }
      }

      if (!fp.IsAxisZoomed("x") && !fp.IsAxisZoomed("y") && (histo.fTsumw > 0)) {
         stat_sum0 = histo.fTsumw;
         stat_sumx1 = histo.fTsumwx;
         stat_sumx2 = histo.fTsumwx2;
         stat_sumy1 = histo.fTsumwy;
         stat_sumy2 = histo.fTsumwy2;
         stat_sumxy = histo.fTsumwxy;
      }

      if (stat_sum0 > 0) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.rmsx = Math.sqrt(Math.abs(stat_sumx2 / stat_sum0 - res.meanx * res.meanx));
         res.rmsy = Math.sqrt(Math.abs(stat_sumy2 / stat_sum0 - res.meany * res.meany));
      }

      if (res.wmax===null) res.wmax = 0;
      res.integral = stat_sum0;

      if (histo.fEntries > 1) res.entries = histo.fEntries;

      return res;
   }

   TH2Painter.prototype.FillStatistic = function(stat, dostat, dofit) {

      // no need to refill statistic if histogram is dummy
      if (this.IgnoreStatsFill()) return false;

      var data = this.CountStat(),
          print_name = Math.floor(dostat % 10),
          print_entries = Math.floor(dostat / 10) % 10,
          print_mean = Math.floor(dostat / 100) % 10,
          print_rms = Math.floor(dostat / 1000) % 10,
          print_under = Math.floor(dostat / 10000) % 10,
          print_over = Math.floor(dostat / 100000) % 10,
          print_integral = Math.floor(dostat / 1000000) % 10,
          print_skew = Math.floor(dostat / 10000000) % 10,
          print_kurt = Math.floor(dostat / 100000000) % 10;

      stat.ClearPave();

      if (print_name > 0)
         stat.AddText(data.name);

      if (print_entries > 0)
         stat.AddText("Entries = " + stat.Format(data.entries,"entries"));

      if (print_mean > 0) {
         stat.AddText("Mean x = " + stat.Format(data.meanx));
         stat.AddText("Mean y = " + stat.Format(data.meany));
      }

      if (print_rms > 0) {
         stat.AddText("Std Dev x = " + stat.Format(data.rmsx));
         stat.AddText("Std Dev y = " + stat.Format(data.rmsy));
      }

      if (print_integral > 0)
         stat.AddText("Integral = " + stat.Format(data.matrix[4],"entries"));

      if (print_skew > 0) {
         stat.AddText("Skewness x = <undef>");
         stat.AddText("Skewness y = <undef>");
      }

      if (print_kurt > 0)
         stat.AddText("Kurt = <undef>");

      if ((print_under > 0) || (print_over > 0)) {
         var m = data.matrix;

         stat.AddText("" + m[6].toFixed(0) + " | " + m[7].toFixed(0) + " | "  + m[7].toFixed(0));
         stat.AddText("" + m[3].toFixed(0) + " | " + m[4].toFixed(0) + " | "  + m[5].toFixed(0));
         stat.AddText("" + m[0].toFixed(0) + " | " + m[1].toFixed(0) + " | "  + m[2].toFixed(0));
      }

      if (dofit) stat.FillFunctionStat(this.FindFunction('TF1'), dofit);

      return true;
   }

   TH2Painter.prototype.DrawBinsColor = function(w,h) {
      var histo = this.GetHisto(),
          handle = this.PrepareColorDraw(),
          colPaths = [], currx = [], curry = [],
          colindx, cmd1, cmd2, i, j, binz, x1, dx, y2, dy;

      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {

         dx = handle.grx[i+1] - handle.grx[i];
         x1 = Math.round(handle.grx[i] + dx*handle.xbar1);
         dx = Math.round(dx*(handle.xbar2-handle.xbar1));

         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            colindx = this.getContourColor(binz, true);
            if (binz===0) {
               if (!this.options.Zero) continue;
               if ((colindx === null) && this._show_empty_bins) colindx = 0;
            }
            if (colindx === null) continue;

            dy = handle.gry[j]-handle.gry[j+1];
            y2 = Math.round(handle.gry[j+1] + dy*handle.ybar1);
            dy = Math.round(dy*(handle.ybar2-handle.ybar1));

            cmd1 = "M"+x1+","+y2;
            if (colPaths[colindx] === undefined) {
               colPaths[colindx] = cmd1;
            } else{
               cmd2 = "m" + (x1-currx[colindx]) + "," + (y2-curry[colindx]);
               colPaths[colindx] += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
            }

            currx[colindx] = x1;
            curry[colindx] = y2;

            colPaths[colindx] += "v"+dy + "h"+dx + "v"+(-dy) + "z";
         }
      }

      for (colindx=0;colindx<colPaths.length;++colindx)
        if (colPaths[colindx] !== undefined)
           this.draw_g
               .append("svg:path")
               .attr("palette-index", colindx)
               .attr("fill", this.fPalette.getColor(colindx))
               .attr("d", colPaths[colindx]);

      return handle;
   }

   TH2Painter.prototype.BuildContour = function(handle, levels, palette, contour_func) {
      var histo = this.GetObject(), ddd = 0,
          painter = this,
          kMAXCONTOUR = 2004,
          kMAXCOUNT = 2000,
          // arguments used in the PaintContourLine
          xarr = new Float32Array(2*kMAXCONTOUR),
          yarr = new Float32Array(2*kMAXCONTOUR),
          itarr = new Int32Array(2*kMAXCONTOUR),
          lj = 0, ipoly, poly, polys = [], np, npmax = 0,
          x = [0.,0.,0.,0.], y = [0.,0.,0.,0.], zc = [0.,0.,0.,0.], ir = [0,0,0,0],
          i, j, k, n, m, ix, ljfill, count,
          xsave, ysave, itars, ix, jx;

      function BinarySearch(zc) {
         for (var kk=0;kk<levels.length;++kk)
            if (zc<levels[kk]) return kk-1;
         return levels.length-1;
      }

      function PaintContourLine(elev1, icont1, x1, y1,  elev2, icont2, x2, y2) {
         /* Double_t *xarr, Double_t *yarr, Int_t *itarr, Double_t *levels */
         var vert = (x1 === x2),
             tlen = vert ? (y2 - y1) : (x2 - x1),
             n = icont1 +1,
             tdif = elev2 - elev1,
             ii = lj-1,
             maxii = kMAXCONTOUR/2 -3 + lj,
             icount = 0,
             xlen, pdif, diff, elev;

         while (n <= icont2 && ii <= maxii) {
//          elev = fH->GetContourLevel(n);
            elev = levels[n];
            diff = elev - elev1;
            pdif = diff/tdif;
            xlen = tlen*pdif;
            if (vert) {
               xarr[ii] = x1;
               yarr[ii] = y1 + xlen;
            } else {
               xarr[ii] = x1 + xlen;
               yarr[ii] = y1;
            }
            itarr[ii] = n;
            icount++;
            ii +=2;
            n++;
         }
         return icount;
      }

      var arrx = handle.original ? handle.origx : handle.grx,
          arry = handle.original ? handle.origy : handle.gry;

      for (j = handle.j1; j < handle.j2-1; ++j) {

         y[1] = y[0] = (arry[j] + arry[j+1])/2;
         y[3] = y[2] = (arry[j+1] + arry[j+2])/2;

         for (i = handle.i1; i < handle.i2-1; ++i) {

            zc[0] = histo.getBinContent(i+1, j+1);
            zc[1] = histo.getBinContent(i+2, j+1);
            zc[2] = histo.getBinContent(i+2, j+2);
            zc[3] = histo.getBinContent(i+1, j+2);

            for (k=0;k<4;k++)
               ir[k] = BinarySearch(zc[k]);

            if ((ir[0] !== ir[1]) || (ir[1] !== ir[2]) || (ir[2] !== ir[3]) || (ir[3] !== ir[0])) {
               x[3] = x[0] = (arrx[i] + arrx[i+1])/2;
               x[2] = x[1] = (arrx[i+1] + arrx[i+2])/2;

               if (zc[0] <= zc[1]) n = 0; else n = 1;
               if (zc[2] <= zc[3]) m = 2; else m = 3;
               if (zc[n] > zc[m]) n = m;
               n++;
               lj=1;
               for (ix=1;ix<=4;ix++) {
                  m = n%4 + 1;
                  ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],
                        zc[m-1],ir[m-1],x[m-1],y[m-1]);
                  lj += 2*ljfill;
                  n = m;
               }

               if (zc[0] <= zc[1]) n = 0; else n = 1;
               if (zc[2] <= zc[3]) m = 2; else m = 3;
               if (zc[n] > zc[m]) n = m;
               n++;
               lj=2;
               for (ix=1;ix<=4;ix++) {
                  if (n == 1) m = 4;
                  else        m = n-1;
                  ljfill = PaintContourLine(zc[n-1],ir[n-1],x[n-1],y[n-1],
                        zc[m-1],ir[m-1],x[m-1],y[m-1]);
                  lj += 2*ljfill;
                  n = m;
               }
               //     Re-order endpoints

               count = 0;
               for (ix=1; ix<=lj-5; ix +=2) {
                  //count = 0;
                  while (itarr[ix-1] != itarr[ix]) {
                     xsave = xarr[ix];
                     ysave = yarr[ix];
                     itars = itarr[ix];
                     for (jx=ix; jx<=lj-5; jx +=2) {
                        xarr[jx]  = xarr[jx+2];
                        yarr[jx]  = yarr[jx+2];
                        itarr[jx] = itarr[jx+2];
                     }
                     xarr[lj-3]  = xsave;
                     yarr[lj-3]  = ysave;
                     itarr[lj-3] = itars;
                     if (count > kMAXCOUNT) break;
                     count++;
                  }
               }

               if (count > kMAXCOUNT) continue;

               for (ix=1; ix<=lj-2; ix +=2) {

                  ipoly = itarr[ix-1];

                  if ((ipoly >= 0) && (ipoly < levels.length)) {
                     poly = polys[ipoly];
                     if (!poly)
                        poly = polys[ipoly] = JSROOT.CreateTPolyLine(kMAXCONTOUR*4, true);

                     np = poly.fLastPoint;
                     if (np < poly.fN-2) {
                        poly.fX[np+1] = Math.round(xarr[ix-1]); poly.fY[np+1] = Math.round(yarr[ix-1]);
                        poly.fX[np+2] = Math.round(xarr[ix]); poly.fY[np+2] = Math.round(yarr[ix]);
                        poly.fLastPoint = np+2;
                        npmax = Math.max(npmax, poly.fLastPoint+1);
                     } else {
                        // console.log('reject point??', poly.fLastPoint);
                     }
                  }
               }
            } // end of if (ir[0]
         } // end of j
      } // end of i

      var polysort = new Int32Array(levels.length), first = 0;
      //find first positive contour
      for (ipoly=0;ipoly<levels.length;ipoly++) {
         if (levels[ipoly] >= 0) { first = ipoly; break; }
      }
      //store negative contours from 0 to minimum, then all positive contours
      k = 0;
      for (ipoly=first-1;ipoly>=0;ipoly--) {polysort[k] = ipoly; k++;}
      for (ipoly=first;ipoly<levels.length;ipoly++) { polysort[k] = ipoly; k++;}

      var xp = new Float32Array(2*npmax),
          yp = new Float32Array(2*npmax);

      for (k=0;k<levels.length;++k) {

         ipoly = polysort[k];
         poly = polys[ipoly];
         if (!poly) continue;

         var colindx = palette.calcColorIndex(ipoly, levels.length),
             xx = poly.fX, yy = poly.fY, np = poly.fLastPoint+1,
             istart = 0, iminus, iplus, xmin = 0, ymin = 0, nadd;

         while (true) {

            iminus = npmax;
            iplus  = iminus+1;
            xp[iminus]= xx[istart];   yp[iminus] = yy[istart];
            xp[iplus] = xx[istart+1]; yp[iplus]  = yy[istart+1];
            xx[istart] = xx[istart+1] = xmin;
            yy[istart] = yy[istart+1] = ymin;
            while (true) {
               nadd = 0;
               for (i=2;i<np;i+=2) {
                  if ((iplus < 2*npmax-1) && (xx[i] === xp[iplus]) && (yy[i] === yp[iplus])) {
                     iplus++;
                     xp[iplus] = xx[i+1]; yp[iplus] = yy[i+1];
                     xx[i] = xx[i+1] = xmin;
                     yy[i] = yy[i+1] = ymin;
                     nadd++;
                  }
                  if ((iminus > 0) && (xx[i+1] === xp[iminus]) && (yy[i+1] === yp[iminus])) {
                     iminus--;
                     xp[iminus] = xx[i]; yp[iminus] = yy[i];
                     xx[i] = xx[i+1] = xmin;
                     yy[i] = yy[i+1] = ymin;
                     nadd++;
                  }
               }
               if (nadd == 0) break;
            }

            if ((iminus+1 < iplus) && (iminus>=0))
               contour_func(colindx, xp, yp, iminus, iplus, ipoly);

            istart = 0;
            for (i=2;i<np;i+=2) {
               if (xx[i] !== xmin && yy[i] !== ymin) {
                  istart = i;
                  break;
               }
            }

            if (istart === 0) break;
         }
      }
   }

   TH2Painter.prototype.DrawBinsContour = function(frame_w,frame_h) {
      var handle = this.PrepareColorDraw({ rounding: false, extra: 100, original: this.options.Proj != 0 });

      // get levels
      var levels = this.GetContour(),
          palette = this.GetPalette(),
          painter = this, main = this.frame_painter();

      function BuildPath(xp,yp,iminus,iplus) {
         var cmd = "", last = null, pnt = null, i;
         for (i=iminus;i<=iplus;++i) {
            pnt = null;
            switch (main.projection) {
               case 1: pnt = main.ProjectAitoff2xy(xp[i], yp[i]); break;
               case 2: pnt = main.ProjectMercator2xy(xp[i], yp[i]); break;
               case 3: pnt = main.ProjectSinusoidal2xy(xp[i], yp[i]); break;
               case 4: pnt = main.ProjectParabolic2xy(xp[i], yp[i]); break;
            }
            if (pnt) {
               pnt.x = main.grx(pnt.x);
               pnt.y = main.gry(pnt.y);
            } else {
               pnt = { x: xp[i], y: yp[i] };
            }
            pnt.x = Math.round(pnt.x);
            pnt.y = Math.round(pnt.y);
            if (!cmd) cmd = "M" + pnt.x + "," + pnt.y;
            else if ((pnt.x != last.x) && (pnt.y != last.y)) cmd +=  "l" + (pnt.x - last.x) + "," + (pnt.y - last.y);
            else if (pnt.x != last.x) cmd +=  "h" + (pnt.x - last.x);
            else if (pnt.y != last.y) cmd +=  "v" + (pnt.y - last.y);
            last = pnt;
         }
         return cmd;
      }

      if (this.options.Contour===14) {
         var dd = "M0,0h"+frame_w+"v"+frame_h+"h-"+frame_w;
         if (this.options.Proj) {
            var sz = handle.j2 - handle.j1, xd = new Float32Array(sz*2), yd = new Float32Array(sz*2);
            for (var i=0;i<sz;++i) {
               xd[i] = handle.origx[handle.i1];
               yd[i] = (handle.origy[handle.j1]*(i+0.5) + handle.origy[handle.j2]*(sz-0.5-i))/sz;
               xd[i+sz] = handle.origx[handle.i2];
               yd[i+sz] = (handle.origy[handle.j2]*(i+0.5) + handle.origy[handle.j1]*(sz-0.5-i))/sz;
            }
            dd = BuildPath(xd,yd,0,2*sz-1);
         }

         this.draw_g
             .append("svg:path")
             .attr("d", dd + "z")
             .style('stroke','none')
             .style("fill", palette.calcColor(0, levels.length));
      }

      this.BuildContour(handle, levels, palette,
         function(colindx,xp,yp,iminus,iplus) {
            var icol = palette.getColor(colindx),
                fillcolor = icol, lineatt = null;

            switch (painter.options.Contour) {
               case 1: break;
               case 11: fillcolor = 'none'; lineatt = new JSROOT.TAttLineHandler({ color: icol } ); break;
               case 12: fillcolor = 'none'; lineatt = new JSROOT.TAttLineHandler({ color: 1, style: (colindx%5 + 1), width: 1 }); break;
               case 13: fillcolor = 'none'; lineatt = painter.lineatt; break;
               case 14: break;
            }

            var elem = painter.draw_g
                          .append("svg:path")
                          .attr("class","th2_contour")
                          .attr("d", BuildPath(xp,yp,iminus,iplus) + (fillcolor == 'none' ? "" : "z"))
                          .style("fill", fillcolor);

            if (lineatt!==null)
               elem.call(lineatt.func);
            else
               elem.style('stroke','none');
         }
      );

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return handle;
   }

   TH2Painter.prototype.CreatePolyBin = function(pmain, bin, text_pos) {
      var cmd = "", ngr, ngraphs = 1, gr = null;

      if (bin.fPoly._typename=='TMultiGraph')
         ngraphs = bin.fPoly.fGraphs.arr.length;
      else
         gr = bin.fPoly;

      if (text_pos)
         bin._sumx = bin._sumy = bin._suml = 0;

      function AddPoint(x1,y1,x2,y2) {
         var len = Math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
         bin._sumx += (x1+x2)*len/2;
         bin._sumy += (y1+y2)*len/2;
         bin._suml += len;
      }

      for (ngr = 0; ngr < ngraphs; ++ ngr) {
         if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];

         var npnts = gr.fNpoints, n,
             x = gr.fX, y = gr.fY,
             grx = Math.round(pmain.grx(x[0])),
             gry = Math.round(pmain.gry(y[0])),
             nextx, nexty;

         if ((npnts>2) && (x[0]==x[npnts-1]) && (y[0]==y[npnts-1])) npnts--;

         cmd += "M"+grx+","+gry;

         for (n=1;n<npnts;++n) {
            nextx = Math.round(pmain.grx(x[n]));
            nexty = Math.round(pmain.gry(y[n]));
            if (text_pos) AddPoint(grx,gry, nextx, nexty);
            if ((grx!==nextx) || (gry!==nexty)) {
               if (grx===nextx) cmd += "v" + (nexty - gry); else
                  if (gry===nexty) cmd += "h" + (nextx - grx); else
                     cmd += "l" + (nextx - grx) + "," + (nexty - gry);
            }
            grx = nextx; gry = nexty;
         }

         if (text_pos) AddPoint(grx, gry, Math.round(pmain.grx(x[0])), Math.round(pmain.gry(y[0])));
         cmd += "z";
      }

      if (text_pos) {
         if (bin._suml > 0) {
            bin._midx = Math.round(bin._sumx / bin._suml);
            bin._midy = Math.round(bin._sumy / bin._suml);
         } else {
            bin._midx = Math.round(pmain.grx((bin.fXmin + bin.fXmax)/2));
            bin._midy = Math.round(pmain.gry((bin.fYmin + bin.fYmax)/2));
         }
      }

      return cmd;
   }

   TH2Painter.prototype.DrawPolyBinsColor = function(w,h) {
      var histo = this.GetObject(),
          pmain = this.frame_painter(),
          colPaths = [], textbins = [],
          colindx, cmd, bin, item,
          i, len = histo.fBins.arr.length;

      // force recalculations of contours
      this.fContour = null;
      this.fCustomContour = false;

      // use global coordinates
      this.maxbin = this.gmaxbin;
      this.minbin = this.gminbin;
      this.minposbin = this.gminposbin;

      for (i = 0; i < len; ++ i) {
         bin = histo.fBins.arr[i];
         colindx = this.getContourColor(bin.fContent, true);
         if (colindx === null) continue;
         if (bin.fContent === 0) {
            if (!this.options.Zero || !this.options.Line) continue;
            colindx = 0; // make dummy fill color to draw only line
         }

         // check if bin outside visible range
         if ((bin.fXmin > pmain.scale_xmax) || (bin.fXmax < pmain.scale_xmin) ||
             (bin.fYmin > pmain.scale_ymax) || (bin.fYmax < pmain.scale_ymin)) continue;

         cmd = this.CreatePolyBin(pmain, bin, this.options.Text && bin.fContent);

         if (colPaths[colindx] === undefined)
            colPaths[colindx] = cmd;
         else
            colPaths[colindx] += cmd;

         if (this.options.Text && bin.fContent) textbins.push(bin);
      }

      for (colindx=0;colindx<colPaths.length;++colindx)
         if (colPaths[colindx]) {
            item = this.draw_g
                     .append("svg:path")
                     .attr("palette-index", colindx)
                     .attr("fill", colindx ? this.fPalette.getColor(colindx) : 'none')
                     .attr("d", colPaths[colindx]);
            if (this.options.Line)
               item.call(this.lineatt.func);
         }

      if (textbins.length > 0) {
         var text_col = this.get_color(histo.fMarkerColor),
             text_angle = -1*this.options.TextAngle,
             text_g = this.draw_g.append("svg:g").attr("class","th2poly_text"),
             text_size = 12;

         if ((histo.fMarkerSize!==1) && text_angle)
             text_size = Math.round(0.02*h*histo.fMarkerSize);

         this.StartTextDrawing(42, text_size, text_g, text_size);

         for (i = 0; i < textbins.length; ++ i) {
            bin = textbins[i];

            var lbl = "";

            if (!this.options.TextKind) {
               lbl = (Math.round(bin.fContent) === bin.fContent) ? bin.fContent.toString() :
                          JSROOT.FFormat(bin.fContent, JSROOT.gStyle.fPaintTextFormat);
            } else {
               if (bin.fPoly) lbl = bin.fPoly.fName;
               if (lbl === "Graph") lbl = "";
               if (!lbl) lbl = bin.fNumber;
            }

            this.DrawText({ align: 22, x: bin._midx, y: bin._midy, rotate: text_angle, text: lbl, color: text_col, latex: 0, draw_g: text_g });
         }

         this.FinishTextDrawing(text_g, null);
      }

      return { poly: true };
   }

   TH2Painter.prototype.DrawBinsText = function(w, h, handle) {
      var histo = this.GetObject(),
          i,j,binz,errz,colindx,binw,binh,lbl,lble,posx,posy,sizex,sizey,
          text_col = this.get_color(histo.fMarkerColor),
          text_angle = -1*this.options.TextAngle,
          text_g = this.draw_g.append("svg:g").attr("class","th2_text"),
          text_size = 20, text_offset = 0,
          profile2d = this.MatchObjectType('TProfile2D') && (typeof histo.getBinEntries=='function'),
          show_err = (this.options.TextKind == "E"),
          use_latex = (show_err && !this.options.TextLine) ? 1 : 0;

      if (handle===null) handle = this.PrepareColorDraw({ rounding: false });

      if ((histo.fMarkerSize!==1) && text_angle)
         text_size = Math.round(0.02*h*histo.fMarkerSize);

      if (histo.fBarOffset!==0) text_offset = histo.fBarOffset*1e-3;

      this.StartTextDrawing(42, text_size, text_g, text_size);

      for (i = handle.i1; i < handle.i2; ++i)
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i+1, j+1);
            if ((binz === 0) && !this._show_empty_bins) continue;

            binw = handle.grx[i+1] - handle.grx[i];
            binh = handle.gry[j] - handle.gry[j+1];

            if (profile2d)
               binz = histo.getBinEntries(i+1, j+1);

            lbl = (binz === Math.round(binz)) ? binz.toString() :
                      JSROOT.FFormat(binz, JSROOT.gStyle.fPaintTextFormat);

            if (show_err) {
               errz = histo.getBinError(histo.getBin(i+1,j+1));
               lble = (errz === Math.round(errz)) ? errz.toString() :
                            JSROOT.FFormat(errz, JSROOT.gStyle.fPaintTextFormat);
               if (this.options.TextLine)
                  lbl += '\xB1' + lble;
               else
                  lbl = "#splitline{" + lbl + "}{#pm" + lble + "}";
            }

            if (text_angle /*|| (histo.fMarkerSize!==1)*/) {
               posx = Math.round(handle.grx[i] + binw*0.5);
               posy = Math.round(handle.gry[j+1] + binh*(0.5 + text_offset));
               sizex = 0;
               sizey = 0;
            } else {
               posx = Math.round(handle.grx[i] + binw*0.1);
               posy = Math.round(handle.gry[j+1] + binh*(0.1 + text_offset));
               sizex = Math.round(binw*0.8);
               sizey = Math.round(binh*0.8);
            }

            this.DrawText({ align: 22, x: posx, y: posy, width: sizex, height: sizey, rotate: text_angle, text: lbl, color: text_col, latex: use_latex, draw_g: text_g });
         }

      this.FinishTextDrawing(text_g, null);

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return handle;
   }

   TH2Painter.prototype.DrawBinsArrow = function(w, h) {
      var histo = this.GetObject(), cmd = "",
          i,j,binz,colindx,binw,binh,lbl, loop, dn = 1e-30, dx, dy, xc,yc,
          dxn,dyn,x1,x2,y1,y2, anr,si,co,
          handle = this.PrepareColorDraw({ rounding: false }),
          scale_x  = (handle.grx[handle.i2] - handle.grx[handle.i1])/(handle.i2 - handle.i1 + 1-0.03)/2,
          scale_y  = (handle.gry[handle.j2] - handle.gry[handle.j1])/(handle.j2 - handle.j1 + 1-0.03)/2;

      for (var loop=0;loop<2;++loop)
         for (i = handle.i1; i < handle.i2; ++i)
            for (j = handle.j1; j < handle.j2; ++j) {

               if (i === handle.i1) {
                  dx = histo.getBinContent(i+2, j+1) - histo.getBinContent(i+1, j+1);
               } else if (i === handle.i2-1) {
                  dx = histo.getBinContent(i+1, j+1) - histo.getBinContent(i, j+1);
               } else {
                  dx = 0.5*(histo.getBinContent(i+2, j+1) - histo.getBinContent(i, j+1));
               }
               if (j === handle.j1) {
                  dy = histo.getBinContent(i+1, j+2) - histo.getBinContent(i+1, j+1);
               } else if (j === handle.j2-1) {
                  dy = histo.getBinContent(i+1, j+1) - histo.getBinContent(i+1, j);
               } else {
                  dy = 0.5*(histo.getBinContent(i+1, j+2) - histo.getBinContent(i+1, j));
               }

               if (loop===0) {
                  dn = Math.max(dn, Math.abs(dx), Math.abs(dy));
               } else {
                  xc = (handle.grx[i] + handle.grx[i+1])/2;
                  yc = (handle.gry[j] + handle.gry[j+1])/2;
                  dxn = scale_x*dx/dn;
                  dyn = scale_y*dy/dn;
                  x1  = xc - dxn;
                  x2  = xc + dxn;
                  y1  = yc - dyn;
                  y2  = yc + dyn;
                  dx = Math.round(x2-x1);
                  dy = Math.round(y2-y1);

                  if ((dx!==0) || (dy!==0)) {
                     cmd += "M"+Math.round(x1)+","+Math.round(y1)+"l"+dx+","+dy;

                     if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                        anr = Math.sqrt(2/(dx*dx + dy*dy));
                        si  = Math.round(anr*(dx + dy));
                        co  = Math.round(anr*(dx - dy));
                        if ((si!==0) && (co!==0))
                           cmd+="l"+(-si)+","+co + "m"+si+","+(-co) + "l"+(-co)+","+(-si);
                     }
                  }
               }
            }

      this.draw_g
         .append("svg:path")
         .attr("class","th2_arrows")
         .attr("d", cmd)
         .style("fill", "none")
         .call(this.lineatt.func);

      return handle;
   }


   TH2Painter.prototype.DrawBinsBox = function(w,h) {

      var histo = this.GetObject(),
          handle = this.PrepareColorDraw({ rounding: false }),
          main = this.main_painter();

      if (main===this) {
         if (main.maxbin === main.minbin) {
            main.maxbin = main.gmaxbin;
            main.minbin = main.gminbin;
            main.minposbin = main.gminposbin;
         }
         if (main.maxbin === main.minbin)
            main.minbin = Math.min(0, main.maxbin-1);
      }

      var absmax = Math.max(Math.abs(main.maxbin), Math.abs(main.minbin)),
          absmin = Math.max(0, main.minbin),
          i, j, binz, absz, res = "", cross = "", btn1 = "", btn2 = "",
          colindx, zdiff, dgrx, dgry, xx, yy, ww, hh, cmd1, cmd2,
          xyfactor = 1, uselogz = false, logmin = 0, logmax = 1;

      if (this.root_pad().fLogz && (absmax>0)) {
         uselogz = true;
         logmax = Math.log(absmax);
         if (absmin>0) logmin = Math.log(absmin); else
         if ((main.minposbin>=1) && (main.minposbin<100)) logmin = Math.log(0.7); else
            logmin = (main.minposbin > 0) ? Math.log(0.7*main.minposbin) : logmax - 10;
         if (logmin >= logmax) logmin = logmax - 10;
         xyfactor = 1. / (logmax - logmin);
      } else {
         xyfactor = 1. / (absmax - absmin);
      }

      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            absz = Math.abs(binz);
            if ((absz === 0) || (absz < absmin)) continue;

            zdiff = uselogz ? ((absz>0) ? Math.log(absz) - logmin : 0) : (absz - absmin);
            // area of the box should be proportional to absolute bin content
            zdiff = 0.5 * ((zdiff < 0) ? 1 : (1 - Math.sqrt(zdiff * xyfactor)));
            // avoid oversized bins
            if (zdiff < 0) zdiff = 0;

            ww = handle.grx[i+1] - handle.grx[i];
            hh = handle.gry[j] - handle.gry[j+1];

            dgrx = zdiff * ww;
            dgry = zdiff * hh;

            xx = Math.round(handle.grx[i] + dgrx);
            yy = Math.round(handle.gry[j+1] + dgry);

            ww = Math.max(Math.round(ww - 2*dgrx), 1);
            hh = Math.max(Math.round(hh - 2*dgry), 1);

            res += "M"+xx+","+yy + "v"+hh + "h"+ww + "v-"+hh + "z";

            if ((binz<0) && (this.options.BoxStyle === 10))
               cross += "M"+xx+","+yy + "l"+ww+","+hh + "M"+(xx+ww)+","+yy + "l-"+ww+","+hh;

            if ((this.options.BoxStyle === 11) && (ww>5) && (hh>5)) {
               var pww = Math.round(ww*0.1),
                   phh = Math.round(hh*0.1),
                   side1 = "M"+xx+","+yy + "h"+ww + "l"+(-pww)+","+phh + "h"+(2*pww-ww) +
                           "v"+(hh-2*phh)+ "l"+(-pww)+","+phh + "z",
                   side2 = "M"+(xx+ww)+","+(yy+hh) + "v"+(-hh) + "l"+(-pww)+","+phh + "v"+(hh-2*phh)+
                           "h"+(2*pww-ww) + "l"+(-pww)+","+phh + "z";
               if (binz<0) { btn2+=side1; btn1+=side2; }
                      else { btn1+=side1; btn2+=side2; }
            }
         }
      }

      if (res.length > 0) {
         var elem = this.draw_g.append("svg:path")
                               .attr("d", res)
                               .call(this.fillatt.func);
         if ((this.options.BoxStyle === 11) || !this.fillatt.empty())
            elem.style('stroke','none');
         else
            elem.call(this.lineatt.func);
      }

      if ((btn1.length>0) && (this.fillatt.color !== 'none'))
         this.draw_g.append("svg:path")
                    .attr("d", btn1)
                    .style("stroke","none")
                    .call(this.fillatt.func)
                    .style("fill", d3.rgb(this.fillatt.color).brighter(0.5).toString());

      if (btn2.length>0)
         this.draw_g.append("svg:path")
                    .attr("d", btn2)
                    .style("stroke","none")
                    .call(this.fillatt.func)
                    .style("fill", this.fillatt.color === 'none' ? 'red' : d3.rgb(this.fillatt.color).darker(0.5).toString());

      if (cross.length > 0) {
         var elem = this.draw_g.append("svg:path")
                               .attr("d", cross)
                               .style("fill", "none");
         if (this.lineatt.color !== 'none')
            elem.call(this.lineatt.func);
         else
            elem.style('stroke','black');
      }

      return handle;
   }

   TH2Painter.prototype.DrawCandle = function(w,h) {
      var histo = this.GetHisto(),
          handle = this.PrepareColorDraw(),
          pmain = this.frame_painter(), // used for axis values conversions
          i, j, y, sum0, sum1, sum2, cont, center, counter, integral, w, pnt,
          bars = "", markers = "", posy;

      if (histo.fMarkerColor === 1) histo.fMarkerColor = histo.fLineColor;

      // create attribute only when necessary
      this.createAttMarker({ attr: histo, style: 5 });

      // reset absolution position for markers
      this.markeratt.reset_pos();

      handle.candle = []; // array of drawn points

      // loop over visible x-bins
      for (i = handle.i1; i < handle.i2; ++i) {
         sum1 = 0;
         //estimate integral
         integral = 0;
         counter = 0;
         for (j = 0; j < this.nbinsy; ++j) {
            integral += histo.getBinContent(i+1,j+1);
         }
         pnt = { bin:i, meany:0, m25y:0, p25y:0, median:0, iqr:0, whiskerp:0, whiskerm:0};
         //estimate quantiles... simple function... not so nice as GetQuantiles
         for (j = 0; j < this.nbinsy; ++j) {
            cont = histo.getBinContent(i+1,j+1);
            posy = histo.fYaxis.GetBinCenter(j+1);
            if (counter/integral < 0.001 && (counter + cont)/integral >=0.001) pnt.whiskerm = posy; // Lower whisker
            if (counter/integral < 0.25 && (counter + cont)/integral >=0.25) pnt.m25y = posy; // Lower edge of box
            if (counter/integral < 0.5 && (counter + cont)/integral >=0.5) pnt.median = posy; //Median
            if (counter/integral < 0.75 && (counter + cont)/integral >=0.75) pnt.p25y = posy; //Upper edge of box
            if (counter/integral < 0.999 && (counter + cont)/integral >=0.999) pnt.whiskerp = posy; // Upper whisker
            counter += cont;
            y = posy; // center of y bin coordinate
            sum1 += cont*y;
         }
         if (counter > 0) {
            pnt.meany = sum1/counter;
         }
         pnt.iqr = pnt.p25y-pnt.m25y;

         //Whiskers cannot exceed 1.5*iqr from box
         if ((pnt.m25y-1.5*pnt.iqr) > pnt.whsikerm)  {
            pnt.whiskerm = pnt.m25y-1.5*pnt.iqr;
         }
         if ((pnt.p25y+1.5*pnt.iqr) < pnt.whiskerp) {
            pnt.whiskerp = pnt.p25y+1.5*pnt.iqr;
         }

         // exclude points with negative y when log scale is specified
         if (pmain.logy && (pnt.whiskerm<=0)) continue;

         w = handle.grx[i+1] - handle.grx[i];
         w *= 0.66;
         center = (handle.grx[i+1] + handle.grx[i]) / 2 + histo.fBarOffset/1000*w;
         if (histo.fBarWidth>0) w = w * histo.fBarWidth / 1000;

         pnt.x1 = Math.round(center - w/2);
         pnt.x2 = Math.round(center + w/2);
         center = Math.round(center);

         pnt.y0 = Math.round(pmain.gry(pnt.median));
         // mean line
         bars += "M" + pnt.x1 + "," + pnt.y0 + "h" + (pnt.x2-pnt.x1);

         pnt.y1 = Math.round(pmain.gry(pnt.p25y));
         pnt.y2 = Math.round(pmain.gry(pnt.m25y));

         // rectangle
         bars += "M" + pnt.x1 + "," + pnt.y1 +
         "v" + (pnt.y2-pnt.y1) + "h" + (pnt.x2-pnt.x1) + "v-" + (pnt.y2-pnt.y1) + "z";

         pnt.yy1 = Math.round(pmain.gry(pnt.whiskerp));
         pnt.yy2 = Math.round(pmain.gry(pnt.whiskerm));

         // upper part
         bars += "M" + center + "," + pnt.y1 + "v" + (pnt.yy1-pnt.y1);
         bars += "M" + pnt.x1 + "," + pnt.yy1 + "h" + (pnt.x2-pnt.x1);

         // lower part
         bars += "M" + center + "," + pnt.y2 + "v" + (pnt.yy2-pnt.y2);
         bars += "M" + pnt.x1 + "," + pnt.yy2 + "h" + (pnt.x2-pnt.x1);

         //estimate outliers
         for (j = 0; j < this.nbinsy; ++j) {
            cont = histo.getBinContent(i+1,j+1);
            posy = histo.fYaxis.GetBinCenter(j+1);
            if (cont > 0 && posy < pnt.whiskerm) markers += this.markeratt.create(center, posy);
            if (cont > 0 && posy > pnt.whiskerp) markers += this.markeratt.create(center, posy);
         }

         handle.candle.push(pnt); // keep point for the tooltip
      }

      if (bars.length > 0)
         this.draw_g.append("svg:path")
             .attr("d", bars)
             .call(this.lineatt.func)
             .call(this.fillatt.func);

      if (markers.length > 0)
         this.draw_g.append("svg:path")
             .attr("d", markers)
             .call(this.markeratt.func);

      return handle;
   }

   TH2Painter.prototype.DrawBinsScatter = function(w,h) {
      var histo = this.GetObject(),
          handle = this.PrepareColorDraw({ rounding: true, pixel_density: true }),
          colPaths = [], currx = [], curry = [], cell_w = [], cell_h = [],
          colindx, cmd1, cmd2, i, j, binz, cw, ch, factor = 1.,
          scale = this.options.ScatCoef * ((this.gmaxbin) > 2000 ? 2000. / this.gmaxbin : 1.);

      JSROOT.seed(handle.sumz);

      if (scale*handle.sumz < 1e5) {
         // one can use direct drawing of scatter plot without any patterns

         this.createAttMarker({ attr: histo });

         this.markeratt.reset_pos();

         var path = "", k, npix;
         for (i = handle.i1; i < handle.i2; ++i) {
            cw = handle.grx[i+1] - handle.grx[i];
            for (j = handle.j1; j < handle.j2; ++j) {
               ch = handle.gry[j] - handle.gry[j+1];
               binz = histo.getBinContent(i + 1, j + 1);

               npix = Math.round(scale*binz);
               if (npix<=0) continue;

               for (k=0;k<npix;++k)
                  path += this.markeratt.create(
                            Math.round(handle.grx[i] + cw * JSROOT.random()),
                            Math.round(handle.gry[j+1] + ch * JSROOT.random()));
            }
         }

         this.draw_g
              .append("svg:path")
              .attr("d", path)
              .call(this.markeratt.func);

         return handle;
      }

      // limit filling factor, do not try to produce as many points as filled area;
      if (this.maxbin > 0.7) factor = 0.7/this.maxbin;

      var nlevels = Math.round(handle.max - handle.min);
      this.CreateContour((nlevels > 50) ? 50 : nlevels, this.minposbin, this.maxbin, this.minposbin);

      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            if ((binz <= 0) || (binz < this.minbin)) continue;

            cw = handle.grx[i+1] - handle.grx[i];
            ch = handle.gry[j] - handle.gry[j+1];
            if (cw*ch <= 0) continue;

            colindx = this.getContourIndex(binz/cw/ch);
            if (colindx < 0) continue;

            cmd1 = "M"+handle.grx[i]+","+handle.gry[j+1];
            if (colPaths[colindx] === undefined) {
               colPaths[colindx] = cmd1;
               cell_w[colindx] = cw;
               cell_h[colindx] = ch;
            } else{
               cmd2 = "m" + (handle.grx[i]-currx[colindx]) + "," + (handle.gry[j+1] - curry[colindx]);
               colPaths[colindx] += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
               cell_w[colindx] = Math.max(cell_w[colindx], cw);
               cell_h[colindx] = Math.max(cell_h[colindx], ch);
            }

            currx[colindx] = handle.grx[i];
            curry[colindx] = handle.gry[j+1];

            colPaths[colindx] += "v"+ch+"h"+cw+"v-"+ch+"z";
         }
      }

      var layer = this.svg_frame().select('.main_layer'),
          defs = layer.select("defs");
      if (defs.empty() && (colPaths.length>0))
         defs = layer.insert("svg:defs",":first-child");

      this.createAttMarker({ attr: histo });

      for (colindx=0;colindx<colPaths.length;++colindx)
        if ((colPaths[colindx] !== undefined) && (colindx<this.fContour.length)) {
           var pattern_class = "scatter_" + colindx,
               pattern = defs.select('.' + pattern_class);
           if (pattern.empty())
              pattern = defs.append('svg:pattern')
                            .attr("class", pattern_class)
                            .attr("id", "jsroot_scatter_pattern_" + JSROOT.id_counter++)
                            .attr("patternUnits","userSpaceOnUse");
           else
              pattern.selectAll("*").remove();

           var npix = Math.round(factor*this.fContour[colindx]*cell_w[colindx]*cell_h[colindx]);
           if (npix<1) npix = 1;

           var arrx = new Float32Array(npix), arry = new Float32Array(npix);

           if (npix===1) {
              arrx[0] = arry[0] = 0.5;
           } else {
              for (var n=0;n<npix;++n) {
                 arrx[n] = JSROOT.random();
                 arry[n] = JSROOT.random();
              }
           }

           // arrx.sort();

           this.markeratt.reset_pos();

           var path = "";

           for (var n=0;n<npix;++n)
              path += this.markeratt.create(arrx[n] * cell_w[colindx], arry[n] * cell_h[colindx]);

           pattern.attr("width", cell_w[colindx])
                  .attr("height", cell_h[colindx])
                  .append("svg:path")
                  .attr("d",path)
                  .call(this.markeratt.func);

           this.draw_g
               .append("svg:path")
               .attr("scatter-index", colindx)
               .attr("fill", 'url(#' + pattern.attr("id") + ')')
               .attr("d", colPaths[colindx]);
        }

      return handle;
   }

   TH2Painter.prototype.DrawBins = function() {

      if (!this.draw_content)
         return this.RemoveDrawG();

      this.CheckHistDrawAttributes();

      this.CreateG(true);

      var w = this.frame_width(),
          h = this.frame_height(),
          handle = null;

      if (this.IsTH2Poly()) {
         handle = this.DrawPolyBinsColor(w, h);
      } else {
         if (this.options.Scat)
            handle = this.DrawBinsScatter(w, h);
         else if (this.options.Color)
            handle = this.DrawBinsColor(w, h);
         else if (this.options.Box)
            handle = this.DrawBinsBox(w, h);
         else if (this.options.Arrow)
            handle = this.DrawBinsArrow(w, h);
         else if (this.options.Contour)
            handle = this.DrawBinsContour(w, h);
         else if (this.options.Candle)
            handle = this.DrawCandle(w, h);

         if (this.options.Text)
            handle = this.DrawBinsText(w, h, handle);

         if (!handle)
            handle = this.DrawBinsScatter(w, h);
      }

      this.tt_handle = handle;
   }

   /** Return text information about histogram bin */
   TH2Painter.prototype.GetBinTips = function (i, j) {
      var lines = [], pmain = this.frame_painter(),
          histo = this.GetHisto(),
          binz = histo.getBinContent(i+1,j+1)

      lines.push(this.GetTipName());

      if (pmain.x_kind == 'labels')
         lines.push("x = " + pmain.AxisAsText("x", histo.fXaxis.GetBinLowEdge(i+1)));
      else
         lines.push("x = [" + pmain.AxisAsText("x", histo.fXaxis.GetBinLowEdge(i+1)) + ", " + pmain.AxisAsText("x", histo.fXaxis.GetBinLowEdge(i+2)) + ")");

      if (pmain.y_kind == 'labels')
         lines.push("y = " + pmain.AxisAsText("y", histo.fYaxis.GetBinLowEdge(j+1)));
      else
         lines.push("y = [" + pmain.AxisAsText("y", histo.fYaxis.GetBinLowEdge(j+1)) + ", " + pmain.AxisAsText("y", histo.fYaxis.GetBinLowEdge(j+2)) + ")");

      lines.push("bin = " + i + ", " + j);

      if (histo.$baseh) binz -= histo.$baseh.getBinContent(i+1,j+1);

      lines.push("entries = " + ((binz === Math.round(binz)) ? binz : JSROOT.FFormat(binz, JSROOT.gStyle.fStatFormat)));

      if ((this.options.TextKind == "E") || this.MatchObjectType('TProfile2D')) {
         var errz = histo.getBinError(histo.getBin(i+1,j+1));
         lines.push("error = " + ((errz === Math.round(errz)) ? errz.toString() : JSROOT.FFormat(errz, JSROOT.gStyle.fPaintTextFormat)));
      }

      return lines;
   }

   TH2Painter.prototype.GetCandleTips = function(p) {
      var lines = [], main = this.frame_painter(), histo = this.GetHisto();

      lines.push(this.GetTipName());

      lines.push("x = " + main.AxisAsText("x", histo.fXaxis.GetBinLowEdge(p.bin+1)));

      lines.push('mean y = ' + JSROOT.FFormat(p.meany, JSROOT.gStyle.fStatFormat))
      lines.push('m25 = ' + JSROOT.FFormat(p.m25y, JSROOT.gStyle.fStatFormat))
      lines.push('p25 = ' + JSROOT.FFormat(p.p25y, JSROOT.gStyle.fStatFormat))

      return lines;
   }

   TH2Painter.prototype.ProvidePolyBinHints = function(binindx, realx, realy) {

      var histo = this.GetHisto(),
          bin = histo.fBins.arr[binindx],
          pmain = this.frame_painter(),
          binname = bin.fPoly.fName,
          lines = [], numpoints = 0;

      if (binname === "Graph") binname = "";
      if (binname.length === 0) binname = bin.fNumber;

      if ((realx===undefined) && (realy===undefined)) {
         realx = realy = 0;
         var gr = bin.fPoly, numgraphs = 1;
         if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

         for (var ngr=0;ngr<numgraphs;++ngr) {
            if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];

            for (var n=0;n<gr.fNpoints;++n) {
               ++numpoints;
               realx += gr.fX[n];
               realy += gr.fY[n];
            }
         }

         if (numpoints > 1) {
            realx = realx / numpoints;
            realy = realy / numpoints;
         }
      }

      lines.push(this.GetTipName());
      lines.push("x = " + pmain.AxisAsText("x", realx));
      lines.push("y = " + pmain.AxisAsText("y", realy));
      if (numpoints > 0) lines.push("npnts = " + numpoints);
      lines.push("bin = " + binname);
      if (bin.fContent === Math.round(bin.fContent))
         lines.push("content = " + bin.fContent);
      else
         lines.push("content = " + JSROOT.FFormat(bin.fContent, JSROOT.gStyle.fStatFormat));
      return lines;
   }

   TH2Painter.prototype.ProcessTooltip = function(pnt) {
      if (!pnt || !this.draw_content || !this.draw_g || !this.tt_handle || this.options.Proj) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      var histo = this.GetHisto(),
          h = this.tt_handle,
          ttrect = this.draw_g.select(".tooltip_bin");

      if (h.poly) {
         // process tooltips from TH2Poly

         var pmain = this.frame_painter(),
             realx, realy, foundindx = -1;

         if (pmain.grx === pmain.x) realx = pmain.x.invert(pnt.x);
         if (pmain.gry === pmain.y) realy = pmain.y.invert(pnt.y);

         if ((realx!==undefined) && (realy!==undefined)) {
            var i, len = histo.fBins.arr.length, bin;

            for (i = 0; (i < len) && (foundindx < 0); ++ i) {
               bin = histo.fBins.arr[i];

               // found potential bins candidate
               if ((realx < bin.fXmin) || (realx > bin.fXmax) ||
                    (realy < bin.fYmin) || (realy > bin.fYmax)) continue;

               // ignore empty bins with col0 option
               if ((bin.fContent === 0) && !this.options.Zero) continue;

               var gr = bin.fPoly, numgraphs = 1;
               if (gr._typename === 'TMultiGraph') { numgraphs = bin.fPoly.fGraphs.arr.length; gr = null; }

               for (var ngr=0;ngr<numgraphs;++ngr) {
                  if (!gr || (ngr>0)) gr = bin.fPoly.fGraphs.arr[ngr];
                  if (gr.IsInside(realx,realy)) {
                     foundindx = i;
                     break;
                  }
               }
            }
         }

         if (foundindx < 0) {
            ttrect.remove();
            return null;
         }

         var res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.fillcoloralt('blue') : "blue",
                     exact: true, menu: true,
                     lines: this.ProvidePolyBinHints(foundindx, realx, realy) };

         if (pnt.disabled) {
            ttrect.remove();
            res.changed = true;
         } else {

            if (ttrect.empty())
               ttrect = this.draw_g.append("svg:path")
                            .attr("class","tooltip_bin h1bin")
                            .style("pointer-events","none");

            res.changed = ttrect.property("current_bin") !== foundindx;

            if (res.changed)
                  ttrect.attr("d", this.CreatePolyBin(pmain, bin))
                        .style("opacity", "0.7")
                        .property("current_bin", foundindx);
         }

         if (res.changed)
            res.user_info = { obj: histo, name: histo.fName,
                              bin: foundindx,
                              cont: bin.fContent,
                              grx: pnt.x, gry: pnt.y };

         return res;

      } else if (h.candle) {
         // process tooltips for candle

         var p, i;

         for (i=0;i<h.candle.length;++i) {
            p = h.candle[i];
            if ((p.x1 <= pnt.x) && (pnt.x <= p.x2) && (p.yy1 <= pnt.y) && (pnt.y <= p.yy2)) break;
         }

         if (i>=h.candle.length) {
            ttrect.remove();
            return null;
         }

         var res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.fillcoloralt('blue') : "blue",
                     lines: this.GetCandleTips(p), exact: true, menu: true };

         if (pnt.disabled) {
            ttrect.remove();
            res.changed = true;
         } else {

            if (ttrect.empty())
               ttrect = this.draw_g.append("svg:rect")
                                   .attr("class","tooltip_bin h1bin")
                                   .style("pointer-events","none");

            res.changed = ttrect.property("current_bin") !== i;

            if (res.changed)
               ttrect.attr("x", p.x1)
                     .attr("width", p.x2-p.x1)
                     .attr("y", p.yy1)
                     .attr("height", p.yy2-p.yy1)
                     .style("opacity", "0.7")
                     .property("current_bin", i);
         }

         if (res.changed)
            res.user_info = { obj: histo,  name: histo.fName,
                              bin: i+1, cont: p.median, binx: i+1, biny: 1,
                              grx: pnt.x, gry: pnt.y };

         return res;
      }

      var i, j, binz = 0, colindx = null,
          i1, i2, j1, j2, x1, x2, y1, y2,
          pmain = this.frame_painter();

      // search bins position
      if (pmain.reverse_x) {
         for (i = h.i1; i < h.i2; ++i)
            if ((pnt.x<=h.grx[i]) && (pnt.x>=h.grx[i+1])) break;
      } else {
         for (i = h.i1; i < h.i2; ++i)
            if ((pnt.x>=h.grx[i]) && (pnt.x<=h.grx[i+1])) break;
      }

      if (pmain.reverse_y) {
         for (j = h.j1; j < h.j2; ++j)
            if ((pnt.y<=h.gry[j+1]) && (pnt.y>=h.gry[j])) break;
      } else {
         for (j = h.j1; j < h.j2; ++j)
            if ((pnt.y>=h.gry[j+1]) && (pnt.y<=h.gry[j])) break;
      }

      if ((i < h.i2) && (j < h.j2)) {

         i1 = i; i2 = i+1; j1 = j; j2 = j+1;
         x1 = h.grx[i1]; x2 = h.grx[i2];
         y1 = h.gry[j2]; y2 = h.gry[j1];

         var match = true;

         if (this.options.Color) {
            // take into account bar settings
            var dx = x2 - x1, dy = y2 - y1;
            x2 = Math.round(x1 + dx*h.xbar2);
            x1 = Math.round(x1 + dx*h.xbar1);
            y2 = Math.round(y1 + dy*h.ybar2);
            y1 = Math.round(y1 + dy*h.ybar1);
            if (pmain.reverse_x) {
               if ((pnt.x>x1) || (pnt.x<=x2)) match = false;
            } else {
               if ((pnt.x<x1) || (pnt.x>=x2)) match = false;
            }
            if (pmain.reverse_y) {
               if ((pnt.y>y1) || (pnt.y<=y2)) match = false;
            } else {
               if ((pnt.y<y1) || (pnt.y>=y2)) match = false;
            }
         }

         binz = histo.getBinContent(i+1,j+1);
         if (this.is_projection) {
            colindx = 0; // just to avoid hide
         } else if (!match) {
            colindx = null;
         } else if (h.hide_only_zeros) {
            colindx = (binz === 0) && !this._show_empty_bins ? null : 0;
         } else {
            colindx = this.getContourColor(binz, true);
            if ((colindx === null) && (binz === 0) && this._show_empty_bins) colindx = 0;
         }
      }

      if (colindx === null) {
         ttrect.remove();
         return null;
      }

      var res = { name: histo.fName, title: histo.fTitle,
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.fillcoloralt('blue') : "blue",
                  lines: this.GetBinTips(i, j), exact: true, menu: true };

      if (this.options.Color) res.color2 = this.GetPalette().getColor(colindx);

      if (pnt.disabled && !this.is_projection) {
         ttrect.remove();
         res.changed = true;
      } else {
         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:path")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         var binid = i*10000 + j;

         if (this.is_projection == "X") {
            x1 = 0; x2 = this.frame_width();
            if (this.projection_width > 1) {
               var dd = (this.projection_width-1)/2;
               if (j2+dd >= h.j2) { j2 = Math.min(Math.round(j2+dd), h.j2); j1 = Math.max(j2 - this.projection_width, h.j1); }
                             else { j1 = Math.max(Math.round(j1-dd), h.j1); j2 = Math.min(j1 + this.projection_width, h.j2); }
            }
            y1 = h.gry[j2]; y2 = h.gry[j1];
            binid = j1*777 + j2*333;
         } else if (this.is_projection == "Y") {
            y1 = 0; y2 = this.frame_height();
            if (this.projection_width > 1) {
               var dd = (this.projection_width-1)/2;
               if (i2+dd >= h.i2) { i2 = Math.min(Math.round(i2+dd), h.i2); i1 = Math.max(i2 - this.projection_width, h.i1); }
                             else { i1 = Math.max(Math.round(i1-dd), h.i1); i2 = Math.min(i1 + this.projection_width, h.i2); }
            }
            x1 = h.grx[i1], x2 = h.grx[i2],
            binid = i1*777 + i2*333;
         }

         res.changed = ttrect.property("current_bin") !== binid;

         if (res.changed)
            ttrect.attr("d", "M"+x1+","+y1 + "h"+(x2-x1) + "v"+(y2-y1) + "h"+(x1-x2) + "z")
                  .style("opacity", "0.7")
                  .property("current_bin", binid);

         if (this.is_projection && res.changed)
            this.RedrawProjection(i1, i2, j1, j2);
      }

      if (res.changed)
         res.user_info = { obj: histo, name: histo.fName,
                           bin: histo.getBin(i+1, j+1), cont: binz, binx: i+1, biny: j+1,
                           grx: pnt.x, gry: pnt.y };

      return res;
   }

   TH2Painter.prototype.CanZoomIn = function(axis,min,max) {
      // check if it makes sense to zoom inside specified axis range
      if (axis=="z") return true;

      var obj = this.GetHisto();
      if (obj) obj = (axis=="y") ? obj.fYaxis : obj.fXaxis;

      return !obj || (obj.FindBin(max,0.5) - obj.FindBin(min,0) > 1);
   }

   TH2Painter.prototype.Draw2D = function(call_back, resize) {

      this.Clear3DScene();
      this.mode3d = false;

      this.CreateXY();

      // draw new palette, resize frame if required
      var pp = this.DrawColorPalette(this.options.Zscale && (this.options.Color || this.options.Contour), true);

      this.DrawAxes();

      this.DrawBins();

      // redraw palette till the end when contours are available
      if (pp) pp.DrawPave();

      this.DrawTitle();

      this.UpdateStatWebCanvas();

      this.AddInteractive();

      JSROOT.CallBack(call_back);
   }

   TH2Painter.prototype.Draw3D = function(call_back, resize) {
      this.mode3d = true;
      JSROOT.AssertPrerequisites('hist3d', function() {
         this.Draw3D(call_back, resize);
      }.bind(this));
   }

   TH2Painter.prototype.CallDrawFunc = function(callback, reason) {

      var main = this.main_painter(),
          fp = this.frame_painter();

     if ((main !== this) && fp && (fp.mode3d !== this.options.Mode3D))
        this.CopyOptionsFrom(main);

      var funcname = this.options.Mode3D ? "Draw3D" : "Draw2D";
      this[funcname](callback, reason);
   }

   TH2Painter.prototype.Redraw = function(reason) {
      this.CallDrawFunc(null, reason);
   }

   function drawHistogram2D(divid, histo, opt) {
      // create painter and add it to canvas
      var painter = new JSROOT.TH2Painter(histo);

      if (!painter.SetDivId(divid, 1)) return null;

      // here we deciding how histogram will look like and how will be shown
      painter.DecodeOptions(opt);

      if (painter.IsTH2Poly()) {
         if (painter.options.Mode3D) painter.options.Lego = 12; // lego always 12
         else if (!painter.options.Color) painter.options.Color = true; // default is color
      }

      painter._show_empty_bins = false;

      painter._can_move_colz = true;

      // special case for root 3D drawings - pad range is wired
      painter.CheckPadRange(!painter.options.Mode3D && (painter.options.Contour != 14));

      painter.ScanContent();

      painter.CreateStat(); // only when required

      painter.CallDrawFunc(function() {
         this.DrawNextFunction(0, function() {
            if (!this.Mode3D && this.options.AutoZoom) this.AutoZoom();
            this.FillToolbar();
            if (this.options.Project && !this.mode3d)
               this.ToggleProjection(this.options.Project);
            this.DrawingReady();
         }.bind(this));
      }.bind(painter));

      return painter;
   }

   // =================================================================================


   function createTF2Histogram(func, nosave, hist) {
      var nsave = 0, npx = 0, npy = 0;
      if (!nosave) {
         nsave = func.fSave.length;
         npx = Math.round(func.fSave[nsave-2]);
         npy = Math.round(func.fSave[nsave-1]);
         if (nsave !== (npx+1)*(npy+1) + 6) nsave = 0;
      }

      if (nsave > 6) {
         var dx = (func.fSave[nsave-5] - func.fSave[nsave-6]) / npx / 2,
             dy = (func.fSave[nsave-3] - func.fSave[nsave-4]) / npy / 2;

         if (!hist) hist = JSROOT.CreateHistogram("TH2F", npx+1, npy+1);

         hist.fXaxis.fXmin = func.fSave[nsave-6] - dx;
         hist.fXaxis.fXmax = func.fSave[nsave-5] + dx;

         hist.fYaxis.fXmin = func.fSave[nsave-4] - dy;
         hist.fYaxis.fXmax = func.fSave[nsave-3] + dy;

         for (var k=0,j=0;j<=npy;++j)
            for (var i=0;i<=npx;++i)
               hist.setBinContent(hist.getBin(i+1,j+1), func.fSave[k++]);

      } else {
         npx = Math.max(func.fNpx, 2);
         npy = Math.max(func.fNpy, 2);

         if (!hist) hist = JSROOT.CreateHistogram("TH2F", npx, npy);

         hist.fXaxis.fXmin = func.fXmin;
         hist.fXaxis.fXmax = func.fXmax;

         hist.fYaxis.fXmin = func.fYmin;
         hist.fYaxis.fXmax = func.fYmax;

         for (var j=0;j<npy;++j)
           for (var i=0;i<npx;++i) {
               var x = func.fXmin + (i + 0.5) * (func.fXmax - func.fXmin) / npx,
                   y = func.fYmin + (j + 0.5) * (func.fYmax - func.fYmin) / npy;

               hist.setBinContent(hist.getBin(i+1,j+1), func.evalPar(x,y));
            }
      }

      hist.fName = "Func";
      hist.fTitle = func.fTitle;
      hist.fMinimum = func.fMinimum;
      hist.fMaximum = func.fMaximum;
      //fHistogram->SetContour(fContour.fN, levels);
      hist.fLineColor = func.fLineColor;
      hist.fLineStyle = func.fLineStyle;
      hist.fLineWidth = func.fLineWidth;
      hist.fFillColor = func.fFillColor;
      hist.fFillStyle = func.fFillStyle;
      hist.fMarkerColor = func.fMarkerColor;
      hist.fMarkerStyle = func.fMarkerStyle;
      hist.fMarkerSize = func.fMarkerSize;

      hist.fBits |= JSROOT.TH1StatusBits.kNoStats;

      // only for testing - unfortunately, axis settings are not stored with TF2
      // hist.fXaxis.fTitle = "axis X";
      // hist.fXaxis.InvertBit(JSROOT.EAxisBits.kCenterTitle);
      // hist.fYaxis.fTitle = "axis Y";
      // hist.fYaxis.InvertBit(JSROOT.EAxisBits.kCenterTitle);
      // hist.fZaxis.fTitle = "axis Z";
      // hist.fZaxis.InvertBit(JSROOT.EAxisBits.kCenterTitle);

      return hist;
   }

   // TF2 always drawn via temporary TH2 object,
   // therefore there is no special painter class

   function drawTF2(divid, func, opt) {

      var d = new JSROOT.DrawOptions(opt);

      var hist = createTF2Histogram(func, d.check('NOSAVE'));

      if (d.empty()) opt = "cont3"; else
      if (d.opt === "SAME") opt = "cont2 same";
      else opt = d.opt;

      var hpainter = drawHistogram2D(divid, hist, opt);

      hpainter.tf2_typename = func._typename;
      hpainter.tf2_nosave = d.check('NOSAVE');

      hpainter.UpdateObject = function(obj, opt) {
         if (!obj || (this.tf2_typename != obj._typename)) return false;

         createTF2Histogram(obj, this.tf2_nosave, this.GetHisto());

         return true;
      }

      return hpainter;
   }

   // ====================================================================

   function THStackPainter(stack, opt) {
      JSROOT.TObjectPainter.call(this, stack, opt);

      this.firstpainter = null;
      this.painters = []; // keep painters to be able update objects
   }

   THStackPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   THStackPainter.prototype.Cleanup = function() {
      var pp = this.pad_painter();
      if (pp) pp.CleanPrimitives(this.Selector.bind(this, true));
      delete this.firstpainter;
      delete this.painters;
      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   THStackPainter.prototype.HasErrors = function(hist) {
      if (hist.fSumw2 && (hist.fSumw2.length > 0))
         for (var n=0;n<hist.fSumw2.length;++n)
            if (hist.fSumw2[n] > 0) return true;
      return false;
   }

   THStackPainter.prototype.BuildStack = function(stack) {
      //  build sum of all histograms
      //  Build a separate list fStack containing the running sum of all histograms

      if (!stack.fHists) return false;
      var nhists = stack.fHists.arr.length;
      if (nhists <= 0) return false;
      var lst = JSROOT.Create("TList");
      lst.Add(JSROOT.clone(stack.fHists.arr[0]), stack.fHists.opt[0]);
      for (var i=1;i<nhists;++i) {
         var hnext = JSROOT.clone(stack.fHists.arr[i]),
             hnextopt = stack.fHists.opt[i],
             hprev = lst.arr[i-1];

         if ((hnext.fNbins != hprev.fNbins) ||
             (hnext.fXaxis.fXmin != hprev.fXaxis.fXmin) ||
             (hnext.fXaxis.fXmax != hprev.fXaxis.fXmax)) {
            JSROOT.console("When drawing THStack, cannot sum-up histograms " + hnext.fName + " and " + hprev.fName);
            lst.Clear();
            return false;
         }

         // trivial sum of histograms
         for (var n = 0; n < hnext.fArray.length; ++n)
            hnext.fArray[n] += hprev.fArray[n];

         lst.Add(hnext, hnextopt);
      }
      stack.fStack = lst;
      return true;
   }

   THStackPainter.prototype.GetHistMinMax = function(hist, witherr) {
      var res = { min : 0, max : 0 },
          domin = true, domax = true;
      if (hist.fMinimum !== -1111) {
         res.min = hist.fMinimum;
         domin = false;
      }
      if (hist.fMaximum !== -1111) {
         res.max = hist.fMaximum;
         domax = false;
      }

      if (!domin && !domax) return res;

      var i1 = 1, i2 = hist.fXaxis.fNbins, j1 = 1, j2 = 1, first = true;

      if (hist.fXaxis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
         i1 = hist.fXaxis.fFirst;
         i2 = hist.fXaxis.fLast;
      }

      if (hist._typename.indexOf("TH2")===0) {
         j2 = hist.fYaxis.fNbins;
         if (hist.fYaxis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
            j1 = hist.fYaxis.fFirst;
            j2 = hist.fYaxis.fLast;
         }
      }
      for (var j=j1; j<=j2;++j)
         for (var i=i1; i<=i2;++i) {
            var val = hist.getBinContent(i, j),
                err = witherr ? hist.getBinError(hist.getBin(i,j)) : 0;
            if (domin && (first || (val-err < res.min))) res.min = val-err;
            if (domax && (first || (val+err > res.max))) res.max = val+err;
            first = false;
        }

      return res;
   }

   THStackPainter.prototype.GetMinMax = function(iserr, pad) {
      var res = { min: 0, max: 0 },
          stack = this.GetObject();

      if (this.options.nostack) {
         for (var i = 0; i < stack.fHists.arr.length; ++i) {
            var resh = this.GetHistMinMax(stack.fHists.arr[i], iserr);
            if (i==0) {
               res = resh;
             } else {
               res.min = Math.min(res.min, resh.min);
               res.max = Math.max(res.max, resh.max);
            }
         }
      } else {
         res.min = this.GetHistMinMax(stack.fStack.arr[0], iserr).min;
         res.max = this.GetHistMinMax(stack.fStack.arr[stack.fStack.arr.length-1], iserr).max;
      }

      if (stack.fMaximum != -1111) res.max = stack.fMaximum;
      res.max *= 1.05;
      if (stack.fMinimum != -1111) res.min = stack.fMinimum;

      if (pad && (this.options.ndim == 1 ? pad.fLogy : pad.fLogz)) {
         if (res.max<=0) res.max = 1;
         if (res.min<=0) res.min = 1e-4*res.max;
         var kmin = 1/(1 + 0.5*JSROOT.log10(res.max / res.min)),
             kmax = 1 + 0.2*JSROOT.log10(res.max / res.min);
         res.min *= kmin;
         res.max *= kmax;
      } else {
         if ((res.min>0) && (res.min < 0.05*res.max)) res.min = 0;
      }

      return res;
   }

   THStackPainter.prototype.DrawNextHisto = function(indx, mm, subp, reenter) {
      if (mm === "callback") {
         mm = null; // just misuse min/max argument to indicate callback
         if (indx<0) this.firstpainter = subp;
                else this.painters.push(subp);
         indx++;
      }

      var stack = this.GetObject(),
          hist = stack.fHistogram, hopt = "",
          hlst = this.options.nostack ? stack.fHists : stack.fStack,
          nhists = (hlst && hlst.arr) ? hlst.arr.length : 0, rindx = 0;

      if (indx>=nhists) return this.DrawingReady();

      if ((indx % 500 === 499) && !reenter)
         return setTimeout(this.DrawNextHisto.bind(this, indx, mm, subp, true), 0);

      if (indx>=0) {
         rindx = this.options.horder ? indx : nhists-indx-1;
         hist = hlst.arr[rindx];
         hopt = hlst.opt[rindx] || hist.fOption || this.options.hopt;
         if (hopt.toUpperCase().indexOf(this.options.hopt)<0) hopt += this.options.hopt;
         if (this.options.draw_errors && !hopt) hopt = "E";
         hopt += " same nostat";

         // if there is auto colors assignment, try to provide it
         if (this.options._pfc || this.options._plc || this.options._pmc) {
            if (!this.palette && JSROOT.Painter.GetColorPalette)
               this.palette = JSROOT.Painter.GetColorPalette();
            if (this.palette) {
               var color = this.palette.calcColor(rindx, nhists+1);
               var icolor = this.add_color(color);

               if (this.options._pfc) hist.fFillColor = icolor;
               if (this.options._plc) hist.fLineColor = icolor;
               if (this.options._pmc) hist.fMarkerColor = icolor;
            }
         }

      } else {
         hopt = this.options.hopt + " axis";
         // if (mm && (!this.options.nostack || (hist.fMinimum==-1111 && hist.fMaximum==-1111))) hopt += ";minimum:" + mm.min + ";maximum:" + mm.max;
         if (mm) hopt += ";minimum:" + mm.min + ";maximum:" + mm.max;
      }

      // special handling of stacked histograms - set $baseh object for correct drawing
      // also used to provide tooltips
      if ((rindx > 0) && !this.options.nostack) hist.$baseh = hlst.arr[rindx - 1];

      JSROOT.draw(this.divid, hist, hopt, this.DrawNextHisto.bind(this, indx, "callback"));
   }

   THStackPainter.prototype.DecodeOptions = function(opt) {
      if (!this.options) this.options = {};
      JSROOT.extend(this.options, { ndim: 1, nostack: false, same: false, horder: true, has_errors: false, draw_errors: false, hopt: "" });

      var stack = this.GetObject(),
          hist = stack.fHistogram || (stack.fHists ? stack.fHists.arr[0] : null) || (stack.fStack ? stack.fStack.arr[0] : null);

      if (hist && (hist._typename.indexOf("TH2")==0)) this.options.ndim = 2;

      if ((this.options.ndim==2) && !opt) opt = "lego1";

      if (stack.fHists && !this.options.nostack) {
         for (var k=0;k<stack.fHists.arr.length;++k)
            this.options.has_errors = this.options.has_errors || this.HasErrors(stack.fHists.arr[k]);
      }

      var d = new JSROOT.DrawOptions(opt);

      this.options.nostack = d.check("NOSTACK");
      if (d.check("STACK")) this.options.nostack = false;
      this.options.same = d.check("SAME");

      this.options._pfc = d.check("PFC");
      this.options._plc = d.check("PLC");
      this.options._pmc = d.check("PMC");

      this.options.hopt = d.remain(); // use remaining draw options for histogram draw

      var dolego = d.check("LEGO");

      this.options.errors = d.check("E");

      // if any histogram appears with pre-calculated errors, use E for all histograms
      if (!this.options.nostack && this.options.has_errors && !dolego && !d.check("HIST") && (this.options.hopt.indexOf("E")<0)) this.options.draw_errors = true;

      this.options.horder = this.options.nostack || dolego;
   }

   THStackPainter.prototype.CreateHistogram = function(stack) {
      var histos = stack.fHists,
          numhistos = histos ? histos.arr.length : 0;

      if (!numhistos) {
         var histo = JSROOT.CreateHistogram("TH1I", 100);
         histo.fTitle = stack.fTitle;
         return histo;
      }

      var h0 = histos.arr[0];
      var histo = JSROOT.CreateHistogram((this.options.ndim==1) ? "TH1I" : "TH2I", h0.fXaxis.fNbins, h0.fYaxis.fNbins);
      histo.fName = "axis_hist";
      JSROOT.extend(histo.fXaxis, h0.fXaxis);
      if (this.options.ndim==2)
         JSROOT.extend(histo.fYaxis, h0.fYaxis);

      // this code is not exists in ROOT painter, can be skipped?
      for (var n=1;n<numhistos;++n) {
         var h = histos.arr[n];

         if (!histo.fXaxis.fLabels) {
            histo.fXaxis.fXmin = Math.min(histo.fXaxis.fXmin, h.fXaxis.fXmin);
            histo.fXaxis.fXmax = Math.max(histo.fXaxis.fXmax, h.fXaxis.fXmax);
         }

         if ((this.options.ndim==2) && !histo.fYaxis.fLabels) {
            histo.fYaxis.fXmin = Math.min(histo.fYaxis.fXmin, h.fYaxis.fXmin);
            histo.fYaxis.fXmax = Math.max(histo.fYaxis.fXmax, h.fYaxis.fXmax);
         }
      }

      histo.fTitle = stack.fTitle;

      return histo;
   }

   THStackPainter.prototype.drawStack = function() {

      var stack = this.GetObject();

      // when building stack, one could fail to sum up histograms
      if (!this.options.nostack)
         this.options.nostack = ! this.BuildStack(stack);

      if (!stack.fHistogram && !this.options.same)
         stack.fHistogram = this.CreateHistogram(stack);

      var mm = this.GetMinMax(this.options.errors || this.options.draw_errors, this.root_pad());

      this.DrawNextHisto(this.options.same ? 0 : -1, mm);
      return this;
   }

   THStackPainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;

      var stack = this.GetObject();

      stack.fHists = obj.fHists;
      stack.fStack = obj.fStack;

      if (!this.options.nostack) {
         this.options.nostack = !this.BuildStack(stack);
      }

      var isany = false;
      if (this.firstpainter) {
         var src = obj.fHistogram;
         if (!src)
            src = stack.fHistogram = this.CreateHistogram(stack);

         this.firstpainter.UpdateObject(src);

         var mm = this.GetMinMax(this.options.errors || this.options.draw_errors, this.root_pad());

         this.firstpainter.options.minimum = mm.min;
         this.firstpainter.options.maximum = mm.max;

         if (this.options.ndim == 1) {
            this.firstpainter.ymin = mm.min;
            this.firstpainter.ymax = mm.max;
         }

         isany = true;
      }

      // try fully remove old histogram painters
      var pp = this.pad_painter();
      if (pp) pp.CleanPrimitives(this.Selector.bind(this, false));
      this.painters = [];

      this.did_update = isany;

      return isany;
   }

   /** @summary Returns true if painter belongs to stack, used in cleanup */
   THStackPainter.prototype.Selector = function(fullclear, painter) {
      if (fullclear && (painter===this.firstpainter)) return true;
      return this.painters.indexOf(painter) >= 0;
   }

   /** @summary Redraw THStack, changes output only if Update was performed before */
   THStackPainter.prototype.Redraw = function() {
      // do nothing in case of simple redraw
      if (!this.did_update) return;

      // remove flag set in update
      delete this.did_update;

      if (this.firstpainter)
         this.firstpainter.Redraw();

      this.DrawNextHisto(0, null);
   }

   function drawHStack(divid, stack, opt) {
      // paint the list of histograms
      // By default, histograms are shown stacked.
      // - the first histogram is paint
      // - then the sum of the first and second, etc

      var painter = new THStackPainter(stack, opt);
      painter.SetDivId(divid, -1); // it maybe no element to set divid
      painter.DecodeOptions(opt);

      if (!stack.fHists || (stack.fHists.arr.length == 0)) return painter.DrawingReady();

      painter.drawStack();

      painter.SetDivId(divid); // only when first histogram drawn, we could assign divid

      return painter;
   }

   // =================================================================================

   // kept for backward compatibility, will be removed in future JSROOT versions
   JSROOT.Painter.drawLegend = drawPave;
   JSROOT.Painter.drawPaveText = drawPave;

   JSROOT.Painter.drawPave = drawPave;
   JSROOT.Painter.produceLegend = produceLegend;
   JSROOT.Painter.drawHistogram1D = drawHistogram1D;
   JSROOT.Painter.drawHistogram2D = drawHistogram2D;
   JSROOT.Painter.drawTF2 = drawTF2;
   JSROOT.Painter.drawHStack = drawHStack;

   JSROOT.TPavePainter = TPavePainter;
   JSROOT.THistPainter = THistPainter;
   JSROOT.TH1Painter = TH1Painter;
   JSROOT.TH2Painter = TH2Painter;
   JSROOT.THStackPainter = THStackPainter;

   return JSROOT;

}));
