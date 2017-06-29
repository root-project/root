/// @file JSRootPainter.hist.js
/// JavaScript ROOT graphics for histogram classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"), require("./d3.min.js"));
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

   JSROOT.sources.push("hist");

   JSROOT.ToolbarIcons.th2color = {
       recs: [{x:0,y:256,w:13,h:39,f:'rgb(38,62,168)'},{x:13,y:371,w:39,h:39},{y:294,h:39},{y:256,h:39},{y:218,h:39},{x:51,y:410,w:39,h:39},{y:371,h:39},{y:333,h:39},{y:294},{y:256,h:39},{y:218,h:39},{y:179,h:39},{y:141,h:39},{y:102,h:39},{y:64},{x:90,y:448,w:39,h:39},{y:410},{y:371,h:39},{y:333,h:39,f:'rgb(22,82,205)'},{y:294},{y:256,h:39,f:'rgb(16,100,220)'},{y:218,h:39},{y:179,h:39,f:'rgb(22,82,205)'},{y:141,h:39},{y:102,h:39,f:'rgb(38,62,168)'},{y:64},{y:0,h:27},{x:128,y:448,w:39,h:39},{y:410},{y:371,h:39},{y:333,h:39,f:'rgb(22,82,205)'},{y:294,f:'rgb(20,129,214)'},{y:256,h:39,f:'rgb(9,157,204)'},{y:218,h:39,f:'rgb(14,143,209)'},{y:179,h:39,f:'rgb(20,129,214)'},{y:141,h:39,f:'rgb(16,100,220)'},{y:102,h:39,f:'rgb(22,82,205)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{y:0,h:27},{x:166,y:486,h:14},{y:448,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39,f:'rgb(20,129,214)'},{y:294,f:'rgb(82,186,146)'},{y:256,h:39,f:'rgb(179,189,101)'},{y:218,h:39,f:'rgb(116,189,129)'},{y:179,h:39,f:'rgb(82,186,146)'},{y:141,h:39,f:'rgb(14,143,209)'},{y:102,h:39,f:'rgb(16,100,220)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:205,y:486,w:39,h:14},{y:448,h:39},{y:410},{y:371,h:39,f:'rgb(16,100,220)'},{y:333,h:39,f:'rgb(9,157,204)'},{y:294,f:'rgb(149,190,113)'},{y:256,h:39,f:'rgb(244,198,59)'},{y:218,h:39},{y:179,h:39,f:'rgb(226,192,75)'},{y:141,h:39,f:'rgb(13,167,195)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(22,82,205)'},{y:26,h:39,f:'rgb(38,62,168)'},{x:243,y:448,w:39,h:39},{y:410},{y:371,h:39,f:'rgb(18,114,217)'},{y:333,h:39,f:'rgb(30,175,179)'},{y:294,f:'rgb(209,187,89)'},{y:256,h:39,f:'rgb(251,230,29)'},{y:218,h:39,f:'rgb(249,249,15)'},{y:179,h:39,f:'rgb(226,192,75)'},{y:141,h:39,f:'rgb(30,175,179)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:282,y:448,h:39},{y:410},{y:371,h:39,f:'rgb(18,114,217)'},{y:333,h:39,f:'rgb(14,143,209)'},{y:294,f:'rgb(149,190,113)'},{y:256,h:39,f:'rgb(226,192,75)'},{y:218,h:39,f:'rgb(244,198,59)'},{y:179,h:39,f:'rgb(149,190,113)'},{y:141,h:39,f:'rgb(9,157,204)'},{y:102,h:39,f:'rgb(18,114,217)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:320,y:448,w:39,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39,f:'rgb(20,129,214)'},{y:294,f:'rgb(46,183,164)'},{y:256,h:39},{y:218,h:39,f:'rgb(82,186,146)'},{y:179,h:39,f:'rgb(9,157,204)'},{y:141,h:39,f:'rgb(20,129,214)'},{y:102,h:39,f:'rgb(16,100,220)'},{y:64,f:'rgb(38,62,168)'},{y:26,h:39},{x:358,y:448,h:39},{y:410},{y:371,h:39,f:'rgb(22,82,205)'},{y:333,h:39},{y:294,f:'rgb(16,100,220)'},{y:256,h:39,f:'rgb(20,129,214)'},{y:218,h:39,f:'rgb(14,143,209)'},{y:179,h:39,f:'rgb(18,114,217)'},{y:141,h:39,f:'rgb(22,82,205)'},{y:102,h:39,f:'rgb(38,62,168)'},{y:64},{y:26,h:39},{x:397,y:448,w:39,h:39},{y:371,h:39},{y:333,h:39},{y:294,f:'rgb(22,82,205)'},{y:256,h:39},{y:218,h:39},{y:179,h:39,f:'rgb(38,62,168)'},{y:141,h:39},{y:102,h:39},{y:64},{y:26,h:39},{x:435,y:410,h:39},{y:371,h:39},{y:333,h:39},{y:294},{y:256,h:39},{y:218,h:39},{y:179,h:39},{y:141,h:39},{y:102,h:39},{y:64},{x:474,y:256,h:39},{y:179,h:39}]
   };

   JSROOT.ToolbarIcons.th2colorz = {
      recs: [{x:128,y:486,w:256,h:26,f:'rgb(38,62,168)'},{y:461,f:'rgb(22,82,205)'},{y:435,f:'rgb(16,100,220)'},{y:410,f:'rgb(18,114,217)'},{y:384,f:'rgb(20,129,214)'},{y:358,f:'rgb(14,143,209)'},{y:333,f:'rgb(9,157,204)'},{y:307,f:'rgb(13,167,195)'},{y:282,f:'rgb(30,175,179)'},{y:256,f:'rgb(46,183,164)'},{y:230,f:'rgb(82,186,146)'},{y:205,f:'rgb(116,189,129)'},{y:179,f:'rgb(149,190,113)'},{y:154,f:'rgb(179,189,101)'},{y:128,f:'rgb(209,187,89)'},{y:102,f:'rgb(226,192,75)'},{y:77,f:'rgb(244,198,59)'},{y:51,f:'rgb(253,210,43)'},{y:26,f:'rgb(251,230,29)'},{y:0,f:'rgb(249,249,15)'}]
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
         return 'rgb(' + Math.round(r * 255) + ',' + Math.round(g * 255) + ',' + Math.round(b * 255) + ')';
      }

      var palette = [], saturation = 1, lightness = 0.5, maxHue = 280, minHue = 0, maxPretty = 50;
      for (var i = 0; i < maxPretty; ++i) {
         var hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty)) / 360.0;
         var rgbval = HLStoRGB(hue, lightness, saturation);
         palette.push(rgbval);
      }
      return palette;
   }

   JSROOT.Painter.CreateGrayPalette = function() {
      var palette = [];
      for (var i = 0; i < 50; ++i) {
         var code = Math.round((i+2)/60 * 255 );
         palette.push('rgb('+code+','+code+','+code+')');
      }
      return palette;
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

       return palette;
   }

   JSROOT.Painter.GetColorPalette = function(col,alfa) {
      if ((col===null) || (col===0)) col = JSROOT.gStyle.Palette;
      if ((col>0) && (col<10)) return JSROOT.Painter.CreateGrayPalette();
      if (col < 51) return JSROOT.Painter.CreateDefaultPalette();
      if (col > 112) col = 57;
      var red, green, blue,
          stops = [ 0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000 ];
      switch(col) {

         // Deep Sea
         case 51:
            red   = [ 0,  9, 13, 17, 24,  32,  27,  25,  29];
            green = [ 0,  0,  0,  2, 37,  74, 113, 160, 221];
            blue  = [ 28, 42, 59, 78, 98, 129, 154, 184, 221];
            break;

         // Grey Scale
         case 52:
            red = [ 0, 32, 64, 96, 128, 160, 192, 224, 255];
            green = [ 0, 32, 64, 96, 128, 160, 192, 224, 255];
            blue = [ 0, 32, 64, 96, 128, 160, 192, 224, 255];
            break;

         // Dark Body Radiator
         case 53:
            red = [ 0, 45, 99, 156, 212, 230, 237, 234, 242];
            green = [ 0,  0,  0,  45, 101, 168, 238, 238, 243];
            blue = [ 0,  1,  1,   3,   9,   8,  11,  95, 230];
            break;

         // Two-color hue (dark blue through neutral gray to bright yellow)
         case 54:
            red = [  0,  22, 44, 68, 93, 124, 160, 192, 237];
            green = [  0,  16, 41, 67, 93, 125, 162, 194, 241];
            blue = [ 97, 100, 99, 99, 93,  68,  44,  26,  74];
            break;

         // Rain Bow
         case 55:
            red = [  0,   5,  15,  35, 102, 196, 208, 199, 110];
            green = [  0,  48, 124, 192, 206, 226,  97,  16,   0];
            blue = [ 99, 142, 198, 201,  90,  22,  13,   8,   2];
            break;

         // Inverted Dark Body Radiator
         case 56:
            red = [ 242, 234, 237, 230, 212, 156, 99, 45, 0];
            green = [ 243, 238, 238, 168, 101,  45,  0,  0, 0];
            blue = [ 230,  95,  11,   8,   9,   3,  1,  1, 0];
            break;

         // Bird
         case 57:
            red = [ 0.2082*255, 0.0592*255, 0.0780*255, 0.0232*255, 0.1802*255, 0.5301*255, 0.8186*255, 0.9956*255, 0.9764*255];
            green = [ 0.1664*255, 0.3599*255, 0.5041*255, 0.6419*255, 0.7178*255, 0.7492*255, 0.7328*255, 0.7862*255, 0.9832*255];
            blue = [ 0.5293*255, 0.8684*255, 0.8385*255, 0.7914*255, 0.6425*255, 0.4662*255, 0.3499*255, 0.1968*255, 0.0539*255];
            break;

         // Cubehelix
         case 58:
            red = [ 0.0000, 0.0956*255, 0.0098*255, 0.2124*255, 0.6905*255, 0.9242*255, 0.7914*255, 0.7596*255, 1.0000*255];
            green = [ 0.0000, 0.1147*255, 0.3616*255, 0.5041*255, 0.4577*255, 0.4691*255, 0.6905*255, 0.9237*255, 1.0000*255];
            blue = [ 0.0000, 0.2669*255, 0.3121*255, 0.1318*255, 0.2236*255, 0.6741*255, 0.9882*255, 0.9593*255, 1.0000*255];
            break;

         // Green Red Violet
         case 59:
            red = [13, 23, 25, 63, 76, 104, 137, 161, 206];
            green = [95, 67, 37, 21,  0,  12,  35,  52,  79];
            blue = [ 4,  3,  2,  6, 11,  22,  49,  98, 208];
            break;

         // Blue Red Yellow
         case 60:
            red = [0,  61,  89, 122, 143, 160, 185, 204, 231];
            green = [0,   0,   0,   0,  14,  37,  72, 132, 235];
            blue = [0, 140, 224, 144,   4,   5,   6,   9,  13];
            break;

         // Ocean
         case 61:
            red = [ 14,  7,  2,  0,  5,  11,  55, 131, 229];
            green = [105, 56, 26,  1, 42,  74, 131, 171, 229];
            blue = [  2, 21, 35, 60, 92, 113, 160, 185, 229];
            break;

         // Color Printable On Grey
         case 62:
            red = [ 0,   0,   0,  70, 148, 231, 235, 237, 244];
            green = [ 0,   0,   0,   0,   0,  69,  67, 216, 244];
            blue = [ 0, 102, 228, 231, 177, 124, 137,  20, 244];
            break;

         // Alpine
         case 63:
            red = [ 50, 56, 63, 68,  93, 121, 165, 192, 241];
            green = [ 66, 81, 91, 96, 111, 128, 155, 189, 241];
            blue = [ 97, 91, 75, 65,  77, 103, 143, 167, 217];
            break;

         // Aquamarine
         case 64:
            red = [ 145, 166, 167, 156, 131, 114, 101, 112, 132];
            green = [ 158, 178, 179, 181, 163, 154, 144, 152, 159];
            blue = [ 190, 199, 201, 192, 176, 169, 160, 166, 190];
            break;

         // Army
         case 65:
            red = [ 93,   91,  99, 108, 130, 125, 132, 155, 174];
            green = [ 126, 124, 128, 129, 131, 121, 119, 153, 173];
            blue = [ 103,  94,  87,  85,  80,  85, 107, 120, 146];
            break;

         // Atlantic
         case 66:
            red = [ 24, 40, 69,  90, 104, 114, 120, 132, 103];
            green = [ 29, 52, 94, 127, 150, 162, 159, 151, 101];
            blue = [ 29, 52, 96, 132, 162, 181, 184, 186, 131];
            break;

         // Aurora
         case 67:
            red = [ 46, 38, 61, 92, 113, 121, 132, 150, 191];
            green = [ 46, 36, 40, 69, 110, 135, 131,  92,  34];
            blue = [ 46, 80, 74, 70,  81, 105, 165, 211, 225];
            break;

         // Avocado
         case 68:
            red = [ 0,  4, 12,  30,  52, 101, 142, 190, 237];
            green = [ 0, 40, 86, 121, 140, 172, 187, 213, 240];
            blue = [ 0,  9, 14,  18,  21,  23,  27,  35, 101];
            break;

         // Beach
         case 69:
            red = [ 198, 206, 206, 211, 198, 181, 161, 171, 244];
            green = [ 103, 133, 150, 172, 178, 174, 163, 175, 244];
            blue = [  49,  54,  55,  66,  91, 130, 184, 224, 244];
            break;

         // Black Body
         case 70:
            red = [ 243, 243, 240, 240, 241, 239, 186, 151, 129];
            green = [ 0,  46,  99, 149, 194, 220, 183, 166, 147];
            blue = [ 6,   8,  36,  91, 169, 235, 246, 240, 233];
            break;

         // Blue Green Yellow
         case 71:
            red = [ 22, 19,  19,  25,  35,  53,  88, 139, 210];
            green = [ 0, 32,  69, 108, 135, 159, 183, 198, 215];
            blue = [ 77, 96, 110, 116, 110, 100,  90,  78,  70];
            break;

         // Brown Cyan
         case 72:
            red = [ 68, 116, 165, 182, 189, 180, 145, 111,  71];
            green = [ 37,  82, 135, 178, 204, 225, 221, 202, 147];
            blue = [ 16,  55, 105, 147, 196, 226, 232, 224, 178];
            break;

         // CMYK
         case 73:
            red = [ 61,  99, 136, 181, 213, 225, 198, 136, 24];
            green = [ 149, 140,  96,  83, 132, 178, 190, 135, 22];
            blue = [ 214, 203, 168, 135, 110, 100, 111, 113, 22];
            break;

         // Candy
         case 74:
            red = [ 76, 120, 156, 183, 197, 180, 162, 154, 140];
            green = [ 34,  35,  42,  69, 102, 137, 164, 188, 197];
            blue = [ 64,  69,  78, 105, 142, 177, 205, 217, 198];
            break;

         // Cherry
         case 75:
            red = [ 37, 102, 157, 188, 196, 214, 223, 235, 251];
            green = [ 37,  29,  25,  37,  67,  91, 132, 185, 251];
            blue = [ 37,  32,  33,  45,  66,  98, 137, 187, 251];
            break;

         // Coffee
         case 76:
            red = [ 79, 100, 119, 137, 153, 172, 192, 205, 250];
            green = [ 63,  79,  93, 103, 115, 135, 167, 196, 250];
            blue = [ 51,  59,  66,  61,  62,  70, 110, 160, 250];
            break;

         // Dark Rain Bow
         case 77:
            red = [  43,  44, 50,  66, 125, 172, 178, 155, 157];
            green = [  63,  63, 85, 101, 138, 163, 122,  51,  39];
            blue = [ 121, 101, 58,  44,  47,  55,  57,  44,  43];
            break;

         // Dark Terrain
         case 78:
            red = [  0, 41, 62, 79, 90, 87, 99, 140, 228];
            green = [  0, 57, 81, 93, 85, 70, 71, 125, 228];
            blue = [ 95, 91, 91, 82, 60, 43, 44, 112, 228];
            break;

         // Fall
         case 79:
            red = [ 49, 59, 72, 88, 114, 141, 176, 205, 222];
            green = [ 78, 72, 66, 57,  59,  75, 106, 142, 173];
            blue = [ 78, 55, 46, 40,  39,  39,  40,  41,  47];
            break;

         // Fruit Punch
         case 80:
            red = [ 243, 222, 201, 185, 165, 158, 166, 187, 219];
            green = [  94, 108, 132, 135, 125,  96,  68,  51,  61];
            blue = [   7,  9,   12,  19,  45,  89, 118, 146, 118];
            break;

         // Fuchsia
         case 81:
            red = [ 19, 44, 74, 105, 137, 166, 194, 206, 220];
            green = [ 19, 28, 40,  55,  82, 110, 159, 181, 220];
            blue = [ 19, 42, 68,  96, 129, 157, 188, 203, 220];
            break;

         // Grey Yellow
         case 82:
            red = [ 33, 44, 70,  99, 140, 165, 199, 211, 216];
            green = [ 38, 50, 76, 105, 140, 165, 191, 189, 167];
            blue = [ 55, 67, 97, 124, 140, 166, 163, 129,  52];
            break;

         // Green Brown Terrain
         case 83:
            red = [ 0, 33, 73, 124, 136, 152, 159, 171, 223];
            green = [ 0, 43, 92, 124, 134, 126, 121, 144, 223];
            blue = [ 0, 43, 68,  76,  73,  64,  72, 114, 223];
            break;

         // Green Pink
         case 84:
            red = [  5,  18,  45, 124, 193, 223, 205, 128, 49];
            green = [ 48, 134, 207, 230, 193, 113,  28,   0,  7];
            blue = [  6,  15,  41, 121, 193, 226, 208, 130, 49];
            break;

         // Island
         case 85:
            red = [ 180, 106, 104, 135, 164, 188, 189, 165, 144];
            green = [  72, 126, 154, 184, 198, 207, 205, 190, 179];
            blue = [  41, 120, 158, 188, 194, 181, 145, 100,  62];
            break;

         // Lake
         case 86:
            red = [  57,  72,  94, 117, 136, 154, 174, 192, 215];
            green = [   0,  33,  68, 109, 140, 171, 192, 196, 209];
            blue = [ 116, 137, 173, 201, 200, 201, 203, 190, 187];
            break;

         // Light Temperature
         case 87:
            red = [  31,  71, 123, 160, 210, 222, 214, 199, 183];
            green = [  40, 117, 171, 211, 231, 220, 190, 132,  65];
            blue = [ 234, 214, 228, 222, 210, 160, 105,  60,  34];
            break;

         // Light Terrain
         case 88:
            red = [ 123, 108, 109, 126, 154, 172, 188, 196, 218];
            green = [ 184, 138, 130, 133, 154, 175, 188, 196, 218];
            blue = [ 208, 130, 109,  99, 110, 122, 150, 171, 218];
            break;

         // Mint
         case 89:
            red = [ 105, 106, 122, 143, 159, 172, 176, 181, 207];
            green = [ 252, 197, 194, 187, 174, 162, 153, 136, 125];
            blue = [ 146, 133, 144, 155, 163, 167, 166, 162, 174];
            break;

         // Neon
         case 90:
            red = [ 171, 141, 145, 152, 154, 159, 163, 158, 177];
            green = [ 236, 143, 100,  63,  53,  55,  44,  31,   6];
            blue = [  59,  48,  46,  44,  42,  54,  82, 112, 179];
            break;

         // Pastel
         case 91:
            red = [ 180, 190, 209, 223, 204, 228, 205, 152,  91];
            green = [  93, 125, 147, 172, 181, 224, 233, 198, 158];
            blue = [ 236, 218, 160, 133, 114, 132, 162, 220, 218];
            break;

         // Pearl
         case 92:
            red = [ 225, 183, 162, 135, 115, 111, 119, 145, 211];
            green = [ 205, 177, 166, 135, 124, 117, 117, 132, 172];
            blue = [ 186, 165, 155, 135, 126, 130, 150, 178, 226];
            break;

         // Pigeon
         case 93:
            red = [ 39, 43, 59, 63, 80, 116, 153, 177, 223];
            green = [ 39, 43, 59, 74, 91, 114, 139, 165, 223];
            blue = [ 39, 50, 59, 70, 85, 115, 151, 176, 223];
            break;

         // Plum
         case 94:
            red = [ 0, 38, 60, 76, 84, 89, 101, 128, 204];
            green = [ 0, 10, 15, 23, 35, 57,  83, 123, 199];
            blue = [ 0, 11, 22, 40, 63, 86,  97,  94,  85];
            break;

         // Red Blue
         case 95:
            red = [ 94, 112, 141, 165, 167, 140,  91,  49,  27];
            green = [ 27,  46,  88, 135, 166, 161, 135,  97,  58];
            blue = [ 42,  52,  81, 106, 139, 158, 155, 137, 116];
            break;

         // Rose
         case 96:
            red = [ 30, 49, 79, 117, 135, 151, 146, 138, 147];
            green = [ 63, 60, 72,  90,  94,  94,  68,  46,  16];
            blue = [ 18, 28, 41,  56,  62,  63,  50,  36,  21];
            break;

         // Rust
         case 97:
            red = [  0, 30, 63, 101, 143, 152, 169, 187, 230];
            green = [  0, 14, 28,  42,  58,  61,  67,  74,  91];
            blue = [ 39, 26, 21,  18,  15,  14,  14,  13,  13];
            break;

         // Sandy Terrain
         case 98:
            red = [ 149, 140, 164, 179, 182, 181, 131, 87, 61];
            green = [  62,  70, 107, 136, 144, 138, 117, 87, 74];
            blue = [  40,  38,  45,  49,  49,  49,  38, 32, 34];
            break;

         // Sienna
         case 99:
            red = [ 99, 112, 148, 165, 179, 182, 183, 183, 208];
            green = [ 39,  40,  57,  79, 104, 127, 148, 161, 198];
            blue = [ 15,  16,  18,  33,  51,  79, 103, 129, 177];
            break;

         // Solar
         case 100:
            red = [ 99, 116, 154, 174, 200, 196, 201, 201, 230];
            green = [  0,   0,   8,  32,  58,  83, 119, 136, 173];
            blue = [  5,   6,   7,   9,   9,  14,  17,  19,  24];
            break;

         // South West
         case 101:
            red = [ 82, 106, 126, 141, 155, 163, 142, 107,  66];
            green = [ 62,  44,  69, 107, 135, 152, 149, 132, 119];
            blue = [ 39,  25,  31,  60,  73,  68,  49,  72, 188];
            break;

         // Starry Night
         case 102:
            red = [ 18, 29, 44,  72, 116, 158, 184, 208, 221];
            green = [ 27, 46, 71, 105, 146, 177, 189, 190, 183];
            blue = [ 39, 55, 80, 108, 130, 133, 124, 100,  76];
            break;

         // Sunset
         case 103:
            red = [ 0, 48, 119, 173, 212, 224, 228, 228, 245];
            green = [ 0, 13,  30,  47,  79, 127, 167, 205, 245];
            blue = [ 0, 68,  75,  43,  16,  22,  55, 128, 245];
            break;

         // Temperature Map
         case 104:
            red = [  34,  70, 129, 187, 225, 226, 216, 193, 179];
            green = [  48,  91, 147, 194, 226, 229, 196, 110,  12];
            blue = [ 234, 212, 216, 224, 206, 110,  53,  40,  29];
            break;

         // Thermometer
         case 105:
            red = [  30,  55, 103, 147, 174, 203, 188, 151, 105];
            green = [   0,  65, 138, 182, 187, 175, 121,  53,   9];
            blue = [ 191, 202, 212, 208, 171, 140,  97,  57,  30];
            break;

         // Valentine
         case 106:
            red = [ 112, 97, 113, 125, 138, 159, 178, 188, 225];
            green = [  16, 17,  24,  37,  56,  81, 110, 136, 189];
            blue = [  38, 35,  46,  59,  78, 103, 130, 152, 201];
            break;

         // Visible Spectrum
         case 107:
            red = [ 18,  72,   5,  23,  29, 201, 200, 98, 29];
            green = [  0,   0,  43, 167, 211, 117,   0,  0,  0];
            blue = [ 51, 203, 177,  26,  10,   9,   8,  3,  0];
            break;

         // Water Melon
         case 108:
            red = [ 19, 42, 64,  88, 118, 147, 175, 187, 205];
            green = [ 19, 55, 89, 125, 154, 169, 161, 129,  70];
            blue = [ 19, 32, 47,  70, 100, 128, 145, 130,  75];
            break;

         // Cool
         case 109:
            red = [  33,  31,  42,  68,  86, 111, 141, 172, 227];
            green = [ 255, 175, 145, 106,  88,  55,  15,   0,   0];
            blue = [ 255, 205, 202, 203, 208, 205, 203, 206, 231];
            break;

         // Copper
         case 110:
            red = [ 0, 25, 50, 79, 110, 145, 181, 201, 254];
            green = [ 0, 16, 30, 46,  63,  82, 101, 124, 179];
            blue = [ 0, 12, 21, 29,  39,  49,  61,  74, 103];
            break;

         // Gist Earth
         case 111:
            red = [ 0, 13,  30,  44,  72, 120, 156, 200, 247];
            green = [ 0, 36,  84, 117, 141, 153, 151, 158, 247];
            blue = [ 0, 94, 100,  82,  56,  66,  76, 131, 247];
            break;

         // Viridis
         case 112:
            red = [ 26, 51,  43,  33,  28,  35,  74, 144, 246];
            green = [  9, 24,  55,  87, 118, 150, 180, 200, 222];
            blue = [ 30, 96, 112, 114, 112, 101,  72,  35,   0];
            break;

         default:
            return JSROOT.Painter.CreateDefaultPalette();
      }

      return JSROOT.Painter.CreateGradientColorTable(stops, red, green, blue, 255, alfa);
   }

   // ===========================================================================

   JSROOT.Painter.drawPaletteAxis = function(divid,palette,opt) {

      // disable draw of shadow element of TPave
      palette.fBorderSize = 1;
      palette.fShadowColor = 0;

      var painter = new JSROOT.TPavePainter(palette);

      painter.SetDivId(divid);

      painter.z_handle = new JSROOT.TAxisPainter(palette.fAxis, true);
      painter.z_handle.SetDivId(divid, -1);

      painter.Cleanup = function() {
         if (this.z_handle) {
            this.z_handle.Cleanup();
            delete this.z_handle;
         }

         JSROOT.TObjectPainter.prototype.Cleanup.call(this);
      }

      painter.PaveDrawFunc = function(s_width, s_height, arg) {

         var pthis = this,
             palette = this.GetObject(),
             axis = palette.fAxis,
             can_move = (typeof arg == 'string') && (arg.indexOf('canmove')>0),
             postpone_draw = (typeof arg == 'string') && (arg.indexOf('postpone')>0),
             nbr1 = axis.fNdiv % 100,
             pos_x = parseInt(this.draw_g.attr("x")), // pave position
             pos_y = parseInt(this.draw_g.attr("y")),
             width = this.pad_width(),
             height = this.pad_height(),
             axisOffset = axis.fLabelOffset * width,
             main = this.main_painter(),
             zmin = 0, zmax = 100,
             contour = main.fContour;

         if (nbr1<=0) nbr1 = 8;
         axis.fTickSize = 0.6 * s_width / width; // adjust axis ticks size

         if (contour) {
            zmin = Math.min(contour[0], main.zmin);
            zmax = Math.max(contour[contour.length-1], main.zmax);
         } else
         if ((main.gmaxbin!==undefined) && (main.gminbin!==undefined)) {
            // this is case of TH2 (needs only for size adjustment)
            zmin = main.gminbin; zmax = main.gmaxbin;
         } else
         if ((main.hmin!==undefined) && (main.hmax!==undefined)) {
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
                   col = main.getValueColor((contour[i]+contour[i+1])/2);

               var r = this.draw_g.append("svg:rect")
                          .attr("x", 0)
                          .attr("y",  z1.toFixed(1))
                          .attr("width", s_width)
                          .attr("height", (z0-z1).toFixed(1))
                          .style("fill", col)
                          .style("stroke", col)
                          .property("fill0", col)
                          .property("fill1", d3.rgb(col).darker(0.5).toString())

               if (JSROOT.gStyle.Tooltip > 0)
                  r.on('mouseover', function() {
                     d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill1'));
                  }).on('mouseout', function() {
                     d3.select(this).transition().duration(100).style("fill", d3.select(this).property('fill0'));
                  }).append("svg:title").text(contour[i].toFixed(2) + " - " + contour[i+1].toFixed(2));

               if (JSROOT.gStyle.Zooming)
                  r.on("dblclick", function() { pthis.main_painter().Unzoom("z"); });
            }


         this.z_handle.SetAxisConfig("zaxis", z_kind, z, zmin, zmax, zmin, zmax);

         this.z_handle.DrawAxis(true, this.draw_g, s_width, s_height, "translate(" + s_width + ", 0)");

         if (can_move && ('getBoundingClientRect' in this.draw_g.node())) {
            var rect = this.draw_g.node().getBoundingClientRect();

            var shift = (pos_x + parseInt(rect.width)) - Math.round(0.995*width) + 3;

            if (shift>0) {
               this.draw_g.attr("x", pos_x - shift).attr("y", pos_y)
                          .attr("transform", "translate(" + (pos_x-shift) + ", " + pos_y + ")");
               palette.fX1NDC -= shift/width;
               palette.fX2NDC -= shift/width;
            }
         }

         if (!JSROOT.gStyle.Zooming) return;

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

            pthis.main_painter().Zoom("z", zmin, zmax);
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

         this.draw_g.select(".axis_zoom")
                    .on("mousedown", startRectSel)
                    .on("dblclick", function() { pthis.main_painter().Unzoom("z"); });
      }

      painter.ShowContextMenu = function(evnt) {
         this.main_painter().ShowContextMenu("z", evnt, this.GetObject().fAxis);
      }

      painter.UseContextMenu = true;

      painter.DrawPave(opt);

      return painter.DrawingReady();
   }

   // =======================================================================

   function TAxisPainter(axis, embedded) {
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
      this.invert_side = false;
   }

   TAxisPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TAxisPainter.prototype.Cleanup = function() {

      this.ticks = [];
      this.func = null;
      delete this.format;
      delete this.range;

      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   TAxisPainter.prototype.SetAxisConfig = function(name, kind, func, min, max, smin, smax) {
      this.name = name;
      this.kind = kind;
      this.func = func;

      this.full_min = min;
      this.full_max = max;
      this.scale_min = smin;
      this.scale_max = smax;
   }

   TAxisPainter.prototype.CreateFormatFuncs = function() {

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

         this.tfunc1 = this.tfunc2 = d3.timeFormat(tf1);
         if (tf2!==tf1)
            this.tfunc2 = d3.timeFormat(tf2);

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
                  val = (this.ndig>10) ? val.toExponential(4) : val.toFixed(this.ndig > 0 ? this.ndig : 0);
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

            return this.ndig>10 ? val.toExponential(4) : val.toFixed(this.ndig+2 > 0 ? this.ndig+2 : 0);
         }
      }
   }

   TAxisPainter.prototype.CreateTicks = function(only_major_as_array) {
      // function used to create array with minor/middle/major ticks

      var handle = { nminor: 0, nmiddle: 0, nmajor: 0, func: this.func };

      handle.minor = handle.middle = handle.major = this.func.ticks(this.nticks);

      if (only_major_as_array) {
         var res = handle.major;
         var delta = (this.scale_max - this.scale_min)*1e-5;
         if (res[0] > this.scale_min + delta) res.unshift(this.scale_min);
         if (res[res.length-1] < this.scale_max - delta) res.push(this.scale_max);
         return res;
      }

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

   TAxisPainter.prototype.IsCenterLabels = function() {
      if (this.kind === 'labels') return true;
      if (this.kind === 'log') return false;
      var axis = this.GetObject();
      return axis && axis.TestBit(JSROOT.EAxisBits.kCenterLabels);
   }

   TAxisPainter.prototype.AddTitleDrag = function(title_g, vertical, offset_k, reverse, axis_length) {
      if (!JSROOT.gStyle.MoveResize) return;

      var pthis = this,  drag_rect = null, prefix = "", drag_move,
          acc_x, acc_y, new_x, new_y, sign_0, center_0, alt_pos;
      if (JSROOT._test_d3_ === 3) {
         prefix = "drag";
         drag_move = d3.behavior.drag().origin(Object);
      } else {
         drag_move = d3.drag().subject(Object);
      }

      drag_move
         .on(prefix+"start",  function() {

            d3.event.sourceEvent.preventDefault();
            d3.event.sourceEvent.stopPropagation();

            var box = title_g.node().getBBox(), // check that elements visible, request precise value
                axis = pthis.GetObject();

            new_x = acc_x = title_g.property('shift_x');
            new_y = acc_y = title_g.property('shift_y');

            sign_0 = vertical ? (acc_x>0) : (acc_y>0); // sign should remain

            if (axis.TestBit(JSROOT.EAxisBits.kCenterTitle))
               alt_pos = (reverse === vertical) ? axis_length : 0;
            else
               alt_pos = Math.round(axis_length/2);

            drag_rect = title_g.append("rect")
                 .classed("zoom", true)
                 .attr("x", box.x)
                 .attr("y", box.y)
                 .attr("width", box.width)
                 .attr("height", box.height)
                 .style("cursor", "move");
//                 .style("pointer-events","none"); // let forward double click to underlying elements
          }).on("drag", function() {
               if (!drag_rect) return;

               d3.event.sourceEvent.preventDefault();
               d3.event.sourceEvent.stopPropagation();

               acc_x += d3.event.dx;
               acc_y += d3.event.dy;

               var set_x = title_g.property('shift_x'),
                   set_y = title_g.property('shift_y');

               if (vertical) {
                  set_x = acc_x;
                  if (Math.abs(acc_y - set_y) > Math.abs(acc_y - alt_pos)) set_y = alt_pos;
               } else {
                  set_y = acc_y;
                  if (Math.abs(acc_x - set_x) > Math.abs(acc_x - alt_pos)) set_x = alt_pos;
               }

               if (sign_0 === (vertical ? (set_x>0) : (set_y>0))) {
                  new_x = set_x; new_y = set_y;
                  title_g.attr('transform', 'translate(' + new_x + ',' + new_y +  ')');
               }

          }).on(prefix+"end", function() {
               if (!drag_rect) return;

               d3.event.sourceEvent.preventDefault();
               d3.event.sourceEvent.stopPropagation();

               title_g.property('shift_x', new_x)
                      .property('shift_y', new_y);

               var axis = pthis.GetObject();

               axis.fTitleOffset = (vertical ? new_x : new_y) / offset_k;
               if ((vertical ? new_y : new_x) === alt_pos) axis.InvertBit(JSROOT.EAxisBits.kCenterTitle);

               drag_rect.remove();
               drag_rect = null;
            });

      title_g.style("cursor", "move").call(drag_move);
   }

   TAxisPainter.prototype.DrawAxis = function(vertical, layer, w, h, transform, reverse, second_shift) {
      // function draw complete TAxis
      // later will be used to draw TGaxis

      var axis = this.GetObject(),
          is_gaxis = (axis && axis._typename === 'TGaxis'),
          side = (this.name === "zaxis") ? -1  : 1, both_sides = 0,
          axis_g = layer, tickSize = 10, scaling_size = 100, text_scaling_size = 100,
          pad_w = this.pad_width() || 10,
          pad_h = this.pad_height() || 10;

      this.vertical = vertical;

      // shift for second ticks set (if any)
      if (!second_shift) second_shift = 0; else
      if (this.invert_side) second_shift = -second_shift;

      if (is_gaxis) {
         if (!this.lineatt) this.lineatt = JSROOT.Painter.createAttLine(axis);
         scaling_size = (vertical ? pad_w : pad_h);
         tickSize = Math.round(axis.fTickSize * scaling_size);
      } else {
         if (!this.lineatt) this.lineatt = JSROOT.Painter.createAttLine(axis.fAxisColor, 1);
         scaling_size = (vertical ? w : h);
         tickSize = Math.round(axis.fTickLength * scaling_size);
      }

      text_scaling_size = Math.min(pad_w, pad_h);

      if (!is_gaxis || (this.name === "zaxis")) {
         axis_g = layer.select("." + this.name + "_container");
         if (axis_g.empty())
            axis_g = layer.append("svg:g").attr("class",this.name+"_container");
         else
            axis_g.selectAll("*").remove();
         if (this.invert_side) side = -side;
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
          res = "", res2 = "", lastpos = 0, lasth = 0,
          textscale = 1, maxtextlen = 0;

      // first draw ticks

      this.ticks = [];

      var handle = this.CreateTicks();

      while (handle.next(true)) {
         var h1 = Math.round(tickSize/4), h2 = 0;

         if (handle.kind < 3)
            h1 = Math.round(tickSize/2);

         if (handle.kind == 1) {
            // if not showing labels, not show large tick
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
      if ((this.kind=="normal") && /*vertical && */ !axis.TestBit(JSROOT.EAxisBits.kNoExponent)) {
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
         var pos = Math.round(this.func(handle.major[nmajor])),
             lbl = this.format(handle.major[nmajor], true);
         if (lbl === null) continue;

         var t = label_g.append("svg:text").attr("fill", label_color).text(lbl);

         maxtextlen = Math.max(maxtextlen, lbl.length);

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

         var tsize = !JSROOT.nodejs ? this.GetBoundarySizes(t.node()) :
                      { height: Math.round(labelfont.size*1.2), width: Math.round(lbl.length*labelfont.size*0.4) },
             space_before = (nmajor > 0) ? (pos - last) : (vertical ? h/2 : w/2),
             space_after = (nmajor < handle.major.length-1) ? (Math.round(this.func(handle.major[nmajor+1])) - pos) : space_before,
             space = Math.min(Math.abs(space_before), Math.abs(space_after));

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

     if (this.order!==0)
        label_g.append("svg:text")
               .attr("fill", label_color)
               .attr("x", vertical ? labeloffset : w+5)
               .attr("y", 0)
               .style("text-anchor", "start")
               .style("dominant-baseline", "middle")
               .attr("dy", "-.5em")
               .text('\xD7' + JSROOT.Painter.formatExp(Math.pow(10,this.order).toExponential(0)));

     if ((textscale>0) && (textscale<1.)) {
        // rotate X labels if they are too big
        if ((textscale < 0.7) && !vertical && (side>0) && (maxtextlen > 5)) {
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

     if (JSROOT.gStyle.Zooming && !this.disable_zooming) {
        var r =  axis_g.append("svg:rect")
                       .attr("class", "axis_zoom")
                       .style("opacity", "0")
                       .style("cursor", "crosshair");

        if (vertical)
           r.attr("x", (side>0) ? (-2*labelfont.size - 3) : 3)
            .attr("y", 0)
            .attr("width", 2*labelfont.size + 3)
            .attr("height", h)
        else
           r.attr("x", 0).attr("y", (side>0) ? 0 : -labelfont.size-3)
            .attr("width", w).attr("height", labelfont.size + 3);
      }

      if (axis.fTitle.length > 0) {
         var title_g = axis_g.append("svg:g").attr("class", "axis_title"),
             title_fontsize = (axis.fTitleSize >= 1) ? axis.fTitleSize : Math.round(axis.fTitleSize * text_scaling_size),
             title_offest_k = 1.6*(axis.fTitleSize<1 ? axis.fTitleSize : axis.fTitleSize/text_scaling_size),
             center = axis.TestBit(JSROOT.EAxisBits.kCenterTitle),
             rotate = axis.TestBit(JSROOT.EAxisBits.kRotateTitle) ? -1 : 1,
             title_color = JSROOT.Painter.root_colors[axis.fTitleColor],
             shift_x = 0, shift_y = 0;

         this.StartTextDrawing(axis.fTitleFont, title_fontsize, title_g);

         var myxor = ((rotate<0) && !reverse) || ((rotate>=0) && reverse);

         if (vertical) {
            title_offest_k *= -side*pad_w;

            shift_x = Math.round(title_offest_k*axis.fTitleOffset);

            if ((this.name == "zaxis") && is_gaxis && ('getBoundingClientRect' in axis_g.node())) {
               // special handling for color palette labels - draw them always on right side
               var rect = axis_g.node().getBoundingClientRect();
               if (shift_x < rect.width - tickSize) shift_x = Math.round(rect.width - tickSize);
            }

            shift_y = Math.round(center ? h/2 : (reverse ? h : 0));

            this.DrawText((center ? "middle" : (myxor ? "begin" : "end" ))+ ";middle",
                           0, 0, 0, (rotate<0 ? -90 : -270),
                           axis.fTitle, title_color, 1, title_g);
         } else {
            title_offest_k *= side*pad_h;

            shift_x = Math.round(center ? w/2 : (reverse ? 0 : w));
            shift_y = Math.round(title_offest_k*axis.fTitleOffset);
            this.DrawText((center ? 'middle' : (myxor ? 'begin' : 'end')) + ";middle",
                          0, 0, 0, (rotate<0 ? -180 : 0),
                          axis.fTitle, title_color, 1, title_g);
         }

         this.FinishTextDrawing(title_g);

         title_g.attr('transform', 'translate(' + shift_x + ',' + shift_y +  ')')
                .property('shift_x',shift_x)
                .property('shift_y',shift_y);

         this.AddTitleDrag(title_g, vertical, title_offest_k, reverse, vertical ? h : w);
      }

      this.position = 0;

      if ('getBoundingClientRect' in axis_g.node()) {
         var rect1 = axis_g.node().getBoundingClientRect(),
             rect2 = this.svg_pad().node().getBoundingClientRect();

         this.position = rect1.left - rect2.left; // use to control left position of Y scale
      }
   }

   TAxisPainter.prototype.Redraw = function() {

      var gaxis = this.GetObject(),
          x1 = this.AxisToSvg("x", gaxis.fX1),
          y1 = this.AxisToSvg("y", gaxis.fY1),
          x2 = this.AxisToSvg("x", gaxis.fX2),
          y2 = this.AxisToSvg("y", gaxis.fY2),
          w = x2 - x1, h = y1 - y2,
          vertical = w < 5, kind = "normal", func = null,
          min = gaxis.fWmin, max = gaxis.fWmax, reverse = false;

      if (gaxis.fChopt.indexOf("G")>=0) {
         func = d3.scaleLog();
         kind = "log";
      } else {
         func = d3.scaleLinear();
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

   JSROOT.Painter.drawGaxis = function(divid, obj, opt) {
      var painter = new JSROOT.TAxisPainter(obj, false);

      painter.SetDivId(divid);

      painter.disable_zooming = true;

      painter.Redraw();

      return painter.DrawingReady();
   }

   // ============================================================

   // base class for all objects, derived from TPave
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
         // function used to signal drawing ready, required when text drawing posponed due to mathjax
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

      this.draw_g.attr("transform", "translate(" + pos_x + "," + pos_y + ")");

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

      var rect =
         this.draw_g.append("svg:path")
          .attr("d", "M0,0h"+width+"v"+height+"h-"+width+"z")
          .call(this.fillatt.func)
          .call(this.lineatt.func);

      if ('PaveDrawFunc' in this)
         this.PaveDrawFunc(width, height, arg);

      if (JSROOT.BatchMode) return;

      // here all kind of interactive settings

      rect.style("pointer-events", "visibleFill")
          .on("mouseenter", this.ShowObjectStatus.bind(this))

      // position and size required only for drag functions
      this.draw_g.attr("x", pos_x)
                 .attr("y", pos_y)
                 .attr("width", width)
                 .attr("height", height);

      this.AddDrag({ obj: pt, minwidth: 10, minheight: 20,
                     redraw: this.DrawPave.bind(this),
                     ctxmenu: JSROOT.touches && JSROOT.gStyle.ContextMenu && this.UseContextMenu });

      if (this.UseContextMenu && JSROOT.gStyle.ContextMenu)
         this.draw_g.on("contextmenu", this.ShowContextMenu.bind(this));
   }

   TPavePainter.prototype.DrawPaveLabel = function(width, height) {
      this.UseTextColor = true;

      var pave = this.GetObject();

      this.StartTextDrawing(pave.fTextFont, height/1.2);

      this.DrawText(pave.fTextAlign, 0, 0, width, height, pave.fLabel, JSROOT.Painter.root_colors[pave.fTextColor]);

      this.FinishTextDrawing(null, this.FinishPave);
   }

   TPavePainter.prototype.DrawPaveText = function(width, height, refill) {

      if (refill && this.IsStats()) this.FillStatistic();

      var pt = this.GetObject(),
          tcolor = JSROOT.Painter.root_colors[pt.fTextColor],
          lwidth = pt.fBorderSize,
          first_stat = 0,
          num_cols = 0,
          nlines = pt.fLines.arr.length,
          lines = [],
          maxlen = 0,
          draw_header = (pt.fLabel.length>0) && !this.IsStats();

      if (draw_header) this.FirstRun++; // increment finish counter

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

      if ((nlines===1) && !this.IsStats() &&
          (lines[0].indexOf("#splitline{")===0) && (lines[0][lines[0].length-1]=="}")) {
            var pos = lines[0].indexOf("}{");
            if ((pos>0) && (pos == lines[0].lastIndexOf("}{"))) {
               lines[1] = lines[0].substr(pos+2, lines[0].length - pos - 3);
               lines[0] = lines[0].substr(11, pos - 11);
               nlines = 2;
               this.UseTextColor = true;
            }
         }

      // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
      var stepy = height / nlines, has_head = false, margin_x = pt.fMargin * width;

      this.StartTextDrawing(pt.fTextFont, height/(nlines * 1.2));

      if (nlines == 1) {
         this.DrawText(pt.fTextAlign, 0, 0, width, height, lines[0], tcolor);
         this.UseTextColor = true;
      } else {
         for (var j = 0; j < nlines; ++j) {
            var posy = j*stepy, jcolor = tcolor;
            if (!this.UseTextColor && (j<pt.fLines.arr.length) && (pt.fLines.arr[j].fTextColor!==0))
               jcolor = JSROOT.Painter.root_colors[pt.fLines.arr[j].fTextColor];
            if (jcolor===undefined) {
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

      this.FinishTextDrawing(undefined, this.FinishPave);

      var lpath = "";

      if ((lwidth > 0) && has_head)
         lpath += "M0," + Math.round(stepy) + "h" + width;

      if ((first_stat > 0) && (num_cols > 1)) {
         for (var nrow = first_stat; nrow < nlines; ++nrow)
            lpath += "M0," + Math.round(nrow * stepy) + "h" + width;
         for (var ncol = 0; ncol < num_cols - 1; ++ncol)
            lpath += "M" + Math.round(width / num_cols * (ncol + 1)) + "," + Math.round(first_stat * stepy) + "V" + height;
      }

      if (lpath) this.draw_g.append("svg:path").attr("d",lpath).call(this.lineatt.func);

      if (draw_header) {
         var x = Math.round(width*0.25),
             y = Math.round(-height*0.02),
             w = Math.round(width*0.5),
             h = Math.round(height*0.04);

         var lbl_g = this.draw_g.append("svg:g");

         lbl_g.append("svg:path")
               .attr("d", "M" + x + "," + y + "h" + w + "v" + h + "h-" + w + "z")
               .call(this.fillatt.func)
               .call(this.lineatt.func);

         this.StartTextDrawing(pt.fTextFont, h/1.5, lbl_g);

         this.DrawText(22, x, y, w, h, pt.fLabel, tcolor, 1, lbl_g);

         this.FinishTextDrawing(lbl_g, this.FinishPave);

         this.UseTextColor = true;
      }
   }

   TPavePainter.prototype.Format = function(value, fmt) {
      // method used to convert value to string according specified format
      // format can be like 5.4g or 4.2e or 6.4f
      if (!fmt) fmt = "stat";

      var pave = this.GetObject();

      if (fmt=="stat") {
         fmt = pave.fStatFormat;
         if (!fmt) fmt = JSROOT.gStyle.fStatFormat;
      } else
      if (fmt=="fit") {
         fmt = pave.fFitFormat;
         if (!fmt) fmt = JSROOT.gStyle.fFitFormat;
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
            this.Redraw();
         });

         menu.add("SetStatFormat", function() {
            var fmt = prompt("Enter StatFormat", pave.fStatFormat);
            if (fmt!=null) {
               pave.fStatFormat = fmt;
               this.Redraw();
            }
         });
         menu.add("SetFitFormat", function() {
            var fmt = prompt("Enter FitFormat", pave.fFitFormat);
            if (fmt!=null) {
               pave.fFitFormat = fmt;
               this.Redraw();
            }
         });
         menu.add("separator");
         menu.add("sub:SetOptStat", function() {
            // todo - use jqury dialog here
            var fmt = prompt("Enter OptStat", pave.fOptStat);
            if (fmt!=null) { pave.fOptStat = parseInt(fmt); this.Redraw(); }
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
               this.Redraw();
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
            if (fmt!=null) { pave.fOptFit = parseInt(fmt); this.Redraw(); }
         });
         AddStatOpt(10, "Fit parameters");
         AddStatOpt(11, "Par errors");
         AddStatOpt(12, "Chi square / NDF");
         AddStatOpt(13, "Probability");
         menu.add("endsub:");

         menu.add("separator");
      } else
      if (pave.fName === "title")
         menu.add("Default position", function() {
            pave.fX1NDC = 0.28;
            pave.fY1NDC = 0.94;
            pave.fX2NDC = 0.72;
            pave.fY2NDC = 0.99;
            pave.fInit = 1;
            this.Redraw();
         });

      if (this.UseTextColor)
         this.TextAttContextMenu(menu);

      this.FillAttContextMenu(menu);

      return menu.size() > 0;
   }

   TPavePainter.prototype.ShowContextMenu = function(evnt) {
      if (!evnt) {
         d3.event.stopPropagation(); // disable main context menu
         d3.event.preventDefault();  // disable browser context menu

         // one need to copy event, while after call back event may be changed
         evnt = d3.event;
      }

      JSROOT.Painter.createMenu(this, function(menu) {
         menu.painter.FillContextMenu(menu);
         menu.show(evnt);
      }); // end menu creation
   }

   TPavePainter.prototype.IsStats = function() {
      return this.MatchObjectType('TPaveStats');
   }

   TPavePainter.prototype.FillStatistic = function() {
      var pave = this.GetObject(), main = this.main_painter();

      if (pave.fName !== "stats") return false;
      if ((main===null) || !('FillStatistic' in main)) return false;

      // no need to refill statistic if histogram is dummy
      if (main.IsDummyHisto()) return true;

      var dostat = parseInt(pave.fOptStat), dofit = parseInt(pave.fOptFit);
      if (isNaN(dostat)) dostat = JSROOT.gStyle.fOptStat;
      if (isNaN(dofit)) dofit = JSROOT.gStyle.fOptFit;

      // make empty at the beginning
      pave.Clear();

      // we take statistic from first painter
      main.FillStatistic(this, dostat, dofit);

      return true;
   }

   TPavePainter.prototype.UpdateObject = function(obj) {
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
      } else
      if (obj._typename === 'TLegend') {
         pave.fPrimitives = obj.fPrimitives;
         pave.fNColumns = obj.fNColumns;
         return true;
      }

      return false;
   }

   TPavePainter.prototype.Redraw = function() {
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

      // drawing ready handled in special painters, if not exists - drawing is done
      return this.PaveDrawFunc ? painter.DrawingReady() : painter;
   }

   // ==============================================================================

   JSROOT.Painter.drawLegend = function(divid, obj, opt) {

      var painter = new JSROOT.TPavePainter(obj);

      painter.SetDivId(divid, 2);

      painter.AssignFinishPave(); // do it once again while this is changed

      painter.PaveDrawFunc = function(w, h) {

         var legend = this.GetObject(),
             nlines = legend.fPrimitives.arr.length,
             ncols = legend.fNColumns,
             nrows = nlines;

         if (ncols<2) ncols = 1; else { while ((nrows-1)*ncols >= nlines) nrows--; }

         this.StartTextDrawing(legend.fTextFont, h / (nlines * 1.2));

         var tcolor = JSROOT.Painter.root_colors[legend.fTextColor],
             column_width = Math.round(w/ncols),
             padding_x = Math.round(0.03*w/ncols),
             padding_y = Math.round(0.03*h),
             step_y = (h - 2*padding_y)/nrows,
             any_opt = false;

         for (var i = 0; i < nlines; ++i) {
            var leg = legend.fPrimitives.arr[i],
                lopt = leg.fOption.toLowerCase(),
                icol = i % ncols, irow = (i - icol) / ncols,
                x0 = icol * column_width,
                tpos_x = x0 + Math.round(legend.fMargin*column_width),
                pos_y = Math.round(padding_y + irow*step_y), // top corner
                mid_y = Math.round(padding_y + (irow+0.5)*step_y), // center line
                o_fill = leg, o_marker = leg, o_line = leg,
                mo = leg.fObject,
                painter = null,
                isany = false;

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
               this.draw_g.append("svg:rect")
                      .attr("x", x0 + padding_x)
                      .attr("y", Math.round(pos_y+step_y*0.1))
                      .attr("width", tpos_x - 2*padding_x - x0)
                      .attr("height", Math.round(step_y*0.8))
                      .call(fillatt.func);
               if (fillatt.color !== 'none') isany = true;
            }

            // Draw line
            if (lopt.indexOf('l') != -1) {
               var lineatt = (painter && painter.lineatt) ? painter.lineatt : JSROOT.Painter.createAttLine(o_line)
               this.draw_g.append("svg:line")
                  .attr("x1", x0 + padding_x)
                  .attr("y1", mid_y)
                  .attr("x2", tpos_x - padding_x)
                  .attr("y2", mid_y)
                  .call(lineatt.func);
               if (lineatt.color !== 'none') isany = true;
            }

            // Draw error
            if (lopt.indexOf('e') != -1  && (lopt.indexOf('l') == -1 || lopt.indexOf('f') != -1)) {
            }

            // Draw Polymarker
            if (lopt.indexOf('p') != -1) {
               var marker = (painter && painter.markeratt) ? painter.markeratt : JSROOT.Painter.createAttMarker(o_marker);
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

            this.DrawText("start", pos_x, pos_y, x0+column_width-pos_x-padding_x, step_y, leg.fLabel, tcolor);
         }

         // rescale after all entries are shown
         this.FinishTextDrawing(null, this.FinishPave);
      }

      painter.Redraw();

      return painter;
   }

   // =============================================================

   function THistPainter(histo) {
      JSROOT.TObjectPainter.call(this, histo);
      this.histo = histo;
      this.shrink_frame_left = 0.;
      this.draw_content = true;
      this.nbinsx = 0;
      this.nbinsy = 0;
      this.x_kind = 'normal'; // 'normal', 'time', 'labels'
      this.y_kind = 'normal'; // 'normal', 'time', 'labels'
      this.keys_handler = null;
      this.accept_drops = true; // indicate that one can drop other objects like doing Draw("same")
      this.mode3d = false;
      this.zoom_changed_interactive = 0;
   }

   THistPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   THistPainter.prototype.IsDummyHisto = function() {
      return !this.histo || (!this.draw_content && !this.create_stats) || (this.options.Axis>0);
   }

   THistPainter.prototype.IsTProfile = function() {
      return this.MatchObjectType('TProfile');
   }

   THistPainter.prototype.IsTH2Poly = function() {
      return this.histo && this.histo._typename.match(/^TH2Poly/);
   }

   THistPainter.prototype.Cleanup = function() {
      if (this.keys_handler) {
         window.removeEventListener( 'keydown', this.keys_handler, false );
         this.keys_handler = null;
      }

      // clear all 3D buffers
      if (typeof this.Create3DScene === 'function')
         this.Create3DScene(-1);

      this.histo = null; // cleanup histogram reference
      delete this.x; delete this.grx;
      delete this.ConvertX; delete this.RevertX;
      delete this.y; delete this.gry;
      delete this.ConvertY; delete this.RevertY;
      delete this.z; delete this.grz;

      if (this.x_handle) {
         this.x_handle.Cleanup();
         delete this.x_handle;
      }

      if (this.y_handle) {
         this.y_handle.Cleanup();
         delete this.y_handle;
      }

      if (this.z_handle) {
         this.z_handle.Cleanup();
         delete this.z_handle;
      }

      delete this.fPalette;
      delete this.fContour;
      delete this.options;

      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   THistPainter.prototype.Dimension = function() {
      if (!this.histo) return 0;
      if (this.histo._typename.indexOf("TH2")==0) return 2;
      if (this.histo._typename.indexOf("TProfile2D")==0) return 2;
      if (this.histo._typename.indexOf("TH3")==0) return 3;
      return 1;
   }

   THistPainter.prototype.DecodeOptions = function(opt, interactive) {

      /* decode string 'opt' and fill the option structure */
      var option = { Axis: 0, Bar: 0, Curve: 0, Hist: 0, Line: 0,
             Error: 0, errorX: JSROOT.gStyle.fErrorX,
             Mark: 0, Fill: 0, Same: 0, Scat: 0, ScatCoef: 1., Func: 1, Star: 0,
             Arrow: 0, Box: 0, Text: 0, Char: 0, Color: 0, Contour: 0,
             Lego: 0, Surf: 0, Off: 0, Tri: 0, Proj: 0, AxisPos: 0,
             Spec: 0, Pie: 0, List: 0, Zscale: 0, FrontBox: 1, BackBox: 1, Candle: "",
             GLBox: 0, GLColor: 0,
             System: JSROOT.Painter.Coord.kCARTESIAN,
             AutoColor: 0, NoStat: 0, AutoZoom: false,
             HighRes: 0, Zero: 1, Palette: 0, BaseLine: false,
             Optimize: JSROOT.gStyle.OptimizeDraw,
             minimum: -1111, maximum: -1111 },
           d = new JSROOT.DrawOptions(opt ? opt : this.histo.fOption),
           hdim = this.Dimension(),
           pad = this.root_pad(),
           need_fillcol = false;

      // use error plot only when any sumw2 bigger than 0
      if ((hdim===1) && (this.histo.fSumw2.length > 0))
         for (var n=0;n<this.histo.fSumw2.length;++n)
            if (this.histo.fSumw2[n] > 0) { option.Error = 2; option.Zero = 0; break; }

      if (d.check('PAL', true)) option.Palette = d.partAsInt();
      if (d.check('MINIMUM:', true)) option.minimum = parseFloat(d.part); else option.minimum = this.histo.fMinimum;
      if (d.check('MAXIMUM:', true)) option.maximum = parseFloat(d.part); else option.maximum = this.histo.fMaximum;

      if (d.check('NOOPTIMIZE')) option.Optimize = 0;
      if (d.check('OPTIMIZE')) option.Optimize = 2;

      if (d.check('AUTOCOL')) { option.AutoColor = 1; option.Hist = 1; }
      if (d.check('AUTOZOOM')) { option.AutoZoom = 1; option.Hist = 1; }

      if (d.check('NOSTAT')) option.NoStat = 1;

      var tooltip = null;
      if (d.check('NOTOOLTIP')) tooltip = false;
      if (d.check('TOOLTIP')) tooltip = true;
      if ((tooltip!==null) && this.frame_painter()) this.frame_painter().tooltip_allowed = tooltip;

      if (d.check('LOGX')) pad.fLogx = 1;
      if (d.check('LOGY')) pad.fLogy = 1;
      if (d.check('LOGZ')) pad.fLogz = 1;
      if (d.check('GRIDXY')) pad.fGridx = pad.fGridy = 1;
      if (d.check('GRIDX')) pad.fGridx = 1;
      if (d.check('GRIDY')) pad.fGridy = 1;
      if (d.check('TICKXY')) pad.fTickx = pad.fTicky = 1;
      if (d.check('TICKX')) pad.fTickx = 1;
      if (d.check('TICKY')) pad.fTicky = 1;

      if (d.check('FILL_', true)) {
         if (d.partAsInt(1)>0) this.histo.fFillColor = d.partAsInt(); else
         for (var col=0;col<8;++col)
            if (JSROOT.Painter.root_colors[col].toUpperCase() === d.part) this.histo.fFillColor = col;
      }
      if (d.check('LINE_', true)) {
         if (d.partAsInt(1)>0) this.histo.fLineColor = d.partAsInt(); else
         for (var col=0;col<8;++col)
            if (JSROOT.Painter.root_colors[col].toUpperCase() === d.part) this.histo.fLineColor = col;
      }

      if (d.check('X+')) option.AxisPos = 10;
      if (d.check('Y+')) option.AxisPos += 1;

      if (d.check('SAMES')) option.Same = 2;
      if (d.check('SAME')) option.Same = 1;

      // if here rest option is empty, draw histograms by default
      if (d.empty()) option.Hist = 1;

      if (d.check('SPEC')) { option.Scat = 0; option.Spec = 1; }

      if (d.check('BASE0')) option.BaseLine = 0; else
      if (JSROOT.gStyle.fHistMinimumZero) option.BaseLine = 0;

      if (d.check('PIE')) option.Pie = 1;

      if (d.check('CANDLE', true)) option.Candle = d.part;

      if (d.check('GLBOX',true)) option.GLBox = 10 + d.partAsInt();
      if (d.check('GLCOL')) option.GLColor = 1;

      d.check('GL'); // suppress GL

      if (d.check('LEGO', true)) {
         option.Scat = 0;
         option.Lego = 1;
         if (d.part.indexOf('0') >= 0) option.Zero = 0;
         if (d.part.indexOf('1') >= 0) option.Lego = 11;
         if (d.part.indexOf('2') >= 0) option.Lego = 12;
         if (d.part.indexOf('3') >= 0) option.Lego = 13;
         if (d.part.indexOf('4') >= 0) option.Lego = 14;
         if (d.part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (d.part.indexOf('BB') >= 0) option.BackBox = 0;
         if (d.part.indexOf('Z') >= 0) option.Zscale = 1;
      }

      if (d.check('SURF', true)) {
         option.Scat = 0;
         option.Surf = d.partAsInt(10, 1);
         if (d.part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (d.part.indexOf('BB') >= 0) option.BackBox = 0;
         if (d.part.indexOf('Z')>=0) option.Zscale = 1;
      }

      if (d.check('TF3', true)) {
         if (d.part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (d.part.indexOf('BB') >= 0) option.BackBox = 0;
      }

      if (d.check('ISO', true)) {
         if (d.part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (d.part.indexOf('BB') >= 0) option.BackBox = 0;
      }

      if (d.check('LIST')) option.List = 1;

      if (d.check('CONT', true)) {
         if (hdim > 1) {
            option.Scat = 0;
            option.Contour = 1;
            if (d.part.indexOf('Z') >= 0) option.Zscale = 1;
            if (d.part.indexOf('1') >= 0) option.Contour = 11; else
            if (d.part.indexOf('2') >= 0) option.Contour = 12; else
            if (d.part.indexOf('3') >= 0) option.Contour = 13; else
            if (d.part.indexOf('4') >= 0) option.Contour = 14;
         } else {
            option.Hist = 1;
         }
      }

      // decode bar/hbar option
      if (d.check('HBAR', true)) option.Bar = 20; else
      if (d.check('BAR', true)) option.Bar = 10;
      if (option.Bar > 0) {
         option.Hist = 0; need_fillcol = true;
         option.Bar += d.partAsInt();
      }

      if (d.check('ARR')) {
         if (hdim > 1) {
            option.Arrow = 1;
            option.Scat = 0;
         } else {
            option.Hist = 1;
         }
      }

      if (d.check('BOX',true)) option.Box = 10 + d.partAsInt();

      if (option.Box)
         if (hdim > 1) option.Scat = 0;
                  else option.Hist = 1;

      if (d.check('COL', true)) {
         option.Color = 1;

         if (d.part.indexOf('0')>=0) option.Color = 11;
         if (d.part.indexOf('1')>=0) option.Color = 11;
         if (d.part.indexOf('2')>=0) option.Color = 12;
         if (d.part.indexOf('3')>=0) option.Color = 13;

         if (d.part.indexOf('Z')>=0) option.Zscale = 1;
         if (hdim == 1) option.Hist = 1;
                   else option.Scat = 0;
      }

      if (d.check('CHAR')) { option.Char = 1; option.Scat = 0; }
      if (d.check('FUNC')) { option.Func = 2; option.Hist = 0; }
      if (d.check('AXIS')) option.Axis = 1;
      if (d.check('AXIG')) option.Axis = 2;

      if (d.check('TEXT', true)) {
         option.Text = 1;
         option.Scat = 0;
         option.Hist = 0;

         var angle = Math.min(d.partAsInt(), 90);
         if (angle) option.Text = 1000 + angle;

         if (d.part.indexOf('N')>=0 && this.IsTH2Poly())
            option.Text = 3000 + angle;

         if (d.part.indexOf('E')>=0)
            option.Text = 2000 + angle;
      }

      if (d.check('SCAT=', true)) {
         option.Scat = 1;
         option.ScatCoef = parseFloat(d.part);
         if (isNaN(option.ScatCoef) || (option.ScatCoef<=0)) option.ScatCoef = 1.;
      }

      if (d.check('SCAT')) option.Scat = 1;
      if (d.check('POL')) option.System = JSROOT.Painter.Coord.kPOLAR;
      if (d.check('CYL')) option.System = JSROOT.Painter.Coord.kCYLINDRICAL;
      if (d.check('SPH')) option.System = JSROOT.Painter.Coord.kSPHERICAL;
      if (d.check('PSR')) option.System = JSROOT.Painter.Coord.kRAPIDITY;

      if (d.check('TRI', true)) {
         option.Scat = 0;
         option.Color = 0;
         option.Tri = 1;
         if (d.part.indexOf('FB') >= 0) option.FrontBox = 0;
         if (d.part.indexOf('BB') >= 0) option.BackBox = 0;
         if (d.part.indexOf('ERR') >= 0) option.Error = 1;
      }
      if (d.check('AITOFF')) option.Proj = 1;
      if (d.check('MERCATOR')) option.Proj = 2;
      if (d.check('SINUSOIDAL')) option.Proj = 3;
      if (d.check('PARABOLIC')) option.Proj = 4;

      if (option.Proj > 0) { option.Scat = 0; option.Contour = 14; }

      if ((hdim==3) && d.check('FB')) option.FrontBox = 0;
      if ((hdim==3) && d.check('BB')) option.BackBox = 0;

      if (d.check('LF2')) { option.Line = 2; option.Hist = -1; option.Error = 0; need_fillcol = true; }
      if (d.check('L')) { option.Line = 1; option.Hist = -1; option.Error = 0; }

      if (d.check('A')) option.Axis = -1;
      if (d.check('B1')) { option.Bar = 1; option.BaseLine = 0; option.Hist = -1; need_fillcol = true; }
      if (d.check('B')) { option.Bar = 1; option.Hist = -1; need_fillcol = true; }
      if (d.check('C')) { option.Curve = 1; option.Hist = -1; }
      if (d.check('][')) { option.Off = 1; option.Hist = 1; }
      if (d.check('F')) option.Fill = 1;

      if (d.check('P0')) { option.Mark = 1; option.Hist = -1; option.Zero = 1; }
      if (d.check('P')) { option.Mark = 1; option.Hist = -1; option.Zero = 0; }
      if (d.check('Z')) option.Zscale = 1;
      if (d.check('*H') || d.check('*')) { option.Mark = 23; option.Hist = -1; }

      if (d.check('HIST')) { option.Hist = 2; option.Func = 0; option.Error = 0; }

      if (this.IsTH2Poly()) {
         if (option.Fill + option.Line + option.Mark != 0) option.Scat = 0;
      }

      if (d.check('E', true)) {
         if (hdim == 1) {
            option.Error = 1;
            option.Zero = 0; // do not draw empty bins with erros
            if (!isNaN(parseInt(d.part[0]))) option.Error = 10 + parseInt(d.part[0]);
            if ((option.Error === 13) || (option.Error === 14)) need_fillcol = true;
            if (option.Error === 10) option.Zero = 1; // enable drawing of empty bins
            if (d.part.indexOf('X0')>=0) option.errorX = 0;
         } else {
            if (option.Error == 0) {
               option.Error = 100;
               option.Scat = 0;
            }
         }
      }
      if (d.check('9')) option.HighRes = 1;
      if (d.check('0')) option.Zero = 0;

      if (interactive) {
         if (need_fillcol && this.fillatt && (this.fillatt.color=='none'))
            this.fillatt.Change(5,1001);
      }

      //if (option.Surf == 15)
      //   if (option.System == JSROOT.Painter.Coord.kPOLAR || option.System == JSROOT.Painter.Coord.kCARTESIAN)
      //      option.Surf = 13;

      return option;
   }

   THistPainter.prototype.GetAutoColor = function(col) {
      if (this.options.AutoColor<=0) return col;

      var id = this.options.AutoColor;
      this.options.AutoColor = id % 8 + 1;
      return JSROOT.Painter.root_colors[id];
   }

   THistPainter.prototype.ScanContent = function(when_axis_changed) {
      // function will be called once new histogram or
      // new histogram content is assigned
      // one should find min,max,nbins, maxcontent values
      // if when_axis_changed === true specified, content will be scanned after axis zoom changed

      alert("HistPainter.prototype.ScanContent not implemented");
   }

   THistPainter.prototype.CheckPadRange = function() {

      if (!this.is_main_painter()) return;

      this.zoom_xmin = this.zoom_xmax = 0;
      this.zoom_ymin = this.zoom_ymax = 0;
      this.zoom_zmin = this.zoom_zmax = 0;


      var ndim = this.Dimension(),
          xaxis = this.histo.fXaxis,
          yaxis = this.histo.fYaxis,
          zaxis = this.histo.fXaxis;

      // apply selected user range from histogram itself
      if (xaxis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
         //xaxis.InvertBit(JSROOT.EAxisBits.kAxisRange); // axis range is not used for main painter
         if ((xaxis.fFirst !== xaxis.fLast) && ((xaxis.fFirst > 1) || (xaxis.fLast < xaxis.fNbins))) {
            this.zoom_xmin = xaxis.fFirst > 1 ? xaxis.GetBinLowEdge(xaxis.fFirst) : xaxis.fXmin;
            this.zoom_xmax = xaxis.fLast < xaxis.fNbins ? xaxis.GetBinLowEdge(xaxis.fLast+1) : xaxis.fXmax;
         }
      }

      if ((ndim>1) && yaxis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
         //yaxis.InvertBit(JSROOT.EAxisBits.kAxisRange); // axis range is not used for main painter
         if ((yaxis.fFirst !== yaxis.fLast) && ((yaxis.fFirst > 1) || (yaxis.fLast < yaxis.fNbins))) {
            this.zoom_ymin = yaxis.fFirst > 1 ? yaxis.GetBinLowEdge(yaxis.fFirst) : yaxis.fXmin;
            this.zoom_ymax = yaxis.fLast < yaxis.fNbins ? yaxis.GetBinLowEdge(yaxis.fLast+1) : yaxis.fXmax;
         }
      }

      if ((ndim>2) && zaxis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
         //zaxis.InvertBit(JSROOT.EAxisBits.kAxisRange); // axis range is not used for main painter
         if ((zaxis.fFirst !== zaxis.fLast) && ((zaxis.fFirst > 1) || (zaxis.fLast < zaxis.fNbins))) {
            this.zoom_zmin = zaxis.fFirst > 1 ? zaxis.GetBinLowEdge(zaxis.fFirst) : zaxis.fXmin;
            this.zoom_zmax = zaxis.fLast < zaxis.fNbins ? zaxis.GetBinLowEdge(zaxis.fLast+1) : zaxis.fXmax;
         }
      }

      var pad = this.root_pad();

      if (!pad || !('fUxmin' in pad) || this.create_canvas) return;

      var min = pad.fUxmin, max = pad.fUxmax;

      // first check that non-default values are there
      if ((ndim < 3) && ((min !== 0) || (max !== 1))) {
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

      min = pad.fUymin; max = pad.fUymax;

      if ((ndim == 2) && ((min !== 0) || (max !== 1))) {
         if (pad.fLogy > 0) {
            min = Math.exp(min * Math.log(10));
            max = Math.exp(max * Math.log(10));
         }

         if (min !== yaxis.fXmin || max !== yaxis.fXmax)
            if (min >= yaxis.fXmin && max <= yaxis.fXmax) {
               // set zoom values if only inside range
               this.zoom_ymin = min;
               this.zoom_ymax = max;
            }
      }
   }

   THistPainter.prototype.CheckHistDrawAttributes = function() {

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

   THistPainter.prototype.UpdateObject = function(obj) {

      var histo = this.GetObject();

      if (obj !== histo) {

         if (!this.MatchObjectType(obj)) return false;

         // TODO: simple replace of object does not help - one can have different
         // complex relations between histo and stat box, histo and colz axis,
         // one could have THStack or TMultiGraph object
         // The only that could be done is update of content

         // this.histo = obj;

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
         }
         CopyAxis(histo.fXaxis, obj.fXaxis);
         CopyAxis(histo.fYaxis, obj.fYaxis);
         CopyAxis(histo.fZaxis, obj.fZaxis);
         if (!this.main_painter().zoom_changed_interactive) {
            function CopyZoom(tgt,src) {
               tgt.fXmin = src.fXmin;
               tgt.fXmax = src.fXmax;
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
         }

         if (obj.fFunctions && !this.options.Same && this.options.Func)
            for (var n=0;n<obj.fFunctions.arr.length;++n) {
               var func = obj.fFunctions.arr[n];
               if (!func || !func._typename || !func.fName) continue;
               var funcpainter = this.FindPainterFor(null, func.fName, func._typename);
               if (funcpainter) funcpainter.UpdateObject(func);
            }
      }

      if (!this.zoom_changed_interactive) this.CheckPadRange();

      this.ScanContent();

      this.histogram_updated = true; // indicate that object updated

      return true;
   }

   THistPainter.prototype.CreateAxisFuncs = function(with_y_axis, with_z_axis) {
      // here functions are defined to convert index to axis value and back
      // introduced to support non-equidistant bins

      this.xmin = this.histo.fXaxis.fXmin;
      this.xmax = this.histo.fXaxis.fXmax;

      if (this.histo.fXaxis.fXbins.length == this.nbinsx+1) {
         this.regularx = false;
         this.GetBinX = function(bin) {
            var indx = Math.round(bin);
            if (indx <= 0) return this.xmin;
            if (indx > this.nbinsx) return this.xmax;
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
            if (indx > this.nbinsy) return this.ymax;
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
            if (indx > this.nbinsz) return this.zmax;
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

   THistPainter.prototype.CreateXY = function() {
      // here we create x,y objects which maps our physical coordinates into pixels
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
         this.x = d3.scaleTime();
      } else
      if (this.swap_xy ? pad.fLogy : pad.fLogx) {
         this.logx = true;

         if (this.scale_xmax <= 0) this.scale_xmax = 0;

         if ((this.scale_xmin <= 0) && (this.nbinsx>0))
            for (var i=0;i<this.nbinsx;++i) {
               this.scale_xmin = Math.max(this.scale_xmin, this.GetBinX(i));
               if (this.scale_xmin>0) break;
            }

         if ((this.scale_xmin <= 0) || (this.scale_xmin >= this.scale_xmax))
            this.scale_xmin = this.scale_xmax * 0.0001;

         this.xmin_log = this.scale_xmin;

         this.x = d3.scaleLog();
      } else {
         this.x = d3.scaleLinear();
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

         // this is for 2/3 dim histograms - find first non-negative bin
         if ((this.scale_ymin <= 0) && (this.nbinsy>0) && (this.Dimension()>1))
            for (var i=0;i<this.nbinsy;++i) {
               this.scale_ymin = Math.max(this.scale_ymin, this.GetBinY(i));
               if (this.scale_ymin>0) break;
            }

         if ((this.scale_ymin <= 0) && ('ymin_nz' in this) && (this.ymin_nz > 0) && (this.ymin_nz < 1e-2*this.ymax))
            this.scale_ymin = 0.3*this.ymin_nz;

         if ((this.scale_ymin <= 0) || (this.scale_ymin >= this.scale_ymax))
            this.scale_ymin = 3e-4 * this.scale_ymax;

         this.ymin_log = this.scale_ymin;

         this.y = d3.scaleLog();
      } else
      if (this.y_kind=='time') {
         this.y = d3.scaleTime();
      } else {
         this.y = d3.scaleLinear()
      }

      this.y.domain([ this.ConvertY(this.scale_ymin), this.ConvertY(this.scale_ymax) ])
            .range(this.swap_xy ? [ 0, w ] : [ h, 0 ]);

      if (this.y_kind=='time') {
         // we emulate scale functionality
         this.gry = function(val) { return this.y(this.ConvertY(val)); }
      } else
      if (this.logy) {
         // make protection for log
         this.gry = function(val) { return (val < this.scale_ymin) ? (this.swap_xy ? -5 : this.y.range()[0]+5) : this.y(val); }
      } else {
         this.gry = this.y;
      }

      this.SetRootPadRange(pad);
   }

   /** Set selected range back to TPad object */
   THistPainter.prototype.SetRootPadRange = function(pad) {
      if (!pad) return;

      if (this.logx) {
         pad.fUxmin = JSROOT.log10(this.scale_xmin);
         pad.fUxmax = JSROOT.log10(this.scale_xmax);
      } else {
         pad.fUxmin = this.scale_xmin;
         pad.fUxmax = this.scale_xmax;
      }
      if (this.logy) {
         pad.fUymin = JSROOT.log10(this.scale_ymin);
         pad.fUymax = JSROOT.log10(this.scale_ymax);
      } else {
         pad.fUymin = this.scale_ymin;
         pad.fUymax = this.scale_ymax;
      }

      var rx = pad.fUxmax - pad.fUxmin,
          mx = 1 - pad.fLeftMargin - pad.fRightMargin,
          ry = pad.fUymax - pad.fUymin,
          my = 1 - pad.fBottomMargin - pad.fTopMargin;

      pad.fX1 = pad.fUxmin - rx/mx*pad.fLeftMargin;
      pad.fX2 = pad.fUxmax + rx/mx*pad.fRightMargin;
      pad.fY1 = pad.fUymin - ry/my*pad.fBottomMargin;
      pad.fY2 = pad.fUymax + ry/my*pad.fTopMargin;
   }

   THistPainter.prototype.DrawGrids = function() {
      // grid can only be drawn by first painter
      if (!this.is_main_painter()) return;

      var layer = this.svg_frame().select(".grid_layer");

      layer.selectAll(".xgrid").remove();
      layer.selectAll(".ygrid").remove();

      var pad = this.root_pad(), h = this.frame_height(), w = this.frame_width(),
          grid, grid_style = JSROOT.gStyle.fGridStyle, grid_color = "black";

      if (JSROOT.Painter.fGridColor > 0)
         grid_color = JSROOT.Painter.root_colors[JSROOT.Painter.fGridColor];

      if ((grid_style < 0) || (grid_style >= JSROOT.Painter.root_line_styles.length)) grid_style = 11;

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
               .style('stroke',grid_color).style("stroke-width",JSROOT.gStyle.fGridWidth)
               .style("stroke-dasharray",JSROOT.Painter.root_line_styles[grid_style]);
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
               .style('stroke',grid_color).style("stroke-width",JSROOT.gStyle.fGridWidth)
               .style("stroke-dasharray", JSROOT.Painter.root_line_styles[grid_style]);
      }
   }

   THistPainter.prototype.DrawBins = function() {
      alert("HistPainter.DrawBins not implemented");
   }

   THistPainter.prototype.AxisAsText = function(axis, value) {
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

   THistPainter.prototype.DrawAxes = function(shrink_forbidden) {
      // axes can be drawn only for main histogram

      if (!this.is_main_painter()) return;

      var layer = this.svg_frame().select(".axis_layer"),
          w = this.frame_width(),
          h = this.frame_height(),
          pad = this.root_pad();

      this.x_handle = new JSROOT.TAxisPainter(this.histo.fXaxis, true);
      this.x_handle.SetDivId(this.divid, -1);
      this.x_handle.pad_name = this.pad_name;

      this.x_handle.SetAxisConfig("xaxis",
                                  (this.logx && (this.x_kind !== "time")) ? "log" : this.x_kind,
                                  this.x, this.xmin, this.xmax, this.scale_xmin, this.scale_xmax);
      this.x_handle.invert_side = (this.options.AxisPos>=10);

      this.y_handle = new JSROOT.TAxisPainter(this.histo.fYaxis, true);
      this.y_handle.SetDivId(this.divid, -1);
      this.y_handle.pad_name = this.pad_name;

      this.y_handle.SetAxisConfig("yaxis",
                                  (this.logy && this.y_kind !== "time") ? "log" : this.y_kind,
                                  this.y, this.ymin, this.ymax, this.scale_ymin, this.scale_ymax);
      this.y_handle.invert_side = (this.options.AxisPos % 10) === 1;

      var draw_horiz = this.swap_xy ? this.y_handle : this.x_handle,
          draw_vertical = this.swap_xy ? this.x_handle : this.y_handle;

      draw_horiz.DrawAxis(false, layer, w, h, draw_horiz.invert_side ? undefined : "translate(0," + h + ")",
                          false, pad.fTickx ? -h : 0);
      draw_vertical.DrawAxis(true, layer, w, h, draw_vertical.invert_side ? "translate(" + w + ",0)" : undefined,
                             false, pad.fTicky ? w : 0);

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

   THistPainter.prototype.ToggleTitle = function(arg) {
      if (!this.is_main_painter()) return false;
      if (arg==='only-check') return !this.histo.TestBit(JSROOT.TH1StatusBits.kNoTitle);
      this.histo.InvertBit(JSROOT.TH1StatusBits.kNoTitle);
      this.DrawTitle();
   }

   THistPainter.prototype.DrawTitle = function() {

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
         if (tpainter) tpainter.Redraw();
      } else
      if (draw_title && !tpainter && (this.histo.fTitle.length > 0)) {
         pavetext = JSROOT.Create("TPaveText");

         JSROOT.extend(pavetext, { fName: "title", fX1NDC: 0.28, fY1NDC: 0.94, fX2NDC: 0.72, fY2NDC: 0.99 } );
         pavetext.AddText(this.histo.fTitle);

         JSROOT.Painter.drawPaveText(this.divid, pavetext);
      }
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

      if (stat == null) {
         if (arg.indexOf('-check')>0) return false;
         // when statbox created first time, one need to draw it
         stat = this.CreateStat();
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

   THistPainter.prototype.IsAxisZoomed = function(axis) {
      var obj = this.main_painter() || this;
      return obj['zoom_'+axis+'min'] !== obj['zoom_'+axis+'max'];
   }

   THistPainter.prototype.GetSelectIndex = function(axis, size, add) {
      // be aware - here indexes starts from 0
      var indx = 0, obj = this.main_painter();
      if (!obj) obj = this;
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

   THistPainter.prototype.FindStat = function() {
      if (this.histo.fFunctions !== null)
         for (var i = 0; i < this.histo.fFunctions.arr.length; ++i) {
            var func = this.histo.fFunctions.arr[i];

            if ((func._typename == 'TPaveStats') &&
                (func.fName == 'stats')) return func;
         }

      return null;
   }

   THistPainter.prototype.CreateStat = function(opt_stat) {

      if (!this.draw_content || !this.is_main_painter()) return null;

      this.create_stats = true;

      var stats = this.FindStat();
      if (stats) return stats;

      var st = JSROOT.gStyle;

      stats = JSROOT.Create('TPaveStats');
      JSROOT.extend(stats, { fName : 'stats',
                             fOptStat: opt_stat || st.fOptStat,
                             fOptFit: st.fOptFit,
                             fBorderSize : 1} );

      stats.fX1NDC = st.fStatX - st.fStatW;
      stats.fY1NDC = st.fStatY - st.fStatH;
      stats.fX2NDC = st.fStatX;
      stats.fY2NDC = st.fStatY;

      stats.fFillColor = st.fStatColor;
      stats.fFillStyle = st.fStatStyle;

      stats.fTextAngle = 0;
      stats.fTextSize = st.fStatFontSize; // 9 ??
      stats.fTextAlign = 12;
      stats.fTextColor = st.fStatTextColor;
      stats.fTextFont = st.fStatFont;

//      st.fStatBorderSize : 1,

      if (this.histo._typename.match(/^TProfile/) || this.histo._typename.match(/^TH2/))
         stats.fY1NDC = 0.67;

      stats.AddText(this.histo.fName);

      if (!this.histo.fFunctions)
         this.histo.fFunctions = JSROOT.Create("TList");

      this.histo.fFunctions.Add(stats,"");

      return stats;
   }

   THistPainter.prototype.AddFunction = function(obj, asfirst) {
      var histo = this.GetObject();
      if (!histo || !obj) return;

      if (histo.fFunctions == null)
         histo.fFunctions = JSROOT.Create("TList");

      if (asfirst)
         histo.fFunctions.AddFirst(obj);
      else
         histo.fFunctions.Add(obj);

   }

   THistPainter.prototype.FindFunction = function(type_name) {
      var funcs = this.GetObject().fFunctions;
      if (funcs === null) return null;

      for (var i = 0; i < funcs.arr.length; ++i)
         if (funcs.arr[i]._typename === type_name) return funcs.arr[i];

      return null;
   }

   THistPainter.prototype.DrawNextFunction = function(indx, callback) {
      // method draws next function from the functions list

      if (this.options.Same || !this.options.Func || !this.histo.fFunctions ||
           (indx >= this.histo.fFunctions.arr.length)) return JSROOT.CallBack(callback);

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

      if (do_draw)
         return JSROOT.draw(this.divid, func, opt, this.DrawNextFunction.bind(this, indx+1, callback));

      this.DrawNextFunction(indx+1, callback);
   }

   THistPainter.prototype.UnzoomUserRange = function(dox, doy, doz) {

      if (!this.histo) return false;

      var res = false, painter = this;

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

      if (dox && UnzoomTAxis(this.histo.fXaxis)) res = true;
      if (doy && (UnzoomTAxis(this.histo.fYaxis) || UzoomMinMax(1, this.histo))) res = true;
      if (doz && (UnzoomTAxis(this.histo.fZaxis) || UzoomMinMax(2, this.histo))) res = true;

      return res;
   }

   THistPainter.prototype.ToggleLog = function(axis) {
      var obj = this.main_painter(), pad = this.root_pad();
      if (!obj) obj = this;
      var curr = pad["fLog" + axis];
      // do not allow log scale for labels
      if (!curr) {
         var kind = this[axis+"_kind"];
         if (this.swap_xy && axis==="x") kind = this["y_kind"]; else
         if (this.swap_xy && axis==="y") kind = this["x_kind"];
         if (kind === "labels") return;
      }
      var pp = this.pad_painter();
      if (pp && pp._websocket) {
         pp._websocket.send("EXEC:SetLog" + axis + (curr ? "(0)" : "(1)"));
      } else {
         pad["fLog" + axis] = curr ? 0 : 1;
         obj.RedrawPad();
      }
   }

   THistPainter.prototype.Zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {
      // function can be used for zooming into specified range
      // if both limits for each axis 0 (like xmin==xmax==0), axis will be unzoomed

      if (xmin==="x") { xmin = xmax; xmax = ymin; ymin = undefined; } else
      if (xmin==="y") { ymax = ymin; ymin = xmax; xmin = xmax = undefined; } else
      if (xmin==="z") { zmin = xmax; zmax = ymin; xmin = xmax = ymin = undefined; }

      var main = this.main_painter(),
          zoom_x = (xmin !== xmax), zoom_y = (ymin !== ymax), zoom_z = (zmin !== zmax),
          unzoom_x = false, unzoom_y = false, unzoom_z = false;

      if (zoom_x) {
         var cnt = 0, main_xmin = main.xmin;
         if (main.logx && main.xmin_log) main_xmin = main.xmin_log;
         if (xmin <= main_xmin) { xmin = main_xmin; cnt++; }
         if (xmax >= main.xmax) { xmax = main.xmax; cnt++; }
         if (cnt === 2) { zoom_x = false; unzoom_x = true; }
      } else {
         unzoom_x = (xmin === xmax) && (xmin === 0);
      }

      if (zoom_y) {
         var cnt = 0, main_ymin = main.ymin;
         if (main.logy && main.ymin_log) main_ymin = main.ymin_log;
         if (ymin <= main_ymin) { ymin = main_ymin; cnt++; }
         if (ymax >= main.ymax) { ymax = main.ymax; cnt++; }
         if (cnt === 2) { zoom_y = false; unzoom_y = true; }
      } else {
         unzoom_y = (ymin === ymax) && (ymin === 0);
      }

      if (zoom_z) {
         var cnt = 0, main_zmin = main.zmin;
         // if (main.logz && main.ymin_nz && main.Dimension()===2) main_zmin = 0.3*main.ymin_nz;
         if (zmin <= main_zmin) { zmin = main_zmin; cnt++; }
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

   THistPainter.prototype.Unzoom = function(dox, doy, doz) {
      if (typeof dox === 'undefined') { dox = true; doy = true; doz = true; } else
      if (typeof dox === 'string') { doz = dox.indexOf("z")>=0; doy = dox.indexOf("y")>=0; dox = dox.indexOf("x")>=0; }

      var last = this.zoom_changed_interactive;

      if (dox || doy || dox) this.zoom_changed_interactive = 2;

      var changed = this.Zoom(dox ? 0 : undefined, dox ? 0 : undefined,
                              doy ? 0 : undefined, doy ? 0 : undefined,
                              doz ? 0 : undefined, doz ? 0 : undefined);

      // if unzooming has no effect, decrease counter
      if ((dox || doy || dox) && !changed)
         this.zoom_changed_interactive = (!isNaN(last) && (last>0)) ? last - 1 : 0;

      return changed;

   }

   THistPainter.prototype.clearInteractiveElements = function() {
      JSROOT.Painter.closeMenu();
      if (this.zoom_rect != null) { this.zoom_rect.remove(); this.zoom_rect = null; }
      this.zoom_kind = 0;

      // enable tooltip in frame painter
      this.SwitchTooltip(true);
   }

   THistPainter.prototype.mouseDoubleClick = function() {
      d3.event.preventDefault();
      var m = d3.mouse(this.svg_frame().node());
      this.clearInteractiveElements();
      var kind = "xyz";
      if ((m[0] < 0) || (m[0] > this.frame_width())) kind = this.swap_xy ? "x" : "y"; else
      if ((m[1] < 0) || (m[1] > this.frame_height())) kind = this.swap_xy ? "y" : "x";
      this.Unzoom(kind);
   }

   THistPainter.prototype.startRectSel = function() {
      // ignore when touch selection is activated

      if (this.zoom_kind > 100) return;

      // ignore all events from non-left button
      if ((d3.event.which || d3.event.button) !== 1) return;

      d3.event.preventDefault();

      this.clearInteractiveElements();
      this.zoom_origin = d3.mouse(this.svg_frame().node());

      var w = this.frame_width(), h = this.frame_height();

      this.zoom_curr = [ Math.max(0, Math.min(w, this.zoom_origin[0])),
                         Math.max(0, Math.min(h, this.zoom_origin[1])) ];

      if ((this.zoom_origin[0] < 0) || (this.zoom_origin[0] > w)) {
         this.zoom_kind = 3; // only y
         this.zoom_origin[0] = 0;
         this.zoom_origin[1] = this.zoom_curr[1];
         this.zoom_curr[0] = w;
         this.zoom_curr[1] += 1;
      } else if ((this.zoom_origin[1] < 0) || (this.zoom_origin[1] > h)) {
         this.zoom_kind = 2; // only x
         this.zoom_origin[0] = this.zoom_curr[0];
         this.zoom_origin[1] = 0;
         this.zoom_curr[0] += 1;
         this.zoom_curr[1] = h;
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

   THistPainter.prototype.moveRectSel = function() {

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

   THistPainter.prototype.endRectSel = function() {
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
         this.zoom_changed_interactive = 2;
         this.Zoom(xmin, xmax, ymin, ymax);
      }
   }

   THistPainter.prototype.startTouchZoom = function() {
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

      if ((arr.length != 2) || !JSROOT.gStyle.Zooming || !JSROOT.gStyle.ZoomTouch) return;

      d3.event.preventDefault();
      d3.event.stopPropagation();

      this.clearInteractiveElements();

      this.svg_frame().on("touchcancel", null)
                      .on("touchend", null);

      var pnt1 = arr[0], pnt2 = arr[1], w = this.frame_width(), h = this.frame_height();

      this.zoom_curr = [ Math.min(pnt1[0], pnt2[0]), Math.min(pnt1[1], pnt2[1]) ];
      this.zoom_origin = [ Math.max(pnt1[0], pnt2[0]), Math.max(pnt1[1], pnt2[1]) ];

      if ((this.zoom_curr[0] < 0) || (this.zoom_curr[0] > w)) {
         this.zoom_kind = 103; // only y
         this.zoom_curr[0] = 0;
         this.zoom_origin[0] = w;
      } else if ((this.zoom_origin[1] > h) || (this.zoom_origin[1] < 0)) {
         this.zoom_kind = 102; // only x
         this.zoom_curr[1] = 0;
         this.zoom_origin[1] = h;
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

   THistPainter.prototype.moveTouchSel = function() {
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

   THistPainter.prototype.endTouchSel = function() {

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
         this.zoom_changed_interactive = 2;
         this.Zoom(xmin, xmax, ymin, ymax);
      }

      d3.event.stopPropagation();
   }

   THistPainter.prototype.AllowDefaultYZooming = function() {
      // return true if default Y zooming should be enabled
      // it is typically for 2-Dim histograms or
      // when histogram not draw, defined by other painters

      if (this.Dimension()>1) return true;
      if (this.draw_content) return false;

      var pad_painter = this.pad_painter(true);
      if (pad_painter &&  pad_painter.painters)
         for (var k = 0; k < pad_painter.painters.length; ++k) {
            var subpainter = pad_painter.painters[k];
            if ((subpainter!==this) && subpainter.wheel_zoomy!==undefined)
               return subpainter.wheel_zoomy;
         }

      return false;
   }

   THistPainter.prototype.AnalyzeMouseWheelEvent = function(event, item, dmin, ignore) {

      item.min = item.max = undefined;
      item.changed = false;
      if (ignore && item.ignore) return;

      var delta = 0, delta_left = 1, delta_right = 1;

      if ('dleft' in item) { delta_left = item.dleft; delta = 1; }
      if ('dright' in item) { delta_right = item.dright; delta = 1; }

      if ('delta' in item)
         delta = item.delta;
      else
      if (event && event.wheelDelta !== undefined ) {
         // WebKit / Opera / Explorer 9
         delta = -event.wheelDelta;
      } else if (event && event.deltaY !== undefined ) {
         // Firefox
         delta = event.deltaY;
      } else if (event && event.detail !== undefined) {
         delta = event.detail;
      }

      if (delta===0) return;
      delta = (delta<0) ? -0.2 : 0.2;

      delta_left *= delta
      delta_right *= delta;

      var lmin = item.min = this["scale_"+item.name+"min"],
          lmax = item.max = this["scale_"+item.name+"max"],
          gmin = this[item.name+"min"],
          gmax = this[item.name+"max"];

      if ((item.min === item.max) && (delta<0)) {
         item.min = gmin;
         item.max = gmax;
      }

      if (item.min >= item.max) return;

      if ((dmin>0) && (dmin<1)) {
         if (this['log'+item.name]) {
            var factor = (item.min>0) ? JSROOT.log10(item.max/item.min) : 2;
            if (factor>10) factor = 10; else if (factor<0.01) factor = 0.01;
            item.min = item.min / Math.pow(10, factor*delta_left*dmin);
            item.max = item.max * Math.pow(10, factor*delta_right*(1-dmin));
         } else {
            var rx_left = (item.max - item.min),
                rx_right = rx_left;
            if (delta_left>0) rx_left = 1.001 * rx_left / (1-delta_left);
            item.min += -delta_left*dmin*rx_left;

            if (delta_right>0) rx_right = 1.001 * rx_right / (1-delta_right);

            item.max -= -delta_right*(1-dmin)*rx_right;
         }
         if (item.min >= item.max)
            item.min = item.max = undefined;
         else
         if (delta_left !== delta_right) {
            // extra check case when moving left or right
            if (((item.min < gmin) && (lmin===gmin)) ||
                ((item.max > gmax) && (lmax==gmax)))
                   item.min = item.max = undefined;
         }

      } else {
         item.min = item.max = undefined;
      }

      item.changed = ((item.min !== undefined) && (item.max !== undefined));
   }

   THistPainter.prototype.mouseWheel = function() {
      d3.event.stopPropagation();

      d3.event.preventDefault();
      this.clearInteractiveElements();

      var itemx = { name: "x", ignore: false },
          itemy = { name: "y", ignore: !this.AllowDefaultYZooming() },
          cur = d3.mouse(this.svg_frame().node()),
          w = this.frame_width(), h = this.frame_height();

      this.AnalyzeMouseWheelEvent(d3.event, this.swap_xy ? itemy : itemx, cur[0] / w, (cur[1] >=0) && (cur[1] <= h));

      this.AnalyzeMouseWheelEvent(d3.event, this.swap_xy ? itemx : itemy, 1 - cur[1] / h, (cur[0] >= 0) && (cur[0] <= w));

      this.Zoom(itemx.min, itemx.max, itemy.min, itemy.max);

      if (itemx.changed || itemy.changed) this.zoom_changed_interactive = 2;
   }

   THistPainter.prototype.ShowAxisStatus = function(axis_name) {
      // method called normally when mouse enter main object element

      var status_func = this.GetShowStatusFunc();

      if (!status_func) return;

      var taxis = this.histo ? this.histo['f'+axis_name.toUpperCase()+"axis"] : null;

      var hint_name = axis_name, hint_title = "TAxis";

      if (taxis) { hint_name = taxis.fName; hint_title = taxis.fTitle || "histogram TAxis object"; }

      var m = d3.mouse(this.svg_frame().node());

      var id = (axis_name=="x") ? 0 : 1;
      if (this.swap_xy) id = 1-id;

      var axis_value = (axis_name=="x") ? this.RevertX(m[id]) : this.RevertY(m[id]);

      status_func(hint_name, hint_title, axis_name + " : " + this.AxisAsText(axis_name, axis_value),
                  m[0].toFixed(0)+","+ m[1].toFixed(0));
   }

   THistPainter.prototype.AddInteractive = function() {
      // only first painter in list allowed to add interactive functionality to the frame

      if ((!JSROOT.gStyle.Zooming && !JSROOT.gStyle.ContextMenu) || !this.is_main_painter()) return;

      var svg = this.svg_frame();

      if (svg.empty() || svg.property('interactive_set')) return;

      this.AddKeysHandler();

      this.last_touch = new Date(0);
      this.zoom_kind = 0; // 0 - none, 1 - XY, 2 - only X, 3 - only Y, (+100 for touches)
      this.zoom_rect = null;
      this.zoom_origin = null;  // original point where zooming started
      this.zoom_curr = null;    // current point for zooming
      this.touch_cnt = 0;

      if (JSROOT.gStyle.Zooming) {
         if (JSROOT.gStyle.ZoomMouse) {
            svg.on("mousedown", this.startRectSel.bind(this));
            svg.on("dblclick", this.mouseDoubleClick.bind(this));
         }
         if (JSROOT.gStyle.ZoomWheel) {
            svg.on("wheel", this.mouseWheel.bind(this));
         }
      }

      if (JSROOT.touches && ((JSROOT.gStyle.Zooming && JSROOT.gStyle.ZoomTouch) || JSROOT.gStyle.ContextMenu))
         svg.on("touchstart", this.startTouchZoom.bind(this));

      if (JSROOT.gStyle.ContextMenu) {
         if (JSROOT.touches) {
            svg.selectAll(".xaxis_container")
               .on("touchstart", this.startTouchMenu.bind(this,"x"));
            svg.selectAll(".yaxis_container")
                .on("touchstart", this.startTouchMenu.bind(this,"y"));
         }
         svg.on("contextmenu", this.ShowContextMenu.bind(this));
         svg.selectAll(".xaxis_container")
             .on("contextmenu", this.ShowContextMenu.bind(this,"x"));
         svg.selectAll(".yaxis_container")
             .on("contextmenu", this.ShowContextMenu.bind(this,"y"));
      }

      svg.selectAll(".xaxis_container")
         .on("mousemove", this.ShowAxisStatus.bind(this,"x"));
      svg.selectAll(".yaxis_container")
         .on("mousemove", this.ShowAxisStatus.bind(this,"y"));

      svg.property('interactive_set', true);
   }

   THistPainter.prototype.AddKeysHandler = function() {
      if (this.keys_handler || !this.is_main_painter() || JSROOT.BatchMode || (typeof window == 'undefined')) return;

      this.keys_handler = this.ProcessKeyPress.bind(this);

      window.addEventListener('keydown', this.keys_handler, false);
   }

   THistPainter.prototype.ProcessKeyPress = function(evnt) {

      var main = this.select_main();
      if (main.empty()) return;
      var isactive = main.attr('frame_active');
      if (isactive && isactive!=='true') return;

      var key = "";
      switch (evnt.keyCode) {
         case 33: key = "PageUp"; break;
         case 34: key = "PageDown"; break;
         case 37: key = "ArrowLeft"; break;
         case 38: key = "ArrowUp"; break;
         case 39: key = "ArrowRight"; break;
         case 40: key = "ArrowDown"; break;
         case 42: key = "PrintScreen"; break;
         case 106: key = "*"; break;
         default: return false;
      }

      if (evnt.shiftKey) key = "Shift " + key;
      if (evnt.altKey) key = "Alt " + key;
      if (evnt.ctrlKey) key = "Ctrl " + key;

      var zoom = { name: "x", dleft: 0, dright: 0 };

      switch (key) {
         case "ArrowLeft":  zoom.dleft = -1; zoom.dright = 1; break;
         case "ArrowRight":  zoom.dleft = 1; zoom.dright = -1; break;
         case "Ctrl ArrowLeft": zoom.dleft = zoom.dright = -1; break;
         case "Ctrl ArrowRight": zoom.dleft = zoom.dright = 1; break;
         case "ArrowUp":  zoom.name = "y"; zoom.dleft = 1; zoom.dright = -1; break;
         case "ArrowDown":  zoom.name = "y"; zoom.dleft = -1; zoom.dright = 1; break;
         case "Ctrl ArrowUp": zoom.name = "y"; zoom.dleft = zoom.dright = 1; break;
         case "Ctrl ArrowDown": zoom.name = "y"; zoom.dleft = zoom.dright = -1; break;
      }

      if (zoom.dleft || zoom.dright) {
         if (!JSROOT.gStyle.Zooming) return false;
         // in 3dmode with orbit control ignore simple arrows
         if (this.mode3d && (key.indexOf("Ctrl")!==0)) return false;
         this.AnalyzeMouseWheelEvent(null, zoom, 0.5);
         this.Zoom(zoom.name, zoom.min, zoom.max);
         if (zoom.changed) this.zoom_changed_interactive = 2;
         evnt.stopPropagation();
         evnt.preventDefault();
      } else {
         var pp = this.pad_painter(true),
             func = pp ? pp.FindButton(key) : "";
         if (func) {
            pp.PadButtonClick(func);
            evnt.stopPropagation();
            evnt.preventDefault();
         }
      }

      return true; // just process any key press
   }

   THistPainter.prototype.ShowContextMenu = function(kind, evnt, obj) {
      // ignore context menu when touches zooming is ongoing
      if (('zoom_kind' in this) && (this.zoom_kind > 100)) return;

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
                pp = this.pad_painter(true),
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
         }
      }

      // one need to copy event, while after call back event may be changed
      menu_painter.ctx_menu_evnt = evnt;

      JSROOT.Painter.createMenu(menu_painter, function(menu) {
         var domenu = menu.painter.FillContextMenu(menu, kind, obj);

         // fill frame menu by default - or append frame elements when activated in the frame corner
         if (fp && (!domenu || (frame_corner && (kind!=="frame"))))
            domenu = fp.FillContextMenu(menu);

         if (domenu)
            menu.painter.FillObjectExecMenu(menu, function() {
                // suppress any running zooming
                menu.painter.SwitchTooltip(false);
                menu.show(menu.painter.ctx_menu_evnt, menu.painter.SwitchTooltip.bind(menu.painter, true) );
            });

      });  // end menu creation
   }


   THistPainter.prototype.ChangeUserRange = function(arg) {
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

   THistPainter.prototype.FillContextMenu = function(menu, kind, obj) {

      // when fill and show context menu, remove all zooming
      this.clearInteractiveElements();

      if ((kind=="x") || (kind=="y") || (kind=="z")) {
         var faxis = this.histo.fXaxis;
         if (kind=="y") faxis = this.histo.fYaxis;  else
         if (kind=="z") faxis = obj ? obj : this.histo.fZaxis;
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

      if ((this.options.Lego > 0) || (this.options.Surf > 0) || (this.Dimension() === 3)) {
         // menu for 3D drawings

         if (menu.size() > 0)
            menu.add("separator");

         var main = this.main_painter() || this;

         menu.addchk(main.tooltip_allowed, 'Show tooltips', function() {
            main.tooltip_allowed = !main.tooltip_allowed;
         });

         menu.addchk(main.enable_highlight, 'Highlight bins', function() {
            main.enable_highlight = !main.enable_highlight;
            if (!main.enable_highlight && main.BinHighlight3D && main.mode3d) main.BinHighlight3D(null);
         });

         menu.addchk(main.options.FrontBox, 'Front box', function() {
            main.options.FrontBox = !main.options.FrontBox;
            if (main.Render3D) main.Render3D();
         });
         menu.addchk(main.options.BackBox, 'Back box', function() {
            main.options.BackBox = !main.options.BackBox;
            if (main.Render3D) main.Render3D();
         });

         if (this.draw_content) {
            menu.addchk(!this.options.Zero, 'Suppress zeros', function() {
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

         if (main.control && typeof main.control.reset === 'function')
            menu.add('Reset camera', function() {
               main.control.reset();
            });
      }

      this.FillAttContextMenu(menu);

      if (this.histogram_updated && this.zoom_changed_interactive)
         menu.add('Let update zoom', function() {
            this.zoom_changed_interactive = 0;
         });

      return true;
   }

   THistPainter.prototype.ButtonClick = function(funcname) {
      if (!this.is_main_painter()) return false;
      switch(funcname) {
         case "ToggleZoom":
            if ((this.zoom_xmin !== this.zoom_xmax) || (this.zoom_ymin !== this.zoom_ymax) || (this.zoom_zmin !== this.zoom_zmax)) {
               this.Unzoom();
               return true;
            }
            if (this.draw_content && (typeof this.AutoZoom === 'function')) {
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

   THistPainter.prototype.FillToolbar = function() {
      var pp = this.pad_painter(true);
      if (pp===null) return;

      pp.AddButton(JSROOT.ToolbarIcons.auto_zoom, 'Toggle between unzoom and autozoom-in', 'ToggleZoom', "Ctrl *");
      pp.AddButton(JSROOT.ToolbarIcons.arrow_right, "Toggle log x", "ToggleLogX", "PageDown");
      pp.AddButton(JSROOT.ToolbarIcons.arrow_up, "Toggle log y", "ToggleLogY", "PageUp");
      if (this.Dimension() > 1)
         pp.AddButton(JSROOT.ToolbarIcons.arrow_diag, "Toggle log z", "ToggleLogZ");
      if (this.draw_content)
         pp.AddButton(JSROOT.ToolbarIcons.statbox, 'Toggle stat box', "ToggleStatBox");
   }

   THistPainter.prototype.Get3DToolTip = function(indx) {
      var tip = { bin: indx, name: this.GetObject().fName, title: this.GetObject().fTitle };
      switch (this.Dimension()) {
         case 1:
            tip.ix = indx; tip.iy = 1;
            tip.value = this.histo.getBinContent(tip.ix);
            tip.error = this.histo.getBinError(indx);
            tip.lines = this.GetBinTips(indx-1);
            break;
         case 2:
            tip.ix = indx % (this.nbinsx + 2);
            tip.iy = (indx - tip.ix) / (this.nbinsx + 2);
            tip.value = this.histo.getBinContent(tip.ix, tip.iy);
            tip.error = this.histo.getBinError(indx);
            tip.lines = this.GetBinTips(tip.ix-1, tip.iy-1);
            break;
         case 3:
            tip.ix = indx % (this.nbinsx+2);
            tip.iy = ((indx - tip.ix) / (this.nbinsx+2)) % (this.nbinsy+2);
            tip.iz = (indx - tip.ix - tip.iy * (this.nbinsx+2)) / (this.nbinsx+2) / (this.nbinsy+2);
            tip.value = this.GetObject().getBinContent(tip.ix, tip.iy, tip.iz);
            tip.error = this.histo.getBinError(indx);
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

      if (this.Dimension() < 3) {
         this.zmin = this.colzmin;
         this.zmax = this.colzmax;
      }

      return this.fContour;
   }

   THistPainter.prototype.GetContour = function() {
      if (this.fContour) return this.fContour;

      var main = this.main_painter();
      if ((main !== this) && main.fContour) {
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
      if (this.zoom_zmin != this.zoom_zmax) {
         zmin = this.zoom_zmin;
         zmax = this.zoom_zmax;
      }

      if (histo.fContour && (histo.fContour.length>1) && histo.TestBit(JSROOT.TH1StatusBits.kUserContour)) {
         this.fContour = JSROOT.clone(histo.fContour);
         this.fCustomContour = true;
         this.colzmin = zmin;
         this.colzmax = zmax;
         if (zmax > this.fContour[this.fContour.length-1]) this.fContour.push(zmax);
         if (this.Dimension()<3) {
            this.zmin = this.colzmin;
            this.zmax = this.colzmax;
         }
         return this.fContour;
      }

      this.fCustomContour = false;

      return this.CreateContour(nlevels, zmin, zmax, zminpos);
   }

   THistPainter.prototype.getContourIndex = function(zc) {
      // return contour index, which corresponds to the z content value

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
      if (zc < this.colzmin) return (this.options.Color === 11) ? 0 : -1;

      // if bin content exactly zmin, draw it when col0 specified or when content is positive
      if (zc===this.colzmin) return ((this.colzmin != 0) || (this.options.Color === 11) || this.IsTH2Poly()) ? 0 : -1;

      return Math.floor(0.01+(zc-this.colzmin)*(cntr.length-1)/(this.colzmax-this.colzmin));
   }

   THistPainter.prototype.getIndexColor = function(index, asindx) {
      if (index < 0) return null;

      var cntr = this.GetContour();

      var palette = this.GetPalette();
      var theColor = Math.floor((index+0.99)*palette.length/(cntr.length-1));
      if (theColor > palette.length-1) theColor = palette.length-1;
      return asindx ? theColor : palette[theColor];
   }

   THistPainter.prototype.getValueColor = function(zc, asindx) {

      var index = this.getContourIndex(zc);

      if (index < 0) return null;

      var palette = this.GetPalette();

      var theColor = Math.floor((index+0.99)*palette.length/(this.fContour.length-1));
      if (theColor > palette.length-1) theColor = palette.length-1;
      return asindx ? theColor : palette[theColor];
   }

   THistPainter.prototype.GetPalette = function(force) {
      if (!this.fPalette || force)
         this.fPalette = JSROOT.Painter.GetColorPalette(this.options.Palette);
      return this.fPalette;
   }

   THistPainter.prototype.FillPaletteMenu = function(menu) {

      var curr = this.options.Palette;
      if ((curr===null) || (curr===0)) curr = JSROOT.gStyle.Palette;

      function change(arg) {
         this.options.Palette = parseInt(arg);
         this.GetPalette(true);
         this.Redraw(); // redraw histogram
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
      add(57, "Bird", (curr>112));
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

      if (pal === null) {
         pal = JSROOT.Create('TPave');

         JSROOT.extend(pal, { _typename: "TPaletteAxis", fName: "TPave", fH: null, fAxis: null,
                               fX1NDC: 0.91, fX2NDC: 0.95, fY1NDC: 0.1, fY2NDC: 0.9, fInit: 1 } );

         pal.fAxis = JSROOT.Create('TGaxis');

         // set values from base classes

         JSROOT.extend(pal.fAxis, { fTitle: this.GetObject().fZaxis.fTitle,
                                    fLineColor: 1, fLineSyle: 1, fLineWidth: 1,
                                    fTextAngle: 0, fTextSize: 0.04, fTextAlign: 11, fTextColor: 1, fTextFont: 42 });

         // place colz in the beginning, that stat box is always drawn on the top
         this.AddFunction(pal, true);

         can_move = true;
      }

      var frame_painter = this.frame_painter();

      // keep palette width
      if (can_move) {
         pal.fX2NDC = frame_painter.fX2NDC + 0.01 + (pal.fX2NDC - pal.fX1NDC);
         pal.fX1NDC = frame_painter.fX2NDC + 0.01;
         pal.fY1NDC = frame_painter.fY1NDC;
         pal.fY2NDC = frame_painter.fY2NDC;
      }

      var arg = "";
      if (postpone_draw) arg+=";postpone";
      if (can_move && !this.do_redraw_palette) arg+=";canmove"

      if (pal_painter === null) {
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

      // make dummy redraw, palette will be updated only from histogram painter
      pal_painter.Redraw = function() {};

      if ((pal.fX1NDC-0.005 < frame_painter.fX2NDC) && !this.do_redraw_palette && can_move) {

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
      if (this.options.Zscale > 0) {
         this.options.Zscale = 0;
      } else {
         this.options.Zscale = 1;
      }

      this.DrawColorPalette(this.options.Zscale > 0, false, true);
   }

   THistPainter.prototype.PrepareColorDraw = function(args) {

      if (!args) args = { rounding: true, extra: 0, middle: 0 };

      if (args.extra === undefined) args.extra = 0;
      if (args.middle === undefined) args.middle = 0;

      var histo = this.GetObject(),
          pad = this.root_pad(),
          pmain = this.main_painter(),
          hdim = this.Dimension(),
          i, j, x, y, binz, binarea,
          res = {
             i1: this.GetSelectIndex("x", "left", 0 - args.extra),
             i2: this.GetSelectIndex("x", "right", 1 + args.extra),
             j1: (hdim===1) ? 0 : this.GetSelectIndex("y", "left", 0 - args.extra),
             j2: (hdim===1) ? 1 : this.GetSelectIndex("y", "right", 1 + args.extra),
             min: 0, max: 0, sumz: 0
          };
      res.grx = new Float32Array(res.i2+1);
      res.gry = new Float32Array(res.j2+1);

      if (args.pixel_density) args.rounding = true;

       // calculate graphical coordinates in advance
      for (i = res.i1; i <= res.i2; ++i) {
         x = this.GetBinX(i + args.middle);
         if (pmain.logx && (x <= 0)) { res.i1 = i+1; continue; }
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
         y = this.GetBinY(j + args.middle);
         if (pmain.logy && (y <= 0)) { res.j1 = j+1; continue; }
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

   // ======= TH1 painter================================================

   function TH1Painter(histo) {
      THistPainter.call(this, histo);
   }

   TH1Painter.prototype = Object.create(THistPainter.prototype);

   TH1Painter.prototype.ScanContent = function(when_axis_changed) {
      // if when_axis_changed === true specified, content will be scanned after axis zoom changed

      if (!this.nbinsx && when_axis_changed) when_axis_changed = false;

      if (!when_axis_changed) {
         this.nbinsx = this.histo.fXaxis.fNbins;
         this.nbinsy = 0;
         this.CreateAxisFuncs(false);
      }

      var left = this.GetSelectIndex("x", "left"),
          right = this.GetSelectIndex("x", "right");

      if (when_axis_changed) {
         if ((left === this.scan_xleft) && (right === this.scan_xright)) return;
      }

      this.scan_xleft = left;
      this.scan_xright = right;

      var hmin = 0, hmin_nz = 0, hmax = 0, hsum = 0, first = true,
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

      if (this.draw_content) {
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
      }

      hmin = hmax = null;
      var set_zoom = false;

      if (this.options.minimum !== -1111) {
         hmin = this.options.minimum;
         if (hmin < this.ymin)
            this.ymin = hmin;
         else
            set_zoom = true;
      }

      if (this.options.maximum !== -1111) {
         hmax = this.options.maximum;
         if (hmax > this.ymax)
            this.ymax = hmax;
         else
            set_zoom = true;
      }

      if (set_zoom && this.draw_content) {
         this.zoom_ymin = (hmin === null) ? this.ymin : hmin;
         this.zoom_ymax = (hmax === null) ? this.ymax : hmax;
      }

      // If no any draw options specified, do not try draw histogram
      if (!this.options.Bar && !this.options.Hist && !this.options.Line &&
          !this.options.Error && !this.options.Same && !this.options.Lego && !this.options.Text) {
         this.draw_content = false;
      }
      if (this.options.Axis > 0) { // Paint histogram axis only
         this.draw_content = false;
      }
   }

   TH1Painter.prototype.CountStat = function(cond) {
      var profile = this.IsTProfile(),
          left = this.GetSelectIndex("x", "left"),
          right = this.GetSelectIndex("x", "right"),
          stat_sumw = 0, stat_sumwx = 0, stat_sumwx2 = 0, stat_sumwy = 0, stat_sumwy2 = 0,
          i, xx = 0, w = 0, xmax = null, wmax = null,
          res = { meanx: 0, meany: 0, rmsx: 0, rmsy: 0, integral: 0, entries: this.stat_entries, xmax:0, wmax:0 };

      for (i = left; i < right; ++i) {
         xx = this.GetBinX(i+0.5);

         if (cond && !cond(xx)) continue;

         if (profile) {
            w = this.histo.fBinEntries[i + 1];
            stat_sumwy += this.histo.fArray[i + 1];
            stat_sumwy2 += this.histo.fSumw2[i + 1];
         } else {
            w = this.histo.getBinContent(i + 1);
         }

         if ((xmax===null) || (w>wmax)) { xmax = xx; wmax = w; }

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

   TH1Painter.prototype.FillStatistic = function(stat, dostat, dofit) {
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

         if (print_mean > 0)
            pave.AddText("Mean = " + stat.Format(data.meanx));

         if (print_rms > 0)
            pave.AddText("Std Dev = " + stat.Format(data.rmsx));

         if (print_under > 0)
            pave.AddText("Underflow = " + stat.Format((this.histo.fArray.length > 0) ? this.histo.fArray[0] : 0,"entries"));

         if (print_over > 0)
            pave.AddText("Overflow = " + stat.Format((this.histo.fArray.length > 0) ? this.histo.fArray[this.histo.fArray.length - 1] : 0,"entries"));

         if (print_integral > 0)
            pave.AddText("Integral = " + stat.Format(data.integral,"entries"));

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

   TH1Painter.prototype.DrawBars = function(width, height) {

      this.RecreateDrawG(false, "main_layer");

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          pmain = this.main_painter(),
          pad = this.root_pad(),
          pthis = this,
          i, x1, x2, grx1, grx2, y, gry1, gry2, w,
          bars = "", barsl = "", barsr = "",
          side = (this.options.Bar > 10) ? this.options.Bar % 10 : 0;

      if (side>4) side = 4;
      gry2 = pmain.swap_xy ? 0 : height;
      if ((this.options.BaseLine !== false) && !isNaN(this.options.BaseLine))
         if (this.options.BaseLine >= pmain.scale_ymin)
            gry2 = Math.round(pmain.gry(this.options.BaseLine));

      for (i = left; i < right; ++i) {
         x1 = this.GetBinX(i);
         x2 = this.GetBinX(i+1);

         if (pmain.logx && (x2 <= 0)) continue;

         grx1 = Math.round(pmain.grx(x1));
         grx2 = Math.round(pmain.grx(x2));

         y = this.histo.getBinContent(i+1);
         if (pmain.logy && (y < pmain.scale_ymin)) continue;
         gry1 = Math.round(pmain.gry(y));

         w = grx2 - grx1;
         grx1 += Math.round(this.histo.fBarOffset/1000*w);
         w = Math.round(this.histo.fBarWidth/1000*w);

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

   TH1Painter.prototype.DrawFilledErrors = function(width, height) {
      this.RecreateDrawG(false, "main_layer");

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          pmain = this.main_painter(),
          i, x, grx, y, yerr, gry1, gry2,
          bins1 = [], bins2 = [];

      for (i = left; i < right; ++i) {
         x = this.GetBinX(i+0.5);
         if (pmain.logx && (x <= 0)) continue;
         grx = Math.round(pmain.grx(x));

         y = this.histo.getBinContent(i+1);
         yerr = this.histo.getBinError(i+1);
         if (pmain.logy && (y-yerr < pmain.scale_ymin)) continue;

         gry1 = Math.round(pmain.gry(y + yerr));
         gry2 = Math.round(pmain.gry(y - yerr));

         bins1.push({grx:grx, gry: gry1});
         bins2.unshift({grx:grx, gry: gry2});
      }

      var kind = (this.options.Error == 14) ? "bezier" : "line";

      var path1 = JSROOT.Painter.BuildSvgPath(kind, bins1),
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

      if (this.options.Bar > 0)
         return this.DrawBars(width, height);

      if ((this.options.Error == 13) || (this.options.Error == 14))
         return this.DrawFilledErrors(width, height);

      this.RecreateDrawG(false, "main_layer");

      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 2),
          pmain = this.main_painter(),
          pad = this.root_pad(),
          pthis = this,
          res = "", lastbin = false,
          startx, currx, curry, x, grx, y, gry, curry_min, curry_max, prevy, prevx, i, besti,
          exclude_zero = !this.options.Zero,
          show_errors = (this.options.Error > 0),
          show_markers = (this.options.Mark > 0),
          show_line = (this.options.Line > 0),
          show_text = (this.options.Text > 0),
          path_fill = null, path_err = null, path_marker = null, path_line = null,
          endx = "", endy = "", dend = 0, my, yerr1, yerr2, bincont, binerr, mx1, mx2, midx,
          mpath = "", text_col, text_angle, text_size;

      if (show_errors && !show_markers && (this.histo.fMarkerStyle > 1))
         show_markers = true;

      if (this.options.Error == 12) {
         if (this.fillatt.color=='none') show_markers = true;
                                    else path_fill = "";
      } else
      if (this.options.Error > 0) path_err = "";

      if (show_line) path_line = "";

      if (show_markers) {
         // draw markers also when e2 option was specified
         if (!this.markeratt)
            this.markeratt = JSROOT.Painter.createAttMarker(this.histo, this.options.Mark - 20);
         if (this.markeratt.size > 0) {
            // simply use relative move from point, can optimize in the future
            path_marker = "";
            this.markeratt.reset_pos();
         } else {
            show_markers = false;
         }
      }

      if (show_text) {
         text_col = JSROOT.Painter.root_colors[this.histo.fMarkerColor];
         text_angle = (this.options.Text>1000) ? this.options.Text % 1000 : 0;
         text_size = 20;

         if ((this.histo.fMarkerSize!==1) && (text_angle!==0))
            text_size = 0.02*height*this.histo.fMarkerSize;

         if ((text_angle === 0) && (this.options.Text<1000)) {
             var space = width / (right - left + 1);
             if (space < 3 * text_size) {
                text_angle = 90;
                text_size = Math.round(space*0.7);
             }
         }

         this.StartTextDrawing(42, text_size, this.draw_g, text_size);
      }

      // if there are too many points, exclude many vertical drawings at the same X position
      // instead define min and max value and made min-max drawing
      var use_minmax = ((right-left) > 3*width);

      if (this.options.Error == 11) {
         var lw = this.lineatt.width + JSROOT.gStyle.fEndErrorSize;
         endx = "m0," + lw + "v-" + 2*lw + "m0," + lw;
         endy = "m" + lw + ",0h-" + 2*lw + "m" + lw + ",0";
         dend = Math.floor((this.lineatt.width-1)/2);
      }

      var draw_markers = show_errors || show_markers;

      if (draw_markers || show_text || show_line) use_minmax = true;

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

               if (draw_markers || show_text || show_line) {
                  bincont = this.histo.getBinContent(besti+1);
                  if (!exclude_zero || (bincont!==0)) {
                     mx1 = Math.round(pmain.grx(this.GetBinX(besti)));
                     mx2 = Math.round(pmain.grx(this.GetBinX(besti+1)));
                     midx = Math.round((mx1+mx2)/2);
                     my = Math.round(pmain.gry(bincont));
                     yerr1 = yerr2 = 20;
                     if (show_errors) {
                        binerr = this.histo.getBinError(besti+1);
                        yerr1 = Math.round(my - pmain.gry(bincont + binerr)); // up
                        yerr2 = Math.round(pmain.gry(bincont - binerr) - my); // down
                     }

                     if (show_text) {
                        var cont = bincont;
                        if ((this.options.Text>=2000) && (this.options.Text < 3000) &&
                             this.IsTProfile() && this.histo.fBinEntries)
                           cont = this.histo.fBinEntries[besti+1];

                        var posx = Math.round(mx1 + (mx2-mx1)*0.1),
                            posy = Math.round(my-2-text_size),
                            sizex = Math.round((mx2-mx1)*0.8),
                            sizey = text_size,
                            lbl = Math.round(cont),
                            talign = 22;

                        if (lbl === cont)
                           lbl = cont.toString();
                        else
                           lbl = JSROOT.FFormat(cont, JSROOT.gStyle.fPaintTextFormat);

                        if (text_angle!==0) {
                           posx = midx;
                           posy = Math.round(my - 2 - text_size/5);
                           sizex = 0;
                           sizey = text_angle-360;
                           talign = 12;
                        }

                        if (cont!==0)
                           this.DrawText(talign, posx, posy, sizex, sizey, lbl, text_col, 0);
                     }

                     if (show_line && (path_line !== null))
                        path_line += ((path_line.length===0) ? "M" : "L") + midx + "," + my;

                     if (draw_markers) {
                        if ((my >= -yerr1) && (my <= height + yerr2)) {
                           if (path_fill !== null)
                              path_fill += "M" + mx1 +","+(my-yerr1) +
                                           "h" + (mx2-mx1) + "v" + (yerr1+yerr2+1) + "h-" + (mx2-mx1) + "z";
                           if (path_marker !== null)
                              path_marker += this.markeratt.create(midx, my);
                           if (path_err !== null) {
                              if (this.options.errorX > 0) {
                                 var mmx1 = Math.round(midx - (mx2-mx1)*this.options.errorX),
                                     mmx2 = Math.round(midx + (mx2-mx1)*this.options.errorX);
                                 path_err += "M" + (mmx1+dend) +","+ my + endx + "h" + (mmx2-mmx1-2*dend) + endx;
                              }
                              path_err += "M" + midx +"," + (my-yerr1+dend) + endy + "v" + (yerr1+yerr2-2*dend) + endy;
                           }
                        }
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

      var close_path = "";

      if (!this.fillatt.empty()) {
         var h0 = (height+3);
         if ((this.hmin>=0) && (pmain.gry(0) < height)) h0 = Math.round(pmain.gry(0));
         close_path = "L"+currx+","+h0 + "L"+startx+","+h0 + "Z";
         if (res.length>0) res += close_path;
      }

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
                     .attr("d", this.options.Line===2 ? (path_line + close_path) : res)
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

      } else
      if ((res.length > 0) && (this.options.Hist>0)) {
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
          pmain = this.main_painter(),
          histo = this.GetObject(),
          x1 = this.GetBinX(bin),
          x2 = this.GetBinX(bin+1),
          cont = histo.getBinContent(bin+1);

      if (name.length>0) tips.push(name);

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

         if (histo['$baseh']) cont -= histo['$baseh'].getBinContent(bin+1);

         if (cont === Math.round(cont))
            tips.push("entries = " + cont);
         else
            tips.push("entries = " + JSROOT.FFormat(cont, JSROOT.gStyle.fStatFormat));
      }

      return tips;
   }

   TH1Painter.prototype.ProcessTooltip = function(pnt) {
      if ((pnt === null) || !this.draw_content || (this.options.Lego > 0) || (this.options.Surf > 0)) {
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
          grx1, midx, grx2, gry1, midy, gry2, gapx = 2,
          left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 2),
          l = left, r = right;

      function GetBinGrX(i) {
         var xx = painter.GetBinX(i);
         return (pmain.logx && (xx<=0)) ? null : pmain.grx(xx);
      }

      function GetBinGrY(i) {
         var yy = painter.histo.getBinContent(i + 1);
         if (pmain.logy && (yy < painter.scale_ymin))
            return pmain.swap_xy ? -1000 : 10*height;
         return Math.round(pmain.gry(yy));
      }

      var pnt_x = pmain.swap_xy ? pnt.y : pnt.x,
          pnt_y = pmain.swap_xy ? pnt.x : pnt.y;

      while (l < r-1) {
         var m = Math.round((l+r)*0.5);

         var xx = GetBinGrX(m);
         if ((xx === null) || (xx < pnt_x - 0.5)) {
            if (pmain.swap_xy) r = m; else l = m;
         } else
         if (xx > pnt_x + 0.5) {
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

      if (this.options.Bar > 0) {
         var w = grx2 - grx1;
         grx1 += Math.round(this.histo.fBarOffset/1000*w);
         grx2 = grx1 + Math.round(this.histo.fBarWidth/1000*w);
      }

      if (grx1 > grx2) { var d = grx1; grx1 = grx2; grx2 = d; }

      midx = Math.round((grx1+grx2)/2);

      midy = gry1 = gry2 = GetBinGrY(findbin);

      if (this.options.Bar > 0) {
         show_rect = true;

         gapx = 0;

         gry1 = Math.round(pmain.gry(((this.options.BaseLine!==false) && (this.options.BaseLine > pmain.scale_ymin)) ? this.options.BaseLine : pmain.scale_ymin));

         if (gry1 > gry2) { var d = gry1; gry1 = gry2; gry2 = d; }

         if (!pnt.touch && (pnt.nproc === 1))
            if ((pnt_y<gry1) || (pnt_y>gry2)) findbin = null;
      } else
      if ((this.options.Error > 0) || (this.options.Mark > 0) || (this.options.Line > 0))  {

         show_rect = true;

         var msize = 3;
         if (this.markeratt) msize = Math.max(msize, 2+Math.round(this.markeratt.size * 4));

         if (this.options.Error > 0) {
            var cont = this.histo.getBinContent(findbin+1),
                binerr = this.histo.getBinError(findbin+1);

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
         if ((findbin === left) && (grx1 > pnt_x + gapx))  findbin = null; else
         if ((findbin === right-1) && (grx2 < pnt_x - gapx)) findbin = null; else
         // if bars option used check that bar is not match
         if ((pnt_x < grx1 - gapx) || (pnt_x > grx2 + gapx)) findbin = null; else
         // exclude empty bin if empty bins suppressed
         if (!this.options.Zero && (this.histo.getBinContent(findbin+1)===0)) findbin = null;
      }

      var ttrect = this.draw_g.select(".tooltip_bin");

      if ((findbin === null) || ((gry2 <= 0) || (gry1 >= height))) {
         ttrect.remove();
         this.ProvideUserTooltip(null);
         return null;
      }

      var res = { name: this.histo.fName, title: this.histo.fTitle,
                  x: midx, y: midy,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.color : 'blue',
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

      if (this.IsUserTooltipCallback() && res.changed) {
         this.ProvideUserTooltip({ obj: this.histo,  name: this.histo.fName,
                                   bin: findbin, cont: this.histo.getBinContent(findbin+1),
                                   grx: midx, gry: midy });
      }

      return res;
   }


   TH1Painter.prototype.FillHistContextMenu = function(menu) {

      menu.add("Auto zoom-in", this.AutoZoom);

      var sett = JSROOT.getDrawSettings("ROOT." + this.GetObject()._typename, 'nosame');

      menu.addDrawMenu("Draw with", sett.opts, function(arg) {
         if (arg==='inspect')
            return JSROOT.draw(this.divid, this.GetObject(), arg);

         this.options = this.DecodeOptions(arg, true);

         // redraw all objects
         this.RedrawPad();
      });
   }

   TH1Painter.prototype.AutoZoom = function() {
      var left = this.GetSelectIndex("x", "left", -1),
          right = this.GetSelectIndex("x", "right", 1),
          dist = right - left;

      if (dist == 0) return;

      // first find minimum
      var min = this.histo.getBinContent(left + 1);
      for (var indx = left; indx < right; ++indx)
         min = Math.min(min, this.histo.getBinContent(indx+1));
      if (min > 0) return; // if all points positive, no chance for autoscale

      while ((left < right) && (this.histo.getBinContent(left+1) <= min)) ++left;
      while ((left < right) && (this.histo.getBinContent(right) <= min)) --right;

      // if singular bin
      if ((left === right-1) && (left > 2) && (right < this.nbinsx-2)) {
         --left; ++right;
      }

      if ((right - left < dist) && (left < right))
         this.Zoom(this.GetBinX(left), this.GetBinX(right));
   }

   TH1Painter.prototype.CanZoomIn = function(axis,min,max) {
      if ((axis=="x") && (this.GetIndexX(max,0.5) - this.GetIndexX(min,0) > 1)) return true;

      if ((axis=="y") && (Math.abs(max-min) > Math.abs(this.ymax-this.ymin)*1e-6)) return true;

      // check if it makes sense to zoom inside specified axis range
      return false;
   }

   TH1Painter.prototype.CallDrawFunc = function(callback, resize) {
      var is3d = (this.options.Lego > 0) ? true : false,
          main = this.main_painter();

      if ((main !== this) && (main.mode3d !== is3d)) {
         // that to do with that case
         is3d = main.mode3d;
         this.options.Lego = main.options.Lego;
      }

      var funcname = is3d ? "Draw3D" : "Draw2D";

      this[funcname](callback, resize);
   }


   TH1Painter.prototype.Draw2D = function(call_back) {
      if (typeof this.Create3DScene === 'function')
         this.Create3DScene(-1);

      this.mode3d = false;

      this.ScanContent(true);

      this.CreateXY();

      if (typeof this.DrawColorPalette === 'function')
         this.DrawColorPalette(false);

      this.DrawAxes();
      this.DrawGrids();
      this.DrawBins();
      this.DrawTitle();
      this.UpdateStatWebCanvas();
      this.AddInteractive();
      JSROOT.CallBack(call_back);
   }

   TH1Painter.prototype.Draw3D = function(call_back) {
      this.mode3d = true;
      JSROOT.AssertPrerequisites('hist3d', function() {
         this.Draw3D(call_back);
      }.bind(this));
   }

   TH1Painter.prototype.Redraw = function(resize) {
      this.CallDrawFunc(null, resize);
   }

   JSROOT.Painter.drawHistogram1D = function(divid, histo, opt) {
      // create painter and add it to canvas
      var painter = new TH1Painter(histo);

      painter.SetDivId(divid, 1);

      // here we deciding how histogram will look like and how will be shown
      painter.options = painter.DecodeOptions(opt);

      if (!painter.options.Lego) painter.CheckPadRange();

      painter.ScanContent();

      if (JSROOT.gStyle.AutoStat && (painter.create_canvas || histo.$snapid))
         painter.CreateStat(histo.$custom_stat);

      painter.CallDrawFunc(function() {
         painter.DrawNextFunction(0, function() {
            if ((painter.options.Lego === 0) && painter.options.AutoZoom) painter.AutoZoom();
            painter.FillToolbar();
            painter.DrawingReady();
         });
      });

      return painter;
   }

   // ==================== painter for TH2 histograms ==============================

   function TH2Painter(histo) {
      THistPainter.call(this, histo);
      this.fContour = null; // contour levels
      this.fCustomContour = false; // are this user-defined levels (can be irregular)
      this.fPalette = null;
   }

   TH2Painter.prototype = Object.create(THistPainter.prototype);

   TH2Painter.prototype.Cleanup = function() {
      delete this.fCustomContour;
      delete this.tt_handle;

      THistPainter.prototype.Cleanup.call(this);
   }

   TH2Painter.prototype.FillHistContextMenu = function(menu) {
      // painter automatically bind to menu callbacks
      menu.add("Auto zoom-in", this.AutoZoom);

      var sett = JSROOT.getDrawSettings("ROOT." + this.GetObject()._typename, 'nosame');

      menu.addDrawMenu("Draw with", sett.opts, function(arg) {
         if (arg==='inspect')
            return JSROOT.draw(this.divid, this.GetObject(), arg);
         this.options = this.DecodeOptions(arg);
         this.RedrawPad();
      });

      if (this.options.Color > 0)
         this.FillPaletteMenu(menu);
   }

   TH2Painter.prototype.ButtonClick = function(funcname) {
      if (THistPainter.prototype.ButtonClick.call(this, funcname)) return true;

      if (this !== this.main_painter()) return false;

      switch(funcname) {
         case "ToggleColor": this.ToggleColor(); break;
         case "ToggleColorZ":
            if (this.options.Lego === 12 || this.options.Lego === 14 ||
                this.options.Color > 0 || this.options.Contour > 0 ||
                this.options.Surf === 11 || this.options.Surf === 12) this.ToggleColz();
            break;
         case "Toggle3D":
            if (this.options.Surf > 0) {
               this.options.Surf = -this.options.Surf;
            } else
            if (this.options.Lego > 0) {
               this.options.Lego = 0;
            } else
            if (this.options.Surf < 0) {
               this.options.Surf = -this.options.Surf;
            } else {
               if ((this.nbinsx>=50) || (this.nbinsy>=50))
                  this.options.Lego = (this.options.Color > 0) ? 14 : 13;
               else
                  this.options.Lego = (this.options.Color > 0) ? 12 : 1;

               this.options.Zero = 0; // do not show zeros by default
            }

            this.RedrawPad();
            break;
         default: return false;
      }

      // all methods here should not be processed further
      return true;
   }

   TH2Painter.prototype.FillToolbar = function() {
      THistPainter.prototype.FillToolbar.call(this);

      var pp = this.pad_painter(true);
      if (pp===null) return;

      if (!this.IsTH2Poly())
         pp.AddButton(JSROOT.ToolbarIcons.th2color, "Toggle color", "ToggleColor");
      pp.AddButton(JSROOT.ToolbarIcons.th2colorz, "Toggle color palette", "ToggleColorZ");
      pp.AddButton(JSROOT.ToolbarIcons.th2draw3d, "Toggle 3D mode", "Toggle3D");
   }

   TH2Painter.prototype.ToggleColor = function() {

      var toggle = true;

      if (this.options.Lego > 0) { this.options.Lego = 0; toggle = false; } else
      if (this.options.Surf > 0) { this.options.Surf = -this.options.Surf; toggle = false; }

      if (this.options.Color == 0) {
         this.options.Color = ('LastColor' in this.options) ?  this.options.LastColor : 1;
      } else
      if (toggle) {
         this.options.LastColor = this.options.Color;
         this.options.Color = 0;
      }

      this._can_move_colz = true; // indicate that next redraw can move Z scale

      this.Redraw();

      // this.DrawColorPalette((this.options.Color > 0) && (this.options.Zscale > 0));
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
         xmin = this.GetBinX(ileft);
         xmax = this.GetBinX(iright);
         isany = true;
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = this.GetBinY(jleft);
         ymax = this.GetBinY(jright);
         isany = true;
      }

      if (isany) this.Zoom(xmin, xmax, ymin, ymax);
   }

   TH2Painter.prototype.ScanContent = function(when_axis_changed) {

      // no need to rescan histogram while result does not depend from axis selection
      if (when_axis_changed && this.nbinsx && this.nbinsy) return;

      var i,j,histo = this.GetObject();

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
         // used to enable/disable stat box
         this.draw_content = this.gmaxbin > 0;
      }
   }

   TH2Painter.prototype.CountStat = function(cond) {
      var histo = this.GetObject(),
          stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0,
          stat_sumx2 = 0, stat_sumy2 = 0, stat_sumxy = 0,
          xside, yside, xx, yy, zz,
          res = { entries: 0, integral: 0, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix: [0,0,0,0,0,0,0,0,0], xmax: 0, ymax:0, wmax: null };

      if (this.IsTH2Poly()) {

         var len = histo.fBins.arr.length, i, bin, n, gr, ngr, numgraphs, numpoints,
             pmain = this.main_painter();

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

            if ((cond!=null) && !cond(xx,yy)) continue;

            if ((res.wmax==null) || (zz>res.wmax)) { res.wmax = zz; res.xmax = xx; res.ymax = yy; }

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
            xx = this.GetBinX(xi - 0.5);

            for (yi = 0; yi <= this.nbinsy + 1; ++yi) {
               yside = (yi <= yleft) ? 0 : (yi > yright ? 2 : 1);
               yy = this.GetBinY(yi - 0.5);

               zz = histo.getBinContent(xi, yi);

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
               stat_sumxy += xx * yy * zz;
            }
         }
      }

      if (!this.IsAxisZoomed("x") && !this.IsAxisZoomed("y") && (histo.fTsumw > 0)) {
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
      if (this.GetObject() === null) return false;

      var pave = stat.GetObject(),
          data = this.CountStat(),
          print_name = Math.floor(dostat % 10),
          print_entries = Math.floor(dostat / 10) % 10,
          print_mean = Math.floor(dostat / 100) % 10,
          print_rms = Math.floor(dostat / 1000) % 10,
          print_under = Math.floor(dostat / 10000) % 10,
          print_over = Math.floor(dostat / 100000) % 10,
          print_integral = Math.floor(dostat / 1000000) % 10,
          print_skew = Math.floor(dostat / 10000000) % 10,
          print_kurt = Math.floor(dostat / 100000000) % 10;

      if (print_name > 0)
         pave.AddText(this.GetObject().fName);

      if (print_entries > 0)
         pave.AddText("Entries = " + stat.Format(data.entries,"entries"));

      if (print_mean > 0) {
         pave.AddText("Mean x = " + stat.Format(data.meanx));
         pave.AddText("Mean y = " + stat.Format(data.meany));
      }

      if (print_rms > 0) {
         pave.AddText("Std Dev x = " + stat.Format(data.rmsx));
         pave.AddText("Std Dev y = " + stat.Format(data.rmsy));
      }

      if (print_integral > 0)
         pave.AddText("Integral = " + stat.Format(data.matrix[4],"entries"));

      if (print_skew > 0) {
         pave.AddText("Skewness x = <undef>");
         pave.AddText("Skewness y = <undef>");
      }

      if (print_kurt > 0)
         pave.AddText("Kurt = <undef>");

      if ((print_under > 0) || (print_over > 0)) {
         var m = data.matrix;

         pave.AddText("" + m[6].toFixed(0) + " | " + m[7].toFixed(0) + " | "  + m[7].toFixed(0));
         pave.AddText("" + m[3].toFixed(0) + " | " + m[4].toFixed(0) + " | "  + m[5].toFixed(0));
         pave.AddText("" + m[0].toFixed(0) + " | " + m[1].toFixed(0) + " | "  + m[2].toFixed(0));
      }

      // adjust the size of the stats box wrt the number of lines
      var nlines = pave.fLines.arr.length,
          stath = nlines * JSROOT.gStyle.StatFontSize;
      if (stath <= 0 || 3 == (JSROOT.gStyle.StatFont % 10)) {
         stath = 0.25 * nlines * JSROOT.gStyle.StatH;
         pave.fY1NDC = 0.93 - stath;
         pave.fY2NDC = 0.93;
      }

      return true;
   }

   TH2Painter.prototype.DrawBinsColor = function(w,h) {
      var histo = this.GetObject(),
          handle = this.PrepareColorDraw(),
          colPaths = [], currx = [], curry = [],
          colindx, cmd1, cmd2, i, j, binz;

      // now start build
      for (i = handle.i1; i < handle.i2; ++i) {
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i + 1, j + 1);
            colindx = this.getValueColor(binz, true);
            if (binz===0) {
               if (this.options.Color===11) continue;
               if ((colindx === null) && this._show_empty_bins) colindx = 0;
            }
            if (colindx === null) continue;

            cmd1 = "M"+handle.grx[i]+","+handle.gry[j+1];
            if (colPaths[colindx] === undefined) {
               colPaths[colindx] = cmd1;
            } else{
               cmd2 = "m" + (handle.grx[i]-currx[colindx]) + "," + (handle.gry[j+1]-curry[colindx]);
               colPaths[colindx] += (cmd2.length < cmd1.length) ? cmd2 : cmd1;
            }

            currx[colindx] = handle.grx[i];
            curry[colindx] = handle.gry[j+1];

            colPaths[colindx] += "v" + (handle.gry[j] - handle.gry[j+1]) +
                                 "h" + (handle.grx[i+1] - handle.grx[i]) +
                                 "v" + (handle.gry[j+1] - handle.gry[j]) + "z";
         }
      }

      for (colindx=0;colindx<colPaths.length;++colindx)
        if (colPaths[colindx] !== undefined)
           this.draw_g
               .append("svg:path")
               .attr("palette-index", colindx)
               .attr("fill", this.fPalette[colindx])
               .attr("d", colPaths[colindx]);

      return handle;
   }

   TH2Painter.prototype.BuildContour = function(handle, levels, palette, contour_func) {
      var histo = this.GetObject(),
          kMAXCONTOUR = 404,
          kMAXCOUNT = 400,
      // arguments used in he PaintContourLine
          xarr = new Float32Array(2*kMAXCONTOUR),
          yarr = new Float32Array(2*kMAXCONTOUR),
          itarr = new Int32Array(2*kMAXCONTOUR),
          lj = 0, ipoly, poly, polys = [], np, npmax = 0,
          x = new Float32Array(4),
          y = new Float32Array(4),
          zc = new Float32Array(4),
          ir =  new Int32Array(4),
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

      for (j = handle.j1; j < handle.j2-1; ++j) {

         y[1] = y[0] = (handle.gry[j] + handle.gry[j+1])/2;
         y[3] = y[2] = (handle.gry[j+1] + handle.gry[j+2])/2;

         for (i = handle.i1; i < handle.i2-1; ++i) {

            zc[0] = histo.getBinContent(i+1, j+1);
            zc[1] = histo.getBinContent(i+2, j+1);
            zc[2] = histo.getBinContent(i+2, j+2);
            zc[3] = histo.getBinContent(i+1, j+2);

            for (k=0;k<4;k++)
               ir[k] = BinarySearch(zc[k]);

            if ((ir[0] !== ir[1]) || (ir[1] !== ir[2]) || (ir[2] !== ir[3]) || (ir[3] !== ir[0])) {
               x[3] = x[0] = (handle.grx[i] + handle.grx[i+1])/2;
               x[2] = x[1] = (handle.grx[i+1] + handle.grx[i+2])/2;

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

      var xp = new Int32Array(2*npmax),
          yp = new Int32Array(2*npmax);

      for (k=0;k<levels.length;++k) {

         ipoly = polysort[k];
         poly = polys[ipoly];
         if (!poly) continue;

         var colindx = Math.floor((ipoly+0.99)*palette.length/(levels.length-1));
         if (colindx > palette.length-1) colindx = palette.length-1;

         var xx = poly.fX, yy = poly.fY, np = poly.fLastPoint+1,
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
                  if (xx[i] === xp[iplus] && yy[i] === yp[iplus]) {
                     iplus++;
                     xp[iplus] = xx[i+1]; yp[iplus]  = yy[i+1];
                     xx[i] = xx[i+1] = xmin;
                     yy[i] = yy[i+1] = ymin;
                     nadd++;
                  }
                  if (xx[i+1] === xp[iminus] && yy[i+1] === yp[iminus]) {
                     iminus--;
                     xp[iminus] = xx[i];   yp[iminus]  = yy[i];
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
      var handle = this.PrepareColorDraw({ rounding: false, extra: 100 });

      // initialize contour
      this.getContourIndex(0);

      // get levels
      var levels = this.fContour,
          palette = this.GetPalette(),
          painter = this;

      if (this.options.Contour===14)
         this.draw_g
             .append("svg:rect")
             .attr("x", 0).attr("y", 0)
             .attr("width", frame_w).attr("height", frame_h)
             .style("fill", palette[Math.floor(0.99*palette.length/(levels.length-1))]);

      this.BuildContour(handle, levels, palette,
         function(colindx,xp,yp,iminus,iplus) {
            var icol = palette[colindx],
                fillcolor = icol, lineatt = null;

            switch(painter.options.Contour) {
               case 1: break;
               case 11: fillcolor = 'none'; lineatt = JSROOT.Painter.createAttLine(icol); break;
               case 12: fillcolor = 'none'; lineatt = JSROOT.Painter.createAttLine({fLineColor:1, fLineStyle: (colindx%5 + 1), fLineWidth: 1 }); break;
               case 13: fillcolor = 'none'; lineatt = painter.lineatt; break;
               case 14: break;
            }

            var cmd = "M" + xp[iminus] + "," + yp[iminus];
            for (i = iminus+1;i<=iplus;++i)
               cmd +=  "l" + (xp[i] - xp[i-1]) + "," + (yp[i] - yp[i-1]);
            if (fillcolor !== 'none') cmd += "Z";

            var elem = painter.draw_g
                          .append("svg:path")
                          .attr("class","th2_contour")
                          .attr("d", cmd)
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

   TH2Painter.prototype.CreatePolyBin = function(pmain, bin) {
      var cmd = "", ngr, ngraphs = 1, gr = null;

      if (bin.fPoly._typename=='TMultiGraph')
         ngraphs = bin.fPoly.fGraphs.arr.length;
      else
         gr = bin.fPoly;

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
            if ((grx!==nextx) || (gry!==nexty)) {
               if (grx===nextx) cmd += "v" + (nexty - gry); else
                  if (gry===nexty) cmd += "h" + (nextx - grx); else
                     cmd += "l" + (nextx - grx) + "," + (nexty - gry);
            }
            grx = nextx; gry = nexty;
         }

         cmd += "z";
      }

      return cmd;
   }

   TH2Painter.prototype.DrawPolyBinsColor = function(w,h) {
      var histo = this.GetObject(),
          pmain = this.main_painter(),
          colPaths = [],
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
         colindx = this.getValueColor(bin.fContent, true);
         if (colindx === null) continue;
         if ((bin.fContent === 0) && (this.options.Color === 11)) continue;

         // check if bin outside visible range
         if ((bin.fXmin > pmain.scale_xmax) || (bin.fXmax < pmain.scale_xmin) ||
             (bin.fYmin > pmain.scale_ymax) || (bin.fYmax < pmain.scale_ymin)) continue;

         cmd = this.CreatePolyBin(pmain, bin);

         if (colPaths[colindx] === undefined)
            colPaths[colindx] = cmd;
         else
            colPaths[colindx] += cmd;
      }

      for (colindx=0;colindx<colPaths.length;++colindx)
         if (colPaths[colindx] !== undefined) {
            item = this.draw_g
                     .append("svg:path")
                     .attr("palette-index", colindx)
                     .attr("fill", this.fPalette[colindx])
                     .attr("d", colPaths[colindx])
            if (this.options.Line)
               item.call(this.lineatt.func);
         }

      return { poly: true };
   }

   TH2Painter.prototype.DrawBinsText = function(w, h, handle) {
      var histo = this.GetObject(),
          i,j,binz,colindx,binw,binh,lbl,posx,posy,sizex,sizey;

      if (handle===null) handle = this.PrepareColorDraw({ rounding: false });

      var text_col = JSROOT.Painter.root_colors[histo.fMarkerColor],
          text_angle = (this.options.Text > 1000) ? this.options.Text % 1000 : 0,
          text_g = this.draw_g.append("svg:g").attr("class","th2_text"),
          text_size = 20, text_offset = 0;

      if ((histo.fMarkerSize!==1) && (text_angle!==0))
         text_size = Math.round(0.02*h*histo.fMarkerSize);

      if (histo.fBarOffset!==0) text_offset = histo.fBarOffset*1e-3;

      this.StartTextDrawing(42, text_size, text_g, text_size);

      for (i = handle.i1; i < handle.i2; ++i)
         for (j = handle.j1; j < handle.j2; ++j) {
            binz = histo.getBinContent(i+1, j+1);
            if ((binz === 0) && !this._show_empty_bins) continue;

            binw = handle.grx[i+1] - handle.grx[i];
            binh = handle.gry[j] - handle.gry[j+1];

            if ((this.options.Text >= 2000) && (this.options.Text < 3000) &&
                 this.MatchObjectType('TProfile2D') && (typeof histo.getBinEntries=='function'))
                   binz = histo.getBinEntries(i+1, j+1);

            lbl = Math.round(binz);
            if (lbl === binz)
               lbl = binz.toString();
            else
               lbl = JSROOT.FFormat(binz, JSROOT.gStyle.fPaintTextFormat);

            if ((text_angle!==0) /*|| (histo.fMarkerSize!==1)*/) {
               posx = Math.round(handle.grx[i] + binw*0.5);
               posy = Math.round(handle.gry[j+1] + binh*(0.5 + text_offset));
               sizex = 0;
               sizey = text_angle-360;
            } else {
               posx = Math.round(handle.grx[i] + binw*0.1);
               posy = Math.round(handle.gry[j+1] + binh*(0.1 + text_offset));
               sizex = Math.round(binw*0.8);
               sizey = Math.round(binh*0.8);
            }

            this.DrawText(22, posx, posy, sizex, sizey, lbl, text_col, 0, text_g);
         }

      this.FinishTextDrawing(text_g, null);

      handle.hide_only_zeros = true; // text drawing suppress only zeros

      return handle;
   }

   TH2Painter.prototype.DrawBinsArrow = function(w, h) {
      var histo = this.GetObject(),
          i,j,binz,colindx,binw,binh,lbl, loop, dn = 1e-30, dx, dy, xc,yc,
          dxn,dyn,x1,x2,y1,y2, anr,si,co;

      var handle = this.PrepareColorDraw({ rounding: false });

      var scale_x  = (handle.grx[handle.i2] - handle.grx[handle.i1])/(handle.i2 - handle.i1 + 1-0.03)/2,
          scale_y  = (handle.gry[handle.j2] - handle.gry[handle.j1])/(handle.j2 - handle.j1 + 1-0.03)/2;

      var cmd = "";

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

            if ((binz<0) && (this.options.Box === 10))
               cross += "M"+xx+","+yy + "l"+ww+","+hh + "M"+(xx+ww)+","+yy + "l-"+ww+","+hh;

            if ((this.options.Box === 11) && (ww>5) && (hh>5)) {
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
        if ((this.options.Box === 11) || (this.fillatt.color !== 'none'))
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
      var histo = this.GetObject(),
          handle = this.PrepareColorDraw(),
          pad = this.root_pad(),
          pmain = this.main_painter(), // used for axis values conversions
          i, j, y, sum0, sum1, sum2, cont, center, counter, integral, w, pnt;

      // candle option coded into string, which comes after candle identifier
      // console.log('Draw candle plot with option', this.options.Candle);

      var bars = "", markers = "";

      // create attribute only when necessary
      if (!this.markeratt) {
         if (histo.fMarkerColor === 1) histo.fMarkerColor = histo.fLineColor;
         this.markeratt = JSROOT.Painter.createAttMarker(histo, 5);
      }

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
            if (counter/integral < 0.001 && (counter + cont)/integral >=0.001) pnt.whiskerm = this.GetBinY(j + 0.5); // Lower whisker
            if (counter/integral < 0.25 && (counter + cont)/integral >=0.25) pnt.m25y = this.GetBinY(j + 0.5); // Lower edge of box
            if (counter/integral < 0.5 && (counter + cont)/integral >=0.5) pnt.median = this.GetBinY(j + 0.5); //Median
            if (counter/integral < 0.75 && (counter + cont)/integral >=0.75) pnt.p25y = this.GetBinY(j + 0.5); //Upper edge of box
            if (counter/integral < 0.999 && (counter + cont)/integral >=0.999) pnt.whiskerp = this.GetBinY(j + 0.5); // Upper whisker
            counter += cont;
            y = this.GetBinY(j + 0.5); // center of y bin coordinate
            sum1 += cont*y;
         }
         if (counter > 0) {
            pnt.meany = sum1/counter;
         }
         pnt.iqr = pnt.p25y-pnt.m25y;

//       console.log('Whisker before ' + pnt.whiskerm + '/' + pnt.whiskerp);

         //Whiskers cannot exceed 1.5*iqr from box
         if ((pnt.m25y-1.5*pnt.iqr) > pnt.whsikerm)  {
            pnt.whiskerm = pnt.m25y-1.5*pnt.iqr;
         }
         if ((pnt.p25y+1.5*pnt.iqr) < pnt.whiskerp) {
            pnt.whiskerp = pnt.p25y+1.5*pnt.iqr;
         }
//       console.log('Whisker after ' + pnt.whiskerm + '/' + pnt.whiskerp);

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

//       console.log('Whisker-: '+ pnt.whiskerm + ' Whisker+:' + pnt.whiskerp);
         //estimate outliers
         for (j = 0; j < this.nbinsy; ++j) {
            cont = histo.getBinContent(i+1,j+1);
            if (cont > 0 && this.GetBinY(j + 0.5) < pnt.whiskerm) markers += this.markeratt.create(center, this.GetBinY(j + 0.5));
            if (cont > 0 && this.GetBinY(j + 0.5) > pnt.whiskerp) markers += this.markeratt.create(center, this.GetBinY(j + 0.5));
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

      if (scale*handle.sumz < 1e5) {
         // one can use direct drawing of scatter plot without any patterns

         if (!this.markeratt)
            this.markeratt = JSROOT.Painter.createAttMarker(histo);

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
                            Math.round(handle.grx[i] + cw * Math.random()),
                            Math.round(handle.gry[j+1] + ch * Math.random()));
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

      var layer = this.svg_frame().select('.main_layer');
      var defs = layer.select("defs");
      if (defs.empty() && (colPaths.length>0))
         defs = layer.insert("svg:defs",":first-child");

      if (!this.markeratt)
         this.markeratt = JSROOT.Painter.createAttMarker(histo);

      for (colindx=0;colindx<colPaths.length;++colindx)
        if ((colPaths[colindx] !== undefined) && (colindx<this.fContour.length)) {
           var pattern_class = "scatter_" + colindx;
           var pattern = defs.select('.'+pattern_class);
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
                 arrx[n] = Math.random();
                 arry[n] = Math.random();
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

      this.CheckHistDrawAttributes();

      this.RecreateDrawG(false, "main_layer");

      var w = this.frame_width(),
          h = this.frame_height(),
          handle = null;

      // if (this.lineatt.color == 'none') this.lineatt.color = 'cyan';

      if (this.options.Color + this.options.Box + this.options.Scat + this.options.Text +
          this.options.Contour + this.options.Arrow + this.options.Candle.length == 0)
         this.options.Scat = 1;

      if (this.draw_content)
      if (this.IsTH2Poly())
         handle = this.DrawPolyBinsColor(w, h);
      else
      if (this.options.Color > 0)
         handle = this.DrawBinsColor(w, h);
      else
      if (this.options.Scat > 0)
         handle = this.DrawBinsScatter(w, h);
      else
      if (this.options.Box > 0)
         handle = this.DrawBinsBox(w, h);
      else
      if (this.options.Arrow > 0)
         handle = this.DrawBinsArrow(w,h);
      else
      if (this.options.Contour > 0)
         handle = this.DrawBinsContour(w,h);
      else
      if (this.options.Candle.length > 0)
         handle = this.DrawCandle(w, h);

      if (this.options.Text > 0)
         handle = this.DrawBinsText(w, h, handle);

      this.tt_handle = handle;
   }

   TH2Painter.prototype.GetBinTips = function (i, j) {
      var lines = [], pmain = this.main_painter();

      lines.push(this.GetTipName());

      if (pmain.x_kind == 'labels')
         lines.push("x = " + pmain.AxisAsText("x", this.GetBinX(i)));
      else
         lines.push("x = [" + pmain.AxisAsText("x", this.GetBinX(i)) + ", " + pmain.AxisAsText("x", this.GetBinX(i+1)) + ")");

      if (pmain.y_kind == 'labels')
         lines.push("y = " + pmain.AxisAsText("y", this.GetBinY(j)));
      else
         lines.push("y = [" + pmain.AxisAsText("y", this.GetBinY(j)) + ", " + pmain.AxisAsText("y", this.GetBinY(j+1)) + ")");

      lines.push("bin = " + i + ", " + j);

      var histo = this.GetObject(),
          binz = histo.getBinContent(i+1,j+1);
      if (histo.$baseh) binz -= histo.$baseh.getBinContent(i+1,j+1);

      if (binz === Math.round(binz))
         lines.push("entries = " + binz);
      else
         lines.push("entries = " + JSROOT.FFormat(binz, JSROOT.gStyle.fStatFormat));

      return lines;
   }

   TH2Painter.prototype.GetCandleTips = function(p) {
      var lines = [], main = this.main_painter();

      lines.push(this.GetTipName());

      lines.push("x = " + main.AxisAsText("x", this.GetBinX(p.bin)));
      // lines.push("x = [" + main.AxisAsText("x", this.GetBinX(p.bin)) + ", " + main.AxisAsText("x", this.GetBinX(p.bin+1)) + ")");

      lines.push('mean y = ' + JSROOT.FFormat(p.meany, JSROOT.gStyle.fStatFormat))
      lines.push('m25 = ' + JSROOT.FFormat(p.m25y, JSROOT.gStyle.fStatFormat))
      lines.push('p25 = ' + JSROOT.FFormat(p.p25y, JSROOT.gStyle.fStatFormat))

      return lines;
   }

   TH2Painter.prototype.ProvidePolyBinHints = function(binindx, realx, realy) {

      var bin = this.histo.fBins.arr[binindx],
          pmain = this.main_painter(),
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

            for (n=0;n<gr.fNpoints;++n) {
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
      lines.push("content = " + JSROOT.FFormat(bin.fContent, JSROOT.gStyle.fStatFormat));
      return lines;
   }

   TH2Painter.prototype.ProcessTooltip = function(pnt) {
      if (!pnt || !this.draw_content || !this.draw_g || !this.tt_handle) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         this.ProvideUserTooltip(null);
         return null;
      }

      var histo = this.GetObject(),
          h = this.tt_handle, i,
          ttrect = this.draw_g.select(".tooltip_bin");

      if (h.poly) {
         // process tooltips from TH2Poly

         var pmain = this.main_painter(),
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
               if ((bin.fContent === 0) && (this.options.Color === 11)) continue;

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
            this.ProvideUserTooltip(null);
            return null;
         }

         var res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.color : 'blue',
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

         if (this.IsUserTooltipCallback() && res.changed)
            this.ProvideUserTooltip({ obj: histo,  name: histo.fName,
                                      bin: foundindx,
                                      cont: bin.fContent,
                                      grx: pnt.x, gry: pnt.y });

         return res;

      } else

      if (h.candle) {
         // process tooltips for candle

         var p;

         for (i=0;i<h.candle.length;++i) {
            p = h.candle[i];
            if ((p.x1 <= pnt.x) && (pnt.x <= p.x2) && (p.yy1 <= pnt.y) && (pnt.y <= p.yy2)) break;
         }

         if (i>=h.candle.length) {
            ttrect.remove();
            this.ProvideUserTooltip(null);
            return null;
         }

         var res = { name: histo.fName, title: histo.fTitle,
                     x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.color : 'blue',
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
                     .attr("height", p.yy2- p.yy1)
                     .style("opacity", "0.7")
                     .property("current_bin", i);
         }

         if (this.IsUserTooltipCallback() && res.changed) {
            this.ProvideUserTooltip({ obj: histo,  name: histo.fName,
                                      bin: i+1, cont: p.median, binx: i+1, biny: 1,
                                      grx: pnt.x, gry: pnt.y });
         }

         return res;
      }


      var i, j, find = 0, binz = 0, colindx = null;

      // search bin position
      for (i = h.i1; i < h.i2; ++i)
         if ((pnt.x>=h.grx[i]) && (pnt.x<=h.grx[i+1])) { ++find; break; }

      for (j = h.j1; j <= h.j2; ++j)
         if ((pnt.y>=h.gry[j+1]) && (pnt.y<=h.gry[j])) { ++find; break; }

      if (find === 2) {
         binz = histo.getBinContent(i+1,j+1);
         if (h.hide_only_zeros) {
            colindx = (binz === 0) && !this._show_empty_bins ? null : 0;
         } else {
            colindx = this.getValueColor(binz, true);
            if ((colindx === null) && (binz === 0) && this._show_empty_bins) colindx = 0;
         }
      }

      if (colindx === null) {
         ttrect.remove();
         this.ProvideUserTooltip(null);
         return null;
      }

      var res = { name: histo.fName, title: histo.fTitle,
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.color : 'blue',
                  lines: this.GetBinTips(i, j), exact: true, menu: true };

      if (this.options.Color > 0) res.color2 = this.GetPalette()[colindx];

      if (pnt.disabled) {
         ttrect.remove();
         res.changed = true;
      } else {
         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:rect")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         res.changed = ttrect.property("current_bin") !== i*10000 + j;

         if (res.changed)
            ttrect.attr("x", h.grx[i])
                  .attr("width", h.grx[i+1] - h.grx[i])
                  .attr("y", h.gry[j+1])
                  .attr("height", h.gry[j] - h.gry[j+1])
                  .style("opacity", "0.7")
                  .property("current_bin", i*10000 + j);
      }

      if (this.IsUserTooltipCallback() && res.changed) {
         this.ProvideUserTooltip({ obj: histo,  name: histo.fName,
                                   bin: histo.getBin(i+1, j+1), cont: binz, binx: i+1, biny: j+1,
                                   grx: pnt.x, gry: pnt.y });
      }

      return res;
   }

   TH2Painter.prototype.CanZoomIn = function(axis,min,max) {
      // check if it makes sense to zoom inside specified axis range
      if ((axis=="x") && (this.GetIndexX(max,0.5) - this.GetIndexX(min,0) > 1)) return true;

      if ((axis=="y") && (this.GetIndexY(max,0.5) - this.GetIndexY(min,0) > 1)) return true;

      if (axis=="z") return true;

      return false;
   }

   TH2Painter.prototype.Draw2D = function(call_back, resize) {

      this.mode3d = false;

      if (typeof this.Create3DScene == 'function')
         this.Create3DScene(-1);

      this.CreateXY();

      // draw new palette, resize frame if required
      var pp = this.DrawColorPalette((this.options.Zscale > 0) && ((this.options.Color > 0) || (this.options.Contour > 0)), true);

      this.DrawAxes();

      this.DrawGrids();

      this.DrawBins();

      // redraw palette till the end when contours are available
      if (pp) pp.DrawPave();

      this.DrawTitle();

      this.UpdateStatWebCanvas();

      this.AddInteractive();

      JSROOT.CallBack(call_back);
   }

   TH2Painter.prototype.Draw3D = function(call_back) {
      this.mode3d = true;
      JSROOT.AssertPrerequisites('hist3d', function() {
         this.Draw3D(call_back);
      }.bind(this));
   }

   TH2Painter.prototype.CallDrawFunc = function(callback, resize) {
      var main = this.main_painter(), is3d = false;

      if ((this.options.Contour > 0) && (main !== this)) is3d = main.mode3d; else
      if ((this.options.Lego > 0) || (this.options.Surf > 0) || (this.options.Error > 0)) is3d = true;

      if ((main!==this) && (is3d !== main.mode3d)) {
         is3d = main.mode3d;

         this.options.Lego = main.options.Lego;
         this.options.Surf = main.options.Surf;
         this.options.Error = main.options.Error;
      }

      var funcname = is3d ? "Draw3D" : "Draw2D";

      this[funcname](callback, resize);
   }

   TH2Painter.prototype.Redraw = function(resize) {
      this.CallDrawFunc(null, resize);
   }

   JSROOT.Painter.drawHistogram2D = function(divid, histo, opt) {
      // create painter and add it to canvas
      var painter = new JSROOT.TH2Painter(histo);

      painter.SetDivId(divid, 1);

      // here we deciding how histogram will look like and how will be shown
      painter.options = painter.DecodeOptions(opt);

      if (painter.IsTH2Poly()) {
         if (painter.options.Lego) painter.options.Lego = 12; else // and lego always 12
         if (!painter.options.Color) painter.options.Color = 1; // default color
      }

      painter._show_empty_bins = false;

      painter._can_move_colz = true;

      // special case for root 3D drawings - user range is wired
      if ((painter.options.Contour !==14) && !painter.options.Lego && !painter.options.Surf)
         painter.CheckPadRange();

      painter.ScanContent();

      // check if we need to create statbox
      if (JSROOT.gStyle.AutoStat && (painter.create_canvas || histo.$snapid) /* && !this.IsTH2Poly()*/)
         painter.CreateStat(histo.$custom_stat);

      painter.CallDrawFunc(function() {
         this.DrawNextFunction(0, function() {
            if ((this.options.Lego <= 0) && (this.options.Surf <= 0)) {
               if (this.options.AutoZoom) this.AutoZoom();
            }
            this.FillToolbar();
            this.DrawingReady();
         }.bind(this));
      }.bind(painter));

      return painter;
   }


   // ====================================================================

   function THStackPainter(stack) {
      JSROOT.TObjectPainter.call(this, stack);

      this.nostack = false;
      this.firstpainter = null;
      this.painters = []; // keep painters to be able update objects
   }

   THStackPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   THStackPainter.prototype.Cleanup = function() {
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

   THStackPainter.prototype.BuildStack = function() {
      //  build sum of all histograms
      //  Build a separate list fStack containing the running sum of all histograms

      var stack = this.GetObject();
      if (!stack.fHists) return false;
      var nhists = stack.fHists.arr.length;
      if (nhists <= 0) return false;
      var lst = JSROOT.Create("TList");
      lst.Add(JSROOT.clone(stack.fHists.arr[0]));
      this.haserrors = this.HasErrors(stack.fHists.arr[0]);
      for (var i=1;i<nhists;++i) {
         var hnext = JSROOT.clone(stack.fHists.arr[i]);
         var hprev = lst.arr[i-1];

         if ((hnext.fNbins != hprev.fNbins) ||
             (hnext.fXaxis.fXmin != hprev.fXaxis.fXmin) ||
             (hnext.fXaxis.fXmax != hprev.fXaxis.fXmax)) {
            JSROOT.console("When drawing THStack, cannot sum-up histograms " + hnext.fName + " and " + hprev.fName);
            delete hnext;
            lst.Clear();
            delete lst;
            return false;
         }

         this.haserrors = this.haserrors || this.HasErrors(stack.fHists.arr[i]);

         // trivial sum of histograms
         for (var n = 0; n < hnext.fArray.length; ++n)
            hnext.fArray[n] += hprev.fArray[n];

         lst.Add(hnext);
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

   THStackPainter.prototype.GetMinMax = function(iserr) {
      var res = { min : 0, max : 0 },
          stack = this.GetObject();

      if (this.nostack) {
         for (var i = 0; i < stack.fHists.arr.length; ++i) {
            var resh = this.GetHistMinMax(stack.fHists.arr[i], iserr);
            if (i==0) res = resh; else {
               if (resh.min < res.min) res.min = resh.min;
               if (resh.max > res.max) res.max = resh.max;
            }
         }

         if (stack.fMaximum != -1111)
            res.max = stack.fMaximum;
         else
            res.max *= 1.05;

         if (stack.fMinimum != -1111) res.min = stack.fMinimum;
      } else {
         res.min = this.GetHistMinMax(stack.fStack.arr[0], iserr).min;
         res.max = this.GetHistMinMax(stack.fStack.arr[stack.fStack.arr.length-1], iserr).max * 1.05;
      }

      var pad = this.root_pad();
      if (pad && pad.fLogy) {
         if (res.min<0) res.min = res.max * 1e-4;
      }

      return res;
   }

   THStackPainter.prototype.DrawNextHisto = function(indx, opt, mm, subp) {
      if (mm === "callback") {
         mm = null; // just misuse min/max argument to indicate callback
         if (indx<0) this.firstpainter = subp;
                else this.painters.push(subp);
         indx++;
      }

      var stack = this.GetObject(),
          hist = stack.fHistogram, hopt = "",
          hlst = this.nostack ? stack.fHists : stack.fStack,
          nhists = (hlst && hlst.arr) ? hlst.arr.length : 0, rindx = 0;

      if (indx>=nhists) return this.DrawingReady();

      if (indx>=0) {
         rindx = this.horder ? indx : nhists-indx-1;
         hist = hlst.arr[rindx];
         hopt = hlst.opt[rindx] || hist.fOption || opt;
         if (hopt.toUpperCase().indexOf(opt)<0) hopt += opt;
         hopt += " same";
      } else {
         hopt = (opt || "") + " axis";
         if (mm) hopt += ";minimum:" + mm.min + ";maximum:" + mm.max;
      }

      // special handling of stacked histograms - set $baseh object for correct drawing
      // also used to provide tooltips
      if ((rindx > 0) && !this.nostack) hist.$baseh = hlst.arr[rindx - 1];

      JSROOT.draw(this.divid, hist, hopt, this.DrawNextHisto.bind(this, indx, opt, "callback"));
   }

   THStackPainter.prototype.drawStack = function(opt) {

      var pad = this.root_pad(),
          stack = this.GetObject(),
          histos = stack.fHists,
          nhists = histos.arr.length,
          d = new JSROOT.DrawOptions(opt),
          lsame = d.check("SAME");

      this.nostack = d.check("NOSTACK");

      opt = d.opt; // use remaining draw options for histogram draw

      // when building stack, one could fail to sum up histograms
      if (!this.nostack)
         this.nostack = ! this.BuildStack();

      // if any histogram appears with pre-calculated errors, use E for all histograms
      if (!this.nostack && this.haserrors && !d.check("HIST")) opt+= " E";

      // order used to display histograms in stack direct - true, reverse - false
      this.dolego = d.check("LEGO");
      this.horder = this.nostack || this.dolego;

      var mm = this.GetMinMax(d.check("E"));

      var histo = stack.fHistogram;

      if (!histo) {

         // compute the min/max of each axis
         var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
         for (var i = 0; i < nhists; ++i) {
            var h = histos.arr[i];
            if (i == 0 || h.fXaxis.fXmin < xmin)
               xmin = h.fXaxis.fXmin;
            if (i == 0 || h.fXaxis.fXmax > xmax)
               xmax = h.fXaxis.fXmax;
            if (i == 0 || h.fYaxis.fXmin < ymin)
               ymin = h.fYaxis.fXmin;
            if (i == 0 || h.fYaxis.fXmax > ymax)
               ymax = h.fYaxis.fXmax;
         }

         var h = stack.fHists.arr[0];
         stack.fHistogram = histo = JSROOT.CreateHistogram("TH1I", h.fXaxis.fNbins);
         histo.fName = h.fName;
         histo.fXaxis = JSROOT.clone(h.fXaxis);
         histo.fYaxis = JSROOT.clone(h.fYaxis);
         histo.fXaxis.fXmin = xmin;
         histo.fXaxis.fXmax = xmax;
         histo.fYaxis.fXmin = ymin;
         histo.fYaxis.fXmax = ymax;
      }
      histo.fTitle = stack.fTitle;

      if (pad && pad.fLogy) {
         if (mm.max<=0) mm.max = 1;
         if (mm.min<=0) mm.min = 1e-4*mm.max;
         var kmin = 1/(1 + 0.5*JSROOT.log10(mm.max / mm.min)),
             kmax = 1 + 0.2*JSROOT.log10(mm.max / mm.min);
         mm.min*=kmin;
         mm.max*=kmax;
      }

      this.DrawNextHisto(!lsame ? -1 : 0, opt, mm);
      return this;
   }

   THStackPainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;

      var isany = false;
      if (this.firstpainter)
         if (this.firstpainter.UpdateObject(obj.fHistogram)) isany = true;

      var harr = this.nostack ? obj.fHists.arr : obj.fStack.arr,
          nhists = Math.min(harr.length, this.painters.length);

      for (var i = 0; i < nhists; ++i) {
         var hist = harr[this.horder ? i : nhists - i - 1];
         if (this.painters[i].UpdateObject(hist)) isany = true;
      }

      return isany;
   }


   JSROOT.Painter.drawHStack = function(divid, stack, opt) {
      // paint the list of histograms
      // By default, histograms are shown stacked.
      // - the first histogram is paint
      // - then the sum of the first and second, etc

      var painter = new JSROOT.THStackPainter(stack);

      painter.SetDivId(divid, -1); // it maybe no element to set divid

      if (!stack.fHists || (stack.fHists.arr.length == 0)) return painter.DrawingReady();

      painter.drawStack(opt);

      painter.SetDivId(divid); // only when first histogram drawn, we could assign divid

      return painter;
   }


   // =================================================================================

   JSROOT.Painter.drawTF2 = function(divid, func, opt) {
      // TF2 always drawn via temporary TH2 object,
      // therefore there is no special painter class

      var hist = null, npx = 0, npy = 0, nsave = 1,
          d = new JSROOT.DrawOptions(opt);

      if (d.check('NOSAVE')) nsave = 0;

      if (!func.fSave || func.fSave.length<7 || !nsave) {
         nsave = 0;
      } else {
          nsave = func.fSave.length;
          npx = Math.round(func.fSave[nsave-2]);
          npy = Math.round(func.fSave[nsave-1]);
          if (nsave !== (npx+1)*(npy+1) + 6) nsave = 0;
      }

      if (nsave > 6) {
         var dx = (func.fSave[nsave-5] - func.fSave[nsave-6]) / npx / 2,
             dy = (func.fSave[nsave-3] - func.fSave[nsave-4]) / npy / 2;

         hist = JSROOT.CreateHistogram("TH2F", npx+1, npy+1);

         hist.fXaxis.fXmin = func.fSave[nsave-6] - dx;
         hist.fXaxis.fXmax = func.fSave[nsave-5] + dx;

         hist.fYaxis.fXmin = func.fSave[nsave-4] - dy;
         hist.fYaxis.fXmax = func.fSave[nsave-3] + dy;

         var k = 0;
         for (var j=0;j<=npy;++j)
            for (var i=0;i<=npx;++i)
               hist.setBinContent(hist.getBin(i+1,j+1), func.fSave[k++]);

      } else {
         npx = Math.max(func.fNpx, 2);
         npy = Math.max(func.fNpy, 2);

         hist = JSROOT.CreateHistogram("TH2F", npx, npy);

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

      if (d.empty()) opt = "cont3"; else
      if (d.opt === "SAME") opt = "cont2 same";
      else opt = d.opt;

      return JSROOT.Painter.drawHistogram2D(divid, hist, opt);
   }

   JSROOT.TPavePainter = TPavePainter;
   JSROOT.TAxisPainter = TAxisPainter;
   JSROOT.THistPainter = THistPainter;
   JSROOT.TH1Painter = TH1Painter;
   JSROOT.TH2Painter = TH2Painter;
   JSROOT.THStackPainter = THStackPainter;

   return JSROOT;

}));
