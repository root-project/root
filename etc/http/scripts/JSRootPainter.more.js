/// @file JSRootPainter.more.js
/// Part of JavaScript ROOT graphics with more classes like TEllipse, TLine, ...
/// Such classes are rarely used and therefore loaded only on demand

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['d3', 'JSRootPainter', 'JSRootMath'], factory );
   } else {

      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.v3.js', 'JSRootPainter.more.js');

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.more.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.more.js');

      // Browser globals
      factory(d3, JSROOT);
   }
} (function(d3, JSROOT) {

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
            green = [   0,  46,  99, 149, 194, 220, 183, 166, 147];
            blue = [   6,   8,  36,  91, 169, 235, 246, 240, 233];
            break;

         // Blue Green Yellow
         case 71:
            red = [ 22, 19,  19,  25,  35,  53,  88, 139, 210];
            green = [  0, 32,  69, 108, 135, 159, 183, 198, 215];
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
            red = [  61,  99, 136, 181, 213, 225, 198, 136, 24];
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

   // ==============================================================================


   JSROOT.Painter.drawEllipse = function(divid, obj, opt) {

      this.SetDivId(divid, 2);

      this.Redraw = function() {
         var ellipse = this.GetObject();

         if(!this.lineatt) this.lineatt = JSROOT.Painter.createAttLine(ellipse);
         if (!this.fillatt) this.fillatt = this.createAttFill(ellipse);

         // create svg:g container for ellipse drawing
         this.RecreateDrawG(true, "text_layer");

         var x = this.AxisToSvg("x", ellipse.fX1),
             y = this.AxisToSvg("y", ellipse.fY1),
             rx = this.AxisToSvg("x", ellipse.fX1 + ellipse.fR1) - x,
             ry = y - this.AxisToSvg("y", ellipse.fY1 + ellipse.fR2);

         if ((ellipse.fPhimin == 0) && (ellipse.fPhimax == 360) && (ellipse.fTheta == 0)) {
            // this is simple case, which could be drawn with svg:ellipse
            this.draw_g
                .append("svg:ellipse")
                .attr("cx", x).attr("cy", y)
                .attr("rx", rx).attr("ry", ry)
                .call(this.lineatt.func).call(this.fillatt.func);
            return;
         }

         // here svg:path is used to draw more complex figure

         var ct = Math.cos(Math.PI*ellipse.fTheta/180.);
         var st = Math.sin(Math.PI*ellipse.fTheta/180.);

         var dx1 =  rx * Math.cos(ellipse.fPhimin*Math.PI/180.);
         var dy1 =  ry * Math.sin(ellipse.fPhimin*Math.PI/180.);
         var x1 =  dx1*ct - dy1*st;
         var y1 = -dx1*st - dy1*ct;

         var dx2 = rx * Math.cos(ellipse.fPhimax*Math.PI/180.);
         var dy2 = ry * Math.sin(ellipse.fPhimax*Math.PI/180.);
         var x2 =  dx2*ct - dy2*st;
         var y2 = -dx2*st - dy2*ct;

         this.draw_g
            .attr("transform","translate("+x.toFixed(1)+","+y.toFixed(1)+")")
            .append("svg:path")
            .attr("d", "M 0,0" +
                       " L " + x1.toFixed(1) + "," + y1.toFixed(1) +
                       " A " + rx.toFixed(1) + " " + ry.toFixed(1) + " " + -ellipse.fTheta.toFixed(1) + " 1 0 " + x2.toFixed(1) + "," + y2.toFixed(1) +
                       " L 0,0 Z")
            .call(this.lineatt.func).call(this.fillatt.func);
      }

      this.Redraw(); // actual drawing
      return this.DrawingReady();
   }

   // =============================================================================

   JSROOT.Painter.drawLine = function(divid, obj, opt) {

      this.SetDivId(divid, 2);

      this.Redraw = function() {
         var line = this.GetObject(),
             lineatt = JSROOT.Painter.createAttLine(line);

         // create svg:g container for line drawing
         this.RecreateDrawG(true, "text_layer");

         this.draw_g
             .append("svg:line")
             .attr("x1", this.AxisToSvg("x", line.fX1))
             .attr("y1", this.AxisToSvg("y", line.fY1))
             .attr("x2", this.AxisToSvg("x", line.fX2))
             .attr("y2", this.AxisToSvg("y", line.fY2))
             .call(lineatt.func);
      }

      this.Redraw(); // actual drawing

      return this.DrawingReady();
   }

   // =============================================================================

   JSROOT.Painter.drawPolyLine = function(divid, obj, opt) {

      this.SetDivId(divid, 2);

      this.Redraw = function() {
         var polyline = this.GetObject(),
             lineatt = JSROOT.Painter.createAttLine(polyline),
             fillatt = this.createAttFill(polyline);

         // create svg:g container for polyline drawing
         this.RecreateDrawG(true, "text_layer");

         var cmd = "M";
         for (var n=0;n<=polyline.fLastPoint;++n) {
            if (n>0) cmd += "L";
            cmd += this.AxisToSvg("x", polyline.fX[n]) + "," +
                   this.AxisToSvg("y", polyline.fY[n]);
         }
         if (fillatt.color!=='none') cmd+="Z";

         this.draw_g
             .append("svg:path")
             .attr("d", cmd)
             .call(lineatt.func)
             .call(fillatt.func);
      }

      this.Redraw(); // actual drawing

      return this.DrawingReady();
   }

   // =============================================================================

   JSROOT.Painter.drawBox = function(divid, obj, opt) {

      this.SetDivId(divid, 2);

      this.Redraw = function() {
         var box = this.GetObject(),
             lineatt = JSROOT.Painter.createAttLine(box),
             fillatt = this.createAttFill(box);

         // create svg:g container for box drawing
         this.RecreateDrawG(true, "text_layer");

         var x1 = this.AxisToSvg("x", box.fX1),
             x2 = this.AxisToSvg("x", box.fX2),
             y1 = this.AxisToSvg("y", box.fY1),
             y2 = this.AxisToSvg("y", box.fY2);

         // if box filled, contor line drawn only with "L" draw option:
         if ((fillatt.color != 'none') && !this.draw_line) lineatt.color = "none";

         this.draw_g
             .append("svg:rect")
             .attr("x", Math.min(x1,x2))
             .attr("y", Math.min(y1,y2))
             .attr("width", Math.abs(x2-x1))
             .attr("height", Math.abs(y1-y2))
             .call(lineatt.func)
             .call(fillatt.func);
      }

      this.draw_line = (typeof opt=='string') && (opt.toUpperCase().indexOf("L")>=0);

      this.Redraw(); // actual drawing

      return this.DrawingReady();
   }

   // =============================================================================

   JSROOT.Painter.drawMarker = function(divid, obj) {

      this.SetDivId(divid, 2);

      this.Redraw = function() {
         var marker = this.GetObject(),
             att = JSROOT.Painter.createAttMarker(marker);

         // create svg:g container for box drawing
         this.RecreateDrawG(true, "text_layer");

         var x = this.AxisToSvg("x", marker.fX),
             y = this.AxisToSvg("y", marker.fY);

         var path = att.create(x,y); 
         
         if (path && path.length > 0)
            this.draw_g.append("svg:path")
                .attr("d", path)
                .call(att.func);
      }

      this.Redraw(); // actual drawing

      return this.DrawingReady();
   }

   // ======================================================================================

   JSROOT.Painter.drawArrow = function(divid, obj, opt) {

      this.SetDivId(divid, 2);

      this.Redraw = function() {
         var arrow = this.GetObject();
         if (!this.lineatt) this.lineatt = JSROOT.Painter.createAttLine(arrow);
         if (!this.fillatt) this.fillatt = this.createAttFill(arrow);

         var wsize = Math.max(this.pad_width(), this.pad_height()) * arrow.fArrowSize;
         if (wsize<3) wsize = 3;
         var hsize = wsize * Math.tan(arrow.fAngle/2 * (Math.PI/180));

         // create svg:g container for line drawing
         this.RecreateDrawG(true, "text_layer");

         var x1 = this.AxisToSvg("x", arrow.fX1),
             y1 = this.AxisToSvg("y", arrow.fY1),
             x2 = this.AxisToSvg("x", arrow.fX2),
             y2 = this.AxisToSvg("y", arrow.fY2),
             right_arrow = "M0,0" + " L"+wsize.toFixed(1) +","+hsize.toFixed(1) + " L0," + (hsize*2).toFixed(1),
             left_arrow =  "M" + wsize.toFixed(1) + ", 0" + " L 0," + hsize.toFixed(1) + " L " + wsize.toFixed(1) + "," + (hsize*2).toFixed(1),
             m_start = null, m_mid = null, m_end = null, defs = null,
             oo = arrow.fOption, len = oo.length;

         if (oo.indexOf("<")==0) {
            var closed = (oo.indexOf("<|") == 0);
            if (!defs) defs = this.draw_g.append("defs");
            m_start = "jsroot_arrowmarker_" +  JSROOT.Painter.arrowcnt++;
            var beg = defs.append("svg:marker")
                .attr("id", m_start)
                .attr("markerWidth", wsize.toFixed(1))
                .attr("markerHeight", (hsize*2).toFixed(1))
                .attr("refX", "0")
                .attr("refY", hsize.toFixed(1))
                .attr("orient", "auto")
                .attr("markerUnits", "userSpaceOnUse")
                .append("svg:path")
                .style("fill","none")
                .attr("d", left_arrow + (closed ? " Z" : ""))
                .call(this.lineatt.func);
            if (closed) beg.call(this.fillatt.func);
         }

         var midkind = 0;
         if (oo.indexOf("->-")>=0)  midkind = 1; else
         if (oo.indexOf("-|>-")>=0) midkind = 11; else
         if (oo.indexOf("-<-")>=0) midkind = 2; else
         if (oo.indexOf("-<|-")>=0) midkind = 12;

         if (midkind > 0) {
            var closed = midkind > 10;
            if (!defs) defs = this.draw_g.append("defs");
            m_mid = "jsroot_arrowmarker_" +  JSROOT.Painter.arrowcnt++;

            var mid = defs.append("svg:marker")
              .attr("id", m_mid)
              .attr("markerWidth", wsize.toFixed(1))
              .attr("markerHeight", (hsize*2).toFixed(1))
              .attr("refX", (wsize*0.5).toFixed(1))
              .attr("refY", hsize.toFixed(1))
              .attr("orient", "auto")
              .attr("markerUnits", "userSpaceOnUse")
              .append("svg:path")
              .style("fill","none")
              .attr("d", ((midkind % 10 == 1) ? right_arrow : left_arrow) +
                         ((midkind > 10) ? " Z" : ""))
              .call(this.lineatt.func);
            if (midkind > 10) mid.call(this.fillatt.func);
         }

         if (oo.lastIndexOf(">") == len-1) {
            var closed = (oo.lastIndexOf("|>") == len-2) && (len>1);
            if (!defs) defs = this.draw_g.append("defs");
            m_end = "jsroot_arrowmarker_" +  JSROOT.Painter.arrowcnt++;
            var end = defs.append("svg:marker")
              .attr("id", m_end)
              .attr("markerWidth", wsize.toFixed(1))
              .attr("markerHeight", (hsize*2).toFixed(1))
              .attr("refX", wsize.toFixed(1))
              .attr("refY", hsize.toFixed(1))
              .attr("orient", "auto")
              .attr("markerUnits", "userSpaceOnUse")
              .append("svg:path")
              .style("fill","none")
              .attr("d", right_arrow + (closed ? " Z" : ""))
              .call(this.lineatt.func);
            if (closed) end.call(this.fillatt.func);
         }

         var path = this.draw_g
             .append("svg:path")
             .attr("d",  "M" + x1 + "," + y1 +
                      ((m_mid == null) ? "" : "L" + (x1/2+x2/2).toFixed(1) + "," + (y1/2+y2/2).toFixed(1)) +
                        " L" + x2 + "," + y2)
             .call(this.lineatt.func);

         if (m_start) path.style("marker-start","url(#" + m_start + ")");
         if (m_mid) path.style("marker-mid","url(#" + m_mid + ")");
         if (m_end) path.style("marker-end","url(#" + m_end + ")");
      }

      if (!('arrowcnt' in JSROOT.Painter)) JSROOT.Painter.arrowcnt = 0;

      this.Redraw(); // actual drawing
      return this.DrawingReady();
   }

   // =================================================================================

   JSROOT.Painter.drawTF2 = function(divid, func, opt) {
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
         var dx = (func.fSave[nsave-5] - func.fSave[nsave-6]) / (npx-1) / 2,
             dy = (func.fSave[nsave-3] - func.fSave[nsave-4]) / (npy-1) / 2;

         hist = JSROOT.CreateTH2(npx+1, npy+1);

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

         hist = JSROOT.CreateTH2(npx, npy);

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

      return JSROOT.Painter.drawHistogram2D.call(this, divid, hist, opt);
   };


   // ===================================================================================

   JSROOT.Painter.drawFunction = function(divid, tf1, opt) {
      this.bins = null;

      this.Eval = function(x) {
         return this.GetObject().evalPar(x);
      }

      this.CreateBins = function(ignore_zoom) {
         var main = this.main_painter(), gxmin = 0, gxmax = 0, tf1 = this.GetObject();

         if ((main!==null) && !ignore_zoom)  {
            if (main.zoom_xmin !== main.zoom_xmax) {
               gxmin = main.zoom_xmin;
               gxmax = main.zoom_xmax;
            } else {
               gxmin = main.xmin;
               gxmax = main.xmax;
            }
         }

         if ((tf1.fSave.length > 0) && !this.nosave) {
            // in the case where the points have been saved, useful for example
            // if we don't have the user's function
            var np = tf1.fSave.length - 2,
                xmin = tf1.fSave[np],
                xmax = tf1.fSave[np+1],
                dx = (xmax - xmin) / (np-1),
                res = [];

            for (var n=0; n < np; ++n) {
               var xx = xmin + dx*n;
               // check if points need to be displayed at all, keep at least 4-5 points for Bezier curves
               if ((gxmin !== gxmax) && ((xx + 2*dx < gxmin) || (xx - 2*dx > gxmax))) continue;

               res.push({ x: xx, y: tf1.fSave[n] });
            }
            return res;
         }

         var xmin = tf1.fXmin, xmax = tf1.fXmax, logx = false, pad = this.root_pad();

         if (gxmin !== gxmax) {
            if (gxmin > xmin) xmin = gxmin;
            if (gxmax < xmax) xmax = gxmax;
         }

         if ((main!==null) && main.logx && (xmin>0) && (xmax>0)) {
            logx = true;
            xmin = Math.log(xmin);
            xmax = Math.log(xmax);
         }

         var np = Math.max(tf1.fNpx, 101),
            dx = (xmax - xmin) / (np - 1),
            res = [];

         for (var n=0; n < np; n++) {
            var xx = xmin + n*dx;
            if (logx) xx = Math.exp(xx);
            var yy = this.Eval(xx);
            if (!isNaN(yy)) res.push({ x : xx, y : yy });
         }
         return res;
      }

      this.CreateDummyHisto = function() {

         var xmin = 0, xmax = 1, ymin = 0, ymax = 1,
             bins = this.CreateBins(true);

         if (bins!==null) {

            xmin = xmax = bins[0].x;
            ymin = ymax = bins[0].y;

            bins.forEach(function(bin) {
               xmin = Math.min(bin.x, xmin);
               xmax = Math.max(bin.x, xmax);
               ymin = Math.min(bin.y, ymin);
               ymax = Math.max(bin.y, ymax);
            });

            if (ymax > 0.0) ymax *= 1.05;
            if (ymin < 0.0) ymin *= 1.05;
         }

         var histo = JSROOT.Create("TH1I"),
             tf1 = this.GetObject();

         histo.fName = tf1.fName + "_hist";
         histo.fTitle = tf1.fTitle;

         histo.fXaxis.fXmin = xmin;
         histo.fXaxis.fXmax = xmax;
         histo.fYaxis.fXmin = ymin;
         histo.fYaxis.fXmax = ymax;

         return histo;
      }

      this.ProcessTooltipFunc = function(pnt) {
         var cleanup = false;

         if ((pnt === null) || (this.bins===null)) {
            cleanup = true;
         } else
         if ((this.bins.length==0) || (pnt.x < this.bins[0].grx) || (pnt.x > this.bins[this.bins.length-1].grx)) {
            cleanup = true;
         }

         if (cleanup) {
            if (this.draw_g !== null)
               this.draw_g.select(".tooltip_bin").remove();
            return null;
         }

         var min = 100000, best = -1, bin;

         for(var n=0; n<this.bins.length; ++n) {
            bin = this.bins[n];
            var dist = Math.abs(bin.grx - pnt.x);
            if (dist < min) { min = dist; best = n; }
         }

         bin = this.bins[best];

         var gbin = this.draw_g.select(".tooltip_bin"),
             radius = this.lineatt.width + 3;

         if (gbin.empty())
            gbin = this.draw_g.append("svg:circle")
                              .attr("class","tooltip_bin")
                              .style("pointer-events","none")
                              .attr("r", radius)
                              .call(this.lineatt.func)
                              .call(this.fillatt.func);

         var res = { x: bin.grx,
                     y: bin.gry,
                     color1: this.lineatt.color,
                     color2: this.fillatt.color,
                     lines: [],
                     exact : (Math.abs(bin.grx - pnt.x) < radius) && (Math.abs(bin.gry - pnt.y) < radius) };

         res.changed = gbin.property("current_bin") !== best;
         res.menu = res.exact;
         res.menu_dist = Math.sqrt((bin.grx-pnt.x)*(bin.grx-pnt.x) + (bin.gry-pnt.y)*(bin.grx-pnt.x));

         if (res.changed)
            gbin.attr("cx", bin.grx)
                .attr("cy", bin.gry)
                .property("current_bin", best);

         var name = this.GetTipName();
         if (name.length > 0) res.lines.push(name);

         var pmain = this.main_painter();
         if (pmain!==null)
            res.lines.push("x = " + pmain.AxisAsText("x",bin.x) + " y = " + pmain.AxisAsText("y",bin.y));

         return res;
      }

      this.Redraw = function() {

         var w = this.frame_width(), h = this.frame_height(), tf1 = this.GetObject();

         this.RecreateDrawG(false, "main_layer");

         // recalculate drawing bins when necessary
         this.bins = this.CreateBins(false);

         var pthis = this;
         var pmain = this.main_painter();
         var name = this.GetTipName("\n");

         if (!this.lineatt)
            this.lineatt = JSROOT.Painter.createAttLine(tf1);
         this.lineatt.used = false;
         if (!this.fillatt)
            this.fillatt = this.createAttFill(tf1, undefined, undefined, 1);
         this.fillatt.used = false;

         var n, bin;
         // first calculate graphical coordinates
         for(n=0; n<this.bins.length; ++n) {
            bin = this.bins[n];
            //bin.grx = Math.round(pmain.grx(bin.x));
            //bin.gry = Math.round(pmain.gry(bin.y));
            bin.grx = pmain.grx(bin.x);
            bin.gry = pmain.gry(bin.y);
         }

         if (this.bins.length > 2) {

            var h0 = h;  // use maximal frame height for filling
            if ((pmain.hmin!==undefined) && (pmain.hmin>=0)) {
               h0 = Math.round(pmain.gry(0));
               if ((h0 > h) || (h0 < 0)) h0 = h;
            }

            var path = JSROOT.Painter.BuildSvgPath("bezier", this.bins, h0, 2);

            if (this.lineatt.color != "none")
               this.draw_g.append("svg:path")
                  .attr("class", "line")
                  .attr("d", path.path)
                  .style("fill", "none")
                  .call(this.lineatt.func);

            if (this.fillatt.color != "none")
               this.draw_g.append("svg:path")
                  .attr("class", "area")
                  .attr("d", path.path + path.close)
                  .style("stroke", "none")
                  .call(this.fillatt.func);
         }

         delete this.ProcessTooltip;

        if (JSROOT.gStyle.Tooltip > 0)
           this.ProcessTooltip = this.ProcessTooltipFunc;
      }

      this.CanZoomIn = function(axis,min,max) {
         if (axis!=="x") return false;

         var tf1 = this.GetObject();

         if (tf1.fSave.length > 0) {
            // in the case where the points have been saved, useful for example
            // if we don't have the user's function
            var nb_points = tf1.fNpx;

            var xmin = tf1.fSave[nb_points + 1];
            var xmax = tf1.fSave[nb_points + 2];

            return Math.abs(xmin - xmax) / nb_points < Math.abs(min - max);
         }

         // if function calculated, one always could zoom inside
         return true;
      }
      
      this.PerformDraw = function() {
         if (this.main_painter() === null) {
            var histo = this.CreateDummyHisto();
            JSROOT.Painter.drawHistogram1D(this.divid, histo, "AXIS");
         }

         this.SetDivId(this.divid);
         this.Redraw();
         return this.DrawingReady();
      }

      this.SetDivId(divid, -1);
      var d = new JSROOT.DrawOptions(opt);
      this.nosave = d.check('NOSAVE');
      
      if (JSROOT.Math !== undefined) 
         return this.PerformDraw();
      
      JSROOT.AssertPrerequisites("math", this.PerformDraw.bind(this));
      return this;
   }

   // ====================================================================

   JSROOT.Painter.drawHStack = function(divid, stack, opt) {
      // paint the list of histograms
      // By default, histograms are shown stacked.
      // -the first histogram is paint
      // -then the sum of the first and second, etc

      // 'this' pointer set to created painter instance
      this.nostack = false;
      this.firstpainter = null;
      this.painters = []; // keep painters to be able update objects

      this.SetDivId(divid);

      if (!stack.fHists || (stack.fHists.arr.length == 0)) return this.DrawingReady();

      this.Cleanup = function() {
         delete this.firstpainter;
         delete this.painters;
         JSROOT.TObjectPainter.prototype.Cleanup.call(this);
      }

      this.HasErrors = function(hist) {
         if (hist.fSumw2 && (hist.fSumw2.length > 0))
            for (var n=0;n<hist.fSumw2.length;++n)
               if (hist.fSumw2[n] > 0) return true;
         return false;
      }

      this.BuildStack = function() {
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

      this.GetHistMinMax = function(hist, witherr) {
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

      this.GetMinMax = function(iserr) {
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

      this.DrawNextHisto = function(indx, opt) {
         var stack = this.GetObject(),
             hist = stack.fHistogram,
             harr = this.nostack ? stack.fHists.arr : stack.fStack.arr,
             nhists = harr ? harr.length : 0,
             rindx = 0;

         if (indx>=nhists) return this.DrawingReady();

         if (indx>=0) {
            rindx = this.horder ? indx : nhists-indx-1;
            hist = harr[rindx];
         }

         // special handling of stacked histograms - set $baseh object for correct drawing
         // also used to provide tooltips
         if ((rindx > 0) && !this.nostack) hist['$baseh'] = harr[rindx - 1];

         var hopt = hist.fOption.toUpperCase();
         if (hopt.indexOf(opt) < 0) hopt += opt;
         if (indx>=0) hopt += " SAME";

         var subp = JSROOT.draw(this.divid, hist, hopt);

         if (indx<0) this.firstpainter = subp;
                else this.painters.push(subp);
         subp.WhenReady(this.DrawNextHisto.bind(this, indx+1, opt));
      }

      this.drawStack = function(opt) {

         var pad = this.root_pad(),
             stack = this.GetObject(),
             histos = stack.fHists,
             nhists = histos.arr.length,
             d = new JSROOT.DrawOptions(opt),
             lsame = d.check("SAME");

         opt = d.opt; // use remaining draw options for histogram draw

         this.nostack = d.check("NOSTACK");

         // when building stack, one could fail to sum up histograms
         if (!this.nostack)
            this.nostack = ! this.BuildStack();

         // if any histogram appears with precalculated errors, use E for all histograms
         if (!this.nostack && this.haserrors && !d.check("HIST")) opt+= " E";

         // order used to display histograms in stack direct - true, reverse - false
         this.dolego = d.check("LEGO");
         this.horder = this.nostack || this.dolego;

         var mm = this.GetMinMax(d.check("E"));

         if (stack.fHistogram === null) {
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
            stack.fHistogram = JSROOT.Create("TH1I");
            stack.fHistogram.fName = "unnamed";
            stack.fHistogram.fXaxis = JSROOT.clone(h.fXaxis);
            stack.fHistogram.fYaxis = JSROOT.clone(h.fYaxis);
            stack.fHistogram.fXaxis.fXmin = xmin;
            stack.fHistogram.fXaxis.fXmax = xmax;
            stack.fHistogram.fYaxis.fXmin = ymin;
            stack.fHistogram.fYaxis.fXmax = ymax;
         }
         stack.fHistogram.fTitle = stack.fTitle;
         var histo = stack.fHistogram;
         if (!histo.TestBit(JSROOT.TH1StatusBits.kIsZoomed)) {
            if (pad && pad.fLogy)
                histo.fMaximum = mm.max * (1 + 0.2 * JSROOT.log10(mm.max / mm.min));
             else
                histo.fMaximum = mm.max;
            if (pad && pad.fLogy)
               histo.fMinimum = mm.min / (1 + 0.5 * JSROOT.log10(mm.max / mm.min));
            else
               histo.fMinimum = mm.min;
         }

         this.DrawNextHisto(!lsame ? -1 : 0, opt);
         return this;
      }

      this.UpdateObject = function(obj) {
         if (!this.MatchObjectType(obj)) return false;

         var isany = false;
         if (this.firstpainter)
            if (this.firstpainter.UpdateObject(obj.fHistogram)) isany = true;

         var nhists = obj.fHists.arr.length,
             harr = this.nostack ? obj.fHists.arr : obj.fStack.arr;
         for (var i = 0; i < nhists; ++i) {
            var hist = harr[this.horder ? i : nhists - i - 1];
            if (this.painters[i].UpdateObject(hist)) isany = true;
         }

         return isany;
      }

      return this.drawStack(opt);
   }

   // =======================================================================

   JSROOT.TGraphPainter = function(graph) {
      JSROOT.TObjectPainter.call(this, graph);
      this.ownhisto = false; // indicate if graph histogram was drawn for axes
      this.bins = null;
      this.xmin = this.ymin = this.xmax = this.ymax = 0;
      this.wheel_zoomy = true;
      this.is_bent = (graph._typename == 'TGraphBentErrors');
      this.has_errors = (graph._typename == 'TGraphErrors') ||
                        (graph._typename == 'TGraphAsymmErrors') ||
                         this.is_bent || graph._typename.match(/^RooHist/);
   }

   JSROOT.TGraphPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   JSROOT.TGraphPainter.prototype.Redraw = function() {
      this.DrawBins();
   }

   JSROOT.TGraphPainter.prototype.DecodeOptions = function(opt) {

      var d = new JSROOT.DrawOptions(opt);

      var res = { Line:0, Curve:0, Rect:0, Mark:0, Bar:0, OutRange: 0,  EF:0, Fill:0,
                  Errors: 0, MainError: 1, Ends: 1, Axis: "AXIS" };

      var graph = this.GetObject();

      if (this.has_errors) res.Errors = 1;

      if (d.check('L')) res.Line = 1;
      if (d.check('F')) res.Fill = 1;
      if (d.check('A')) res.Axis = "AXIS";
      if (d.check('X+')) res.Axis += "X+";
      if (d.check('Y+')) res.Axis += "Y+";
      if (d.check('C')) { res.Curve = 1; if (!res.Fill) res.Line = 1; }
      if (d.check('*')) res.Mark = 103;
      if (d.check('P0')) res.Mark = 104;
      if (d.check('P')) res.Mark = 1;
      if (d.check('B')) { res.Bar = 1; res.Errors = 0; }
      if (d.check('Z')) { res.Errors = 1; res.Ends = 0; }
      if (d.check('||')) { res.Errors = 1; res.MainError = 0; res.Ends = 1; }
      if (d.check('[]')) { res.Errors = 1; res.MainError = 0; res.Ends = 2; }
      if (d.check('|>')) { res.Errors = 1; res.Ends = 3; }
      if (d.check('>')) { res.Errors = 1; res.Ends = 4; }
      if (d.check('0')) { res.Mark = 1; res.Errors = 1; res.OutRange = 1; }
      if (d.check('1')) { if (res.Bar == 1) res.Bar = 2; }
      if (d.check('2')) { res.Rect = 1; res.Line = 0; res.Errors = 0; }
      if (d.check('3')) { res.EF = 1; res.Line = 0; res.Errors = 0; }
      if (d.check('4')) { res.EF = 2; res.Line = 0; res.Errors = 0; }
      if (d.check('5')) { res.Rect = 2; res.Line = 0; res.Errors = 0; }
      if (d.check('X')) res.Errors = 0;

      // special case - one could use svg:path to draw many pixels (
      if ((res.Mark==1) && (graph.fMarkerStyle==1)) res.Mark = 101;

      // if no drawing option is selected and if opt=='' nothing is done.
      if (res.Line + res.Fill + res.Mark + res.Bar + res.EF + res.Rect + res.Errors == 0) {
         if (d.empty()) res.Line = 1;
      }

      if (graph._typename == 'TGraphErrors') {
         if (d3.max(graph.fEX) < 1.0e-300 && d3.max(graph.fEY) < 1.0e-300)
            res.Errors = 0;
      }

      return res;
   }

   JSROOT.TGraphPainter.prototype.CreateBins = function() {
      var gr = this.GetObject();
      if (!gr) return;

      var p, kind = 0, npoints = gr.fNpoints;
      if ((gr._typename==="TCutG") && (npoints>3)) npoints--;

      if (gr._typename == 'TGraphErrors') kind = 1; else
      if (gr._typename == 'TGraphAsymmErrors' || gr._typename == 'TGraphBentErrors'
          || gr._typename.match(/^RooHist/)) kind = 2;

      this.bins = [];

      for (p=0;p<npoints;++p) {
         var bin = { x: gr.fX[p], y: gr.fY[p] };
         switch(kind) {
            case 1:
              bin.exlow = bin.exhigh = gr.fEX[p];
              bin.eylow = bin.eyhigh = gr.fEY[p];
              break;
            case 2:
               bin.exlow  = gr.fEXlow[p];
               bin.exhigh  = gr.fEXhigh[p];
               bin.eylow  = gr.fEYlow[p];
               bin.eyhigh = gr.fEYhigh[p];
               break;
         }
         this.bins.push(bin);

         if (p===0) {
            this.xmin = this.xmax = bin.x;
            this.ymin = this.ymax = bin.y;
         }

         if (kind > 0) {
            this.xmin = Math.min(this.xmin, bin.x - bin.exlow, bin.x + bin.exhigh);
            this.xmax = Math.max(this.xmax, bin.x - bin.exlow, bin.x + bin.exhigh);
            this.ymin = Math.min(this.ymin, bin.y - bin.eylow, bin.y + bin.eyhigh);
            this.ymax = Math.max(this.ymax, bin.y - bin.eylow, bin.y + bin.eyhigh);
         } else {
            this.xmin = Math.min(this.xmin, bin.x);
            this.xmax = Math.max(this.xmax, bin.x);
            this.ymin = Math.min(this.ymin, bin.y);
            this.ymax = Math.max(this.ymax, bin.y);
         }

      }
   }

   JSROOT.TGraphPainter.prototype.CreateHistogram = function() {
      // bins should be created

      var xmin = this.xmin, xmax = this.xmax, ymin = this.ymin, ymax = this.ymax;

      if (xmin >= xmax) xmax = xmin+1;
      if (ymin >= ymax) ymax = ymin+1;
      var dx = (xmax-xmin)*0.1, dy = (ymax-ymin)*0.1,
          uxmin = xmin - dx, uxmax = xmax + dx,
          minimum = ymin - dy, maximum = ymax + dy;

      if ((uxmin<0) && (xmin>=0)) uxmin = xmin*0.9;
      if ((uxmax>0) && (xmax<=0)) uxmax = 0;

      var graph = this.GetObject();

      if (graph.fMinimum != -1111) minimum = ymin = graph.fMinimum;
      if (graph.fMaximum != -1111) maximum = ymax = graph.fMaximum;
      if ((minimum < 0) && (ymin >=0)) minimum = 0.9*ymin;

      var histo = JSROOT.CreateTH1(100);
      histo.fName = graph.fName + "_h";
      histo.fTitle = graph.fTitle;
      histo.fXaxis.fXmin = uxmin;
      histo.fXaxis.fXmax = uxmax;
      histo.fYaxis.fXmin = minimum;
      histo.fYaxis.fXmax = maximum;
      histo.fMinimum = minimum;
      histo.fMaximum = maximum;
      histo.fBits = histo.fBits | JSROOT.TH1StatusBits.kNoStats;
      return histo;
   }


   JSROOT.TGraphPainter.prototype.OptimizeBins = function(filter_func) {
      if ((this.bins.length < 30) && !filter_func) return this.bins;

      var selbins = null;
      if (typeof filter_func == 'function') {
         for (var n = 0; n < this.bins.length; ++n) {
            if (filter_func(this.bins[n],n)) {
               if (selbins==null)
                  selbins = (n==0) ? [] : this.bins.slice(0, n);
            } else {
               if (selbins != null) selbins.push(this.bins[n]);
            }
         }
      }
      if (selbins == null) selbins = this.bins;

      if ((selbins.length < 5000) || (JSROOT.gStyle.OptimizeDraw == 0)) return selbins;
      var step = Math.floor(selbins.length / 5000);
      if (step < 2) step = 2;
      var optbins = [];
      for (var n = 0; n < selbins.length; n+=step)
         optbins.push(selbins[n]);

      return optbins;
   }

   JSROOT.TGraphPainter.prototype.TooltipText = function(d, asarray) {
      var pmain = this.main_painter(), lines = [];

      lines.push(this.GetTipName());
      lines.push("x = " + pmain.AxisAsText("x", d.x));
      lines.push("y = " + pmain.AxisAsText("y", d.y));

      if (this.options.Errors && (pmain.x_kind=='normal') && ('exlow' in d) && ((d.exlow!=0) || (d.exhigh!=0)))
         lines.push("error x = -" + pmain.AxisAsText("x", d.exlow) +
                              "/+" + pmain.AxisAsText("x", d.exhigh));

      if ((this.options.Errors || (this.options.EF > 0)) && (pmain.y_kind=='normal') && ('eylow' in d) && ((d.eylow!=0) || (d.eyhigh!=0)) )
         lines.push("error y = -" + pmain.AxisAsText("y", d.eylow) +
                           "/+" + pmain.AxisAsText("y", d.eyhigh));

      if (asarray) return lines;

      var res = "";
      for (var n=0;n<lines.length;++n) res += ((n>0 ? "\n" : "") + lines[n]);
      return res;
   }

   JSROOT.TGraphPainter.prototype.DrawBins = function() {

      this.RecreateDrawG(false, "main_layer");

      var pthis = this,
          pmain = this.main_painter(),
          w = this.frame_width(),
          h = this.frame_height(),
          graph = this.GetObject(),
          excl_width = 0;

      if (!this.lineatt)
         this.lineatt = JSROOT.Painter.createAttLine(graph, undefined, true);
      if (!this.fillatt)
         this.fillatt = this.createAttFill(graph, undefined, undefined, 1);
      this.fillatt.used = false;

      if (this.fillatt) this.fillatt.used = false; // mark used only when really used
      this.draw_kind = "none"; // indicate if special svg:g were created for each bin
      this.marker_size = 0; // indicate if markers are drawn

      if (this.lineatt.excl_side!=0) {
         excl_width = this.lineatt.excl_side * this.lineatt.excl_width;
         if (this.lineatt.width>0) this.options.Line = 1;
      }

      var drawbins = null;

      if (this.options.EF) {

         drawbins = this.OptimizeBins();

         // build lower part
         for (var n=0;n<drawbins.length;++n) {
            var bin = drawbins[n];
            bin.grx = pmain.grx(bin.x);
            bin.gry = pmain.gry(bin.y - bin.eylow);
         }

         var path1 = JSROOT.Painter.BuildSvgPath(this.options.EF > 1 ? "bezier" : "line", drawbins),
             bins2 = [];

         for (var n=drawbins.length-1;n>=0;--n) {
            var bin = drawbins[n];
            bin.gry = pmain.gry(bin.y + bin.eyhigh);
            bins2.push(bin);
         }

         // build upper part (in reverse direction)
         var path2 = JSROOT.Painter.BuildSvgPath(this.options.EF > 1 ? "Lbezier" : "Lline", bins2);

         this.draw_g.append("svg:path")
                    .attr("d", path1.path + path2.path + "Z")
                    .style("stroke", "none")
                    .call(this.fillatt.func);
         this.draw_kind = "lines";
      }

      if (this.options.Line == 1 || this.options.Fill == 1 || (excl_width!==0)) {

         var close_symbol = "";
         if (graph._typename=="TCutG") this.options.Fill = 1;

         if (this.options.Fill == 1) {
            close_symbol = "Z"; // always close area if we want to fill it
            excl_width=0;
         }

         if (drawbins===null) drawbins = this.OptimizeBins();

         for (var n=0;n<drawbins.length;++n) {
            var bin = drawbins[n];
            bin.grx = pmain.grx(bin.x);
            bin.gry = pmain.gry(bin.y);
         }

         var kind = "line"; // simple line
         if (this.options.Curve === 1) kind = "bezier"; else
         if (excl_width!==0) kind+="calc"; // we need to calculated deltas to build exclusion points

         var path = JSROOT.Painter.BuildSvgPath(kind, drawbins);

         if (excl_width!==0) {
            var extrabins = [];
            for (var n=drawbins.length-1;n>=0;--n) {
               var bin = drawbins[n];
               var dlen = Math.sqrt(bin.dgrx*bin.dgrx + bin.dgry*bin.dgry);
               // shift point, using
               bin.grx += excl_width*bin.dgry/dlen;
               bin.gry -= excl_width*bin.dgrx/dlen;
               extrabins.push(bin);
            }

            var path2 = JSROOT.Painter.BuildSvgPath("L" + ((this.options.Curve === 1) ? "bezier" : "line"), extrabins);

            this.draw_g.append("svg:path")
                       .attr("d", path.path + path2.path + "Z")
                       .style("stroke", "none")
                       .call(this.fillatt.func)
                       .style('opacity', 0.75);
         }

         if (this.options.Line || this.options.Fill) {
            var elem = this.draw_g.append("svg:path")
                           .attr("d", path.path + close_symbol);
            if (this.options.Line)
               elem.call(this.lineatt.func);
            else
               elem.style('stroke','none');

            if (this.options.Fill)
               elem.call(this.fillatt.func);
            else
               elem.style('fill','none');
         }

         this.draw_kind = "lines";
      }

      var nodes = null;

      if (this.options.Errors || this.options.Rect || this.options.Bar) {

         drawbins = this.OptimizeBins(function(pnt,i) {

            var grx = pmain.grx(pnt.x);

            // when drawing bars, take all points
            if (!pthis.options.Bar && ((grx<0) || (grx>w))) return true;

            var gry = pmain.gry(pnt.y);

            if (!pthis.options.Bar && !pthis.options.OutRange && ((gry<0) || (gry>h))) return true;

            pnt.grx1 = Math.round(grx);
            pnt.gry1 = Math.round(gry);

            if (pthis.has_errors) {
               pnt.grx0 = Math.round(pmain.grx(pnt.x - pnt.exlow) - grx);
               pnt.grx2 = Math.round(pmain.grx(pnt.x + pnt.exhigh) - grx);
               pnt.gry0 = Math.round(pmain.gry(pnt.y - pnt.eylow) - gry);
               pnt.gry2 = Math.round(pmain.gry(pnt.y + pnt.eyhigh) - gry);

               if (pthis.is_bent) {
                  pnt.grdx0 = Math.round(pmain.gry(pnt.y + graph.fEXlowd[i]) - gry);
                  pnt.grdx2 = Math.round(pmain.gry(pnt.y + graph.fEXhighd[i]) - gry);
                  pnt.grdy0 = Math.round(pmain.grx(pnt.x + graph.fEYlowd[i]) - grx);
                  pnt.grdy2 = Math.round(pmain.grx(pnt.x + graph.fEYhighd[i]) - grx);
               } else {
                  pnt.grdx0 = pnt.grdx2 = pnt.grdy0 = pnt.grdy2 = 0;
               }
            }

            return false;
         });

         this.draw_kind = "nodes";

         // here are up to five elements are collected, try to group them
         nodes = this.draw_g.selectAll(".grpoint")
                     .data(drawbins)
                     .enter()
                     .append("svg:g")
                     .attr("class", "grpoint")
                     .attr("transform", function(d) { return "translate(" + d.grx1 + "," + d.gry1 + ")"; });
      }

      if (this.options.Bar) {
         // calculate bar width
         for (var i=1;i<drawbins.length-1;++i)
            drawbins[i].width = Math.max(2, (drawbins[i+1].grx1 - drawbins[i-1].grx1) / 2 - 2);

         // first and last bins
         switch (drawbins.length) {
            case 0: break;
            case 1: drawbins[0].width = w/4; break; // patalogic case of single bin
            case 2: drawbins[0].width = drawbins[1].width = (drawbins[1].grx1-drawbins[0].grx1)/2; break;
            default:
               drawbins[0].width = drawbins[1].width;
               drawbins[drawbins.length-1].width = drawbins[drawbins.length-2].width;
         }

         var yy0 = Math.round(pmain.gry(0));

         nodes.append("svg:rect")
            .attr("x", function(d) { return Math.round(-d.width/2); })
            .attr("y", function(d) {
                d.bar = true; // element drawn as bar
                if (pthis.options.Bar!==1) return 0;
                return (d.gry1 > yy0) ? yy0-d.gry1 : 0;
             })
            .attr("width", function(d) { return Math.round(d.width); })
            .attr("height", function(d) {
                if (pthis.options.Bar!==1) return h > d.gry1 ? h - d.gry1 : 0;
                return Math.abs(yy0 - d.gry1);
             })
            .call(this.fillatt.func);
      }

      if (this.options.Rect)
         nodes.filter(function(d) { return (d.exlow > 0) && (d.exhigh > 0) && (d.eylow > 0) && (d.eyhigh > 0); })
           .append("svg:rect")
           .attr("x", function(d) { d.rect = true; return d.grx0; })
           .attr("y", function(d) { return d.gry2; })
           .attr("width", function(d) { return d.grx2 - d.grx0; })
           .attr("height", function(d) { return d.gry0 - d.gry2; })
           .call(this.fillatt.func)
           .call(this.options.Rect === 2 ? this.lineatt.func : function() {});

      this.error_size = 0;

      if (this.options.Errors) {
         // to show end of error markers, use line width attribute
         var lw = this.lineatt.width + JSROOT.gStyle.fEndErrorSize, bb = 0,
             vv = this.options.Ends ? "m0," + lw + "v-" + 2*lw : "",
             hh = this.options.Ends ? "m" + lw + ",0h-" + 2*lw : "",
             vleft = vv, vright = vv, htop = hh, hbottom = hh,
             mm = this.options.MainError ? "M0,0L" : "M"; // command to draw main errors

         switch (this.options.Ends) {
            case 2:  // option []
               bb = Math.max(this.lineatt.width+1, Math.round(lw*0.66));
               vleft = "m"+bb+","+lw + "h-"+bb + "v-"+2*lw + "h"+bb;
               vright = "m-"+bb+","+lw + "h"+bb + "v-"+2*lw + "h-"+bb;
               htop = "m-"+lw+","+bb + "v-"+bb + "h"+2*lw + "v"+bb;
               hbottom = "m-"+lw+",-"+bb + "v"+bb + "h"+2*lw + "v-"+bb;
               break;
            case 3: // option |>
               lw = Math.max(lw, Math.round(graph.fMarkerSize*8*0.66));
               bb = Math.max(this.lineatt.width+1, Math.round(lw*0.66));
               vleft = "l"+bb+","+lw + "v-"+2*lw + "l-"+bb+","+lw;
               vright = "l-"+bb+","+lw + "v-"+2*lw + "l"+bb+","+lw;
               htop = "l-"+lw+","+bb + "h"+2*lw + "l-"+lw+",-"+bb;
               hbottom = "l-"+lw+",-"+bb + "h"+2*lw + "l-"+lw+","+bb;
               break;
            case 4: // option >
               lw = Math.max(lw, Math.round(graph.fMarkerSize*8*0.66));
               bb = Math.max(this.lineatt.width+1, Math.round(lw*0.66));
               vleft = "l"+bb+","+lw + "m0,-"+2*lw + "l-"+bb+","+lw;
               vright = "l-"+bb+","+lw + "m0,-"+2*lw + "l"+bb+","+lw;
               htop = "l-"+lw+","+bb + "m"+2*lw + ",0l-"+lw+",-"+bb;
               hbottom = "l-"+lw+",-"+bb + "m"+2*lw + ",0l-"+lw+","+bb;
               break;
         }

         this.error_size = lw;

         lw = Math.floor((this.lineatt.width-1)/2); // one shoud take into account half of end-cup line width
         nodes.filter(function(d) { return (d.exlow > 0) || (d.exhigh > 0) || (d.eylow > 0) || (d.eyhigh > 0); })
             .append("svg:path")
             .call(this.lineatt.func)
             .style('fill', "none")
             .attr("d", function(d) {
                d.error = true;
                return ((d.exlow > 0)  ? mm + (d.grx0+lw) + "," + d.grdx0 + vleft : "") +
                       ((d.exhigh > 0) ? mm + (d.grx2-lw) + "," + d.grdx2 + vright : "") +
                       ((d.eylow > 0)  ? mm + d.grdy0 + "," + (d.gry0-lw) + hbottom : "") +
                       ((d.eyhigh > 0) ? mm + d.grdy2 + "," + (d.gry2+lw) + htop : "");
              });
      }

      if (this.options.Mark) {
         // for tooltips use markers only if nodes where not created
         var step = Math.max(1, Math.round(this.bins.length / 50000)),
             path = "", n, pnt, grx, gry;

         if (!this.markeratt)
            this.markeratt = JSROOT.Painter.createAttMarker(graph, this.options.Mark - 100);
         else
            this.markeratt.Change(undefined, this.options.Mark - 100);

         this.marker_size = this.markeratt.size;

         this.markeratt.reset_pos();

         for (n=0;n<this.bins.length;n+=step) {
            pnt = this.bins[n];
            grx = pmain.grx(pnt.x);
            if ((grx > -this.marker_size) && (grx < w+this.marker_size)) {
               gry = pmain.gry(pnt.y);
               if ((gry >-this.marker_size) && (gry < h+this.marker_size)) {
                  path += this.markeratt.create(grx, gry);
               }
            }
         }

         if (path.length>0) {
            this.draw_g.append("svg:path")
                       .attr("d", path)
                       .call(this.markeratt.func);
            if ((nodes===null) && (this.draw_kind=="none"))
               this.draw_kind = (this.options.Mark==101) ? "path" : "mark";

         }
      }
   }

   JSROOT.TGraphPainter.prototype.ProcessTooltip = function(pnt) {
      if (pnt === null) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      if ((this.draw_kind=="lines") || (this.draw_kind=="path") || (this.draw_kind=="mark"))
         return this.ProcessTooltipForPath(pnt);

      if (this.draw_kind!="nodes") return null;

      var width = this.frame_width(),
          height = this.frame_height(),
          pmain = this.main_painter(),
          painter = this,
          findbin = null, best_dist2 = 1e10, best = null;

      this.draw_g.selectAll('.grpoint').each(function() {
         var d = d3.select(this).datum();
         if (d===undefined) return;
         var dist2 = Math.pow(pnt.x - d.grx1, 2);
         if (pnt.nproc===1) dist2 += Math.pow(pnt.y - d.gry1, 2);
         if (dist2 >= best_dist2) return;

         var rect = null;

         if (d.error || d.rect || d.marker) {
            rect = { x1: Math.min(-painter.error_size, d.grx0),
                     x2: Math.max(painter.error_size, d.grx2),
                     y1: Math.min(-painter.error_size, d.gry2),
                     y2: Math.max(painter.error_size, d.gry0) };
         } else
         if (d.bar) {
             rect = { x1: -d.width/2, x2: d.width/2, y1: 0, y2: height - d.gry1 };

             if (painter.options.Bar===1) {
                var yy0 = pmain.gry(0);
                rect.y1 = (d.gry1 > yy0) ? yy0-d.gry1 : 0;
                rect.y2 = (d.gry1 > yy0) ? 0 : yy0-d.gry1;
             }
          } else {
             rect = { x1: -5, x2: 5, y1: -5, y2: 5 };
          }
          var matchx = (pnt.x >= d.grx1 + rect.x1) && (pnt.x <= d.grx1 + rect.x2);
          var matchy = (pnt.y >= d.gry1 + rect.y1) && (pnt.y <= d.gry1 + rect.y2);

          if (matchx && (matchy || (pnt.nproc > 1))) {
             best_dist2 = dist2;
             findbin = this;
             best = rect;
             best.exact = matchx && matchy;
          }
       });

      var ttrect = this.draw_g.select(".tooltip_bin");

      if (findbin == null) {
         ttrect.remove();
         return null;
      }

      var d = d3.select(findbin).datum();

      var res = { x: d.grx1, y: d.gry1,
                  color1: this.lineatt.color,
                  lines: this.TooltipText(d, true) };
      if (this.fillatt && this.fillatt.used) res.color2 = this.fillatt.color;

      if (best.exact) res.exact = true;
      res.menu = res.exact; // activate menu only when exactly locate bin
      res.menu_dist = 3; // distance alwyas fixed

      if (ttrect.empty())
         ttrect = this.draw_g.append("svg:rect")
                             .attr("class","tooltip_bin h1bin")
                             .style("pointer-events","none");

      res.changed = ttrect.property("current_bin") !== findbin;

      if (res.changed)
         ttrect.attr("x", d.grx1 + best.x1)
               .attr("width", best.x2 - best.x1)
               .attr("y", d.gry1 + best.y1)
               .attr("height", best.y2 - best.y1)
               .style("opacity", "0.3")
               .property("current_bin", findbin);

      return res;
   }

   JSROOT.TGraphPainter.prototype.ProcessTooltipForPath = function(pnt) {

      if (this.bins === null) return null;

      var islines = (this.draw_kind=="lines"),
          ismark = (this.draw_kind=="mark"),
          bestbin = null,
          bestdist = 1e10,
          pmain = this.main_painter(),
          dist, grx, gry, n, bin;

      for (n=0;n<this.bins.length;++n) {
         bin = this.bins[n];

         grx = pmain.grx(bin.x);
         gry = pmain.gry(bin.y);

         dist = (pnt.x-grx)*(pnt.x-grx) + (pnt.y-gry)*(pnt.y-gry);

         if (dist < bestdist) {
            bestdist = dist;
            bestbin = bin;
         }
      }

      // check last point
      if ((bestdist > 100) && islines) bestbin = null;

      var radius = Math.max(this.lineatt.width + 3, 4);

      if (this.marker_size > 0) radius = Math.max(Math.round(this.marker_size*7), radius);

      if (bestbin !== null)
         bestdist = Math.sqrt(Math.pow(pnt.x-pmain.grx(bestbin.x),2) + Math.pow(pnt.y-pmain.gry(bestbin.y),2));

      if (!islines && !ismark && (bestdist>radius)) bestbin = null;

      if (ismark && (bestbin!==null)) {
         if ((pnt.nproc == 1) && (bestdist>radius)) bestbin = null; else
         if ((this.bins.length==1) && (bestdist>3*radius)) bestbin = null;
      }

      var ttbin = this.draw_g.select(".tooltip_bin");

      if (bestbin===null) {
         ttbin.remove();
         return null;
      }

      var res = { x: pmain.grx(bestbin.x), y: pmain.gry(bestbin.y),
                  color1: this.lineatt.color,
                  lines: this.TooltipText(bestbin, true) };

      if (this.fillatt && this.fillatt.used) res.color2 = this.fillatt.color;

      if (!islines) {
         res.color1 = JSROOT.Painter.root_colors[this.GetObject().fMarkerColor];
         if (!res.color2) res.color2 = res.color1;
      }

      if (ttbin.empty())
         ttbin = this.draw_g.append("svg:g")
                             .attr("class","tooltip_bin");

      var gry1, gry2;

      if (this.options.EF && islines) {
         gry1 = pmain.gry(bestbin.y - bestbin.eylow);
         gry2 = pmain.gry(bestbin.y + bestbin.eyhigh);
      } else {
         gry1 = gry2 = pmain.gry(bestbin.y);
      }

      res.exact = (Math.abs(pnt.x - res.x) <= radius) &&
                  ((Math.abs(pnt.y - gry1) <= radius) || (Math.abs(pnt.y - gry2) <= radius));

      res.menu = res.exact;
      res.menu_dist = Math.sqrt((pnt.x-res.x)*(pnt.x-res.x) + Math.pow(Math.min(Math.abs(pnt.y-gry1),Math.abs(pnt.y-gry2)),2));

      res.changed = ttbin.property("current_bin") !== bestbin;

      if (res.changed) {
         ttbin.selectAll("*").remove(); // first delete all childs
         ttbin.property("current_bin", bestbin);

         if (ismark) {
            ttbin.append("svg:rect")
                 .attr("class","h1bin")
                 .style("pointer-events","none")
                 .style("opacity", "0.3")
                 .attr("x", (res.x - radius).toFixed(1))
                 .attr("y", (res.y - radius).toFixed(1))
                 .attr("width", (2*radius).toFixed(1))
                 .attr("height", (2*radius).toFixed(1));
         } else {
            ttbin.append("svg:circle").attr("cy", gry1.toFixed(1))
            if (Math.abs(gry1-gry2) > 1)
               ttbin.append("svg:circle").attr("cy", gry2.toFixed(1));

            var elem = ttbin.selectAll("circle")
                            .attr("r", radius)
                            .attr("cx", res.x.toFixed(1));

            if (!islines) {
               elem.style('stroke', res.color1 == 'black' ? 'green' : 'black').style('fill','none');
            } else {
               if (this.options.Line)
                  elem.call(this.lineatt.func);
               else
                  elem.style('stroke','black');
               if (this.options.Fill)
                  elem.call(this.fillatt.func);
               else
                  elem.style('fill','none');
            }
         }
      }

      return res;
   }

   JSROOT.TGraphPainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;

      // if our own histogram was used as axis drawing, we need update histogram  as well
      if (this.ownhisto)
         this.main_painter().UpdateObject(obj.fHistogram);

      var graph = this.GetObject();
      // TODO: make real update of TGraph object content
      graph.fX = obj.fX;
      graph.fY = obj.fY;
      graph.fNpoints = obj.fNpoints;
      this.CreateBins();
      return true;
   }

   JSROOT.TGraphPainter.prototype.CanZoomIn = function(axis,min,max) {
      // allow to zoom TGraph only when at least one point in the range

      var gr = this.GetObject();
      if ((gr===null) || (axis!=="x")) return false;

      for (var n=0; n < gr.fNpoints; ++n)
         if ((min < gr.fX[n]) && (gr.fX[n] < max)) return true;

      return false;
   }

   JSROOT.TGraphPainter.prototype.ButtonClick = function(funcname) {

      if (funcname !== "ToggleZoom") return false;

      var main = this.main_painter();
      if (main === null) return false;

      if ((this.xmin===this.xmax) && (this.ymin = this.ymax)) return false;

      main.Zoom(this.xmin, this.xmax, this.ymin, this.ymax);

      return true;
   }

   JSROOT.TGraphPainter.prototype.DrawNextFunction = function(indx, callback) {
      // method draws next function from the functions list

      var graph = this.GetObject();

      if ((graph.fFunctions === null) || (indx >= graph.fFunctions.arr.length))
         return JSROOT.CallBack(callback);

      var func = graph.fFunctions.arr[indx];
      var opt = graph.fFunctions.opt[indx];

      var painter = JSROOT.draw(this.divid, func, opt);
      if (painter) return painter.WhenReady(this.DrawNextFunction.bind(this, indx+1, callback));

      this.DrawNextFunction(indx+1, callback);
   }

   JSROOT.Painter.drawGraph = function(divid, graph, opt) {
      JSROOT.extend(this, new JSROOT.TGraphPainter(graph));

      this.options = this.DecodeOptions(opt);

      this.SetDivId(divid, -1); // just to get access to existing elements

      this.CreateBins();

      if (this.main_painter() == null) {
         if (graph.fHistogram == null)
            graph.fHistogram = this.CreateHistogram();
         JSROOT.Painter.drawHistogram1D(divid, graph.fHistogram, this.options.Axis);
         this.ownhisto = true;
      }

      this.SetDivId(divid);
      this.DrawBins();

      this.DrawNextFunction(0, this.DrawingReady.bind(this));

      return this;
   }

   // =============================================================

   JSROOT.Painter.drawMultiGraph = function(divid, mgraph, opt) {
      // function call with bind(painter)

      this.firstpainter = null;
      this.autorange = false;
      this.painters = []; // keep painters to be able update objects

      this.SetDivId(divid, -1); // it may be no element to set divid

      this.Cleanup = function() {
         this.painters = [];
         JSROOT.TObjectPainter.prototype.Cleanup.call(this);
      }

      this.UpdateObject = function(obj) {
         if (!this.MatchObjectType(obj)) return false;

         var mgraph = this.GetObject(),
             graphs = obj.fGraphs;

         mgraph.fTitle = obj.fTitle;

         var isany = false;
         if (this.firstpainter) {
            var histo = obj.fHistogram;
            if (this.autorange && !histo)
               histo = this.ScanGraphsRange(graphs);

            if (this.firstpainter.UpdateObject(histo)) isany = true;
         }

         for (var i = 0; i <  graphs.arr.length; ++i) {
            if (i<this.painters.length)
               if (this.painters[i].UpdateObject(graphs.arr[i])) isany = true;
         }

         return isany;
      }

      this.ComputeGraphRange = function(res, gr) {
         // Compute the x/y range of the points in this graph
         if (gr.fNpoints == 0) return;
         if (res.first) {
            res.xmin = res.xmax = gr.fX[0];
            res.ymin = res.ymax = gr.fY[0];
            res.first = false;
         }
         for (var i=0; i < gr.fNpoints; ++i) {
            res.xmin = Math.min(res.xmin, gr.fX[i]);
            res.xmax = Math.max(res.xmax, gr.fX[i]);
            res.ymin = Math.min(res.ymin, gr.fY[i]);
            res.ymax = Math.max(res.ymax, gr.fY[i]);
         }
         return res;
      }

      this.padtoX = function(pad, x) {
         // Convert x from pad to X.
         if (pad.fLogx && (x < 50))
            return Math.exp(2.302585092994 * x);
         return x;
      }

      this.ScanGraphsRange = function(graphs, histo, pad) {
         var mgraph = this.GetObject(),
             maximum, minimum, dx, dy, uxmin = 0, uxmax = 0, logx = false, logy = false,
             rw = {  xmin: 0, xmax: 0, ymin: 0, ymax: 0, first: true };

         if (pad!=null) {
            logx = pad.fLogx;
            logy = pad.fLogy;
            rw.xmin = pad.fUxmin;
            rw.xmax = pad.fUxmax;
            rw.ymin = pad.fUymin;
            rw.ymax = pad.fUymax;
            rw.first = false;
         }
         if (histo!=null) {
            minimum = histo.fYaxis.fXmin;
            maximum = histo.fYaxis.fXmax;
            if (pad!=null) {
               uxmin = this.padtoX(pad, rw.xmin);
               uxmax = this.padtoX(pad, rw.xmax);
            }
         } else {
            this.autorange = true;

            for (var i = 0; i < graphs.arr.length; ++i)
               this.ComputeGraphRange(rw, graphs.arr[i]);

            if (rw.xmin == rw.xmax) rw.xmax += 1.;
            if (rw.ymin == rw.ymax) rw.ymax += 1.;
            dx = 0.05 * (rw.xmax - rw.xmin);
            dy = 0.05 * (rw.ymax - rw.ymin);
            uxmin = rw.xmin - dx;
            uxmax = rw.xmax + dx;
            if (logy) {
               if (rw.ymin <= 0) rw.ymin = 0.001 * rw.ymax;
               minimum = rw.ymin / (1 + 0.5 * JSROOT.log10(rw.ymax / rw.ymin));
               maximum = rw.ymax * (1 + 0.2 * JSROOT.log10(rw.ymax / rw.ymin));
            } else {
               minimum = rw.ymin - dy;
               maximum = rw.ymax + dy;
            }
            if (minimum < 0 && rw.ymin >= 0)
               minimum = 0;
            if (maximum > 0 && rw.ymax <= 0)
               maximum = 0;
         }

         if (uxmin < 0 && rw.xmin >= 0) {
            if (logx) uxmin = 0.9 * rw.xmin;
                 else uxmin = 0;
         }
         if (uxmax > 0 && rw.xmax <= 0) {
            if (logx) uxmax = 1.1 * rw.xmax;
                 else uxmax = 0;
         }

         if (mgraph.fMinimum != -1111)
            rw.ymin = minimum = mgraph.fMinimum;
         if (mgraph.fMaximum != -1111)
            rw.ymax = maximum = mgraph.fMaximum;

         if (minimum < 0 && rw.ymin >= 0) {
            if (logy) minimum = 0.9 * rw.ymin;
         }
         if (maximum > 0 && rw.ymax <= 0) {
            if (logy) maximum = 1.1 * rw.ymax;
         }
         if (minimum <= 0 && logy)
            minimum = 0.001 * maximum;
         if (uxmin <= 0 && logx) {
            if (uxmax > 1000)
               uxmin = 1;
            else
               uxmin = 0.001 * uxmax;
         }

         // Create a temporary histogram to draw the axis (if necessary)
         if (!histo) {
            histo = JSROOT.Create("TH1I");
            histo.fTitle = mgraph.fTitle;
            histo.fXaxis.fXmin = uxmin;
            histo.fXaxis.fXmax = uxmax;
         }

         histo.fYaxis.fXmin = minimum;
         histo.fYaxis.fXmax = maximum;

         return histo;
      }

      this.DrawAxis = function() {
         // draw special histogram

         var mgraph = this.GetObject();

         var histo = this.ScanGraphsRange(mgraph.fGraphs, mgraph.fHistogram, this.root_pad());

         // histogram painter will be first in the pad, will define axis and
         // interactive actions
         this.firstpainter = JSROOT.Painter.drawHistogram1D(this.divid, histo, "AXIS");
      }

      this.DrawNextFunction = function(indx, callback) {
         // method draws next function from the functions list

         var mgraph = this.GetObject();

         if ((mgraph.fFunctions == null) || (indx >= mgraph.fFunctions.arr.length))
            return JSROOT.CallBack(callback);

         var func = mgraph.fFunctions.arr[indx],
             opt = mgraph.fFunctions.opt[indx];

         var painter = JSROOT.draw(this.divid, func, opt);
         if (painter) return painter.WhenReady(this.DrawNextFunction.bind(this, indx+1, callback));

         this.DrawNextFunction(indx+1, callback);
      }

      this.DrawNextGraph = function(indx, opt) {
         var graphs = this.GetObject().fGraphs;

         // at the end of graphs drawing draw functions (if any)
         if (indx >= graphs.arr.length)
            return this.DrawNextFunction(0, this.DrawingReady.bind(this));

         var drawopt = graphs.opt[indx];
         if ((drawopt==null) || (drawopt == "")) drawopt = opt;
         var subp = JSROOT.draw(this.divid, graphs.arr[indx], drawopt);
         this.painters.push(subp);
         subp.WhenReady(this.DrawNextGraph.bind(this, indx+1, opt));
      }

      if (opt == null) opt = "";
      opt = opt.toUpperCase().replace("3D","").replace("FB",""); // no 3D supported, FB not clear

      if ((opt.indexOf("A") >= 0) || (this.main_painter()==null)) {
         opt = opt.replace("A","");
         this.DrawAxis();
      }
      this.SetDivId(divid);

      this.DrawNextGraph(0, opt);

      return this;
   }

   // ==============================================================================

   JSROOT.Painter.drawLegend = function(divid, obj, opt) {

      JSROOT.extend(this, new JSROOT.TPavePainter(obj));

      this.SetDivId(divid);

      this.DrawLegendItems = function(w, h) {

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
         this.FinishTextDrawing();
      }

      this.PaveDrawFunc = this.DrawLegendItems;

      this.Redraw();

      return this.DrawingReady();
   }

   // ===========================================================================

   JSROOT.Painter.drawPaletteAxis = function(divid,palette,opt) {

      // disable draw of shadow element of TPave
      palette.fBorderSize = 1;
      palette.fShadowColor = 0;

      JSROOT.extend(this, new JSROOT.TPavePainter(palette));

      this.SetDivId(divid);

      this.z_handle = new JSROOT.TAxisPainter(palette.fAxis, true);
      this.z_handle.SetDivId(divid, -1);

      this.Cleanup = function() {
         if (this.z_handle) {
            this.z_handle.Cleanup();
            delete this.z_handle;
         }

         JSROOT.TObjectPainter.prototype.Cleanup.call(this);
      }

      this.DrawAxisPalette = function(s_width, s_height, arg) {

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
            z = d3.scale.log();
            z_kind = "log";
         } else {
            z = d3.scale.linear();
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
            // ignore when touch selection is actiavated
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

      this.ShowContextMenu = function(evnt) {
         this.main_painter().ShowContextMenu("z", evnt, this.GetObject().fAxis);
      }

      this.PaveDrawFunc = this.DrawAxisPalette;

      this.UseContextMenu = true;

      this.DrawPave(opt);

      return this.DrawingReady();
   }

   // ================= some functions for basic histogram painter =======================

   JSROOT.THistPainter.prototype.CreateContour = function(nlevels, zmin, zmax, zminpositive) {

      if (nlevels<1) nlevels = JSROOT.gStyle.fNumberContours;
      this.fContour = [];
      this.zmin = zmin;
      this.zmax = zmax;

      if (this.root_pad().fLogz) {
         if (this.zmax <= 0) this.zmax = 1.;
         if (this.zmin <= 0)
            if ((zminpositive===undefined) || (zminpositive <= 0))
               this.zmin = 0.0001*this.zmax;
            else
               this.zmin = ((zminpositive < 3) || (zminpositive>100)) ? 0.3*zminpositive : 1;
         if (this.zmin >= this.zmax) this.zmin = 0.0001*this.zmax;

         var logmin = Math.log(this.zmin)/Math.log(10);
         var logmax = Math.log(this.zmax)/Math.log(10);
         var dz = (logmax-logmin)/nlevels;
         this.fContour.push(this.zmin);
         for (var level=1; level<nlevels; level++)
            this.fContour.push(Math.exp((logmin + dz*level)*Math.log(10)));
         this.fContour.push(this.zmax);
         this.fCustomContour = true;
      } else {
         if ((this.zmin === this.zmax) && (this.zmin !== 0)) {
            this.zmax += 0.01*Math.abs(this.zmax);
            this.zmin -= 0.01*Math.abs(this.zmin);
         }
         var dz = (this.zmax-this.zmin)/nlevels;
         for (var level=0; level<=nlevels; level++)
            this.fContour.push(this.zmin + dz*level);
      }

      return this.fContour;
   }

   JSROOT.THistPainter.prototype.GetContour = function() {
      if (this.fContour) return this.fContour;

      var main = this.main_painter();
      if ((main !== this) && main.fContour) {
         this.fContour = main.fContour;
         this.fCustomContour = main.fCustomContour;
         this.zmin = main.zmin;
         this.zmax = main.zmax;
         return this.fContour;
      }

      // if not initialized, first create contour array
      // difference from ROOT - fContour includes also last element with maxbin, which makes easier to build logz
      var histo = this.GetObject(), nlevels = JSROOT.gStyle.fNumberContours,
          zmin = this.minbin, zmax = this.maxbin, zminpos = this.minposbin;
      if (zmin === zmax) { zmin = this.gminbin; zmax = this.gmaxbin; zminpos = this.gminposbin }
      if (histo.fContour) nlevels = histo.fContour.length;
      if ((this.histo.fMinimum != -1111) && (this.histo.fMaximum != -1111)) {
         zmin = this.histo.fMinimum;
         zmax = this.histo.fMaximum;
      }
      if (this.zoom_zmin != this.zoom_zmax) {
         zmin = this.zoom_zmin;
         zmax = this.zoom_zmax;
      }

      if (histo.fContour && (histo.fContour.length>1) && histo.TestBit(JSROOT.TH1StatusBits.kUserContour)) {
         this.fContour = JSROOT.clone(histo.fContour);
         this.fCustomContour = true;
         this.zmin = zmin;
         this.zmax = zmax;
         if (zmax > this.fContour[this.fContour.length-1]) this.fContour.push(zmax);
         return this.fContour;
      }

      this.fCustomContour = false;

      return this.CreateContour(nlevels, zmin, zmax, zminpos);
   }

   JSROOT.THistPainter.prototype.getContourIndex = function(zc) {
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
      if (zc < this.zmin) return (this.options.Color === 11) ? 0 : -1;

      // if bin content exactly zmin, draw it when col0 specified or when content is positive
      if (zc===this.zmin) return ((this.zmin != 0) || (this.options.Color === 11) || this.IsTH2Poly()) ? 0 : -1;

      return Math.floor(0.01+(zc-this.zmin)*(cntr.length-1)/(this.zmax-this.zmin));
   }

   JSROOT.THistPainter.prototype.getIndexColor = function(index, asindx) {
      if (index < 0) return null;

      var cntr = this.GetContour();

      var palette = this.GetPalette();
      var theColor = Math.floor((index+0.99)*palette.length/(cntr.length-1));
      if (theColor > palette.length-1) theColor = palette.length-1;
      return asindx ? theColor : palette[theColor];
   }

   JSROOT.THistPainter.prototype.getValueColor = function(zc, asindx) {

      var index = this.getContourIndex(zc);

      if (index < 0) return null;

      var palette = this.GetPalette();

      var theColor = Math.floor((index+0.99)*palette.length/(this.fContour.length-1));
      if (theColor > palette.length-1) theColor = palette.length-1;
      return asindx ? theColor : palette[theColor];
   }

   JSROOT.THistPainter.prototype.GetPalette = function(force) {
      if (!this.fPalette || force)
         this.fPalette = JSROOT.Painter.GetColorPalette(this.options.Palette);
      return this.fPalette;
   }

   JSROOT.THistPainter.prototype.FillPaletteMenu = function(menu) {

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

   JSROOT.THistPainter.prototype.DrawColorPalette = function(enabled, postpone_draw, can_move) {
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

   JSROOT.THistPainter.prototype.ToggleColz = function() {
      if (this.options.Zscale > 0) {
         this.options.Zscale = 0;
      } else {
         this.options.Zscale = 1;
      }

      this.DrawColorPalette(this.options.Zscale > 0, false, true);
   }


   // ==================== painter for TH2 histograms ==============================

   JSROOT.TH2Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);
      this.fContour = null; // contour levels
      this.fCustomContour = false; // are this user-defined levels (can be irregular)
      this.fPalette = null;
   }

   JSROOT.TH2Painter.prototype = Object.create(JSROOT.THistPainter.prototype);

   JSROOT.TH2Painter.prototype.Cleanup = function() {
      delete this.fCustomContour;
      delete this.tt_handle;

      JSROOT.THistPainter.prototype.Cleanup.call(this);
   }

   JSROOT.TH2Painter.prototype.FillHistContextMenu = function(menu) {
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

   JSROOT.TH2Painter.prototype.ButtonClick = function(funcname) {
      if (JSROOT.THistPainter.prototype.ButtonClick.call(this, funcname)) return true;

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

   JSROOT.TH2Painter.prototype.FillToolbar = function() {
      JSROOT.THistPainter.prototype.FillToolbar.call(this);

      var pp = this.pad_painter(true);
      if (pp===null) return;

      if (!this.IsTH2Poly())
         pp.AddButton(JSROOT.ToolbarIcons.th2color, "Toggle color", "ToggleColor");
      pp.AddButton(JSROOT.ToolbarIcons.th2colorz, "Toggle color palette", "ToggleColorZ");
      pp.AddButton(JSROOT.ToolbarIcons.th2draw3d, "Toggle 3D mode", "Toggle3D");
   }

   JSROOT.TH2Painter.prototype.ToggleColor = function() {

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

   JSROOT.TH2Painter.prototype.AutoZoom = function() {
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


   JSROOT.TH2Painter.prototype.ScanContent = function() {
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

      // used to enable/disable stat box
      this.draw_content = this.gmaxbin > 0;

/*
      // apply selected user X range if no other range selection was done
      if (this.is_main_painter() && (this.zoom_xmin === this.zoom_xmax) &&
          this.histo.fXaxis.TestBit(JSROOT.EAxisBits.kAxisRange) &&
          (this.histo.fXaxis.fFirst !== this.histo.fXaxis.fLast) &&
          ((this.histo.fXaxis.fFirst>1) || (this.histo.fXaxis.fLast <= this.nbinsx))) {
         this.zoom_xmin = this.histo.fXaxis.fFirst > 1 ? this.GetBinX(this.histo.fXaxis.fFirst-1) : this.xmin;
         this.zoom_xmax = this.histo.fXaxis.fLast <= this.nbinsx ? this.GetBinX(this.histo.fXaxis.fLast) : this.xmax;
      }

      // apply selected user Y range if no other range selection was done
      if (this.is_main_painter() && (this.zoom_ymin === this.zoom_ymax) &&
          this.histo.fYaxis.TestBit(JSROOT.EAxisBits.kAxisRange) &&
          (this.histo.fYaxis.fFirst !== this.histo.fYaxis.fLast) &&
          ((this.histo.fYaxis.fFirst>1) || (this.histo.fYaxis.fLast <= this.nbinsy))) {
         this.zoom_ymin = this.histo.fYaxis.fFirst > 1 ? this.GetBinY(this.histo.fYaxis.fFirst-1) : this.ymin;
         this.zoom_ymax = this.histo.fYaxis.fLast <= this.nbinsy ? this.GetBinY(this.histo.fYaxis.fLast) : this.ymax;
      }
*/
   }

   JSROOT.TH2Painter.prototype.CountStat = function(cond) {
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

   JSROOT.TH2Painter.prototype.FillStatistic = function(stat, dostat, dofit) {
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

   JSROOT.TH2Painter.prototype.PrepareColorDraw = function(args) {

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
            if (res.grx[i] < -this.size_xy3d) { res.i1 = i; res.grx[i] = -this.size_xy3d; }
            if (res.grx[i] > this.size_xy3d) { res.i2 = i; res.grx[i] = this.size_xy3d; }
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
            if (res.gry[j] < -this.size_xy3d) { res.j1 = j; res.gry[j] = -this.size_xy3d; }
            if (res.gry[j] > this.size_xy3d) { res.j2 = j; res.gry[j] = this.size_xy3d; }
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

      JSROOT.TH2Painter.prototype.DrawBinsColor = function(w,h) {
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

   JSROOT.TH2Painter.prototype.BuildContour = function(handle, levels, palette, call_back) {
      var histo = this.GetObject(),
          kMAXCONTOUR = 404,
          kMAXCOUNT = 400,
      // arguemnts used in he PaintContourLine
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

            if (iminus+1 < iplus)
               call_back(colindx, xp, yp, iminus, iplus, ipoly);

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

   JSROOT.TH2Painter.prototype.DrawBinsContour = function(frame_w,frame_h) {
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

   JSROOT.TH2Painter.prototype.CreatePolyBin = function(pmain, bin) {
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

   JSROOT.TH2Painter.prototype.DrawPolyBinsColor = function(w,h) {
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

   JSROOT.TH2Painter.prototype.DrawBinsText = function(w, h, handle) {
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

   JSROOT.TH2Painter.prototype.DrawBinsArrow = function(w, h) {
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


   JSROOT.TH2Painter.prototype.DrawBinsBox = function(w,h) {

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
            // area of the box should be propotional to absolute bin content
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

            if ((binz<0) && (this.options.Box === 1))
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

   JSROOT.TH2Painter.prototype.DrawCandle = function(w,h) {
      var histo = this.GetObject(),
          handle = this.PrepareColorDraw(),
          pad = this.root_pad(),
          pmain = this.main_painter(), // used for axis values conversions
          i, j, y, sum0, sum1, sum2, cont, center, counter, integral, w, pnt;

      // candle option coded into string, which comes after candle indentifier
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
            if (counter/integral < 0.75 && (counter + cont)/integral >=0.75) pnt.p25y = this.GetBinY(j + 0.5); //Uppeder edge of box
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

         //Whsikers cannot exceed 1.5*iqr from box
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

   JSROOT.TH2Painter.prototype.DrawBinsScatter = function(w,h) {
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
//              .style("fill","none")
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
            if ((binz == 0) || (binz < this.minbin)) continue;

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

   JSROOT.TH2Painter.prototype.DrawBins = function() {

      this.CheckHistDrawAttributes();

      this.RecreateDrawG(false, "main_layer");

      var w = this.frame_width(),
          h = this.frame_height(),
          handle = null;

      // if (this.lineatt.color == 'none') this.lineatt.color = 'cyan';

      if (this.options.Color + this.options.Box + this.options.Scat + this.options.Text +
          this.options.Contour + this.options.Arrow + this.options.Candle.length == 0)
         this.options.Scat = 1;

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

   JSROOT.TH2Painter.prototype.GetBinTips = function (i, j) {
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
      if (histo['$baseh']) binz -= histo['$baseh'].getBinContent(i+1,j+1);

      if (binz === Math.round(binz))
         lines.push("entries = " + binz);
      else
         lines.push("entries = " + JSROOT.FFormat(binz, JSROOT.gStyle.fStatFormat));

      return lines;
   }

   JSROOT.TH2Painter.prototype.GetCandleTips = function(p) {
      var lines = [], main = this.main_painter();

      lines.push(this.GetTipName());

      lines.push("x = " + main.AxisAsText("x", this.GetBinX(p.bin)));
      // lines.push("x = [" + main.AxisAsText("x", this.GetBinX(p.bin)) + ", " + main.AxisAsText("x", this.GetBinX(p.bin+1)) + ")");

      lines.push('mean y = ' + JSROOT.FFormat(p.meany, JSROOT.gStyle.fStatFormat))
      lines.push('m25 = ' + JSROOT.FFormat(p.m25y, JSROOT.gStyle.fStatFormat))
      lines.push('p25 = ' + JSROOT.FFormat(p.p25y, JSROOT.gStyle.fStatFormat))

      return lines;
   }

   JSROOT.TH2Painter.prototype.ProvidePolyBinHints = function(binindx, realx, realy) {

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

   JSROOT.TH2Painter.prototype.ProcessTooltip = function(pnt) {
      if ((pnt==null) || !this.draw_content || !this.draw_g || !this.tt_handle) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         this.ProvideUserTooltip(null);
         return null;
      }

      var histo = this.GetObject(),
          h = this.tt_handle, i,
          ttrect = this.draw_g.select(".tooltip_bin");

      if (h.poly) {
         // process toolltips from TH2Poly

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

         var res = { x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.color : 'blue',
                     exact: true, menu: true,
                     lines: this.ProvidePolyBinHints(foundindx, realx, realy) };

         if (ttrect.empty())
            ttrect = this.draw_g.append("svg:path")
                                .attr("class","tooltip_bin h1bin")
                                .style("pointer-events","none");

         res.changed = ttrect.property("current_bin") !== foundindx;

         if (res.changed)
            ttrect.attr("d", this.CreatePolyBin(pmain, bin))
                  .style("opacity", "0.7")
                  .property("current_bin", foundindx);

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

         var res = { x: pnt.x, y: pnt.y,
                     color1: this.lineatt ? this.lineatt.color : 'green',
                     color2: this.fillatt ? this.fillatt.color : 'blue',
                     lines: this.GetCandleTips(p), exact: true, menu: true };

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

      var res = { x: pnt.x, y: pnt.y,
                  color1: this.lineatt ? this.lineatt.color : 'green',
                  color2: this.fillatt ? this.fillatt.color : 'blue',
                  lines: this.GetBinTips(i, j), exact: true, menu: true };

      if (this.options.Color > 0) res.color2 = this.GetPalette()[colindx];

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

      if (this.IsUserTooltipCallback() && res.changed) {
         this.ProvideUserTooltip({ obj: histo,  name: histo.fName,
                                   bin: histo.getBin(i+1, j+1), cont: binz, binx: i+1, biny: j+1,
                                   grx: pnt.x, gry: pnt.y });
      }

      return res;
   }

   JSROOT.TH2Painter.prototype.CanZoomIn = function(axis,min,max) {
      // check if it makes sense to zoom inside specified axis range
      if ((axis=="x") && (this.GetIndexX(max,0.5) - this.GetIndexX(min,0) > 1)) return true;

      if ((axis=="y") && (this.GetIndexY(max,0.5) - this.GetIndexY(min,0) > 1)) return true;

      if (axis=="z") return true;

      return false;
   }

   JSROOT.TH2Painter.prototype.Draw2D = function(call_back, resize) {

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
      // if (pp) pp.WhenReady( function() { pp.DrawPave(); });
      if (pp) pp.DrawPave();

      this.DrawTitle();

      this.AddInteractive();

      JSROOT.CallBack(call_back);
   }

   JSROOT.TH2Painter.prototype.Draw3D = function(call_back) {
      this.mode3d = true;
      JSROOT.AssertPrerequisites('3d', function() {
         this.Create3DScene = JSROOT.Painter.HPainter_Create3DScene;
         this.Draw3D = JSROOT.Painter.TH2Painter_Draw3D;
         this.Draw3D(call_back);
      }.bind(this));
   }

   JSROOT.TH2Painter.prototype.CallDrawFunc = function(callback, resize) {
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

   JSROOT.TH2Painter.prototype.Redraw = function(resize) {
      this.CallDrawFunc(null, resize);
   }

   JSROOT.Painter.drawHistogram2D = function(divid, histo, opt) {
      // create painter and add it to canvas
      JSROOT.extend(this, new JSROOT.TH2Painter(histo));

      this.SetDivId(divid, 1);

      // here we deciding how histogram will look like and how will be shown
      this.options = this.DecodeOptions(opt);

      if (this.IsTH2Poly()) {
         this.options.Color = 1; // default color
         if (this.options.Lego) this.options.Lego = 12; // and lego always 12
      }

      this._show_empty_bins = false; // this.MatchObjectType('TProfile2D');

      this._can_move_colz = true;

      // special case for root 3D drawings - user range is wired
      if ((this.options.Contour !==14) && !this.options.Lego && !this.options.Surf)
         this.CheckPadRange();

      this.ScanContent();

      // check if we need to create statbox
      if (JSROOT.gStyle.AutoStat && this.create_canvas /* && !this.IsTH2Poly()*/)
         this.CreateStat();

      this.CallDrawFunc(function() {
         this.DrawNextFunction(0, function() {
            if ((this.options.Lego <= 0) && (this.options.Surf <= 0)) {
               if (this.options.AutoZoom) this.AutoZoom();
            }
            this.FillToolbar();
            this.DrawingReady();
         }.bind(this));

      }.bind(this));

      return this;
   }

   return JSROOT.Painter;

}));
