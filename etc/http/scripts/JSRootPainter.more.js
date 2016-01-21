/// @file JSRootPainter.more.js
/// Part of JavaScript ROOT graphics with more classes like TEllipse, TLine, ...
/// Such classes are rarely used and therefore loaded only on demand

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( ['d3', 'JSRootPainter'], factory );
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

      var palette = [];
      var saturation = 1, lightness = 0.5, maxHue = 280, minHue = 0, maxPretty = 50;
      for (var i = 0; i < maxPretty; ++i) {
         var hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty)) / 360.0;
         var rgbval = HLStoRGB(hue, lightness, saturation);
         palette.push(rgbval);
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
      if ((col == null) || (col==0)) col = JSROOT.gStyle.Palette;
      var stops = [ 0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000 ];
      var red, green, blue;
      switch(col) {
         // Deep Sea
         case 51:
            var red   = [ 0,  9, 13, 17, 24,  32,  27,  25,  29];
            var green = [ 0,  0,  0,  2, 37,  74, 113, 160, 221];
            var blue  = [ 28, 42, 59, 78, 98, 129, 154, 184, 221];
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

      this.ellipse = obj;
      this.SetDivId(divid);

      // function used for live update of object
      this['UpdateObject'] = function(obj) {
         // copy all fields
         JSROOT.extend(this.ellipse, obj);
      }

      this['Redraw'] = function() {
         var lineatt = JSROOT.Painter.createAttLine(this.ellipse);
         var fillatt = this.createAttFill(this.ellipse);

         // create svg:g container for ellipse drawing
         this.RecreateDrawG(this.main_painter() == null);

         var x = this.AxisToSvg("x", this.ellipse.fX1);
         var y = this.AxisToSvg("y", this.ellipse.fY1);
         var rx = this.AxisToSvg("x", this.ellipse.fX1 + this.ellipse.fR1) - x;
         var ry = y - this.AxisToSvg("y", this.ellipse.fY1 + this.ellipse.fR2);

         if ((this.ellipse.fPhimin == 0) && (this.ellipse.fPhimax == 360) && (this.ellipse.fTheta == 0)) {
            // this is simple case, which could be drawn with svg:ellipse
            this.draw_g
                .append("svg:ellipse")
                .attr("cx", x.toFixed(1)).attr("cy", y.toFixed(1))
                .attr("rx", rx.toFixed(1)).attr("ry", ry.toFixed(1))
                .call(lineatt.func).call(fillatt.func);
            return;
         }

         // here svg:path is used to draw more complex figure

         var ct = Math.cos(Math.PI*this.ellipse.fTheta/180.);
         var st = Math.sin(Math.PI*this.ellipse.fTheta/180.);

         var dx1 =  rx * Math.cos(this.ellipse.fPhimin*Math.PI/180.);
         var dy1 =  ry * Math.sin(this.ellipse.fPhimin*Math.PI/180.);
         var x1 =  dx1*ct - dy1*st;
         var y1 = -dx1*st - dy1*ct;

         var dx2 = rx * Math.cos(this.ellipse.fPhimax*Math.PI/180.);
         var dy2 = ry * Math.sin(this.ellipse.fPhimax*Math.PI/180.);
         var x2 =  dx2*ct - dy2*st;
         var y2 = -dx2*st - dy2*ct;

         this.draw_g
            .attr("transform","translate("+x.toFixed(1)+","+y.toFixed(1)+")")
            .append("svg:path")
            .attr("d", "M 0,0" +
                       " L " + x1.toFixed(1) + "," + y1.toFixed(1) +
                       " A " + rx.toFixed(1) + " " + ry.toFixed(1) + " " + -this.ellipse.fTheta.toFixed(1) + " 1 0 " + x2.toFixed(1) + "," + y2.toFixed(1) +
                       " L 0,0 Z")
            .call(lineatt.func).call(fillatt.func);
      }

      this.Redraw(); // actual drawing
      return this.DrawingReady();
   }

   // =============================================================================

   JSROOT.Painter.drawLine = function(divid, obj, opt) {

      this.line = obj;
      this.SetDivId(divid);

      // function used for live update of object
      this['UpdateObject'] = function(obj) {
         // copy all fields
         JSROOT.extend(this.line, obj);
      }

      this['Redraw'] = function() {
         var lineatt = JSROOT.Painter.createAttLine(this.line);

         // create svg:g container for line drawing
         this.RecreateDrawG(this.main_painter() == null);

         var x1 = this.AxisToSvg("x", this.line.fX1);
         var y1 = this.AxisToSvg("y", this.line.fY1);
         var x2 = this.AxisToSvg("x", this.line.fX2);
         var y2 = this.AxisToSvg("y", this.line.fY2);

         this.draw_g
             .append("svg:line")
             .attr("x1", x1.toFixed(1))
             .attr("y1", y1.toFixed(1))
             .attr("x2", x2.toFixed(1))
             .attr("y2", y2.toFixed(1))
             .call(lineatt.func);
      }

      this.Redraw(); // actual drawing
      return this.DrawingReady();
   }

   // ======================================================================================

   JSROOT.Painter.drawArrow = function(divid, obj, opt) {

      this.arrow = obj;
      this.SetDivId(divid);

      // function used for live update of object
      this['UpdateObject'] = function(obj) {
         // copy all fields
         JSROOT.extend(this.arrow, obj);
      }

      this['Redraw'] = function() {
         var lineatt = JSROOT.Painter.createAttLine(this.arrow);
         var fillatt = this.createAttFill(this.arrow);

         var wsize = Math.max(this.pad_width(), this.pad_height()) * this.arrow.fArrowSize;
         if (wsize<3) wsize = 3;
         var hsize = wsize * Math.tan(this.arrow.fAngle/2 * (Math.PI/180));

         // create svg:g container for line drawing
         this.RecreateDrawG(this.main_painter() == null);

         var x1 = this.AxisToSvg("x", this.arrow.fX1);
         var y1 = this.AxisToSvg("y", this.arrow.fY1);
         var x2 = this.AxisToSvg("x", this.arrow.fX2);
         var y2 = this.AxisToSvg("y", this.arrow.fY2);

         var right_arrow = "M0,0" + " L"+wsize.toFixed(1) +","+hsize.toFixed(1) + " L0," + (hsize*2).toFixed(1);
         var left_arrow =  "M" + wsize.toFixed(1) + ", 0" + " L 0," + hsize.toFixed(1) + " L " + wsize.toFixed(1) + "," + (hsize*2).toFixed(1);

         var m_start = null, m_mid = null, m_end = null, defs = null;

         var oo = this.arrow.fOption;
         var len = oo.length;

         if (oo.indexOf("<")==0) {
            var closed = (oo.indexOf("<|") == 0);
            if (!defs) defs = this.draw_g.append("defs");
            m_start = "jsroot_arrowmarker_" +  JSROOT.Painter['arrowcnt']++;
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
                .call(lineatt.func);
            if (closed) beg.call(fillatt.func);
         }

         var midkind = 0;
         if (oo.indexOf("->-")>=0)  midkind = 1; else
         if (oo.indexOf("-|>-")>=0) midkind = 11; else
         if (oo.indexOf("-<-")>=0) midkind = 2; else
         if (oo.indexOf("-<|-")>=0) midkind = 12;

         if (midkind > 0) {
            var closed = midkind > 10;
            if (!defs) defs = this.draw_g.append("defs");
            m_mid = "jsroot_arrowmarker_" +  JSROOT.Painter['arrowcnt']++;

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
              .call(lineatt.func);
            if (midkind > 10) mid.call(fillatt.func);
         }

         if (oo.lastIndexOf(">") == len-1) {
            var closed = (oo.lastIndexOf("|>") == len-2) && (len>1);
            if (!defs) defs = this.draw_g.append("defs");
            m_end = "jsroot_arrowmarker_" +  JSROOT.Painter['arrowcnt']++;
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
              .call(lineatt.func);
            if (closed) end.call(fillatt.func);
         }

         var path = this.draw_g
             .append("svg:path")
             .attr("d",  "M" + x1.toFixed(1) + "," + y1.toFixed(1) +
                      ((m_mid == null) ? "" : "L" + (x1/2+x2/2).toFixed(1) + "," + (y1/2+y2/2).toFixed(1)) +
                        " L" + x2.toFixed(1) + "," + y2.toFixed(1))
             .call(lineatt.func);

         if (m_start!=null) path.style("marker-start","url(#" + m_start + ")");
         if (m_mid!=null) path.style("marker-mid","url(#" + m_mid + ")");
         if (m_end!=null) path.style("marker-end","url(#" + m_end + ")");
      }

      if (!('arrowcnt' in JSROOT.Painter)) JSROOT.Painter['arrowcnt'] = 0;

      this.Redraw(); // actual drawing
      return this.DrawingReady();
   }

   // ===================================================================================

   JSROOT.Painter.drawFunction = function(divid, tf1, opt) {

      this['tf1'] = tf1;
      this['bins'] = null;

      this['GetObject'] = function() {
         return this.tf1;
      }

      this['Redraw'] = function() {
         this.DrawBins();
      }

      this['Eval'] = function(x) {
         return this.tf1.evalPar(x);
      }

      this['CreateDummyHisto'] = function() {
         var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
         if (this.tf1['fSave'].length > 0) {
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

               if (h > ymax) ymax = h;
               if (h < ymin) ymin = h;
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

      this['CreateBins'] = function() {

         var pthis = this;

         if (this.tf1['fSave'].length > 0) {
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
            var main = this.main_painter();

            if (this.tf1['fNpfits'] <= 103)
               this.tf1['fNpfits'] = 333;
            var xmin = this.tf1['fXmin'], xmax = this.tf1['fXmax'], logx = false;

            if (main['zoom_xmin'] != main['zoom_xmax']) {
               if (main['zoom_xmin'] > xmin) xmin = main['zoom_xmin'];
               if (main['zoom_xmax'] < xmax) xmax = main['zoom_xmax'];
            }

            if (main.options.Logx && (xmin>0) && (xmax>0)) {
               logx = true;
               xmin = Math.log(xmin);
               xmax = Math.log(xmax);
            }

            var nb_points = Math.max(this.tf1['fNpx'], this.tf1['fNpfits']);
            var binwidthx = (xmax - xmin) / nb_points;
            this['bins'] = d3.range(nb_points).map(function(p) {
               var xx = xmin + (p * binwidthx);
               if (logx) xx = Math.exp(xx);
               var yy = pthis.Eval(xx);
               if (isNaN(yy)) yy = 0;
               return { x : xx, y : yy };
            });

            this['interpolate_method'] = 'monotone';
         }
      }

      this['DrawBins'] = function() {
         var w = this.frame_width(), h = this.frame_height();

         this.RecreateDrawG(false, ".main_layer");

         // recalculate drawing bins when necessary
         if ((this['bins']==null) || (this.tf1['fSave'].length==0)) this.CreateBins();

         var pthis = this;
         var pmain = this.main_painter();

         var name = this.GetItemName();
         if ((name==null) || (name=="")) name = this.tf1.fName;
         if (name.length > 0) name += "\n";

         var attline = JSROOT.Painter.createAttLine(this.tf1);
         var fill = this.createAttFill(this.tf1);
         if (fill.color == 'white') fill.color = 'none';

         var line = d3.svg.line()
                      .x(function(d) { return pmain.grx(d.x).toFixed(1); })
                      .y(function(d) { return pmain.gry(d.y).toFixed(1); })
                      .interpolate(this.interpolate_method);

         var area = d3.svg.area()
                     .x(function(d) { return pmain.grx(d.x).toFixed(1); })
                     .y1(h)
                     .y0(function(d) { return pmain.gry(d.y).toFixed(1); });

         if (attline.color != "none")
            this.draw_g.append("svg:path")
               .attr("class", "line")
               .attr("d",line(pthis.bins))
               .style("fill", "none")
               .call(attline.func);

         if (fill.color != "none")
            this.draw_g.append("svg:path")
                   .attr("class", "area")
                   .attr("d",area(pthis.bins))
                   .style("stroke", "none")
                   .style("pointer-events", "none")
                   .call(fill.func);

         // add tooltips
         if (JSROOT.gStyle.Tooltip)
            this.draw_g.selectAll()
                      .data(this.bins).enter()
                      .append("svg:circle")
                      .attr("cx", function(d) { return pmain.grx(d.x).toFixed(1); })
                      .attr("cy", function(d) { return pmain.gry(d.y).toFixed(1); })
                      .attr("r", 4)
                      .style("opacity", 0)
                      .append("svg:title")
                      .text( function(d) { return name + "x = " + pmain.AxisAsText("x",d.x) + " \ny = " + pmain.AxisAsText("y", d.y); });
      }

      this['UpdateObject'] = function(obj) {
         if (obj['_typename'] != this.tf1['_typename']) return false;
         // TODO: realy update object content
         this.tf1 = obj;
         return true;
      }

      this['CanZoomIn'] = function(axis,min,max) {
         if (axis!="x") return false;

         if (this.tf1['fSave'].length > 0) {
            // in the case where the points have been saved, useful for example
            // if we don't have the user's function
            var nb_points = this.tf1['fNpx'];

            var xmin = this.tf1['fSave'][nb_points + 1];
            var xmax = this.tf1['fSave'][nb_points + 2];

            return Math.abs(xmin - xmax) / nb_points < Math.abs(min - max);
         }

         // if function calculated, one always could zoom inside
         return true;
      }

      this.SetDivId(divid, -1);
      if (this.main_painter() == null) {
         var histo = this.CreateDummyHisto();
         JSROOT.Painter.drawHistogram1D(divid, histo, "AXIS");
      }
      this.SetDivId(divid);
      this.DrawBins();
      return this.DrawingReady();
   }

   // ====================================================================

   JSROOT.Painter.drawHStack = function(divid, stack, opt) {
      // paint the list of histograms
      // By default, histograms are shown stacked.
      // -the first histogram is paint
      // -then the sum of the first and second, etc

      // 'this' pointer set to created painter instance
      this.stack = stack;
      this.nostack = false;
      this.firstpainter = null;
      this.painters = new Array; // keep painters to be able update objects

      this.SetDivId(divid);

      if (!('fHists' in stack) || (stack['fHists'].arr.length == 0)) return this.DrawingReady();

      this['GetObject'] = function() {
         return this.stack;
      }

      this['BuildStack'] = function() {
         //  build sum of all histograms
         //  Build a separate list fStack containing the running sum of all histograms

         if (!('fHists' in this.stack)) return false;
         var nhists = this.stack['fHists'].arr.length;
         if (nhists <= 0) return false;
         var lst = JSROOT.Create("TList");
         lst.Add(JSROOT.clone(this.stack['fHists'].arr[0]));
         for (var i=1;i<nhists;++i) {
            var hnext = JSROOT.clone(this.stack['fHists'].arr[i]);
            var hprev = lst.arr[i-1];

            if ((hnext.fNbins != hprev.fNbins) ||
                (hnext['fXaxis']['fXmin'] != hprev['fXaxis']['fXmin']) ||
                (hnext['fXaxis']['fXmax'] != hprev['fXaxis']['fXmax'])) {
               JSROOT.console("When drawing THStack, cannot sum-up histograms " + hnext.fName + " and " + hprev.fName);
               delete hnext;
               delete lst;
               return false;
            }

            // trivial sum of histograms
            for (var n = 0; n < hnext.fArray.length; ++n)
               hnext.fArray[n] += hprev.fArray[n];

            lst.Add(hnext);
         }
         this.stack['fStack'] = lst;
         return true;
      }

      this['GetHistMinMax'] = function(hist, witherr) {
         var res = { min : 0, max : 0 };
         var domin = false, domax = false;
         if (hist['fMinimum'] != -1111)
            res.min = hist['fMinimum'];
         else
            domin = true;
         if (hist['fMaximum'] != -1111)
            res.max = hist['fMaximum'];
         else
            domax = true;

         if (domin || domax) {
            var left = 1, right = hist.fXaxis.fNbins;

            if (hist.fXaxis.TestBit(JSROOT.EAxisBits.kAxisRange)) {
               left = hist.fXaxis.fFirst;
               right = hist.fXaxis.fLast;
            }
            for (var bin = left; bin<=right; ++bin) {
               var val = hist.getBinContent(bin);
               var err = witherr ? hist.getBinError(bin) : 0;
               if (domin && ((bin==left) || (val-err < res.min))) res.min = val-err;
               if (domax && ((bin==left) || (val+err > res.max))) res.max = val+err;
            }
         }

         return res;
      }

      this['GetMinMax'] = function(opt) {
         var res = { min : 0, max : 0 };
         var iserr = opt.indexOf('e')>=0;

         if (this.nostack) {
            for (var i = 0; i < this.stack['fHists'].arr.length; ++i) {
               var resh = this.GetHistMinMax(this.stack['fHists'].arr[i], iserr);
               if (i==0) res = resh; else {
                  if (resh.min < res.min) res.min = resh.min;
                  if (resh.max > res.max) res.max = resh.max;
               }
            }

            if (this.stack['fMaximum'] != -1111)
               res.max = this.stack['fMaximum'];
            else
               res.max = res.max * 1.05;

            if (this.stack['fMinimum'] != -1111) res.min = this.stack['fMinimum'];
         } else {
            res.min = this.GetHistMinMax(this.stack['fStack'].arr[0], iserr).min;
            res.max = this.GetHistMinMax(this.stack['fStack'].arr[this.stack['fStack'].arr.length-1], iserr).max * 1.05;
         }

         var pad = this.root_pad();
         if ((pad!=null) && (pad['fLogy']>0)) {
            if (res.min<0) res.min = res.max * 1e-4;
         }

         return res;
      }

      this['DrawNextHisto'] = function(indx, opt) {
         var nhists = this.stack['fHists'].arr.length;
         if (indx>=nhists) return this.DrawingReady();

         var hist = null;
         if (indx<0) hist = this.stack['fHistogram']; else
         if (this.nostack) hist = this.stack['fHists'].arr[indx];
                     else  hist = this.stack['fStack'].arr[nhists - indx - 1];

         var hopt = hist['fOption'];
         if ((opt != "") && (hopt.indexOf(opt) == -1)) hopt += opt;
         if (indx>=0) hopt += "same";
         var subp = JSROOT.draw(this.divid, hist, hopt);
         if (indx<0) this.firstpainter = subp;
                else this.painters.push(subp);
         subp.WhenReady(this.DrawNextHisto.bind(this, indx+1, opt));
      }

      this['drawStack'] = function(opt) {
         var pad = this.root_pad();
         var histos = this.stack['fHists'];
         var nhists = histos.arr.length;
         var pad = this.root_pad();

         if (opt == null) opt = "";
                     else opt = opt.toLowerCase();

         var lsame = false;
         if (opt.indexOf("same") != -1) {
            lsame = true;
            opt.replace("same", "");
         }
         this.nostack = opt.indexOf("nostack") < 0 ? false : true;

         // when building stack, one could fail to sum up histograms
         if (!this.nostack)
            this.nostack = ! this.BuildStack();

         var mm = this.GetMinMax(opt);

         if (this.stack['fHistogram'] == null) {
            // compute the min/max of each axis
            var xmin = 0, xmax = 0, ymin = 0, ymax = 0;
            for (var i = 0; i < nhists; ++i) {
               var h = histos.arr[i];
               if (i == 0 || h['fXaxis']['fXmin'] < xmin)
                  xmin = h['fXaxis']['fXmin'];
               if (i == 0 || h['fXaxis']['fXmax'] > xmax)
                  xmax = h['fXaxis']['fXmax'];
               if (i == 0 || h['fYaxis']['fXmin'] < ymin)
                  ymin = h['fYaxis']['fXmin'];
               if (i == 0 || h['fYaxis']['fXmax'] > ymax)
                  ymax = h['fYaxis']['fXmax'];
            }

            var h = this.stack['fHists'].arr[0];
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
            if (pad && pad['fLogy'])
                histo['fMaximum'] = mm.max * (1 + 0.2 * JSROOT.log10(mm.max / mm.min));
             else
                histo['fMaximum'] = mm.max;
            if (pad && pad['fLogy'])
               histo['fMinimum'] = mm.min / (1 + 0.5 * JSROOT.log10(mm.max / mm.min));
            else
               histo['fMinimum'] = mm.min;
         }

         console.log('Drawing histograms ' + !lsame);

         this.DrawNextHisto(!lsame ? -1 : 0, opt);
         return this;
      }

      this['UpdateObject'] = function(obj) {
         var isany = false;
         if (this.firstpainter)
            if (this.firstpainter.UpdateObject(obj['fHistogram'])) isany = true;

         var nhists = obj['fHists'].arr.length;
         for (var i = 0; i < nhists; ++i) {
            var hist = this.nostack ? obj['fHists'].arr[i] : obj['fStack'].arr[nhists - i - 1];
            if (this.painters[i].UpdateObject(hist)) isany = true;
         }

         return isany;
      }

      return this.drawStack(opt);
   }

   // =============================================================

   JSROOT.Painter.drawMultiGraph = function(divid, mgraph, opt) {
      // function call with bind(painter)

      this.mgraph = mgraph;

      this.firstpainter = null;
      this.autorange = false;
      this.painters = new Array; // keep painters to be able update objects

      this.SetDivId(divid, -1); // it may be no element to set divid

      this['GetObject'] = function() {
         return this.mgraph;
      }

      this['UpdateObject'] = function(obj) {

         if ((obj==null) || (obj['_typename'] != 'TMultiGraph')) return false;

         this.mgraph.fTitle = obj.fTitle;
         var graphs = obj['fGraphs'];

         var isany = false;
         if (this.firstpainter) {
            var histo = obj['fHistogram'];
            if (this.autorange && (histo == null))
               histo = this.ScanGraphsRange(graphs);
            if (this.firstpainter.UpdateObject(histo)) isany = true;
         }

         for (var i = 0; i <  graphs.arr.length; ++i) {
            if (i<this.painters.length)
               if (this.painters[i].UpdateObject(graphs.arr[i])) isany = true;
         }

         return isany;
      }

      this['ComputeGraphRange'] = function(res, gr) {
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

      this['padtoX'] = function(pad, x) {
         // Convert x from pad to X.
         if (pad['fLogx'] && x < 50)
            return Math.exp(2.302585092994 * x);
         return x;
      }

      this['ScanGraphsRange'] = function(graphs, histo, pad) {
         var maximum, minimum, dx, dy;
         var uxmin = 0, uxmax = 0;
         var logx = false, logy = false;
         var rw = {  xmin: 0, xmax: 0, ymin: 0, ymax: 0, first: true };

         if (pad!=null) {
            logx = pad['fLogx'];
            logy = pad['fLogy'];
            rw.xmin = pad.fUxmin;
            rw.xmax = pad.fUxmax;
            rw.ymin = pad.fUymin;
            rw.ymax = pad.fUymax;
            rw.first = false;
         }
         if (histo!=null) {
            minimum = histo['fYaxis']['fXmin'];
            maximum = histo['fYaxis']['fXmax'];
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

         if (this.mgraph['fMinimum'] != -1111)
            rw.ymin = minimum = this.mgraph['fMinimum'];
         if (this.mgraph['fMaximum'] != -1111)
            rw.ymax = maximum = this.mgraph['fMaximum'];

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
            histo['fTitle'] = this.mgraph['fTitle'];
            histo['fXaxis']['fXmin'] = uxmin;
            histo['fXaxis']['fXmax'] = uxmax;
         }

         histo['fYaxis']['fXmin'] = minimum;
         histo['fYaxis']['fXmax'] = maximum;

         return histo;
      }

      this['DrawAxis'] = function() {
         // draw special histogram

         var histo = this.ScanGraphsRange(this.mgraph.fGraphs, this.mgraph.fHistogram, this.root_pad());

         // histogram painter will be first in the pad, will define axis and
         // interactive actions
         this.firstpainter = JSROOT.Painter.drawHistogram1D(this.divid, histo, "AXIS");
      }

      this['DrawNextFunction'] = function(indx, callback) {
         // method draws next function from the functions list

         if ((this.mgraph['fFunctions'] == null) || (indx >= this.mgraph.fFunctions.arr.length))
            return JSROOT.CallBack(callback);

         var func = this.mgraph.fFunctions.arr[indx];
         var opt = this.mgraph.fFunctions.opt[indx];

         var painter = JSROOT.draw(this.divid, func, opt);
         if (painter) return painter.WhenReady(this.DrawNextFunction.bind(this, indx+1, callback));

         this.DrawNextFunction(indx+1, callback);
      }

      this['DrawNextGraph'] = function(indx, opt) {
         var graphs = this.mgraph['fGraphs'];
         // at the end of graphs drawing draw functions (if any)
         if (indx >= graphs.arr.length)
            return this.DrawNextFunction(0, this.DrawingReady.bind(this));

         var drawopt = graphs.opt[indx];
         if ((drawopt==null) || (drawopt == "")) drawopt = opt;
         var subp = JSROOT.Painter.drawGraph(this.divid, graphs.arr[indx], drawopt);
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

      this.legend = obj;
      this.SetDivId(divid);

      this['GetObject'] = function() {
         return this.legend;
      }

      this['Redraw'] = function() {
         this.RecreateDrawG(true, ".text_layer");

         var pave = this.legend;

         var x = 0, y = 0, w = 0, h = 0;

         if (pave['fInit'] == 0) {
            x = pave['fX1'] * this.pad_width();
            y = (1 - pave['fY2']) * this.pad_height();
            w = (pave['fX2'] - pave['fX1']) * this.pad_width();
            h = (pave['fY2'] - pave['fY1']) * this.pad_height();
         } else {
            x = pave['fX1NDC'] * this.pad_width();
            y = (1 - pave['fY2NDC']) * this.pad_height();
            w = (pave['fX2NDC'] - pave['fX1NDC']) * this.pad_width();
            h = (pave['fY2NDC'] - pave['fY1NDC']) * this.pad_height();
         }
         var lwidth = pave['fBorderSize'] ? pave['fBorderSize'] : 0;
         var boxfill = this.createAttFill(pave);
         var lineatt = JSROOT.Painter.createAttLine(pave, lwidth);
         var ncols = pave.fNColumns, nlines = pave.fPrimitives.arr.length;
         var nrows = nlines;
         if (ncols>1) { while ((nrows-1)*ncols>=nlines) nrows--; } else ncols = 1;

         this.draw_g.attr("x", x.toFixed(1))
                    .attr("y", y.toFixed(1))
                    .attr("width", w.toFixed(1))
                    .attr("height", h.toFixed(1))
                    .attr("transform", "translate(" + x.toFixed(1) + "," + y.toFixed(1) + ")");

         this.StartTextDrawing(pave['fTextFont'], h / (nlines * 1.2));

         this.draw_g
              .append("svg:rect")
              .attr("x", 0)
              .attr("y", 0)
              .attr("width", w.toFixed(1))
              .attr("height", h.toFixed(1))
              .call(boxfill.func)
              .style("stroke-width", lwidth ? 1 : 0)
              .style("stroke", lineatt.color);

         var tcolor = JSROOT.Painter.root_colors[pave['fTextColor']];
         var column_width = Math.round(w/ncols);
         var padding_x = Math.round(0.03 * w/ncols);
         var padding_y = Math.round(0.03 * h);

         var leg_painter = this;
         var step_y = (h - 2*padding_y)/nrows;
         var any_opt = false;

         for (var i = 0; i < nlines; ++i) {
            var leg = pave.fPrimitives.arr[i];
            var lopt = leg['fOption'].toLowerCase();

            var icol = i % ncols, irow = (i - icol) / ncols;

            var x0 = icol * column_width;
            var tpos_x = x0 + Math.round(pave['fMargin']*column_width);

            var pos_y = Math.round(padding_y + irow*step_y); // top corner
            var mid_y = Math.round(padding_y + (irow+0.5)*step_y); // center line

            var attfill = leg;
            var attmarker = leg;
            var attline = leg;

            var mo = leg['fObject'];

            if ((mo != null) && (typeof mo == 'object')) {
               if ('fLineColor' in mo) attline = mo;
               if ('fFillColor' in mo) attfill = mo;
               if ('fMarkerColor' in mo) attmarker = mo;
            }

            var fill = this.createAttFill(attfill);
            var llll = JSROOT.Painter.createAttLine(attline);

            // Draw fill pattern (in a box)
            if (lopt.indexOf('f') != -1) {
               // box total height is yspace*0.7
               // define x,y as the center of the symbol for this entry
               this.draw_g.append("svg:rect")
                      .attr("x", x0 + padding_x)
                      .attr("y", Math.round(pos_y+step_y*0.1))
                      .attr("width", tpos_x - 2*padding_x - x0)
                      .attr("height", Math.round(step_y*0.8))
                      .call(fill.func);
            }

            // Draw line
            if (lopt.indexOf('l') != -1) {
               this.draw_g.append("svg:line")
                  .attr("x1", x0 + padding_x)
                  .attr("y1", mid_y)
                  .attr("x2", tpos_x - padding_x)
                  .attr("y2", mid_y)
                  .call(llll.func);
            }
            // Draw error only
            if (lopt.indexOf('e') != -1  && (lopt.indexOf('l') == -1 || lopt.indexOf('f') != -1)) {
            }
            // Draw Polymarker
            if (lopt.indexOf('p') != -1) {
               var marker = JSROOT.Painter.createAttMarker(attmarker);
               this.draw_g.append(marker.kind)
                   .attr("transform", function(d) { return "translate(" + (x0 + tpos_x)/2 + "," + mid_y + ")"; })
                   .call(marker.func);
            }

            var pos_x = tpos_x;
            if (lopt.length>0) any_opt = true;
                          else if (!any_opt) pos_x = x0 + padding_x;

            this.DrawText("start", pos_x, pos_y, x0+column_width-pos_x-padding_x, step_y, leg['fLabel'], tcolor);
         }

         if (lwidth && lwidth > 1) {
            this.draw_g.append("svg:line")
               .attr("x1", w + (lwidth / 2))
               .attr("y1", lwidth + 1)
               .attr("x2", w + (lwidth / 2))
               .attr("y2",  h + lwidth - 1)
               .call(lineatt.func);
            this.draw_g.append("svg:line")
               .attr("x1", lwidth + 1)
               .attr("y1", h + (lwidth / 2))
               .attr("x2", w + lwidth - 1)
               .attr("y2", h + (lwidth / 2))
               .call(lineatt.func);
         }

         this.AddDrag({ obj:pave, redraw: this.Redraw.bind(this) });

         // rescale after all entries are shown
         this.FinishTextDrawing();
      }

      this.Redraw();

      return this.DrawingReady();
   }

   // ===========================================================================

   JSROOT.Painter.drawPaletteAxis = function(divid, palette) {

      this.palette = palette;
      this.SetDivId(divid);

      this['GetObject'] = function() {
         return this.palette;
      }

      this['DrawPalette'] = function() {
         var palette = this.palette;
         var axis = palette['fAxis'];

         var nbr1 = axis['fNdiv'] % 100;
         if (nbr1<=0) nbr1 = 8;

         var width = this.pad_width(), height = this.pad_height();

         var s_height = Math.round(Math.abs(palette['fY2NDC'] - palette['fY1NDC']) * height);

         var axisOffset = axis['fLabelOffset'] * width;
         var tickSize = axis['fTickSize'] * width;

         // force creation of contour array if missing
         if (this.main_painter().fContour == null)
            this.main_painter().getValueColor(this.main_painter().minbin);

         var contour = this.main_painter().fContour;
         var zmin = contour[0], zmax = contour[contour.length-1];

         var z = null;

         if (this.main_painter().options.Logz) {
            z = d3.scale.log();
            this['noexpz'] = ((zmax < 300) && (zmin > 0.3));

            this['formatz'] = function(d) {
               var val = parseFloat(d);
               var vlog = JSROOT.log10(val);
               if (Math.abs(vlog - Math.round(vlog))<0.001) {
                  if (!this['noexpz'])
                     return JSROOT.Painter.formatExp(val.toExponential(0));
                  else
                  if (vlog<0)
                     return val.toFixed(Math.round(-vlog+0.5));
                  else
                     return val.toFixed(0);
               }
               return null;
            }

         } else {
            z = d3.scale.linear();
            this['formatz'] = function(d) {
               if ((Math.abs(d) < 1e-14) && (Math.abs(zmax - zmin) > 1e-5)) d = 0;
               return parseFloat(d.toPrecision(12));
            }
         }
         z.domain([zmin, zmax]).range([s_height,0]);

         var labelfont = JSROOT.Painter.getFontDetails(axis['fLabelFont'], axis['fLabelSize'] * height);

         var pos_x = Math.round(palette['fX1NDC'] * width);
         var pos_y = Math.round(height*(1 - palette['fY1NDC']));

         var s_width = Math.round(Math.abs(palette['fX2NDC'] - palette['fX1NDC']) * width);
         pos_y -= s_height;
         if (tickSize > s_width*0.8) tickSize = s_width*0.8;

         // Draw palette pad
         this.RecreateDrawG(true, ".text_layer");

         this.draw_g
                .attr("x", pos_x).attr("y", pos_y)               // position required only for drag functions
                .attr("width", s_width).attr("height", s_height) // dimension required only for drag functions
                .attr("transform", "translate(" + pos_x + ", " + pos_y + ")");

         for (var i=0;i<contour.length-1;++i) {
            var z0 = z(contour[i]);
            var z1 = z(contour[i+1]);
            var col = this.main_painter().getValueColor(contour[i]);
            var r = this.draw_g
               .append("svg:rect")
               .attr("x", 0)
               .attr("y",  z1.toFixed(1))
               .attr("width", s_width)
               .attr("height", (z0-z1).toFixed(1))
               .attr("fill", col)
               .property("fill0", col)
               .attr("stroke", col);

            if (JSROOT.gStyle.Tooltip)
               r.on('mouseover', function() {
                  if (JSROOT.gStyle.Tooltip)
                     d3.select(this).transition().duration(100).style("fill", 'grey');
               }).on('mouseout', function() {
                  d3.select(this).transition().duration(100).style("fill", this['fill0']);
               }).append("svg:title").text(contour[i].toFixed(2) + " - " + contour[i+1].toFixed(2));
         }

         // Build and draw axes
         var z_axis = d3.svg.axis().scale(z)
                       .orient("right")
                       .tickPadding(axisOffset)
                       .tickSize(-tickSize, -tickSize / 2, 0)
                       .ticks(nbr1)
                       .tickFormat(this.formatz.bind(this));

         var zax = this.draw_g.append("svg:g")
                      .attr("class", "zaxis")
                      .attr("transform", "translate(" + s_width + ", 0)")
                      .call(z_axis);

         zax.selectAll("text")
                 .call(labelfont.func)
                 .attr("fill", JSROOT.Painter.root_colors[axis['fLabelColor']]);

         /** Add palette axis title */
         if ((axis['fTitle'] != "") && (typeof axis['fTextFont'] != 'undefined')) {
            // offest in width of colz drawings
            var xoffset = axis['fTitleOffset'] * s_width;
            if ('getBoundingClientRect' in this.draw_g.node()) {
               var rect1 = this.draw_g.node().getBoundingClientRect();
               // offset in portion of real text width produced by axis
               xoffset = axis['fTitleOffset'] * (rect1.width-s_width);
            }
            // add font size
            xoffset += s_width + axis['fTitleSize'] * height * 1.3;
            if (pos_x + xoffset > width-3) xoffset = width - 3 - pos_x;
            var tcolor = JSROOT.Painter.root_colors[axis['fTextColor']];
            this.StartTextDrawing(axis['fTextFont'], axis['fTitleSize'] * height);
            this.DrawText(33, 0, xoffset, 0, -270, axis['fTitle'], tcolor);
            this.FinishTextDrawing();
         }

         this.AddDrag({ obj: palette, redraw: this.DrawPalette.bind(this), ctxmenu : JSROOT.touches && JSROOT.gStyle.ContextMenu });

         if (JSROOT.gStyle.ContextMenu && !JSROOT.touches)
            this.draw_g.on("contextmenu", this.ShowContextMenu.bind(this) );

         if (!JSROOT.gStyle.Zooming) return;

         var pthis = this, evnt = null, doing_zoom = false, sel1 = 0, sel2 = 0, zoom_rect = null;

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
            // d3.select(window).on("touchmove.zoomRect",
            // null).on("touchend.zoomRect", null);
            d3.select(window).on("mousemove.colzoomRect", null)
                             .on("mouseup.colzoomRect", null);
            // d3.select("body").classed("noselect", false);
            // d3.select("body").style("-webkit-user-select", "auto");
            zoom_rect.remove();
            zoom_rect = null;
            doing_zoom = false;

            var zmin = Math.min(z.invert(sel1), z.invert(sel2));
            var zmax = Math.max(z.invert(sel1), z.invert(sel2));

            pthis.main_painter().Zoom(0, 0, 0, 0, zmin, zmax);
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

         this.draw_g.append("svg:rect")
                    .attr("x", s_width)
                    .attr("y", 0)
                    .attr("width", 20)
                    .attr("height", s_height)
                    .style("cursor", "crosshair")
                    .style("opacity", "0")
                    .on("mousedown", startRectSel);
      }

      this['ShowContextMenu'] = function(kind, evnt) {
         this.main_painter().ShowContextMenu("z", evnt);
      }

      this['Redraw'] = function() {
         var enabled = true;
         if ('options' in this.main_painter())
            enabled = (this.main_painter().options.Zscale > 0) && (this.main_painter().options.Color > 0);

         if (enabled)
            this.DrawPalette();
         else
            this.RemoveDrawG(); // if palette artificially disabled, do not redraw it
      }

      this.DrawPalette();

      return this.DrawingReady();
   }

   // ==================== painter for TH2 histograms ==============================

   JSROOT.TH2Painter = function(histo) {
      JSROOT.THistPainter.call(this, histo);
      this.fContour = null; // contour levels
      this.fUserContour = false; // are this user-defined levels
      this.fPalette = null;
   }

   JSROOT.TH2Painter.prototype = Object.create(JSROOT.THistPainter.prototype);

   JSROOT.TH2Painter.prototype.FillContextMenu = function(menu) {
      JSROOT.THistPainter.prototype.FillContextMenu.call(this, menu);
      if (this.options.Lego > 0) {
         menu.add("Draw in 2D", function() { this.options.Lego = 0; this.Redraw(); });
      } else {
         menu.add("Auto zoom-in", function() { this.AutoZoom(); });
         menu.add("Draw in 3D", function() { this.options.Lego = 1; this.Redraw(); });
         menu.add("Toggle col", function() {
            if (this.options.Color == 0)
               this.options.Color = JSROOT.gStyle.DefaultCol;
            else
               this.options.Color = - this.options.Color;
            this.RedrawPad();
         });
         if (this.options.Color > 0)
            menu.add("Toggle colz", this.ToggleColz.bind(this));
      }
   }

   JSROOT.TH2Painter.prototype.FindPalette = function(remove) {
      if ('fFunctions' in this.histo)
         for (var i = 0; i < this.histo.fFunctions.arr.length; ++i) {
            var func = this.histo.fFunctions.arr[i];
            if (func['_typename'] != 'TPaletteAxis') continue;
            if (remove) {
               this.histo.fFunctions.RemoveAt(i);
               return null;
            }

            return func;
         }

      return null;
   }

   JSROOT.TH2Painter.prototype.ToggleColz = function() {
      if (this.FindPalette() == null) {
         var shrink = this.CreatePalette(0.04);
         this.svg_frame().property('frame_painter').Shrink(0, shrink);
         this.options.Zscale = 1;
         // one should draw palette
         JSROOT.draw(this.divid, this.FindPalette());
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

      // first find minimum
      var min = this.histo.getBinContent(i1 + 1, j1 + 1);
      for (var i = i1; i < i2; ++i)
         for (var j = j1; j < j2; ++j)
            if (this.histo.getBinContent(i + 1, j + 1) < min)
               min = this.histo.getBinContent(i + 1, j + 1);
      if (min>0) return; // if all points positive, no chance for autoscale

      var ileft = i2, iright = i1, jleft = j2, jright = j1;

      for (var i = i1; i < i2; ++i)
         for (var j = j1; j < j2; ++j)
            if (this.histo.getBinContent(i + 1, j + 1) > min) {
               if (i < ileft) ileft = i;
               if (i >= iright) iright = i + 1;
               if (j < jleft) jleft = j;
               if (j >= jright) jright = j + 1;
            }

      var xmin = 0, xmax = 0, ymin = 0, ymax = 0;

      if ((ileft > i1 || iright < i2) && (ileft < iright - 1)) {
         xmin = this.GetBinX(ileft);
         xmax = this.GetBinX(iright);
      }

      if ((jleft > j1 || jright < j2) && (jleft < jright - 1)) {
         ymin = this.GetBinY(jleft);
         ymax = this.GetBinY(jright);
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

      var ndc = this.svg_frame().property('NDC');

      pal['fX1NDC'] = ndc.fX2NDC - rel_width;
      pal['fY1NDC'] = ndc.fY1NDC;
      pal['fX2NDC'] = ndc.fX2NDC;
      pal['fY2NDC'] = ndc.fY2NDC;
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
      axis['fTitle'] = this.histo.fZaxis.fTitle;
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
      this.histo.fFunctions.AddFirst(pal);

      // and at the end try to check how much place will be used by the labels
      // in the palette

      var width = this.frame_width(), height = this.frame_height();

      var axisOffset = Math.round(axis['fLabelOffset'] * width);
      var tickSize = Math.round(axis['fTickSize'] * width);
      var axisfont = JSROOT.Painter.getFontDetails(axis['fLabelFont'], axis['fLabelSize'] * height);

      var ticks = d3.scale.linear().clamp(true)
                  .domain([ this.gminbin, this.gmaxbin ])
                  .range([ height, 0 ]).nice().ticks(axis['fNdiv'] % 100);

      var maxlen = 0;
      for (var i = 0; i < ticks.length; ++i) {
         var len = axisfont.stringWidth(this.svg_frame(), ticks[i]);
         if (len > maxlen) maxlen = len;
      }

      var rel = (maxlen + 5 + axisOffset) / width;

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
      // if (this.histo['fFillColor'] == 0) this.fillcolor = '#4572A7'; // why?

      this.attline = JSROOT.Painter.createAttLine(this.histo);
      if (this.attline.color == 'none') this.attline.color = '#4572A7';

      this.nbinsx = this.histo['fXaxis']['fNbins'];
      this.nbinsy = this.histo['fYaxis']['fNbins'];

      // used in CreateXY method

      this.CreateAxisFuncs(true);

      // global min/max, used at the moment in 3D drawing
      this.gminbin = this.gmaxbin = this.histo.getBinContent(1, 1);
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
      var stat_sum0 = 0, stat_sumx1 = 0, stat_sumy1 = 0, stat_sumx2 = 0, stat_sumy2 = 0, stat_sumxy = 0;

      var res = { entries: 0, integral: 0, meanx: 0, meany: 0, rmsx: 0, rmsy: 0, matrix : [], xmax: 0, ymax:0, wmax: null };
      for (var n = 0; n < 9; ++n) res.matrix.push(0);

      var xleft = this.GetSelectIndex("x", "left");
      var xright = this.GetSelectIndex("x", "right");

      var yleft = this.GetSelectIndex("y", "left");
      var yright = this.GetSelectIndex("y", "right");

      for (var xi = 0; xi <= this.nbinsx + 1; ++xi) {
         var xside = (xi <= xleft) ? 0 : (xi > xright ? 2 : 1);
         var xx = this.GetBinX(xi - 0.5);

         for (var yi = 0; yi <= this.nbinsy + 1; ++yi) {
            var yside = (yi <= yleft) ? 0 : (yi > yright ? 2 : 1);
            var yy = this.ymin + this.GetBinY(yi - 0.5);

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
            stat_sumxy += xx * yy * zz;
         }
      }

      if (!this.IsAxisZoomed("x") && !this.IsAxisZoomed("y") && (this.histo.fTsumw>0)) {
         stat_sum0 = this.histo.fTsumw;
         stat_sumx1 = this.histo.fTsumwx;
         stat_sumx2 = this.histo.fTsumwx2;
         stat_sumy1 = this.histo.fTsumwy;
         stat_sumy2 = this.histo.fTsumwy2;
         stat_sumxy = this.histo.fTsumwxy;
      }

      if (stat_sum0 > 0) {
         res.meanx = stat_sumx1 / stat_sum0;
         res.meany = stat_sumy1 / stat_sum0;
         res.rmsx = Math.sqrt(stat_sumx2 / stat_sum0 - res.meanx * res.meanx);
         res.rmsy = Math.sqrt(stat_sumy2 / stat_sum0 - res.meany * res.meany);
      }

      if (res.wmax==null) res.wmax = 0;
      res.integral = stat_sum0;

      if (this.histo.fEntries > 1) res.entries = this.histo.fEntries;

      return res;
   }

   JSROOT.TH2Painter.prototype.FillStatistic = function(stat, dostat, dofit) {
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
         stat.AddLine("Entries = " + stat.Format(data.entries,"entries"));

      if (print_mean > 0) {
         stat.AddLine("Mean x = " + stat.Format(data.meanx));
         stat.AddLine("Mean y = " + stat.Format(data.meany));
      }

      if (print_rms > 0) {
         stat.AddLine("Std Dev x = " + stat.Format(data.rmsx));
         stat.AddLine("Std Dev y = " + stat.Format(data.rmsy));
      }

      if (print_integral > 0) {
         stat.AddLine("Integral = " + stat.Format(data.matrix[4],"entries"));
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
      if (this.fContour == null) {
         // if not initialized, first create controur array
         // difference from ROOT - fContour includes also last element with maxbin, which makes easier to build logz
         this.fUserContour = false;
         if ((this.histo.fContour!=null) && (this.histo.fContour.length>1) && this.histo.TestBit(JSROOT.TH1StatusBits.kUserContour)) {
            this.fContour = JSROOT.clone(this.histo.fContour);
            this.fUserContour = true;
         } else {
            var nlevels = 20;
            if (this.histo.fContour != null) nlevels = this.histo.fContour.length;
            if (nlevels<1) nlevels = 20;
            this.fContour = [];
            this.zmin = this.minbin;
            this.zmax = this.maxbin;
            if (this.zoom_zmin != this.zoom_zmax) {
               this.zmin = this.zoom_zmin;
               this.zmax = this.zoom_zmax;
            }

            if (this.options.Logz) {
               if (this.zmax <= 0) this.zmax = 1.;
               if (this.zmin <= 0) this.zmin = 0.001*this.zmax;
               var logmin = Math.log(this.zmin)/Math.log(10);
               var logmax = Math.log(this.zmax)/Math.log(10);
               var dz = (logmax-logmin)/nlevels;
               this.fContour.push(this.zmin);
               for (var level=1; level<nlevels; level++)
                  this.fContour.push(Math.exp((logmin + dz*level)*Math.log(10)));
               this.fContour.push(this.zmax);
            } else {
               if ((this.zmin == this.zmax) && (this.zmin != 0)) {
                  this.zmax += 0.01*Math.abs(this.zmax);
                  this.zmin -= 0.01*Math.abs(this.zmin);
               }
               var dz = (this.zmax-this.zmin)/nlevels;
               for (var level=0; level<=nlevels; level++)
                  this.fContour.push(this.zmin + dz*level);
            }
         }
      }

      var color = -1;
      if (this.fUserContour || this.options.Logz) {
         for (var k = 0; k < this.fContour.length; ++k) {
            if (zc >= this.fContour[k]) color++;
         }
      } else {
         color = Math.floor(0.01+(zc-this.zmin)*(this.fContour.length-1)/(this.zmax-this.zmin));
      }

      if (color<0) {
      // do not draw bin where color is negative
         if (this.options.Color != 111) return null;
         color = 0;
      }

      if (this.fPalette == null)
         this.fPalette = JSROOT.Painter.GetColorPalette(this.options.Palette);

      var theColor = Math.floor((color+0.99)*this.fPalette.length/(this.fContour.length-1));
      if (theColor > this.fPalette.length-1) theColor = this.fPalette.length-1;
      return this.fPalette[theColor];
   }

   JSROOT.TH2Painter.prototype.CompressAxis = function(arr, maxlen, regular) {
      if (arr.length <= maxlen) return;

      // check filled bins
      var left = 0, right = arr.length-2;
      while ((left < right) && (arr[left].cnt===0)) ++left;
      while ((left < right) && (arr[right].cnt===0)) --right;
      if ((left==right) || (right-left < maxlen)) return;

      function RemoveNulls() {
         var j = right;
         while (j>left) {
            while ((j>left) && (arr[j]!=null)) --j;
            var j2 = j;
            while ((j>0) && (arr[j]==null)) --j;
            if (j < j2) arr.splice(j+1, j2-j);
            --j;
         }
      };

      if (!regular) {
         var grdist = Math.abs(arr[right+1].gr - arr[left].gr) / maxlen;
         var i = 0;
         while (i <= right) {
            var gr0 = arr[i++].gr;
            // remove points which are not far away from current
            while ((i <= right) && (Math.abs(arr[i+1].gr - gr0) < grdist)) arr[i++] = null;
         }
         RemoveNulls();
      }

      if (regular || ((right-left) > 1.5*maxlen)) {
         // just remove regular number of bins
         var period = Math.floor((right-left) / maxlen);
         if (period<2) period = 2;
         var i = left;
         while (++i <= right) {
            for (var k=1;k<period;++k)
               if (++i <= right) arr[i] = null;
         }
         RemoveNulls();
      }
   }

   JSROOT.TH2Painter.prototype.CreateDrawBins = function(w, h, coordinates_kind, tipkind) {
      var i1 = this.GetSelectIndex("x", "left", 0);
      var i2 = this.GetSelectIndex("x", "right", 1);
      var j1 = this.GetSelectIndex("y", "left", 0);
      var j2 = this.GetSelectIndex("y", "right", 1);

      var name = this.GetItemName();
      if ((name==null) || (name=="")) name = this.histo.fName;
      if (name.length > 0) name += "\n";

      var xx = [], yy = [];
      for (var i = i1; i <= i2; ++i) {
         var x = this.GetBinX(i);
         if (this.options.Logx && (x <= 0)) continue;
         xx.push({indx:i, axis: x, gr: this.grx(x), cnt:0});
      }

      for (var j = j1; j <= j2; ++j) {
         var y = this.GetBinY(j);
         if (this.options.Logy && (y <= 0)) continue;
         yy.push({indx:j, axis: y, gr: this.gry(y), cnt:0});
      }

      // first found min/max values in selected range, and number of non-zero bins
      this.maxbin = this.minbin = this.histo.getBinContent(i1 + 1, j1 + 1);
      var nbins = 0, binz = 0, sumz = 0, zdiff, dgrx, dgry;
      for (var i = i1; i < i2; ++i) {
         for (var j = j1; j < j2; ++j) {
            binz = this.histo.getBinContent(i + 1, j + 1);
            if (binz != 0) nbins++;
            if (binz>this.maxbin) this.maxbin = binz; else
            if (binz<this.minbin) this.minbin = binz;
         }
      }

      var xfactor = 1, yfactor = 1, uselogz = false, logmin  = 0, logmax = 1;
      if (coordinates_kind == 1)
         if (this.options.Logz && (this.maxbin>0)) {
            uselogz = true;
            logmax = Math.log(this.maxbin);
            logmin = (this.minbin > 0) ? Math.log(this.minbin) : logmax - 10;
            xfactor = 0.5 / (logmax - logmin);
            yfactor = 0.5 / (logmax - logmin);
         } else {
            xfactor = 0.5 / (this.maxbin - this.minbin);
            yfactor = 0.5 / (this.maxbin - this.minbin);
         }

      if ((this.options.Optimize > 0) && (nbins>1000) && (coordinates_kind<2)) {
         // if there are many non-empty points, check if all of them are selected
         // probably we do not need to optimize axis

         this.fContour = null; // z-scale ranges when drawing with color
         this.fUserContour = false;
         nbins = 0;
         for (var i = i1; i < i2; ++i) {
            for (var j = j1; j < j2; ++j) {
               binz = this.histo.getBinContent(i + 1, j + 1);
               if ((binz == 0) || (binz < this.minbin)) continue;

               var show = false;

               if (coordinates_kind == 0) {
                  if (this.getValueColor(binz) != null) show = true;
               } else {
                  zdiff = uselogz ? (logmax - ((binz>0) ? Math.log(binz) : logmin)) : this.maxbin - binz;
                  dgrx = zdiff * xfactor;
                  dgry = zdiff * yfactor;

                  if (((1 - 2*zdiff*xfactor)*(xx[i-i1+1].gr - xx[i-i1].gr) > 0.05) ||
                      ((1 - 2*zdiff*yfactor)*(yy[j-j1].gr - yy[j-j1+1].gr) > 0.05)) show = true;
               }

               if (show) {
                  nbins++;
                  xx[i-i1].cnt+=1;
                  yy[j-j1].cnt+=1;
               }
            }
         }
      }

      if ((this.options.Optimize > 0) && (nbins>1000) && (coordinates_kind<2)) {
         var numx = 40, numy = 40;

         var coef = Math.abs(xx[0].gr - xx[xx.length-1].gr) / Math.abs(yy[0].gr - yy[yy.length-1].gr);
         if (coef > 1.) numy = Math.max(10, Math.round(numx / coef));
                   else numx = Math.max(10, Math.round(numy * coef));

         if ((this.options.Optimize > 1) || (xx.length > 50))
            this.CompressAxis(xx, numx, !this.options.Logx && this.regularx);

         if ((this.options.Optimize > 1) || (yy.length > 50))
            this.CompressAxis(yy, numy, !this.options.Logy && this.regulary);
      }

      this.fContour = null; // z-scale ranges when drawing with color
      this.fUserContour = false;

      var local_bins = [];

      for (var i = 0; i < xx.length-1; ++i) {
         var grx1 = xx[i].gr, grx2 = xx[i+1].gr;

         for (var j = 0; j < yy.length-1; ++j) {
            var gry1 = yy[j].gr, gry2 = yy[j+1].gr;

            sumz = binz = this.histo.getBinContent(xx[i].indx + 1, yy[j].indx + 1);

            if ((xx[i+1].indx > xx[i].indx+1) || (yy[j+1].indx > yy[j].indx+1)) {
               sumz = 0;
               // check all other pixels inside range
               for (var i1 = xx[i].indx;i1 < xx[i+1].indx;++i1)
                  for (var j1 = yy[j].indx;j1 < yy[j+1].indx;++j1) {
                     var morez = this.histo.getBinContent(i1 + 1, j1 + 1);
                     binz = Math.max(binz, morez);
                     sumz += morez;
                  }
            }

            if ((binz == 0) || (binz < this.minbin)) continue;

            var point = null;

            switch (coordinates_kind) {
            case 0: {
               var fillcol = this.getValueColor(binz);
               if (fillcol!=null)
                 point = {
                   x : grx1,
                   y : gry2,
                   width : grx2 - grx1 + 1,  // +1 to fill gaps between colored bins
                   height : gry1 - gry2 + 1,
                   stroke : "none",
                   fill : fillcol,
                   tipcolor: (fillcol == 'black') ? "grey" : "black"
                 };
               break;
            }
            case 1:
               zdiff = uselogz ? (logmax - ((binz>0) ? Math.log(binz) : logmin)) : this.maxbin - binz;
               dgrx = zdiff * xfactor * (grx2 - grx1);
               dgry = zdiff * yfactor * (gry1 - gry2);
               point = {
                  x : grx1 + dgrx,
                  y : gry2 + dgry,
                  width : grx2 - grx1 - 2 * dgrx,
                  height : gry1 - gry2 - 2 * dgry,
                  stroke : this.attline.color,
                  fill : this.fillcolor,
                  tipcolor: this.fillcolor == 'black' ? "grey" : "black"
               }
               if ((point.width < 0.05) || (point.height < 0.05)) point = null;
               break;

            case 2:
               point = {
                  x : (xx[i].axis + xx[i+1].axis) / 2,
                  y : (yy[j].axis + yy[j+1].axis) / 2,
                  z : binz
               }
               break;
            }

            if (point==null) continue;

            if (tipkind == 1) {
               if (this.x_kind == 'labels')
                  point['tip'] = name + "x = " + this.AxisAsText("x", xx[i].axis) + "\n";
               else {
                  point['tip'] = name + "x = [" + this.AxisAsText("x", xx[i].axis) + ", " + this.AxisAsText("x", xx[i+1].axis) + "]";

                  if (xx[i].indx + 1 == xx[i+1].indx)
                     point['tip'] += " bin=" + xx[i].indx + "\n";
                  else
                     point['tip'] += " bins=[" + xx[i].indx + "," + (xx[i+1].indx-1) + "]\n";
               }
               if (this.y_kind == 'labels')
                  point['tip'] += "y = " + this.AxisAsText("y", yy[j].axis) + "\n";
               else {
                  point['tip'] += "y = [" + this.AxisAsText("y", yy[j].axis) + ", " + this.AxisAsText("y", yy[j+1].axis) + "]";
                  if (yy[j].indx + 1 == yy[j+1].indx)
                     point['tip'] += " bin=" + yy[j].indx + "\n";
                  else
                     point['tip'] += " bins=[" + yy[j].indx + "," + (yy[j+1].indx-1) + "]\n";
               }

               if (sumz == binz)
                  point['tip'] += "entries = " + JSROOT.FFormat(sumz, JSROOT.gStyle.StatFormat);
               else
                  point['tip'] += "sum = " + JSROOT.FFormat(sumz, JSROOT.gStyle.StatFormat) +
                                  " max = " + JSROOT.FFormat(binz, JSROOT.gStyle.StatFormat);
            } else if (tipkind == 2)
               point['tip'] = name +
                              "x = " + this.AxisAsText("x", xx[i].axis) + "\n" +
                              "y = " + this.AxisAsText("y", yy[j].axis) + "\n" +
                              "entries = " + JSROOT.FFormat(sumz, JSROOT.gStyle.StatFormat);

            local_bins.push(point);
         }
      }

      return local_bins;
   }

   JSROOT.TH2Painter.prototype.DrawSimpleCanvas = function(w,h) {
      var i1 = this.GetSelectIndex("x", "left", 0);
      var i2 = this.GetSelectIndex("x", "right", 1);
      var j1 = this.GetSelectIndex("y", "left", 0);
      var j2 = this.GetSelectIndex("y", "right", 1);

      this.maxbin = this.minbin = this.histo.getBinContent(i1 + 1, j1 + 1);
      for (var i = i1; i < i2; ++i) {
         for (var j = j1; j < j2; ++j) {
            binz = this.histo.getBinContent(i + 1, j + 1);
            if (binz>this.maxbin) this.maxbin = binz; else
            if (binz<this.minbin) this.minbin = binz;
         }
      }

      var dx = i2-i1, dy = j2-j1;

      var fo = this.draw_g.append("foreignObject").attr("width", w).attr("height", h);
      this.SetForeignObjectPosition(fo);

      var canvas = fo.append("xhtml:canvas")
                     .attr("width", dx).attr("height", dy)
                     .attr("style", "width: " + w + "px; height: "+ h + "px");

      var context = canvas.node().getContext("2d");
      var image = context.createImageData(dx, dy);

      var p = -1;

      for (var j = j2-1; j >= j1; j--) {
         for (var i = i1; i < i2; ++i) {
            var bin = this.histo.getBinContent(i + 1, j + 1);
            var col = bin>this.minbin ? this.getValueColor(bin) : 'white';
            var c = d3.rgb(col);
            image.data[++p] = c.r;
            image.data[++p] = c.g;
            image.data[++p] = c.b;
            image.data[++p] = 255;
         }
      }

      context.putImageData(image, 0, 0);
   }

   JSROOT.TH2Painter.prototype.DrawNormalCanvas = function(w,h) {

      var local_bins = this.CreateDrawBins(w, h, 0, 0);

      var fo = this.draw_g.append("foreignObject").attr("width", w).attr("height", h);
      this.SetForeignObjectPosition(fo);

      var canvas = fo.append("xhtml:canvas").attr("width", w).attr("height", h);

      var ctx = canvas.node().getContext("2d");

      for (var i = 0; i < local_bins.length; ++i) {
         var bin = local_bins[i];
         ctx.fillStyle = bin.fill;
         ctx.fillRect(bin.x,bin.y,bin.width,bin.height);
      }

      ctx.stroke();
   }

   JSROOT.TH2Painter.prototype.DrawBins = function() {

      this.RecreateDrawG(false, ".main_layer");

      var w = this.frame_width(), h = this.frame_height();

      if ((this.options.Color==2) && !JSROOT.browser.isIE)
         return this.DrawSimpleCanvas(w,h);

      if ((this.options.Color==3) && !JSROOT.browser.isIE)
         return this.DrawNormalCanvas(w,h);

      // this.options.Scat =1;
      // this.histo['fMarkerStyle'] = 2;

      var draw_markers = (this.options.Scat > 0 && this.histo['fMarkerStyle'] > 1);
      var normal_coordinates = (this.options.Color > 0) || draw_markers;

      var tipkind = 0;
      if (JSROOT.gStyle.Tooltip) tipkind = draw_markers ? 2 : 1;

      var local_bins = this.CreateDrawBins(w, h, normal_coordinates ? 0 : 1, tipkind);

      if (draw_markers) {
         // Add markers
         var marker = JSROOT.Painter.createAttMarker(this.histo);

         var markers =
            this.draw_g.selectAll(".marker")
                  .data(local_bins)
                  .enter().append(marker.kind)
                  .attr("class", "marker")
                  .attr("transform", function(d) { return "translate(" + d.x.toFixed(1) + "," + d.y.toFixed(1) + ")" })
                  .call(marker.func);

         if (JSROOT.gStyle.Tooltip)
            markers.append("svg:title").text(function(d) { return d.tip; });
      } else {

         this.draw_g.selectAll(".bins")
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
             })
             .filter(function() { return JSROOT.gStyle.Tooltip ? this : null } )
             .on('mouseover', function() {
                if (JSROOT.gStyle.Tooltip)
                   d3.select(this).transition().duration(100).style("fill", this['f1']);
             })
             .on('mouseout', function() {
                d3.select(this).transition().duration(100).style("fill", this['f0']);
             })
             .append("svg:title").text(function(d) { return d.tip; });

      }

      delete local_bins;
   }

   JSROOT.TH2Painter.prototype.CanZoomIn = function(axis,min,max) {
      // check if it makes sense to zoom inside specified axis range
      if ((axis=="x") && (this.GetIndexX(max,0.5) - this.GetIndexX(min,0) > 1)) return true;

      if ((axis=="y") && (this.GetIndexY(max,0.5) - this.GetIndexY(min,0) > 1)) return true;

      if (axis=="z") return true;

      return false;
   }

   JSROOT.TH2Painter.prototype.Draw2D = function(call_back) {
      if (typeof this['Create3DScene'] == 'function')
         this.Create3DScene(-1);

      this.DrawAxes();

      this.DrawGrids();

      this.DrawBins();

      this.AddInteractive();

      JSROOT.CallBack(call_back);
   }

   JSROOT.TH2Painter.prototype.Draw3D = function(call_back) {
      JSROOT.AssertPrerequisites('3d', function() {
         this['Create3DScene'] = JSROOT.Painter.HPainter_Create3DScene;
         this['Draw3DBins'] = JSROOT.Painter.TH2Painter_Draw3DBins;
         this['Draw3D'] = JSROOT.Painter.TH2Painter_Draw3D;
         this['Draw3D'](call_back);
      }.bind(this));
   }

   JSROOT.TH2Painter.prototype.Redraw = function() {
      this.CreateXY();

      var func_name = this.options.Lego > 0 ? "Draw3D" : "Draw2D";

      this[func_name](function() {
         if (this.create_canvas) this.DrawTitle();
      }.bind(this));
   }

   JSROOT.Painter.drawHistogram2D = function(divid, histo, opt) {

      // create painter and add it to canvas
      JSROOT.extend(this, new JSROOT.TH2Painter(histo));

      this.SetDivId(divid, 1);

      // here we deciding how histogram will look like and how will be shown
      this.options = this.DecodeOptions(opt);

      this.CheckPadOptions();

      this.ScanContent();

      this.CreateXY();

      // check if we need to create palette
      if ((this.FindPalette() == null) && this.create_canvas && (this.options.Zscale > 0)) {
         // create pallette

         var shrink = this.CreatePalette(0.04);
         this.svg_frame().property('frame_painter').Shrink(0, shrink);
         this.svg_frame().property('frame_painter').Redraw();
         this.CreateXY();
      } else if (this.options.Zscale == 0) {
         // delete palette - it may appear there due to previous draw options
         this.FindPalette(true);
      }

      // check if we need to create statbox
      if (JSROOT.gStyle.AutoStat && this.create_canvas)
         this.CreateStat();

      var func_name = this.options.Lego > 0 ? "Draw3D" : "Draw2D";

      this[func_name](function() {
         if (this.create_canvas) this.DrawTitle();

         this.DrawNextFunction(0, function() {
            if (this.options.Lego == 0) {
               if (this.options.AutoZoom) this.AutoZoom();
            }
            this.DrawingReady();
         }.bind(this));

      }.bind(this));

      return this;
   }


   return JSROOT.Painter;

}));
