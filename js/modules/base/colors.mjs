import { clTColor, settings } from '../core.mjs';
import { color as d3_color } from '../d3.mjs';

const clTLinearGradient = 'TLinearGradient', clTRadialGradient = 'TRadialGradient',
      kWhite = 0, kBlack = 1, kRed = 2, kGreen = 3, kBlue = 4, kYellow = 5, kMagenta = 6, kCyan = 7;

/** @summary Covert value between 0 and 1 into hex, used for colors coding
  * @private */
function toHex(num, scale = 255) {
   const s = Math.round(num * scale).toString(16);
   return s.length === 1 ? '0'+s : s;
}

/** @summary list of global root colors
  * @private */
let gbl_colors_list = [];

/** @summary Generates all root colors, used also in jstests to reset colors
  * @private */
function createRootColors() {
   const colorMap = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', '#59d454', '#5954d9', 'white'];
   colorMap[110] = 'white';

   const moreCol = [
      { n: 11, s: 'c1b7ad4d4d4d6666668080809a9a9ab3b3b3cdcdcde6e6e6f3f3f3cdc8accdc8acc3c0a9bbb6a4b3a697b8a49cae9a8d9c8f83886657b1cfc885c3a48aa9a1839f8daebdc87b8f9a768a926983976e7b857d9ad280809caca6c0d4cf88dfbb88bd9f83c89a7dc08378cf5f61ac8f94a6787b946971d45a549300ff7b00ff6300ff4b00ff3300ff1b00ff0300ff0014ff002cff0044ff005cff0074ff008cff00a4ff00bcff00d4ff00ecff00fffd00ffe500ffcd00ffb500ff9d00ff8500ff6d00ff5500ff3d00ff2600ff0e0aff0022ff003aff0052ff006aff0082ff009aff00b1ff00c9ff00e1ff00f9ff00ffef00ffd700ffbf00ffa700ff8f00ff7700ff6000ff4800ff3000ff1800ff0000' },
      { n: 201, s: '5c5c5c7b7b7bb8b8b8d7d7d78a0f0fb81414ec4848f176760f8a0f14b81448ec4876f1760f0f8a1414b84848ec7676f18a8a0fb8b814ecec48f1f1768a0f8ab814b8ec48ecf176f10f8a8a14b8b848ecec76f1f1' },
      { n: 390, s: 'ffffcdffff9acdcd9affff66cdcd669a9a66ffff33cdcd339a9a33666633ffff00cdcd009a9a00666600333300' },
      { n: 406, s: 'cdffcd9aff9a9acd9a66ff6666cd66669a6633ff3333cd33339a3333663300ff0000cd00009a00006600003300' },
      { n: 422, s: 'cdffff9affff9acdcd66ffff66cdcd669a9a33ffff33cdcd339a9a33666600ffff00cdcd009a9a006666003333' },
      { n: 590, s: 'cdcdff9a9aff9a9acd6666ff6666cd66669a3333ff3333cd33339a3333660000ff0000cd00009a000066000033' },
      { n: 606, s: 'ffcdffff9affcd9acdff66ffcd66cd9a669aff33ffcd33cd9a339a663366ff00ffcd00cd9a009a660066330033' },
      { n: 622, s: 'ffcdcdff9a9acd9a9aff6666cd66669a6666ff3333cd33339a3333663333ff0000cd00009a0000660000330000' },
      { n: 791, s: 'ffcd9acd9a669a66339a6600cd9a33ffcd66ff9a00ffcd33cd9a00ffcd00ff9a33cd66006633009a3300cd6633ff9a66ff6600ff6633cd3300ff33009aff3366cd00336600339a0066cd339aff6666ff0066ff3333cd0033ff00cdff9a9acd66669a33669a009acd33cdff669aff00cdff339acd00cdff009affcd66cd9a339a66009a6633cd9a66ffcd00ff6633ffcd00cd9a00ffcd33ff9a00cd66006633009a3333cd6666ff9a00ff9a33ff6600cd3300ff339acdff669acd33669a00339a3366cd669aff0066ff3366ff0033cd0033ff339aff0066cd00336600669a339acd66cdff009aff33cdff009acd00cdffcd9aff9a66cd66339a66009a9a33cdcd66ff9a00ffcd33ff9a00cdcd00ff9a33ff6600cd33006633009a6633cd9a66ff6600ff6633ff3300cd3300ffff339acd00666600339a0033cd3366ff669aff0066ff3366cd0033ff0033ff9acdcd669a9a33669a0066cd339aff66cdff009acd009aff33cdff009a' },
      { n: 920, s: 'cdcdcd9a9a9a666666333333' }];

   moreCol.forEach(entry => {
      const s = entry.s;
      for (let n = 0; n < s.length; n += 6) {
         const num = entry.n + n / 6;
         colorMap[num] = '#' + s.slice(n, n+6);
      }
   });

   gbl_colors_list = colorMap;
}

/** @summary Get list of colors
  * @private */
function getRootColors() {
   return gbl_colors_list;
}

/** @summary Produces rgb code for TColor object
  * @private */
function getRGBfromTColor(col) {
   if (col?._typename !== clTColor) return null;

   let rgb = '#' + toHex(col.fRed) + toHex(col.fGreen) + toHex(col.fBlue);
   if ((col.fAlpha !== undefined) && (col.fAlpha !== 1))
      rgb += toHex(col.fAlpha);

   switch (rgb) {
      case '#ffffff': return 'white';
      case '#000000': return 'black';
      case '#ff0000': return 'red';
      case '#00ff00': return 'green';
      case '#0000ff': return 'blue';
      case '#ffff00': return 'yellow';
      case '#ff00ff': return 'magenta';
      case '#00ffff': return 'cyan';
   }
   return rgb;
}

/** @ummary Return list of grey colors for the original array
  * @private */
function getGrayColors(rgb_array) {
   const gray_colors = [];

   if (!rgb_array) rgb_array = getRootColors();

   for (let n = 0; n < rgb_array.length; ++n) {
      if (!rgb_array[n]) continue;
      const rgb = d3_color(rgb_array[n]),
            gray = 0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b;
      rgb.r = rgb.g = rgb.b = gray;
      gray_colors[n] = rgb.hex();
   }

   return gray_colors;
}

/** @summary Add new colors from object array
  * @private */
function extendRootColors(jsarr, objarr, grayscale) {
   if (!jsarr) {
      jsarr = [];
      for (let n = 0; n < gbl_colors_list.length; ++n)
         jsarr[n] = gbl_colors_list[n];
   }

   if (!objarr) return jsarr;

   let rgb_array = objarr;
   if (objarr._typename && objarr.arr) {
      rgb_array = [];
      for (let n = 0; n < objarr.arr.length; ++n) {
         const col = objarr.arr[n];
         if ((col?._typename === clTLinearGradient) || (col?._typename === clTRadialGradient)) {
            rgb_array[col.fNumber] = col;
            col.toString = () => 'white';
            continue;
         }

         if (col?._typename !== clTColor)
            continue;

         if ((col.fNumber >= 0) && (col.fNumber <= 10000))
            rgb_array[col.fNumber] = getRGBfromTColor(col);
      }
   }

   for (let n = 0; n < rgb_array.length; ++n) {
      if (rgb_array[n] && (jsarr[n] !== rgb_array[n]))
         jsarr[n] = rgb_array[n];
   }

   return grayscale ? getGrayColors(jsarr) : jsarr;
}

/** @ummary Set global list of colors.
  * @desc Either TObjArray of TColor instances or just plain array with rgb() code.
  * List of colors typically stored together with TCanvas primitives
  * @private */
function adoptRootColors(objarr) {
   extendRootColors(gbl_colors_list, objarr);
}

/** @summary Return ROOT color by index
  * @desc Color numbering corresponds typical ROOT colors
  * @return {String} with RGB color code or existing color name like 'cyan'
  * @private */
function getColor(indx) {
   return gbl_colors_list[indx];
}

/** @summary Search for specified color in the list of colors
  * @return Color index or -1 if fails
  * @private */
function findColor(name) {
   if (!name) return -1;
   for (let indx = 0; indx < gbl_colors_list.length; ++indx) {
      if (gbl_colors_list[indx] === name)
         return indx;
   }
   return -1;
}

/** @summary Add new color
  * @param {string} rgb - color name or just string with rgb value
  * @param {array} [lst] - optional colors list, to which add colors
  * @return {number} index of new color
  * @private */
function addColor(rgb, lst) {
   if (!lst) lst = gbl_colors_list;
   const indx = lst.indexOf(rgb);
   if (indx >= 0) return indx;
   lst.push(rgb);
   return lst.length-1;
}

/**
 * @summary Color palette handle
 *
 * @private
 */

class ColorPalette {

   /** @summary constructor */
   constructor(arr, grayscale) {
      this.palette = grayscale ? getGrayColors(arr) : arr;
   }

   /** @summary Returns color index which correspond to contour index of provided length */
   calcColorIndex(i, len) {
      const plen = this.palette.length, theColor = Math.floor((i + 0.99) * plen / (len - 1));
      return (theColor > plen - 1) ? plen - 1 : theColor;
    }

   /** @summary Returns color with provided index */
   getColor(indx) { return this.palette[indx]; }

   /** @summary Returns number of colors in the palette */
   getLength() { return this.palette.length; }

   /** @summary Calculate color for given i and len */
   calcColor(i, len) { return this.getColor(this.calcColorIndex(i, len)); }

} // class ColorPalette

function createDefaultPalette(grayscale) {
   const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2/3 - t) * 6;
      return p;
   }, HLStoRGB = (h, l, s) => {
      const q = (l < 0.5) ? l * (1 + s) : l + s - l * s,
            p = 2 * l - q,
            r = hue2rgb(p, q, h + 1/3),
            g = hue2rgb(p, q, h),
            b = hue2rgb(p, q, h - 1/3);
      return '#' + toHex(r) + toHex(g) + toHex(b);
   }, minHue = 0, maxHue = 280, maxPretty = 50, palette = [];
   for (let i = 0; i < maxPretty; ++i) {
      const hue = (maxHue - (i + 1) * ((maxHue - minHue) / maxPretty)) / 360;
      palette.push(HLStoRGB(hue, 0.5, 1));
   }
   return new ColorPalette(palette, grayscale);
}

function createGrayPalette() {
   const palette = [];
   for (let i = 0; i < 50; ++i) {
      const code = toHex((i+2)/60);
      palette.push('#'+code+code+code);
   }
   return new ColorPalette(palette);
}

/* eslint-disable comma-spacing */


/** @summary Create color palette
  * @private */
function getColorPalette(id, grayscale) {
   id = id || settings.Palette;
   if ((id > 0) && (id < 10)) return createGrayPalette();
   if (id < 51) return createDefaultPalette(grayscale);
   if (id > 113) id = 57;
   const stops = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1];
   let rgb;
   switch (id) {
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
      // Bird (default, keep float for backward compatibility)
      case 57: rgb = [[53.091,15.096,19.89,5.916,45.951,135.1755,208.743,253.878,248.982],[42.432,91.7745,128.5455,163.6845,183.039,191.046,186.864,200.481,250.716],[134.9715,221.442,213.8175,201.807,163.8375,118.881,89.2245,50.184,13.7445]]; break;
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
      case 74: rgb = [[76,120,156,183,197,180,162,154,140],[34,35,42,69,102,137,164,188,197],[64,69,78,105,142,177,205,217,198]]; break;
      // Cherry
      case 75: rgb = [[37,102,157,188,196,214,223,235,251],[37,29,25,37,67,91,132,185,251],[37,32,33,45,66,98,137,187,251]]; break;
      // Coffee
      case 76: rgb = [[79,100,119,137,153,172,192,205,250],[63,79,93,103,115,135,167,196,250],[51,59,66,61,62,70,110,160,250]]; break;
      // Dark Rain Bow
      case 77: rgb = [[43,44,50,66,125,172,178,155,157],[63,63,85,101,138,163,122,51,39],[121,101,58,44,47,55,57,44,43]]; break;
      // Dark Terrain
      case 78: rgb = [[0,41,62,79,90,87,99,140,228],[0,57,81,93,85,70,71,125,228],[95,91,91,82,60,43,44,112,228]]; break;
      // Fall
      case 79: rgb = [[49,59,72,88,114,141,176,205,222],[78,72,66,57,59,75,106,142,173],[78,55,46,40,39,39,40,41,47]]; break;
      // Fruit Punch
      case 80: rgb = [[243,222,201,185,165,158,166,187,219],[94,108,132,135,125,96,68,51,61],[7,9,12,19,45,89,118,146,118]]; break;
      // Fuchsia
      case 81: rgb = [[19,44,74,105,137,166,194,206,220],[19,28,40,55,82,110,159,181,220],[19,42,68,96,129,157,188,203,220]]; break;
      // Grey Yellow
      case 82: rgb = [[33,44,70,99,140,165,199,211,216],[38,50,76,105,140,165,191,189,167],[55,67,97,124,140,166,163,129,52]]; break;
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
      case 93: rgb = [[39,43,59,63,80,116,153,177,223],[39,43,59,74,91,114,139,165,223],[39,50,59,70,85,115,151,176,223]]; break;
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
      case 101: rgb = [[82,106,126,141,155,163,142,107,66],[62,44,69,107,135,152,149,132,119],[39,25,31,60,73,68,49,72,188]]; break;
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
      default: return createDefaultPalette();
   }

   const NColors = 255, Red = rgb[0], Green = rgb[1], Blue = rgb[2], palette = [];

   for (let g = 1; g < stops.length; g++) {
       // create the colors...
       const nColorsGradient = Math.round(Math.floor(NColors*stops[g]) - Math.floor(NColors*stops[g-1]));
       for (let c = 0; c < nColorsGradient; c++) {
          const col = '#' + toHex(Red[g-1] + c * (Red[g] - Red[g-1]) / nColorsGradient, 1) +
                            toHex(Green[g-1] + c * (Green[g] - Green[g-1]) / nColorsGradient, 1) +
                            toHex(Blue[g-1] + c * (Blue[g] - Blue[g-1]) / nColorsGradient, 1);
          palette.push(col);
       }
    }

    return new ColorPalette(palette, grayscale);
}


/** @summary Decode list of ROOT colors coded by TWebCanvas
  * @private */
function decodeWebCanvasColors(oper) {
   const colors = [], arr = oper.split(';');
   for (let n = 0; n < arr.length; ++n) {
      const name = arr[n];
      let p = name.indexOf(':');
      if (p > 0) {
         colors[parseInt(name.slice(0, p))] = d3_color(`rgb(${name.slice(p+1)})`).formatHex();
         continue;
      }
      p = name.indexOf('=');
      if (p > 0) {
         colors[parseInt(name.slice(0, p))] = d3_color(`rgba(${name.slice(p+1)})`).formatHex8();
         continue;
      }
      p = name.indexOf('#');
      if (p < 0) continue;

      const colindx = parseInt(name.slice(0, p)),
            data = JSON.parse(name.slice(p+1)),
            grad = { _typename: data[0] === 10 ? clTLinearGradient : clTRadialGradient, fNumber: colindx, fType: data[0] };

      let cnt = 1;

      grad.fCoordinateMode = Math.round(data[cnt++]);
      const nsteps = Math.round(data[cnt++]);
      grad.fColorPositions = data.slice(cnt, cnt + nsteps); cnt += nsteps;
      grad.fColors = data.slice(cnt, cnt + 4*nsteps); cnt += 4*nsteps;
      grad.fStart = { fX: data[cnt++], fY: data[cnt++] };
      grad.fEnd = { fX: data[cnt++], fY: data[cnt++] };
      if (grad._typename === clTRadialGradient && cnt < data.length) {
         grad.fR1 = data[cnt++];
         grad.fR2 = data[cnt++];
      }

      colors[colindx] = grad;
   }

   return colors;
}


createRootColors();

export { getColor, findColor, addColor, adoptRootColors,
         getRootColors, getGrayColors,
         extendRootColors, getRGBfromTColor, createRootColors, toHex,
         kWhite, kBlack, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan,
         ColorPalette, getColorPalette, clTLinearGradient, clTRadialGradient, decodeWebCanvasColors };
