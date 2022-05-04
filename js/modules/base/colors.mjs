/** @summary Covert value between 0 and 1 into hex, used for colors coding
  * @private */
function toHex(num,scale) {
   let s = Math.round(num*(scale || 255)).toString(16);
   return s.length == 1 ? '0'+s : s;
}

/** @summary list of global root colors
  * @private */
let gbl_colors_list = [];

/** @summary Generates all root colors, used also in jstests to reset colors
  * @private */
function createRootColors() {
   let colorMap = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', '#59d454', '#5954d9', 'white'];
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
      let s = entry.s;
      for (let n = 0; n < s.length; n += 6) {
         let num = entry.n + n / 6;
         colorMap[num] = '#' + s.slice(n,n+6);
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
   if (!col || (col._typename != 'TColor')) return null;

   let rgb = '#' + toHex(col.fRed) + toHex(col.fGreen) + toHex(col.fBlue);
   if ((col.fAlpha !==undefined) && (col.fAlpha !== 1.))
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


/** @summary Add new colors from object array
  * @private */
function extendRootColors(jsarr, objarr) {
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
         let col = objarr.arr[n];
         if (!col || (col._typename != 'TColor')) continue;

         if ((col.fNumber >= 0) && (col.fNumber <= 10000))
            rgb_array[col.fNumber] = getRGBfromTColor(col);
      }
   }

   for (let n = 0; n < rgb_array.length; ++n)
      if (rgb_array[n] && (jsarr[n] != rgb_array[n]))
         jsarr[n] = rgb_array[n];

   return jsarr;
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
  * @returns {String} with RGB color code or existing color name like 'cyan'
  * @private */
function getColor(indx) {
   return gbl_colors_list[indx];
}

/** @summary Add new color
  * @param {string} rgb - color name or just string with rgb value
  * @param {array} [lst] - optional colors list, to which add colors
  * @returns {number} index of new color
  * @private */
function addColor(rgb, lst) {
   if (!lst) lst = gbl_colors_list;
   let indx = lst.indexOf(rgb);
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
   constructor(arr) {
      this.palette = arr;
   }

   /** @summary Returns color index which correspond to contour index of provided length */
   calcColorIndex(i, len) {
      let plen = this.palette.length, theColor = Math.floor((i + 0.99) * plen / (len - 1));
      return (theColor > plen - 1) ? plen - 1 : theColor;
    }

   /** @summary Returns color with provided index */
   getColor(indx) { return this.palette[indx]; }

   /** @summary Returns number of colors in the palette */
   getLength() { return this.palette.length; }

   /** @summary Calculate color for given i and len */
   calcColor(i, len) { return this.getColor(this.calcColorIndex(i, len)); }

} // class ColorPalette

createRootColors();

export { ColorPalette, getColor, addColor, adoptRootColors, getRootColors, extendRootColors, getRGBfromTColor, createRootColors, toHex };


