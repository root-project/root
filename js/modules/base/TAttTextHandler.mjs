import { getColor } from './colors.mjs';


function calcTextSize(sz, sz0, fact, pp) {
   if (!sz)
      sz = sz0 || 0;

   if (sz >= 1)
      return Math.round(sz * (pp?.getPadScale() || 1));

   return Math.round(sz * Math.min(pp?.getPadWidth() ?? 1000, pp?.getPadHeight() ?? 1000) * (fact || 1));
}

/**
  * @summary Handle for text attributes
  * @private
  */

class TAttTextHandler {

   /** @summary constructor
     * @param {object} attr - attributes, see {@link TAttTextHandler#setArgs} */
   constructor(args) {
      this.used = true;
      if (args._typename && (args.fTextFont !== undefined)) args = { attr: args };
      this.setArgs(args);
   }

   /** @summary Set text attributes.
     * @param {object} args - specify attributes by different ways
     * @param {object} args.attr - TAttText object with appropriate data members or
     * @param {object} args.attr_alt - alternative TAttText object with appropriate data members if values are 0
     * @param {string} args.color - color in html like rgb(255,0,0) or 'red' or '#ff0000'
     * @param {number} args.align - text align
     * @param {number} args.angle - text angle
     * @param {number} args.font  - font index
     * @param {number} args.size  - text size */
   setArgs(args) {
      if (args.attr) {
         args.font = args.attr.fTextFont || args.attr_alt?.fTextFont || 0;
         args.size = args.attr.fTextSize || args.attr_alt?.fTextSize || 0;
         this.color_index = args.attr.fTextColor || args.attr_alt?.fTextColor || 0;
         args.color = args.painter?.getColor(this.color_index) ?? getColor(this.color_index);
         args.align = args.attr.fTextAlign || args.attr_alt?.fTextAlign || 0;
         args.angle = args.attr.fTextAngle || args.attr_alt?.fTextAngle || 0;
      } else if (typeof args.color === 'number') {
         this.color_index = args.color;
         args.color = args.painter?.getColor(args.color) ?? getColor(args.color);
      }

      this.font = args.font;
      this.size = args.size;
      this.color = args.color;
      this.align = args.align;
      this.angle = args.angle;

      this.can_rotate = args.can_rotate ?? true;
      this.angle_used = false;
      this.align_used = false;
   }

   /** @summary returns true if line attribute is empty and will not be applied. */
   empty() { return this.color === 'none'; }

   /** @summary Change text attributes */
   change(font, size, color, align, angle) {
      if (font !== undefined)
         this.font = font;
      if (size !== undefined)
         this.size = size;
      if (color !== undefined) {
         if (this.color !== color)
            delete this.color_index;
         this.color = color;
      }
      if (align !== undefined)
         this.align = align;
      if (angle !== undefined)
         this.angle = angle;
      this.changed = true;
   }

   /** @summary Method used when color or pattern were changed with OpenUi5 widgets.
     * @private */
   verifyDirectChange(/* painter */) {
      this.change(parseInt(this.font), parseFloat(this.size), this.color, parseInt(this.align), parseInt(this.angle));
   }

   /** @summary Create argument for drawText method */
   createArg(arg) {
      if (!arg) arg = {};
      this.align_used = !arg.noalign && !arg.align;
      if (this.align_used)
         arg.align = this.align;
      this.angle_used = !arg.norotate && this.can_rotate;
      if (this.angle_used && this.angle)
         arg.rotate = -this.angle; // SVG rotation angle has different sign
      arg.color = this.color || 'black';
      return arg;
   }

   /** @summary Provides pixel size */
   getSize(pp, fact, zero_size) { return calcTextSize(this.size, zero_size, fact, pp); }

   /** @summary Returns alternating size - which defined by sz1 variable */
   getAltSize(sz1, pp) { return calcTextSize(sz1, this.size, 1, pp); }

   /** @summary Get font index - without precision */
   getGedFont() { return Math.floor(this.font / 10); }

   /** @summary Change text font from GED */
   setGedFont(value) {
      const v = parseInt(value);
      if ((v > 0) && (v < 17))
         this.change(v*10 + (this.font % 10));
      return this.font;
   }

} // class TAttTextHandler

export { TAttTextHandler };
