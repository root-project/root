import { select as d3_select } from '../d3.mjs';

import { getColor } from './colors.mjs';

const root_markers = [
      0, 1, 2, 3, 4,           //  0..4
      5, 106, 107, 104, 1,     //  5..9
      1, 1, 1, 1, 1,           // 10..14
      1, 1, 1, 1, 1,           // 15..19
      104, 125, 126, 132, 4,   // 20..24
      25, 26, 27, 28, 130,     // 25..29
      30, 3, 32, 127, 128,     // 30..34
      35, 36, 37, 38, 137,     // 35..39
      40, 140, 42, 142, 44,    // 40..44
      144, 46, 146, 148, 149]; // 45..49


/**
  * @summary Handle for marker attributes
  * @private
  */

class TAttMarkerHandler {

   /** @summary constructor
     * @param {object} args - attributes, see {@link TAttMarkerHandler#setArgs} for details */
   constructor(args) {
      this.x0 = this.y0 = 0;
      this.color = 'black';
      this.style = 1;
      this.size = 8;
      this.scale = 1;
      this.stroke = true;
      this.fill = true;
      this.marker = "";
      this.ndig = 0;
      this.used = true;
      this.changed = false;
      this.func = this.apply.bind(this);
      this.setArgs(args);
      this.changed = false;
   }

   /** @summary Set marker attributes.
     * @param {object} args - arguments can be
     * @param {object} args.attr - instance of TAttrMarker (or derived class) or
     * @param {string} args.color - color in HTML form like grb(1,4,5) or 'green'
     * @param {number} args.style - marker style
     * @param {number} args.size - marker size
     * @param {number} [args.refsize] - when specified and marker size < 1, marker size will be calculated relative to that size */
   setArgs(args) {
      if ((typeof args == 'object') && (typeof args.fMarkerStyle == 'number')) args = { attr: args };

      if (args.attr) {
         if (args.color === undefined)
            args.color = args.painter ? args.painter.getColor(args.attr.fMarkerColor) : getColor(args.attr.fMarkerColor);
         if (!args.style || (args.style < 0)) args.style = args.attr.fMarkerStyle;
         if (!args.size) args.size = args.attr.fMarkerSize;
      }

      this.color = args.color;
      this.style = args.style;
      this.size = args.size;
      this.refsize = args.refsize;

      this._configure();
   }

   /** @summary Reset position, used for optimization of drawing of multiple markers
    * @private */
   resetPos() { this.lastx = this.lasty = null; }

   /** @summary Create marker path for given position.
     * @desc When drawing many elementary points, created path may depend from previously produced markers.
     * @param {number} x - first coordinate
     * @param {number} y - second coordinate
     * @returns {string} path string */
   create(x, y) {
      if (!this.optimized)
         return `M${(x + this.x0).toFixed(this.ndig)},${(y + this.y0).toFixed(this.ndig)}${this.marker}`;

      // use optimized handling with relative position
      let xx = Math.round(x), yy = Math.round(y), mv = `M${xx},${yy}`;
      if (this.lastx !== null) {
         if ((xx == this.lastx) && (yy == this.lasty)) {
            mv = ""; // pathological case, but let exclude it
         } else {
            let m2 = `m${xx-this.lastx},${yy - this.lasty}`;
            if (m2.length < mv.length) mv = m2;
         }
      }
      this.lastx = xx + 1; this.lasty = yy;
      return mv + "h1";
   }

   /** @summary Returns full size of marker */
   getFullSize() { return this.scale * this.size; }

   /** @summary Returns approximate length of produced marker string */
   getMarkerLength() { return this.marker ? this.marker.length : 10; }

   /** @summary Change marker attributes.
    *  @param {string} color - marker color
    *  @param {number} style - marker style
    *  @param {number} size - marker size */
   change(color, style, size) {
      this.changed = true;

      if (color !== undefined) this.color = color;
      if ((style !== undefined) && (style >= 0)) this.style = style;
      if (size !== undefined) this.size = size;

      this._configure();
   }

   /** @summary Prepare object to create marker
     * @private */
    _configure() {

      this.x0 = this.y0 = 0;

      if ((this.style === 1) || (this.style === 777)) {
         this.fill = false;
         this.marker = "h1";
         this.size = 1;
         this.optimized = true;
         this.resetPos();
         return true;
      }

      this.optimized = false;

      let marker_kind = root_markers[this.style] ?? 104;
      let shape = marker_kind % 100;

      this.fill = (marker_kind >= 100);

      this.scale = this.refsize || 8; // v7 defines refsize as 1 or pad height

      let size = this.getFullSize();

      this.ndig = (size > 7) ? 0 : ((size > 2) ? 1 : 2);
      if (shape == 30) this.ndig++; // increase precision for star
      let s1 = size.toFixed(this.ndig),
          s2 = (size/2).toFixed(this.ndig),
          s3 = (size/3).toFixed(this.ndig),
          s4 = (size/4).toFixed(this.ndig),
          s8 = (size/8).toFixed(this.ndig),
          s38 = (size*3/8).toFixed(this.ndig);

      switch (shape) {
         case 1: // dot
            this.marker = "h1";
            break;
         case 2: // plus
            this.y0 = -size / 2;
            this.marker = `v${s1}m-${s2},-${s2}h${s1}`;
            break;
         case 3: // asterisk
            this.x0 = this.y0 = -size / 2;
            this.marker = `l${s1},${s1}m0,-${s1}l-${s1},${s1}m0,-${s2}h${s1}m-${s2},-${s2}v${s1}`;
            break;
         case 4: // circle
            this.x0 = -parseFloat(s2);
            s1 = (parseFloat(s2) * 2).toFixed(this.ndig);
            this.marker = `a${s2},${s2},0,1,0,${s1},0a${s2},${s2},0,1,0,-${s1},0z`;
            break;
         case 5: // mult
            this.x0 = this.y0 = -size / 2;
            this.marker = `l${s1},${s1}m0,-${s1}l-${s1},${s1}`;
            break;
         case 6: // small dot
            this.x0 = -1;
            this.marker = "a1,1,0,1,0,2,0a1,1,0,1,0,-2,0z";
            break;
         case 7: // medium dot
            this.x0 = -1.5;
            this.marker = "a1.5,1.5,0,1,0,3,0a1.5,1.5,0,1,0,-3,0z";
            break;
         case 25: // square
            this.x0 = this.y0 = -size / 2;
            this.marker = `v${s1}h${s1}v-${s1}z`;
            break;
         case 26: // triangle-up
            this.y0 = -size / 2;
            this.marker = `l-${s2},${s1}h${s1}z`;
            break;
         case 27: // diamand
            this.y0 = -size / 2;
            this.marker = `l${s3},${s2}l-${s3},${s2}l-${s3},-${s2}z`;
            break;
         case 28: // cross
            this.x0 = this.y0 = size / 6;
            this.marker = `h${s3}v-${s3}h-${s3}v-${s3}h-${s3}v${s3}h-${s3}v${s3}h${s3}v${s3}h${s3}z`;
            break;
         case 30: // star
            this.y0 = -size / 2;
            let s56 = (size*5/6).toFixed(this.ndig), s58 = (size*5/8).toFixed(this.ndig);
            this.marker = `l${s3},${s1}l-${s56},-${s58}h${s1}l-${s56},${s58}z`;
            break;
         case 32: // triangle-down
            this.y0 = size / 2;
            this.marker = `l-${s2},-${s1}h${s1}z`;
            break;
         case 35:
            this.x0 = -size / 2;
            this.marker = `l${s2},${s2}l${s2},-${s2}l-${s2},-${s2}zh${s1}m-${s2},-${s2}v${s1}`;
            break;
         case 36:
            this.x0 = this.y0 = -size / 2;
            this.marker = `h${s1}v${s1}h-${s1}zl${s1},${s1}m0,-${s1}l-${s1},${s1}`;
            break;
         case 37:
            this.x0 = -size/2;
            this.marker = `h${s1}l-${s4},-${s2}l-${s2},${s1}h${s2}l-${s2},-${s1}z`;
            break;
         case 38:
            this.x0 = -size/4; this.y0 = -size/2;
            this.marker = `h${s2}l${s4},${s4}v${s2}l-${s4},${s4}h-${s2}l-${s4},-${s4}v-${s2}zm${s4},0v${s1}m-${s2},-${s2}h${s1}`;
            break;
         case 40:
            this.x0 = -size/4; this.y0 = -size/2;
            this.marker = `l${s2},${s1}l${s4},-${s4}l-${s1},-${s2}zm${s2},0l-${s2},${s1}l-${s4},-${s4}l${s1},-${s2}z`;
            break;
         case 42:
            this.y0 = -size/2;
            this.marker = `l${s8},${s38}l${s38},${s8}l-${s38},${s8}l-${s8},${s38}l-${s8},-${s38}l-${s38},-${s8}l${s38},-${s8}z`;
            break;
         case 44:
            this.x0 = -size/4; this.y0 = -size/2;
            this.marker = `h${s2}l-${s8},${s38}l${s38},-${s8}v${s2}l-${s38},-${s8}l${s8},${s38}h-${s2}l${s8},-${s38}l-${s38},${s8}v-${s2}l${s38},${s8}z`;
            break;
         case 46:
            this.x0 = -size/4; this.y0 = -size/2;
            this.marker = `l${s4},${s4}l${s4},-${s4}l${s4},${s4}l-${s4},${s4}l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}l-${s4},${s4}l-${s4},-${s4}l${s4},-${s4}l-${s4},-${s4}z`;
            break;
         case 48:
            this.x0 = -size/4; this.y0 = -size/2;
            this.marker = `l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}zm${s2},0l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}zm0,${s2}l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}zm-${s2},0l${s4},${s4}l-${s4},${s4}l-${s4},-${s4}z`;
            break;
         case 49:
            this.x0 = -size/6; this.y0 = -size/2;
            this.marker = `h${s3}v${s3}h-${s3}zm${s3},${s3}h${s3}v${s3}h-${s3}zm-${s3},${s3}h${s3}v${s3}h-${s3}zm-${s3},-${s3}h${s3}v${s3}h-${s3}z`;
            break;
         default: // diamand
            this.y0 = -size / 2;
            this.marker = `l${s3},${s2}l-${s3},${s2}l-${s3},-${s2}z`;
            break;
      }

      return true;
   }

   /** @summary get stroke color */
   getStrokeColor() { return this.stroke ? this.color : "none"; }

   /** @summary get fill color */
   getFillColor() { return this.fill ? this.color : "none"; }

   /** @summary returns true if marker attributes will produce empty (invisible) output */
   empty() {
      return (this.color === 'none') || (!this.fill && !this.stroke);
   }

   /** @summary Apply marker styles to created element */
   apply(selection) {
      selection.style('stroke', this.stroke ? this.color : "none")
               .style('fill', this.fill ? this.color : "none");
   }

   /** @summary Method used when color or pattern were changed with OpenUi5 widgets.
    * @private */
   verifyDirectChange(/* painter */) {
      this.change(this.color, parseInt(this.style), parseFloat(this.size));
   }

   /** @summary Create sample with marker in given SVG element
     * @param {selection} svg - SVG element
     * @param {number} width - width of sample SVG
     * @param {number} height - height of sample SVG
     * @private */
   createSample(svg, width, height, plain) {
      if (plain) svg = d3_select(svg);
      this.resetPos();
      svg.append("path")
         .attr("d", this.create(width / 2, height / 2))
         .call(this.func);
   }

} // class TAttMarkerHandler

export { TAttMarkerHandler };
