import { gStyle, isObject, isStr } from '../core.mjs';
import { color as d3_color, rgb as d3_rgb, select as d3_select } from '../d3.mjs';
import { getColor, findColor, clTLinearGradient, clTRadialGradient, toHex } from './colors.mjs';


/**
  * @summary Handle for fill attributes
  * @private
  */

class TAttFillHandler {

   /** @summary constructor
     * @param {object} args - arguments see {@link TAttFillHandler#setArgs} for more info
     * @param {number} [args.kind = 2] - 1 means object drawing where combination fillcolor == 0 and fillstyle == 1001 means no filling,  2 means all other objects where such combination is white-color filling */
   constructor(args) {
      this.color = 'none';
      this.colorindx = 0;
      this.pattern = 0;
      this.used = true;
      this.kind = args.kind || 2;
      this.changed = false;
      this.func = this.apply.bind(this);
      this.setArgs(args);
      this.changed = false; // unset change property
   }

   /** @summary Set fill style as arguments
     * @param {object} args - different arguments to set fill attributes
     * @param {object} [args.attr] - TAttFill object
     * @param {number} [args.color] - color id
     * @param {number} [args.pattern] - filll pattern id
     * @param {object} [args.svg] - SVG element to store newly created patterns
     * @param {string} [args.color_as_svg] - color in SVG format */
   setArgs(args) {
      if (isObject(args.attr)) {
         if ((args.pattern === undefined) && (args.attr.fFillStyle !== undefined)) args.pattern = args.attr.fFillStyle;
         if ((args.color === undefined) && (args.attr.fFillColor !== undefined)) args.color = args.attr.fFillColor;
      }

      const was_changed = this.changed; // preserve changed state
      this.change(args.color, args.pattern, args.svg, args.color_as_svg, args.painter);
      this.changed = was_changed;
   }

   /** @summary Apply fill style to selection */
   apply(selection) {
      this.used = true;

      selection.style('fill', this.getFillColor());

      if ('opacity' in this)
         selection.style('opacity', this.opacity);

      if ('antialias' in this)
         selection.style('antialias', this.antialias);
   }

   /** @summary Returns fill color (or pattern url) */
   getFillColor() { return this.pattern_url || this.color; }

   /** @summary Returns fill color without pattern url.
     * @desc If empty, alternative color will be provided
     * @param {string} [altern] - alternative color which returned when fill color not exists
     * @private */
   getFillColorAlt(altern) { return this.color && (this.color !== 'none') ? this.color : altern; }

   /** @summary Returns true if color not specified or fill style not specified */
   empty() {
      const fill = this.getFillColor();
      return !fill || (fill === 'none');
   }

   /** @summary Set usage flag of attribute */
   setUsed(flag) {
      this.used = flag;
   }

   /** @summary Returns true if fill attributes has real color */
   hasColor() {
      return this.color && (this.color !== 'none');
   }

   /** @summary Set solid fill color as fill pattern
     * @param {string} col - solid color */
   setSolidColor(col) {
      delete this.pattern_url;
      this.color = col;
      this.pattern = 1001;
   }

   /** @summary Set fill color opacity */
   setOpacity(o) {
      this.opacity = o;
   }

   /** @summary Check if solid fill is used, also color can be checked
     * @param {string} [solid_color] - when specified, checks if fill color matches */
   isSolid(solid_color) {
      if ((this.pattern !== 1001) || this.gradient) return false;
      return !solid_color || (solid_color === this.color);
   }

   /** @summary Method used when color or pattern were changed with OpenUi5 widgets
     * @private */
   verifyDirectChange(painter) {
      if (isStr(this.pattern))
         this.pattern = parseInt(this.pattern);
      if (!Number.isInteger(this.pattern))
         this.pattern = 0;

      this.change(this.color, this.pattern, painter ? painter.getCanvSvg() : null, true, painter);
   }

   /** @summary Method to change fill attributes.
     * @param {number} color - color index
     * @param {number} pattern - pattern index
     * @param {selection} svg - top canvas element for pattern storages
     * @param {string} [color_as_svg] - when color is string, interpret as normal SVG color
     * @param {object} [painter] - when specified, used to extract color by index */
   change(color, pattern, svg, color_as_svg, painter) {
      delete this.pattern_url;
      delete this.gradient;
      this.changed = true;

      if ((color !== undefined) && Number.isInteger(parseInt(color)) && !color_as_svg)
         this.colorindx = parseInt(color);

      if ((pattern !== undefined) && Number.isInteger(parseInt(pattern))) {
         this.pattern = parseInt(pattern);
         delete this.opacity;
         delete this.antialias;
      }

      if ((this.pattern === 1000) && (this.colorindx === 0)) {
         this.pattern_url = 'white';
         return true;
      }

      if (this.pattern === 1000)
         this.pattern = 1001;

      if (this.pattern < 1001) {
         this.pattern_url = 'none';
         return true;
      }

      if (this.isSolid() && (this.colorindx === 0) && (this.kind === 1) && !color_as_svg) {
         this.pattern_url = 'none';
         return true;
      }

      let indx = this.colorindx;

      if (color_as_svg) {
         this.color = color;
         if (color !== 'none') indx = d3_color(color).hex().slice(1); // fictional index produced from color code
      } else
         this.color = painter ? painter.getColor(indx) : getColor(indx);

      if (!isStr(this.color)) {
         if (isObject(this.color) && (this.color?._typename === clTLinearGradient || this.color?._typename === clTRadialGradient))
            this.gradient = this.color;
         this.color = 'none';
      }

      if (this.isSolid()) return true;

      if (!this.gradient) {
         if ((this.pattern >= 4000) && (this.pattern <= 4100)) {
            // special transparent colors (use for subpads)
            this.opacity = (this.pattern - 4000) / 100;
            return true;
         }
         if ((this.pattern < 3000) || (this.color === 'none'))
            return false;
      }

      if (!svg || svg.empty()) return false;

      let id = '', lines = '', lfill = null, fills = '', fills2 = '', w = 2, h = 2;

      if (this.gradient)
         id = `grad_${this.gradient.fNumber}`;
      else {
         id = `pat_${this.pattern}_${indx}`;

         switch (this.pattern) {
            case 3001: w = h = 2; fills = 'M0,0h1v1h-1zM1,1h1v1h-1z'; break;
            case 3002: w = 4; h = 2; fills = 'M1,0h1v1h-1zM3,1h1v1h-1z'; break;
            case 3003: w = h = 4; fills = 'M2,1h1v1h-1zM0,3h1v1h-1z'; break;
            case 3004: w = h = 8; lines = 'M8,0L0,8'; break;
            case 3005: w = h = 8; lines = 'M0,0L8,8'; break;
            case 3006: w = h = 4; lines = 'M1,0v4'; break;
            case 3007: w = h = 4; lines = 'M0,1h4'; break;
            case 3008:
               w = h = 10;
               fills = 'M0,3v-3h3ZM7,0h3v3ZM0,7v3h3ZM7,10h3v-3ZM5,2l3,3l-3,3l-3,-3Z';
               lines = 'M0,3l5,5M3,10l5,-5M10,7l-5,-5M7,0l-5,5';
               break;
            case 3009: w = 12; h = 12; lines = 'M0,0A6,6,0,0,0,12,0M6,6A6,6,0,0,0,12,12M6,6A6,6,0,0,1,0,12'; lfill = 'none'; break;
            case 3010: w = h = 10; lines = 'M0,2h10M0,7h10M2,0v2M7,2v5M2,7v3'; break; // bricks
            case 3011: w = 9; h = 18; lines = 'M5,0v8M2,1l6,6M8,1l-6,6M9,9v8M6,10l3,3l-3,3M0,9v8M3,10l-3,3l3,3'; lfill = 'none'; break;
            case 3012: w = 10; h = 20; lines = 'M5,1A4,4,0,0,0,5,9A4,4,0,0,0,5,1M0,11A4,4,0,0,1,0,19M10,11A4,4,0,0,0,10,19'; lfill = 'none'; break;
            case 3013: w = h = 7; lines = 'M0,0L7,7M7,0L0,7'; lfill = 'none'; break;
            case 3014: w = h = 16; lines = 'M0,0h16v16h-16v-16M0,12h16M12,0v16M4,0v8M4,4h8M0,8h8M8,4v8'; lfill = 'none'; break;
            case 3015: w = 6; h = 12; lines = 'M2,1A2,2,0,0,0,2,5A2,2,0,0,0,2,1M0,7A2,2,0,0,1,0,11M6,7A2,2,0,0,0,6,11'; lfill = 'none'; break;
            case 3016: w = 12; h = 7; lines = 'M0,1A3,2,0,0,1,3,3A3,2,0,0,0,9,3A3,2,0,0,1,12,1'; lfill = 'none'; break;
            case 3017: w = h = 4; lines = 'M3,1l-2,2'; break;
            case 3018: w = h = 4; lines = 'M1,1l2,2'; break;
            case 3019:
               w = h = 12;
               lines = 'M1,6A5,5,0,0,0,11,6A5,5,0,0,0,1,6h-1h1A5,5,0,0,1,6,11v1v-1A5,5,0,0,1,11,6h1h-1A5,5,0,0,1,6,1v-1v1A5,5,0,0,1,1,6';
               lfill = 'none';
               break;
            case 3020: w = 7; h = 12; lines = 'M1,0A2,3,0,0,0,3,3A2,3,0,0,1,3,9A2,3,0,0,0,1,12'; lfill = 'none'; break;
            case 3021: w = h = 8; lines = 'M8,2h-2v4h-4v2M2,0v2h-2'; lfill = 'none'; break; // left stairs
            case 3022: w = h = 8; lines = 'M0,2h2v4h4v2M6,0v2h2'; lfill = 'none'; break; // right stairs
            case 3023: w = h = 8; fills = 'M4,0h4v4zM8,4v4h-4z'; fills2 = 'M4,0L0,4L4,8L8,4Z'; break;
            case 3024: w = h = 16; fills = 'M0,8v8h2v-8zM8,0v8h2v-8M4,14v2h12v-2z'; fills2 = 'M0,2h8v6h4v-6h4v12h-12v-6h-4z'; break;
            case 3025: w = h = 18; fills = 'M5,13v-8h8ZM18,0v18h-18l5,-5h8v-8Z'; break;
            default: {
               if ((this.pattern > 3025) && (this.pattern < 3100)) {
                  // same as 3002, see TGX11.cxx, line 2234
                  w = 4; h = 2; fills = 'M1,0h1v1h-1zM3,1h1v1h-1z'; break;
               }

               const code = this.pattern % 1000,
                     k = code % 10,
                     j = ((code - k) % 100) / 10,
                     i = (code - j * 10 - k) / 100;
               if (!i) break;

               // use flexible hatches only possible when single pattern is used,
               // otherwise it is not possible to adjust pattern dimension that both hatches match with each other
               const use_new = (j === k) || (j === 0) || (j === 5) || (j === 9) || (k === 0) || (k === 5) || (k === 9),
                     pp = painter?.getPadPainter(),
                     scale_size = pp ? Math.max(pp.getPadWidth(), pp.getPadHeight()) : 600,
                     spacing_original = Math.max(0.1, gStyle.fHatchesSpacing * scale_size * 0.001),
                     hatches_spacing = Math.max(1, Math.round(spacing_original)) * 6,
                     sz = i * hatches_spacing; // axis distance between lines

               id += use_new ? `_hn${Math.round(spacing_original*100)}` : `_ho${hatches_spacing}`;

               w = h = 6 * sz; // we use at least 6 steps

               const produce_old = (dy, swap) => {
                  const pos = [];
                  let step = sz, y1 = 0, max = h, y2, x1, x2;

                  // reduce step for smaller angles to keep normal distance approx same
                  if (Math.abs(dy) < 3)
                     step = Math.round(sz / 12 * 9);
                  if (dy === 0) {
                     step = Math.round(sz / 12 * 8);
                     y1 = step / 2;
                  } else if (dy > 0)
                     max -= step;
                  else
                     y1 = step;

                  while (y1 <= max) {
                     y2 = y1 + dy * step;
                     if (y2 < 0) {
                        x2 = Math.round(y1 / (y1 - y2) * w);
                        pos.push(0, y1, x2, 0);
                        pos.push(w, h - y1, w - x2, h);
                     } else if (y2 > h) {
                        x2 = Math.round((h - y1) / (y2 - y1) * w);
                        pos.push(0, y1, x2, h);
                        pos.push(w, h - y1, w - x2, 0);
                     } else
                        pos.push(0, y1, w, y2);
                     y1 += step;
                  }
                  for (let k = 0; k < pos.length; k += 4) {
                     if (swap) {
                        x1 = pos[k+1];
                        y1 = pos[k];
                        x2 = pos[k+3];
                        y2 = pos[k+2];
                     } else {
                        x1 = pos[k];
                        y1 = pos[k+1];
                        x2 = pos[k+2];
                        y2 = pos[k+3];
                     }
                     lines += `M${x1},${y1}`;
                     if (y2 === y1)
                        lines += `h${x2-x1}`;
                     else if (x2 === x1)
                        lines += `v${y2-y1}`;
                     else
                        lines += `L${x2},${y2}`;
                  }
               },

               produce_new = (_aa, _bb, angle, swapx) => {
                  if ((angle === 0) || (angle === 90)) {
                     const dy = i*spacing_original*3,
                           nsteps = Math.round(h / dy),
                           dyreal = h / nsteps;
                     let yy = dyreal/2;

                     while (yy < h) {
                        if (angle === 0)
                           lines += `M0,${Math.round(yy)}h${w}`;
                        else
                           lines += `M${Math.round(yy)},0v${h}`;
                        yy += dyreal;
                     }

                     return;
                  }

                  const a = angle/180*Math.PI,
                        dy = i*spacing_original*3/Math.cos(a),
                        hside = Math.tan(a) * w,
                        hside_steps = Math.round(hside / dy),
                        dyreal = hside / hside_steps,
                        nsteps = Math.floor(h / dyreal);

                  h = Math.round(nsteps * dyreal);

                  let yy = nsteps * dyreal;

                  while (Math.abs(yy-h) < 0.1) yy -= dyreal;

                  while (yy + hside > 0) {
                     let x1 = 0, y1 = yy, x2 = w, y2 = yy + hside;

                     if (y1 < -0.00001) {
                        // cut at the begin
                        x1 = -y1 / hside * w;
                        y1 = 0;
                     } else if (y2 > h) {
                        // cut at the end
                        x2 = (h - y1) / hside * w;
                        y2 = h;
                     }

                     if (swapx) {
                        x1 = w - x1;
                        x2 = w - x2;
                     }

                     lines += `M${Math.round(x1)},${Math.round(y1)}L${Math.round(x2)},${Math.round(y2)}`;
                     yy -= dyreal;
                  }
               },

               func = use_new ? produce_new : produce_old;

               let horiz = false, vertical = false;

               switch (j) {
                  case 0: horiz = true; break;
                  case 1: func(1, false, 10); break;
                  case 2: func(2, false, 20); break;
                  case 3: func(3, false, 30); break;
                  case 4: func(6, false, 45); break;
                  case 6: func(3, true, 60); break;
                  case 7: func(2, true, 70); break;
                  case 8: func(1, true, 80); break;
                  case 9: vertical = true; break;
               }

               switch (k) {
                  case 0: horiz = true; break;
                  case 1: func(-1, false, 10, true); break;
                  case 2: func(-2, false, 20, true); break;
                  case 3: func(-3, false, 30, true); break;
                  case 4: func(-6, false, 45, true); break;
                  case 6: func(-3, true, 60, true); break;
                  case 7: func(-2, true, 70, true); break;
                  case 8: func(-1, true, 80, true); break;
                  case 9: vertical = true; break;
               }

               if (horiz) func(0, false, 0);
               if (vertical) func(0, true, 90);

               break;
            }
         }

         if (!fills && !lines) return false;
      }

      this.pattern_url = `url(#${id})`;
      this.antialias = false;

      let defs = svg.selectChild('.canvas_defs');
      if (defs.empty())
         defs = svg.insert('svg:defs', ':first-child').attr('class', 'canvas_defs');

      if (defs.selectChild('.' + id).empty()) {
         if (this.gradient) {
            const is_linear = this.gradient._typename === clTLinearGradient,
                  grad = defs.append(is_linear ? 'svg:linearGradient' : 'svg:radialGradient')
                             .attr('id', id).attr('class', id),
                  conv = v => { return v === Math.round(v) ? v.toFixed(0) : v.toFixed(2); };
            if (is_linear) {
               grad.attr('x1', conv(this.gradient.fStart.fX))
                   .attr('y1', conv(1 - this.gradient.fStart.fY))
                   .attr('x2', conv(this.gradient.fEnd.fX))
                   .attr('y2', conv(1 - this.gradient.fEnd.fY));
            } else {
               grad.attr('cx', conv(this.gradient.fStart.fX))
                   .attr('cy', conv(1 - this.gradient.fStart.fY))
                   .attr('cr', conv(this.gradient.fR1));
            }
            for (let n = 0; n < this.gradient.fColorPositions.length; ++n) {
               const pos = this.gradient.fColorPositions[n],
                     col = '#' + toHex(this.gradient.fColors[n*4]) + toHex(this.gradient.fColors[n*4+1]) + toHex(this.gradient.fColors[n*4+2]);
               grad.append('svg:stop').attr('offset', `${Math.round(pos*100)}%`)
                                      .attr('stop-color', col)
                                      .attr('stop-opacity', `${Math.round(this.gradient.fColors[n*4+3]*100)}%`);
            }
         } else {
            const patt = defs.append('svg:pattern')
                             .attr('id', id).attr('class', id).attr('patternUnits', 'userSpaceOnUse')
                             .attr('width', w).attr('height', h);

            if (fills2) {
               const col = d3_rgb(this.color);
               col.r = Math.round((col.r + 255) / 2); col.g = Math.round((col.g + 255) / 2); col.b = Math.round((col.b + 255) / 2);
               patt.append('svg:path').attr('d', fills2).style('fill', col);
            }
            if (fills) patt.append('svg:path').attr('d', fills).style('fill', this.color);
            if (lines) patt.append('svg:path').attr('d', lines).style('stroke', this.color).style('stroke-width', gStyle.fHatchesLineWidth || 1).style('fill', lfill);
         }
      }

      return true;
   }

   /** @summary Create sample of fill pattern inside SVG
     * @private */
   createSample(svg, width, height, plain) {
      // we need to create extra handle to change
      if (plain) svg = d3_select(svg);

      const sample = new TAttFillHandler({ svg, pattern: this.pattern, color: this.color, color_as_svg: true });

      svg.append('path')
         .attr('d', `M0,0h${width}v${height}h${-width}z`)
         .call(sample.func);
   }

   /** @summary Save fill attributes to style
     * @private */
   saveToStyle(name_color, name_pattern) {
      if (name_color) {
         const indx = this.colorindx ?? findColor(this.color);
         if (indx >= 0) gStyle[name_color] = indx;
      }
      if (name_pattern)
         gStyle[name_pattern] = this.pattern;
   }

} // class TAttFillHandler

export { TAttFillHandler };
