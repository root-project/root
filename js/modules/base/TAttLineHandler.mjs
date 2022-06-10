import { select as d3_select } from '../d3.mjs';
import { gStyle } from '../core.mjs';
import { getColor, findColor } from './colors.mjs';


const root_line_styles = [
      "", "", "3,3", "1,2",
      "3,4,1,4", "5,3,1,3", "5,3,1,3,1,3,1,3", "5,5",
      "5,3,1,3,1,3", "20,5", "20,10,1,10", "1,3"];

/**
  * @summary Handle for line attributes
  * @private
  */

class TAttLineHandler {

   /** @summary constructor
     * @param {object} attr - attributes, see {@link TAttLineHandler#setArgs} */
   constructor(args) {
      this.func = this.apply.bind(this);
      this.used = true;
      if (args._typename && (args.fLineStyle !== undefined)) args = { attr: args };
      this.setArgs(args);
   }

   /** @summary Set line attributes.
     * @param {object} args - specify attributes by different ways
     * @param {object} args.attr - TAttLine object with appropriate data members or
     * @param {string} args.color - color in html like rgb(255,0,0) or "red" or "#ff0000"
     * @param {number} args.style - line style number
     * @param {number} args.width - line width */
   setArgs(args) {
      if (args.attr) {
         this.color_index = args.attr.fLineColor;
         args.color = args.color0 || (args.painter ? args.painter.getColor(this.color_index) : getColor(this.color_index));
         if (args.width === undefined) args.width = args.attr.fLineWidth;
         if (args.style === undefined) args.style = args.attr.fLineStyle;
      } else if (typeof args.color == 'string') {
         if ((args.color !== 'none') && !args.width) args.width = 1;
      } else if (typeof args.color == 'number') {
         this.color_index = args.color;
         args.color = args.painter ? args.painter.getColor(args.color) : getColor(args.color);
      }

      if (args.width === undefined)
         args.width = (args.color && args.color != 'none') ? 1 : 0;

      this.color = (args.width === 0) ? 'none' : args.color;
      this.width = args.width;
      this.style = args.style;
      this.pattern = args.pattern || root_line_styles[this.style] || null;

      if (args.can_excl) {
         this.excl_side = this.excl_width = 0;
         if (Math.abs(this.width) > 99) {
            // exclusion graph
            this.excl_side = (this.width < 0) ? -1 : 1;
            this.excl_width = Math.floor(this.width / 100) * 5;
            this.width = Math.abs(this.width % 100); // line width
         }
      }

      // if custom color number used, use lightgrey color to show lines
      if (!this.color && (this.width > 0))
         this.color = 'lightgrey';
   }

   /** @summary Change exclusion attributes */
   changeExcl(side, width) {
      if (width !== undefined)
         this.excl_width = width;
      if (side !== undefined) {
         this.excl_side = side;
         if ((this.excl_width === 0) && (this.excl_side !== 0)) this.excl_width = 20;
      }
      this.changed = true;
   }

   /** @summary returns true if line attribute is empty and will not be applied. */
   empty() { return this.color == 'none'; }

   /** @summary set border parameters, used for rect drawing */
   setBorder(rx, ry) {
      this.rx = rx;
      this.ry = ry;
      this.func = this.applyBorder.bind(this);
   }

   /** @summary Applies line attribute to selection.
     * @param {object} selection - d3.js selection */
   apply(selection) {
      this.used = true;
      if (this.empty())
         selection.style('stroke', null)
                  .style('stroke-width', null)
                  .style('stroke-dasharray', null);
      else
         selection.style('stroke', this.color)
                  .style('stroke-width', this.width)
                  .style('stroke-dasharray', this.pattern);
   }

   /** @summary Applies line and border attribute to selection.
     * @param {object} selection - d3.js selection */
   applyBorder(selection) {
      this.used = true;
      if (this.empty())
         selection.style('stroke', null)
                  .style('stroke-width', null)
                  .style('stroke-dasharray', null)
                  .attr("rx", null).attr("ry", null);
      else
         selection.style('stroke', this.color)
                  .style('stroke-width', this.width)
                  .style('stroke-dasharray', this.pattern)
                  .attr("rx", this.rx || null).attr("ry", this.ry || null);
   }

   /** @summary Change line attributes */
   change(color, width, style) {
      if (color !== undefined) {
         if (this.color !== color)
            delete this.color_index;
         this.color = color;
      }
      if (width !== undefined)
         this.width = width;
      if (style !== undefined) {
         this.style = style;
         this.pattern = root_line_styles[this.style] || null;
      }
      this.changed = true;
   }

   /** @summary Create sample element inside primitive SVG - used in context menu */
   createSample(svg, width, height, plain) {
      if (plain) svg = d3_select(svg);
      svg.append("path")
         .attr("d", `M0,${height/2}h${width}`)
         .call(this.func);
   }

   saveToStyle(name_color, name_width, name_style) {
      if (name_color) {
         let indx = (this.color_index !== undefined) ? this.color_index : findColor(this.color);
         if (indx >= 0)
            gStyle[name_color] = indx;
      }
      if (name_width)
        gStyle[name_width] = this.width;
      if (name_style)
        gStyle[name_style] = this.style;
   }

} // class TAttLineHandler

/** @summary Get svg string for specified line style
  * @private */
function getSvgLineStyle(indx) {
   if ((indx < 0) || (indx >= root_line_styles.length)) indx = 11;
   return root_line_styles[indx];
}


export { TAttLineHandler, getSvgLineStyle, root_line_styles };

