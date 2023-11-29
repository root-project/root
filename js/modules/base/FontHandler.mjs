const kArial = 'Arial', kTimes = 'Times New Roman', kCourier = 'Courier New', kVerdana = 'Verdana', kSymbol = 'Symbol', kWingdings = 'Wingdings',
// average width taken from symbols.html, counted only for letters and digits
root_fonts = [null,  // index 0 not exists
      { n: kTimes, s: 'italic', aw: 0.5314 },
      { n: kTimes, w: 'bold', aw: 0.5809 },
      { n: kTimes, s: 'italic', w: 'bold', aw: 0.5540 },
      { n: kArial, aw: 0.5778 },
      { n: kArial, s: 'oblique', aw: 0.5783 },
      { n: kArial, w: 'bold', aw: 0.6034 },
      { n: kArial, s: 'oblique', w: 'bold', aw: 0.6030 },
      { n: kCourier, aw: 0.6003 },
      { n: kCourier, s: 'oblique', aw: 0.6004 },
      { n: kCourier, w: 'bold', aw: 0.6003 },
      { n: kCourier, s: 'oblique', w: 'bold', aw: 0.6005 },
      { n: kSymbol, aw: 0.5521 },
      { n: kTimes, aw: 0.5521 },
      { n: kWingdings, aw: 0.5664 },
      { n: kSymbol, s: 'italic', aw: 0.5314 },
      { n: kVerdana, aw: 0.5664 },
      { n: kVerdana, s: 'italic', aw: 0.5495 },
      { n: kVerdana, w: 'bold', aw: 0.5748 },
      { n: kVerdana, s: 'italic', w: 'bold', aw: 0.5578 }];

/**
 * @summary Helper class for font handling
 * @private
 */

class FontHandler {

   /** @summary constructor */
   constructor(fontIndex, size, scale) {
      if (scale && (size < 1)) {
         size *= scale;
         this.scaled = true;
      }

      this.size = Math.round(size || 11);
      this.scale = scale;

      this.func = this.setFont.bind(this);

      const indx = (fontIndex && Number.isInteger(fontIndex)) ? Math.floor(fontIndex / 10) : 0,
            cfg = root_fonts[indx];

      if (cfg)
         this.setNameStyleWeight(cfg.n, cfg.s, cfg.w, cfg.aw, cfg.format, cfg.base64);
      else
         this.setNameStyleWeight(kArial);
   }

   /** @summary Directly set name, style and weight for the font
    * @private */
   setNameStyleWeight(name, style, weight, aver_width, format, base64) {
      this.name = name;
      this.style = style || null;
      this.weight = weight || null;
      this.aver_width = aver_width || (weight ? 0.58 : 0.55);
      this.format = format; // format of custom font, ttf by default
      this.base64 = base64; // indication of custom font
      if ((this.name === kSymbol) || (this.name === kWingdings)) {
         this.isSymbol = this.name;
         this.name = kTimes;
      } else
         this.isSymbol = '';
   }

   /** @summary Set painter for which font will be applied */
   setPainter(painter) {
      this.painter = painter;
   }

   /** @summary Assigns font-related attributes */
   addCustomFontToSvg(svg) {
      if (!this.base64 || !this.name)
         return;
      const clname = 'custom_font_' + this.name, fmt = 'ttf';
      let defs = svg.selectChild('.canvas_defs');
      if (defs.empty())
         defs = svg.insert('svg:defs', ':first-child').attr('class', 'canvas_defs');
      const entry = defs.selectChild('.' + clname);
      if (entry.empty()) {
         console.log('Adding style entry for class', clname);
         defs.append('style')
               .attr('class', clname)
               .property('$fonthandler', this)
               .text(`@font-face { font-family: "${this.name}"; font-weight: normal; font-style: normal; src: url(data:application/font-${fmt};charset=utf-8;base64,${this.base64}); }`);
      }
   }

   /** @summary Assigns font-related attributes */
   setFont(selection) {
      if (this.base64 && this.painter)
         this.addCustomFontToSvg(this.painter.getCanvSvg());

      selection.attr('font-family', this.name)
               .attr('font-size', this.size)
               .attr('xml:space', 'preserve')
               .attr('font-weight', this.weight || null)
               .attr('font-style', this.style || null);
   }

   /** @summary Set font size (optional) */
   setSize(size) { this.size = Math.round(size); }

   /** @summary Set text color (optional) */
   setColor(color) { this.color = color; }

   /** @summary Set text align (optional) */
   setAlign(align) { this.align = align; }

   /** @summary Set text angle (optional) */
   setAngle(angle) { this.angle = angle; }

   /** @summary Allign angle to step raster, add optional offset */
   roundAngle(step, offset) {
      this.angle = parseInt(this.angle || 0);
      if (!Number.isInteger(this.angle)) this.angle = 0;
      this.angle = Math.round(this.angle/step) * step + (offset || 0);
      if (this.angle < 0)
         this.angle += 360;
      else if (this.angle >= 360)
         this.angle -= 360;
   }

   /** @summary Clears all font-related attributes */
   clearFont(selection) {
      selection.attr('font-family', null)
               .attr('font-size', null)
               .attr('xml:space', null)
               .attr('font-weight', null)
               .attr('font-style', null);
   }

   /** @summary Returns true in case of monospace font
     * @private */
   isMonospace() {
      const n = this.name.toLowerCase();
      return (n.indexOf('courier') === 0) || (n === 'monospace') || (n === 'monaco');
   }

   /** @summary Return full font declaration which can be set as font property like '12pt Arial bold'
     * @private */
   getFontHtml() {
      let res = Math.round(this.size) + 'pt ' + this.name;
      if (this.weight) res += ' ' + this.weight;
      if (this.style) res += ' ' + this.style;
      return res;
   }

   /** @summary Returns font name */
   getFontName() {
      return this.isSymbol || this.name || 'none';
   }

} // class FontHandler

/** @summary Register custom font
  * @private */
function addCustomFont(index, name, format, base64) {
   if (!Number.isInteger(index))
      console.error(`Wrong index ${index} for custom font`);
   else
      root_fonts[index] = { n: name, format, base64 };
}

/** @summary Return handle with custom font
  * @private */
function getCustomFont(name) {
   return root_fonts.find(h => (h?.n === name) && h?.base64);
}

/** @summary Try to detect and create font handler for SVG text node
  * @private */
function detectFont(node) {
   const sz = node.getAttribute('font-size'),
         family = node.getAttribute('font-family'),
         p = sz.indexOf('px'),
         sz_pixels = p > 0 ? Number.parseInt(sz.slice(0, p)) : 12;
   let style = node.getAttribute('font-style'),
       weight = node.getAttribute('font-weight'),
      fontIndx = null, name = '';
   if (weight === 'normal')
      weight = '';
   else if (weight === 'bold')
      name += 'b';
   if (style === 'normal')
      style = '';
   else if (style === 'italic')
      name += 'i';
   else if (style === 'oblique')
      name += 'o';

   if (family === 'arial')
      name += 'Arial';
   else if (family === 'times')
      name += 'Times New Roman';
   else if (family === 'verdana')
      name += 'Verdana';

   for (let n = 1; n < root_fonts.length; ++n) {
      if (name === root_fonts[n]) {
         fontIndx = n*10 + 2;
         break;
      }
   }

   const handler = new FontHandler(fontIndx, sz_pixels);
   if (!fontIndx)
      handler.setNameStyleWeight(family, style, weight);
   return handler;
}

export { FontHandler, addCustomFont, getCustomFont, detectFont };
