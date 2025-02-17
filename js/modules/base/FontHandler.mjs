import { isNodeJs, httpRequest, btoa_func, source_dir, settings, isObject } from '../core.mjs';


const kArial = 'Arial', kTimes = 'Times New Roman', kCourier = 'Courier New', kVerdana = 'Verdana', kSymbol = 'RootSymbol', kWingdings = 'Wingdings',
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
      { n: kSymbol, aw: 0.5521, file: 'symbol.ttf' },
      { n: kTimes, aw: 0.5521 },
      { n: kWingdings, aw: 0.5664, file: 'wingding.ttf' },
      { n: kSymbol, s: 'oblique', aw: 0.5314, file: 'symbol.ttf' },
      { n: kVerdana, aw: 0.5664 },
      { n: kVerdana, s: 'italic', aw: 0.5495 },
      { n: kVerdana, w: 'bold', aw: 0.5748 },
      { n: kVerdana, s: 'italic', w: 'bold', aw: 0.5578 }],
   // list of loaded fonts including handling of multiple simultaneous requests
   gFontFiles = {};

/** @summary Read font file from some pre-configured locations
  * @return {Promise} with base64 code of the font
  * @private */
async function loadFontFile(fname) {
   let entry = gFontFiles[fname];
   if (entry?.base64)
      return entry?.base64;

   if (entry?.promises !== undefined) {
      return new Promise(resolveFunc => {
         cfg.promises.push(resolveFunc);
      });
   }

   entry = gFontFiles[fname] = { promises: [] };

   const locations = [];
   if (fname.indexOf('/') >= 0)
      locations.push(''); // just use file name as is
   else {
      locations.push(source_dir + 'fonts/');
      if (isNodeJs())
         locations.push('../../fonts/');
      else if (source_dir.indexOf('jsrootsys/') >= 0) {
         locations.unshift(source_dir.replace(/jsrootsys/g, 'rootsys_fonts'));
         locations.unshift(source_dir.replace(/jsrootsys/g, 'rootsys/fonts'));
      }
   }

   function completeReading(base64) {
      entry.base64 = base64;
      const arr = entry.promises;
      delete entry.promises;
      arr.forEach(func => func(base64));
      return base64;
   }

   async function tryNext() {
      if (locations.length === 0) {
         completeReading(null);
         throw new Error(`Fail to load ${fname} font`);
      }
      let path = locations.shift() + fname;
      console.log('loading font', path);
      const pr = isNodeJs() ? import('fs').then(fs => {
         const prefix = 'file://' + (process?.platform === 'win32' ? '/' : '');
         if (path.indexOf(prefix) === 0)
            path = path.slice(prefix.length);
         return fs.readFileSync(path).toString('base64');
      }) : httpRequest(path, 'bin').then(buf => btoa_func(buf));

      return pr.then(res => res ? completeReading(res) : tryNext()).catch(() => tryNext());
   }

   return tryNext();
}


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

      this.size = Math.round(size);
      this.scale = scale;
      this.index = 0;

      this.func = this.setFont.bind(this);

      let cfg = null;

      if (fontIndex && isObject(fontIndex))
         cfg = fontIndex;
      else {
         if (fontIndex && Number.isInteger(fontIndex))
            this.index = Math.floor(fontIndex / 10);
         cfg = root_fonts[this.index];
      }

      if (cfg) {
         this.cfg = cfg;
         this.setNameStyleWeight(cfg.n, cfg.s, cfg.w, cfg.aw, cfg.format, cfg.base64);
      } else
         this.setNameStyleWeight(kArial);
   }

   /** @summary Should returns true if font has to be loaded before
    * @private */
   needLoad() { return this.cfg?.file && !this.isSymbol && !this.base64; }

   /** @summary Async function to load font
    * @private */
   async load() {
      if (!this.needLoad())
         return true;
      return loadFontFile(this.cfg.file).then(base64 => {
         this.cfg.base64 = this.base64 = base64;
         this.format = 'ttf';
         return !!base64;
      });
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
      if (!settings.LoadSymbolTtf && ((this.name === kSymbol) || (this.name === kWingdings))) {
         this.isSymbol = this.name;
         this.name = kTimes;
      } else
         this.isSymbol = '';
   }

   /** @summary Set painter for which font will be applied */
   setPainter(painter) {
      this.painter = painter;
   }

   /** @summary Force setting of style and weight, used in latex */
   setUseFullStyle(flag) {
      this.full_style = flag;
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
         defs.append('style')
               .attr('class', clname)
               .property('$fontcfg', this.cfg || null)
               .text(`@font-face { font-family: "${this.name}"; font-weight: normal; font-style: normal; src: url(data:application/font-${fmt};charset=utf-8;base64,${this.base64}); }`);
      }
   }

   /** @summary Assigns font-related attributes */
   setFont(selection) {
      if (this.base64 && this.painter)
         this.addCustomFontToSvg(this.painter.getCanvSvg());

      selection.attr('font-family', this.name)
               .attr('font-size', this.size)
               .attr(':xml:space', 'preserve');
      this.setFontStyle(selection);
   }

   /** @summary Assigns only font style attributes */
   setFontStyle(selection) {
      selection.attr('font-weight', this.weight || (this.full_style ? 'normal' : null))
               .attr('font-style', this.style || (this.full_style ? 'normal' : null));
   }

   /** @summary Set font size (optional) */
   setSize(size) { this.size = Math.round(size); }

   /** @summary Set text color (optional) */
   setColor(color) { this.color = color; }

   /** @summary Set text align (optional) */
   setAlign(align) { this.align = align; }

   /** @summary Set text angle (optional) */
   setAngle(angle) { this.angle = angle; }

   /** @summary Align angle to step raster, add optional offset */
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
               .attr(':xml:space', null)
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
function detectPdfFont(node) {
   const sz = node.getAttribute('font-size'),
         p = sz.indexOf('px'),
         sz_pixels = p > 0 ? Number.parseInt(sz.slice(0, p)) : 12;

   let family = node.getAttribute('font-family'),
       style = node.getAttribute('font-style'),
       weight = node.getAttribute('font-weight');

   if (family === 'times')
      family = kTimes;
   else if (family === 'symbol')
      family = kSymbol;
   else if (family === 'arial')
      family = kArial;
   else if (family === 'verdana')
      family = kVerdana;
   if (weight === 'normal')
      weight = '';
   if (style === 'normal')
      style = '';

   const fcfg = root_fonts.find(elem => {
      return (elem?.n === family) &&
             ((!weight && !elem.w) || (elem.w === weight)) &&
             ((!style && !elem.s) || (elem.s === style));
   });

   return new FontHandler(fcfg || root_fonts[13], sz_pixels);
}


export { kArial, kCourier, kSymbol, kWingdings, kTimes,
         FontHandler, addCustomFont, getCustomFont, detectPdfFont };
