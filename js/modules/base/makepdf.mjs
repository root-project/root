import { select as d3_select } from '../d3.mjs';
import { jsPDF } from './jspdf.mjs';
import { svg2pdf } from './svg2pdf.mjs';
import { isNodeJs, internals, settings } from '../core.mjs';
import { FontHandler, detectPdfFont, kArial, kCourier, kSymbol, kWingdings } from './FontHandler.mjs';
import { approximateLabelWidth, replaceSymbolsInTextNode } from './latex.mjs';


/** @summary Create pdf for existing SVG element
  * @return {Promise} with produced PDF file as url string
  * @private */
async function makePDF(svg, args) {
   const nodejs = isNodeJs();
   let need_symbols = false;

   const restore_fonts = [], restore_symb = [], restore_wing = [], restore_dominant = [], restore_oblique = [], restore_text = [],
         node_transform = svg.node.getAttribute('transform'), custom_fonts = {};

   if (svg.reset_tranform)
      svg.node.removeAttribute('transform');

   d3_select(svg.node).selectAll('g').each(function() {
      if (this.hasAttribute('font-family')) {
         const name = this.getAttribute('font-family');
         if (name === kCourier) {
            this.setAttribute('font-family', 'courier');
            if (!svg.can_modify) restore_fonts.push(this); // keep to restore it
         }
         if (name === kSymbol) {
            this.setAttribute('font-family', 'symbol');
            if (!svg.can_modify) restore_symb.push(this); // keep to restore it
         }
         if (name === kWingdings) {
            this.setAttribute('font-family', 'zapfdingbats');
            if (!svg.can_modify) restore_wing.push(this); // keep to restore it
         }

         if (((name === kArial) || (name === kCourier)) && (this.getAttribute('font-weight') === 'bold') && (this.getAttribute('font-style') === 'oblique')) {
            this.setAttribute('font-style', 'italic');
            if (!svg.can_modify) restore_oblique.push(this); // keep to restore it
         } else if ((name === kCourier) && (this.getAttribute('font-style') === 'oblique')) {
            this.setAttribute('font-style', 'italic');
            if (!svg.can_modify) restore_oblique.push(this); // keep to restore it
         }
      }
   });

   d3_select(svg.node).selectAll('text').each(function() {
      if (this.hasAttribute('dominant-baseline')) {
         this.setAttribute('dy', '.2em'); // slightly different as in plain text
         this.removeAttribute('dominant-baseline');
         if (!svg.can_modify) restore_dominant.push(this); // keep to restore it
      } else if (svg.can_modify && nodejs && this.getAttribute('dy') === '.4em')
         this.setAttribute('dy', '.2em'); // better alignment in PDF

      if (replaceSymbolsInTextNode(this)) {
         need_symbols = true;
         if (!svg.can_modify) restore_text.push(this); // keep to restore it
      }
   });

   if (nodejs) {
      const doc = internals.nodejs_document;
      doc.originalCreateElementNS = doc.createElementNS;
      globalThis.document = doc;
      globalThis.CSSStyleSheet = internals.nodejs_window.CSSStyleSheet;
      globalThis.CSSStyleRule = internals.nodejs_window.CSSStyleRule;
      doc.createElementNS = function(ns, kind) {
         const res = doc.originalCreateElementNS(ns, kind);
         res.getBBox = function() {
            let width = 50, height = 10;
            if (this.tagName === 'text') {
               // TODO: use jsDOC fonts for label width estimation
               const font = detectPdfFont(this);
               width = approximateLabelWidth(this.textContent, font);
               height = font.size * 1.2;
            }

            return { x: 0, y: 0, width, height };
         };
         return res;
      };
   }

   const orientation = (svg.width < svg.height) ? 'portrait' : 'landscape';

   let doc = args?.as_doc ? args.doc : null;

   if (doc) {
      doc.addPage({
         orientation,
         unit: 'px',
         format: [svg.width + 10, svg.height + 10]
      });
   } else {
      doc = new jsPDF({
         orientation,
         unit: 'px',
         format: [svg.width + 10, svg.height + 10]
      });
      if (args?.as_doc)
         args.doc = doc;
   }

   // add custom fonts to PDF document, only TTF format supported
   d3_select(svg.node).selectAll('style').each(function() {
      const fcfg = this.$fontcfg;
      if (!fcfg?.n || !fcfg?.base64) return;
      const name = fcfg.n;
      if ((name === kSymbol) || (name === kWingdings)) return;
      if (custom_fonts[name]) return;
      custom_fonts[name] = true;

      const filename = name.toLowerCase().replace(/\s/g, '') + '.ttf';
      doc.addFileToVFS(filename, fcfg.base64);
      doc.addFont(filename, fcfg.n, fcfg.s || 'normal');
   });

   let pr = Promise.resolve();
   if (need_symbols && !custom_fonts[kSymbol] && settings.LoadSymbolTtf) {
      const handler = new FontHandler(122, 10);
      pr = handler.load().then(() => {
         handler.addCustomFontToSvg(d3_select(svg.node));
         doc.addFileToVFS(kSymbol + '.ttf', handler.base64);
         doc.addFont(kSymbol + '.ttf', kSymbol, 'normal');
      });
   }

   return pr.then(() => svg2pdf(svg.node, doc, { x: 5, y: 5, width: svg.width, height: svg.height })).then(() => {
      if (svg.reset_tranform && !svg.can_modify && node_transform)
         svg.node.setAttribute('transform', node_transform);

      restore_fonts.forEach(node => node.setAttribute('font-family', kCourier));
      restore_symb.forEach(node => node.setAttribute('font-family', kSymbol));
      restore_wing.forEach(node => node.setAttribute('font-family', kWingdings));
      restore_oblique.forEach(node => node.setAttribute('font-style', 'oblique'));
      restore_dominant.forEach(node => {
         node.setAttribute('dominant-baseline', 'middle');
         node.removeAttribute('dy');
      });

      restore_text.forEach(node => {
         node.innerHTML = node.$originalHTML;
         if (node.$originalFont)
            node.setAttribute('font-family', node.$originalFont);
         else
            node.removeAttribute('font-family');
      });

      const res = args?.as_buffer ? doc.output('arraybuffer') : doc.output('dataurlstring');
      if (nodejs) {
         globalThis.document = undefined;
         globalThis.CSSStyleSheet = undefined;
         globalThis.CSSStyleRule = undefined;
         internals.nodejs_document.createElementNS = internals.nodejs_document.originalCreateElementNS;
         if (args?.as_buffer) return Buffer.from(res);
      }

      return res;
   });
}

export { makePDF };
