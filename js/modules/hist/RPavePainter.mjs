import { settings } from '../core.mjs';
import { makeTranslate } from '../base/BasePainter.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';
import { ensureRCanvas } from '../gpad/RCanvasPainter.mjs';
import { addDragHandler } from '../gpad/TFramePainter.mjs';


const ECorner = { kTopLeft: 1, kTopRight: 2, kBottomLeft: 3, kBottomRight: 4 };

/**
 * @summary Painter for RPave class
 *
 * @private
 */

class RPavePainter extends RObjectPainter {

   /** @summary Draw pave content
     * @desc assigned depending on pave class */
   async drawContent() { return this; }

   async drawPaveContent() {
      return this.drawContent().then(() => {
         if (!this.isBatchMode()) {
            // TODO: provide pave context menu as in v6
            if (settings.ContextMenu && this.paveContextMenu)
               this.getG().on('contextmenu', evnt => this.paveContextMenu(evnt));

            addDragHandler(this, {
               x: this.pave_x, y: this.pave_y, width: this.pave_width, height: this.pave_height,
               minwidth: 20, minheight: 20, redraw: d => this.sizeChanged(d)
            });
         }

         return this;
      });
   }


   /** @summary Draw pave */
   async drawPave() {
      const rect = this.getPadPainter().getPadRect(),
            fp = this.getFramePainter();

      this.onFrame = fp && this.v7EvalAttr('onFrame', true);
      this.corner = this.v7EvalAttr('corner', ECorner.kTopRight);

      const visible = this.v7EvalAttr('visible', true),
            offsetx = this.v7EvalLength('offsetX', rect.width, 0.02),
            offsety = this.v7EvalLength('offsetY', rect.height, 0.02),
            pave_width = this.v7EvalLength('width', rect.width, 0.3),
            pave_height = this.v7EvalLength('height', rect.height, 0.3),
            g = this.createG();

      g.classed('most_upper_primitives', true); // this primitive will remain on top of list

      if (!visible)
         return this;

      this.createv7AttLine('border_');

      this.createv7AttFill();

      const fr = this.onFrame ? fp.getFrameRect() : rect;
      switch (this.corner) {
         case ECorner.kTopLeft:
            this.pave_x = fr.x + offsetx;
            this.pave_y = fr.y + offsety;
            break;
         case ECorner.kBottomLeft:
            this.pave_x = fr.x + offsetx;
            this.pave_y = fr.y + fr.height - offsety - pave_height;
            break;
         case ECorner.kBottomRight:
            this.pave_x = fr.x + fr.width - offsetx - pave_width;
            this.pave_y = fr.y + fr.height - offsety - pave_height;
            break;
         case ECorner.kTopRight:
         default:
            this.pave_x = fr.x + fr.width - offsetx - pave_width;
            this.pave_y = fr.y + offsety;
      }

      makeTranslate(g, this.pave_x, this.pave_y);

      g.append('svg:rect')
       .attr('x', 0)
       .attr('width', pave_width)
       .attr('y', 0)
       .attr('height', pave_height)
       .call(this.lineatt.func)
       .call(this.fillatt.func);

      this.pave_width = pave_width;
      this.pave_height = pave_height;

      // draw content and append interactivity
      return this.drawPaveContent();
   }

   /** @summary Process interactive moving of the stats box */
   sizeChanged(drag) {
      this.pave_width = drag.width;
      this.pave_height = drag.height;
      this.pave_x = drag.x;
      this.pave_y = drag.y;

      const rect = this.getPadPainter().getPadRect(),
            fr = this.onFrame ? this.getFramePainter().getFrameRect() : rect,
            changes = {};
      let offsetx, offsety;

      switch (this.corner) {
         case ECorner.kTopLeft:
            offsetx = this.pave_x - fr.x;
            offsety = this.pave_y - fr.y;
            break;
         case ECorner.kBottomLeft:
            offsetx = this.pave_x - fr.x;
            offsety = fr.y + fr.height - this.pave_y - this.pave_height;
            break;
         case ECorner.kBottomRight:
            offsetx = fr.x + fr.width - this.pave_x - this.pave_width;
            offsety = fr.y + fr.height - this.pave_y - this.pave_height;
            break;
         case ECorner.kTopRight:
         default:
            offsetx = fr.x + fr.width - this.pave_x - this.pave_width;
            offsety = this.pave_y - fr.y;
      }

      this.v7AttrChange(changes, 'offsetX', offsetx / rect.width);
      this.v7AttrChange(changes, 'offsetY', offsety / rect.height);
      this.v7AttrChange(changes, 'width', this.pave_width / rect.width);
      this.v7AttrChange(changes, 'height', this.pave_height / rect.height);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      const g = this.getG(),
            main = g.selectChild('rect');
      main.attr('width', this.pave_width)
          .attr('height', this.pave_height);
      g.selectAll('*').filter(function() { return this !== main.node(); }).remove();

      return this.drawPaveContent();
   }

   /** @summary Redraw RPave object */
   async redraw(/* reason */) {
      return this.drawPave();
   }

   /** @summary draw RPave object */
   static async draw(dom, pave, opt) {
      const painter = new RPavePainter(dom, pave, opt, 'pave');
      return ensureRCanvas(painter).then(() => painter.drawPave());
   }

}


/**
 * @summary Painter for RLegend class
 *
 * @private
 */

class RLegendPainter extends RPavePainter {

   /** @summary draw RLegend content */
   async drawContent() {
      const legend = this.getObject(),
            textFont = this.v7EvalFont('text', { size: 12, color: 'black', align: 22 }),
            width = this.pave_width,
            height = this.pave_height,
            pp = this.getPadPainter();

      let nlines = legend.fEntries.length;
      if (legend.fTitle)
         nlines++;

      if (!nlines || !pp)
         return this;

      const stepy = height / nlines, margin_x = 0.02 * width;

      textFont.setSize(height / (nlines * 1.2));
      return this.startTextDrawingAsync(textFont, 'font').then(() => {
         let posy = 0;

         if (legend.fTitle) {
            this.drawText({ latex: 1, width: width - 2 * margin_x, height: stepy, x: margin_x, y: posy, text: legend.fTitle });
            posy += stepy;
         }

         for (let i = 0; i < legend.fEntries.length; ++i) {
            const entry = legend.fEntries[i], w4 = Math.round(width / 4);
            let objp = null;

            this.drawText({ latex: 1, width: 0.75 * width - 3 * margin_x, height: stepy, x: 2 * margin_x + w4, y: posy, text: entry.fLabel });

            if (entry.fDrawableId !== 'custom')
               objp = pp.findSnap(entry.fDrawableId, true);
            else if (entry.fDrawable.fIO) {
               objp = new RObjectPainter(this.getPadPainter(), entry.fDrawable.fIO);
               if (entry.fLine)
                  objp.createv7AttLine();
               if (entry.fFill)
                  objp.createv7AttFill();
               if (entry.fMarker)
                  objp.createv7AttMarker();
            }

            if (entry.fFill && objp?.fillatt) {
               this.appendPath(`M${Math.round(margin_x)},${Math.round(posy + stepy * 0.1)}h${w4}v${Math.round(stepy * 0.8)}h${-w4}z`)
                   .call(objp.fillatt.func);
            }

            if (entry.fLine && objp?.lineatt) {
               this.appendPath(`M${Math.round(margin_x)},${Math.round(posy + stepy / 2)}h${w4}`)
                   .call(objp.lineatt.func);
            }

            if (entry.fError && objp?.lineatt) {
               this.appendPath(`M${Math.round(margin_x + width / 8)},${Math.round(posy + stepy * 0.2)}v${Math.round(stepy * 0.6)}`)
                   .call(objp.lineatt.func);
            }

            if (entry.fMarker && objp?.markeratt) {
               this.appendPath(objp.markeratt.create(margin_x + width / 8, posy + stepy / 2))
                   .call(objp.markeratt.func);
            }

            posy += stepy;
         }

         return this.finishTextDrawing();
      });
   }

   /** @summary draw RLegend object */
   static async draw(dom, legend, opt) {
      const painter = new RLegendPainter(dom, legend, opt, 'legend');
      return ensureRCanvas(painter).then(() => painter.drawPave());
   }

} // class RLegendPainter


/**
 * @summary Painter for RPaveText class
 *
 * @private
 */

class RPaveTextPainter extends RPavePainter {

   /** @summary draw RPaveText content */
   async drawContent() {
      const pavetext = this.getObject(),
            nlines = pavetext?.fText.length;

      if (!nlines)
         return;

      const textFont = this.v7EvalFont('text', { size: 12, color: 'black', align: 22 }),
            width = this.pave_width,
            height = this.pave_height,
            stepy = height / nlines, margin_x = 0.02 * width;

      textFont.setSize(height / (nlines * 1.2));

      return this.startTextDrawingAsync(textFont, 'font').then(() => {
         for (let i = 0, posy = 0; i < pavetext.fText.length; ++i, posy += stepy)
            this.drawText({ latex: 1, width: width - 2 * margin_x, height: stepy, x: margin_x, y: posy, text: pavetext.fText[i] });

         return this.finishTextDrawing(undefined, true);
      });
   }

   /** @summary draw RPaveText object */
   static async draw(dom, pave, opt) {
      const painter = new RPaveTextPainter(dom, pave, opt, 'pavetext');
      return ensureRCanvas(painter).then(() => painter.drawPave());
   }

} // class RPaveTextPainter

export { RPavePainter, RLegendPainter, RPaveTextPainter };
