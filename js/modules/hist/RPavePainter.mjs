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
      let pave_x = 0, pave_y = 0;
      switch (this.corner) {
         case ECorner.kTopLeft:
            pave_x = fr.x + offsetx;
            pave_y = fr.y + offsety;
            break;
         case ECorner.kBottomLeft:
            pave_x = fr.x + offsetx;
            pave_y = fr.y + fr.height - offsety - pave_height;
            break;
         case ECorner.kBottomRight:
            pave_x = fr.x + fr.width - offsetx - pave_width;
            pave_y = fr.y + fr.height - offsety - pave_height;
            break;
         case ECorner.kTopRight:
         default:
            pave_x = fr.x + fr.width - offsetx - pave_width;
            pave_y = fr.y + offsety;
      }

      makeTranslate(g, pave_x, pave_y);

      g.append('svg:rect')
       .attr('x', 0)
       .attr('width', pave_width)
       .attr('y', 0)
       .attr('height', pave_height)
       .call(this.lineatt.func)
       .call(this.fillatt.func);

      this.pave_width = pave_width;
      this.pave_height = pave_height;

      // here should be fill and draw of text

      return this.drawContent().then(() => {
         if (!this.isBatchMode()) {
            // TODO: provide pave context menu as in v6
            if (settings.ContextMenu && this.paveContextMenu)
               g.on('contextmenu', evnt => this.paveContextMenu(evnt));

            addDragHandler(this, { x: pave_x, y: pave_y, width: pave_width, height: pave_height,
                                   minwidth: 20, minheight: 20, redraw: d => this.sizeChanged(d) });
         }

         return this;
      });
   }

   /** @summary Process interactive moving of the stats box */
   sizeChanged(drag) {
      this.pave_width = drag.width;
      this.pave_height = drag.height;

      const pave_x = drag.x,
            pave_y = drag.y,
            rect = this.getPadPainter().getPadRect(),
            fr = this.onFrame ? this.getFramePainter().getFrameRect() : rect,
            changes = {};
      let offsetx, offsety;

      switch (this.corner) {
         case ECorner.kTopLeft:
            offsetx = pave_x - fr.x;
            offsety = pave_y - fr.y;
            break;
         case ECorner.kBottomLeft:
            offsetx = pave_x - fr.x;
            offsety = fr.y + fr.height - pave_y - this.pave_height;
            break;
         case ECorner.kBottomRight:
            offsetx = fr.x + fr.width - pave_x - this.pave_width;
            offsety = fr.y + fr.height - pave_y - this.pave_height;
            break;
         case ECorner.kTopRight:
         default:
            offsetx = fr.x + fr.width - pave_x - this.pave_width;
            offsety = pave_y - fr.y;
      }

      this.v7AttrChange(changes, 'offsetX', offsetx / rect.width);
      this.v7AttrChange(changes, 'offsetY', offsety / rect.height);
      this.v7AttrChange(changes, 'width', this.pave_width / rect.width);
      this.v7AttrChange(changes, 'height', this.pave_height / rect.height);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.getG().selectChild('rect')
                 .attr('width', this.pave_width)
                 .attr('height', this.pave_height);

      this.drawContent();
   }

   /** @summary Redraw RPave object */
   async redraw(/* reason */) {
      return this.drawPave();
   }

   /** @summary draw RPave object */
   static async draw(dom, pave, opt) {
      const painter = new RPavePainter(dom, pave, opt, 'pave');
      return ensureRCanvas(painter, false).then(() => painter.drawPave());
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
      if (legend.fTitle) nlines++;

      if (!nlines || !pp) return this;

      const stepy = height / nlines, margin_x = 0.02 * width;

      textFont.setSize(height/(nlines * 1.2));
      return this.startTextDrawingAsync(textFont, 'font').then(() => {
         let posy = 0;

         if (legend.fTitle) {
            this.drawText({ latex: 1, width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: legend.fTitle });
            posy += stepy;
         }

         for (let i = 0; i < legend.fEntries.length; ++i) {
            const entry = legend.fEntries[i], w4 = Math.round(width/4);
            let objp = null;

            this.drawText({ latex: 1, width: 0.75*width - 3*margin_x, height: stepy, x: 2*margin_x + w4, y: posy, text: entry.fLabel });

            if (entry.fDrawableId !== 'custom')
               objp = pp.findSnap(entry.fDrawableId, true);
            else if (entry.fDrawable.fIO) {
               objp = new RObjectPainter(this.getPadPainter(), entry.fDrawable.fIO);
               if (entry.fLine) objp.createv7AttLine();
               if (entry.fFill) objp.createv7AttFill();
               if (entry.fMarker) objp.createv7AttMarker();
            }

            if (entry.fFill && objp?.fillatt) {
               this.appendPath(`M${Math.round(margin_x)},${Math.round(posy + stepy*0.1)}h${w4}v${Math.round(stepy*0.8)}h${-w4}z`)
                   .call(objp.fillatt.func);
            }

            if (entry.fLine && objp?.lineatt) {
               this.appendPath(`M${Math.round(margin_x)},${Math.round(posy + stepy/2)}h${w4}`)
                   .call(objp.lineatt.func);
            }

            if (entry.fError && objp?.lineatt) {
               this.appendPath(`M${Math.round(margin_x + width/8)},${Math.round(posy + stepy*0.2)}v${Math.round(stepy*0.6)}`)
                   .call(objp.lineatt.func);
            }

            if (entry.fMarker && objp?.markeratt) {
               this.appendPath(objp.markeratt.create(margin_x + width/8, posy + stepy/2))
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
      return ensureRCanvas(painter, false).then(() => painter.drawPave());
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
            textFont = this.v7EvalFont('text', { size: 12, color: 'black', align: 22 }),
            width = this.pave_width,
            height = this.pave_height,
            nlines = pavetext.fText.length;

      if (!nlines) return;

      const stepy = height / nlines, margin_x = 0.02 * width;

      textFont.setSize(height/(nlines * 1.2));

      return this.startTextDrawingAsync(textFont, 'font').then(() => {
         for (let i = 0, posy = 0; i < pavetext.fText.length; ++i, posy += stepy)
            this.drawText({ latex: 1, width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: pavetext.fText[i] });

         return this.finishTextDrawing(undefined, true);
      });
   }

   /** @summary draw RPaveText object */
   static async draw(dom, pave, opt) {
      const painter = new RPaveTextPainter(dom, pave, opt, 'pavetext');
      return ensureRCanvas(painter, false).then(() => painter.drawPave());
   }

} // class RPaveTextPainter

export { RPavePainter, RLegendPainter, RPaveTextPainter };
