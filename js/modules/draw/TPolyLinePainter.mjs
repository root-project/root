import { BIT, isStr, clTPolyLine } from '../core.mjs';
import { makeTranslate } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu } from '../gui/menu.mjs';


const kPolyLineNDC = BIT(14);

class TPolyLinePainter extends ObjectPainter {

   /** @summary Dragging object
    *  @private */
   moveDrag(dx, dy) {
      this.dx += dx;
      this.dy += dy;
      makeTranslate(this.draw_g.select('path'), this.dx, this.dy);
   }

   /** @summary End dragging object
    * @private */
   moveEnd(not_changed) {
      if (not_changed) return;
      const polyline = this.getObject(),
            func = this.getAxisToSvgFunc(this.isndc);
      let exec = '';

      for (let n = 0; n <= polyline.fLastPoint; ++n) {
         const x = this.svgToAxis('x', func.x(polyline.fX[n]) + this.dx, this.isndc),
               y = this.svgToAxis('y', func.y(polyline.fY[n]) + this.dy, this.isndc);
         polyline.fX[n] = x;
         polyline.fY[n] = y;
         exec += `SetPoint(${n},${x},${y});;`;
      }
      this.submitCanvExec(exec + 'Notify();;');
      this.redraw();
   }

   /** @summary Returns object ranges
    * @desc Can be used for newly created canvas */
   getUserRanges() {
      const polyline = this.getObject(),
            isndc = polyline.TestBit(kPolyLineNDC);
      if (isndc || !polyline.fLastPoint)
         return null;
      let minx = polyline.fX[0], maxx = minx,
          miny = polyline.fY[0], maxy = miny;
      for (let n = 1; n <= polyline.fLastPoint; ++n) {
         minx = Math.min(minx, polyline.fX[n]);
         maxx = Math.max(maxx, polyline.fX[n]);
         miny = Math.min(miny, polyline.fY[n]);
         maxy = Math.max(maxy, polyline.fY[n]);
      }
      return { minx, miny, maxx, maxy };
   }

   /** @summary Redraw poly line */
   redraw() {
      this.createG();

      const polyline = this.getObject(),
            isndc = polyline.TestBit(kPolyLineNDC),
            opt = this.getDrawOpt() || polyline.fOption,
            dofill = (polyline._typename === clTPolyLine) && (isStr(opt) && opt.toLowerCase().indexOf('f') >= 0),
            func = this.getAxisToSvgFunc(isndc);

      this.createAttLine({ attr: polyline });
      this.createAttFill({ attr: polyline, enable: dofill });

      let cmd = '';
      for (let n = 0; n <= polyline.fLastPoint; ++n)
         cmd += `${n > 0?'L':'M'}${func.x(polyline.fX[n])},${func.y(polyline.fY[n])}`;

      this.draw_g.append('svg:path')
                 .attr('d', cmd + (dofill ? 'Z' : ''))
                 .call(dofill ? () => {} : this.lineatt.func)
                 .call(this.fillatt.func);

      assignContextMenu(this);

      addMoveHandler(this);

      this.dx = this.dy = 0;
      this.isndc = isndc;

      return this;
   }

   /** @summary Draw TPolyLine object */
   static async draw(dom, obj, opt) {
      const painter = new TPolyLinePainter(dom, obj, opt);
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TPolyLinePainter


export { TPolyLinePainter };
