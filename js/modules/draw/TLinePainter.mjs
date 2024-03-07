import { BIT } from '../core.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu, kToFront } from '../gui/menu.mjs';


const kLineNDC = BIT(14);

class TLinePainter extends ObjectPainter {

   /** @summary Start interactive moving */
   moveStart(x, y) {
      const fullsize = Math.sqrt((this.x1-this.x2)**2 + (this.y1-this.y2)**2),
          sz1 = Math.sqrt((x-this.x1)**2 + (y-this.y1)**2)/fullsize,
          sz2 = Math.sqrt((x-this.x2)**2 + (y-this.y2)**2)/fullsize;
      if (sz1 > 0.9)
         this.side = 1;
      else if (sz2 > 0.9)
         this.side = -1;
      else
         this.side = 0;
   }

   /** @summary Continue interactive moving */
   moveDrag(dx, dy) {
      if (this.side !== 1) { this.x1 += dx; this.y1 += dy; }
      if (this.side !== -1) { this.x2 += dx; this.y2 += dy; }
      this.draw_g.select('path').attr('d', this.createPath());
   }

   /** @summary Finish interactive moving */
   moveEnd(not_changed) {
      if (not_changed) return;
      const line = this.getObject();
      let exec = '';
      line.fX1 = this.svgToAxis('x', this.x1, this.isndc);
      line.fX2 = this.svgToAxis('x', this.x2, this.isndc);
      line.fY1 = this.svgToAxis('y', this.y1, this.isndc);
      line.fY2 = this.svgToAxis('y', this.y2, this.isndc);
      if (this.side !== 1) exec += `SetX1(${line.fX1});;SetY1(${line.fY1});;`;
      if (this.side !== -1) exec += `SetX2(${line.fX2});;SetY2(${line.fY2});;`;
      this.submitCanvExec(exec + 'Notify();;');
   }

   /** @summary Calculate line coordinates */
   prepareDraw() {
      const line = this.getObject();

      this.isndc = line.TestBit(kLineNDC);

      const func = this.getAxisToSvgFunc(this.isndc, true, true);

      this.x1 = func.x(line.fX1);
      this.y1 = func.y(line.fY1);
      this.x2 = func.x(line.fX2);
      this.y2 = func.y(line.fY2);

      this.createAttLine({ attr: line });
   }

   /** @summary Create path */
   createPath() {
      const x1 = Math.round(this.x1), x2 = Math.round(this.x2), y1 = Math.round(this.y1), y2 = Math.round(this.y2);
      return `M${x1},${y1}` + (x1 === x2 ? `V${y2}` : (y1 === y2 ? `H${x2}` : `L${x2},${y2}`));
   }

   /** @summary Add extras - used for TArrow */
   addExtras() {}

   /** @summary Redraw line */
   redraw() {
      this.createG();

      this.prepareDraw();

      const elem = this.draw_g.append('svg:path')
                       .attr('d', this.createPath())
                       .call(this.lineatt.func);

      if (this.getObject()?.$do_not_draw)
         elem.remove();
      else {
         this.addExtras(elem);
         addMoveHandler(this);
         assignContextMenu(this, kToFront);
      }

      return this;
   }

   /** @summary Draw TLine object */
   static async draw(dom, obj, opt) {
      const painter = new TLinePainter(dom, obj, opt);
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TLinePainter


export { TLinePainter };
