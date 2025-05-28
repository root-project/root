import { BIT } from '../core.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu } from '../gui/menu.mjs';


const kLineNDC = BIT(14);

/**
 * @summary Painter for TLine class
 * @private
 */

class TLinePainter extends ObjectPainter {

   #side; // side which is interactively moved

   /** @summary Start interactive moving */
   moveStart(x, y) {
      const fullsize = Math.max(1, Math.sqrt((this.x1 - this.x2)**2 + (this.y1 - this.y2)**2)),
            sz1 = Math.sqrt((x - this.x1)**2 + (y - this.y1)**2)/fullsize,
            sz2 = Math.sqrt((x - this.x2)**2 + (y - this.y2)**2)/fullsize;
      if (sz1 > 0.9)
         this.#side = 1;
      else if (sz2 > 0.9)
         this.#side = -1;
      else
         this.#side = 0;
   }

   /** @summary Continue interactive moving */
   moveDrag(dx, dy) {
      if (this.#side !== 1) { this.x1 += dx; this.y1 += dy; }
      if (this.#side !== -1) { this.x2 += dx; this.y2 += dy; }
      this.getG().select('path').attr('d', this.createPath());
   }

   /** @summary Finish interactive moving */
   moveEnd(not_changed) {
      if (not_changed) return;
      const line = this.getObject();
      let exec = '',
          fx1 = this.svgToAxis('x', this.x1, this.isndc),
          fx2 = this.svgToAxis('x', this.x2, this.isndc),
          fy1 = this.svgToAxis('y', this.y1, this.isndc),
          fy2 = this.svgToAxis('y', this.y2, this.isndc);
      if (this.swap_xy)
         [fx1, fy1, fx2, fy2] = [fy1, fx1, fy2, fx2];
      line.fX1 = fx1;
      line.fX2 = fx2;
      line.fY1 = fy1;
      line.fY2 = fy2;
      if (this.#side !== 1) exec += `SetX1(${fx1});;SetY1(${fy1});;`;
      if (this.#side !== -1) exec += `SetX2(${fx2});;SetY2(${fy2});;`;
      this.submitCanvExec(exec + 'Notify();;');
   }

   /** @summary Returns object ranges
     * @desc Can be used for newly created canvas */
   getUserRanges() {
      const line = this.getObject(),
            isndc = line.TestBit(kLineNDC);
      if (isndc)
         return null;
      const minx = Math.min(line.fX1, line.fX2),
            maxx = Math.max(line.fX1, line.fX2),
            miny = Math.min(line.fY1, line.fY2),
            maxy = Math.max(line.fY1, line.fY2);
      return { minx, miny, maxx, maxy };
   }

   /** @summary Calculate line coordinates */
   prepareDraw() {
      const line = this.getObject();

      this.isndc = line.TestBit(kLineNDC);

      const use_frame = this.isndc ? false : new DrawOptions(this.getDrawOpt()).check('FRAME');

      this.createG(use_frame ? 'frame2d' : undefined);

      this.swap_xy = use_frame && this.getFramePainter()?.swap_xy();

      const func = this.getAxisToSvgFunc(this.isndc, true);

      this.x1 = func.x(line.fX1);
      this.y1 = func.y(line.fY1);
      this.x2 = func.x(line.fX2);
      this.y2 = func.y(line.fY2);

      if (this.swap_xy)
         [this.x1, this.y1, this.x2, this.y2] = [this.y1, this.x1, this.y2, this.x2];

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
      this.prepareDraw();

      const elem = this.appendPath(this.createPath())
                       .call(this.lineatt.func);

      if (this.getObject()?.$do_not_draw)
         elem.remove();
      else {
         this.addExtras(elem);
         addMoveHandler(this);
         assignContextMenu(this);
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
