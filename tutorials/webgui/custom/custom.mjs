import { ObjectPainter, addMoveHandler, addDrawFunc, ensureTCanvas, assignContextMenu, kToFront } from 'jsroot';

// $$jsroot_batch_conform$$
// specially mark script that it can be converted into the batch mode
// only for 'simple' scripts it is possible - without any extra include beside main jsroot module
// it is workaround until proper use of ES6 modules in headless browser will be possible

class TTrianglePainter extends ObjectPainter {

   /** @summary Start interactive moving */
   moveStart(x, y) {
   }

   /** @summary Continue interactive moving */
   moveDrag(dx, dy) {
      for (let n = 0; n < 3; ++n) {
         this.x[n] += dx;
         this.y[n] += dy;
      }
      this.draw_g.select('path').attr('d', this.createPath());
   }

   /** @summary Finish interactive moving */
   moveEnd(not_changed) {
      if (not_changed) return;
      const tr = this.getObject();
      for (let n = 0; n < 3; ++n) {
         tr.fX[n] = this.svgToAxis('x', this.x[n], this.isndc);
         tr.fY[n] = this.svgToAxis('y', this.y[n], this.isndc);
      }
      // submit to server method which will be executed
      this.submitCanvExec(`SetPoints(${tr.fX[0]},${tr.fY[0]},${tr.fX[1]},${tr.fY[1]},${tr.fX[2]},${tr.fY[2]});`);
   }

   /** @summary Create path */
   createPath() {
      return `M${this.x[0]},${this.y[0]}L${this.x[1]},${this.y[1]}L${this.x[2]},${this.y[2]}` + (this.fill ? 'Z' : '');
   }

   /** @summary Redraw triangle */
   redraw() {
      this.createG();

      const tr = this.getObject();

      this.isndc = true;

      this.x = []; this.y = [];
      for (let n = 0; n < 3; ++n) {
         this.x[n] = this.axisToSvg('x', tr.fX[n], this.isndc);
         this.y[n] = this.axisToSvg('y', tr.fY[n], this.isndc);
      }

      this.createAttLine();
      this.createAttFill();

      this.fillatt.enable(this.fill);

      this.draw_g.append('svg:path')
                 .attr('d', this.createPath())
                 .call(this.lineatt.func)
                 .call(this.fillatt.func);

      addMoveHandler(this);

      assignContextMenu(this, kToFront);

      return this;
   }

   /** @summary Draw TTriangle object */
   static async draw(dom, obj, opt) {
      const painter = new TTrianglePainter(dom, obj, opt);
      painter.fill = (opt === 'f');
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TTrianglePainter

addDrawFunc({ name: 'TTriangle', func: TTrianglePainter.draw, opt: ";f" });

