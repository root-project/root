import { ObjectPainter, addMoveHandler, addDrawFunc, ensureTCanvas } from 'jsroot';

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
      const exec = `SetPoints(${tr.fX[0]},${tr.fY[0]},${tr.fX[1]},${tr.fY[1]},${tr.fX[2]},${tr.fY[2]});;`;
      this.submitCanvExec(exec);
   }

   /** @summary Calculate coordinates */
   prepareDraw() {
      const tr = this.getObject();

      this.isndc = true;

      const func = this.getAxisToSvgFunc(this.isndc);
      this.x = []; this.y = [];
      for (let n = 0; n < 3; ++n) {
         this.x[n] = func.x(tr.fX[n]);
         this.y[n] = func.y(tr.fY[n]);
      }

      this.createAttLine({ attr: tr });
      this.createAttFill({ attr: tr });
   }

   /** @summary Create path */
   createPath() {
      return `M${this.x[0]},${this.y[0]}L${this.x[1]},${this.y[1]}L${this.x[2]},${this.y[2]}` + (this.fill ? 'Z' : '');
   }

   /** @summary Redraw line */
   redraw() {
      this.createG();

      this.prepareDraw();

      const elem = this.draw_g.append('svg:path')
                       .attr('d', this.createPath())
                       .call(this.lineatt.func);
      if (this.fill)
         elem.call(this.fillatt.func);
      else
         elem.style('fill', 'none');

      addMoveHandler(this);

      // assignContextMenu(this, kToFront);

      return this;
   }

   /** @summary Draw TLine object */
   static async draw(dom, obj, opt) {
      const painter = new TTrianglePainter(dom, obj, opt);
      painter.fill = (opt === 'f');
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TTrianglePainter

addDrawFunc({ name: 'TTriangle', func: TTrianglePainter.draw });

