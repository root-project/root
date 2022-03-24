import { BIT, isBatchMode } from '../core.mjs';

import { ObjectPainter } from '../base/ObjectPainter.mjs';

import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';

import { addMoveHandler } from '../gui/utils.mjs';

/** @summary Draw TArrow
  * @private */
class TArrowPainter extends ObjectPainter {

   rotate(angle, x0, y0) {
      let dx = this.wsize * Math.cos(angle), dy = this.wsize * Math.sin(angle), res = "";
      if ((x0 !== undefined) && (y0 !== undefined)) {
         res =  `M${Math.round(x0-dx)},${Math.round(y0-dy)}`;
      } else {
         dx = -dx; dy = -dy;
      }
      res += `l${Math.round(dx)},${Math.round(dy)}`;
      if (x0 && (y0===undefined)) res+="z";
      return res;
   }

   createPath() {
      let angle = Math.atan2(this.y2 - this.y1, this.x2 - this.x1),
          dlen = this.wsize * Math.cos(this.angle2),
          dx = dlen*Math.cos(angle), dy = dlen*Math.sin(angle),
          path = "";

      if (this.beg)
         path += this.rotate(angle - Math.PI - this.angle2, this.x1, this.y1) +
                 this.rotate(angle - Math.PI + this.angle2, this.beg > 10);

      if (this.mid % 10 == 2)
         path += this.rotate(angle - Math.PI - this.angle2, (this.x1+this.x2-dx)/2, (this.y1+this.y2-dy)/2) +
                 this.rotate(angle - Math.PI + this.angle2, this.mid > 10);

      if (this.mid % 10 == 1)
         path += this.rotate(angle - this.angle2, (this.x1+this.x2+dx)/2, (this.y1+this.y2+dy)/2) +
                 this.rotate(angle + this.angle2, this.mid > 10);

      if (this.end)
         path += this.rotate(angle - this.angle2, this.x2, this.y2) +
                 this.rotate(angle + this.angle2, this.end > 10);

      return `M${Math.round(this.x1 + (this.beg > 10 ? dx : 0))},${Math.round(this.y1 + (this.beg > 10 ? dy : 0))}` +
             `L${Math.round(this.x2 - (this.end > 10 ? dx : 0))},${Math.round(this.y2 - (this.end > 10 ? dy : 0))}` +
              path;
   }

   moveStart(x,y) {
      let fullsize = Math.sqrt(Math.pow(this.x1-this.x2,2) + Math.pow(this.y1-this.y2,2)),
          sz1 = Math.sqrt(Math.pow(x-this.x1,2) + Math.pow(y-this.y1,2))/fullsize,
          sz2 = Math.sqrt(Math.pow(x-this.x2,2) + Math.pow(y-this.y2,2))/fullsize;
      if (sz1>0.9) this.side = 1; else if (sz2>0.9) this.side = -1; else this.side = 0;
   }

   moveDrag(dx,dy) {
      if (this.side != 1) { this.x1 += dx; this.y1 += dy; }
      if (this.side != -1) { this.x2 += dx; this.y2 += dy; }
      this.draw_g.select('path').attr("d", this.createPath());
   }

   moveEnd(not_changed) {
      if (not_changed) return;
      let arrow = this.getObject(), exec = "";
      arrow.fX1 = this.svgToAxis("x", this.x1, this.isndc);
      arrow.fX2 = this.svgToAxis("x", this.x2, this.isndc);
      arrow.fY1 = this.svgToAxis("y", this.y1, this.isndc);
      arrow.fY2 = this.svgToAxis("y", this.y2, this.isndc);
      if (this.side != 1) exec += `SetX1(${arrow.fX1});;SetY1(${arrow.fY1});;`;
      if (this.side != -1) exec += `SetX2(${arrow.fX2});;SetY2(${arrow.fY2});;`;
      this.submitCanvExec(exec + "Notify();;");
   }

   redraw() {
      let arrow = this.getObject(), kLineNDC = BIT(14),
          oo = arrow.fOption, rect = this.getPadPainter().getPadRect();

      this.wsize = Math.max(3, Math.round(Math.max(rect.width, rect.height) * arrow.fArrowSize*0.8));
      this.isndc = arrow.TestBit(kLineNDC);
      this.angle2 = arrow.fAngle/2/180 * Math.PI;
      this.beg = this.mid = this.end = 0;

      if (oo.indexOf("<")==0)
         this.beg = (oo.indexOf("<|") == 0) ? 12 : 2;
      if (oo.indexOf("->-")>=0)
         this.mid = 1;
      else if (oo.indexOf("-|>-")>=0)
         this.mid = 11;
      else if (oo.indexOf("-<-")>=0)
         this.mid = 2;
      else if (oo.indexOf("-<|-")>=0)
         this.mid = 12;

      let p1 = oo.lastIndexOf(">"), p2 = oo.lastIndexOf("|>"), len = oo.length;
      if ((p1 >= 0) && (p1 == len-1))
         this.end = ((p2 >= 0) && (p2 == len-2)) ? 11 : 1;

      this.createAttLine({ attr: arrow });

      this.createG();

      this.x1 = this.axisToSvg("x", arrow.fX1, this.isndc, true);
      this.y1 = this.axisToSvg("y", arrow.fY1, this.isndc, true);
      this.x2 = this.axisToSvg("x", arrow.fX2, this.isndc, true);
      this.y2 = this.axisToSvg("y", arrow.fY2, this.isndc, true);

      let elem = this.draw_g.append("svg:path")
                     .attr("d", this.createPath())
                     .call(this.lineatt.func);

      if ((this.beg > 10) || (this.end > 10)) {
         this.createAttFill({ attr: arrow });
         elem.call(this.fillatt.func);
      } else {
         elem.style('fill','none');
      }

     if (!isBatchMode())
        addMoveHandler(this);

      return this;
   }

   static draw(dom, obj, opt) {
      let painter = new TArrowPainter(dom, obj,opt);
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

}

export { TArrowPainter };
