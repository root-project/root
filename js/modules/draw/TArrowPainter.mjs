import { TLinePainter } from './TLinePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';


/** @summary Drawing TArrow
  * @private */
class TArrowPainter extends TLinePainter {

   /** @summary Create line segment with rotation */
   rotate(angle, x0, y0) {
      let dx = this.wsize * Math.cos(angle), dy = this.wsize * Math.sin(angle), res = '';
      if ((x0 !== undefined) && (y0 !== undefined))
         res = `M${Math.round(x0-dx)},${Math.round(y0-dy)}`;
      else {
         dx = -dx; dy = -dy;
      }
      res += `l${Math.round(dx)},${Math.round(dy)}`;
      if (x0 && (y0 === undefined)) res += 'z';
      return res;
   }

   /** @summary Create SVG path for the arrow */
   createPath() {
      const angle = Math.atan2(this.y2 - this.y1, this.x2 - this.x1),
            dlen = this.wsize * Math.cos(this.angle2),
            dx = dlen*Math.cos(angle), dy = dlen*Math.sin(angle);

      let path = '';
      if (this.beg) {
         path += this.rotate(angle - Math.PI - this.angle2, this.x1, this.y1) +
                 this.rotate(angle - Math.PI + this.angle2, this.beg > 10);
      }

      if (this.mid % 10 === 2) {
         path += this.rotate(angle - Math.PI - this.angle2, (this.x1+this.x2-dx)/2, (this.y1+this.y2-dy)/2) +
                 this.rotate(angle - Math.PI + this.angle2, this.mid > 10);
      }

      if (this.mid % 10 === 1) {
         path += this.rotate(angle - this.angle2, (this.x1+this.x2+dx)/2, (this.y1+this.y2+dy)/2) +
                 this.rotate(angle + this.angle2, this.mid > 10);
      }

      if (this.end) {
         path += this.rotate(angle - this.angle2, this.x2, this.y2) +
                 this.rotate(angle + this.angle2, this.end > 10);
      }

      return `M${Math.round(this.x1 + (this.beg > 10 ? dx : 0))},${Math.round(this.y1 + (this.beg > 10 ? dy : 0))}` +
             `L${Math.round(this.x2 - (this.end > 10 ? dx : 0))},${Math.round(this.y2 - (this.end > 10 ? dy : 0))}` +
              path;
   }

   /** @summary calculate all TArrow coordinates */
   prepareDraw() {
      super.prepareDraw();

      const arrow = this.getObject(),
            oo = arrow.fOption,
            rect = this.getPadPainter().getPadRect();

      this.wsize = Math.max(3, Math.round(Math.max(rect.width, rect.height) * arrow.fArrowSize * 0.8));
      this.angle2 = arrow.fAngle/2/180 * Math.PI;
      this.beg = this.mid = this.end = 0;

      if (oo.indexOf('<') === 0)
         this.beg = (oo.indexOf('<|') === 0) ? 12 : 2;
      if (oo.indexOf('->-') >= 0)
         this.mid = 1;
      else if (oo.indexOf('-|>-') >= 0)
         this.mid = 11;
      else if (oo.indexOf('-<-') >= 0)
         this.mid = 2;
      else if (oo.indexOf('-<|-') >= 0)
         this.mid = 12;

      const p1 = oo.lastIndexOf('>'), p2 = oo.lastIndexOf('|>'), len = oo.length;
      if ((p1 >= 0) && (p1 === len-1))
         this.end = ((p2 >= 0) && (p2 === len-2)) ? 11 : 1;

      this.createAttFill({ attr: arrow });
   }

   /** @summary Add extras to path for TArrow */
   addExtras(elem) {
      if ((this.beg > 10) || (this.end > 10))
         elem.call(this.fillatt.func);
      else
         elem.style('fill', 'none');
   }

   /** @summary Draw TArrow object */
   static async draw(dom, obj, opt) {
      const painter = new TArrowPainter(dom, obj, opt);
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TArrowPainter

export { TArrowPainter };
