import { rgb as d3_rgb, select as d3_select } from '../d3.mjs';
import { DrawOptions, getBoxDecorations } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu } from '../gui/menu.mjs';

class TBoxPainter extends ObjectPainter {

   /** @summary start of drag handler
     * @private */
   moveStart(x, y) {
      const ww = Math.abs(this.x2 - this.x1), hh = Math.abs(this.y1 - this.y2);

      this.c_x1 = Math.abs(x - this.x2) > ww * 0.1;
      this.c_x2 = Math.abs(x - this.x1) > ww * 0.1;
      this.c_y1 = Math.abs(y - this.y2) > hh * 0.1;
      this.c_y2 = Math.abs(y - this.y1) > hh * 0.1;
      if (this.c_x1 !== this.c_x2 && this.c_y1 && this.c_y2)
         this.c_y1 = this.c_y2 = false;
      if (this.c_y1 !== this.c_y2 && this.c_x1 && this.c_x2)
         this.c_x1 = this.c_x2 = false;
   }

   /** @summary drag handler
     * @private */
   moveDrag(dx, dy) {
      if (this.c_x1) this.x1 += dx;
      if (this.c_x2) this.x2 += dx;
      if (this.c_y1) this.y1 += dy;
      if (this.c_y2) this.y2 += dy;

      const nodes = this.draw_g.selectAll('path').nodes(),
            pathes = this.getPathes();

      pathes.forEach((path, i) => d3_select(nodes[i]).attr('d', path));
   }

   /** @summary end of drag handler
     * @private */
   moveEnd(not_changed) {
      if (not_changed) return;
      const box = this.getObject(), X = this.swap_xy ? 'Y' : 'X', Y = this.swap_xy ? 'X' : 'Y';
      let exec = '';
      if (this.c_x1) { const v = this.svgToAxis('x', this.x1); box[`f${X}1`] = v; exec += `Set${X}1(${v});;`; }
      if (this.c_x2) { const v = this.svgToAxis('x', this.x2); box[`f${X}2`] = v; exec += `Set${X}2(${v});;`; }
      if (this.c_y1) { const v = this.svgToAxis('y', this.y1); box[`f${Y}1`] = v; exec += `Set${Y}1(${v});;`; }
      if (this.c_y2) { const v = this.svgToAxis('y', this.y2); box[`f${Y}2`] = v; exec += `Set${Y}2(${v});;`; }
      this.submitCanvExec(exec + 'Notify();;');
   }

   /** @summary Returns object ranges
     * @desc Can be used for newly created canvas */
   getUserRanges() {
      const box = this.getObject(),
            minx = Math.min(box.fX1, box.fX2),
            maxx = Math.max(box.fX1, box.fX2),
            miny = Math.min(box.fY1, box.fY2),
            maxy = Math.max(box.fY1, box.fY2);
      return { minx, miny, maxx, maxy };
   }

   /** @summary Create path */
   getPathes() {
      const xx = Math.round(Math.min(this.x1, this.x2)),
            yy = Math.round(Math.min(this.y1, this.y2)),
            ww = Math.round(Math.abs(this.x2 - this.x1)),
            hh = Math.round(Math.abs(this.y1 - this.y2)),
            path = `M${xx},${yy}h${ww}v${hh}h${-ww}z`;
      if (!this.borderMode)
         return [path];
      return [path].concat(getBoxDecorations(xx, yy, ww, hh, this.borderMode, this.borderSize, this.borderSize));
   }

   /** @summary Redraw box */
   redraw() {
      const box = this.getObject(),
            d = new DrawOptions(this.getDrawOpt()),
            fp = d.check('FRAME') ? this.getFramePainter() : null,
            draw_line = d.check('L');

      this.createAttLine({ attr: box });
      this.createAttFill({ attr: box });

      this.swap_xy = fp?.swap_xy;

      // if box filled, contour line drawn only with 'L' draw option:
      if (!this.fillatt.empty() && !draw_line)
         this.lineatt.color = 'none';

      this.createG(fp);

      this.x1 = this.axisToSvg('x', box.fX1);
      this.x2 = this.axisToSvg('x', box.fX2);
      this.y1 = this.axisToSvg('y', box.fY1);
      this.y2 = this.axisToSvg('y', box.fY2);

      if (this.swap_xy)
         [this.x1, this.x2, this.y1, this.y2] = [this.y1, this.y2, this.x1, this.x2];

      this.borderMode = (box.fBorderMode && this.fillatt.hasColor()) ? box.fBorderMode : 0;
      this.borderSize = box.fBorderSize || 2;

      const paths = this.getPathes();

      this.draw_g
          .append('svg:path')
          .attr('d', paths[0])
          .call(this.lineatt.func)
          .call(this.fillatt.func);

      if (this.borderMode) {
         this.draw_g.append('svg:path')
                    .attr('d', paths[1])
                    .call(this.fillatt.func)
                    .style('fill', d3_rgb(this.fillatt.color).brighter(0.5).formatRgb());

         this.draw_g.append('svg:path')
                    .attr('d', paths[2])
                    .call(this.fillatt.func)
                    .style('fill', d3_rgb(this.fillatt.color).darker(0.5).formatRgb());
      }

      assignContextMenu(this);

      addMoveHandler(this);

      return this;
   }

   /** @summary Draw TLine object */
   static async draw(dom, obj, opt) {
      const painter = new TBoxPainter(dom, obj, opt);
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TBoxPainter


export { TBoxPainter };
