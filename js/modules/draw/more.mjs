import { BIT } from '../core.mjs';
import { BasePainter, makeTranslate, DrawOptions } from '../base/BasePainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu } from '../gui/menu.mjs';


/** @summary Draw TEllipse
  * @private */
function drawEllipse() {
   const ellipse = this.getObject(),
         closed_ellipse = (ellipse.fPhimin === 0) && (ellipse.fPhimax === 360),
         is_crown = (ellipse._typename === 'TCrown');

   this.createAttLine({ attr: ellipse });
   this.createAttFill({ attr: ellipse });

   const g = this.createG(),
         funcs = this.getAxisToSvgFunc(),
         x = funcs.x(ellipse.fX1),
         y = funcs.y(ellipse.fY1),
         rx = is_crown && (ellipse.fR1 <= 0) ? (funcs.x(ellipse.fX1 + ellipse.fR2) - x) : (funcs.x(ellipse.fX1 + ellipse.fR1) - x),
         ry = y - funcs.y(ellipse.fY1 + ellipse.fR2),
         dr = Math.PI / 180;

   let path = '';

   if (is_crown && (ellipse.fR1 > 0)) {
      const ratio = ellipse.fYXRatio ?? 1,
            rx1 = rx, ry2 = ratio * ry,
            ry1 = ratio * (y - funcs.y(ellipse.fY1 + ellipse.fR1)),
            rx2 = funcs.x(ellipse.fX1 + ellipse.fR2) - x;

      if (closed_ellipse) {
         path = `M${-rx1},0A${rx1},${ry1},0,1,0,${rx1},0A${rx1},${ry1},0,1,0,${-rx1},0` +
                `M${-rx2},0A${rx2},${ry2},0,1,0,${rx2},0A${rx2},${ry2},0,1,0,${-rx2},0`;
      } else {
         const large_arc = (ellipse.fPhimax - ellipse.fPhimin >= 180) ? 1 : 0,
               a1 = ellipse.fPhimin * dr, a2 = ellipse.fPhimax * dr,
               dx1 = Math.round(rx1 * Math.cos(a1)), dy1 = Math.round(ry1 * Math.sin(a1)),
               dx2 = Math.round(rx1 * Math.cos(a2)), dy2 = Math.round(ry1 * Math.sin(a2)),
               dx3 = Math.round(rx2 * Math.cos(a1)), dy3 = Math.round(ry2 * Math.sin(a1)),
               dx4 = Math.round(rx2 * Math.cos(a2)), dy4 = Math.round(ry2 * Math.sin(a2));

         path = `M${dx2},${dy2}A${rx1},${ry1},0,${large_arc},0,${dx1},${dy1}` +
                `L${dx3},${dy3}A${rx2},${ry2},0,${large_arc},1,${dx4},${dy4}Z`;
      }
   } else if (ellipse.fTheta === 0) {
      if (closed_ellipse)
         path = `M${-rx},0A${rx},${ry},0,1,0,${rx},0A${rx},${ry},0,1,0,${-rx},0Z`;
      else {
         const x1 = Math.round(rx * Math.cos(ellipse.fPhimin * dr)),
               y1 = Math.round(ry * Math.sin(ellipse.fPhimin * dr)),
               x2 = Math.round(rx * Math.cos(ellipse.fPhimax * dr)),
               y2 = Math.round(ry * Math.sin(ellipse.fPhimax * dr));
         path = `M0,0L${x1},${y1}A${rx},${ry},0,1,1,${x2},${y2}Z`;
      }
   } else {
      const ct = Math.cos(ellipse.fTheta * dr),
            st = Math.sin(ellipse.fTheta * dr),
            phi1 = ellipse.fPhimin * dr,
            phi2 = ellipse.fPhimax * dr,
            np = 200,
            dphi = (phi2 - phi1) / (np - (closed_ellipse ? 0 : 1));
      let lastx = 0, lasty = 0;
      if (!closed_ellipse)
         path = 'M0,0';
      for (let n = 0; n < np; ++n) {
         const angle = phi1 + n * dphi,
               dx = ellipse.fR1 * Math.cos(angle),
               dy = ellipse.fR2 * Math.sin(angle),
               px = funcs.x(ellipse.fX1 + dx * ct - dy * st) - x,
               py = funcs.y(ellipse.fY1 + dx * st + dy * ct) - y;
         if (!path)
            path = `M${px},${py}`;
         else if (lastx === px)
            path += `v${py - lasty}`;
         else if (lasty === py)
            path += `h${px - lastx}`;
         else
            path += `l${px - lastx},${py - lasty}`;
         lastx = px;
         lasty = py;
      }
      path += 'Z';
   }

   this.x = x;
   this.y = y;

   makeTranslate(g.append('svg:path'), x, y)
      .attr('d', path)
      .call(this.lineatt.func)
      .call(this.fillatt.func);

   assignContextMenu(this);

   addMoveHandler(this);

   this.moveDrag = function(dx, dy) {
      this.x += dx;
      this.y += dy;
      makeTranslate(this.getG().select('path'), this.x, this.y);
   };

   this.moveEnd = function(not_changed) {
      if (not_changed)
         return;
      const ell = this.getObject();
      ell.fX1 = this.svgToAxis('x', this.x);
      ell.fY1 = this.svgToAxis('y', this.y);
      this.submitCanvExec(`SetX1(${ell.fX1});;SetY1(${ell.fY1});;Notify();;`);
   };
}

/** @summary Draw TMarker
  * @private */
function drawMarker() {
   const marker = this.getObject(),
         kMarkerNDC = BIT(14);

   this.isndc = marker.TestBit(kMarkerNDC);

   const d = new DrawOptions(this.getDrawOpt()),
         use_frame = this.isndc ? false : d.check('FRAME'),
         swap_xy = use_frame && this.getFramePainter()?.swap_xy();

   this.createAttMarker({ attr: marker });

   const g = this.createG(use_frame ? 'frame2d' : undefined);

   let x = this.axisToSvg('x', marker.fX, this.isndc),
       y = this.axisToSvg('y', marker.fY, this.isndc);
   if (swap_xy)
      [x, y] = [y, x];

   const path = this.markeratt.create(x, y);

   if (path) {
      g.append('svg:path')
       .attr('d', path)
       .call(this.markeratt.func);
   }

   if (d.check('NO_INTERACTIVE'))
      return;

   assignContextMenu(this);

   addMoveHandler(this);

   this.dx = this.dy = 0;

   this.moveDrag = function(dx, dy) {
      this.dx += dx;
      this.dy += dy;
      if (this.getG())
         makeTranslate(this.getG().select('path'), this.dx, this.dy);
   };

   this.moveEnd = function(not_changed) {
      if (not_changed || !this.getG())
         return;
      const mrk = this.getObject();
      let fx = this.svgToAxis('x', this.axisToSvg('x', mrk.fX, this.isndc) + this.dx, this.isndc),
          fy = this.svgToAxis('y', this.axisToSvg('y', mrk.fY, this.isndc) + this.dy, this.isndc);
      if (swap_xy)
         [fx, fy] = [fy, fx];
      mrk.fX = fx;
      mrk.fY = fy;
      this.submitCanvExec(`SetX(${fx});;SetY(${fy});;Notify();;`);
      this.redraw();
   };
}

/** @summary Draw TPolyMarker
  * @private */
function drawPolyMarker() {
   const poly = this.getObject(),
         func = this.getAxisToSvgFunc();

   this.createAttMarker({ attr: poly });

   const g = this.createG();

   let path = '';
   for (let n = 0; n <= poly.fLastPoint; ++n)
      path += this.markeratt.create(func.x(poly.fX[n]), func.y(poly.fY[n]));

   if (path) {
      g.append('svg:path')
       .attr('d', path)
       .call(this.markeratt.func);
   }

   assignContextMenu(this);

   addMoveHandler(this);

   this.dx = this.dy = 0;

   this.moveDrag = function(dx, dy) {
      this.dx += dx;
      this.dy += dy;
      if (this.getG())
         makeTranslate(this.getG().select('path'), this.dx, this.dy);
   };

   this.moveEnd = function(not_changed) {
      if (not_changed || !this.getG())
         return;
      const poly2 = this.getObject(),
            func2 = this.getAxisToSvgFunc();
      let exec = '';
      for (let n = 0; n <= poly2.fLastPoint; ++n) {
         const x = this.svgToAxis('x', func2.x(poly2.fX[n]) + this.dx),
               y = this.svgToAxis('y', func2.y(poly2.fY[n]) + this.dy);
         poly2.fX[n] = x;
         poly2.fY[n] = y;
         exec += `SetPoint(${n},${x},${y});;`;
      }
      this.submitCanvExec(exec + 'Notify();;');
      this.redraw();
   };
}

/** @summary Draw JS image
  * @private */
function drawJSImage(dom, obj, opt) {
   const painter = new BasePainter(dom),
         main = painter.selectDom(),
         img = main.append('img').attr('src', obj.fName).attr('title', obj.fTitle || obj.fName);

   if (opt && opt.indexOf('scale') >= 0)
      img.style('width', '100%').style('height', '100%');
   else if (opt && opt.indexOf('center') >= 0) {
      main.style('position', 'relative');
      img.attr('style', 'margin: 0; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);');
   }

   painter.setTopPainter();

   return painter;
}

export { drawEllipse, drawMarker, drawPolyMarker, drawJSImage };
