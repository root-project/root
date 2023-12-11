import { BIT, isFunc, clTLatex, clTMathText, clTAnnotation, clTPolyLine } from '../core.mjs';
import { rgb as d3_rgb, select as d3_select } from '../d3.mjs';
import { BasePainter, makeTranslate } from '../base/BasePainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu, kToFront } from '../gui/menu.mjs';


/** @summary Draw TText
  * @private */
async function drawText() {
   const text = this.getObject(),
         pp = this.getPadPainter(),
         w = pp.getPadWidth(),
         h = pp.getPadHeight(),
         fp = this.getFramePainter();
   let pos_x = text.fX, pos_y = text.fY,
       fact = 1,
       annot = this.matchObjectType(clTAnnotation);

   this.createAttText({ attr: text });

   if (annot && fp?.mode3d && isFunc(fp?.convert3DtoPadNDC)) {
      const pos = fp.convert3DtoPadNDC(text.fX, text.fY, text.fZ);
      pos_x = pos.x;
      pos_y = pos.y;
      this.isndc = true;
      annot = '3d';
   } else if (text.TestBit(BIT(14))) {
      // NDC coordinates
      this.isndc = true;
   } else if (pp.getRootPad(true)) {
      // force pad coordiantes
   } else {
      // place in the middle
      this.isndc = true;
      pos_x = pos_y = 0.5;
      text.fTextAlign = 22;
   }

   this.createG();

   this.draw_g.attr('transform', null); // remove transofrm from interactive changes

   this.pos_x = this.axisToSvg('x', pos_x, this.isndc);
   this.pos_y = this.axisToSvg('y', pos_y, this.isndc);

   const arg = this.textatt.createArg({ x: this.pos_x, y: this.pos_y, text: text.fTitle, latex: 0 });

   if ((text._typename === clTLatex) || annot) {
      arg.latex = 1;
      fact = 0.9;
   } else if (text._typename === clTMathText) {
      arg.latex = 2;
      fact = 0.8;
   }

   this.startTextDrawing(this.textatt.font, this.textatt.getSize(w, h, fact, 0.05));

   this.drawText(arg);

   return this.finishTextDrawing().then(() => {
      if (this.isBatchMode()) return this;

      this.pos_dx = this.pos_dy = 0;

      if (!this.moveDrag) {
         this.moveDrag = function(dx, dy) {
            this.pos_dx += dx;
            this.pos_dy += dy;
            makeTranslate(this.draw_g, this.pos_dx, this.pos_dy);
         };
      }

      if (!this.moveEnd) {
         this.moveEnd = function(not_changed) {
            if (not_changed) return;
            const text = this.getObject();
            text.fX = this.svgToAxis('x', this.pos_x + this.pos_dx, this.isndc);
            text.fY = this.svgToAxis('y', this.pos_y + this.pos_dy, this.isndc);
            this.submitCanvExec(`SetX(${text.fX});;SetY(${text.fY});;`);
         };
      }

      if (annot !== '3d')
         addMoveHandler(this);
      else {
         fp.processRender3D = true;
         this.handleRender3D = () => {
            const pos = fp.convert3DtoPadNDC(text.fX, text.fY, text.fZ),
                  new_x = this.axisToSvg('x', pos.x, true),
                  new_y = this.axisToSvg('y', pos.y, true);
            makeTranslate(this.draw_g, new_x - this.pos_x, new_y - this.pos_y);
         };
      }

      assignContextMenu(this, kToFront);

      return this;
   });
}


/** @summary Draw TPolyLine
  * @private */
function drawPolyLine() {
   this.createG();

   const polyline = this.getObject(),
         kPolyLineNDC = BIT(14),
         isndc = polyline.TestBit(kPolyLineNDC),
         opt = this.getDrawOpt() || polyline.fOption,
         dofill = (polyline._typename === clTPolyLine) && ((opt === 'f') || (opt === 'F')),
         func = this.getAxisToSvgFunc(isndc);

   this.createAttLine({ attr: polyline });
   this.createAttFill({ attr: polyline });

   let cmd = '';
   for (let n = 0; n <= polyline.fLastPoint; ++n)
      cmd += `${n>0?'L':'M'}${func.x(polyline.fX[n])},${func.y(polyline.fY[n])}`;

   if (dofill)
      cmd += 'Z';

   const elem = this.draw_g.append('svg:path').attr('d', cmd);

   if (dofill)
      elem.call(this.fillatt.func);
   else
      elem.call(this.lineatt.func).style('fill', 'none');

   assignContextMenu(this, kToFront);

   addMoveHandler(this);

   this.dx = this.dy = 0;
   this.isndc = isndc;

   this.moveDrag = function(dx, dy) {
      this.dx += dx;
      this.dy += dy;
      makeTranslate(this.draw_g.select('path'), this.dx, this.dy);
   };

   this.moveEnd = function(not_changed) {
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
   };
}

/** @summary Draw TEllipse
  * @private */
function drawEllipse() {
   const ellipse = this.getObject(),
         closed_ellipse = (ellipse.fPhimin === 0) && (ellipse.fPhimax === 360),
         is_crown = (ellipse._typename === 'TCrown');

   this.createAttLine({ attr: ellipse });
   this.createAttFill({ attr: ellipse });

   this.createG();

   const funcs = this.getAxisToSvgFunc(),
         x = funcs.x(ellipse.fX1),
         y = funcs.y(ellipse.fY1),
         rx = is_crown && (ellipse.fR1 <= 0) ? (funcs.x(ellipse.fX1 + ellipse.fR2) - x) : (funcs.x(ellipse.fX1 + ellipse.fR1) - x),
         ry = y - funcs.y(ellipse.fY1 + ellipse.fR2);

   let path = '';

   if (is_crown && (ellipse.fR1 > 0)) {
      const rx1 = rx, ry2 = ry,
            ry1 = y - funcs.y(ellipse.fY1 + ellipse.fR1),
            rx2 = funcs.x(ellipse.fX1 + ellipse.fR2) - x;

      if (closed_ellipse) {
         path = `M${-rx1},0A${rx1},${ry1},0,1,0,${rx1},0A${rx1},${ry1},0,1,0,${-rx1},0` +
                `M${-rx2},0A${rx2},${ry2},0,1,0,${rx2},0A${rx2},${ry2},0,1,0,${-rx2},0`;
      } else {
         const large_arc = (ellipse.fPhimax-ellipse.fPhimin>=180) ? 1 : 0,
               a1 = ellipse.fPhimin*Math.PI/180, a2 = ellipse.fPhimax*Math.PI/180,
               dx1 = Math.round(rx1*Math.cos(a1)), dy1 = Math.round(ry1*Math.sin(a1)),
               dx2 = Math.round(rx1*Math.cos(a2)), dy2 = Math.round(ry1*Math.sin(a2)),
               dx3 = Math.round(rx2*Math.cos(a1)), dy3 = Math.round(ry2*Math.sin(a1)),
               dx4 = Math.round(rx2*Math.cos(a2)), dy4 = Math.round(ry2*Math.sin(a2));

         path = `M${dx2},${dy2}A${rx1},${ry1},0,${large_arc},0,${dx1},${dy1}` +
                `L${dx3},${dy3}A${rx2},${ry2},0,${large_arc},1,${dx4},${dy4}Z`;
      }
   } else if (ellipse.fTheta === 0) {
      if (closed_ellipse)
         path = `M${-rx},0A${rx},${ry},0,1,0,${rx},0A${rx},${ry},0,1,0,${-rx},0Z`;
      else {
         const x1 = Math.round(rx * Math.cos(ellipse.fPhimin*Math.PI/180)),
               y1 = Math.round(ry * Math.sin(ellipse.fPhimin*Math.PI/180)),
               x2 = Math.round(rx * Math.cos(ellipse.fPhimax*Math.PI/180)),
               y2 = Math.round(ry * Math.sin(ellipse.fPhimax*Math.PI/180));
         path = `M0,0L${x1},${y1}A${rx},${ry},0,1,1,${x2},${y2}Z`;
      }
   } else {
     const ct = Math.cos(ellipse.fTheta*Math.PI/180),
           st = Math.sin(ellipse.fTheta*Math.PI/180),
           phi1 = ellipse.fPhimin*Math.PI/180,
           phi2 = ellipse.fPhimax*Math.PI/180,
           np = 200,
           dphi = (phi2-phi1) / (np - (closed_ellipse ? 0 : 1));
     let lastx = 0, lasty = 0;
     if (!closed_ellipse) path = 'M0,0';
     for (let n = 0; n < np; ++n) {
         const angle = phi1 + n*dphi,
               dx = ellipse.fR1 * Math.cos(angle),
               dy = ellipse.fR2 * Math.sin(angle),
               px = funcs.x(ellipse.fX1 + dx*ct - dy*st) - x,
               py = funcs.y(ellipse.fY1 + dx*st + dy*ct) - y;
         if (!path)
            path = `M${px},${py}`;
         else if (lastx === px)
            path += `v${py-lasty}`;
         else if (lasty === py)
            path += `h${px-lastx}`;
         else
            path += `l${px-lastx},${py-lasty}`;
         lastx = px; lasty = py;
     }
     path += 'Z';
   }

   this.x = x;
   this.y = y;

   makeTranslate(this.draw_g.append('svg:path'), x, y)
      .attr('d', path)
      .call(this.lineatt.func)
      .call(this.fillatt.func);

   assignContextMenu(this, kToFront);

   addMoveHandler(this);

   this.moveDrag = function(dx, dy) {
      this.x += dx;
      this.y += dy;
      makeTranslate(this.draw_g.select('path'), this.x, this.y);
   };

   this.moveEnd = function(not_changed) {
      if (not_changed) return;
      const ellipse = this.getObject();
      ellipse.fX1 = this.svgToAxis('x', this.x);
      ellipse.fY1 = this.svgToAxis('y', this.y);
      this.submitCanvExec(`SetX1(${ellipse.fX1});;SetY1(${ellipse.fY1});;Notify();;`);
   };
}

/** @summary Draw TPie
  * @private */
function drawPie() {
   this.createG();

   const pie = this.getObject(),
         nb = pie.fPieSlices.length,
         xc = this.axisToSvg('x', pie.fX),
         yc = this.axisToSvg('y', pie.fY),
         rx = this.axisToSvg('x', pie.fX + pie.fRadius) - xc,
         ry = this.axisToSvg('y', pie.fY + pie.fRadius) - yc;

   makeTranslate(this.draw_g, xc, yc);

   // Draw the slices
   let total = 0,
       af = (pie.fAngularOffset*Math.PI)/180,
       x1 = Math.round(rx*Math.cos(af)),
       y1 = Math.round(ry*Math.sin(af));

   for (let n = 0; n < nb; n++)
      total += pie.fPieSlices[n].fValue;

   for (let n = 0; n < nb; n++) {
      const slice = pie.fPieSlices[n];

      this.createAttLine({ attr: slice });
      this.createAttFill({ attr: slice });

      af += slice.fValue/total*2*Math.PI;
      const x2 = Math.round(rx*Math.cos(af)),
            y2 = Math.round(ry*Math.sin(af));

      this.draw_g
          .append('svg:path')
          .attr('d', `M0,0L${x1},${y1}A${rx},${ry},0,0,0,${x2},${y2}z`)
          .call(this.lineatt.func)
          .call(this.fillatt.func);
      x1 = x2; y1 = y2;
   }
}

/** @summary Draw TBox
  * @private */
function drawBox() {
   const box = this.getObject(),
         opt = this.getDrawOpt(),
         draw_line = (opt.toUpperCase().indexOf('L') >= 0);

   this.createAttLine({ attr: box });
   this.createAttFill({ attr: box });

   // if box filled, contour line drawn only with 'L' draw option:
   if (!this.fillatt.empty() && !draw_line)
      this.lineatt.color = 'none';

   this.createG();

   this.x1 = this.axisToSvg('x', box.fX1);
   this.x2 = this.axisToSvg('x', box.fX2);
   this.y1 = this.axisToSvg('y', box.fY1);
   this.y2 = this.axisToSvg('y', box.fY2);
   this.borderMode = (box.fBorderMode && box.fBorderSize && this.fillatt.hasColor()) ? box.fBorderMode : 0;
   this.borderSize = box.fBorderSize;

   this.getPathes = () => {
      const xx = Math.min(this.x1, this.x2), yy = Math.min(this.y1, this.y2),
            ww = Math.abs(this.x2 - this.x1), hh = Math.abs(this.y1 - this.y2),
            path = `M${xx},${yy}h${ww}v${hh}h${-ww}z`;
      if (!this.borderMode)
         return [path];
      const pww = this.borderSize, phh = this.borderSize,
            side1 = `M${xx},${yy}h${ww}l${-pww},${phh}h${2*pww-ww}v${hh-2*phh}l${-pww},${phh}z`,
            side2 = `M${xx+ww},${yy+hh}v${-hh}l${-pww},${phh}v${hh-2*phh}h${2*pww-ww}l${-pww},${phh}z`;

      return (this.borderMode > 0) ? [path, side1, side2] : [path, side2, side1];
   };

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
                 .style('fill', d3_rgb(this.fillatt.color).brighter(0.5).formatHex());

      this.draw_g.append('svg:path')
                 .attr('d', paths[2])
                 .call(this.fillatt.func)
                 .style('fill', d3_rgb(this.fillatt.color).darker(0.5).formatHex());
   }

   assignContextMenu(this, kToFront);

   addMoveHandler(this);

   this.moveStart = function(x, y) {
      const ww = Math.abs(this.x2 - this.x1), hh = Math.abs(this.y1 - this.y2);

      this.c_x1 = Math.abs(x - this.x2) > ww*0.1;
      this.c_x2 = Math.abs(x - this.x1) > ww*0.1;
      this.c_y1 = Math.abs(y - this.y2) > hh*0.1;
      this.c_y2 = Math.abs(y - this.y1) > hh*0.1;
      if (this.c_x1 !== this.c_x2 && this.c_y1 && this.c_y2)
         this.c_y1 = this.c_y2 = false;
      if (this.c_y1 !== this.c_y2 && this.c_x1 && this.c_x2)
         this.c_x1 = this.c_x2 = false;
   };

   this.moveDrag = function(dx, dy) {
      if (this.c_x1) this.x1 += dx;
      if (this.c_x2) this.x2 += dx;
      if (this.c_y1) this.y1 += dy;
      if (this.c_y2) this.y2 += dy;

      const nodes = this.draw_g.selectAll('path').nodes(),
            pathes = this.getPathes();

      pathes.forEach((path, i) => d3_select(nodes[i]).attr('d', path));
   };

   this.moveEnd = function(not_changed) {
      if (not_changed) return;
      const box = this.getObject();
      let exec = '';
      if (this.c_x1) { box.fX1 = this.svgToAxis('x', this.x1); exec += `SetX1(${box.fX1});;`; }
      if (this.c_x2) { box.fX2 = this.svgToAxis('x', this.x2); exec += `SetX2(${box.fX2});;`; }
      if (this.c_y1) { box.fY1 = this.svgToAxis('y', this.y1); exec += `SetY1(${box.fY1});;`; }
      if (this.c_y2) { box.fY2 = this.svgToAxis('y', this.y2); exec += `SetY2(${box.fY2});;`; }
      this.submitCanvExec(exec + 'Notify();;');
   };
}

/** @summary Draw TMarker
  * @private */
function drawMarker() {
   const marker = this.getObject(),
         kMarkerNDC = BIT(14);

   this.isndc = marker.TestBit(kMarkerNDC);

   this.createAttMarker({ attr: marker });

   this.createG();

   const x = this.axisToSvg('x', marker.fX, this.isndc),
         y = this.axisToSvg('y', marker.fY, this.isndc),
         path = this.markeratt.create(x, y);

   if (path) {
      this.draw_g.append('svg:path')
          .attr('d', path)
          .call(this.markeratt.func);
   }

   assignContextMenu(this, kToFront);

   addMoveHandler(this);

   this.dx = this.dy = 0;

   this.moveDrag = function(dx, dy) {
      this.dx += dx;
      this.dy += dy;
      makeTranslate(this.draw_g.select('path'), this.dx, this.dy);
   };

   this.moveEnd = function(not_changed) {
      if (not_changed) return;
      const marker = this.getObject();
      marker.fX = this.svgToAxis('x', this.axisToSvg('x', marker.fX, this.isndc) + this.dx, this.isndc);
      marker.fY = this.svgToAxis('y', this.axisToSvg('y', marker.fY, this.isndc) + this.dy, this.isndc);
      this.submitCanvExec(`SetX(${marker.fX});;SetY(${marker.fY});;Notify();;`);
      this.redraw();
   };
}

/** @summary Draw TPolyMarker
  * @private */
function drawPolyMarker() {
   const poly = this.getObject(),
         func = this.getAxisToSvgFunc();

   this.createAttMarker({ attr: poly });

   this.createG();

   let path = '';
   for (let n = 0; n <= poly.fLastPoint; ++n)
      path += this.markeratt.create(func.x(poly.fX[n]), func.y(poly.fY[n]));

   if (path) {
      this.draw_g.append('svg:path')
          .attr('d', path)
          .call(this.markeratt.func);
   }

   assignContextMenu(this, kToFront);

   addMoveHandler(this);

   this.dx = this.dy = 0;

   this.moveDrag = function(dx, dy) {
      this.dx += dx;
      this.dy += dy;
      makeTranslate(this.draw_g.select('path'), this.dx, this.dy);
   };

   this.moveEnd = function(not_changed) {
      if (not_changed) return;
      const poly = this.getObject(),
            func = this.getAxisToSvgFunc();

      let exec = '';
      for (let n = 0; n <= poly.fLastPoint; ++n) {
         const x = this.svgToAxis('x', func.x(poly.fX[n]) + this.dx),
               y = this.svgToAxis('y', func.y(poly.fY[n]) + this.dy);
         poly.fX[n] = x;
         poly.fY[n] = y;
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

export { drawText, drawPolyLine, drawEllipse, drawPie, drawBox,
         drawMarker, drawPolyMarker, drawJSImage };
