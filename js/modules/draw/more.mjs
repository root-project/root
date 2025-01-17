import { BIT, isFunc, clTLatex, clTLink, clTMathText, clTAnnotation } from '../core.mjs';
import { BasePainter, makeTranslate, DrawOptions } from '../base/BasePainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';
import { assignContextMenu } from '../gui/menu.mjs';


/** @summary Draw TText
  * @private */
async function drawText() {
   const text = this.getObject(),
         pp = this.getPadPainter(),
         fp = this.getFramePainter(),
         is_url = text.fName.startsWith('http://') || text.fName.startsWith('https://');
   let pos_x = text.fX, pos_y = text.fY, use_frame = false,
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
      // force pad coordinates
      const d = new DrawOptions(this.getDrawOpt());
      use_frame = d.check('FRAME');
   } else {
      // place in the middle
      this.isndc = true;
      pos_x = pos_y = 0.5;
      text.fTextAlign = 22;
   }

   this.createG(use_frame ? 'frame2d' : undefined, is_url);

   this.draw_g.attr('transform', null); // remove transform from interactive changes

   this.pos_x = this.axisToSvg('x', pos_x, this.isndc);
   this.pos_y = this.axisToSvg('y', pos_y, this.isndc);
   this.swap_xy = use_frame && fp?.swap_xy;

   if (this.swap_xy)
      [this.pos_x, this.pos_y] = [this.pos_y, this.pos_x];

   const arg = this.textatt.createArg({ x: this.pos_x, y: this.pos_y, text: text.fTitle, latex: 0 });

   if ((text._typename === clTLatex) || annot)
      arg.latex = 1;
   else if (text._typename === clTMathText) {
      arg.latex = 2;
      fact = 0.8;
   }

   if (is_url) {
      this.draw_g.attr('href', text.fName);
      if (!this.isBatchMode())
         this.draw_g.append('svg:title').text(`link on ${text.fName}`);
   }

   return this.startTextDrawingAsync(this.textatt.font, this.textatt.getSize(pp, fact /* , 0.05 */))
              .then(() => this.drawText(arg))
              .then(() => this.finishTextDrawing())
              .then(() => {
      if (this.isBatchMode())
         return this;

      if (pp.isButton() && !pp.isEditable()) {
         this.draw_g.on('click', () => this.getCanvPainter().selectActivePad(pp));
         return this;
      }

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
            let fx = this.svgToAxis('x', this.pos_x + this.pos_dx, this.isndc),
                fy = this.svgToAxis('y', this.pos_y + this.pos_dy, this.isndc);
            if (this.swap_xy)
               [fx, fy] = [fy, fx];

            text.fX = fx;
            text.fY = fy;
            this.submitCanvExec(`SetX(${fx});;SetY(${fy});;`);
         };
      }

      if (annot !== '3d')
         addMoveHandler(this, true, is_url);
      else {
         fp.processRender3D = true;
         this.handleRender3D = () => {
            const pos = fp.convert3DtoPadNDC(text.fX, text.fY, text.fZ),
                  new_x = this.axisToSvg('x', pos.x, true),
                  new_y = this.axisToSvg('y', pos.y, true);
            makeTranslate(this.draw_g, new_x - this.pos_x, new_y - this.pos_y);
         };
      }

      assignContextMenu(this);

      this.fillContextMenuItems = function(menu) {
         menu.add('Change text', () => menu.input('Enter new text', text.fTitle).then(t => {
            text.fTitle = t;
            this.interactiveRedraw('pad', `exec:SetTitle("${t}")`);
         }));
      };

      if (this.matchObjectType(clTLink)) {
         this.draw_g.style('cursor', 'pointer')
                    .on('click', () => this.submitCanvExec('ExecuteEvent(kButton1Up, 0, 0);;'));
      }

      return this;
   });
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

   assignContextMenu(this);

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

/** @summary Draw TMarker
  * @private */
function drawMarker() {
   const marker = this.getObject(),
         kMarkerNDC = BIT(14);

   this.isndc = marker.TestBit(kMarkerNDC);

   const use_frame = this.isndc ? false : new DrawOptions(this.getDrawOpt()).check('FRAME'),
         swap_xy = use_frame && this.getFramePainter()?.swap_xy;

   this.createAttMarker({ attr: marker });

   this.createG(use_frame ? 'frame2d' : undefined);

   let x = this.axisToSvg('x', marker.fX, this.isndc),
       y = this.axisToSvg('y', marker.fY, this.isndc);
   if (swap_xy)
      [x, y] = [y, x];

   const path = this.markeratt.create(x, y);

   if (path) {
      this.draw_g.append('svg:path')
          .attr('d', path)
          .call(this.markeratt.func);
   }

   assignContextMenu(this);

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
      let fx = this.svgToAxis('x', this.axisToSvg('x', marker.fX, this.isndc) + this.dx, this.isndc),
          fy = this.svgToAxis('y', this.axisToSvg('y', marker.fY, this.isndc) + this.dy, this.isndc);
      if (swap_xy)
         [fx, fy] = [fy, fx];
      marker.fX = fx;
      marker.fY = fy;
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

   this.createG();

   let path = '';
   for (let n = 0; n <= poly.fLastPoint; ++n)
      path += this.markeratt.create(func.x(poly.fX[n]), func.y(poly.fY[n]));

   if (path) {
      this.draw_g.append('svg:path')
          .attr('d', path)
          .call(this.markeratt.func);
   }

   assignContextMenu(this);

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

export { drawText, drawEllipse, drawPie, drawMarker, drawPolyMarker, drawJSImage };
