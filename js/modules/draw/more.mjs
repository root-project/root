import { BIT, isBatchMode } from '../core.mjs';
import { rgb as d3_rgb } from '../d3.mjs';
import { BasePainter } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TAttMarkerHandler } from '../base/TAttMarkerHandler.mjs';
import { TAttLineHandler } from '../base/TAttLineHandler.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { addMoveHandler } from '../gui/utils.mjs';


/** @summary Draw TText
  * @private */
function drawText() {
   let text = this.getObject(),
       pp = this.getPadPainter(),
       w = pp.getPadWidth(),
       h = pp.getPadHeight(),
       pos_x = text.fX, pos_y = text.fY,
       tcolor = this.getColor(text.fTextColor),
       use_frame = false,
       fact = 1., textsize = text.fTextSize || 0.05,
       main = this.getFramePainter();

   if (text.TestBit(BIT(14))) {
      // NDC coordinates
      this.isndc = true;
   } else if (main && !main.mode3d) {
      // frame coordiantes
      w = main.getFrameWidth();
      h = main.getFrameHeight();
      use_frame = "upper_layer";
   } else if (pp.getRootPad(true)) {
      // force pad coordiantes
   } else {
      // place in the middle
      this.isndc = true;
      pos_x = pos_y = 0.5;
      text.fTextAlign = 22;
      if (!tcolor) tcolor = 'black';
   }

   this.createG(use_frame);

   this.draw_g.attr("transform", null); // remove transofrm from interactive changes

   this.pos_x = this.axisToSvg("x", pos_x, this.isndc);
   this.pos_y = this.axisToSvg("y", pos_y, this.isndc);

   let arg = { align: text.fTextAlign, x: this.pos_x, y: this.pos_y, text: text.fTitle, color: tcolor, latex: 0 };

   if (text.fTextAngle) arg.rotate = -text.fTextAngle;

   if (text._typename == 'TLatex') { arg.latex = 1; fact = 0.9; } else
   if (text._typename == 'TMathText') { arg.latex = 2; fact = 0.8; }

   this.startTextDrawing(text.fTextFont, Math.round((textsize>1) ? textsize : textsize*Math.min(w,h)*fact));

   this.drawText(arg);

   return this.finishTextDrawing().then(() => {
      if (isBatchMode()) return this;

      this.pos_dx = this.pos_dy = 0;

      if (!this.moveDrag)
         this.moveDrag = function(dx,dy) {
            this.pos_dx += dx;
            this.pos_dy += dy;
            this.draw_g.attr("transform", `translate(${this.pos_dx},${this.pos_dy})`);
        }

      if (!this.moveEnd)
         this.moveEnd = function(not_changed) {
            if (not_changed) return;
            let text = this.getObject();
            text.fX = this.svgToAxis("x", this.pos_x + this.pos_dx, this.isndc),
            text.fY = this.svgToAxis("y", this.pos_y + this.pos_dy, this.isndc);
            this.submitCanvExec(`SetX(${text.fX});;SetY(${text.fY});;`);
         }

      addMoveHandler(this);

      return this;
   });
}

/** @summary Draw TLine
  * @private */
function drawTLine(dom, obj) {

   let painter = new ObjectPainter(dom, obj);

   painter.redraw = function() {
      const kLineNDC = BIT(14),
            line = this.getObject(),
            lineatt = new TAttLineHandler(line),
            isndc = line.TestBit(kLineNDC);

      // create svg:g container for line drawing
      this.createG();

      this.draw_g
          .append("svg:path")
          .attr("d", `M${this.axisToSvg("x", line.fX1, isndc)},${this.axisToSvg("y", line.fY1, isndc)}L${this.axisToSvg("x", line.fX2, isndc)},${this.axisToSvg("y", line.fY2, isndc)}`)
          .call(lineatt.func);

      return this;
   }

   return ensureTCanvas(painter, false).then(() => painter.redraw());
}

/** @summary Draw TPolyLine
  * @private */
function drawPolyLine() {

   // create svg:g container for polyline drawing
   this.createG();

   let polyline = this.getObject(),
       lineatt = new TAttLineHandler(polyline),
       fillatt = this.createAttFill(polyline),
       kPolyLineNDC = BIT(14),
       isndc = polyline.TestBit(kPolyLineNDC),
       cmd = "", func = this.getAxisToSvgFunc(isndc);

   for (let n = 0; n <= polyline.fLastPoint; ++n)
      cmd += ((n > 0) ? "L" : "M") + func.x(polyline.fX[n]) + "," + func.y(polyline.fY[n]);

   if (polyline._typename != "TPolyLine") fillatt.setSolidColor("none");

   if (!fillatt.empty()) cmd+="Z";

   this.draw_g
       .append("svg:path")
       .attr("d", cmd)
       .call(lineatt.func)
       .call(fillatt.func);
}

/** @summary Draw TEllipse
  * @private */
function drawEllipse() {

   let ellipse = this.getObject();

   this.createAttLine({ attr: ellipse });
   this.createAttFill({ attr: ellipse });

   // create svg:g container for ellipse drawing
   this.createG();

   let funcs = this.getAxisToSvgFunc(),
       x = funcs.x(ellipse.fX1),
       y = funcs.y(ellipse.fY1),
       rx = funcs.x(ellipse.fX1 + ellipse.fR1) - x,
       ry = y - funcs.y(ellipse.fY1 + ellipse.fR2),
       path = "", closed_ellipse = (ellipse.fPhimin == 0) && (ellipse.fPhimax == 360);

   // handle same as ellipse with equal radius
   if ((ellipse._typename == "TCrown") && (ellipse.fR1 <= 0))
      rx = funcs.x(ellipse.fX1 + ellipse.fR2) - x;

   if ((ellipse._typename == "TCrown") && (ellipse.fR1 > 0)) {
      let rx1 = rx, ry2 = ry,
          ry1 = y - funcs.y(ellipse.fY1 + ellipse.fR1),
          rx2 = funcs.x(ellipse.fX1 + ellipse.fR2) - x;

      if (closed_ellipse) {
         path = `M${-rx1},0A${rx1},${ry1},0,1,0,${rx1},0A${rx1},${ry1},0,1,0,${-rx1},0` +
                `M${-rx2},0A${rx2},${ry2},0,1,0,${rx2},0A${rx2},${ry2},0,1,0,${-rx2},0`;
      } else {
         let large_arc = (ellipse.fPhimax-ellipse.fPhimin>=180) ? 1 : 0,
             a1 = ellipse.fPhimin*Math.PI/180, a2 = ellipse.fPhimax*Math.PI/180,
             dx1 = Math.round(rx1*Math.cos(a1)), dy1 = Math.round(ry1*Math.sin(a1)),
             dx2 = Math.round(rx1*Math.cos(a2)), dy2 = Math.round(ry1*Math.sin(a2)),
             dx3 = Math.round(rx2*Math.cos(a1)), dy3 = Math.round(ry2*Math.sin(a1)),
             dx4 = Math.round(rx2*Math.cos(a2)), dy4 = Math.round(ry2*Math.sin(a2));

         path = `M${dx2},${dy2}A${rx1},${ry1},0,${large_arc},0,${dx1},${dy1}` +
                `L${dx3},${dy3}A${rx2},${ry2},0,${large_arc},1,${dx4},${dy4}Z`;
      }
   } else if (ellipse.fTheta == 0) {
      if (closed_ellipse) {
         path = `M${-rx},0A${rx},${ry},0,1,0,${rx},0A${rx},${ry},0,1,0,${-rx},0Z`;
      } else {
         let x1 = Math.round(rx * Math.cos(ellipse.fPhimin*Math.PI/180)),
             y1 = Math.round(ry * Math.sin(ellipse.fPhimin*Math.PI/180)),
             x2 = Math.round(rx * Math.cos(ellipse.fPhimax*Math.PI/180)),
             y2 = Math.round(ry * Math.sin(ellipse.fPhimax*Math.PI/180));
         path = `M0,0L${x1},${y1}A${rx},${ry},0,1,1,${x2},${y2}Z`;
      }
   } else {
     let ct = Math.cos(ellipse.fTheta*Math.PI/180),
         st = Math.sin(ellipse.fTheta*Math.PI/180),
         phi1 = ellipse.fPhimin*Math.PI/180,
         phi2 = ellipse.fPhimax*Math.PI/180,
         np = 200,
         dphi = (phi2-phi1) / (np - (closed_ellipse ? 0 : 1)),
         lastx = 0, lasty = 0;
     if (!closed_ellipse) path = "M0,0";
     for (let n = 0; n < np; ++n) {
         let angle = phi1 + n*dphi,
             dx = ellipse.fR1 * Math.cos(angle),
             dy = ellipse.fR2 * Math.sin(angle),
             px = funcs.x(ellipse.fX1 + dx*ct - dy*st) - x,
             py = funcs.y(ellipse.fY1 + dx*st + dy*ct) - y;
         if (!path)
            path = `M${px},${py}`;
         else if (lastx == px)
            path += `v${py-lasty}`;
         else if (lasty == py)
            path += `h${px-lastx}`;
         else
            path += `l${px-lastx},${py-lasty}`;
         lastx = px; lasty = py;
     }
     path += "Z";
   }

   this.draw_g
      .append("svg:path")
      .attr("transform",`translate(${x},${y})`)
      .attr("d", path)
      .call(this.lineatt.func).call(this.fillatt.func);
}

/** @summary Draw TPie
  * @private */
function drawPie() {
   let pie = this.getObject();

   // create svg:g container for ellipse drawing
   this.createG();

   let xc = this.axisToSvg("x", pie.fX),
       yc = this.axisToSvg("y", pie.fY),
       rx = this.axisToSvg("x", pie.fX + pie.fRadius) - xc,
       ry = this.axisToSvg("y", pie.fY + pie.fRadius) - yc;

   this.draw_g.attr("transform",`translate(${xc},${yc})`);

   // Draw the slices
   let nb = pie.fPieSlices.length, total = 0,
       af = (pie.fAngularOffset*Math.PI)/180,
       x1 = Math.round(rx*Math.cos(af)), y1 = Math.round(ry*Math.sin(af));

   for (let n = 0; n < nb; n++)
      total += pie.fPieSlices[n].fValue;

   for (let n = 0; n < nb; n++) {
      let slice = pie.fPieSlices[n],
          lineatt = new TAttLineHandler({attr: slice}),
          fillatt = this.createAttFill(slice);

      af += slice.fValue/total*2*Math.PI;
      let x2 = Math.round(rx*Math.cos(af)), y2 = Math.round(ry*Math.sin(af));

      this.draw_g
          .append("svg:path")
          .attr("d", `M0,0L${x1},${y1}A${rx},${ry},0,0,0,${x2},${y2}z`)
          .call(lineatt.func)
          .call(fillatt.func);
      x1 = x2; y1 = y2;
   }
}

/** @summary Draw TBox
  * @private */
function drawBox() {

   let box = this.getObject(),
       opt = this.getDrawOpt(),
       draw_line = (opt.toUpperCase().indexOf("L")>=0),
       lineatt = this.createAttLine(box),
       fillatt = this.createAttFill(box);

   // create svg:g container for box drawing
   this.createG();

   let x1 = this.axisToSvg("x", box.fX1),
       x2 = this.axisToSvg("x", box.fX2),
       y1 = this.axisToSvg("y", box.fY1),
       y2 = this.axisToSvg("y", box.fY2),
       xx = Math.min(x1,x2), yy = Math.min(y1,y2),
       ww = Math.abs(x2-x1), hh = Math.abs(y1-y2);

   // if box filled, contour line drawn only with "L" draw option:
   if (!fillatt.empty() && !draw_line) lineatt.color = "none";

   this.draw_g
       .append("svg:path")
       .attr("d", `M${xx},${yy}h${ww}v${hh}h${-ww}z`)
       .call(lineatt.func)
       .call(fillatt.func);

   if (box.fBorderMode && box.fBorderSize && fillatt.hasColor()) {
      let pww = box.fBorderSize, phh = box.fBorderSize,
          side1 = `M${xx},${yy}h${ww}l${-pww},${phh}h${2*pww-ww}v${hh-2*phh}l${-pww},${phh}z`,
          side2 = `M${xx+ww},${yy+hh}v${-hh}l${-pww},${phh}v${hh-2*phh}h${2*pww-ww}l${-pww},${phh}z`;

      if (box.fBorderMode < 0) { let s = side1; side1 = side2; side2 = s; }

      this.draw_g.append("svg:path")
                 .attr("d", side1)
                 .call(fillatt.func)
                 .style("fill", d3_rgb(fillatt.color).brighter(0.5).formatHex());

      this.draw_g.append("svg:path")
          .attr("d", side2)
          .call(fillatt.func)
          .style("fill", d3_rgb(fillatt.color).darker(0.5).formatHex());
   }
}

/** @summary Draw TMarker
  * @private */
function drawMarker() {
   let marker = this.getObject(),
       att = new TAttMarkerHandler(marker),
       kMarkerNDC = BIT(14),
       isndc = marker.TestBit(kMarkerNDC);

   // create svg:g container for box drawing
   this.createG();

   let x = this.axisToSvg("x", marker.fX, isndc),
       y = this.axisToSvg("y", marker.fY, isndc),
       path = att.create(x,y);

   if (path)
      this.draw_g.append("svg:path")
          .attr("d", path)
          .call(att.func);
}

/** @summary Draw TPolyMarker
  * @private */
function drawPolyMarker() {

   // create svg:g container for box drawing
   this.createG();

   let poly = this.getObject(),
       att = new TAttMarkerHandler(poly),
       path = "",
       func = this.getAxisToSvgFunc();

   for (let n = 0; n < poly.fN; ++n)
      path += att.create(func.x(poly.fX[n]), func.y(poly.fY[n]));

   if (path)
      this.draw_g.append("svg:path")
          .attr("d", path)
          .call(att.func);
}

/** @summary Draw JS image
  * @private */
function drawJSImage(dom, obj, opt) {
   let painter = new BasePainter(dom),
       main = painter.selectDom(),
       img = main.append("img").attr("src", obj.fName).attr("title", obj.fTitle || obj.fName);

   if (opt && opt.indexOf("scale") >= 0) {
      img.style("width","100%").style("height","100%");
   } else if (opt && opt.indexOf("center") >= 0) {
      main.style("position", "relative");
      img.attr("style", "margin: 0; position: absolute;  top: 50%; left: 50%; transform: translate(-50%, -50%);");
   }

   painter.setTopPainter();

   return painter;
}

export { drawText, drawTLine, drawPolyLine, drawEllipse, drawPie, drawBox,
         drawMarker, drawPolyMarker, drawJSImage };
