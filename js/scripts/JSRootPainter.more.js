/// @file JSRootPainter.more.js
/// Part of JavaScript ROOT graphics with more classes like TEllipse, TLine, ...
/// Such classes are rarely used and therefore loaded only on demand

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3', 'JSRootMath'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"), require("d3"), require("./JSRootMath.js"));
   } else {

      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.v3.js', 'JSRootPainter.more.js');

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.more.js');

      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.more.js');

      factory(JSROOT, d3);
   }
} (function(JSROOT, d3) {

   "use strict";

   JSROOT.sources.push("more2d");

   function drawText() {
      var text = this.GetObject(),
          w = this.pad_width(), h = this.pad_height(),
          pos_x = text.fX, pos_y = text.fY,
          tcolor = this.get_color(text.fTextColor),
          use_frame = false,
          fact = 1., textsize = text.fTextSize || 0.05,
          main = this.frame_painter();

      if (text.TestBit(JSROOT.BIT(14))) {
         // NDC coordinates
         pos_x = pos_x * w;
         pos_y = (1 - pos_y) * h;
      } else if (main && !main.mode3d) {
         w = this.frame_width(); h = this.frame_height(); use_frame = "upper_layer";
         pos_x = main.grx(pos_x);
         pos_y = main.gry(pos_y);
      } else if (this.root_pad() !== null) {
         pos_x = this.ConvertToNDC("x", pos_x) * w;
         pos_y = (1 - this.ConvertToNDC("y", pos_y)) * h;
      } else {
         text.fTextAlign = 22;
         pos_x = w/2;
         pos_y = h/2;
         if (!tcolor) tcolor = 'black';
      }

      this.CreateG(use_frame);

      var arg = { align: text.fTextAlign, x: Math.round(pos_x), y: Math.round(pos_y), text: text.fTitle, color: tcolor, latex: 0 };

      if (text.fTextAngle) arg.rotate = -text.fTextAngle;

      if (text._typename == 'TLatex') { arg.latex = 1; fact = 0.9; } else
      if (text._typename == 'TMathText') { arg.latex = 2; fact = 0.8; }

      this.StartTextDrawing(text.fTextFont, Math.round((textsize>1) ? textsize : textsize*Math.min(w,h)*fact));

      this.DrawText(arg);

      this.FinishTextDrawing();
   }

   // =====================================================================================

   function drawLine() {

      var line = this.GetObject(),
          lineatt = new JSROOT.TAttLineHandler(line),
          kLineNDC = JSROOT.BIT(14),
          isndc = line.TestBit(kLineNDC);

      // create svg:g container for line drawing
      this.CreateG();

      this.draw_g
          .append("svg:line")
          .attr("x1", this.AxisToSvg("x", line.fX1, isndc))
          .attr("y1", this.AxisToSvg("y", line.fY1, isndc))
          .attr("x2", this.AxisToSvg("x", line.fX2, isndc))
          .attr("y2", this.AxisToSvg("y", line.fY2, isndc))
          .call(lineatt.func);
   }

   // =============================================================================

   function drawPolyLine() {

      var polyline = this.GetObject(),
          lineatt = new JSROOT.TAttLineHandler(polyline),
          fillatt = this.createAttFill(polyline),
          kPolyLineNDC = JSROOT.BIT(14),
          isndc = polyline.TestBit(kPolyLineNDC),
          cmd = "", func = this.AxisToSvgFunc(isndc);

      // create svg:g container for polyline drawing
      this.CreateG();

      for (var n=0;n<=polyline.fLastPoint;++n)
         cmd += ((n>0) ? "L" : "M") + func.x(polyline.fX[n]) + "," + func.y(polyline.fY[n]);

      if (polyline._typename != "TPolyLine") fillatt.SetSolidColor("none");

      if (!fillatt.empty()) cmd+="Z";

      this.draw_g
          .append("svg:path")
          .attr("d", cmd)
          .call(lineatt.func)
          .call(fillatt.func);
   }

   // ==============================================================================

   function drawEllipse() {

      var ellipse = this.GetObject();

      this.createAttLine({ attr: ellipse });
      this.createAttFill({ attr: ellipse });

      // create svg:g container for ellipse drawing
      this.CreateG();

      var x = this.AxisToSvg("x", ellipse.fX1, false),
          y = this.AxisToSvg("y", ellipse.fY1, false),
          rx = this.AxisToSvg("x", ellipse.fX1 + ellipse.fR1, false) - x,
          ry = y - this.AxisToSvg("y", ellipse.fY1 + ellipse.fR2, false);

      if (ellipse._typename == "TCrown") {
         if (ellipse.fR1 <= 0) {
            // handle same as ellipse with equal radius
            rx = this.AxisToSvg("x", ellipse.fX1 + ellipse.fR2, false) - x;
         } else {
            var rx1 = rx, ry2 = ry,
                ry1 = y - this.AxisToSvg("y", ellipse.fY1 + ellipse.fR1, false),
                rx2 = this.AxisToSvg("x", ellipse.fX1 + ellipse.fR2, false) - x;

            var elem = this.draw_g
                          .attr("transform","translate("+x+","+y+")")
                          .append("svg:path")
                          .call(this.lineatt.func).call(this.fillatt.func);

            if ((ellipse.fPhimin == 0) && (ellipse.fPhimax == 360)) {
               elem.attr("d", "M-"+rx1+",0" +
                              "A"+rx1+","+ry1+",0,1,0,"+rx1+",0" +
                              "A"+rx1+","+ry1+",0,1,0,-"+rx1+",0" +
                              "M-"+rx2+",0" +
                              "A"+rx2+","+ry2+",0,1,0,"+rx2+",0" +
                              "A"+rx2+","+ry2+",0,1,0,-"+rx2+",0");

            } else {
               var large_arc = (ellipse.fPhimax-ellipse.fPhimin>=180) ? 1 : 0;

               var a1 = ellipse.fPhimin*Math.PI/180, a2 = ellipse.fPhimax*Math.PI/180,
                   dx1 = rx1 * Math.cos(a1), dy1 = ry1 * Math.sin(a1),
                   dx2 = rx1 * Math.cos(a2), dy2 = ry1 * Math.sin(a2),
                   dx3 = rx2 * Math.cos(a1), dy3 = ry2 * Math.sin(a1),
                   dx4 = rx2 * Math.cos(a2), dy4 = ry2 * Math.sin(a2);

               elem.attr("d", "M"+Math.round(dx2)+","+Math.round(dy2)+
                              "A"+rx1+","+ry1+",0,"+large_arc+",0,"+Math.round(dx1)+","+Math.round(dy1)+
                              "L"+Math.round(dx3)+","+Math.round(dy3) +
                              "A"+rx2+","+ry2+",0,"+large_arc+",1,"+Math.round(dx4)+","+Math.round(dy4)+"Z");
            }

            return;
         }
      }

      if ((ellipse.fPhimin == 0) && (ellipse.fPhimax == 360) && (ellipse.fTheta == 0)) {
            // this is simple case, which could be drawn with svg:ellipse
         this.draw_g.append("svg:ellipse")
                    .attr("cx", x).attr("cy", y)
                    .attr("rx", rx).attr("ry", ry)
                    .call(this.lineatt.func).call(this.fillatt.func);
         return;
      }

      // here svg:path is used to draw more complex figure

      var ct = Math.cos(Math.PI*ellipse.fTheta/180),
          st = Math.sin(Math.PI*ellipse.fTheta/180),
          dx1 = rx * Math.cos(ellipse.fPhimin*Math.PI/180),
          dy1 = ry * Math.sin(ellipse.fPhimin*Math.PI/180),
          x1 =  dx1*ct - dy1*st,
          y1 = -dx1*st - dy1*ct,
          dx2 = rx * Math.cos(ellipse.fPhimax*Math.PI/180),
          dy2 = ry * Math.sin(ellipse.fPhimax*Math.PI/180),
          x2 =  dx2*ct - dy2*st,
          y2 = -dx2*st - dy2*ct;

      this.draw_g
         .attr("transform","translate("+x+","+y+")")
         .append("svg:path")
         .attr("d", "M0,0" +
                    "L" + Math.round(x1) + "," + Math.round(y1) +
                    "A"+rx+ ","+ry + "," + Math.round(-ellipse.fTheta) + ",1,0," + Math.round(x2) + "," + Math.round(y2) +
                    "Z")
         .call(this.lineatt.func).call(this.fillatt.func);
   }

   // ==============================================================================

   function drawPie() {
      var pie = this.GetObject();

      // create svg:g container for ellipse drawing
      this.CreateG();

      var xc = this.AxisToSvg("x", pie.fX, false),
          yc = this.AxisToSvg("y", pie.fY, false),
          rx = this.AxisToSvg("x", pie.fX + pie.fRadius, false) - xc,
          ry = this.AxisToSvg("y", pie.fY + pie.fRadius, false) - yc;

      this.draw_g.attr("transform","translate("+xc+","+yc+")");

      // Draw the slices
      var nb = pie.fPieSlices.length,
          slice, title, value, total = 0, lineatt, fillatt;

      for (var n=0;n<nb; n++) {
         slice = pie.fPieSlices[n];
         total = total + slice.fValue;
      }

      var af = (pie.fAngularOffset*Math.PI)/180.;
      var x1 = rx*Math.cos(af);
      var y1 = ry*Math.sin(af);
      var x2, y2, a = af;

      for (var n=0;n<nb; n++) {
         slice = pie.fPieSlices[n];
         lineatt = new JSROOT.TAttLineHandler(slice);
         fillatt = this.createAttFill(slice);
         value   = slice.fValue;
         a       = a + ((2*Math.PI)/total)*value;
         x2      = rx*Math.cos(a);
         y2      = ry*Math.sin(a);
         title   = slice.fTitle;
         this.draw_g
             .append("svg:path")
             .attr("d", "M0,0L"+x1.toFixed(1)+","+y1.toFixed(1)+"A"+
                         rx.toFixed(1)+","+ry.toFixed(1)+",0,0,0,"+x2.toFixed(1)+","+y2.toFixed(1)+
                        "Z")
             .call(lineatt.func)
             .call(fillatt.func);
         x1 = x2;
         y1 = y2;
      }
   }

   // =============================================================================

   function drawBox() {

      var box = this.GetObject(),
          opt = this.OptionsAsString(),
          draw_line = (opt.toUpperCase().indexOf("L")>=0),
          lineatt = this.createAttLine(box),
          fillatt = this.createAttFill(box);

      // create svg:g container for box drawing
      this.CreateG();

      var x1 = this.AxisToSvg("x", box.fX1, false),
          x2 = this.AxisToSvg("x", box.fX2, false),
          y1 = this.AxisToSvg("y", box.fY1, false),
          y2 = this.AxisToSvg("y", box.fY2, false),
          xx = Math.min(x1,x2), yy = Math.min(y1,y2),
          ww = Math.abs(x2-x1), hh = Math.abs(y1-y2);

      // if box filled, contour line drawn only with "L" draw option:
      if (!fillatt.empty() && !draw_line) lineatt.color = "none";

      this.draw_g
          .append("svg:rect")
          .attr("x", xx).attr("y", yy)
          .attr("width", ww)
          .attr("height", hh)
          .call(lineatt.func)
          .call(fillatt.func);

      if (box.fBorderMode && box.fBorderSize && (fillatt.color!=='none')) {
         var pww = box.fBorderSize, phh = box.fBorderSize,
             side1 = "M"+xx+","+yy + "h"+ww + "l"+(-pww)+","+phh + "h"+(2*pww-ww) +
                     "v"+(hh-2*phh)+ "l"+(-pww)+","+phh + "z",
             side2 = "M"+(xx+ww)+","+(yy+hh) + "v"+(-hh) + "l"+(-pww)+","+phh + "v"+(hh-2*phh)+
                     "h"+(2*pww-ww) + "l"+(-pww)+","+phh + "z";

         if (box.fBorderMode<0) { var s = side1; side1 = side2; side2 = s; }

         this.draw_g.append("svg:path")
                    .attr("d", side1)
                    .style("stroke","none")
                    .call(fillatt.func)
                    .style("fill", d3.rgb(fillatt.color).brighter(0.5).toString());

         this.draw_g.append("svg:path")
             .attr("d", side2)
             .style("stroke","none")
             .call(fillatt.func)
             .style("fill", d3.rgb(fillatt.color).darker(0.5).toString());
      }
   }

   // =============================================================================

   function drawMarker() {
      var marker = this.GetObject(),
          att = new JSROOT.TAttMarkerHandler(marker),
          kMarkerNDC = JSROOT.BIT(14),
          isndc = marker.TestBit(kMarkerNDC);

      // create svg:g container for box drawing
      this.CreateG();

      var x = this.AxisToSvg("x", marker.fX, isndc),
          y = this.AxisToSvg("y", marker.fY, isndc),
          path = att.create(x,y);

      if (path)
         this.draw_g.append("svg:path")
             .attr("d", path)
             .call(att.func);
   }

   // =============================================================================

   function drawPolyMarker() {
      var poly = this.GetObject(),
          att = new JSROOT.TAttMarkerHandler(poly),
          func = this.AxisToSvgFunc(false),
          path = "";

      // create svg:g container for box drawing
      this.CreateG();

      for (var n=0;n<poly.fN;++n)
         path += att.create(func.x(poly.fX[n]), func.y(poly.fY[n]));

      if (path)
         this.draw_g.append("svg:path")
             .attr("d", path)
             .call(att.func);
   }

   // ======================================================================================

   function drawArrow() {
      var arrow = this.GetObject(),
          wsize = Math.max(3, Math.round(Math.max(this.pad_width(), this.pad_height()) * arrow.fArrowSize)),
          hsize = Math.round(wsize * Math.tan(arrow.fAngle/2*Math.PI/180));

      this.createAttLine({ attr: arrow });
      this.createAttFill({ attr: arrow });

      // create svg:g container for line drawing
      this.CreateG();

      var x1 = this.AxisToSvg("x", arrow.fX1, false),
          y1 = this.AxisToSvg("y", arrow.fY1, false),
          x2 = this.AxisToSvg("x", arrow.fX2, false),
          y2 = this.AxisToSvg("y", arrow.fY2, false),
          right_arrow = "M0,0" + "L"+wsize+","+hsize + "L0,"+(2*hsize),
          left_arrow =  "M"+wsize+",0" + "L0,"+hsize + "L"+wsize+","+(2*hsize),
          m_start = null, m_mid = null, m_end = null, defs = null,
          oo = arrow.fOption, oolen = oo.length;

      if (oo.indexOf("<")==0) {
         var closed = (oo.indexOf("<|") == 0);
         if (!defs) defs = this.draw_g.append("defs");
         m_start = "jsroot_arrowmarker_" +  JSROOT.id_counter++;
         var mbeg = defs.append("svg:marker")
                       .attr("id", m_start)
                       .attr("markerWidth", wsize)
                       .attr("markerHeight", 2*hsize)
                       .attr("refX", 0)
                       .attr("refY", hsize)
                       .attr("orient", "auto")
                       .attr("markerUnits", "userSpaceOnUse");
         var pbeg = mbeg.append("svg:path")
                       .style("fill","none")
                       .attr("d", left_arrow + (closed ? "Z" : ""))
                       .call(this.lineatt.func);
         if (closed) {
            pbeg.call(this.fillatt.func);
            if ((this.fillatt.color == this.lineatt.color) && this.fillatt.isSolid()) pbeg.style('stroke-width',1);
            var dx = x2-x1, dy = y2-y1, len = Math.sqrt(dx*dx + dy*dy);
            if (len>wsize) {
               var ratio = wsize/len;
               x1 += ratio*dx;
               y1 += ratio*dy;
               mbeg.attr("refX", wsize);
            }
         }
      }

      var midkind = 0;
      if (oo.indexOf("->-")>=0)  midkind = 1; else
      if (oo.indexOf("-|>-")>=0) midkind = 11; else
      if (oo.indexOf("-<-")>=0) midkind = 2; else
      if (oo.indexOf("-<|-")>=0) midkind = 12;

      if (midkind > 0) {
         var closed = midkind > 10;
         if (!defs) defs = this.draw_g.append("defs");
         m_mid = "jsroot_arrowmarker_" + JSROOT.id_counter++;

         var pmid = defs.append("svg:marker")
                      .attr("id", m_mid)
                      .attr("markerWidth", wsize)
                      .attr("markerHeight", 2*hsize)
                      .attr("refX", Math.round(wsize*0.5))
                      .attr("refY", hsize)
                      .attr("orient", "auto")
                      .attr("markerUnits", "userSpaceOnUse")
                      .append("svg:path")
                      .style("fill","none")
                      .attr("d", ((midkind % 10 == 1) ? right_arrow : left_arrow) +
                            ((midkind > 10) ? "Z" : ""))
                            .call(this.lineatt.func);
         if (midkind > 10) {
            pmid.call(this.fillatt.func);
            if (this.fillatt.isSolid(this.lineatt.color)) pmid.style('stroke-width',1);
         }
      }

      if (oo.lastIndexOf(">") == oolen-1) {
         var closed = (oo.lastIndexOf("|>") == oolen-2) && (oolen>1);
         if (!defs) defs = this.draw_g.append("defs");
         m_end = "jsroot_arrowmarker_" + JSROOT.id_counter++;
         var mend = defs.append("svg:marker")
                       .attr("id", m_end)
                       .attr("markerWidth", wsize)
                       .attr("markerHeight", 2*hsize)
                       .attr("refX", wsize)
                       .attr("refY", hsize)
                       .attr("orient", "auto")
                       .attr("markerUnits", "userSpaceOnUse");
         var pend = mend.append("svg:path")
                       .style("fill","none")
                       .attr("d", right_arrow + (closed ? "Z" : ""))
                       .call(this.lineatt.func);
         if (closed) {
            pend.call(this.fillatt.func);
            if (this.fillatt.isSolid(this.lineatt.color)) pend.style('stroke-width',1);
            var dx = x2-x1, dy = y2-y1, len = Math.sqrt(dx*dx + dy*dy);
            if (len>wsize) {
               var ratio = wsize/len;
               x2 -= ratio*dx;
               y2 -= ratio*dy;
               mend.attr("refX", 0);
            }
         }
      }

      var path = this.draw_g
           .append("svg:path")
           .attr("d",  "M"+Math.round(x1)+","+Math.round(y1) +
                       ((m_mid == null) ? "" : "L" + Math.round(x1/2+x2/2) + "," + Math.round(y1/2+y2/2)) +
                       "L"+Math.round(x2)+","+Math.round(y2))
            .call(this.lineatt.func);

      if (m_start) path.style("marker-start","url(#" + m_start + ")");
      if (m_mid) path.style("marker-mid","url(#" + m_mid + ")");
      if (m_end) path.style("marker-end","url(#" + m_end + ")");
   }

   // =================================================================================

   function drawRooPlot(divid, plot, opt) {

      var painter = new JSROOT.TObjectPainter(plot), cnt = -1;

      function DrawNextItem() {
         if (++cnt >= plot._items.arr.length) return painter.DrawingReady();

         JSROOT.draw(divid, plot._items.arr[cnt], plot._items.opt[cnt], DrawNextItem);
      }

      JSROOT.draw(divid, plot._hist, "hist", DrawNextItem);

      return painter;
   }

   // ===================================================================================

   /**
    * @summary Painter for TF1 object.
    *
    * @constructor
    * @memberof JSROOT
    * @augments JSROOT.TObjectPainter
    * @param {object} tf1 - TF1 object to draw
    */

   function TF1Painter(tf1) {
      JSROOT.TObjectPainter.call(this, tf1);
      this.bins = null;
   }

   TF1Painter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TF1Painter.prototype.Eval = function(x) {
      return this.GetObject().evalPar(x);
   }

   //TF1Painter.prototype.UpdateObject = function(obj, opt) {
   //   if (!this.MatchObjectType(obj)) return false;
   //   var tf1 = this.GetObject();
   //   tf1.fSave = obj.fSave;
   //   return true;
   //}

   TF1Painter.prototype.CreateBins = function(ignore_zoom) {
      var main = this.frame_painter(),
          gxmin = 0, gxmax = 0, tf1 = this.GetObject();

      if (main && !ignore_zoom)  {
         if (main.zoom_xmin !== main.zoom_xmax) {
            gxmin = main.zoom_xmin;
            gxmax = main.zoom_xmax;
         } else {
            gxmin = main.xmin;
            gxmax = main.xmax;
         }
      }

      if ((tf1.fSave.length > 0) && !this.nosave) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         var np = tf1.fSave.length - 2,
             xmin = tf1.fSave[np],
             xmax = tf1.fSave[np+1],
             dx = (xmax - xmin) / (np-1),
             res = [];

         for (var n=0; n < np; ++n) {
            var xx = xmin + dx*n;
            // check if points need to be displayed at all, keep at least 4-5 points for Bezier curves
            if ((gxmin !== gxmax) && ((xx + 2*dx < gxmin) || (xx - 2*dx > gxmax))) continue;
            var yy = tf1.fSave[n];

            if (!isNaN(yy)) res.push({ x : xx, y : yy });
         }
         return res;
      }

      var xmin = tf1.fXmin, xmax = tf1.fXmax, logx = false;

      if (gxmin !== gxmax) {
         if (gxmin > xmin) xmin = gxmin;
         if (gxmax < xmax) xmax = gxmax;
      }

      if (main && main.logx && (xmin>0) && (xmax>0)) {
         logx = true;
         xmin = Math.log(xmin);
         xmax = Math.log(xmax);
      }

      var np = Math.max(tf1.fNpx, 101),
         dx = (xmax - xmin) / (np - 1),
         res = [];

      for (var n=0; n < np; n++) {
         var xx = xmin + n*dx;
         if (logx) xx = Math.exp(xx);
         var yy = this.Eval(xx);
         if (!isNaN(yy)) res.push({ x: xx, y: yy });
      }
      return res;
   }

   TF1Painter.prototype.CreateDummyHisto = function() {

      var xmin = 0, xmax = 1, ymin = 0, ymax = 1,
          bins = this.CreateBins(true);

      if (bins && (bins.length > 0)) {

         xmin = xmax = bins[0].x;
         ymin = ymax = bins[0].y;

         bins.forEach(function(bin) {
            xmin = Math.min(bin.x, xmin);
            xmax = Math.max(bin.x, xmax);
            ymin = Math.min(bin.y, ymin);
            ymax = Math.max(bin.y, ymax);
         });

         if (ymax > 0.0) ymax *= 1.05;
         if (ymin < 0.0) ymin *= 1.05;
      }

      var histo = JSROOT.Create("TH1I"),
          tf1 = this.GetObject();

      histo.fName = tf1.fName + "_hist";
      histo.fTitle = tf1.fTitle;

      histo.fXaxis.fXmin = xmin;
      histo.fXaxis.fXmax = xmax;
      histo.fYaxis.fXmin = ymin;
      histo.fYaxis.fXmax = ymax;

      return histo;
   }

   TF1Painter.prototype.ProcessTooltip = function(pnt) {
      var cleanup = false;

      if ((pnt === null) || (this.bins === null)) {
         cleanup = true;
      } else
      if ((this.bins.length==0) || (pnt.x < this.bins[0].grx) || (pnt.x > this.bins[this.bins.length-1].grx)) {
         cleanup = true;
      }

      if (cleanup) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      var min = 100000, best = -1, bin;

      for(var n=0; n<this.bins.length; ++n) {
         bin = this.bins[n];
         var dist = Math.abs(bin.grx - pnt.x);
         if (dist < min) { min = dist; best = n; }
      }

      bin = this.bins[best];

      var gbin = this.draw_g.select(".tooltip_bin"),
          radius = this.lineatt.width + 3;

      if (gbin.empty())
         gbin = this.draw_g.append("svg:circle")
                           .attr("class","tooltip_bin")
                           .style("pointer-events","none")
                           .attr("r", radius)
                           .call(this.lineatt.func)
                           .call(this.fillatt.func);

      var res = { name: this.GetObject().fName,
                  title: this.GetObject().fTitle,
                  x: bin.grx,
                  y: bin.gry,
                  color1: this.lineatt.color,
                  color2: this.fillatt.fillcolor(),
                  lines: [],
                  exact: (Math.abs(bin.grx - pnt.x) < radius) && (Math.abs(bin.gry - pnt.y) < radius) };

      res.changed = gbin.property("current_bin") !== best;
      res.menu = res.exact;
      res.menu_dist = Math.sqrt((bin.grx-pnt.x)*(bin.grx-pnt.x) + (bin.gry-pnt.y)*(bin.gry-pnt.y));

      if (res.changed)
         gbin.attr("cx", bin.grx)
             .attr("cy", bin.gry)
             .property("current_bin", best);

      var name = this.GetTipName();
      if (name.length > 0) res.lines.push(name);

      var pmain = this.frame_painter();
      if (pmain)
         res.lines.push("x = " + pmain.AxisAsText("x",bin.x) + " y = " + pmain.AxisAsText("y",bin.y));

      return res;
   }

   TF1Painter.prototype.Redraw = function() {

      var w = this.frame_width(),
          h = this.frame_height(),
          tf1 = this.GetObject(),
          fp = this.frame_painter(),
          pmain = this.main_painter(),
          name = this.GetTipName("\n");

      this.CreateG(true);

      // recalculate drawing bins when necessary
      this.bins = this.CreateBins(false);

      this.createAttLine({ attr: tf1 });
      this.lineatt.used = false;

      this.createAttFill({ attr: tf1, kind: 1 });
      this.fillatt.used = false;

      // first calculate graphical coordinates
      for(var n=0; n<this.bins.length; ++n) {
         var bin = this.bins[n];
         bin.grx = fp.grx(bin.x);
         bin.gry = fp.gry(bin.y);
      }

      if (this.bins.length > 2) {

         var h0 = h;  // use maximal frame height for filling
         if ((pmain.hmin!==undefined) && (pmain.hmin>=0)) {
            h0 = Math.round(fp.gry(0));
            if ((h0 > h) || (h0 < 0)) h0 = h;
         }

         var path = JSROOT.Painter.BuildSvgPath("bezier", this.bins, h0, 2);

         if (this.lineatt.color != "none")
            this.draw_g.append("svg:path")
               .attr("class", "line")
               .attr("d", path.path)
               .style("fill", "none")
               .call(this.lineatt.func);

         if (!this.fillatt.empty())
            this.draw_g.append("svg:path")
               .attr("class", "area")
               .attr("d", path.path + path.close)
               .style("stroke", "none")
               .call(this.fillatt.func);
      }
   }

   TF1Painter.prototype.CanZoomIn = function(axis,min,max) {
      if (axis!=="x") return false;

      var tf1 = this.GetObject();

      if (tf1.fSave.length > 0) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         var nb_points = tf1.fNpx;

         var xmin = tf1.fSave[nb_points + 1];
         var xmax = tf1.fSave[nb_points + 2];

         return Math.abs(xmin - xmax) / nb_points < Math.abs(min - max);
      }

      // if function calculated, one always could zoom inside
      return true;
   }

   TF1Painter.prototype.PerformDraw = function() {
      if (this.main_painter() === null) {
         var histo = this.CreateDummyHisto(), pthis = this;
         JSROOT.draw(this.divid, histo, "AXIS", function(hpainter) {
            pthis.SetDivId(pthis.divid);
            pthis.Redraw();
            return pthis.DrawingReady();
         });
         return pthis;
      }

      this.SetDivId(this.divid);
      this.Redraw();
      return this.DrawingReady();
   }

   function drawFunction(divid, tf1, opt) {

      var painter = new TF1Painter(tf1);

      painter.SetDivId(divid, -1);
      var d = new JSROOT.DrawOptions(opt);
      painter.nosave = d.check('NOSAVE');

      if (JSROOT.Math !== undefined)
         return painter.PerformDraw();

      JSROOT.AssertPrerequisites("math", painter.PerformDraw.bind(painter));
      return painter;
   }

   // =======================================================================

   /**
    * @summary Painter for TGraph object.
    *
    * @constructor
    * @memberof JSROOT
    * @augments JSROOT.TObjectPainter
    * @param {object} graph - TGraph object to draw
    */

   function TGraphPainter(graph) {
      JSROOT.TObjectPainter.call(this, graph);
      this.axes_draw = false; // indicate if graph histogram was drawn for axes
      this.bins = null;
      this.xmin = this.ymin = this.xmax = this.ymax = 0;
      this.wheel_zoomy = true;
      this.is_bent = (graph._typename == 'TGraphBentErrors');
      this.has_errors = (graph._typename == 'TGraphErrors') ||
                        (graph._typename == 'TGraphAsymmErrors') ||
                         this.is_bent || graph._typename.match(/^RooHist/);
   }

   TGraphPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TGraphPainter.prototype.Redraw = function() {
      this.DrawBins();
   }

   TGraphPainter.prototype.DecodeOptions = function(opt) {

      if (!opt) opt = this.main_painter() ? "lp" : "alp";

      if ((typeof opt == "string") && (opt.indexOf("same ")==0))
         opt = opt.substr(5);

      var graph = this.GetObject(),
          d = new JSROOT.DrawOptions(opt);

      if (!this.options) this.options = {};

      JSROOT.extend(this.options, {
         Line: 0, Curve: 0, Rect: 0, Mark: 0, Bar: 0, OutRange: 0,  EF:0, Fill: 0, NoOpt: 0,
         MainError: 1, Ends: 1, Axis: "", PadStats: false, PadTitle: false, original: opt
       });

      var res = this.options;

      res.PadStats = d.check("USE_PAD_STATS");
      res.PadTitle = d.check("USE_PAD_TITLE");

      res._pfc = d.check("PFC");
      res._plc = d.check("PLC");
      res._pmc = d.check("PMC");

      if (d.check('NOOPT')) res.NoOpt = 1;
      if (d.check('L')) res.Line = 1;
      if (d.check('F')) res.Fill = 1;
      if (d.check('A')) res.Axis = d.check("I") ? "A" : "AXIS"; // I means invisible axis
      if (d.check('X+')) res.Axis += "X+";
      if (d.check('Y+')) res.Axis += "Y+";
      if (d.check('RX')) res.Axis += "RX";
      if (d.check('RY')) res.Axis += "RY";
      if (d.check('C')) res.Curve = res.Line = 1;
      if (d.check('*')) res.Mark = 103;
      if (d.check('P0')) res.Mark = 104;
      if (d.check('P')) res.Mark = 1;
      if (d.check('B')) { res.Bar = 1; res.Errors = 0; }
      if (d.check('Z')) { res.Errors = 1; res.Ends = 0; }
      if (d.check('||')) { res.Errors = 1; res.MainError = 0; res.Ends = 1; }
      if (d.check('[]')) { res.Errors = 1; res.MainError = 0; res.Ends = 2; }
      if (d.check('|>')) { res.Errors = 1; res.Ends = 3; }
      if (d.check('>')) { res.Errors = 1; res.Ends = 4; }
      if (d.check('0')) { res.Mark = 1; res.Errors = 1; res.OutRange = 1; }
      if (d.check('1')) { if (res.Bar == 1) res.Bar = 2; }
      if (d.check('2')) { res.Rect = 1; res.Errors = 0; }
      if (d.check('3')) { res.EF = 1; res.Errors = 0;  }
      if (d.check('4')) { res.EF = 2; res.Errors = 0; }
      if (d.check('5')) { res.Rect = 2; res.Errors = 0; }
      if (d.check('X')) res.Errors = 0;
      // if (d.check('E')) res.Errors = 1; // E option only defined for TGraphPolar

      if (res.Errors === undefined)
         res.Errors = this.has_errors ? 1 : 0;

      // special case - one could use svg:path to draw many pixels (
      if ((res.Mark == 1) && (graph.fMarkerStyle==1)) res.Mark = 101;

      // if no drawing option is selected and if opt=='' nothing is done.
      if (res.Line + res.Fill + res.Mark + res.Bar + res.EF + res.Rect + res.Errors == 0) {
         if (d.empty()) res.Line = 1;
      }

      if (graph._typename == 'TGraphErrors') {
         if (d3.max(graph.fEX) < 1.0e-300 && d3.max(graph.fEY) < 1.0e-300)
            res.Errors = 0;
      }

      if (!res.Axis) {
         // check if axis should be drawn
         // either graph drawn directly or
         // graph is first object in list of primitives
         var pad = this.root_pad();
         if (!pad || (pad.fPrimitives && (pad.fPrimitives.arr[0] === graph))) res.Axis = "AXIS";
      } else if (res.Axis.indexOf("A")<0) {
         res.Axis = "AXIS," + res.Axis;
      }

      if (res.PadTitle) res.Axis += ";USE_PAD_TITLE";
   }

   TGraphPainter.prototype.CreateBins = function() {
      var gr = this.GetObject();
      if (!gr) return;

      var p, kind = 0, npoints = gr.fNpoints;
      if ((gr._typename==="TCutG") && (npoints>3)) npoints--;

      if (gr._typename == 'TGraphErrors') kind = 1; else
      if (gr._typename == 'TGraphAsymmErrors' || gr._typename == 'TGraphBentErrors'
          || gr._typename.match(/^RooHist/)) kind = 2;

      this.bins = [];

      for (p=0;p<npoints;++p) {
         var bin = { x: gr.fX[p], y: gr.fY[p], indx: p };
         switch(kind) {
            case 1:
               bin.exlow = bin.exhigh = gr.fEX[p];
               bin.eylow = bin.eyhigh = gr.fEY[p];
               break;
            case 2:
               bin.exlow  = gr.fEXlow[p];
               bin.exhigh = gr.fEXhigh[p];
               bin.eylow  = gr.fEYlow[p];
               bin.eyhigh = gr.fEYhigh[p];
               break;
         }
         this.bins.push(bin);

         if (p===0) {
            this.xmin = this.xmax = bin.x;
            this.ymin = this.ymax = bin.y;
         }

         if (kind > 0) {
            this.xmin = Math.min(this.xmin, bin.x - bin.exlow, bin.x + bin.exhigh);
            this.xmax = Math.max(this.xmax, bin.x - bin.exlow, bin.x + bin.exhigh);
            this.ymin = Math.min(this.ymin, bin.y - bin.eylow, bin.y + bin.eyhigh);
            this.ymax = Math.max(this.ymax, bin.y - bin.eylow, bin.y + bin.eyhigh);
         } else {
            this.xmin = Math.min(this.xmin, bin.x);
            this.xmax = Math.max(this.xmax, bin.x);
            this.ymin = Math.min(this.ymin, bin.y);
            this.ymax = Math.max(this.ymax, bin.y);
         }
      }
   }

   TGraphPainter.prototype.CreateHistogram = function(only_set_ranges) {
      // bins should be created when calling this function

      var xmin = this.xmin, xmax = this.xmax, ymin = this.ymin, ymax = this.ymax, set_x = true, set_y = true;

      if (xmin >= xmax) xmax = xmin+1;
      if (ymin >= ymax) ymax = ymin+1;
      var dx = (xmax-xmin)*0.1, dy = (ymax-ymin)*0.1,
          uxmin = xmin - dx, uxmax = xmax + dx,
          minimum = ymin - dy, maximum = ymax + dy;

      if ((uxmin<0) && (xmin>=0)) uxmin = xmin*0.9;
      if ((uxmax>0) && (xmax<=0)) uxmax = 0;

      var graph = this.GetObject();

      if (graph.fMinimum != -1111) minimum = ymin = graph.fMinimum;
      if (graph.fMaximum != -1111) maximum = ymax = graph.fMaximum;
      if ((minimum < 0) && (ymin >=0)) minimum = 0.9*ymin;

      var kResetHisto = JSROOT.BIT(17);   ///< fHistogram must be reset in GetHistogram

      var histo = graph.fHistogram;

      if (only_set_ranges) {
         set_x = only_set_ranges.indexOf("x") >= 0;
         set_y = only_set_ranges.indexOf("y") >= 0;
      } else if (histo) {
         // make logic like in the TGraph::GetHistogram
         if (!graph.TestBit(kResetHisto)) return histo;
         graph.InvertBit(kResetHisto);
      } else {
         graph.fHistogram = histo = JSROOT.CreateHistogram("TH1F", 100);
         histo.fName = graph.fName + "_h";
         histo.fTitle = graph.fTitle;
         histo.fBits = histo.fBits | JSROOT.TH1StatusBits.kNoStats;
         this._own_histogram = true;
      }

      if (set_x) {
         histo.fXaxis.fXmin = uxmin;
         histo.fXaxis.fXmax = uxmax;
      }
      if (set_y) {
         histo.fYaxis.fXmin = minimum;
         histo.fYaxis.fXmax = maximum;
         histo.fMinimum = minimum;
         histo.fMaximum = maximum;
      }

      return histo;
   }

   /** @summary Check if user range can be unzommed
    * @desc Used when graph points covers larger range than provided histogram
    * @private*/
   TGraphPainter.prototype.UnzoomUserRange = function(dox, doy, doz) {
      var graph = this.GetObject();
      if (this._own_histogram || !graph) return false;

      var histo = graph.fHistogram;
      if (!histo) return false;

      var arg = "";
      if (dox && (histo.fXaxis.fXmin > this.xmin) || (histo.fXaxis.fXmax < this.xmax)) arg += "x";
      if (doy && (histo.fYaxis.fXmin > this.ymin) || (histo.fYaxis.fXmax < this.ymax)) arg += "y";
      if (!arg) return false;

      this.CreateHistogram(arg);
      var hpainter = this.main_painter();
      if (hpainter) hpainter.CreateAxisFuncs(false);

      return true;
   }

   /** Returns true if graph drawing can be optimize */
   TGraphPainter.prototype.CanOptimize = function() {
      return JSROOT.gStyle.OptimizeDraw > 0 && !this.options.NoOpt;
   }

   /** Returns optimized bins - if optimization enabled */
   TGraphPainter.prototype.OptimizeBins = function(maxpnt, filter_func) {
      if ((this.bins.length < 30) && !filter_func) return this.bins;

      var selbins = null;
      if (typeof filter_func == 'function') {
         for (var n = 0; n < this.bins.length; ++n) {
            if (filter_func(this.bins[n],n)) {
               if (!selbins) selbins = (n==0) ? [] : this.bins.slice(0, n);
            } else {
               if (selbins) selbins.push(this.bins[n]);
            }
         }
      }
      if (!selbins) selbins = this.bins;

      if (!maxpnt) maxpnt = 500000;

      if ((selbins.length < maxpnt) || !this.CanOptimize()) return selbins;
      var step = Math.floor(selbins.length / maxpnt);
      if (step < 2) step = 2;
      var optbins = [];
      for (var n = 0; n < selbins.length; n+=step)
         optbins.push(selbins[n]);

      return optbins;
   }

   TGraphPainter.prototype.TooltipText = function(d) {
      var pmain = this.frame_painter(), lines = [];

      lines.push(this.GetTipName());

      if (d && pmain) {
         lines.push("x = " + pmain.AxisAsText("x", d.x));
         lines.push("y = " + pmain.AxisAsText("y", d.y));

         if (this.options.Errors && (pmain.x_kind=='normal') && ('exlow' in d) && ((d.exlow!=0) || (d.exhigh!=0)))
            lines.push("error x = -" + pmain.AxisAsText("x", d.exlow) + "/+" + pmain.AxisAsText("x", d.exhigh));

         if ((this.options.Errors || (this.options.EF > 0)) && (pmain.y_kind=='normal') && ('eylow' in d) && ((d.eylow!=0) || (d.eyhigh!=0)))
            lines.push("error y = -" + pmain.AxisAsText("y", d.eylow) + "/+" + pmain.AxisAsText("y", d.eyhigh));
      }
      return lines;
   }

   TGraphPainter.prototype.get_main = function() {
      var pmain = this.frame_painter();

      if (pmain && pmain.grx && pmain.gry) return pmain;

      pmain = {
          pad_layer: true,
          pad: this.root_pad(),
          pw: this.pad_width(),
          ph: this.pad_height(),
          grx: function(value) {
             if (this.pad.fLogx)
                value = (value>0) ? JSROOT.log10(value) : this.pad.fUxmin;
             else
                value = (value - this.pad.fX1) / (this.pad.fX2 - this.pad.fX1);
             return value*this.pw;
          },
          gry: function(value) {
             if (this.pad.fLogy)
                value = (value>0) ? JSROOT.log10(value) : this.pad.fUymin;
             else
                value = (value - this.pad.fY1) / (this.pad.fY2 - this.pad.fY1);
             return (1-value)*this.ph;
          }
      }

      return pmain.pad ? pmain : null;
   }

   TGraphPainter.prototype.DrawBins = function() {

      var pthis = this,
          pmain = this.get_main(),
          w = this.frame_width(),
          h = this.frame_height(),
          graph = this.GetObject(),
          excl_width = 0,
          pp = this.pad_painter();

      if (!pmain) return;

      this.CreateG(!pmain.pad_layer);

      if (pp && (this.options._pfc || this.options._plc || this.options._pmc)) {
         var icolor = pp.CreateAutoColor(this);

         if (this.options._pfc) { graph.fFillColor = icolor; delete this.fillatt; }
         if (this.options._plc) { graph.fLineColor = icolor; delete this.lineatt; }
         if (this.options._pmc) { graph.fMarkerColor = icolor; delete this.markeratt; }

         this.options._pfc = this.options._plc = this.options._pmc = false;
      }

      this.createAttLine({ attr: graph, can_excl: true });

      this.createAttFill({ attr: graph, kind: 1 });
      this.fillatt.used = false; // mark used only when really used

      this.draw_kind = "none"; // indicate if special svg:g were created for each bin
      this.marker_size = 0; // indicate if markers are drawn

      if (this.lineatt.excl_side != 0) {
         excl_width = this.lineatt.excl_width;
         if (this.lineatt.width > 0) this.options.Line = 1;
      }

      var drawbins = null;

      if (this.options.EF) {

         drawbins = this.OptimizeBins((this.options.EF > 1) ? 5000 : 0);

         // build lower part
         for (var n=0;n<drawbins.length;++n) {
            var bin = drawbins[n];
            bin.grx = pmain.grx(bin.x);
            bin.gry = pmain.gry(bin.y - bin.eylow);
         }

         var path1 = JSROOT.Painter.BuildSvgPath((this.options.EF > 1) ? "bezier" : "line", drawbins),
             bins2 = [];

         for (var n=drawbins.length-1;n>=0;--n) {
            var bin = drawbins[n];
            bin.gry = pmain.gry(bin.y + bin.eyhigh);
            bins2.push(bin);
         }

         // build upper part (in reverse direction)
         var path2 = JSROOT.Painter.BuildSvgPath((this.options.EF > 1) ? "Lbezier" : "Lline", bins2);

         this.draw_g.append("svg:path")
                    .attr("d", path1.path + path2.path + "Z")
                    .style("stroke", "none")
                    .call(this.fillatt.func);
         this.draw_kind = "lines";
      }

      if (this.options.Line == 1 || this.options.Fill == 1 || (excl_width!==0)) {

         var close_symbol = "";
         if (graph._typename=="TCutG") this.options.Fill = 1;

         if (this.options.Fill == 1) {
            close_symbol = "Z"; // always close area if we want to fill it
            excl_width = 0;
         }

         if (!drawbins) drawbins = this.OptimizeBins(this.options.Curve ? 5000 : 0);

         for (var n=0;n<drawbins.length;++n) {
            var bin = drawbins[n];
            bin.grx = pmain.grx(bin.x);
            bin.gry = pmain.gry(bin.y);
         }

         var kind = "line"; // simple line
         if (this.options.Curve === 1) kind = "bezier"; else
         if (excl_width!==0) kind+="calc"; // we need to calculated deltas to build exclusion points

         var path = JSROOT.Painter.BuildSvgPath(kind, drawbins);

         if (excl_width!==0) {
            var extrabins = [];
            for (var n=drawbins.length-1;n>=0;--n) {
               var bin = drawbins[n];
               var dlen = Math.sqrt(bin.dgrx*bin.dgrx + bin.dgry*bin.dgry);
               // shift point, using
               bin.grx += excl_width*bin.dgry/dlen;
               bin.gry -= excl_width*bin.dgrx/dlen;
               extrabins.push(bin);
            }

            var path2 = JSROOT.Painter.BuildSvgPath("L" + ((this.options.Curve === 1) ? "bezier" : "line"), extrabins);

            this.draw_g.append("svg:path")
                       .attr("d", path.path + path2.path + "Z")
                       .style("stroke", "none")
                       .call(this.fillatt.func)
                       .style('opacity', 0.75);
         }

         if (this.options.Line || this.options.Fill) {
            var elem = this.draw_g.append("svg:path")
                           .attr("d", path.path + close_symbol);
            if (this.options.Line)
               elem.call(this.lineatt.func);
            else
               elem.style('stroke','none');

            if (this.options.Fill)
               elem.call(this.fillatt.func);
            else
               elem.style('fill','none');
         }

         this.draw_kind = "lines";
      }

      var nodes = null;

      if (this.options.Errors || this.options.Rect || this.options.Bar) {

         drawbins = this.OptimizeBins(5000, function(pnt,i) {

            var grx = pmain.grx(pnt.x);

            // when drawing bars, take all points
            if (!pthis.options.Bar && ((grx<0) || (grx>w))) return true;

            var gry = pmain.gry(pnt.y);

            if (!pthis.options.Bar && !pthis.options.OutRange && ((gry<0) || (gry>h))) return true;

            pnt.grx1 = Math.round(grx);
            pnt.gry1 = Math.round(gry);

            if (pthis.has_errors) {
               pnt.grx0 = Math.round(pmain.grx(pnt.x - pnt.exlow) - grx);
               pnt.grx2 = Math.round(pmain.grx(pnt.x + pnt.exhigh) - grx);
               pnt.gry0 = Math.round(pmain.gry(pnt.y - pnt.eylow) - gry);
               pnt.gry2 = Math.round(pmain.gry(pnt.y + pnt.eyhigh) - gry);

               if (pthis.is_bent) {
                  pnt.grdx0 = Math.round(pmain.gry(pnt.y + graph.fEXlowd[i]) - gry);
                  pnt.grdx2 = Math.round(pmain.gry(pnt.y + graph.fEXhighd[i]) - gry);
                  pnt.grdy0 = Math.round(pmain.grx(pnt.x + graph.fEYlowd[i]) - grx);
                  pnt.grdy2 = Math.round(pmain.grx(pnt.x + graph.fEYhighd[i]) - grx);
               } else {
                  pnt.grdx0 = pnt.grdx2 = pnt.grdy0 = pnt.grdy2 = 0;
               }
            }

            return false;
         });

         this.draw_kind = "nodes";

         // here are up to five elements are collected, try to group them
         nodes = this.draw_g.selectAll(".grpoint")
                     .data(drawbins)
                     .enter()
                     .append("svg:g")
                     .attr("class", "grpoint")
                     .attr("transform", function(d) { return "translate(" + d.grx1 + "," + d.gry1 + ")"; });
      }

      if (this.options.Bar) {
         // calculate bar width
         for (var i=1;i<drawbins.length-1;++i)
            drawbins[i].width = Math.max(2, (drawbins[i+1].grx1 - drawbins[i-1].grx1) / 2 - 2);

         // first and last bins
         switch (drawbins.length) {
            case 0: break;
            case 1: drawbins[0].width = w/4; break; // pathologic case of single bin
            case 2: drawbins[0].width = drawbins[1].width = (drawbins[1].grx1-drawbins[0].grx1)/2; break;
            default:
               drawbins[0].width = drawbins[1].width;
               drawbins[drawbins.length-1].width = drawbins[drawbins.length-2].width;
         }

         var yy0 = Math.round(pmain.gry(0));

         nodes.append("svg:rect")
            .attr("x", function(d) { return Math.round(-d.width/2); })
            .attr("y", function(d) {
                d.bar = true; // element drawn as bar
                if (pthis.options.Bar!==1) return 0;
                return (d.gry1 > yy0) ? yy0-d.gry1 : 0;
             })
            .attr("width", function(d) { return Math.round(d.width); })
            .attr("height", function(d) {
                if (pthis.options.Bar!==1) return h > d.gry1 ? h - d.gry1 : 0;
                return Math.abs(yy0 - d.gry1);
             })
            .call(this.fillatt.func);
      }

      if (this.options.Rect)
         nodes.filter(function(d) { return (d.exlow > 0) && (d.exhigh > 0) && (d.eylow > 0) && (d.eyhigh > 0); })
           .append("svg:rect")
           .attr("x", function(d) { d.rect = true; return d.grx0; })
           .attr("y", function(d) { return d.gry2; })
           .attr("width", function(d) { return d.grx2 - d.grx0; })
           .attr("height", function(d) { return d.gry0 - d.gry2; })
           .call(this.fillatt.func)
           .call(this.options.Rect === 2 ? this.lineatt.func : function() {});

      this.error_size = 0;

      if (this.options.Errors) {
         // to show end of error markers, use line width attribute
         var lw = this.lineatt.width + JSROOT.gStyle.fEndErrorSize, bb = 0,
             vv = this.options.Ends ? "m0," + lw + "v-" + 2*lw : "",
             hh = this.options.Ends ? "m" + lw + ",0h-" + 2*lw : "",
             vleft = vv, vright = vv, htop = hh, hbottom = hh,
             mm = this.options.MainError ? "M0,0L" : "M"; // command to draw main errors

         switch (this.options.Ends) {
            case 2:  // option []
               bb = Math.max(this.lineatt.width+1, Math.round(lw*0.66));
               vleft = "m"+bb+","+lw + "h-"+bb + "v-"+2*lw + "h"+bb;
               vright = "m-"+bb+","+lw + "h"+bb + "v-"+2*lw + "h-"+bb;
               htop = "m-"+lw+","+bb + "v-"+bb + "h"+2*lw + "v"+bb;
               hbottom = "m-"+lw+",-"+bb + "v"+bb + "h"+2*lw + "v-"+bb;
               break;
            case 3: // option |>
               lw = Math.max(lw, Math.round(graph.fMarkerSize*8*0.66));
               bb = Math.max(this.lineatt.width+1, Math.round(lw*0.66));
               vleft = "l"+bb+","+lw + "v-"+2*lw + "l-"+bb+","+lw;
               vright = "l-"+bb+","+lw + "v-"+2*lw + "l"+bb+","+lw;
               htop = "l-"+lw+","+bb + "h"+2*lw + "l-"+lw+",-"+bb;
               hbottom = "l-"+lw+",-"+bb + "h"+2*lw + "l-"+lw+","+bb;
               break;
            case 4: // option >
               lw = Math.max(lw, Math.round(graph.fMarkerSize*8*0.66));
               bb = Math.max(this.lineatt.width+1, Math.round(lw*0.66));
               vleft = "l"+bb+","+lw + "m0,-"+2*lw + "l-"+bb+","+lw;
               vright = "l-"+bb+","+lw + "m0,-"+2*lw + "l"+bb+","+lw;
               htop = "l-"+lw+","+bb + "m"+2*lw + ",0l-"+lw+",-"+bb;
               hbottom = "l-"+lw+",-"+bb + "m"+2*lw + ",0l-"+lw+","+bb;
               break;
         }

         this.error_size = lw;

         lw = Math.floor((this.lineatt.width-1)/2); // one should take into account half of end-cup line width
         nodes.filter(function(d) { return (d.exlow > 0) || (d.exhigh > 0) || (d.eylow > 0) || (d.eyhigh > 0); })
             .append("svg:path")
             .call(this.lineatt.func)
             .style('fill', "none")
             .attr("d", function(d) {
                d.error = true;
                return ((d.exlow > 0)  ? mm + (d.grx0+lw) + "," + d.grdx0 + vleft : "") +
                       ((d.exhigh > 0) ? mm + (d.grx2-lw) + "," + d.grdx2 + vright : "") +
                       ((d.eylow > 0)  ? mm + d.grdy0 + "," + (d.gry0-lw) + hbottom : "") +
                       ((d.eyhigh > 0) ? mm + d.grdy2 + "," + (d.gry2+lw) + htop : "");
              });
      }

      if (this.options.Mark) {
         // for tooltips use markers only if nodes where not created
         var path = "", pnt, grx, gry;

         this.createAttMarker({ attr: graph, style: this.options.Mark - 100 });

         this.marker_size = this.markeratt.GetFullSize();

         this.markeratt.reset_pos();

         // let produce SVG at maximum 1MB
         var maxnummarker = 1000000 / (this.markeratt.MarkerLength() + 7), step = 1;

         if (!drawbins) drawbins = this.OptimizeBins(maxnummarker); else
            if (this.CanOptimize() && (drawbins.length > 1.5*maxnummarker))
               step = Math.min(2, Math.round(drawbins.length/maxnummarker));

         for (var n=0;n<drawbins.length;n+=step) {
            pnt = drawbins[n];
            grx = pmain.grx(pnt.x);
            if ((grx > -this.marker_size) && (grx < w + this.marker_size)) {
               gry = pmain.gry(pnt.y);
               if ((gry > -this.marker_size) && (gry < h + this.marker_size))
                  path += this.markeratt.create(grx, gry);
            }
         }

         if (path.length>0) {
            this.draw_g.append("svg:path")
                       .attr("d", path)
                       .call(this.markeratt.func);
            if ((nodes===null) && (this.draw_kind=="none"))
               this.draw_kind = (this.options.Mark==101) ? "path" : "mark";

         }
      }
   }

   TGraphPainter.prototype.ExtractTooltip = function(pnt) {
      if (!pnt) return null;

      if ((this.draw_kind=="lines") || (this.draw_kind=="path") || (this.draw_kind=="mark"))
         return this.ExtractTooltipForPath(pnt);

      if (this.draw_kind!="nodes") return null;

      var width = this.frame_width(),
          height = this.frame_height(),
          pmain = this.frame_painter(),
          painter = this,
          findbin = null, best_dist2 = 1e10, best = null,
          msize = this.marker_size ? Math.round(this.marker_size/2 + 1.5) : 0;

      this.draw_g.selectAll('.grpoint').each(function() {
         var d = d3.select(this).datum();
         if (d===undefined) return;
         var dist2 = Math.pow(pnt.x - d.grx1, 2);
         if (pnt.nproc===1) dist2 += Math.pow(pnt.y - d.gry1, 2);
         if (dist2 >= best_dist2) return;

         var rect = null;

         if (d.error || d.rect || d.marker) {
            rect = { x1: Math.min(-painter.error_size, d.grx0, -msize),
                     x2: Math.max(painter.error_size, d.grx2, msize),
                     y1: Math.min(-painter.error_size, d.gry2, -msize),
                     y2: Math.max(painter.error_size, d.gry0, msize) };
         } else if (d.bar) {
             rect = { x1: -d.width/2, x2: d.width/2, y1: 0, y2: height - d.gry1 };

             if (painter.options.Bar===1) {
                var yy0 = pmain.gry(0);
                rect.y1 = (d.gry1 > yy0) ? yy0-d.gry1 : 0;
                rect.y2 = (d.gry1 > yy0) ? 0 : yy0-d.gry1;
             }
          } else {
             rect = { x1: -5, x2: 5, y1: -5, y2: 5 };
          }
          var matchx = (pnt.x >= d.grx1 + rect.x1) && (pnt.x <= d.grx1 + rect.x2),
              matchy = (pnt.y >= d.gry1 + rect.y1) && (pnt.y <= d.gry1 + rect.y2);

          if (matchx && (matchy || (pnt.nproc > 1))) {
             best_dist2 = dist2;
             findbin = this;
             best = rect;
             best.exact = matchx && matchy;
          }
       });

      if (findbin === null) return null;

      var d = d3.select(findbin).datum();

      var res = { name: this.GetObject().fName, title: this.GetObject().fTitle,
                  x: d.grx1, y: d.gry1,
                  color1: this.lineatt.color,
                  lines: this.TooltipText(d),
                  rect: best, d3bin: findbin  };

      if (this.fillatt && this.fillatt.used && !this.fillatt.empty()) res.color2 = this.fillatt.fillcolor();

      if (best.exact) res.exact = true;
      res.menu = res.exact; // activate menu only when exactly locate bin
      res.menu_dist = 3; // distance always fixed
      res.bin = d;
      res.binindx = d.indx;

      if (pnt.click_handler && res.exact && this.TestEditable())
         res.click_handler = this.InvokeClickHandler.bind(this);

      return res;
   }

   TGraphPainter.prototype.ShowTooltip = function(hint) {

      if (!hint) {
         if (this.draw_g) this.draw_g.select(".tooltip_bin").remove();
         return;
      }

      if (hint.usepath) return this.ShowTooltipForPath(hint);

      var d = d3.select(hint.d3bin).datum();

      var ttrect = this.draw_g.select(".tooltip_bin");

      if (ttrect.empty())
         ttrect = this.draw_g.append("svg:rect")
                             .attr("class","tooltip_bin h1bin")
                             .style("pointer-events","none");

      hint.changed = ttrect.property("current_bin") !== hint.d3bin;

      if (hint.changed)
         ttrect.attr("x", d.grx1 + hint.rect.x1)
               .attr("width", hint.rect.x2 - hint.rect.x1)
               .attr("y", d.gry1 + hint.rect.y1)
               .attr("height", hint.rect.y2 - hint.rect.y1)
               .style("opacity", "0.3")
               .property("current_bin", hint.d3bin);
   }

   TGraphPainter.prototype.ProcessTooltip = function(pnt) {
      var hint = this.ExtractTooltip(pnt);
      if (!pnt || !pnt.disabled) this.ShowTooltip(hint);
      return hint;
   }

   TGraphPainter.prototype.FindBestBin = function(pnt) {
      if (!this.bins) return null;

      var islines = (this.draw_kind=="lines"),
          ismark = (this.draw_kind=="mark"),
          bestindx = -1,
          bestbin = null,
          bestdist = 1e10,
          pmain = this.frame_painter(),
          dist, grx, gry, n, bin;

      for (n=0;n<this.bins.length;++n) {
         bin = this.bins[n];

         grx = pmain.grx(bin.x);
         gry = pmain.gry(bin.y);

         dist = (pnt.x-grx)*(pnt.x-grx) + (pnt.y-gry)*(pnt.y-gry);

         if (dist < bestdist) {
            bestdist = dist;
            bestbin = bin;
            bestindx = n;
         }
      }

      // check last point
      if ((bestdist > 100) && islines) bestbin = null;

      var radius = Math.max(this.lineatt.width + 3, 4);

      if (this.marker_size > 0) radius = Math.max(this.marker_size, radius);

      if (bestbin)
         bestdist = Math.sqrt(Math.pow(pnt.x-pmain.grx(bestbin.x),2) + Math.pow(pnt.y-pmain.gry(bestbin.y),2));

      if (!islines && (bestdist > radius)) bestbin = null;

      if (!bestbin) bestindx = -1;

      var res = { bin: bestbin, indx: bestindx, dist: bestdist, radius: Math.round(radius) };

      if (!bestbin && islines) {

         bestdist = 10000;

         function IsInside(x, x1, x2) {
            return ((x1>=x) && (x>=x2)) || ((x1<=x) && (x<=x2));
         }

         var bin0 = this.bins[0], grx0 = pmain.grx(bin0.x), gry0, posy = 0;
         for (n=1;n<this.bins.length;++n) {
            bin = this.bins[n];
            grx = pmain.grx(bin.x);

            if (IsInside(pnt.x, grx0, grx)) {
               // if inside interval, check Y distance
               gry0 = pmain.gry(bin0.y)
               gry = pmain.gry(bin.y);

               if (Math.abs(grx - grx0) < 1) {
                  // very close x - check only y
                  posy = pnt.y;
                  dist = IsInside(pnt.y, gry0, gry) ? 0 : Math.min(Math.abs(pnt.y-gry0), Math.abs(pnt.y-gry));
               } else {
                  posy = gry0 + (pnt.x - grx0) / (grx - grx0) * (gry - gry0);
                  dist = Math.abs(posy - pnt.y);
               }

               if (dist < bestdist) {
                  bestdist = dist;
                  res.linex = pnt.x;
                  res.liney = posy;
               }
            }

            bin0 = bin;
            grx0 = grx;
         }

         if (bestdist < radius*0.5) {
            res.linedist = bestdist;
            res.closeline = true;
         }
      }

      return res;
   }

   TGraphPainter.prototype.TestEditable = function(toggle) {
      var obj = this.GetObject(),
          kNotEditable = JSROOT.BIT(18);   // bit set if graph is non editable

      if (!obj) return false;
      if (toggle) obj.InvertBit(kNotEditable);
      return !obj.TestBit(kNotEditable);
   }

   TGraphPainter.prototype.ExtractTooltipForPath = function(pnt) {

      if (this.bins === null) return null;

      var best = this.FindBestBin(pnt);

      if (!best || (!best.bin && !best.closeline)) return null;

      var islines = (this.draw_kind=="lines"),
          ismark = (this.draw_kind=="mark"),
          pmain = this.frame_painter(),
          gr = this.GetObject(),
          res = { name: gr.fName, title: gr.fTitle,
                  x: best.bin ? pmain.grx(best.bin.x) : best.linex,
                  y: best.bin ? pmain.gry(best.bin.y) : best.liney,
                  color1: this.lineatt.color,
                  lines: this.TooltipText(best.bin),
                  usepath: true };

      res.ismark = ismark;
      res.islines = islines;

      if (best.closeline) {
         res.menu = res.exact = true;
         res.menu_dist = best.linedist;
      } else if (best.bin) {
         if (this.options.EF && islines) {
            res.gry1 = pmain.gry(best.bin.y - best.bin.eylow);
            res.gry2 = pmain.gry(best.bin.y + best.bin.eyhigh);
         } else {
            res.gry1 = res.gry2 = pmain.gry(best.bin.y);
         }

         res.binindx = best.indx;
         res.bin = best.bin;
         res.radius = best.radius;

         res.exact = (Math.abs(pnt.x - res.x) <= best.radius) &&
            ((Math.abs(pnt.y - res.gry1) <= best.radius) || (Math.abs(pnt.y - res.gry2) <= best.radius));

         res.menu = res.exact;
         res.menu_dist = Math.sqrt((pnt.x-res.x)*(pnt.x-res.x) + Math.pow(Math.min(Math.abs(pnt.y-res.gry1),Math.abs(pnt.y-res.gry2)),2));

         if (pnt.click_handler && res.exact && this.TestEditable())
            res.click_handler = this.InvokeClickHandler.bind(this);
      }

      if (this.fillatt && this.fillatt.used && !this.fillatt.empty()) res.color2 = this.fillatt.fillcolor();

      if (!islines) {
         res.color1 = this.get_color(gr.fMarkerColor);
         if (!res.color2) res.color2 = res.color1;
      }

      return res;
   }

   TGraphPainter.prototype.ShowTooltipForPath = function(hint) {

      var ttbin = this.draw_g.select(".tooltip_bin");

      if (!hint || !hint.bin) {
         ttbin.remove();
         return;
      }

      if (ttbin.empty())
         ttbin = this.draw_g.append("svg:g")
                             .attr("class","tooltip_bin");

      hint.changed = ttbin.property("current_bin") !== hint.bin;

      if (hint.changed) {
         ttbin.selectAll("*").remove(); // first delete all children
         ttbin.property("current_bin", hint.bin);

         if (hint.ismark) {
            ttbin.append("svg:rect")
                 .attr("class","h1bin")
                 .style("pointer-events","none")
                 .style("opacity", "0.3")
                 .attr("x", Math.round(hint.x - hint.radius))
                 .attr("y", Math.round(hint.y - hint.radius))
                 .attr("width", 2*hint.radius)
                 .attr("height", 2*hint.radius);
         } else {
            ttbin.append("svg:circle").attr("cy", Math.round(hint.gry1))
            if (Math.abs(hint.gry1-hint.gry2) > 1)
               ttbin.append("svg:circle").attr("cy", Math.round(hint.gry2));

            var elem = ttbin.selectAll("circle")
                            .attr("r", hint.radius)
                            .attr("cx", Math.round(hint.x));

            if (!hint.islines) {
               elem.style('stroke', hint.color1 == 'black' ? 'green' : 'black').style('fill','none');
            } else {
               if (this.options.Line)
                  elem.call(this.lineatt.func);
               else
                  elem.style('stroke','black');
               if (this.options.Fill)
                  elem.call(this.fillatt.func);
               else
                  elem.style('fill','none');
            }
         }
      }
   }

   TGraphPainter.prototype.movePntHandler = function(first_time) {
      var pos = d3.mouse(this.svg_frame().node()),
          main = this.frame_painter();

      if (!main || !this.interactive_bin) return;

      this.interactive_bin.x = main.RevertX(pos[0] + this.interactive_delta_x);
      this.interactive_bin.y = main.RevertY(pos[1] + this.interactive_delta_y);
      this.DrawBins();
   }

   TGraphPainter.prototype.endPntHandler = function() {
      if (this.snapid && this.interactive_bin) {
         var exec = "SetPoint(" + this.interactive_bin.indx + "," + this.interactive_bin.x + "," + this.interactive_bin.y + ")";
         var canp = this.canv_painter();
         if (canp && !canp._readonly)
            canp.SendWebsocket("OBJEXEC:" + this.snapid + ":" + exec);
      }

      delete this.interactive_bin;
      d3.select(window).on("mousemove.graphPnt", null)
                       .on("mouseup.graphPnt", null);
   }

   TGraphPainter.prototype.InvokeClickHandler = function(hint) {
      if (!hint.bin) return; //

      this.interactive_bin = hint.bin;

      d3.select(window).on("mousemove.graphPnt", this.movePntHandler.bind(this))
                       .on("mouseup.graphPnt", this.endPntHandler.bind(this), true);

      var pos = d3.mouse(this.svg_frame().node()),
          main = this.frame_painter();

      this.interactive_delta_x = main ? main.x(this.interactive_bin.x) - pos[0] : 0;
      this.interactive_delta_y = main ? main.y(this.interactive_bin.y) - pos[1] : 0;
   }

   TGraphPainter.prototype.FillContextMenu = function(menu) {
      JSROOT.TObjectPainter.prototype.FillContextMenu.call(this, menu);

      if (!this.snapid)
         menu.addchk(this.TestEditable(), "Editable", this.TestEditable.bind(this, true));

      return menu.size() > 0;
   }

   TGraphPainter.prototype.ExecuteMenuCommand = function(method, args) {
      if (JSROOT.TObjectPainter.prototype.ExecuteMenuCommand.call(this,method,args)) return true;

      var canp = this.canv_painter(), fp = this.frame_painter();

      if ((method.fName == 'RemovePoint') || (method.fName == 'InsertPoint')) {
         var pnt = fp ? fp.GetLastEventPos() : null;

         if (!canp || canp._readonly || !fp || !pnt) return true; // ignore function

         var hint = this.ExtractTooltip(pnt);

         if (method.fName == 'InsertPoint') {
            var main = this.frame_painter(),
                userx = main && main.RevertX ? main.RevertX(pnt.x) : 0,
                usery = main && main.RevertY ? main.RevertY(pnt.y) : 0;
            canp.ShowMessage('InsertPoint(' + userx.toFixed(3) + ',' + usery.toFixed(3) + ') not yet implemented');
         } else if (this.args_menu_id && hint && (hint.binindx !== undefined)) {
            var exec = "RemovePoint(" + hint.binindx + ")";
            console.log('execute ' + exec + ' for object ' + this.args_menu_id);
            canp.SendWebsocket('OBJEXEC:' + this.args_menu_id + ":" + exec);
         }

         return true; // call is processed
      }

      return false;
   }

   TGraphPainter.prototype.UpdateObject = function(obj, opt) {
      if (!this.MatchObjectType(obj)) return false;

      if ((opt !== undefined) && (opt != this.options.original))
         this.DecodeOptions(opt);

      var graph = this.GetObject();
      // TODO: make real update of TGraph object content
      graph.fBits = obj.fBits;
      graph.fTitle = obj.fTitle;
      graph.fX = obj.fX;
      graph.fY = obj.fY;
      graph.fNpoints = obj.fNpoints;
      this.CreateBins();

      // if our own histogram was used as axis drawing, we need update histogram as well
      if (this.axes_draw) {
         var main = this.main_painter(),
             fp = this.frame_painter();

         // if zoom was changed - do not update histogram
         if (!fp.zoom_changed_interactive)
            main.UpdateObject(obj.fHistogram || this.CreateHistogram());

         main.GetObject().fTitle = graph.fTitle; // copy title
      }

      return true;
   }

   TGraphPainter.prototype.CanZoomIn = function(axis,min,max) {
      // allow to zoom TGraph only when at least one point in the range

      var gr = this.GetObject();
      if ((gr===null) || (axis!=="x")) return false;

      for (var n=0; n < gr.fNpoints; ++n)
         if ((min < gr.fX[n]) && (gr.fX[n] < max)) return true;

      return false;
   }

   TGraphPainter.prototype.ButtonClick = function(funcname) {

      if (funcname !== "ToggleZoom") return false;

      var main = this.frame_painter();
      if (!main) return false;

      if ((this.xmin===this.xmax) && (this.ymin===this.ymax)) return false;

      main.Zoom(this.xmin, this.xmax, this.ymin, this.ymax);

      return true;
   }

   TGraphPainter.prototype.FindFunc = function() {
      var gr = this.GetObject();
      if (gr && gr.fFunctions)
         for (var i = 0; i < gr.fFunctions.arr.length; ++i) {
            var func = gr.fFunctions.arr[i];
            if ((func._typename == 'TF1') || (func._typename == 'TF2')) return func;
         }
      return null;
   }

   TGraphPainter.prototype.FindStat = function() {
      var gr = this.GetObject();
      if (gr && gr.fFunctions)
         for (var i = 0; i < gr.fFunctions.arr.length; ++i) {
            var func = gr.fFunctions.arr[i];
            if ((func._typename == 'TPaveStats') && (func.fName == 'stats')) return func;
         }

      return null;
   }

   TGraphPainter.prototype.CreateStat = function() {
      var func = this.FindFunc();
      if (!func) return null;

      var stats = this.FindStat();
      if (stats) return stats;

      // do not create stats box when drawing canvas
      var pp = this.canv_painter();
      if (pp && pp.normal_canvas) return null;

      if (this.options.PadStats) return null;

      this.create_stats = true;

      var st = JSROOT.gStyle;

      stats = JSROOT.Create('TPaveStats');
      JSROOT.extend(stats, { fName : 'stats',
                             fOptStat: 0,
                             fOptFit: st.fOptFit || 111,
                             fBorderSize : 1} );

      stats.fX1NDC = st.fStatX - st.fStatW;
      stats.fY1NDC = st.fStatY - st.fStatH;
      stats.fX2NDC = st.fStatX;
      stats.fY2NDC = st.fStatY;

      stats.fFillColor = st.fStatColor;
      stats.fFillStyle = st.fStatStyle;

      stats.fTextAngle = 0;
      stats.fTextSize = st.fStatFontSize; // 9 ??
      stats.fTextAlign = 12;
      stats.fTextColor = st.fStatTextColor;
      stats.fTextFont = st.fStatFont;

      stats.AddText(func.fName);

      // while TF1 was found, one can be sure that stats is existing
      this.GetObject().fFunctions.Add(stats);

      return stats;
   }

   TGraphPainter.prototype.FillStatistic = function(stat, dostat, dofit) {

      // cannot fill stats without func
      var func = this.FindFunc();

      if (!func || !dofit || !this.create_stats) return false;

      stat.ClearPave();

      stat.FillFunctionStat(func, dofit);

      return true;
   }

   TGraphPainter.prototype.DrawNextFunction = function(indx, callback) {
      // method draws next function from the functions list

      var graph = this.GetObject();

      if (!graph.fFunctions || (indx >= graph.fFunctions.arr.length))
         return JSROOT.CallBack(callback);

      var func = graph.fFunctions.arr[indx], opt = graph.fFunctions.opt[indx];

      //  required for stats filling
      // TODO: use weak reference (via pad list of painters and any kind of string)
      func.$main_painter = this;

      JSROOT.draw(this.divid, func, opt, this.DrawNextFunction.bind(this, indx+1, callback));
   }

   TGraphPainter.prototype.PerformDrawing = function(divid, hpainter) {
      if (hpainter) {
         this.axes_draw = true;
         if (!this._own_histogram) this.$primary = true;
         hpainter.$secondary = true;
      }
      this.SetDivId(divid);
      this.DrawBins();
      this.DrawNextFunction(0, this.DrawingReady.bind(this));
      return this;
   }

   function drawGraph(divid, graph, opt) {

      var painter = new TGraphPainter(graph);

      painter.SetDivId(divid, -1); // just to get access to existing elements

      painter.DecodeOptions(opt);

      painter.CreateBins();

      painter.CreateStat();

      if (!painter.main_painter() && painter.options.Axis) {
         JSROOT.draw(divid, painter.CreateHistogram(), painter.options.Axis, painter.PerformDrawing.bind(painter, divid));
      } else {
         painter.PerformDrawing(divid);
      }

      return painter;
   }

   // ==============================================================

   function TGraphPolargramPainter(polargram) {
      JSROOT.TooltipHandler.call(this, polargram);
      this.$polargram = true; // indicate that this is polargram
      this.zoom_rmin = this.zoom_rmax = 0;
   }

   TGraphPolargramPainter.prototype = Object.create(JSROOT.TooltipHandler.prototype);

   TGraphPolargramPainter.prototype.translate = function(angle, radius, keep_float) {
      var _rx = this.r(radius), _ry = _rx/this.szx*this.szy,
          pos = {
            x: _rx * Math.cos(-angle - this.angle),
            y: _ry * Math.sin(-angle - this.angle),
            rx: _rx,
            ry: _ry
         };

      if (!keep_float) {
         pos.x = Math.round(pos.x);
         pos.y = Math.round(pos.y);
         pos.rx =  Math.round(pos.rx);
         pos.ry =  Math.round(pos.ry);
      }
      return pos;
   }

   TGraphPolargramPainter.prototype.format = function(radius) {
      // used to format label for radius ticks

      if (radius === Math.round(radius)) return radius.toString();
      if (this.ndig>10) return radius.toExponential(4);

      return radius.toFixed((this.ndig > 0) ? this.ndig : 0);
   }

   TGraphPolargramPainter.prototype.AxisAsText = function(axis, value) {

      if (axis == "r") {
         if (value === Math.round(value)) return value.toString();
         if (this.ndig>10) return value.toExponential(4);
         return value.toFixed(this.ndig+2);
      }

      value *= 180/Math.PI;
      return (value === Math.round(value)) ? value.toString() : value.toFixed(1);
   }

   TGraphPolargramPainter.prototype.MouseEvent = function(kind) {
      var layer = this.svg_layer("primitives_layer"),
          interactive = layer.select(".interactive_ellipse");
      if (interactive.empty()) return;

      var pnt = null;

      if (kind !== 'leave') {
         var pos = d3.mouse(interactive.node());
         pnt = { x: pos[0], y: pos[1], touch: false };
      }

      this.ProcessTooltipEvent(pnt);
   }

   TGraphPolargramPainter.prototype.GetFrameRect = function() {
      var pad = this.root_pad(),
          w = this.pad_width(),
          h = this.pad_height(),
          rect = {};

      rect.szx = Math.round(Math.max(0.1, 0.5 - Math.max(pad.fLeftMargin, pad.fRightMargin))*w);
      rect.szy = Math.round(Math.max(0.1, 0.5 - Math.max(pad.fBottomMargin, pad.fTopMargin))*h);

      rect.width = 2*rect.szx;
      rect.height = 2*rect.szy;
      rect.midx = Math.round(w/2);
      rect.midy = Math.round(h/2);
      rect.x = rect.midx - rect.szx;
      rect.y = rect.midy - rect.szy;

      rect.hint_delta_x = rect.szx;
      rect.hint_delta_y = rect.szy;

      rect.transform = "translate(" + rect.x + "," + rect.y + ")";

      return rect;
   }

   TGraphPolargramPainter.prototype.MouseWheel = function() {
      d3.event.stopPropagation();
      d3.event.preventDefault();

      this.ProcessTooltipEvent(null); // remove all tooltips

      var polar = this.GetObject();

      if (!d3.event || !polar) return;

      var delta = d3.event.wheelDelta ? -d3.event.wheelDelta : (d3.event.deltaY || d3.event.detail);
      if (!delta) return;

      delta = (delta<0) ? -0.2 : 0.2;

      var rmin = this.scale_rmin, rmax = this.scale_rmax, range = rmax - rmin;

      // rmin -= delta*range;
      rmax += delta*range;

      if ((rmin<polar.fRwrmin) || (rmax>polar.fRwrmax)) rmin = rmax = 0;

      if ((this.zoom_rmin != rmin) || (this.zoom_rmax != rmax)) {
         this.zoom_rmin = rmin;
         this.zoom_rmax = rmax;
         this.RedrawPad();
      }
   }

   TGraphPolargramPainter.prototype.Redraw = function() {
      if (!this.is_main_painter()) return;

      var pad = this.root_pad(),
          polar = this.GetObject(),
          rect = this.GetFrameRect();

      this.CreateG();

      this.draw_g.attr("transform", "translate(" + rect.midx + "," + rect.midy + ")");
      this.szx = rect.szx;
      this.szy = rect.szy;

      this.scale_rmin = polar.fRwrmin;
      this.scale_rmax = polar.fRwrmax;
      if (this.zoom_rmin != this.zoom_rmax) {
         this.scale_rmin = this.zoom_rmin;
         this.scale_rmax = this.zoom_rmax;
      }

      this.r = d3.scaleLinear().domain([this.scale_rmin, this.scale_rmax]).range([ 0, this.szx ]);
      this.angle = polar.fAxisAngle || 0;

      var ticks = this.r.ticks(5),
          nminor = Math.floor((polar.fNdivRad % 10000) / 100);

      this.createAttLine({ attr: polar });
      if (!this.gridatt) this.gridatt = new JSROOT.TAttLineHandler({ color: polar.fLineColor, style: 2, width: 1 });

      var range = Math.abs(polar.fRwrmax - polar.fRwrmin);
      this.ndig = (range <= 0) ? -3 : Math.round(JSROOT.log10(ticks.length / range));

      // verify that all radius labels are unique
      var lbls = [], indx = 0;
      while (indx<ticks.length) {
         var lbl = this.format(ticks[indx]);
         if (lbls.indexOf(lbl)>=0) {
            if (++this.ndig>10) break;
            lbls = []; indx = 0; continue;
          }
         lbls.push(lbl);
         indx++;
      }

      var exclude_last = false;

      if ((ticks[ticks.length-1] < polar.fRwrmax) && (this.zoom_rmin == this.zoom_rmax)) {
         ticks.push(polar.fRwrmax);
         exclude_last = true;
      }

      this.StartTextDrawing(polar.fRadialLabelFont, Math.round(polar.fRadialTextSize * this.szy * 2));

      for (var n=0;n<ticks.length;++n) {
         var rx = this.r(ticks[n]), ry = rx/this.szx*this.szy;
         this.draw_g.append("ellipse")
             .attr("cx",0)
             .attr("cy",0)
             .attr("rx",Math.round(rx))
             .attr("ry",Math.round(ry))
             .style("fill", "none")
             .call(this.lineatt.func);

         if ((n < ticks.length-1) || !exclude_last)
            this.DrawText({ align: 23, x: Math.round(rx), y: Math.round(polar.fRadialTextSize * this.szy * 0.5),
                            text: this.format(ticks[n]), color: this.get_color[polar.fRadialLabelColor], latex: 0 });

         if ((nminor>1) && ((n < ticks.length-1) || !exclude_last)) {
            var dr = (ticks[1] - ticks[0]) / nminor;
            for (var nn=1;nn<nminor;++nn) {
               var gridr = ticks[n] + dr*nn;
               if (gridr > this.scale_rmax) break;
               rx = this.r(gridr); ry = rx/this.szx*this.szy;
               this.draw_g.append("ellipse")
                   .attr("cx",0)
                   .attr("cy",0)
                   .attr("rx",Math.round(rx))
                   .attr("ry",Math.round(ry))
                   .style("fill", "none")
                   .call(this.gridatt.func);
            }
         }
      }

      this.FinishTextDrawing();

      var fontsize = Math.round(polar.fPolarTextSize * this.szy * 2);
      this.StartTextDrawing(polar.fPolarLabelFont, fontsize);

      var nmajor = polar.fNdivPol % 100;
      if ((nmajor !== 8) && (nmajor !== 3)) nmajor = 8;

      var lbls = (nmajor==8) ? ["0", "#frac{#pi}{4}", "#frac{#pi}{2}", "#frac{3#pi}{4}", "#pi", "#frac{5#pi}{4}", "#frac{3#pi}{2}", "#frac{7#pi}{4}"] : ["0", "#frac{2#pi}{3}", "#frac{4#pi}{3}"],
          aligns = [12, 11, 21, 31, 32, 33, 23, 13 ];

      for (var n=0;n<nmajor;++n) {
         var angle = -n*2*Math.PI/nmajor - this.angle;
         this.draw_g.append("line")
             .attr("x1",0)
             .attr("y1",0)
             .attr("x2", Math.round(this.szx*Math.cos(angle)))
             .attr("y2", Math.round(this.szy*Math.sin(angle)))
             .call(this.lineatt.func);

         var aindx = Math.round(16 -angle/Math.PI*4) % 8; // index in align table, here absolute angle is important

         this.DrawText({ align: aligns[aindx],
                         x: Math.round((this.szx+fontsize)*Math.cos(angle)),
                         y: Math.round((this.szy + fontsize/this.szx*this.szy)*(Math.sin(angle))),
                         text: lbls[n],
                          color: this.get_color[polar.fPolarLabelColor], latex: 1 });
      }

      this.FinishTextDrawing();

      var nminor = Math.floor((polar.fNdivPol % 10000) / 100);

      if (nminor > 1)
         for (var n=0;n<nmajor*nminor;++n) {
            if (n % nminor === 0) continue;
            var angle = -n*2*Math.PI/nmajor/nminor - this.angle;
            this.draw_g.append("line")
                .attr("x1",0)
                .attr("y1",0)
                .attr("x2", Math.round(this.szx*Math.cos(angle)))
                .attr("y2", Math.round(this.szy*Math.sin(angle)))
                .call(this.gridatt.func);
         }


      if (JSROOT.BatchMode) return;

      var layer = this.svg_layer("primitives_layer"),
          interactive = layer.select(".interactive_ellipse");

      if (interactive.empty())
         interactive = layer.append("g")
                            .classed("most_upper_primitives", true)
                            .append("ellipse")
                            .classed("interactive_ellipse", true)
                            .attr("cx",0)
                            .attr("cy",0)
                            .style("fill", "none")
                            .style("pointer-events","visibleFill")
                            .on('mouseenter', this.MouseEvent.bind(this,'enter'))
                            .on('mousemove', this.MouseEvent.bind(this,'move'))
                            .on('mouseleave', this.MouseEvent.bind(this,'leave'));

      interactive.attr("rx", this.szx).attr("ry", this.szy);

      d3.select(interactive.node().parentNode).attr("transform", this.draw_g.attr("transform"));

      if (JSROOT.gStyle.Zooming && JSROOT.gStyle.ZoomWheel)
         interactive.on("wheel", this.MouseWheel.bind(this));
   }

   function drawGraphPolargram(divid, polargram, opt) {

      var painter = new TGraphPolargramPainter(polargram);

      painter.SetDivId(divid, -1); // just to get access to existing elements

      var main = painter.main_painter();

      if (main) {
         if (main.GetObject() !== polargram)
             console.error('Cannot superimpose TGraphPolargram with any other drawings');
          return null;
      }

      painter.SetDivId(divid, 4); // main object without need of frame
      painter.Redraw();
      return painter.DrawingReady();
   }

   // ==============================================================

   function TGraphPolarPainter(graph) {
      JSROOT.TObjectPainter.call(this, graph);
   }

   TGraphPolarPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TGraphPolarPainter.prototype.Redraw = function() {
      this.DrawBins();
   }

   TGraphPolarPainter.prototype.DecodeOptions = function(opt) {

      var d = new JSROOT.DrawOptions(opt || "L");

      if (!this.options) this.options = {};

      JSROOT.extend(this.options, {
          mark: d.check("P"),
          err: d.check("E"),
          fill: d.check("F"),
          line: d.check("L"),
          curve: d.check("C")
      });

      this.OptionsStore(opt);
   }

   TGraphPolarPainter.prototype.DrawBins = function() {
      var graph = this.GetObject(),
          main = this.main_painter();

      if (!graph || !main || !main.$polargram) return;

      if (this.options.mark) this.createAttMarker({ attr: graph });
      if (this.options.err || this.options.line || this.options.curve) this.createAttLine({ attr: graph });
      if (this.options.fill) this.createAttFill({ attr: graph });

      this.CreateG();

      this.draw_g.attr("transform", main.draw_g.attr("transform"));

      var mpath = "", epath = "", lpath = "", bins = [];

      for (var n=0;n<graph.fNpoints;++n) {

         if (graph.fY[n] > main.scale_rmax) continue;

         if (this.options.err) {
            var pos1 = main.translate(graph.fX[n], graph.fY[n] - graph.fEY[n]),
                pos2 = main.translate(graph.fX[n], graph.fY[n] + graph.fEY[n]);
            epath += "M" + pos1.x + "," + pos1.y + "L" + pos2.x + "," + pos2.y;

            pos1 = main.translate(graph.fX[n] + graph.fEX[n], graph.fY[n]);
            pos2 = main.translate(graph.fX[n] - graph.fEX[n], graph.fY[n]);

            epath += "M" + pos1.x + "," + pos1.y + "A" + pos2.rx + "," + pos2.ry+ ",0,0,1," + pos2.x + "," + pos2.y;
         }

         var pos = main.translate(graph.fX[n], graph.fY[n]);

         if (this.options.mark) {
            mpath += this.markeratt.create(pos.x, pos.y);
         }

         if (this.options.line || this.options.fill) {
            lpath += (lpath ? "L" : "M") + pos.x + "," + pos.y;
         }

         if (this.options.curve) {
            pos.grx = pos.x;
            pos.gry = pos.y;
            bins.push(pos);
         }
      }

      if (this.options.fill && lpath)
         this.draw_g.append("svg:path")
             .attr("d",lpath + "Z")
             .style("stroke","none")
             .call(this.fillatt.func);

      if (this.options.line && lpath)
         this.draw_g.append("svg:path")
             .attr("d", lpath)
             .style("fill", "none")
             .call(this.lineatt.func);

      if (this.options.curve && bins.length)
         this.draw_g.append("svg:path")
                 .attr("d", JSROOT.Painter.BuildSvgPath("bezier", bins).path)
                 .style("fill", "none")
                 .call(this.lineatt.func);

      if (epath)
         this.draw_g.append("svg:path")
             .attr("d",epath)
             .style("fill","none")
             .call(this.lineatt.func);

      if (mpath)
         this.draw_g.append("svg:path")
               .attr("d",mpath)
               .call(this.markeratt.func);

   }

   TGraphPolarPainter.prototype.CreatePolargram = function() {
      var polargram = JSROOT.Create("TGraphPolargram"),
          gr = this.GetObject();

      var rmin = gr.fY[0] || 0, rmax = rmin;
      for (var n=0;n<gr.fNpoints;++n) {
         rmin = Math.min(rmin, gr.fY[n] - gr.fEY[n]);
         rmax = Math.max(rmax, gr.fY[n] + gr.fEY[n]);
      }

      polargram.fRwrmin = rmin - (rmax-rmin)*0.1;
      polargram.fRwrmax = rmax + (rmax-rmin)*0.1;

      return polargram;
   }

   TGraphPolarPainter.prototype.ExtractTooltip = function(pnt) {
      if (!pnt) return null;

      var graph = this.GetObject(),
          main = this.main_painter(),
          best_dist2 = 1e10, bestindx = -1, bestpos = null;

      for (var n=0;n<graph.fNpoints;++n) {
         var pos = main.translate(graph.fX[n], graph.fY[n]);

         var dist2 = (pos.x-pnt.x)*(pos.x-pnt.x) + (pos.y-pnt.y)*(pos.y-pnt.y);
         if (dist2<best_dist2) { best_dist2 = dist2; bestindx = n; bestpos = pos; }
      }

      var match_distance = 5;
      if (this.markeratt && this.markeratt.used) match_distance = this.markeratt.GetFullSize();

      if (Math.sqrt(best_dist2) > match_distance) return null;

      var res = { name: this.GetObject().fName, title: this.GetObject().fTitle,
                  x: bestpos.x, y: bestpos.y,
                  color1: this.markeratt && this.markeratt.used ? this.markeratt.color : this.lineatt.color,
                  exact: Math.sqrt(best_dist2) < 4,
                  lines: [ this.GetTipName() ],
                  binindx: bestindx,
                  menu_dist: match_distance,
                  radius: match_distance
                };

      res.lines.push("r = " + main.AxisAsText("r", graph.fY[bestindx]));
      res.lines.push("phi = " + main.AxisAsText("phi",graph.fX[bestindx]));

      if (graph.fEY && graph.fEY[bestindx])
         res.lines.push("error r = " + main.AxisAsText("r", graph.fEY[bestindx]));

      if (graph.fEX && graph.fEX[bestindx])
         res.lines.push("error phi = " + main.AxisAsText("phi", graph.fEX[bestindx]));

      return res;
   }

   TGraphPolarPainter.prototype.ShowTooltip = function(hint) {

      if (!this.draw_g) return;

      var ttcircle = this.draw_g.select(".tooltip_bin");

      if (!hint) {
         ttcircle.remove();
         return;
      }

      if (ttcircle.empty())
         ttcircle = this.draw_g.append("svg:ellipse")
                             .attr("class","tooltip_bin")
                             .style("pointer-events","none");

      hint.changed = ttcircle.property("current_bin") !== hint.binindx;

      if (hint.changed)
         ttcircle.attr("cx", hint.x)
               .attr("cy", hint.y)
               .attr("rx", Math.round(hint.radius))
               .attr("ry", Math.round(hint.radius))
               .style("fill", "none")
               .style("stroke", hint.color1)
               .property("current_bin", hint.binindx);
   }

   TGraphPolarPainter.prototype.ProcessTooltip = function(pnt) {
      var hint = this.ExtractTooltip(pnt);
      if (!pnt || !pnt.disabled) this.ShowTooltip(hint);
      return hint;
   }

   TGraphPolarPainter.prototype.PerformDrawing = function(divid) {
      this.SetDivId(divid);
      this.DrawBins();
      this.DrawingReady();
   }

   function drawGraphPolar(divid, graph, opt) {

      var painter = new TGraphPolarPainter(graph);

      painter.DecodeOptions(opt);

      painter.SetDivId(divid, -1); // just to get access to existing elements

      var main = painter.main_painter();
      if (main) {
         if (!main.$polargram) {
            console.error('Cannot superimpose TGraphPolar with plain histograms');
            return null;
         }
         painter.PerformDrawing(divid);

         return painter;
      }

      if (!graph.fPolargram) graph.fPolargram = painter.CreatePolargram();

      return JSROOT.draw(divid, graph.fPolargram, "", painter.PerformDrawing.bind(painter, divid));
   }

   // ==============================================================

   function TSplinePainter(spline) {
      JSROOT.TObjectPainter.call(this, spline);
      this.bins = null;
   }

   TSplinePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TSplinePainter.prototype.UpdateObject = function(obj, opt) {
      var spline = this.GetObject();

      if (spline._typename != obj._typename) return false;

      if (spline !== obj) JSROOT.extend(spline, obj);

      if (opt !== undefined) this.DecodeOptions(opt);

      return true;
   }

   TSplinePainter.prototype.Eval = function(knot, x) {
      var dx = x - knot.fX;

      if (knot._typename == "TSplinePoly3")
         return knot.fY + dx*(knot.fB + dx*(knot.fC + dx*knot.fD));

      if (knot._typename == "TSplinePoly5")
         return knot.fY + dx*(knot.fB + dx*(knot.fC + dx*(knot.fD + dx*(knot.fE + dx*knot.fF))));

      return knot.fY + dx;
   }

   TSplinePainter.prototype.FindX = function(x) {
      var spline = this.GetObject(),
          klow = 0, khig = spline.fNp - 1;

      if (x <= spline.fXmin) return 0;
      if (x >= spline.fXmax) return khig;

      if(spline.fKstep) {
         // Equidistant knots, use histogramming
         klow = Math.round((x - spline.fXmin)/spline.fDelta);
         // Correction for rounding errors
         if (x < spline.fPoly[klow].fX) {
            klow = Math.max(klow-1,0);
         } else if (klow < khig) {
            if (x > spline.fPoly[klow+1].fX) ++klow;
         }
      } else {
         // Non equidistant knots, binary search
         while(khig-klow>1) {
            var khalf = Math.round((klow+khig)/2);
            if(x > spline.fPoly[khalf].fX) klow = khalf;
                                      else khig = khalf;
         }
      }
      return klow;
   }

   TSplinePainter.prototype.CreateDummyHisto = function() {

      var xmin = 0, xmax = 1, ymin = 0, ymax = 1,
          spline = this.GetObject();

      if (spline && spline.fPoly) {

         xmin = xmax = spline.fPoly[0].fX;
         ymin = ymax = spline.fPoly[0].fY;

         spline.fPoly.forEach(function(knot) {
            xmin = Math.min(knot.fX, xmin);
            xmax = Math.max(knot.fX, xmax);
            ymin = Math.min(knot.fY, ymin);
            ymax = Math.max(knot.fY, ymax);
         });

         if (ymax > 0.0) ymax *= 1.05;
         if (ymin < 0.0) ymin *= 1.05;
      }

      var histo = JSROOT.Create("TH1I");

      histo.fName = spline.fName + "_hist";
      histo.fTitle = spline.fTitle;

      histo.fXaxis.fXmin = xmin;
      histo.fXaxis.fXmax = xmax;
      histo.fYaxis.fXmin = ymin;
      histo.fYaxis.fXmax = ymax;

      return histo;
   }

   TSplinePainter.prototype.ProcessTooltip = function(pnt) {

      var cleanup = false,
          spline = this.GetObject(),
          main = this.frame_painter(),
          xx, yy, knot = null, indx = 0;

      if ((pnt === null) || !spline || !main) {
         cleanup = true;
      } else {
         xx = main.RevertX(pnt.x);
         indx = this.FindX(xx);
         knot = spline.fPoly[indx];
         yy = this.Eval(knot, xx);

         if ((indx < spline.fN-1) && (Math.abs(spline.fPoly[indx+1].fX-xx) < Math.abs(xx-knot.fX))) knot = spline.fPoly[++indx];

         if (Math.abs(main.grx(knot.fX) - pnt.x) < 0.5*this.knot_size) {
            xx = knot.fX; yy = knot.fY;
         } else {
            knot = null;
            if ((xx < spline.fXmin) || (xx > spline.fXmax)) cleanup = true;
         }
      }

      if (cleanup) {
         if (this.draw_g !== null)
            this.draw_g.select(".tooltip_bin").remove();
         return null;
      }

      var gbin = this.draw_g.select(".tooltip_bin"),
          radius = this.lineatt.width + 3;

      if (gbin.empty())
         gbin = this.draw_g.append("svg:circle")
                           .attr("class", "tooltip_bin")
                           .style("pointer-events","none")
                           .attr("r", radius)
                           .style("fill", "none")
                           .call(this.lineatt.func);

      var res = { name: this.GetObject().fName,
                  title: this.GetObject().fTitle,
                  x: main.grx(xx),
                  y: main.gry(yy),
                  color1: this.lineatt.color,
                  lines: [],
                  exact: (knot !== null) || (Math.abs(main.gry(yy) - pnt.y) < radius) };

      res.changed = gbin.property("current_xx") !== xx;
      res.menu = res.exact;
      res.menu_dist = Math.sqrt((res.x-pnt.x)*(res.x-pnt.x) + (res.y-pnt.y)*(res.y-pnt.y));

      if (res.changed)
         gbin.attr("cx", Math.round(res.x))
             .attr("cy", Math.round(res.y))
             .property("current_xx", xx);

      var name = this.GetTipName();
      if (name.length > 0) res.lines.push(name);
      res.lines.push("x = " + main.AxisAsText("x", xx))
      res.lines.push("y = " + main.AxisAsText("y", yy));
      if (knot !== null) {
         res.lines.push("knot = " + indx);
         res.lines.push("B = " + JSROOT.FFormat(knot.fB, JSROOT.gStyle.fStatFormat));
         res.lines.push("C = " + JSROOT.FFormat(knot.fC, JSROOT.gStyle.fStatFormat));
         res.lines.push("D = " + JSROOT.FFormat(knot.fD, JSROOT.gStyle.fStatFormat));
         if ((knot.fE!==undefined) && (knot.fF!==undefined)) {
            res.lines.push("E = " + JSROOT.FFormat(knot.fE, JSROOT.gStyle.fStatFormat));
            res.lines.push("F = " + JSROOT.FFormat(knot.fF, JSROOT.gStyle.fStatFormat));
         }
      }

      return res;
   }

   TSplinePainter.prototype.Redraw = function() {

      var w = this.frame_width(),
          h = this.frame_height(),
          spline = this.GetObject(),
          pmain = this.frame_painter(),
          name = this.GetTipName("\n");

      this.CreateG(true);

      this.knot_size = 5; // used in tooltip handling

      this.createAttLine({ attr: spline });

      if (this.options.Line || this.options.Curve) {

         var npx = Math.max(10, spline.fNpx);

         var xmin = Math.max(pmain.scale_xmin, spline.fXmin),
             xmax = Math.min(pmain.scale_xmax, spline.fXmax),
             indx = this.FindX(xmin),
             bins = []; // index of current knot

         if (pmain.logx) {
            xmin = Math.log(xmin);
            xmax = Math.log(xmax);
         }

         for (var n=0;n<npx;++n) {
            var xx = xmin + (xmax-xmin)/npx*(n-1);
            if (pmain.logx) xx = Math.exp(xx);

            while ((indx < spline.fNp-1) && (xx > spline.fPoly[indx+1].fX)) ++indx;

            var yy = this.Eval(spline.fPoly[indx], xx);

            bins.push({ x: xx, y: yy, grx: pmain.grx(xx), gry: pmain.gry(yy) });
         }

         var h0 = h;  // use maximal frame height for filling
         if ((pmain.hmin!==undefined) && (pmain.hmin>=0)) {
            h0 = Math.round(pmain.gry(0));
            if ((h0 > h) || (h0 < 0)) h0 = h;
         }

         var path = JSROOT.Painter.BuildSvgPath("bezier", bins, h0, 2);

         this.draw_g.append("svg:path")
             .attr("class", "line")
             .attr("d", path.path)
             .style("fill", "none")
             .call(this.lineatt.func);
      }

      if (this.options.Mark) {

         // for tooltips use markers only if nodes where not created
         var path = "";

         this.createAttMarker({ attr: spline })

         this.markeratt.reset_pos();

         this.knot_size = this.markeratt.GetFullSize();

         for (var n=0; n<spline.fPoly.length; n++) {
            var knot = spline.fPoly[n],
                grx = pmain.grx(knot.fX);
            if ((grx > -this.knot_size) && (grx < w + this.knot_size)) {
               var gry = pmain.gry(knot.fY);
               if ((gry > -this.knot_size) && (gry < h + this.knot_size)) {
                  path += this.markeratt.create(grx, gry);
               }
            }
         }

         if (path)
            this.draw_g.append("svg:path")
                       .attr("d", path)
                       .call(this.markeratt.func);
      }

   }

   TSplinePainter.prototype.CanZoomIn = function(axis,min,max) {
      if (axis!=="x") return false;

      var spline = this.GetObject();
      if (!spline) return false;

      // if function calculated, one always could zoom inside
      return true;
   }

   TSplinePainter.prototype.DecodeOptions = function(opt) {
      var d = new JSROOT.DrawOptions(opt);

      if (!this.options) this.options = {};

      JSROOT.extend(this.options, {
         Same: d.check('SAME'),
         Line: d.check('L'),
         Curve: d.check('C'),
         Mark: d.check('P')
      });

      this.OptionsStore(opt);
   }

   TSplinePainter.prototype.FirstDraw = function() {
      this.SetDivId(this.divid);
      this.Redraw();
      return this.DrawingReady();
   }

   JSROOT.Painter.drawSpline = function(divid, spline, opt) {

      var painter = new TSplinePainter(spline);

      painter.SetDivId(divid, -1);
      painter.DecodeOptions(opt);

      if (!painter.main_painter()) {
         if (painter.options.Same) {
            console.warn('TSpline painter requires histogram to be drawn');
            return null;
         }
         var histo = painter.CreateDummyHisto();
         JSROOT.draw(divid, histo, "AXIS", painter.FirstDraw.bind(painter));
         return painter;
      }

      return painter.FirstDraw();
   }

   // =============================================================

   function TGraphTimePainter(gr) {
      JSROOT.TObjectPainter.call(this, gr);
   }

   TGraphTimePainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TGraphTimePainter.prototype.Redraw = function() {
      if (this.step === undefined) this.StartDrawing(false);
   }

   TGraphTimePainter.prototype.DecodeOptions = function(opt) {

      var d = new JSROOT.DrawOptions(opt || "REPEAT");

      if (!this.options) this.options = {};

      JSROOT.extend(this.options, {
          once: d.check("ONCE"),
          repeat: d.check("REPEAT"),
          first: d.check("FIRST")
      });

      this.OptionsStore(opt);
   }

   TGraphTimePainter.prototype.DrawPrimitives = function(indx, callback, ppainter) {

      if (indx===0) {
         this._doing_primitives = true;
      }

      var lst = this.GetObject().fSteps.arr[this.step];

      while (true) {
         if (ppainter) ppainter.$grtimeid = this.selfid; // indicator that painter created by ourself

         if (!lst || (indx >= lst.arr.length)) {
            delete this._doing_primitives;
            return JSROOT.CallBack(callback);
         }

         // handle use to invoke callback only when necessary
         var handle = { func: this.DrawPrimitives.bind(this, indx+1, callback) };

         ppainter = JSROOT.draw(this.divid, lst.arr[indx], lst.opt[indx], handle);

         if (!handle.completed) return;
         indx++;
      }
   }

   TGraphTimePainter.prototype.Selector = function(p) {
      return p && (p.$grtimeid === this.selfid);
   }

   TGraphTimePainter.prototype.ContineDrawing = function() {
      if (!this.options) return;

      var gr = this.GetObject();

      if (!this.ready_called) {
         this.ready_called = true;
         this.DrawingReady(); // do it already here, animation will continue in background
      }

      if (this.options.first) {
         // draw only single frame, cancel all others
         delete this.step;
         return;
      }

      if (this.wait_animation_frame) {
         delete this.wait_animation_frame;

         // clear pad
         var pp = this.pad_painter();
         if (!pp) {
            // most probably, pad is cleared
            delete this.step;
            return;
         }

         // clear primitives produced by the TGraphTime
         pp.CleanPrimitives(this.Selector.bind(this));

         // draw ptrimitives again
         this.DrawPrimitives(0, this.ContineDrawing.bind(this));
      } else if (this.running_timeout) {
         clearTimeout(this.running_timeout);
         delete this.running_timeout;

         this.wait_animation_frame = true;
         // use animation frame to disable update in inactive form
         requestAnimationFrame(this.ContineDrawing.bind(this));
      } else {

         var sleeptime = gr.fSleepTime;
         if (!sleeptime || (sleeptime<100)) sleeptime = 10;

         if (++this.step > gr.fSteps.arr.length) {
            if (this.options.repeat) {
               this.step = 0; // start again
               sleeptime = Math.max(5000, 5*sleeptime); // increase sleep time
            } else {
               delete this.step;    // clear indicator that animation running
               return;
            }
         }

         this.running_timeout = setTimeout(this.ContineDrawing.bind(this), sleeptime);
      }
   }

   TGraphTimePainter.prototype.StartDrawing = function(once_again) {
      if (once_again!==false) this.SetDivId(this.divid);

      this.step = 0;

      this.DrawPrimitives(0, this.ContineDrawing.bind(this));
   }

   JSROOT.Painter.drawGraphTime = function(divid,gr,opt) {

      var painter = new TGraphTimePainter(gr);
      painter.SetDivId(divid,-1);

      if (painter.main_painter()) {
         console.error('Cannot draw graph time on top of other histograms');
         return null;
      }

      if (!gr.fFrame) {
         console.error('Frame histogram not exists');
         return null;
      }

      painter.DecodeOptions(opt);

      if (!gr.fFrame.fTitle && gr.fTitle) gr.fFrame.fTitle = gr.fTitle;

      painter.selfid = "grtime" + JSROOT.id_counter++; // use to identify primitives which should be clean

      JSROOT.draw(divid, gr.fFrame, "AXIS", painter.StartDrawing.bind(painter));

      return painter;

   }

   // =============================================================

   function TEfficiencyPainter(eff) {
      JSROOT.TObjectPainter.call(this, eff);
      this.fBoundary = 'Normal';
   }

   TEfficiencyPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TEfficiencyPainter.prototype.GetEfficiency = function(bin) {
      var obj = this.GetObject(),
          total = obj.fTotalHistogram.getBinContent(bin),
          passed = obj.fPassedHistogram.getBinContent(bin);

      return total ? passed/total : 0;
   }

/**  implementing of  beta_quantile requires huge number of functions in JSRootMath.js

   TEfficiencyPainter.prototype.ClopperPearson = function(total,passed,level,bUpper) {
      var alpha = (1.0 - level) / 2;
      if(bUpper)
         return ((passed == total) ? 1.0 : JSROOT.Math.beta_quantile(1 - alpha,passed + 1,total-passed));
      else
         return ((passed == 0) ? 0.0 : JSROOT.Math.beta_quantile(alpha,passed,total-passed+1.0));
   }
*/

   TEfficiencyPainter.prototype.Normal = function(total,passed,level,bUpper) {
      if (total == 0) return bUpper ? 1 : 0;

      var alpha = (1.0 - level)/2,
          average = passed / total,
          sigma = Math.sqrt(average * (1 - average) / total),
         delta = JSROOT.Math.normal_quantile(1 - alpha,sigma);

      if(bUpper)
         return ((average + delta) > 1) ? 1.0 : (average + delta);

      return ((average - delta) < 0) ? 0.0 : (average - delta);
   }

   TEfficiencyPainter.prototype.GetEfficiencyErrorLow = function(bin) {
      var obj = this.GetObject(),
          total = obj.fTotalHistogram.getBinContent(bin),
          passed = obj.fPassedHistogram.getBinContent(bin),
          eff = this.GetEfficiency(bin);

      return eff - this[this.fBoundary](total,passed, obj.fConfLevel, false);
   }

   TEfficiencyPainter.prototype.GetEfficiencyErrorUp = function(bin) {
      var obj = this.GetObject(),
          total = obj.fTotalHistogram.getBinContent(bin),
          passed = obj.fPassedHistogram.getBinContent(bin),
          eff = this.GetEfficiency(bin);

      return this[this.fBoundary]( total, passed, obj.fConfLevel, true) - eff;
   }

   TEfficiencyPainter.prototype.CreateGraph = function() {
      var gr = JSROOT.Create('TGraphAsymmErrors');
      gr.fName = "eff_graph";
      return gr;
   }

   TEfficiencyPainter.prototype.FillGraph = function(gr, opt) {
      var eff = this.GetObject(),
          npoints = eff.fTotalHistogram.fXaxis.fNbins,
          option = opt.toLowerCase(),
          plot0Bins = false, j = 0;
      if (option.indexOf("e0")>=0) plot0Bins = true;
      for (var n=0;n<npoints;++n) {
         if (!plot0Bins && eff.fTotalHistogram.getBinContent(n+1) === 0) continue;
         gr.fX[j] = eff.fTotalHistogram.fXaxis.GetBinCenter(n+1);
         gr.fY[j] = this.GetEfficiency(n+1);
         gr.fEXlow[j] = eff.fTotalHistogram.fXaxis.GetBinCenter(n+1) - eff.fTotalHistogram.fXaxis.GetBinLowEdge(n+1);
         gr.fEXhigh[j] = eff.fTotalHistogram.fXaxis.GetBinLowEdge(n+2) - eff.fTotalHistogram.fXaxis.GetBinCenter(n+1);
         gr.fEYlow[j] = this.GetEfficiencyErrorLow(n+1);
         gr.fEYhigh[j] = this.GetEfficiencyErrorUp(n+1);
         ++j;
      }
      gr.fNpoints = j;
   }

   JSROOT.Painter.drawEfficiency = function(divid, eff, opt) {

      if (!eff || !eff.fTotalHistogram || (eff.fTotalHistogram._typename.indexOf("TH1")!=0)) return null;

      var painter = new TEfficiencyPainter(eff);
      painter.options = opt;

      var gr = painter.CreateGraph();
      painter.FillGraph(gr, opt);

      JSROOT.draw(divid, gr, opt, function() {
         painter.SetDivId(divid);
         painter.DrawingReady();
      });

      return painter;
   }


   // =============================================================

   function TMultiGraphPainter(mgraph) {
      JSROOT.TObjectPainter.call(this, mgraph);
      this.firstpainter = null;
      this.autorange = false;
      this.painters = []; // keep painters to be able update objects
   }

   TMultiGraphPainter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TMultiGraphPainter.prototype.Cleanup = function() {
      this.painters = [];
      JSROOT.TObjectPainter.prototype.Cleanup.call(this);
   }

   TMultiGraphPainter.prototype.UpdateObject = function(obj) {
      if (!this.MatchObjectType(obj)) return false;

      var mgraph = this.GetObject(),
          graphs = obj.fGraphs;

      mgraph.fTitle = obj.fTitle;

      var isany = false;
      if (this.firstpainter) {
         var histo = obj.fHistogram;
         if (this.autorange && !histo)
            histo = this.ScanGraphsRange(graphs);

         if (this.firstpainter.UpdateObject(histo)) isany = true;
      }

      for (var i = 0; i < graphs.arr.length; ++i) {
         if (i<this.painters.length)
            if (this.painters[i].UpdateObject(graphs.arr[i])) isany = true;
      }

      if (obj.fFunctions)
         for (var i = 0; i < obj.fFunctions.arr.length; ++i) {
            var func = obj.fFunctions.arr[i];
            if (!func || !func._typename || !func.fName) continue;
            var funcpainter = this.FindPainterFor(null, func.fName, func._typename);
            if (funcpainter) funcpainter.UpdateObject(func);
         }

      return isany;
   }

   TMultiGraphPainter.prototype.ComputeGraphRange = function(res, gr) {
      // Compute the x/y range of the points in this graph
      if (gr.fNpoints == 0) return;
      if (res.first) {
         res.xmin = res.xmax = gr.fX[0];
         res.ymin = res.ymax = gr.fY[0];
         res.first = false;
      }
      for (var i=0; i < gr.fNpoints; ++i) {
         res.xmin = Math.min(res.xmin, gr.fX[i]);
         res.xmax = Math.max(res.xmax, gr.fX[i]);
         res.ymin = Math.min(res.ymin, gr.fY[i]);
         res.ymax = Math.max(res.ymax, gr.fY[i]);
      }
      return res;
   }

   TMultiGraphPainter.prototype.padtoX = function(pad, x) {
      // Convert x from pad to X.
      if (pad.fLogx && (x < 50))
         return Math.exp(2.302585092994 * x);
      return x;
   }

   TMultiGraphPainter.prototype.ScanGraphsRange = function(graphs, histo, pad) {
      var mgraph = this.GetObject(),
          maximum, minimum, dx, dy, uxmin = 0, uxmax = 0, logx = false, logy = false,
          time_display = false, time_format = "",
          rw = {  xmin: 0, xmax: 0, ymin: 0, ymax: 0, first: true };

      if (pad) {
         logx = pad.fLogx;
         logy = pad.fLogy;
         rw.xmin = pad.fUxmin;
         rw.xmax = pad.fUxmax;
         rw.ymin = pad.fUymin;
         rw.ymax = pad.fUymax;
         rw.first = false;
      }
      if (histo) {
         minimum = histo.fYaxis.fXmin;
         maximum = histo.fYaxis.fXmax;
         if (pad) {
            uxmin = this.padtoX(pad, rw.xmin);
            uxmax = this.padtoX(pad, rw.xmax);
         }
      } else {
         this.autorange = true;

         for (var i = 0; i < graphs.arr.length; ++i)
            this.ComputeGraphRange(rw, graphs.arr[i]);

         if (graphs.arr[0] && graphs.arr[0].fHistogram && graphs.arr[0].fHistogram.fXaxis.fTimeDisplay) {
            time_display = true;
            time_format = graphs.arr[0].fHistogram.fXaxis.fTimeFormat;
         }

         if (rw.xmin == rw.xmax) rw.xmax += 1.;
         if (rw.ymin == rw.ymax) rw.ymax += 1.;
         dx = 0.05 * (rw.xmax - rw.xmin);
         dy = 0.05 * (rw.ymax - rw.ymin);
         uxmin = rw.xmin - dx;
         uxmax = rw.xmax + dx;
         if (logy) {
            if (rw.ymin <= 0) rw.ymin = 0.001 * rw.ymax;
            minimum = rw.ymin / (1 + 0.5 * JSROOT.log10(rw.ymax / rw.ymin));
            maximum = rw.ymax * (1 + 0.2 * JSROOT.log10(rw.ymax / rw.ymin));
         } else {
            minimum = rw.ymin - dy;
            maximum = rw.ymax + dy;
         }
         if (minimum < 0 && rw.ymin >= 0)
            minimum = 0;
         if (maximum > 0 && rw.ymax <= 0)
            maximum = 0;
      }

      if (uxmin < 0 && rw.xmin >= 0)
         uxmin = logx ? 0.9 * rw.xmin : 0;
      if (uxmax > 0 && rw.xmax <= 0)
         uxmax = logx? 1.1 * rw.xmax : 0;

      if (mgraph.fMinimum != -1111)
         rw.ymin = minimum = mgraph.fMinimum;
      if (mgraph.fMaximum != -1111)
         rw.ymax = maximum = mgraph.fMaximum;

      if (minimum < 0 && rw.ymin >= 0 && logy) minimum = 0.9 * rw.ymin;
      if (maximum > 0 && rw.ymax <= 0 && logy) maximum = 1.1 * rw.ymax;
      if (minimum <= 0 && logy) minimum = 0.001 * maximum;
      if (!logy && minimum > 0 && minimum < 0.05*maximum) minimum = 0;
      if (uxmin <= 0 && logx)
         uxmin = (uxmax > 1000) ? 1 : 0.001 * uxmax;

      // Create a temporary histogram to draw the axis (if necessary)
      if (!histo) {
         histo = JSROOT.Create("TH1I");
         histo.fTitle = mgraph.fTitle;
         histo.fXaxis.fXmin = uxmin;
         histo.fXaxis.fXmax = uxmax;
         histo.fXaxis.fTimeDisplay = time_display;
         if (time_display) histo.fXaxis.fTimeFormat = time_format;
     }

      histo.fYaxis.fXmin = minimum;
      histo.fYaxis.fXmax = maximum;

      return histo;
   }

   TMultiGraphPainter.prototype.DrawAxis = function(callback) {
      // draw special histogram

      var mgraph = this.GetObject(),
          histo = this.ScanGraphsRange(mgraph.fGraphs, mgraph.fHistogram, this.root_pad());

      // histogram painter will be first in the pad, will define axis and
      // interactive actions
      JSROOT.draw(this.divid, histo, "AXIS", callback);
   }

   TMultiGraphPainter.prototype.DrawNextFunction = function(indx, callback) {
      // method draws next function from the functions list

      var mgraph = this.GetObject();

      if (!mgraph.fFunctions || (indx >= mgraph.fFunctions.arr.length))
         return JSROOT.CallBack(callback);

      JSROOT.draw(this.divid, mgraph.fFunctions.arr[indx], mgraph.fFunctions.opt[indx],
                  this.DrawNextFunction.bind(this, indx+1, callback));
   }

   TMultiGraphPainter.prototype.DrawNextGraph = function(indx, opt, subp, used_timeout) {
      if (subp) this.painters.push(subp);

      var graphs = this.GetObject().fGraphs;

      // at the end of graphs drawing draw functions (if any)
      if (indx >= graphs.arr.length) {
         this._pfc = this._plc = this._pmc = false; // disable auto coloring at the end
         return this.DrawNextFunction(0, this.DrawingReady.bind(this));
      }

      // when too many graphs are drawn, avoid deep stack with timeout
      if ((indx % 500 === 499) && !used_timeout)
         return setTimeout(this.DrawNextGraph.bind(this,indx,opt,null,true),0);

      // if there is auto colors assignment, try to provide it
      if (this._pfc || this._plc || this._pmc) {
         if (!this.pallette && JSROOT.Painter.GetColorPalette)
            this.palette = JSROOT.Painter.GetColorPalette();
         if (this.palette) {
            var color = this.palette.calcColor(indx, graphs.arr.length+1);
            var icolor = this.add_color(color);

            if (this._pfc) graphs.arr[indx].fFillColor = icolor;
            if (this._plc) graphs.arr[indx].fLineColor = icolor;
            if (this._pmc) graphs.arr[indx].fMarkerColor = icolor;
         }
      }

      JSROOT.draw(this.divid, graphs.arr[indx], graphs.opt[indx] || opt,
                  this.DrawNextGraph.bind(this, indx+1, opt));
   }

   JSROOT.Painter.drawMultiGraph = function(divid, mgraph, opt) {

      var painter = new TMultiGraphPainter(mgraph);

      painter.SetDivId(divid, -1); // it may be no element to set divid

      var d = new JSROOT.DrawOptions(opt);
      d.check("3D"); d.check("FB"); // no 3D supported, FB not clear

      painter._pfc = d.check("PFC");
      painter._plc = d.check("PLC");
      painter._pmc = d.check("PMC");

      if (d.check("A") || !painter.main_painter()) {
         painter.DrawAxis(function(hpainter) {
            painter.firstpainter = hpainter;
            painter.SetDivId(divid);
            painter.DrawNextGraph(0, d.remain());
         });
      } else {
         painter.SetDivId(divid);
         painter.DrawNextGraph(0, d.remain());
      }

      return painter;
   }

   // =========================================================================================

   function drawWebPainting(divid, obj, opt) {

      var painter = new JSROOT.TObjectPainter(obj, opt);

      painter.UpdateObject = function(obj) {
         if (!this.MatchObjectType(obj)) return false;
         this.draw_object = obj;
         return true;
      }

      painter.ReadAttr = function(str, names) {
         var lastp = 0, obj = { _typename: "any" };
         for (var k=0;k<names.length;++k) {
            var p = str.indexOf(":", lastp+1);
            obj[names[k]] = parseInt(str.substr(lastp+1, (p>lastp) ? p-lastp-1 : undefined));
            lastp = p;
         }
         return obj;
      }

      painter.Redraw = function() {

         var obj = this.GetObject(), func = this.AxisToSvgFunc(false);

         if (!obj || !obj.fOper || !func) return;

         var indx = 0, attr = {}, lastpath = null, lastkind = "none", d = "",
             oper, k, npoints, n, arr = obj.fOper.split(";");

         function check_attributes(painter, kind) {
            if (kind == lastkind) return;

            if (lastpath) {
               lastpath.attr("d", d); // flush previous
               d = ""; lastpath = null; lastkind = "none";
            }

            if (!kind) return;

            lastkind = kind;
            lastpath = painter.draw_g.append("svg:path");
            switch (kind) {
               case "f": lastpath.call(painter.fillatt.func).attr('stroke', 'none'); break;
               case "l": lastpath.call(painter.lineatt.func).attr('fill', 'none'); break;
               case "m": lastpath.call(painter.markeratt.func); break;
            }
         }

         this.CreateG();

         for (k=0;k<arr.length;++k) {
            oper = arr[k][0];
            switch (oper) {
               case "z":
                  this.createAttLine({ attr: this.ReadAttr(arr[k], ["fLineColor", "fLineStyle", "fLineWidth"]), force: true });
                  check_attributes();
                  continue;
               case "y":
                  this.createAttFill({ attr: this.ReadAttr(arr[k], ["fFillColor", "fFillStyle"]), force: true });
                  check_attributes();
                  continue;
               case "x":
                  this.createAttMarker({ attr: this.ReadAttr(arr[k], ["fMarkerColor", "fMarkerStyle", "fMarkerSize"]), force: true })
                  check_attributes();
                  continue;
               case "o":
                  attr = this.ReadAttr(arr[k], ["fTextColor", "fTextFont", "fTextSize", "fTextAlign", "fTextAngle" ]);
                  if (attr.fTextSize < 0) attr.fTextSize *= -0.001;
                  check_attributes();
                  continue;
               case "r":
               case "b": {

                  check_attributes(this, (oper == "b") ? "f" : "l");

                  var x1 = func.x(obj.fBuf[indx++]),
                      y1 = func.y(obj.fBuf[indx++]),
                      x2 = func.x(obj.fBuf[indx++]),
                      y2 = func.y(obj.fBuf[indx++]);

                  d += "M"+x1+","+y1+"h"+(x2-x1)+"v"+(y2-y1)+"h"+(x1-x2)+"z";

                  continue;
               }
               case "l":
               case "f": {

                  check_attributes(this, oper);

                  npoints = parseInt(arr[k].substr(1));

                  for (n=0;n<npoints;++n)
                     d += ((n>0) ? "L" : "M") +
                           func.x(obj.fBuf[indx++]) + "," + func.y(obj.fBuf[indx++]);

                  if (oper == "f") d+="Z";

                  continue;
               }

               case "m": {

                  check_attributes(this, oper);

                  npoints = parseInt(arr[k].substr(1));

                  this.markeratt.reset_pos();
                  for (n=0;n<npoints;++n)
                     d += this.markeratt.create(func.x(obj.fBuf[indx++]), func.y(obj.fBuf[indx++]));

                  continue;
               }

               case "h":
               case "t": {
                  if (attr.fTextSize) {

                     check_attributes();

                     var height = (attr.fTextSize > 1) ? attr.fTextSize : this.pad_height() * attr.fTextSize;

                     var group = this.draw_g.append("svg:g");

                     this.StartTextDrawing(attr.fTextFont, height, group);

                     var angle = attr.fTextAngle;
                     if (angle >= 360) angle -= Math.floor(angle/360) * 360;

                     var txt = arr[k].substr(1);

                     if (oper == "h") {
                        var res = "";
                        for (n=0;n<txt.length;n+=2)
                           res += String.fromCharCode(parseInt(txt.substr(n,2), 16));
                        txt = res;
                     }

                     // todo - correct support of angle
                     this.DrawText({ align: attr.fTextAlign,
                                     x: func.x(obj.fBuf[indx++]),
                                     y: func.y(obj.fBuf[indx++]),
                                     rotate: -angle,
                                     text: txt,
                                     color: JSROOT.Painter.root_colors[attr.fTextColor], latex: 0, draw_g: group });

                     this.FinishTextDrawing(group);
                  }
                  continue;
               }

               default:
                  console.log('unsupported operation ' + oper);
            }
         }

         check_attributes();
      }

      painter.SetDivId(divid);

      painter.Redraw();

      return painter.DrawingReady();
   }

   JSROOT.Painter.drawASImage = function(divid, obj, opt) {
      var painter = new JSROOT.TBasePainter();
      painter.SetDivId(divid, -1);

      var main = painter.select_main(); // this is d3 selection of main element for image drawing

      // from here one should insert PNG image

      // this is example how external image can be inserted
      // main.append("img").attr("src","https://root.cern/js/files/img/tf1.png");

      // this is potential example how image can be generated
      // one could use TASImage member like obj.fPngBuf
      // main.append("img").attr("src","data:image/png;base64,xxxxxxxxx..");

      painter.SetDivId(divid);

      return painter.DrawingReady();
   }

   JSROOT.Painter.drawJSImage = function(divid, obj, opt) {
      var painter = new JSROOT.TBasePainter();
      painter.SetDivId(divid, -1);

      var main = painter.select_main();

      // this is example how external image can be inserted
      var img = main.append("img").attr("src", obj.fName).attr("title", obj.fTitle || obj.fName);

      if (opt && opt.indexOf("scale")>=0) {
         img.style("width","100%").style("height","100%");
      } else if (opt && opt.indexOf("center")>=0) {
         main.style("position", "relative");
         img.attr("style", "margin: 0; position: absolute;  top: 50%; left: 50%; transform: translate(-50%, -50%);");
      }

      painter.SetDivId(divid);

      return painter.DrawingReady();
   }


   // ==================================================================================================

   JSROOT.Painter.CreateBranchItem = function(node, branch, tree, parent_branch) {
      if (!node || !branch) return false;

      var nb_branches = branch.fBranches ? branch.fBranches.arr.length : 0,
          nb_leaves = branch.fLeaves ? branch.fLeaves.arr.length : 0;

      function ClearName(arg) {
         var pos = arg.indexOf("[");
         if (pos>0) arg = arg.substr(0, pos);
         if (parent_branch && arg.indexOf(parent_branch.fName)==0) {
            arg = arg.substr(parent_branch.fName.length);
            if (arg[0]===".") arg = arg.substr(1);
         }
         return arg;
      }

      branch.$tree = tree; // keep tree pointer, later do it more smart

      var subitem = {
            _name : ClearName(branch.fName),
            _kind : "ROOT." + branch._typename,
            _title : branch.fTitle,
            _obj : branch
      };

      if (!node._childs) node._childs = [];

      node._childs.push(subitem);

      if (branch._typename==='TBranchElement')
         subitem._title += " from " + branch.fClassName + ";" + branch.fClassVersion;

      if (nb_branches > 0) {
         subitem._more = true;
         subitem._expand = function(bnode,bobj) {
            // really create all sub-branch items
            if (!bobj) return false;

            if (!bnode._childs) bnode._childs = [];

            if (bobj.fLeaves && (bobj.fLeaves.arr.length === 1) &&
                ((bobj.fType === JSROOT.BranchType.kClonesNode) || (bobj.fType === JSROOT.BranchType.kSTLNode))) {
                 bobj.fLeaves.arr[0].$branch = bobj;
                 bnode._childs.push({
                    _name: "@size",
                    _title: "container size",
                    _kind: "ROOT.TLeafElement",
                    _icon: "img_leaf",
                    _obj: bobj.fLeaves.arr[0],
                    _more : false
                 });
              }

            for (var i=0; i<bobj.fBranches.arr.length; ++i)
               JSROOT.Painter.CreateBranchItem(bnode, bobj.fBranches.arr[i], bobj.$tree, bobj);

            var object_class = JSROOT.IO.GetBranchObjectClass(bobj, bobj.$tree, true),
                methods = object_class ? JSROOT.getMethods(object_class) : null;

            if (methods && (bobj.fBranches.arr.length>0))
               for (var key in methods) {
                  if (typeof methods[key] !== 'function') continue;
                  var s = methods[key].toString();
                  if ((s.indexOf("return")>0) && (s.indexOf("function ()")==0))
                     bnode._childs.push({
                        _name: key+"()",
                        _title: "function " + key + " of class " + object_class,
                        _kind: "ROOT.TBranchFunc", // fictional class, only for drawing
                        _obj: { _typename: "TBranchFunc", branch: bobj, func: key },
                        _more : false
                     });

               }

            return true;
         }
         return true;
      } else if (nb_leaves === 1) {
         subitem._icon = "img_leaf";
         subitem._more = false;
      } else if (nb_leaves > 1) {
         subitem._childs = [];
         for (var j = 0; j < nb_leaves; ++j) {
            branch.fLeaves.arr[j].$branch = branch; // keep branch pointer for drawing
            var leafitem = {
               _name : ClearName(branch.fLeaves.arr[j].fName),
               _kind : "ROOT." + branch.fLeaves.arr[j]._typename,
               _obj: branch.fLeaves.arr[j]
            }
            subitem._childs.push(leafitem);
         }
      }

      return true;
   }

   JSROOT.Painter.TreeHierarchy = function(node, obj) {
      if (obj._typename != 'TTree' && obj._typename != 'TNtuple' && obj._typename != 'TNtupleD' ) return false;

      node._childs = [];
      node._tree = obj;  // set reference, will be used later by TTree::Draw

      for (var i=0; i<obj.fBranches.arr.length; ++i)
         JSROOT.Painter.CreateBranchItem(node, obj.fBranches.arr[i], obj);

      return true;
   }

   /** @summary function called from JSROOT.draw()
    * @desc just envelope for real TTree::Draw method which do the main job
    * Can be also used for the branch and leaf object
    * @private */
   JSROOT.Painter.drawTree = function(divid, obj, opt) {

      var painter = new JSROOT.TObjectPainter(obj),
          tree = obj, args = opt;

      if (obj._typename == "TBranchFunc") {
         // fictional object, created only in browser
         args = { expr: "." + obj.func + "()", branch: obj.branch };
         if (opt && opt.indexOf("dump")==0) args.expr += ">>" + opt; else
         if (opt) args.expr += opt;
         tree = obj.branch.$tree;
      } else if (obj.$branch) {
         // this is drawing of the single leaf from the branch
         args = { expr: "." + obj.fName + (opt || ""), branch: obj.$branch };
         if ((args.branch.fType === JSROOT.BranchType.kClonesNode) || (args.branch.fType === JSROOT.BranchType.kSTLNode)) {
            // special case of size
            args.expr = opt;
            args.direct_branch = true;
         }

         tree = obj.$branch.$tree;
      } else if (obj.$tree) {
         // this is drawing of the branch

         // if generic object tried to be drawn without specifying any options, it will be just dump
         if (!opt && obj.fStreamerType && (obj.fStreamerType !== JSROOT.IO.kTString) &&
             (obj.fStreamerType >= JSROOT.IO.kObject) && (obj.fStreamerType <= JSROOT.IO.kAnyP)) opt = "dump";

         args = { expr: opt, branch: obj };
         tree = obj.$tree;
      } else {

         if ((args==='player') || !args) {
            JSROOT.AssertPrerequisites("jq2d", function() {
               JSROOT.CreateTreePlayer(painter);
               painter.ConfigureTree(tree);
               painter.Show(divid);
               painter.DrawingReady();
            });
            return painter;
         }

         if (typeof args === 'string') args = { expr: args };
      }

      if (!tree) {
         console.error('No TTree object available for TTree::Draw');
         return painter.DrawingReady();
      }

      var callback = painter.DrawingReady.bind(painter);
      painter._return_res_painter = true; // indicate that TTree::Draw painter returns not itself but drawing of result object

      JSROOT.cleanup(divid);

      tree.Draw(args, function(histo, hopt, intermediate) {

         var drawid = "";

         if (!args.player) drawid = divid; else
         if (args.create_player === 2) drawid = painter.drawid;

         if (drawid)
            return JSROOT.redraw(drawid, histo, hopt, intermediate ? null : callback);

         if (args.create_player === 1) { args.player_intermediate = intermediate; return; }

         // redirect drawing to the player
         args.player_create = 1;
         args.player_intermediate = intermediate;
         JSROOT.AssertPrerequisites("jq2d", function() {
            JSROOT.CreateTreePlayer(painter);
            painter.ConfigureTree(tree);
            painter.Show(divid, args);
            args.create_player = 2;
            JSROOT.redraw(painter.drawid, histo, hopt, args.player_intermediate ? null : callback);
            painter.SetItemName("TreePlayer"); // item name used by MDI when process resize
         });
      });

      return painter;
   }

   // ==================================================================================================


   JSROOT.Painter.drawText = drawText;
   JSROOT.Painter.drawLine = drawLine;
   JSROOT.Painter.drawPolyLine = drawPolyLine;
   JSROOT.Painter.drawArrow = drawArrow;
   JSROOT.Painter.drawEllipse = drawEllipse;
   JSROOT.Painter.drawPie = drawPie;
   JSROOT.Painter.drawBox = drawBox;
   JSROOT.Painter.drawMarker = drawMarker;
   JSROOT.Painter.drawPolyMarker = drawPolyMarker;
   JSROOT.Painter.drawWebPainting = drawWebPainting;
   JSROOT.Painter.drawRooPlot = drawRooPlot;
   JSROOT.Painter.drawGraph = drawGraph;
   JSROOT.Painter.drawFunction = drawFunction;
   JSROOT.Painter.drawGraphPolar = drawGraphPolar;
   JSROOT.Painter.drawGraphPolargram = drawGraphPolargram;

   JSROOT.TF1Painter = TF1Painter;
   JSROOT.TGraphPainter = TGraphPainter;
   JSROOT.TGraphPolarPainter = TGraphPolarPainter;
   JSROOT.TMultiGraphPainter = TMultiGraphPainter;
   JSROOT.TSplinePainter = TSplinePainter;

   return JSROOT;

}));
