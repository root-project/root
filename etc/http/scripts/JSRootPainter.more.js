/// @file JSRootPainter.more.js
/// Part of JavaScript ROOT graphics with more classes like TEllipse, TLine, ...
/// Such classes are rarely used and therefore loaded only on demand

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3', 'JSRootMath'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"), require("./d3.min.js"), require("./JSRootMath.js"));
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
          tcolor = JSROOT.Painter.root_colors[text.fTextColor],
          use_pad = true, latex_kind = 0, fact = 1.;

      if (text.TestBit(JSROOT.BIT(14))) {
         // NDC coordinates
         pos_x = pos_x * w;
         pos_y = (1 - pos_y) * h;
      } else
      if (this.main_painter() !== null) {
         w = this.frame_width(); h = this.frame_height(); use_pad = false;
         pos_x = this.main_painter().grx(pos_x);
         pos_y = this.main_painter().gry(pos_y);
      } else
      if (this.root_pad() !== null) {
         pos_x = this.ConvertToNDC("x", pos_x) * w;
         pos_y = (1 - this.ConvertToNDC("y", pos_y)) * h;
      } else {
         text.fTextAlign = 22;
         pos_x = w/2;
         pos_y = h/2;
         if (text.fTextSize === 0) text.fTextSize = 0.05;
         if (text.fTextColor === 0) text.fTextColor = 1;
      }

      this.RecreateDrawG(use_pad, use_pad ? "text_layer" : "upper_layer");

      if (text._typename == 'TLatex') { latex_kind = 1; fact = 0.9; } else
      if (text._typename == 'TMathText') { latex_kind = 2; fact = 0.8; }

      this.StartTextDrawing(text.fTextFont, Math.round(text.fTextSize*Math.min(w,h)*fact));

      this.DrawText(text.fTextAlign, Math.round(pos_x), Math.round(pos_y), 0, 0, text.fTitle, tcolor, latex_kind);

      this.FinishTextDrawing();
   }

   // =====================================================================================

   function drawLine() {

      var line = this.GetObject(),
          lineatt = new JSROOT.TAttLineHandler(line),
          kLineNDC = JSROOT.BIT(14),
          isndc = line.TestBit(kLineNDC);

      // create svg:g container for line drawing
      this.RecreateDrawG(true, "text_layer");

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
          cmd = "";

      // create svg:g container for polyline drawing
      this.RecreateDrawG(true, "text_layer");

      for (var n=0;n<=polyline.fLastPoint;++n)
         cmd += ((n>0) ? "L" : "M") +
                this.AxisToSvg("x", polyline.fX[n], isndc) + "," +
                this.AxisToSvg("y", polyline.fY[n], isndc);

      if (fillatt.color!=='none') cmd+="Z";

      this.draw_g
          .append("svg:path")
          .attr("d", cmd)
          .call(lineatt.func)
          .call(fillatt.func);
   }

   // ==============================================================================

   function drawEllipse() {

      var ellipse = this.GetObject();

      if(!this.lineatt) this.lineatt = new JSROOT.TAttLineHandler(ellipse);
      if (!this.fillatt) this.fillatt = this.createAttFill(ellipse);

      // create svg:g container for ellipse drawing
      this.RecreateDrawG(true, "text_layer");

      var x = this.AxisToSvg("x", ellipse.fX1, false),
          y = this.AxisToSvg("y", ellipse.fY1, false),
          rx = this.AxisToSvg("x", ellipse.fX1 + ellipse.fR1, false) - x,
          ry = y - this.AxisToSvg("y", ellipse.fY1 + ellipse.fR2, false);

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
         .attr("transform","translate("+x.toFixed(1)+","+y.toFixed(1)+")")
         .append("svg:path")
         .attr("d", "M 0,0" +
                    " L " + x1.toFixed(1) + "," + y1.toFixed(1) +
                    " A " + rx.toFixed(1) + " " + ry.toFixed(1) + " " + -ellipse.fTheta.toFixed(1) + " 1 0 " + x2.toFixed(1) + "," + y2.toFixed(1) +
                    " L 0,0 Z")
         .call(this.lineatt.func).call(this.fillatt.func);
   }

   // =============================================================================

   function drawBox(divid, obj, opt) {

      var box = this.GetObject(),
          draw_line = (typeof this._drawopt == 'string') && (this._drawopt.toUpperCase().indexOf("L")>=0),
          lineatt = new JSROOT.TAttLineHandler(box),
          fillatt = this.createAttFill(box);

      // create svg:g container for box drawing
      this.RecreateDrawG(true, "text_layer");

      var x1 = this.AxisToSvg("x", box.fX1, false),
          x2 = this.AxisToSvg("x", box.fX2, false),
          y1 = this.AxisToSvg("y", box.fY1, false),
          y2 = this.AxisToSvg("y", box.fY2, false);

      // if box filled, contour line drawn only with "L" draw option:
      if ((fillatt.color != 'none') && !draw_line) lineatt.color = "none";

      this.draw_g
          .append("svg:rect")
          .attr("x", Math.min(x1,x2))
          .attr("y", Math.min(y1,y2))
          .attr("width", Math.abs(x2-x1))
          .attr("height", Math.abs(y1-y2))
          .call(lineatt.func)
          .call(fillatt.func);
   }

   // =============================================================================

   function drawMarker() {
      var marker = this.GetObject(),
          att = new JSROOT.TAttMarkerHandler(marker),
          kMarkerNDC = JSROOT.BIT(14),
          isndc = marker.TestBit(kMarkerNDC);

      // create svg:g container for box drawing
      this.RecreateDrawG(true, "text_layer");

      var x = this.AxisToSvg("x", marker.fX, isndc),
          y = this.AxisToSvg("y", marker.fY, isndc),
          path = att.create(x,y);

      if (path && path.length > 0)
         this.draw_g.append("svg:path")
             .attr("d", path)
             .call(att.func);
   }

   // =============================================================================

   function drawPolyMarker() {
      var poly = this.GetObject(),
          att = new JSROOT.TAttMarkerHandler(poly),
          isndc = false;

      // create svg:g container for box drawing
      this.RecreateDrawG(true, "text_layer");

      var path = "";

      for (var n=0;n<poly.fN;++n)
         path += att.create(this.AxisToSvg("x", poly.fX[n], isndc),
                            this.AxisToSvg("y", poly.fY[n], isndc));

      if (path)
         this.draw_g.append("svg:path")
             .attr("d", path)
             .call(att.func);
   }

   // ======================================================================================

   function drawArrow() {
      var arrow = this.GetObject();
      if (!this.lineatt) this.lineatt = new JSROOT.TAttLineHandler(arrow);
      if (!this.fillatt) this.fillatt = this.createAttFill(arrow);

      var wsize = Math.max(this.pad_width(), this.pad_height()) * arrow.fArrowSize;
      if (wsize<3) wsize = 3;
      var hsize = wsize * Math.tan(arrow.fAngle/2 * (Math.PI/180));

      // create svg:g container for line drawing
      this.RecreateDrawG(true, "text_layer");

      var x1 = this.AxisToSvg("x", arrow.fX1, false),
          y1 = this.AxisToSvg("y", arrow.fY1, false),
          x2 = this.AxisToSvg("x", arrow.fX2, false),
          y2 = this.AxisToSvg("y", arrow.fY2, false),
          right_arrow = "M0,0" + " L"+wsize.toFixed(1) +","+hsize.toFixed(1) + " L0," + (hsize*2).toFixed(1),
          left_arrow =  "M" + wsize.toFixed(1) + ", 0" + " L 0," + hsize.toFixed(1) + " L " + wsize.toFixed(1) + "," + (hsize*2).toFixed(1),
          m_start = null, m_mid = null, m_end = null, defs = null,
          oo = arrow.fOption, len = oo.length;

      if (oo.indexOf("<")==0) {
         var closed = (oo.indexOf("<|") == 0);
         if (!defs) defs = this.draw_g.append("defs");
         m_start = "jsroot_arrowmarker_" +  JSROOT.id_counter++;
         var beg = defs.append("svg:marker")
                       .attr("id", m_start)
                       .attr("markerWidth", wsize.toFixed(1))
                       .attr("markerHeight", (hsize*2).toFixed(1))
                       .attr("refX", "0")
                       .attr("refY", hsize.toFixed(1))
                       .attr("orient", "auto")
                       .attr("markerUnits", "userSpaceOnUse")
                       .append("svg:path")
                       .style("fill","none")
                       .attr("d", left_arrow + (closed ? " Z" : ""))
                       .call(this.lineatt.func);
         if (closed) beg.call(this.fillatt.func);
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

         var mid = defs.append("svg:marker")
                      .attr("id", m_mid)
                      .attr("markerWidth", wsize.toFixed(1))
                      .attr("markerHeight", (hsize*2).toFixed(1))
                      .attr("refX", (wsize*0.5).toFixed(1))
                      .attr("refY", hsize.toFixed(1))
                      .attr("orient", "auto")
                      .attr("markerUnits", "userSpaceOnUse")
                      .append("svg:path")
                      .style("fill","none")
                      .attr("d", ((midkind % 10 == 1) ? right_arrow : left_arrow) +
                            ((midkind > 10) ? " Z" : ""))
                            .call(this.lineatt.func);
         if (midkind > 10) mid.call(this.fillatt.func);
      }

      if (oo.lastIndexOf(">") == len-1) {
         var closed = (oo.lastIndexOf("|>") == len-2) && (len>1);
         if (!defs) defs = this.draw_g.append("defs");
         m_end = "jsroot_arrowmarker_" + JSROOT.id_counter++;
         var end = defs.append("svg:marker")
                       .attr("id", m_end)
                       .attr("markerWidth", wsize.toFixed(1))
                       .attr("markerHeight", (hsize*2).toFixed(1))
                       .attr("refX", wsize.toFixed(1))
                       .attr("refY", hsize.toFixed(1))
                       .attr("orient", "auto")
                       .attr("markerUnits", "userSpaceOnUse")
                       .append("svg:path")
                       .style("fill","none")
                       .attr("d", right_arrow + (closed ? " Z" : ""))
                       .call(this.lineatt.func);
         if (closed) end.call(this.fillatt.func);
      }

      var path = this.draw_g
           .append("svg:path")
           .attr("d",  "M" + x1 + "," + y1 +
                       ((m_mid == null) ? "" : "L" + (x1/2+x2/2).toFixed(1) + "," + (y1/2+y2/2).toFixed(1)) +
                       " L" + x2 + "," + y2)
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

   function TF1Painter(tf1) {
      JSROOT.TObjectPainter.call(this, tf1);
      this.bins = null;
   }

   TF1Painter.prototype = Object.create(JSROOT.TObjectPainter.prototype);

   TF1Painter.prototype.Eval = function(x) {
      return this.GetObject().evalPar(x);
   }

   TF1Painter.prototype.CreateBins = function(ignore_zoom) {
      var main = this.main_painter(), gxmin = 0, gxmax = 0, tf1 = this.GetObject();

      if ((main!==null) && !ignore_zoom)  {
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

      var xmin = tf1.fXmin, xmax = tf1.fXmax, logx = false, pad = this.root_pad();

      if (gxmin !== gxmax) {
         if (gxmin > xmin) xmin = gxmin;
         if (gxmax < xmax) xmax = gxmax;
      }

      if ((main!==null) && main.logx && (xmin>0) && (xmax>0)) {
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
         if (!isNaN(yy)) res.push({ x : xx, y : yy });
      }
      return res;
   }

   TF1Painter.prototype.CreateDummyHisto = function() {

      var xmin = 0, xmax = 1, ymin = 0, ymax = 1,
          bins = this.CreateBins(true);

      if ((bins!==null) && (bins.length > 0)) {

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
      if (tf1.fTitle.indexOf(';')!==0) {
         var array = tf1.fTitle.split(';');
         histo.fTitle = array[0];
         if (array.length>1)
            histo.fXaxis.fTitle = array[1];
         if (array.length>2)
            histo.fYaxis.fTitle = array[2];
      }
      else histo.fTitle = tf1.fTitle;

      histo.fXaxis.fXmin = xmin;
      histo.fXaxis.fXmax = xmax;
      histo.fYaxis.fXmin = ymin;
      histo.fYaxis.fXmax = ymax;

      return histo;
   }

   TF1Painter.prototype.ProcessTooltipFunc = function(pnt) {
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
                  color2: this.fillatt.color,
                  lines: [],
                  exact : (Math.abs(bin.grx - pnt.x) < radius) && (Math.abs(bin.gry - pnt.y) < radius) };

      res.changed = gbin.property("current_bin") !== best;
      res.menu = res.exact;
      res.menu_dist = Math.sqrt((bin.grx-pnt.x)*(bin.grx-pnt.x) + (bin.gry-pnt.y)*(bin.grx-pnt.x));

      if (res.changed)
         gbin.attr("cx", bin.grx)
             .attr("cy", bin.gry)
             .property("current_bin", best);

      var name = this.GetTipName();
      if (name.length > 0) res.lines.push(name);

      var pmain = this.main_painter();
      if (pmain!==null)
         res.lines.push("x = " + pmain.AxisAsText("x",bin.x) + " y = " + pmain.AxisAsText("y",bin.y));

      return res;
   }

   TF1Painter.prototype.Redraw = function() {

      var w = this.frame_width(), h = this.frame_height(), tf1 = this.GetObject();

      this.RecreateDrawG(false, "main_layer");

      // recalculate drawing bins when necessary
      this.bins = this.CreateBins(false);

      var pthis = this;
      var pmain = this.main_painter();
      var name = this.GetTipName("\n");

      if (!this.lineatt)
         this.lineatt = new JSROOT.TAttLineHandler(tf1);
      this.lineatt.used = false;
      if (!this.fillatt)
         this.fillatt = this.createAttFill(tf1, undefined, undefined, 1);
      this.fillatt.used = false;

      var n, bin;
      // first calculate graphical coordinates
      for(n=0; n<this.bins.length; ++n) {
         bin = this.bins[n];
         //bin.grx = Math.round(pmain.grx(bin.x));
         //bin.gry = Math.round(pmain.gry(bin.y));
         bin.grx = pmain.grx(bin.x);
         bin.gry = pmain.gry(bin.y);
      }

      if (this.bins.length > 2) {

         var h0 = h;  // use maximal frame height for filling
         if ((pmain.hmin!==undefined) && (pmain.hmin>=0)) {
            h0 = Math.round(pmain.gry(0));
            if ((h0 > h) || (h0 < 0)) h0 = h;
         }

         var path = JSROOT.Painter.BuildSvgPath("bezier", this.bins, h0, 2);

         if (this.lineatt.color != "none")
            this.draw_g.append("svg:path")
               .attr("class", "line")
               .attr("d", path.path)
               .style("fill", "none")
               .call(this.lineatt.func);

         if (this.fillatt.color != "none")
            this.draw_g.append("svg:path")
               .attr("class", "area")
               .attr("d", path.path + path.close)
               .style("stroke", "none")
               .call(this.fillatt.func);
      }

      delete this.ProcessTooltip;

     if (JSROOT.gStyle.Tooltip > 0)
        this.ProcessTooltip = this.ProcessTooltipFunc;
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

   JSROOT.Painter.drawFunction = function(divid, tf1, opt) {

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

   function TGraphPainter(graph) {
      JSROOT.TObjectPainter.call(this, graph);
      this.ownhisto = false; // indicate if graph histogram was drawn for axes
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

      var d = new JSROOT.DrawOptions(opt);

      var res = { Line:0, Curve:0, Rect:0, Mark:0, Bar:0, OutRange: 0,  EF:0, Fill:0,
                  Errors: 0, MainError: 1, Ends: 1, Axis: "AXIS", original: opt };

      var graph = this.GetObject();

      if (this.has_errors) res.Errors = 1;

      if (d.check('L')) res.Line = 1;
      if (d.check('F')) res.Fill = 1;
      if (d.check('A')) res.Axis = "AXIS";
      if (d.check('X+')) res.Axis += "X+";
      if (d.check('Y+')) res.Axis += "Y+";
      if (d.check('C')) { res.Curve = 1; if (!res.Fill) res.Line = 1; }
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
      if (d.check('2')) { res.Rect = 1; res.Line = 0; res.Errors = 0; }
      if (d.check('3')) { res.EF = 1; res.Line = 0; res.Errors = 0; }
      if (d.check('4')) { res.EF = 2; res.Line = 0; res.Errors = 0; }
      if (d.check('5')) { res.Rect = 2; res.Line = 0; res.Errors = 0; }
      if (d.check('X')) res.Errors = 0;

      // special case - one could use svg:path to draw many pixels (
      if ((res.Mark==1) && (graph.fMarkerStyle==1)) res.Mark = 101;

      // if no drawing option is selected and if opt=='' nothing is done.
      if (res.Line + res.Fill + res.Mark + res.Bar + res.EF + res.Rect + res.Errors == 0) {
         if (d.empty()) res.Line = 1;
      }

      if (graph._typename == 'TGraphErrors') {
         if (d3.max(graph.fEX) < 1.0e-300 && d3.max(graph.fEY) < 1.0e-300)
            res.Errors = 0;
      }

      return res;
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
               bin.exhigh  = gr.fEXhigh[p];
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

   TGraphPainter.prototype.CreateHistogram = function() {
      // bins should be created

      var xmin = this.xmin, xmax = this.xmax, ymin = this.ymin, ymax = this.ymax;

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

      var histo = JSROOT.CreateHistogram("TH1I", 100);
      histo.fName = graph.fName + "_h";
      histo.fTitle = graph.fTitle;
      histo.fXaxis.fXmin = uxmin;
      histo.fXaxis.fXmax = uxmax;
      histo.fYaxis.fXmin = minimum;
      histo.fYaxis.fXmax = maximum;
      histo.fMinimum = minimum;
      histo.fMaximum = maximum;
      histo.fBits = histo.fBits | JSROOT.TH1StatusBits.kNoStats;
      return histo;
   }

   TGraphPainter.prototype.OptimizeBins = function(filter_func) {
      if ((this.bins.length < 30) && !filter_func) return this.bins;

      var selbins = null;
      if (typeof filter_func == 'function') {
         for (var n = 0; n < this.bins.length; ++n) {
            if (filter_func(this.bins[n],n)) {
               if (selbins==null)
                  selbins = (n==0) ? [] : this.bins.slice(0, n);
            } else {
               if (selbins != null) selbins.push(this.bins[n]);
            }
         }
      }
      if (selbins == null) selbins = this.bins;

      if ((selbins.length < 5000) || (JSROOT.gStyle.OptimizeDraw == 0)) return selbins;
      var step = Math.floor(selbins.length / 5000);
      if (step < 2) step = 2;
      var optbins = [];
      for (var n = 0; n < selbins.length; n+=step)
         optbins.push(selbins[n]);

      return optbins;
   }

   TGraphPainter.prototype.TooltipText = function(d, asarray) {
      var pmain = this.main_painter(), lines = [];

      lines.push(this.GetTipName());

      if (d) {
         lines.push("x = " + pmain.AxisAsText("x", d.x));
         lines.push("y = " + pmain.AxisAsText("y", d.y));

         if (this.options.Errors && (pmain.x_kind=='normal') && ('exlow' in d) && ((d.exlow!=0) || (d.exhigh!=0)))
            lines.push("error x = -" + pmain.AxisAsText("x", d.exlow) + "/+" + pmain.AxisAsText("x", d.exhigh));

         if ((this.options.Errors || (this.options.EF > 0)) && (pmain.y_kind=='normal') && ('eylow' in d) && ((d.eylow!=0) || (d.eyhigh!=0)))
            lines.push("error y = -" + pmain.AxisAsText("y", d.eylow) + "/+" + pmain.AxisAsText("y", d.eyhigh));
      }
      if (asarray) return lines;

      var res = "";
      for (var n=0;n<lines.length;++n) res += ((n>0 ? "\n" : "") + lines[n]);
      return res;
   }

   TGraphPainter.prototype.DrawBins = function() {

      this.RecreateDrawG(false, "main_layer");

      var pthis = this,
          pmain = this.main_painter(),
          w = this.frame_width(),
          h = this.frame_height(),
          graph = this.GetObject(),
          excl_width = 0;

      if (!this.lineatt)
         this.lineatt = new JSROOT.TAttLineHandler(graph, undefined, true);
      if (!this.fillatt)
         this.fillatt = this.createAttFill(graph, undefined, undefined, 1);
      this.fillatt.used = false;

      if (this.fillatt) this.fillatt.used = false; // mark used only when really used
      this.draw_kind = "none"; // indicate if special svg:g were created for each bin
      this.marker_size = 0; // indicate if markers are drawn

      if (this.lineatt.excl_side!=0) {
         excl_width = this.lineatt.excl_side * this.lineatt.excl_width;
         if (this.lineatt.width>0) this.options.Line = 1;
      }

      var drawbins = null;

      if (this.options.EF) {

         drawbins = this.OptimizeBins();

         // build lower part
         for (var n=0;n<drawbins.length;++n) {
            var bin = drawbins[n];
            bin.grx = pmain.grx(bin.x);
            bin.gry = pmain.gry(bin.y - bin.eylow);
         }

         var path1 = JSROOT.Painter.BuildSvgPath(this.options.EF > 1 ? "bezier" : "line", drawbins),
             bins2 = [];

         for (var n=drawbins.length-1;n>=0;--n) {
            var bin = drawbins[n];
            bin.gry = pmain.gry(bin.y + bin.eyhigh);
            bins2.push(bin);
         }

         // build upper part (in reverse direction)
         var path2 = JSROOT.Painter.BuildSvgPath(this.options.EF > 1 ? "Lbezier" : "Lline", bins2);

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
            excl_width=0;
         }

         if (drawbins===null) drawbins = this.OptimizeBins();

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

         drawbins = this.OptimizeBins(function(pnt,i) {

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
         var step = Math.max(1, Math.round(this.bins.length / 50000)),
             path = "", n, pnt, grx, gry;

         if (!this.markeratt)
            this.markeratt = new JSROOT.TAttMarkerHandler(graph, this.options.Mark - 100);
         else
            this.markeratt.Change(undefined, this.options.Mark - 100);

         this.marker_size = this.markeratt.size;

         this.markeratt.reset_pos();

         for (n=0;n<this.bins.length;n+=step) {
            pnt = this.bins[n];
            grx = pmain.grx(pnt.x);
            if ((grx > -this.marker_size) && (grx < w+this.marker_size)) {
               gry = pmain.gry(pnt.y);
               if ((gry >-this.marker_size) && (gry < h+this.marker_size)) {
                  path += this.markeratt.create(grx, gry);
               }
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
          pmain = this.main_painter(),
          painter = this,
          findbin = null, best_dist2 = 1e10, best = null;

      this.draw_g.selectAll('.grpoint').each(function() {
         var d = d3.select(this).datum();
         if (d===undefined) return;
         var dist2 = Math.pow(pnt.x - d.grx1, 2);
         if (pnt.nproc===1) dist2 += Math.pow(pnt.y - d.gry1, 2);
         if (dist2 >= best_dist2) return;

         var rect = null;

         if (d.error || d.rect || d.marker) {
            rect = { x1: Math.min(-painter.error_size, d.grx0),
                     x2: Math.max(painter.error_size, d.grx2),
                     y1: Math.min(-painter.error_size, d.gry2),
                     y2: Math.max(painter.error_size, d.gry0) };
         } else
         if (d.bar) {
             rect = { x1: -d.width/2, x2: d.width/2, y1: 0, y2: height - d.gry1 };

             if (painter.options.Bar===1) {
                var yy0 = pmain.gry(0);
                rect.y1 = (d.gry1 > yy0) ? yy0-d.gry1 : 0;
                rect.y2 = (d.gry1 > yy0) ? 0 : yy0-d.gry1;
             }
          } else {
             rect = { x1: -5, x2: 5, y1: -5, y2: 5 };
          }
          var matchx = (pnt.x >= d.grx1 + rect.x1) && (pnt.x <= d.grx1 + rect.x2);
          var matchy = (pnt.y >= d.gry1 + rect.y1) && (pnt.y <= d.gry1 + rect.y2);

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
                  lines: this.TooltipText(d, true),
                  rect: best, d3bin: findbin  };

      if (this.fillatt && this.fillatt.used) res.color2 = this.fillatt.color;

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
          pmain = this.main_painter(),
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

      if (this.marker_size > 0) radius = Math.max(Math.round(this.marker_size*7), radius);

      if (bestbin !== null)
         bestdist = Math.sqrt(Math.pow(pnt.x-pmain.grx(bestbin.x),2) + Math.pow(pnt.y-pmain.gry(bestbin.y),2));

      if (!islines && !ismark && (bestdist > radius)) bestbin = null;

      if (ismark && (bestbin!==null)) {
         if ((pnt.nproc == 1) && (bestdist > radius)) bestbin = null; else
         if ((this.bins.length==1) && (bestdist > 3*radius)) bestbin = null;
      }

      if (bestbin === null) bestindx = -1;

      var res = { bin: bestbin, indx: bestindx, dist: bestdist, radius: radius };

      if ((bestbin===null) && islines) {

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
          pmain = this.main_painter();

      var res = { name: this.GetObject().fName, title: this.GetObject().fTitle,
                  x: best.bin ? pmain.grx(best.bin.x) : best.linex,
                  y: best.bin ? pmain.gry(best.bin.y) : best.liney,
                  color1: this.lineatt.color,
                  lines: this.TooltipText(best.bin, true),
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

      if (this.fillatt && this.fillatt.used) res.color2 = this.fillatt.color;

      if (!islines) {
         res.color1 = this.get_color(this.GetObject().fMarkerColor);
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
                 .attr("x", (hint.x - hint.radius).toFixed(1))
                 .attr("y", (hint.y - hint.radius).toFixed(1))
                 .attr("width", (2*hint.radius).toFixed(1))
                 .attr("height", (2*hint.radius).toFixed(1));
         } else {
            ttbin.append("svg:circle").attr("cy", hint.gry1.toFixed(1))
            if (Math.abs(hint.gry1-hint.gry2) > 1)
               ttbin.append("svg:circle").attr("cy", hint.gry2.toFixed(1));

            var elem = ttbin.selectAll("circle")
                            .attr("r", hint.radius)
                            .attr("cx", hint.x.toFixed(1));

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
      var pos = d3.mouse(this.svg_frame().node());

      var main = this.main_painter();
      if (!main || !this.interactive_bin) return;

      this.interactive_bin.x = main.RevertX(pos[0] + this.interactive_delta_x);
      this.interactive_bin.y = main.RevertY(pos[1] + this.interactive_delta_y);
      this.DrawBins();
   }

   TGraphPainter.prototype.endPntHandler = function() {
      if (this.snapid && this.interactive_bin) {
         var exec = "SetPoint(" + this.interactive_bin.indx + "," + this.interactive_bin.x + "," + this.interactive_bin.y + ")";
         var canp = this.pad_painter();
         if (canp) canp.SendWebsocket("OBJEXEC:" + this.snapid + ":" + exec);
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

      var pos = d3.mouse(this.svg_frame().node());
      var main = this.main_painter();

      this.interactive_delta_x = main ? main.x(this.interactive_bin.x)-pos[0] : 0;
      this.interactive_delta_y = main ? main.y(this.interactive_bin.y)-pos[1] : 0;
   }

   TGraphPainter.prototype.FillContextMenu = function(menu) {
      JSROOT.TObjectPainter.prototype.FillContextMenu.call(this, menu);

      if (!this.snapid)
         menu.addchk(this.TestEditable(), "Editable", this.TestEditable.bind(this, true));

      return menu.size() > 0;
   }


   TGraphPainter.prototype.ExecuteMenuCommand = function(item) {
      if (JSROOT.TObjectPainter.prototype.ExecuteMenuCommand.call(this,item)) return true;

      var canp = this.pad_painter(), fp = this.frame_painter();

      if ((item.fName == 'RemovePoint') || (item.fName == 'InsertPoint')) {
         var pnt = fp ? fp.GetLastEventPos() : null;

         if (!canp || !fp || !pnt) return true; // ignore function

         var hint = this.ExtractTooltip(pnt);

         if (item.fName == 'InsertPoint') {
            var main = this.main_painter(),
                userx = main && main.RevertX ? main.RevertX(pnt.x) : 0,
                usery = main && main.RevertY ? main.RevertY(pnt.y) : 0;
            canp.ShowMessage('InsertPoint(' + userx.toFixed(3) + ',' + usery.toFixed(3) + ') not yet implemented');
         } else
         if (this.args_menu_id && hint && (hint.binindx !== undefined)) {
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
         this.options = this.DecodeOptions(opt);

      var graph = this.GetObject();
      // TODO: make real update of TGraph object content
      graph.fBits = obj.fBits;
      graph.fTitle = obj.fTitle;
      graph.fX = obj.fX;
      graph.fY = obj.fY;
      graph.fNpoints = obj.fNpoints;
      this.CreateBins();

      // if our own histogram was used as axis drawing, we need update histogram  as well
      if (this.ownhisto) {
         var main = this.main_painter();
         if (obj.fHistogram) main.UpdateObject(obj.fHistogram);
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

      var main = this.main_painter();
      if (main === null) return false;

      if ((this.xmin===this.xmax) && (this.ymin = this.ymax)) return false;

      main.Zoom(this.xmin, this.xmax, this.ymin, this.ymax);

      return true;
   }

   TGraphPainter.prototype.DrawNextFunction = function(indx, callback) {
      // method draws next function from the functions list

      var graph = this.GetObject();

      if (!graph.fFunctions || (indx >= graph.fFunctions.arr.length))
         return JSROOT.CallBack(callback);

      JSROOT.draw(this.divid, graph.fFunctions.arr[indx], graph.fFunctions.opt[indx],
                  this.DrawNextFunction.bind(this, indx+1, callback));
   }

   TGraphPainter.prototype.PerformDrawing = function(divid, hpainter) {
      if (hpainter) this.ownhisto = true;
      this.SetDivId(divid);
      this.DrawBins();
      this.DrawNextFunction(0, this.DrawingReady.bind(this));
      return this;
   }

   JSROOT.Painter.drawGraph = function(divid, graph, opt) {

      var painter = new TGraphPainter(graph);

      painter.options = painter.DecodeOptions(opt);

      painter.SetDivId(divid, -1); // just to get access to existing elements

      painter.CreateBins();

      if (!painter.main_painter()) {
         if (!graph.fHistogram)
            graph.fHistogram = painter.CreateHistogram();
         JSROOT.draw(divid, graph.fHistogram, painter.options.Axis, painter.PerformDrawing.bind(painter, divid));
      } else {
         painter.PerformDrawing(divid);
      }

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
          rw = {  xmin: 0, xmax: 0, ymin: 0, ymax: 0, first: true };

      if (pad!=null) {
         logx = pad.fLogx;
         logy = pad.fLogy;
         rw.xmin = pad.fUxmin;
         rw.xmax = pad.fUxmax;
         rw.ymin = pad.fUymin;
         rw.ymax = pad.fUymax;
         rw.first = false;
      }
      if (histo!=null) {
         minimum = histo.fYaxis.fXmin;
         maximum = histo.fYaxis.fXmax;
         if (pad!=null) {
            uxmin = this.padtoX(pad, rw.xmin);
            uxmax = this.padtoX(pad, rw.xmax);
         }
      } else {
         this.autorange = true;

         for (var i = 0; i < graphs.arr.length; ++i)
            this.ComputeGraphRange(rw, graphs.arr[i]);

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

      if (minimum < 0 && rw.ymin >= 0 && logy)
         minimum = 0.9 * rw.ymin;
      if (maximum > 0 && rw.ymax <= 0 && logy)
         maximum = 1.1 * rw.ymax;
      if (minimum <= 0 && logy)
         minimum = 0.001 * maximum;
      if (uxmin <= 0 && logx)
         uxmin = (uxmax > 1000) ? 1 : 0.001 * uxmax;

      // Create a temporary histogram to draw the axis (if necessary)
      if (!histo) {
         histo = JSROOT.Create("TH1I");
         histo.fTitle = mgraph.fTitle;
         histo.fXaxis.fXmin = uxmin;
         histo.fXaxis.fXmax = uxmax;
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

   TMultiGraphPainter.prototype.DrawNextGraph = function(indx, opt, subp) {
      if (subp) this.painters.push(subp);

      var graphs = this.GetObject().fGraphs;

      // at the end of graphs drawing draw functions (if any)
      if (indx >= graphs.arr.length)
         return this.DrawNextFunction(0, this.DrawingReady.bind(this));

      // if there is auto colors assignment, try to provide it
      if (this._pfc || this._plc || this._pmc) {
         if (!this.pallette && JSROOT.Painter.GetColorPalette)
            this.palette = JSROOT.Painter.GetColorPalette();
         if (this.palette) {
            var color = this.palette.calcColor(indx, graphs.arr.length+1);
            var icolor = JSROOT.Painter.root_colors.indexOf(color);
            if (icolor<0) {
               icolor = JSROOT.Painter.root_colors.length;
               JSROOT.Painter.root_colors.push(color);
            }
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

      var painter = new JSROOT.TObjectPainter(obj);

      painter.UpdateObject = function(obj) {
         if (!this.MatchObjectType(obj)) return false;
         this.draw_object = obj;
         return true;
      }

      painter.Redraw = function() {
         var obj = this.GetObject(), attr = null, indx = 0,
             lineatt = null, fillatt = null, markeratt = null;
         if (!obj || !obj.fOper) return;

         this.RecreateDrawG(true, "special_layer");

         for (var k=0;k<obj.fOper.arr.length;++k) {
            var oper = obj.fOper.opt[k];
            switch (oper) {
               case "attr":
                  attr = obj.fOper.arr[k];
                  lineatt = fillatt = markeratt = null;
                  continue;
               case "rect":
               case "box": {
                  var x1 = this.AxisToSvg("x", obj.fBuf[indx++]),
                      y1 = this.AxisToSvg("y", obj.fBuf[indx++]),
                      x2 = this.AxisToSvg("x", obj.fBuf[indx++]),
                      y2 = this.AxisToSvg("y", obj.fBuf[indx++]);

                  if (!lineatt) lineatt = new JSROOT.TAttLineHandler(attr);

                  var rect = this.draw_g
                     .append("svg:rect")
                     .attr("x", Math.min(x1,x2))
                     .attr("y", Math.min(y1,y2))
                     .attr("width", Math.abs(x2-x1))
                     .attr("height", Math.abs(y1-y2))
                     .call(lineatt.func);

                  if (oper === "box") {
                     if (!fillatt) fillatt = this.createAttFill(attr);
                     rect.call(fillatt.func);
                  }
                  continue;
               }
               case "line":
               case "linendc": {

                  var isndc = (oper==="linendc"),
                      x1 = this.AxisToSvg("x", obj.fBuf[indx++], isndc),
                      y1 = this.AxisToSvg("y", obj.fBuf[indx++], isndc),
                      x2 = this.AxisToSvg("x", obj.fBuf[indx++], isndc),
                      y2 = this.AxisToSvg("y", obj.fBuf[indx++], isndc);

                  if (!lineatt) lineatt = new JSROOT.TAttLineHandler(attr);

                  this.draw_g
                      .append("svg:line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
                      .call(lineatt.func);

                  continue;
               }
               case "polyline":
               case "polylinendc":
               case "fillarea": {

                  var npoints = parseInt(obj.fOper.arr[k].fString), cmd = "";

                  if (!lineatt) lineatt = new JSROOT.TAttLineHandler(attr);

                  for (var n=0;n<npoints;++n)
                     cmd += ((n>0) ? "L" : "M") +
                            this.AxisToSvg("x", obj.fBuf[indx++], false) + "," +
                            this.AxisToSvg("y", obj.fBuf[indx++], false);

                  if (oper == "fillarea") cmd+="Z";
                  var path = this.draw_g
                          .append("svg:path")
                          .attr("d", cmd)
                          .call(lineatt.func);

                  if (oper == "fillarea") {
                     if (!fillatt) fillatt = this.createAttFill(attr);
                     path.call(fillatt.func);
                  }

                  continue;
               }

               case "polymarker": {
                  var npoints = parseInt(obj.fOper.arr[k].fString), cmd = "";

                  if (!markeratt) markeratt = new JSROOT.TAttMarkerHandler(attr);

                  markeratt.reset_pos();
                  for (var n=0;n<npoints;++n)
                     cmd += markeratt.create(this.AxisToSvg("x", obj.fBuf[indx++], false),
                                             this.AxisToSvg("y", obj.fBuf[indx++], false));

                  if (cmd)
                     this.draw_g.append("svg:path").attr("d", cmd).call(markeratt.func);

                  continue;
               }

               case "text":
               case "textndc": {
                  var isndc = (oper==="textndc"),
                      xx = this.AxisToSvg("x", obj.fBuf[indx++], isndc),
                      yy = this.AxisToSvg("y", obj.fBuf[indx++], isndc);

                  if (attr) {
                     var height = (attr.fTextSize > 1) ? attr.fTextSize : this.pad_height() * attr.fTextSize;

                     var group = this.draw_g.append("svg:g");

                     this.StartTextDrawing(attr.fTextFont, height, group);

                     var angle = attr.fTextAngle;
                     angle -= Math.floor(angle/360) * 360;
                     if (angle>0) angle = -360 + angle; // rotation angle in ROOT and SVG has different meaning

                     var enable_latex = 0; // 0-off, 1 - when make sense, 2 - always

                     // todo - correct support of angle
                     this.DrawText(attr.fTextAlign, xx, yy, 0, angle, obj.fOper.arr[k].fString, JSROOT.Painter.root_colors[attr.fTextColor], enable_latex, group);

                     this.FinishTextDrawing(group);
                  }
                  continue;
               }

               default:
                  console.log('unsupported operation', oper);
            }

         }
      }

      painter.SetDivId(divid);

      painter.options = opt;

      painter.Redraw();

      return painter.DrawingReady();
   }


   JSROOT.Painter.drawText = drawText;
   JSROOT.Painter.drawLine = drawLine;
   JSROOT.Painter.drawPolyLine = drawPolyLine;
   JSROOT.Painter.drawArrow = drawArrow;
   JSROOT.Painter.drawEllipse = drawEllipse;
   JSROOT.Painter.drawBox = drawBox;
   JSROOT.Painter.drawMarker = drawMarker;
   JSROOT.Painter.drawPolyMarker = drawPolyMarker;
   JSROOT.Painter.drawWebPainting = drawWebPainting;
   JSROOT.Painter.drawRooPlot = drawRooPlot;

   JSROOT.TF1Painter = TF1Painter;
   JSROOT.TGraphPainter = TGraphPainter;
   JSROOT.TMultiGraphPainter = TMultiGraphPainter;

   return JSROOT;

}));
