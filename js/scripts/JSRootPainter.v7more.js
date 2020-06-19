/// @file JSRootPainter.v7more.js
/// JavaScript ROOT v7 graphics for different classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3'], factory );
   } else if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"), require("d3"));
   } else {
      if (typeof d3 != 'object')
         throw new Error('This extension requires d3.js', 'JSRootPainter.v7hist.js');
      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootPainter.v7hist.js');
      if (typeof JSROOT.Painter != 'object')
         throw new Error('JSROOT.Painter not defined', 'JSRootPainter.v7hist.js');
      factory(JSROOT, d3);
   }
} (function(JSROOT, d3) {

   "use strict";

   JSROOT.sources.push("v7more");

   // =================================================================================

   function drawText() {
      var text         = this.GetObject(),
          pp           = this.pad_painter(),
          use_frame    = false,
          p            = pp.GetCoordinate(text.fPos),
          text_size    = this.v7EvalAttr( "text_size", 12),
          text_angle   = -1 * this.v7EvalAttr( "text_angle", 0),
          text_align   = this.v7EvalAttr( "text_align", 22),
          text_color   = this.v7EvalColor( "text_color", "black"),
          text_font    = this.v7EvalAttr( "text_font", 41);

      this.CreateG(use_frame);

      var arg = { align: text_align, x: p.x, y: p.y, text: text.fText, rotate: text_angle, color: text_color, latex: 1 };

      // if (text.fTextAngle) arg.rotate = -text.fTextAngle;
      // if (text._typename == 'TLatex') { arg.latex = 1; fact = 0.9; } else
      // if (text._typename == 'TMathText') { arg.latex = 2; fact = 0.8; }

      this.StartTextDrawing(text_font, text_size);

      this.DrawText(arg);

      this.FinishTextDrawing();
   }


   // =================================================================================

   function drawLine() {

       var line         = this.GetObject(),
           pp           = this.pad_painter(),
           p1           = pp.GetCoordinate(line.fP1),
           p2           = pp.GetCoordinate(line.fP2),
           line_width   = this.v7EvalAttr("line_width", 1),
           line_style   = this.v7EvalAttr("line_style", 1),
           line_color   = this.v7EvalColor("line_color", "black");

       this.CreateG();

       this.draw_g
           .append("svg:line")
           .attr("x1", p1.x)
           .attr("y1", p1.y)
           .attr("x2", p2.x)
           .attr("y2", p2.y)
           .style("stroke", line_color)
           .attr("stroke-width", line_width)
//        .attr("stroke-opacity", line_opacity)
           .style("stroke-dasharray", JSROOT.Painter.root_line_styles[line_style]);
   }

   // =================================================================================

   function drawBox() {

       var box          = this.GetObject(),
           pp           = this.pad_painter(),
           p1           = pp.GetCoordinate(box.fP1),
           p2           = pp.GetCoordinate(box.fP2),
           line_width   = this.v7EvalAttr( "box_border_width", 1),
           line_style   = this.v7EvalAttr( "box_border_style", 1),
           line_color   = this.v7EvalColor( "box_border_color", "black"),
           fill_color   = this.v7EvalColor( "box_fill_color", "white"),
           fill_style   = this.v7EvalAttr( "box_fill_style", 1),
           round_width  = this.v7EvalAttr( "box_round_width", 0), // not yet exists
           round_height = this.v7EvalAttr( "box_round_height", 0); // not yet exists

    this.CreateG();

    if (fill_style == 0) fill_color = "none";

    this.draw_g
        .append("svg:rect")
        .attr("x", p1.x)
        .attr("width", p2.x-p1.x)
        .attr("y", p2.y)
        .attr("height", p1.y-p2.y)
        .attr("rx", round_width)
        .attr("ry", round_height)
        .style("stroke", line_color)
        .attr("stroke-width", line_width)
        .attr("fill", fill_color)
        .style("stroke-dasharray", JSROOT.Painter.root_line_styles[line_style]);
   }

   // =================================================================================

   function drawMarker() {
       var marker       = this.GetObject(),
           pp           = this.pad_painter(),
           p            = pp.GetCoordinate(marker.fP),
           marker_size  = this.v7EvalAttr( "marker_size", 1),
           marker_style = this.v7EvalAttr( "marker_style", 1),
           marker_color = this.v7EvalColor( "marker_color", "black"),
           att          = new JSROOT.TAttMarkerHandler({ style: marker_style, color: marker_color, size: marker_size }),
           path         = att.create(p.x, p.y);

       this.CreateG();

       if (path)
          this.draw_g.append("svg:path")
                     .attr("d", path)
                     .call(att.func);
   }

   // =================================================================================

   function drawLegendContent() {
      var legend     = this.GetObject(),
          text_size  = this.v7EvalAttr( "legend_text_size", 20),
          text_angle = -1 * this.v7EvalAttr( "legend_text_angle", 0),
          text_align = this.v7EvalAttr( "legend_text_align", 12),
          text_color = this.v7EvalColor( "legend_text_color", "black"),
          text_font  = this.v7EvalAttr( "legend_text_font", 41),
          width      = this.pave_width,
          height     = this.pave_height,
          nlines     = legend.fEntries.length,
          pp         = this.pad_painter();

      if (legend.fTitle) nlines++;

      if (!nlines || !pp) return;

      var arg = { align: text_align, rotate: text_angle, color: text_color, latex: 1 },
          stepy = height / nlines, posy = 0, margin_x = 0.02 * width;

      this.StartTextDrawing(text_font, height/(nlines * 1.2));

      if (legend.fTitle) {
         this.DrawText({ align: 22, rotate: text_angle, color: text_color, latex: 1,
                         width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: legend.fTitle });
         posy += stepy;
      }

      for (var i=0; i<legend.fEntries.length; ++i) {
         var objp = null, entry = legend.fEntries[i];

         this.DrawText({ align: text_align, rotate: text_angle, color: text_color, latex: 1,
                         width: 0.75*width - 3*margin_x, height: stepy, x: 2*margin_x + width*0.25, y: posy, text: entry.fLabel });

         if (entry.fDrawableId != "custom") {
            objp = pp.FindSnap(entry.fDrawableId, true);
         } else if (entry.fDrawable.fIO) {
            objp = new JSROOT.TObjectPainter(entry.fDrawable.fIO);
            if (entry.fLine) objp.createv7AttLine();
            if (entry.fFill) objp.createv7AttFill();
            if (entry.fMarker) objp.createv7AttMarker();
         }

         if (objp && entry.fFill && objp.fillatt)
            this.draw_g
              .append("svg:rect")
              .attr("x", Math.round(margin_x))
              .attr("y", Math.round(posy + stepy*0.1))
              .attr("width", Math.round(width/4))
              .attr("height", Math.round(stepy*0.8))
              .call(objp.fillatt.func);

         if (objp && entry.fLine && objp.lineatt)
            this.draw_g
              .append("svg:line")
              .attr("x1", Math.round(margin_x))
              .attr("y1", Math.round(posy + stepy/2))
              .attr("x2", Math.round(margin_x + width/4))
              .attr("y2", Math.round(posy + stepy/2))
              .call(objp.lineatt.func);

         if (objp && entry.fMarker && objp.markeratt)
            this.draw_g.append("svg:path")
                .attr("d", objp.markeratt.create(margin_x + width/8, posy + stepy/2))
                .call(objp.markeratt.func);

         posy += stepy;
      }

      this.FinishTextDrawing();
   }

   function drawLegend(divid, legend, opt) {
      var painter = new JSROOT.v7.RPavePainter(legend, opt, "legend");

      painter.SetDivId(divid);

      painter.DrawContent = drawLegendContent;

      painter.DrawPave();

      return painter.DrawingReady();
   }

   // =================================================================================

   function drawPaveTextContent() {
      var pavetext   = this.GetObject(),
          text_size  = this.v7EvalAttr( "pavetext_text_size", 20),
          text_angle = -1 * this.v7EvalAttr( "pavetext_text_angle", 0),
          text_align = this.v7EvalAttr( "pavetext_text_align", 12),
          text_color = this.v7EvalColor( "pavetext_text_color", "black"),
          text_font  = this.v7EvalAttr( "pavetext_text_font", 41),
          width      = this.pave_width,
          height     = this.pave_height,
          nlines     = pavetext.fText.length;

      if (!nlines) return;

      var stepy = height / nlines, posy = 0, margin_x = 0.02 * width;

      this.StartTextDrawing(text_font, height/(nlines * 1.2));

      for (var i=0; i < pavetext.fText.length; ++i) {
         var line = pavetext.fText[i];

         this.DrawText({ align: text_align, rotate: text_angle, color: text_color, latex: 1,
                         width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: line });
         posy += stepy;
      }

      this.FinishTextDrawing();
   }

   function drawPaveText(divid, pave, opt) {
      var painter = new JSROOT.v7.RPavePainter(pave, opt, "pavetext");

      painter.SetDivId(divid);

      painter.DrawContent = drawPaveTextContent;

      painter.DrawPave();

      return painter.DrawingReady();
   }


   // ================================================================================

   JSROOT.v7.drawText     = drawText;
   JSROOT.v7.drawLine     = drawLine;
   JSROOT.v7.drawBox      = drawBox;
   JSROOT.v7.drawMarker   = drawMarker;
   JSROOT.v7.drawLegend   = drawLegend;
   JSROOT.v7.drawPaveText = drawPaveText;

   return JSROOT;

}));
