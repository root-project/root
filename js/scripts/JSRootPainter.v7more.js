/// @file JSRootPainter.v7more.js
/// JavaScript ROOT v7 graphics for different classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
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
          opts         = text.fOpts,
          pp           = this.canv_painter(),
          w            = this.pad_width(),
          h            = this.pad_height(),
          use_frame    = false,
          text_size    = pp.GetNewOpt(opts, "text.size", 12),
          text_angle   = -pp.GetNewOpt(opts, "text.angle", 0),
          text_align   = pp.GetNewOpt(opts, "text.align", 22),
          text_color   = pp.GetNewColor(opts, "text.color", "black"),
          text_font    = pp.GetNewOpt(opts, "text.font", 41),
          p            = this.GetCoordinate(text.fP);

      this.CreateG(use_frame);

      var arg = { align: text_align, x: p.x, y: p.y, text: text.fText, rotate: text_angle, color: text_color, latex: 1 };

      // if (text.fTextAngle) arg.rotate = -text.fTextAngle;
      // if (text._typename == 'TLatex') { arg.latex = 1; fact = 0.9; } else
      // if (text._typename == 'TMathText') { arg.latex = 2; fact = 0.8; }

      this.StartTextDrawing(text_font, text_size);

      this.DrawText(arg);

      this.FinishTextDrawing();
   }


   function drawLine() {

       var line         = this.GetObject(),
           opts         = line.fOpts,
           pp           = this.canv_painter(),

           line_width   = pp.GetNewOpt(opts, "line.width", 1),
           line_style   = pp.GetNewOpt(opts, "line.style", 1),
           line_color   = pp.GetNewColor(opts, "line.color", "black"),
           line_opacity = pp.GetNewOpt(opts, "line.opacity", 1),
           p1           = this.GetCoordinate(line.fP1),
           p2           = this.GetCoordinate(line.fP2);

    this.CreateG();

    this.draw_g
        .append("svg:line")
        .attr("x1", p1.x)
        .attr("y1", p1.y)
        .attr("x2", p2.x)
        .attr("y2", p2.y)
        .style("stroke", line_color)
        .attr("stroke-width", line_width)
        .attr("stroke-opacity", line_opacity)
        .style("stroke-dasharray", JSROOT.Painter.root_line_styles[line_style]);
   }


   function drawBox() {

       var box          = this.GetObject(),
           opts         = box.fOpts,
           pp           = this.canv_painter(),
           line_width   = pp.GetNewOpt(opts, "box.border.width", 1),
           line_opacity = pp.GetNewOpt(opts, "box.border.opacity", 1),
           line_style   = pp.GetNewOpt(opts, "box.border.style", 1),
           line_color   = pp.GetNewColor(opts, "box.border.color", "black"),
           fill_opacity = pp.GetNewOpt(opts, "box.fill.opacity", 1),
           fill_color   = pp.GetNewColor(opts, "box.fill.color", "white"),
           fill_style   = pp.GetNewOpt(opts, "box.fill.style", 1),
           round_width  = pp.GetNewOpt(opts, "box.round.width", 0),
           round_height = pp.GetNewOpt(opts, "box.round.height", 0),
           p1           = this.GetCoordinate(box.fP1),
           p2           = this.GetCoordinate(box.fP2);

    this.CreateG();

    if (fill_style == 0 ) fill_color = "none";

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
        .attr("stroke-opacity", line_opacity)
        .attr("fill", fill_color)
        .attr("fill-opacity", fill_opacity)
        .style("stroke-dasharray", JSROOT.Painter.root_line_styles[line_style]);
   }


   function drawMarker() {
       var marker         = this.GetObject(),
           opts           = marker.fOpts,
           pp             = this.canv_painter(),
           marker_size    = pp.GetNewOpt(opts, "marker.size", 1),
           marker_opacity = pp.GetNewOpt(opts, "marker.opacity", 1),
           marker_style   = pp.GetNewOpt(opts, "marker.style", 1),
           marker_color   = pp.GetNewColor(opts, "marker.color", "black"),
           p              = this.GetCoordinate(marker.fP);

           var att = new JSROOT.TAttMarkerHandler({ style: marker_style, color: marker_color, size: marker_size });

           this.CreateG();

           var  path = att.create(p.x, p.y);

           if (path)
              this.draw_g.append("svg:path")
                  .attr("d", path)
                  .call(att.func);
   }

   // ================================================================================

   JSROOT.v7.drawText   = drawText;
   JSROOT.v7.drawLine   = drawLine;
   JSROOT.v7.drawBox    = drawBox;
   JSROOT.v7.drawMarker = drawMarker;

   return JSROOT;

}));
