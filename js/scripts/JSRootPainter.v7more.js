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
          attr         = text.fAttr,
          pp           = this.pad_painter(),
          w            = this.pad_width(),
          h            = this.pad_height(),
          use_frame    = false,
          p            = pp.GetCoordinate(text.fPos),
          text_size    = pp.GetNewOpt(attr, "text_size", 12),
          text_angle   = -pp.GetNewOpt(attr, "text_angle", 0),
          text_align   = pp.GetNewOpt(attr, "text_align", 22),
          text_color   = pp.GetNewColor(attr, "text_color", "black"),
          text_font    = pp.GetNewOpt(attr, "text_font", 41);

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
           attr         = line.fAttr,
           pp           = this.pad_painter(),
           p1           = pp.GetCoordinate(line.fP1),
           p2           = pp.GetCoordinate(line.fP2),
           line_width   = pp.GetNewOpt(attr, "line_width", 1),
           line_style   = pp.GetNewOpt(attr, "line_style", 1),
           line_color   = pp.GetNewColor(attr, "line_color", "black");

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


   function drawBox() {

       var box          = this.GetObject(),
           attr         = box.fAttr,
           pp           = this.pad_painter(),
           p1           = pp.GetCoordinate(box.fP1),
           p2           = pp.GetCoordinate(box.fP2),
           line_width   = pp.GetNewOpt(attr, "box_border_width", 1),
           line_style   = pp.GetNewOpt(attr, "box_border_style", 1),
           line_color   = pp.GetNewColor(attr, "box_border_color", "black"),
           fill_color   = pp.GetNewColor(attr, "box_fill_color", "white"),
           fill_style   = pp.GetNewOpt(attr, "box_fill_style", 1),
           round_width  = pp.GetNewOpt(attr, "box_round_width", 0), // not yet exists
           round_height = pp.GetNewOpt(attr, "box_round_height", 0); // not yet exists

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
        .attr("fill", fill_color)
        .style("stroke-dasharray", JSROOT.Painter.root_line_styles[line_style]);
   }


   function drawMarker() {
       var marker         = this.GetObject(),
           attr           = marker.fAttr,
           pp             = this.pad_painter(),
           p              = pp.GetCoordinate(marker.fP),
           marker_size    = pp.GetNewOpt(attr, "marker.size", 1),
           marker_style   = pp.GetNewOpt(attr, "marker.style", 1),
           marker_color   = pp.GetNewColor(attr, "marker.color", "black");

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
