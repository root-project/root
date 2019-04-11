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
          text_size    = opts.fTextSize.fAttr,
          text_angle   = -opts.fTextAngle.fAttr,
          text_align   = opts.fTextAlign.fAttr,
          text_color   = pp.GetOldColor(opts.fTextColor),
          text_font    = opts.fTextFont.fAttr,
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
           line_width   = opts.fWidth.fAttr,
           line_opacity = opts.fOpacity.fAttr,
           line_style   = opts.fStyle.fAttr,
           line_color   = pp.GetOldColor(opts.fColor),
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
           line_width   = opts.fLineWidth.fAttr,
           line_opacity = opts.fLineOpacity.fAttr,
           line_style   = opts.fLineStyle.fAttr,
           line_color   = pp.GetOldColor(opts.fLineColor),
           fill_opacity = opts.fFillOpacity.fAttr,
           fill_color   = pp.GetOldColor(opts.fFillColor),
           fill_style   = opts.fFillStyle.fAttr,
           round_width  = opts.fRoundWidth.fAttr,
           round_height = opts.fRoundHeight.fAttr,
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
           marker_size    = opts.fSize.fAttr,
           marker_opacity = opts.fOpacity.fAttr,
           marker_style   = opts.fStyle.fAttr,
           marker_color   = pp.GetOldColor(opts.fColor),
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
