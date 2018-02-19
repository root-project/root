/// @file JSRootPainter.v7more.js
/// JavaScript ROOT v7 graphics for different classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( ['JSRootPainter', 'd3'], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
       factory(require("./JSRootCore.js"), require("./d3.min.js"));
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
      var drawable = this.GetObject(),
          pp = this.canv_painter(),
          w = this.pad_width(),
          h = this.pad_height(),
          use_frame = false,
          text_font = 42,
          text_size = 40;

      var text = drawable && drawable.fText ? drawable.fText.fWeakForIO : null,
          opts = drawable.fOpts;

      var fillcolor = pp.GetNewColor(opts.fFillColor);

      this.CreateG(use_frame);

      var arg = { align: 13, x: Math.round(text.fX*w), y: Math.round(text.fY*h), text: text.fText, color: fillcolor, latex: 1 };

      // if (text.fTextAngle) arg.rotate = -text.fTextAngle;
      // if (text._typename == 'TLatex') { arg.latex = 1; fact = 0.9; } else
      // if (text._typename == 'TMathText') { arg.latex = 2; fact = 0.8; }

      this.StartTextDrawing(text_font, text_size);

      this.DrawText(arg);

      this.FinishTextDrawing();
   }

   // ================================================================================

   JSROOT.v7.drawText = drawText;

   return JSROOT;

}));
