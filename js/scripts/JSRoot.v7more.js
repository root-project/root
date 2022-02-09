/// @file JSRoot.v7more.js
/// JavaScript ROOT v7 graphics for different classes

JSROOT.define(['painter', 'v7gpad'], jsrp => {

   "use strict";

   /** @summary draw RText object
     * @memberof JSROOT.v7
     * @private */
   function drawText() {
      let text      = this.getObject(),
          pp        = this.getPadPainter(),
          onframe   = this.v7EvalAttr("onFrame", false) ? pp.getFramePainter() : null,
          clipping  = onframe ? this.v7EvalAttr("clipping", false) : false,
          p         = pp.getCoordinate(text.fPos, onframe),
          textFont  = this.v7EvalFont("text", { size: 12, color: "black", align: 22 });

      this.createG(clipping ? "main_layer" : (onframe ? "upper_layer" : false));

      this.startTextDrawing(textFont, 'font');

      this.drawText({ x: p.x, y: p.y, text: text.fText, latex: 1 });

      return this.finishTextDrawing();
   }

   /** @summary draw RLine object
     * @memberof JSROOT.v7
     * @private */
   function drawLine() {

       let line         = this.getObject(),
           pp           = this.getPadPainter(),
           onframe      = this.v7EvalAttr("onFrame", false) ? pp.getFramePainter() : null,
           clipping     = onframe ? this.v7EvalAttr("clipping", false) : false,
           p1           = pp.getCoordinate(line.fP1, onframe),
           p2           = pp.getCoordinate(line.fP2, onframe);

       this.createG(clipping ? "main_layer" : (onframe ? "upper_layer" : false));

       this.createv7AttLine();

       this.draw_g
           .append("svg:path")
           .attr("d",`M${p1.x},${p1.y}L${p2.x},${p2.y}`)
           .call(this.lineatt.func);
   }

   /** @summary draw RBox object
     * @memberof JSROOT.v7
     * @private */
   function drawBox() {

       let box          = this.getObject(),
           pp           = this.getPadPainter(),
           onframe      = this.v7EvalAttr("onFrame", false) ? pp.getFramePainter() : null,
           clipping     = onframe ? this.v7EvalAttr("clipping", false) : false,
           p1           = pp.getCoordinate(box.fP1, onframe),
           p2           = pp.getCoordinate(box.fP2, onframe);

    this.createG(clipping ? "main_layer" : (onframe ? "upper_layer" : false));

    this.createv7AttLine("border_");

    this.createv7AttFill();

    this.draw_g
        .append("svg:path")
        .attr("d",`M${p1.x},${p1.y}H${p2.x}V${p2.y}H${p1.x}Z`)
        .call(this.lineatt.func)
        .call(this.fillatt.func);
   }

   /** @summary draw RMarker object
     * @memberof JSROOT.v7
     * @private */
   function drawMarker() {
       let marker       = this.getObject(),
           pp           = this.getPadPainter(),
           onframe      = this.v7EvalAttr("onFrame", false) ? pp.getFramePainter() : null,
           clipping     = onframe ? this.v7EvalAttr("clipping", false) : false,
           p            = pp.getCoordinate(marker.fP, onframe);

       this.createG(clipping ? "main_layer" : (onframe ? "upper_layer" : false));

       this.createv7AttMarker();

       let path = this.markeratt.create(p.x, p.y);

       if (path)
          this.draw_g.append("svg:path")
                     .attr("d", path)
                     .call(this.markeratt.func);
   }

   /**
    * @summary Painter for RLegend class
    *
    * @memberof JSROOT
    * @private
    */

   class RLegendPainter extends JSROOT.RPavePainter {

      /** @summary draw RLegend content */
      drawContent() {
         let legend     = this.getObject(),
             textFont   = this.v7EvalFont("text", { size: 12, color: "black", align: 22 }),
             width      = this.pave_width,
             height     = this.pave_height,
             nlines     = legend.fEntries.length,
             pp         = this.getPadPainter();

         if (legend.fTitle) nlines++;

         if (!nlines || !pp) return Promise.resolve(this);

         let stepy = height / nlines, posy = 0, margin_x = 0.02 * width;

         textFont.setSize(height/(nlines * 1.2));
         this.startTextDrawing(textFont, 'font' );

         if (legend.fTitle) {
            this.drawText({ latex: 1, width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: legend.fTitle });
            posy += stepy;
         }

         for (let i = 0; i < legend.fEntries.length; ++i) {
            let objp = null, entry = legend.fEntries[i], w4 = Math.round(width/4);

            this.drawText({ latex: 1, width: 0.75*width - 3*margin_x, height: stepy, x: 2*margin_x + w4, y: posy, text: entry.fLabel });

            if (entry.fDrawableId != "custom") {
               objp = pp.findSnap(entry.fDrawableId, true);
            } else if (entry.fDrawable.fIO) {
               objp = new JSROOT.RObjectPainter(this.getDom(), entry.fDrawable.fIO);
               if (entry.fLine) objp.createv7AttLine();
               if (entry.fFill) objp.createv7AttFill();
               if (entry.fMarker) objp.createv7AttMarker();
            }

            if (objp && entry.fFill && objp.fillatt)
               this.draw_g
                 .append("svg:path")
                 .attr("d", `M${Math.round(margin_x)},${Math.round(posy + stepy*0.1)}h${w4}v${Math.round(stepy*0.8)}h${-w4}z`)
                 .call(objp.fillatt.func);

            if (objp && entry.fLine && objp.lineatt)
               this.draw_g
                 .append("svg:path")
                 .attr("d", `M${Math.round(margin_x)},${Math.round(posy + stepy/2)}h${w4}`)
                 .call(objp.lineatt.func);

            if (objp && entry.fError && objp.lineatt)
               this.draw_g
                 .append("svg:path")
                 .attr("d", `M${Math.round(margin_x + width/8)},${Math.round(posy + stepy*0.2)}v${Math.round(stepy*0.6)}`)
                 .call(objp.lineatt.func);

            if (objp && entry.fMarker && objp.markeratt)
               this.draw_g.append("svg:path")
                   .attr("d", objp.markeratt.create(margin_x + width/8, posy + stepy/2))
                   .call(objp.markeratt.func);

            posy += stepy;
         }

         return this.finishTextDrawing();
      }

      /** @summary draw RLegend object */
      static draw(dom, legend, opt) {
         let painter = new RLegendPainter(dom, legend, opt, "legend");

        return jsrp.ensureRCanvas(painter, false).then(() => painter.drawPave());
      }

   } // class RLegendPainter


   /**
    * @summary Painter for RPaveText class
    *
    * @memberof JSROOT
    * @private
    */

   class RPaveTextPainter extends JSROOT.RPavePainter {

      /** @summary draw RPaveText content */
      drawContent() {
         let pavetext  = this.getObject(),
             textFont  = this.v7EvalFont("text", { size: 12, color: "black", align: 22 }),
             width     = this.pave_width,
             height    = this.pave_height,
             nlines    = pavetext.fText.length;

         if (!nlines) return;

         let stepy = height / nlines, posy = 0, margin_x = 0.02 * width;

         textFont.setSize(height/(nlines * 1.2))

         this.startTextDrawing(textFont, 'font');

         for (let i = 0; i < pavetext.fText.length; ++i) {
            let line = pavetext.fText[i];

            this.drawText({ latex: 1, width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: line });
            posy += stepy;
         }

         return this.finishTextDrawing(undefined, true);
      }

      /** @summary draw RPaveText object */
      static draw(dom, pave, opt) {
         let painter = new RPaveTextPainter(dom, pave, opt, "pavetext");

         return jsrp.ensureRCanvas(painter, false).then(() => painter.drawPave());
      }

   } // class RPaveTextPainter


   JSROOT.v7.drawText      = drawText;
   JSROOT.v7.drawLine      = drawLine;
   JSROOT.v7.drawBox       = drawBox;
   JSROOT.v7.drawMarker    = drawMarker;
   JSROOT.RLegendPainter   = RLegendPainter;
   JSROOT.RPaveTextPainter = RPaveTextPainter;

   if (JSROOT.nodejs) module.exports = JSROOT;
   return JSROOT;

});
