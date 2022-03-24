/// more ROOT classes

import { create } from '../core.mjs';

import { ObjectPainter } from '../base/ObjectPainter.mjs';

import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';

import { drawTLine } from './more.mjs';


/**
 * @summary Painter class for TRatioPlot
 *
 * @private
 */

class TRatioPlotPainter extends ObjectPainter {

   /** @summary Set grids range */
   setGridsRange(xmin, xmax) {
      let ratio = this.getObject(),
          pp = this.getPadPainter();
      if (xmin === xmax) {
         let low_p = pp.findPainterFor(ratio.fLowerPad, "lower_pad", "TPad"),
             low_fp = low_p ? low_p.getFramePainter() : null;
         if (!low_fp || !low_fp.x_handle) return;
         xmin = low_fp.x_handle.full_min;
         xmax = low_fp.x_handle.full_max;
      }

      ratio.fGridlines.forEach(line => {
         line.fX1 = xmin;
         line.fX2 = xmax;
      });
   }

   /** @summary Redraw TRatioPlot */
   redraw() {
      let ratio = this.getObject(),
          pp = this.getPadPainter();

      let top_p = pp.findPainterFor(ratio.fTopPad, "top_pad", "TPad");
      if (top_p) top_p.disablePadDrawing();

      let up_p = pp.findPainterFor(ratio.fUpperPad, "upper_pad", "TPad"),
          up_main = up_p ? up_p.getMainPainter() : null,
          up_fp = up_p ? up_p.getFramePainter() : null,
          low_p = pp.findPainterFor(ratio.fLowerPad, "lower_pad", "TPad"),
          low_main = low_p ? low_p.getMainPainter() : null,
          low_fp = low_p ? low_p.getFramePainter() : null,
          lbl_size = 20, promise_up = Promise.resolve(true);

      if (up_p && up_main && up_fp && low_fp && !up_p._ratio_configured) {
         up_p._ratio_configured = true;
         up_main.options.Axis = 0; // draw both axes

         lbl_size = up_main.getHisto().fYaxis.fLabelSize;
         if (lbl_size < 1) lbl_size = Math.round(lbl_size*Math.min(up_p.getPadWidth(), up_p.getPadHeight()));

         let h = up_main.getHisto();
         h.fXaxis.fLabelSize = 0; // do not draw X axis labels
         h.fXaxis.fTitle = ""; // do not draw X axis title
         h.fYaxis.fLabelSize = lbl_size;
         h.fYaxis.fTitleSize = lbl_size;

         up_p.getRootPad().fTicky = 1;

         promise_up = up_p.redrawPad().then(() => {
            up_fp.o_zoom = up_fp.zoom;
            up_fp._ratio_low_fp = low_fp;
            up_fp._ratio_painter = this;

            up_fp.zoom = function(xmin,xmax,ymin,ymax,zmin,zmax) {
               this._ratio_painter.setGridsRange(xmin, xmax);
               this._ratio_low_fp.o_zoom(xmin,xmax);
               return this.o_zoom(xmin,xmax,ymin,ymax,zmin,zmax);
            }

            up_fp.o_sizeChanged = up_fp.sizeChanged;
            up_fp.sizeChanged = function() {
               this.o_sizeChanged();
               this._ratio_low_fp.fX1NDC = this.fX1NDC;
               this._ratio_low_fp.fX2NDC = this.fX2NDC;
               this._ratio_low_fp.o_sizeChanged();
            }
            return true;
         });
      }

      return promise_up.then(() => {

         if (!low_p || !low_main || !low_fp || !up_fp || low_p._ratio_configured)
            return this;

         low_p._ratio_configured = true;
         low_main.options.Axis = 0; // draw both axes
         let h = low_main.getHisto();
         h.fXaxis.fTitle = "x";
         h.fXaxis.fLabelSize = lbl_size;
         h.fXaxis.fTitleSize = lbl_size;
         h.fYaxis.fLabelSize = lbl_size;
         h.fYaxis.fTitleSize = lbl_size;
         low_p.getRootPad().fTicky = 1;

         low_p.forEachPainterInPad(objp => {
            if (typeof objp.testEditable == 'function')
               objp.testEditable(false);
         });

         let arr = [], currpad;

         if ((ratio.fGridlinePositions.length > 0) && (ratio.fGridlines.length < ratio.fGridlinePositions.length)) {
            ratio.fGridlinePositions.forEach(gridy => {
               let found = false;
               ratio.fGridlines.forEach(line => {
                  if ((line.fY1 == line.fY2) && (Math.abs(line.fY1 - gridy) < 1e-6)) found = true;
               });
               if (!found) {
                  let line = create("TLine");
                  line.fX1 = up_fp.scale_xmin;
                  line.fX2 = up_fp.scale_xmax;
                  line.fY1 = line.fY2 = gridy;
                  line.fLineStyle = 2;
                  ratio.fGridlines.push(line);
                  if (currpad === undefined)
                     currpad = this.selectCurrentPad(ratio.fLowerPad.fName);
                  arr.push(drawTLine(this.getDom(), line));
               }
            });
         }

         return Promise.all(arr).then(() => low_fp.zoom(up_fp.scale_xmin,  up_fp.scale_xmax)).then(() => {

            low_fp.o_zoom = low_fp.zoom;
            low_fp._ratio_up_fp = up_fp;
            low_fp._ratio_painter = this;

            low_fp.zoom = function(xmin,xmax,ymin,ymax,zmin,zmax) {
               this._ratio_painter.setGridsRange(xmin, xmax);
               this._ratio_up_fp.o_zoom(xmin,xmax);
               return this.o_zoom(xmin,xmax,ymin,ymax,zmin,zmax);
            }

            low_fp.o_sizeChanged = low_fp.sizeChanged;
            low_fp.sizeChanged = function() {
               this.o_sizeChanged();
               this._ratio_up_fp.fX1NDC = this.fX1NDC;
               this._ratio_up_fp.fX2NDC = this.fX2NDC;
               this._ratio_up_fp.o_sizeChanged();
            }
            return this;
         });
      });
   }

   /** @summary Draw TRatioPlot */
   static draw(dom, ratio, opt) {
      let painter = new TRatioPlotPainter(dom, ratio, opt);

      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TRatioPlotPainter

export { TRatioPlotPainter }
