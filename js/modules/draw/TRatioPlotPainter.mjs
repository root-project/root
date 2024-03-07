import { create, clTPad, clTLine, isFunc } from '../core.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';
import { TLinePainter } from './TLinePainter.mjs';


/**
 * @summary Painter class for TRatioPlot
 *
 * @private
 */

class TRatioPlotPainter extends ObjectPainter {

   /** @summary Set grids range */
   setGridsRange(xmin, xmax, ymin, ymax, low_p) {
      const ratio = this.getObject();
      if (xmin === xmax) {
         const x_handle = this.getPadPainter()?.findPainterFor(ratio.fLowerPad, 'lower_pad', clTPad)?.getFramePainter()?.x_handle;
         if (!x_handle) return;
         if (xmin === 0) {
            // in case of unzoom full range should be used
            xmin = x_handle.full_min;
            xmax = x_handle.full_max;
         } else {
            // in case of y-scale zooming actual range has to be used
            xmin = x_handle.scale_min;
            xmax = x_handle.scale_max;
         }
      }
      ratio.fGridlines.forEach(line => {
         line.fX1 = xmin;
         line.fX2 = xmax;
      });
      const nlines = Math.min(ratio.fGridlines.length, ratio.fGridlinePositions.length);
      for (let i = 0; i < nlines; ++i) {
         const y = ratio.fGridlinePositions[i],
               line = ratio.fGridlines[i];
         if (ymin !== 'ignorey') {
            line.$do_not_draw = (ymin !== ymax) && ((y < ymin) || (y > ymax));
            line.fY1 = line.fY2 = y;
         }
         low_p?.findPainterFor(line)?.redraw();
      }
   }

   /** @summary Configure custom interactive handlers for ratio plot
    * @desc Should work for both new and old code */
   configureInteractive() {
      const ratio = this.getObject(),
            pp = this.getPadPainter(),
            up_p = pp.findPainterFor(ratio.fUpperPad, 'upper_pad', clTPad),
            up_fp = up_p?.getFramePainter(),
            low_p = pp.findPainterFor(ratio.fLowerPad, 'lower_pad', clTPad),
            low_fp = low_p?.getFramePainter();

      if (!up_p || !low_p)
         return;

      low_p.forEachPainterInPad(objp => {
         if (isFunc(objp?.testEditable))
            objp.testEditable(false);
      });

      this.setGridsRange(low_fp.scale_xmin, low_fp.scale_xmax, low_fp.scale_ymin, low_fp.scale_ymax, low_p);

      if (up_p._ratio_interactive && low_p._ratio_interactive)
         return;

      up_p._ratio_interactive = true;
      low_p._ratio_interactive = true;

      up_fp.o_zoom = up_fp.zoom;
      up_fp._ratio_low_fp = low_fp;
      up_fp._ratio_painter = this;

      up_fp.zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {
         return this.o_zoom(xmin, xmax, ymin, ymax, zmin, zmax).then(res => {
            this._ratio_painter.setGridsRange(up_fp.scale_xmin, up_fp.scale_xmax, 'ignory');
            return this._ratio_low_fp.o_zoom(up_fp.scale_xmin, up_fp.scale_xmax).then(() => res);
         });
      };

      up_fp.o_sizeChanged = up_fp.sizeChanged;
      up_fp.sizeChanged = function() {
         this.o_sizeChanged();
         this._ratio_low_fp.fX1NDC = this.fX1NDC;
         this._ratio_low_fp.fX2NDC = this.fX2NDC;
         this._ratio_low_fp.o_sizeChanged();
      };

      low_fp.o_zoom = low_fp.zoom;
      low_fp._ratio_up_fp = up_fp;
      low_fp._ratio_painter = this;

      low_fp.zoom = function(xmin, xmax, ymin, ymax, zmin, zmax) {
         if (xmin === xmax) {
            xmin = up_fp.xmin;
            xmax = up_fp.xmax;
         } else {
            if (xmin < up_fp.xmin) xmin = up_fp.xmin;
            if (xmax > up_fp.xmax) xmax = up_fp.xmax;
         }
         this._ratio_painter.setGridsRange(xmin, xmax, ymin, ymax);
         return this._ratio_up_fp.o_zoom(xmin, xmax).then(() => this.o_zoom(xmin, xmax, ymin, ymax, zmin, zmax));
      };

      low_fp.o_sizeChanged = low_fp.sizeChanged;
      low_fp.sizeChanged = function() {
         this.o_sizeChanged();
         this._ratio_up_fp.fX1NDC = this.fX1NDC;
         this._ratio_up_fp.fX2NDC = this.fX2NDC;
         this._ratio_up_fp.o_sizeChanged();
      };
   }

   /** @summary Redraw old TRatioPlot where object was in very end of list of primitives */
   async redrawOld() {
      const ratio = this.getObject(),
            pp = this.getPadPainter(),
            top_p = pp.findPainterFor(ratio.fTopPad, 'top_pad', clTPad),
            pad = pp.getRootPad(),
            mirrow_axis = (pad.fFrameFillStyle === 0) ? 1 : 0,
            tick_x = pad.fTickx || mirrow_axis,
            tick_y = pad.fTicky || mirrow_axis;

      top_p?.disablePadDrawing();

      const up_p = pp.findPainterFor(ratio.fUpperPad, 'upper_pad', clTPad),
            up_main = up_p?.getMainPainter(),
            up_fp = up_p?.getFramePainter(),
            low_p = pp.findPainterFor(ratio.fLowerPad, 'lower_pad', clTPad),
            low_main = low_p?.getMainPainter(),
            low_fp = low_p?.getFramePainter();
      let promise_up = Promise.resolve(true);

      if (up_p && up_main && up_fp && low_fp && !up_p._ratio_configured) {
         up_p._ratio_configured = true;

         up_main.options.Axis = 0; // draw both axes

         const h = up_main.getHisto();

         h.fYaxis.$use_top_pad = true; // workaround to use same scaling
         h.fXaxis.fLabelSize = 0; // do not draw X axis labels
         h.fXaxis.fTitle = ''; // do not draw X axis title

         up_p.getRootPad().fTickx = tick_x;
         up_p.getRootPad().fTicky = tick_y;

         promise_up = up_p.redrawPad();
      }

      return promise_up.then(() => {
         if (!low_p || !low_main || !low_fp || !up_fp || low_p._ratio_configured)
            return this;

         low_p._ratio_configured = true;
         low_main.options.Axis = 0; // draw both axes
         const h = low_main.getHisto();
         h.fXaxis.fTitle = 'x';

         h.fXaxis.$use_top_pad = true;
         h.fYaxis.$use_top_pad = true;
         low_p.getRootPad().fTickx = tick_x;
         low_p.getRootPad().fTicky = tick_y;

         const arr = [];
         let currpad;

         // add missing lines in old ratio painter
         if ((ratio.fGridlinePositions.length > 0) && (ratio.fGridlines.length < ratio.fGridlinePositions.length)) {
            ratio.fGridlinePositions.forEach(gridy => {
               let found = false;
               ratio.fGridlines.forEach(line => {
                  if ((line.fY1 === line.fY2) && (Math.abs(line.fY1 - gridy) < 1e-6)) found = true;
               });
               if (!found) {
                  const line = create(clTLine);
                  line.fX1 = up_fp.scale_xmin;
                  line.fX2 = up_fp.scale_xmax;
                  line.fY1 = line.fY2 = gridy;
                  line.fLineStyle = 2;
                  ratio.fGridlines.push(line);
                  if (currpad === undefined)
                     currpad = this.selectCurrentPad(ratio.fLowerPad.fName);
                  arr.push(TLinePainter.draw(this.getDom(), line));
               }
            });
         }

         return Promise.all(arr).then(() => {
            if (currpad !== undefined)
               this.selectCurrentPad(currpad);
            return low_fp.zoom(up_fp.scale_xmin, up_fp.scale_xmax);
         });
      });
   }

   /** @summary Redraw TRatioPlot */
   async redraw() {
      const ratio = this.getObject(),
            pp = this.getPadPainter();

      if (this.$oldratio === undefined)
         this.$oldratio = !!pp.findPainterFor(ratio.fTopPad, 'top_pad', clTPad);

      // configure ratio interactive at the end
      pp.$userInteractive = () => this.configureInteractive();

      if (this.$oldratio)
         return this.redrawOld();

      const pad = pp.getRootPad(),
            mirrow_axis = (pad.fFrameFillStyle === 0) ? 1 : 0,
            tick_x = pad.fTickx || mirrow_axis,
            tick_y = pad.fTicky || mirrow_axis;

      // do not draw primitives and pad itself
      ratio.fTopPad.$disable_drawing = true;

      ratio.fUpperPad.$ratio_pad = 'up'; // indicate drawing of the axes for main painter
      ratio.fUpperPad.fTickx = tick_x;
      ratio.fUpperPad.fTicky = tick_y;

      ratio.fLowerPad.$ratio_pad = 'low'; // indicate drawing of the axes for main painter
      ratio.fLowerPad.fTickx = tick_x;
      ratio.fLowerPad.fTicky = tick_y;

      return this;
   }

   /** @summary Draw TRatioPlot */
   static async draw(dom, ratio, opt) {
      const painter = new TRatioPlotPainter(dom, ratio, opt);

      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }

} // class TRatioPlotPainter

export { TRatioPlotPainter };
