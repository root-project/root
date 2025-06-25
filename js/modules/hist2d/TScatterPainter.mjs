import { clTPaletteAxis, isFunc, create, kNoZoom } from '../core.mjs';
import { getColorPalette } from '../base/colors.mjs';
import { TAttMarkerHandler } from '../base/TAttMarkerHandler.mjs';
import { TGraphPainter } from './TGraphPainter.mjs';
import { HistContour } from './THistPainter.mjs';
import { TH2Painter } from './TH2Painter.mjs';

/**
 * @summary Painter for TScatter object.
 *
 * @private
 */

class TScatterPainter extends TGraphPainter {

   #color_palette; // color palette
   #contour; // colors contour

   /** @summary Cleanup painter */
   cleanup() {
      this.clearHistPalette();
      this.#contour = undefined;
      super.cleanup();
   }

   /** @summary Return drawn graph object */
   getGraph() { return this.getObject()?.fGraph; }

   /** @summary Return colors contour */
   getContour() { return this.#contour; }

   /** @summary Is TScatter object */
   isScatter() { return true; }

   /** @summary Return margins for histogram ranges */
   getHistRangeMargin() { return this.getObject()?.fMargin ?? 0.1; }

  /** @summary Draw axis histogram
    * @private */
   async drawAxisHisto() {
      const need_histo = !this.getHistogram(),
            histo = this.createHistogram(need_histo, need_histo);
      return TH2Painter.draw(this.getDrawDom(), histo, this.getOptions().Axis + ';IGNORE_PALETTE');
   }

  /** @summary Provide palette, create if necessary
    * @private */
   getPalette() {
      const gr = this.getGraph();
      let pal = gr?.fFunctions?.arr?.find(func => (func._typename === clTPaletteAxis));

      if (!pal && gr) {
         pal = create(clTPaletteAxis);

         const fp = this.get_fp();
         Object.assign(pal, { fX1NDC: fp.fX2NDC + 0.005, fX2NDC: fp.fX2NDC + 0.05, fY1NDC: fp.fY1NDC, fY2NDC: fp.fY2NDC, fInit: 1, $can_move: true });
         Object.assign(pal.fAxis, { fChopt: '+', fLineColor: 1, fLineSyle: 1, fLineWidth: 1, fTextAngle: 0, fTextAlign: 11, fNdiv: 510 });
         gr.fFunctions.AddFirst(pal, '');
      }

      return pal;
   }

   /** @summary Update TScatter members
    * @private */
   _updateMembers(scatter, obj) {
      scatter.fBits = obj.fBits;
      scatter.fTitle = obj.fTitle;
      scatter.fNpoints = obj.fNpoints;
      scatter.fColor = obj.fColor;
      scatter.fSize = obj.fSize;
      scatter.fMargin = obj.fMargin;
      scatter.fMinMarkerSize = obj.fMinMarkerSize;
      scatter.fMaxMarkerSize = obj.fMaxMarkerSize;
      return super._updateMembers(scatter.fGraph, obj.fGraph);
   }

   /** @summary Return Z axis used for palette drawing
    * @private */
   getZaxis() {
      return this.getHistogram()?.fZaxis;
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      if (axis !== 'z')
         return super.canZoomInside(axis, min, max);

      const levels = this.#contour?.getLevels();
      if (!levels)
         return false;
      // match at least full color level inside
      for (let i = 0; i < levels.length - 1; ++i) {
         if ((min <= levels[i]) && (max >= levels[i+1]))
            return true;
      }
      return false;
   }

   /** @summary Returns color palette associated with histogram
    * @desc Create if required, checks pad and canvas for custom palette */
   getHistPalette(force) {
      let pal = force ? null : this.#color_palette;
      if (pal)
         return pal;
      const pp = this.getPadPainter();
      if (isFunc(pp?.getCustomPalette))
         pal = pp.getCustomPalette();
      if (!pal)
         pal = getColorPalette(this.getOptions().Palette, pp?.isGrayscale());
      this.#color_palette = pal;
      return pal;
   }

   /** @summary Remove palette */
   clearHistPalette() {
      this.#color_palette = undefined;
   }

   /** @summary Actual drawing of TScatter */
   async drawGraph() {
      const fp = this.get_fp(),
            hpainter = this.getMainPainter(),
            scatter = this.getObject(),
            hist = this.getHistogram();

      let scale = 1, offset = 0, palette;
      if (!fp || !hpainter || !scatter)
         return;

      if (scatter.fColor) {
         const pal = this.getPalette();
         if (pal)
            pal.$main_painter = this;

         palette = this.getHistPalette();

         let minc = scatter.fColor[0], maxc = scatter.fColor[0];
         for (let i = 1; i < scatter.fColor.length; ++i) {
             minc = Math.min(minc, scatter.fColor[i]);
             maxc = Math.max(maxc, scatter.fColor[i]);
         }
         if (maxc <= minc)
            maxc = minc < 0 ? 0.9*minc : (minc > 0 ? 1.1*minc : 1);
         else if ((minc > 0) && (minc < 0.3*maxc))
            minc = 0;
         this.#contour = new HistContour(minc, maxc);
         this.#contour.createNormal(30);
         this.#contour.configIndicies(0, 0);

         fp.zmin = minc;
         fp.zmax = maxc;

         if (!fp.zoomChangedInteractive('z') && hist && hist.fMinimum !== kNoZoom && hist.fMaximum !== kNoZoom) {
            fp.zoom_zmin = hist.fMinimum;
            fp.zoom_zmax = hist.fMaximum;
         }
      }

      if (scatter.fSize) {
         let mins = scatter.fSize[0], maxs = scatter.fSize[0];

         for (let i = 1; i < scatter.fSize.length; ++i) {
             mins = Math.min(mins, scatter.fSize[i]);
             maxs = Math.max(maxs, scatter.fSize[i]);
         }

         if (maxs <= mins)
            maxs = mins < 0 ? 0.9*mins : (mins > 0 ? 1.1*mins : 1);

         scale = (scatter.fMaxMarkerSize - scatter.fMinMarkerSize) / (maxs - mins);
         offset = mins;
      }

      const g = this.createG(!fp.pad_layer),
            funcs = fp.getGrFuncs(),
            is_zoom = (fp.zoom_zmin !== fp.zoom_zmax) && scatter.fColor,
            bins = this._getBins();

      for (let i = 0; i < bins.length; ++i) {
         if (is_zoom && ((scatter.fColor[i] < fp.zoom_zmin) || (scatter.fColor[i] > fp.zoom_zmax)))
            continue;

         const pnt = bins[i],
               grx = funcs.grx(pnt.x),
               gry = funcs.gry(pnt.y),
               size = scatter.fSize ? scatter.fMinMarkerSize + scale * (scatter.fSize[i] - offset) : scatter.fMarkerSize,
               color = scatter.fColor ? this.#contour.getPaletteColor(palette, scatter.fColor[i]) : this.getColor(scatter.fMarkerColor),
               handle = new TAttMarkerHandler({ color, size, style: scatter.fMarkerStyle });

         g.append('svg:path')
          .attr('d', handle.create(grx, gry))
          .call(handle.func);
      }

      return this;
   }

   /** @summary Draw TScatter object */
   static async draw(dom, obj, opt) {
      return TGraphPainter._drawGraph(new TScatterPainter(dom, obj), opt);
   }

} // class TScatterPainter

export { TScatterPainter };
