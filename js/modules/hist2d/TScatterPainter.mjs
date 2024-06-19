import { clTPaletteAxis, isFunc, create } from '../core.mjs';
import { getColorPalette } from '../base/colors.mjs';
import { TAttMarkerHandler } from '../base/TAttMarkerHandler.mjs';
import { TGraphPainter } from './TGraphPainter.mjs';
import { HistContour } from './THistPainter.mjs';
import { TH2Painter } from './TH2Painter.mjs';


class TScatterPainter extends TGraphPainter {

   constructor(dom, obj) {
      super(dom, obj);
      this._need_2dhist = true;
      this._not_adjust_hrange = true;
   }

   /** @summary Return drawn graph object */
   getGraph() { return this.getObject()?.fGraph; }

   /** @summary Return margins for histogram ranges */
   getHistRangeMargin() { return this.getObject()?.fMargin ?? 0.1; }

  /** @summary Draw axis histogram
    * @private */
   async drawAxisHisto() {
      const histo = this.createHistogram();
      return TH2Painter.draw(this.getDom(), histo, this.options.Axis + ';IGNORE_PALETTE');
   }

  /** @summary Provide palette, create if necessary
    * @private */
   getPalette() {
      const gr = this.getGraph();
      let pal = gr?.fFunctions?.arr?.find(func => (func._typename === clTPaletteAxis));

      if (pal) return pal;

      if (gr) {
         pal = create(clTPaletteAxis);

         const fp = this.get_main();
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

   /** @summary Actual drawing of TScatter */
   async drawGraph() {
      const fpainter = this.get_main(),
          hpainter = this.getMainPainter(),
          scatter = this.getObject();
      let scale = 1, offset = 0;
      if (!fpainter || !hpainter || !scatter) return;

      if (scatter.fColor) {
         const pal = this.getPalette();
         if (pal)
            pal.$main_painter = this;

         const pp = this.getPadPainter();
         if (!this._color_palette && isFunc(pp?.getCustomPalette))
            this._color_palette = pp.getCustomPalette();
         if (!this._color_palette)
            this._color_palette = getColorPalette(this.options.Palette, pp?.isGrayscale());

         let minc = scatter.fColor[0], maxc = scatter.fColor[0];
         for (let i = 1; i < scatter.fColor.length; ++i) {
             minc = Math.min(minc, scatter.fColor[i]);
             maxc = Math.max(maxc, scatter.fColor[i]);
         }
         if (maxc <= minc)
            maxc = minc < 0 ? 0.9*minc : (minc > 0 ? 1.1*minc : 1);
         this.fContour = new HistContour(minc, maxc);
         this.fContour.createNormal(30);
         this.fContour.configIndicies(0, 0);

         fpainter.zmin = minc;
         fpainter.zmax = maxc;
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

      this.createG(!fpainter.pad_layer);

      const funcs = fpainter.getGrFuncs();

      for (let i = 0; i < this.bins.length; ++i) {
         const pnt = this.bins[i],
               grx = funcs.grx(pnt.x),
               gry = funcs.gry(pnt.y),
               size = scatter.fSize ? scatter.fMinMarkerSize + scale * (scatter.fSize[i] - offset) : scatter.fMarkerSize,
               color = scatter.fColor ? this.fContour.getPaletteColor(this._color_palette, scatter.fColor[i]) : this.getColor(scatter.fMarkerColor),
               handle = new TAttMarkerHandler({ color, size, style: scatter.fMarkerStyle });

          this.draw_g.append('svg:path')
                     .attr('d', handle.create(grx, gry))
                     .call(handle.func);
      }

      return this;
   }

   static async draw(dom, obj, opt) {
      return TGraphPainter._drawGraph(new TScatterPainter(dom, obj), opt);
   }

} // class TScatterPainter

export { TScatterPainter };
