import { TGraphPainter as TGraphPainter2D } from '../hist2d/TGraphPainter.mjs';
import { TH1Painter } from './TH1Painter.mjs';
import { createLineSegments, create3DLineMaterial } from '../base/base3d.mjs';


class TGraphPainter extends TGraphPainter2D {

   /** @summary Draw TGraph points in 3D
     * @private */
   drawBins3D(fp, graph) {
      if (!fp.mode3d || !fp.grx || !fp.gry || !fp.grz || !fp.toplevel)
         return console.log('Frame painter missing base 3d elements');

      if (fp.zoom_xmin !== fp.zoom_xmax)
        if ((this.options.pos3d < fp.zoom_xmin) || (this.options.pos3d > fp.zoom_xmax)) return;

      this.createGraphDrawAttributes(true);

      const drawbins = this.optimizeBins(1000);
      let first = 0, last = drawbins.length-1;

      if (fp.zoom_ymin !== fp.zoom_ymax) {
         while ((first < last) && (drawbins[first].x < fp.zoom_ymin)) first++;
         while ((first < last) && (drawbins[last].x > fp.zoom_ymax)) last--;
      }

      if (first === last) return;

      const pnts = [], grx = fp.grx(this.options.pos3d);
      let p0 = drawbins[first];

      for (let n = first + 1; n <= last; ++n) {
         const p1 = drawbins[n];
         pnts.push(grx, fp.gry(p0.x), fp.grz(p0.y),
                   grx, fp.gry(p1.x), fp.grz(p1.y));
         p0 = p1;
      }

      const lines = createLineSegments(pnts, create3DLineMaterial(this, graph));

      fp.add3DMesh(lines, this, true);

      fp.render3D(100);
   }

   /** @summary Draw axis histogram
     * @private */
   async drawAxisHisto() {
      return TH1Painter.draw(this.getDom(), this.createHistogram(), this.options.Axis);
   }

   static async draw(dom, graph, opt) {
      return TGraphPainter._drawGraph(new TGraphPainter(dom, graph), opt);
   }

} // class TGraphPainter

export { TGraphPainter };
