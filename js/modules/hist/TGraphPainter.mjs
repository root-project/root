import { TGraphPainter as TGraphPainter2D } from '../hist2d/TGraphPainter.mjs';

import { TH1Painter } from './TH1Painter.mjs';

import { createLineSegments, create3DLineMaterial } from '../base/base3d.mjs';

class TGraphPainter extends TGraphPainter2D {

   drawBins3D(fp, graph) {

      if (!fp.mode3d || !fp.grx || !fp.gry || !fp.grz || !fp.toplevel)
         return console.log('Frame painter missing base 3d elements');

      if (fp.zoom_xmin != fp.zoom_xmax)
        if ((this.options.pos3d < fp.zoom_xmin) || (this.options.pos3d > fp.zoom_xmax)) return;

      let drawbins = this.optimizeBins(1000),
          first = 0, last = drawbins.length-1;

      if (fp.zoom_ymin != fp.zoom_ymax) {
         while ((first < last) && (drawbins[first].x < fp.zoom_ymin)) first++;
         while ((first < last) && (drawbins[last].x > fp.zoom_ymax)) last--;
      }

      if (first == last) return;

      let pnts = [], grx = fp.grx(this.options.pos3d),
          p0 = drawbins[first];

      for (let n = first + 1; n <= last; ++n) {
         let p1 = drawbins[n];
         pnts.push(grx, fp.gry(p0.x), fp.grz(p0.y),
                   grx, fp.gry(p1.x), fp.grz(p1.y));
         p0 = p1;
      }

      let lines = createLineSegments(pnts, create3DLineMaterial(this, graph));

      fp.toplevel.add(lines);

      fp.render3D(100);
   }

   /** @summary Draw axis histogram
     * @private */
   drawAxisHisto() {
      let histo = this.createHistogram();
      return TH1Painter.draw(this.getDom(), histo, this.options.Axis)
   }

   static draw(dom, graph, opt) {
      return TGraphPainter._drawGraph(new TGraphPainter(dom, graph), opt);
   }

}

export { TGraphPainter };
