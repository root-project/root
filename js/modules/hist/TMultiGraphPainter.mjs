import { TMultiGraphPainter as TMultiGraphPainter2D } from '../hist2d/TMultiGraphPainter.mjs';
import { TH1Painter } from '../hist2d/TH1Painter.mjs';
import { TH2Painter } from './TH2Painter.mjs';
import { TGraphPainter } from './TGraphPainter.mjs';


class TMultiGraphPainter extends TMultiGraphPainter2D {

   /** @summary draw speical histogram for axis
     * @returns {Promise} when ready */
   drawAxisHist(histo, hopt) {
      return this._3d ? TH2Painter.draw(this.getDom(), histo, "AXIS3D" + hopt)
                      : TH1Painter.draw(this.getDom(), histo, "AXIS" + hopt);
   }

   drawGraph(gr, opt, pos3d) {
      if (this._3d) opt += "pos3d_"+pos3d;
      return TGraphPainter.draw(this.getDom(), gr, opt);
   }

   /** @summary Draw TMultiGraph object */
   static draw(dom, mgraph, opt) {
      return TMultiGraphPainter._drawMG(new TMultiGraphPainter(dom, mgraph), opt);
   }

} // class TMultiGraphPainter

export { TMultiGraphPainter };
