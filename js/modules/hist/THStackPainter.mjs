import { THStackPainter as THStackPainter2D } from '../hist2d/THStackPainter.mjs';
import { TH1Painter } from './TH1Painter.mjs';
import { TH2Painter } from './TH2Painter.mjs';


class THStackPainter extends THStackPainter2D {

   /** @summary Invoke histogram drawing */
   drawHist(dom, hist, hopt) {
      const func = (this.getOptions().ndim === 1) ? TH1Painter.draw : TH2Painter.draw;
      return func(dom, hist, hopt);
   }

   /** @summary draw THStack object */
   static async draw(dom, stack, opt) {
      if (!stack.fHists || !stack.fHists.arr)
         return null; // drawing not needed

      const painter = new THStackPainter(dom, stack, opt);

      return painter.redrawWith(opt, true);
   }

} // class THStackPainter

export { THStackPainter };
