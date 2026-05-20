import { gStyle } from '../core.mjs';
import { crete3DFrame, drawBinsLego } from './hist3d.mjs';
import { TAxisPainter } from '../gpad/TAxisPainter.mjs';
import { TFramePainter } from '../gpad/TFramePainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';
import { TH1Painter as TH1Painter2D } from '../hist2d/TH1Painter.mjs';


/** @summary Draw 1-D histogram in 3D
  * @private */

class TH1Painter extends TH1Painter2D {

   /** @summary draw TH1 object in 3D mode */
   async draw3D(reason) {
      this.mode3d = true;

      const fp = this.getFramePainter(), // who makes axis drawing
            is_main = this.isMainPainter(), // is main histogram
            o = this.getOptions();

      o.zmult = 1 + 2 * gStyle.fHistTopMargin;
      let pr = Promise.resolve(true), full_draw = true;

      if (reason === 'resize') {
         const res = is_main ? fp.resize3D() : false;
         if (res !== 1) {
            full_draw = false;
            if (res)
               fp.render3D();
         }
      }

      if (full_draw) {
         this.createHistDrawAttributes(true);

         this.scanContent(reason === 'zoom'); // may be required for axis drawings

         if (is_main)
            pr = crete3DFrame(this, TAxisPainter, o.Render3D);

         if (fp.mode3d) {
            pr = pr.then(() => {
               drawBinsLego(this);
               fp.render3D();
               this.updateStatWebCanvas();
               fp.addKeysHandler();
            });
         }
      }

      if (is_main)
         pr = pr.then(() => this.drawColorPalette(o.Zscale && o.canHavePalette()));

      return pr.then(() => this.updateFunctions())
               .then(() => this.updateHistTitle())
               .then(() => this);
   }

   /** @summary Build three.js object for the histogram */
   static async build3d(histo, opt) {
      const painter = new TH1Painter(null, histo);
      painter.decodeOptions(opt);
      painter.scanContent();

      painter.createHistDrawAttributes(true);
      painter.options.zmult = 1 + 2 * gStyle.fHistTopMargin;

      const fp = new TFramePainter(null, null);
      painter.getFramePainter = () => fp;

      return crete3DFrame(painter, TAxisPainter)
             .then(() => drawBinsLego(painter))
             .then(() => fp.create3DScene(-1, true));
   }


   /** @summary draw TH1 object */
   static async draw(dom, histo, opt) {
      return THistPainter._drawHist(new TH1Painter(dom, histo), opt);
   }

} // class TH1Painter

export { TH1Painter };
