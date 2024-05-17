import { settings, gStyle } from '../core.mjs';
import { assignFrame3DMethods, drawBinsLego } from './hist3d.mjs';
import { TAxisPainter } from '../gpad/TAxisPainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';
import { TH1Painter as TH1Painter2D } from '../hist2d/TH1Painter.mjs';


/** @summary Draw 1-D histogram in 3D
  * @private */

class TH1Painter extends TH1Painter2D {

   /** @summary draw TH1 object in 3D mode */
   draw3D(reason) {
      this.mode3d = true;

      const main = this.getFramePainter(), // who makes axis drawing
            is_main = this.isMainPainter(), // is main histogram
            histo = this.getHisto(),
            zmult = 1 + 2*gStyle.fHistTopMargin;
      let pr = Promise.resolve(true);

      if (reason === 'resize') {
         if (is_main && main.resize3D()) main.render3D();
      } else {
         this.createHistDrawAttributes(true);

         this.scanContent(true); // may be required for axis drawings

         if (is_main) {
            assignFrame3DMethods(main);
            pr = main.create3DScene(this.options.Render3D, this.options.x3dscale, this.options.y3dscale, this.options.Ortho).then(() => {
               main.setAxesRanges(histo.fXaxis, this.xmin, this.xmax, histo.fYaxis, this.ymin, this.ymax, histo.fZaxis, 0, 0, this);
               main.set3DOptions(this.options);
               main.drawXYZ(main.toplevel, TAxisPainter, { use_y_for_z: true, zmult, zoom: settings.Zooming, ndim: 1,
                  draw: (this.options.Axis !== -1), drawany: this.options.isCartesian() });
            });
         }

         if (main.mode3d) {
            pr = pr.then(() => {
               drawBinsLego(this);
               main.render3D();
               this.updateStatWebCanvas();
               main.addKeysHandler();
            });
         }
      }

      if (is_main)
         pr = pr.then(() => this.drawColorPalette(this.options.Zscale && ((this.options.Lego === 12) || (this.options.Lego === 14))));

      return pr.then(() => this.updateFunctions())
               .then(() => this.updateHistTitle())
               .then(() => this);
   }

   /** @summary draw TH1 object */
   static async draw(dom, histo, opt) {
      return THistPainter._drawHist(new TH1Painter(dom, histo), opt);
   }

} // class TH1Painter

export { TH1Painter };
