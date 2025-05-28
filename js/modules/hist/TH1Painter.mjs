import { settings, gStyle } from '../core.mjs';
import { assignFrame3DMethods, drawBinsLego } from './hist3d.mjs';
import { TAxisPainter } from '../gpad/TAxisPainter.mjs';
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
            histo = this.getHisto(),
            o = this.getOptions(),
            zmult = 1 + 2*gStyle.fHistTopMargin;
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

         if (is_main) {
            assignFrame3DMethods(fp);
            pr = fp.create3DScene(o.Render3D, o.x3dscale, o.y3dscale, o.Ortho).then(() => {
               fp.setAxesRanges(histo.fXaxis, this.xmin, this.xmax, histo.fYaxis, this.ymin, this.ymax, histo.fZaxis, 0, 0, this);
               fp.set3DOptions(o);
               fp.drawXYZ(fp.toplevel, TAxisPainter, {
                  ndim: 1, hist_painter: this, use_y_for_z: true, zmult, zoom: settings.Zooming,
                  draw: (o.Axis !== -1), drawany: o.isCartesian()
               });
            });
         }

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

   /** @summary draw TH1 object */
   static async draw(dom, histo, opt) {
      return THistPainter._drawHist(new TH1Painter(dom, histo), opt);
   }

} // class TH1Painter

export { TH1Painter };
