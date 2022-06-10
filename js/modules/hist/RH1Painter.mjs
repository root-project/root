import { settings, gStyle } from '../core.mjs';
import { RH1Painter as RH1Painter2D } from '../hist2d/RH1Painter.mjs';
import { RAxisPainter } from '../gpad/RAxisPainter.mjs';
import { assignFrame3DMethods, drawBinsLego } from './hist3d.mjs';

class RH1Painter extends RH1Painter2D {

   /** @summary Draw 1-D histogram in 3D mode */
   draw3D(reason) {

      this.mode3d = true;

      let main = this.getFramePainter(), // who makes axis drawing
          is_main = this.isMainPainter(), // is main histogram
          zmult = 1 + 2*gStyle.fHistTopMargin;

      if (reason == "resize")  {
         if (is_main && main.resize3D()) main.render3D();
         return Promise.resolve(this);
      }

      this.deleteAttr();

      this.scanContent(true); // may be required for axis drawings

      if (is_main) {
         assignFrame3DMethods(main);
         main.create3DScene(this.options.Render3D);
         main.setAxesRanges(this.getAxis("x"), this.xmin, this.xmax, null, this.ymin, this.ymax, null, 0, 0);
         main.set3DOptions(this.options);
         main.drawXYZ(main.toplevel, RAxisPainter, { use_y_for_z: true, zmult, zoom: settings.Zooming, ndim: 1, draw: true, v7: true });
      }

      if (!main.mode3d)
         return Promise.resolve(this);

      return this.drawingBins(reason).then(() => {

         // called when bins received from server, must be reentrant
         let main = this.getFramePainter();

         drawBinsLego(this, true);
         this.updatePaletteDraw();
         main.render3D();
         main.addKeysHandler();
         return this;
      });
   }

      /** @summary draw RH1 object */
   static draw(dom, histo, opt) {
      return RH1Painter._draw(new RH1Painter(dom, histo), opt);
   }

} // class RH1Painter

export { RH1Painter };

