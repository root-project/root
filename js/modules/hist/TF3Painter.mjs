import { createHistogram, setHistogramTitle, kNoStats, settings, clTF3, clTH2F } from '../core.mjs';
import { TH2Painter } from '../hist/TH2Painter.mjs';
import { proivdeEvalPar } from '../base/func.mjs';
import { produceTAxisLogScale, scanTF1Options } from '../hist/TF1Painter.mjs';
import { ObjectPainter, getElementMainPainter } from '../base/ObjectPainter.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';


function findZValue(arrz, arrv, cross = 0) {
   for (let i = arrz.length - 2; i >= 0; --i) {
      const v1 = arrv[i], v2 = arrv[i + 1],
            z1 = arrz[i], z2 = arrz[i + 1];
      if (v1 === cross) return z1;
      if (v2 === cross) return z2;
      if ((v1 < cross) !== (v2 < cross))
         return z1 + (cross - v1) / (v2 - v1) * (z2 - z1);
   }

   return arrz[0] - 1;
}


/**
  * @summary Painter for TF3 object
  *
  * @private
  */

class TF3Painter extends TH2Painter {

   /** @summary Returns drawn object name */
   getObjectName() { return this.$func?.fName ?? 'func'; }

   /** @summary Returns drawn object class name */
   getClassName() { return this.$func?._typename ?? clTF3; }

   /** @summary Returns true while function is drawn */
   isTF1() { return true; }

   /** @summary Returns primary function which was then drawn as histogram */
   getPrimaryObject() { return this.$func; }

   /** @summary Update histogram */
   updateObject(obj /*, opt */) {
      if (!obj || (this.getClassName() !== obj._typename)) return false;
      delete obj.evalPar;
      const histo = this.getHisto();

      if (this.webcanv_hist) {
         const h0 = this.getPadPainter()?.findInPrimitives('Func', clTH2F);
         if (h0) this.updateAxes(histo, h0, this.getFramePainter());
      }

      this.$func = obj;
      this.createTF3Histogram(obj, histo);
      this.scanContent();
      return true;
   }

   /** @summary Redraw TF2
     * @private */
   redraw(reason) {
      if (!this._use_saved_points && (reason === 'logx' || reason === 'logy' || reason === 'logy' || reason === 'zoom')) {
         this.createTF3Histogram(this.$func, this.getHisto());
         this.scanContent();
      }

      return super.redraw(reason);
   }

   /** @summary Create histogram for TF3 drawing
     * @private */
   createTF3Histogram(func, hist) {
      const nsave = func.fSave.length - 9;

      this._use_saved_points = (nsave > 0) && (settings.PreferSavedPoints || (this.use_saved > 1));

      const fp = this.getFramePainter(),
            pad = this.getPadPainter()?.getRootPad(true),
            logx = pad?.fLogx, logy = pad?.fLogy,
            gr = fp?.getGrFuncs(this.second_x, this.second_y);
      let xmin = func.fXmin, xmax = func.fXmax,
          ymin = func.fYmin, ymax = func.fYmax,
          zmin = func.fZmin, zmax = func.fZmax,
          npx = Math.max(func.fNpx, 20),
          npy = Math.max(func.fNpy, 20),
          npz = Math.max(func.fNpz, 20);

      if (gr?.zoom_xmin !== gr?.zoom_xmax) {
         const dx = (xmax - xmin) / npx;
         if ((xmin < gr.zoom_xmin) && (gr.zoom_xmin < xmax))
            xmin = Math.max(xmin, gr.zoom_xmin - dx);
         if ((xmin < gr.zoom_xmax) && (gr.zoom_xmax < xmax))
            xmax = Math.min(xmax, gr.zoom_xmax + dx);
      }

      if (gr?.zoom_ymin !== gr?.zoom_ymax) {
         const dy = (ymax - ymin) / npy;
         if ((ymin < gr.zoom_ymin) && (gr.zoom_ymin < ymax))
            ymin = Math.max(ymin, gr.zoom_ymin - dy);
         if ((ymin < gr.zoom_ymax) && (gr.zoom_ymax < ymax))
            ymax = Math.min(ymax, gr.zoom_ymax + dy);
      }

      if (gr?.zoom_zmin !== gr?.zoom_zmax) {
         // no need for dz here - TH2 is not binned over Z axis
         if ((zmin < gr.zoom_zmin) && (gr.zoom_zmin < zmax))
            zmin = gr.zoom_zmin;
         if ((zmin < gr.zoom_zmax) && (gr.zoom_zmax < zmax))
            zmax = gr.zoom_zmax;
      }

      const ensureBins = (nx, ny) => {
         if (hist.fNcells !== (nx + 2) * (ny + 2)) {
            hist.fNcells = (nx + 2) * (ny + 2);
            hist.fArray = new Float32Array(hist.fNcells);
         }
         hist.fArray.fill(0);
         hist.fXaxis.fNbins = nx;
         hist.fXaxis.fXbins = [];
         hist.fYaxis.fNbins = ny;
         hist.fYaxis.fXbins = [];
         hist.fXaxis.fXmin = xmin;
         hist.fXaxis.fXmax = xmax;
         hist.fYaxis.fXmin = ymin;
         hist.fYaxis.fXmax = ymax;
         hist.fMinimum = zmin;
         hist.fMaximum = zmax;
      };

      delete this._fail_eval;

      if (!this._use_saved_points) {
         let iserror = false;

         if (!func.evalPar && !proivdeEvalPar(func))
            iserror = true;

         ensureBins(npx, npy);

         if (logx)
            produceTAxisLogScale(hist.fXaxis, npx, xmin, xmax);
         if (logy)
            produceTAxisLogScale(hist.fYaxis, npy, ymin, ymax);

         const arrv = new Array(npz), arrz = new Array(npz);
         for (let k = 0; k < npz; ++k)
            arrz[k] = zmin + k / (npz - 1) * (zmax - zmin);

         for (let j = 0; (j < npy) && !iserror; ++j) {
            for (let i = 0; (i < npx) && !iserror; ++i) {
               const x = hist.fXaxis.GetBinCenter(i+1),
                     y = hist.fYaxis.GetBinCenter(j+1);
               let z = 0;

               try {
                  for (let k = 0; k < npz; ++k)
                     arrv[k] = func.evalPar(x, y, arrz[k]);

                  z = findZValue(arrz, arrv);
               } catch {
                  iserror = true;
               }

               if (!iserror)
                  hist.setBinContent(hist.getBin(i + 1, j + 1), Number.isFinite(z) ? z : 0);
            }
         }

         if (iserror)
            this._fail_eval = true;

         if (iserror && (nsave > 0))
            this._use_saved_points = true;
      }

      if (this._use_saved_points) {
         xmin = func.fSave[nsave]; xmax = func.fSave[nsave+1];
         ymin = func.fSave[nsave+2]; ymax = func.fSave[nsave+3];
         zmin = func.fSave[nsave+4]; zmax = func.fSave[nsave+5];
         npx = Math.round(func.fSave[nsave+6]);
         npy = Math.round(func.fSave[nsave+7]);
         npz = Math.round(func.fSave[nsave+8]);
         // dx = (xmax - xmin) / npx,
         // dy = (ymax - ymin) / npy,
         const dz = (zmax - zmin) / npz;

         ensureBins(npx + 1, npy + 1);

         const arrv = new Array(npz + 1), arrz = new Array(npz + 1);
         for (let k = 0; k <= npz; k++)
            arrz[k] = zmin + k*dz;

         for (let i = 0; i <= npx; ++i) {
            for (let j = 0; j <= npy; ++j) {
               for (let k = 0; k <= npz; k++)
                  arrv[k] = func.fSave[i + (npx + 1)*(j + (npy + 1)*k)];
               const z = findZValue(arrz, arrv);
               hist.setBinContent(hist.getBin(i + 1, j + 1), Number.isFinite(z) ? z : 0);
            }
         }
      }

      hist.fName = 'Func';
      setHistogramTitle(hist, func.fTitle);


      // hist.fMinimum = func.fMinimum;
      // hist.fMaximum = func.fMaximum;
      // fHistogram->SetContour(fContour.fN, levels);
      hist.fLineColor = func.fLineColor;
      hist.fLineStyle = func.fLineStyle;
      hist.fLineWidth = func.fLineWidth;
      hist.fFillColor = func.fFillColor;
      hist.fFillStyle = func.fFillStyle;
      hist.fMarkerColor = func.fMarkerColor;
      hist.fMarkerStyle = func.fMarkerStyle;
      hist.fMarkerSize = func.fMarkerSize;
      hist.fBits |= kNoStats;

      return hist;
   }

   /** @summary Extract function ranges */
   extractAxesProperties(ndim) {
      super.extractAxesProperties(ndim);

      const func = this.$func, nsave = func?.fSave.length ?? 0;

      if (nsave > 9 && this._use_saved_points) {
         this.xmin = Math.min(this.xmin, func.fSave[nsave-9]);
         this.xmax = Math.max(this.xmax, func.fSave[nsave-8]);
         this.ymin = Math.min(this.ymin, func.fSave[nsave-7]);
         this.ymax = Math.max(this.ymax, func.fSave[nsave-6]);
         this.zmin = Math.min(this.zmin, func.fSave[nsave-5]);
         this.zmax = Math.max(this.zmax, func.fSave[nsave-4]);
      }
      if (func) {
         this.xmin = Math.min(this.xmin, func.fXmin);
         this.xmax = Math.max(this.xmax, func.fXmax);
         this.ymin = Math.min(this.ymin, func.fYmin);
         this.ymax = Math.max(this.ymax, func.fYmax);
         this.zmin = Math.min(this.zmin, func.fZmin);
         this.zmax = Math.max(this.zmax, func.fZmax);
      }
   }

   /** @summary fill information for TWebCanvas
    * @desc Used to inform webcanvas when evaluation failed
     * @private */
   fillWebObjectOptions(opt) {
      opt.fcust = this._fail_eval && !this.use_saved ? 'func_fail' : '';
   }

   /** @summary draw TF3 object */
   static async draw(dom, tf3, opt) {
      const web = scanTF1Options(opt);
      opt = web.opt;
      delete web.opt;

      const d = new DrawOptions(opt);
      if (d.empty() || (opt === 'gl'))
         opt = 'surf1';
      else if (d.opt === 'SAME')
         opt = 'surf1 same';

      if ((opt.indexOf('same') === 0) || (opt.indexOf('SAME') === 0)) {
         if (!getElementMainPainter(dom))
            opt = 'A_ADJUST_FRAME_' + opt.slice(4);
      }

      let hist;

      if (web.webcanv_hist) {
         const dummy = new ObjectPainter(dom);
         hist = dummy.getPadPainter()?.findInPrimitives('Func', clTH2F);
      }

      if (!hist) {
         hist = createHistogram(clTH2F, 20, 20);
         hist.fBits |= kNoStats;
      }

      const painter = new TF3Painter(dom, hist);

      painter.$func = tf3;
      Object.assign(painter, web);
      painter.createTF3Histogram(tf3, hist);
      return THistPainter._drawHist(painter, opt);
   }

} // class TF3Painter

export { TF3Painter };
