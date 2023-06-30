import { createHistogram, kNoStats, settings, clTF2, clTH2F, isStr } from '../core.mjs';
import { TH2Painter } from '../hist/TH2Painter.mjs';
import { proivdeEvalPar } from '../hist/TF1Painter.mjs';
import { ObjectPainter, getElementMainPainter } from '../base/ObjectPainter.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';


/**
  * @summary Painter for TF2 object
  *
  * @private
  */

class TF2Painter extends TH2Painter {

   /** @summary Returns drawn object name */
   getObjectName() { return this.$func?.fName ?? 'func'; }

   /** @summary Returns drawn object class name */
   getClassName() { return this.$func?._typename ?? clTF2; }

   /** @summary Returns true while function is drawn */
   isTF1() { return true; }

   /** @summary Update histogram */
   updateObject(obj /*, opt*/) {
      if (!obj || (this.getClassName() != obj._typename)) return false;
      delete obj.evalPar;
      let histo = this.getHisto();

      if (this.webcanv_hist) {
         let h0 = this.getPadPainter()?.findInPrimitives('Func', clTH2F);
         if (h0) {
            histo.fXaxis.fTitle = h0.fXaxis.fTitle;
            histo.fYaxis.fTitle = h0.fYaxis.fTitle;
            histo.fZaxis.fTitle = h0.fZaxis.fTitle;
         }
      }

      this.createTF2Histogram(obj, histo);
      return true;
   };

   /** @summary Create histogram for TF2 drawing
     * @private */
   createTF2Histogram(func, hist = undefined) {
      let nsave = func.fSave.length;
      if ((nsave > 6) && (nsave !== (func.fSave[nsave-2]+1)*(func.fSave[nsave-1]+1) + 6)) nsave = 0;

      let npx = Math.max(func.fNpx, 2),
          npy = Math.max(func.fNpy, 2),
          iserr = false, isany = false,
          dx = (func.fXmax - func.fXmin) / npx,
          dy = (func.fYmax - func.fYmin) / npy,
          use_saved_points = (nsave > 6) && settings.PreferSavedPoints;

      const ensureBins = (nx, ny) => {
         if (hist.fNcells !== (nx + 2) * (ny + 2)) {
            hist.fNcells = (nx + 2) * (ny + 2);
            hist.fArray = new Float32Array(hist.fNcells);
            hist.fArray.fill(0);
         }
         hist.fXaxis.fNbins = nx;
         hist.fYaxis.fNbins = ny;
      }

      if (!use_saved_points) {
         if (!func.evalPar && !proivdeEvalPar(func))
            iserr = true;

         ensureBins(npx, npy);
         hist.fXaxis.fXmin = func.fXmin;
         hist.fXaxis.fXmax = func.fXmax;
         hist.fYaxis.fXmin = func.fYmin;
         hist.fYaxis.fXmax = func.fYmax;

         for (let j = 0; (j < npy) && !iserr; ++j)
            for (let i = 0; (i < npx) && !iserr; ++i) {

               let x = func.fXmin + (i + 0.5) * dx,
                   y = func.fYmin + (j + 0.5) * dy,
                   z = 0;

               try {
                  z = func.evalPar(x, y);
               } catch {
                  iserr = true;
               }

               if (!iserr && Number.isFinite(z)) {
                  if (!hist) hist = createHistogram(clTH2F, npx, npy);
                  isany = true;
                  hist.setBinContent(hist.getBin(i + 1, j + 1), z);
               }
            }

         if ((iserr || !isany) && (nsave > 6))
            use_saved_points = true;
      }

      if (use_saved_points) {
         npx = Math.round(func.fSave[nsave-2]);
         npy = Math.round(func.fSave[nsave-1]);
         dx = (func.fSave[nsave-5] - func.fSave[nsave-6]) / npx;
         dy = (func.fSave[nsave-3] - func.fSave[nsave-4]) / npy;

         ensureBins(npx+1, npy+1);
         hist.fXaxis.fXmin = func.fSave[nsave-6] - dx/2;
         hist.fXaxis.fXmax = func.fSave[nsave-5] + dx/2;
         hist.fYaxis.fXmin = func.fSave[nsave-4] - dy/2;
         hist.fYaxis.fXmax = func.fSave[nsave-3] + dy/2;

         for (let k = 0, j = 0; j <= npy; ++j)
            for (let i = 0; i <= npx; ++i)
               hist.setBinContent(hist.getBin(i+1,j+1), func.fSave[k++]);
      }

      hist.fName = 'Func';
      hist.fTitle = func.fTitle;
      hist.fMinimum = func.fMinimum;
      hist.fMaximum = func.fMaximum;
      //fHistogram->SetContour(fContour.fN, levels);
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

   /** @summary draw TF2 object */
   static async draw(dom, tf2, opt) {
      if (!isStr(opt)) opt = '';
      let p = opt.indexOf(';webcanv_hist'), webcanv_hist = false;
      if (p >= 0) {
         webcanv_hist = true;
         opt = opt.slice(0, p);
      }

      let d = new DrawOptions(opt);
      if (d.empty())
         opt = 'cont3';
      else if (d.opt === 'SAME')
         opt = 'cont2 same';
      else
         opt = d.opt;

      // workaround for old waves.C
      if (opt == 'SAMECOLORZ' || opt == 'SAMECOLOR' || opt == 'SAMECOLZ')
         opt = 'SAMECOL';

      if (opt.indexOf('SAME') == 0)
         if (!getElementMainPainter(dom))
            opt = 'A_ADJUST_FRAME_' + opt.slice(4);

      let hist;

      if (webcanv_hist) {
         let dummy = new ObjectPainter(dom);

         hist = dummy.getPadPainter()?.findInPrimitives('Func', clTH2F);
      }

      if (!hist) hist = createHistogram(clTH2F, 20, 20);

      let painter = new TF2Painter(dom, hist);

      painter.$func = tf2;
      painter.webcanv_hist = webcanv_hist;
      painter.createTF2Histogram(tf2, hist);

      return THistPainter._drawHist(painter, opt);
   }

} // class TF2Painter

export { TF2Painter };
