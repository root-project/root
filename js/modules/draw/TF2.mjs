import { createHistogram, BIT } from '../core.mjs';
import { TH2Painter } from '../hist/TH2Painter.mjs';
import { proivdeEvalPar } from '../hist/TF1Painter.mjs';
import { getElementMainPainter } from '../base/ObjectPainter.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';


/** @summary Create histogram for TF2 drawing
  * @private */
function createTF2Histogram(func, hist = undefined) {
   let nsave = func.fSave.length, use_middle = true;
   if ((nsave > 6) && (nsave !== (func.fSave[nsave-2]+1)*(func.fSave[nsave-1]+1) + 6)) nsave = 0;

   // check if exact min/max range is used or created histogram has to be extended
   if ((nsave > 6) && (func.fXmin < func.fXmax) && (func.fSave[nsave-6] < func.fSave[nsave-5]) &&
      ((func.fSave[nsave-5] - func.fSave[nsave-6]) / (func.fXmax - func.fXmin) > 0.99999)) use_middle = false;

   let npx = Math.max(func.fNpx, 2),
       npy = Math.max(func.fNpy, 2),
       iserr = false, isany = false,
       dx = (func.fXmax - func.fXmin) / (use_middle ? npx : (npx-1)),
       dy = (func.fYmax - func.fYmin) / (use_middle ? npy : (npy-1)),
       extra = use_middle ? 0.5 : 0;

   for (let j = 0; j < npy; ++j)
     for (let i = 0; (i < npx) && !iserr; ++i) {
         let x = func.fXmin + (i + extra) * dx,
             y = func.fYmin + (j + extra) * dy,
             z = 0;

         try {
            z = func.evalPar(x, y);
         } catch {
            iserr = true;
         }

         if (!iserr && Number.isFinite(z)) {
            if (!hist) hist = createHistogram("TH2F", npx, npy);
            isany = true;
            hist.setBinContent(hist.getBin(i+1,j+1), z);
         }
      }

   let use_saved_points = (iserr || !isany) && (nsave > 6);
   if (!use_saved_points && !hist)
      hist = createHistogram("TH2F", npx, npy);

   if (!iserr && isany) {
      hist.fXaxis.fXmin = func.fXmin - (use_middle ? 0 : dx/2);
      hist.fXaxis.fXmax = func.fXmax + (use_middle ? 0 : dx/2);

      hist.fYaxis.fXmin = func.fYmin - (use_middle ? 0 : dy/2);
      hist.fYaxis.fXmax = func.fYmax + (use_middle ? 0 : dy/2);
   }

   if (use_saved_points) {
      npx = Math.round(func.fSave[nsave-2]);
      npy = Math.round(func.fSave[nsave-1]);
      dx = (func.fSave[nsave-5] - func.fSave[nsave-6]) / npx;
      dy = (func.fSave[nsave-3] - func.fSave[nsave-4]) / npy;

      if (!hist) hist = createHistogram("TH2F", npx+1, npy+1);

      hist.fXaxis.fXmin = func.fSave[nsave-6] - dx/2;
      hist.fXaxis.fXmax = func.fSave[nsave-5] + dx/2;

      hist.fYaxis.fXmin = func.fSave[nsave-4] - dy/2;
      hist.fYaxis.fXmax = func.fSave[nsave-3] + dy/2;

      for (let k = 0, j = 0; j <= npy; ++j)
         for (let i = 0; i <= npx; ++i)
            hist.setBinContent(hist.getBin(i+1,j+1), func.fSave[k++]);
   }

   hist.fName = "Func";
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
   const kNoStats = BIT(9);
   hist.fBits |= kNoStats;

   return hist;
}

/** @summary draw TF2 object
  * @desc TF2 always drawn via temporary TH2 object,
  * therefore there is no special painter class
  * @private */
function drawTF2(dom, func, opt) {

   proivdeEvalPar(func);

   let hist = createTF2Histogram(func);
   if (!hist) return;

   let d = new DrawOptions(opt);

   if (d.empty())
      opt = "cont3";
   else if (d.opt === "SAME")
      opt = "cont2 same";
   else
      opt = d.opt;

   // workaround for old waves.C
   if (opt == "SAMECOLORZ" || opt == "SAMECOLOR" || opt == "SAMECOLZ") opt = "SAMECOL";

   if (opt.indexOf("SAME") == 0)
      if (!getElementMainPainter(dom))
         opt = "A_ADJUST_FRAME_" + opt.slice(4);

   return TH2Painter.draw(dom, hist, opt).then(hpainter => {

      hpainter.tf2_typename = func._typename;

      hpainter.updateObject = function(obj /*, opt*/) {
         if (!obj || (this.tf2_typename != obj._typename)) return false;
         proivdeEvalPar(obj);
         createTF2Histogram(obj, this.getHisto());
         return true;
      };

      return hpainter;
   });
}

export { drawTF2 };
