import { create, createHistogram, clTH1I, clTH2I, clTObjString, clTHashList, kNoZoom, kNoStats } from '../core.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { FunctionsHandler } from './THistPainter.mjs';
import { TH1Painter, PadDrawOptions } from './TH1Painter.mjs';
import { TGraphPainter } from './TGraphPainter.mjs';


/**
 * @summary Painter for TMultiGraph object.
 *
 * @private
 */

class TMultiGraphPainter extends ObjectPainter {

   /** @summary Create painter
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} obj - TMultiGraph object to draw */
   constructor(dom, mgraph) {
      super(dom, mgraph);
      this.firstpainter = null;
      this.autorange = false;
      this.painters = []; // keep painters to be able update objects
   }

   /** @summary Cleanup multigraph painter */
   cleanup() {
      this.painters = [];
      super.cleanup();
   }

   /** @summary Update multigraph object */
   updateObject(obj) {
      if (!this.matchObjectType(obj)) return false;

      const mgraph = this.getObject(),
            graphs = obj.fGraphs,
            pp = this.getPadPainter();

      mgraph.fTitle = obj.fTitle;

      let isany = false;
      if (this.firstpainter) {
         let histo = obj.fHistogram;
         if (this.autorange && !histo)
            histo = this.scanGraphsRange(graphs);

         if (this.firstpainter.updateObject(histo))
            isany = true;
      }

      const ngr = Math.min(graphs.arr.length, this.painters.length);

      for (let i = 0; i < ngr; ++i) {
         if (this.painters[i].updateObject(graphs.arr[i], (graphs.opt[i] || this._restopt) + this._auto))
            isany = true;
      }

      this._funcHandler = new FunctionsHandler(this, pp, obj.fFunctions);

      return isany;
   }

   /** @summary Redraw multigraph
     * @desc may redraw histogram which was used to draw axes
     * @return {Promise} for ready */
    async redraw(reason) {
       const promise = this.firstpainter?.redraw(reason) ?? Promise.resolve(true),
             redrawNext = async indx => {
                if (indx >= this.painters.length)
                   return this;
                return this.painters[indx].redraw(reason).then(() => redrawNext(indx + 1));
             };

       return promise.then(() => redrawNext(0)).then(() => {
          const res = this._funcHandler?.drawNext(0) ?? this;
          delete this._funcHandler;
          return res;
       });
    }

   /** @summary Scan graphs range
     * @return {object} histogram for axes drawing */
   scanGraphsRange(graphs, histo, pad) {
      const mgraph = this.getObject(),
            rw = { xmin: 0, xmax: 0, ymin: 0, ymax: 0, first: true };
      let maximum, minimum, logx = false, logy = false,
          time_display = false, time_format = '';

      if (pad) {
         logx = pad.fLogx;
         logy = pad.fLogv ?? pad.fLogy;
         rw.xmin = pad.fUxmin;
         rw.xmax = pad.fUxmax;
         rw.ymin = pad.fUymin;
         rw.ymax = pad.fUymax;
         rw.first = false;
      }

      // ignore existing histo in 3d case
      if (this._3d && histo && !histo.fXaxis.fLabels)
         histo = null;

      if (!histo) {
         this.autorange = true;

         if (graphs.arr[0]?.fHistogram?.fXaxis?.fTimeDisplay) {
            time_display = true;
            time_format = graphs.arr[0].fHistogram.fXaxis.fTimeFormat;
         }
      }

      graphs.arr.forEach(gr => {
         if (gr.fNpoints === 0) return;
         if (rw.first) {
            rw.xmin = rw.xmax = gr.fX[0];
            rw.ymin = rw.ymax = gr.fY[0];
            rw.first = false;
         }
         for (let i = 0; i < gr.fNpoints; ++i) {
            rw.xmin = Math.min(rw.xmin, gr.fX[i]);
            rw.xmax = Math.max(rw.xmax, gr.fX[i]);
            rw.ymin = Math.min(rw.ymin, gr.fY[i]);
            rw.ymax = Math.max(rw.ymax, gr.fY[i]);
         }
      });

      if (rw.xmin === rw.xmax)
         rw.xmax += 1;
      if (rw.ymin === rw.ymax)
         rw.ymax += 1;
      const dx = 0.05 * (rw.xmax - rw.xmin),
            dy = 0.05 * (rw.ymax - rw.ymin);
      let uxmin = rw.xmin - dx,
          uxmax = rw.xmax + dx;
      if (logy) {
         if (rw.ymin <= 0)
            rw.ymin = 0.001 * rw.ymax;
         minimum = rw.ymin / (1 + 0.5 * Math.log10(rw.ymax / rw.ymin));
         maximum = rw.ymax * (1 + 0.2 * Math.log10(rw.ymax / rw.ymin));
      } else {
         minimum = rw.ymin - dy;
         maximum = rw.ymax + dy;
      }
      if (minimum < 0 && rw.ymin >= 0)
         minimum = 0;
      if (maximum > 0 && rw.ymax <= 0)
         maximum = 0;

       const glob_minimum = minimum, glob_maximum = maximum;

      if (uxmin < 0 && rw.xmin >= 0)
         uxmin = logx ? 0.9 * rw.xmin : 0;
      if (uxmax > 0 && rw.xmax <= 0)
         uxmax = logx? 1.1 * rw.xmax : 0;

      if (mgraph.fMinimum !== kNoZoom)
         rw.ymin = minimum = mgraph.fMinimum;
      if (mgraph.fMaximum !== kNoZoom)
         rw.ymax = maximum = mgraph.fMaximum;

      if (minimum < 0 && rw.ymin >= 0 && logy)
         minimum = 0.9 * rw.ymin;
      if (maximum > 0 && rw.ymax <= 0 && logy)
         maximum = 1.1 * rw.ymax;
      if (minimum <= 0 && logy)
         minimum = 0.001 * maximum;
      if (!logy && minimum > 0 && minimum < 0.05*maximum)
         minimum = 0;
      if (uxmin <= 0 && logx)
         uxmin = (uxmax > 1000) ? 1 : 0.001 * uxmax;

      // Create a temporary histogram to draw the axis (if necessary)
      if (!histo) {
         let xaxis, yaxis;
         if (this._3d) {
            histo = createHistogram(clTH2I, graphs.arr.length, 10);
            xaxis = histo.fXaxis;
            xaxis.fXmin = 0;
            xaxis.fXmax = graphs.arr.length;
            xaxis.fLabels = create(clTHashList);
            for (let i = 0; i < graphs.arr.length; i++) {
               const lbl = create(clTObjString);
               lbl.fString = graphs.arr[i].fTitle || `gr${i}`;
               lbl.fUniqueID = graphs.arr.length - i; // graphs drawn in reverse order
               xaxis.fLabels.Add(lbl, '');
            }
            xaxis = histo.fYaxis;
            yaxis = histo.fZaxis;
         } else {
            histo = createHistogram(clTH1I, 10);
            xaxis = histo.fXaxis;
            yaxis = histo.fYaxis;
         }
         histo.fTitle = mgraph.fTitle;
         if (histo.fTitle.indexOf(';') >= 0) {
            const t = histo.fTitle.split(';');
            histo.fTitle = t[0];
            if (t[1]) xaxis.fTitle = t[1];
            if (t[2]) yaxis.fTitle = t[2];
         }

         xaxis.fXmin = uxmin;
         xaxis.fXmax = uxmax;
         xaxis.fTimeDisplay = time_display;
         if (time_display) xaxis.fTimeFormat = time_format;
      }

      const axis = this._3d ? histo.fZaxis : histo.fYaxis;
      axis.fXmin = Math.min(minimum, glob_minimum);
      axis.fXmax = Math.max(maximum, glob_maximum);
      histo.fMinimum = minimum;
      histo.fMaximum = maximum;
      histo.fBits |= kNoStats;

      return histo;
   }

   /** @summary draw speical histogram for axis
     * @return {Promise} when ready */
   async drawAxisHist(histo, hopt) {
      return TH1Painter.draw(this.getDom(), histo, hopt);
   }

   /** @summary Draw graph  */
   async drawGraph(gr, opt /*, pos3d */) {
      return TGraphPainter.draw(this.getDom(), gr, opt);
   }

   /** @summary method draws next graph  */
   async drawNextGraph(indx) {
      const graphs = this.getObject().fGraphs;

      // at the end of graphs drawing draw functions (if any)
      if (indx >= graphs.arr.length)
         return this;

      const gr = graphs.arr[indx],
            draw_opt = (graphs.opt[indx] || this._restopt) + this._auto;

      // used in automatic colors numbering
      if (this._auto)
         gr.$num_graphs = graphs.arr.length;

      return this.drawGraph(gr, draw_opt, graphs.arr.length - indx).then(subp => {
         if (subp) {
            subp.setSecondaryId(this, `graphs_${indx}`);
            this.painters.push(subp);
         }

         return this.drawNextGraph(indx+1);
      });
   }

   /** @summary Draw multigraph object using painter instance
     * @private */
   static async _drawMG(painter, opt) {
      const d = new DrawOptions(opt);

      painter._3d = d.check('3D');
      painter._auto = ''; // extra options for auto colors
      ['PFC', 'PLC', 'PMC'].forEach(f => { if (d.check(f)) painter._auto += ' ' + f; });

      let hopt = '';
      if (d.check('FB') && painter._3d) hopt += 'FB'; // will be directly combined with LEGO
      PadDrawOptions.forEach(name => { if (d.check(name)) hopt += ';' + name; });

      painter._restopt = d.remain();

      let promise = Promise.resolve(true);
      if (d.check('A') || !painter.getMainPainter()) {
          const mgraph = painter.getObject(),
                histo = painter.scanGraphsRange(mgraph.fGraphs, mgraph.fHistogram, painter.getPadPainter()?.getRootPad(true));

         promise = painter.drawAxisHist(histo, hopt).then(ap => {
            ap.setSecondaryId(painter, 'hist'); // mark that axis painter generated from mg
            painter.firstpainter = ap;
         });
      }

      return promise.then(() => {
         painter.addToPadPrimitives();
         return painter.drawNextGraph(0);
      }).then(() => {
         const handler = new FunctionsHandler(painter, painter.getPadPainter(), painter.getObject().fFunctions, true);
         return handler.drawNext(0); // returns painter
      });
   }

   /** @summary Draw TMultiGraph object */
   static async draw(dom, mgraph, opt) {
      return TMultiGraphPainter._drawMG(new TMultiGraphPainter(dom, mgraph), opt);
   }

} // class TMultiGraphPainter

export { TMultiGraphPainter };
