import { internals, clTMarker } from '../core.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TH1Painter } from '../hist2d/TH1Painter.mjs';
import { draw } from '../draw.mjs';


/**
 * @summary Painter for TGraphTime object
 *
 * @private
 */

class TGraphTimePainter extends ObjectPainter {

   #step; // step number
   #selfid; // use to identify primitives which should be clean
   #wait_animation_frame; // animation flag
   #running_timeout; // timeout handle

   constructor(dom, gr, opt) {
      super(dom, gr, opt);
      this.decodeOptions(opt);
      this.#selfid = 'grtime_' + internals.id_counter++;
   }

   /** @summary Redraw object */
   redraw() {
      if (this.#step === undefined)
         this.startDrawing();
   }

   /** @summary Decode drawing options */
   decodeOptions(opt) {
      const d = new DrawOptions(opt || 'REPEAT');

      this.setOptions({
         once: d.check('ONCE'),
         repeat: d.check('REPEAT'),
         first: d.check('FIRST')
      });

      this.storeDrawOpt(opt);
   }

   /** @summary Draw primitives */
   async drawPrimitives(indx) {
      const lst = this.getObject()?.fSteps.arr[this.#step];

      if (!lst || (indx >= lst.arr.length))
         return;

      const obj = lst.arr[indx],
            opt = lst.opt[indx] + (obj._typename === clTMarker ? ';no_interactive' : '');

      return draw(this.getPadPainter(), obj, opt).then(p => {
         if (p) {
            p.$grtimeid = this.#selfid; // indicator that painter created by ourself
            p.$grstep = this.#step; // remember step
         }
         return this.drawPrimitives(indx + 1);
      });
   }

   /** @summary Continue drawing */
   continueDrawing() {
      const gr = this.getObject(),
            o = this.getOptions();

      if (o.first) {
         // draw only single frame, cancel all others
         this.#step = undefined;
         return;
      }

      if (this.#wait_animation_frame) {
         this.#wait_animation_frame = undefined;

         // clear pad
         const pp = this.getPadPainter();
         if (!pp) {
            // most probably, pad is cleared
            this.#step = undefined;
            return;
         }

         // draw primitives again
         this.drawPrimitives(0).then(() => {
            // clear primitives produced by previous drawing to avoid flicking
            pp.cleanPrimitives(p => { return (p.$grtimeid === this.#selfid) && (p.$grstep !== this.#step); });

            this.continueDrawing();
         });
      } else if (this.#running_timeout) {
         clearTimeout(this.#running_timeout);
         this.#running_timeout = undefined;

         this.#wait_animation_frame = true;
         // use animation frame to disable update in inactive form
         requestAnimationFrame(() => this.continueDrawing());
      } else {
         let sleeptime = Math.max(gr.fSleepTime, 10);

         if (++this.#step > gr.fSteps.arr.length) {
            if (o.repeat) {
               this.#step = 0; // start again
               sleeptime = Math.max(5000, 5*sleeptime); // increase sleep time
            } else {
               this.#step = undefined;    // clear indicator that animation running
               return;
            }
         }

         this.#running_timeout = setTimeout(() => this.continueDrawing(), sleeptime);
      }
   }

   /** @summary Start drawing of TGraphTime */
   startDrawing() {
      this.#step = 0;

      return this.drawPrimitives(0).then(() => {
         this.continueDrawing();
         return this;
      });
   }

   /** @summary Draw TGraphTime object */
   static async draw(dom, gr, opt) {
      if (!gr.fFrame) {
        console.error('Frame histogram not exists');
        return null;
      }

      const painter = new TGraphTimePainter(dom, gr, opt);

      if (painter.getMainPainter()) {
         console.error('Cannot draw graph time on top of other histograms');
         return null;
      }

      if (!gr.fFrame.fTitle && gr.fTitle) {
         const arr = gr.fTitle.split(';');
         gr.fFrame.fTitle = arr[0];
         if (arr[1]) gr.fFrame.fXaxis.fTitle = arr[1];
         if (arr[2]) gr.fFrame.fYaxis.fTitle = arr[2];
      }

      return TH1Painter.draw(dom, gr.fFrame, '').then(() => {
         painter.addToPadPrimitives();
         return painter.startDrawing();
      });
   }

} // class TGraphTimePainter


/** @summary Draw TRooPlot
  * @private */
async function drawRooPlot(dom, plot) {
   return draw(dom, plot._hist, 'hist').then(async hp => {
      const arr = [];
      for (let i = 0; i < plot._items.arr.length; ++i)
         arr.push(draw(dom, plot._items.arr[i], plot._items.opt[i]));
      return Promise.all(arr).then(() => hp);
   });
}

export { TGraphTimePainter, drawRooPlot };
