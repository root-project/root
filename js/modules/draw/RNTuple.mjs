import { isStr } from '../core.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { createTreePlayer, treeDrawProgress } from './TTree.mjs';
import { rntupleDraw } from '../rntuple.mjs';

/** @summary function called from draw()
  * @desc just envelope for real TTree::Draw method which do the main job
  * Can be also used for the branch and leaf object
  * @private */
async function drawRNTuple(dom, obj, opt) {
   const args = {};
   let tuple;

   if (obj?.$tuple && obj.$field) {
      // case of fictional ROOT::RNTupleField
      tuple = obj.$tuple;
      args.expr = obj._name;
      if (isStr(opt) && opt.indexOf('dump') === 0) {
         args.expr += '>>' + opt;
         args.branch = obj.$field;
         args.copy_fields = false; // no need to copy fields, reading is simple
      } else if (opt)
         args.expr += opt;
   } else {
      tuple = obj;
      args.expr = opt;
      if (args.expr && isStr(args.expr)) {
         const p = args.expr.indexOf('player');
         if (p === 0) {
            args.player = true;
            args.expr = args.expr.slice(6);
            if (args.expr[0] === ':')
               args.expr = args.expr.slice(1);
         } else if ((p >= 0) && (p === args.expr.length - 6)) {
            args.player = true;
            args.expr = args.expr.slice(0, p);
            if ((p > 0) && (args.expr[p - 1] === ';'))
               args.expr = args.expr.slice(0, p - 1);
         }
      }
   }

   if (!tuple)
      throw Error('No RNTuple object available for drawing');

   let painter;

   if (args.player) {
      painter = new ObjectPainter(dom, obj, opt);
      createTreePlayer(painter, null, tuple);
      painter.rntupleDraw = rntupleDraw;
      painter.showPlayer(args);
      args.drawid = painter.drawid;
   } else
      args.drawid = dom;

   args.progress = treeDrawProgress.bind(args);

   if (args.expr || args.branch)
      return rntupleDraw(tuple, args).then(res => args.progress(res, true));

   return painter;
}

export { drawRNTuple };
