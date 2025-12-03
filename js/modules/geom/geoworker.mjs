import { runGeoWorker } from './geobase.mjs';

const ctxt = {};

onmessage = function(e) {
   if (!e?.data)
      return;

   if (typeof e.data === 'string') {
      console.log(`Worker get message ${e.data}`);
      return;
   }

   runGeoWorker(ctxt, e.data, postMessage);
};
