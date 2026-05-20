import { runGeoWorker } from './geobase.mjs';
import { parentPort } from 'node:worker_threads';

const ctxt = {};

parentPort.on('message', msg => runGeoWorker(ctxt, msg, reply => parentPort.postMessage(reply)));
