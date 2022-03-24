// initialization of EVE

import { settings, parse, browser, decodeUrl } from '../core.mjs';
import { draw, redraw } from '../draw.mjs';
import { root_line_styles } from '../base/TAttLineHandler.mjs';
import { createMenu } from '../gui/menu.mjs';
import * as three from '../three.mjs';
import * as geo1 from '../base/colors.mjs';
import * as geo2 from '../base/base3d.mjs';
import * as geo3 from './geobase.mjs';
import * as geo4 from './TGeoPainter.mjs';

function initEVE() {
   globalThis.THREE = Object.assign({}, three);
   // placeholder for all global functions used by EVE
   globalThis.EVE = Object.assign({ settings, browser, parse, decodeUrl, draw, redraw, root_line_styles, createMenu }, geo1, geo2, geo3, geo4);
}

export { initEVE };
