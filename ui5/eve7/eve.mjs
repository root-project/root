// initialization of EVE

function initEVE() {
   if (globalThis.EVE)
      return Promise.resolve(globalThis.EVE);

   return Promise.all([import('jsrootsys/modules/three.mjs'),
                       import('jsrootsys/modules/three_addons.mjs'),
                       import('jsrootsys/modules/core.mjs'),
                       import('jsrootsys/modules/draw.mjs'),
                       import('jsrootsys/modules/base/TAttLineHandler.mjs'),
                       import('jsrootsys/modules/gui/menu.mjs'),
                       import('jsrootsys/modules/base/colors.mjs'),
                       import('jsrootsys/modules/base/base3d.mjs'),
                       import('jsrootsys/modules/geom/geobase.mjs'),
                       import('jsrootsys/modules/geom/TGeoPainter.mjs')])
    .then(arr => {
       globalThis.THREE = Object.assign({}, arr.shift(), arr.shift());

       if (globalThis.THREE.OrbitControls) {
          globalThis.THREE.OrbitControls.prototype.resetOrthoPanZoom = function() {
            this._panOffset.set(0, 0, 0);
            this.object.zoom = 1;
            this.object.updateProjectionMatrix();
          }
       }

       globalThis.EVE = {};
       globalThis.EVE.JSR = Object.assign({}, ...arr); // JSROOT functionality
       return globalThis.EVE;
     });
}

export { initEVE };
