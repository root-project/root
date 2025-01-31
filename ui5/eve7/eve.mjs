// initialization of EVE

function initEVE(source_dir) {
   if (globalThis.EVE)
      return Promise.resolve(globalThis.EVE);

   let mpath = source_dir + 'modules/';

   return Promise.all([import(mpath+'three.mjs'),
                       import(mpath+'three_addons.mjs'),
                       import(mpath+'core.mjs'),
                       import(mpath+'draw.mjs'),
                       import(mpath+'base/TAttLineHandler.mjs'),
                       import(mpath+'gui/menu.mjs'),
                       import(mpath+'base/colors.mjs'),
                       import(mpath+'base/base3d.mjs'),
                       import(mpath+'geom/geobase.mjs'),
                       import(mpath+'geom/TGeoPainter.mjs')])
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
