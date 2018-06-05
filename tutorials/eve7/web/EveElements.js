/** @file JSRootGeoBase.js */
/// Basic functions for work with TGeo classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( [ 'JSRootCore', 'threejs' ], factory );
   } else
   if (typeof exports === 'object' && typeof module !== 'undefined') {
      factory(require("./JSRootCore.js"), require("./three.min.js"));
   } else {

      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'EveElements.js');

      if (typeof JSROOT.EVE == 'undefined')
         throw new Error('JSROOT.EVE is not defined', 'EveElements.js');

      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'EveElements.js');

      factory(JSROOT, THREE);
   }
} (function( JSROOT, THREE ) {

   "use strict";

   function EveElements() {
      
   }
   
   JSROOT.EVE.EveElements = EveElements;
   
   console.log("LOADING EVE ELEMENTS");
   
   return JSROOT;

}));

