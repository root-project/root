sap.ui.define([
   'rootui5/eve7/lib/GlViewer',
   'rootui5/eve7/lib/EveElementsRCore'
], function(GlViewer, EveElements) {

   "use strict";

   function GlViewerRCore(viewer_class)
   {
      GlViewer.call(this, viewer_class);
   }

   var RC;

   GlViewerRCore.prototype = Object.assign(Object.create(GlViewer.prototype), {

      constructor: GlViewerRCore,

      init: function(controller)
      {
         GlViewer.prototype.init.call(this, controller);
         // super.init(controller);

         var pthis = this;

         import("../../eve7/rnr_core/RenderCore.js").then((module) => {

            console.log("GlViewerRCore.onInit - RenderCore.js loaded");

            RC = module;

            pthis.bootstrap();
         });
      },

      bootstrap: function()
      {
         this.creator = new EveElements(RC);
         // this.creator.useIndexAsIs = JSROOT.decodeUrl().has('useindx');

         this.createRCoreRenderer();
         this.controller.createScenes();
         this.controller.redrawScenes();
         this.setupRCoreDomAndEventHandlers();

         this.controller.glViewerInitDone();

         // As RC gets loaded asynchronously we've probably missed the initial resize.
         this.onResizeTimeout();
      },

      //==============================================================================

      make_object: function(name)
      {
         let c = new RC.Group();
         c.name = name || "<no-name>";
         return c;
      },

      get_top_scene: function()
      {
         return this.scene;
      },

      //==============================================================================

      createRCoreRenderer: function()
      {
         var w = this.get_width();
         var h = this.get_height();

         //this.canvas = document.createElement('canvas');
         this.canvas = document.createElementNS( 'http://www.w3.org/1999/xhtml', 'canvas' );

         this.get_view().getDomRef().appendChild(this.canvas);

         this.renderer = new RC.MeshRenderer(this.canvas, RC.WEBGL2);
         this.renderer.clearColor = "#FFFFFFFF";
         this.renderer.addShaderLoaderUrls("rootui5sys/eve7/rnr_core/shaders");

         if (this.controller.kind === "3D")
         {
            this.camera = new RC.PerspectiveCamera(75, w / h, 1, 5000);
            this.camera.position = new RC.Vector3(-500, 0, 0);
            this.camera.lookAt(new RC.Vector3(0, 0, 0), new RC.Vector3(0, 1, 0));
         }
         else
         {
            this.camera = new RC.OrthographicCamera(-w/2, w/2, -h/2, h/2, 0, 2000);
            this.camera.position = new RC.Vector3(0, 0, 500);
            this.camera.lookAt(new RC.Vector3(0, 0, 0), new RC.Vector3(0, 1, 0));
         }

         this.scene = new RC.Scene();

         this.rot_center = new THREE.Vector3(0,0,0);

         // Lights are positioned in resetRenderer

         this.point_lights = this.make_object("Lamp container");
         this.point_lights.add( new RC.PointLight( 0xff5050, 0.7 )); // R
         this.point_lights.add( new RC.PointLight( 0x50ff50, 0.7 )); // G
         this.point_lights.add( new RC.PointLight( 0x5050ff, 0.7 )); // B
         this.scene.add(this.point_lights);
      },

      setupRCoreDomAndEventHandlers: function()
      {
         // this.get_view().getDomRef().appendChild( this.renderer.domElement );

         // This will also call render().
         // this.resetRCoreRenderer();

         // this.onResizeTimeout();

         this.render();
      },

      resetRCoreRenderer: function()
      {
         let THREE = RC;

         let sbbox = new THREE.Box3();
         //sbbox.setFromObject( this.scene );
         // XXXX infinity ... no traversal?
         sbbox.expandByPoint(new RC.Point(-1000,-1000,-1000));
         sbbox.expandByPoint(new RC.Point( 1000, 1000, 1000));

         let posV = new THREE.Vector3; posV.subVectors(sbbox.max, this.rot_center);
         let negV = new THREE.Vector3; negV.subVectors(sbbox.min, this.rot_center);

         let extV = new THREE.Vector3; extV = negV; extV.negate(); extV.max(posV);
         let extR = extV.length();

         let lc = this.point_lights.children;
         lc[0].position.set( extR, extR, -extR);
         lc[1].position.set(-extR, extR,  extR);
         lc[2].position.set( extR, extR,  extR);

         if (this.controller.kind === "3D") // (this.camera.isPerspectiveCamera)
         {
            let posC = new THREE.Vector3(-0.7 * extR, 0.5 * extR, -0.7 * extR);

            this.camera.position.copy(posC);

            // this.controls.screenSpacePanning = true;

            // console.log("resetThreejsRenderer 3D scene bbox ", sbbox, ", camera_pos ", posC, ", look_at ", this.rot_center);
         }
         else
         {
            let posC = new THREE.Vector3(0, 0, 1000);

            this.camera.position.copy(posC);

            let ey = 1.02 * extV.y;
            let ex = ey / this.get_height() * this.get_width();
            this.camera.left   = -ex;
            this.camera.right  =  ex;
            this.camera.top    =  ey;
            this.camera.bottom = -ey;

            // this.controls.resetOrthoPanZoom();

            // this.controls.screenSpacePanning = true;
            // this.controls.enableRotate = false;

            // console.log("resetThreejsRenderer 2D scene bbox ex ey", sbbox, ex, ey, ", camera_pos ", posC, ", look_at ", this.rot_center);
         }
         // this.controls.target.copy( this.rot_center );

         // this.composer.reset();

         // this.controls.update();

         this.render();
      },

      //==============================================================================

      render: function()
      {
         console.log("RENDER", this.scene, this.camera, this.canvas, this.renderer);

         this.renderer.render( this.scene, this.camera );
      },

      //==============================================================================

      onResizeTimeout: function()
      {
         console.log("GlViewerRCore RESIZE ", this.get_width(), this.get_height(), "canvas=", this.canvas);

         let w = this.get_width();
         let h = this.get_height();

         this.canvas.width  = w;
         this.canvas.height = h;

         if (this.controller.kind === "3D")
         {
            this.camera.aspect = w / h;
         }
         else
         {
            this.camera.left  =  this.camera.bottom / h * w;
            this.camera.right = -this.camera.left;
            this.camera.updateProjectionMatrix();
         }
         // this.camera.updateProjectionMatrix();

         // this.renderer.setSize(w, h);
         this.renderer.updateViewport(w, h);

         //this.outline_pass.setSize(w, h);
         //this.fxaa_pass.uniforms.resolution.value.set(0.5 / w, 0.5 / h);

         //this.composer.reset();
         //this.controls.update();

         this.render();
      },

   });

   return GlViewerRCore;
});
