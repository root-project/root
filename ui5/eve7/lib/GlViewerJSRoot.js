sap.ui.define([
   'rootui5/eve7/lib/GlViewer',
   'rootui5/eve7/lib/EveElements',
   'rootui5/eve7/lib/OutlinePass',
   'rootui5/eve7/lib/FXAAShader'
], function(GlViewer, EveElements) {

   "use strict";

   function GlViewerJSRoot(viewer_class)
   {
      GlViewer.call(this, viewer_class);
   };

   GlViewerJSRoot.prototype = Object.assign(Object.create(GlViewer.prototype), {

      constructor: GlViewerJSRoot,

      init: function(controller)
      {
         GlViewer.prototype.init.call(this, controller);
         //super.init(controller);

         this.creator = new EveElements(controller);
         this.creator.useIndexAsIs = (JSROOT.GetUrlOption('useindx') !== null);

         this.createGeoPainter();
      },

      //==============================================================================

      make_object: function(name)
      {
         return new THREE.Object3D;
      },

      get_top_scene: function()
      {
         return this.geo_painter.getExtrasContainer();
      },

      //==============================================================================

      createGeoPainter: function()
      {
         let options = "outline";
         // options += " black, ";
         if (this.controller.kind != "3D") options += ", ortho_camera";

         // TODO: should be specified somehow in XML file
         this.get_view().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");

         this.geo_painter = JSROOT.Painter.CreateGeoPainter(this.get_view().getDomRef(), null, options);

         // function used by TGeoPainter to create OutlineShader - for the moment remove from JSROOT
         this.geo_painter.createOutline = function(w,h)
         {
            // this here will be TGeoPainter!

            this.outline_pass = new THREE.OutlinePass( new THREE.Vector2( w, h ), this._scene, this._camera );
            this.outline_pass.edgeStrength = 5.5;
            this.outline_pass.edgeGlow = 0.7;
            this.outline_pass.edgeThickness = 1.5;
            this.outline_pass.usePatternTexture = false;
            this.outline_pass.downSampleRatio = 1;
            this.outline_pass.glowDownSampleRatio = 3;

            // const sh = THREE.OutlinePass.selection_enum["select"]; // doesnt stand for spherical harmonics :P
            // THREE.OutlinePass.selection_atts[sh].visibleEdgeColor.set('#dd1111');
            // THREE.OutlinePass.selection_atts[sh].hiddenEdgeColor.set('#1111dd');

            this._effectComposer.addPass( this.outline_pass );

            this.fxaa_pass = new THREE.ShaderPass( THREE.FXAAShader );
            this.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / w, 1 / h );
            this.fxaa_pass.renderToScreen = true;
            this._effectComposer.addPass( this.fxaa_pass );
         }

         // assign callback function - when needed
         this.geo_painter.WhenReady(this.onGeoPainterReady.bind(this));

         this.geo_painter.AssignObject(null);

         this.geo_painter.prepareObjectDraw(null); // and now start everything
      },

      onGeoPainterReady: function(painter)
      {
         console.log("GL_controller::onGeoPainterReady");

         // AMT temporary here, should be set in camera instantiation time
         if (this.geo_painter._camera.type == "OrthographicCamera")
         {
            this.geo_painter._camera.left   = -this.get_width();
            this.geo_painter._camera.right  =  this.get_width();
            this.geo_painter._camera.top    =  this.get_height();
            this.geo_painter._camera.bottom = -this.get_height();
            this.geo_painter._camera.updateProjectionMatrix();
         }

         painter.eveGLcontroller = this.controller;
         painter._controls.ProcessMouseMove = function(intersects)
         {
            var active_mesh = null, tooltip = null, resolve = null, names = [], geo_object, geo_index;

            // try to find mesh from intersections
            for (var k=0;k<intersects.length;++k)
            {
               var obj = intersects[k].object, info = null;
               if (!obj) continue;
               if (obj.geo_object) info = obj.geo_name; else
                  if (obj.stack) info = painter.GetStackFullName(obj.stack);
               if (info===null) continue;

               if (info.indexOf("<prnt>")==0)
                  info = painter.GetItemName() + info.substr(6);

               names.push(info);

               if (!active_mesh) {
                  active_mesh = obj;
                  tooltip = info;
                  geo_object = obj.geo_object;
                  if (obj.get_ctrl) {
                     geo_index = obj.get_ctrl().extractIndex(intersects[k]);
                     if ((geo_index !== undefined) && (typeof tooltip == "string")) tooltip += " indx:" + JSON.stringify(geo_index);
                  }
                  if (active_mesh.stack) resolve = painter.ResolveStack(active_mesh.stack);
               }
            }

            // painter.HighlightMesh(active_mesh, undefined, geo_object, geo_index); AMT override
            if (active_mesh && active_mesh.get_ctrl())
            {
               active_mesh.get_ctrl().elementHighlighted( 0xffaa33, geo_index);
            }
            else
            {
               var sl = painter.eveGLcontroller.created_scenes;
               for (var k=0; k < sl.length; ++k)
                  sl[k].clearHighlight();
            }


            if (painter.options.update_browser) {
               if (painter.options.highlight && tooltip) names = [ tooltip ];
               painter.ActivateInBrowser(names);
            }

            if (!resolve || !resolve.obj) return tooltip;

            var lines = JSROOT.GEO.provideInfo(resolve.obj);
            lines.unshift(tooltip);

            return { name: resolve.obj.fName, title: resolve.obj.fTitle || resolve.obj._typename, lines: lines };
         }

         // this.geo_painter._highlight_handlers = [ this ]; // register ourself for highlight handling
         this.last_highlight = null;

         // outline_pass passthrough
         this.outline_pass = this.geo_painter.outline_pass;

         var sz = this.geo_painter.size_for_3d();
         this.geo_painter._effectComposer.setSize( sz.width, sz.height);
         this.geo_painter.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / sz.width, 1 / sz.height );

         // create only when geo painter is ready
         this.controller.createScenes();
         this.controller.redrawScenes();

         this.geo_painter.adjustCameraPosition(true);
         this.render();
      },

      //==============================================================================

      render: function()
      {
         this.geo_painter.Render3D();
      },

      //==============================================================================

      onResizeTimeout: function()
      {
         this.geo_painter.CheckResize();
         if (this.geo_painter.fxaa_pass)
            this.geo_painter.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / this.geo_painter._scene_width, 1 / this.geo_painter._scene_height );
      },
   });

   return GlViewerJSRoot;
});
